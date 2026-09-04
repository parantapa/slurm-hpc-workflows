"""Batch Bayesian optimization of search spaces, with botorch.

Fits a Gaussian process to everything measured so far and asks it for a
whole batch of points at once, chosen jointly so they do not stack on one
spot; evaluates that batch across a pilot pool; refits; repeats.

Exploration is not done here.
An `OptimizeSpaceBotorch` is handed the results files that
`ExploreSpaceSobolQMC.save` wrote and starts from those observations,
which is also how a search is resumed:
pass the exploration file and every optimization file since,
and the new instance picks up where the last one stopped.

Several spaces are optimized at once rather than one after another.
Each is an `OptimizationTask`, with its own space, objective and queues,
and one round fits every task's model together and then evaluates every
task's batch together, so the pool is filled by all of them rather than
by whichever one is having its turn.
"""

from __future__ import annotations

import gzip
import pickle
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from .slurm_pilot_executor import SlurmPilotExecutor, RaiseOnError, Task
from .search_space import SearchSpace, space_dim, to_params, to_unit
from .explore_space import load_results
from .utils import RemoteExecutionError, format_mapping, objective_value

# Botorch recommendation:
# single precision makes the GP fits
# and the acquisition optimization numerically fragile.
DTYPE = torch.double


ObjectiveOutput = Mapping[str, Any]
ObjectiveFunction = Callable[..., ObjectiveOutput]


# The keys `fit_and_propose` returns,
# read back by the driver in `_fit_and_propose`.
# How far outside the unit cube a saved point may land before it is taken
# for a point from another space. Standardizing a value at the edge of a
# log range is a couple of floating point operations off exactly 0 or 1.
SAVED_POINT_TOLERANCE = 1e-9

CANDIDATES_KEY = "candidates"
FIT_SECONDS_KEY = "fit_seconds"
PROPOSE_SECONDS_KEY = "propose_seconds"


@dataclass
class OptimizationTask:
    """One space to optimize, and everything needed to optimize it.

    name: names this task. It labels the progress lines, keys the results,
        and is what links the task to its observations in the results files,
        so it has to be the name the exploration ran under.
    space: the search space.
    objective: the function to minimize.
        Its argument names must match the keys of `space`.
        It returns a mapping: the value to minimize under `objective_key`,
        plus anything else worth recording about the evaluation.
        Only the value under `objective_key` is modelled,
        but the whole mapping is kept.
    objective_queue: queue(s) the objective evaluations are submitted to.
    optimizer_queue: queue(s) the model fit and acquisition optimization are
        submitted to, one task per round.
    search_parallelism: points evaluated per round.
        Left as None, the count `OptimizeSpaceBotorch` was constructed with
        is used instead.
        Match it to the pool: a bigger batch queues behind the workers,
        a smaller one leaves workers idle.

    The search runs between min_search_iterations and max_search_iterations
    rounds, stopping early when it stops improving:

    min_search_iterations: rounds that always run.
        Stalled rounds below it count towards patience
        but cannot be the round that ends the search,
        so a search that never improves runs exactly this many rounds
        and the earliest stop is max(min_search_iterations, patience).
    max_search_iterations: hard ceiling,
        reached even if the search is still improving.
    patience: consecutive stalled rounds that end the search.
        A round that improves resets the count,
        so this bounds a run of stalled rounds, not their total.
    min_improvement: fractional improvement in the best value that a round
        must deliver to count as improving, measured relative to the
        magnitude of the incumbent. 0.05 requires each round to beat it by 5%.

    objective_key: the key in the objective's result holding the value to
        minimize. Set it when the objective already reports under another
        name, such as "loss" or "rmse". Every other key is recorded and not
        modelled. Lower is better: negate a score you would rather maximize.

    The rest tune the fit and the acquisition optimization:

    num_restarts: multi-start count for `optimize_acqf`.
        The acquisition surface is multimodal,
        so a single start routinely lands in a local optimum.
    raw_samples: candidates `optimize_acqf` draws
        to pick those starting points from.
    mc_samples: quasi-MC draws used to estimate the acquisition value
        at a candidate.
        Sobol' draws are stratified, so they carry further
        than the same number of independent normal samples.
    acqf_timeout_s: wall-clock budget for one `optimize_acqf` call.
        Proposal cost grows with the number of observations,
        so an unbounded search can spend longer choosing a batch
        than evaluating one.
        Hitting the limit is not an error:
        `optimize_acqf` returns the best candidates found so far
        --- a full batch, finite and inside the bounds,
        just less thoroughly optimized.

    extra_objective_kwargs: extra keyword arguments for the objective.
        Forwarded verbatim, and may not shadow a parameter of the space.
    """

    name: str
    space: SearchSpace
    objective: ObjectiveFunction
    objective_queue: str | list[str]
    optimizer_queue: str | list[str]
    search_parallelism: int | None = None
    min_search_iterations: int = 5
    max_search_iterations: int = 30
    patience: int = 3
    min_improvement: float = 0.05
    objective_key: str = "objective"
    num_restarts: int = 10
    raw_samples: int = 128
    mc_samples: int = 128
    acqf_timeout_s: float = 10.0
    extra_objective_kwargs: dict = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """What one task measured, in submission order.

    The four lists are index-aligned:
    `points[i]` was evaluated, returned `outputs[i]`,
    was modelled by `values[i]`, and sits at `unit_points[i]`
    in the unit cube.

    `unit_points` holds the standardized coordinates of the point the
    objective actually ran at, after rounding rather than the continuous
    proposal, so the model is never told about a location that was not
    evaluated.
    """

    points: list[dict[str, Any]] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    # The objective's whole result, not just the number modelled from it.
    outputs: list[dict[str, Any]] = field(default_factory=list)
    unit_points: list[list[float]] = field(default_factory=list)


def fit_and_propose(
    unit_points: list[list[float]],
    values: list[float],
    batch: int,
    *,
    num_restarts: int,
    raw_samples: int,
    mc_samples: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Fit the GP and optimize the acquisition, returning `batch` unit points."""
    train_x = torch.tensor(unit_points, dtype=DTYPE)

    # Botorch maximizes, the objective is minimized:
    # the model is fit to -f,
    # and every acquisition value below is in that negated space too.
    train_y = torch.tensor([[-v] for v in values], dtype=DTYPE)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    started = time.monotonic()
    fit_gpytorch_mll(mll)
    fit_seconds = time.monotonic() - started

    dim = train_x.shape[-1]
    bounds = torch.stack([torch.zeros(dim, dtype=DTYPE), torch.ones(dim, dtype=DTYPE)])

    acqf = qLogNoisyExpectedImprovement(
        model,
        train_x,
        sampler=SobolQMCNormalSampler(torch.Size([mc_samples])),
    )

    started = time.monotonic()

    # The whole batch in one call, optimized jointly rather than greedily:
    # joint optimization keeps the parallel proposals from stacking
    # on one spot, and `sequential=True` measured far slower here.
    candidates, _ = optimize_acqf(
        acqf,
        bounds=bounds,
        q=batch,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        timeout_sec=timeout_s,
    )

    propose_seconds = time.monotonic() - started

    return {
        CANDIDATES_KEY: candidates.tolist(),
        FIT_SECONDS_KEY: fit_seconds,
        PROPOSE_SECONDS_KEY: propose_seconds,
    }


class OptimizeSpaceBotorch:
    """Botorch batch optimization of one or more search spaces, run together.

    Integer and categorical parameters are handled
    by rounding a continuous proposal,
    so distinct proposals can collapse onto the same point.
    The GP's learned noise term absorbs the repeated observations;
    on a mostly-discrete space with few levels
    the search re-evaluates points it has already seen.
    """

    def __init__(
        self,
        tasks: list[OptimizationTask],
        executor: SlurmPilotExecutor,
        files: Iterable[Path | str],
        search_parallelism: int | None = None,
    ):
        """Initialize.

        tasks: the spaces to optimize, one `OptimizationTask` each.
            They run together rather than one after another,
            so a two dimensional space does not hold the pool
            while a twenty dimensional one waits its turn.
        executor: executor for parallelizing objective execution.
        files: results files to start from, as written by
            `ExploreSpaceSobolQMC.save` or by this class's own `save`.
            A task is modelled on every observation these hold under its
            name; a task with none of them is an error, since there is
            nothing to fit. Pass the exploration file and every optimization
            file since to carry on from where the last run stopped.
        search_parallelism: batch size for tasks that do not carry their own.
            A task with neither is an error rather than a guess.

        Every task is checked here rather than when it runs,
        so a misspelled keyword or an empty space
        is reported before anything reaches the cluster.
        """
        if not tasks:
            raise ValueError("no optimization tasks given")

        names = [task.name for task in tasks]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"optimization task names must be unique: {duplicates}")

        self.executor = executor
        self.tasks = [self._resolve(task, search_parallelism) for task in tasks]

        # What the files hold, keyed by task name: the starting point of the
        # model and not this run's work, so `save` does not write it back out.
        self.prior: dict[str, OptimizationResult] = {}
        # What this instance has evaluated, which is what `save` writes.
        self.results: dict[str, OptimizationResult] = {}

        loaded = load_results(files)
        for task in self.tasks:
            self.prior[task.name] = self._observed(task, loaded.get(task.name))
            self.results[task.name] = OptimizationResult()

    @staticmethod
    def _resolve(
        task: OptimizationTask, default_parallelism: int | None
    ) -> OptimizationTask:
        """Validate one task and fill in what it left to the run."""
        if not task.space:
            raise ValueError(f"{task.name}: search space is empty")

        overlap = set(task.extra_objective_kwargs) & set(task.space)
        if overlap:
            raise ValueError(
                f"{task.name}: extra_objective_kwargs may not shadow search "
                f"space parameters: {sorted(overlap)}"
            )

        parallelism = task.search_parallelism
        if parallelism is None:
            parallelism = default_parallelism
        if parallelism is None:
            raise ValueError(
                f"{task.name}: no search_parallelism, on the task or on the run"
            )
        if parallelism < 1:
            raise ValueError(
                f"{task.name}: search_parallelism must be >= 1, got {parallelism}"
            )

        if task.min_search_iterations < 0:
            raise ValueError(
                f"{task.name}: min_search_iterations must be >= 0, "
                f"got {task.min_search_iterations}"
            )
        if task.max_search_iterations < task.min_search_iterations:
            raise ValueError(
                f"{task.name}: max_search_iterations must be >= "
                f"min_search_iterations, got {task.max_search_iterations} < "
                f"{task.min_search_iterations}"
            )
        if task.patience < 1:
            raise ValueError(f"{task.name}: patience must be >= 1, got {task.patience}")
        if task.min_improvement < 0.0:
            raise ValueError(
                f"{task.name}: min_improvement must be >= 0, "
                f"got {task.min_improvement}"
            )
        if task.num_restarts < 1:
            raise ValueError(
                f"{task.name}: num_restarts must be >= 1, got {task.num_restarts}"
            )
        if task.raw_samples < 1:
            raise ValueError(
                f"{task.name}: raw_samples must be >= 1, got {task.raw_samples}"
            )
        if task.mc_samples < 1:
            raise ValueError(
                f"{task.name}: mc_samples must be >= 1, got {task.mc_samples}"
            )
        if task.acqf_timeout_s <= 0.0:
            raise ValueError(
                f"{task.name}: acqf_timeout_s must be > 0, got {task.acqf_timeout_s}"
            )

        return replace(task, space=dict(task.space), search_parallelism=parallelism)

    @staticmethod
    def _observed(task: OptimizationTask, saved) -> OptimizationResult:
        """A task's starting observations, standardized into its own space.

        The file carries the points as the objective saw them,
        so the unit cube coordinates are recomputed here
        against the space this task declares.
        A point the space cannot place is a file for another search,
        which is worth saying plainly rather than fitting a model on.
        """
        if saved is None or not saved.values:
            raise RuntimeError(
                f"{task.name}: no observations in the given files; "
                "explore the space first, and pass what "
                "ExploreSpaceSobolQMC.save() wrote"
            )

        unit_points = [
            OptimizeSpaceBotorch._saved_unit_point(task, params)
            for params in saved.points
        ]

        for value in saved.values:
            objective_value(task.name, "saved value", {}, {"saved value": value})

        return OptimizationResult(
            points=list(saved.points),
            values=[float(value) for value in saved.values],
            outputs=list(saved.outputs),
            unit_points=unit_points,
        )

    @staticmethod
    def _saved_unit_point(
        task: OptimizationTask, params: Mapping[str, Any]
    ) -> list[float]:
        """One saved point, standardized into a task's own space.

        Checked rather than trusted, because a results file says nothing
        about the space it was measured over.
        A point from another search would otherwise be fit on:
        a parameter the space no longer has is dropped silently by
        `to_unit`, and a range that has since been narrowed standardizes
        to a coordinate outside the unit cube the acquisition searches,
        so the model would be told about a point the search cannot reach.
        """
        mismatch = set(params) ^ set(task.space)
        if mismatch:
            raise RuntimeError(
                f"{task.name}: a saved point has parameters {sorted(params)}, "
                f"which do not match the search space {sorted(task.space)}"
            )

        try:
            unit = to_unit(task.space, params)
        except ValueError as e:
            # A log range cannot standardize a value at or below zero.
            raise RuntimeError(
                f"{task.name}: a saved point {dict(params)} cannot be placed "
                f"in the search space {sorted(task.space)}: {e}"
            ) from e

        outside = [
            name
            for name, coordinate in zip(task.space, unit)
            if not -SAVED_POINT_TOLERANCE <= coordinate <= 1.0 + SAVED_POINT_TOLERANCE
        ]
        if outside:
            raise RuntimeError(
                f"{task.name}: a saved point {dict(params)} lies outside the "
                f"search space in {sorted(outside)}; it was measured over a "
                "different range, and the model may not be fit on it"
            )

        return unit

    def _task(self, name: str) -> OptimizationTask:
        """The named task, or a `KeyError` naming the ones there are."""
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(
            f"no optimization task named {name!r}; have {sorted(self.results)}"
        )

    def dim(self, name: str) -> int:
        """Dimensionality of a task's search space."""
        return space_dim(self._task(name).space)

    def observations(self, name: str) -> tuple[list[list[float]], list[float]]:
        """Everything a task's model is fit on: the files, then this run."""
        self._task(name)
        prior, results = self.prior[name], self.results[name]
        return (
            prior.unit_points + results.unit_points,
            prior.values + results.values,
        )

    def num_observations(self, name: str) -> int:
        """How many points a task's model is fit on."""
        return len(self.observations(name)[1])

    def run_search_jobs(self) -> None:
        """Run search rounds for every task until each one is done.

        A round is two batches, each submitted whole and waited for once:

        * one `fit_and_propose` task per still-running search, which trains a
          SingleTaskGP on that task's observations and proposes
          `search_parallelism` points jointly using the
          qLogNoisyExpectedImprovement acquisition;
        * every proposed point of every task, evaluated on its own
          objective queue.

        Tasks therefore advance in step, one round each, and drop out
        independently: a task stops when `patience` consecutive rounds fail
        to improve its best value by `min_improvement`, or when it reaches
        its own `max_search_iterations`. Stalled rounds are counted from the
        first round, including those below `min_search_iterations`: the floor
        holds off the stop, not the counting, so the earliest stop is
        max(`min_search_iterations`, `patience`).

        Called again, it runs another set of rounds from where this one
        stopped, modelling everything measured so far.
        """
        active = list(self.tasks)
        stalled = {task.name: 0 for task in self.tasks}

        round_number = 0
        while active:
            round_number += 1
            desc = f"search round {round_number}"

            previous_best = {task.name: self._best_value(task.name) for task in active}

            proposals = self._fit_and_propose(active, desc)
            self._evaluate(
                {
                    task.name: [
                        to_params(task.space, candidate)
                        for candidate in proposals[task.name]
                    ]
                    for task in active
                },
                desc,
            )

            still_running = []
            for task in active:
                self._report_best(task.name)

                if self._improved_enough(
                    task, previous_best[task.name], self._best_value(task.name)
                ):
                    stalled[task.name] = 0
                    still_running.append(task)
                    continue

                stalled[task.name] += 1

                # A stalled round ends the search only when both bounds are
                # met: the streak reaches `patience` and the round count
                # reaches the floor.
                # Report the larger gap: `stalled`/`patience` alone runs past
                # its own denominator when the floor outlasts the streak.
                remaining = max(
                    task.patience - stalled[task.name],
                    task.min_search_iterations - round_number,
                )

                if remaining > 0:
                    print(
                        f"{task.name}: round {round_number} improved by less "
                        f"than {task.min_improvement:.0%} "
                        f"--- {stalled[task.name]} in a row, "
                        f"{remaining} more to stop",
                        flush=True,
                    )
                    still_running.append(task)
                    continue

                print(
                    f"{task.name}: stopping after {round_number} rounds "
                    f"--- {stalled[task.name]} in a row without a "
                    f"{task.min_improvement:.0%} improvement",
                    flush=True,
                )

            active = [
                task
                for task in still_running
                if round_number < task.max_search_iterations
            ]
            for task in still_running:
                if task not in active:
                    print(
                        f"{task.name}: stopping after {round_number} rounds "
                        f"--- the ceiling on this search",
                        flush=True,
                    )

    def _fit_and_propose(
        self, tasks: list[OptimizationTask], desc: str
    ) -> dict[str, list[list[float]]]:
        """One fit per task, all of them submitted before any is waited for."""
        submissions: list[tuple[str, Task]] = []
        for task in tasks:
            unit_points, values = self.observations(task.name)
            print(
                f"{task.name}: fitting GP on {len(values)} points ...",
                flush=True,
            )
            submissions.append(
                (
                    task.name,
                    self.executor.submit(
                        task.optimizer_queue,
                        fit_and_propose,
                        unit_points,
                        values,
                        task.search_parallelism,
                        num_restarts=task.num_restarts,
                        raw_samples=task.raw_samples,
                        mc_samples=task.mc_samples,
                        timeout_s=task.acqf_timeout_s,
                    ),
                )
            )

        try:
            self.executor.wait(
                [submission for _, submission in submissions],
                desc=desc,
                unit="fit",
                raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED,
            )
        except RuntimeError as e:
            # Named in full, because the traceback is in a worker log the
            # driver never reads, and the usual cause is that the optimizer
            # queue's workers cannot import botorch.
            broken = [
                f"{name} on queue {self._task(name).optimizer_queue!r} "
                f"--- {submission.output}"
                for name, submission in submissions
                if isinstance(submission.output, RemoteExecutionError)
            ]
            detail = f": {'; '.join(broken)}" if broken else ""
            raise RuntimeError(f"fitting the model failed during {desc}{detail}") from e

        proposals = {}
        for task, (_, submission) in zip(tasks, submissions):
            proposals[task.name] = self._candidates(task, submission, desc)
        return proposals

    def _candidates(
        self, task: OptimizationTask, submission: Task, desc: str
    ) -> list[list[float]]:
        """The batch one fit proposed, checked before it is evaluated."""
        # Check every key read below, not just the candidates:
        # a version-skewed worker would otherwise reach the timings
        # as a bare KeyError, skipping this message.
        result = submission.output
        expected = (CANDIDATES_KEY, FIT_SECONDS_KEY, PROPOSE_SECONDS_KEY)
        if isinstance(result, Mapping):
            missing = [key for key in expected if key not in result]
        else:
            missing = list(expected)
        if missing:
            raise RuntimeError(
                f"{task.name}: the optimizer queue returned {result!r} "
                f"during {desc}, with no "
                f"{', '.join(repr(key) for key in missing)} in it; "
                "check that its workers run the same slurm-workflows "
                "as this driver"
            )

        candidates = result[CANDIDATES_KEY]

        # Every round is the full width of the pool;
        # a short batch would silently narrow it.
        if len(candidates) != task.search_parallelism:
            raise RuntimeError(
                f"{task.name}: the optimizer queue proposed "
                f"{len(candidates)} points during {desc}, not the "
                f"{task.search_parallelism} asked for"
            )

        print(
            f"{task.name}: GP fit took {result[FIT_SECONDS_KEY]:.2f}s, "
            f"proposed {len(candidates)} points in "
            f"{result[PROPOSE_SECONDS_KEY]:.2f}s",
            flush=True,
        )

        return candidates

    def _evaluate(self, batches: dict[str, list[dict[str, Any]]], desc: str) -> None:
        """Evaluate every task's batch together and record the results."""
        submitted: list[tuple[OptimizationTask, dict[str, Any], Task]] = []
        for name, points in batches.items():
            task = self._task(name)
            for params in points:
                submitted.append(
                    (
                        task,
                        params,
                        self.executor.submit(
                            task.objective_queue,
                            task.objective,
                            **params,
                            **task.extra_objective_kwargs,
                        ),
                    )
                )

        try:
            self._wait(
                [(task.name, submission) for task, _, submission in submitted],
                desc,
                unit="point",
                what="objective evaluations",
            )
        except RuntimeError:
            # Keep what did come back before reporting the failure.
            # A round is every task's batch, so one bad point would
            # otherwise cost the rounds of the tasks that were fine,
            # and `save()` is what the next run reads.
            self._record_returned(submitted)
            raise

        for task, params, submission in submitted:
            self._record(task, params, submission)

    def _wait(
        self, submissions: list[tuple[str, Task]], desc: str, unit: str, what: str
    ) -> None:
        """Wait for a whole batch, and say which tasks failed if it did.

        Deferred rather than immediate:
        one failed evaluation should not hide the other 39,
        and with several searches in flight
        the interesting question is which of them broke.
        """
        try:
            self.executor.wait(
                [submission for _, submission in submissions],
                desc=desc,
                unit=unit,
                raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED,
            )
        except RuntimeError as e:
            failed = sorted(
                {
                    name
                    for name, submission in submissions
                    if isinstance(submission.output, RemoteExecutionError)
                }
            )
            # Empty when nothing came back at all, a canceled task say,
            # in which case the cause is in the exception this chains to.
            named = f" of {failed}" if failed else ""
            raise RuntimeError(f"{what} failed during {desc}{named}") from e

    def _record_returned(
        self, submitted: list[tuple[OptimizationTask, dict[str, Any], Task]]
    ) -> None:
        """Record every evaluation that did come back, ignoring the rest.

        For the failure path only, where an exception is already on its way:
        a point that failed, or returned something unusable, is skipped
        rather than raising over the report of what went wrong.
        """
        for task, params, submission in submitted:
            try:
                self._record(task, params, submission)
            except RuntimeError:
                continue

    def _record(
        self, task: OptimizationTask, params: dict[str, Any], submission: Task
    ) -> None:
        """Check one evaluation's result and add it to its task's record."""
        output = submission.output
        value = objective_value(task.name, task.objective_key, params, output)

        result = self.results[task.name]
        result.points.append(params)
        result.values.append(value)
        # Copied, so a later mutation of the returned mapping
        # cannot rewrite what the run recorded.
        result.outputs.append(dict(output))
        result.unit_points.append(to_unit(task.space, params))

    def _improved_enough(
        self, task: OptimizationTask, previous: float, current: float
    ) -> bool:
        """Whether `current` beats `previous` by at least `min_improvement`.

        The threshold is fractional,
        measured against the magnitude of the incumbent,
        so it means the same thing
        whether the objective is scaled in seconds or in nanoseconds.

        A negative incumbent works the same way:
        -10 to -11 is a 10% improvement.
        An incumbent of exactly zero has no magnitude
        to take a fraction of, so any strict decrease counts.
        """
        if current >= previous:
            return False

        magnitude = abs(previous)
        if magnitude == 0.0:
            return True

        return (previous - current) / magnitude >= task.min_improvement

    def _all(self, name: str) -> OptimizationResult:
        """One task's observations, the files and this run together."""
        prior, results = self.prior[name], self.results[name]
        return OptimizationResult(
            points=prior.points + results.points,
            values=prior.values + results.values,
            outputs=prior.outputs + results.outputs,
            unit_points=prior.unit_points + results.unit_points,
        )

    def _best_value(self, name: str) -> float:
        """The lowest value a task knows of, from the files or this run."""
        return min(self.prior[name].values + self.results[name].values)

    def _report_best(self, name: str) -> None:
        """Print the best point a task knows of.

        Printed after every round:
        several rounds with an unchanged best
        mean the budget is being spent without improving anything.
        """
        known = self._all(name)
        best = min(range(len(known.values)), key=known.values.__getitem__)
        params = format_mapping(known.points[best])
        # The whole result, not just the objective value.
        output = format_mapping(known.outputs[best])
        print(
            f"{name}: best after {len(known.values)} points "
            f"at {params} -> {output}",
            flush=True,
        )

    def best_point(self, name: str) -> tuple[dict[str, Any], float]:
        """A task's best point (params, objective value) known so far.

        Over everything the task knows: the observations it started from
        as well as the points this run evaluated.
        """
        known = self._all(self._task(name).name)
        best = min(range(len(known.values)), key=known.values.__getitem__)
        return dict(known.points[best]), known.values[best]

    def best_output(self, name: str) -> dict[str, Any]:
        """The objective's whole result at a task's best point so far."""
        known = self._all(self._task(name).name)
        best = min(range(len(known.values)), key=known.values.__getitem__)
        return dict(known.outputs[best])

    def save(self, path: Path | str) -> None:
        """Write what this run measured to a gzipped pickle.

        Only this run: the observations it started from are already in the
        files it was given, and writing them back out would double them the
        next time both files are passed together.

        Same shape as `ExploreSpaceSobolQMC.save` writes --- one dict keyed
        by task name, each entry holding `points`, `values` and `outputs`
        --- so the exploration file and every optimization file since can be
        handed to the next `OptimizeSpaceBotorch` as one list.

        Overwrites `path`. Saving before anything has been evaluated writes
        each task's three empty lists rather than raising: the file says
        what this run measured, and that is nothing yet.
        """
        results = {
            name: {
                "points": result.points,
                "values": result.values,
                "outputs": result.outputs,
            }
            for name, result in self.results.items()
        }
        with gzip.open(path, "wb") as fobj:
            pickle.dump(results, fobj, protocol=pickle.HIGHEST_PROTOCOL)
