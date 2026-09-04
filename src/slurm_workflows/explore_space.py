"""Sobol' QMC exploration of search spaces.

Draws a low-discrepancy design over each `SearchSpace` it is given,
evaluates every point of every design across a pilot pool,
and keeps what came back.

This is the exploration phase of `OptimizeSpaceBotorch` standing on its own,
for a sweep that is not going to be followed by a model:
a first look at a space,
a baseline to compare a search against,
or a set of points to hand to something else.
The Sobol' engine is scipy's rather than botorch's,
so exploring a space needs neither torch nor botorch installed.

Several spaces are explored at once rather than one after another.
Each is an `ExplorationTask`, with its own space, objective and queue,
and one `ExploreSpaceSobolQMC` submits the whole lot before waiting for any
of it: a sweep of 16 points and a sweep of 512 fill the pool together
instead of the first leaving it idle.

Sobol' is used rather than a uniform random sample
because a low-discrepancy sequence covers the space more evenly
at the sample sizes a cluster run can afford:
a random design of 64 points leaves clumps and gaps that a Sobol' one does not.
"""

from __future__ import annotations

import gzip
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace

from scipy.stats import qmc

from .slurm_pilot_executor import SlurmPilotExecutor, RaiseOnError, Task
from .search_space import SearchSpace, space_dim, to_params, to_unit
from .utils import (
    RemoteExecutionError,
    floor_power_of_two,
    format_mapping,
    objective_value,
)

ObjectiveOutput = Mapping[str, Any]
ObjectiveFunction = Callable[..., ObjectiveOutput]


@dataclass
class ExplorationTask:
    """One space to explore, and everything needed to explore it.

    name: names this task. It labels the progress lines and keys the
        results, so no two tasks of one sweep may share one.
    space: the search space.
    objective: the function to evaluate.
        Its argument names must match the keys of `space`.
        It returns a mapping: the value of interest under `objective_key`,
        plus anything else worth recording about the evaluation.
        Only the value under `objective_key` is ranked,
        but the whole mapping is kept.
    objective_queue: queue(s) the evaluations are submitted to.
        Tasks may name different queues, and are still submitted together.
    num_exploration_points: number of points to sample.
        Truncated to the nearest lower power of two.
        Left as None, the count `ExploreSpaceSobolQMC` was constructed with
        is used instead.
    seed: seed for this task's design.
        Left as None, one is drawn from os.urandom and printed,
        so the run can still be repeated afterwards.
    objective_key: the key in the objective's result
        holding the value to rank points by.
        Set it when the objective already reports under another name,
        such as "loss" or "rmse".
        Every other key is recorded and not ranked.
        Lower is better, as in `OptimizeSpaceBotorch`:
        negate a score you would rather maximize.
    extra_objective_kwargs: extra keyword arguments for the objective.
        Forwarded verbatim, and may not shadow a parameter of the space.
    """

    name: str
    space: SearchSpace
    objective: ObjectiveFunction
    objective_queue: str | list[str]
    num_exploration_points: int | None = None
    seed: int | None = None
    objective_key: str = "objective"
    extra_objective_kwargs: dict = field(default_factory=dict)


@dataclass
class SavedResults:
    """Exactly what a results file holds for one task.

    `unit_points` is not in the file:
    it is derivable from the points and the space,
    and only whoever holds the space can derive it,
    so a reader recomputes it rather than trusting a stored copy
    that may have been written against another space.
    """

    points: list[dict[str, Any]] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)


def load_results(paths: Iterable[Path | str]) -> dict[str, SavedResults]:
    """Read back results files, merged by task name in the order given.

    Reads what `ExploreSpaceSobolQMC.save` and `OptimizeSpaceBotorch.save`
    write: a gzipped pickle of one dict keyed by task name,
    each entry holding `points`, `values` and `outputs`.

    Merging is what makes a run resumable:
    the exploration file and every optimization file since
    are one argument list, and a task's observations are their concatenation.
    """
    merged: dict[str, SavedResults] = {}

    for path in paths:
        with gzip.open(path, "rb") as fobj:
            loaded = pickle.load(fobj)

        if not isinstance(loaded, Mapping):
            raise ValueError(f"{path}: expected a mapping of task name to results")

        for name, results in loaded.items():
            if not isinstance(results, Mapping) or not {
                "points",
                "values",
                "outputs",
            } <= set(results):
                raise ValueError(
                    f"{path}: {name!r} does not hold points, values and outputs"
                )

            lengths = {len(results[key]) for key in ("points", "values", "outputs")}
            if len(lengths) != 1:
                raise ValueError(
                    f"{path}: {name!r} holds lists of different lengths, "
                    "so its points, values and outputs do not line up"
                )

            saved = merged.setdefault(name, SavedResults())
            saved.points.extend(results["points"])
            saved.values.extend(results["values"])
            saved.outputs.extend(results["outputs"])

    return merged


@dataclass
class ExplorationResult:
    """What one task measured, in submission order.

    The four lists are index-aligned:
    `points[i]` was evaluated, returned `outputs[i]`,
    was ranked by `values[i]`, and sits at `unit_points[i]` in the unit cube.

    `unit_points` holds the standardized coordinates of the point the
    objective actually ran at, after rounding rather than the drawn
    coordinate, so what is recorded is always a location that was evaluated.
    """

    points: list[dict[str, Any]] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    # The objective's whole result, not just the number ranked from it.
    outputs: list[dict[str, Any]] = field(default_factory=list)
    unit_points: list[list[float]] = field(default_factory=list)


class ExploreSpaceSobolQMC:
    """Sobol' QMC sweeps of one or more search spaces, run together."""

    def __init__(
        self,
        tasks: list[ExplorationTask],
        executor: SlurmPilotExecutor,
        num_exploration_points: int | None = None,
    ):
        """Initialize.

        tasks: the spaces to explore, one `ExplorationTask` each.
            They run together rather than one after another,
            so a small sweep does not hold the pool
            while a large one waits its turn.
        executor: executor for parallelizing objective execution.
        num_exploration_points: point count for tasks that do not carry
            their own. A task with neither is an error rather than a guess.

        Every task is checked here rather than when it runs,
        so a misspelled keyword or an empty space
        is reported before anything reaches the cluster.
        The tasks are copied, with the point count and the seed filled in,
        so `self.tasks` says what will actually be run
        and the caller's own objects are left alone.
        """
        if not tasks:
            raise ValueError("no exploration tasks given")

        names = [task.name for task in tasks]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"exploration task names must be unique: {duplicates}")

        self.executor = executor
        self.tasks = [self._resolve(task, num_exploration_points) for task in tasks]
        self.results: dict[str, ExplorationResult] = {
            task.name: ExplorationResult() for task in self.tasks
        }

    @staticmethod
    def _resolve(task: ExplorationTask, default_points: int | None) -> ExplorationTask:
        """Validate one task and fill in what it left to the sweep."""
        if not task.space:
            raise ValueError(f"{task.name}: search space is empty")

        overlap = set(task.extra_objective_kwargs) & set(task.space)
        if overlap:
            raise ValueError(
                f"{task.name}: extra_objective_kwargs may not shadow search "
                f"space parameters: {sorted(overlap)}"
            )

        points = task.num_exploration_points
        if points is None:
            points = default_points
        if points is None:
            raise ValueError(
                f"{task.name}: no num_exploration_points, on the task or on "
                "the sweep"
            )

        seed = task.seed
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "big")
            print(
                f"{task.name}: no seed given, drew {seed} "
                f"--- pass it back to repeat this run",
                flush=True,
            )

        # Sobol' is only balanced on power-of-two prefixes of the sequence;
        # truncating is what keeps the design low-discrepancy.
        return replace(
            task,
            space=dict(task.space),
            num_exploration_points=floor_power_of_two(points),
            seed=seed,
        )

    def _task(self, name: str) -> ExplorationTask:
        """The named task, or a `KeyError` naming the ones there are."""
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(
            f"no exploration task named {name!r}; have {sorted(self.results)}"
        )

    def dim(self, name: str) -> int:
        """Dimensionality of a task's search space."""
        return space_dim(self._task(name).space)

    def design(self, name: str) -> list[dict[str, Any]]:
        """The points a task will evaluate, without evaluating them.

        Drawing is reproducible: the same seed redraws the same design,
        so this is also how to see what a run will do before it runs.
        """
        task = self._task(name)
        assert task.num_exploration_points is not None  # filled in by _resolve
        assert task.seed is not None

        # `random_base2` rather than `random`, which warns that a count
        # that is not a power of two loses the balance property.
        # The count is already floored to one, so it is the same draw
        # asked for in the terms scipy states it in.
        engine = qmc.Sobol(d=space_dim(task.space), scramble=True, rng=task.seed)
        design = engine.random_base2(m=task.num_exploration_points.bit_length() - 1)
        return [to_params(task.space, row.tolist()) for row in design]

    def run_exploration_jobs(self) -> None:
        """Evaluate every task's design, all of them in one batch.

        Everything is submitted before anything is waited for,
        so the tasks share the pool rather than queueing behind each other,
        and a task with its own queue is served by that queue's workers
        while the others run elsewhere.

        Blocks until every point of every task is back.
        Calling it a second time draws the same designs again,
        so a repeated call re-evaluates the same points
        rather than exploring further.
        """
        submitted: list[tuple[ExplorationTask, dict[str, Any], Task]] = []
        for task in self.tasks:
            for params in self.design(task.name):
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
            self._wait(submitted)
        except RuntimeError:
            # Keep what did come back before reporting the failure.
            # A sweep of a few thousand points must not lose all of them
            # to one that failed, since `save()` is what the next run reads.
            self._record_returned(submitted)
            raise

        for task, params, submission in submitted:
            self._record(task, params, submission)

        for task in self.tasks:
            self._report_best(task.name)

    def _wait(
        self, submitted: list[tuple[ExplorationTask, dict[str, Any], Task]]
    ) -> None:
        """Wait for the whole batch, and say which tasks failed if it did.

        Deferred rather than immediate:
        one failed evaluation should not hide the rest of the sweep,
        and with several spaces in flight
        the interesting question is which of them broke.
        """
        try:
            self.executor.wait(
                [submission for _, _, submission in submitted],
                desc="explore",
                unit="point",
                raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED,
            )
        except RuntimeError as e:
            failed = sorted(
                {
                    task.name
                    for task, _, submission in submitted
                    if isinstance(submission.output, RemoteExecutionError)
                }
            )
            # Empty when nothing came back at all, a canceled task say,
            # in which case the cause is in the exception this chains to.
            named = f" of {failed}" if failed else ""
            raise RuntimeError(
                f"objective evaluations failed during exploration{named}"
            ) from e

    def _record_returned(
        self, submitted: list[tuple[ExplorationTask, dict[str, Any], Task]]
    ) -> None:
        """Record every evaluation that did come back, ignoring the rest.

        For the failure path only, where an exception is already on its way:
        a task that failed, or returned something unusable, is skipped
        rather than raising over the report of what went wrong.
        """
        for task, params, submission in submitted:
            try:
                self._record(task, params, submission)
            except RuntimeError:
                continue

    def _record(
        self, task: ExplorationTask, params: dict[str, Any], submission: Task
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

    def _report_best(self, name: str) -> None:
        """Print the best point one task measured."""
        best = self._best_index(name)
        result = self.results[name]
        params = format_mapping(result.points[best])
        # The whole result, not just the ranked value.
        output = format_mapping(result.outputs[best])
        print(
            f"{name}: best of {len(result.values)} points " f"at {params} -> {output}",
            flush=True,
        )

    def _best_index(self, name: str) -> int:
        """Index of the lowest objective value one task has seen."""
        values = self.results[self._task(name).name].values
        if not values:
            raise RuntimeError(f"{name}: nothing has been evaluated yet")

        return min(range(len(values)), key=values.__getitem__)

    def best_point(self, name: str) -> tuple[dict[str, Any], float]:
        """Returns a task's best point (params, objective value) so far."""
        best = self._best_index(name)
        result = self.results[name]
        return dict(result.points[best]), result.values[best]

    def best_output(self, name: str) -> dict[str, Any]:
        """The objective's whole result at a task's best point so far."""
        return dict(self.results[name].outputs[self._best_index(name)])

    def save(self, path: Path | str) -> None:
        """Write what every task measured to a gzipped pickle.

        The file holds one dict keyed by task name,
        each entry holding `points`, `values` and `outputs`
        under those names, each a list in submission order and index-aligned
        with the other two.
        Read it back with:

            with gzip.open(path, "rb") as fobj:
                results = pickle.load(fobj)

        Plain `pickle`, not `cloudpickle`:
        this is the measurements, not the code that produced them,
        so anything an objective returns that a plain unpickler
        cannot reconstruct does not belong in the file.
        Gzipped because an exploration of a few thousand points
        is a few thousand near-identical small mappings,
        which is what compresses well.

        Overwrites `path`. Saving before anything has been evaluated
        writes each task's three empty lists rather than raising:
        the file says what the sweep measured, and that is nothing yet.
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
