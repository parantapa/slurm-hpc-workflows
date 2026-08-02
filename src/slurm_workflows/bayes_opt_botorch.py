"""A botorch based parallel bayesian optimizer."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Any, Mapping, Sequence

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from .slurm_pilot_executor import SlurmPilotExecutor, check_for_error

# Botorch recommendation:
# single precision makes the GP fits
# and the acquisition optimization numerically fragile.
DTYPE = torch.double


@dataclass
class IntRange:
    """Integer range."""

    min: int
    max: int

    def __post_init__(self) -> None:
        if self.max <= self.min:
            raise ValueError(f"IntRange needs max > min, got {self.min}, {self.max}")

    def standardize(self, x: int) -> float:
        """Move from [min, max] range to [0, 1] range."""
        return (x - self.min) / (self.max - self.min)

    def unstandardize(self, y: float) -> int:
        """Move from [0, 1] range to nearest integer in [min, max]."""
        x = round(self.min + y * (self.max - self.min))
        return int(min(max(x, self.min), self.max))


@dataclass
class FloatRange:
    """Floating point range.

    if log_range is true,
    the space is first scaled into a logarithmic space,
    the sample is generated in that space
    and then scaled back.
    """

    min: float
    max: float
    log_range: bool = False

    def __post_init__(self) -> None:
        if self.max <= self.min:
            raise ValueError(f"FloatRange needs max > min, got {self.min}, {self.max}")
        if self.log_range and self.min <= 0.0:
            raise ValueError(f"log_range needs min > 0, got {self.min}")

    def standardize(self, x: float) -> float:
        """Move from [min, max] range to [0, 1] range."""
        if self.log_range:
            lo, hi, x = math.log(self.min), math.log(self.max), math.log(x)
        else:
            lo, hi = self.min, self.max
        return (x - lo) / (hi - lo)

    def unstandardize(self, y: float) -> float:
        """Move from [0, 1] range to [min, max] range."""
        y = min(max(y, 0.0), 1.0)
        if self.log_range:
            lo, hi = math.log(self.min), math.log(self.max)
            return math.exp(lo + y * (hi - lo))
        return self.min + y * (self.max - self.min)


@dataclass
class CategoricalRange:
    """Categorical range."""

    num_categories: int

    def __post_init__(self) -> None:
        if self.num_categories < 1:
            raise ValueError(f"num_categories must be >= 1, got {self.num_categories}")

    def standardize(self, x: int) -> float:
        """Move from [0, num_categories -1] range to [0, 1] range."""
        if self.num_categories == 1:
            return 0.0
        return x / (self.num_categories - 1)

    def unstandardize(self, y: float) -> int:
        """Move from [0, 1] range to [0, num_categories - 1] range."""
        if self.num_categories == 1:
            return 0
        i = round(y * (self.num_categories - 1))
        return int(min(max(i, 0), self.num_categories - 1))


ParameterRange = IntRange | FloatRange | CategoricalRange

SearchSpace = Mapping[str, ParameterRange]

ObjectiveOutput = Mapping[str, Any]
ObjectiveFunction = Callable[..., ObjectiveOutput]


def _format_param(value: Any) -> str:
    """Render one value for a progress line.

    Floats get a fixed precision
    so the columns do not jump around between rounds;
    everything else prints as itself,
    since an objective's result may carry values of any type.
    """
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _format_mapping(mapping: Mapping[str, Any]) -> str:
    """Render a whole mapping for a progress line."""
    return ", ".join(f"{k}={_format_param(v)}" for k, v in mapping.items())


def floor_power_of_two(n: int) -> int:
    """Largest power of two <= n."""
    if n < 1:
        raise ValueError(f"expected a positive integer, got {n}")
    return 1 << (n.bit_length() - 1)


# The keys `fit_and_propose` returns,
# read back by the driver in `_fit_and_propose`.
CANDIDATES_KEY = "candidates"
FIT_SECONDS_KEY = "fit_seconds"
PROPOSE_SECONDS_KEY = "propose_seconds"


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

    # `batch` points jointly, not `batch` independent optima:
    # the batch acquisition keeps the parallel proposals
    # from stacking on one spot.
    # Optimized jointly rather than greedily one at a time.
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


class BayesOptBotorch:
    """A botorch based parallel bayesian optimzier.

    Integer and categorical parameters are handled
    by rounding a continuous proposal,
    so distinct proposals can collapse onto the same point.
    The GP's learned noise term absorbs the resulting repeated observations;
    on a mostly-discrete space with few levels,
    expect the search to re-evaluate points it has already seen.
    """

    def __init__(
        self,
        name: str,
        space: SearchSpace,
        objective: ObjectiveFunction,
        executor: SlurmPilotExecutor,
        objective_queue: str | list[str],
        optimizer_queue: str | list[str],
        num_exploration_points: int,
        search_parallelism: int,
        seed: int | None = None,
        /,
        *,
        min_search_iterations: int = 5,
        max_search_iterations: int = 30,
        patience: int = 3,
        min_improvement: float = 0.05,
        objective_key: str = "objective",
        num_restarts: int = 10,
        raw_samples: int = 128,
        mc_samples: int = 128,
        acqf_timeout_s: float = 10.0,
        **extra_objective_kwargs,
    ):
        """Initialize.

        name: name of the optimization job.
        space: search space.
        objective: objective function to minimize.
            Argument names in the objective function must match those in search space.
            It returns a mapping:
            the value to minimize under `objective_key`,
            plus anything else worth recording about the evaluation.
            The optimizer models only the objective value
            but keeps the whole mapping in `outputs`.
        executor: executor for parallelizing objective execution.
        objective_queue: queue(s) the objective evaluations are submitted to.
        optimizer_queue: queue(s) the model fit
            and acquisition optimization are submitted to,
            one task per search round.
        num_exploration_points: number of initial points to sample
            using Sobol QMC method.
            if not a power of two
            it will be truncated to the nearest lower power of two.
        search_parallelism: number of points evaluated per search round.
        seed: seed for the exploration design.
            Omit it to draw one from os.urandom.

        Everything up to and including the seed is positional-only,
        which is what leaves those names free for the objective:
        an objective that takes its own `seed` or `name`
        still receives it through extra_objective_kwargs.
        Everything after is keyword-only.

        The search runs between min_search_iterations
        and max_search_iterations rounds,
        stopping early when it stops paying:

        min_search_iterations: rounds always run,
            whatever they achieve.
            Stalled rounds below it still count towards patience,
            they just cannot be the round that ends the search,
            so a search that never improves runs exactly this many rounds
            and the earliest possible stop is max(min_search_iterations, patience).
        max_search_iterations: hard ceiling;
            the search stops here even if it is still improving.
        patience: consecutive stalled rounds that end the search.
            A round that improves resets the count,
            so this bounds a run of bad rounds, not their total.
        min_improvement: fractional improvement in the best value
            that a round must deliver to count as improving.
            0.05 asks each round to beat the incumbent by 5%,
            measured relative to its magnitude.

        objective_key: the key in the objective's result
            holding the value to minimize.
            Worth changing when the objective is shared with something else
            --- an evaluation that already reports "loss" or "rmse"
            can be searched as it is,
            rather than wrapped to rename one of its keys.
            Every other key is recorded and not modelled, as usual.

        The rest tune the fit and the acquisition optimization.

        num_restarts: multi-start count for `optimize_acqf`.
            The acquisition surface is multimodal,
            so a single start routinely lands in a local optimum.
        raw_samples: candidates `optimize_acqf` draws
            to pick those starting points from.
        mc_samples: quasi-MC draws used to estimate the acquisition value
            at a candidate.
            Sobol' draws are stratified,
            so they carry further
            than the same number of independent normal samples.
        acqf_timeout_s: wall-clock budget for one `optimize_acqf` call.
            Proposal cost grows with the number of observations,
            so an unbounded search can end up spending longer
            choosing the next batch
            than the batch takes to evaluate.
            Hitting the limit is not an error:
            `optimize_acqf` returns the best candidates it has found so far
            --- still a full batch, still finite and inside the bounds ---
            just less thoroughly optimized.
            A slightly worse proposal costs one round;
            a stalled driver costs the run.

        extra_objective_kwargs: extra keyword arguments to pass to objective function.
            These are forwarded verbatim,
            so a misspelled early-stopping argument lands here
            and fails when the objective rejects it.
        """

        if not space:
            raise ValueError("search space is empty")
        if min_search_iterations < 0:
            raise ValueError(
                f"min_search_iterations must be >= 0, got {min_search_iterations}"
            )
        if max_search_iterations < min_search_iterations:
            raise ValueError(
                f"max_search_iterations must be >= min_search_iterations, got "
                f"{max_search_iterations} < {min_search_iterations}"
            )
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_improvement < 0.0:
            raise ValueError(f"min_improvement must be >= 0, got {min_improvement}")
        if search_parallelism < 1:
            raise ValueError(
                f"search_parallelism must be >= 1, got {search_parallelism}"
            )
        if num_restarts < 1:
            raise ValueError(f"num_restarts must be >= 1, got {num_restarts}")
        if raw_samples < 1:
            raise ValueError(f"raw_samples must be >= 1, got {raw_samples}")
        if mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1, got {mc_samples}")
        if acqf_timeout_s <= 0.0:
            raise ValueError(f"acqf_timeout_s must be > 0, got {acqf_timeout_s}")

        overlap = set(extra_objective_kwargs) & set(space)
        if overlap:
            raise ValueError(
                "extra_objective_kwargs may not shadow search space parameters: "
                f"{sorted(overlap)}"
            )

        self.name = name
        self.space: SearchSpace = dict(space)
        self.param_names: list[str] = list(self.space)
        self.objective = objective
        self.executor = executor
        self.objective_queue = objective_queue
        self.optimizer_queue = optimizer_queue

        # Sobol' is only balanced on power-of-two prefixes of the sequence;
        # truncating is what keeps the exploration phase low-discrepancy.
        self.num_exploration_points = floor_power_of_two(num_exploration_points)
        self.search_parallelism = search_parallelism
        self.min_search_iterations = min_search_iterations
        self.max_search_iterations = max_search_iterations
        self.patience = patience
        self.min_improvement = min_improvement
        self.objective_key = objective_key
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.mc_samples = mc_samples
        self.acqf_timeout_s = acqf_timeout_s
        self.extra_objective_kwargs = extra_objective_kwargs

        if seed is None:
            # 63 bits: torch's SobolEngine unpacks the seed
            # as a signed long long
            # and overflows on anything wider.
            seed = int.from_bytes(os.urandom(8), "big") >> 1
            print(
                f"{name}: no seed given, drew {seed} "
                f"--- pass it back to repeat this run",
                flush=True,
            )

        self.seed = seed

        # Everything evaluated so far, in submission order.
        # `unit_points` holds the standardized coordinates of the point
        # the objective *actually* ran at --
        # after rounding, not the continuous proposal --
        # so the model is never told about a location that was never evaluated.
        self.points: list[dict[str, Any]] = []
        self.values: list[float] = []
        # The objective's whole result, not just the number modelled from it.
        self.outputs: list[dict[str, Any]] = []
        self.unit_points: list[list[float]] = []

    @property
    def dim(self) -> int:
        """Dimensionality of the search space."""
        return len(self.param_names)

    def _to_params(self, unit: Sequence[float]) -> dict[str, Any]:
        """Unit cube coordinates -> objective keyword arguments."""
        return {
            name: self.space[name].unstandardize(float(u))
            for name, u in zip(self.param_names, unit)
        }

    def _to_unit(self, params: dict[str, Any]) -> list[float]:
        """Objective keyword arguments -> unit cube coordinates."""
        return [self.space[name].standardize(params[name]) for name in self.param_names]

    def _evaluate(self, points: list[dict[str, Any]], desc: str) -> None:
        """Evaluate `points` on the executor and record the results."""
        tasks = [
            self.executor.submit(
                self.objective_queue,
                self.objective,
                **params,
                **self.extra_objective_kwargs,
            )
            for params in points
        ]
        self.executor.wait(tasks, desc=f"{self.name}:{desc}", unit="point")

        failed = check_for_error(tasks)
        if failed:
            raise RuntimeError(
                f"{self.name}: {len(failed)} of {len(tasks)} objective "
                f"evaluations failed during {desc}"
            )

        for params, task in zip(points, tasks):
            output = task.output

            if not isinstance(output, Mapping):
                raise RuntimeError(
                    f"{self.name}: objective returned {output!r} at {params}; "
                    f"expected a mapping carrying an {self.objective_key!r} key"
                )
            if self.objective_key not in output:
                raise RuntimeError(
                    f"{self.name}: objective returned keys {sorted(output)} "
                    f"at {params}, with no {self.objective_key!r} among them"
                )

            try:
                value = float(output[self.objective_key])
            except (TypeError, ValueError) as e:
                raise RuntimeError(
                    f"{self.name}: {self.objective_key!r} was "
                    f"{output[self.objective_key]!r} at {params}, "
                    "which is not a float"
                ) from e
            if not math.isfinite(value):
                raise RuntimeError(
                    f"{self.name}: {self.objective_key!r} was {value} "
                    f"at {params}; the GP cannot be fit on non-finite values"
                )

            self.points.append(params)
            self.values.append(value)
            # Copied, so a later mutation of the returned mapping
            # cannot rewrite what the run recorded.
            self.outputs.append(dict(output))
            self.unit_points.append(self._to_unit(params))

    def run_exploration_jobs(self):
        """Run exploration jobs.

        Samples num_exploration_points to explore using Sobol QMC method.
        Then submits all of them to the executor for evaluation.
        Waits until all the tasks are finished.
        """
        engine = torch.quasirandom.SobolEngine(
            dimension=self.dim, scramble=True, seed=self.seed
        )
        design = engine.draw(self.num_exploration_points, dtype=DTYPE)
        points = [self._to_params(row.tolist()) for row in design]
        self._evaluate(points, "explore")
        self._report_best()

    def _fit_and_propose(self, batch: int, desc: str) -> list[list[float]]:
        """Fit the GP and get `batch` unit cube points, on the optimizer queue."""
        print(
            f"{self.name}: fitting GP on {len(self.values)} points ...",
            flush=True,
        )

        task = self.executor.submit(
            self.optimizer_queue,
            fit_and_propose,
            list(self.unit_points),
            list(self.values),
            batch,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            mc_samples=self.mc_samples,
            timeout_s=self.acqf_timeout_s,
        )
        self.executor.wait([task], desc=f"{self.name}:{desc}", unit="fit")

        if check_for_error([task]):
            raise RuntimeError(
                f"{self.name}: fitting the model on queue "
                f"{self.optimizer_queue!r} failed during {desc} "
                f"--- {task.output}"
            )

        # Every key this reads below, not just the candidates:
        # a worker one version behind is exactly what this message is for,
        # and it would otherwise reach the timings as a bare KeyError.
        result = task.output
        expected = (CANDIDATES_KEY, FIT_SECONDS_KEY, PROPOSE_SECONDS_KEY)
        if isinstance(result, Mapping):
            missing = [key for key in expected if key not in result]
        else:
            missing = list(expected)
        if missing:
            raise RuntimeError(
                f"{self.name}: the optimizer queue returned {result!r} "
                f"during {desc}, with no "
                f"{', '.join(repr(key) for key in missing)} in it; "
                "check that its workers run the same slurm-workflows "
                "as this driver"
            )

        candidates = result[CANDIDATES_KEY]

        # Every round is the full width of the pool.
        # A short batch would quietly narrow the round instead,
        # which is a worse way to find out the far end disagrees.
        if len(candidates) != batch:
            raise RuntimeError(
                f"{self.name}: the optimizer queue proposed "
                f"{len(candidates)} points during {desc}, not the {batch} "
                "asked for"
            )

        print(
            f"{self.name}: GP fit took {result[FIT_SECONDS_KEY]:.2f}s, "
            f"proposed {len(candidates)} points in "
            f"{result[PROPOSE_SECONDS_KEY]:.2f}s",
            flush=True,
        )

        return candidates

    def run_search_jobs(self):
        """Run search jobs.

        At each round:
            * First submits one task to the optimizer queue,
              which trains a SingleTaskGp model using known results
              and samples search_parallelism points jointly
              using the qLogNoisyExpectedImprovement acquisition.
            * Submits those search_parallelism points
              to the objective queue for evaluation.
            * Waits until the jobs are complete.

        Every round is the full width of the pool,
        so a round always costs search_parallelism evaluations.

        Rounds run until either the search stops paying
        --- `patience` consecutive rounds that fail to improve the best value
        by `min_improvement` --- or `max_search_iterations` is reached.
        Stalled rounds are counted from the first round,
        including the ones below `min_search_iterations`:
        the floor holds off the *stop*, not the counting,
        so the earliest possible stop is max(`min_search_iterations`, `patience`).
        """
        if not self.values:
            raise RuntimeError(
                f"{self.name}: no observations to model; "
                "run_exploration_jobs() must run first"
            )

        stalled = 0
        for iteration in range(1, self.max_search_iterations + 1):
            previous_best = min(self.values)
            desc = f"search {iteration}/{self.max_search_iterations}"

            candidates = self._fit_and_propose(self.search_parallelism, desc)
            points = [self._to_params(candidate) for candidate in candidates]

            self._evaluate(points, desc)
            self._report_best()

            if self._improved_enough(previous_best, min(self.values)):
                stalled = 0
                continue

            stalled += 1

            # Two things hold the search open,
            # and a stalled round ends it only once both have given way:
            # the streak has to reach `patience`,
            # and the round count has to reach the floor.
            # Reporting the larger of the two gaps
            # is what keeps the progress line honest
            # when the floor outlasts the streak
            # --- a bare `stalled`/`patience` ratio runs past its own
            # denominator, and says the search should have stopped rounds ago.
            remaining = max(
                self.patience - stalled,
                self.min_search_iterations - iteration,
            )

            if remaining > 0:
                print(
                    f"{self.name}: round {iteration} improved by less than "
                    f"{self.min_improvement:.0%} "
                    f"--- {stalled} in a row, {remaining} more to stop",
                    flush=True,
                )
                continue

            print(
                f"{self.name}: stopping after {iteration} rounds "
                f"--- {stalled} in a row without a "
                f"{self.min_improvement:.0%} improvement",
                flush=True,
            )
            return

    def _improved_enough(self, previous: float, current: float) -> bool:
        """Whether `current` beats `previous` by at least `min_improvement`.

        The threshold is fractional,
        measured against the magnitude of the incumbent,
        so it means the same thing
        whether the objective is scaled in seconds or in nanoseconds.

        A negative incumbent works the same way
        --- going from -10 to -11 is a 10% improvement ---
        but an incumbent of exactly zero
        has no magnitude to take a fraction of,
        so there any strict decrease counts.
        """
        if current >= previous:
            return False

        magnitude = abs(previous)
        if magnitude == 0.0:
            return True

        return (previous - current) / magnitude >= self.min_improvement

    def _report_best(self) -> None:
        """Print the best point measured so far.

        Printed after every batch comes back
        so a long search shows whether it is still improving,
        which is the thing you actually want to watch:
        several rounds with an unchanged best
        say the budget is being spent without buying anything.
        """
        best = self._best_index()
        params = _format_mapping(self.points[best])
        # The whole result, not just the objective value:
        # whatever else the evaluation reported is usually the thing
        # that explains *why* this point is winning.
        output = _format_mapping(self.outputs[best])
        print(
            f"{self.name}: best after {len(self.values)} points "
            f"at {params} -> {output}",
            flush=True,
        )

    def _best_index(self) -> int:
        """Index of the lowest objective value seen so far."""
        if not self.values:
            raise RuntimeError(f"{self.name}: nothing has been evaluated yet")

        return min(range(len(self.values)), key=self.values.__getitem__)

    def best_point(self) -> tuple[dict[str, Any], float]:
        """Returns the best point (params, objective value) known so far."""
        best = self._best_index()
        return dict(self.points[best]), self.values[best]

    def best_output(self) -> dict[str, Any]:
        """The objective's whole result at the best point known so far."""
        return dict(self.outputs[self._best_index()])
