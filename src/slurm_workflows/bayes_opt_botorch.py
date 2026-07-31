"""A botorch based parallel bayesian optimizer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Any, Mapping, Sequence

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogExpectedImprovement, qProbabilityOfImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

from .slurm_pilot_executor import SlurmPilotExecutor, check_for_error

# Botorch recommendation:
# single precision makes the GP fits and the acquisition optimization
# numerically fragile.
DTYPE = torch.double

# Multi-start settings for `optimize_acqf`.
# The acquisition surface is multimodal,
# so a single start routinely lands in a local optimum.
NUM_RESTARTS: int = 10
RAW_SAMPLES: int = 512


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

    if log_range is true, the space is first scaled into a logarithmic space,
    the sample is generated in that space and then scaled back.
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

# Mapping, not dict: dict's value type is invariant,
# so a plain `{"x": FloatRange(...)}` --- inferred as dict[str, FloatRange] ---
# would not satisfy dict[str, ParameterRange],
# and every caller would have to annotate its search space.
SearchSpace = Mapping[str, ParameterRange]

ObjectiveFunction = Callable[..., float]


def floor_power_of_two(n: int) -> int:
    """Largest power of two <= n."""
    if n < 1:
        raise ValueError(f"expected a positive integer, got {n}")
    return 1 << (n.bit_length() - 1)


class BayesOptBotorch:
    """A botorch based parallel bayesian optimzier.

    The optimizer works entirely in the unit cube:
    every parameter is mapped into [0, 1] by its `ParameterRange`
    before the GP sees it,
    and candidates come back out through `unstandardize`.

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
        queue: str | list[str],
        num_exploration_points: int,
        num_search_points: int,
        search_parallelism: int,
        seed: int,
        /,
        **extra_objective_kwargs,
    ):
        """Initialize.

        name: name of the optimization job.
        space: search space.
        objective: objective function to minimize.
            Argument names in the objective function must match those in search space.
        executor: executor for parallelizing objective execution.
        queue: queue(s) to use to submit job to executor.
        num_exploration_points: number of initial points to sample using Sobol QMC method.
            if not a power of two it will be truncated to the nearest lower power of two.
        num_search_points: number of total points to search using bayesian optimization.
        search_parallelism: number of points to explore in parallel.
        extra_objective_kwargs: extra keyword arguments to pass to objective function.
        """

        if not space:
            raise ValueError("search space is empty")
        if num_search_points < 0:
            raise ValueError(f"num_search_points must be >= 0, got {num_search_points}")
        if search_parallelism < 1:
            raise ValueError(
                f"search_parallelism must be >= 1, got {search_parallelism}"
            )

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
        self.queue = queue

        # Sobol' is only balanced on power-of-two prefixes of the sequence;
        # truncating is what keeps the exploration phase low-discrepancy.
        self.num_exploration_points = floor_power_of_two(num_exploration_points)
        self.num_search_points = num_search_points
        self.search_parallelism = search_parallelism
        self.extra_objective_kwargs = extra_objective_kwargs

        self.seed = seed

        # Everything evaluated so far, in submission order.
        # `unit_points` holds the standardized coordinates of the point
        # the objective *actually* ran at --
        # after rounding, not the continuous proposal --
        # so the model is never told about a location that was never evaluated.
        self.points: list[dict[str, Any]] = []
        self.values: list[float] = []
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
                self.queue, self.objective, **params, **self.extra_objective_kwargs
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
            try:
                value = float(task.output)
            except (TypeError, ValueError) as e:
                raise RuntimeError(
                    f"{self.name}: objective returned {task.output!r}, "
                    "which is not a float"
                ) from e
            if not math.isfinite(value):
                raise RuntimeError(
                    f"{self.name}: objective returned {value} at {params}; "
                    "the GP cannot be fit on non-finite values"
                )

            self.points.append(params)
            self.values.append(value)
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

    def _fit_model(self) -> SingleTaskGP:
        """Fit a SingleTaskGP to everything evaluated so far."""
        train_x = torch.tensor(self.unit_points, dtype=DTYPE)

        # Botorch maximizes, the objective is minimized:
        # the model is fit to -f,
        # and every acquisition value below is in that negated space too.
        train_y = torch.tensor([[-v] for v in self.values], dtype=DTYPE)

        model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def _propose(self, model: SingleTaskGP, batch: int) -> list[list[float]]:
        """Propose `batch` unit cube points, split over the two acquisitions."""
        # Odd batches give the extra point to qLogEI,
        # the better-behaved of the two.
        # qPI's improvement probability saturates
        # once the model is confident,
        # and stops discriminating between candidates.
        num_ei = batch - batch // 2
        num_pi = batch // 2

        best_f = max(-v for v in self.values)
        bounds = torch.stack(
            [torch.zeros(self.dim, dtype=DTYPE), torch.ones(self.dim, dtype=DTYPE)]
        )

        candidates: list[list[float]] = []
        for acqf_class, q in (
            (qLogExpectedImprovement, num_ei),
            (qProbabilityOfImprovement, num_pi),
        ):
            if q == 0:
                continue
            acqf = acqf_class(model, best_f=best_f)
            # q points jointly, not q independent optima:
            # the batch acquisition keeps the parallel proposals
            # from stacking on one spot.
            batch_candidates, _ = optimize_acqf(
                acqf,
                bounds=bounds,
                q=q,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
            candidates.extend(batch_candidates.tolist())

        return candidates

    def run_search_jobs(self):
        """Run search jobs.

        Iteratively evaluates a total of num_search_points to evaluate.
        At each step:
            * First trains a SingleTaskGp model using known results.
            * Samples search_parallelism / 2 points using qLogExpectedImprovement objective.
            * Samples search_parallelism / 2 points using qProbabilityOfImprovement objective.
            * Submits search_parallelism points to executor for evaluation.
            * Waits until the jobs are complete.
        """
        if not self.values:
            raise RuntimeError(
                f"{self.name}: no observations to model; "
                "run_exploration_jobs() must run first"
            )

        done = 0
        while done < self.num_search_points:
            # The last round is short rather than overshooting the budget.
            batch = min(self.search_parallelism, self.num_search_points - done)

            model = self._fit_model()
            candidates = self._propose(model, batch)
            points = [self._to_params(candidate) for candidate in candidates]

            done += batch
            self._evaluate(points, f"search {done}/{self.num_search_points}")

    def best_point(self) -> tuple[dict[str, Any], float]:
        """Returns the best point (params, objective value) known so far."""
        if not self.values:
            raise RuntimeError(f"{self.name}: nothing has been evaluated yet")

        best = min(range(len(self.values)), key=self.values.__getitem__)
        return dict(self.points[best]), self.values[best]
