"""Tests for the botorch-based parallel optimizer.

The optimizer's contract with the executor is two calls wide ---
`submit()` returning a `Task`, and `wait()` filling in that task's `output` ---
so most tests here drive that contract through `LocalExecutor`,
which runs the objective inline.
Fitting a GP and optimizing an acquisition function
is the expensive part of every one of these tests;
paying for a queue round trip on top would buy nothing,
because the optimizer cannot tell the difference.

`TestRealExecutor` is what keeps that stand-in honest:
it runs a whole optimization through the real executor,
the real ds-service queue and a real worker,
so the two-call contract is pinned against the thing it stands in for.

The four tests in `TestSearchBehaviour` assert *behaviour of the search*,
not just its bookkeeping.
They are the ones that would catch the sign of the objective being flipped
--- botorch maximizes and this optimizer minimizes.
They are stochastic
(torch's global RNG is left unseeded, so each run is a fresh sample),
and their thresholds come from measured spreads:
the monotone case put every search point below 0.1 against a threshold of 0.5,
and `test_search_beats_random_search` won 12/12 with a 4.6x margin.
All four use unimodal objectives on purpose
--- an earlier multimodal version of the random-search comparison
lost 1 run in 10.
"""

from __future__ import annotations

import math
import threading
from typing import cast

import pytest

pytest.importorskip("botorch")

import torch  # noqa: E402
from botorch.acquisition import (  # noqa: E402
    qLogExpectedImprovement,
    qProbabilityOfImprovement,
)

from slurm_workflows import bayes_opt_botorch as bo  # noqa: E402
from slurm_workflows.bayes_opt_botorch import (  # noqa: E402
    BayesOptBotorch,
    CategoricalRange,
    FloatRange,
    IntRange,
    floor_power_of_two,
)
from slurm_workflows.slurm_pilot_executor import (  # noqa: E402
    SlurmPilotExecutor,
    Task,
)
from slurm_workflows.utils import RemoteExecutionError, gen_error_id  # noqa: E402

from worker_harness import make_worker, run_worker  # noqa: E402

# --------------------------------------------------------------------------
# Test doubles and objectives
# --------------------------------------------------------------------------


class LocalExecutor:
    """Stands in for SlurmPilotExecutor, running each callable inline.

    Records what it was asked to submit
    so tests can assert on the queue
    and on the keyword arguments the objective was called with.
    A raising objective comes back as a `RemoteExecutionError` output,
    exactly as a real worker reports it.
    """

    def __init__(self) -> None:
        self.queues: list[str | list[str]] = []
        self.kwargs: list[dict] = []
        self.waits: list[str | None] = []

    def submit(self, queue, fn, *args, **kwargs) -> Task:
        self.queues.append(queue)
        self.kwargs.append(dict(kwargs))
        try:
            output = fn(*args, **kwargs)
        except Exception as e:
            output = RemoteExecutionError(str(e), gen_error_id())
        return Task(
            task_id=str(len(self.queues)),
            queue=[queue] if isinstance(queue, str) else list(queue),
            priority=0.0,
            function=fn,
            input=(args, kwargs),
            output=output,
        )

    def wait(self, tasks, desc=None, unit="task") -> None:
        self.waits.append(desc)

    @property
    def num_submitted(self) -> int:
        return len(self.queues)


def as_executor(executor: LocalExecutor) -> SlurmPilotExecutor:
    """Type the stand-in as the executor it stands in for.

    `LocalExecutor` satisfies the whole contract the optimizer uses
    --- `submit()` returning a `Task`, `wait()` filling in its `output` ---
    but does not inherit from `SlurmPilotExecutor`,
    whose `__init__` would open a real queue connection.
    The cast is the assertion that the two-call contract is all that is needed;
    `TestRealExecutor` is what proves it.
    """
    return cast(SlurmPilotExecutor, executor)


def sphere(x, y):
    """Convex, minimum f = 0 at the origin."""
    return x * x + y * y


def identity(x):
    """Monotone: the minimum is at the low edge of the box."""
    return x


BOX_2D = {"x": FloatRange(-5.0, 5.0), "y": FloatRange(-5.0, 5.0)}


SEED = 20260730


def make_opt(
    objective=sphere, space=None, explore=8, search=8, parallel=4, seed=SEED, **extra
):
    """An optimizer wired to a fresh LocalExecutor.

    `seed` is named explicitly
    so it is not swallowed by `**extra`, which goes to the objective.
    """
    space = BOX_2D if space is None else space
    executor = LocalExecutor()
    opt = BayesOptBotorch(
        "test",
        space,
        objective,
        as_executor(executor),
        "cpu",
        explore,
        search,
        parallel,
        seed,
        **extra,
    )
    return opt, executor


# --------------------------------------------------------------------------
# Parameter ranges
# --------------------------------------------------------------------------


class TestIntRange:
    def test_endpoints_map_to_the_unit_interval(self):
        r = IntRange(3, 11)
        assert r.standardize(3) == 0.0
        assert r.standardize(11) == 1.0
        assert r.unstandardize(0.0) == 3
        assert r.unstandardize(1.0) == 11

    def test_every_integer_round_trips(self):
        r = IntRange(-4, 7)
        for x in range(-4, 8):
            assert r.unstandardize(r.standardize(x)) == x

    def test_returns_an_int_not_a_float(self):
        # The value is handed to the objective as a keyword argument;
        # a 3.0 where the objective expects 3 is a bug the caller has to debug.
        value = IntRange(0, 10).unstandardize(0.5)
        assert isinstance(value, int)

    def test_rounds_to_nearest(self):
        r = IntRange(0, 10)
        assert r.unstandardize(0.44) == 4
        assert r.unstandardize(0.46) == 5

    def test_clamps_outside_the_unit_interval(self):
        # optimize_acqf can return a point a hair outside the bounds.
        r = IntRange(3, 11)
        assert r.unstandardize(-0.4) == 3
        assert r.unstandardize(1.7) == 11

    def test_rejects_a_degenerate_range(self):
        with pytest.raises(ValueError):
            IntRange(5, 5)
        with pytest.raises(ValueError):
            IntRange(5, 4)


class TestFloatRange:
    def test_linear_mapping(self):
        r = FloatRange(-5.0, 5.0)
        assert r.standardize(0.0) == 0.5
        assert r.unstandardize(0.5) == 0.0
        assert r.unstandardize(0.0) == -5.0
        assert r.unstandardize(1.0) == 5.0

    def test_linear_round_trip(self):
        r = FloatRange(-5.0, 5.0)
        for y in (0.0, 0.13, 0.5, 0.87, 1.0):
            assert math.isclose(r.standardize(r.unstandardize(y)), y)

    def test_log_range_midpoint_is_the_geometric_mean(self):
        # The point of log_range:
        # half the budget goes to each decade,
        # not to each half of the interval.
        r = FloatRange(1e-4, 1e-1, log_range=True)
        assert math.isclose(r.unstandardize(0.5), math.sqrt(1e-4 * 1e-1))
        assert math.isclose(r.unstandardize(1 / 3), 1e-3)

    def test_log_range_round_trip(self):
        r = FloatRange(1e-5, 1e2, log_range=True)
        for y in (0.0, 0.25, 0.5, 1.0):
            assert math.isclose(r.standardize(r.unstandardize(y)), y, abs_tol=1e-12)

    def test_defaults_to_linear(self):
        assert FloatRange(0.0, 1.0).log_range is False

    def test_clamps_outside_the_unit_interval(self):
        r = FloatRange(-5.0, 5.0)
        assert r.unstandardize(-0.2) == -5.0
        assert r.unstandardize(1.2) == 5.0

    def test_rejects_a_degenerate_range(self):
        with pytest.raises(ValueError):
            FloatRange(1.0, 1.0)
        with pytest.raises(ValueError):
            FloatRange(1.0, 0.0)

    def test_rejects_a_log_range_that_reaches_zero(self):
        # log(0) is not a number the search can work in.
        with pytest.raises(ValueError):
            FloatRange(0.0, 1.0, log_range=True)
        with pytest.raises(ValueError):
            FloatRange(-1.0, 1.0, log_range=True)


class TestCategoricalRange:
    def test_endpoints_map_to_the_unit_interval(self):
        r = CategoricalRange(4)
        assert r.standardize(0) == 0.0
        assert r.standardize(3) == 1.0

    def test_every_category_round_trips(self):
        r = CategoricalRange(7)
        for i in range(7):
            assert r.unstandardize(r.standardize(i)) == i

    def test_categories_partition_the_unit_interval(self):
        r = CategoricalRange(4)
        assert [r.unstandardize(y) for y in (0.0, 0.34, 0.66, 1.0)] == [0, 1, 2, 3]

    def test_a_single_category_is_not_a_division_by_zero(self):
        r = CategoricalRange(1)
        assert r.standardize(0) == 0.0
        assert r.unstandardize(0.7) == 0

    def test_clamps_outside_the_unit_interval(self):
        r = CategoricalRange(3)
        assert r.unstandardize(-0.5) == 0
        assert r.unstandardize(1.5) == 2

    def test_rejects_an_empty_range(self):
        with pytest.raises(ValueError):
            CategoricalRange(0)


class TestFloorPowerOfTwo:
    @pytest.mark.parametrize(
        "n,expected",
        [(1, 1), (2, 2), (3, 2), (4, 4), (7, 4), (8, 8), (9, 8), (63, 32), (64, 64)],
    )
    def test_truncates_down(self, n, expected):
        assert floor_power_of_two(n) == expected

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            floor_power_of_two(0)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


class TestConstruction:
    def test_exploration_budget_is_truncated_to_a_power_of_two(self):
        # Sobol' is only balanced on power-of-two prefixes of the sequence.
        opt, _ = make_opt(explore=20)
        assert opt.num_exploration_points == 16

    def test_a_power_of_two_is_left_alone(self):
        opt, _ = make_opt(explore=16)
        assert opt.num_exploration_points == 16

    def test_dim_follows_the_space(self):
        space = {"a": FloatRange(0, 1), "b": IntRange(0, 3), "c": CategoricalRange(2)}
        opt, _ = make_opt(space=space)
        assert opt.dim == 3
        assert opt.param_names == ["a", "b", "c"]

    def test_rejects_an_empty_space(self):
        with pytest.raises(ValueError):
            make_opt(space={})

    def test_rejects_a_negative_search_budget(self):
        with pytest.raises(ValueError):
            make_opt(search=-1)

    def test_rejects_zero_parallelism(self):
        with pytest.raises(ValueError):
            make_opt(parallel=0)

    def test_rejects_extra_kwargs_that_shadow_a_parameter(self):
        # Silently overriding a searched parameter
        # would make the recorded point and the evaluated point disagree.
        with pytest.raises(ValueError, match="shadow"):
            make_opt(x=1.0)

    def test_nothing_is_evaluated_at_construction(self):
        opt, executor = make_opt()
        assert executor.num_submitted == 0
        assert opt.values == []


# --------------------------------------------------------------------------
# Exploration
# --------------------------------------------------------------------------


class TestExploration:
    def test_evaluates_the_truncated_budget(self):
        opt, executor = make_opt(explore=9)
        opt.run_exploration_jobs()
        assert executor.num_submitted == 8
        assert len(opt.values) == 8
        assert len(opt.points) == 8

    def test_submits_to_the_configured_queue(self):
        opt, executor = make_opt(explore=4)
        opt.run_exploration_jobs()
        assert executor.queues == ["cpu"] * 4

    def test_accepts_a_list_of_queues(self):
        executor = LocalExecutor()
        opt = BayesOptBotorch(
            "t", BOX_2D, sphere, as_executor(executor), ["a", "b"], 2, 0, 1, SEED
        )
        opt.run_exploration_jobs()
        assert executor.queues == [["a", "b"]] * 2

    def test_passes_extra_kwargs_to_the_objective(self):
        def objective(x, y, *, scale):
            return scale * sphere(x, y)

        opt, executor = make_opt(objective=objective, explore=4, scale=3.0)
        opt.run_exploration_jobs()
        assert all(kw["scale"] == 3.0 for kw in executor.kwargs)
        assert all(set(kw) == {"x", "y", "scale"} for kw in executor.kwargs)

    def test_points_lie_inside_the_declared_space(self):
        space = {
            "f": FloatRange(-5.0, 5.0),
            "l": FloatRange(1e-4, 1e-1, log_range=True),
            "i": IntRange(2, 9),
            "c": CategoricalRange(3),
        }
        opt, _ = make_opt(
            objective=lambda f, l, i, c: f + l + i + c, space=space, explore=16
        )
        opt.run_exploration_jobs()
        for p in opt.points:
            assert -5.0 <= p["f"] <= 5.0
            assert 1e-4 <= p["l"] <= 1e-1
            assert isinstance(p["i"], int) and 2 <= p["i"] <= 9
            assert isinstance(p["c"], int) and p["c"] in (0, 1, 2)

    def test_the_design_covers_the_box(self):
        # A Sobol' design, not 16 draws at one spot:
        # every axis should have points in both halves.
        opt, _ = make_opt(explore=16)
        opt.run_exploration_jobs()
        for name in ("x", "y"):
            values = [p[name] for p in opt.points]
            assert min(values) < 0.0 < max(values)

    def test_the_design_is_reproducible_from_the_seed(self):
        first, _ = make_opt(explore=8, seed=7)
        second, _ = make_opt(explore=8, seed=7)
        first.run_exploration_jobs()
        second.run_exploration_jobs()
        assert first.points == second.points

    def test_a_different_seed_gives_a_different_design(self):
        first, _ = make_opt(explore=8, seed=7)
        second, _ = make_opt(explore=8, seed=8)
        first.run_exploration_jobs()
        second.run_exploration_jobs()
        assert first.points != second.points

    def test_the_design_does_not_depend_on_the_job_name(self):
        # The seed is the only thing that decides the design.
        # The name is for progress bars and error messages.
        first = BayesOptBotorch(
            "one", BOX_2D, sphere, as_executor(LocalExecutor()), "cpu", 8, 0, 1, 7
        )
        second = BayesOptBotorch(
            "two", BOX_2D, sphere, as_executor(LocalExecutor()), "cpu", 8, 0, 1, 7
        )
        first.run_exploration_jobs()
        second.run_exploration_jobs()
        assert first.points == second.points

    def test_the_seed_is_kept_as_given(self):
        opt, _ = make_opt(seed=1234)
        assert opt.seed == 1234

    def test_recorded_coordinates_are_the_point_actually_evaluated(self):
        # An integer parameter rounds;
        # the model must be told where the objective really ran,
        # not where the continuous proposal landed.
        space = {"i": IntRange(0, 4)}
        opt, _ = make_opt(objective=lambda i: float(i), space=space, explore=8)
        opt.run_exploration_jobs()
        for params, unit in zip(opt.points, opt.unit_points):
            assert unit == [params["i"] / 4]


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


class TestSearchBudget:
    def test_evaluates_exactly_the_search_budget(self):
        opt, executor = make_opt(explore=4, search=8, parallel=4)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert len(opt.values) == 4 + 8
        assert executor.num_submitted == 4 + 8

    def test_the_last_round_is_short_rather_than_overshooting(self):
        opt, executor = make_opt(explore=4, search=5, parallel=3)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert len(opt.values) == 4 + 5

    def test_a_zero_budget_does_nothing(self):
        opt, executor = make_opt(explore=4, search=0)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert executor.num_submitted == 4

    def test_requires_observations_to_model(self):
        # Fitting a GP to nothing is not a useful error message.
        opt, _ = make_opt()
        with pytest.raises(RuntimeError, match="run_exploration_jobs"):
            opt.run_search_jobs()

    def test_can_be_resumed_for_more_search(self):
        opt, _ = make_opt(explore=4, search=4, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        opt.run_search_jobs()
        # The budget is per call, and the second call models the first's results.
        assert len(opt.values) == 4 + 4 + 4


class TestAcquisitionSplit:
    """The batch split, asserted without paying for a real acqf optimization."""

    @pytest.fixture
    def record(self, monkeypatch):
        calls: list[tuple[str, int]] = []

        def fake_optimize_acqf(acqf, **kwargs):
            q = kwargs["q"]
            calls.append((type(acqf).__name__, q))
            dim = kwargs["bounds"].shape[1]
            return torch.rand(q, dim, dtype=bo.DTYPE), None

        monkeypatch.setattr(bo, "optimize_acqf", fake_optimize_acqf)
        return calls

    def test_splits_the_batch_between_the_two_acquisitions(self, record):
        opt, _ = make_opt(explore=4, search=4, parallel=4)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert record == [
            (qLogExpectedImprovement.__name__, 2),
            (qProbabilityOfImprovement.__name__, 2),
        ]

    def test_an_odd_batch_gives_the_extra_point_to_log_ei(self, record):
        opt, _ = make_opt(explore=4, search=3, parallel=3)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert record == [
            (qLogExpectedImprovement.__name__, 2),
            (qProbabilityOfImprovement.__name__, 1),
        ]

    def test_parallelism_of_one_skips_the_empty_acquisition(self, record):
        # q=0 is not a legal batch size for optimize_acqf.
        opt, _ = make_opt(explore=4, search=2, parallel=1)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert record == [(qLogExpectedImprovement.__name__, 1)] * 2

    def test_a_short_final_round_is_split_too(self, record):
        opt, _ = make_opt(explore=4, search=5, parallel=3)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert record == [
            (qLogExpectedImprovement.__name__, 2),
            (qProbabilityOfImprovement.__name__, 1),
            (qLogExpectedImprovement.__name__, 1),
            (qProbabilityOfImprovement.__name__, 1),
        ]

    def test_the_model_is_refit_every_round(self, record, monkeypatch):
        fits = []
        real_fit = bo.fit_gpytorch_mll
        monkeypatch.setattr(
            bo,
            "fit_gpytorch_mll",
            lambda mll, **kw: (fits.append(1), real_fit(mll, **kw))[1],
        )
        opt, _ = make_opt(explore=4, search=6, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert len(fits) == 3


class TestSearchBehaviour:
    """The optimizer minimizes. Botorch maximizes, so this is worth asserting."""

    def test_search_moves_toward_the_minimum(self):
        # f(x) = x on [0, 1]: a flipped sign sends every point to 1.0 instead.
        opt, _ = make_opt(
            objective=identity,
            space={"x": FloatRange(0.0, 1.0)},
            explore=4,
            search=8,
            parallel=2,
        )
        opt.run_exploration_jobs()
        n_explored = len(opt.points)
        opt.run_search_jobs()

        searched = [p["x"] for p in opt.points[n_explored:]]
        assert max(searched) < 0.5, searched
        assert opt.best_point()[0]["x"] < 0.5

    def test_search_finds_the_optimum(self):
        opt, _ = make_opt(objective=sphere, explore=8, search=12, parallel=4)
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        params, value = opt.best_point()
        assert value < 0.5, value
        assert abs(params["x"]) < 1.0 and abs(params["y"]) < 1.0

    def test_search_beats_random_search(self):
        # Sobol' alone over the same total budget is the thing BO has to beat.
        # A unimodal objective deliberately:
        # on a multimodal one this margin closes at these budgets
        # and the test starts flaking.
        opt, _ = make_opt(objective=sphere, explore=8, search=16, parallel=4)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        guided = opt.best_point()[1]

        blind, _ = make_opt(objective=sphere, explore=24, search=0)
        blind.run_exploration_jobs()

        assert guided < blind.best_point()[1]

    def test_a_mixed_space_optimizes(self):
        # Float, integer and categorical in one space.
        # The categorical offsets are large,
        # so picking the wrong category cannot look like a good point
        # however well the continuous part is tuned.
        offsets = [0.0, 10.0, 25.0]

        def objective(x, y, cat):
            return sphere(x, y) + offsets[cat]

        space = {
            "x": FloatRange(-5.0, 5.0),
            "y": IntRange(-5, 5),
            "cat": CategoricalRange(3),
        }
        opt, _ = make_opt(
            objective=objective, space=space, explore=16, search=16, parallel=4
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        params, value = opt.best_point()
        assert params["cat"] == 0, params
        assert params["y"] == 0, params
        assert value < 1.0, value


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------


class TestFailures:
    def test_a_failed_evaluation_raises(self):
        # Workers report exceptions as the task's output, not by raising.
        def boom(x, y):
            raise RuntimeError("worker exploded")

        opt, _ = make_opt(objective=boom, explore=2)
        with pytest.raises(RuntimeError, match="failed"):
            opt.run_exploration_jobs()

    def test_a_remote_error_is_never_recorded_as_a_value(self):
        def boom(x, y):
            raise RuntimeError("worker exploded")

        opt, _ = make_opt(objective=boom, explore=2)
        with pytest.raises(RuntimeError):
            opt.run_exploration_jobs()
        assert opt.values == []

    def test_a_non_finite_objective_raises(self):
        # NaN silently poisons the GP fit; the failure has to surface here.
        opt, _ = make_opt(objective=lambda x, y: float("nan"), explore=2)
        with pytest.raises(RuntimeError, match="non-finite"):
            opt.run_exploration_jobs()

    def test_an_infinite_objective_raises(self):
        opt, _ = make_opt(objective=lambda x, y: float("inf"), explore=2)
        with pytest.raises(RuntimeError, match="non-finite"):
            opt.run_exploration_jobs()

    def test_a_non_numeric_objective_raises(self):
        opt, _ = make_opt(objective=lambda x, y: "not a number", explore=2)
        with pytest.raises(RuntimeError, match="not a float"):
            opt.run_exploration_jobs()

    def test_an_integer_objective_is_accepted(self):
        opt, _ = make_opt(objective=lambda x, y: 1, explore=2)
        opt.run_exploration_jobs()
        assert opt.values == [1.0, 1.0]


# --------------------------------------------------------------------------
# best_point
# --------------------------------------------------------------------------


class TestBestPoint:
    def test_returns_the_minimum(self):
        opt, _ = make_opt(objective=sphere, explore=8)
        opt.run_exploration_jobs()
        params, value = opt.best_point()
        assert value == min(opt.values)
        assert value == sphere(**params)

    def test_raises_before_anything_is_evaluated(self):
        opt, _ = make_opt()
        with pytest.raises(RuntimeError):
            opt.best_point()

    def test_returns_a_copy(self):
        # A caller mutating the returned dict must not corrupt the history.
        opt, _ = make_opt(explore=4)
        opt.run_exploration_jobs()
        params, _ = opt.best_point()
        params["x"] = 999.0
        assert 999.0 not in [p["x"] for p in opt.points]

    def test_improves_or_holds_across_the_search(self):
        opt, _ = make_opt(objective=sphere, explore=8, search=8, parallel=4)
        opt.run_exploration_jobs()
        after_exploration = opt.best_point()[1]
        opt.run_search_jobs()
        assert opt.best_point()[1] <= after_exploration


# --------------------------------------------------------------------------
# The real executor
# --------------------------------------------------------------------------


class TestRealExecutor:
    """One end-to-end run against the real queue, executor and worker.

    This is what keeps `LocalExecutor` honest
    --- everything above assumes `submit()` returns a Task
    whose `output` `wait()` fills in,
    and this is where that assumption meets the real implementation.
    """

    def test_optimizes_through_a_real_worker(
        self, executor, ds_service_address, tmp_path
    ):
        explore, search, parallel = 4, 4, 2
        total = explore + search

        opt = BayesOptBotorch(
            "e2e", BOX_2D, sphere, executor, "cpu", explore, search, parallel, SEED
        )

        # A real worker in a thread:
        # the optimizer blocks in wait() as soon as it submits,
        # so nothing can play the worker's part after the fact.
        worker = make_worker(ds_service_address, tmp_path / "worker", group="cpu")
        thread = threading.Thread(target=run_worker, args=(worker, total), daemon=True)
        thread.start()
        try:
            opt.run_exploration_jobs()
            opt.run_search_jobs()
        finally:
            thread.join(timeout=30)
            worker.close()

        assert not thread.is_alive(), "worker thread did not finish"
        assert len(opt.values) == total

        params, value = opt.best_point()
        assert value == min(opt.values)
        assert math.isclose(value, sphere(**params))
        assert value < BOX_2D["x"].max ** 2 + BOX_2D["y"].max ** 2

    def test_a_raising_objective_surfaces_from_a_real_worker(
        self, executor, ds_service_address, tmp_path
    ):
        def boom(x, y):
            raise RuntimeError("worker exploded")

        opt = BayesOptBotorch("e2e-fail", BOX_2D, boom, executor, "cpu", 2, 0, 1, SEED)

        worker = make_worker(ds_service_address, tmp_path / "worker", group="cpu")
        thread = threading.Thread(target=run_worker, args=(worker, 2), daemon=True)
        thread.start()
        try:
            with pytest.raises(RuntimeError, match="failed"):
                opt.run_exploration_jobs()
        finally:
            thread.join(timeout=30)
            worker.close()

        assert opt.values == []
