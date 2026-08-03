"""Tests for the botorch-based parallel optimizer.

The optimizer's contract with the executor is two calls wide ---
`submit()` returning a `Task`, and `wait()` filling in that task's `output` ---
so most tests here drive that contract through `LocalExecutor`,
which runs whatever it is handed inline ---
both the objective and, since the fit became a task of its own,
`fit_and_propose`.
The GP fit and the acquisition optimization
are the expensive part of these tests;
a queue round trip on top would add nothing the optimizer can tell apart.

Running the fit inline is also what makes the monkeypatching work:
a patched `bo.optimize_acqf` reaches the task
because the task ran in this process.
`TestOptimizerQueue` covers what only shows up once it does not:
which queue the fit went to,
and what a failure on the far end reports.

`TestRealExecutor` keeps the stand-in honest.
It runs a whole optimization through the real executor,
the real ds-service queue and a real worker, the fit included,
pinning the two-call contract against the real implementation.

The four tests in `TestSearchBehaviour` assert behaviour of the search
rather than its bookkeeping,
and are what would catch the objective's sign being flipped
--- botorch maximizes and this optimizer minimizes.
They are stochastic
(torch's global RNG is left unseeded, so each run is a fresh sample),
and their thresholds come from measured spreads:
the monotone case put every search point below 0.1 against a threshold of 0.5,
and `test_search_beats_random_search` won 12/12 with a 4.6x margin.
All four use unimodal objectives:
an earlier multimodal version of the random-search comparison
lost 1 run in 10.
"""

from __future__ import annotations

import math
import re
import statistics
import threading
from typing import cast

import pytest

pytest.importorskip("botorch")

import torch  # noqa: E402
from botorch.acquisition import qLogNoisyExpectedImprovement  # noqa: E402

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
    """Convex, minimum f = 0 at the origin.

    Objectives return a mapping, not a bare number: the value to minimize
    under "objective", plus whatever else is worth recording. The extra key
    here keeps the tests honest about the optimizer carrying it through.
    """
    return {"objective": x * x + y * y, "note": "sphere"}


def identity(x):
    """Monotone: the minimum is at the low edge of the box."""
    return {"objective": x}


BOX_2D = {"x": FloatRange(-5.0, 5.0), "y": FloatRange(-5.0, 5.0)}


SEED = 20260730


def make_opt(
    objective=sphere,
    space=None,
    explore=8,
    iterations=2,
    parallel=4,
    seed: int | None = SEED,
    objective_queue: str | list[str] = "cpu",
    optimizer_queue: str | list[str] = "opt",
    **extra,
):
    """An optimizer wired to a fresh LocalExecutor.

    `seed` and the two queue names are named explicitly
    so they are not swallowed by `**extra`, which goes to the objective.

    The two queues are given different names throughout,
    so a test that asserts on `executor.queues`
    is asserting which kind of work went where,
    not just that something was submitted.

    `iterations` pins the round count by setting both search bounds to it,
    which switches early stopping off:
    a stall can only end the search at a round at or above the floor,
    and with floor == ceiling that is the round the loop ends on anyway,
    so the search runs exactly that many rounds.
    Tests that are *about* early stopping pass the bounds themselves.
    """
    space = BOX_2D if space is None else space
    executor = LocalExecutor()

    early = {"min_search_iterations": iterations, "max_search_iterations": iterations}
    for name in (
        "min_search_iterations",
        "max_search_iterations",
        "patience",
        "min_improvement",
        "objective_key",
        # The acquisition knobs travel the same way: named here so they
        # reach the optimizer instead of being handed to the objective.
        "num_restarts",
        "raw_samples",
        "mc_samples",
        "acqf_timeout_s",
    ):
        if name in extra:
            early[name] = extra.pop(name)

    opt = BayesOptBotorch(
        "test",
        space,
        objective,
        as_executor(executor),
        objective_queue,
        optimizer_queue,
        explore,
        parallel,
        seed,
        **early,
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

    def test_rejects_negative_search_iterations(self):
        with pytest.raises(ValueError, match="min_search_iterations"):
            make_opt(iterations=-1)

    def test_rejects_a_ceiling_below_the_floor(self):
        with pytest.raises(ValueError, match="max_search_iterations"):
            make_opt(min_search_iterations=5, max_search_iterations=4)

    def test_rejects_zero_patience(self):
        # Zero would end the search before a round could reset the counter.
        with pytest.raises(ValueError, match="patience"):
            make_opt(patience=0)

    def test_rejects_negative_min_improvement(self):
        with pytest.raises(ValueError, match="min_improvement"):
            make_opt(min_improvement=-0.1)

    def test_rejects_zero_parallelism(self):
        with pytest.raises(ValueError):
            make_opt(parallel=0)

    @pytest.mark.parametrize(
        "knob, value",
        [
            ("num_restarts", 0),
            ("raw_samples", 0),
            ("mc_samples", 0),
            ("acqf_timeout_s", 0.0),
            ("acqf_timeout_s", -1.0),
        ],
    )
    def test_rejects_a_degenerate_acquisition_knob(self, knob, value):
        # Each of these reaches botorch on a compute node.
        # Caught here, the message names the argument;
        # caught there, it is a stack trace in a worker log.
        with pytest.raises(ValueError, match=knob):
            make_opt(**{knob: value})

    def test_the_acquisition_knobs_are_kept_as_given(self):
        # They are the run's settings, carried on the instance
        # and read again for every fit task.
        opt, _ = make_opt(
            num_restarts=3, raw_samples=7, mc_samples=11, acqf_timeout_s=1.5
        )
        assert opt.num_restarts == 3
        assert opt.raw_samples == 7
        assert opt.mc_samples == 11
        assert opt.acqf_timeout_s == 1.5

    def test_the_acquisition_knobs_have_usable_defaults(self):
        # The values themselves live in the signature and are not pinned here;
        # what matters is that an unconfigured run has them at all.
        opt, _ = make_opt()
        assert opt.num_restarts >= 1
        assert opt.raw_samples >= opt.num_restarts
        assert opt.mc_samples >= 1
        assert opt.acqf_timeout_s > 0.0

    def test_an_objective_may_take_the_optimizers_own_argument_names(self):
        """The leading arguments are positional-only, which is what this buys.

        An objective with its own `seed` must receive it,
        rather than rebinding the exploration design's seed.
        """

        def objective(x, y, *, seed):
            return {"objective": x * x + y * y, "seed": seed}

        executor = LocalExecutor()
        opt = BayesOptBotorch(
            "t",
            BOX_2D,
            objective,
            as_executor(executor),
            "cpu",
            "opt",
            2,
            1,
            7,
            min_search_iterations=0,
            max_search_iterations=0,
            seed=99,
        )
        opt.run_exploration_jobs()

        assert opt.seed == 7
        assert opt.extra_objective_kwargs == {"seed": 99}
        assert all(kw["seed"] == 99 for kw in executor.kwargs)
        assert opt.best_output()["seed"] == 99

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
            "t",
            BOX_2D,
            sphere,
            as_executor(executor),
            ["a", "b"],
            "opt",
            2,
            1,
            SEED,
            min_search_iterations=0,
            max_search_iterations=0,
        )
        opt.run_exploration_jobs()
        assert executor.queues == [["a", "b"]] * 2

    def test_passes_extra_kwargs_to_the_objective(self):
        def objective(x, y, *, scale):
            return {"objective": scale * sphere(x, y)["objective"]}

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
            objective=lambda f, l, i, c: {"objective": f + l + i + c},
            space=space,
            explore=16,
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

    def test_an_omitted_seed_is_drawn(self):
        first, _ = make_opt(explore=4, seed=None)
        second, _ = make_opt(explore=4, seed=None)

        assert first.seed is not None
        assert first.seed != second.seed, "two unseeded runs must not coincide"

    def test_a_drawn_seed_fits_what_sobol_accepts(self):
        """torch unpacks the seed as a signed long long and overflows above it."""
        for _ in range(10):
            opt, _ = make_opt(explore=4, seed=None)
            assert 0 <= opt.seed < 2**63

    def test_a_drawn_seed_reproduces_its_own_run(self):
        """Otherwise an unseeded run could never be repeated."""
        first, _ = make_opt(explore=8, seed=None)
        first.run_exploration_jobs()

        second, _ = make_opt(explore=8, seed=first.seed)
        second.run_exploration_jobs()

        assert first.points == second.points

    def test_a_drawn_seed_is_reported(self, capsys):
        """It is only reproducible if the user can see what was used."""
        opt, _ = make_opt(explore=4, seed=None)

        out = capsys.readouterr().out
        assert str(opt.seed) in out
        assert "no seed given" in out

    def test_an_explicit_seed_is_not_announced(self, capsys):
        make_opt(explore=4, seed=7)

        assert "no seed given" not in capsys.readouterr().out

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
            "one",
            BOX_2D,
            sphere,
            as_executor(LocalExecutor()),
            "cpu",
            "opt",
            8,
            1,
            7,
            min_search_iterations=0,
            max_search_iterations=0,
        )
        second = BayesOptBotorch(
            "two",
            BOX_2D,
            sphere,
            as_executor(LocalExecutor()),
            "cpu",
            "opt",
            8,
            1,
            7,
            min_search_iterations=0,
            max_search_iterations=0,
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
        opt, _ = make_opt(
            objective=lambda i: {"objective": float(i)}, space=space, explore=8
        )
        opt.run_exploration_jobs()
        for params, unit in zip(opt.points, opt.unit_points):
            assert unit == [params["i"] / 4]


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def constant(x, y):
    """Flat: nothing a search does can ever improve on the incumbent."""
    return {"objective": 1.0}


def rounds_run(opt, explored: int) -> int:
    """How many search rounds actually ran."""
    return (len(opt.values) - explored) // opt.search_parallelism


class TestEarlyStopping:
    def test_stops_after_patience_stalled_rounds(self):
        opt, _ = make_opt(
            objective=constant,
            explore=4,
            parallel=2,
            min_search_iterations=2,
            max_search_iterations=30,
            patience=3,
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        # Stalls count from the first round, so patience alone decides
        # once it is the larger of the two.
        assert rounds_run(opt, 4) == 3

    def test_the_floor_runs_even_when_nothing_improves(self):
        """Patience below the floor cannot cut the search short."""
        opt, _ = make_opt(
            objective=constant,
            explore=4,
            parallel=2,
            min_search_iterations=6,
            max_search_iterations=30,
            patience=1,
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        assert rounds_run(opt, 4) == 6

    def test_stalls_below_the_floor_are_carried_past_it(self):
        """The floor holds off the stop, not the counting.

        Patience is already spent by the time the floor is reached,
        so the search ends at the floor rather than at floor + patience.
        """
        opt, _ = make_opt(
            objective=constant,
            explore=4,
            parallel=2,
            min_search_iterations=3,
            max_search_iterations=30,
            patience=2,
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        assert rounds_run(opt, 4) == 3

    def test_the_ceiling_stops_a_search_that_keeps_improving(self):
        opt, _ = make_opt(
            objective=sphere,
            explore=4,
            parallel=2,
            min_search_iterations=0,
            max_search_iterations=4,
            patience=100,
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        assert rounds_run(opt, 4) == 4

    def test_an_improving_round_resets_the_counter(self, monkeypatch):
        """Patience bounds a *run* of bad rounds, not their total."""
        seen = []

        def improved(self, previous, current):
            # stall, stall, improve, stall, stall, stall -> stop at 6
            pattern = [False, False, True, False, False, False]
            seen.append(len(seen))
            return pattern[min(len(seen) - 1, len(pattern) - 1)]

        monkeypatch.setattr(bo.BayesOptBotorch, "_improved_enough", improved)
        opt, _ = make_opt(
            objective=sphere,
            explore=4,
            parallel=2,
            min_search_iterations=0,
            max_search_iterations=30,
            patience=3,
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        assert rounds_run(opt, 4) == 6

    def test_a_zero_floor_leaves_patience_in_charge(self):
        opt, _ = make_opt(
            objective=constant,
            explore=4,
            parallel=2,
            min_search_iterations=0,
            max_search_iterations=30,
            patience=2,
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        assert rounds_run(opt, 4) == 2

    def test_the_progress_line_counts_towards_the_real_stop(self, capsys):
        """A floor outlasting patience must not print a ratio past its own end.

        These are the shipped defaults, so it is the common case:
        counting `stalled` against `patience` alone reaches "(4/3)",
        and then claims a 3-round streak for a search that stalled 5 times.
        """
        opt, _ = make_opt(
            objective=constant,
            explore=4,
            parallel=2,
            min_search_iterations=5,
            max_search_iterations=30,
            patience=3,
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        out = capsys.readouterr().out

        # The gap shrinks by one a round, and the floor sets it:
        # patience alone would have run out after three.
        assert re.findall(r"(\d+) in a row, (\d+) more to stop", out) == [
            ("1", "4"),
            ("2", "3"),
            ("3", "2"),
            ("4", "1"),
        ]
        assert "stopping after 5 rounds --- 5 in a row" in out

    def test_it_says_why_it_stopped(self, capsys):
        opt, _ = make_opt(
            objective=constant,
            explore=4,
            parallel=2,
            min_search_iterations=0,
            max_search_iterations=30,
            patience=2,
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        out = capsys.readouterr().out
        assert "stopping after 2 rounds" in out
        assert "5%" in out


class TestImprovementTest:
    """`_improved_enough` decides every stall, so its edges matter."""

    @pytest.fixture
    def opt(self):
        opt, _ = make_opt(min_improvement=0.05)
        return opt

    def test_a_big_enough_drop_counts(self, opt):
        assert opt._improved_enough(1.0, 0.94)

    def test_a_drop_below_the_threshold_does_not(self, opt):
        assert not opt._improved_enough(1.0, 0.96)

    def test_the_threshold_is_inclusive(self, opt):
        assert opt._improved_enough(1.0, 0.95)

    def test_no_change_is_not_improvement(self, opt):
        assert not opt._improved_enough(1.0, 1.0)

    def test_getting_worse_is_not_improvement(self, opt):
        assert not opt._improved_enough(1.0, 2.0)

    def test_it_is_relative_not_absolute(self, opt):
        """The same absolute step is decisive at one scale and noise at another."""
        assert opt._improved_enough(1.0, 0.9)
        assert not opt._improved_enough(1000.0, 999.9)

    def test_a_negative_incumbent_uses_its_magnitude(self, opt):
        # -10 -> -11 is a 10% improvement; -10 -> -10.1 is 1%.
        assert opt._improved_enough(-10.0, -11.0)
        assert not opt._improved_enough(-10.0, -10.1)

    def test_a_zero_incumbent_accepts_any_decrease(self, opt):
        """Zero has no magnitude to take a fraction of."""
        assert opt._improved_enough(0.0, -1e-9)
        assert not opt._improved_enough(0.0, 0.0)


class TestSearchBudget:
    def test_evaluates_iterations_times_parallelism(self):
        opt, executor = make_opt(explore=4, iterations=2, parallel=4)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert len(opt.values) == 4 + 2 * 4
        # The evaluations, plus one fit-and-propose task per round ---
        # which is submitted like any other task and counts here too.
        assert executor.num_submitted == 4 + 2 * 4 + 2

    def test_every_round_is_the_full_width_of_the_pool(self):
        """No short final round: the budget is rounds, not points."""
        opt, executor = make_opt(explore=4, iterations=3, parallel=3)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert len(opt.values) == 4 + 3 * 3

    def test_a_zero_budget_does_nothing(self):
        opt, executor = make_opt(explore=4, iterations=0)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert executor.num_submitted == 4

    def test_requires_observations_to_model(self):
        # Fitting a GP to nothing is not a useful error message.
        opt, _ = make_opt()
        with pytest.raises(RuntimeError, match="run_exploration_jobs"):
            opt.run_search_jobs()

    def test_can_be_resumed_for_more_search(self):
        opt, _ = make_opt(explore=4, iterations=2, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        opt.run_search_jobs()
        # The budget is per call, and the second call models the first's results.
        assert len(opt.values) == 4 + 4 + 4


class TestAcquisition:
    """One acquisition per round, asserted without paying for a real optimization."""

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

    def test_one_call_per_round_for_the_whole_batch(self, record):
        opt, _ = make_opt(explore=4, iterations=1, parallel=4)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert record == [(qLogNoisyExpectedImprovement.__name__, 4)]

    def test_an_odd_batch_is_not_split(self, record):
        opt, _ = make_opt(explore=4, iterations=1, parallel=3)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert record == [(qLogNoisyExpectedImprovement.__name__, 3)]

    def test_a_parallelism_of_one_still_asks_for_one_point(self, record):
        opt, _ = make_opt(explore=4, iterations=2, parallel=1)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert record == [(qLogNoisyExpectedImprovement.__name__, 1)] * 2

    def test_every_round_asks_for_the_full_parallelism(self, record):
        """The budget is rounds, so no round is short."""
        opt, _ = make_opt(explore=4, iterations=3, parallel=3)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert record == [(qLogNoisyExpectedImprovement.__name__, 3)] * 3

    def test_a_timeout_is_passed_to_the_optimizer(self, monkeypatch):
        """Unbounded, one round can outlast the batch it is choosing points for."""
        timeouts = []

        def fake_optimize_acqf(acqf, **kwargs):
            timeouts.append(kwargs.get("timeout_sec"))
            q, dim = kwargs["q"], kwargs["bounds"].shape[1]
            return torch.rand(q, dim, dtype=bo.DTYPE), None

        monkeypatch.setattr(bo, "optimize_acqf", fake_optimize_acqf)
        opt, _ = make_opt(explore=4, iterations=2, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        # Against the run's own setting rather than a literal:
        # the default lives in the constructor signature,
        # and pinning its value here would only mean editing this test
        # whenever it is retuned.
        assert timeouts == [opt.acqf_timeout_s] * 2
        assert timeouts[0] is not None

    def test_a_timed_out_proposal_is_still_usable(self):
        """The limit degrades the proposal; it must not break the round.

        botorch returns its best-so-far rather than raising,
        so a round that runs out of time still yields a full batch
        of finite, in-bounds points.
        """
        opt, _ = make_opt(explore=4, iterations=1, parallel=3, acqf_timeout_s=0.001)
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        searched = opt.points[4:]
        assert len(searched) == 3
        for params in searched:
            for name, value in params.items():
                assert math.isfinite(value)
                assert BOX_2D[name].min <= value <= BOX_2D[name].max

    def test_the_sampler_is_passed_explicitly(self, monkeypatch):
        """Left to botorch the default is larger, and every round pays for it."""
        shapes = []
        real_acqf = bo.qLogNoisyExpectedImprovement

        def spy(model, x_baseline, *a, sampler=None, **kw):
            shapes.append(None if sampler is None else tuple(sampler.sample_shape))
            return real_acqf(model, x_baseline, *a, sampler=sampler, **kw)

        monkeypatch.setattr(bo, "qLogNoisyExpectedImprovement", spy)
        opt, _ = make_opt(explore=4, iterations=2, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        # Compared against the run's own setting, not a literal:
        # the sample count is a tuning knob, and pinning its value here
        # only means this test has to be edited whenever it is retuned.
        assert shapes == [(opt.mc_samples,)] * 2

    def test_the_baseline_is_every_point_measured_so_far(self, monkeypatch):
        """qLogNEI reads its incumbent off these, so they must be up to date."""
        baselines: list[int] = []
        real_acqf = bo.qLogNoisyExpectedImprovement

        def spy(model, x_baseline, *a, **kw):
            baselines.append(len(x_baseline))
            return real_acqf(model, x_baseline, *a, **kw)

        monkeypatch.setattr(bo, "qLogNoisyExpectedImprovement", spy)
        opt, _ = make_opt(explore=4, iterations=2, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        # Four exploration points, then those plus the first round's two.
        assert baselines == [4, 6]

    def test_each_round_reports_how_long_proposing_took(self, capsys):
        opt, _ = make_opt(explore=4, iterations=3, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        out = capsys.readouterr().out

        # Three rounds, and exploration proposes nothing --- it is a Sobol'
        # draw, not an acquisition optimization.
        proposals = re.findall(r"proposed (\d+) points in ([0-9.]+)s", out)
        assert len(proposals) == 3
        assert [int(n) for n, _ in proposals] == [2, 2, 2]
        assert all(float(t) >= 0.0 for _, t in proposals)

    def test_the_best_so_far_is_reported_after_every_batch(self, capsys):
        opt, _ = make_opt(explore=4, iterations=3, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        out = capsys.readouterr().out
        counts = [int(n) for n in re.findall(r"best after (\d+) points", out)]

        # Once for exploration, then once per search round,
        # each covering everything measured up to that point.
        assert counts == [4, 6, 8, 10]

    def test_the_best_is_reported_after_exploration_alone(self, capsys):
        opt, _ = make_opt(explore=4, iterations=0, parallel=2)
        opt.run_exploration_jobs()

        out = capsys.readouterr().out
        _, value = opt.best_point()
        assert "best after 4 points" in out
        assert f"{value:.6g}" in out

    def test_reported_parameters_keep_their_type(self, capsys):
        """An int parameter must not be printed as a float."""
        opt, _ = make_opt(
            objective=lambda x, n: {"objective": x + n},
            space={"x": FloatRange(0.0, 1.0), "n": IntRange(1, 8)},
            explore=4,
            iterations=0,
        )
        opt.run_exploration_jobs()

        out = capsys.readouterr().out
        params, _ = opt.best_point()
        assert f"n={params['n']}" in out, out
        assert f"n={float(params['n'])}" not in out

    def test_each_fit_reports_its_size_and_duration(self, capsys):
        opt, _ = make_opt(explore=4, iterations=2, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        out = capsys.readouterr().out

        # Two rounds, so two fits, each announced before and timed after.
        assert out.count("fitting GP on") == 2
        assert out.count("GP fit took") == 2

        # The count is the observations the fit actually sees:
        # four exploration points, then those plus the first round's two.
        assert "test: fitting GP on 4 points" in out
        assert "test: fitting GP on 6 points" in out

        durations = re.findall(r"GP fit took ([0-9.]+)s", out)
        assert len(durations) == 2
        assert all(float(d) >= 0.0 for d in durations)

    def test_the_size_is_reported_before_the_fit_runs(self, monkeypatch, capsys):
        """The count has to be visible even if the fit then hangs or dies."""

        def explode(mll, **kw):
            raise RuntimeError("fit blew up")

        opt, _ = make_opt(explore=4, iterations=1, parallel=2)
        opt.run_exploration_jobs()
        monkeypatch.setattr(bo, "fit_gpytorch_mll", explode)

        with pytest.raises(RuntimeError, match="fit blew up"):
            opt.run_search_jobs()

        out = capsys.readouterr().out
        assert "test: fitting GP on 4 points" in out
        assert "GP fit took" not in out, "no duration for a fit that never finished"

    def test_the_model_is_refit_every_round(self, record, monkeypatch):
        fits = []
        real_fit = bo.fit_gpytorch_mll
        monkeypatch.setattr(
            bo,
            "fit_gpytorch_mll",
            lambda mll, **kw: (fits.append(1), real_fit(mll, **kw))[1],
        )
        opt, _ = make_opt(explore=4, iterations=3, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert len(fits) == 3


class TestOptimizerQueue:
    """The fit is a task too, and it goes somewhere else.

    Everything the driver used to do inline now crosses the executor,
    so what is asserted here is *where* each kind of work was sent
    and what happens when the far end cannot do it.
    """

    def test_the_fit_goes_to_the_optimizer_queue(self):
        opt, executor = make_opt(explore=2, iterations=2, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        # Exploration is objective-only; then each round is one fit
        # followed by that round's evaluations.
        assert executor.queues == ["cpu"] * 2 + ["opt", "cpu", "cpu"] * 2

    def test_the_fit_accepts_a_list_of_queues(self):
        opt, executor = make_opt(
            explore=2, iterations=1, parallel=1, optimizer_queue=["a", "b"]
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert executor.queues == ["cpu", "cpu", ["a", "b"], "cpu"]

    def test_one_queue_may_serve_both(self):
        """Nothing deadlocks: the two kinds are never in flight together."""
        opt, executor = make_opt(
            explore=2, iterations=1, parallel=2, optimizer_queue="cpu"
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        assert executor.queues == ["cpu"] * 5
        assert len(opt.values) == 4

    def test_the_run_carries_its_own_tuning_to_the_worker(self, monkeypatch):
        """How the run was configured has to decide, not what the worker has.

        The four knobs belong to the optimizer object;
        reading the module globals on the compute node instead
        would silently ignore everything the caller asked for.
        """
        seen = self._record_kwargs(monkeypatch)

        opt, _ = make_opt(
            explore=4,
            iterations=1,
            parallel=2,
            num_restarts=3,
            raw_samples=7,
            mc_samples=11,
            acqf_timeout_s=1.5,
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        assert seen == [
            {
                "num_restarts": 3,
                "raw_samples": 7,
                "mc_samples": 11,
                "timeout_s": 1.5,
            }
        ]

    def test_an_unconfigured_run_carries_its_defaults(self, monkeypatch):
        """The defaults travel too --- the worker is told, never left to guess."""
        seen = self._record_kwargs(monkeypatch)

        opt, _ = make_opt(explore=4, iterations=1, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        assert seen == [
            {
                "num_restarts": opt.num_restarts,
                "raw_samples": opt.raw_samples,
                "mc_samples": opt.mc_samples,
                "timeout_s": opt.acqf_timeout_s,
            }
        ]

    @staticmethod
    def _record_kwargs(monkeypatch) -> list[dict]:
        """Record what each fit was asked for, without paying for one."""
        seen: list[dict] = []

        def fake(unit_points, values, batch, **kwargs):
            seen.append(kwargs)
            return {
                bo.CANDIDATES_KEY: [[0.5] * len(unit_points[0]) for _ in range(batch)],
                bo.FIT_SECONDS_KEY: 0.0,
                bo.PROPOSE_SECONDS_KEY: 0.0,
            }

        monkeypatch.setattr(bo, "fit_and_propose", fake)
        return seen

    def test_a_fit_that_fails_names_the_queue_and_botorch(self, monkeypatch):
        """The traceback is in a worker log the driver never reads."""

        def explode(mll, **kw):
            raise RuntimeError("No module named 'botorch'")

        opt, _ = make_opt(explore=4, iterations=1, parallel=2)
        opt.run_exploration_jobs()
        monkeypatch.setattr(bo, "fit_gpytorch_mll", explode)

        with pytest.raises(RuntimeError, match="'opt'.*botorch") as excinfo:
            opt.run_search_jobs()
        assert "search 1/1" in str(excinfo.value)

    def test_an_unusable_result_is_reported_rather_than_unpacked(self, monkeypatch):
        """The case is a worker running a different slurm-workflows."""
        monkeypatch.setattr(bo, "fit_and_propose", lambda *a, **kw: {"points": []})

        opt, _ = make_opt(explore=4, iterations=1, parallel=2)
        opt.run_exploration_jobs()

        with pytest.raises(RuntimeError, match="no 'candidates'"):
            opt.run_search_jobs()

    def test_a_result_missing_only_the_timings_is_reported_too(self, monkeypatch):
        """Every key the driver goes on to read, not just the candidates.

        Half-right output is what version skew actually looks like,
        and reaching the timings as a bare KeyError
        would skip the message written for exactly this.
        """
        monkeypatch.setattr(
            bo,
            "fit_and_propose",
            lambda unit_points, values, batch, **kw: {
                bo.CANDIDATES_KEY: [[0.5] * len(unit_points[0])] * batch
            },
        )

        opt, _ = make_opt(explore=4, iterations=1, parallel=2)
        opt.run_exploration_jobs()

        with pytest.raises(RuntimeError, match="'fit_seconds', 'propose_seconds'"):
            opt.run_search_jobs()

    def test_a_batch_that_is_not_the_full_width_is_rejected(self, monkeypatch):
        """A short round would otherwise pass as a normal one."""
        monkeypatch.setattr(
            bo,
            "fit_and_propose",
            lambda unit_points, values, batch, **kw: {
                bo.CANDIDATES_KEY: [[0.5] * len(unit_points[0])] * (batch - 1),
                bo.FIT_SECONDS_KEY: 0.0,
                bo.PROPOSE_SECONDS_KEY: 0.0,
            },
        )

        opt, _ = make_opt(explore=4, iterations=1, parallel=3)
        opt.run_exploration_jobs()

        with pytest.raises(RuntimeError, match=r"proposed 2 points.*not the 3"):
            opt.run_search_jobs()

    def test_the_fit_sees_every_point_measured_so_far(self, monkeypatch):
        """It is given the observations, not a handle to the driver's state."""
        sizes = []
        real = bo.fit_and_propose

        def spy(unit_points, values, batch, **kwargs):
            sizes.append((len(unit_points), len(values)))
            return real(unit_points, values, batch, **kwargs)

        monkeypatch.setattr(bo, "fit_and_propose", spy)

        opt, _ = make_opt(explore=4, iterations=2, parallel=2)
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        assert sizes == [(4, 4), (6, 6)]


class TestSearchBehaviour:
    """The optimizer minimizes while botorch maximizes, so assert the sign."""

    def test_search_moves_toward_the_minimum(self):
        # f(x) = x on [0, 1]: a flipped sign sends the search to 1.0 instead.
        #
        # Asserted on the *median* search point, not these two:
        #
        # `max(searched)`: qLogNEI treats the objective as noisy and keeps
        # probing away from the incumbent, so over 15 correct runs the max
        # reached 1.0 and breached a 0.5 bound 10 times. That is exploration,
        # not a wrong sign.
        #
        # `best_point()`: with the sign flipped the best stays at 0.057,
        # because the Sobol' exploration already sampled near the minimum
        # and the search never beats it.
        #
        # The median separates the two. Measured over 12 runs each:
        # correct 0.00 (max 0.00), flipped 1.00 (min 0.97).
        opt, _ = make_opt(
            objective=identity,
            space={"x": FloatRange(0.0, 1.0)},
            explore=4,
            iterations=2,
            parallel=2,
        )
        opt.run_exploration_jobs()
        n_explored = len(opt.points)
        opt.run_search_jobs()

        searched = [p["x"] for p in opt.points[n_explored:]]
        assert statistics.median(searched) < 0.5, searched

    def test_search_finds_the_optimum(self):
        opt, _ = make_opt(objective=sphere, explore=8, iterations=3, parallel=4)
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
        opt, _ = make_opt(objective=sphere, explore=8, iterations=4, parallel=4)
        opt.run_exploration_jobs()
        opt.run_search_jobs()
        guided = opt.best_point()[1]

        blind, _ = make_opt(objective=sphere, explore=24, iterations=0)
        blind.run_exploration_jobs()

        assert guided < blind.best_point()[1]

    def test_a_mixed_space_optimizes(self):
        # Float, integer and categorical in one space.
        # The categorical offsets are large,
        # so picking the wrong category cannot look like a good point
        # however well the continuous part is tuned.
        offsets = [0.0, 10.0, 25.0]

        def objective(x, y, cat):
            return {"objective": sphere(x, y)["objective"] + offsets[cat]}

        space = {
            "x": FloatRange(-5.0, 5.0),
            "y": IntRange(-5, 5),
            "cat": CategoricalRange(3),
        }
        opt, _ = make_opt(
            objective=objective, space=space, explore=16, iterations=4, parallel=4
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
        opt, _ = make_opt(objective=lambda x, y: {"objective": float("nan")}, explore=2)
        with pytest.raises(RuntimeError, match="non-finite"):
            opt.run_exploration_jobs()

    def test_an_infinite_objective_raises(self):
        opt, _ = make_opt(objective=lambda x, y: {"objective": float("inf")}, explore=2)
        with pytest.raises(RuntimeError, match="non-finite"):
            opt.run_exploration_jobs()

    def test_a_non_numeric_objective_raises(self):
        opt, _ = make_opt(
            objective=lambda x, y: {"objective": "not a number"}, explore=2
        )
        with pytest.raises(RuntimeError, match="not a float"):
            opt.run_exploration_jobs()

    def test_a_bare_number_is_rejected(self):
        """The old contract returned a float; that must fail loudly, not coerce."""
        opt, _ = make_opt(objective=lambda x, y: 1.0, explore=2)
        with pytest.raises(RuntimeError, match="expected a mapping"):
            opt.run_exploration_jobs()

    def test_a_mapping_without_the_objective_key_is_rejected(self):
        opt, _ = make_opt(objective=lambda x, y: {"loss": 1.0}, explore=2)
        with pytest.raises(RuntimeError, match="no 'objective'"):
            opt.run_exploration_jobs()

    def test_a_configured_key_is_what_the_message_names(self):
        """The default key is not what a run with its own key is missing."""
        opt, _ = make_opt(
            objective=lambda x, y: {"objective": 1.0},
            explore=2,
            objective_key="rmse",
        )
        with pytest.raises(RuntimeError, match="no 'rmse'"):
            opt.run_exploration_jobs()

    def test_the_error_names_the_keys_that_were_returned(self):
        """So a misspelled key is obvious from the message alone."""
        opt, _ = make_opt(objective=lambda x, y: {"objectiv": 1.0}, explore=2)
        with pytest.raises(RuntimeError, match=r"\['objectiv'\]"):
            opt.run_exploration_jobs()

    def test_an_integer_objective_is_accepted(self):
        opt, _ = make_opt(objective=lambda x, y: {"objective": 1}, explore=2)
        opt.run_exploration_jobs()
        assert opt.values == [1.0, 1.0]


class TestObjectiveKey:
    """Which key of the result is modelled is the run's to choose."""

    def test_the_default_key_is_objective(self):
        opt, _ = make_opt()
        assert opt.objective_key == "objective"

    def test_a_configured_key_is_the_one_modelled(self):
        """An evaluation that already reports `loss` is searched as it is."""

        def objective(x, y):
            return {"loss": x * x + y * y, "note": "sphere"}

        opt, _ = make_opt(objective=objective, explore=4, objective_key="loss")
        opt.run_exploration_jobs()

        assert opt.values == [objective(**p)["loss"] for p in opt.points]
        _, value = opt.best_point()
        assert value == min(opt.values)

    def test_the_default_key_is_then_just_another_recorded_key(self):
        """Only the configured key is modelled; the rest are carried along."""

        def objective(x, y):
            return {"loss": x * x + y * y, "objective": 999.0}

        opt, _ = make_opt(objective=objective, explore=2, objective_key="loss")
        opt.run_exploration_jobs()

        assert 999.0 not in opt.values
        assert opt.best_output()["objective"] == 999.0

    def test_the_search_models_the_configured_key_too(self):
        """Not just exploration: every round reads the same key."""

        def objective(x, y):
            return {"score": x * x + y * y}

        opt, _ = make_opt(
            objective=objective,
            explore=4,
            iterations=1,
            parallel=2,
            objective_key="score",
        )
        opt.run_exploration_jobs()
        opt.run_search_jobs()

        assert len(opt.values) == 6
        assert opt.values == [objective(**p)["score"] for p in opt.points]


# --------------------------------------------------------------------------
# best_point
# --------------------------------------------------------------------------


class TestBestPoint:
    def test_returns_the_minimum(self):
        opt, _ = make_opt(objective=sphere, explore=8)
        opt.run_exploration_jobs()
        params, value = opt.best_point()
        assert value == min(opt.values)
        assert value == sphere(**params)["objective"]

    def test_best_output_is_the_whole_mapping(self):
        opt, _ = make_opt(objective=sphere, explore=8)
        opt.run_exploration_jobs()
        params, value = opt.best_point()

        output = opt.best_output()
        assert output == sphere(**params)
        assert output["objective"] == value
        assert output["note"] == "sphere", "keys beyond the objective are kept"

    def test_every_output_is_recorded(self):
        opt, _ = make_opt(objective=sphere, explore=8)
        opt.run_exploration_jobs()

        assert len(opt.outputs) == len(opt.values) == len(opt.points)
        for params, value, output in zip(opt.points, opt.values, opt.outputs):
            assert output == sphere(**params)
            assert output["objective"] == value

    def test_a_stored_output_is_a_copy(self):
        """Mutating what the objective returned must not rewrite the record."""
        returned = {}

        def objective(x, y):
            nonlocal returned
            returned = {"objective": x * x + y * y, "trace": [1, 2, 3]}
            return returned

        opt, _ = make_opt(objective=objective, explore=2)
        opt.run_exploration_jobs()
        recorded = dict(opt.outputs[-1])

        returned["objective"] = -999.0
        returned["trace"] = []

        assert opt.outputs[-1] == recorded
        assert opt.best_output() != returned

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
        opt, _ = make_opt(objective=sphere, explore=8, iterations=2, parallel=4)
        opt.run_exploration_jobs()
        after_exploration = opt.best_point()[1]
        opt.run_search_jobs()
        assert opt.best_point()[1] <= after_exploration


# --------------------------------------------------------------------------
# The real executor
# --------------------------------------------------------------------------


class TestRealExecutor:
    """One end-to-end run against the real queue, executor and worker.

    Everything above assumes `submit()` returns a Task
    whose `output` `wait()` fills in;
    this is where that assumption meets the real implementation.
    """

    @pytest.fixture(autouse=True)
    def _pilot_jobs(self, pilot_jobs):
        pilot_jobs("cpu")

    def test_optimizes_through_a_real_worker(
        self, executor, ds_service_address, tmp_path
    ):
        explore, iterations, parallel = 4, 2, 2
        # One fit-and-propose task per round on top of the evaluations,
        # and both kinds go to the one queue this worker serves ---
        # which is also what pins that a real worker can run the fit at all.
        total = explore + iterations * (parallel + 1)

        opt = BayesOptBotorch(
            "e2e",
            BOX_2D,
            sphere,
            executor,
            "cpu",
            "cpu",
            explore,
            parallel,
            SEED,
            min_search_iterations=iterations,
            max_search_iterations=iterations,
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
        assert len(opt.values) == explore + iterations * parallel

        params, value = opt.best_point()
        assert value == min(opt.values)
        assert math.isclose(value, sphere(**params)["objective"])
        # The whole mapping survives the round trip through the real queue,
        # not just the number the model was fit on.
        assert opt.best_output() == {"objective": value, "note": "sphere"}
        assert value < BOX_2D["x"].max ** 2 + BOX_2D["y"].max ** 2

    def test_a_raising_objective_surfaces_from_a_real_worker(
        self, executor, ds_service_address, tmp_path
    ):
        def boom(x, y):
            raise RuntimeError("worker exploded")

        opt = BayesOptBotorch(
            "e2e-fail",
            BOX_2D,
            boom,
            executor,
            "cpu",
            "cpu",
            2,
            1,
            SEED,
            min_search_iterations=0,
            max_search_iterations=0,
        )

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
