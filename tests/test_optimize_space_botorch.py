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
a patched `osb.optimize_acqf` reaches the task
because the task ran in this process.
`TestOptimizerQueue` covers what only shows up once it does not:
which queue the fit went to,
and what a failure on the far end reports.

Observations come from a results file, as they do in a real run:
`explored()` runs a real `ExploreSpaceSobolQMC` sweep and saves it,
so the file the optimizer reads is the file the explorer writes.

`TestRealExecutor` keeps the stand-in honest.
It runs a whole exploration and optimization through the real executor,
the real ds-service queue and a real worker, the fit included,
pinning the two-call contract against the real implementation.

The four tests in `TestSearchBehaviour` assert behaviour of the search
rather than its bookkeeping,
and are what would catch the objective's sign being flipped
--- botorch maximizes and this optimizer minimizes.
They are stochastic
(torch's global RNG is left unseeded, so each run is a fresh sample),
and their thresholds come from measured spreads.
All four use unimodal objectives:
an earlier multimodal version of the random-search comparison
lost 1 run in 10.
"""

from __future__ import annotations

import gzip
import math
import pickle
import re
import statistics
import threading
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("botorch")

import torch  # noqa: E402
from botorch.acquisition import qLogNoisyExpectedImprovement  # noqa: E402

from slurm_workflows import optimize_space_botorch as osb  # noqa: E402
from slurm_workflows.optimize_space_botorch import (  # noqa: E402
    OptimizationTask,
    OptimizeSpaceBotorch,
)
from slurm_workflows.explore_space import (  # noqa: E402
    ExplorationTask,
    ExploreSpaceSobolQMC,
    load_results,
)
from slurm_workflows.search_space import (  # noqa: E402
    CategoricalRange,
    FloatRange,
    IntRange,
)
from slurm_workflows.slurm_pilot_executor import (  # noqa: E402
    RaiseOnError,
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
        self.batch_sizes: list[int] = []

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

    def wait(
        self,
        tasks,
        desc=None,
        unit="task",
        raise_on_error=RaiseOnError.RAISE_ON_FIRST_ERROR,
    ) -> None:
        """The real `wait`'s contract, minus the waiting.

        The callables ran in `submit`,
        so all this has left to do is the raise policy,
        which is what the optimizer relies on
        to turn a failed evaluation into an exception.
        """
        self.waits.append(desc)
        self.batch_sizes.append(len(tasks))

        if raise_on_error is RaiseOnError.RAISE_NEVER:
            return

        failed = [t for t in tasks if isinstance(t.output, RemoteExecutionError)]
        if failed:
            raise RuntimeError(f"{len(failed)} of {len(tasks)} tasks did not succeed")

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


def constant(x, y):
    """Never improves, so every round stalls."""
    return {"objective": 1.0}


def benign(**params):
    """An objective for the *prior* file, whatever the space.

    Tests of a broken objective still need observations to start from,
    and the file only carries numbers: what ranked them is not recorded,
    so the search's own key and objective are free to differ from this.
    """
    return {"objective": float(sum(params.values()))}


BOX_2D = {"x": FloatRange(-5.0, 5.0), "y": FloatRange(-5.0, 5.0)}


SEED = 20260730


def explored(
    tmp_path: Path,
    name: str = "test",
    space=None,
    objective=sphere,
    points: int = 4,
    seed: int = SEED,
    filename: str | None = None,
) -> Path:
    """A results file, written by a real exploration sweep.

    The optimizer starts from what `ExploreSpaceSobolQMC.save` wrote,
    so the tests start from that too rather than from a hand-built file.
    """
    space = BOX_2D if space is None else space
    sweep = ExploreSpaceSobolQMC(
        [ExplorationTask(name, space, objective, "cpu", points, seed)],
        as_executor(LocalExecutor()),
    )
    sweep.run_exploration_jobs()

    path = tmp_path / (filename or f"{name}-explore.pkl.gz")
    sweep.save(path)
    return path


def make_task(
    name: str = "test",
    space=None,
    objective=sphere,
    parallel: int | None = 4,
    iterations: int = 2,
    objective_queue: str | list[str] = "cpu",
    optimizer_queue: str | list[str] = "opt",
    **extra,
) -> OptimizationTask:
    """One optimization task, with the test defaults filled in.

    `iterations` pins the round count by setting both search bounds to it,
    which switches early stopping off:
    a stall can only end the search at a round at or above the floor,
    and with floor == ceiling that is the round the loop ends on anyway,
    so the search runs exactly that many rounds.
    Tests that are *about* early stopping pass the bounds themselves.
    """
    space = BOX_2D if space is None else space

    settings: dict[str, Any] = {
        "min_search_iterations": iterations,
        "max_search_iterations": iterations,
    }
    for field in (
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
        if field in extra:
            settings[field] = extra.pop(field)

    return OptimizationTask(
        name=name,
        space=space,
        objective=objective,
        objective_queue=objective_queue,
        optimizer_queue=optimizer_queue,
        search_parallelism=parallel,
        extra_objective_kwargs=extra,
        **settings,
    )


def make_opt(
    tmp_path: Path,
    objective=sphere,
    space=None,
    explore: int = 4,
    files: list[Path] | None = None,
    prior_objective=None,
    **task_kwargs,
):
    """An optimizer over one task, wired to a fresh LocalExecutor.

    The task's observations come from a file an exploration sweep wrote,
    unless the test supplies its own `files`.
    The sweep runs the task's own objective, so the file and the search
    measure the same thing --- except where the test is *about* an objective
    that cannot be evaluated, which passes `prior_objective=benign`.
    """
    space = BOX_2D if space is None else space
    task = make_task(objective=objective, space=space, **task_kwargs)

    if files is None:
        files = [
            explored(
                tmp_path,
                task.name,
                space,
                prior_objective or objective,
                explore,
                filename="prior.pkl.gz",
            )
        ]

    executor = LocalExecutor()
    return OptimizeSpaceBotorch(list([task]), as_executor(executor), files), executor


def rounds_run(opt, name: str = "test") -> int:
    """How many search rounds one task actually ran."""
    parallelism = opt._task(name).search_parallelism
    return len(opt.results[name].values) // parallelism


# --------------------------------------------------------------------------
# Construction and the observations it starts from
# --------------------------------------------------------------------------


class TestConstruction:
    def test_it_starts_from_the_observations_in_the_files(self, tmp_path):
        opt, _ = make_opt(tmp_path, explore=8)

        assert opt.num_observations("test") == 8
        assert opt.results["test"].values == [], "no round has run yet"

    def test_observations_are_merged_across_files(self, tmp_path):
        first = explored(tmp_path, points=4, seed=1, filename="a.pkl.gz")
        second = explored(tmp_path, points=8, seed=2, filename="b.pkl.gz")

        opt, _ = make_opt(tmp_path, files=[first, second])

        assert opt.num_observations("test") == 12

    def test_a_task_with_no_observations_is_rejected(self, tmp_path):
        other = explored(tmp_path, name="somebody-else")

        with pytest.raises(RuntimeError, match="no observations"):
            make_opt(tmp_path, files=[other])

    def test_the_unit_points_are_recomputed_against_the_space(self, tmp_path):
        """The file carries the parameters; only the space can place them."""
        opt, _ = make_opt(tmp_path, explore=4)

        prior = opt.prior["test"]
        for params, unit in zip(prior.points, prior.unit_points):
            assert math.isclose(unit[0], (params["x"] + 5.0) / 10.0)

    def test_a_file_from_another_space_is_reported(self, tmp_path):
        elsewhere = explored(
            tmp_path,
            space={"a": FloatRange(0.0, 1.0)},
            objective=lambda a: {"objective": a},
        )

        with pytest.raises(RuntimeError, match="do not match the search space"):
            make_opt(tmp_path, files=[elsewhere])

    def test_no_tasks_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no optimization tasks"):
            OptimizeSpaceBotorch([], as_executor(LocalExecutor()), [])

    def test_duplicate_task_names_are_rejected(self, tmp_path):
        path = explored(tmp_path)

        with pytest.raises(ValueError, match="unique"):
            OptimizeSpaceBotorch(
                [make_task(), make_task()], as_executor(LocalExecutor()), [path]
            )

    def test_the_runs_parallelism_fills_in_for_a_task_without_one(self, tmp_path):
        path = explored(tmp_path)

        opt = OptimizeSpaceBotorch(
            [make_task(parallel=None)], as_executor(LocalExecutor()), [path], 8
        )

        assert opt.tasks[0].search_parallelism == 8

    def test_a_parallelism_from_neither_is_rejected(self, tmp_path):
        path = explored(tmp_path)

        with pytest.raises(ValueError, match="search_parallelism"):
            OptimizeSpaceBotorch(
                [make_task(parallel=None)], as_executor(LocalExecutor()), [path]
            )

    def test_an_empty_space_is_rejected(self, tmp_path):
        path = explored(tmp_path)

        with pytest.raises(ValueError, match="empty"):
            OptimizeSpaceBotorch(
                [make_task(space={})], as_executor(LocalExecutor()), [path]
            )

    def test_extra_kwargs_may_not_shadow_a_parameter(self, tmp_path):
        path = explored(tmp_path)

        with pytest.raises(ValueError, match="shadow"):
            OptimizeSpaceBotorch(
                [make_task(x=1.0)], as_executor(LocalExecutor()), [path]
            )

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"parallel": 0}, "search_parallelism"),
            ({"min_search_iterations": -1}, "min_search_iterations"),
            ({"min_search_iterations": 5, "max_search_iterations": 2}, "must be >="),
            ({"patience": 0}, "patience"),
            ({"min_improvement": -0.1}, "min_improvement"),
            ({"num_restarts": 0}, "num_restarts"),
            ({"raw_samples": 0}, "raw_samples"),
            ({"mc_samples": 0}, "mc_samples"),
            ({"acqf_timeout_s": 0.0}, "acqf_timeout_s"),
        ],
    )
    def test_a_nonsensical_setting_is_rejected(self, tmp_path, kwargs, match):
        path = explored(tmp_path)
        iterations = kwargs.pop("iterations", 2)

        with pytest.raises(ValueError, match=match):
            OptimizeSpaceBotorch(
                [make_task(iterations=iterations, **kwargs)],
                as_executor(LocalExecutor()),
                [path],
            )

    def test_the_callers_task_is_left_alone(self, tmp_path):
        path = explored(tmp_path)
        original = make_task(parallel=None)

        opt = OptimizeSpaceBotorch([original], as_executor(LocalExecutor()), [path], 8)

        assert original.search_parallelism is None
        assert opt.tasks[0].search_parallelism == 8

    def test_an_unknown_task_name_says_what_there_is(self, tmp_path):
        opt, _ = make_opt(tmp_path)

        with pytest.raises(KeyError, match="'test'"):
            opt.best_point("nope")

    def test_the_dimension_is_the_tasks_own(self, tmp_path):
        opt, _ = make_opt(tmp_path)

        assert opt.dim("test") == 2


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------


class TestEarlyStopping:
    def test_stops_after_patience_stalled_rounds(self, tmp_path):
        opt, _ = make_opt(
            tmp_path,
            objective=constant,
            explore=4,
            parallel=2,
            min_search_iterations=2,
            max_search_iterations=30,
            patience=3,
        )
        opt.run_search_jobs()

        # Stalls count from the first round, so patience alone decides
        # once it is the larger of the two.
        assert rounds_run(opt) == 3

    def test_the_floor_runs_even_when_nothing_improves(self, tmp_path):
        """Patience below the floor cannot cut the search short."""
        opt, _ = make_opt(
            tmp_path,
            objective=constant,
            explore=4,
            parallel=2,
            min_search_iterations=6,
            max_search_iterations=30,
            patience=1,
        )
        opt.run_search_jobs()

        assert rounds_run(opt) == 6

    def test_stalls_below_the_floor_are_carried_past_it(self, tmp_path):
        """The floor holds off the stop, not the counting."""
        opt, _ = make_opt(
            tmp_path,
            objective=constant,
            explore=4,
            parallel=2,
            min_search_iterations=3,
            max_search_iterations=30,
            patience=2,
        )
        opt.run_search_jobs()

        assert rounds_run(opt) == 3

    def test_the_ceiling_stops_a_search_that_keeps_improving(self, tmp_path):
        opt, _ = make_opt(
            tmp_path,
            objective=sphere,
            explore=4,
            parallel=2,
            min_search_iterations=0,
            max_search_iterations=4,
            patience=100,
        )
        opt.run_search_jobs()

        assert rounds_run(opt) == 4

    def test_an_improving_round_resets_the_counter(self, tmp_path, monkeypatch):
        """Patience bounds a *run* of bad rounds, not their total."""
        seen = []

        def improved(self, task, previous, current):
            # stall, stall, improve, stall, stall, stall -> stop at 6
            pattern = [False, False, True, False, False, False]
            seen.append(len(seen))
            return pattern[min(len(seen) - 1, len(pattern) - 1)]

        monkeypatch.setattr(OptimizeSpaceBotorch, "_improved_enough", improved)
        opt, _ = make_opt(
            tmp_path,
            objective=sphere,
            explore=4,
            parallel=2,
            min_search_iterations=0,
            max_search_iterations=30,
            patience=3,
        )
        opt.run_search_jobs()

        assert rounds_run(opt) == 6

    def test_a_zero_floor_leaves_patience_in_charge(self, tmp_path):
        opt, _ = make_opt(
            tmp_path,
            objective=constant,
            explore=4,
            parallel=2,
            min_search_iterations=0,
            max_search_iterations=30,
            patience=2,
        )
        opt.run_search_jobs()

        assert rounds_run(opt) == 2

    def test_the_progress_line_counts_towards_the_real_stop(self, tmp_path, capsys):
        """A floor outlasting patience must not print a ratio past its own end."""
        opt, _ = make_opt(
            tmp_path,
            objective=constant,
            explore=4,
            parallel=2,
            min_search_iterations=5,
            max_search_iterations=30,
            patience=3,
        )
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

    def test_it_says_why_it_stopped(self, tmp_path, capsys):
        opt, _ = make_opt(
            tmp_path,
            objective=constant,
            explore=4,
            parallel=2,
            min_search_iterations=0,
            max_search_iterations=30,
            patience=2,
        )
        opt.run_search_jobs()

        out = capsys.readouterr().out
        assert "stopping after 2 rounds" in out
        assert "5%" in out

    def test_the_ceiling_says_so_too(self, tmp_path, capsys):
        opt, _ = make_opt(
            tmp_path,
            objective=sphere,
            explore=4,
            parallel=2,
            min_search_iterations=0,
            max_search_iterations=2,
            patience=100,
        )
        opt.run_search_jobs()

        assert "the ceiling on this search" in capsys.readouterr().out


class TestImprovementTest:
    """`_improved_enough` decides every stall, so its edges matter."""

    @pytest.fixture
    def improved(self, tmp_path):
        opt, _ = make_opt(tmp_path, min_improvement=0.05)
        task = opt.tasks[0]
        return lambda previous, current: opt._improved_enough(task, previous, current)

    def test_a_big_enough_drop_counts(self, improved):
        assert improved(1.0, 0.94)

    def test_a_drop_below_the_threshold_does_not(self, improved):
        assert not improved(1.0, 0.96)

    def test_the_threshold_is_inclusive(self, improved):
        assert improved(1.0, 0.95)

    def test_no_change_is_not_improvement(self, improved):
        assert not improved(1.0, 1.0)

    def test_getting_worse_is_not_improvement(self, improved):
        assert not improved(1.0, 2.0)

    def test_it_is_relative_not_absolute(self, improved):
        """The same absolute step is decisive at one scale and noise at another."""
        assert improved(1.0, 0.9)
        assert not improved(1000.0, 999.9)

    def test_a_negative_incumbent_uses_its_magnitude(self, improved):
        # -10 -> -11 is a 10% improvement; -10 -> -10.1 is 1%.
        assert improved(-10.0, -11.0)
        assert not improved(-10.0, -10.1)

    def test_a_zero_incumbent_accepts_any_decrease(self, improved):
        """Zero has no magnitude to take a fraction of."""
        assert improved(0.0, -1e-9)
        assert not improved(0.0, 0.0)


# --------------------------------------------------------------------------
# The acquisition
# --------------------------------------------------------------------------


class TestAcquisition:
    """One acquisition per round, asserted without paying for a real optimization."""

    @pytest.fixture
    def record(self, monkeypatch):
        calls: list[tuple[str, int]] = []

        def fake_optimize_acqf(acqf, **kwargs):
            q = kwargs["q"]
            calls.append((type(acqf).__name__, q))
            dim = kwargs["bounds"].shape[1]
            return torch.rand(q, dim, dtype=osb.DTYPE), None

        monkeypatch.setattr(osb, "optimize_acqf", fake_optimize_acqf)
        return calls

    def test_one_call_per_round_for_the_whole_batch(self, tmp_path, record):
        opt, _ = make_opt(tmp_path, explore=4, iterations=1, parallel=4)
        opt.run_search_jobs()
        assert record == [(qLogNoisyExpectedImprovement.__name__, 4)]

    def test_an_odd_batch_is_not_split(self, tmp_path, record):
        opt, _ = make_opt(tmp_path, explore=4, iterations=1, parallel=3)
        opt.run_search_jobs()
        assert record == [(qLogNoisyExpectedImprovement.__name__, 3)]

    def test_a_parallelism_of_one_still_asks_for_one_point(self, tmp_path, record):
        opt, _ = make_opt(tmp_path, explore=4, iterations=2, parallel=1)
        opt.run_search_jobs()
        assert record == [(qLogNoisyExpectedImprovement.__name__, 1)] * 2

    def test_every_round_asks_for_the_full_parallelism(self, tmp_path, record):
        """The budget is rounds, so no round is short."""
        opt, _ = make_opt(tmp_path, explore=4, iterations=3, parallel=3)
        opt.run_search_jobs()
        assert record == [(qLogNoisyExpectedImprovement.__name__, 3)] * 3

    def test_a_timeout_is_passed_to_the_optimizer(self, tmp_path, monkeypatch):
        """Unbounded, one round can outlast the batch it is choosing points for."""
        timeouts = []

        def fake_optimize_acqf(acqf, **kwargs):
            timeouts.append(kwargs.get("timeout_sec"))
            q, dim = kwargs["q"], kwargs["bounds"].shape[1]
            return torch.rand(q, dim, dtype=osb.DTYPE), None

        monkeypatch.setattr(osb, "optimize_acqf", fake_optimize_acqf)
        opt, _ = make_opt(tmp_path, explore=4, iterations=2, parallel=2)
        opt.run_search_jobs()

        # Against the task's own setting rather than a literal:
        # the default lives in the dataclass,
        # and pinning its value here would only mean editing this test
        # whenever it is retuned.
        assert timeouts == [opt.tasks[0].acqf_timeout_s] * 2
        assert timeouts[0] is not None

    def test_a_timed_out_proposal_is_still_usable(self, tmp_path):
        """The limit degrades the proposal; it must not break the round."""
        opt, _ = make_opt(
            tmp_path, explore=4, iterations=1, parallel=3, acqf_timeout_s=0.001
        )
        opt.run_search_jobs()

        searched = opt.results["test"].points
        assert len(searched) == 3
        for params in searched:
            for name, value in params.items():
                assert math.isfinite(value)
                assert BOX_2D[name].min <= value <= BOX_2D[name].max

    def test_the_sampler_is_passed_explicitly(self, tmp_path, monkeypatch):
        """Left to botorch the default is larger, and every round pays for it."""
        shapes = []
        real_acqf = osb.qLogNoisyExpectedImprovement

        def spy(model, x_baseline, *a, sampler=None, **kw):
            shapes.append(None if sampler is None else tuple(sampler.sample_shape))
            return real_acqf(model, x_baseline, *a, sampler=sampler, **kw)

        monkeypatch.setattr(osb, "qLogNoisyExpectedImprovement", spy)
        opt, _ = make_opt(tmp_path, explore=4, iterations=2, parallel=2)
        opt.run_search_jobs()

        assert shapes == [(opt.tasks[0].mc_samples,)] * 2

    def test_the_baseline_is_every_point_measured_so_far(self, tmp_path, monkeypatch):
        """qLogNEI reads its incumbent off these, so they must be up to date."""
        baselines: list[int] = []
        real_acqf = osb.qLogNoisyExpectedImprovement

        def spy(model, x_baseline, *a, **kw):
            baselines.append(len(x_baseline))
            return real_acqf(model, x_baseline, *a, **kw)

        monkeypatch.setattr(osb, "qLogNoisyExpectedImprovement", spy)
        opt, _ = make_opt(tmp_path, explore=4, iterations=2, parallel=2)
        opt.run_search_jobs()

        # Four points from the file, then those plus the first round's two.
        assert baselines == [4, 6]

    def test_each_round_reports_how_long_proposing_took(self, tmp_path, capsys):
        opt, _ = make_opt(tmp_path, explore=4, iterations=3, parallel=2)
        opt.run_search_jobs()

        out = capsys.readouterr().out

        proposals = re.findall(r"proposed (\d+) points in ([0-9.]+)s", out)
        assert len(proposals) == 3
        assert [int(n) for n, _ in proposals] == [2, 2, 2]
        assert all(float(t) >= 0.0 for _, t in proposals)

    def test_the_best_so_far_is_reported_after_every_round(self, tmp_path, capsys):
        opt, _ = make_opt(tmp_path, explore=4, iterations=3, parallel=2)
        opt.run_search_jobs()

        out = capsys.readouterr().out
        counts = [int(n) for n in re.findall(r"best after (\d+) points", out)]

        # Once per round, each covering everything measured up to that point,
        # the file's four included.
        assert counts == [6, 8, 10]

    def test_reported_parameters_keep_their_type(self, tmp_path, capsys):
        """An int parameter must not be printed as a float."""
        space = {"x": FloatRange(0.0, 1.0), "n": IntRange(1, 8)}
        objective = lambda x, n: {"objective": x + n}  # noqa: E731
        opt, _ = make_opt(
            tmp_path,
            objective=objective,
            space=space,
            explore=4,
            iterations=1,
            parallel=2,
        )
        opt.run_search_jobs()

        out = capsys.readouterr().out
        params, _ = opt.best_point("test")
        assert f"n={params['n']}" in out, out
        assert f"n={float(params['n'])}" not in out

    def test_each_fit_reports_its_size_and_duration(self, tmp_path, capsys):
        opt, _ = make_opt(tmp_path, explore=4, iterations=2, parallel=2)
        opt.run_search_jobs()

        out = capsys.readouterr().out

        assert out.count("fitting GP on") == 2
        assert out.count("GP fit took") == 2

        # The count is the observations the fit actually sees:
        # the file's four, then those plus the first round's two.
        assert "test: fitting GP on 4 points" in out
        assert "test: fitting GP on 6 points" in out

        durations = re.findall(r"GP fit took ([0-9.]+)s", out)
        assert len(durations) == 2
        assert all(float(d) >= 0.0 for d in durations)

    def test_the_size_is_reported_before_the_fit_runs(
        self, tmp_path, monkeypatch, capsys
    ):
        """The count has to be visible even if the fit then hangs or dies."""

        def explode(mll, **kw):
            raise RuntimeError("fit blew up")

        opt, _ = make_opt(tmp_path, explore=4, iterations=1, parallel=2)
        monkeypatch.setattr(osb, "fit_gpytorch_mll", explode)

        with pytest.raises(RuntimeError, match="fit blew up"):
            opt.run_search_jobs()

        out = capsys.readouterr().out
        assert "test: fitting GP on 4 points" in out
        assert "GP fit took" not in out, "no duration for a fit that never finished"

    def test_the_model_is_refit_every_round(self, tmp_path, record, monkeypatch):
        fits = []
        real_fit = osb.fit_gpytorch_mll
        monkeypatch.setattr(
            osb,
            "fit_gpytorch_mll",
            lambda mll, **kw: (fits.append(1), real_fit(mll, **kw))[1],
        )
        opt, _ = make_opt(tmp_path, explore=4, iterations=3, parallel=2)
        opt.run_search_jobs()
        assert len(fits) == 3


# --------------------------------------------------------------------------
# Where the work is sent
# --------------------------------------------------------------------------


class TestOptimizerQueue:
    """The fit is a task too, and it goes somewhere else."""

    def test_the_fit_goes_to_the_optimizer_queue(self, tmp_path):
        opt, executor = make_opt(tmp_path, explore=2, iterations=2, parallel=2)
        opt.run_search_jobs()

        # Each round is one fit followed by that round's evaluations.
        assert executor.queues == ["opt", "cpu", "cpu"] * 2

    def test_the_fit_accepts_a_list_of_queues(self, tmp_path):
        opt, executor = make_opt(
            tmp_path, explore=2, iterations=1, parallel=1, optimizer_queue=["a", "b"]
        )
        opt.run_search_jobs()
        assert executor.queues == [["a", "b"], "cpu"]

    def test_one_queue_may_serve_both(self, tmp_path):
        """Nothing deadlocks: the two kinds are never in flight together."""
        opt, executor = make_opt(
            tmp_path, explore=2, iterations=1, parallel=2, optimizer_queue="cpu"
        )
        opt.run_search_jobs()
        assert executor.queues == ["cpu"] * 3
        assert len(opt.results["test"].values) == 2

    def test_the_task_carries_its_own_tuning_to_the_worker(self, tmp_path, monkeypatch):
        """How the task was configured has to decide, not what the worker has."""
        seen = self._record_kwargs(monkeypatch)

        opt, _ = make_opt(
            tmp_path,
            explore=4,
            iterations=1,
            parallel=2,
            num_restarts=3,
            raw_samples=7,
            mc_samples=11,
            acqf_timeout_s=1.5,
        )
        opt.run_search_jobs()

        assert seen == [
            {
                "num_restarts": 3,
                "raw_samples": 7,
                "mc_samples": 11,
                "timeout_s": 1.5,
            }
        ]

    def test_an_unconfigured_task_carries_its_defaults(self, tmp_path, monkeypatch):
        """The defaults travel too --- the worker is told, never left to guess."""
        seen = self._record_kwargs(monkeypatch)

        opt, _ = make_opt(tmp_path, explore=4, iterations=1, parallel=2)
        opt.run_search_jobs()

        task = opt.tasks[0]
        assert seen == [
            {
                "num_restarts": task.num_restarts,
                "raw_samples": task.raw_samples,
                "mc_samples": task.mc_samples,
                "timeout_s": task.acqf_timeout_s,
            }
        ]

    @staticmethod
    def _record_kwargs(monkeypatch) -> list[dict]:
        """Record what each fit was asked for, without paying for one."""
        seen: list[dict] = []

        def fake(unit_points, values, batch, **kwargs):
            seen.append(kwargs)
            return {
                osb.CANDIDATES_KEY: [[0.5] * len(unit_points[0]) for _ in range(batch)],
                osb.FIT_SECONDS_KEY: 0.0,
                osb.PROPOSE_SECONDS_KEY: 0.0,
            }

        monkeypatch.setattr(osb, "fit_and_propose", fake)
        return seen

    def test_a_fit_that_fails_names_the_task_queue_and_botorch(
        self, tmp_path, monkeypatch
    ):
        """The traceback is in a worker log the driver never reads."""

        def explode(mll, **kw):
            raise RuntimeError("No module named 'botorch'")

        opt, _ = make_opt(tmp_path, explore=4, iterations=1, parallel=2)
        monkeypatch.setattr(osb, "fit_gpytorch_mll", explode)

        with pytest.raises(RuntimeError, match="'opt'.*botorch") as excinfo:
            opt.run_search_jobs()
        assert "search round 1" in str(excinfo.value)
        assert "test on queue" in str(excinfo.value)

    def test_an_unusable_result_is_reported_rather_than_unpacked(
        self, tmp_path, monkeypatch
    ):
        """The case is a worker running a different slurm-workflows."""
        monkeypatch.setattr(osb, "fit_and_propose", lambda *a, **kw: {"points": []})

        opt, _ = make_opt(tmp_path, explore=4, iterations=1, parallel=2)

        with pytest.raises(RuntimeError, match="no 'candidates'"):
            opt.run_search_jobs()

    def test_a_result_missing_only_the_timings_is_reported_too(
        self, tmp_path, monkeypatch
    ):
        """Every key the driver goes on to read, not just the candidates."""
        monkeypatch.setattr(
            osb,
            "fit_and_propose",
            lambda unit_points, values, batch, **kw: {
                osb.CANDIDATES_KEY: [[0.5] * len(unit_points[0])] * batch
            },
        )

        opt, _ = make_opt(tmp_path, explore=4, iterations=1, parallel=2)

        with pytest.raises(RuntimeError, match="'fit_seconds', 'propose_seconds'"):
            opt.run_search_jobs()

    def test_a_batch_that_is_not_the_full_width_is_rejected(
        self, tmp_path, monkeypatch
    ):
        """A short round would otherwise pass as a normal one."""
        monkeypatch.setattr(
            osb,
            "fit_and_propose",
            lambda unit_points, values, batch, **kw: {
                osb.CANDIDATES_KEY: [[0.5] * len(unit_points[0])] * (batch - 1),
                osb.FIT_SECONDS_KEY: 0.0,
                osb.PROPOSE_SECONDS_KEY: 0.0,
            },
        )

        opt, _ = make_opt(tmp_path, explore=4, iterations=1, parallel=3)

        with pytest.raises(RuntimeError, match=r"proposed 2 points.*not the 3"):
            opt.run_search_jobs()

    def test_the_fit_sees_every_point_measured_so_far(self, tmp_path, monkeypatch):
        """It is given the observations, not a handle to the driver's state."""
        sizes = []
        real = osb.fit_and_propose

        def spy(unit_points, values, batch, **kwargs):
            sizes.append((len(unit_points), len(values)))
            return real(unit_points, values, batch, **kwargs)

        monkeypatch.setattr(osb, "fit_and_propose", spy)

        opt, _ = make_opt(tmp_path, explore=4, iterations=2, parallel=2)
        opt.run_search_jobs()

        assert sizes == [(4, 4), (6, 6)]


# --------------------------------------------------------------------------
# Several spaces at once
# --------------------------------------------------------------------------


class TestSeveralSpacesAtOnce:
    @pytest.fixture
    def two(self, tmp_path):
        """Two tasks over one results file, and the executor they share."""
        sweep = ExploreSpaceSobolQMC(
            [
                ExplorationTask("a", BOX_2D, sphere, "cpu", 4, SEED),
                ExplorationTask("b", BOX_2D, sphere, "cpu", 4, SEED + 1),
            ],
            as_executor(LocalExecutor()),
        )
        sweep.run_exploration_jobs()
        path = tmp_path / "explore.pkl.gz"
        sweep.save(path)

        executor = LocalExecutor()
        opt = OptimizeSpaceBotorch(
            [
                make_task(name="a", iterations=2, parallel=2),
                make_task(name="b", iterations=2, parallel=3, optimizer_queue="opt2"),
            ],
            as_executor(executor),
            [path],
        )
        return opt, executor

    def test_every_task_runs(self, two):
        opt, _ = two

        opt.run_search_jobs()

        assert len(opt.results["a"].values) == 4
        assert len(opt.results["b"].values) == 6

    def test_a_round_is_one_batch_of_fits_and_one_of_evaluations(self, two):
        """The point of running them together: two waits a round, not four."""
        opt, executor = two

        opt.run_search_jobs()

        # Per round: both fits, then both batches of points.
        assert executor.batch_sizes == [2, 5, 2, 5]
        assert len(executor.waits) == 4

    def test_each_task_uses_its_own_queues(self, two):
        opt, executor = two

        opt.run_search_jobs()

        assert executor.queues[:2] == ["opt", "opt2"]
        assert executor.queues[2:7] == ["cpu"] * 5

    def test_a_task_that_stops_early_leaves_the_others_running(self, tmp_path):
        sweep = ExploreSpaceSobolQMC(
            [
                ExplorationTask("short", BOX_2D, constant, "cpu", 4, SEED),
                ExplorationTask("long", BOX_2D, sphere, "cpu", 4, SEED),
            ],
            as_executor(LocalExecutor()),
        )
        sweep.run_exploration_jobs()
        path = tmp_path / "explore.pkl.gz"
        sweep.save(path)

        opt = OptimizeSpaceBotorch(
            [
                make_task(
                    name="short",
                    objective=constant,
                    parallel=2,
                    min_search_iterations=0,
                    max_search_iterations=10,
                    patience=1,
                ),
                make_task(
                    name="long",
                    objective=sphere,
                    parallel=2,
                    min_search_iterations=4,
                    max_search_iterations=4,
                    patience=100,
                ),
            ],
            as_executor(LocalExecutor()),
            [path],
        )

        opt.run_search_jobs()

        assert rounds_run(opt, "short") == 1
        assert rounds_run(opt, "long") == 4

    def test_a_broken_task_names_itself(self, tmp_path):
        def boom(x, y):
            raise RuntimeError("worker exploded")

        sweep = ExploreSpaceSobolQMC(
            [
                ExplorationTask("fine", BOX_2D, sphere, "cpu", 4, SEED),
                ExplorationTask("broken", BOX_2D, sphere, "cpu", 4, SEED),
            ],
            as_executor(LocalExecutor()),
        )
        sweep.run_exploration_jobs()
        path = tmp_path / "explore.pkl.gz"
        sweep.save(path)

        opt = OptimizeSpaceBotorch(
            [
                make_task(name="fine", iterations=1, parallel=2),
                make_task(name="broken", objective=boom, iterations=1, parallel=2),
            ],
            as_executor(LocalExecutor()),
            [path],
        )

        with pytest.raises(RuntimeError, match=r"failed during .* of \['broken'\]"):
            opt.run_search_jobs()


# --------------------------------------------------------------------------
# Saving and resuming
# --------------------------------------------------------------------------


class TestPartialFailure:
    """One bad point must not cost a whole round, across every task."""

    @staticmethod
    def fails_at(threshold: float):
        """An objective that raises on the points past `threshold`."""

        def objective(x, y):
            if x > threshold:
                raise ValueError("objective blew up")
            return {"objective": x * x + y * y}

        return objective

    def test_the_points_that_came_back_are_kept(self, tmp_path):
        opt, _ = make_opt(
            tmp_path,
            objective=self.fails_at(0.0),
            prior_objective=sphere,
            explore=8,
            iterations=1,
            parallel=8,
        )

        with pytest.raises(RuntimeError, match="objective evaluations failed"):
            opt.run_search_jobs()

        result = opt.results["test"]
        assert result.values, "the successful points were thrown away"
        assert all(p["x"] <= 0.0 for p in result.points)
        assert len(result.points) == len(result.values) == len(result.outputs)

    def test_a_task_whose_points_all_worked_keeps_its_round(self, tmp_path):
        sweep = ExploreSpaceSobolQMC(
            [
                ExplorationTask("fine", BOX_2D, sphere, "cpu", 4, SEED),
                ExplorationTask("broken", BOX_2D, sphere, "cpu", 4, SEED),
            ],
            as_executor(LocalExecutor()),
        )
        sweep.run_exploration_jobs()
        path = tmp_path / "explore.pkl.gz"
        sweep.save(path)

        opt = OptimizeSpaceBotorch(
            [
                make_task(name="fine", iterations=1, parallel=2),
                make_task(
                    name="broken",
                    objective=self.fails_at(-10.0),
                    iterations=1,
                    parallel=2,
                ),
            ],
            as_executor(LocalExecutor()),
            [path],
        )

        with pytest.raises(RuntimeError, match=r"of \['broken'\]"):
            opt.run_search_jobs()

        assert len(opt.results["fine"].values) == 2
        assert opt.results["broken"].values == []

    def test_a_saved_round_is_resumable_after_a_failure(self, tmp_path):
        opt, _ = make_opt(
            tmp_path,
            objective=self.fails_at(0.0),
            prior_objective=sphere,
            explore=8,
            iterations=1,
            parallel=8,
        )

        with pytest.raises(RuntimeError):
            opt.run_search_jobs()
        opt.save(tmp_path / "search.pkl.gz")

        saved = load_results([tmp_path / "search.pkl.gz"])
        assert saved["test"].values == opt.results["test"].values


class TestSavedObservations:
    """A results file says nothing about the space it was measured over."""

    def test_a_point_missing_a_parameter_is_rejected(self, tmp_path):
        elsewhere = explored(
            tmp_path,
            space={"x": FloatRange(-5.0, 5.0)},
            objective=lambda x: {"objective": x},
        )

        with pytest.raises(RuntimeError, match="do not match the search space"):
            make_opt(tmp_path, files=[elsewhere])

    def test_a_point_with_a_parameter_too_many_is_rejected(self, tmp_path):
        """A wider space is not a narrower one with spare columns."""
        elsewhere = explored(
            tmp_path,
            space={
                "x": FloatRange(-5.0, 5.0),
                "y": FloatRange(-5.0, 5.0),
                "z": FloatRange(-5.0, 5.0),
            },
            objective=lambda x, y, z: {"objective": x + y + z},
        )

        with pytest.raises(RuntimeError, match="do not match the search space"):
            make_opt(tmp_path, files=[elsewhere])

    def test_a_point_outside_a_narrowed_range_is_rejected(self, tmp_path):
        """Otherwise the GP is fit outside the cube the acquisition searches."""
        wider = explored(
            tmp_path,
            space={"x": FloatRange(-5.0, 5.0), "y": FloatRange(-5.0, 5.0)},
            points=8,
        )

        with pytest.raises(RuntimeError, match="lies outside the search space"):
            make_opt(
                tmp_path,
                space={"x": FloatRange(1.0, 5.0), "y": FloatRange(-5.0, 5.0)},
                files=[wider],
            )

    def test_a_point_a_log_range_cannot_place_is_rejected(self, tmp_path):
        """log(0) is not a coordinate, and must not escape as a bare error."""
        linear = explored(
            tmp_path,
            space={"x": FloatRange(-5.0, 5.0), "y": FloatRange(-5.0, 5.0)},
        )

        with pytest.raises(RuntimeError, match="cannot be placed"):
            make_opt(
                tmp_path,
                space={
                    "x": FloatRange(1e-3, 5.0, log_range=True),
                    "y": FloatRange(-5.0, 5.0),
                },
                files=[linear],
            )

    def test_the_same_space_is_accepted(self, tmp_path):
        opt, _ = make_opt(tmp_path, explore=8)

        assert opt.num_observations("test") == 8
        assert all(
            all(0.0 <= u <= 1.0 for u in unit) for unit in opt.prior["test"].unit_points
        )


class TestSaveAndResume:
    def test_it_saves_only_what_this_run_measured(self, tmp_path):
        opt, _ = make_opt(tmp_path, explore=4, iterations=2, parallel=2)
        opt.run_search_jobs()

        opt.save(tmp_path / "search.pkl.gz")

        saved = load_results([tmp_path / "search.pkl.gz"])
        assert len(saved["test"].values) == 4, "the four searched points, not eight"
        assert saved["test"].points == opt.results["test"].points

    def test_the_file_has_the_shape_the_explorer_writes(self, tmp_path):
        opt, _ = make_opt(tmp_path, explore=4, iterations=1, parallel=2)
        opt.run_search_jobs()
        opt.save(tmp_path / "search.pkl.gz")

        path: Path = tmp_path / "search.pkl.gz"
        with gzip.open(path, "rb") as fobj:
            results = pickle.load(fobj)

        assert sorted(results) == ["test"]
        assert sorted(results["test"]) == ["outputs", "points", "values"]

    def test_saving_before_running_writes_empty_lists(self, tmp_path):
        opt, _ = make_opt(tmp_path)

        opt.save(tmp_path / "search.pkl.gz")

        assert load_results([tmp_path / "search.pkl.gz"])["test"].values == []

    def test_a_search_resumes_from_its_own_file(self, tmp_path):
        """The whole point of saving: the next run picks up where this stopped."""
        prior = explored(tmp_path, points=4, filename="prior.pkl.gz")

        first, _ = make_opt(
            tmp_path, files=[prior], iterations=2, parallel=2, explore=4
        )
        first.run_search_jobs()
        first.save(tmp_path / "round-one.pkl.gz")

        second, _ = make_opt(
            tmp_path,
            files=[prior, tmp_path / "round-one.pkl.gz"],
            iterations=2,
            parallel=2,
        )

        # Four from the exploration, four from the first search.
        assert second.num_observations("test") == 8

        second.run_search_jobs()

        assert second.num_observations("test") == 12
        assert len(second.results["test"].values) == 4

    def test_resuming_does_not_double_count_the_earlier_run(self, tmp_path):
        prior = explored(tmp_path, points=4, filename="prior.pkl.gz")
        first, _ = make_opt(tmp_path, files=[prior], iterations=1, parallel=2)
        first.run_search_jobs()
        first.save(tmp_path / "round-one.pkl.gz")

        second, _ = make_opt(
            tmp_path, files=[prior, tmp_path / "round-one.pkl.gz"], iterations=1
        )
        second.run_search_jobs()
        second.save(tmp_path / "round-two.pkl.gz")

        third, _ = make_opt(
            tmp_path,
            files=[
                prior,
                tmp_path / "round-one.pkl.gz",
                tmp_path / "round-two.pkl.gz",
            ],
            iterations=1,
        )

        # 4 explored + 2 + 4, each counted once.
        assert third.num_observations("test") == 10

    def test_the_best_covers_the_files_as_well_as_this_run(self, tmp_path):
        opt, _ = make_opt(tmp_path, explore=8, iterations=1, parallel=2)

        opt.run_search_jobs()
        _, value = opt.best_point("test")

        everything = opt.prior["test"].values + opt.results["test"].values
        assert value == min(everything)


# --------------------------------------------------------------------------
# Search behaviour
# --------------------------------------------------------------------------


class TestSearchBehaviour:
    """The optimizer minimizes while botorch maximizes, so assert the sign."""

    def test_search_moves_toward_the_minimum(self, tmp_path):
        # f(x) = x on [0, 1]: a flipped sign sends the search to 1.0 instead.
        # Asserted on the median search point: qLogNEI keeps probing away
        # from the incumbent, so the max is not a reliable signal, and the
        # best is already near the minimum from the exploration file.
        space = {"x": FloatRange(0.0, 1.0)}
        opt, _ = make_opt(
            tmp_path,
            objective=identity,
            space=space,
            explore=4,
            iterations=2,
            parallel=2,
        )
        opt.run_search_jobs()

        searched = [p["x"] for p in opt.results["test"].points]
        assert statistics.median(searched) < 0.5, searched

    def test_search_finds_the_optimum(self, tmp_path):
        opt, _ = make_opt(
            tmp_path, objective=sphere, explore=8, iterations=3, parallel=4
        )
        opt.run_search_jobs()

        params, value = opt.best_point("test")
        assert value < 0.5, value
        assert abs(params["x"]) < 1.0 and abs(params["y"]) < 1.0

    def test_search_beats_random_search(self, tmp_path):
        # Sobol' alone over the same total budget is the thing BO has to beat.
        opt, _ = make_opt(
            tmp_path, objective=sphere, explore=8, iterations=4, parallel=4
        )
        opt.run_search_jobs()
        guided = opt.best_point("test")[1]

        blind = ExploreSpaceSobolQMC(
            [ExplorationTask("blind", BOX_2D, sphere, "cpu", 24, SEED)],
            as_executor(LocalExecutor()),
        )
        blind.run_exploration_jobs()

        assert guided < blind.best_point("blind")[1]

    def test_a_mixed_space_optimizes(self, tmp_path):
        # Float, integer and categorical in one space.
        offsets = [0.0, 10.0, 25.0]

        def objective(x, y, cat):
            return {"objective": sphere(x, y)["objective"] + offsets[cat]}

        space = {
            "x": FloatRange(-5.0, 5.0),
            "y": IntRange(-5, 5),
            "cat": CategoricalRange(3),
        }
        opt, _ = make_opt(
            tmp_path,
            objective=objective,
            space=space,
            explore=16,
            iterations=4,
            parallel=4,
        )
        opt.run_search_jobs()

        params, value = opt.best_point("test")
        assert params["cat"] == 0, params
        assert params["y"] == 0, params
        assert value < 1.0, value


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------


class TestFailures:
    def test_a_failed_evaluation_raises(self, tmp_path):
        # Workers report exceptions as the task's output, not by raising.
        def boom(x, y):
            raise RuntimeError("worker exploded")

        opt, _ = make_opt(
            tmp_path,
            objective=boom,
            prior_objective=benign,
            explore=4,
            iterations=1,
            parallel=2,
        )
        with pytest.raises(RuntimeError, match="failed"):
            opt.run_search_jobs()

    def test_a_remote_error_is_never_recorded_as_a_value(self, tmp_path):
        def boom(x, y):
            raise RuntimeError("worker exploded")

        opt, _ = make_opt(
            tmp_path,
            objective=boom,
            prior_objective=benign,
            explore=4,
            iterations=1,
            parallel=2,
        )
        with pytest.raises(RuntimeError):
            opt.run_search_jobs()
        assert opt.results["test"].values == []

    @pytest.mark.parametrize(
        "objective,match",
        [
            (lambda x, y: {"objective": float("nan")}, "non-finite"),
            (lambda x, y: {"objective": float("inf")}, "non-finite"),
            (lambda x, y: {"objective": "not a number"}, "not a float"),
            (lambda x, y: 1.0, "expected a mapping"),
            (lambda x, y: {"loss": 1.0}, "no 'objective'"),
            (lambda x, y: {"objectiv": 1.0}, r"\['objectiv'\]"),
        ],
    )
    def test_an_unusable_result_raises(self, tmp_path, objective, match):
        opt, _ = make_opt(
            tmp_path,
            objective=objective,
            prior_objective=benign,
            explore=4,
            iterations=1,
            parallel=2,
        )

        with pytest.raises(RuntimeError, match=match):
            opt.run_search_jobs()

    def test_a_configured_key_is_what_the_message_names(self, tmp_path):
        """The default key is not what a run with its own key is missing."""
        opt, _ = make_opt(
            tmp_path,
            prior_objective=benign,
            objective=lambda x, y: {"objective": 1.0},
            explore=4,
            iterations=1,
            parallel=2,
            objective_key="rmse",
        )
        with pytest.raises(RuntimeError, match="no 'rmse'"):
            opt.run_search_jobs()

    def test_an_integer_objective_is_accepted(self, tmp_path):
        opt, _ = make_opt(
            tmp_path,
            prior_objective=benign,
            objective=lambda x, y: {"objective": 1},
            explore=4,
            iterations=1,
            parallel=2,
        )
        opt.run_search_jobs()
        assert opt.results["test"].values == [1.0, 1.0]


class TestObjectiveKey:
    """Which key of the result is modelled is the task's to choose."""

    def test_the_default_key_is_objective(self, tmp_path):
        opt, _ = make_opt(tmp_path)
        assert opt.tasks[0].objective_key == "objective"

    def test_a_configured_key_is_the_one_modelled(self, tmp_path):
        """An evaluation that already reports `loss` is searched as it is."""

        def objective(x, y):
            return {"loss": x * x + y * y, "note": "sphere"}

        opt, _ = make_opt(
            tmp_path,
            prior_objective=benign,
            objective=objective,
            explore=4,
            iterations=1,
            parallel=2,
            objective_key="loss",
        )
        opt.run_search_jobs()

        result = opt.results["test"]
        assert result.values == [objective(**p)["loss"] for p in result.points]

    def test_the_default_key_is_then_just_another_recorded_key(self, tmp_path):
        """Only the configured key is modelled; the rest are carried along."""

        def objective(x, y):
            return {"loss": x * x + y * y, "objective": 999.0}

        opt, _ = make_opt(
            tmp_path,
            prior_objective=benign,
            objective=objective,
            explore=4,
            iterations=1,
            parallel=2,
            objective_key="loss",
        )
        opt.run_search_jobs()

        result = opt.results["test"]
        assert 999.0 not in result.values
        assert result.values == [objective(**p)["loss"] for p in result.points]
        assert all(output["objective"] == 999.0 for output in result.outputs)


class TestBestPoint:
    def test_returns_the_minimum(self, tmp_path):
        opt, _ = make_opt(
            tmp_path, objective=sphere, explore=8, iterations=1, parallel=2
        )
        opt.run_search_jobs()

        params, value = opt.best_point("test")
        assert value == sphere(**params)["objective"]

    def test_best_output_is_the_whole_mapping(self, tmp_path):
        opt, _ = make_opt(
            tmp_path, objective=sphere, explore=8, iterations=1, parallel=2
        )
        opt.run_search_jobs()
        params, value = opt.best_point("test")

        output = opt.best_output("test")
        assert output == sphere(**params)
        assert output["objective"] == value
        assert output["note"] == "sphere", "keys beyond the objective are kept"

    def test_every_output_is_recorded(self, tmp_path):
        opt, _ = make_opt(
            tmp_path, objective=sphere, explore=4, iterations=2, parallel=2
        )
        opt.run_search_jobs()

        result = opt.results["test"]
        assert len(result.outputs) == len(result.values) == len(result.points)
        for params, value, output in zip(result.points, result.values, result.outputs):
            assert output == sphere(**params)
            assert output["objective"] == value

    def test_a_stored_output_is_a_copy(self, tmp_path):
        """Mutating what the objective returned must not rewrite the record."""
        returned = {}

        def objective(x, y):
            nonlocal returned
            returned = {"objective": x * x + y * y, "trace": [1, 2, 3]}
            return returned

        opt, _ = make_opt(
            tmp_path, objective=objective, explore=4, iterations=1, parallel=2
        )
        opt.run_search_jobs()
        recorded = dict(opt.results["test"].outputs[-1])

        returned["objective"] = -999.0
        returned["trace"] = []

        assert opt.results["test"].outputs[-1] == recorded

    def test_returns_a_copy(self, tmp_path):
        # A caller mutating the returned dict must not corrupt the history.
        opt, _ = make_opt(tmp_path, explore=4, iterations=1, parallel=2)
        opt.run_search_jobs()

        params, _ = opt.best_point("test")
        params["x"] = 999.0

        known = opt.prior["test"].points + opt.results["test"].points
        assert 999.0 not in [p["x"] for p in known]

    def test_improves_or_holds_across_the_search(self, tmp_path):
        opt, _ = make_opt(
            tmp_path, objective=sphere, explore=8, iterations=2, parallel=4
        )
        before = min(opt.prior["test"].values)

        opt.run_search_jobs()

        assert opt.best_point("test")[1] <= before


# --------------------------------------------------------------------------
# The real executor
# --------------------------------------------------------------------------


class TestRealExecutor:
    """One end-to-end run against the real queue, executor and worker."""

    @pytest.fixture(autouse=True)
    def _pilot_jobs(self, pilot_jobs):
        pilot_jobs("cpu")

    def test_explores_then_optimizes_through_a_real_worker(
        self, executor, ds_service_address, tmp_path
    ):
        explore, iterations, parallel = 4, 2, 2
        # The exploration's points, then per round one fit-and-propose task
        # on top of the evaluations. Both kinds go to the one queue this
        # worker serves, which is also what pins that a real worker can run
        # the fit at all.
        total = explore + iterations * (parallel + 1)

        worker = make_worker(ds_service_address, tmp_path / "worker", group="cpu")
        thread = threading.Thread(target=run_worker, args=(worker, total), daemon=True)
        thread.start()
        try:
            sweep = ExploreSpaceSobolQMC(
                [ExplorationTask("e2e", BOX_2D, sphere, "cpu", explore, SEED)],
                executor,
            )
            sweep.run_exploration_jobs()
            path = tmp_path / "explore.pkl.gz"
            sweep.save(path)

            opt = OptimizeSpaceBotorch(
                [
                    OptimizationTask(
                        "e2e",
                        BOX_2D,
                        sphere,
                        "cpu",
                        "cpu",
                        parallel,
                        min_search_iterations=iterations,
                        max_search_iterations=iterations,
                    )
                ],
                executor,
                [path],
            )
            opt.run_search_jobs()
        finally:
            thread.join(timeout=30)
            worker.close()

        assert not thread.is_alive(), "worker thread did not finish"
        assert opt.num_observations("e2e") == explore + iterations * parallel

        params, value = opt.best_point("e2e")
        assert math.isclose(value, sphere(**params)["objective"])
        # The whole mapping survives the round trip through the real queue,
        # not just the number the model was fit on.
        assert opt.best_output("e2e") == {"objective": value, "note": "sphere"}
        assert value < BOX_2D["x"].max ** 2 + BOX_2D["y"].max ** 2

    def test_a_raising_objective_surfaces_from_a_real_worker(
        self, executor, ds_service_address, tmp_path
    ):
        def boom(x, y):
            raise RuntimeError("worker exploded")

        prior = explored(tmp_path, name="e2e-fail", points=2, filename="prior.pkl.gz")

        opt = OptimizeSpaceBotorch(
            [
                OptimizationTask(
                    "e2e-fail",
                    BOX_2D,
                    boom,
                    "cpu",
                    "cpu",
                    1,
                    min_search_iterations=1,
                    max_search_iterations=1,
                )
            ],
            executor,
            [prior],
        )

        # One fit, then the one evaluation that fails.
        worker = make_worker(ds_service_address, tmp_path / "worker", group="cpu")
        thread = threading.Thread(target=run_worker, args=(worker, 2), daemon=True)
        thread.start()
        try:
            with pytest.raises(RuntimeError, match="failed"):
                opt.run_search_jobs()
        finally:
            thread.join(timeout=30)
            worker.close()

        assert opt.results["e2e-fail"].values == []
