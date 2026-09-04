"""Tests for the Sobol' QMC sweep.

Needs neither botorch nor torch, which is the point of the module:
scipy draws the designs, and a stand-in executor runs the objectives inline.
`TestRealExecutor` is what keeps that stand-in honest.
"""

from __future__ import annotations

import gzip
import math
import pickle
import threading
from pathlib import Path
from typing import cast

import pytest

from slurm_workflows.explore_space import ExplorationTask, ExploreSpaceSobolQMC
from slurm_workflows.search_space import CategoricalRange, FloatRange, IntRange
from slurm_workflows.slurm_pilot_executor import SlurmPilotExecutor, Task
from slurm_workflows.utils import RemoteExecutionError, gen_error_id
from worker_harness import make_worker, run_worker

SEED = 20260904

BOX_2D = {"x": FloatRange(-5.0, 5.0), "y": FloatRange(-5.0, 5.0)}

MIXED = {
    "x": FloatRange(-1.0, 1.0),
    "n": IntRange(2, 10),
    "kind": CategoricalRange(3),
}


def sphere(x, y):
    return {"objective": x * x + y * y}


def plane(x, y):
    return {"objective": x + y}


def mixed_objective(x, n, kind):
    return {"objective": abs(x) + n + kind, "n": n, "kind": kind}


class LocalExecutor:
    """Stands in for SlurmPilotExecutor, running each callable inline.

    Records what it was asked to submit,
    and turns a raising objective into a `RemoteExecutionError` output,
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

    def wait(self, tasks, desc=None, unit="task", raise_on_error=None) -> None:
        self.waits.append(desc)
        self.batch_sizes.append(len(tasks))
        failed = [t for t in tasks if isinstance(t.output, RemoteExecutionError)]
        if failed:
            raise RuntimeError(f"{len(failed)} of {len(tasks)} tasks did not succeed")


def as_executor(executor: LocalExecutor) -> SlurmPilotExecutor:
    """Type the stand-in as the executor it stands in for."""
    return cast(SlurmPilotExecutor, executor)


def task(
    name: str = "sweep",
    space=BOX_2D,
    objective=sphere,
    queue: str | list[str] = "cpu",
    points: int | None = 8,
    seed: int | None = SEED,
    objective_key: str = "objective",
    **extra,
) -> ExplorationTask:
    """One exploration task, with the test defaults filled in."""
    return ExplorationTask(
        name=name,
        space=space,
        objective=objective,
        objective_queue=queue,
        num_exploration_points=points,
        seed=seed,
        objective_key=objective_key,
        extra_objective_kwargs=extra,
    )


def explorer(*tasks: ExplorationTask, points: int | None = None):
    """A sweep of `tasks`, or of one default task, on a stand-in executor."""
    return ExploreSpaceSobolQMC(
        list(tasks) or [task()], as_executor(LocalExecutor()), points
    )


class TestConstruction:
    def test_the_point_count_is_floored_to_a_power_of_two(self):
        sweep = explorer(task(points=100))

        assert sweep.tasks[0].num_exploration_points == 64

    def test_the_sweeps_count_fills_in_for_a_task_without_one(self):
        sweep = explorer(task(points=None), points=16)

        assert sweep.tasks[0].num_exploration_points == 16

    def test_a_tasks_own_count_wins(self):
        sweep = explorer(task(points=4), points=64)

        assert sweep.tasks[0].num_exploration_points == 4

    def test_a_count_from_neither_is_rejected(self):
        with pytest.raises(ValueError, match="num_exploration_points"):
            explorer(task(points=None))

    def test_a_point_count_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            explorer(task(points=0))

    def test_no_tasks_is_rejected(self):
        with pytest.raises(ValueError, match="no exploration tasks"):
            ExploreSpaceSobolQMC([], as_executor(LocalExecutor()))

    def test_duplicate_task_names_are_rejected(self):
        """The name keys the results, so two of them would lose one."""
        with pytest.raises(ValueError, match="unique"):
            explorer(task(name="a"), task(name="b"), task(name="a"))

    def test_an_empty_space_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            explorer(task(space={}))

    def test_extra_kwargs_may_not_shadow_a_parameter(self):
        with pytest.raises(ValueError, match="shadow"):
            explorer(task(x=1.0))

    def test_the_failing_task_is_named(self):
        with pytest.raises(ValueError, match="second"):
            explorer(task(name="first"), task(name="second", space={}))

    def test_a_missing_seed_is_drawn_and_printed(self, capsys):
        sweep = explorer(task(seed=None))

        assert isinstance(sweep.tasks[0].seed, int)
        assert str(sweep.tasks[0].seed) in capsys.readouterr().out

    def test_the_callers_task_is_left_alone(self):
        """`tasks` says what will run; the caller's own object is not touched."""
        original = task(points=100, seed=None)

        sweep = explorer(original)

        assert original.num_exploration_points == 100
        assert original.seed is None
        assert sweep.tasks[0].num_exploration_points == 64
        assert sweep.tasks[0].seed is not None

    def test_nothing_is_submitted_until_it_runs(self):
        sweep = explorer()

        assert sweep.results["sweep"].points == []
        with pytest.raises(RuntimeError, match="nothing has been evaluated"):
            sweep.best_point("sweep")

    def test_an_unknown_task_name_says_what_there_is(self):
        sweep = explorer(task(name="a"))

        with pytest.raises(KeyError, match="'a'"):
            sweep.design("b")


class TestDesign:
    def test_it_draws_the_asked_for_number_of_points(self):
        assert len(explorer(task(points=8)).design("sweep")) == 8

    def test_the_same_seed_draws_the_same_design(self):
        one = explorer(task(seed=7)).design("sweep")
        two = explorer(task(seed=7)).design("sweep")

        assert one == two

    def test_a_different_seed_draws_a_different_design(self):
        assert explorer(task(seed=7)).design("sweep") != explorer(task(seed=8)).design(
            "sweep"
        )

    def test_each_task_draws_in_its_own_space(self):
        sweep = explorer(
            task(name="box"),
            task(name="mixed", space=MIXED, objective=mixed_objective),
        )

        assert set(sweep.design("box")[0]) == {"x", "y"}
        assert set(sweep.design("mixed")[0]) == {"x", "n", "kind"}

    def test_every_point_is_inside_its_range(self):
        sweep = explorer(task(space=MIXED, objective=mixed_objective))

        for params in sweep.design("sweep"):
            assert -1.0 <= params["x"] <= 1.0
            assert 2 <= params["n"] <= 10
            assert params["kind"] in (0, 1, 2)

    def test_discrete_parameters_come_back_as_integers(self):
        sweep = explorer(task(space=MIXED, objective=mixed_objective))

        params = sweep.design("sweep")[0]

        assert isinstance(params["n"], int)
        assert isinstance(params["kind"], int)

    def test_the_design_covers_the_space_more_evenly_than_it_clumps(self):
        """What Sobol' buys over a uniform draw: no half is left empty."""
        design = explorer(task(points=64)).design("sweep")

        assert sum(1 for p in design if p["x"] < 0.0) == 32
        assert sum(1 for p in design if p["y"] < 0.0) == 32

    def test_drawing_does_not_evaluate(self):
        sweep = explorer()

        sweep.design("sweep")

        assert sweep.results["sweep"].values == []

    def test_the_dimension_is_the_tasks_own(self):
        sweep = explorer(
            task(name="box"),
            task(name="mixed", space=MIXED, objective=mixed_objective),
        )

        assert sweep.dim("box") == 2
        assert sweep.dim("mixed") == 3


class TestRun:
    def test_it_evaluates_every_point_of_the_design(self):
        sweep = explorer(task(points=8))

        sweep.run_exploration_jobs()

        result = sweep.results["sweep"]
        assert len(result.points) == 8
        assert len(result.values) == 8
        assert len(result.outputs) == 8
        assert len(result.unit_points) == 8

    def test_the_values_are_the_objectives_own(self):
        sweep = explorer(task(points=8))

        sweep.run_exploration_jobs()

        result = sweep.results["sweep"]
        for params, value in zip(result.points, result.values):
            assert math.isclose(value, sphere(**params)["objective"])

    def test_the_whole_output_is_kept_not_just_the_ranked_value(self):
        sweep = explorer(task(space=MIXED, objective=mixed_objective, points=4))

        sweep.run_exploration_jobs()

        outputs = sweep.results["sweep"].outputs
        assert all(set(o) == {"objective", "n", "kind"} for o in outputs)

    def test_the_best_point_is_the_lowest_value(self):
        sweep = explorer(task(points=8))

        sweep.run_exploration_jobs()
        params, value = sweep.best_point("sweep")

        result = sweep.results["sweep"]
        assert value == min(result.values)
        assert params == result.points[result.values.index(value)]
        assert sweep.best_output("sweep")["objective"] == value

    def test_the_best_of_each_task_is_reported(self, capsys):
        sweep = explorer(task(name="first", points=4), task(name="second", points=4))

        sweep.run_exploration_jobs()

        out = capsys.readouterr().out
        assert "first: best of 4 points" in out
        assert "second: best of 4 points" in out

    def test_the_recorded_unit_point_is_the_one_evaluated(self):
        """Discrete parameters are rounded; the rounded point is the record."""
        sweep = explorer(task(space=MIXED, objective=mixed_objective, points=4))

        sweep.run_exploration_jobs()

        result = sweep.results["sweep"]
        for params, unit in zip(result.points, result.unit_points):
            assert math.isclose(unit[1], (params["n"] - 2) / (10 - 2))

    def test_extra_kwargs_reach_the_objective(self):
        def objective(x, y, scale):
            return {"objective": scale * (x * x + y * y)}

        sweep = explorer(task(objective=objective, points=4, scale=2.0))

        sweep.run_exploration_jobs()

        assert all("scale" in kw for kw in sweep.executor.kwargs)  # type: ignore[attr-defined]

    def test_a_second_run_repeats_the_design(self):
        """The seed decides the design, so this is a re-evaluation."""
        sweep = explorer(task(points=4))

        sweep.run_exploration_jobs()
        sweep.run_exploration_jobs()

        points = sweep.results["sweep"].points
        assert points[:4] == points[4:]


class TestSeveralSpacesAtOnce:
    def test_every_task_is_evaluated(self):
        sweep = explorer(
            task(name="small", points=4),
            task(name="large", objective=plane, points=16),
        )

        sweep.run_exploration_jobs()

        assert len(sweep.results["small"].values) == 4
        assert len(sweep.results["large"].values) == 16

    def test_the_whole_sweep_is_submitted_before_any_of_it_is_waited_for(self):
        """The point of running them together: one batch, not one each."""
        sweep = explorer(
            task(name="small", points=4),
            task(name="large", objective=plane, points=16),
        )

        sweep.run_exploration_jobs()

        assert sweep.executor.batch_sizes == [20]  # type: ignore[attr-defined]
        assert len(sweep.executor.waits) == 1  # type: ignore[attr-defined]

    def test_each_task_goes_to_its_own_queue(self):
        sweep = explorer(
            task(name="on_cpu", queue="cpu", points=4),
            task(name="on_gpu", queue="gpu", objective=plane, points=4),
        )

        sweep.run_exploration_jobs()

        assert sweep.executor.queues == ["cpu"] * 4 + ["gpu"] * 4  # type: ignore[attr-defined]

    def test_results_are_kept_apart(self):
        sweep = explorer(
            task(name="sphere", points=4),
            task(name="plane", objective=plane, points=4),
        )

        sweep.run_exploration_jobs()

        for params, value in zip(
            sweep.results["plane"].points, sweep.results["plane"].values
        ):
            assert math.isclose(value, plane(**params)["objective"])
        assert all(v >= 0 for v in sweep.results["sphere"].values)

    def test_each_task_ranks_by_its_own_objective_key(self):
        sweep = explorer(
            task(name="a", points=4),
            task(
                name="b",
                objective=lambda x, y: {"loss": x * x + y * y},
                objective_key="loss",
                points=4,
            ),
        )

        sweep.run_exploration_jobs()

        assert len(sweep.results["b"].values) == 4

    def test_one_broken_task_names_itself(self):
        def boom(x, y):
            raise ValueError("objective blew up")

        sweep = explorer(
            task(name="fine", points=4),
            task(name="broken", objective=boom, points=4),
        )

        with pytest.raises(RuntimeError, match=r"exploration of \['broken'\]"):
            sweep.run_exploration_jobs()


class TestPartialFailure:
    """One bad point must not cost a whole sweep."""

    @staticmethod
    def fails_at(threshold: float):
        """An objective that raises on the points past `threshold`."""

        def objective(x, y):
            if x > threshold:
                raise ValueError("objective blew up")
            return {"objective": x * x + y * y}

        return objective

    def test_the_points_that_came_back_are_kept(self):
        sweep = explorer(task(objective=self.fails_at(0.0), points=8))

        with pytest.raises(RuntimeError, match="objective evaluations failed"):
            sweep.run_exploration_jobs()

        result = sweep.results["sweep"]
        assert result.values, "the successful points were thrown away"
        assert len(result.values) < 8
        assert all(x <= 0.0 for x in (p["x"] for p in result.points))

    def test_what_is_kept_is_index_aligned(self):
        sweep = explorer(task(objective=self.fails_at(0.0), points=8))

        with pytest.raises(RuntimeError):
            sweep.run_exploration_jobs()

        result = sweep.results["sweep"]
        assert len(result.points) == len(result.values) == len(result.outputs)
        assert len(result.unit_points) == len(result.values)

    def test_a_saved_sweep_is_resumable_after_a_failure(self, tmp_path):
        """`save()` is what the next run reads, so it must not be empty."""
        sweep = explorer(task(objective=self.fails_at(0.0), points=8))

        with pytest.raises(RuntimeError):
            sweep.run_exploration_jobs()
        path: Path = tmp_path / "sweep.pkl.gz"
        sweep.save(path)

        with gzip.open(path, "rb") as fobj:
            saved = pickle.load(fobj)
        assert saved["sweep"]["values"] == sweep.results["sweep"].values

    def test_a_task_whose_points_all_worked_keeps_all_of_them(self):
        """A failure in one task must not empty another's record."""
        sweep = explorer(
            task(name="fine", points=4),
            task(name="broken", objective=self.fails_at(-10.0), points=4),
        )

        with pytest.raises(RuntimeError, match=r"exploration of \['broken'\]"):
            sweep.run_exploration_jobs()

        assert len(sweep.results["fine"].values) == 4
        assert sweep.results["broken"].values == []


class TestSave:
    @staticmethod
    def load(path: Path):
        with gzip.open(path, "rb") as fobj:
            return pickle.load(fobj)

    def test_it_writes_what_each_task_measured(self, tmp_path):
        sweep = explorer(
            task(name="a", points=8), task(name="b", objective=plane, points=4)
        )
        sweep.run_exploration_jobs()

        sweep.save(tmp_path / "sweep.pkl.gz")

        results = self.load(tmp_path / "sweep.pkl.gz")
        assert sorted(results) == ["a", "b"]
        for name in ("a", "b"):
            assert results[name]["points"] == sweep.results[name].points
            assert results[name]["values"] == sweep.results[name].values
            assert results[name]["outputs"] == sweep.results[name].outputs

    def test_the_three_lists_stay_index_aligned(self, tmp_path):
        sweep = explorer(task(space=MIXED, objective=mixed_objective, points=4))
        sweep.run_exploration_jobs()

        sweep.save(tmp_path / "sweep.pkl.gz")

        saved = self.load(tmp_path / "sweep.pkl.gz")["sweep"]
        for params, value, output in zip(
            saved["points"], saved["values"], saved["outputs"]
        ):
            assert output["objective"] == value
            assert output["n"] == params["n"]

    def test_the_file_is_gzipped(self, tmp_path):
        sweep = explorer(task(points=4))
        sweep.run_exploration_jobs()

        sweep.save(tmp_path / "sweep.pkl.gz")

        assert (tmp_path / "sweep.pkl.gz").read_bytes()[:2] == b"\x1f\x8b"

    def test_a_path_may_be_a_string(self, tmp_path):
        sweep = explorer(task(points=4))
        sweep.run_exploration_jobs()

        sweep.save(str(tmp_path / "sweep.pkl.gz"))

        saved = self.load(tmp_path / "sweep.pkl.gz")["sweep"]
        assert saved["values"] == sweep.results["sweep"].values

    def test_saving_again_replaces_the_file(self, tmp_path):
        sweep = explorer(task(points=4))
        sweep.run_exploration_jobs()
        sweep.save(tmp_path / "sweep.pkl.gz")

        sweep.run_exploration_jobs()
        sweep.save(tmp_path / "sweep.pkl.gz")

        assert len(self.load(tmp_path / "sweep.pkl.gz")["sweep"]["values"]) == 8

    def test_saving_before_running_writes_empty_lists(self, tmp_path):
        """The file says what was measured, and that is nothing yet."""
        explorer(task(points=4)).save(tmp_path / "sweep.pkl.gz")

        results = self.load(tmp_path / "sweep.pkl.gz")
        assert results == {"sweep": {"points": [], "values": [], "outputs": []}}


class TestObjectiveResults:
    def test_a_failed_evaluation_is_reported(self):
        def objective(x, y):
            raise ValueError("objective blew up")

        sweep = explorer(task(objective=objective, points=4))

        with pytest.raises(RuntimeError, match="objective evaluations failed"):
            sweep.run_exploration_jobs()

    def test_a_result_that_is_not_a_mapping_is_rejected(self):
        sweep = explorer(task(objective=lambda x, y: x + y, points=4))

        with pytest.raises(RuntimeError, match="expected a mapping"):
            sweep.run_exploration_jobs()

    def test_a_missing_objective_key_is_rejected(self):
        sweep = explorer(task(objective=lambda x, y: {"loss": x}, points=4))

        with pytest.raises(RuntimeError, match="no 'objective' among them"):
            sweep.run_exploration_jobs()

    def test_another_objective_key_can_be_named(self):
        sweep = explorer(
            task(
                objective=lambda x, y: {"loss": x * x + y * y},
                objective_key="loss",
                points=4,
            )
        )

        sweep.run_exploration_jobs()

        assert len(sweep.results["sweep"].values) == 4

    def test_a_value_that_is_not_a_float_is_rejected(self):
        sweep = explorer(task(objective=lambda x, y: {"objective": "cheap"}, points=4))

        with pytest.raises(RuntimeError, match="not a float"):
            sweep.run_exploration_jobs()

    def test_a_non_finite_value_is_rejected(self):
        sweep = explorer(
            task(objective=lambda x, y: {"objective": float("nan")}, points=4)
        )

        with pytest.raises(RuntimeError, match="non-finite"):
            sweep.run_exploration_jobs()


class TestRealExecutor:
    """One end-to-end sweep against the real queue, executor and worker."""

    @pytest.fixture(autouse=True)
    def _pilot_jobs(self, pilot_jobs):
        pilot_jobs("cpu")

    def test_it_sweeps_two_spaces_through_a_real_worker(
        self, executor, ds_service_address, tmp_path
    ):
        points = 4
        sweep = ExploreSpaceSobolQMC(
            [
                ExplorationTask("box", BOX_2D, sphere, "cpu", points, SEED),
                ExplorationTask("flat", BOX_2D, plane, "cpu", points, SEED),
            ],
            executor,
        )

        # A real worker in a thread: the sweep blocks in wait() as soon as
        # it submits, so nothing can play the worker's part after the fact.
        worker = make_worker(ds_service_address, tmp_path / "worker", group="cpu")
        thread = threading.Thread(
            target=run_worker, args=(worker, 2 * points), daemon=True
        )
        thread.start()
        try:
            sweep.run_exploration_jobs()
        finally:
            thread.join(timeout=30)
            worker.close()

        assert not thread.is_alive(), "worker thread did not finish"
        for name in ("box", "flat"):
            result = sweep.results[name]
            assert len(result.values) == points
            assert sweep.best_point(name)[1] == min(result.values)
