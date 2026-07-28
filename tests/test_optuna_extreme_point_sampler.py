"""Tests for the extreme-point (corner) sampler.

Two properties carry the sampler: it is deterministic, so the corner a
given id maps to is asserted directly; and it is safe distributed, so the
concurrent tests assert that every corner came out exactly once with real
workers racing for them.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest

optuna = pytest.importorskip("optuna")

import cloudpickle  # noqa: E402
from optuna.distributions import (  # noqa: E402
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from slurm_workflows.optuna_storage import (  # noqa: E402
    DsServiceJournalBackend,
    create_optuna_storage,
)
from slurm_workflows.optuna_extreme_point_sampler import (  # noqa: E402
    CORNER_ID_ATTR,
    ExtremePointSampler,
)

optuna.logging.set_verbosity(optuna.logging.ERROR)

# Five parameters -> 32 corners, which divides evenly among four workers.
SPACE = {
    "a": FloatDistribution(-1, 1),
    "b": FloatDistribution(0, 10),
    "c": IntDistribution(1, 8),
    "d": FloatDistribution(1e-5, 1e-1, log=True),
    "e": IntDistribution(0, 3),
}
N_CORNERS = 32
WORKERS = 4


def objective(trial):
    return (
        trial.suggest_float("a", -1, 1)
        + trial.suggest_float("b", 0, 10)
        + trial.suggest_int("c", 1, 8)
        + trial.suggest_float("d", 1e-5, 1e-1, log=True)
        + trial.suggest_int("e", 0, 3)
    )


def corner_ids(study) -> list[int]:
    return [t.system_attrs[CORNER_ID_ATTR] for t in study.trials]


@pytest.fixture
def backend(ds_service_address: str):
    b = DsServiceJournalBackend(ds_service_address, prefix="corners")
    yield b
    b.close()


@pytest.fixture
def storage(ds_service_address: str):
    return create_optuna_storage(ds_service_address, prefix="corners")


# --------------------------------------------------------------------------
# The corners themselves
# --------------------------------------------------------------------------


class TestCornerEnumeration:
    def test_the_count_is_two_to_the_dimension(self, backend):
        assert ExtremePointSampler(backend, SPACE).n_corners == 2**5

    def test_corner_zero_is_every_low(self, backend):
        sampler = ExtremePointSampler(backend, SPACE)

        assert sampler.corner(0) == {
            "a": -1.0,
            "b": 0.0,
            "c": 1,
            "d": 1e-5,
            "e": 0,
        }

    def test_the_last_corner_is_every_high(self, backend):
        sampler = ExtremePointSampler(backend, SPACE)

        assert sampler.corner(N_CORNERS - 1) == {
            "a": 1.0,
            "b": 10.0,
            "c": 8,
            "d": 1e-1,
            "e": 3,
        }

    def test_bit_i_selects_the_high_of_parameter_i(self, backend):
        """Parameters sorted by name, least significant bit first."""
        sampler = ExtremePointSampler(backend, SPACE)

        assert sampler.corner(0b00001)["a"] == 1.0
        assert sampler.corner(0b00001)["b"] == 0.0
        assert sampler.corner(0b00010)["b"] == 10.0
        assert sampler.corner(0b10000)["e"] == 3
        assert sampler.corner(0b10000)["a"] == -1.0

    def test_every_corner_is_distinct(self, backend):
        sampler = ExtremePointSampler(backend, SPACE)

        corners = [tuple(sampler.corner(i).items()) for i in range(N_CORNERS)]

        assert len(set(corners)) == N_CORNERS

    def test_it_is_deterministic_across_instances(self, backend):
        first = ExtremePointSampler(backend, SPACE)
        second = ExtremePointSampler(backend, SPACE)

        assert [first.corner(i) for i in range(N_CORNERS)] == [
            second.corner(i) for i in range(N_CORNERS)
        ]

    def test_a_degenerate_parameter_does_not_double_the_count(self, backend):
        space = {"x": FloatDistribution(-1, 1), "fixed": IntDistribution(3, 3)}

        sampler = ExtremePointSampler(backend, space)

        assert sampler.n_corners == 2
        assert sampler.corner(0) == {"x": -1.0, "fixed": 3}
        assert sampler.corner(1) == {"x": 1.0, "fixed": 3}

    def test_stepped_distributions_use_their_attainable_high(self, backend):
        """Optuna clamps `high` to the step grid; the corner must match it."""
        space = {"x": IntDistribution(0, 10, step=3)}

        sampler = ExtremePointSampler(backend, space)

        assert sampler.corner(1) == {"x": 9}

    def test_out_of_range_ids_are_rejected(self, backend):
        sampler = ExtremePointSampler(backend, SPACE)

        with pytest.raises(IndexError):
            sampler.corner(N_CORNERS)
        with pytest.raises(IndexError):
            sampler.corner(-1)

    def test_it_does_not_materialize_the_corners(self, backend):
        """A 60-parameter box has more corners than memory; it must not care."""
        space = {f"p{i}": FloatDistribution(0, 1) for i in range(60)}

        sampler = ExtremePointSampler(backend, space)

        assert sampler.n_corners == 2**60
        assert sampler.corner(2**60 - 1) == {f"p{i}": 1.0 for i in range(60)}


# --------------------------------------------------------------------------
# Running a study
# --------------------------------------------------------------------------


class TestSequentialStudy:
    def test_it_visits_every_corner_exactly_once(self, storage, backend):
        sampler = ExtremePointSampler(backend, SPACE)
        study = optuna.create_study(storage=storage, study_name="s", sampler=sampler)

        study.optimize(objective, n_trials=N_CORNERS)

        assert sorted(corner_ids(study)) == list(range(N_CORNERS))

    def test_the_recorded_params_are_the_corners(self, storage, backend):
        sampler = ExtremePointSampler(backend, SPACE)
        study = optuna.create_study(storage=storage, study_name="s", sampler=sampler)

        study.optimize(objective, n_trials=N_CORNERS)

        for trial in study.trials:
            assert trial.params == sampler.corner(trial.system_attrs[CORNER_ID_ATTR])

    def test_it_stops_once_the_walk_is_done(self, storage, backend):
        """Asking for more trials than corners must not keep going."""
        study = optuna.create_study(
            storage=storage, study_name="s", sampler=ExtremePointSampler(backend, SPACE)
        )

        study.optimize(objective, n_trials=N_CORNERS * 3)

        assert len(study.trials) == N_CORNERS

    def test_re_running_a_finished_study_warns(self, storage, backend, caplog):
        study = optuna.create_study(
            storage=storage, study_name="s", sampler=ExtremePointSampler(backend, SPACE)
        )
        study.optimize(objective, n_trials=N_CORNERS)

        study.optimize(objective, n_trials=1)

        assert len(study.trials) == N_CORNERS + 1
        assert "re-evaluating" in caplog.text


# --------------------------------------------------------------------------
# Distributed
# --------------------------------------------------------------------------


class TestDistributed:
    def run_workers(self, address, trials_each=N_CORNERS // WORKERS):
        optuna.create_study(
            storage=create_optuna_storage(address, prefix="corners"),
            study_name="par",
            load_if_exists=True,
        )

        def worker(_):
            storage = create_optuna_storage(address, prefix="corners")
            study = optuna.load_study(
                study_name="par",
                storage=storage,
                sampler=ExtremePointSampler(storage, SPACE),
            )
            study.optimize(objective, n_trials=trials_each)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(worker, range(WORKERS)))

        return optuna.load_study(
            study_name="par",
            storage=create_optuna_storage(address, prefix="corners"),
        )

    def test_concurrent_workers_split_the_corners_exactly(
        self, ds_service_address
    ):
        """No corner twice, no corner missed, across four racing workers."""
        study = self.run_workers(ds_service_address)

        assert sorted(corner_ids(study)) == list(range(N_CORNERS))

    def test_the_params_are_still_the_right_corners(self, ds_service_address):
        study = self.run_workers(ds_service_address)
        reference = ExtremePointSampler(
            create_optuna_storage(ds_service_address, prefix="corners"), SPACE
        )

        for trial in study.trials:
            assert trial.params == reference.corner(
                trial.system_attrs[CORNER_ID_ATTR]
            )

    def test_a_resumed_study_is_no_different(self, ds_service_address):
        """The case that defeats GridSampler: trial numbers past the count.

        GridSampler indexes by trial number while it can and only then
        falls back to scanning; this sampler allocates the same way always,
        so pre-existing trials change nothing.
        """
        storage = create_optuna_storage(ds_service_address, prefix="corners")
        study = optuna.create_study(storage=storage, study_name="par")
        # Unrelated prior trials, more of them than there are corners.
        study.optimize(objective, n_trials=N_CORNERS + 8)
        before = len(study.trials)

        walked = self.run_workers(ds_service_address)

        fresh = [t for t in walked.trials if CORNER_ID_ATTR in t.system_attrs]
        assert len(walked.trials) == before + N_CORNERS
        assert sorted(t.system_attrs[CORNER_ID_ATTR] for t in fresh) == list(
            range(N_CORNERS)
        )

    def test_grid_sampler_misses_points_in_that_case(self, ds_service_address):
        """Guards the premise: the atomic counter is fixing a real defect.

        Timing-dependent by nature --- it races four workers through
        GridSampler's scan path. If Optuna makes that path safe, this fails
        and the counter has one less job to justify it.
        """
        grid = {"a": [-1, 1], "b": [0, 10], "c": [1, 8], "d": [1e-5, 1e-1], "e": [0, 3]}
        storage = create_optuna_storage(ds_service_address, prefix="grid")
        study = optuna.create_study(storage=storage, study_name="grid")
        study.optimize(objective, n_trials=N_CORNERS + 8)

        def worker(_):
            s = create_optuna_storage(ds_service_address, prefix="grid")
            optuna.load_study(
                study_name="grid", storage=s, sampler=optuna.samplers.GridSampler(grid)
            ).optimize(objective, n_trials=N_CORNERS // WORKERS)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(worker, range(WORKERS)))

        loaded = optuna.load_study(
            study_name="grid",
            storage=create_optuna_storage(ds_service_address, prefix="grid"),
        )
        visited = {
            t.system_attrs["grid_id"]
            for t in loaded.trials
            if "grid_id" in t.system_attrs
        }
        assert len(visited) < N_CORNERS  # points silently skipped

    def test_workers_wind_down_together(self, ds_service_address):
        """Each worker asks for the whole walk; between them they do it once."""
        study = self.run_workers(ds_service_address, trials_each=N_CORNERS)

        assert set(corner_ids(study)) == set(range(N_CORNERS))
        # A few workers may be mid-trial when the last corner lands, so
        # allow the overshoot -- but not a second pass over the box.
        assert N_CORNERS <= len(study.trials) < N_CORNERS + WORKERS


class TestUnfinishedCorners:
    def test_a_failed_corner_is_retried_before_any_is_repeated(
        self, storage, backend
    ):
        """A corner whose worker died is worth more than a duplicate."""
        doomed = 5
        sampler = ExtremePointSampler(backend, SPACE)
        doomed_corner = sampler.corner(doomed)

        def flaky(trial):
            value = objective(trial)
            if trial.params == doomed_corner:
                raise RuntimeError("this worker died")
            return value

        study = optuna.create_study(
            storage=storage, study_name="s", sampler=sampler
        )
        study.optimize(flaky, n_trials=N_CORNERS, catch=(RuntimeError,))

        failed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
        ]
        assert [t.system_attrs[CORNER_ID_ATTR] for t in failed] == [doomed]

        # The walk is not done: one corner has no result.
        study.optimize(objective, n_trials=1)

        assert corner_ids(study)[-1] == doomed
        assert study.trials[-1].state == optuna.trial.TrialState.COMPLETE


# --------------------------------------------------------------------------
# Construction and plumbing
# --------------------------------------------------------------------------


class TestConstruction:
    def test_a_categorical_is_rejected(self, backend):
        with pytest.raises(ValueError, match="unordered"):
            ExtremePointSampler(
                backend, {"c": CategoricalDistribution(["a", "b"])}
            )

    def test_an_empty_space_is_rejected(self, backend):
        with pytest.raises(ValueError, match="must not be empty"):
            ExtremePointSampler(backend, {})

    def test_a_foreign_storage_is_rejected(self, tmp_path):
        other = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(tmp_path / "j.log"))
        )

        with pytest.raises(TypeError, match="DsServiceJournalBackend"):
            ExtremePointSampler(other, SPACE)

    def test_it_accepts_the_storage_or_the_backend(self, storage, backend):
        assert ExtremePointSampler(storage, SPACE)._backend is storage._backend
        assert ExtremePointSampler(backend, SPACE)._backend is backend

    def test_constructing_does_not_connect(self, ds_service_address):
        backend = DsServiceJournalBackend(ds_service_address, prefix="corners")

        ExtremePointSampler(backend, SPACE)

        assert backend._client is None


class TestParametersOutsideTheSpace:
    def test_they_raise_by_default(self, storage, backend):
        """Sampling them would make a deterministic sampler non-deterministic."""
        study = optuna.create_study(
            storage=storage, study_name="s", sampler=ExtremePointSampler(backend, SPACE)
        )

        def extra(trial):
            return objective(trial) + trial.suggest_float("undeclared", 0, 1)

        with pytest.raises(ValueError, match="not in this sampler's search space"):
            study.optimize(extra, n_trials=1)

    def test_an_independent_sampler_opts_into_them(self, storage, backend):
        sampler = ExtremePointSampler(
            backend, SPACE, independent_sampler=optuna.samplers.RandomSampler(seed=1)
        )
        study = optuna.create_study(storage=storage, study_name="s", sampler=sampler)

        def extra(trial):
            return objective(trial) + trial.suggest_float("undeclared", 0, 1)

        study.optimize(extra, n_trials=4)

        assert len(study.trials) == 4
        # The declared parameters stay on their corners regardless.
        for trial in study.trials:
            corner = sampler.corner(trial.system_attrs[CORNER_ID_ATTR])
            assert {k: trial.params[k] for k in corner} == corner
            assert 0 <= trial.params["undeclared"] <= 1


class TestKeying:
    def test_a_changed_space_starts_a_new_walk(self, storage, backend):
        study = optuna.create_study(
            storage=storage, study_name="s", sampler=ExtremePointSampler(backend, SPACE)
        )
        study.optimize(objective, n_trials=4)

        other_space = {"z": FloatDistribution(0, 1)}
        other = ExtremePointSampler(backend, other_space)
        study.sampler = other

        study.optimize(lambda t: t.suggest_float("z", 0, 1), n_trials=1)

        assert corner_ids(study)[-1] == 0

    def test_separate_studies_walk_independently(self, storage, backend):
        first = optuna.create_study(
            storage=storage, study_name="one", sampler=ExtremePointSampler(backend, SPACE)
        )
        second = optuna.create_study(
            storage=storage, study_name="two", sampler=ExtremePointSampler(backend, SPACE)
        )

        first.optimize(objective, n_trials=3)
        second.optimize(objective, n_trials=3)

        assert corner_ids(first) == [0, 1, 2]
        assert corner_ids(second) == [0, 1, 2]


class TestPickling:
    """The sampler rides along with the study into a pilot worker."""

    @pytest.mark.parametrize("dumps", [pickle.dumps, cloudpickle.dumps])
    def test_a_round_tripped_sampler_keeps_walking(self, ds_service_address, dumps):
        storage = create_optuna_storage(ds_service_address, prefix="corners")
        study = optuna.create_study(
            storage=storage,
            study_name="pickled",
            sampler=ExtremePointSampler(storage, SPACE),
        )
        study.optimize(objective, n_trials=4)

        revived = pickle.loads(dumps(study))
        revived.optimize(objective, n_trials=4)

        loaded = optuna.load_study(
            study_name="pickled",
            storage=create_optuna_storage(ds_service_address, prefix="corners"),
        )
        assert sorted(corner_ids(loaded)) == list(range(8))

    def test_the_connection_does_not_travel(self, ds_service_address):
        storage = create_optuna_storage(ds_service_address, prefix="corners")
        sampler = ExtremePointSampler(storage, SPACE)
        sampler._backend._get_client()

        revived = pickle.loads(pickle.dumps(sampler))

        assert revived._backend._client is None
        assert revived._backend.server_address == ds_service_address
