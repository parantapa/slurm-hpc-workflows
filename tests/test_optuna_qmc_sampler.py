"""Tests for the distributed-safe QMC sampler.

Every claim this sampler makes is about what happens when several workers
run at once, so the tests that matter here run real concurrent workers
against a real ds-service and assert on the trials that came out. Where a
fix exists to repair a specific base-class behaviour, the base class is run
alongside it, so a test failing tells you which of the two changed.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest

optuna = pytest.importorskip("optuna")
pytest.importorskip("scipy")  # QMC engines live in scipy.stats.qmc

import cloudpickle  # noqa: E402
from optuna.distributions import (  # noqa: E402
    CategoricalDistribution,
    FloatDistribution,
)

from slurm_workflows.optuna_storage import (  # noqa: E402
    DsServiceJournalBackend,
    create_optuna_storage,
)
from slurm_workflows.optuna_qmc_sampler import DsServiceQMCSampler  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.ERROR)

SEARCH_SPACE = {
    "x": FloatDistribution(-1, 1),
    "y": FloatDistribution(-1, 1),
}

WORKERS = 4
TRIALS_PER_WORKER = 8

# Sample id 0 of an unscrambled Sobol sequence is the origin of the unit
# cube, which the search-space transform maps to the lower corner. Nothing
# else in the sequence lands there, and a random draw hits it with
# probability zero, so this point identifies "this trial was QMC id 0".
SAMPLE_ID_0 = (-1.0, -1.0)


def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_float("y", -1, 1)
    return x**2 + y**2


def points(study) -> list[tuple]:
    return [(t.params.get("x"), t.params.get("y")) for t in study.trials]


def qmc_trial_count(address, prefix, study) -> int:
    """How many trials of `study` were drawn from the QMC sequence.

    The counter is incremented exactly once per QMC-sampled trial, so
    reading it tells the QMC trials apart from independent-sampler
    fallbacks without having to reconstruct the lattice.
    """
    backend = DsServiceJournalBackend(address, prefix=prefix)
    try:
        key = DsServiceQMCSampler(backend)._sample_id_key(study, SEARCH_SPACE)
        return backend._get_client().counter_get_current_value(key)
    finally:
        backend.close()


def run_workers(address, study_name, make_sampler, prefix="qmc", **create_kwargs):
    """Run WORKERS workers concurrently, each with its own storage."""
    optuna.create_study(
        storage=create_optuna_storage(address, prefix=prefix),
        study_name=study_name,
        load_if_exists=True,
        **create_kwargs,
    )

    def worker(_):
        storage = create_optuna_storage(address, prefix=prefix)
        study = optuna.load_study(
            study_name=study_name, storage=storage, sampler=make_sampler(storage)
        )
        study.optimize(objective, n_trials=TRIALS_PER_WORKER)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(worker, range(WORKERS)))

    return optuna.load_study(
        study_name=study_name, storage=create_optuna_storage(address, prefix=prefix)
    )


@pytest.fixture
def backend(ds_service_address: str):
    b = DsServiceJournalBackend(ds_service_address, prefix="qmc")
    yield b
    b.close()


# --------------------------------------------------------------------------
# Issue 1: sample ids must be allocated atomically
# --------------------------------------------------------------------------


class TestSampleIdAllocation:
    def test_concurrent_workers_never_repeat_a_point(self, ds_service_address):
        """The whole point: no two workers may evaluate the same sample id."""
        study = run_workers(
            ds_service_address,
            "atomic",
            lambda storage: DsServiceQMCSampler(storage, search_space=SEARCH_SPACE),
        )

        pts = points(study)
        assert len(pts) == WORKERS * TRIALS_PER_WORKER
        assert len(set(pts)) == len(pts)

    def test_the_base_sampler_does_repeat_points(self, ds_service_address):
        """Guards the premise: the fix is repairing a real defect.

        If Optuna ever makes this allocation atomic, this fails and the
        subclass can lose its counter.
        """
        study = run_workers(
            ds_service_address,
            "base",
            lambda storage: optuna.samplers.QMCSampler(seed=7),
            prefix="base",
        )

        pts = points(study)
        assert len(set(pts)) < len(pts)

    def test_ids_are_consecutive_from_zero(self, ds_service_address, backend):
        """A gap or a repeat in the ids means a hole in the sequence."""
        sampler = DsServiceQMCSampler(backend, search_space=SEARCH_SPACE)
        study = optuna.create_study(
            storage=create_optuna_storage(ds_service_address, prefix="qmc"),
            study_name="ids",
            sampler=sampler,
        )

        ids = [sampler._find_sample_id(study, SEARCH_SPACE) for _ in range(5)]

        assert ids == [0, 1, 2, 3, 4]

    def test_a_changed_search_space_starts_a_new_sequence(
        self, ds_service_address, backend
    ):
        study = optuna.create_study(
            storage=create_optuna_storage(ds_service_address, prefix="qmc"),
            study_name="rekey",
            sampler=DsServiceQMCSampler(backend, search_space=SEARCH_SPACE),
        )
        sampler = DsServiceQMCSampler(backend, search_space=SEARCH_SPACE)

        assert sampler._find_sample_id(study, SEARCH_SPACE) == 0
        assert sampler._find_sample_id(study, SEARCH_SPACE) == 1
        # A different space is a different sequence, so it starts over.
        other = {"x": FloatDistribution(-5, 5)}
        assert sampler._find_sample_id(study, other) == 0

    def test_separate_studies_have_separate_counters(
        self, ds_service_address, backend
    ):
        storage = create_optuna_storage(ds_service_address, prefix="qmc")
        first = optuna.create_study(storage=storage, study_name="one")
        second = optuna.create_study(storage=storage, study_name="two")
        sampler = DsServiceQMCSampler(backend, search_space=SEARCH_SPACE)

        assert sampler._find_sample_id(first, SEARCH_SPACE) == 0
        assert sampler._find_sample_id(first, SEARCH_SPACE) == 1
        assert sampler._find_sample_id(second, SEARCH_SPACE) == 0


# --------------------------------------------------------------------------
# Issue 2: workers must scramble with one shared seed
# --------------------------------------------------------------------------


class TestSeedAgreement:
    def test_workers_converge_on_one_seed_without_being_told_it(
        self, ds_service_address, backend
    ):
        storage = create_optuna_storage(ds_service_address, prefix="qmc")
        study = optuna.create_study(storage=storage, study_name="seed")
        samplers = [
            DsServiceQMCSampler(backend, scramble=True) for _ in range(WORKERS)
        ]

        # Each starts with its own random seed ...
        assert len({int(s._seed) for s in samplers}) > 1

        for sampler in samplers:
            sampler.before_trial(study, study.ask())

        # ... and ends up on the one the first of them published.
        assert len({s._seed for s in samplers}) == 1

    def test_an_explicit_seed_is_left_alone(self, ds_service_address, backend):
        storage = create_optuna_storage(ds_service_address, prefix="qmc")
        study = optuna.create_study(storage=storage, study_name="explicit")
        sampler = DsServiceQMCSampler(backend, scramble=True, seed=1234)

        sampler.before_trial(study, study.ask())

        assert sampler._seed == 1234

    def test_no_negotiation_happens_without_scrambling(self, backend):
        """Unscrambled Sobol/Halton ignore the seed; don't pay for a round trip."""
        sampler = DsServiceQMCSampler(backend, scramble=False)

        assert sampler._seed_agreed
        assert backend._client is None

    def test_scrambled_workers_produce_one_coherent_sequence(
        self, ds_service_address
    ):
        study = run_workers(
            ds_service_address,
            "scrambled",
            lambda storage: DsServiceQMCSampler(
                storage, scramble=True, search_space=SEARCH_SPACE
            ),
        )

        pts = points(study)
        assert len(set(pts)) == len(pts)
        # Scrambling is on, so nothing should land on the unscrambled lattice.
        assert SAMPLE_ID_0 not in pts


# --------------------------------------------------------------------------
# Issue 3: the fallback sampler must not be seeded in lockstep
# --------------------------------------------------------------------------


class TestIndependentSampler:
    def test_workers_sharing_a_seed_still_draw_different_fallbacks(
        self, ds_service_address, backend
    ):
        """A shared QMC seed must not make the random fallback identical."""
        samplers = [
            DsServiceQMCSampler(backend, scramble=True, seed=7) for _ in range(3)
        ]
        storage = create_optuna_storage(ds_service_address, prefix="qmc")

        drawn = []
        for i, sampler in enumerate(samplers):
            study = optuna.create_study(
                storage=storage, study_name=f"fallback-{i}", sampler=sampler
            )
            study.optimize(objective, n_trials=1)
            drawn.append(points(study)[0])

        assert len(set(drawn)) == len(drawn)

    def test_the_base_sampler_draws_them_identically(self):
        """Guards the premise for the test above."""
        drawn = []
        for _ in range(3):
            study = optuna.create_study(
                sampler=optuna.samplers.QMCSampler(scramble=True, seed=7)
            )
            study.optimize(objective, n_trials=1)
            drawn.append(points(study)[0])

        assert len(set(drawn)) == 1

    def test_an_explicit_independent_sampler_is_respected(self, backend):
        given = optuna.samplers.TPESampler()

        sampler = DsServiceQMCSampler(backend, independent_sampler=given)

        assert sampler._independent_sampler is given


# --------------------------------------------------------------------------
# Declared search space: QMC from trial 0
# --------------------------------------------------------------------------


class TestDeclaredSearchSpace:
    def test_the_first_trial_is_already_qmc(self, ds_service_address, backend):
        """Undeclared, trial 0 is random; declared, it is sample id 0."""
        storage = create_optuna_storage(ds_service_address, prefix="qmc")
        study = optuna.create_study(
            storage=storage,
            study_name="declared",
            sampler=DsServiceQMCSampler(backend, search_space=SEARCH_SPACE),
        )

        study.optimize(objective, n_trials=1)

        assert points(study)[0] == SAMPLE_ID_0

    def test_without_it_the_first_trial_falls_back(
        self, ds_service_address, backend
    ):
        storage = create_optuna_storage(ds_service_address, prefix="qmc")
        study = optuna.create_study(
            storage=storage,
            study_name="undeclared",
            sampler=DsServiceQMCSampler(backend),
        )

        study.optimize(objective, n_trials=1)

        assert points(study)[0] != SAMPLE_ID_0

    def test_every_worker_starts_on_qmc(self, ds_service_address):
        """With N workers, the first N trials are the ones to lose to random."""
        total = WORKERS * TRIALS_PER_WORKER

        declared = run_workers(
            ds_service_address,
            "declared-par",
            lambda storage: DsServiceQMCSampler(storage, search_space=SEARCH_SPACE),
        )
        undeclared = run_workers(
            ds_service_address,
            "undeclared-par",
            lambda storage: DsServiceQMCSampler(storage),
            prefix="qmc2",
        )

        assert len(points(declared)) == len(points(undeclared)) == total
        # Declared, every trial came from the sequence.
        assert qmc_trial_count(ds_service_address, "qmc", declared) == total
        # Undeclared, at least the worker that got there first had no
        # finished trial to infer a search space from, and sampled randomly.
        assert qmc_trial_count(ds_service_address, "qmc2", undeclared) < total

    def test_a_categorical_distribution_is_rejected(self, backend):
        with pytest.raises(ValueError, match="CategoricalDistribution"):
            DsServiceQMCSampler(
                backend,
                search_space={"c": CategoricalDistribution(["a", "b"])},
            )

    def test_an_empty_search_space_is_rejected(self, backend):
        with pytest.raises(ValueError, match="must not be empty"):
            DsServiceQMCSampler(backend, search_space={})

    def test_a_parameter_outside_the_declared_space_still_works(
        self, ds_service_address, backend
    ):
        """Anything undeclared goes to the independent sampler, as usual."""
        storage = create_optuna_storage(ds_service_address, prefix="qmc")
        study = optuna.create_study(
            storage=storage,
            study_name="extra",
            sampler=DsServiceQMCSampler(backend, search_space=SEARCH_SPACE),
        )

        def with_extra(trial):
            base = objective(trial)
            return base + trial.suggest_categorical("kind", ["a", "b"]).count("a")

        study.optimize(with_extra, n_trials=4)

        assert len(study.trials) == 4
        assert all(t.params["kind"] in ("a", "b") for t in study.trials)


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


class TestConstruction:
    def test_it_accepts_the_storage(self, ds_service_address):
        storage = create_optuna_storage(ds_service_address, prefix="qmc")

        sampler = DsServiceQMCSampler(storage)

        assert sampler._backend is storage._backend

    def test_it_accepts_the_backend(self, backend):
        sampler = DsServiceQMCSampler(backend)

        assert sampler._backend is backend

    def test_it_rejects_a_foreign_storage(self, tmp_path):
        other = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(tmp_path / "j.log"))
        )

        with pytest.raises(TypeError, match="DsServiceJournalBackend"):
            DsServiceQMCSampler(other)

    def test_it_rejects_a_bad_qmc_type(self, backend):
        with pytest.raises(ValueError, match="halton"):
            DsServiceQMCSampler(backend, qmc_type="lattice")

    def test_constructing_does_not_connect(self, ds_service_address):
        backend = DsServiceJournalBackend(ds_service_address, prefix="qmc")

        DsServiceQMCSampler(backend, scramble=True, search_space=SEARCH_SPACE)

        assert backend._client is None


class TestPickling:
    """The sampler rides along with the study into a pilot worker."""

    @pytest.mark.parametrize("dumps", [pickle.dumps, cloudpickle.dumps])
    def test_a_round_tripped_sampler_still_samples(self, ds_service_address, dumps):
        storage = create_optuna_storage(ds_service_address, prefix="qmc")
        study = optuna.create_study(
            storage=storage,
            study_name="pickled",
            sampler=DsServiceQMCSampler(storage, search_space=SEARCH_SPACE),
        )
        study.optimize(objective, n_trials=2)

        revived = pickle.loads(dumps(study))
        revived.optimize(objective, n_trials=2)

        pts = points(
            optuna.load_study(
                study_name="pickled",
                storage=create_optuna_storage(ds_service_address, prefix="qmc"),
            )
        )
        assert len(pts) == 4
        # The revived sampler kept counting from where the original left off.
        assert len(set(pts)) == 4

    def test_the_connection_does_not_travel(self, ds_service_address):
        storage = create_optuna_storage(ds_service_address, prefix="qmc")
        sampler = DsServiceQMCSampler(storage, scramble=True)
        sampler._backend._get_client()

        revived = pickle.loads(pickle.dumps(sampler))

        assert revived._backend._client is None
        assert revived._backend.server_address == ds_service_address
