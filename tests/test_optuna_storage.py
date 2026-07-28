"""Tests for the ds-service-backed Optuna storage.

These run against a real ds-service and a real `optuna.storages.JournalStorage`
--- the backend's whole job is to satisfy a contract owned by Optuna, so a
stand-in for either side would only test this file's idea of that contract.
"""

from __future__ import annotations

import pickle

import pytest

optuna = pytest.importorskip("optuna")

import cloudpickle  # noqa: E402

from slurm_workflows.optuna_storage import (  # noqa: E402
    DsServiceJournalBackend,
    create_optuna_storage,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture
def backend(ds_service_address: str):
    b = DsServiceJournalBackend(ds_service_address, prefix="test")
    yield b
    b.close()


def make_backend(address: str, **kwargs):
    return DsServiceJournalBackend(address, **kwargs)


# --------------------------------------------------------------------------
# Backend contract
# --------------------------------------------------------------------------


class TestReadAppend:
    def test_an_empty_journal_reads_as_nothing(self, backend):
        assert list(backend.read_logs(0)) == []

    def test_logs_come_back_in_order(self, backend):
        logs = [{"op_code": i, "value": f"v{i}"} for i in range(5)]
        backend.append_logs(logs)

        assert list(backend.read_logs(0)) == logs

    def test_log_number_from_skips_the_prefix(self, backend):
        logs = [{"op_code": i} for i in range(5)]
        backend.append_logs(logs)

        assert list(backend.read_logs(2)) == logs[2:]
        assert list(backend.read_logs(5)) == []
        # Past the end is empty, not an error: another process may have read
        # a longer journal than this one has caught up with.
        assert list(backend.read_logs(99)) == []

    def test_appends_accumulate_across_calls(self, backend):
        backend.append_logs([{"n": 0}])
        backend.append_logs([{"n": 1}, {"n": 2}])

        assert list(backend.read_logs(0)) == [{"n": 0}, {"n": 1}, {"n": 2}]

    def test_reading_is_chunked_but_seamless(self, ds_service_address):
        """A chunk size below the journal length must not change the result."""
        writer = make_backend(ds_service_address, prefix="chunked")
        logs = [{"n": i} for i in range(25)]
        writer.append_logs(logs)

        reader = make_backend(ds_service_address, prefix="chunked", read_chunk_size=4)
        try:
            assert list(reader.read_logs(0)) == logs
            assert list(reader.read_logs(23)) == logs[23:]
        finally:
            reader.close()
            writer.close()

    def test_a_second_backend_sees_the_first_ones_logs(self, ds_service_address):
        first = make_backend(ds_service_address, prefix="shared")
        second = make_backend(ds_service_address, prefix="shared")
        try:
            first.append_logs([{"from": "first"}])
            assert list(second.read_logs(0)) == [{"from": "first"}]

            second.append_logs([{"from": "second"}])
            assert list(first.read_logs(1)) == [{"from": "second"}]
        finally:
            second.close()
            first.close()

    def test_prefixes_are_independent(self, ds_service_address):
        a = make_backend(ds_service_address, prefix="a")
        b = make_backend(ds_service_address, prefix="b")
        try:
            a.append_logs([{"who": "a"}])

            assert list(a.read_logs(0)) == [{"who": "a"}]
            assert list(b.read_logs(0)) == []
        finally:
            b.close()
            a.close()

    def test_non_ascii_survives_the_round_trip(self, backend):
        backend.append_logs([{"study_name": "café-λ"}])

        assert list(backend.read_logs(0)) == [{"study_name": "café-λ"}]


class TestSnapshot:
    def test_absent_snapshot_reads_as_none(self, backend):
        assert backend.load_snapshot() is None

    def test_a_saved_snapshot_comes_back(self, backend):
        backend.save_snapshot(b"\x00 binary payload \xff")

        assert backend.load_snapshot() == b"\x00 binary payload \xff"

    def test_the_latest_snapshot_wins(self, backend):
        backend.save_snapshot(b"old")
        backend.save_snapshot(b"new")

        assert backend.load_snapshot() == b"new"

    def test_snapshots_follow_the_prefix(self, ds_service_address):
        a = make_backend(ds_service_address, prefix="a")
        b = make_backend(ds_service_address, prefix="b")
        try:
            a.save_snapshot(b"only-a")

            assert b.load_snapshot() is None
        finally:
            b.close()
            a.close()

    def test_an_unstorable_snapshot_warns_instead_of_raising(self, backend, caplog):
        """A snapshot is a replay shortcut; losing one must not fail a run."""

        def too_large(key, value):
            raise ValueError("Sent message larger than max")

        backend._get_client().map_set = too_large

        backend.save_snapshot(b"x" * 16)

        assert backend.load_snapshot() is None
        assert "too large" in caplog.text


class TestConstruction:
    def test_the_address_defaults_to_the_environment(self, monkeypatch):
        monkeypatch.setenv("DS_SERVER_ADDRESS", "10.0.0.1:5000")

        backend = DsServiceJournalBackend()

        assert backend.server_address == "10.0.0.1:5000"

    def test_a_missing_address_is_rejected_at_construction(self, monkeypatch):
        monkeypatch.delenv("DS_SERVER_ADDRESS", raising=False)

        with pytest.raises(ValueError, match="DS_SERVER_ADDRESS"):
            DsServiceJournalBackend()

    def test_a_nonsense_chunk_size_is_rejected(self):
        with pytest.raises(ValueError, match="read_chunk_size"):
            DsServiceJournalBackend("127.0.0.1:1", read_chunk_size=0)

    def test_constructing_does_not_connect(self):
        """Nothing may touch the network until the storage is used."""
        backend = DsServiceJournalBackend("127.0.0.1:1", prefix="unreachable")

        assert backend._client is None


class TestPickling:
    """The backend has to reach compute nodes inside a cloudpickled task."""

    @pytest.mark.parametrize("dumps", [pickle.dumps, cloudpickle.dumps])
    def test_a_round_tripped_backend_still_works(self, ds_service_address, dumps):
        original = make_backend(ds_service_address, prefix="pickled")
        original.append_logs([{"before": "pickling"}])

        revived = pickle.loads(dumps(original))
        try:
            assert revived.server_address == ds_service_address
            assert revived.prefix == "pickled"
            # Reconnects on first use rather than carrying a dead channel.
            assert revived._client is None
            assert list(revived.read_logs(0)) == [{"before": "pickling"}]

            revived.append_logs([{"after": "pickling"}])
            assert list(original.read_logs(1)) == [{"after": "pickling"}]
        finally:
            revived.close()
            original.close()

    def test_the_connection_is_not_part_of_the_state(self, backend):
        backend.append_logs([{"connect": True}])  # force a connection
        assert backend._client is not None

        assert backend.__getstate__()["_client"] is None
        # Introspection must not disturb the live backend.
        assert backend._client is not None


# --------------------------------------------------------------------------
# Through Optuna
# --------------------------------------------------------------------------


class TestOptunaStorage:
    def test_a_study_optimizes_and_records_its_trials(self, ds_service_address):
        storage = create_optuna_storage(ds_service_address, prefix="opt")
        study = optuna.create_study(storage=storage, study_name="s")

        study.optimize(lambda t: (t.suggest_float("x", -10, 10) - 2) ** 2, n_trials=15)

        assert len(study.trials) == 15
        assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        assert study.best_value == min(t.value for t in study.trials)
        assert "x" in study.best_params

    def test_a_study_is_visible_to_another_process_view(self, ds_service_address):
        """The point of a shared storage: reopen it and the trials are there."""
        writer = create_optuna_storage(ds_service_address, prefix="opt")
        study = optuna.create_study(storage=writer, study_name="s")
        study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=5)

        reader = create_optuna_storage(ds_service_address, prefix="opt")
        loaded = optuna.load_study(storage=reader, study_name="s")

        assert len(loaded.trials) == 5
        assert loaded.best_value == study.best_value
        assert [t.params for t in loaded.trials] == [t.params for t in study.trials]

    def test_two_storages_interleave_trials_on_one_study(self, ds_service_address):
        """Two workers against one study, as a pilot job would run them."""
        a = create_optuna_storage(ds_service_address, prefix="opt")
        study_a = optuna.create_study(storage=a, study_name="s")

        b = create_optuna_storage(ds_service_address, prefix="opt")
        study_b = optuna.load_study(storage=b, study_name="s")

        for _ in range(4):
            study_a.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)
            study_b.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)

        assert len(study_a.trials) == 8
        assert len(study_b.trials) == 8
        assert sorted(t.number for t in study_a.trials) == list(range(8))

    def test_attributes_and_intermediate_values_round_trip(self, ds_service_address):
        storage = create_optuna_storage(ds_service_address, prefix="attrs")
        study = optuna.create_study(storage=storage, study_name="s")
        study.set_user_attr("owner", "pb")

        def objective(trial):
            trial.set_user_attr("tag", "first")
            trial.report(0.5, step=0)
            trial.report(0.25, step=1)
            return trial.suggest_int("n", 1, 3)

        study.optimize(objective, n_trials=1)

        loaded = optuna.load_study(
            storage=create_optuna_storage(ds_service_address, prefix="attrs"),
            study_name="s",
        )
        assert loaded.user_attrs == {"owner": "pb"}
        assert loaded.trials[0].user_attrs == {"tag": "first"}
        assert loaded.trials[0].intermediate_values == {0: 0.5, 1: 0.25}

    def test_studies_under_different_prefixes_do_not_collide(self, ds_service_address):
        first = create_optuna_storage(ds_service_address, prefix="one")
        optuna.create_study(storage=first, study_name="same-name")

        second = create_optuna_storage(ds_service_address, prefix="two")
        # A colliding name would raise DuplicatedStudyError.
        optuna.create_study(storage=second, study_name="same-name")

        assert len(optuna.get_all_study_names(storage=second)) == 1

    def test_a_multi_objective_study_round_trips(self, ds_service_address):
        storage = create_optuna_storage(ds_service_address, prefix="multi")
        study = optuna.create_study(
            storage=storage, study_name="s", directions=["minimize", "maximize"]
        )

        study.optimize(lambda t: (t.suggest_float("x", 0, 1), 1.0), n_trials=3)

        loaded = optuna.load_study(
            storage=create_optuna_storage(ds_service_address, prefix="multi"),
            study_name="s",
        )
        assert loaded.directions == study.directions
        assert [t.values for t in loaded.trials] == [t.values for t in study.trials]

    def test_a_small_chunk_size_does_not_change_what_optuna_sees(
        self, ds_service_address
    ):
        """Chunked replay must reconstruct the same state as one big read."""
        storage = create_optuna_storage(ds_service_address, prefix="chunk")
        study = optuna.create_study(storage=storage, study_name="s")
        study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=10)

        chunked = create_optuna_storage(
            ds_service_address, prefix="chunk", read_chunk_size=3
        )
        loaded = optuna.load_study(storage=chunked, study_name="s")

        assert [t.params for t in loaded.trials] == [t.params for t in study.trials]
        assert loaded.best_value == study.best_value

    def test_a_snapshot_is_taken_and_reused(self, ds_service_address, monkeypatch):
        """Optuna snapshots every SNAPSHOT_INTERVAL trials; we must store it."""
        monkeypatch.setattr(
            optuna.storages.journal._storage, "SNAPSHOT_INTERVAL", 5, raising=True
        )
        backend = DsServiceJournalBackend(ds_service_address, prefix="snap")
        study = optuna.create_study(
            storage=optuna.storages.JournalStorage(backend), study_name="s"
        )
        study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=12)

        assert backend.load_snapshot() is not None

        # Restoring from the snapshot must land in the same place as replaying
        # the journal from scratch.
        loaded = optuna.load_study(
            storage=create_optuna_storage(ds_service_address, prefix="snap"),
            study_name="s",
        )
        assert len(loaded.trials) == 12
        assert loaded.best_value == study.best_value
