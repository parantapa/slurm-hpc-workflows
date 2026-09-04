"""Tests for the `swtop` monitor.

The server is real, as everywhere else here,
so what the collector reports is what a live queue would tell it.
Slurm is mocked, since a worker's identity comes from the store
rather than from a running job.
"""

from __future__ import annotations

import json
from typing import cast
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner
from ds_service_client import DsServiceClient

from slurm_workflows import swtop as swtop_mod
from slurm_workflows.swtop import Collector, Snapshot, render, swtop
from worker_harness import make_worker
from test_monitors import wait_for


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def square(x):
    return x * x


@pytest.fixture
def collector(ds_client, ds_service_address):
    return Collector(ds_client, ds_service_address)


class CountingClient:
    """Counts key reads, so the identity cache can be checked."""

    def __init__(self, inner):
        self._inner = inner
        self.map_gets = 0

    def map_get(self, key):
        self.map_gets += 1
        return self._inner.map_get(key)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# --------------------------------------------------------------------------
# collecting
# --------------------------------------------------------------------------


class TestCollectTasks:
    @pytest.fixture(autouse=True)
    def _pilot_jobs(self, pilot_jobs):
        pilot_jobs("cpu")

    def test_an_idle_server_reports_nothing(self, collector):
        snapshot = collector.snapshot()

        assert snapshot.counts == {
            "ready": 0,
            "running": 0,
            "complete": 0,
            "canceled": 0,
        }
        assert snapshot.workers == []
        assert snapshot.tasks == []

    def test_counts_cover_every_task_named_or_not(self, collector, executor):
        [executor.submit("cpu", square, i) for i in range(3)]

        snapshot = collector.snapshot()

        assert snapshot.counts["ready"] == 3
        assert snapshot.tasks == [], "unnamed tasks are counted, not listed"

    def test_named_tasks_are_listed(self, collector, executor):
        task = executor.submit("cpu", square, 1)
        executor.submit("cpu", square, 2)
        executor.set_task_name(task, "the-named-one")

        (listed,) = collector.snapshot().tasks

        assert listed.name == "the-named-one"
        assert listed.task_id == task.task_id
        assert listed.state == "Ready"
        assert listed.worker == ""

    def test_a_running_task_names_its_worker(
        self, collector, executor, ds_service_address, tmp_path
    ):
        task = executor.submit("cpu", square, 1)
        executor.set_task_name(task, "in-flight")

        worker = make_worker(ds_service_address, tmp_path, group="cpu", name="w-1")
        worker.client.task_get(worker.worker_id, "cpu")

        (listed,) = collector.snapshot().tasks

        assert listed.state == "Running"
        assert listed.worker == "w-1"
        worker.close()

    def test_an_unregistered_holder_is_shown_by_its_id(
        self, collector, executor, ds_client
    ):
        task = executor.submit("cpu", square, 1)
        executor.set_task_name(task, "in-flight")

        # Nothing published an identity for this one.
        ds_client.task_get("a-stranger", "cpu")

        (listed,) = collector.snapshot().tasks

        assert listed.worker == "a-stranger"

    def test_running_tasks_are_listed_first(
        self, collector, executor, ds_client, ds_service_address, tmp_path
    ):
        ready = executor.submit("cpu", square, 1)
        running = executor.submit("cpu", square, 2)
        executor.set_task_name(ready, "b-ready")
        executor.set_task_name(running, "a-running")

        # The queue is oldest first, so this claims `ready` --
        # claim both and let the second one stay Running.
        ds_client.task_get("stranger", "cpu")
        ds_client.task_get("stranger", "cpu")
        ds_client.task_done(ready.task_id, "stranger", b"")

        states = [(t.name, t.state) for t in collector.snapshot().tasks]

        assert states == [("a-running", "Running"), ("b-ready", "Complete")]


class TestCollectWorkers:
    def test_a_worker_appears_once_it_registers(
        self, collector, ds_service_address, tmp_path
    ):
        assert collector.snapshot().workers == []

        worker = make_worker(ds_service_address, tmp_path, group="cpu", name="w-1")

        (listed,) = collector.snapshot().workers
        assert listed.worker_id == worker.worker_id
        assert listed.name == "w-1"
        assert listed.group == "cpu"
        assert listed.hostname == "testhost"
        assert listed.slurm_job_id == "42"
        assert listed.pid == "4242"
        worker.close()

    def test_workers_are_ordered_by_group_then_name(
        self, collector, ds_service_address, tmp_path
    ):
        # Named as the executor names them, because the worker id is built
        # from the name: two workers of one job and pid
        # are told apart by their names alone.
        workers = [
            make_worker(
                ds_service_address, tmp_path, group="gpu", name="run.worker.gpu.0"
            ),
            make_worker(
                ds_service_address, tmp_path, group="cpu", name="run.worker.cpu.1"
            ),
            make_worker(
                ds_service_address, tmp_path, group="cpu", name="run.worker.cpu.0"
            ),
        ]

        listed = [(w.group, w.name) for w in collector.snapshot().workers]

        assert listed == [
            ("cpu", "run.worker.cpu.0"),
            ("cpu", "run.worker.cpu.1"),
            ("gpu", "run.worker.gpu.0"),
        ]
        for worker in workers:
            worker.close()

    def test_an_identity_is_read_once_however_long_it_runs(
        self, ds_client, ds_service_address, tmp_path
    ):
        """The published description never changes, so re-reading it is waste."""
        worker = make_worker(ds_service_address, tmp_path, group="cpu", name="w-1")
        counting = CountingClient(ds_client)
        # Forwards everything it does not count, as the worker harness's
        # doubles do, so it stands in for a client without subclassing one.
        collector = Collector(cast(DsServiceClient, counting), ds_service_address)

        collector.snapshot()
        after_first = counting.map_gets
        collector.snapshot()

        assert after_first == 1, "the whole description is one key"
        assert counting.map_gets == after_first
        worker.close()

    def test_a_description_that_cannot_be_read_is_shown_as_unknown(
        self, collector, ds_client
    ):
        ds_client.map_set("worker_info:something-else", b"not json")

        (listed,) = collector.snapshot().workers

        assert listed.worker_id == "something-else"
        assert listed.name == "?"

    def test_an_unreadable_description_is_not_cached(self, collector, ds_client):
        """It may be a writer this reader arrived in the middle of."""
        ds_client.map_set("worker_info:w", b"not json")
        collector.snapshot()

        ds_client.map_set(
            "worker_info:w",
            json.dumps(
                {
                    "group": "cpu",
                    "name": "w-1",
                    "slurm_job_id": 42,
                    "hostname": "testhost",
                    "pid": 1,
                }
            ).encode(),
        )

        (listed,) = collector.snapshot().workers
        assert listed.name == "w-1"


class TestCollectMonitored:
    """Host and job readings, as the monitors leave them in the store."""

    def test_nothing_is_monitored_on_an_idle_server(self, collector):
        snapshot = collector.snapshot()

        assert snapshot.hosts == []
        assert snapshot.jobs == []

    def test_a_monitored_host_and_job_appear(
        self, collector, ds_client, ds_service_address, tmp_path
    ):
        worker = make_worker(ds_service_address, tmp_path)
        assert wait_for(
            lambda: bool(ds_client.time_series_get("host_free_memory:testhost"))
        )

        snapshot = collector.snapshot()

        (host,) = snapshot.hosts
        assert host.subject == "testhost"
        assert host.values["free_memory"] > 0
        assert not host.stale

        (job,) = snapshot.jobs
        assert job.subject == "42"
        assert job.values["memory"] > 0
        worker.close()

    def test_the_latest_point_is_the_one_shown(self, collector, ds_client):
        for value in [1.0, 2.0, 3.0]:
            ds_client.time_series_append("host_load_average:node-1", value, _now_utc())
        ds_client.time_series_append("host_free_memory:node-1", 5.0, _now_utc())

        (host,) = collector.snapshot().hosts

        assert host.values["load_average"] == 3.0

    def test_a_subject_with_only_old_points_is_stale(self, collector, ds_client):
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        ds_client.time_series_append("host_free_memory:node-1", 5.0, old)

        (host,) = collector.snapshot().hosts

        assert host.subject == "node-1"
        assert host.values == {}
        assert host.stale

    def test_a_series_that_never_started_leaves_its_column_out(
        self, collector, ds_client
    ):
        """Only one of a host's four series has to exist for it to be listed."""
        ds_client.time_series_append("host_free_memory:node-1", 5.0, _now_utc())

        (host,) = collector.snapshot().hosts

        assert set(host.values) == {"free_memory"}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


class TestRender:
    @pytest.fixture(autouse=True)
    def _pilot_jobs(self, pilot_jobs):
        pilot_jobs("cpu")

    def test_an_idle_server_says_so(self, collector):
        out = render(collector.snapshot())

        assert "total 0" in out
        assert "no workers have registered" in out
        assert "no tasks have been named" in out

    def test_the_tables_carry_the_data(
        self, collector, executor, ds_service_address, tmp_path
    ):
        task = executor.submit("cpu", square, 1)
        executor.set_task_name(task, "the-named-one")
        worker = make_worker(ds_service_address, tmp_path, group="cpu", name="w-1")

        out = render(collector.snapshot())

        assert "workers (1)" in out
        assert "named tasks (1)" in out
        for expected in ["w-1", "testhost", "4242", "the-named-one", task.task_id]:
            assert expected in out
        worker.close()

    def test_an_unmonitored_server_says_so(self, collector):
        out = render(collector.snapshot())

        assert "no host is being monitored" in out
        assert "no slurm job is being monitored" in out

    def test_the_readings_are_shown_in_readable_units(self, collector, ds_client):
        for key, value in [
            ("host_free_memory:node-1", 2 * 1024**3),
            ("host_load_average:node-1", 3.5),
            ("host_dev_shm_used:node-1", 12.25),
            ("host_tmp_used:node-1", 46.9),
            ("slurm_job_memory:1846231", 1024**3),
            ("slurm_job_cpu:1846231", 12.4),
        ]:
            ds_client.time_series_append(key, value, _now_utc())

        out = render(collector.snapshot())

        assert "2.0G" in out
        assert "3.50" in out
        assert "12.2%" in out
        assert "1.0G" in out
        assert "12.4 cores" in out

    def test_a_stale_subject_is_labelled_not_dropped(self, collector, ds_client):
        """A monitor that died is worth seeing, not hiding."""
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        ds_client.time_series_append("host_free_memory:node-1", 5.0, old)

        out = render(collector.snapshot())

        assert "node-1 (stale)" in out
        assert "hosts (1)" in out

    def test_a_missing_measurement_is_a_dash(self, collector, ds_client):
        # A subject name without a dash of its own,
        # so the dashes asserted below can only be the empty columns.
        ds_client.time_series_append("host_free_memory:nodeone", 5.0, _now_utc())

        out = render(collector.snapshot())

        row = next(line for line in out.splitlines() if line.startswith("nodeone"))
        assert row.split() == ["nodeone", "5.0B", "-", "-", "-"]

    def test_a_failed_poll_is_reported_in_place_of_the_tables(self, collector):
        snapshot = collector.snapshot()
        snapshot.error = "TimeoutError: server unreachable"

        out = render(snapshot)

        assert "cannot read the server: TimeoutError" in out
        assert "workers" not in out

    def test_every_line_fits_together(self, collector):
        """Columns are padded, so no row may be ragged or unterminated."""
        out = render(collector.snapshot())

        assert out.endswith("\n")
        assert not any(line.endswith(" ") for line in out.splitlines())


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------


class TestCli:
    @pytest.fixture
    def stop_after_one_poll(self, monkeypatch):
        """Let one frame be drawn, then interrupt as a user would."""

        def sleep(seconds):
            raise KeyboardInterrupt

        monkeypatch.setattr(swtop_mod.time, "sleep", sleep)

    def test_it_polls_and_stops_on_interrupt(
        self, ds_service_address, stop_after_one_poll
    ):
        result = CliRunner().invoke(swtop, [ds_service_address])

        assert result.exit_code == 0
        assert "swtop" in result.output
        assert "total 0" in result.output

    def test_the_interval_defaults_to_two_seconds(
        self, ds_service_address, monkeypatch
    ):
        slept = []

        def sleep(seconds):
            slept.append(seconds)
            raise KeyboardInterrupt

        monkeypatch.setattr(swtop_mod.time, "sleep", sleep)

        CliRunner().invoke(swtop, [ds_service_address], catch_exceptions=False)

        assert slept == [2.0]

    def test_the_interval_can_be_set(self, ds_service_address, monkeypatch):
        slept = []

        def sleep(seconds):
            slept.append(seconds)
            raise KeyboardInterrupt

        monkeypatch.setattr(swtop_mod.time, "sleep", sleep)

        CliRunner().invoke(swtop, [ds_service_address, "-i", "0.5"])

        assert slept == [0.5]

    @pytest.mark.parametrize("interval", ["0", "-1"])
    def test_a_non_positive_interval_is_rejected(self, ds_service_address, interval):
        result = CliRunner().invoke(swtop, [ds_service_address, "-i", interval])

        assert result.exit_code != 0

    def test_the_address_is_required(self):
        result = CliRunner().invoke(swtop, [])

        assert result.exit_code != 0

    def test_an_unreachable_server_is_reported_not_fatal(self, stop_after_one_poll):
        """A monitor that quits when the server blinks is not much of a monitor."""
        result = CliRunner().invoke(swtop, ["127.0.0.1:1"])

        assert result.exit_code == 0
        assert "cannot read the server" in result.output


def test_a_snapshot_needs_only_an_address_and_a_time():
    """The error path builds one without ever reaching the server."""
    snapshot = Snapshot(address="host:1", when=swtop_mod.datetime.now())

    assert snapshot.counts == {}
    assert snapshot.workers == []
    assert snapshot.tasks == []
