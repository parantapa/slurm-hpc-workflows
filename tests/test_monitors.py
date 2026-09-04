"""Tests for the host and Slurm job monitors.

The samplers read this machine, so the assertions are about shape and
plausibility rather than exact numbers.
The cgroup reader is pointed at files written by the test,
which is the only way to assert on values a kernel decides.
"""

from __future__ import annotations

import time

import psutil
import pytest

from slurm_workflows import monitors as monitors_mod
from slurm_workflows.monitors import (
    HOST_SERIES,
    JOB_SERIES,
    CgroupSampler,
    Monitor,
    sample_host,
    start_host_monitor,
    start_slurm_job_monitor,
)


def wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll `predicate` until it holds, or the timeout runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestSampleHost:
    def test_it_reports_memory_load_and_scratch(self):
        values = sample_host()

        assert values["free_memory"] > 0
        assert values["load_average"] >= 0
        for name in ["dev_shm_used", "tmp_used"]:
            assert 0 <= values[name] <= 100

    def test_an_unmounted_filesystem_is_left_out(self, monkeypatch):
        """Absent is not the same as empty, so it gets no value at all."""

        # Kept before patching: `monitors_mod.psutil` is the psutil module
        # itself, so the patch would otherwise replace what this calls.
        real_disk_usage = psutil.disk_usage

        def disk_usage(path):
            if path == "/dev/shm":
                raise OSError("not mounted")
            return real_disk_usage(path)

        monkeypatch.setattr(monitors_mod.psutil, "disk_usage", disk_usage)

        values = sample_host()

        assert "dev_shm_used" not in values
        assert "tmp_used" in values


class TestCgroupSampler:
    @staticmethod
    def write_cgroup(root, memory: int, cpu_usec: int) -> None:
        (root / "memory.current").write_text(f"{memory}\n")
        (root / "cpu.stat").write_text(
            f"usage_usec {cpu_usec}\nuser_usec {cpu_usec}\nsystem_usec 0\n"
        )

    def test_it_reads_the_cgroups_own_accounting(self, tmp_path):
        self.write_cgroup(tmp_path, memory=4096, cpu_usec=1_000_000)

        values = CgroupSampler(tmp_path).sample()

        assert values["memory"] == 4096
        assert values["cpu"] == 0.0, "the first sample has nothing to difference"

    def test_cpu_is_the_rate_between_two_samples(self, tmp_path):
        sampler = CgroupSampler(tmp_path)
        self.write_cgroup(tmp_path, memory=4096, cpu_usec=0)
        sampler.sample()

        # Two cores' worth of CPU seconds over the elapsed wall time.
        time.sleep(0.05)
        elapsed = 0.05
        self.write_cgroup(tmp_path, memory=4096, cpu_usec=int(2 * elapsed * 1e6))
        values = sampler.sample()

        assert values["cpu"] > 0.5, "a busy cgroup reports cores in use"

    def test_a_counter_that_restarts_reports_no_time(self, tmp_path):
        """A recreated cgroup starts from zero; that is not negative CPU."""
        sampler = CgroupSampler(tmp_path)
        self.write_cgroup(tmp_path, memory=4096, cpu_usec=10_000_000)
        sampler.sample()

        self.write_cgroup(tmp_path, memory=4096, cpu_usec=0)
        values = sampler.sample()

        assert values["cpu"] == 0.0

    def test_it_falls_back_to_counting_processes(self, tmp_path):
        """No cgroup files: a login node, cgroup v1, or a plain container."""
        values = CgroupSampler(tmp_path).sample()

        assert values["memory"] > 0
        assert values["cpu"] == 0.0

    def test_a_malformed_cpu_stat_falls_back_too(self, tmp_path):
        (tmp_path / "memory.current").write_text("4096\n")
        (tmp_path / "cpu.stat").write_text("nr_periods 0\n")

        values = CgroupSampler(tmp_path).sample()

        assert values["memory"] > 0
        assert values["memory"] != 4096, "the partial cgroup reading is not used"

    def test_the_real_cgroup_is_readable_or_falls_back(self):
        """Whatever this machine is, a sample comes back."""
        values = CgroupSampler().sample()

        assert values["memory"] > 0
        assert values["cpu"] == 0.0


class TestMonitor:
    def test_one_reading_lands_in_the_subjects_series(
        self, ds_client, ds_service_address
    ):
        monitor = Monitor(
            client=ds_client,
            subject="node-1",
            prefixes=HOST_SERIES,
            sampler=lambda: {"free_memory": 17.0, "load_average": 0.5},
        )

        monitor.append_sample()

        (point,) = ds_client.time_series_get("host_free_memory:node-1")
        assert point.value == 17.0
        assert ds_client.time_series_get("host_load_average:node-1")[-1].value == 0.5

    def test_the_readings_of_one_sample_share_a_timestamp(self, ds_client):
        monitor = Monitor(
            client=ds_client,
            subject="node-1",
            prefixes=HOST_SERIES,
            sampler=lambda: {"free_memory": 1.0, "load_average": 2.0},
        )

        monitor.append_sample()

        memory = ds_client.time_series_get("host_free_memory:node-1")[-1]
        load = ds_client.time_series_get("host_load_average:node-1")[-1]
        assert memory.datetime == load.datetime

    def test_it_keeps_sampling_until_stopped(self, ds_client):
        monitor = Monitor(
            client=ds_client,
            subject="node-1",
            prefixes=HOST_SERIES,
            sampler=lambda: {"load_average": 1.0},
            interval=0.01,
        )
        monitor.start()

        assert wait_for(
            lambda: len(ds_client.time_series_get("host_load_average:node-1")) >= 3
        )
        monitor.stop()

        assert not monitor.is_alive()

    def test_a_sampler_that_raises_does_not_end_it(self, ds_client, caplog):
        """A node that is briefly unreachable must not stop the series."""

        def boom():
            raise OSError("no /proc today")

        monitor = Monitor(
            client=ds_client,
            subject="node-1",
            prefixes=HOST_SERIES,
            sampler=boom,
            interval=0.01,
        )
        monitor.start()

        assert wait_for(lambda: "failed to sample" in caplog.text.lower())
        assert monitor.is_alive()
        monitor.stop()

    def test_stopping_one_that_never_ran_is_fine(self, ds_client):
        monitor = Monitor(
            client=ds_client,
            subject="node-1",
            prefixes=HOST_SERIES,
            sampler=sample_host,
        )

        monitor.stop()  # must not raise

        assert not monitor.is_alive()


class TestStartHelpers:
    def test_the_host_monitor_writes_the_host_series(
        self, ds_client, ds_service_address
    ):
        monitor = start_host_monitor(ds_client, "node-1", interval=60.0)

        assert wait_for(
            lambda: bool(ds_client.time_series_get("host_free_memory:node-1"))
        )
        monitor.stop()

    def test_the_job_monitor_writes_the_job_series(self, ds_client):
        monitor = start_slurm_job_monitor(ds_client, 12345, interval=60.0)

        assert wait_for(
            lambda: bool(ds_client.time_series_get("slurm_job_memory:12345"))
        )
        assert ds_client.time_series_get("slurm_job_cpu:12345")[-1].value == 0.0
        monitor.stop()

    def test_a_numeric_job_id_keys_the_series_as_text(self, ds_client):
        monitor = start_slurm_job_monitor(ds_client, 7, interval=60.0)
        monitor.stop()

        assert ds_client.time_series_search_key("^slurm_job_memory:") == [
            "slurm_job_memory:7"
        ]


def test_the_two_series_maps_do_not_overlap():
    """A subject's prefix has to say which monitor wrote it."""
    assert set(HOST_SERIES.values()).isdisjoint(JOB_SERIES.values())


@pytest.mark.parametrize("prefixes", [HOST_SERIES, JOB_SERIES])
def test_every_series_prefix_ends_with_a_separator(prefixes):
    """The subject is appended raw, so the prefix carries the colon."""
    assert all(prefix.endswith(":") for prefix in prefixes.values())
