"""Background sampling of a compute node and of a Slurm job.

A pilot worker knows things nothing else on the queue does:
which node it landed on, and which job it belongs to.
Many workers share both, so they elect a sampler between themselves
with a `ds-service` counter (see `PilotWorkerProcess._start_monitors`)
and the elected one runs the threads here.

Each thread appends to `ds-service` time series,
one series per measurement per subject,
so `swtop` (or anything else) can read a node's load
without logging in to it.
"""

from __future__ import annotations

import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable

import psutil

from ds_service_client import DsServiceClient

DEFAULT_MONITOR_INTERVAL_S: float = 5.0

# Where the cgroup v2 files of the current process live.
# Slurm puts a job in its own cgroup and accounts it there,
# so this is the whole job on this node, not just this process tree.
CGROUP_ROOT = Path("/sys/fs/cgroup")

# The filesystems worth watching on a compute node:
# both are node-local scratch that a job can fill,
# and a full one fails jobs in ways that look like anything but a full disk.
HOST_FILESYSTEMS = {"dev_shm": "/dev/shm", "tmp": "/tmp"}

# Time series keys, as `<prefix><subject>`.
# The subject is the hostname for a host and the job id for a job.
HOST_SERIES = {
    "free_memory": "host_free_memory:",  # bytes
    "load_average": "host_load_average:",  # 1 minute load average
    "dev_shm_used": "host_dev_shm_used:",  # percent of /dev/shm in use
    "tmp_used": "host_tmp_used:",  # percent of /tmp in use
}
JOB_SERIES = {
    "memory": "slurm_job_memory:",  # bytes, the cgroup's own total
    "cpu": "slurm_job_cpu:",  # cores in use, averaged over the interval
}


def sample_host() -> dict[str, float]:
    """One reading of this node: free memory, load, and scratch usage.

    A filesystem that is not mounted is left out rather than reported as
    zero, since a node without `/dev/shm` and a node with an empty one
    are not the same thing.
    """
    values = {
        "free_memory": float(psutil.virtual_memory().available),
        "load_average": float(psutil.getloadavg()[0]),
    }

    for name, path in HOST_FILESYSTEMS.items():
        try:
            values[f"{name}_used"] = float(psutil.disk_usage(path).percent)
        except OSError:
            continue

    return values


class CgroupSampler:
    """Total memory and CPU of everything in this process's cgroup.

    The cgroup is what Slurm accounts and enforces a job against,
    so it covers every process and thread of the job on this node,
    including ones this worker never started.

    CPU is reported as cores in use, averaged since the previous sample,
    which is why this is a class and not a function:
    the kernel reports CPU as a total that only rises,
    and the rate has to be differenced out of two readings.
    The first sample therefore reports 0 cores,
    having nothing to difference against.
    """

    def __init__(self, root: Path = CGROUP_ROOT) -> None:
        self.root = root
        self._last: tuple[float, float] | None = None

    def sample(self) -> dict[str, float]:
        now = time.monotonic()
        reading = self._read_cgroup()
        if reading is None:
            reading = self._read_processes()
        memory, cpu_seconds = reading

        cores = 0.0
        if self._last is not None:
            last_now, last_cpu = self._last
            elapsed = now - last_now
            if elapsed > 0:
                # Clamped, because a cgroup can be recreated under us
                # and a counter that restarts would otherwise read
                # as a large negative rate.
                cores = max(0.0, (cpu_seconds - last_cpu) / elapsed)
        self._last = (now, cpu_seconds)

        return {"memory": memory, "cpu": cores}

    def _read_cgroup(self) -> tuple[float, float] | None:
        """Memory in bytes and cumulative CPU seconds, from cgroup v2.

        Returns None where those files are not readable
        -- cgroup v1, a container that does not mount them,
        or a login node -- and the caller falls back to counting processes.
        """
        try:
            memory = float((self.root / "memory.current").read_text().strip())
            cpu_stat = (self.root / "cpu.stat").read_text()
        except (OSError, ValueError):
            return None

        for line in cpu_stat.splitlines():
            key, _, value = line.partition(" ")
            if key == "usage_usec":
                try:
                    return memory, float(value) / 1e6
                except ValueError:
                    return None
        return None

    def _read_processes(self) -> tuple[float, float]:
        """The same two numbers, summed over the processes in the cgroup.

        Less accurate than the cgroup's own accounting:
        summed RSS counts shared pages once per process.
        It is a fallback, not an equivalent.
        """
        memory = 0.0
        cpu_seconds = 0.0

        for proc in self._processes():
            try:
                with proc.oneshot():
                    memory += float(proc.memory_info().rss)
                    times = proc.cpu_times()
                    cpu_seconds += times.user + times.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # A process that exited between the listing and the read.
                continue

        return memory, cpu_seconds

    def _processes(self) -> list[psutil.Process]:
        """The cgroup's processes, or this one's own tree if it has no cgroup."""
        try:
            pids = [
                int(line)
                for line in (self.root / "cgroup.procs").read_text().split()
                if line
            ]
        except (OSError, ValueError):
            pids = []

        if pids:
            procs = []
            for pid in pids:
                try:
                    procs.append(psutil.Process(pid))
                except psutil.NoSuchProcess:
                    continue
            return procs

        this = psutil.Process()
        return [this, *this.children(recursive=True)]


class Monitor(threading.Thread):
    """Appends one sampler's readings to `ds-service`, on a timer.

    A daemon thread: a worker killed by Slurm at the end of its walltime
    must not be held open by its monitor.
    """

    def __init__(
        self,
        client: DsServiceClient,
        subject: str,
        prefixes: dict[str, str],
        sampler: Callable[[], dict[str, float]],
        interval: float = DEFAULT_MONITOR_INTERVAL_S,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"monitor:{subject}")
        self.client = client
        self.subject = subject
        self.prefixes = prefixes
        self.sampler = sampler
        self.interval = interval
        self.logger = logger or logging.getLogger("worker_process")
        self._stopping = threading.Event()

    def run(self) -> None:
        while True:
            try:
                self.append_sample()
            except Exception:
                # A sampling or network failure must not end the monitor:
                # the node it is watching is usually still there,
                # and a series that stops for good is worse than a gap.
                self.logger.exception("Monitor %s failed to sample", self.subject)

            if self._stopping.wait(self.interval):
                return

    def append_sample(self) -> None:
        """Take one reading and append it to this subject's series."""
        values = self.sampler()
        # One timestamp for the whole reading,
        # so the series of one subject line up point for point.
        stamp = datetime.now(timezone.utc).isoformat()

        for name, value in values.items():
            self.client.time_series_append(
                f"{self.prefixes[name]}{self.subject}", float(value), stamp
            )

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the thread to finish its wait and end, and wait for it to.

        Idempotent, and safe on a thread that was never started.
        """
        self._stopping.set()
        if self.is_alive():
            self.join(timeout)


def start_host_monitor(
    client: DsServiceClient,
    hostname: str,
    interval: float = DEFAULT_MONITOR_INTERVAL_S,
    logger: logging.Logger | None = None,
) -> Monitor:
    """Start sampling this node, and return the running thread."""
    monitor = Monitor(
        client=client,
        subject=hostname,
        prefixes=HOST_SERIES,
        sampler=sample_host,
        interval=interval,
        logger=logger,
    )
    monitor.start()
    return monitor


def start_slurm_job_monitor(
    client: DsServiceClient,
    slurm_job_id: str | int,
    interval: float = DEFAULT_MONITOR_INTERVAL_S,
    logger: logging.Logger | None = None,
) -> Monitor:
    """Start sampling this job's cgroup, and return the running thread."""
    monitor = Monitor(
        client=client,
        subject=str(slurm_job_id),
        prefixes=JOB_SERIES,
        sampler=CgroupSampler().sample,
        interval=interval,
        logger=logger,
    )
    monitor.start()
    return monitor
