"""Helpers for driving a real PilotWorkerProcess in-process."""

from __future__ import annotations

from pathlib import Path

from ds_service_client import DsServiceClient
from slurm_workflows.slurm_pilot_worker import PilotWorkerProcess


class StopWorker(BaseException):
    """Breaks the worker's otherwise-infinite main loop.

    Deliberately a BaseException, not an Exception:
    `main()` catches every Exception so a worker survives bad tasks,
    so only a BaseException can end the loop from inside a client call.
    """


class _StoppingClient:
    """Wraps a real client, raising StopWorker after N tasks are reported done."""

    def __init__(self, inner: DsServiceClient, limit: int) -> None:
        self._inner = inner
        self._limit = limit
        self.completed = 0

    def task_done(self, task_id: str, output: bytes):
        result = self._inner.task_done(task_id, output)
        self.completed += 1
        if self.completed >= self._limit:
            raise StopWorker
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


def make_worker(
    address: str,
    work_dir: Path,
    group: str = "cpu",
    name: str = "worker-0",
    actor_class_name: str = "",
) -> PilotWorkerProcess:
    return PilotWorkerProcess(
        group=group,
        name=name,
        actor_class_name=actor_class_name,
        server_address=address,
        work_dir=Path(work_dir),
        slurm_job_id=42,
        hostname="testhost",
        pid=4242,
    )


def run_worker(worker: PilotWorkerProcess, expect_tasks: int) -> None:
    """Run the worker's real main loop until `expect_tasks` are completed.

    Tasks must already be queued:
    an empty queue makes `task_get` block until its deadline,
    which would just slow the test down.
    """
    worker.client = _StoppingClient(worker.client, expect_tasks)
    try:
        worker.main()
    except StopWorker:
        pass
