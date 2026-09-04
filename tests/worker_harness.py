"""Helpers for driving a real PilotWorkerProcess in-process."""

from __future__ import annotations

from pathlib import Path
from typing import cast

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

    def task_done(self, task_id: str, worker_id: str, output: bytes):
        result = self._inner.task_done(task_id, worker_id, output)
        self.completed += 1
        if self.completed >= self._limit:
            raise StopWorker
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _IdlingClient:
    """Wraps a real client, raising StopWorker after N `task_get` calls.

    The `task_get` calls themselves are the real ones,
    so an empty queue answers with the server's own `NoTaskAvailable`
    rather than one this double invented.
    """

    def __init__(self, inner: DsServiceClient, limit: int) -> None:
        self._inner = inner
        self._limit = limit
        self.polls = 0

    def task_get(self, worker_id: str, queue: str | list[str]):
        self.polls += 1
        if self.polls > self._limit:
            raise StopWorker
        return self._inner.task_get(worker_id, queue)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def make_worker(
    address: str,
    work_dir: Path,
    group: str = "cpu",
    name: str = "worker-0",
    actor_class_name: str = "",
    slurm_job_id: int = 42,
    hostname: str = "testhost",
    monitor_interval: float = 60.0,
) -> PilotWorkerProcess:
    """A real worker against a real server.

    The monitor interval is long by default:
    the first worker on a host samples once as it starts,
    which is what the tests look at,
    and nothing here wants a thread sampling again mid-test.
    """
    return PilotWorkerProcess(
        group=group,
        name=name,
        actor_class_name=actor_class_name,
        server_address=address,
        work_dir=Path(work_dir),
        slurm_job_id=slurm_job_id,
        hostname=hostname,
        pid=4242,
        monitor_interval=monitor_interval,
    )


def run_worker(worker: PilotWorkerProcess, expect_tasks: int) -> None:
    """Run the worker's real main loop until `expect_tasks` are completed.

    Tasks must already be queued:
    `task_get` on an empty queue raises `NoTaskAvailable` at once,
    which the worker answers by sleeping and asking again,
    so a worker started with nothing to do spins until the hang guard fires.
    """
    # `_StoppingClient` forwards everything it does not override,
    # so it satisfies the worker's use of the client without subclassing it.
    worker.client = cast(DsServiceClient, _StoppingClient(worker.client, expect_tasks))
    try:
        worker.main()
    except StopWorker:
        pass


def poll_worker(worker: PilotWorkerProcess, polls: int) -> int:
    """Run the worker's real main loop for `polls` fetches, and count them.

    For the empty-queue case, where no task is ever completed
    and `run_worker` would therefore never stop.
    Returns the number of fetches the worker actually made,
    so a loop that gave up early is distinguishable
    from one that kept polling.
    """
    client = _IdlingClient(worker.client, polls)
    worker.client = cast(DsServiceClient, client)
    try:
        worker.main()
    except StopWorker:
        pass
    return client.polls - 1
