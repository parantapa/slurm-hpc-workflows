"""Pilot workers for slurm."""

from __future__ import annotations

import time
import json
import pickle
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, Iterable, Any, cast

import platformdirs
import cloudpickle
from typeguard import typechecked
from tqdm import tqdm
from ds_service_client import DsServiceClient, TaskState

from .slurm_utils import (
    get_running_jobids,
    cancel_jobs,
    submit_sbatch_job,
    SlurmJob,
)

from .utils import (
    RemoteExecutionError,
    gen_random_string,
    LOG_FORMAT,
    LOG_LEVEL,
)

from .templates import render_template

NoOutput = object()

POLL_INTERVAL_S: float = 0.1

# How often `_as_completed` checks
# that pending tasks still have a pilot job that could run them.
# Kept well above POLL_INTERVAL_S because each check costs an `squeue` call.
LIVE_QUEUE_CHECK_INTERVAL_S: float = 60.0


@dataclass
class Task:
    task_id: str
    queue: list[str]
    priority: float
    function: Callable | str
    input: tuple
    output: Any


@dataclass
class WorkerGroup:
    name: str
    sbatch_args: list[str]
    is_batch_worker: bool
    worker_exe: str
    actor_class_name: str
    setup_script: str
    python_paths: list[str]
    workers: dict[str, SlurmJob] = field(default_factory=dict, compare=False)
    next_worker_index: int = field(default=0, compare=False)


class SlurmPilotExecutor:
    def __init__(
        self,
        server_address: str,
        work_dir: Path | str | None = None,
    ):
        self.executor_id = gen_random_string()
        self.next_task_index = 0

        self.server_address = server_address
        self.client = DsServiceClient(server_address)

        if work_dir is None:
            now = datetime.now().isoformat()
            work_dir = platformdirs.user_cache_path(appname=f"slurm-workflows") / now
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("excutor")
        self.logger.setLevel(LOG_LEVEL)
        handler = logging.FileHandler(self.work_dir / "executor.log", delay=True)
        handler.setLevel(LOG_LEVEL)
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.groups: dict[str, WorkerGroup] = {}

    @typechecked
    def define_worker(
        self,
        name: str,
        sbatch_args: list[str],
        setup_script: str = "",
        worker_exe: str = "slurm-pilot-worker",
        is_batch_worker: bool = False,
        actor_class_name: str | None = None,
        python_paths: list[str | Path] | None = None,
        add_cwd_to_python_path: bool = True,
    ) -> None:
        python_str_paths: list[str] = []
        if python_paths is not None:
            for path in python_paths:
                python_str_paths.append(str(path))
        if add_cwd_to_python_path:
            python_str_paths.append(str(Path.cwd()))

        if actor_class_name is None:
            actor_class_name = ""

        group = WorkerGroup(
            name=name,
            sbatch_args=sbatch_args,
            worker_exe=worker_exe,
            is_batch_worker=is_batch_worker,
            actor_class_name=actor_class_name,
            setup_script=setup_script,
            python_paths=python_str_paths,
        )

        if group.name in self.groups:
            assert self.groups[group.name] == group
        else:
            self.groups[group.name] = group

    def _add_worker(self, group: WorkerGroup) -> None:
        worker_index = group.next_worker_index
        group.next_worker_index += 1
        name = f"slurm_pilot_worker.{group.name}.{worker_index}"

        worker_script = render_template(
            "slurm_pilot:worker_script",
            group=group.name,
            name=name,
            server_address=self.server_address,
            worker_exe=group.worker_exe,
            work_dir=self.work_dir,
            python_paths_json=json.dumps(group.python_paths),
            setup_script=group.setup_script,
            actor_class_name=group.actor_class_name,
        )
        worker_script_path = self.work_dir / f"{name}.sh"
        worker_script_path.write_text(worker_script)
        worker_script_path.chmod(0o755)

        worker_sbatch_script = render_template(
            "slurm_pilot:worker_sbatch_script",
            name=name,
            work_dir=self.work_dir,
            is_batch_worker=group.is_batch_worker,
            worker_script_path=worker_script_path,
        )

        self.logger.info("Starting worker %s", name)
        try:
            slurm_job = submit_sbatch_job(
                name=name,
                sbatch_args=group.sbatch_args,
                script=worker_sbatch_script,
                work_dir=self.work_dir,
            )
            group.workers[name] = slurm_job
        except subprocess.CalledProcessError as cp:
            print(f"Failed to submit slurm job: returncode={cp.returncode}")
            if cp.stdout.strip():
                print(cp.stdout)
            if cp.stderr.strip():
                print(cp.stderr)
            raise cp

    @typechecked
    def scale_workers(self, name: str, count: int) -> None:
        assert name in self.groups, "Unknown worker type"

        group = self.groups[name]
        if len(group.workers) < count:
            to_hire = count - len(group.workers)
            for _ in range(to_hire):
                self._add_worker(group)

        if len(group.workers) > count:
            to_retire = len(group.workers) - count

            try:
                running_jobids = get_running_jobids()
            except subprocess.CalledProcessError as cp:
                print(
                    f"Failed to get running slurm job ids: returncode={cp.returncode}"
                )
                if cp.stdout.strip():
                    print(cp.stdout)
                if cp.stderr.strip():
                    print(cp.stderr)
                raise RuntimeError("Failed to get running slurm job ids")
            except Exception:
                raise RuntimeError("Failed to get running slurm job ids")

            to_cancel_jobids = []
            for _ in range(to_retire):
                _, worker = group.workers.popitem()
                self.logger.info("Canceling worker: %s", worker.name)
                if worker.job_id in running_jobids:
                    to_cancel_jobids.append(worker.job_id)

            if not to_cancel_jobids:
                return

            try:
                cancel_jobs(to_cancel_jobids)
            except subprocess.CalledProcessError as cp:
                print(f"Failed to cancel slurm jobs: returncode={cp.returncode}")
                if cp.stdout.strip():
                    print(cp.stdout)
                if cp.stderr.strip():
                    print(cp.stderr)
                raise RuntimeError("Failed to cancel slurm jobs")
            except Exception:
                raise RuntimeError("Failed to cancel slurm jobs")

    def _submit(
        self,
        queue: list[str],
        fn: Callable | str,
        *args,
        **kwargs,
    ) -> Task:
        priority = time.perf_counter()
        function_bytes = cloudpickle.dumps(fn, protocol=pickle.HIGHEST_PROTOCOL)
        input_bytes = cloudpickle.dumps(
            (args, kwargs), protocol=pickle.HIGHEST_PROTOCOL
        )

        task_id = f"{self.executor_id}:{self.next_task_index}"
        self.next_task_index += 1

        task = Task(
            task_id=task_id,
            queue=queue,
            priority=priority,
            function=fn,
            input=(args, kwargs),
            output=NoOutput,
        )

        self.client.task_add(
            task_id=task_id,
            queue=queue,
            priority=priority,
            function=function_bytes,
            input=input_bytes,
        )
        return task

    @typechecked
    def submit(
        self, queue: str | list[str], fn: Callable | str, *args, **kwargs
    ) -> Task:
        if isinstance(queue, str):
            queue = [queue]

        return self._submit(queue, fn, *args, **kwargs)

    def _as_completed(self, tasks: list[Task]) -> Iterable[Task]:
        pending: list[Task] = []
        finished: list[Task] = []
        for task in tasks:
            if task.output is NoOutput:
                pending.append(task)
            else:
                finished.append(task)

        # Before anything is yielded, so a caller that never scaled a group
        # is told at once rather than after the first result.
        if pending:
            self._raise_if_no_worker_started(pending)

        yield from finished

        # The first check is one interval away rather than immediate.
        # Submitting before any worker exists is a supported pattern
        # --- tasks queue up and are picked up as pilot jobs start ---
        # so a queue with no job on the cluster *yet* is normal here,
        # and checking straight away would reject it.
        next_liveness_check = time.monotonic() + LIVE_QUEUE_CHECK_INTERVAL_S

        while pending:
            # Status for every pending task comes back in a single request,
            # in the same order as the ids we sent.
            states = self.client.task_get_status([t.task_id for t in pending])
            states = cast(list[Task], states)

            next_pending: list[Task] = []
            completed = 0
            for task, state in zip(pending, states):
                if state == TaskState.Complete:
                    output = self.client.task_get_output(task.task_id)
                    task.output = cloudpickle.loads(output)
                    completed += 1
                    yield task
                elif state == TaskState.Undefined:
                    raise RuntimeError(
                        f"Task {task.task_id} is unknown to the task queue server"
                    )
                else:
                    next_pending.append(task)

            pending = next_pending

            if pending and time.monotonic() >= next_liveness_check:
                self._raise_if_no_live_queue(pending)
                next_liveness_check = time.monotonic() + LIVE_QUEUE_CHECK_INTERVAL_S

            if pending and not completed:
                time.sleep(POLL_INTERVAL_S)

    @typechecked
    def as_completed(
        self, tasks: Iterable[Task], desc: str | None = None, unit: str = "task"
    ) -> Iterable[Task]:
        tasks = list(tasks)
        iterable = self._as_completed(tasks)
        iterable = tqdm(iterable, total=len(tasks), desc=desc, unit=unit)
        return iterable

    @typechecked
    def wait(
        self, tasks: Iterable[Task], desc: str | None = None, unit: str = "task"
    ) -> None:
        for _ in self.as_completed(tasks, desc, unit):
            pass

    def num_groups(self):
        return len(self.groups)

    def num_workers(self, detail: bool = False):
        if detail:
            return {g.name: len(g.workers) for g in self.groups.values()}
        else:
            return sum(len(g.workers) for g in self.groups.values())

    def _live_queues(self, queues: Iterable[str] | None = None) -> set[str]:
        """Names of the queues that still have a pilot job on the cluster.

        Queue name and worker group name are the same thing,
        so a queue is live when at least one job submitted for that group
        is still known to Slurm.

        "Still known to Slurm" means `squeue` lists it,
        which covers a job that is pending as well as one that is running.
        A pending job counts as live on purpose:
        it has not started yet,
        but tasks on its queue will be served once it does,
        and treating it as dead would abandon work that is merely waiting
        for an allocation.

        `queues` restricts the answer to the names given
        --- unknown names are simply absent from the result ---
        and the default considers every defined group.

        Whatever `get_running_jobids` raises propagates.
        A failed `squeue` means liveness is *unknown*,
        and an empty set would claim the stronger "nothing is live",
        which a caller could act on by giving up on live work.
        """
        job_ids = get_running_jobids()

        groups = self.groups.values()
        if queues is not None:
            wanted = set(queues)
            groups = [g for g in groups if g.name in wanted]

        return {
            group.name
            for group in groups
            if any(worker.job_id in job_ids for worker in group.workers.values())
        }

    def _raise_if_no_worker_started(self, pending: list[Task]) -> None:
        """Fail at once on tasks no worker has ever been started for.

        This reads local bookkeeping rather than asking the cluster:
        a queue is covered when `scale_workers` has submitted at least one job
        for the group of that name.
        Whether those jobs are *still* alive is `_raise_if_no_live_queue`'s
        question, asked periodically from then on.

        Checking here turns the two commonest mistakes
        --- never scaling a group, and mistyping a queue name ---
        into an immediate error
        rather than a wait that lasts until the first liveness check.

        Note this only knows about workers *this* executor submitted.
        """
        started = {name for name, group in self.groups.items() if group.workers}

        starved = [task for task in pending if not set(task.queue) & started]
        if not starved:
            return

        queues = sorted({q for task in starved for q in task.queue})
        raise RuntimeError(
            f"{len(starved)} of {len(pending)} pending tasks are on queues with "
            f"no worker started: {queues}. "
            f"Call scale_workers() for a worker group of that name "
            f"before waiting on them."
        )

    def _raise_if_no_live_queue(self, pending: list[Task]) -> None:
        """Fail fast on pending tasks whose queues have no pilot job left.

        A task is stranded when none of its queues
        still has a job on the cluster:
        nothing is left to pull it,
        so waiting on it would block until the caller gives up.

        A `squeue` that cannot be reached leaves liveness *unknown*,
        which is not the same as dead,
        so that case is logged and retried at the next interval
        rather than aborting a wait that may be perfectly healthy.
        """
        try:
            live = self._live_queues({q for task in pending for q in task.queue})
        except (subprocess.SubprocessError, OSError):
            self.logger.warning(
                "Could not check whether queues are still live; "
                "will retry at the next interval",
                exc_info=True,
            )
            return

        stranded = [task for task in pending if not set(task.queue) & live]
        if not stranded:
            return

        queues = sorted({q for task in stranded for q in task.queue})
        raise RuntimeError(
            f"{len(stranded)} of {len(pending)} pending tasks are on queues with "
            f"no live pilot job, so they can never run: {queues}. "
            f"Scale up a worker group named after one of those queues, "
            f"or cancel the wait."
        )

    def _cleanup_all_workers(self):
        try:
            job_ids = get_running_jobids()
        except subprocess.CalledProcessError as cp:
            print(f"Failed to get running slurm job ids: returncode={cp.returncode}")
            if cp.stdout.strip():
                print(cp.stdout)
            if cp.stderr.strip():
                print(cp.stderr)
            job_ids = None
        except Exception:
            self.logger.exception("Failed to get running slurm job ids")
            job_ids = None

        if job_ids is None:
            return

        to_cancel_jobids = []
        for group in self.groups.values():
            for worker in group.workers.values():
                if worker.job_id in job_ids:
                    to_cancel_jobids.append(worker.job_id)

        if not to_cancel_jobids:
            return

        try:
            cancel_jobs(to_cancel_jobids)
        except subprocess.CalledProcessError as cp:
            print(f"Failed to cancel slurm jobs: returncode={cp.returncode}")
            if cp.stdout.strip():
                print(cp.stdout)
            if cp.stderr.strip():
                print(cp.stderr)
        except Exception:
            self.logger.exception("Failed to cancel slurm jobs")

    def close(self):
        self._cleanup_all_workers()
        for group in self.groups.values():
            group.workers.clear()

        self.client.close()

    def stop(self):
        self._cleanup_all_workers()
        for group in self.groups.values():
            group.workers.clear()

    def __enter__(self) -> "SlurmPilotExecutor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Returning None (not False-y-by-accident, but explicitly nothing)
        # so an exception raised in the body still propagates:
        # pilot jobs are cancelled on the way out either way,
        # but a failure in the body must not be swallowed.
        self.close()


def check_for_error(tasks: list[Task], verbose: bool = True) -> list[Task]:
    ret = []
    for task in tasks:
        if isinstance(task.output, RemoteExecutionError):
            ret.append(task)

            if verbose:
                print(f"task_id={task.task_id}")
                print(f"  error={task.output.error}")
                print(f"  error_id={task.output.error_id}")

    return ret
