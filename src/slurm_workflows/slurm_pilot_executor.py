"""Pilot workers for slurm."""

from __future__ import annotations

import re
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
    LOG_FORMAT,
    LOG_LEVEL,
)

from .templates import render_template

NoOutput = object()

# What an executor may be called.
# The name ends up in task ids, a logger name, a directory name
# and every worker's job name,
# so it is restricted to characters that are safe in all four.
EXECUTOR_NAME_REGEX = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
MIN_EXECUTOR_NAME_LEN = 3

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
    _task_name: str | None = None

    @property
    def task_name(self) -> str | None:
        """The name given by `SlurmPilotExecutor.set_task_name`, or None."""
        return self._task_name


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
    @typechecked
    def __init__(
        self,
        name: str,
        server_address: str,
        work_dir: Path | str | None = None,
    ):
        if len(name) < MIN_EXECUTOR_NAME_LEN:
            raise ValueError(
                f"Executor name {name!r} is shorter than "
                f"{MIN_EXECUTOR_NAME_LEN} characters"
            )
        if EXECUTOR_NAME_REGEX.fullmatch(name) is None:
            raise ValueError(
                f"Executor name {name!r} must start with a letter "
                f"and hold only letters, digits, '_' and '-'"
            )

        self.name = name
        self.next_task_index = 0

        self.server_address = server_address
        self.client = DsServiceClient(server_address)

        if work_dir is None:
            now = datetime.now().isoformat()
            work_dir = (
                platformdirs.user_cache_path(appname="slurm-workflows") / name / now
            )
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        print(f"work directory: '{self.work_dir}'")

        # A logger of this executor's own, keyed on its name.
        # A name shared between live executors would collect one handler per
        # executor, and every line would then be written to every work dir
        # that had ever been opened in this process,
        # which is one more reason two of them must not share a name.
        self.logger = logging.getLogger(f"slurm_workflows.executor.{self.name}")
        self.logger.setLevel(LOG_LEVEL)
        # These records belong in the work dir, not in whatever handler the
        # importing program happens to have put on the root logger.
        self.logger.propagate = False
        handler = logging.FileHandler(self.work_dir / "executor.log", delay=True)
        handler.setLevel(LOG_LEVEL)
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # Detached and closed by `close()`;
        # until then the handler holds the log file open.
        self._log_handler: logging.FileHandler | None = handler

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
        actor_class_args: list[Any] | None = None,
        actor_class_kwargs: dict[str, Any] | None = None,
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
            if actor_class_args is not None or actor_class_kwargs is not None:
                raise ValueError(
                    "actor_class_args and actor_class_kwargs "
                    "need an actor_class_name to construct"
                )
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

        # The constructor arguments go to the workers through the key value
        # store rather than the command line:
        # they are arbitrary Python objects, so they are cloudpickled,
        # and the worker reads them back keyed on its own group name.
        #
        # The arguments are not part of the group's identity,
        # so redefining a group with different ones is not a conflict:
        # the store simply takes the new values,
        # which only the workers started after this call will read.
        if actor_class_args is not None:
            self.client.map_set(
                f"actor_class_args:{name}",
                cloudpickle.dumps(actor_class_args, protocol=pickle.HIGHEST_PROTOCOL),
            )
        if actor_class_kwargs is not None:
            self.client.map_set(
                f"actor_class_kwargs:{name}",
                cloudpickle.dumps(actor_class_kwargs, protocol=pickle.HIGHEST_PROTOCOL),
            )

    def _add_worker(self, group: WorkerGroup) -> None:
        worker_index = group.next_worker_index
        group.next_worker_index += 1
        # The executor's name leads, so a worker's job name says which
        # executor submitted it, and two executors sharing a cluster
        # do not produce identically named jobs and script files.
        worker_name = f"{self.name}.worker.{group.name}.{worker_index}"

        worker_script = render_template(
            "slurm_pilot:worker_script",
            group=group.name,
            name=worker_name,
            server_address=self.server_address,
            worker_exe=group.worker_exe,
            work_dir=self.work_dir,
            python_paths_json=json.dumps(group.python_paths),
            setup_script=group.setup_script,
            actor_class_name=group.actor_class_name,
        )
        worker_script_path = self.work_dir / f"{worker_name}.sh"
        worker_script_path.write_text(worker_script)
        worker_script_path.chmod(0o755)

        worker_sbatch_script = render_template(
            "slurm_pilot:worker_sbatch_script",
            name=worker_name,
            work_dir=self.work_dir,
            is_batch_worker=group.is_batch_worker,
            worker_script_path=worker_script_path,
        )

        self.logger.info("Starting worker %s", worker_name)
        try:
            slurm_job = submit_sbatch_job(
                name=worker_name,
                sbatch_args=group.sbatch_args,
                script=worker_sbatch_script,
                work_dir=self.work_dir,
            )
            group.workers[worker_name] = slurm_job
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
        # ds-service dispatches the highest priority first,
        # so the timestamp is negated to keep submission order:
        # a task submitted earlier outranks one submitted later.
        #
        # Wall clock rather than a monotonic counter, whose zero is the
        # start of *this* process: two executors sharing a queue would
        # otherwise be ordered by which of them started more recently
        # rather than by when each task was submitted.
        priority = -time.time()
        function_bytes = cloudpickle.dumps(fn, protocol=pickle.HIGHEST_PROTOCOL)
        input_bytes = cloudpickle.dumps(
            (args, kwargs), protocol=pickle.HIGHEST_PROTOCOL
        )

        task_id = f"{self.name}.task.{self.next_task_index}"
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
    def set_task_name(self, task: Task, name: str) -> None:
        """Give `task` a name, on the queue server as well as locally.

        The name is for whoever is reading the queue:
        nothing in this library dispatches on it.
        """
        # Recorded on the server first,
        # so a failed write leaves the task without a local name
        # rather than with one nothing else can see.
        self.client.map_set(f"task_name:{task.task_id}", name.encode("utf-8"))
        task._task_name = name

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
            states = cast(list[TaskState], states)

            next_pending: list[Task] = []
            completed = 0
            for task, state in zip(pending, states):
                if state == TaskState.Complete:
                    output = self.client.task_get_output(task.task_id)
                    task.output = cloudpickle.loads(output)
                    completed += 1
                    yield task
                elif state == TaskState.Canceled:
                    # Nothing here cancels tasks,
                    # so this is somebody cancelling out of band.
                    # A canceled task is never dispatched again,
                    # so it has to be reported rather than waited on.
                    raise RuntimeError(
                        f"Task {task.task_id} was canceled on the task queue "
                        f"server, so it will never produce an output"
                    )
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

    def _close_log_handler(self) -> None:
        """Detach this executor's log handler and close its file.

        The logger holds the handler and the handler holds the file,
        so neither is released while the logger is in the logging module's
        registry --- which is for the life of the process.
        Idempotent, because `close()` is.
        """
        handler = self._log_handler
        if handler is None:
            return

        # Cleared first, so a failure below cannot leave a detached handler
        # to be closed a second time.
        self._log_handler = None
        self.logger.removeHandler(handler)
        handler.close()

    def close(self):
        self._cleanup_all_workers()
        for group in self.groups.values():
            group.workers.clear()

        self.client.close()
        self._close_log_handler()

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
