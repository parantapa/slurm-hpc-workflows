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
from typing import Callable, Iterable, Any

import platformdirs
import cloudpickle
from typeguard import typechecked
from tqdm import tqdm
from ds_service_client import Client, TaskState

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
    workers: dict[str, SlurmJob] = field(default_factory=dict)
    next_worker_index: int = 0


class SlurmPilotExecutor:
    def __init__(
        self,
        server_address: str,
        work_dir: Path | str | None = None,
    ):
        self.executor_id = gen_random_string()
        self.next_task_index = 0

        self.server_address = server_address
        self.client = Client(server_address)

        if work_dir is None:
            now = datetime.now().isoformat()
            work_dir = platformdirs.user_cache_path(appname=f"slurm-pilot") / now
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("pilot_coordinator")
        self.logger.setLevel(LOG_LEVEL)
        handler = logging.FileHandler(self.work_dir / "coordinator.log", delay=True)
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
        setup_script: str,
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
            work_dir=str(self.work_dir),
            python_paths_json=json.dumps(group.python_paths),
            setup_script=group.setup_script,
            actor_class_name=group.actor_class_name,
        )
        worker_script_path = self.work_dir / f"{name}.sh"
        worker_script_path.write_text(worker_script)
        worker_script_path.chmod(0o755)

        worker_sbatch_script = render_template(
            "slurm_pilot:worker_sbatch_script",
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
            print(f"Failed to cancel slurm jobs: returncode={cp.returncode}")
            if cp.stdout.strip():
                print(cp.stdout)
            if cp.stderr.strip():
                print(cp.stdout)
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
        cur_tasks = tasks
        next_tasks = []

        while cur_tasks:
            for task in cur_tasks:
                if task.output is NoOutput:
                    status = self.client.task_status(task.task_id)
                    if status.state == TaskState.Complete:
                        task.output = cloudpickle.loads(status.output)
                        yield task
                        continue
                else:
                    yield task
                    continue

                next_tasks.append(task)

            cur_tasks = next_tasks
            next_tasks = []

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
