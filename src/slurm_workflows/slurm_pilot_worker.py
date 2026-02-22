"""Pilot workers for Slurm pilot."""

import os
import sys
import json
import time
import pickle
import socket
import logging
from pathlib import Path

import click
import cloudpickle

from ds_service_client import Client

from .utils import gen_error_id, RemoteExecutionError, LOG_FORMAT, LOG_LEVEL

NEXT_TASK_RETRY_TIME_S: float = 1.0


class PilotWorkerProcess:
    def __init__(
        self,
        group: str,
        name: str,
        server_address: str,
        work_dir: Path,
        slurm_job_id: int,
        hostname: str,
        pid: int,
    ):
        self.group = group
        self.server_address = server_address
        self.work_dir = work_dir

        self.worker_id = "%s:%s:%s:%s:%s" % (group, name, slurm_job_id, hostname, pid)
        self.logger = logging.getLogger("worker_process")
        self.client = Client(self.server_address)

    def close(self):
        self.client.close()

    def main(self):
        self.logger.info("Starting worker: %s" % self.worker_id)

        while True:
            try:
                task = self.client.task_get(self.worker_id, self.group)
                try:
                    self.logger.info(
                        "task_id=%s: Deserializing function and inputs ...",
                        task.task_id,
                    )
                    function = cloudpickle.loads(task.function)
                    args, kwargs = cloudpickle.loads(task.input)

                    self.logger.info("task_id=%s: Executing ...", task.task_id)
                    retval = function(*args, **kwargs)

                    self.logger.info("task_id=%s: Serializng output ...", task.task_id)
                    output = cloudpickle.dumps(retval, protocol=pickle.HIGHEST_PROTOCOL)

                    self.client.task_done(task.task_id, output)
                except Exception as e:
                    eid = gen_error_id()
                    self.logger.exception(
                        "Error executing %s: %s: %s", task.task_id, eid, e
                    )

                    retval = RemoteExecutionError(error=str(e), error_id=eid)
                    output = cloudpickle.dumps(retval, protocol=pickle.HIGHEST_PROTOCOL)
                    self.client.task_done(task.task_id, output)
            except TimeoutError:
                time.sleep(NEXT_TASK_RETRY_TIME_S)
            except Exception:
                self.logger.exception("Unexpected exception")


@click.command()
@click.option("--group", type=str, required=True, help="Worker group")
@click.option("--name", type=str, required=True, help="Worker job name")
@click.option("--server-address", type=str, required=True, help="Pilot server address")
@click.option(
    "--work-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Work directory",
)
@click.option(
    "--python-paths-json",
    type=str,
    required=True,
    help="JSON encoded Python paths",
)
def slurm_pilot_worker(
    group: str, name: str, server_address: str, work_dir: Path, python_paths_json: str
):
    """Start a slurm pilot worker."""
    slurm_job_id = int(os.environ.get("SLURM_JOB_ID", -1))
    hostname = socket.gethostname()
    pid = os.getpid()
    log_file = work_dir / f"{name}-{slurm_job_id}-{hostname}-{pid}.log"

    print(f"Redirecting standard output and standard error to {log_file}")
    sys.stdout.flush()
    sys.stderr.flush()

    with open(log_file, "wt") as fout:
        sys.stdout = fout
        sys.stderr = fout

        logging.basicConfig(stream=fout, format=LOG_FORMAT, level=LOG_LEVEL)

        os.environ["PILOT_WORKER_NAME"] = name
        os.environ["PILOT_WORKER_GROUP"] = group
        os.environ["DS_SERVER_ADDRESS"] = server_address

        python_paths: list[str] = json.loads(python_paths_json)
        for path in python_paths:
            sys.path.insert(0, path)

        worker = PilotWorkerProcess(
            group=group,
            name=name,
            server_address=server_address,
            work_dir=work_dir,
            slurm_job_id=slurm_job_id,
            hostname=hostname,
            pid=pid,
        )

        try:
            worker.main()
        finally:
            worker.close()
