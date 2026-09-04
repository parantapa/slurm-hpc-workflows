"""`swtop`: a live view of a `ds-service` task queue.

Polls the server and redraws a summary of its tasks and of the pilot
workers that have registered with it.

What can be shown is decided by what the server can be asked.
Task counts by state come from a single RPC,
so they are always complete.
Individual tasks and workers have to be discovered
through the key value store instead,
which means a worker appears once it has published its identity
(`PilotWorkerProcess` does that at startup)
and a task appears once it has been named
with `SlurmPilotExecutor.set_task_name`.
An unnamed task is counted, but has no row.

Host and job readings come from the monitor threads in `monitors.py`,
which one worker per node and one per job runs.
A subject whose series have stopped is shown as stale
rather than dropped, since a monitor that died is worth noticing.
"""

from __future__ import annotations

import sys
import json
import time
from typing import cast
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

import click
from ds_service_client import DsServiceClient, TaskState, TaskStateError

from .monitors import HOST_SERIES, JOB_SERIES
from .slurm_pilot_worker import WORKER_INFO_PREFIX

DEFAULT_INTERVAL_S: float = 2.0

# The fields a worker publishes about itself, in display order.
WORKER_INFO_FIELDS = ["group", "name", "slurm_job_id", "hostname", "pid"]

TASK_NAME_PREFIX = "task_name:"

# How far back a monitored value is still worth showing.
# A monitor samples every 5 seconds by default,
# so a subject with nothing this recent has lost the worker watching it.
STALE_AFTER_S = 60.0

# Shown when a worker published its id but not the field being read,
# which is what a worker caught mid-startup looks like.
UNKNOWN = "?"

# The order tasks are listed in: what is happening now, first.
STATE_ORDER = ["Running", "Ready", "Complete", "Canceled", "Undefined"]


@dataclass
class WorkerInfo:
    """One registered worker, as it describes itself in the store."""

    worker_id: str
    group: str
    name: str
    slurm_job_id: str
    hostname: str
    pid: str


@dataclass
class TaskInfo:
    """One named task, with the state the server reports for it."""

    task_id: str
    name: str
    state: str
    worker: str = ""


@dataclass
class SubjectInfo:
    """The latest reading of one monitored host or job.

    `values` is empty when the subject's series exist
    but hold nothing recent, which is what a dead monitor looks like.
    """

    subject: str
    values: dict[str, float] = field(default_factory=dict)

    @property
    def stale(self) -> bool:
        return not self.values


@dataclass
class Snapshot:
    """One poll's worth of server state."""

    address: str
    when: datetime
    counts: dict[str, int] = field(default_factory=dict)
    workers: list[WorkerInfo] = field(default_factory=list)
    tasks: list[TaskInfo] = field(default_factory=list)
    hosts: list[SubjectInfo] = field(default_factory=list)
    jobs: list[SubjectInfo] = field(default_factory=list)
    error: str | None = None


class Collector:
    """Turns the server's RPCs into a `Snapshot`.

    Holds the identity of every worker it has seen
    and the name of every task it has seen,
    both of which are written once and never change,
    so a steady state costs one status batch and two key searches
    rather than a read per worker per poll.
    """

    def __init__(self, client: DsServiceClient, address: str) -> None:
        self.client = client
        self.address = address
        self._workers: dict[str, WorkerInfo] = {}
        self._task_names: dict[str, str] = {}

    def snapshot(self) -> Snapshot:
        counts = self.client.task_get_count_by_state()
        workers = self._collect_workers()
        tasks = self._collect_tasks(workers)
        hosts = self._collect_subjects(HOST_SERIES)
        jobs = self._collect_subjects(JOB_SERIES)

        return Snapshot(
            address=self.address,
            when=datetime.now(),
            counts={
                "ready": counts.ready,
                "running": counts.running,
                "complete": counts.complete,
                "canceled": counts.canceled,
            },
            workers=workers,
            tasks=tasks,
            hosts=hosts,
            jobs=jobs,
        )

    def _text(self, key: str) -> str:
        try:
            return self.client.map_get(key).decode("utf-8", errors="replace")
        except KeyError:
            return UNKNOWN

    def _collect_workers(self) -> list[WorkerInfo]:
        worker_ids = [
            key[len(WORKER_INFO_PREFIX) :]
            for key in self.client.map_search_key(f"^{WORKER_INFO_PREFIX}")
        ]

        listed = []
        for worker_id in worker_ids:
            info = self._workers.get(worker_id) or self._worker_info(worker_id)
            if info is None:
                # Not cached: whatever is under the key now
                # is not what a worker writes, and might be later.
                listed.append(_unknown_worker(worker_id))
                continue
            self._workers[worker_id] = info
            listed.append(info)

        return sorted(listed, key=lambda w: (w.group, w.name))

    def _worker_info(self, worker_id: str) -> WorkerInfo | None:
        """One worker's published description, or None if it is not readable.

        A worker writes it as a single key,
        so what comes back is either all of it or none of it,
        which is what makes it safe to cache:
        there is no half-written state to be remembered as final.
        """
        try:
            published = json.loads(self._text(f"{WORKER_INFO_PREFIX}{worker_id}"))
            fields = {name: str(published[name]) for name in WORKER_INFO_FIELDS}
        except (ValueError, TypeError, KeyError):
            return None

        return WorkerInfo(worker_id=worker_id, **fields)

    def _collect_tasks(self, workers: list[WorkerInfo]) -> list[TaskInfo]:
        task_ids = [
            key[len(TASK_NAME_PREFIX) :]
            for key in self.client.map_search_key(f"^{TASK_NAME_PREFIX}")
        ]
        if not task_ids:
            return []

        for task_id in task_ids:
            if task_id not in self._task_names:
                self._task_names[task_id] = self._text(f"{TASK_NAME_PREFIX}{task_id}")

        # One batched call for every named task,
        # rather than a status RPC apiece.
        states = self.client.task_get_status(task_ids)
        # A list of ids answers with a list of states, one per id.
        states = cast(list[TaskState], states)
        worker_names = {w.worker_id: w.name for w in workers}

        tasks = []
        for task_id, state in zip(task_ids, states):
            state_name = TaskState.Name(state)
            task = TaskInfo(
                task_id=task_id,
                name=self._task_names[task_id],
                state=state_name,
            )
            if state_name == "Running":
                task.worker = self._holder(task_id, worker_names)
            tasks.append(task)

        return sorted(tasks, key=lambda t: (_state_rank(t.state), t.name, t.task_id))

    def _collect_subjects(self, prefixes: dict[str, str]) -> list[SubjectInfo]:
        """The latest reading of every subject one monitor writes about.

        The subjects are discovered from one of the series
        rather than from a list of nodes or jobs,
        because a monitor is the only thing that knows either exists.
        """
        first = next(iter(prefixes.values()))
        subjects = sorted(
            key[len(first) :] for key in self.client.time_series_search_key(f"^{first}")
        )
        if not subjects:
            return []

        # Only the tail of each series is asked for.
        # Reading it whole would grow without bound
        # over a run long enough to be worth watching.
        since = (
            datetime.now(timezone.utc) - timedelta(seconds=STALE_AFTER_S)
        ).isoformat()

        readings = []
        for subject in subjects:
            values = {}
            for name, prefix in prefixes.items():
                points = self.client.time_series_get(
                    f"{prefix}{subject}", start_time=since
                )
                if points:
                    values[name] = points[-1].value
            readings.append(SubjectInfo(subject=subject, values=values))
        return readings

    def _holder(self, task_id: str, worker_names: dict[str, str]) -> str:
        """The name of the worker running `task_id`, as far as it can be told.

        The task can finish between the status call and this one,
        and the worker holding it need not be one that registered
        (anything with the client library can take a task),
        so neither answer is guaranteed.
        """
        try:
            worker_id = self.client.task_get_worker_id(task_id)
        except (KeyError, TaskStateError):
            return ""
        return worker_names.get(worker_id, worker_id)


def _unknown_worker(worker_id: str) -> WorkerInfo:
    """A row for a worker whose description could not be read."""
    return WorkerInfo(
        worker_id=worker_id, **{name: UNKNOWN for name in WORKER_INFO_FIELDS}
    )


def _state_rank(state: str) -> int:
    try:
        return STATE_ORDER.index(state)
    except ValueError:
        return len(STATE_ORDER)


def _bytes(value: float) -> str:
    """Bytes as the largest unit that keeps the number readable."""
    for unit in ["B", "K", "M", "G", "T"]:
        if abs(value) < 1024 or unit == "T":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def _subject(info: SubjectInfo) -> str:
    """The subject's name, said to be stale when its series have stopped."""
    return info.subject if not info.stale else f"{info.subject} (stale)"


def _cell(values: dict[str, float], name: str, fmt) -> str:
    """One measurement, or a dash where the series had nothing recent."""
    if name not in values:
        return "-"
    return fmt(values[name])


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Left-aligned fixed-width columns, sized to their contents."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
    return lines


def render(snapshot: Snapshot) -> str:
    """The whole screen, as text."""
    when = snapshot.when.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"swtop  {snapshot.address}  {when}", ""]

    if snapshot.error is not None:
        lines.append(f"cannot read the server: {snapshot.error}")
        return "\n".join(lines) + "\n"

    counts = snapshot.counts
    total = sum(counts.values())
    lines.append(
        "tasks  "
        + "  ".join(f"{name} {counts[name]}" for name in counts)
        + f"  total {total}"
    )
    lines.append("")

    lines.append(f"workers ({len(snapshot.workers)})")
    if snapshot.workers:
        lines.extend(
            _table(
                ["NAME", "GROUP", "HOST", "JOB", "PID"],
                [
                    [w.name, w.group, w.hostname, w.slurm_job_id, w.pid]
                    for w in snapshot.workers
                ],
            )
        )
    else:
        lines.append("no workers have registered with this server")
    lines.append("")

    lines.append(f"hosts ({len(snapshot.hosts)})")
    if snapshot.hosts:
        lines.extend(
            _table(
                ["HOST", "FREE MEM", "LOAD", "/dev/shm", "/tmp"],
                [
                    [
                        _subject(host),
                        _cell(host.values, "free_memory", _bytes),
                        _cell(host.values, "load_average", lambda v: f"{v:.2f}"),
                        _cell(host.values, "dev_shm_used", lambda v: f"{v:.1f}%"),
                        _cell(host.values, "tmp_used", lambda v: f"{v:.1f}%"),
                    ]
                    for host in snapshot.hosts
                ],
            )
        )
    else:
        lines.append("no host is being monitored")
    lines.append("")

    lines.append(f"slurm jobs ({len(snapshot.jobs)})")
    if snapshot.jobs:
        lines.extend(
            _table(
                ["JOB", "MEMORY", "CPU"],
                [
                    [
                        _subject(job),
                        _cell(job.values, "memory", _bytes),
                        _cell(job.values, "cpu", lambda v: f"{v:.1f} cores"),
                    ]
                    for job in snapshot.jobs
                ],
            )
        )
    else:
        lines.append("no slurm job is being monitored")
    lines.append("")

    lines.append(f"named tasks ({len(snapshot.tasks)})")
    if snapshot.tasks:
        lines.extend(
            _table(
                ["NAME", "TASK ID", "STATE", "WORKER"],
                [[t.name, t.task_id, t.state, t.worker] for t in snapshot.tasks],
            )
        )
    else:
        # Every task is counted above; only named ones can be listed,
        # because there is no RPC that enumerates tasks.
        lines.append("no tasks have been named with set_task_name()")

    return "\n".join(lines) + "\n"


def draw(text: str) -> None:
    """Put `text` on the screen, replacing what was there.

    Only on a terminal:
    piped into a file or a pager, the escape codes would be noise,
    so the polls are simply appended and stay readable.
    """
    if sys.stdout.isatty():
        # Home, then clear: clearing first leaves the old frame visible
        # for a moment on a slow link.
        sys.stdout.write("\x1b[H\x1b[2J")
    sys.stdout.write(text)
    if not sys.stdout.isatty():
        sys.stdout.write("\n")
    sys.stdout.flush()


@click.command()
@click.argument("server_address", type=str)
@click.option(
    "--interval",
    "-i",
    type=float,
    default=DEFAULT_INTERVAL_S,
    show_default=True,
    help="Seconds between polls.",
)
def swtop(server_address: str, interval: float) -> None:
    """Watch the tasks and workers on the ds-service server at SERVER_ADDRESS.

    SERVER_ADDRESS is `host:port`, the same address an executor is given.
    Runs until interrupted.
    """
    if interval <= 0:
        raise click.BadParameter("must be greater than 0", param_hint="'--interval'")

    client = DsServiceClient(server_address)
    collector = Collector(client, server_address)

    try:
        while True:
            try:
                snapshot = collector.snapshot()
            except Exception as e:
                # A server that is down, or not up yet, is worth waiting out:
                # this is a monitor, and quitting would lose the history
                # on the screen along with the view.
                snapshot = Snapshot(
                    address=server_address,
                    when=datetime.now(),
                    error=f"{type(e).__name__}: {e}",
                )

            draw(render(snapshot))
            time.sleep(interval)
    except KeyboardInterrupt:
        # Ctrl-C is how this is meant to end.
        pass
    finally:
        client.close()
