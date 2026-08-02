"""Shared test fixtures.

Two deliberate choices here:

* **ds-service is real.** Each test gets its own server process,
    so task-queue semantics (states, batched status, output retrieval)
    are exercised against the actual implementation
    rather than a stand-in that can drift from it.
* **Slurm is mocked.** `sbatch` / `squeue` / `scancel` are intercepted
    at the `subprocess` boundary inside `slurm_utils`,
    so everything above that ---
    script rendering, job-id parsing, environment scrubbing ---
    is the real code path.
"""

from __future__ import annotations

import sys
import signal
import subprocess
from pathlib import Path
from typing import Generator
from dataclasses import dataclass
from contextlib import contextmanager

import pytest
from ds_service_client import DsServiceClient, DsServiceServer

from slurm_workflows import slurm_utils
from slurm_workflows.slurm_pilot_executor import SlurmPilotExecutor

# Test-support modules (e.g. support_actor) must be importable by name,
# both for `import` here
# and for the worker's importlib-based actor lookup.
sys.path.insert(0, str(Path(__file__).parent))


# --------------------------------------------------------------------------
# Hang guards
# --------------------------------------------------------------------------
#
# Both the executor's result polling and the worker's main loop
# run until a condition is met.
# A regression in either turns a failing test into a hanging one,
# which is far worse in CI,
# so tests are bounded by a wall-clock alarm.


@contextmanager
def _time_limit(seconds: float, message: str):
    def on_alarm(signum, frame):
        raise TimeoutError(message)

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture
def time_limit():
    """Bound a block that could spin forever if the code under test regresses."""

    return _time_limit


@pytest.fixture(autouse=True)
def _hang_guard():
    """Backstop so no single test can wedge the suite."""

    with _time_limit(60.0, "test exceeded its 60s time limit"):
        yield


# --------------------------------------------------------------------------
# ds-service (real)
# --------------------------------------------------------------------------


@pytest.fixture
def ds_service_address() -> Generator[str]:
    """Run a private ds-service for one test and yield its address.

    The server is in-memory,
    so a fresh process per test means no state leaks between tests.
    Startup is ~10ms.

    `DsServiceServer` owns finding the binary, picking a free port,
    waiting for the socket and shutting the process down,
    so none of that is reimplemented here.
    Bound to the loopback interface rather than a routable one:
    nothing outside this machine should reach a test's queue.
    """
    try:
        server = DsServiceServer(interface="lo")
    except FileNotFoundError:
        pytest.skip(
            "ds-service executable not found; "
            "put `ds-service` on PATH or point DS_SERVICE_BIN at it"
        )

    try:
        server.wait_until_ready(timeout=10)
        yield server.address
    finally:
        server.close()


@pytest.fixture
def ds_client(ds_service_address: str):
    """A directly-usable client against the test's ds-service."""
    client = DsServiceClient(ds_service_address)
    yield client
    client.close()


# --------------------------------------------------------------------------
# Slurm (mocked)
# --------------------------------------------------------------------------


@dataclass
class Submission:
    """One captured `sbatch` invocation."""

    job_id: int
    script_path: Path
    script_text: str
    env: dict[str, str]

    @property
    def job_name(self) -> str:
        for line in self.script_text.splitlines():
            if line.startswith("#SBATCH --job-name"):
                return line.split(maxsplit=2)[2].strip('"')
        raise AssertionError("no --job-name in submitted script")

    @property
    def sbatch_directives(self) -> list[str]:
        """`#SBATCH` lines, minus the name/output ones the library adds."""
        out = []
        for line in self.script_text.splitlines():
            if not line.startswith("#SBATCH "):
                continue
            body = line[len("#SBATCH ") :]
            if body.startswith(("--job-name", "--output")):
                continue
            out.append(body)
        return out


class FakeSlurm:
    """Stands in for the `subprocess` module inside `slurm_utils`.

    Implements just `run()` for the three Slurm commands
    and delegates everything else (exception types, etc.)
    to the real `subprocess`.
    """

    def __init__(self) -> None:
        self.submissions: list[Submission] = []
        self.running_job_ids: list[int] = []
        self.cancelled_job_ids: list[int] = []
        self.next_job_id = 1000
        self.fail: dict[str, tuple[int, str, str]] = {}
        self.sbatch_stdout_override: str | None = None

    # -- failure injection --------------------------------------------------

    def fail_command(
        self, exe: str, returncode: int = 1, stdout: str = "", stderr: str = "boom"
    ) -> None:
        """Make future calls to `exe` raise CalledProcessError."""
        self.fail[exe] = (returncode, stdout, stderr)

    # -- the subprocess surface --------------------------------------------

    def run(self, cmd, **kwargs):
        exe = Path(cmd[0]).name

        if exe in self.fail:
            returncode, stdout, stderr = self.fail[exe]
            raise subprocess.CalledProcessError(returncode, cmd, stdout, stderr)

        if exe == "sbatch":
            return self._sbatch(cmd, **kwargs)
        if exe == "squeue":
            return self._squeue(cmd)
        if exe == "scancel":
            return self._scancel(cmd)

        raise AssertionError(f"unexpected command in test: {cmd!r}")

    def __getattr__(self, name):
        return getattr(subprocess, name)

    # -- individual commands ------------------------------------------------

    def _sbatch(self, cmd, **kwargs):
        script_path = Path(cmd[1])
        job_id = self.next_job_id
        self.next_job_id += 1

        self.submissions.append(
            Submission(
                job_id=job_id,
                script_path=script_path,
                script_text=script_path.read_text(),
                env=dict(kwargs.get("env") or {}),
            )
        )
        self.running_job_ids.append(job_id)

        stdout = self.sbatch_stdout_override
        if stdout is None:
            stdout = f"Submitted batch job {job_id}\n"
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    def _squeue(self, cmd):
        stdout = "".join(f"{job_id}\n" for job_id in self.running_job_ids)
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    def _scancel(self, cmd):
        for arg in cmd[1:]:
            if arg.startswith("-"):
                continue
            job_id = int(arg)
            self.cancelled_job_ids.append(job_id)
            if job_id in self.running_job_ids:
                self.running_job_ids.remove(job_id)
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def fake_slurm(monkeypatch: pytest.MonkeyPatch) -> Generator[FakeSlurm]:
    """Intercept Slurm commands; no cluster required."""
    fake = FakeSlurm()
    monkeypatch.setattr(slurm_utils, "subprocess", fake)
    # get_clean_environ is @cache'd; clear it so each test sees its own env.
    slurm_utils.get_clean_environ.cache_clear()
    yield fake
    slurm_utils.get_clean_environ.cache_clear()


# --------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------


@pytest.fixture
def executor(ds_service_address: str, fake_slurm: FakeSlurm, tmp_path: Path):
    """An executor wired to the real queue server and the fake Slurm."""
    ex = SlurmPilotExecutor(
        server_address=ds_service_address, work_dir=tmp_path / "work"
    )
    yield ex
    ex.close()


@pytest.fixture
def pilot_jobs(executor):
    """Declare that pilot jobs exist for the named groups.

    `as_completed` refuses to wait on a queue
    this executor never started a worker for,
    so any test that waits has to say a pilot job was submitted
    --- even when the thing that actually drains the queue is `drain()`
    or a worker running in-process.
    The Slurm job stands in for the allocation;
    those stand in for the process inside it.

    Use as `pilot_jobs("cpu")` in a test, or once in an autouse fixture.
    """

    def declare(*names: str) -> None:
        for name in names:
            executor.define_worker(name, [])
            executor.scale_workers(name, 1)

    return declare


@pytest.fixture
def setup_script() -> str:
    """A setup script body, as define_worker expects."""
    return "module load gcc/14.2.0\nexport TEST_SETUP=1\n"
