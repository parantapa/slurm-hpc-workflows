"""Tests for the pilot worker.

The worker runs for real against a real ds-service; only Slurm is mocked.
`run_worker` stops the otherwise-infinite main loop
once the expected number of tasks have been reported done.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest
import click
from ds_service_client import TaskState

import support_actor
from slurm_workflows import slurm_pilot_worker as worker_mod
from slurm_workflows.slurm_pilot_worker import slurm_pilot_worker
from slurm_workflows.utils import RemoteExecutionError
from worker_harness import make_worker, poll_worker, run_worker


@pytest.fixture(autouse=True)
def reset_actors():
    support_actor.reset()
    yield
    support_actor.reset()


def square(x):
    return x * x


def boom():
    raise ValueError("task blew up")


class TestTaskExecution:
    @pytest.fixture(autouse=True)
    def _pilot_jobs(self, pilot_jobs):
        pilot_jobs("cpu", "gpu")

    def test_runs_a_task_and_posts_the_result(
        self, executor, ds_service_address, tmp_path
    ):
        task = executor.submit("cpu", square, 7)

        worker = make_worker(ds_service_address, tmp_path)
        run_worker(worker, expect_tasks=1)
        worker.close()

        executor.wait([task])
        assert task.output == 49

    def test_idles_quietly_on_an_empty_queue(
        self, ds_service_address, tmp_path, caplog
    ):
        """An empty queue is the normal idle case, not an error.

        `task_get` answers `NoTaskAvailable` at once when nothing is ready,
        and the worker sleeps and asks again.
        Left to the catch-all handler instead,
        the loop would still poll, but log a traceback every time round,
        so the absence of that log is what tells the two apart.
        """
        worker = make_worker(ds_service_address, tmp_path)

        with caplog.at_level(logging.ERROR, logger="worker_process"):
            polls = poll_worker(worker, polls=3)

        worker.close()
        assert polls == 3, "the worker stopped polling an empty queue"
        assert "Unexpected exception" not in caplog.text

    def test_runs_many_tasks_in_sequence(self, executor, ds_service_address, tmp_path):
        tasks = [executor.submit("cpu", square, i) for i in range(5)]

        worker = make_worker(ds_service_address, tmp_path)
        run_worker(worker, expect_tasks=5)
        worker.close()

        executor.wait(tasks)
        assert sorted(t.output for t in tasks) == [0, 1, 4, 9, 16]

    def test_runs_closures(self, executor, ds_service_address, tmp_path):
        offset = 100
        task = executor.submit("cpu", lambda x: x + offset, 5)

        worker = make_worker(ds_service_address, tmp_path)
        run_worker(worker, expect_tasks=1)
        worker.close()

        executor.wait([task])
        assert task.output == 105

    def test_passes_args_and_kwargs(self, executor, ds_service_address, tmp_path):
        def combine(a, b, sep="-"):
            return f"{a}{sep}{b}"

        task = executor.submit("cpu", combine, "x", "y", sep="+")

        worker = make_worker(ds_service_address, tmp_path)
        run_worker(worker, expect_tasks=1)
        worker.close()

        executor.wait([task])
        assert task.output == "x+y"

    def test_only_serves_its_own_group(
        self, executor, ds_service_address, ds_client, tmp_path
    ):
        cpu_task = executor.submit("cpu", square, 2)
        gpu_task = executor.submit("gpu", square, 3)

        worker = make_worker(ds_service_address, tmp_path, group="cpu")
        run_worker(worker, expect_tasks=1)
        worker.close()

        executor.wait([cpu_task])
        assert cpu_task.output == 4
        # The other group's task is untouched.
        assert ds_client.task_get_status(gpu_task.task_id) == TaskState.Ready


class TestRemoteErrors:
    @pytest.fixture(autouse=True)
    def _pilot_jobs(self, pilot_jobs):
        pilot_jobs("cpu")

    def test_exception_is_captured_not_propagated(
        self, executor, ds_service_address, tmp_path
    ):
        task = executor.submit("cpu", boom)

        worker = make_worker(ds_service_address, tmp_path)
        run_worker(worker, expect_tasks=1)  # must not raise
        worker.close()

        executor.wait([task])
        assert isinstance(task.output, RemoteExecutionError)
        assert task.output.error == "task blew up"
        assert task.output.error_id.startswith("ERROR_")

    def test_worker_survives_a_failing_task(
        self, executor, ds_service_address, tmp_path
    ):
        """One bad task must not take the worker down."""
        bad = executor.submit("cpu", boom)
        good = executor.submit("cpu", square, 4)

        worker = make_worker(ds_service_address, tmp_path)
        run_worker(worker, expect_tasks=2)
        worker.close()

        executor.wait([bad, good])
        assert isinstance(bad.output, RemoteExecutionError)
        assert good.output == 16

    def test_error_id_is_logged_with_the_traceback(
        self, executor, ds_service_address, tmp_path, caplog
    ):
        task = executor.submit("cpu", boom)

        with caplog.at_level(logging.ERROR, logger="worker_process"):
            worker = make_worker(ds_service_address, tmp_path)
            run_worker(worker, expect_tasks=1)
            worker.close()

        executor.wait([task])
        error_id = task.output.error_id
        assert error_id in caplog.text
        assert "ValueError: task blew up" in caplog.text

    def test_unserializable_result_is_reported_as_an_error(
        self, executor, ds_service_address, tmp_path
    ):
        """Serialization happens inside the try block, so it is caught too."""
        task = executor.submit("cpu", lambda: (_ for _ in range(3)))  # generator

        worker = make_worker(ds_service_address, tmp_path)
        run_worker(worker, expect_tasks=1)
        worker.close()

        executor.wait([task])
        assert isinstance(task.output, RemoteExecutionError)


class TestActors:
    @pytest.fixture(autouse=True)
    def _pilot_jobs(self, pilot_jobs):
        pilot_jobs("cpu")

    def test_actor_is_instantiated_once_at_startup(self, ds_service_address, tmp_path):
        worker = make_worker(
            ds_service_address, tmp_path, actor_class_name="support_actor.CounterActor"
        )

        assert len(support_actor.INSTANCES) == 1
        worker.close()

    def test_no_actor_by_default(self, ds_service_address, tmp_path):
        worker = make_worker(ds_service_address, tmp_path)

        assert worker.actor_instance is None
        worker.close()

    def test_method_names_dispatch_to_the_actor(
        self, executor, ds_service_address, tmp_path
    ):
        task = executor.submit("cpu", "echo", "hello")

        worker = make_worker(
            ds_service_address, tmp_path, actor_class_name="support_actor.CounterActor"
        )
        run_worker(worker, expect_tasks=1)
        worker.close()

        executor.wait([task])
        assert task.output == "hello"

    def test_state_persists_across_tasks(self, executor, ds_service_address, tmp_path):
        tasks = [executor.submit("cpu", "bump", 1) for _ in range(4)]

        worker = make_worker(
            ds_service_address, tmp_path, actor_class_name="support_actor.CounterActor"
        )
        run_worker(worker, expect_tasks=4)
        worker.close()

        executor.wait(tasks)
        # Same instance served all four, so the counter accumulated.
        assert sorted(t.output for t in tasks) == [1, 2, 3, 4]
        assert len(support_actor.INSTANCES) == 1

    def test_actor_exception_is_captured(self, executor, ds_service_address, tmp_path):
        task = executor.submit("cpu", "boom")

        worker = make_worker(
            ds_service_address, tmp_path, actor_class_name="support_actor.CounterActor"
        )
        run_worker(worker, expect_tasks=1)
        worker.close()

        executor.wait([task])
        assert isinstance(task.output, RemoteExecutionError)
        assert task.output.error == "actor failure"

    def test_unknown_method_is_captured(self, executor, ds_service_address, tmp_path):
        task = executor.submit("cpu", "no_such_method")

        worker = make_worker(
            ds_service_address, tmp_path, actor_class_name="support_actor.CounterActor"
        )
        run_worker(worker, expect_tasks=1)
        worker.close()

        executor.wait([task])
        assert isinstance(task.output, RemoteExecutionError)

    def test_close_calls_actor_close(self, ds_service_address, tmp_path):
        worker = make_worker(
            ds_service_address, tmp_path, actor_class_name="support_actor.CounterActor"
        )
        actor = support_actor.INSTANCES[0]

        worker.close()

        assert actor.closed is True
        assert worker.actor_instance is None

    def test_close_tolerates_actor_without_close(self, ds_service_address, tmp_path):
        worker = make_worker(
            ds_service_address, tmp_path, actor_class_name="support_actor.NoCloseActor"
        )

        worker.close()  # must not raise

    def test_unimportable_actor_fails_at_startup(self, ds_service_address, tmp_path):
        with pytest.raises(ModuleNotFoundError):
            make_worker(
                ds_service_address, tmp_path, actor_class_name="no_such_module.Actor"
            )

    def test_missing_actor_class_fails_at_startup(self, ds_service_address, tmp_path):
        with pytest.raises(AttributeError):
            make_worker(
                ds_service_address, tmp_path, actor_class_name="support_actor.Missing"
            )


class TestWorkerIdentity:
    def test_worker_id_encodes_placement(self, ds_service_address, tmp_path):
        worker = make_worker(ds_service_address, tmp_path, group="cpu", name="w-1")

        assert worker.worker_id == "cpu:w-1:42:testhost:4242"
        worker.close()


class TestCli:
    """The console entry point: env vars, sys.path, and leaving output alone."""

    @pytest.fixture(autouse=True)
    def _restore_process_state(self):
        """Put `os.environ` and `sys.path` back after each case.

        The command writes both directly and undoes neither
        --- it is a process entry point, and the process is the worker ---
        so without this a run leaks `DS_SERVER_ADDRESS` into every later
        test, which is exactly the value `DsServiceClient()` falls back to
        when it is given no address.
        """
        env = dict(os.environ)
        path = list(sys.path)
        yield
        os.environ.clear()
        os.environ.update(env)
        sys.path[:] = path

    @pytest.fixture
    def captured(self, monkeypatch):
        """Replace the worker process so main() returns instead of looping."""
        seen = {}

        class FakeWorker:
            def __init__(self, **kwargs):
                seen["kwargs"] = kwargs

            def main(self):
                seen["env"] = {
                    key: os.environ.get(key)
                    for key in (
                        "PILOT_WORKER_NAME",
                        "PILOT_WORKER_GROUP",
                        "DS_SERVER_ADDRESS",
                    )
                }
                seen["sys_path_head"] = list(sys.path[:2])
                seen["streams"] = (sys.stdout, sys.stderr)

            def close(self):
                seen["closed"] = True

        monkeypatch.setattr(worker_mod, "PilotWorkerProcess", FakeWorker)
        return seen

    def invoke(self, tmp_path: Path, **overrides) -> int:
        """Run the CLI and return its exit code.

        Invoked directly rather than through click's CliRunner,
        which swaps the process streams for buffers of its own
        -- these tests assert on what the command does to those streams,
        so it must not.
        """
        args = {
            "--group": "cpu",
            "--name": "worker-0",
            "--actor-class-name": "",
            "--server-address": "127.0.0.1:5051",
            "--work-dir": str(tmp_path),
            "--python-paths-json": '["/extra/path"]',
        }
        args.update(overrides)
        argv = [item for pair in args.items() for item in pair]

        try:
            slurm_pilot_worker.main(
                args=argv, prog_name="slurm-pilot-worker", standalone_mode=False
            )
            return 0
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1
        except click.UsageError as exc:
            return exc.exit_code

    def test_runs_and_closes_the_worker(self, captured, tmp_path):
        exit_code = self.invoke(tmp_path)

        assert exit_code == 0
        assert captured["closed"] is True

    def test_sets_worker_environment_variables(self, captured, tmp_path):
        self.invoke(tmp_path)

        assert captured["env"]["PILOT_WORKER_NAME"] == "worker-0"
        assert captured["env"]["PILOT_WORKER_GROUP"] == "cpu"
        assert captured["env"]["DS_SERVER_ADDRESS"] == "127.0.0.1:5051"

    def test_prepends_python_paths(self, captured, tmp_path):
        self.invoke(tmp_path)

        assert "/extra/path" in captured["sys_path_head"]

    def test_leaves_the_process_streams_alone(self, captured, tmp_path):
        """Slurm writes the worker's output file itself, via `--output`.

        `logging.basicConfig` leaves the streams on the handles the process
        inherited, and redirecting them here
        would leave the file Slurm writes empty.
        """
        before = (sys.stdout, sys.stderr)

        self.invoke(tmp_path)

        assert captured["streams"] == before

    def test_rejects_missing_work_dir(self, captured, tmp_path):
        exit_code = self.invoke(tmp_path, **{"--work-dir": str(tmp_path / "nope")})

        assert exit_code != 0
