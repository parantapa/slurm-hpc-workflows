"""Tests for SlurmPilotExecutor.

Slurm is mocked; the task queue is a real ds-service.
Where a test needs a task to finish,
it plays the worker's part with `drain()`
rather than launching one,
so executor behaviour is isolated from worker behaviour.
"""

from __future__ import annotations

import itertools
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import pytest
import cloudpickle
from typeguard import TypeCheckError

from slurm_workflows import slurm_pilot_executor as spe
from slurm_workflows.slurm_pilot_executor import (
    NoOutput,
    RaiseOnError,
    SlurmPilotExecutor,
    Task,
)
from slurm_workflows.utils import RemoteExecutionError


def drain(ds_client, queue: str, count: int) -> list[str]:
    """Act as a worker: pull `count` tasks and post their real results."""

    task_ids = []
    for _ in range(count):
        task = ds_client.task_get("test-worker", queue)
        fn = cloudpickle.loads(task.function)
        args, kwargs = cloudpickle.loads(task.input)
        ds_client.task_done(
            task.task_id, "test-worker", cloudpickle.dumps(fn(*args, **kwargs))
        )
        task_ids.append(task.task_id)
    return task_ids


def fail_one(ds_client, queue: str, error_id: str = "ERROR_test") -> str:
    """Complete one task the way a worker reports a remote exception."""

    task = ds_client.task_get("test-worker", queue)
    output = RemoteExecutionError(error="boom", error_id=error_id)
    ds_client.task_done(task.task_id, "test-worker", cloudpickle.dumps(output))
    return task.task_id


def ghost_task() -> Task:
    """A task the server has never heard of, so it polls back as `Undefined`."""

    return Task(
        task_id="does-not-exist",
        queue=["cpu"],
        priority=0.0,
        function=square,
        input=((), {}),
        output=NoOutput,
    )


class CountingClient:
    """Counts RPCs so tests can assert on batching."""

    def __init__(self, inner):
        self._inner = inner
        self.status_calls = 0
        self.status_batch_sizes: list[int] = []
        self.output_calls = 0

    def task_get_status(self, task_id):
        self.status_calls += 1
        self.status_batch_sizes.append(1 if isinstance(task_id, str) else len(task_id))
        return self._inner.task_get_status(task_id)

    def task_get_output(self, task_id):
        self.output_calls += 1
        return self._inner.task_get_output(task_id)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def square(x):
    return x * x


# --------------------------------------------------------------------------
# name
# --------------------------------------------------------------------------


class TestExecutorName:
    """The name identifies the executor, so it has to be usable everywhere."""

    def test_it_prefixes_task_ids(self, executor):
        task = executor.submit("cpu", square, 5)

        assert task.task_id == "testex.task.0"

    def test_it_prefixes_worker_job_names(self, executor, setup_script):
        executor.define_worker(name="cpu", sbatch_args=[], setup_script=setup_script)
        executor.scale_workers("cpu", 1)

        assert list(executor.groups["cpu"].workers) == ["testex.worker.cpu.0"]

    @pytest.mark.parametrize("name", ["ab", "a", ""])
    def test_a_short_name_is_rejected(self, ds_service_address, name):
        with pytest.raises(ValueError):
            SlurmPilotExecutor(name, ds_service_address)

    @pytest.mark.parametrize(
        "name",
        [
            "1abc",  # must start with a letter
            "_abc",
            "-abc",
            "abc def",  # a job name and a directory name
            "abc.def",  # the separator in task ids and worker names
            "abc/def",
            "abc:def",  # the separator in key value store keys
        ],
    )
    def test_an_unusable_name_is_rejected(self, ds_service_address, name):
        with pytest.raises(ValueError):
            SlurmPilotExecutor(name, ds_service_address)

    @pytest.mark.parametrize("name", ["abc", "Run-1", "a_b", "abc123"])
    def test_a_usable_name_is_accepted(self, ds_service_address, tmp_path, name):
        ex = SlurmPilotExecutor(name, ds_service_address, work_dir=tmp_path / name)

        assert ex.name == name
        ex.close()

    def test_rejects_a_non_string_name(self, ds_service_address):
        with pytest.raises(TypeCheckError):
            SlurmPilotExecutor(42, ds_service_address)  # type: ignore[arg-type]

    def test_the_default_work_dir_is_under_the_name(
        self, ds_service_address, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            spe.platformdirs, "user_cache_path", lambda **kwargs: tmp_path
        )

        ex = SlurmPilotExecutor("cached", ds_service_address)

        # <cache>/<name>/<timestamp>, so one executor's runs sit together
        # and two executors do not interleave theirs.
        assert ex.work_dir.parent == tmp_path / "cached"
        assert datetime.fromisoformat(ex.work_dir.name)
        ex.close()


# --------------------------------------------------------------------------
# define_worker
# --------------------------------------------------------------------------


class TestDefineWorker:
    def test_registers_group_without_launching(
        self, executor, fake_slurm, setup_script
    ):
        executor.define_worker(
            name="cpu", sbatch_args=["-A alloc"], setup_script=setup_script
        )

        assert executor.num_groups() == 1
        assert executor.num_workers() == 0
        assert fake_slurm.submissions == [], "define_worker must not submit jobs"

    def test_is_idempotent_for_identical_definitions(self, executor, setup_script):
        for _ in range(3):
            executor.define_worker(
                name="cpu", sbatch_args=["-A alloc"], setup_script=setup_script
            )

        assert executor.num_groups() == 1

    def test_conflicting_redefinition_is_rejected(self, executor, setup_script):
        executor.define_worker(
            name="cpu", sbatch_args=["-A alloc"], setup_script=setup_script
        )

        with pytest.raises(AssertionError):
            executor.define_worker(
                name="cpu", sbatch_args=["-A other"], setup_script=setup_script
            )

    def test_actor_class_args_are_stored_in_the_key_value_store(
        self, executor, ds_client
    ):
        executor.define_worker(
            name="cpu",
            sbatch_args=[],
            actor_class_name="support_actor.ConfiguredActor",
            actor_class_args=[1, "two"],
            actor_class_kwargs={"flag": True},
        )

        args = cloudpickle.loads(ds_client.map_get("actor_class_args:cpu"))
        kwargs = cloudpickle.loads(ds_client.map_get("actor_class_kwargs:cpu"))
        assert args == [1, "two"]
        assert kwargs == {"flag": True}

    def test_the_keys_are_named_for_the_group(self, executor, ds_client):
        executor.define_worker(
            name="gpu",
            sbatch_args=[],
            actor_class_name="support_actor.ConfiguredActor",
            actor_class_args=[1],
        )

        assert cloudpickle.loads(ds_client.map_get("actor_class_args:gpu")) == [1]
        with pytest.raises(KeyError):
            ds_client.map_get("actor_class_args:cpu")

    def test_an_actor_without_arguments_stores_nothing(self, executor, ds_client):
        executor.define_worker(
            name="cpu",
            sbatch_args=[],
            actor_class_name="support_actor.CounterActor",
        )

        with pytest.raises(KeyError):
            ds_client.map_get("actor_class_args:cpu")
        with pytest.raises(KeyError):
            ds_client.map_get("actor_class_kwargs:cpu")

    @pytest.mark.parametrize(
        "extra",
        [
            {"actor_class_args": [1]},
            {"actor_class_kwargs": {"flag": True}},
        ],
    )
    def test_actor_arguments_need_an_actor_class(self, executor, extra):
        with pytest.raises(ValueError):
            executor.define_worker(name="cpu", sbatch_args=[], **extra)

    def test_redefining_a_group_replaces_the_stored_arguments(
        self, executor, ds_client
    ):
        common = dict(
            sbatch_args=[],
            actor_class_name="support_actor.ConfiguredActor",
        )
        executor.define_worker(name="cpu", actor_class_args=[1], **common)

        # The arguments are not part of the group's identity,
        # so this is not the conflict that differing sbatch_args would be.
        executor.define_worker(name="cpu", actor_class_args=[2], **common)

        assert executor.num_groups() == 1
        assert cloudpickle.loads(ds_client.map_get("actor_class_args:cpu")) == [2]

    def test_cwd_is_added_to_python_path_by_default(self, executor, setup_script):
        executor.define_worker(name="cpu", sbatch_args=[], setup_script=setup_script)

        assert executor.groups["cpu"].python_paths == [str(Path.cwd())]

    def test_python_paths_are_stringified_and_ordered(self, executor, setup_script):
        executor.define_worker(
            name="cpu",
            sbatch_args=[],
            setup_script=setup_script,
            python_paths=[Path("/a"), "/b"],
        )

        assert executor.groups["cpu"].python_paths == ["/a", "/b", str(Path.cwd())]

    def test_cwd_can_be_omitted(self, executor, setup_script):
        executor.define_worker(
            name="cpu",
            sbatch_args=[],
            setup_script=setup_script,
            python_paths=["/a"],
            add_cwd_to_python_path=False,
        )

        assert executor.groups["cpu"].python_paths == ["/a"]

    def test_setup_script_body_reaches_the_worker_script(self, executor, setup_script):
        executor.define_worker(name="cpu", sbatch_args=[], setup_script=setup_script)
        executor.scale_workers("cpu", 1)

        script = (executor.work_dir / "testex.worker.cpu.0.sh").read_text()
        assert "module load gcc/14.2.0" in script
        assert "export TEST_SETUP=1" in script

    def test_accepts_an_empty_body(self, executor):
        executor.define_worker(name="cpu", sbatch_args=[], setup_script="")

        assert executor.groups["cpu"].setup_script == ""

    def test_setup_script_is_optional(self, executor):
        executor.define_worker(name="cpu", sbatch_args=[])

        assert executor.groups["cpu"].setup_script == ""

    def test_omitted_setup_script_still_yields_a_runnable_script(self, executor):
        executor.define_worker(name="cpu", sbatch_args=[])
        executor.scale_workers("cpu", 1)

        script = (executor.work_dir / "testex.worker.cpu.0.sh").read_text()
        assert script.startswith("#!/bin/bash")
        assert ". '/etc/profile'" in script
        assert "slurm-pilot-worker \\" in script

    def test_rejects_wrong_argument_types(self, executor, setup_script):
        with pytest.raises(TypeCheckError):
            executor.define_worker(
                name="cpu",
                sbatch_args="-A alloc",  # must be a list
                setup_script=setup_script,
            )


# --------------------------------------------------------------------------
# scale_workers
# --------------------------------------------------------------------------


class TestScaleWorkers:
    @pytest.fixture
    def defined(self, executor, setup_script):
        executor.define_worker(
            name="cpu",
            sbatch_args=["-A alloc", "-t 01:00:00"],
            setup_script=setup_script,
        )
        return executor

    def test_unknown_group_is_rejected(self, executor):
        with pytest.raises(AssertionError, match="Unknown worker type"):
            executor.scale_workers("nope", 1)

    def test_scaling_up_submits_one_job_per_worker(self, defined, fake_slurm):
        defined.scale_workers("cpu", 3)

        assert len(fake_slurm.submissions) == 3
        assert defined.num_workers() == 3
        assert defined.num_workers(detail=True) == {"cpu": 3}

    def test_submitted_script_carries_sbatch_args(self, defined, fake_slurm):
        defined.scale_workers("cpu", 1)

        directives = fake_slurm.submissions[0].sbatch_directives
        assert directives == ["-A alloc", "-t 01:00:00"]

    def test_worker_names_are_unique_and_indexed(self, defined, fake_slurm):
        defined.scale_workers("cpu", 2)

        names = [s.job_name for s in fake_slurm.submissions]
        assert names == ["testex.worker.cpu.0", "testex.worker.cpu.1"]

    def test_worker_script_is_written_and_executable(self, defined, fake_slurm):
        defined.scale_workers("cpu", 1)

        script = defined.work_dir / "testex.worker.cpu.0.sh"
        assert script.exists()
        assert script.stat().st_mode & 0o111
        assert f"--server-address '{defined.server_address}'" in script.read_text()

    def test_scaling_up_again_only_adds_the_difference(self, defined, fake_slurm):
        defined.scale_workers("cpu", 2)
        defined.scale_workers("cpu", 5)

        assert len(fake_slurm.submissions) == 5
        assert defined.num_workers() == 5
        # Indices keep increasing rather than restarting.
        names = [s.job_name for s in fake_slurm.submissions]
        assert names[-1] == "testex.worker.cpu.4"

    def test_scaling_to_same_count_is_a_no_op(self, defined, fake_slurm):
        defined.scale_workers("cpu", 2)
        defined.scale_workers("cpu", 2)

        assert len(fake_slurm.submissions) == 2
        assert fake_slurm.cancelled_job_ids == []

    def test_scaling_down_cancels_jobs(self, defined, fake_slurm):
        defined.scale_workers("cpu", 3)
        defined.scale_workers("cpu", 1)

        assert defined.num_workers() == 1
        assert len(fake_slurm.cancelled_job_ids) == 2

    def test_scaling_to_zero_cancels_everything(self, defined, fake_slurm):
        defined.scale_workers("cpu", 2)
        job_ids = [s.job_id for s in fake_slurm.submissions]

        defined.scale_workers("cpu", 0)

        assert defined.num_workers() == 0
        assert sorted(fake_slurm.cancelled_job_ids) == sorted(job_ids)

    def test_already_finished_jobs_are_not_cancelled(self, defined, fake_slurm):
        """A worker whose job already exited should not be scancel'd."""
        defined.scale_workers("cpu", 2)
        fake_slurm.running_job_ids.clear()  # both jobs finished on their own

        defined.scale_workers("cpu", 0)

        assert fake_slurm.cancelled_job_ids == []
        assert defined.num_workers() == 0

    def test_submission_failure_propagates(self, defined, fake_slurm):
        fake_slurm.fail_command("sbatch", stderr="invalid account")

        with pytest.raises(Exception):
            defined.scale_workers("cpu", 1)

    def test_squeue_failure_becomes_runtime_error(self, defined, fake_slurm):
        defined.scale_workers("cpu", 2)
        fake_slurm.fail_command("squeue")

        with pytest.raises(RuntimeError, match="Failed to get running slurm job ids"):
            defined.scale_workers("cpu", 0)

    def test_scancel_failure_becomes_runtime_error(self, defined, fake_slurm):
        defined.scale_workers("cpu", 2)
        fake_slurm.fail_command("scancel")

        with pytest.raises(RuntimeError, match="Failed to cancel slurm jobs"):
            defined.scale_workers("cpu", 0)

    def test_batch_worker_uses_srun_or_not(
        self, executor, fake_slurm, setup_script, srun_lines
    ):
        executor.define_worker(
            name="batch",
            sbatch_args=[],
            setup_script=setup_script,
            is_batch_worker=True,
        )
        executor.define_worker(name="fanout", sbatch_args=[], setup_script=setup_script)

        executor.scale_workers("batch", 1)
        executor.scale_workers("fanout", 1)

        batch_script, fanout_script = fake_slurm.submissions
        batch_cmds = srun_lines(batch_script.script_text)
        fanout_cmds = srun_lines(fanout_script.script_text)
        assert batch_cmds == []
        # Two, because the choice is the job's to make when it starts:
        # a one-task job writes to the batch file,
        # anything larger takes a file per task
        # instead of interleaving them all into that one.
        assert len(fanout_cmds) == 2
        assert all(cmd.endswith(".sh'") for cmd in fanout_cmds)
        assert "--output" not in fanout_cmds[0]
        assert "--output" in fanout_cmds[1]

    def test_srun_output_files_are_per_task_and_in_the_work_dir(
        self, executor, fake_slurm, setup_script, srun_lines
    ):
        executor.define_worker(name="fanout", sbatch_args=[], setup_script=setup_script)

        executor.scale_workers("fanout", 1)

        (submission,) = fake_slurm.submissions
        _, per_task = srun_lines(submission.script_text)
        assert (
            f"--output '{executor.work_dir}/testex.worker.fanout.0-%j-%t.out'"
            in per_task
        )


# --------------------------------------------------------------------------
# submit / as_completed
# --------------------------------------------------------------------------


class TestSubmit:
    def test_returns_pending_task_handle(self, executor):
        task = executor.submit("cpu", square, 5)

        assert task.output is NoOutput
        assert task.queue == ["cpu"]
        assert task.input == ((5,), {})

    def test_task_ids_are_unique(self, executor):
        tasks = [executor.submit("cpu", square, i) for i in range(10)]

        assert len({t.task_id for t in tasks}) == 10

    def test_accepts_a_list_of_queues(self, executor, ds_client):
        executor.submit(["cpu", "gpu"], square, 5)

        # Enqueued on both, so a worker on either queue can serve it.
        fetched = ds_client.task_get("test-worker", "gpu")
        assert cloudpickle.loads(fetched.input) == ((5,), {})

    def test_rejects_wrong_queue_type(self, executor):
        with pytest.raises(TypeCheckError):
            executor.submit(42, square, 1)

    def test_serializes_closures_and_lambdas(self, executor, ds_client):
        """The captured variable has to survive the trip to the queue."""
        factor = 7
        task = executor.submit("cpu", lambda x: x * factor, 6)

        (drained,) = drain(ds_client, "cpu", 1)

        assert drained == task.task_id
        assert cloudpickle.loads(ds_client.task_get_output(drained)) == 42

    def test_submission_order_is_dispatch_order(self, executor, ds_client):
        """Tasks are served oldest first.

        ds-service dispatches the highest priority first,
        so a priority that rises with time would hand out
        the most recently submitted task and leave the oldest until last.
        """
        tasks = [executor.submit("cpu", square, i) for i in range(6)]

        served = drain(ds_client, "cpu", 6)

        assert served == [t.task_id for t in tasks]

    def test_an_earlier_task_outranks_a_later_one(self, executor):
        """The ordering above, read off the priorities themselves."""
        first = executor.submit("cpu", square, 1)
        second = executor.submit("cpu", square, 2)

        assert first.priority > second.priority


class TestTaskName:
    def test_a_task_is_unnamed_to_start_with(self, executor):
        task = executor.submit("cpu", square, 5)

        assert task.task_name is None

    def test_naming_records_it_on_the_task_and_the_server(self, executor, ds_client):
        task = executor.submit("cpu", square, 5)

        executor.set_task_name(task, "warmup")

        assert task.task_name == "warmup"
        assert ds_client.map_get(f"task_name:{task.task_id}") == b"warmup"

    def test_renaming_replaces_both_copies(self, executor, ds_client):
        task = executor.submit("cpu", square, 5)

        executor.set_task_name(task, "first")
        executor.set_task_name(task, "second")

        assert task.task_name == "second"
        assert ds_client.map_get(f"task_name:{task.task_id}") == b"second"

    def test_each_task_is_named_separately(self, executor, ds_client):
        first = executor.submit("cpu", square, 1)
        second = executor.submit("cpu", square, 2)

        executor.set_task_name(first, "one")

        assert second.task_name is None
        with pytest.raises(KeyError):
            ds_client.map_get(f"task_name:{second.task_id}")

    def test_the_property_is_read_only(self, executor):
        task = executor.submit("cpu", square, 5)

        with pytest.raises(AttributeError):
            task.task_name = "direct"  # type: ignore[misc]

    def test_rejects_a_non_string_name(self, executor):
        task = executor.submit("cpu", square, 5)

        with pytest.raises(TypeCheckError):
            executor.set_task_name(task, 42)  # type: ignore[arg-type]


class TestAsCompleted:
    @pytest.fixture(autouse=True)
    def _pilot_jobs(self, pilot_jobs):
        pilot_jobs("cpu")

    def test_yields_results(self, executor, ds_client):
        tasks = [executor.submit("cpu", square, i) for i in range(5)]
        drain(ds_client, "cpu", 5)

        results = {t.task_id: t.output for t in executor.as_completed(tasks)}

        assert sorted(results.values()) == [0, 1, 4, 9, 16]

    def test_closures_round_trip(self, executor, ds_client):
        factor = 7
        task = executor.submit("cpu", lambda x: x * factor, 6)
        drain(ds_client, "cpu", 1)

        executor.wait([task])

        assert task.output == 42

    def test_kwargs_round_trip(self, executor, ds_client):
        def add(a, b=0):
            return a + b

        task = executor.submit("cpu", add, 1, b=41)
        drain(ds_client, "cpu", 1)

        executor.wait([task])

        assert task.output == 42

    def test_populates_task_output_in_place(self, executor, ds_client):
        tasks = [executor.submit("cpu", square, i) for i in range(3)]
        drain(ds_client, "cpu", 3)

        executor.wait(tasks)

        assert sorted(t.output for t in tasks) == [0, 1, 4]

    def test_yields_incrementally(self, executor, ds_client, time_limit):
        """Finished tasks come back without waiting for the whole batch."""
        tasks = [executor.submit("cpu", square, i) for i in range(5)]
        drain(ds_client, "cpu", 2)

        with time_limit(10, "as_completed waited for unfinished tasks"):
            got = list(itertools.islice(executor.as_completed(tasks), 2))

        assert len(got) == 2

    def test_already_completed_tasks_are_served_from_cache(self, executor, ds_client):
        tasks = [executor.submit("cpu", square, i) for i in range(3)]
        drain(ds_client, "cpu", 3)
        executor.wait(tasks)

        counting = CountingClient(executor.client)
        executor.client = counting
        again = list(executor.as_completed(tasks))

        assert len(again) == 3
        assert counting.status_calls == 0, "cached tasks must not be re-polled"

    def test_status_is_polled_in_one_batched_call(self, executor, ds_client):
        tasks = [executor.submit("cpu", square, i) for i in range(6)]
        drain(ds_client, "cpu", 6)

        counting = CountingClient(executor.client)
        executor.client = counting
        executor.wait(tasks)

        assert counting.status_calls == 1
        assert counting.status_batch_sizes == [6]
        assert counting.output_calls == 6, "output fetched only for finished tasks"

    def test_unknown_task_id_raises(self, executor, time_limit):
        # Without Undefined handling this polls forever instead of raising.
        with time_limit(10, "as_completed never terminated for an unknown task"):
            with pytest.raises(RuntimeError, match="unknown to the task queue server"):
                list(executor.as_completed([ghost_task()]))

    def test_unknown_task_id_raises_from_wait(self, executor, time_limit):
        # `wait` gets this by delegating to `as_completed`,
        # so the guarantee is pinned to both entry points
        # rather than resting on that delegation staying put.
        with time_limit(10, "wait never terminated for an unknown task"):
            with pytest.raises(RuntimeError, match="unknown to the task queue server"):
                executor.wait([ghost_task()])

    def test_canceled_task_raises(self, executor, ds_client, time_limit):
        # `Canceled` arrived with ds-service 4.0.0.
        # Nothing here cancels, so this is an out-of-band cancellation,
        # and a state the poll loop does not name waits forever.
        task = executor.submit("cpu", square, 3)
        assert ds_client.task_cancel(task.task_id)

        with time_limit(10, "as_completed never terminated for a canceled task"):
            with pytest.raises(RuntimeError, match="was canceled"):
                list(executor.as_completed([task]))

    def test_empty_task_list(self, executor):
        # Nothing is pending, so neither the no-worker check
        # nor the poll loop has anything to run against.
        assert list(executor.as_completed([])) == []


class TestRemoteErrors:
    @pytest.fixture(autouse=True)
    def _pilot_jobs(self, pilot_jobs):
        pilot_jobs("cpu")

    def test_a_worker_exception_comes_back_as_the_output(self, executor, ds_client):
        """It never propagates out of the worker; the policy decides the rest."""
        task = executor.submit("cpu", square, 1)
        fail_one(ds_client, "cpu", error_id="ERROR_abc")

        executor.wait([task], raise_on_error=RaiseOnError.RAISE_NEVER)

        assert isinstance(task.output, RemoteExecutionError)
        assert task.output.error_id == "ERROR_abc"

    def test_a_failure_raises_by_default(self, executor, ds_client):
        task = executor.submit("cpu", square, 1)
        fail_one(ds_client, "cpu", error_id="ERROR_xyz")

        with pytest.raises(RuntimeError, match="ERROR_xyz"):
            executor.wait([task])

    def test_the_first_failure_stops_the_wait(self, executor, ds_client):
        tasks = [executor.submit("cpu", square, i) for i in range(3)]
        fail_one(ds_client, "cpu")

        with pytest.raises(RuntimeError):
            executor.wait(tasks)

        # The other two were never waited for.
        assert [t.output is NoOutput for t in tasks] == [False, True, True]

    def test_raise_never_returns_the_failure_as_a_result(self, executor, ds_client):
        tasks = [executor.submit("cpu", square, i) for i in range(3)]
        fail_one(ds_client, "cpu")
        drain(ds_client, "cpu", 2)

        executor.wait(tasks, raise_on_error=RaiseOnError.RAISE_NEVER)

        failed = [t for t in tasks if isinstance(t.output, RemoteExecutionError)]
        assert len(failed) == 1
        assert all(t.output is not NoOutput for t in tasks)

    def test_raise_after_completed_waits_for_the_rest_first(self, executor, ds_client):
        tasks = [executor.submit("cpu", square, i) for i in range(3)]
        fail_one(ds_client, "cpu")
        drain(ds_client, "cpu", 2)

        with pytest.raises(RuntimeError, match="1 of 3 tasks did not succeed"):
            executor.wait(tasks, raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED)

        assert all(t.output is not NoOutput for t in tasks)

    def test_a_deferred_exception_names_every_failure(self, executor, ds_client):
        tasks = [executor.submit("cpu", square, i) for i in range(2)]
        fail_one(ds_client, "cpu", error_id="ERROR_one")
        fail_one(ds_client, "cpu", error_id="ERROR_two")

        with pytest.raises(RuntimeError) as raised:
            executor.wait(tasks, raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED)

        assert "ERROR_one" in str(raised.value)
        assert "ERROR_two" in str(raised.value)

    def test_a_long_list_of_failures_is_summarized(self, executor, ds_client):
        tasks = [executor.submit("cpu", square, i) for i in range(7)]
        for _ in range(7):
            fail_one(ds_client, "cpu")

        with pytest.raises(RuntimeError, match="and 2 more"):
            executor.wait(tasks, raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED)

    @pytest.mark.parametrize(
        "policy",
        [
            RaiseOnError.RAISE_ON_FIRST_ERROR,
            RaiseOnError.RAISE_AFTER_COMPLETED,
            RaiseOnError.RAISE_NEVER,
        ],
    )
    def test_every_failure_is_warned_about_whatever_the_policy(
        self, executor, ds_client, capsys, policy
    ):
        task = executor.submit("cpu", square, 1)
        fail_one(ds_client, "cpu", error_id="ERROR_xyz")

        try:
            executor.wait([task], raise_on_error=policy)
        except RuntimeError:
            pass

        err = capsys.readouterr().err
        assert "ERROR_xyz" in err
        assert task.task_id in err

    def test_as_completed_cannot_defer_and_says_so_by_raising(
        self, executor, ds_client
    ):
        """There is no "after" once results are being handed out."""
        tasks = [executor.submit("cpu", square, i) for i in range(3)]
        fail_one(ds_client, "cpu")
        drain(ds_client, "cpu", 2)

        with pytest.raises(RuntimeError):
            list(
                executor.as_completed(
                    tasks, raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED
                )
            )

    def test_as_completed_yields_failures_when_never_raising(self, executor, ds_client):
        tasks = [executor.submit("cpu", square, i) for i in range(3)]
        fail_one(ds_client, "cpu")
        drain(ds_client, "cpu", 2)

        seen = list(
            executor.as_completed(tasks, raise_on_error=RaiseOnError.RAISE_NEVER)
        )

        assert len(seen) == 3
        assert sum(isinstance(t.output, RemoteExecutionError) for t in seen) == 1


# --------------------------------------------------------------------------
# live queues
# --------------------------------------------------------------------------


class TestLiveQueues:
    def test_no_groups_means_nothing_is_live(self, executor):
        assert executor._live_queues() == set()

    def test_a_defined_but_unscaled_group_is_not_live(self, executor, setup_script):
        executor.define_worker("cpu", [], setup_script)

        assert executor._live_queues() == set()

    def test_a_group_with_a_queued_job_is_live(self, executor, setup_script):
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)

        assert executor._live_queues() == {"cpu"}

    def test_only_groups_with_jobs_still_on_the_cluster_are_live(
        self, executor, fake_slurm, setup_script
    ):
        executor.define_worker("cpu", [], setup_script)
        executor.define_worker("gpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        executor.scale_workers("gpu", 1)
        gpu_job = fake_slurm.submissions[-1].job_id

        # The gpu job ends; its group has nothing left on the cluster.
        fake_slurm.running_job_ids.remove(gpu_job)

        assert executor._live_queues() == {"cpu"}

    def test_a_group_is_live_while_any_of_its_jobs_survives(
        self, executor, fake_slurm, setup_script
    ):
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 3)
        first = fake_slurm.submissions[0].job_id

        fake_slurm.running_job_ids.remove(first)

        assert executor._live_queues() == {"cpu"}

    def test_queues_argument_restricts_the_answer(self, executor, setup_script):
        executor.define_worker("cpu", [], setup_script)
        executor.define_worker("gpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        executor.scale_workers("gpu", 1)

        assert executor._live_queues(["cpu"]) == {"cpu"}
        assert executor._live_queues(["cpu", "gpu"]) == {"cpu", "gpu"}
        assert executor._live_queues([]) == set()

    def test_unknown_queue_names_are_absent_not_an_error(self, executor, setup_script):
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)

        assert executor._live_queues(["nope"]) == set()
        assert executor._live_queues(["cpu", "nope"]) == {"cpu"}

    def test_squeue_failure_propagates(self, executor, fake_slurm, setup_script):
        """Unknown liveness must not be reported as "nothing is live"."""
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        fake_slurm.fail_command("squeue")

        with pytest.raises(subprocess.CalledProcessError):
            executor._live_queues()


# --------------------------------------------------------------------------
# no worker started
# --------------------------------------------------------------------------


class TestNoWorkerStarted:
    """The up-front check, before any polling and without asking Slurm."""

    def test_a_group_that_was_never_scaled_is_rejected(self, executor, setup_script):
        """Defining a group submits nothing, so no worker exists for it."""
        executor.define_worker("cpu", [], setup_script)
        task = executor.submit("cpu", square, 2)

        with pytest.raises(RuntimeError, match="no worker started"):
            list(executor.as_completed([task]))

    def test_a_queue_matching_no_group_is_rejected(self, executor):
        """Queue names are not validated at submit time, so a typo lands here."""
        task = executor.submit("typo-in-queue-name", square, 2)

        with pytest.raises(RuntimeError, match="no worker started"):
            list(executor.as_completed([task]))

    def test_the_error_names_the_queues(self, executor):
        task = executor.submit("ghost", square, 2)

        with pytest.raises(RuntimeError, match=r"\['ghost'\]"):
            list(executor.as_completed([task]))

    def test_wait_rejects_too(self, executor):
        task = executor.submit("ghost", square, 2)

        with pytest.raises(RuntimeError, match="no worker started"):
            executor.wait([task])

    def test_it_raises_before_yielding_anything(
        self, executor, ds_client, setup_script
    ):
        """A finished task in the same batch must not mask the bad one."""
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        done = executor.submit("cpu", square, 3)
        drain(ds_client, "cpu", 1)
        executor.wait([done])

        stranded = executor.submit("ghost", square, 2)

        yielded = []
        with pytest.raises(RuntimeError, match="no worker started"):
            for task in executor.as_completed([done, stranded]):
                yielded.append(task)

        assert yielded == [], "the error must come before any result"

    def test_only_the_starved_tasks_are_given_up_on(
        self, executor, ds_client, setup_script
    ):
        """A queue nobody scaled says nothing about the queues that were."""
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        good = [executor.submit("cpu", square, i) for i in range(3)]
        starved = executor.submit("ghost", square, 9)
        drain(ds_client, "cpu", 3)

        executor.wait(good + [starved], raise_on_error=RaiseOnError.RAISE_NEVER)

        assert [task.output for task in good] == [0, 1, 4]
        assert starved.output is NoOutput

    def test_the_rest_of_the_batch_is_waited_for_before_raising(
        self, executor, ds_client, setup_script
    ):
        """What RAISE_AFTER_COMPLETED promises: everything that can finish."""
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        good = [executor.submit("cpu", square, i) for i in range(3)]
        starved = executor.submit("ghost", square, 9)
        drain(ds_client, "cpu", 3)

        with pytest.raises(RuntimeError, match="1 of 4 tasks did not succeed"):
            executor.wait(
                good + [starved], raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED
            )

        assert all(task.output is not NoOutput for task in good)

    def test_the_count_is_of_tasks_not_of_messages(
        self, executor, ds_client, setup_script
    ):
        """One message covers every task on a dead queue."""
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        done = executor.submit("cpu", square, 1)
        starved = [executor.submit("ghost", square, i) for i in range(3)]
        drain(ds_client, "cpu", 1)

        with pytest.raises(RuntimeError, match="3 of 4 tasks did not succeed"):
            executor.wait(
                [done] + starved, raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED
            )

    def test_one_live_queue_is_enough(self, executor, ds_client, setup_script):
        """A task submitted to several queues needs a worker on only one."""
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        task = executor.submit(["cpu", "ghost"], square, 4)

        drain(ds_client, "cpu", 1)

        (result,) = list(executor.as_completed([task]))
        assert result.output == 16

    def test_finished_tasks_need_no_worker(self, executor, ds_client, setup_script):
        """Nothing is pending, so there is nothing a worker could still run."""
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        task = executor.submit("cpu", square, 5)
        drain(ds_client, "cpu", 1)
        executor.wait([task])

        executor.groups.clear()  # as if this executor never started anything

        assert [t.output for t in executor.as_completed([task])] == [25]


# --------------------------------------------------------------------------
# stranded tasks
# --------------------------------------------------------------------------


@pytest.fixture
def check_immediately(monkeypatch):
    """Collapse the liveness interval so one poll triggers a check.

    The real 60s gap exists so that submitting before scaling workers keeps
    working; these tests are about what happens once the gap has elapsed.
    """

    monkeypatch.setattr(spe, "LIVE_QUEUE_CHECK_INTERVAL_S", 0.0)


class TestStrandedTasks:
    def test_raises_when_the_queue_has_no_live_job(
        self, executor, fake_slurm, setup_script, check_immediately, time_limit
    ):
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        task = executor.submit("cpu", square, 2)

        # Every pilot job for `cpu` ends without draining the queue.
        fake_slurm.running_job_ids.clear()

        with time_limit(10, "as_completed did not notice the dead queue"):
            with pytest.raises(RuntimeError, match="no live pilot job"):
                list(executor.as_completed([task]))

    def test_error_names_the_dead_queues(
        self, executor, fake_slurm, setup_script, check_immediately, time_limit
    ):
        executor.define_worker("gpu", [], setup_script)
        executor.scale_workers("gpu", 1)
        task = executor.submit("gpu", square, 2)
        fake_slurm.running_job_ids.clear()

        with time_limit(10, "as_completed did not notice the dead queue"):
            with pytest.raises(RuntimeError, match=r"\['gpu'\]"):
                list(executor.as_completed([task]))

    def test_a_task_is_fine_while_any_of_its_queues_is_live(
        self, executor, fake_slurm, setup_script, ds_client, check_immediately
    ):
        """Submitting to several queues survives losing one of them."""
        executor.define_worker("cpu", [], setup_script)
        executor.define_worker("gpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        executor.scale_workers("gpu", 1)
        gpu_job = fake_slurm.submissions[-1].job_id

        task = executor.submit(["cpu", "gpu"], square, 3)
        fake_slurm.running_job_ids.remove(gpu_job)

        drain(ds_client, "cpu", 1)

        (done,) = list(executor.as_completed([task]))
        assert done.output == 9

    def test_wait_raises_too(
        self, executor, fake_slurm, setup_script, check_immediately, time_limit
    ):
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        task = executor.submit("cpu", square, 2)
        fake_slurm.running_job_ids.clear()

        with time_limit(10, "wait did not notice the dead queue"):
            with pytest.raises(RuntimeError, match="no live pilot job"):
                executor.wait([task])

    def test_only_the_stranded_tasks_are_given_up_on(
        self, executor, fake_slurm, setup_script, ds_client, check_immediately
    ):
        """One group reaching its walltime must not discard another's work.

        The shape of a real run: a one-job `opt` pool dies
        while the `eval` pool is still working through its round.
        """
        executor.define_worker("eval", [], setup_script)
        executor.define_worker("opt", [], setup_script)
        executor.scale_workers("eval", 1)
        executor.scale_workers("opt", 1)
        opt_job = fake_slurm.submissions[-1].job_id

        working = [executor.submit("eval", square, i) for i in range(3)]
        stranded = executor.submit("opt", square, 9)

        fake_slurm.running_job_ids.remove(opt_job)
        drain(ds_client, "eval", 3)

        executor.wait(working + [stranded], raise_on_error=RaiseOnError.RAISE_NEVER)

        assert [task.output for task in working] == [0, 1, 4]
        assert stranded.output is NoOutput

    def test_the_stranded_count_is_of_tasks(
        self, executor, fake_slurm, setup_script, ds_client, check_immediately
    ):
        executor.define_worker("eval", [], setup_script)
        executor.define_worker("opt", [], setup_script)
        executor.scale_workers("eval", 1)
        executor.scale_workers("opt", 1)
        opt_job = fake_slurm.submissions[-1].job_id

        working = executor.submit("eval", square, 1)
        stranded = [executor.submit("opt", square, i) for i in range(2)]

        fake_slurm.running_job_ids.remove(opt_job)
        drain(ds_client, "eval", 1)

        with pytest.raises(RuntimeError, match="2 of 3 tasks did not succeed"):
            executor.wait(
                [working] + stranded,
                raise_on_error=RaiseOnError.RAISE_AFTER_COMPLETED,
            )

        assert working.output == 1

    def test_already_finished_tasks_are_not_checked(
        self, executor, fake_slurm, setup_script, ds_client, check_immediately
    ):
        """Nothing is pending, so a dead queue is irrelevant."""
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        task = executor.submit("cpu", square, 4)
        drain(ds_client, "cpu", 1)
        executor.wait([task])

        fake_slurm.running_job_ids.clear()

        assert [t.output for t in executor.as_completed([task])] == [16]

    def test_squeue_failure_does_not_abort_the_wait(
        self, executor, fake_slurm, setup_script, ds_client, check_immediately
    ):
        """Unknown liveness is not dead liveness: keep waiting."""
        executor.define_worker("cpu", [], setup_script)
        executor.scale_workers("cpu", 1)
        task = executor.submit("cpu", square, 5)
        fake_slurm.fail_command("squeue")

        drain(ds_client, "cpu", 1)

        (done,) = list(executor.as_completed([task]))
        assert done.output == 25

    def test_not_checked_before_the_interval_elapses(
        self, executor, fake_slurm, setup_script, ds_client
    ):
        """A queue with no job yet is normal right after submitting.

        This is the pattern the README documents:
        submit first, scale workers after.
        Without the initial delay it would raise instead of waiting.
        """
        executor.define_worker("cpu", [], setup_script)
        task = executor.submit("cpu", square, 6)
        assert fake_slurm.running_job_ids == []

        executor.scale_workers("cpu", 1)
        drain(ds_client, "cpu", 1)

        (done,) = list(executor.as_completed([task]))
        assert done.output == 36


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


class TestLogging:
    """One executor, one log file.

    The logger is named after the executor
    because a name shared between them collects a handler per executor,
    and every line then lands in every work dir opened in this process.
    That makes the name the identity here too,
    so each test below uses its own
    rather than inheriting the logger a previous test left in the registry.
    """

    @staticmethod
    def file_handlers(ex) -> list[logging.Handler]:
        """The executor's own handlers.

        Filtered, because the logging module's registry is global
        and pytest's capture plugin puts handlers of its own on loggers
        while a test runs.
        """
        return [h for h in ex.logger.handlers if isinstance(h, logging.FileHandler)]

    def test_two_executors_do_not_share_a_log_file(
        self, ds_service_address, fake_slurm, tmp_path, setup_script
    ):
        first = SlurmPilotExecutor(
            "sharedA", ds_service_address, work_dir=tmp_path / "first"
        )
        second = SlurmPilotExecutor(
            "sharedB", ds_service_address, work_dir=tmp_path / "second"
        )

        # scale_workers is what logs; anything that writes a record will do.
        second.define_worker("cpu", [], setup_script)
        second.scale_workers("cpu", 1)

        assert not (tmp_path / "first" / "executor.log").exists()
        assert "Starting worker" in (tmp_path / "second" / "executor.log").read_text()

        first.close()
        second.close()

    def test_each_executor_gets_exactly_one_handler(
        self, ds_service_address, fake_slurm, tmp_path
    ):
        first = SlurmPilotExecutor(
            "handlerA", ds_service_address, work_dir=tmp_path / "first"
        )
        second = SlurmPilotExecutor(
            "handlerB", ds_service_address, work_dir=tmp_path / "second"
        )

        assert first.logger is not second.logger
        assert len(self.file_handlers(first)) == 1
        assert len(self.file_handlers(second)) == 1

        first.close()
        second.close()

    def test_close_releases_the_log_file(
        self, ds_service_address, fake_slurm, tmp_path, setup_script
    ):
        """The logger outlives the executor, so it must not keep the handler."""
        ex = SlurmPilotExecutor(
            "closes", ds_service_address, work_dir=tmp_path / "work"
        )
        ex.define_worker("cpu", [], setup_script)
        ex.scale_workers("cpu", 1)

        ex.close()

        assert self.file_handlers(ex) == []

    def test_closing_twice_releases_it_once(
        self, ds_service_address, fake_slurm, tmp_path
    ):
        ex = SlurmPilotExecutor("twice", ds_service_address, work_dir=tmp_path / "work")

        ex.close()
        ex.close()  # must not raise

        assert self.file_handlers(ex) == []

    def test_records_do_not_reach_the_root_logger(
        self, ds_service_address, fake_slurm, tmp_path, setup_script, caplog
    ):
        """The work dir is where these belong, not the importer's handlers."""
        ex = SlurmPilotExecutor(
            "rootlog", ds_service_address, work_dir=tmp_path / "work"
        )

        with caplog.at_level(logging.INFO):
            ex.define_worker("cpu", [], setup_script)
            ex.scale_workers("cpu", 1)

        assert "Starting worker" not in caplog.text
        ex.close()


class TestLifecycle:
    def test_work_dir_is_created(self, ds_service_address, fake_slurm, tmp_path):
        work_dir = tmp_path / "nested" / "run"
        ex = SlurmPilotExecutor(
            name="lifecycle", server_address=ds_service_address, work_dir=work_dir
        )

        assert work_dir.is_dir()
        ex.close()

    def test_stop_cancels_jobs_but_keeps_executor_usable(
        self, executor, fake_slurm, setup_script
    ):
        executor.define_worker(name="cpu", sbatch_args=[], setup_script=setup_script)
        executor.scale_workers("cpu", 2)
        job_ids = [s.job_id for s in fake_slurm.submissions]

        executor.stop()

        assert sorted(fake_slurm.cancelled_job_ids) == sorted(job_ids)
        assert executor.num_workers() == 0
        # The client is still open, so the executor can be reused.
        assert executor.submit("cpu", square, 2) is not None

    def test_close_cancels_all_groups(self, executor, fake_slurm, setup_script):
        executor.define_worker("a", [], setup_script)
        executor.define_worker("b", [], setup_script)
        executor.scale_workers("a", 1)
        executor.scale_workers("b", 2)
        job_ids = [s.job_id for s in fake_slurm.submissions]

        executor.close()

        assert sorted(fake_slurm.cancelled_job_ids) == sorted(job_ids)
        assert executor.num_workers() == 0

    def test_close_tolerates_squeue_failure(self, executor, fake_slurm, setup_script):
        """Cleanup must not explode if the cluster is unreachable."""
        executor.define_worker("a", [], setup_script)
        executor.scale_workers("a", 1)
        fake_slurm.fail_command("squeue")

        executor.close()  # must not raise

    def test_close_is_idempotent(self, executor, fake_slurm, setup_script):
        executor.define_worker("a", [], setup_script)
        executor.scale_workers("a", 1)

        executor.close()
        executor.close()  # must not raise

    def test_context_manager_yields_the_executor(self, executor):
        with executor as entered:
            assert entered is executor

    def test_context_manager_closes_on_exit(self, executor, fake_slurm, setup_script):
        with executor:
            executor.define_worker("a", [], setup_script)
            executor.scale_workers("a", 2)
            job_ids = [s.job_id for s in fake_slurm.submissions]

        assert sorted(fake_slurm.cancelled_job_ids) == sorted(job_ids)
        assert executor.num_workers() == 0

    def test_context_manager_closes_after_an_exception(
        self, executor, fake_slurm, setup_script
    ):
        """The jobs still get cancelled, and the exception still escapes."""
        with pytest.raises(ValueError, match="boom"):
            with executor:
                executor.define_worker("a", [], setup_script)
                executor.scale_workers("a", 1)
                raise ValueError("boom")

        assert fake_slurm.cancelled_job_ids == [fake_slurm.submissions[0].job_id]
        assert executor.num_workers() == 0
