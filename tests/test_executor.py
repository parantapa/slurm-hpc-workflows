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

import pytest
import cloudpickle
from typeguard import TypeCheckError

from slurm_workflows import check_for_error
from slurm_workflows import slurm_pilot_executor as spe
from slurm_workflows.slurm_pilot_executor import (
    NoOutput,
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

        script = (executor.work_dir / "slurm_pilot_worker.cpu.0.sh").read_text()
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

        script = (executor.work_dir / "slurm_pilot_worker.cpu.0.sh").read_text()
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
        assert names == ["slurm_pilot_worker.cpu.0", "slurm_pilot_worker.cpu.1"]

    def test_worker_script_is_written_and_executable(self, defined, fake_slurm):
        defined.scale_workers("cpu", 1)

        script = defined.work_dir / "slurm_pilot_worker.cpu.0.sh"
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
        assert names[-1] == "slurm_pilot_worker.cpu.4"

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
            f"--output '{executor.work_dir}/slurm_pilot_worker.fanout.0-%j-%t.out'"
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

    def test_worker_exception_is_returned_not_raised(self, executor, ds_client):
        task = executor.submit("cpu", square, 1)
        fail_one(ds_client, "cpu", error_id="ERROR_abc")

        executor.wait([task])  # must not raise

        assert isinstance(task.output, RemoteExecutionError)
        assert task.output.error_id == "ERROR_abc"

    def test_check_for_error_finds_failed_tasks(self, executor, ds_client):
        tasks = [executor.submit("cpu", square, i) for i in range(3)]
        fail_one(ds_client, "cpu")
        drain(ds_client, "cpu", 2)
        executor.wait(tasks)

        failed = check_for_error(tasks, verbose=False)

        assert len(failed) == 1
        assert isinstance(failed[0].output, RemoteExecutionError)

    def test_check_for_error_returns_empty_when_all_succeed(self, executor, ds_client):
        tasks = [executor.submit("cpu", square, i) for i in range(3)]
        drain(ds_client, "cpu", 3)
        executor.wait(tasks)

        assert check_for_error(tasks, verbose=False) == []

    def test_check_for_error_prints_when_verbose(self, executor, ds_client, capsys):
        task = executor.submit("cpu", square, 1)
        fail_one(ds_client, "cpu", error_id="ERROR_xyz")
        executor.wait([task])

        check_for_error([task], verbose=True)

        out = capsys.readouterr().out
        assert "ERROR_xyz" in out
        assert task.task_id in out


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
    """

    def test_two_executors_do_not_share_a_log_file(
        self, ds_service_address, fake_slurm, tmp_path, setup_script
    ):
        first = SlurmPilotExecutor(ds_service_address, work_dir=tmp_path / "first")
        second = SlurmPilotExecutor(ds_service_address, work_dir=tmp_path / "second")

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
        first = SlurmPilotExecutor(ds_service_address, work_dir=tmp_path / "first")
        second = SlurmPilotExecutor(ds_service_address, work_dir=tmp_path / "second")

        assert first.logger is not second.logger
        assert len(first.logger.handlers) == 1
        assert len(second.logger.handlers) == 1

        first.close()
        second.close()

    def test_close_releases_the_log_file(
        self, ds_service_address, fake_slurm, tmp_path, setup_script
    ):
        """The logger outlives the executor, so it must not keep the handler."""
        ex = SlurmPilotExecutor(ds_service_address, work_dir=tmp_path / "work")
        ex.define_worker("cpu", [], setup_script)
        ex.scale_workers("cpu", 1)

        ex.close()

        assert ex.logger.handlers == []

    def test_closing_twice_releases_it_once(
        self, ds_service_address, fake_slurm, tmp_path
    ):
        ex = SlurmPilotExecutor(ds_service_address, work_dir=tmp_path / "work")

        ex.close()
        ex.close()  # must not raise

        assert ex.logger.handlers == []

    def test_records_do_not_reach_the_root_logger(
        self, ds_service_address, fake_slurm, tmp_path, setup_script, caplog
    ):
        """The work dir is where these belong, not the importer's handlers."""
        ex = SlurmPilotExecutor(ds_service_address, work_dir=tmp_path / "work")

        with caplog.at_level(logging.INFO):
            ex.define_worker("cpu", [], setup_script)
            ex.scale_workers("cpu", 1)

        assert "Starting worker" not in caplog.text
        ex.close()


class TestLifecycle:
    def test_work_dir_is_created(self, ds_service_address, fake_slurm, tmp_path):
        work_dir = tmp_path / "nested" / "run"
        ex = SlurmPilotExecutor(server_address=ds_service_address, work_dir=work_dir)

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
