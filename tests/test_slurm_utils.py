"""Tests for the Slurm command wrappers (sbatch/squeue/scancel are mocked)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from slurm_workflows import slurm_utils
from slurm_workflows.slurm_utils import (
    cancel_jobs,
    get_clean_environ,
    get_running_jobids,
    submit_sbatch_job,
)


class TestGetCleanEnviron:
    def test_strips_slurm_variables(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SLURM_JOB_ID", "1")
        monkeypatch.setenv("SLURMD_NODENAME", "node1")
        monkeypatch.setenv("PMI_RANK", "0")
        monkeypatch.setenv("SRUN_DEBUG", "3")
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("MY_VAR", "keep-me")
        get_clean_environ.cache_clear()

        env = get_clean_environ()

        assert "SLURM_JOB_ID" not in env
        assert "SLURMD_NODENAME" not in env
        assert "PMI_RANK" not in env
        assert "SRUN_DEBUG" not in env
        assert env["PATH"] == "/usr/bin"
        assert env["MY_VAR"] == "keep-me"

        get_clean_environ.cache_clear()

    def test_keeps_variables_merely_containing_slurm(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Only the documented prefixes are stripped, not substrings."""
        monkeypatch.setenv("MY_SLURM_HELPER", "keep-me")
        get_clean_environ.cache_clear()

        assert get_clean_environ()["MY_SLURM_HELPER"] == "keep-me"

        get_clean_environ.cache_clear()


class TestSubmitSbatchJob:
    def test_returns_parsed_job(self, fake_slurm, tmp_path: Path):
        job = submit_sbatch_job(
            name="myjob",
            sbatch_args=["-A alloc", "-p standard"],
            script="echo hello",
            work_dir=tmp_path,
        )

        assert job.job_id == 1000
        assert job.name == "myjob"
        assert job.sbatch_args == ["-A alloc", "-p standard"]

    def test_writes_executable_script_with_directives(self, fake_slurm, tmp_path: Path):
        submit_sbatch_job(
            name="myjob",
            sbatch_args=["-A alloc", "-t 01:00:00"],
            script="echo hello",
            work_dir=tmp_path,
        )

        script_path = tmp_path / "myjob.sbatch"
        assert script_path.exists()
        assert script_path.stat().st_mode & 0o111, "script should be executable"

        text = script_path.read_text()
        assert text.startswith("#!/bin/bash")
        assert '#SBATCH --job-name "myjob"' in text
        assert "#SBATCH -A alloc" in text
        assert "#SBATCH -t 01:00:00" in text
        assert "echo hello" in text

    def test_resolves_job_id_in_output_file(self, fake_slurm, tmp_path: Path):
        job = submit_sbatch_job(
            name="myjob", sbatch_args=[], script="true", work_dir=tmp_path
        )

        assert "%j" not in str(job.output_file)
        assert job.output_file == tmp_path / f"myjob-{job.job_id}.out"

    def test_submits_with_scrubbed_environment(
        self, fake_slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Submitting from inside a Slurm job must not leak SLURM_* through."""
        monkeypatch.setenv("SLURM_JOB_ID", "999")
        monkeypatch.setenv("KEEP_ME", "yes")
        slurm_utils.get_clean_environ.cache_clear()

        submit_sbatch_job(
            name="myjob", sbatch_args=[], script="true", work_dir=tmp_path
        )

        env = fake_slurm.submissions[0].env
        assert "SLURM_JOB_ID" not in env
        assert env["KEEP_ME"] == "yes"

    def test_finds_the_job_id_after_a_banner(self, fake_slurm, tmp_path: Path):
        """Sites put warnings and banners on sbatch's stdout.

        The job id line is searched for rather than matched at the start,
        so anything printed ahead of it is skipped
        instead of failing a submission that in fact succeeded.
        """
        fake_slurm.sbatch_stdout_override = (
            "sbatch: WARNING: your account is nearly out of hours\n"
            "Submitted batch job 4242\n"
        )

        job = submit_sbatch_job(
            name="myjob", sbatch_args=[], script="true", work_dir=tmp_path
        )

        assert job.job_id == 4242

    def test_raises_on_unparsable_sbatch_output(self, fake_slurm, tmp_path: Path):
        fake_slurm.sbatch_stdout_override = "something unexpected\n"

        with pytest.raises(RuntimeError, match="Failed to parse sbatch output"):
            submit_sbatch_job(
                name="myjob", sbatch_args=[], script="true", work_dir=tmp_path
            )

    def test_propagates_sbatch_failure(self, fake_slurm, tmp_path: Path):
        fake_slurm.fail_command("sbatch", returncode=2, stderr="invalid account")

        with pytest.raises(subprocess.CalledProcessError):
            submit_sbatch_job(
                name="myjob", sbatch_args=[], script="true", work_dir=tmp_path
            )


class TestGetRunningJobids:
    def test_parses_job_ids(self, fake_slurm):
        fake_slurm.running_job_ids = [11, 22, 33]

        assert get_running_jobids() == {11, 22, 33}

    def test_empty_when_nothing_queued(self, fake_slurm):
        assert get_running_jobids() == set()


class TestCancelJobs:
    def test_cancels_given_ids(self, fake_slurm):
        fake_slurm.running_job_ids = [11, 22]

        cancel_jobs([11, 22])

        assert fake_slurm.cancelled_job_ids == [11, 22]
        assert fake_slurm.running_job_ids == []

    def test_empty_list_issues_no_command(self, fake_slurm):
        cancel_jobs([])

        assert fake_slurm.cancelled_job_ids == []

    def test_flags_are_passed_through(self, fake_slurm, monkeypatch):
        seen = []
        real_run = fake_slurm.run
        monkeypatch.setattr(
            fake_slurm,
            "run",
            lambda cmd, **kw: (seen.append(cmd), real_run(cmd, **kw))[1],
        )

        cancel_jobs([7], term=True, batch=True, full=True)

        assert "--signal=TERM" in seen[0]
        assert "--batch" in seen[0]
        assert "--full" in seen[0]
