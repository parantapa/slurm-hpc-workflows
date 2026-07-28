"""Tests for the multi-template-per-file Jinja loader and the templates."""

from __future__ import annotations

import json

import jinja2
import pytest

from slurm_workflows.templates import render_template


class TestLoader:
    def test_unknown_template_name_raises(self):
        with pytest.raises(jinja2.TemplateNotFound):
            render_template("slurm_pilot:no_such_template")  # type: ignore[call-overload]

    def test_unknown_file_prefix_raises(self):
        with pytest.raises(jinja2.TemplateNotFound):
            render_template("no_such_file:whatever")  # type: ignore[call-overload]

    def test_missing_variable_is_a_hard_error(self):
        """The environment uses StrictUndefined."""
        with pytest.raises(jinja2.UndefinedError):
            render_template(
                "slurm_pilot:worker_sbatch_script",  # type: ignore[call-overload]
                is_batch_worker=True,
                # worker_script_path deliberately omitted
            )


class TestWorkerSbatchScript:
    def render(self, **overrides):
        kwargs = dict(
            name="slurm_pilot_worker.cpu.0",
            work_dir="/scratch/work",
            is_batch_worker=False,
            worker_script_path="/path/to/worker.sh",
        )
        kwargs.update(overrides)
        return render_template("slurm_pilot:worker_sbatch_script", **kwargs)

    def test_batch_worker_sources_script_directly(self):
        out = self.render(is_batch_worker=True)

        assert ". '/path/to/worker.sh'" in out
        assert "srun" not in out

    def test_batch_worker_leaves_output_to_sbatch(self):
        """One process, one allocation: the job's own --output already has it."""
        out = self.render(is_batch_worker=True)

        assert "--output" not in out

    def test_non_batch_worker_is_wrapped_in_srun(self):
        out = self.render()

        assert (
            "srun --output '/scratch/work/slurm_pilot_worker.cpu.0-%j-%t.out' "
            "/bin/bash '/path/to/worker.sh'" in out
        )

    def test_each_srun_task_gets_its_own_output_file(self):
        """`srun` fans out over every task in the allocation.

        Without a per-task --output they would all interleave into the one
        batch output file, so the pattern has to carry both the job id and
        the task id.
        """
        out = self.render()

        srun_line = next(ln for ln in out.splitlines() if ln.startswith("srun"))
        assert "%j" in srun_line and "%t" in srun_line

    def test_the_output_pattern_lands_in_the_work_dir(self):
        out = self.render(work_dir="/some/other/dir")

        assert "--output '/some/other/dir/" in out


class TestWorkerScript:
    def render(self, **overrides):
        kwargs = dict(
            worker_exe="slurm-pilot-worker",
            setup_script="module load gcc\nconda activate my-env",
            group="cpu",
            name="slurm_pilot_worker.cpu.0",
            actor_class_name="",
            server_address="10.0.0.1:5051",
            work_dir="/scratch/work",
            python_paths_json=json.dumps(["/a", "/b"]),
        )
        kwargs.update(overrides)
        return render_template("slurm_pilot:worker_script", **kwargs)

    def test_setup_script_body_is_inlined_verbatim(self):
        out = self.render()

        assert "module load gcc\nconda activate my-env" in out
        # It is a body, not a path: nothing sources it.
        assert ". 'module load gcc" not in out

    def test_multiline_body_is_not_escaped(self):
        out = self.render(
            setup_script='export A="x y"\nexport B=$HOME/z\n# a comment'
        )

        assert 'export A="x y"' in out
        assert "export B=$HOME/z" in out
        assert "# a comment" in out

    def test_empty_body_is_allowed(self):
        out = self.render(setup_script="")

        assert ". '/etc/profile'" in out
        assert "slurm-pilot-worker \\" in out

    def test_fails_fast_on_setup_errors(self):
        assert "set -Eeuo pipefail" in self.render()

    def test_passes_all_worker_arguments(self):
        out = self.render()

        assert "--group 'cpu'" in out
        assert "--name 'slurm_pilot_worker.cpu.0'" in out
        assert "--server-address '10.0.0.1:5051'" in out
        assert "--work-dir '/scratch/work'" in out
        assert """--python-paths-json '["/a", "/b"]'""" in out

    def test_actor_class_name_is_forwarded(self):
        out = self.render(actor_class_name="pkg.mod.MyActor")

        assert "--actor-class-name 'pkg.mod.MyActor'" in out

    def test_custom_worker_exe(self):
        out = self.render(worker_exe="/opt/bin/my-worker")

        assert "/opt/bin/my-worker \\" in out


class TestSbatchScriptTemplate:
    def test_renders_directives_in_order(self):
        out = render_template(
            "slurm_utils:script_template",
            name="myjob",
            sbatch_args=["-A alloc", "-p gpu", "--gres=gpu:1"],
            output_file="/work/myjob-%j.out",
            script="echo hi",
        )

        assert '#SBATCH --job-name "myjob"' in out
        assert "#SBATCH -A alloc" in out
        assert "#SBATCH -p gpu" in out
        assert "#SBATCH --gres=gpu:1" in out
        assert '#SBATCH --output "/work/myjob-%j.out"' in out
        assert out.rstrip().endswith("echo hi")

    def test_no_sbatch_args(self):
        out = render_template(
            "slurm_utils:script_template",
            name="myjob",
            sbatch_args=[],
            output_file="/work/out",
            script="true",
        )

        assert out.count("#SBATCH") == 2  # job-name and output only


class TestJupyterScript:
    def test_renders_launch_command(self):
        out = render_template(
            "run_jupyter:script_template",
            setup_script="/home/me/setup.sh",
            jupyter_executable="jupyter",
        )

        assert ". '/home/me/setup.sh'" in out
        assert "'jupyter' lab" in out
        assert '--ip "$HOST" --port "$PORT"' in out
        assert "--no-browser" in out
