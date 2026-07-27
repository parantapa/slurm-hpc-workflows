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
    def test_batch_worker_sources_script_directly(self):
        out = render_template(
            "slurm_pilot:worker_sbatch_script",
            is_batch_worker=True,
            worker_script_path="/path/to/worker.sh",
        )

        assert ". '/path/to/worker.sh'" in out
        assert "srun" not in out

    def test_non_batch_worker_is_wrapped_in_srun(self):
        out = render_template(
            "slurm_pilot:worker_sbatch_script",
            is_batch_worker=False,
            worker_script_path="/path/to/worker.sh",
        )

        assert "srun /bin/bash '/path/to/worker.sh'" in out


class TestWorkerScript:
    def render(self, **overrides):
        kwargs = dict(
            worker_exe="slurm-pilot-worker",
            setup_script="/home/me/setup.sh",
            group="cpu",
            name="slurm_pilot_worker.cpu.0",
            actor_class_name="",
            server_address="10.0.0.1:5051",
            work_dir="/scratch/work",
            python_paths_json=json.dumps(["/a", "/b"]),
        )
        kwargs.update(overrides)
        return render_template("slurm_pilot:worker_script", **kwargs)

    def test_sources_profile_and_setup_script(self):
        out = self.render()

        assert ". '/etc/profile'" in out
        assert ". '/home/me/setup.sh'" in out

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
