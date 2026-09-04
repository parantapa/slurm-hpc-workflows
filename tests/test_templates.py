"""Tests for the multi-template-per-file Jinja loader and the templates."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import jinja2
import pytest

from slurm_workflows.templates import line_col_from_pos, parse_file, render_template


def run_sbatch_script(
    script: str, tmp_path: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    """Run a rendered sbatch body against a stub `srun`.

    Which of the two `srun` invocations runs is decided by the shell,
    not by anything Python can see in the rendered text,
    so the only test that can be right about it is one that runs the shell.

    Both streams come back, because the script uses both:
    the stub echoes its command line to stdout,
    among whatever the script itself echoed on the way there
    --- so callers pick that line back out with `srun_lines`
    rather than reading stdout whole ---
    while `set -x` traces to stderr.

    The environment is built from scratch rather than inherited:
    that pins the SLURM variables to exactly what a case sets,
    including when the suite itself is run from inside a Slurm job.
    """

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    srun = bin_dir / "srun"
    srun.write_text('#!/bin/bash\necho "srun $*"\n')
    srun.chmod(0o755)

    script_path = tmp_path / "job.sbatch"
    script_path.write_text(script)

    proc = subprocess.run(
        ["bash", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", **env},
    )
    return proc


class TestParseFile:
    """The multi-template-per-file format itself.

    A file is a run of `{#- <json5 header> -#}` markers,
    each followed by the body of the template it names.
    Nothing else in the file is addressable,
    which is what the cases below pin down.
    """

    def write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "sample.jinja"
        path.write_text(text)
        return path

    def test_reads_every_template_in_the_file(self, tmp_path: Path):
        path = self.write(
            tmp_path,
            '{#- name: "first" -#}\nbody one\n{#- name: "second" -#}\nbody two\n',
        )

        templates = parse_file("sample", path)

        assert sorted(templates) == ["sample:first", "sample:second"]
        assert templates["sample:first"].source == "body one"
        assert templates["sample:second"].source == "body two"

    def test_the_prefix_is_the_callers_not_the_filename(self, tmp_path: Path):
        """`load_template` derives it from the name being looked up.

        The file it then reads is `<prefix>.jinja`,
        so the two agree in practice
        --- but `parse_file` takes the caller's word for it.
        """
        path = self.write(tmp_path, '{#- name: "only" -#}\nbody\n')

        assert list(parse_file("other", path)) == ["other:only"]
        assert list(parse_file("sample", path)) == ["sample:only"]

    def test_a_body_is_stripped(self, tmp_path: Path):
        path = self.write(tmp_path, '{#- name: "only" -#}\n\n\n  body  \n\n')

        assert parse_file("sample", path)["sample:only"].source == "body"

    def test_text_before_the_first_header_is_ignored(self, tmp_path: Path):
        path = self.write(tmp_path, 'a preamble\n{#- name: "only" -#}\nbody\n')

        assert parse_file("sample", path)["sample:only"].source == "body"

    def test_a_file_with_no_headers_yields_nothing(self, tmp_path: Path):
        assert parse_file("sample", self.write(tmp_path, "just text\n")) == {}

    def test_an_empty_file_yields_nothing(self, tmp_path: Path):
        assert parse_file("sample", self.write(tmp_path, "")) == {}

    def test_a_repeated_name_keeps_the_last(self, tmp_path: Path):
        """Not an error today. Pinned so a change to that is a deliberate one."""
        path = self.write(
            tmp_path,
            '{#- name: "dup" -#}\nfirst\n{#- name: "dup" -#}\nsecond\n',
        )

        assert parse_file("sample", path)["sample:dup"].source == "second"

    def test_an_unterminated_header_is_reported(self, tmp_path: Path):
        path = self.write(tmp_path, '{#- name: "only"\nbody with no header end\n')

        with pytest.raises(ValueError, match="Unable to find end of header"):
            parse_file("sample", path)

    def test_a_malformed_header_is_reported(self, tmp_path: Path):
        path = self.write(tmp_path, "{#- name: -#}\nbody\n")

        with pytest.raises(Exception) as excinfo:
            parse_file("sample", path)

        assert "Failed to parse template file" in "".join(excinfo.value.__notes__)

    def test_a_header_without_a_name_is_reported(self, tmp_path: Path):
        path = self.write(tmp_path, '{#- title: "only" -#}\nbody\n')

        with pytest.raises(KeyError) as excinfo:
            parse_file("sample", path)

        assert "Failed to parse template file" in "".join(excinfo.value.__notes__)

    def test_the_error_carries_a_file_position(self, tmp_path: Path):
        path = self.write(tmp_path, '{#- name: "ok" -#}\nbody\n{#- oops -#}\n')

        with pytest.raises(Exception) as excinfo:
            parse_file("sample", path)

        notes = "".join(excinfo.value.__notes__)
        assert str(path) in notes

    def test_a_jinja_comment_in_a_body_is_read_as_the_next_header(self, tmp_path: Path):
        """A limit of the format, not a bug to be fixed by accident.

        `{#-` is how a template body ends,
        so a body cannot also contain a whitespace-trimming Jinja comment
        --- the parser takes it for the header of the next template.
        Use `{#` without the dash for a comment inside a body.
        """
        path = self.write(
            tmp_path,
            '{#- name: "only" -#}\nbody\n{#- a note -#}\nmore body\n',
        )

        with pytest.raises(Exception):
            parse_file("sample", path)


class TestLineColFromPos:
    def test_the_first_character_is_line_one_column_one(self):
        assert line_col_from_pos("hello", 0) == (1, 1)

    def test_counts_lines_and_columns(self):
        text = "one\ntwo\nthree"

        assert line_col_from_pos(text, text.index("two")) == (2, 1)
        assert line_col_from_pos(text, text.index("three") + 2) == (3, 3)

    def test_an_empty_text_is_the_start(self):
        assert line_col_from_pos("", 0) == (1, 1)


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
    def render(
        self,
        name: str = "slurm_pilot_worker.cpu.0",
        work_dir: str = "/scratch/work",
        is_batch_worker: bool = False,
        worker_script_path: str = "/path/to/worker.sh",
    ) -> str:
        # Spelled out rather than gathered into `**overrides`:
        # `render_template` is a set of `@overload`s keyed on the template name,
        # and a `**kwargs` dict erases the per-argument types they match on.
        return render_template(
            "slurm_pilot:worker_sbatch_script",
            name=name,
            work_dir=work_dir,
            is_batch_worker=is_batch_worker,
            worker_script_path=worker_script_path,
        )

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

    def test_both_srun_forms_are_rendered(self, srun_lines):
        """The script carries both; the shell picks between them at run time."""
        out = self.render()

        plain, per_task = srun_lines(out)
        assert plain == "srun /bin/bash '/path/to/worker.sh'"
        assert per_task == (
            "srun --output '/scratch/work/slurm_pilot_worker.cpu.0-%j-%t.out' "
            "/bin/bash '/path/to/worker.sh'"
        )

    def test_each_srun_task_gets_its_own_output_file(self, srun_lines):
        """`srun` fans out over every task in the allocation.

        Without a per-task --output
        they would all interleave into the one batch output file,
        so the pattern has to carry both the job id and the task id.
        """
        out = self.render()

        _, per_task = srun_lines(out)
        assert "%j" in per_task and "%t" in per_task

    def test_the_output_pattern_lands_in_the_work_dir(self):
        out = self.render(work_dir="/some/other/dir")

        assert "--output '/some/other/dir/" in out


class TestOutputRedirectByTaskCount:
    """Which `srun` a job runs, established by running the shell.

    A job of exactly one task writes to the batch job's own output file:
    it has no second task to interleave with,
    so a per-task file would only duplicate what is already there.
    Every other allocation --- and anything that is not a Slurm job at all ---
    keeps the per-task files.

    Each case names the `sbatch` options it stands for,
    and sets the variables Slurm would have set for them.
    """

    def render(self) -> str:
        return render_template(
            "slurm_pilot:worker_sbatch_script",
            name="slurm_pilot_worker.cpu.0",
            work_dir="/scratch/work",
            is_batch_worker=False,
            worker_script_path="/path/to/worker.sh",
        )

    @pytest.mark.parametrize(
        "env",
        [
            pytest.param(
                {"SLURM_NTASKS": "1", "SLURM_JOB_NUM_NODES": "1"},
                id="ntasks-1",
            ),
            pytest.param(
                {"SLURM_NTASKS": "1", "SLURM_JOB_NUM_NODES": "2"},
                id="ntasks-1-over-two-nodes",
            ),
            pytest.param(
                {"SLURM_JOB_NUM_NODES": "1"},
                id="nodes-1-with-no-task-option",
            ),
        ],
    )
    def test_a_single_task_job_writes_to_the_batch_file(
        self, tmp_path: Path, env: dict[str, str], srun_lines
    ):
        # The last case names no task count, so SLURM_NTASKS is unset;
        # one task per node is then the default,
        # which makes the one node one task.
        out = run_sbatch_script(self.render(), tmp_path, **env).stdout

        # One line, because only one of the two branches may run.
        assert srun_lines(out) == ["srun /bin/bash /path/to/worker.sh"]

    @pytest.mark.parametrize(
        "env",
        [
            pytest.param(
                {"SLURM_NTASKS": "8", "SLURM_JOB_NUM_NODES": "4"},
                id="four-nodes-two-tasks-each",
            ),
            pytest.param(
                {"SLURM_NTASKS": "4", "SLURM_JOB_NUM_NODES": "4"},
                id="four-nodes-one-task-each",
            ),
            pytest.param(
                {"SLURM_NTASKS": "4", "SLURM_JOB_NUM_NODES": "1"},
                id="one-node-four-tasks",
            ),
            pytest.param(
                {"SLURM_JOB_NUM_NODES": "4"},
                id="four-nodes-with-no-task-option",
            ),
            pytest.param({}, id="not-a-slurm-job"),
        ],
    )
    def test_every_other_job_keeps_its_per_task_files(
        self, tmp_path: Path, env: dict[str, str], srun_lines
    ):
        # One task *per node* is not one task:
        # `--nodes=4 --ntasks-per-node=1` is four workers on four nodes,
        # and dropping --output there
        # would interleave them into the single batch file.
        out = run_sbatch_script(self.render(), tmp_path, **env).stdout

        assert srun_lines(out) == [
            "srun --output /scratch/work/slurm_pilot_worker.cpu.0-%j-%t.out "
            "/bin/bash /path/to/worker.sh"
        ]

    def test_the_task_count_is_never_overridden_by_the_node_count(
        self, tmp_path: Path, srun_lines
    ):
        """SLURM_NTASKS is the job's task count, so nothing else gets a vote.

        Slurm sets it for `--ntasks` *and* for any `--ntasks-per-*` option,
        which is what makes it the whole answer whenever it is there;
        the node count only stands in when it is absent.
        """
        out = run_sbatch_script(
            self.render(),
            tmp_path,
            SLURM_NTASKS="4",
            SLURM_JOB_NUM_NODES="1",
        ).stdout

        (command,) = srun_lines(out)
        assert "--output" in command

    def test_the_count_it_decided_on_is_echoed(self, tmp_path: Path):
        """The batch output file should say why the branch was taken.

        Which file a worker's log went to is otherwise
        something you can only work out by re-reading the sbatch script
        and guessing what Slurm set.
        """
        proc = run_sbatch_script(
            self.render(),
            tmp_path,
            SLURM_NTASKS="4",
            SLURM_JOB_NUM_NODES="1",
        )

        assert "Num tasks: 4" in proc.stdout

    @pytest.mark.parametrize(
        "env, traced",
        [
            pytest.param({"SLURM_NTASKS": "1"}, "+ srun /bin/bash", id="single-task"),
            pytest.param({"SLURM_NTASKS": "4"}, "+ srun --output", id="several-tasks"),
        ],
    )
    def test_the_srun_command_is_traced(
        self, tmp_path: Path, env: dict[str, str], traced: str
    ):
        """`set -x` is what puts the command line in the batch output file.

        That file is all there is to read
        when a job dies before the worker script gets as far as its own log,
        so tracing has to be on before `srun` runs --- in either branch.
        """
        proc = run_sbatch_script(self.render(), tmp_path, **env)

        assert traced in proc.stderr

    def test_the_guard_survives_a_strict_shell(self, tmp_path: Path, srun_lines):
        """Both variables are read with `:-`.

        Nothing sets `-u` on the generated sbatch script today,
        so an unset variable would merely expand empty;
        the default is only worth anything if it cannot become an error later.
        """
        out = run_sbatch_script("set -u\n" + self.render(), tmp_path).stdout

        (command,) = srun_lines(out)
        assert "--output" in command


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
        out = self.render(setup_script='export A="x y"\nexport B=$HOME/z\n# a comment')

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
