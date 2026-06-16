"""Start a Jupyter Lab instance."""

import subprocess
from datetime import datetime
from pathlib import Path

import click
import platformdirs

from .templates import render_template
from .slurm_utils import submit_sbatch_job

JUPYTER_EXE = "jupyter"


@click.command()
@click.option(
    "--setup-script",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to setup script.",
)
@click.argument("sbatch-args", nargs=-1)
def run_jupyter(
    sbatch_args: list[str],
    setup_script: Path,
):
    """Start a Jupyter Lab instance."""
    print("Sbatch args: ", " ".join(sbatch_args))

    name = "jupyter"

    script = render_template(
        "run_jupyter:script_template",
        setup_script=setup_script,
        jupyter_executable=JUPYTER_EXE,
    )

    now = datetime.now().isoformat()
    work_dir = platformdirs.user_cache_path(appname=f"run-jupyter") / now
    work_dir.mkdir(parents=True)

    try:
        job = submit_sbatch_job(
            name=name, sbatch_args=sbatch_args, script=script, work_dir=work_dir
        )

        print(f"Job ID: {job.job_id}")
        print(f"Output file: {job.output_file!s}")
    except subprocess.CalledProcessError as cp:
        print(f"Failed to submit job: {cp.returncode}")
        if cp.stdout.strip():
            print(cp.stdout)
        if cp.stderr.strip():
            print(cp.stderr)
