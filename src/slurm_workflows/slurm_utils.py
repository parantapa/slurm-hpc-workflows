"""Manage Slurm Jobs."""

import os
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass
from functools import cache

from .templates import render_template

COMMAND_TIMEOUT = 120
SBATCH_OUTPUT_REGEX = re.compile(r"Submitted batch job (?P<id>\S*)")

SBATCH_EXE = "sbatch"
SQUEUE_EXE = "squeue"
SCANCEL_EXE = "scancel"


@cache
def get_clean_environ() -> dict[str, str]:
    """Create environment dict without SLURM set variables.

    This is an issue when submitting Slurm jobs from within Slurm jobs.
    """
    sanitized_env: dict[str, str] = {}
    for k, v in os.environ.items():
        if (
            k.startswith("PMI_")
            or k.startswith("SLURM_")
            or k.startswith("SLURMD_")
            or k.startswith("SRUN_")
        ):
            continue

        sanitized_env[k] = v

    return sanitized_env


def get_running_jobids() -> set[int]:
    """Get the running Slurm job IDs for the given Slurm user."""
    cmd = [SQUEUE_EXE, "--all", "--me", "--noheader", "--format", "%A"]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
    )
    job_ids = proc.stdout.strip().split()
    job_ids = set(int(j) for j in job_ids)
    return job_ids


def cancel_jobs(
    job_ids: list[int],
    term: bool = False,
    batch: bool = False,
    full: bool = False,
):
    """Run scancel command for the given job ids."""
    if not job_ids:
        return

    cmd = [SCANCEL_EXE]
    if term:
        cmd.append("--signal=TERM")
    if batch:
        cmd.append("--batch")
    if full:
        cmd.append("--full")
    cmd.extend([str(id) for id in job_ids])

    subprocess.run(
        cmd, capture_output=True, check=True, text=True, timeout=COMMAND_TIMEOUT
    )


@dataclass
class SlurmJob:
    """A submitted Slurm job."""

    name: str
    sbatch_args: list[str]
    script: str

    job_id: int
    output_file: Path


def submit_sbatch_job(
    name: str,
    sbatch_args: list[str],
    script: str,
    work_dir: Path,
) -> SlurmJob:
    """Submit a sbatch job."""
    # Figure out the output and error file names.
    output_file = str(work_dir / f"{name}-%j.out")

    # Create the sbatch script
    script_path = work_dir / f"{name}.sbatch"
    script_text = render_template(
        "slurm_utils:script_template",
        name=name,
        sbatch_args=sbatch_args,
        script=script,
        output_file=output_file,
    )
    script_path.write_text(script_text)
    os.chmod(script_path, mode=0o755)

    # Run sbatch
    proc = subprocess.run(
        [SBATCH_EXE, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
        env=get_clean_environ(),
    )

    # Extract job id.
    # Searched for rather than matched at the start:
    # a site that prints a banner or a warning on stdout
    # would otherwise turn a successful submission into a parse failure.
    match = SBATCH_OUTPUT_REGEX.search(proc.stdout)
    if match is None:
        raise RuntimeError("Failed to parse sbatch output", proc, match)
    job_id = match.group("id")
    job_id = int(job_id)

    # Resolve the file names
    output_file = Path(output_file.replace("%j", str(job_id)))

    return SlurmJob(
        name=name,
        sbatch_args=sbatch_args,
        script=script,
        job_id=job_id,
        output_file=output_file,
    )
