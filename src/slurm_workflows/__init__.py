"""HPC workflow helpers for Slurm clusters.

`bayes_opt_botorch` is deliberately not imported here,
so that `import slurm_workflows` works without botorch installed.
"""

from .slurm_pilot_executor import SlurmPilotExecutor, check_for_error

__all__ = ["SlurmPilotExecutor", "check_for_error"]
