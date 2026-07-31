# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read `README.md` first — it documents what the library does and how it is used:
the executor API, actors, the worker environment, and log layout /
troubleshooting. `docs/bayesian-optimization-using-botorch.md` covers the
botorch optimizer in full. `tests/README.md` covers the suite: what is mocked,
what is real, and the reasoning behind the trickier tests. This file covers
only what those do not.

## What this is

`slurm-workflows` is a Python library (>=3.12) of helpers for running work on
Slurm HPC clusters. Alongside the pilot executor and the botorch optimizer that
`README.md` describes, there is a third entry point neither README mentions:
**`run-jupyter`**, a CLI that submits a Jupyter Lab server as a Slurm batch job.

## Commands

There is **no linter config or CI** in this repo. Install and test commands are
at the top of `tests/README.md`.

`pyproject.toml` defines two console entry points:

- `slurm-pilot-worker` — internal; invoked by generated sbatch scripts on
  compute nodes, not by users directly.
- `run-jupyter --setup-script <path> -- <sbatch args...>` — user-facing. The
  `--setup-script` flag is **required**, and here it really is a path to an
  existing file (`click.Path(exists=True)`) — unlike `define_worker`'s
  `setup_script`, which takes the snippet's body.

Deploy to clusters with `cpush` (see `.cpush.json5` for the `rivanna` /
`ivy-hip-tricr-2` remotes).

## Where things live

| Module | Holds |
| --- | --- |
| `slurm_pilot_executor.py` | `SlurmPilotExecutor` (coordinator, login node) and `WorkerGroup` |
| `slurm_pilot_worker.py` | `PilotWorkerProcess` (runs inside Slurm jobs) and the `slurm-pilot-worker` CLI |
| `slurm_utils.py` | `sbatch` / `squeue` / `scancel` wrappers, `get_clean_environ()` |
| `bayes_opt_botorch.py` | `BayesOptBotorch` and the `ParameterRange` types |
| `templates/` | Jinja templates and their loader |
| `utils.py` | `RemoteExecutionError`, id and logging helpers |

The coordinator and workers never talk to each other directly — only through
the `ds-service` server, via `DsServiceClient` from the external
`ds-service-client` package. `DsServiceServer` (same package) can launch a
local server process, but the executor is always given the address explicitly.

## Invariants

Things that are easy to break and quiet when broken.

### Task flow

`as_completed` / `wait` poll `task_get_status` with every still-pending id in
one batched call. **Status and output are separate RPCs** — `task_get_status`
returns only states, so each finished task then needs its own
`task_get_output`. `setup_script` is inlined verbatim into the generated
worker script (`{{ setup_script }}`); nothing validates it.

### Templates (`templates/`)

Sbatch and worker shell scripts are generated from Jinja2 templates using a
**custom multi-template-per-file format**: each `.jinja` file holds several named
templates delimited by `{#- name: "..." -#}` JSON5 headers, parsed by
`templates/__init__.py`. Templates are addressed as `"<file_prefix>:<name>"`
(e.g. `"slurm_pilot:worker_script"`). `render_template` carries `@overload`
signatures documenting each template's required kwargs — **keep those overloads
in sync when changing template variables** (the environment uses
`StrictUndefined`, so a missing var is a hard error).

### Batch Bayesian optimization (`bayes_opt_botorch.py`)

- **The model is fit to `-f`**, and `best_f` is `max(-v)` — the same negated
  space. botorch maximizes and this minimizes; get it backwards and the search
  quietly walks uphill instead of failing.
- **`unit_points` holds the point actually evaluated**, re-standardized *after*
  rounding — never the continuous proposal. Otherwise the GP is told about a
  location the objective never ran at.
- **The batch split is exactly** `num_ei = batch - batch // 2` and
  `num_pi = batch // 2`, and a `q == 0` acquisition is skipped — zero is not a
  legal batch size for `optimize_acqf`.
- **Ranges clamp in `unstandardize`**, because `optimize_acqf` can return a
  point a hair outside the bounds.
- **Keep this module out of the package `__init__.py`**, so
  `import slurm_workflows` works without botorch installed.
- `NUM_RESTARTS` / `RAW_SAMPLES` are module-level, and `optimize_acqf` /
  `fit_gpytorch_mll` are called through module globals; the tests monkeypatch
  those to assert the batch split without paying for a real acquisition
  optimization.

### Slurm interaction (`slurm_utils.py`)

- `is_batch_worker=False` wraps the worker script in `srun` and passes
  `--output <work_dir>/<name>-%j-%t.out`. That is why the
  `worker_sbatch_script` template takes `name` and `work_dir`.
- **The worker process must not redirect `sys.stdout` / `sys.stderr`.** Slurm
  writes those files itself via `--output`, and `logging.basicConfig` leaves the
  streams on the inherited handles. Redirecting them again would leave the
  Slurm-written files empty.

## Conventions

- Cleanup is per-class; there is no shared base class for it.
- Version is derived by `setuptools_scm` from git tags (fallback `1.0.0-dev`).
