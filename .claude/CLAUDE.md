# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`slurm-workflows` is a Python library (>=3.12) providing helper utilities to run
work on Slurm HPC clusters. Its three capabilities:

1. **Pilot-job task execution** — a `concurrent.futures`-style executor
   (`SlurmPilotExecutor`) that launches long-lived Slurm "pilot" jobs and
   dispatches Python callables to them via a task queue.
2. **Jupyter launcher** — the `run-jupyter` CLI submits a Jupyter Lab server as
   a Slurm batch job.
3. **Optuna storage** — `optuna_storage.py` backs an Optuna study with a
   ds-service journal, so pilot workers can share one study.

## Commands

```sh
pip install -ve .[test]    # editable install (setuptools + setuptools_scm)
pytest                     # test suite (see tests/README.md)
```

There is **no linter config or CI** in this repo. `pyproject.toml`
defines two console entry points:

- `slurm-pilot-worker` — internal; invoked by generated sbatch scripts on
  compute nodes, not by users directly.
- `run-jupyter --setup-script <path> -- <sbatch args...>` — user-facing. Note the
  `--setup-script` flag is **required** (the `docs/rivanna-setup.md` walkthrough
  predates it and omits it).

Deploy to clusters with `cpush` (see `.cpush.json5` for the `rivanna` /
`ivy-hip-tricr-2` remotes).

## Architecture

The system has three cooperating processes, decoupled through an external task
queue:

- **Coordinator** (`SlurmPilotExecutor`, runs on the login node): defines worker
  groups, submits/cancels pilot Slurm jobs to scale them, and submits tasks.
- **ds-service** (external `ds-service-client` dependency): a separate task-queue
  server holding tasks keyed by named queues. The coordinator and workers only
  talk to each other through this server's `Client` — they never communicate
  directly. `ds_service.py` can launch a local `ds-service` process, but the
  server address is passed to the executor explicitly.
- **Pilot workers** (`PilotWorkerProcess` in `slurm_pilot_worker.py`, run inside
  Slurm jobs on compute nodes): loop calling `client.task_get(...)`, execute the
  task, and post results back with `client.task_done(...)`.

### Task flow

1. `executor.define_worker(name, sbatch_args, setup_script, ...)` registers a
   `WorkerGroup` (does not launch anything). `setup_script` is optional and is
   the **body** of the setup snippet, not a path — it is inlined verbatim into
   each generated worker script (omitted/`None` becomes `""`, like
   `actor_class_name`). `define_worker` rejects a value that is a path to an
   existing file, since that would silently execute the script instead of
   sourcing it.
2. `executor.scale_workers(name, count)` submits or cancels Slurm jobs to reach
   `count` workers for that group. Each worker gets a rendered sbatch script.
3. `executor.submit(queue, fn, *args, **kwargs)` cloudpickles the callable +
   inputs and enqueues a `Task` on the named queue(s).
4. A worker pulls it, `cloudpickle.loads` the function, runs it, and pushes the
   cloudpickled return value back.
5. `executor.as_completed(tasks)` / `wait(tasks)` poll `task_get_status` with
   all still-pending ids in one batched call, then fetch each finished task's
   result with `task_get_output` (tqdm-wrapped). Status and output are separate
   RPCs — `task_get_status` returns only states.

### Serialization & remote errors

Everything crossing the process boundary (functions, args, return values) is
**cloudpickled**, so tasks can be closures/lambdas. Exceptions raised inside a
worker are **not** re-raised in the coordinator — they are caught, wrapped in a
`RemoteExecutionError(error, error_id)`, and returned as the task *output*. Use
`check_for_error(tasks)` (exported from the package root) to find failed tasks;
`error_id` correlates with the worker's log file.

### Actors (stateful workers)

If `define_worker(actor_class_name="pkg.mod.MyClass")` is set, each worker
instantiates that class once at startup. Then `submit(queue, "method_name", ...)`
passes a **string** as `fn`; the worker resolves it to a bound method on the
actor instance. This is how per-worker state (loaded models, DB handles) is kept
warm across tasks. Actor classes may define a `close()` for cleanup.

### Templates (`templates/`)

Sbatch and worker shell scripts are generated from Jinja2 templates using a
**custom multi-template-per-file format**: each `.jinja` file holds several named
templates delimited by `{#- name: "..." -#}` JSON5 headers, parsed by
`templates/__init__.py`. Templates are addressed as `"<file_prefix>:<name>"`
(e.g. `"slurm_pilot:worker_script"`). `render_template` carries `@overload`
signatures documenting each template's required kwargs — **keep those overloads
in sync when changing template variables** (the environment uses
`StrictUndefined`, so a missing var is a hard error).

### Optuna storage (`optuna_storage.py`)

`DsServiceJournalBackend` implements Optuna's `BaseJournalBackend` /
`BaseJournalSnapshot` on top of the ds-service Journal: one journal key
(`<prefix>:log`) holds the whole log, and an entry's journal index *is* its
Optuna log number. Snapshots go in the map under `<prefix>:snapshot`.
`create_optuna_storage(...)` wraps it in an `optuna.storages.JournalStorage`.

Two things to preserve when changing it:

- **No lock object is needed** (unlike `JournalFileBackend`) — ds-service
  serializes journal appends and reads server-side.
- **It must stay picklable.** `__getstate__` drops the `Client` and
  `_get_client()` reconnects lazily, so a study can be cloudpickled into a
  task and reconnect on the compute node. That is the whole point of the
  module; a `Client` held eagerly in `__init__` would break it.

Optuna is an *optional* dependency (`[optuna]` extra) — keep this module out of
the package `__init__.py` so `import slurm_workflows` works without it.

### Slurm interaction (`slurm_utils.py`)

Wraps `sbatch` / `squeue` / `scancel` subprocess calls. Important detail:
`get_clean_environ()` strips all `SLURM_*` / `PMI_*` / `SRUN_*` env vars before
calling `sbatch`, so that submitting jobs *from within* a Slurm job works
correctly.

- `is_batch_worker=True` → worker script is sourced directly (single task per
  node); `False` → wrapped in `srun` (fans out across all tasks in the
  allocation).

## Conventions

- Cleanup uses the `Closeable` ABC (`utils.py`) — context-manager +
  `__del__`-based. `SlurmPilotExecutor.close()`/`stop()` cancel all pilot jobs.
- Logs go to files under a per-run work dir derived from platformdirs
  (`XDG_*`-driven on clusters; see `docs/rivanna-setup.md`).
- Version is derived by `setuptools_scm` from git tags (fallback `1.0.0-dev`).
