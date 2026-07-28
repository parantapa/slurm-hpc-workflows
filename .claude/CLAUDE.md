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

### QMC sampler (`optuna_qmc_sampler.py`)

`DsServiceQMCSampler` subclasses `optuna.samplers.QMCSampler` to make it safe
distributed. It overrides exactly three things, each fixing a base-class
behaviour that only bites across processes:

- `_find_sample_id` — the base does a non-atomic read-modify-write on a study
  system attr, so concurrent workers get the same sequence position. Replaced
  with `counter_get_next_value` (atomic server-side; returns 1 first, ids are
  0-based).
- `before_trial` — negotiates one scramble seed across workers by appending to
  a journal key and reading entry 0 back (ordered appends ⇒ first writer
  wins). It must run before any suggestion, because the counter key's digest
  includes the seed.
- `infer_relative_search_space` — returns the caller's declared `search_space`
  so trial 0 is QMC, instead of waiting for the first trial to *finish*.

Also: the independent (fallback) sampler is seeded per worker, deliberately.
The base passes the QMC `seed` to it, so workers sharing a seed — which the
base requires for scrambling — draw identical fallback points.

The sampler takes the storage/backend rather than an address so its counter
cannot land on a different server or prefix than the study. The counter is not
in the journal, but ds-service is entirely in-memory, so counter and journal
are lost together and can't diverge.

The tests keep a copy of each base-class misbehaviour (`test_the_base_sampler_*`)
— if Optuna fixes one upstream, that test fails and the corresponding override
can be dropped.

### Extreme-point sampler (`optuna_extreme_point_sampler.py`)

`ExtremePointSampler` is a `BaseSampler` (not a subclass — it shares no logic
with `GridSampler`) that walks the `2**d` corners of a declared box, one per
trial, deterministically. It follows `GridSampler`'s *shape*: pick the index in
`before_trial` and record it as a trial system attr, `infer_relative_search_space`
/ `sample_relative` return `{}`, values are handed out by `sample_independent`
(the only place the distribution object is available to validate against), and
`after_trial` calls `study.stop()` when the walk is done.

What differs, and why:

- **Allocation is one `counter_get_next_value`.** `GridSampler` uses the trial
  number while it can and otherwise scans for an unvisited point and picks
  randomly; that fallback races, and it *silently skips* points (measured: 8–13
  of 32 never visited with four workers on a resumed study). The counter has no
  such path.
- **Corners are decoded from the index, never materialized** (mixed-radix
  strides in `_strides`). `GridSampler` builds `itertools.product`, which a
  60-parameter box would not survive.
- **`FAIL` is not a visited state.** A corner whose worker died is retried
  before any corner is repeated. Note a SIGKILLed worker leaves its trial
  `RUNNING` forever — journal storage has no heartbeat — so that corner is not
  recovered.
- **Params outside the declared space raise** unless an `independent_sampler`
  is passed, since sampling them would break determinism.

Two ordering constraints that are easy to break: `CORNER_ID_ATTR` must be
written *before* `SPACE_ATTR` (separate journal appends; readers filter on the
digest, so the digest must arrive last), and `study.stop()` must stay wrapped in
`try/except RuntimeError` (it is only legal inside an optimize loop, not under
ask/tell).

### Slurm interaction (`slurm_utils.py`)

Wraps `sbatch` / `squeue` / `scancel` subprocess calls. Important detail:
`get_clean_environ()` strips all `SLURM_*` / `PMI_*` / `SRUN_*` env vars before
calling `sbatch`, so that submitting jobs *from within* a Slurm job works
correctly.

- `is_batch_worker=True` → worker script is sourced directly (single task per
  node); `False` → wrapped in `srun` (fans out across all tasks in the
  allocation), with `--output <work_dir>/<name>-%j-%t.out` so each task writes
  its own file instead of all of them interleaving into the batch job's one.
  That is why the `worker_sbatch_script` template takes `name` and `work_dir`.

## Conventions

- Cleanup uses the `Closeable` ABC (`utils.py`) — context-manager +
  `__del__`-based. `SlurmPilotExecutor.close()`/`stop()` cancel all pilot jobs.
- Logs go to files under a per-run work dir derived from platformdirs
  (`XDG_*`-driven on clusters; see `docs/rivanna-setup.md`). **Slurm** writes
  the worker files, via `--output` — the worker process does not redirect
  `sys.stdout`/`sys.stderr`, and `logging.basicConfig` leaves them on the
  inherited stream. Redirecting them again would leave the Slurm files empty.
- Version is derived by `setuptools_scm` from git tags (fallback `1.0.0-dev`).
