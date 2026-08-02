# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

All user documentation lives under `docs/`, not in `README.md`
— the README is a landing page
(what the library is, requirements, install, links).
`docs/howto-use-slurm-workflows.md` is the user guide
and the single home for usage docs:
concepts, quick start, actors, `is_batch_worker`,
running the task-queue server, and troubleshooting.
`docs/api-reference.md` lists every public name, argument and return type.
`docs/bayesian-optimization-using-botorch.md` covers the botorch optimizer in full.
`docs/howto-run-tests.md` covers the suite:
what is mocked, what is real,
and the reasoning behind the trickier tests.
This file covers only what those do not.

When adding user-facing documentation,
put it in the how-to
(or a new doc under `docs/` linked from the README's Documentation list)
— not in `README.md`.

## What this is

`slurm-workflows` is a Python library (>=3.12)
of helpers for running work on Slurm HPC clusters.
It does two things, both covered by the `docs/` guides:
the pilot-job executor and the batch Bayesian optimizer built on it.

## Commands

There is **no CI** in this repo, so nothing runs these for you.
Install and test commands are at the top of `docs/howto-run-tests.md`.

```sh
pip install -ve .[test,dev]
black src tests examples    # format
pyright                     # type-check
pytest                      # test suite
```

`black` and `pyright` are configured in `pyproject.toml`
(`[tool.black]` pins `target-version` to the `requires-python` floor
so formatting does not drift with the running interpreter;
`[tool.pyright]` sets the include paths and `pythonVersion`),
so both are run bare — do not pass paths to `pyright`
or it will ignore that config.

`pyproject.toml` defines one console entry point, `slurm-pilot-worker`.
It is internal: generated sbatch scripts invoke it on the compute nodes,
users never call it directly.

Deploy to clusters with `cpush`
(see `.cpush.json5` for the `rivanna` / `ivy-hip-tricr-2` remotes).

## Where things live

| Module | Holds |
| --- | --- |
| `slurm_pilot_executor.py` | `SlurmPilotExecutor` (coordinator, login node) and `WorkerGroup` |
| `slurm_pilot_worker.py` | `PilotWorkerProcess` (runs inside Slurm jobs) and the `slurm-pilot-worker` CLI |
| `slurm_utils.py` | `sbatch` / `squeue` / `scancel` wrappers, `get_clean_environ()` |
| `bayes_opt_botorch.py` | `BayesOptBotorch` and the `ParameterRange` types |
| `templates/` | Jinja templates and their loader |
| `utils.py` | `RemoteExecutionError`, id and logging helpers |

The coordinator and workers never talk to each other directly
— only through the `ds-service` server,
via `DsServiceClient` from the external `ds-service-client` package.
`DsServiceServer` (same package) can launch a local server process,
but the executor is always given the address explicitly.

## Invariants

Things that are easy to break and quiet when broken.

### Task flow

`as_completed` / `wait` poll `task_get_status`
with every still-pending id in one batched call.
**Status and output are separate RPCs**
— `task_get_status` returns only states,
so each finished task then needs its own `task_get_output`.
`setup_script` is inlined verbatim
into the generated worker script (`{{ setup_script }}`);
nothing validates it.

### Templates (`templates/`)

Sbatch and worker shell scripts are generated from Jinja2 templates
using a **custom multi-template-per-file format**:
each `.jinja` file holds several named templates
delimited by `{#- name: "..." -#}` JSON5 headers,
parsed by `templates/__init__.py`.
Templates are addressed as `"<file_prefix>:<name>"`
(e.g. `"slurm_pilot:worker_script"`).
`render_template` carries `@overload` signatures
documenting each template's required kwargs
— **keep those overloads in sync when changing template variables**
(the environment uses `StrictUndefined`, so a missing var is a hard error).

### Batch Bayesian optimization (`bayes_opt_botorch.py`)

- **The model is fit to `-f`**, and `best_f` is `max(-v)`
  — the same negated space.
  botorch maximizes and this minimizes;
  get it backwards and the search quietly walks uphill instead of failing.
- **`unit_points` holds the point actually evaluated**,
  re-standardized *after* rounding — never the continuous proposal.
  Otherwise the GP is told about a location the objective never ran at.
- **One acquisition, one `optimize_acqf` call per round**,
  asking for the whole batch.
  `qLogNoisyExpectedImprovement` takes `X_baseline`
  — every point measured so far — rather than a `best_f` scalar,
  so that argument has to be the current `unit_points`, not a stale copy.
- **Ask for the batch jointly, never `sequential=True`.**
  It is the usual advice for large batches and it is wrong here: measured
  10–15x *slower* on a low-dimensional space, because the greedy path pays
  the restart cost once per point instead of once per batch.
- **Ranges clamp in `unstandardize`**,
  because `optimize_acqf` can return a point a hair outside the bounds.
- **Keep this module out of the package `__init__.py`**,
  so `import slurm_workflows` works without botorch installed.
- `NUM_RESTARTS` / `RAW_SAMPLES` are module-level, and `optimize_acqf` /
  `fit_gpytorch_mll` / `qLogNoisyExpectedImprovement` are called through
  module globals;
  the tests monkeypatch those to assert what was asked for,
  without paying for a real acquisition optimization.
- **`test_search_moves_toward_the_minimum` asserts the *median* search
  point**, not the max and not `best_point()`. Both of those look like
  better guards and neither works:
  qLogNEI explores away from the incumbent
  so the max hits 1.0 on correct runs,
  and exploration alone lands near the minimum
  so `best_point()` passes even with the sign flipped.

### Slurm interaction (`slurm_utils.py`)

- `is_batch_worker=False` wraps the worker script in `srun`
  and passes `--output <work_dir>/<name>-%j-%t.out`.
  That is why the `worker_sbatch_script` template takes `name` and `work_dir`.
- **The worker process must not redirect `sys.stdout` / `sys.stderr`.**
  Slurm writes those files itself via `--output`,
  and `logging.basicConfig` leaves the streams on the inherited handles.
  Redirecting them again would leave the Slurm-written files empty.

## Conventions

- **After changing any Python, run `black` then `pyright`, then `pytest`.**
  All three must be clean before the change is done
  — `black` reports "left unchanged", `pyright` reports "0 errors",
  `pytest` passes.
  Run `black` first: it rewrites lines,
  so type-checking before formatting can report positions that no longer exist.
  Neither tool is advisory here.
  If `pyright` objects to a deliberate test double,
  say so with a `cast` and a comment explaining why the double is sufficient
  (see `as_executor` in `tests/test_bayes_opt_botorch.py`)
  rather than silencing it with a bare `# type: ignore`.
  If it objects to something in `src/`, prefer fixing the annotation
  — the `SearchSpace = Mapping[...]` alias exists because `dict[...]`
  is invariant and made a correct call fail to type-check.
- **Prose uses semantic line breaks.**
  Break at clause boundaries, not at a column limit:
  start a new line after each sentence,
  and at punctuation that already separates clauses (`.` `;` `:` `,` `--`)
  or before a conjunction or preposition that opens a new phrase.
  Never end a line mid-phrase
  — on an article, conjunction, preposition or auxiliary —
  which is what fixed-width wrapping produces.
  The result is a ragged right margin, and that is intended:
  a diff then shows only the clause that actually changed
  instead of a whole reflowed paragraph.
  Keep lines under the usual limit as a ceiling, not a target.

  ```python
  # Wrong -- wrapped at a column, breaking mid-phrase:
  # The seed is the only thing that decides the design. The name is for
  # progress bars and error messages.

  # Right -- one clause per line:
  # The seed is the only thing that decides the design.
  # The name is for progress bars and error messages.
  ```

  This governs `#` comment blocks, docstring prose,
  and every Markdown file in the repo — `README.md`, `docs/*.md`, and this file.
  Exempt: anything whose line structure is already meaningful
  — code inside fences, Markdown tables, headings,
  and ASCII section banners (`# ---- name ----`).
- Cleanup is per-class; there is no shared base class for it.
- Version is derived by `setuptools_scm` from git tags (fallback `1.0.0-dev`).
