# Developer notes

[<- back to the main README](../README.md)

Notes for people working on `slurm-workflows` itself.
Everything here is about the code;
how to *use* the library is in the guides the README indexes.

`slurm-workflows` is a Python library (>=3.12)
of helpers for running work on Slurm HPC clusters.
It does two things, both covered by those guides:
the pilot-job executor, and the batch Bayesian optimizer built on it.

## Where documentation goes

All user documentation lives under `docs/`, not in `README.md`.
The README is a landing page:
what the library is, requirements, install, links.

| Document | Covers |
| --- | --- |
| [`howto-use-slurm-workflows.md`](howto-use-slurm-workflows.md) | The user guide, and the single home for usage docs: concepts, quick start, actors, `is_batch_worker`, running the task-queue server, troubleshooting. |
| [`api-reference.md`](api-reference.md) | Every public name, argument and return type. |
| [`bayesian-optimization-using-botorch.md`](bayesian-optimization-using-botorch.md) | The botorch optimizer in full. |
| [`howto-run-tests.md`](howto-run-tests.md) | The suite: what is mocked, what is real, and the reasoning behind the trickier tests. |
| `developer-notes.md` | This file. Only what the others do not cover. |

When adding user-facing documentation,
put it in the how-to,
or in a new doc under `docs/` added to the README's documentation table.
Not in `README.md`.

## Commands

There is **no CI** in this repository, so nothing runs these for you.
The install and test commands are also at the top of
[`howto-run-tests.md`](howto-run-tests.md).

```sh
pip install -ve .[test,dev]
black src tests examples    # format
pyright                     # type-check
pytest                      # test suite
```

`black` and `pyright` are configured in `pyproject.toml`.
`[tool.black]` pins `target-version` to the `requires-python` floor,
so formatting does not drift with whichever interpreter happens to run it,
and `[tool.pyright]` sets the include paths and `pythonVersion`.
Both are therefore run bare:
**do not pass paths to `pyright`**, or it ignores that configuration.

`pyproject.toml` defines two console entry points.
`slurm-pilot-worker` is internal:
generated sbatch scripts invoke it on the compute nodes,
and users never call it directly.
`swtop` is for users, and is documented in the how-to.

Deploy to clusters with `cpush`.
See `.cpush.json5` for the `rivanna` and `ivy-hip-tricr-2` remotes.

## Where things live

| Module | Holds |
| --- | --- |
| `slurm_pilot_executor.py` | `SlurmPilotExecutor` (the coordinator, on the login node) and `WorkerGroup` |
| `slurm_pilot_worker.py` | `PilotWorkerProcess` (runs inside Slurm jobs) and the `slurm-pilot-worker` CLI |
| `slurm_utils.py` | `sbatch` / `squeue` / `scancel` wrappers, `get_clean_environ()` |
| `optimize_space_botorch.py` | `OptimizationTask` and `OptimizeSpaceBotorch`, the botorch searches |
| `search_space.py` | `SearchSpace`, the `ParameterRange` types, and the unit cube mapping |
| `explore_space.py` | `ExplorationTask` and `ExploreSpaceSobolQMC`, Sobol' sweeps with no model behind them |
| `monitors.py` | Host and cgroup sampling, and the threads that publish it |
| `swtop.py` | The `swtop` monitor: `Collector`, `render` and the CLI |
| `templates/` | Jinja templates and their loader |
| `utils.py` | `RemoteExecutionError`, id and logging helpers, `floor_power_of_two`, the progress-line formatters |

The coordinator and the workers never talk to each other directly,
only through the `ds-service` server,
via `DsServiceClient` from the external `ds-service-client` package.
`DsServiceServer` (same package) can launch a local server process,
but the executor is always given the address explicitly.

The server and the client are versioned together.
The floor is recorded in `pyproject.toml`,
and the README states the server version to match.

## Invariants

Things that are easy to break and quiet when broken.

### Task flow

`as_completed` and `wait` poll `task_get_status`
with every still-pending id in one batched call.

**Status and output are separate RPCs.**
`task_get_status` returns only states,
so each finished task then needs its own `task_get_output`.

**A task that can never run is dropped; the batch is not.**
`_starved_tasks` and `_stranded_tasks` return the subset they object to,
rather than raising, because the answer is always a subset:
a queue nobody scaled says nothing about the queues that were,
and one group's jobs reaching their walltime
says nothing about a task on another group's queue.
Abandoning the rest of `pending` loses results the server already has,
silently under `RAISE_NEVER`,
and breaks what `RAISE_AFTER_COMPLETED` promises.
The failure count in the deferred exception counts *tasks* for the same
reason: one message covers every task on a dead queue.

**What came back is recorded even when the batch failed.**
Both drivers wait with `RAISE_AFTER_COMPLETED` and then record;
on the failure path they record what returned before re-raising.
A sweep of a few thousand points must not lose all of them to one,
and `save()` is what the next run reads.

**Every failure is warned about, whatever `RaiseOnError` says.**
The warning is the part a caller cannot switch off,
because `RAISE_NEVER` otherwise loses a failure entirely:
`task.output` is the only other record, and nothing forces a caller to read it.
It goes to stderr through `tqdm.write`,
so it neither breaks the progress bar nor lands in a caller's stdout.

**Only `wait` can defer.**
`as_completed` yields results as they arrive,
so there is no point at which it has finished but the caller has not,
and `RAISE_AFTER_COMPLETED` collapses to `RAISE_ON_FIRST_ERROR` there.
`wait` therefore drives `_as_completed` itself
rather than going through `as_completed`, which would rewrite the policy.

**The poll loop must name every `TaskState` explicitly.**
Its `else` branch means "keep waiting",
so a state that falls through it waits forever
instead of reporting that the task will never finish.
`Canceled`, added in ds-service 4.0.0, is how this was found.

**`task_done` is per worker.**
The worker passes its own `worker_id`,
and the server refuses the call from any other worker,
so the id given to `task_done`
has to be the one that claimed the task in `task_get`.

**An empty queue is `NoTaskAvailable`, not `TimeoutError`.**
`task_get` answers immediately when no queue has work,
and the worker sleeps and retries on that alone.
A `TimeoutError` there means an unreachable server
and must stay distinguishable.

**`submit` sets `priority` to a *negated* wall clock.**
ds-service dispatches the highest priority first,
so a timestamp that rises with time serves the newest task first
and leaves the oldest until last:
a queue that runs backwards and never says so.
It is a wall clock rather than `perf_counter`
because two executors on one queue have to be comparable,
and `perf_counter`'s zero is the start of whichever process asked.

**Actor constructor arguments travel through the key value store.**
`define_worker` cloudpickles `actor_class_args` and `actor_class_kwargs`
into `actor_class_args:<group>` and `actor_class_kwargs:<group>`,
and `PilotWorkerProcess.__init__` reads them back under the same names.
The two sides agree by convention alone,
so the key format is part of the contract:
change it in one place and workers silently construct actors
with default arguments.
A key is written only when a value is given,
which is why the worker treats `KeyError` as "none were passed"
rather than an error.
The values are deliberately not kept on the `WorkerGroup`,
so they are not part of what a redefinition is checked against:
the store is the only copy, and the last `define_worker` call wins.

**The executor's name is its identity.**
It prefixes task ids, worker job names, script file names and the log,
and it keys the executor's logger,
so two live executors sharing a name collide on all of those
and their log lines land in both work dirs.
`SlurmPilotExecutor` validates it
(`[A-Za-z][A-Za-z0-9_-]*`, at least 3 characters)
because the characters that are safe in a Slurm job name,
a directory name and a task id are the intersection of three sets,
not one.

**Workers publish their identity at startup, as one key.**
`PilotWorkerProcess.__init__` writes `worker_info:<worker-id>`,
a JSON object, before it builds the actor,
so a worker that dies in its actor's constructor
has still recorded which job and node it died on.
One key and not five, because `swtop` caches what it reads:
a reader landing between two writes would otherwise
remember a worker whose host it never learned.
`WORKER_INFO_PREFIX` lives in `slurm_pilot_worker.py`
and is imported by `swtop.py`, so the two cannot drift apart.
Nothing deletes the key:
the store is in memory and dies with the server,
which is the only cleanup there is.

**Anything `__init__` starts, a failed `__init__` has to stop.**
The monitors and the client are live before the actor is constructed,
and an actor constructor that raises means `close()` is never called,
so the worker stops its own monitors and closes its own channel
before re-raising.

**Task names are UTF-8 in the store, not pickles.**
`set_task_name` writes `task_name:<task_id>` as encoded text,
unlike the actor arguments beside it,
because a name is a string and the point of storing it
is that something other than this library can read it.
`Task.task_name` is read only for the same reason:
the store holds the other copy,
and assigning the attribute would rename the task in this process alone.

`setup_script` is inlined verbatim
into the generated worker script (`{{ setup_script }}`).
Nothing validates it.

### Logging

**The executor's logger is named after the executor**
(`slurm_workflows.executor.<name>`), and does not propagate.
A name shared between executors collects one `FileHandler` per executor,
and every line then lands in every work dir opened in this process:
the first executor's log fills up with the second's records.
`close()` removes and closes the handler.
The logger itself stays in the logging registry, inert,
which is why a test that reuses an executor name
inherits whatever handlers the previous one left on it.

### Templates (`templates/`)

Sbatch and worker shell scripts are generated from Jinja2 templates
using a **custom multi-template-per-file format**.
Each `.jinja` file holds several named templates
delimited by `{#- name: "..." -#}` JSON5 headers,
parsed by `templates/__init__.py`.
Templates are addressed as `"<file_prefix>:<name>"`,
for example `"slurm_pilot:worker_script"`.

`render_template` carries `@overload` signatures
documenting each template's required keyword arguments.
**Keep those overloads in sync when changing template variables.**
The environment uses `StrictUndefined`, so a missing variable is a hard error.

`{#-` is also how a template body ends,
so a body cannot contain a whitespace-trimming Jinja comment:
the parser reads it as the header of the next template.
Use `{#` without the dash inside a body.

### Search spaces (`search_space.py`)

**Nothing here may import torch or botorch.**
That is the whole point of the split:
a search space is arithmetic on one value at a time,
so it can be built and tested where the optimizer cannot be installed,
and `tests/test_search_space.py` runs without the `importorskip`
that skips every botorch test.
`optimize_space_botorch` imports only what it uses of it
and re-exports nothing:
`search_space` is the one place a range is imported from.

**`to_unit` and `to_params` agree by the order of the space.**
A `SearchSpace` is an ordered mapping in practice,
and a unit point is a bare list of coordinates,
so the column order is the mapping's own iteration order.
Two spaces holding the same ranges in a different order
are different spaces to a model fit on one of them.

### Sobol' exploration (`explore_space.py`)

**scipy's Sobol', not botorch's.**
`ExploreSpaceSobolQMC` draws with `scipy.stats.qmc.Sobol`,
so a sweep needs neither torch nor botorch,
and `tests/test_explore_space.py` runs without them.
`rng=` is the seed argument (`seed=` is the older spelling),
which is what the `scipy>=1.15` floor in `pyproject.toml` is for.

**The count is floored to a power of two, and drawn with `random_base2`.**
A Sobol' sequence is only balanced on a power-of-two prefix,
and scipy says so with a warning if asked for anything else.
Since the count is floored anyway,
asking in scipy's own terms is the same draw without the warning.

**Every task is submitted before any of them is waited for.**
That is what "simultaneously" means here:
one `submit` loop over every task's design, then a single `wait`.
Submitting and waiting per task would leave the pool idle
whenever a small sweep finished ahead of a large one,
and would serialize tasks that name different queues
even though nothing makes them wait for each other.

**A task is validated when the sweep is built, not when it runs.**
An empty space, a shadowed parameter or a missing point count
is reported before anything reaches the cluster.
`_resolve` returns a copy with the point count and seed filled in,
so `self.tasks` says what will actually run
and the caller's own dataclass is left as they wrote it.

**It shares the shape of `OptimizeSpaceBotorch`, not its code.**
Submitting a batch, waiting with `RAISE_AFTER_COMPLETED`,
recording what came back and reporting the best
are written out in both classes.
The fiddliest part of it is not:
`utils.objective_value` is the one place an objective's result is checked,
so the four rejection messages cannot drift apart.
The rest still can. A fix to one class belongs in the other.

**The results file format is owned by `explore_space`.**
`load_results` reads what both `save` methods write,
and `SavedResults` says what a file holds.
The optimizer imports the reader rather than reimplementing it,
which is what keeps "the shape the explorer writes" true.
`unit_points` is deliberately not in the file:
only the space can place a point in the unit cube,
and a stored copy could have been written against a different space.

### Batch Bayesian optimization (`optimize_space_botorch.py`)

**It never explores.**
An `OptimizeSpaceBotorch` is constructed from results files
and fails if a task has no observations in them.
That is what makes a search resumable:
the state that has to survive a walltime limit is a file,
not an object, and `save` writes only what its own run measured
so the files concatenate without double counting.

**A round is two batches, not two per task.**
Every active task's fit is submitted before any is waited for,
then every active task's proposed points.
Tasks therefore advance in step and drop out independently,
each against its own `patience`, floor and ceiling.

- **The model is fit to `-f`.**
  botorch maximizes and this minimizes,
  so every acquisition value is in that negated space too.
  `qLogNoisyExpectedImprovement` takes no `best_f`:
  it reads its incumbent off the posterior at `X_baseline`,
  which is the same negated space again.
  Get the sign backwards and the search quietly walks uphill
  instead of failing.
- **`unit_points` holds the point actually evaluated**,
  re-standardized *after* rounding, never the continuous proposal.
  Otherwise the GP is told about a location the objective never ran at.
- **The fit runs on a worker, not on the driver.**
  `fit_and_propose` is submitted to `optimizer_queue`
  as one task per round, the fit and the acquisition together:
  shipping a fitted GP back to the driver
  would cost more than the fit did.
  Keep it a module-level function taking and returning plain Python,
  so cloudpickle sends it by reference
  and no torch object has to survive a hop between hosts.
  Its workers need botorch; `objective_queue`'s do not.
- **The four acquisition knobs belong to the task, not to the process.**
  `num_restarts`, `raw_samples`, `mc_samples` and `acqf_timeout_s`
  are `OptimizationTask` fields with literal defaults,
  passed to every `fit_and_propose` task.
  A value read inside `fit_and_propose` would be the *worker's*,
  ignoring how the search was configured.
  Tests assert them by constructing with them
  (`make_task(acqf_timeout_s=...)`) or against `opt.tasks[i].<knob>`,
  never against a literal.
- **The stall counter runs from round 1;
  `min_search_iterations` gates the stop, not the counting.**
  Report the gap to the stop as
  `max(patience - stalled, min_search_iterations - iteration)`.
  A bare `stalled`/`patience` ratio runs past its own denominator
  whenever the floor outlasts the streak, which the defaults do.
- **One acquisition, one `optimize_acqf` call per round**,
  asking for the whole batch.
  `qLogNoisyExpectedImprovement` takes `X_baseline`,
  every point measured so far, rather than a `best_f` scalar,
  so that argument has to be the `train_x` just fit, not a stale copy.
- **Ask for the batch jointly, never `sequential=True`.**
  It is the usual advice for large batches and it is wrong here:
  measured 10-15x *slower* on a low-dimensional space,
  because the greedy path pays
  the restart cost once per point instead of once per batch.
- **Ranges clamp in `unstandardize`**,
  because `optimize_acqf` can return a point a hair outside the bounds.
- **Keep this module out of the package `__init__.py`**,
  so `import slurm_workflows` works without botorch installed.
- `optimize_acqf`, `fit_gpytorch_mll`, `qLogNoisyExpectedImprovement`
  and `fit_and_propose` are called through module globals.
  The tests monkeypatch those to assert what was asked for,
  without paying for a real acquisition optimization.
  That works because `LocalExecutor` runs the submitted task inline,
  in the test's own process, so patching reaches the fit
  only for as long as that stays true.
- **`test_search_moves_toward_the_minimum` asserts the *median* search
  point**, not the max and not `best_point()`.
  Neither of those works.
  qLogNEI explores away from the incumbent,
  so the max hits 1.0 on correct runs,
  and exploration alone lands near the minimum,
  so `best_point()` passes even with the sign flipped.
  [`howto-run-tests.md`](howto-run-tests.md) carries the measured margins.

### Monitoring (`monitors.py`, `swtop.py`)

**One worker per subject samples, and a counter decides which.**
`counter_get_next_value` hands out distinct, gap-free values,
so the worker told 1 for `host_monitor:<hostname>` takes the node
and the one told 1 for `slurm_job_monitor:<job-id>` takes the job.
No lock, no designated rank, and no need for the workers to know each other.
Nothing hands a subject back when that worker dies:
the series stops, and `swtop` marks it stale.
Re-electing would need a heartbeat and a lease, which this does not have.

**Sampling threads are daemons that swallow their errors.**
A worker killed at the end of its walltime must not be held open by a
monitor, and a failed sample must not end the series:
a node briefly unreachable is the common case, and a gap beats a stop.
`close()` stops them before closing the client whose channel they use.

**CPU is a rate, differenced from a total.**
The kernel reports CPU as microseconds that only rise,
so `CgroupSampler` keeps the previous reading
and the first sample of a run necessarily reports 0 cores.
The cgroup's own accounting is preferred over summing processes
because it covers every process and thread Slurm put in the job,
including ones the worker never started.

**`swtop` can only show what an RPC can answer.**
`task_get_count_by_state` covers every task,
but nothing enumerates tasks, workers, hosts or jobs,
so every table is built by searching a key space
for keys the executor, the workers and the monitors publish.
A task nobody named cannot be listed, only counted.
That is a property of the server, not a gap to be worked around here.

**`swtop` reads only the tail of a series.**
`time_series_get` with no bounds returns every point ever appended,
which over a day-long run is most of the memory the server is holding.
It asks for the last minute, and calls a subject with nothing there stale.

**Identities are read once.**
`Collector` caches every worker's fields and every task's name,
because both are written once and never change.
Without the cache a 400 worker pool would cost 2000 reads every 2 seconds.

**A poll that fails is drawn, not raised.**
A monitor that exits when the server blinks
takes the screen down with it.

### Slurm interaction (`slurm_utils.py`)

`is_batch_worker=False` wraps the worker script in `srun`
and passes `--output <work_dir>/<name>-%j-%t.out`.
That is why the `worker_sbatch_script` template takes `name` and `work_dir`.
Which file a worker's log ends up in
is documented for users in
[`howto-use-slurm-workflows.md`](howto-use-slurm-workflows.md#logs-and-troubleshooting);
what follows is why the shell decides it rather than Python.

**The `--output` is dropped for a job of exactly one task**,
which then writes to the batch job's own output file
instead of a per-task file duplicating it.
The job decides at run time, in the shell,
because Python cannot know the answer when the script is rendered.
`SLURM_NTASKS` is the number of tasks in the job
whenever `--ntasks` or any `--ntasks-per-*` option was given,
so it settles the question alone and nothing may override it.
It is unset only when no task count was asked for,
and that is exactly when one task per node is the default,
so `SLURM_JOB_NUM_NODES` stands in there.
The count is per *job*, not per node:
`--nodes=4 --ntasks-per-node=1` is four tasks
and keeps its per-task files.

**The worker process must not redirect `sys.stdout` or `sys.stderr`.**
Slurm writes those files itself via `--output`,
and `logging.basicConfig` leaves the streams on the inherited handles.
Redirecting them again would leave the Slurm-written files empty.

**The job id is searched for in `sbatch`'s stdout, not matched at the start.**
A site that prints a banner or a warning there
would otherwise turn a successful submission into a parse failure.

## Conventions

- **After changing any Python, run `black`, then `pyright`, then `pytest`.**
  All three must be clean before the change is done:
  `black` reports "left unchanged",
  `pyright` reports "0 errors",
  `pytest` passes.
  Run `black` first.
  It rewrites lines,
  so type-checking before formatting
  can report positions that no longer exist.
  Neither tool is advisory here.

  If `pyright` objects to a deliberate test double,
  say so with a `cast` and a comment explaining why the double is sufficient
  (see `as_executor` in `tests/test_optimize_space_botorch.py`)
  rather than silencing it with a bare `# type: ignore`.
  If it objects to something in `src/`, prefer fixing the annotation.
  The `SearchSpace = Mapping[...]` alias exists because `dict[...]`
  is invariant and made a correct call fail to type-check.
- **Deprecation warnings are errors in the test suite**
  (`filterwarnings` in `pyproject.toml`).
  They are how a dependency announces a break one release ahead,
  and a warning nobody reads is a break discovered at the worst moment.
  Other warnings stay warnings:
  botorch is deliberately handed a constant objective in places
  and says so at runtime.
- **Prose uses semantic line breaks.**
  Break at clause boundaries, not at a column limit:
  start a new line after each sentence,
  and at punctuation that already separates clauses (`.` `;` `:` `,` `--`)
  or before a conjunction or preposition that opens a new phrase.
  Never end a line mid-phrase,
  on an article, conjunction, preposition or auxiliary,
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
  and every Markdown file in the repository:
  `README.md`, `docs/*.md`, and this file.
  Exempt: anything whose line structure is already meaningful,
  such as code inside fences, Markdown tables, headings,
  and ASCII section banners (`# ---- name ----`).
- Cleanup is per-class; there is no shared base class for it.
- Version is derived by `setuptools_scm` from git tags,
  with a fallback of `1.0.0-dev`.
