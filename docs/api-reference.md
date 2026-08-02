# API reference

[← back to the how-to guide](howto-use-slurm-workflows.md)

Import from the package root:
`from slurm_workflows import SlurmPilotExecutor, check_for_error`.
(The botorch optimizer is the exception:
it lives in `slurm_workflows.bayes_opt_botorch`,
so that the package keeps working without botorch installed.)

## `SlurmPilotExecutor(server_address, work_dir=None)`

`server_address` is the `host:port` of the `ds-service` server.
`work_dir` defaults to a timestamped directory under the platform cache dir
(`XDG_CACHE_HOME`-driven on Linux); generated scripts and all logs land there.

| Method | Purpose |
| --- | --- |
| `define_worker(name, sbatch_args, ...)` | Register a worker group. Idempotent — redefining a group identically is a no-op, redefining it differently asserts. |
| `scale_workers(name, count)` | Submit or cancel pilot jobs so the group has `count` jobs. |
| `submit(queue, fn, *args, **kwargs) -> Task` | Enqueue a task. `queue` is a group name or a list of them; `fn` is a callable, or a method name (`str`) for actor workers. |
| `as_completed(tasks, desc=None, unit="task")` | Yield tasks as their results arrive, wrapped in a tqdm bar. Raises if a pending task's queues all lose their pilot jobs — see below. |
| `wait(tasks, desc=None, unit="task")` | Same, but discards the iterator — just block until all are done. |
| `num_groups()` / `num_workers(detail=False)` | Counts of defined groups and submitted workers; `detail=True` returns a per-group dict. |
| `stop()` | Cancel all pilot jobs, keep the executor usable. |
| `close()` | Cancel all pilot jobs and close the queue-server connection. |

It is also a context manager,
which is the easiest way to be sure pilot jobs are cancelled
even if the block raises:

```python
with SlurmPilotExecutor(server_address=address) as executor:
    executor.define_worker(name="cpu", sbatch_args=[...])
    executor.scale_workers("cpu", 4)
    ...
# close() has run: every pilot job is cancelled
# and the queue connection is shut.
```

Leaving the block calls `close()`, so the executor is spent afterwards.
An exception raised inside the block still propagates.

### Waiting on tasks nothing can run

A task whose queues have no worker can never finish,
so `as_completed` / `wait` raise `RuntimeError` naming those queues
rather than blocking until you give up.
They check this twice, for two different failure modes.

**Before waiting at all**, and without asking Slurm,
they require that `scale_workers` has been called
for at least one of each pending task's queues.
This catches the two mistakes
that would otherwise cost you a minute of staring at a progress bar:
forgetting to scale a group,
and mistyping a queue name (queue names are not validated at `submit` time).
The error is raised before any result is yielded,
so a finished task in the same batch cannot mask a stranded one.

**Then once a minute while blocked**,
they ask `squeue` whether each pending task's queues
still have a job on the cluster —
catching an allocation that ended, jobs that were cancelled,
and jobs that died before draining their queue.
The first of these checks is a minute in, not immediate,
so the documented submit-then-scale order keeps working:
a queue whose job has not started *yet* is not a queue that has lost it.
A `squeue` that cannot be reached leaves liveness unknown rather than dead,
so that case is logged and retried instead of ending the wait.

Both checks only know about workers **this executor** started.
An executor that submits to a queue served by pilot jobs
some other process launched will be refused;
have the executor that waits be the one that scaled the group.

Remaining `define_worker` options:

| Argument | Default | Meaning |
| --- | --- | --- |
| `setup_script` | `""` | Shell snippet run on the compute node before the worker starts — the text, not a path. Must be a `str`; omit it (or pass `""`) if `/etc/profile` (always sourced) already gives workers the right environment. |
| `is_batch_worker` | `False` | See [One worker per job, or one per task](howto-use-slurm-workflows.md#one-worker-per-job-or-one-per-task). |
| `actor_class_name` | `None` | Fully qualified class name to instantiate once per worker. |
| `python_paths` | `None` | Extra paths prepended to the workers' `sys.path`. |
| `add_cwd_to_python_path` | `True` | Also add the coordinator's cwd. |
| `worker_exe` | `"slurm-pilot-worker"` | Worker entry point, if you've wrapped or renamed it. |

## `Task`

`submit` returns a `Task` with `task_id`, `queue`, `priority`, `function`,
`input`, and `output`. `output` is a sentinel until the task completes; after
that it holds the return value — or a `RemoteExecutionError(error, error_id)`
if the worker raised.

## `check_for_error(tasks, verbose=True)`

Returns the subset of `tasks` whose `output` is a `RemoteExecutionError`,
printing each one's `error` and `error_id` unless `verbose=False`.

## Worker environment

Inside a task, these environment variables are set:

- `PILOT_WORKER_NAME` — e.g. `slurm_pilot_worker.cpu.0`
- `PILOT_WORKER_GROUP` — the group name
- `DS_SERVER_ADDRESS` — the queue server address
- plus the usual Slurm variables (`SLURM_JOB_ID`, …)

| Process | Runs on | Role |
| --- | --- | --- |
| Coordinator (`SlurmPilotExecutor`) | login node | defines worker groups, scales pilot jobs, submits tasks |
| `ds-service` | login node (or elsewhere) | holds tasks on named queues |
| Pilot workers | compute nodes | pull tasks, execute them, return results |

`scale_workers` renders a shell script and an sbatch wrapper
from Jinja templates and submits them.
Each job sources your setup script and launches `slurm-pilot-worker`,
which loops forever: fetch a task from its group's queue,
cloudpickle-load the function, run it, post the cloudpickled result back.

Two details worth knowing:

- **Exceptions are values.** A task that raises on a worker
 does not propagate to the coordinator.
 The worker catches it, logs the traceback under a generated `error_id`,
 and returns a `RemoteExecutionError` as the task's `output`.
 Always run `check_for_error` over a completed batch.
- **Submitting from inside a job works.** `sbatch` is invoked
    with all `SLURM_*` / `SLURMD_*` / `PMI_*` / `SRUN_*` variables
    stripped from the environment,
    so a coordinator running inside a Slurm allocation
    can still submit pilot jobs.
