# API reference

[← back to the how-to guide](howto-use-slurm-workflows.md)

Import from the package root:
`from slurm_workflows import SlurmPilotExecutor, RaiseOnError, RemoteExecutionError`.
(The botorch optimizer is the exception:
it lives in `slurm_workflows.optimize_space_botorch`,
so that the package keeps working without botorch installed.)

## `SlurmPilotExecutor(name, server_address, work_dir=None)`

`name` identifies the executor.
It prefixes every task id (`<name>.task.<n>`)
and every worker's Slurm job name (`<name>.worker.<group>.<index>`),
and it names the executor's log.
So give two executors on one queue, or one cluster, two names:
a shared one collides on all three.
It must start with a letter
and hold only letters, digits, `_` and `-` (`[A-Za-z][A-Za-z0-9_-]*`),
and be at least 3 characters long; anything else raises `ValueError`.

`server_address` is the `host:port` of the `ds-service` server.
`work_dir` defaults to a timestamped directory under
`<platform cache dir>/slurm-workflows/<name>`
(`XDG_CACHE_HOME`-driven on Linux),
so one executor's runs sit together;
generated scripts and all logs land there.

| Method | Purpose |
| --- | --- |
| `define_worker(name, sbatch_args, ...)` | Register a worker group. Idempotent — redefining a group identically is a no-op, redefining it differently asserts. |
| `scale_workers(name, count)` | Submit or cancel pilot jobs so the group has `count` jobs. |
| `submit(queue, fn, *args, **kwargs) -> Task` | Enqueue a task. `queue` is a group name or a list of them; `fn` is a callable, or a method name (`str`) for actor workers. |
| `as_completed(tasks, desc=None, unit="task", raise_on_error=...)` | Yield tasks as their results arrive, wrapped in a tqdm bar. Raises `RuntimeError` rather than blocking forever on a task that can never finish — see below. |
| `wait(tasks, desc=None, unit="task", raise_on_error=...)` | Same, but discards the iterator — just block until all are done. |
| `set_task_name(task, name)` | Name a task, on the queue server as well as locally. |
| `num_groups()` / `num_workers(detail=False)` | Counts of defined groups and submitted workers; `detail=True` returns a per-group dict. |
| `stop()` | Cancel all pilot jobs, keep the executor usable. |
| `close()` | Cancel all pilot jobs and close the queue-server connection. |

It is also a context manager,
which is the easiest way to be sure pilot jobs are cancelled
even if the block raises:

```python
with SlurmPilotExecutor(name="demo", server_address=address) as executor:
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

Two more states end a wait, both read straight off the queue server:

- **the server does not know the task id** —
  `RuntimeError: Task ... is unknown to the task queue server`.
  In practice a `Task` built by hand,
  or one left over from a server that has since been restarted.
- **the task was cancelled** —
  `RuntimeError: Task ... was canceled on the task queue server`.
  Nothing in this library cancels a task,
  so this means somebody called `task_cancel` through the `ds-service`
  client directly.
  A cancelled task is never dispatched again, so waiting cannot help.

Remaining `define_worker` options:

| Argument | Default | Meaning |
| --- | --- | --- |
| `setup_script` | `""` | Shell snippet run on the compute node before the worker starts — the text, not a path. Must be a `str`; omit it (or pass `""`) if `/etc/profile` (always sourced) already gives workers the right environment. |
| `is_batch_worker` | `False` | See [One worker per job, or one per task](howto-use-slurm-workflows.md#one-worker-per-job-or-one-per-task). |
| `actor_class_name` | `None` | Fully qualified class name to instantiate once per worker. |
| `actor_class_args` | `None` | Positional arguments for that class's constructor. Only valid with `actor_class_name`. |
| `actor_class_kwargs` | `None` | Keyword arguments for that class's constructor. Only valid with `actor_class_name`. |
| `python_paths` | `None` | Extra paths prepended to the workers' `sys.path`. |
| `add_cwd_to_python_path` | `True` | Also add the coordinator's cwd. |
| `worker_exe` | `"slurm-pilot-worker"` | Worker entry point, if you've wrapped or renamed it. |

The actor arguments are cloudpickled and put in the `ds-service` key value
store, under `actor_class_args:<name>` and `actor_class_kwargs:<name>`,
where `<name>` is the worker group's name.
Each worker reads them back at startup.
So they must be picklable,
and anything they refer to has to be importable on the compute node,
exactly as for the actor class itself.
They are not part of the group's identity,
so redefining a group with different ones is allowed,
unlike differing `sbatch_args`.
Only the workers started after that call read the new values, though:
an actor is constructed once, when its worker starts.

## `Task`

`submit` returns a `Task` with `task_id`, `queue`, `priority`, `function`,
`input`, and `output`. `output` is a sentinel until the task completes; after
that it holds the return value — or a `RemoteExecutionError(error, error_id)`
if the worker raised.

`task_name` is a read-only property, `None` until
`executor.set_task_name(task, name)` is called.
That call is what makes the name real:
it stores the name on the queue server, under `task_name:<task_id>`,
as UTF-8 rather than a pickle, so anything reading the store can read it too,
and updates the `Task` to match.
Nothing in this library dispatches on the name;
it is there for whoever is looking at the queue.

`priority` is assigned by `submit` and orders the queue.
`ds-service` dispatches the highest value first,
and `submit` sets it from a negated wall clock,
so tasks on one queue are served **oldest first**
however many executors are feeding it.
It is recorded on the `Task` for inspection;
changing it there has no effect,
since the value the server orders by was sent when the task was enqueued.

## `RaiseOnError`

What `as_completed` and `wait` do about a task that fails.
A failure is any of: a task whose worker raised
(its `output` is a `RemoteExecutionError`),
a task canceled on the queue server,
a task the server does not know,
or a pending task whose queues have no pilot job left to run them.
Only the tasks that cannot finish are given up on;
the rest of the batch is still waited for.

| Value | Effect |
| --- | --- |
| `RAISE_ON_FIRST_ERROR` | The default. Stop at the first failure and raise `RuntimeError`. |
| `RAISE_AFTER_COMPLETED` | Wait for every task that can still finish, then raise once for all the failures together. `as_completed` treats this as `RAISE_ON_FIRST_ERROR`. |
| `RAISE_NEVER` | Report and return. |

**Every failure is warned about on stderr as it is met**, whichever value
is used; the value decides only whether an exception follows.
The warning carries the task id and, for a worker that raised, the
`error_id` that appears beside the traceback in that worker's log.

`as_completed` hands out results one at a time,
so there is no point at which it could still be feeding the caller
*and* have finished — which is why `RAISE_AFTER_COMPLETED` collapses to
`RAISE_ON_FIRST_ERROR` there.
Use it with `wait`, where one failed evaluation of a batch
should not hide the other 39.

With `RAISE_NEVER` the caller reads the outcome off the tasks:

```python
from slurm_workflows import RaiseOnError, RemoteExecutionError
from slurm_workflows.slurm_pilot_executor import NoOutput

executor.wait(tasks, raise_on_error=RaiseOnError.RAISE_NEVER)

failed = [t for t in tasks if isinstance(t.output, RemoteExecutionError)]
never_ran = [t for t in tasks if t.output is NoOutput]
```

`output` stays `NoOutput` for a task that was canceled, is unknown to the
server, or was still pending when the last pilot job went away.

## Worker environment

Inside a task, these environment variables are set:

- `PILOT_WORKER_NAME` — e.g. `demo.worker.cpu.0`
- `PILOT_WORKER_GROUP` — the group name
- `DS_SERVER_ADDRESS` — the queue server address
- plus the usual Slurm variables (`SLURM_JOB_ID`, …)

Each worker also publishes where it is running
in the `ds-service` key value store when it starts,
under `worker_info:<worker-id>`,
where the worker id is
`<worker-name>.<slurm-job-id>.<hostname>.<pid>`.
The value is a JSON object, not a pickle,
so anything can read it:

| Field | Value |
| --- | --- |
| `group` | The group whose queue it serves |
| `name` | The worker's name, which is its Slurm job name |
| `slurm_job_id` | The job it is running in |
| `hostname` | The compute node it landed on |
| `pid` | Its process id on that node |

One key rather than one per field,
so a reader can never catch a worker half-described.
The worker id is the handle the queue server hands out
(`task_get_worker_id` says which worker took a task),
so this is how you get from a task
to the process and node that ran it.
Nothing removes the key when a worker exits.

Workers also sample the node they run on and the Slurm job they belong to,
appending to `ds-service` time series every 5 seconds
(`host_free_memory:<hostname>`, `host_load_average:<hostname>`,
`host_dev_shm_used:<hostname>`, `host_tmp_used:<hostname>`,
`slurm_job_memory:<job-id>` and `slurm_job_cpu:<job-id>`).
One worker per node and one per job does this,
elected between them with the `host_monitor:<hostname>`
and `slurm_job_monitor:<job-id>` counters.
`swtop` displays the result;
see [Watching a run](howto-use-slurm-workflows.md#watching-a-run-with-swtop).

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
 `as_completed` and `wait` are what turn that back into an exception,
 under the [`RaiseOnError`](#raiseonerror) policy they are given.
- **Submitting from inside a job works.** `sbatch` is invoked
    with all `SLURM_*` / `SLURMD_*` / `PMI_*` / `SRUN_*` variables
    stripped from the environment,
    so a coordinator running inside a Slurm allocation
    can still submit pilot jobs.
