# How to use slurm-workflows

[← back to the main README](../README.md)

This is the user guide for `slurm-workflows`.
For what the package is and how to install it,
see the [README](../README.md).

## Concepts

**Setup script.** A shell snippet that every worker runs
    on its compute node before starting.
    This is how your environment (`module load`, `conda activate`)
    reaches the compute node — nothing is inherited from the login node.
    You pass the snippet's **text**, not a path to a file;
    it is inlined into each generated worker script.
    It is optional — omit it if `/etc/profile`,
    which every worker sources first, already suffices.

**Worker group.** A named recipe for a worker:
    sbatch arguments, setup script, optional actor class.
    Defining a group does not launch workers.
    `scale_workers` method is used to start/stop workers.

**Queue.** Tasks are submitted to a named queue,
and **a worker group pulls from the queue matching its own name**.
So `submit("gpu", ...)` is served by workers from the group named `gpu`.

## Quick start

```python
from slurm_workflows import SlurmPilotExecutor, check_for_error

def square(x):
    return x * x

DS_SERVICE_ADDRESS = "HOST-IP:5051"

SETUP_SCRIPT = """
module load gcc/14.2.0
conda activate my-env
"""

# server_address points at your running ds-service instance.
executor = SlurmPilotExecutor(server_address=DS_SERVICE_ADDRESS)

# 1. Describe a kind of worker (nothing is launched yet).
executor.define_worker(
    name="cpu",
    sbatch_args=["-A my_alloc", "-p standard", "--cpus-per-task=4", "-t 01:00:00"],
    setup_script=SETUP_SCRIPT,
)

# 2. Launch 4 pilot jobs of that kind.
executor.scale_workers("cpu", 4)

# 3. Submit tasks to a named queue; workers of that group pull from it.
tasks = [executor.submit("cpu", square, i) for i in range(100)]

# 4. Collect results as they complete (tqdm progress bar included).
for task in executor.as_completed(tasks, desc="squaring"):
    ...  # task.output holds the return value

# Surface any tasks that raised on the worker.
for task in check_for_error(tasks):
    print(task.task_id, task.output.error_id)

# 5. Cancel all pilot jobs when done.
executor.close()
```

`sbatch_args` are passed straight through to `sbatch`,
so any Slurm option works.
`submit` returns immediately with a `Task` handle;
`as_completed(tasks)` (or `wait(tasks)`) blocks until results are ready.

If you keep your setup snippet in a file, read it in yourself:

```python
from pathlib import Path

executor.define_worker(..., setup_script=Path("setup.sh").read_text())
```

You don't have to wait for workers before submitting ---
tasks queue up and are picked up as pilot jobs start running.

### Stateful actors

To keep per-worker state warm across tasks,
register an actor class by its importable name.
Each worker instantiates it once at startup,
and you dispatch **method names** (as strings) instead of functions:

```python
# my_pkg/model.py
class Model:
    def __init__(self):
        self.model = load_expensive_model()   # runs once per worker

    def predict(self, x):
        return self.model(x)

    def close(self):                           # optional cleanup hook
        self.model.release()
```

```python
executor.define_worker(
    name="gpu",
    sbatch_args=["-A my_alloc", "-p gpu", "--gres=gpu:1", "-t 02:00:00"],
    setup_script=SETUP_SCRIPT,
    actor_class_name="my_pkg.model.Model",
)
executor.scale_workers("gpu", 2)

tasks = [executor.submit("gpu", "predict", item) for item in dataset]
executor.wait(tasks)
```

The class must be importable on the compute node.
By default the executor's current working directory
is added to the workers' `sys.path`; add more with `python_paths=[...]`.

### One worker per job, or one per task

`is_batch_worker` controls how many worker processes each Slurm job starts:

| Setting | Script is run with | Workers per job |
| --- | --- | --- |
| `is_batch_worker=False` (default) | `srun` | one per Slurm task in the allocation |
| `is_batch_worker=True` | sourced directly | one, on the batch node |

So with the default, `--nodes=4 --ntasks-per-node=2`
gives you 8 worker processes from a single `scale_workers(..., 1)` call.
Use `is_batch_worker=True` when you want a single process
that owns the whole allocation (e.g. an MPI-style or whole-node job).

### Running the task-queue server

The executor and workers communicate only through a `ds-service` server
— they never talk to each other directly.
You can start one on the login node:

```python
from ds_service_client import DsServiceServer
from slurm_workflows import SlurmPilotExecutor

with DsServiceServer(interface="ib0", port=5051) as ds:
    ds.wait_until_ready()      # blocks until it accepts connections

    executor = SlurmPilotExecutor(server_address=ds.address)
    ...
```

`DsServiceServer` comes from the `ds-service-client` package,
and the constructor spawns the process,
so the server is already coming up when it returns.
Call `wait_until_ready()` before handing the address to anything —
a client that connects too early
lands in a gRPC reconnect backoff and is slow to recover.
Omit `port` to get an arbitrary free one.

The server must be reachable from the compute nodes,
so it is bound to the IPv4 address of the `interface` you name
— `ib0` above, the login node's Infiniband interface —
and `ds.address` is the `host:port` the workers then connect to.
Naming an interface that does not exist on that node,
or that has no IPv4 address, raises `ValueError` at construction
rather than starting a server the workers cannot reach.
Use the loopback interface (`lo`) only for a server
that nothing outside the node needs to talk to.

`ds_service_bin` (or the `DS_SERVICE_BIN` environment variable)
sets how the server is started, and may be a whole command line
— `apptainer run ds-service.sif` works as well as a path to a binary.

## Batch Bayesian optimization with botorch

The package ships a botorch optimizer
that fits one Gaussian process to everything measured so far
and proposes a whole batch of points per round,
evaluated across the worker pool
— so a round's points run concurrently on the pilot pool described above.

The model fit is a task as well, on a queue of its own
(`optimizer_queue`, alongside the `objective_queue` the evaluations go to),
so the driver only submits and waits,
and only the fit's workers need botorch installed.

It has its own guide:
**[Batch Bayesian optimization with botorch](bayesian-optimization-using-botorch.md)**.

## API reference

Every public name, argument and return type
is listed in the **[API reference](api-reference.md)**.

## Logs and troubleshooting

Everything for a run lives under the executor's `work_dir`
(printed as `executor.work_dir`):

| File | Contents |
| --- | --- |
| `executor.log` | Worker submission and cancellation from the executor's side |
| `<worker-name>.sh`, `<worker-name>.sbatch` | The generated scripts — read these first when a job dies immediately |
| `<worker-name>-<jobid>-<task>.out` | One per worker process: setup-script trace, task-by-task progress, full tracebacks |
| `<worker-name>-<jobid>.out` | The batch job's own output — and the worker's log too, when the job is a single task |

Slurm writes those files; the worker process doesn't redirect its own output.
Which of the two you want depends on how the group was defined:

- **`is_batch_worker=False`** (the default) runs the worker under `srun`,
    which fans out over every task in the allocation.
    Each task gets `--output <work-dir>/<worker-name>-%j-%t.out`,
    so `<task>` is the task's rank — that file is the worker's log.
    Without the per-task `--output`
    all of them would interleave into the single batch file.
    `<worker-name>-<jobid>.out` then holds
    only what the batch script itself emitted,
    which in practice means `srun`'s own errors.

    The exception is a job of exactly one task
    — `--ntasks=1`, or `--nodes=1` with nothing else said about tasks.
    One task has nothing to interleave with,
    so it keeps `srun` but drops the `--output`
    and writes to `<worker-name>-<jobid>.out` like a batch worker,
    rather than leaving you two files to open per worker.
    Note this counts tasks in the *job*, not per node:
    `--nodes=4 --ntasks-per-node=1` is four tasks
    and still gets four per-task files.

    Which way a job went is not something you have to reconstruct:
    the batch file opens with the task count the job decided on
    (`Num tasks: 4`), says so when it redirects,
    and traces the `srun` command it ran.
- **`is_batch_worker=True`** runs one worker directly on the batch node,
    with no `srun` and so no per-task file.
    Everything lands in `<worker-name>-<jobid>.out`.

The `error_id` inside a `RemoteExecutionError`
appears verbatim next to the traceback
— grep for it across the work dir to find the failing task's stack.

Common failure modes:

- **Tasks never complete, jobs are running.**
    The queue name doesn't match a worker group name,
    or the workers can't reach `ds-service` from the compute nodes.
    Check the worker's `-<jobid>-<task>.out` file.
- **`RuntimeError: ... tasks are on queues with no worker started`.**
    Raised as soon as you wait,
    because `scale_workers` was never called for those queues.
    Either you forgot to scale the group,
    or the queue name is a typo — it is not checked at `submit` time,
    so compare it against your `define_worker` names.
- **`RuntimeError: Task ... was canceled on the task queue server`.**
    Somebody cancelled the task through the `ds-service` client directly
    -- nothing in this library does.
    A canceled task is never dispatched again
    and never produces an output,
    so waiting on it is reported rather than retried.
    Resubmit it if you still want it run.
- **`RuntimeError: ... tasks are on queues with no live pilot job`.**
    Raised while waiting: the group *was* scaled,
    but its jobs have since left the cluster
    — time limit reached, cancelled, or exited before draining the queue.
    The worker's `.out` file will say which.
    Scale the group back up and resubmit.
- **Jobs start and exit within seconds.**
    The setup script failed.
    It runs inside the worker script,
    so its trace is in the same `.out` file as the worker's log
    — not the batch one.
- **`ModuleNotFoundError` on a worker.**
    The module isn't importable on the compute node
    — add `python_paths=[...]`
    or install it into the environment the setup script activates.
