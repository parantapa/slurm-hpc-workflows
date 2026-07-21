# slurm-workflows: HPC workflow helpers for Slurm clusters

`slurm-workflows` lets you run Python functions on a Slurm cluster
without writing sbatch scripts by hand.
It provides a [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) inspired interface
that launches long-lived **pilot jobs** and dispatches tasks to them.
This ensures that the cost of Slurm's queueing latency is paid once per worker
instead of once per task.

## Features

- **Pilot-job task execution** — `SlurmPilotExecutor` submits reusable worker jobs
    and streams Python callables to them through a task queue.
- **Dynamic scaling** — grow or shrink a pool of workers at runtime.
    workers are Slurm jobs you can be sized flexibly.
- **Stateful actors** — keep expensive per-worker state (loaded models, DB
  connections) warm across many tasks.
- **Transparent serialization** — functions, arguments, and return values are
  [cloudpickled](https://github.com/cloudpipe/cloudpickle),
  so closures and lambdas work.

## Requirements

- Python >= 3.12
- Access to a Slurm cluster (`sbatch`, `squeue`, `scancel` on `PATH`)
- A running [`ds-service`](https://github.com/parantapa/ds-service)
    server (see below), reachable from login node and the compute nodes.

## Installation

```sh
git clone https://github.com/parantapa/slurm-hpc-workflows.git
cd slurm-hpc-workflows
pip install -ve .
```

For a full cluster setup (Miniforge environment, XDG directories, Jupyter and
proxy configuration on UVA's Rivanna), see [`docs/rivanna-setup.md`](docs/rivanna-setup.md).

## Quick start

Every worker sources a **setup script** on its compute node before running.
This is a shell snippet that activates your environment
(e.g. `conda activate`, `module load`).
Create one, for example `setup.sh`:

```sh
module load gcc/14.2.0
conda activate my-env
```

Then run tasks against a pilot pool:

```python
from slurm_workflows import SlurmPilotExecutor, check_for_error

def square(x):
    return x * x

DS_SERVICE_ADDRESS = "HOST-IP:5051"

# server_address points at your running ds-service instance.
executor = SlurmPilotExecutor(server_address=DS_SERVICE_ADDRESS)

# 1. Describe a kind of worker (nothing is launched yet).
executor.define_worker(
    name="cpu",
    sbatch_args=["-A my_alloc", "-p standard", "--cpus-per-task=4", "-t 01:00:00"],
    setup_script="setup.sh",
)

# 2. Launch 4 pilot jobs of that kind.
executor.scale_workers("cpu", 4)

# 3. Submit tasks to a named queue; workers pull from it.
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

### Stateful actors

To keep per-worker state warm across tasks,
register an actor class by its importable name.
Each worker instantiates it once,
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
    setup_script="setup.sh",
    actor_class_name="my_pkg.model.Model",
)
executor.scale_workers("gpu", 2)

tasks = [executor.submit("gpu", "predict", item) for item in dataset]
executor.wait(tasks)
```

### Running the task-queue server

The executor and workers communicate only through a `ds-service` server — they
never talk to each other directly. You can start one locally on the login node:

```python
from slurm_workflows.ds_service import DsService

with DsService(host="0.0.0.0", port=5051) as ds:
    executor = SlurmPilotExecutor(server_address=ds.address)
    ...
```

The server must be reachable from the compute nodes, so bind it to an address the
workers can route to.

## How it works

Three processes cooperate through the task queue:

| Process | Runs on | Role |
| --- | --- | --- |
| Coordinator (`SlurmPilotExecutor`) | login node | defines worker groups, scales pilot jobs, submits tasks |
| `ds-service` | login node (or elsewhere) | holds tasks on named queues |
| Pilot workers | compute nodes | pull tasks, execute them, return results |

`scale_workers` submits Slurm jobs whose scripts (generated from Jinja templates)
launch a `slurm-pilot-worker` process. Each worker loops: fetch a task, run it,
post the cloudpickled result back. Exceptions on a worker are **not** re-raised in
the coordinator — they are captured as a `RemoteExecutionError` and returned as
the task's `output`, discoverable with `check_for_error`.

## License

MIT — see [LICENSE](LICENSE).
