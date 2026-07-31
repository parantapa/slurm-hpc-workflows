# slurm-workflows: HPC workflow helpers for Slurm clusters.

![Futuristic banner image.](extra/banner-image.png "Futuristic banner image.")

`slurm-workflows` lets you run Python functions on a Slurm cluster
without writing sbatch scripts by hand.
It provides a [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html)-inspired
interface that launches long-lived **pilot jobs** and dispatches tasks to them,
so Slurm's queueing latency is paid once per worker instead of once per task.

### Features

- **Pilot-job task execution** — pay the queue wait once,
    then dispatch tasks at queue-latency speed.
- **Dynamic scaling** — grow or shrink a pool of workers at runtime
    with `scale_workers`.
- **Stateful actors** — keep expensive per-worker state
    (loaded models, DB connections) warm across many tasks.
- **Transparent serialization** — functions, arguments, and return values
    are [cloudpickled](https://github.com/cloudpipe/cloudpickle),
    so closures and lambdas work.
- **Non-fatal remote errors** — an exception on a worker
    doesn't kill the driver script; it comes back as the task's result.
- **Batch Bayesian optimization** — a [botorch](https://botorch.org/) optimizer
    that proposes a whole batch of points at once
    and evaluates them across the worker pool,
    over mixed integer / float / log-float / categorical spaces.

## Requirements

- Python >= 3.12
- Access to a Slurm cluster (`sbatch`, `squeue`, `scancel` on `PATH`)
- A running [`ds-service`](https://github.com/parantapa/ds-service) server,
  reachable from the login node *and* the compute nodes.
  The client library is installed as a dependency;
  the server is a separate install.

## Installation

```sh
pip install -U slurm-workflows
```

## Documentation

- **[How to use slurm-workflows](docs/howto-use-slurm-workflows.md)** — the user
  guide: concepts, quick start, stateful actors, one worker per job vs per task,
  running the task-queue server, and troubleshooting.
- **[API reference](docs/api-reference.md)** — `SlurmPilotExecutor`, `Task`,
  `check_for_error`, and the worker environment.
- **[Batch Bayesian optimization with botorch](docs/bayesian-optimization-using-botorch.md)**
  — the batch Bayesian optimizer that proposes a whole batch of points per round.

## License

MIT — see [LICENSE](LICENSE).
