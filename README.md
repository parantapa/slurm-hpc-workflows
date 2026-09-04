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
- **Dynamic scaling** — grow or shrink a pool of workers at runtime.
- **Stateful actors** — keep expensive per-worker state
    (loaded models, DB connections) warm across many tasks.
- **Transparent serialization** — functions, arguments, and return values
    are transferred using [cloudpickle](https://github.com/cloudpipe/cloudpickle).
- **Live queue view** — `swtop <server-address>` shows the tasks and the
    pilot workers of a running job, refreshed every couple of seconds.
- **Batch Bayesian optimization** — a [botorch](https://botorch.org/) based optimizer
    that proposes a whole batch of points at once
    and evaluates them across the worker pool,
    over mixed integer / float / log-float / categorical spaces.

## Requirements

- Python >= 3.12
- Access to a Slurm cluster (`sbatch`, `squeue`, `scancel` on `PATH`)
- A running [`ds-service`](https://github.com/parantapa/ds-service) server.

## Installation

```sh
pip install -U slurm-workflows
```

## Documentation

| Document | What it covers |
| --- | --- |
| [How to use slurm-workflows](docs/howto-use-slurm-workflows.md) | The user guide: concepts, quick start, stateful actors, one worker per job vs per task, running the task-queue server, and troubleshooting. |
| [Developer notes](docs/developer-notes.md) | Working on `slurm-workflows` itself: where the code lives, the invariants that are quiet when broken, and the conventions a change is checked against. |
| [Batch Bayesian optimization with botorch](docs/bayesian-optimization-using-botorch.md) | The batch Bayesian optimizer that proposes a whole batch of points per round, over mixed integer / float / log-float / categorical spaces. |
| [API reference](docs/api-reference.md) | `SlurmPilotExecutor`, `Task`, `RaiseOnError`, and the worker environment. |
| [How to run the tests](docs/howto-run-tests.md) | Running the suite, what is mocked and what is real, and notes for changing the tests. |

## License

MIT — see [LICENSE](LICENSE).
