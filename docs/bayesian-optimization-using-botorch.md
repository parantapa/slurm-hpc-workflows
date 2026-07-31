# Batch Bayesian optimization with botorch

[← back to the main README](../README.md)

`slurm_workflows.bayes_opt_botorch`
fits one Gaussian process to everything measured so far
and asks for a *batch* of points at once,
chosen jointly so they do not stack on the same spot.
That is the shape a pilot pool wants —
`search_parallelism` workers busy on one round,
then one model fit, then the next round.

botorch is an optional dependency:

```sh
pip install -U slurm-workflows[botorch]
```

```python
from slurm_workflows import SlurmPilotExecutor
from slurm_workflows.bayes_opt_botorch import (
    BayesOptBotorch, IntRange, FloatRange, CategoricalRange,
)

OPTIMIZERS = ["adam", "sgd"]


def train(lr, batch_size, optimizer, *, epochs):
    # Runs on a compute node. Returns the value to MINIMIZE.
    return validation_loss(lr, batch_size, OPTIMIZERS[optimizer], epochs)


space = {
    "lr": FloatRange(1e-5, 1e-1, log_range=True),
    "batch_size": IntRange(16, 512),
    "optimizer": CategoricalRange(len(OPTIMIZERS)),
}

executor = SlurmPilotExecutor(server_address="10.0.0.1:5051")
executor.define_worker("gpu", ["--partition gpu --gres gpu:1"], setup_script)
executor.scale_workers("gpu", 8)

opt = BayesOptBotorch(
    "lr-search",   # job name, used in progress bars and error messages
    space,
    train,
    executor,
    "gpu",         # queue (or a list of them)
    16,            # exploration points, Sobol'
    64,            # search points, Bayesian
    8,             # points evaluated in parallel per round
    20260730,      # seed for the exploration design
    epochs=20,     # extra kwargs are passed through to the objective
)

opt.run_exploration_jobs()
opt.run_search_jobs()

params, value = opt.best_point()
print(f"best loss {value} at {params}")
```

The objective's argument names must match the keys of `space`,
it takes them as keyword arguments,
and it returns a float to be **minimized**.
It is cloudpickled to the workers like any other task,
so a closure or a lambda is fine.

**Two phases.** `run_exploration_jobs` submits a scrambled Sobol' design
in one batch — a space-filling sweep that costs one round trip
and gives the GP something to model.
The count is truncated down to a power of two,
because that is where a Sobol' sequence is balanced.
The design is scrambled with the `seed` you pass,
so a rerun with the same seed explores the same points
and a second run with a different seed explores fresh ones.
All arguments up to and including the seed are positional-only,
which is what frees `**extra_objective_kwargs`
to use the optimizer's own parameter names.
They may not shadow a key of `space`, which is rejected up front.

`run_search_jobs` then loops until its budget is spent. Each round:
fit a `SingleTaskGP` to every point measured so far,
propose half the batch with `qLogExpectedImprovement`
and half with `qProbabilityOfImprovement`
(an odd batch gives the extra point to qLogEI),
submit all of them, and wait.
The two acquisitions are deliberate:
qLogEI is the workhorse, and qPI is greedier,
so a round spends part of its budget refining the current best.

Both phases block until every point in flight has come back.
A worker that raises does not raise on the driver —
so both phases run `check_for_error` themselves
and raise a `RuntimeError` rather than feeding a
`RemoteExecutionError` into the model.
An objective returning `NaN` or `inf` is rejected the same way,
since either one silently poisons the GP fit.

**Search spaces.** Every parameter is mapped into `[0, 1]` before the GP
sees it and mapped back for the objective,
which is what lets one model span all four kinds at once:

| Range | Objective receives | Notes |
| --- | --- | --- |
| `FloatRange(min, max)` | `float` in `[min, max]` | |
| `FloatRange(min, max, log_range=True)` | `float` in `[min, max]` | Searched in log space, so each decade gets equal budget. Needs `min > 0`. |
| `IntRange(min, max)` | `int` in `[min, max]` | |
| `CategoricalRange(n)` | `int` in `[0, n-1]` | An index — map it to your own values, as with `OPTIMIZERS` above. |

Integer and categorical parameters are handled by rounding a continuous
proposal, so on a mostly-discrete space with few levels
expect the search to re-evaluate points it has already seen.
The model records where the objective *actually* ran, after rounding,
not the continuous proposal.

`run_search_jobs` needs something to model, so it raises unless points have
already been evaluated — call `run_exploration_jobs` first.
It can then be called again to spend another `num_search_points`,
modelling everything the earlier calls measured.
`best_point()` returns the best `(params, value)` seen by either phase,
and raises if nothing has been evaluated yet.

[`examples/optimize_himmelblau_botorch_bii.py`](../examples/optimize_himmelblau_botorch_bii.py)
runs the whole thing on a cluster. The model stays on the login node
and only the evaluations are shipped out,
so it gains a batch chosen jointly and pays for it with a barrier per round.
