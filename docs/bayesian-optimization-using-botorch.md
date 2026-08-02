# Batch Bayesian optimization with botorch

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
    # Runs on a compute node. "objective" is the value to MINIMIZE;
    # anything else is recorded but not modelled.
    loss, seconds = fit_model(lr, batch_size, OPTIMIZERS[optimizer], epochs)
    return {"objective": loss, "train_seconds": seconds}


space = {
    "lr": FloatRange(1e-5, 1e-1, log_range=True),
    "batch_size": IntRange(16, 512),
    "optimizer": CategoricalRange(len(OPTIMIZERS)),
}

executor = SlurmPilotExecutor(server_address="10.0.0.1:5051")
executor.define_worker("gpu", ["--partition gpu --gres gpu:1"], setup_script)
executor.scale_workers("gpu", 8)

# Where the model is fit. Its workers need botorch; the objective's do not.
executor.define_worker("fit", ["--partition standard --cpus-per-task 8"], fit_setup)
executor.scale_workers("fit", 1)

opt = BayesOptBotorch(
    "lr-search",   # job name, used in progress bars and error messages
    space,
    train,
    executor,
    "gpu",         # objective_queue: where evaluations run (or a list)
    "fit",         # optimizer_queue: where the model is fit (or a list)
    16,            # exploration points, Sobol'
    8,             # points evaluated in parallel per round
    20260730,      # seed for the exploration design (optional)
    min_search_iterations=5,    # rounds that always run
    max_search_iterations=30,   # hard ceiling
    patience=3,                 # stalled rounds in a row that end the search
    min_improvement=0.05,       # what a round must beat the incumbent by
    epochs=20,     # extra kwargs are passed through to the objective
)

opt.run_exploration_jobs()
opt.run_search_jobs()

params, value = opt.best_point()
print(f"best loss {value} at {params}")
print(f"whole result: {opt.best_output()}")
```

The objective's argument names must match the keys of `space`,
and it takes them as keyword arguments.
It is cloudpickled to the workers like any other task,
so a closure or a lambda is fine.

**It returns a mapping, not a bare number.**
One key is mandatory: `objective`, the float to be **minimized**
— negate a score you would rather maximize.
Every other key is carried along untouched.
Only `objective` is modelled, but the whole mapping is recorded,
so an expensive evaluation can report the things you will want afterwards
— a runtime, a checkpoint path, the metrics you did not optimize for —
without having to write them somewhere else itself.
A bare float, a mapping without an `objective` key,
or an `objective` that is not a finite float
each raise rather than being coerced.

`objective_key="rmse"` changes which key is read,
and nothing else about the contract:
the named key is modelled, every other key is recorded.
It is there for the evaluation you already have —
one that reports `loss` or `rmse` and is shared with code
that expects to keep calling it that —
so it can be searched as it is rather than wrapped to rename a key.

**Two phases.** `run_exploration_jobs` submits a scrambled Sobol' design
in one batch — a space-filling sweep that costs one round trip
and gives the GP something to model.
The count is truncated down to a power of two,
because that is where a Sobol' sequence is balanced.
The design is scrambled with the `seed` you pass,
so a rerun with the same seed explores the same points
and a second run with a different seed explores fresh ones.

The seed is optional. Omit it and one is drawn from `os.urandom`,
then printed and kept in `opt.seed`:

```
lr-search: no seed given, drew 1778551843323245695 --- pass it back to repeat this run
```

So an exploratory run still costs nothing to start,
and is still repeatable afterwards —
pass that number back as the seed and you get the same design.
It is drawn from `os.urandom` rather than the `random` module,
so seeding the global RNG elsewhere in your script
cannot quietly make every "unseeded" run identical.
All arguments up to and including the seed are positional-only,
which is what frees `**extra_objective_kwargs`
to use the optimizer's own parameter names.
They may not shadow a key of `space`, which is rejected up front.

`run_search_jobs` then runs rounds until the search stops paying. Each round:
fit a `SingleTaskGP` to every point measured so far,
ask `qLogNoisyExpectedImprovement` for the whole batch in one call,
submit all of them, and wait.

The batch is chosen jointly rather than one point at a time,
so the proposals do not stack on the same spot.
The "noisy" variant takes the points already evaluated
instead of a single best-so-far value,
and reads the incumbent off the posterior at those points ---
so an evaluation that came back lucky
cannot become a target the search then chases.
It also means the acquisition carries every point measured so far,
which is why its cost grows with the run.

**Early stopping.** How many rounds run is not fixed up front,
because how many are worth running is not knowable up front.
A round is *stalled* when it fails to improve the best value
by `min_improvement` — a fraction,
measured against the magnitude of the incumbent,
so the same setting means the same thing
whether the objective is scaled in seconds or in dollars.
`patience` stalled rounds **in a row** end the search;
a round that improves resets the count,
so this bounds a run of bad rounds rather than their total.

Two bounds fence that in.
`min_search_iterations` rounds always run,
so a search that starts slowly is not mistaken for a finished one.
It is a floor on rounds *run*, not on rounds counted:
a stalled round below it still counts towards `patience`,
it just cannot be the round that ends the search.
So a search that never improves at all stops
at `min_search_iterations` exactly,
and the earliest possible stop
is max(`min_search_iterations`, `patience`) rounds.
`max_search_iterations` is a hard ceiling,
reached even if the search is still improving.

The search says which of the two ended it:

```
lr-search: round 7 improved by less than 5% --- 1 in a row, 2 more to stop
lr-search: round 8 improved by less than 5% --- 2 in a row, 1 more to stop
lr-search: stopping after 9 rounds --- 3 in a row without a 5% improvement
```

"more to stop" counts whichever bound is still binding —
the streak that has to reach `patience`,
or the rounds that have to reach `min_search_iterations`,
whichever is further away.
Early in a run with a floor above `patience` it is the floor,
which is why the number can be larger than `patience` itself.

Every round prints how many points the fit is over,
then what the fit and the proposal each cost:

```
lr-search: fitting GP on 64 points ...
lr-search: GP fit took 0.21s, proposed 8 points in 1.03s
```

The count is announced before the fit starts,
so it is on screen even if the fit is the thing that stalls.
Watch the duration: it grows superlinearly with the number of observations,
so a search that slows down round after round
is spending its time in the model rather than in your objective.

Both phases block until every point in flight has come back.
A worker that raises does not raise on the driver —
so both phases run `check_for_error` themselves
and raise a `RuntimeError`
rather than feeding a `RemoteExecutionError` into the model.
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

`run_search_jobs` needs something to model,
so it raises unless points have already been evaluated
— call `run_exploration_jobs` first.
It can then be called again for another run of rounds,
modelling everything the earlier calls measured.
`best_point()` returns the best `(params, value)` seen by either phase,
and raises if nothing has been evaluated yet.
`best_output()` returns the whole mapping the objective produced there,
and `outputs` holds one such mapping per evaluation,
in the same order as `points` and `values`.

## Where the work runs

Nothing heavy runs on the driver.
A search round is two kinds of task on two queues:

| Queue | Tasks per round | Needs |
| --- | --- | --- |
| `objective_queue` | `search_parallelism` objective evaluations | whatever the objective needs |
| `optimizer_queue` | one `fit_and_propose` — the GP fit and the acquisition optimization | botorch, cores, and memory for a GP over every point measured so far |

They are separate arguments because the two want different nodes.
An evaluation is one call, and there are `search_parallelism` at a time;
the fit is a single task whose cost grows with the run
— superlinear in the number of observations —
and which threads across cores.
Pointing both at one queue is supported and cannot deadlock,
since a round never has both kinds in flight at once;
the fit then just waits for a slot in a pool sized for the objective.

botorch has to be importable in two places:
on the driver, which imports this module,
and in the `optimizer_queue` workers' environment, which now runs the fit.
Workers serving only `objective_queue` need neither botorch nor torch.
A fit that fails because they cannot import it
raises on the driver naming the queue —
the traceback itself is in a worker log under `executor.work_dir`.

The driver still prints `fitting GP on N points`, `GP fit took Ns`
and `proposed N points in Ns` every round:
the durations are measured around the calls on the worker
and handed back, since a search that slows down round after round
is the thing worth watching, and only the driver sees all of it.

### Tuning the fit

Four keyword arguments tune the acquisition optimization,
all of them settings of one run rather than of the process:

| Argument | Default | What it buys |
| --- | --- | --- |
| `num_restarts` | 10 | Multi-start count for `optimize_acqf`. The acquisition surface is multimodal, so a single start routinely lands in a local optimum. |
| `raw_samples` | 128 | Candidates drawn to pick those starting points from. |
| `mc_samples` | 128 | Quasi-MC draws per acquisition evaluation. Sobol' draws are stratified, so they carry further than the same number of independent normal samples. |
| `acqf_timeout_s` | 10.0 | Wall-clock budget for one `optimize_acqf` call. |

```python
opt = BayesOptBotorch(
    ..., num_restarts=20, acqf_timeout_s=60.0,
)
```

Each is kept on the optimizer and passed to every fit task,
so a run is tuned by constructing it
and nothing has to be redeployed to the compute nodes.
The worker never reads these for itself —
it is told what this run asked for.

The timeout deserves a word.
Proposal cost grows with the number of observations,
so an unbounded search can spend longer choosing the next batch
than the batch takes to evaluate.
Hitting the limit is not an error:
`optimize_acqf` returns the best candidates it has so far
— still a full batch, still finite and inside the bounds —
just less thoroughly optimized.
A slightly worse proposal costs one round; a stalled driver costs the run.
Raise it when an evaluation is expensive enough
that a better batch is worth minutes of thinking.

[`examples/example_optimize_himmelblau.py`](../examples/example_optimize_himmelblau.py)
runs the whole thing on a cluster,
with an `eval` pool for the objective and a one-worker `opt` pool for the fit.
It gains a batch chosen jointly and pays for it with a barrier per round.
