# Batch Bayesian optimization with botorch

`slurm_workflows.optimize_space_botorch`
fits one Gaussian process to everything measured so far
and asks for a *batch* of points at once,
chosen jointly so they do not stack on the same spot.
A round is `search_parallelism` workers busy at once,
then one model fit, then the next round.

A run has two phases, and they are two classes:

1. **Explore.** `ExploreSpaceSobolQMC` sweeps the space with a Sobol'
   design and saves what it measured.
   See [Exploring without a model](#exploring-without-a-model).
2. **Optimize.** `OptimizeSpaceBotorch` reads that file and searches from it.

They are separate because the second is resumable:
the optimizer starts from files, so a search that ran out of walltime
carries on in the next job from what the last one saved.

botorch is an optional dependency:

```sh
pip install -U slurm-workflows[botorch]
```

```python
from slurm_workflows import SlurmPilotExecutor
from slurm_workflows.explore_space import ExplorationTask, ExploreSpaceSobolQMC
from slurm_workflows.optimize_space_botorch import (
    OptimizationTask, OptimizeSpaceBotorch,
)
from slurm_workflows.search_space import (
    IntRange, FloatRange, CategoricalRange,
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

executor = SlurmPilotExecutor(name="my-search", server_address="10.0.0.1:5051")
executor.define_worker("gpu", ["--partition gpu --gres gpu:1"], setup_script)
executor.scale_workers("gpu", 8)

# Where the model is fit. Its workers need botorch; the objective's do not.
executor.define_worker("fit", ["--partition standard --cpus-per-task 8"], fit_setup)
executor.scale_workers("fit", 1)

# Phase 1: a Sobol' sweep, saved to a file.
sweep = ExploreSpaceSobolQMC(
    [
        ExplorationTask(
            "lr-search",   # task name: labels the output, keys the results
            space,
            train,
            "gpu",         # where evaluations run (or a list of queues)
            16,            # exploration points, floored to a power of two
            20260730,      # seed (optional, drawn and printed if omitted)
            extra_objective_kwargs={"epochs": 20},
        )
    ],
    executor,
)
sweep.run_exploration_jobs()
sweep.save("explore.pkl.gz")

# Phase 2: the search, starting from that file.
opt = OptimizeSpaceBotorch(
    [
        OptimizationTask(
            "lr-search",   # the same name: this is how the file is read
            space,
            train,
            "gpu",         # objective_queue: where evaluations run
            "fit",         # optimizer_queue: where the model is fit
            8,             # points evaluated in parallel per round
            min_search_iterations=5,    # rounds that always run
            max_search_iterations=30,   # hard ceiling
            patience=3,                 # stalled rounds in a row that stop it
            min_improvement=0.05,       # what a round must beat the incumbent by
            extra_objective_kwargs={"epochs": 20},
        )
    ],
    executor,
    ["explore.pkl.gz"],
)
opt.run_search_jobs()
opt.save("search.pkl.gz")

params, value = opt.best_point("lr-search")
print(f"best loss {value} at {params}")
print(f"whole result: {opt.best_output('lr-search')}")
```

Both classes take a **list** of tasks and run them together:
two spaces are explored in one batch, and their searches share every round,
rather than one waiting for the other to finish with the pool.

The objective's argument names must match the keys of `space`,
and it takes them as keyword arguments.
It is cloudpickled to the workers like any other task,
so a closure or a lambda is fine.

**It returns a mapping, not a bare number.**
One key is mandatory: `objective`, the float to be **minimized**
— negate a score you would rather maximize.
Only `objective` is modelled; the whole mapping is recorded,
so an evaluation can also report a runtime, a checkpoint path,
or metrics it did not optimize for.
A bare float, a mapping without an `objective` key,
or an `objective` that is not a finite float
each raise rather than being coerced.

`objective_key="rmse"` changes which key is read
and nothing else: the named key is modelled,
every other key is recorded.
Use it for an objective that already reports under another name.

`extra_objective_kwargs` carries anything else the objective needs
but the search should not vary.
It may not shadow a key of `space`, which is rejected up front.

**Where the observations come from.** `OptimizeSpaceBotorch` does not
explore. It is given results files --- what `ExploreSpaceSobolQMC.save`
wrote, and what its own `save` writes --- and models everything they hold
under each task's name. A task with nothing under its name in any of them
is an error rather than a search with no model:

```
lr-search: no observations in the given files; explore the space first,
and pass what ExploreSpaceSobolQMC.save() wrote
```

The file carries the points as the objective saw them, so the unit cube
coordinates are recomputed against the space the task declares,
and a point that space cannot place is reported rather than fit on:
a parameter missing or one too many, a range since narrowed past the point,
or a value a log range cannot take.

**Resuming.** `save` writes only what *that* run measured, so the files
concatenate without double counting:

```python
# a later job, carrying on from both
opt = OptimizeSpaceBotorch(
    tasks, executor, ["explore.pkl.gz", "search.pkl.gz"]
)
opt.run_search_jobs()
opt.save("search-2.pkl.gz")
```

Each run reads every file before it and writes one of its own,
so a search that ran out of walltime resumes with everything it had,
and `best_point(name)` covers the files as well as the current run.

`run_search_jobs` runs rounds until the search stops improving.
Each round: fit a `SingleTaskGP` to every point measured so far,
ask `qLogNoisyExpectedImprovement` for the whole batch in one call,
submit all of them, and wait.

The batch is chosen jointly rather than one point at a time,
so the proposals do not stack on the same spot.
The "noisy" variant takes the points already evaluated
instead of a single best-so-far value,
and reads the incumbent off the posterior at those points,
so an evaluation that came back lucky
does not become a target the search chases.
It also carries every point measured so far,
which is why its cost grows with the run.

**Early stopping.** The number of rounds is not fixed up front.
A round is *stalled* when it fails to improve the best value
by `min_improvement` — a fraction,
measured against the magnitude of the incumbent,
so the same setting means the same thing
whether the objective is scaled in seconds or in dollars.
`patience` stalled rounds **in a row** end the search;
a round that improves resets the count,
so this bounds a run of stalled rounds rather than their total.

Two bounds fence that in.
`min_search_iterations` rounds always run,
so a search that starts slowly is not mistaken for a finished one.
It is a floor on rounds *run*, not on rounds counted:
a stalled round below it still counts towards `patience`,
it just cannot be the round that ends the search.
A search that never improves therefore stops
at `min_search_iterations` exactly,
and the earliest stop is max(`min_search_iterations`, `patience`) rounds.
`max_search_iterations` is a hard ceiling,
reached even if the search is still improving.

The search reports which bound ended it:

```
lr-search: round 7 improved by less than 5% --- 1 in a row, 2 more to stop
lr-search: round 8 improved by less than 5% --- 2 in a row, 1 more to stop
lr-search: stopping after 9 rounds --- 3 in a row without a 5% improvement
```

"more to stop" counts whichever bound is still binding:
the streak that has to reach `patience`,
or the rounds that have to reach `min_search_iterations`,
whichever is further away.
With a floor above `patience` it is the floor,
so the number can exceed `patience` itself.

Every round prints how many points the fit is over,
then what the fit and the proposal each cost:

```
lr-search: fitting GP on 64 points ...
lr-search: GP fit took 0.21s, proposed 8 points in 1.03s
```

The count is printed before the fit starts,
so it is on screen even if the fit is what stalls.
Fit duration grows superlinearly with the number of observations:
a search that slows round after round
is spending its time in the model rather than in the objective.

Both classes block until every point in flight has come back.
A worker that raises does not raise on the driver —
so both wait with `RaiseOnError.RAISE_AFTER_COMPLETED`
and turn what comes back into a `RuntimeError`
rather than feeding a `RemoteExecutionError` into the model.
Deferred rather than immediate,
so one failed evaluation does not hide the rest of its batch,
and the exception names which tasks failed.
What did come back is recorded before the exception is raised,
so `save()` still holds the batch's good points
and the next run resumes from them.
An objective returning `NaN` or `inf` is rejected the same way,
since either one silently poisons the GP fit.

**Search spaces.** They live in `slurm_workflows.search_space`,
which imports neither torch nor botorch,
so a space can be built and unit-tested without an optimizer installed.

Every parameter is mapped into `[0, 1]` before the GP
sees it and mapped back for the objective,
which is what lets one model span all four kinds at once:

| Range | Objective receives | Notes |
| --- | --- | --- |
| `FloatRange(min, max)` | `float` in `[min, max]` | |
| `FloatRange(min, max, log_range=True)` | `float` in `[min, max]` | Searched in log space, so each decade gets equal budget. Needs `min > 0`. |
| `IntRange(min, max)` | `int` in `[min, max]` | |
| `CategoricalRange(n)` | `int` in `[0, n-1]` | An index — map it to your own values, as with `OPTIMIZERS` above. `n=1` is allowed but is a dead dimension: drop the parameter and pass the value through `**extra_objective_kwargs` instead. |

Integer and categorical parameters are handled by rounding a continuous
proposal, so on a mostly-discrete space with few levels
expect the search to re-evaluate points it has already seen.
The model records where the objective *actually* ran, after rounding,
not the continuous proposal.

`run_search_jobs` can be called again for another set of rounds,
modelling everything the earlier calls measured.
Every member takes the task's name, since a run holds several:

| Member | What it is |
| --- | --- |
| `tasks` | The tasks as they will run, with the parallelism filled in. The caller's own `OptimizationTask` objects are left alone. |
| `run_search_jobs()` | Round after round, every task advancing together, until each one stops. |
| `best_point(name)` / `best_output(name)` | That task's best `(params, value)`, and the objective's whole result there, over the files as well as this run. |
| `results[name]` | What *this* run measured: `points`, `values`, `outputs` and `unit_points`, in submission order. |
| `prior[name]` | What the files held, in the same shape. |
| `observations(name)` / `num_observations(name)` | Everything the model is fit on, the two together. |
| `dim(name)` | Dimensionality of that task's space. |
| `save(path)` | Write this run's results, in the shape the files are read in. |

## Where the work runs

Nothing heavy runs on the driver.
A search round is two kinds of task on two queues,
each kind submitted for every task at once and waited for once:

| Queue | Tasks per round | Needs |
| --- | --- | --- |
| `objective_queue` | `search_parallelism` objective evaluations | whatever the objective needs |
| `optimizer_queue` | one `fit_and_propose` — the GP fit and the acquisition optimization | botorch, cores, and memory for a GP over every point measured so far |

They are separate arguments because the two want different nodes.
An evaluation is one call, `search_parallelism` at a time;
the fit is a single task that threads across cores
and whose cost grows superlinearly with the number of observations.
Pointing both at one queue is supported and cannot deadlock,
since a round never has both kinds in flight at once;
the fit then waits for a slot in a pool sized for the objective.

With several tasks in one run, a round submits every task's fit together
and then every task's batch together, so the pool is filled by all of them
rather than by whichever task is having its turn.
Tasks drop out independently as each meets its own stopping rule.

botorch has to be importable in two places:
on the driver, which imports this module,
and in the `optimizer_queue` workers' environment, which runs the fit.
Workers serving only `objective_queue` need neither botorch nor torch.
A fit that fails because they cannot import it
raises on the driver naming the queue;
the traceback is in a worker log under `executor.work_dir`.

The durations in `GP fit took Ns, proposed N points in Ns`
are measured around the calls on the worker and returned to the driver,
which is the only place the whole run is visible.

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
OptimizationTask(..., num_restarts=20, acqf_timeout_s=60.0)
```

Each is kept on the task and passed to every fit task,
so a run is tuned by constructing it
and nothing has to be redeployed to the compute nodes.
The worker never reads these for itself.

On the timeout:
proposal cost grows with the number of observations,
so an unbounded search can spend longer choosing a batch
than evaluating one.
Hitting the limit is not an error:
`optimize_acqf` returns the best candidates it has so far
— a full batch, finite and inside the bounds,
just less thoroughly optimized.
Raise it when one evaluation is expensive enough
that a better batch is worth the extra minutes.

[`examples/example_optimize_himmelblau.py`](../examples/example_optimize_himmelblau.py)
runs this on a cluster,
with an `eval` pool for the objective and a one-worker `opt` pool for the fit.

## Exploring without a model

`slurm_workflows.explore_space.ExploreSpaceSobolQMC`
is the exploration phase on its own:
a Sobol' sweep of a search space, evaluated across the pool,
with nothing fitted afterwards.
Use it for a first look at a space,
for a baseline to judge a search against,
or when the budget is one wave of evaluations and there is no second round.

It draws with scipy rather than botorch,
so it needs neither botorch nor torch — on the driver or on the workers.

A sweep takes a **list of `ExplorationTask`s**, one per space,
and runs them all at once:

```python
from slurm_workflows.explore_space import ExplorationTask, ExploreSpaceSobolQMC

sweep = ExploreSpaceSobolQMC(
    [
        ExplorationTask(
            "coarse",       # names this task: labels its output, keys its results
            SPACE,          # the same SearchSpace the optimizer takes
            train,          # the same objective, returning a mapping
            "cpu",          # the queue this task's evaluations go to
            64,             # points; floored to a power of two
            SEED,           # optional, drawn and printed if omitted
            extra_objective_kwargs={"epochs": 20},
        ),
        ExplorationTask("fine", NARROW_SPACE, train, "cpu", 256, SEED),
    ],
    executor,
    64,                     # the count for any task that did not give one
)

sweep.run_exploration_jobs()

params, value = sweep.best_point("coarse")
print(params, sweep.best_output("coarse"))
```

Every task is submitted before any of them is waited for,
so the whole sweep fills the pool at once
rather than a 64 point task holding it while a 256 point task waits.
Tasks may name different queues and are still submitted together.

| Member | What it is |
| --- | --- |
| `tasks` | The tasks as they will run, with the point count and seed filled in. The caller's own `ExplorationTask` objects are left alone. |
| `design(name)` | The points one task will evaluate, drawn but not submitted. Reproducible from the seed, so it also answers "what would this run do?" |
| `run_exploration_jobs()` | Submit every task's design, block until it is all back, print the best of each. |
| `best_point(name)` / `best_output(name)` | One task's lowest-valued point, and the objective's whole result there. |
| `results[name]` | That task's `points`, `values`, `outputs` and `unit_points`, in submission order. |
| `dim(name)` | Dimensionality of that task's space. |
| `save(path)` | Write every task's `points`, `values` and `outputs` to a gzipped pickle. |

The conventions are the optimizer's, per task:
the objective returns a mapping with its value under `objective_key`
(`"objective"` by default) and lower is better,
a non-finite or non-numeric value is an error rather than a bad point,
and one failed evaluation does not hide the rest of the sweep —
every point of every task is waited for before anything raises,
and the exception then names which tasks failed.

The point count is floored to a power of two,
because that is the prefix length at which a Sobol' sequence is balanced.
64 workers asking for 100 points would evaluate 64 and leave the rest idle,
so size the sweep in powers of two on purpose.

`save()` is for keeping the results past the end of the driver process:

```python
sweep.save("sweep-results.pkl.gz")
```

```python
import gzip, pickle

with gzip.open("sweep-results.pkl.gz", "rb") as fobj:
    results = pickle.load(fobj)

results["coarse"]["points"]    # the parameters of each evaluation
results["coarse"]["values"]    # the ranked value of each
results["coarse"]["outputs"]   # the objective's whole result for each
```

The file is keyed by task name,
and each task's three lists are index-aligned and in submission order.
It is plain `pickle`, not `cloudpickle`:
these are measurements rather than code,
so anything an objective returns
that a plain unpickler cannot rebuild does not belong in the file.

Calling `run_exploration_jobs()` twice re-evaluates the same designs:
the seed decides the draw, so there is no "next 64 points".
Pass a different seed for fresh ground.
