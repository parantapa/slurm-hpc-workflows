"""Batch Bayesian optimization on Rivanna's BII cluster.

A follow-up to `example_compute_pi.py`,
which covers the machinery this one reuses without comment:
starting `ds-service`, the `ib0` address, `setup_script`,
what the `sbatch` arguments mean, and how cleanup works.

The difference is how work is chosen.
`example_compute_pi.py` knew every task up front and submitted them in one go.
An optimization run cannot: which point is worth trying next
depends on what the previous points returned,
so the run is a sequence of *rounds*.

`BayesOptBotorch` fits a Gaussian process to everything measured so far,
asks it for a whole batch of points at once,
evaluates that batch across the pilot pool,
refits, and repeats.
The batch is what keeps the pool busy:
a one-point-at-a-time optimizer would leave all but one worker idle.

Where the work happens:

* the driver here only submits, waits and keeps the record;
* the objective evaluations go to the `eval` pool, a worker per task slot;
* the model fit and the acquisition optimization go to a *second* pool,
  `opt`, as one task per round.

Two worker groups rather than one, because the two want different nodes.
An objective evaluation is one cheap single-threaded call,
40 at a time;
the fit is a single task that wants cores and memory,
and gets more expensive every round as the model grows.
`optimizer_queue="eval"` would also work
--- the fit and the evaluations never run at the same time ---
but the fit would wait for a slot in a pool sized for the objective.

botorch is needed on the login node, to import the optimizer,
and in the `opt` group's environment, which runs the fit.
The `eval` workers need neither.

Run it from a Rivanna login node:

    pip install slurm-workflows[botorch]
    python example_optimize_himmelblau.py

For the full API --- integer, categorical and log-scaled parameters,
resuming a search, the acquisition functions ---
see `docs/bayesian-optimization-using-botorch.md`.
"""

import math

from ds_service_client import DsServiceServer
from slurm_workflows import SlurmPilotExecutor
from slurm_workflows.bayes_opt_botorch import BayesOptBotorch, FloatRange

SETUP_SCRIPT = ""

# The environment the fit runs in.
# Same shape as SETUP_SCRIPT, but it has to bring botorch with it:
# this is the group that imports `bayes_opt_botorch` on a compute node.
#
# Aliased to SETUP_SCRIPT because both are empty here,
# which works only if `/etc/profile` already puts botorch on the path.
# If the `opt` group's fits fail to import it, this is the line to fill in
# --- a `conda activate` of an environment that has botorch, typically.
OPTIMIZER_SETUP_SCRIPT = SETUP_SCRIPT

NUM_NODES = 1
TASKS_PER_NODE = 40

SBATCH_ARGS = [
    "--account=bii_nssac",
    f"--partition=bii --nodes={NUM_NODES}",
    f"--ntasks-per-node={TASKS_PER_NODE} --cpus-per-task=1 --mem=0",
    "--time=1:00:00",
]

# One worker, and nothing else on the node.
# `--ntasks-per-node=1` because the fit is a single task:
# a second worker here would sit idle all run.
# The cores still matter --- torch threads the GP fit's linear algebra.
OPTIMIZER_SBATCH_ARGS = [
    "--account=bii_nssac",
    "--partition=bii --nodes=1",
    "--ntasks-per-node=1 --cpus-per-task=40 --mem=0",
    "--time=1:00:00",
]

JOB_NAME = "himmelblau"

# The exploration design is scrambled, but seeded:
# the same seed redraws the same starting points,
# a different seed explores fresh ground.
# The argument is optional --- leave it out
# and a seed is drawn from os.urandom and printed,
# so the run can still be repeated afterwards.
SEED = 20260730

# One entry per objective argument, keyed by the argument's *name*.
# `FloatRange(-5.0, 5.0)` is a plain continuous interval;
# integer, categorical and log-scaled parameters exist too,
# and one search may mix all four kinds.
SEARCH_SPACE = {
    "x": FloatRange(-5.0, 5.0),
    "y": FloatRange(-5.0, 5.0),
}

# Phase 1: a space-filling Sobol' sweep, evaluated in a single batch.
# It gives the model something to fit before it starts making decisions.
#
# The count is truncated down to a power of two,
# because that is where a Sobol' sequence is balanced.
# Hence a literal rather than the pool size:
# 40 workers would ask for 40, evaluate 32,
# and leave 8 idle for the whole sweep without saying so.
# 64 is the next power of two up: one full wave, then 24.
EXPLORATION_POINTS = 64

# Phase 2: the optimization itself.
# `SEARCH_PARALLELISM` is the batch size, so match it to the pool:
# a bigger batch queues behind the workers,
# a smaller one leaves workers idle.
SEARCH_PARALLELISM = NUM_NODES * TASKS_PER_NODE

# The search budget is counted in *rounds*, not points:
# each round fits the model once and evaluates SEARCH_PARALLELISM points.
#
# It stops on whichever comes first, the ceiling or the early stop.
# A round that fails to beat the incumbent by MIN_IMPROVEMENT is stalled;
# PATIENCE stalled rounds in a row end the search.
# MIN_SEARCH_ITERATIONS rounds always run,
# so a slow start is not mistaken for a finished search.
# Stalled rounds below that floor still count towards PATIENCE
# but cannot be the round that stops the search,
# so the earliest stop is max(MIN_SEARCH_ITERATIONS, PATIENCE) rounds.
MIN_SEARCH_ITERATIONS = 5
MAX_SEARCH_ITERATIONS = 30
PATIENCE = 3
MIN_IMPROVEMENT = 0.05

# Himmelblau's function has four global minima, all with f = 0.
KNOWN_MINIMA = [
    (3.0, 2.0),
    (-2.805118, 3.131312),
    (-3.779310, -3.283186),
    (3.584428, -1.848126),
]


def himmelblau(x, y):
    """The objective. Runs on a compute node, once per point.

    Its argument names have to match the keys of `SEARCH_SPACE`.

    It returns a mapping, not a bare number.
    "objective" is mandatory and is the value to be **minimized**
    --- negate a score you would rather maximize.
    The optimizer models only "objective" but records the whole mapping,
    so an evaluation can also report a runtime,
    a checkpoint path, or intermediate metrics.

    This stands in for something slow.
    Bayesian optimization earns its overhead
    when one evaluation costs minutes;
    on arithmetic this cheap the model fits dominate the runtime.
    """
    value = (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2
    return {"objective": value, "distance_from_origin": math.hypot(x, y)}


def main():
    with DsServiceServer(interface="ib0") as ds_service:
        ds_service.wait_until_ready()
        address = ds_service.address

        with SlurmPilotExecutor(address) as executor:
            executor.define_worker(
                name="eval",
                sbatch_args=SBATCH_ARGS,
                setup_script=SETUP_SCRIPT,
            )

            # The pool the fit runs in. One job, one worker.
            # Defined as a separate group so it can have its own nodes
            # and its own environment --- this is the one that needs botorch.
            executor.define_worker(
                name="opt",
                sbatch_args=OPTIMIZER_SBATCH_ARGS,
                setup_script=OPTIMIZER_SETUP_SCRIPT,
            )

            executor.scale_workers("eval", 1)
            executor.scale_workers("opt", 1)

            # Constructing the optimizer submits nothing.
            # It is handed the executor and the two queue names,
            # and drives `submit` / `wait` itself from here on.
            opt = BayesOptBotorch(
                JOB_NAME,
                SEARCH_SPACE,
                himmelblau,
                executor,
                "eval",
                "opt",
                EXPLORATION_POINTS,
                SEARCH_PARALLELISM,
                SEED,
                min_search_iterations=MIN_SEARCH_ITERATIONS,
                max_search_iterations=MAX_SEARCH_ITERATIONS,
                patience=PATIENCE,
                min_improvement=MIN_IMPROVEMENT,
                # Any further keyword arguments are forwarded to the objective
                # unchanged, for the constants it needs but the search
                # should not vary.
            )

            # Both phases block until every point in flight is back,
            # and both raise if a worker failed
            # rather than letting a broken value into the model.
            #
            # Neither needs a progress report written here:
            # the optimizer prints the best point after every batch,
            # plus how long each fit and each proposal took.
            # A best that stops moving while the fits keep growing
            # means the budget is going to the model, not the search.
            print(
                f"\n=== exploration: {opt.num_exploration_points} points, one batch ==="
            )
            opt.run_exploration_jobs()

            # Each round is a barrier: fit, propose a batch, evaluate, refit.
            # That is the cost of choosing the batch jointly,
            # and why the round should be as wide as the pool.
            print(
                f"\n=== search: up to {MAX_SEARCH_ITERATIONS} rounds"
                f" of {SEARCH_PARALLELISM} points ==="
            )
            opt.run_search_jobs()

            # `run_search_jobs()` may be called again
            # for another run of up to MAX_SEARCH_ITERATIONS rounds,
            # modelling everything measured so far.

    # The pool is cancelled and the queue is gone,
    # but the optimizer kept every point it evaluated,
    # so the result is an ordinary local value.
    params, value = opt.best_point()
    nearest = min(
        KNOWN_MINIMA,
        key=lambda m: (m[0] - params["x"]) ** 2 + (m[1] - params["y"]) ** 2,
    )
    print(f"\nbest f = {value:.6g} (true minimum is 0)")
    print(f"  full result: {opt.best_output()}")
    print(f"  found at x = {params['x']:.4f}, y = {params['y']:.4f}")
    print(f"  nearest known minimum: x = {nearest[0]:.4f}, y = {nearest[1]:.4f}")


if __name__ == "__main__":
    main()
