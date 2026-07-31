"""Batch Bayesian optimization on Rivanna's BII cluster.

A follow-up to `compute_pi_bii.py`.
That example covers the machinery this one reuses without comment ---
starting `ds-service`, the `ib0` address, `setup_script`,
what the `sbatch` arguments mean, and how cleanup works.
Read it first.

What is different here is *why* work is submitted.

`compute_pi_bii.py` knew every task up front:
four thousand independent slices, submitted in one go,
and the only question was how fast the pool could compute them.
An optimization workflow does not work that way.
You are searching for the input that minimizes some expensive function,
you cannot afford to evaluate the whole space,
and which point is worth trying next
depends on what the previous points returned.
That makes the run a sequence of *rounds*
rather than one flat pile of tasks.

`BayesOptBotorch` fits a Gaussian process to everything measured so far,
asks it for a whole batch of promising points at once,
evaluates that batch across the pilot pool,
refits, and repeats.
The batch is what keeps the cluster busy:
a one-point-at-a-time optimizer would leave 9 of 10 workers idle.

Where the work happens:

* the model is fit **on the login node**, so botorch is needed here;
* only the objective evaluations are shipped to the compute nodes,
  which need `slurm-workflows` but not botorch.

Run it from a Rivanna login node:

    module load apptainer/1.4.5
    pip install slurm-workflows[botorch]
    python example_optimize_himmelblau.py

For the full API --- integer, categorical and log-scaled parameters,
resuming a search, the acquisition functions ---
see `docs/bayesian-optimization-using-botorch.md`.
"""

from ds_service_client import DsServiceServer
from slurm_workflows import SlurmPilotExecutor
from slurm_workflows.bayes_opt_botorch import BayesOptBotorch, FloatRange

DS_SERVICE_BIN = "apptainer run /project/bii_nssac/people/pb5gj/shared/ds-service/latest/ds-service.sif"

SETUP_SCRIPT = ""

NUM_NODES = 1
TASKS_PER_NODE = 40

SBATCH_ARGS = [
    "--account=bii_nssac",
    f"--partition=bii --nodes={NUM_NODES}",
    f"--ntasks-per-node={TASKS_PER_NODE} --cpus-per-task=1 --mem=0",
    "--time=1:00:00",
]

JOB_NAME = "himmelblau"

# Scrambling the exploration design is random,
# but seeded: the same seed redraws the same starting points,
# so a rerun is comparable and a different seed explores fresh ground.
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
# It buys the model something to fit before it starts making decisions;
# a GP with no observations has no opinion worth acting on.
# Truncated down to a power of two, because that is where Sobol' is balanced,
# so 64 stays 64 but 100 would quietly become 64.
EXPLORATION_POINTS = NUM_NODES * TASKS_PER_NODE * 10

# Phase 2: the actual optimization.
# `SEARCH_PARALLELISM` is the batch size, so match it to the pool ---
# a bigger batch queues behind the workers,
# a smaller one leaves workers idle.
SEARCH_PARALLELISM = NUM_NODES * TASKS_PER_NODE
SEARCH_POINTS = SEARCH_PARALLELISM * 10

# Himmelblau's function has four global minima, all with f = 0.
KNOWN_MINIMA = [
    (3.0, 2.0),
    (-2.805118, 3.131312),
    (-3.779310, -3.283186),
    (3.584428, -1.848126),
]


def himmelblau(x, y):
    """The objective. Runs on a compute node, once per point.

    Its argument names have to match the keys of `SEARCH_SPACE`,
    and it must return a float to be **minimized**
    --- negate a score you would rather maximize.

    Standing in for something slow here.
    Bayesian optimization earns its overhead
    when one evaluation costs minutes;
    on arithmetic this cheap the model fits dominate the runtime.
    """
    return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2


def report(opt, phase):
    """Print the best point known after a phase."""
    params, value = opt.best_point()
    print(
        f"after {phase}: {len(opt.values)} points, "
        f"best f = {value:.6g} at "
        f"x = {params['x']:.4f}, y = {params['y']:.4f}"
    )


def main():
    with DsServiceServer(ds_service_bin=DS_SERVICE_BIN) as ds_service:
        ds_service.wait_until_ready()
        address = ds_service.get_address_by_interface("ib0")

        with SlurmPilotExecutor(address) as executor:
            executor.define_worker(
                name="bii",
                sbatch_args=SBATCH_ARGS,
                setup_script=SETUP_SCRIPT,
            )

            executor.scale_workers("bii", 1)

            # Constructing the optimizer submits nothing.
            # It is handed the executor and a queue name,
            # and drives `submit` / `wait` itself from here on.
            opt = BayesOptBotorch(
                JOB_NAME,
                SEARCH_SPACE,
                himmelblau,
                executor,
                "bii",
                EXPLORATION_POINTS,
                SEARCH_POINTS,
                SEARCH_PARALLELISM,
                SEED,
                # Any further keyword arguments are forwarded to the objective
                # unchanged, for the constants it needs but the search
                # should not vary.
            )

            # Both phases block until every point in flight is back,
            # and both raise if a worker failed
            # rather than letting a broken value into the model.
            print(
                f"\n=== exploration: {opt.num_exploration_points} points, one batch ==="
            )
            opt.run_exploration_jobs()
            report(opt, "exploration")

            # Each round is a barrier: fit, propose a batch, evaluate, refit.
            # That is the price of choosing the batch jointly,
            # and why the round should be as wide as the pool.
            n_rounds = -(-SEARCH_POINTS // SEARCH_PARALLELISM)  # ceil
            print(f"\n=== search: {SEARCH_POINTS} points over {n_rounds} rounds ===")
            opt.run_search_jobs()
            report(opt, "search")

            # `run_search_jobs()` may be called again to spend another
            # SEARCH_POINTS, modelling everything measured so far.

    # The pool is cancelled and the queue is gone,
    # but the optimizer kept every point it evaluated,
    # so the result is an ordinary local value.
    params, value = opt.best_point()
    nearest = min(
        KNOWN_MINIMA,
        key=lambda m: (m[0] - params["x"]) ** 2 + (m[1] - params["y"]) ** 2,
    )
    print(f"\nbest f = {value:.6g} (true minimum is 0)")
    print(f"  found at x = {params['x']:.4f}, y = {params['y']:.4f}")
    print(f"  nearest known minimum: x = {nearest[0]:.4f}, y = {nearest[1]:.4f}")


if __name__ == "__main__":
    main()
