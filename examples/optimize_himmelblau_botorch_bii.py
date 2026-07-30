"""Optimize a 2-D function on Rivanna with batch Bayesian optimization.

Run it from a Rivanna login node (botorch is needed *here*, not on the compute nodes):

    pip install slurm-workflows[botorch]
    python optimize_himmelblau_botorch_bii.py
"""

from slurm_workflows import DsService, SlurmPilotExecutor
from slurm_workflows.bayes_opt_botorch import BayesOptBotorch, FloatRange

SERVER_EXE = "apptainer run /project/bii_nssac/people/pb5gj/shared/ds-service/latest/ds-service.sif"

SBATCH_ARGS = [
    "--account=bii_nssac --qos=bii-unlimited",
    "--partition=bii --nodes=1",
    "--ntasks-per-node=1 --cpus-per-task=40 --mem=0",
    "--time=1:00:00",
]

NUM_NODES = 10

JOB_NAME = "himmelblau"

SEED = 20260730

SEARCH_SPACE = {
    "x": FloatRange(-5.0, 5.0),
    "y": FloatRange(-5.0, 5.0),
}

EXPLORATION_POINTS = 64
SEARCH_POINTS = NUM_NODES * 10
SEARCH_PARALLELISM = NUM_NODES

# Himmelblau's function has four global minima, all with f = 0.
KNOWN_MINIMA = [
    (3.0, 2.0),
    (-2.805118, 3.131312),
    (-3.779310, -3.283186),
    (3.584428, -1.848126),
]


def himmelblau(x, y):
    """Himmelblau's function --- four global minima, all at f = 0."""
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
    with DsService(server_exe=SERVER_EXE) as ds_service:
        ds_service.start()
        ds_service.wait_until_ready()
        address = ds_service.get_address("ib0")

        executor = SlurmPilotExecutor(address)
        executor.define_worker(name="bii", sbatch_args=SBATCH_ARGS)

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
        )

        executor.scale_workers("bii", NUM_NODES)
        try:
            print(
                f"\n=== exploration: {opt.num_exploration_points} points, one batch ==="
            )
            opt.run_exploration_jobs()
            report(opt, "exploration")

            n_rounds = -(-SEARCH_POINTS // SEARCH_PARALLELISM)  # ceil
            print(f"\n=== search: {SEARCH_POINTS} points over {n_rounds} rounds ===")
            opt.run_search_jobs()
            report(opt, "search")
        finally:
            executor.scale_workers("bii", 0)

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
