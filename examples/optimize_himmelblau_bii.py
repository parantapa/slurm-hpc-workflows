"""Optimize a 2-D function on Rivanna, in three sampling phases.

One Optuna study, shared through ds-service, is worked on by every pilot
worker at once. The study is filled in three passes, each with a different
sampler, and each pass sees everything the previous ones recorded:

1. `ExtremePointSampler` --- the corners of the box. Four trials that cost
   nothing and tell you the objective's range, which parameters it is
   monotone in, and whether it survives its own extremes.
2. `DsServiceQMCSampler` --- a low-discrepancy sweep. Spreads a fixed
   budget over the whole box far more evenly than random search would.
3. `TPESampler` --- the actual optimization, warm-started by the 68 trials
   above. TPE random-samples until a study has `n_startup_trials`
   observations to model; the probe phases have already paid that off, so
   it starts modelling from its very first trial.

The objective here is Himmelblau's function, which is instant to evaluate
--- so this example is dominated by scheduling overhead and tells you
nothing about speed. Its point is the structure. Pilot jobs are worth it
when a single evaluation costs minutes: a training run, a simulation, a
solver.

Run it from a Rivanna login node:

    python optimize_himmelblau_bii.py
"""

import optuna
from optuna.samplers import TPESampler
from optuna.distributions import FloatDistribution

from ds_service_client import DsServiceServer
from slurm_workflows import SlurmPilotExecutor, check_for_error
from slurm_workflows.optuna_storage import create_optuna_storage
from slurm_workflows.optuna_qmc_sampler import DsServiceQMCSampler
from slurm_workflows.optuna_extreme_point_sampler import ExtremePointSampler

SERVER_EXE = "apptainer run /project/bii_nssac/people/pb5gj/shared/ds-service/latest/ds-service.sif"

TASK_PER_NODE = 40
SBATCH_ARGS = [
    "--account=bii_nssac --qos=bii-unlimited",
    "--partition=bii --nodes=1",
    "--ntasks-per-node=40 --cpus-per-task=1 --mem=0",
    "--time=1:00:00",
]

NUM_NODES = 2

# Keys on the ds-service server. The prefix namespaces this study's journal
# and both samplers' counters; a second study on the same server just needs
# a different one.
PREFIX = "himmelblau"
STUDY_NAME = "himmelblau"

# Declared up front because both probe samplers need it: the corner sampler
# to know how many corners there are, the QMC sampler so that its very
# first trial is already on the sequence instead of a random warm-up.
SEARCH_SPACE = {
    "x": FloatDistribution(-5.0, 5.0),
    "y": FloatDistribution(-5.0, 5.0),
}

QMC_TRIALS = 64  # a power of two: where Sobol' has its lowest discrepancy
TPE_TRIALS = 512

# Himmelblau's function has four global minima, all with f = 0.
KNOWN_MINIMA = [(3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]


def objective(trial):
    """Himmelblau's function --- four global minima, all at f = 0."""
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2


def make_sampler(phase, storage):
    """The sampler for one phase.

    Built on the worker rather than shipped to it: the samplers are
    picklable, but constructing them where they run keeps the phase name as
    the only thing that has to travel.
    """
    if phase == "corners":
        return ExtremePointSampler(storage, SEARCH_SPACE)
    if phase == "qmc":
        # scramble=True is safe here: workers agree on the seed through the
        # server, so nobody has to pass the same integer to every worker.
        return DsServiceQMCSampler(storage, scramble=True, search_space=SEARCH_SPACE)
    if phase == "tpe":
        # constant_liar keeps concurrent workers from all proposing the same
        # point while each other's trials are still running.
        return TPESampler(constant_liar=True)
    raise ValueError(f"unknown phase: {phase}")


def run_trials(server_address, phase, n_trials):
    """Run a slice of one phase. This is what executes on a compute node."""
    storage = create_optuna_storage(server_address, prefix=PREFIX)
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=storage,
        sampler=make_sampler(phase, storage),
    )
    study.optimize(objective, n_trials=n_trials)
    return len(study.trials)


def split(n_trials, n_workers):
    """Divide `n_trials` as evenly as possible over at most `n_workers` tasks."""
    n_tasks = min(n_trials, n_workers)
    base, extra = divmod(n_trials, n_tasks)
    return [base + (1 if i < extra else 0) for i in range(n_tasks)]


def run_phase(executor, address, phase, n_trials, n_workers):
    """Submit one phase, wait for it, and report what it found."""
    print(f"\n=== {phase}: {n_trials} trials over {min(n_trials, n_workers)} tasks ===")

    tasks = [
        executor.submit("bii", run_trials, address, phase, chunk)
        for chunk in split(n_trials, n_workers)
    ]
    executor.wait(tasks, desc=phase)

    # A worker that raised does not raise here; its exception comes back as
    # the task's output.
    if check_for_error(tasks):
        raise RuntimeError(f"phase {phase} had failed tasks")


def report(study, phase):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = study.best_trial
    print(
        f"after {phase}: {len(trials)} trials, "
        f"best f = {best.value:.6g} at "
        f"x = {best.params['x']:.4f}, y = {best.params['y']:.4f}"
    )


def main():
    with DsServiceServer(ds_service_bin=SERVER_EXE) as ds_service:
        ds_service.wait_until_ready()
        address = ds_service.get_address_by_interface("ib0")

        # The driver's own handle on the study. Workers open their own
        # against the same address and prefix; the journal is what they
        # share, so this one sees their trials as soon as it reads again.
        storage = create_optuna_storage(address, prefix=PREFIX)
        study = optuna.create_study(
            storage=storage,
            study_name=STUDY_NAME,
            direction="minimize",
            load_if_exists=True,
        )

        n_corners = ExtremePointSampler(storage, SEARCH_SPACE).n_corners

        executor = SlurmPilotExecutor(address)
        executor.define_worker(name="bii", sbatch_args=SBATCH_ARGS)
        executor.scale_workers("bii", NUM_NODES)
        n_workers = NUM_NODES * TASK_PER_NODE

        try:
            run_phase(executor, address, "corners", n_corners, n_workers)
            report(study, "corners")

            run_phase(executor, address, "qmc", QMC_TRIALS, n_workers)
            report(study, "qmc")

            run_phase(executor, address, "tpe", TPE_TRIALS, n_workers)
            report(study, "tpe")
        finally:
            executor.scale_workers("bii", 0)

        best = study.best_trial
        nearest = min(
            KNOWN_MINIMA,
            key=lambda m: (m[0] - best.params["x"]) ** 2 + (m[1] - best.params["y"]) ** 2,
        )
        print(f"\nbest f = {best.value:.6g} (true minimum is 0)")
        print(f"  found at x = {best.params['x']:.4f}, y = {best.params['y']:.4f}")
        print(f"  nearest known minimum: x = {nearest[0]:.4f}, y = {nearest[1]:.4f}")


if __name__ == "__main__":
    main()
