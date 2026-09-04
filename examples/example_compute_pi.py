"""Compute pi on Rivanna's BII cluster --- a worked example.

This is the smallest thing that exercises the whole pipeline:
a login-node driver, a task queue, and a pool of pilot workers
on `bii` compute nodes.
The arithmetic is deliberately trivial
so that what you are reading is the *mechanism*.

Run it from a Rivanna or BII login node:

    pip install slurm-workflows
    python example_compute_pi.py

Before it will work you need two things.

1. `slurm-workflows` importable **on the compute nodes**, not just here.
   The compute nodes do not inherit your login shell,
   so whatever activates your environment
   has to run inside the job --- see `SETUP_SCRIPT` below.
2. `ds-service` on the login node's PATH.
   `DsServiceServer` runs it from there.

What actually happens, in order:

* a `ds-service` task queue starts on the login node;
* one Slurm job is submitted, spanning `NUM_NODES` nodes;
* `srun` starts a worker process on every task slot in that job;
* each worker connects back to the queue over InfiniBand,
  pulls tasks, runs them, and posts results;
* the driver blocks in `wait()` until every task is back.

Numerically: pi is the integral of 4 / (1 + x^2) from 0 to 1,
approximated by a midpoint Riemann sum over `num_steps` slices.
Each task sums a strided subset of the slices,
so the partial sums simply add up.
"""

from ds_service_client import DsServiceServer
from slurm_workflows import SlurmPilotExecutor, check_for_error

# Shell run inside each job before the worker starts.
# Nothing is inherited from the login node,
# so this is the only place your environment can come from.
# Pass the *text* of the snippet, not a path to a file.
# A typical BII value looks like:
#
#     SETUP_SCRIPT = """
#     module load miniforge/26.3.2
#     conda activate my-env
#     """
#
SETUP_SCRIPT = ""

# The size of the pool.
# One worker process will run per task slot,
# so this is NUM_NODES * TASKS_PER_NODE = 400 workers.
NUM_NODES = 10
TASKS_PER_NODE = 40

# Passed through to `sbatch` verbatim, so any Slurm option works.
#   --account            the allocation to charge.
#   --partition          the partition to use.
#   --nodes              how many nodes this *one* job holds.
#   --ntasks-per-node    task slots per node, one worker process per slot.
#   --cpus-per-task=1    this objective is single-threaded.
#   --mem=0              give the job all the memory on each node.
#   --time               walltime; the pool dies when this expires,
#                        so make it longer than the work will take.
SBATCH_ARGS = [
    "--account=bii_nssac",
    f"--partition=bii --nodes={NUM_NODES}",
    f"--ntasks-per-node={TASKS_PER_NODE} --cpus-per-task=1 --mem=0",
    "--time=1:00:00",
]


def do_step_pi(start, stop, step, stepsize):
    """Sum every `step`-th midpoint slice, beginning at `start`.

    This runs on a compute node.
    It is cloudpickled and shipped there,
    so it may be a closure or a lambda,
    but anything it *imports* has to exist on the compute node too.
    """
    x, s = 0.0, 0.0
    for i in range(start, stop, step):
        x = (i + 0.5) * stepsize
        s += 4.0 / (1.0 + x * x)
    return s


def main():
    # The queue server runs on the login node, for as long as this block.
    # Workers connect to it from the compute nodes,
    # so it is bound to an interface they can route to:
    # `ib0`, the login node's InfiniBand interface.
    # A login node without `ib0` raises here,
    # rather than starting a server the workers cannot reach.
    with DsServiceServer(interface="ib0") as ds_service:
        # The server may take a moment to start,
        # and a client that connects too early exits with an error.
        ds_service.wait_until_ready()

        # `ib0`'s address, and the port the server picked.
        address = ds_service.address

        # Leaving this block cancels every pilot job,
        # including if the body raises.
        # The executor's name identifies this run:
        # it prefixes every task id and worker job name,
        # and its work dirs are collected under it in the cache dir.
        with SlurmPilotExecutor("compute-pi", address) as executor:
            # Describes a kind of worker. Nothing is submitted yet.
            # The group name is also the queue name:
            # `submit("bii", ...)` is served only by workers in group `bii`.
            executor.define_worker(
                name="bii",
                sbatch_args=SBATCH_ARGS,
                setup_script=SETUP_SCRIPT,
            )

            # This submits *one* Slurm job,
            # and that single job already spans NUM_NODES nodes
            # because `--nodes` says so.
            # `srun` then starts a worker on every task slot in the allocation,
            # giving NUM_NODES * TASKS_PER_NODE = 400 worker processes.
            #
            # The alternative shape is `--nodes=1` in SBATCH_ARGS
            # with `scale_workers("bii", NUM_NODES)`:
            # the same 400 workers as ten one-node jobs,
            # which start piecemeal as nodes free up
            # instead of waiting for ten nodes to be available at once.
            executor.scale_workers("bii", 1)

            num_steps = 1_000_000_000
            stepsize = 1.0 / num_steps

            # More tasks than workers, on purpose.
            # Slices are equal-cost here, but in general they are not,
            # and handing each worker several small pieces
            # lets a fast worker take extra work
            # instead of the run waiting on the slowest one.
            over_decomp_factor = 10
            par = NUM_NODES * TASKS_PER_NODE * over_decomp_factor

            # `submit` returns immediately with a Task handle;
            # the callable and its arguments are cloudpickled onto the queue.
            # You can submit before the workers exist ---
            # tasks simply wait to be pulled.
            tasks = [
                executor.submit(
                    "bii",
                    do_step_pi,
                    start=i,
                    stop=num_steps,
                    step=par,
                    stepsize=stepsize,
                )
                for i in range(par)
            ]

            # Blocks until every task is executed.
            # Use `as_completed(tasks)` instead
            # if you want to consume results as they land.
            executor.wait(tasks)

            # An exception on a worker does not reach the driver:
            # it comes back as this task's *output*.
            # So always check, or the sum below would fail
            # with something unrelated-looking.
            # `error_id` appears next to the traceback in the worker's log,
            # under `executor.work_dir`.
            failed = check_for_error(tasks)
            if failed:
                raise RuntimeError(f"{len(failed)} of {len(tasks)} tasks failed")

    # Outside both blocks: the pool is cancelled and the queue is gone,
    # but the results were copied into the Task objects by `wait`,
    # so they are ordinary local values now.
    pi = sum(task.output for task in tasks) * stepsize
    print(f"pi = {pi}")


if __name__ == "__main__":
    main()
