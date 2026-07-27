"""Compute PI on Rivanna"""

from slurm_workflows import DsService, SlurmPilotExecutor

SERVER_EXE = "apptainer run /project/bii_nssac/people/pb5gj/shared/ds-service/latest/ds-service.sif"

TASK_PER_NODE = 40
SBATCH_ARGS = [
    "--account=bii_nssac --qos=bii-unlimited",
    "--partition=bii --nodes=1",
    "--ntasks-per-node=40 --cpus-per-task=1 --mem=0",
    "--time=1:00:00",
]


def do_step_pi(start, stop, step, stepsize):
    x, s = 0.0, 0.0
    for i in range(start, stop, step):
        x = (i + 0.5) * stepsize
        s += 4.0 / (1.0 + x * x)
    return s


def main():
    with DsService(server_exe=SERVER_EXE) as ds_service:
        ds_service.start()
        ds_service.wait_until_ready()

        executor = SlurmPilotExecutor(ds_service.get_address("ib0"))
        executor.define_worker(name="bii", sbatch_args=SBATCH_ARGS)

        num_steps = 100_000_000
        stepsize = 1.0 / num_steps
        num_nodes = 10
        tasks_per_node = TASK_PER_NODE
        over_decomp_factor = 10
        par = num_nodes * tasks_per_node * over_decomp_factor

        executor.scale_workers("bii", num_nodes)
        try:
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
            executor.wait(tasks)
        finally:
            executor.scale_workers("bii", 0)

        pi = sum(task.output for task in tasks) * stepsize
        print(f"pi = {pi}")


if __name__ == "__main__":
    main()
