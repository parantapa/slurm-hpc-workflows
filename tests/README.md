# Tests

```sh
pip install -ve .[test]
pytest
```

The whole suite runs in well under a second and needs no Slurm cluster.

## What is real and what is mocked

**Slurm is mocked.** `FakeSlurm` (in `conftest.py`) replaces the `subprocess`
module *inside* `slurm_utils`, intercepting `sbatch` / `squeue` / `scancel`.
Everything above that boundary is the real code path — script rendering, job-id
parsing, environment scrubbing — and tests can inspect the scripts that would
have been submitted (`fake_slurm.submissions`) or inject command failures
(`fake_slurm.fail_command("sbatch")`).

**ds-service is real.** Each test gets its own server process on a random port,
so queue semantics are exercised against the actual implementation rather than
a stand-in that can drift from it. The server is in-memory, so a fresh process
per test also means no state leaks between tests.

The binary is located in this order:

1. `$DS_SERVICE_EXE` — if set but missing, the run fails loudly rather than
   skipping.
2. `ds-service` on `$PATH`.
3. `~/workspace/ds-service/build/Release/ds-service`.

If none is found, the tests that need a queue **skip** (the template and
`slurm_utils` tests still run).

## Layout

| File | Covers |
| --- | --- |
| `test_templates.py` | The multi-template-per-file loader and every template |
| `test_slurm_utils.py` | `sbatch`/`squeue`/`scancel` wrappers, `get_clean_environ` |
| `test_executor.py` | `SlurmPilotExecutor`: worker groups, scaling, submit/poll, lifecycle |
| `test_worker.py` | `PilotWorkerProcess` and the `slurm-pilot-worker` CLI |
| `test_optuna_storage.py` | `DsServiceJournalBackend` and studies through it (skips without Optuna) |
| `test_optuna_qmc_sampler.py` | `DsServiceQMCSampler` under concurrent workers (skips without Optuna/scipy) |
| `test_optuna_extreme_point_sampler.py` | `ExtremePointSampler`: corner enumeration, concurrent allocation |
| `conftest.py` | Fixtures: real ds-service, fake Slurm, executor, hang guards |
| `worker_harness.py` | Runs a real worker's main loop for a bounded number of tasks |
| `support_actor.py` | Actor classes; must stay importable by name for actor tests |

## Notes for future changes

- **Worker tests run a real worker.** `PilotWorkerProcess.main()` loops forever
  by design and swallows every `Exception` so a bad task can't kill a worker.
  `run_worker()` stops it with a `BaseException` raised from `task_done` after
  the expected number of tasks — that's why `StopWorker` is not an `Exception`.
- **Tests are bounded by a wall-clock alarm.** The executor's polling loop and
  the worker's main loop both run until a condition holds, so a regression
  turns a failing test into a hanging one. An autouse 60s alarm backstops every
  test, and the polling tests use a tighter explicit `time_limit` fixture.
- **Three tests assert a defect in *Optuna's* samplers.** `test_the_base_
  sampler_does_repeat_points`, `test_the_base_sampler_draws_them_identically`
  and `test_grid_sampler_misses_points_in_that_case` guard the premise of the
  custom samplers' overrides. If Optuna fixes one upstream, that test fails and
  the matching override should be dropped rather than the test relaxed. All
  three are timing-dependent (they race four threads); they have been stable
  over repeated runs, but a lone flake there is the test, not our sampler.
- **`ExtremePointSampler` overshoots by design.** Workers already mid-trial when
  the last corner lands still finish it, so a distributed walk can end with up
  to `WORKERS - 1` extra trials. Tests assert coverage of every corner, not an
  exact trial count.
- **Queue name == worker group name.** A task submitted to queue `cpu` is only
  served by workers in group `cpu`; tests rely on this to keep groups isolated.
- **Don't wait on an RPC to detect server readiness.** A failed first RPC puts
  the gRPC channel into a ~1s reconnect backoff. `conftest` waits on the TCP
  socket instead; doing otherwise makes the suite ~100x slower.
