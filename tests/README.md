# Tests

```sh
pip install -ve .[test]
pytest
```

The suite needs no Slurm cluster. Everything except the botorch tests runs in
well under a second; those add ~10s, almost all of it GP fits.

The `[test]` extra pulls botorch, and so torch — a large download. Without it
`test_bayes_opt_botorch.py` skips and the rest of the suite still runs.

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
| `test_bayes_opt_botorch.py` | `BayesOptBotorch`: ranges, budgets, batch split, search behaviour (skips without botorch) |
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
- **The botorch tests mostly use a stand-in executor.** `LocalExecutor` runs
  the objective inline. The optimizer's contract with the executor is two calls
  wide (`submit` returns a `Task`, `wait` fills in its `output`), and a GP fit
  already dominates each test, so a queue round trip on top would buy nothing.
  `TestRealExecutor` is what keeps the stand-in honest — it runs a whole
  optimization through the real executor, the real queue and a real worker (in
  a thread, since the optimizer blocks in `wait` the moment it submits).
- **Four botorch tests assert search *behaviour*, not bookkeeping.** They are
  the ones that catch the objective's sign being flipped — botorch maximizes,
  the optimizer minimizes. They are stochastic (torch's global RNG is left
  unseeded, so each run is a fresh sample), and their margins were chosen from
  measured spreads: the monotone case puts every search point below 0.1 against
  a 0.5 threshold, and `test_search_beats_random_search` won 12/12 with a 4.6x
  margin. All four use *unimodal* objectives on purpose — an earlier
  Himmelblau version of the random-search comparison lost 1 run in 10.
- **Queue name == worker group name.** A task submitted to queue `cpu` is only
  served by workers in group `cpu`; tests rely on this to keep groups isolated.
- **Don't wait on an RPC to detect server readiness.** A failed first RPC puts
  the gRPC channel into a ~1s reconnect backoff. `conftest` waits on the TCP
  socket instead; doing otherwise makes the suite ~100x slower.
