# How to run the tests

[← back to the main README](../README.md)

From the repository root:

```sh
pip install -ve .[test,dev]
pytest
```

The suite needs no Slurm cluster.
It takes about 55s end to end:
~26s for everything but the botorch tests,
and the rest is GP fits.
Most of that first 26s is one ds-service process started per test,
which is the price of testing against the real queue.

The `[test]` extra pulls botorch, and so torch — a large download.
Without it `test_optimize_space_botorch.py` skips
and the rest of the suite still runs.
`[dev]` adds `black` and `pyright`,
which the repository conventions require to be clean
alongside a passing suite.

## What is real and what is mocked

**Slurm is mocked.**
`FakeSlurm` (in `tests/conftest.py`) replaces the `subprocess` module
*inside* `slurm_utils`, intercepting `sbatch` / `squeue` / `scancel`.
Everything above that boundary is the real code path
— script rendering, job-id parsing, environment scrubbing —
and tests can inspect the scripts that would have been submitted
(`fake_slurm.submissions`)
or inject command failures (`fake_slurm.fail_command("sbatch")`).

**ds-service is real.**
Each test gets its own server process on a random port,
so queue semantics are exercised against the actual implementation
rather than a stand-in that can drift from it.
The server is in-memory,
so a fresh process per test also means no state leaks between tests.

The server is started by `DsServiceServer` from `ds-service-client`,
which also decides where the binary comes from:
`$DS_SERVICE_BIN` if it is set, otherwise `ds-service` on `$PATH`.
`$DS_SERVICE_BIN` may be a whole command line rather than a path.

If neither finds it, the tests that need a queue **skip**
(the template and `slurm_utils` tests still run).

## Layout

Paths are relative to [`tests/`](../tests).

| File | Covers |
| --- | --- |
| `test_templates.py` | The multi-template-per-file loader and every template |
| `test_slurm_utils.py` | `sbatch`/`squeue`/`scancel` wrappers, `get_clean_environ` |
| `test_executor.py` | `SlurmPilotExecutor`: worker groups, scaling, submit/poll, lifecycle |
| `test_worker.py` | `PilotWorkerProcess` and the `slurm-pilot-worker` CLI |
| `test_monitors.py` | The host and cgroup samplers and the monitor threads |
| `test_swtop.py` | The `swtop` monitor: what it collects, how it renders, and the CLI |
| `test_search_space.py` | The range types and the unit cube mapping (no botorch needed) |
| `test_explore_space.py` | `ExploreSpaceSobolQMC`: the design it draws and what it records (no botorch needed) |
| `test_utils.py` | The shared helpers |
| `test_optimize_space_botorch.py` | `OptimizeSpaceBotorch`: the observations it starts from, rounds, acquisition, search behaviour, resuming (skips without botorch) |
| `conftest.py` | Fixtures: real ds-service, fake Slurm, executor, hang guards |
| `worker_harness.py` | Runs a real worker's main loop for a bounded number of tasks, or of queue polls |
| `support_actor.py` | Actor classes; must stay importable by name for actor tests |

## Notes for future changes

- **Worker tests run a real worker.**
  `PilotWorkerProcess.main()` loops forever by design
  and swallows every `Exception` so a bad task can't kill a worker.
  `run_worker()` stops it with a `BaseException` raised from `task_done`
  after the expected number of tasks
  — that's why `StopWorker` is not an `Exception`.
  `poll_worker()` counts `task_get` calls instead of completions,
  which is the only way to bound a worker with nothing to run:
  an empty queue completes no tasks,
  so `run_worker` would never reach its limit.
- **Tests are bounded by a wall-clock alarm.**
  The executor's polling loop and the worker's main loop
  both run until a condition holds,
  so a regression turns a failing test into a hanging one.
  An autouse 60s alarm backstops every test,
  and the polling tests use a tighter explicit `time_limit` fixture.
- **The botorch tests mostly use a stand-in executor.**
  `LocalExecutor` runs the objective inline.
  The optimizer's contract with the executor is two calls wide
  (`submit` returns a `Task`, `wait` fills in its `output`),
  and a GP fit already dominates each test,
  so a queue round trip on top would buy nothing.
  `TestRealExecutor` is what keeps the stand-in honest
  — it runs a whole optimization
  through the real executor, the real queue and a real worker
  (in a thread, since the optimizer blocks in `wait` the moment it submits).
- **Four botorch tests assert search *behaviour*, not bookkeeping.**
  They are the ones that catch the objective's sign being flipped
  — botorch maximizes, the optimizer minimizes.
  They are stochastic
  (torch's global RNG is left unseeded, so each run is a fresh sample),
  and their margins were chosen from measured spreads:
  the monotone case has a *median* search point of 0.00 against a 0.5
  threshold, where flipping the sign puts it at 0.97+,
  and `test_search_beats_random_search` won 12/12 with a 4.6x margin.
  Assert on that median rather than the max: `qLogNoisyExpectedImprovement`
  deliberately probes away from the incumbent, so single points reach 1.0
  on perfectly correct runs.
  All four use *unimodal* objectives on purpose
  — an earlier Himmelblau version of the random-search comparison
  lost 1 run in 10.
- **Queue name == worker group name.**
  A task submitted to queue `cpu` is only served by workers in group `cpu`;
  tests rely on this to keep groups isolated.
- **Don't wait on an RPC to detect server readiness.**
  A failed first RPC puts the gRPC channel into a ~1s reconnect backoff.
  `DsServiceServer.wait_until_ready()` polls the TCP socket instead,
  which is why `conftest` calls it rather than rolling its own probe;
  doing otherwise makes the suite ~100x slower.
- **Server lifecycle belongs to `ds-service-client`, not to `conftest`.**
  Finding the binary, picking a free port, waiting for the socket
  and terminating the process are all `DsServiceServer`'s job.
  `conftest` only chooses the interface to bind
  and translates a missing binary into a skip.
- **The test server binds `lo`.**
  `DsServiceServer` takes an interface name rather than an address,
  and loopback keeps a test's queue unreachable from outside the machine.
