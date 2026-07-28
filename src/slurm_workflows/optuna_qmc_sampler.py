"""QMC sampler that is safe to run distributed over ds-service.

`optuna.samplers.QMCSampler` keeps no sequence state of its own:
a trial's position in the low-discrepancy sequence
is a counter kept in the study's storage,
so every worker draws from one shared sequence.
That design is right, but three details of it
misbehave once the workers are separate processes:

1. **The counter is allocated non-atomically.**
   `QMCSampler._find_sample_id` reads the counter from the study's system
   attrs, adds one, and writes it back --- three round trips with no
   compare-and-swap. Optuna's own comment concedes the implementation
   "only ensures that each `sample_id` is sampled at least once".
   Concurrent workers are handed the same id and evaluate the same point.
   Measured with four workers and a fast objective,
   a third to a half of the trials were duplicates
   --- on `RDBStorage` just as much as on the journal.
2. **The QMC seed is per-worker.** With `scramble=True` and no explicit
   `seed`, every worker randomizes its own sequence and the
   low-discrepancy property is gone. Optuna's answer is to tell the user
   to pass the same seed to every worker.
3. **That answer breaks the fallback sampler.** The same `seed` also seeds
   the `RandomSampler` used for the first trial of the study and for
   anything outside the relative search space --- so workers sharing a seed
   propose *identical* points there. Nor is it a rare path: the search
   space is inferred from the first finished trial, so until one finishes,
   every worker samples randomly.

`DsServiceQMCSampler` fixes all three:

* Sample ids come from `counter_get_next_value`, which increments and
  returns under the server's counter lock --- one RPC, no read-modify-write,
  so two workers can never receive the same id.
* Workers agree on a scramble seed through the server instead of through
  the caller: the first one to start publishes its seed, the rest adopt it.
* The fallback sampler is seeded independently of the QMC seed, so shared
  scrambling no longer implies identical random draws.

Passing `search_space` additionally removes the random warm-up in point 3
by declaring the space up front,
so trial 0 on every worker is already QMC.

Usage::

    from slurm_workflows.optuna_storage import create_optuna_storage
    from slurm_workflows.optuna_qmc_sampler import DsServiceQMCSampler

    storage = create_optuna_storage(server_address, prefix="tuning")
    study = optuna.create_study(
        storage=storage,
        study_name="tuning",
        load_if_exists=True,   # every worker runs this; first one wins
        sampler=DsServiceQMCSampler(
            storage,
            scramble=True,
            search_space={
                "x": optuna.distributions.FloatDistribution(-10, 10),
                "lr": optuna.distributions.FloatDistribution(1e-5, 1e-1, log=True),
            },
        ),
    )

The sampler takes the storage rather than an address of its own,
so its counter cannot end up on a different server
or under a different prefix than the study it is sampling for.
Both live on that one server, and ds-service is in-memory,
so they are kept or lost together --- a counter can never
outlive, or fall behind, the journal it belongs to.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any, TYPE_CHECKING

from optuna.samplers import QMCSampler, RandomSampler, BaseSampler
from optuna.storages import JournalStorage
from optuna.distributions import BaseDistribution, CategoricalDistribution

from .optuna_storage import DsServiceJournalBackend, resolve_backend

if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial


class DsServiceQMCSampler(QMCSampler):
    """A `QMCSampler` whose shared state lives in ds-service.

    Args:
        storage: The `DsServiceJournalBackend`, or the storage returned by
            `create_optuna_storage`, that the study is using. The sampler's
            keys are namespaced by that backend's prefix.
        qmc_type: `"sobol"` (default) or `"halton"`.
        scramble: Randomize the sequence. Unlike the base class this is safe
            to enable distributed --- workers agree on the seed themselves.
        seed: Scramble seed. Leave as `None` to have the workers negotiate
            one; pass an integer only to reproduce a specific run, in which
            case every worker must be given the same value.
        search_space: Declare the search space up front so that QMC applies
            from the very first trial instead of after the first one
            finishes. Categorical parameters are not supported by QMC and
            are rejected here rather than silently ignored.
        independent_sampler: Sampler for the first trial (when
            `search_space` is not given) and for parameters outside the
            relative search space. Defaults to a `RandomSampler` seeded
            independently per worker, so that workers do not duplicate each
            other's fallback draws.
        warn_independent_sampling: Warn when a parameter falls back to the
            independent sampler.
    """

    def __init__(
        self,
        storage: DsServiceJournalBackend | JournalStorage,
        *,
        qmc_type: str = "sobol",
        scramble: bool = False,
        seed: int | None = None,
        search_space: dict[str, BaseDistribution] | None = None,
        independent_sampler: BaseSampler | None = None,
        warn_independent_sampling: bool = True,
    ):
        self._backend = resolve_backend(storage)

        if search_space is not None:
            if not search_space:
                raise ValueError("search_space must not be empty")
            for name, distribution in search_space.items():
                if isinstance(distribution, CategoricalDistribution):
                    raise ValueError(
                        f"search_space[{name!r}] is a CategoricalDistribution, "
                        "which QMC cannot sample. Leave categorical parameters "
                        "out of search_space; they are sampled by the "
                        "independent sampler."
                    )
            search_space = dict(search_space)
        self._search_space = search_space

        super().__init__(
            qmc_type=qmc_type,
            scramble=scramble,
            seed=seed,
            # Seeded per worker, deliberately: this sampler draws the points
            # that QMC cannot, and workers must not draw the same ones.
            independent_sampler=independent_sampler or RandomSampler(),
            warn_independent_sampling=warn_independent_sampling,
            # The seed is negotiated through the server, so the base class's
            # warning about workers seeding themselves does not apply.
            warn_asynchronous_seeding=False,
        )

        # An explicit seed is already common to every worker, and without
        # scrambling the seed does not reach the QMC engine at all.
        self._seed_agreed: bool = seed is not None or not scramble

    def _key(self, study: Study, kind: str, suffix: str = "") -> str:
        key = f"{self._backend.prefix}:qmc:{kind}:{study.study_name}"
        return f"{key}:{suffix}" if suffix else key

    def _agree_on_seed(self, study: Study) -> None:
        """Converge on one scramble seed across all workers of this study.

        The journal decides it: appends are ordered, so whichever worker
        appends first owns entry 0, and every worker --- including that one
        --- reads its seed back from there. No lock, and no requirement that
        the caller pass the same integer to every worker.
        """
        if self._seed_agreed:
            return

        client = self._backend._get_client()
        key = self._key(study, "seed")
        if client.journal_size(key) == 0:
            client.journal_append(key, str(self._seed).encode("utf-8"))
        self._seed = int(client.journal_read(key, 0, 1)[0])
        self._seed_agreed = True

    def _sample_id_key(self, study: Study, search_space: dict[str, BaseDistribution]) -> str:
        """Counter key, sensitive to everything that defines the sequence.

        Mirrors the digest the base class uses for its system-attr key, so
        that changing the QMC type, the search space or the seed starts a
        fresh sequence rather than continuing an unrelated one.
        """
        qmc_vars: dict[str, Any] = {
            "qmc_type": self._qmc_type,
            "search_space": {
                name: str(distribution)
                for name, distribution in sorted(search_space.items())
            },
        }
        if self._scramble:
            qmc_vars.update(scramble=True, seed=self._seed)
        else:
            qmc_vars.update(scramble=False)

        digest = hashlib.sha256(json.dumps(qmc_vars).encode()).hexdigest()
        return self._key(study, "counter", digest)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        # Runs before any suggestion, so the seed is settled before it can
        # reach either the QMC engine or the counter key that depends on it.
        self._agree_on_seed(study)
        super().before_trial(study, trial)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        if self._search_space is not None:
            return dict(self._search_space)
        return super().infer_relative_search_space(study, trial)

    def _find_sample_id(
        self, study: Study, search_space: dict[str, BaseDistribution]
    ) -> int:
        # One RPC, incremented and returned under the server's counter lock,
        # replacing the base class's read-modify-write over three. The
        # counter starts at 1 on first use; sample ids are 0-based.
        self._agree_on_seed(study)
        client = self._backend._get_client()
        return client.counter_get_next_value(self._sample_id_key(study, search_space)) - 1
