"""Deterministic sampler that visits every corner of the search space.

Given a box of `d` parameters, the extreme points are the `2**d`
combinations of each parameter's low and high value ---
the corners of the box.
Evaluating them is what you do before a real search:
it bounds the objective's range, exposes the parameters it is monotone in,
and catches the corner that makes the code fall over.

`ExtremePointSampler` hands out one corner per trial, with no randomness,
and is safe to run distributed:
corners are allocated with ds-service's atomic counter,
so two workers can never be given the same one.

Optuna's own exhaustive sampler, `GridSampler`, allocates two ways.
On a fresh study it uses the trial number as the grid index,
which is collision-free because trial numbers are unique --- that path is fine.
Every other case falls back to scanning the study for an unvisited point
and picking randomly among them,
which its own comment concedes can hand two workers the same point.
That path is reached whenever the trial number runs past the grid size:
a resumed study, retried failures, more trials than points.
Measured there with four workers on a 32-point grid,
8 to 13 points were **never visited** while as many were evaluated twice
--- an exhaustive search that quietly is not one.
One counter increment replaces both paths,
so a resumed run is no different from a fresh one.

Usage::

    from slurm_workflows.optuna_storage import create_optuna_storage
    from slurm_workflows.optuna_extreme_point_sampler import ExtremePointSampler

    space = {
        "x": optuna.distributions.FloatDistribution(-10, 10),
        "lr": optuna.distributions.FloatDistribution(1e-5, 1e-1, log=True),
        "layers": optuna.distributions.IntDistribution(1, 8),
    }
    storage = create_optuna_storage(server_address, prefix="corners")
    study = optuna.create_study(
        storage=storage,
        study_name="corners",
        load_if_exists=True,          # every worker runs this
        sampler=ExtremePointSampler(storage, space),
    )
    study.optimize(objective, n_trials=study.sampler.n_corners)

Corners are numbered in binary counting order over the parameters sorted
by name: corner 0 is every parameter at its low, and bit *i* of the corner
number selects the high value of the *i*-th parameter. That makes a corner
number meaningful and reproducible, but it also means a run stopped early
covers a face of the box rather than a spread of it --- run all
`n_corners` trials, or don't rely on the subset being representative.

The search space must be given up front. There is nothing to infer it
from: the count of trials to run is a property of the space, and every
worker has to agree on it before the first trial.
"""

from __future__ import annotations

import json
import hashlib
import logging
from math import prod
from typing import Any, TYPE_CHECKING

from optuna.samplers import BaseSampler
from optuna.storages import JournalStorage
from optuna.trial import TrialState
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from .optuna_storage import DsServiceJournalBackend, resolve_backend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.study import Study
    from optuna.trial import FrozenTrial

log = logging.getLogger(__name__)

# Namespaced so they cannot collide with GridSampler's `grid_id`, or with
# anything a user puts in a trial's system attrs.
CORNER_ID_ATTR: str = "extreme_point:corner_id"
SPACE_ATTR: str = "extreme_point:space"

# States that mean a corner has been dealt with and need not be redone.
# FAIL is deliberately absent: a corner whose worker died is worth
# retrying, and re-running one that fails deterministically costs only the
# trials the caller chose to run past `n_corners`.
_VISITED_STATES = (TrialState.COMPLETE, TrialState.PRUNED)


class ExtremePointSampler(BaseSampler):
    """Sampler that walks the corners of a box search space.

    Args:
        storage: The `DsServiceJournalBackend`, or the storage returned by
            `create_optuna_storage`, that the study is using. Corner
            allocation is namespaced by that backend's prefix.
        search_space: The box to take corners of. Distributions with a
            `low` and a `high` only --- a `CategoricalDistribution` has no
            extremes and is rejected. A parameter whose low equals its high
            contributes one value rather than two, so it does not double
            the corner count.
        independent_sampler: Sampler for parameters the objective suggests
            that are *not* in `search_space`. There is none by default and
            such a parameter raises, because sampling it would make this
            sampler non-deterministic; pass one to opt into that.
    """

    def __init__(
        self,
        storage: DsServiceJournalBackend | JournalStorage,
        search_space: dict[str, BaseDistribution],
        *,
        independent_sampler: BaseSampler | None = None,
    ):
        self._backend = resolve_backend(storage)
        self._independent_sampler = independent_sampler

        if not search_space:
            raise ValueError("search_space must not be empty")

        self._param_names: list[str] = sorted(search_space)
        self._search_space = {name: search_space[name] for name in self._param_names}
        self._values: dict[str, list[Any]] = {
            name: _extremes(name, distribution)
            for name, distribution in self._search_space.items()
        }

        # Mixed-radix strides, least significant first, so that corner id
        # `i` decodes one parameter at a time without building the product.
        # With 2**d corners, materializing them the way `GridSampler` does
        # would run out of memory long before the study ran out of trials.
        self._strides: dict[str, int] = {}
        stride = 1
        for name in self._param_names:
            self._strides[name] = stride
            stride *= len(self._values[name])
        self._n_corners: int = stride

        self._space_digest: str = hashlib.sha256(
            json.dumps(
                {name: str(d) for name, d in self._search_space.items()}
            ).encode()
        ).hexdigest()

    @property
    def n_corners(self) -> int:
        """How many extreme points this space has.

        `2**d` for `d` parameters, less any whose low equals their high.
        Pass this as `n_trials` to visit each corner exactly once.
        """
        return self._n_corners

    def corner(self, corner_id: int) -> dict[str, Any]:
        """The parameter values at corner `corner_id`."""
        if not 0 <= corner_id < self._n_corners:
            raise IndexError(
                f"corner_id must be in [0, {self._n_corners}), got {corner_id}"
            )
        return {name: self._value_at(corner_id, name) for name in self._param_names}

    def _value_at(self, corner_id: int, param_name: str) -> Any:
        choices = self._values[param_name]
        return choices[(corner_id // self._strides[param_name]) % len(choices)]

    def _counter_key(self, study: Study) -> str:
        # The digest keys the counter to the space, so that changing the
        # space starts a fresh walk instead of continuing an unrelated one.
        return (
            f"{self._backend.prefix}:extreme-point:counter:"
            f"{study.study_name}:{self._space_digest}"
        )

    def _unvisited_corner_ids(self, study: Study) -> list[int]:
        """Corners no trial has finished or is currently working on."""
        taken = set()
        for trial in study._storage.get_all_trials(study._study_id, deepcopy=False):
            attrs = trial.system_attrs
            if attrs.get(SPACE_ATTR) != self._space_digest:
                continue
            if trial.state in _VISITED_STATES or trial.state == TrialState.RUNNING:
                taken.add(attrs[CORNER_ID_ATTR])
        return sorted(set(range(self._n_corners)) - taken)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        # Like GridSampler, the corner is chosen here and recorded on the
        # trial, and the values are handed out by `sample_independent` ---
        # which is where the distribution object needed to validate them
        # is actually available.
        if CORNER_ID_ATTR in trial.system_attrs or "fixed_params" in trial.system_attrs:
            return  # a retried or enqueued trial already has its corner

        # One RPC, incremented under the server's counter lock. This is the
        # whole distributed story: no scan, no read-modify-write, no chance
        # of two workers being handed the same corner.
        index = self._backend._get_client().counter_get_next_value(
            self._counter_key(study)
        ) - 1

        if index < self._n_corners:
            corner_id = index
        else:
            # Past the end of the walk. Prefer corners that were allocated
            # but never finished -- a worker that died took its corner with
            # it -- and only then start repeating.
            candidates = self._unvisited_corner_ids(study)
            if not candidates:
                log.warning(
                    "Every extreme point of this space has been visited; "
                    "trial %d is re-evaluating one. Pass n_trials=%d to "
                    "visit each corner exactly once.",
                    trial.number,
                    self._n_corners,
                )
                candidates = list(range(self._n_corners))
            # `index` is unique per allocation, so concurrent workers
            # mopping up spread over the candidates instead of colliding.
            corner_id = candidates[index % len(candidates)]

        # Order matters: these are two separate appends, and readers filter
        # on the digest. Writing the corner first means a reader that sees
        # the digest is guaranteed to see the corner too.
        study._storage.set_trial_system_attr(
            trial._trial_id, CORNER_ID_ATTR, corner_id
        )
        study._storage.set_trial_system_attr(
            trial._trial_id, SPACE_ATTR, self._space_digest
        )

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if param_name not in self._search_space:
            if self._independent_sampler is None:
                raise ValueError(
                    f"{param_name!r} is not in this sampler's search space. "
                    "ExtremePointSampler is deterministic and has nothing to "
                    "sample it with; add it to search_space, or pass an "
                    "independent_sampler to have it sampled separately."
                )
            return self._independent_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        corner_id = trial.system_attrs.get(CORNER_ID_ATTR)
        if corner_id is None:
            raise ValueError(
                "This trial has no corner assigned. All parameters must be "
                "specified when enqueueing a trial for ExtremePointSampler."
            )

        value = self._value_at(corner_id, param_name)
        if not param_distribution._contains(
            param_distribution.to_internal_repr(value)
        ):
            # The objective asked for a narrower range than search_space
            # declared. Optuna would silently record the value anyway, so
            # say which parameter drifted.
            log.warning(
                "Extreme point %r for %r is outside the distribution the "
                "objective suggested (%s); the value is used as-is.",
                value,
                param_name,
                param_distribution,
            )
        return value

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if self._independent_sampler is not None:
            self._independent_sampler.after_trial(study, trial, state, values)

        # Wind the workers down once the walk is complete, rather than
        # having them re-evaluate corners to fill out n_trials. Each worker
        # stops itself; there is nothing to coordinate.
        if not self._unvisited_corner_ids(study):
            try:
                study.stop()
            except RuntimeError:
                # Only valid inside an optimize loop; ask/tell callers have
                # nothing to stop.
                pass

    def reseed_rng(self) -> None:
        # Nothing of this sampler's own is random.
        if self._independent_sampler is not None:
            self._independent_sampler.reseed_rng()


def _extremes(param_name: str, distribution: BaseDistribution) -> list[Any]:
    """The low and high of one parameter, deduplicated.

    Optuna normalizes `high` for stepped distributions at construction, so
    both ends are always attainable values.
    """
    if isinstance(distribution, CategoricalDistribution):
        raise ValueError(
            f"search_space[{param_name!r}] is a CategoricalDistribution, which "
            "has no extremes: its choices are unordered. Leave it out and pass "
            "an independent_sampler, or use optuna.samplers.GridSampler to "
            "enumerate categorical values."
        )
    if not isinstance(distribution, (FloatDistribution, IntDistribution)):
        raise ValueError(
            f"search_space[{param_name!r}] is a {type(distribution).__name__}, "
            "which has no low and high to take extremes of."
        )

    if distribution.low == distribution.high:
        return [distribution.low]
    return [distribution.low, distribution.high]
