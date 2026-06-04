"""Pure HPO search-space enumeration + winner selection.

Deliberately dependency-light (stdlib only) so the trial-generation and
selection logic is unit-testable without torch / mlflow / a GPU. The notebook
(`notebooks/02b_hpo_sweep.py`) owns the side-effecting parts: launching each
trial as a nested MLflow run via the `@distributed` trainer and registering the
winner. This module just answers two questions:

* "what hyperparameters does trial *i* use?"  -> ``iter_trials``
* "which finished trial won?"                 -> ``select_best``

Search space format
-------------------
``search_space`` maps a ``TrainerConfig`` field name to a spec:

* ``[a, b, c]``                  -> categorical choice (grid + random)
* ``("uniform", lo, hi)``        -> float in [lo, hi]               (random only)
* ``("loguniform", lo, hi)``     -> float log-uniform in [lo, hi]   (random only)
* ``("int", lo, hi)``            -> integer in [lo, hi] inclusive   (random only)

Grid strategy enumerates the Cartesian product of the categorical lists (it
errors on a continuous spec, which has no finite grid). Random strategy samples
each field independently with a seeded RNG, so a given ``seed`` reproduces the
exact same trial sequence.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

# A field spec is either a list of choices or a (kind, lo, hi) tuple.
FieldSpec = list[Any] | tuple[str, float, float]
SearchSpace = dict[str, FieldSpec]

_CONTINUOUS_KINDS = ("uniform", "loguniform", "int")


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Outcome of one trial. ``metric`` is None when the trial failed."""

    trial_id: int
    params: dict[str, Any]
    metric: float | None = None
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class Trial:
    """A planned trial: an id + the override dict applied to the base config."""

    trial_id: int
    params: dict[str, Any] = field(default_factory=dict)


def _is_continuous(spec: FieldSpec) -> bool:
    return isinstance(spec, tuple) and len(spec) == 3 and spec[0] in _CONTINUOUS_KINDS


def _sample_one(spec: FieldSpec, rng: random.Random) -> Any:
    if isinstance(spec, list):
        if not spec:
            raise ValueError("categorical choice list must be non-empty")
        return rng.choice(spec)
    kind, lo, hi = spec
    if kind == "uniform":
        return rng.uniform(lo, hi)
    if kind == "loguniform":
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    if kind == "int":
        return rng.randint(int(lo), int(hi))
    raise ValueError(f"unknown continuous kind {kind!r}; expected one of {_CONTINUOUS_KINDS}")


def iter_trials(
    search_space: SearchSpace,
    *,
    strategy: str = "random",
    max_trials: int,
    seed: int = 0,
) -> Iterator[Trial]:
    """Yield up to ``max_trials`` planned trials.

    Args:
        search_space: field -> spec (see module docstring).
        strategy: "grid" (Cartesian product of categorical lists) or "random"
            (seeded independent sampling, supports continuous specs).
        max_trials: hard cap on the number of trials yielded.
        seed: RNG seed for the random strategy (deterministic).

    Raises:
        ValueError: grid strategy with a continuous spec, or unknown strategy.
    """
    if max_trials < 1:
        return
    if strategy == "grid":
        keys = list(search_space)
        grids: list[Sequence[Any]] = []
        for k in keys:
            spec = search_space[k]
            if _is_continuous(spec):
                raise ValueError(
                    f"grid strategy cannot enumerate continuous field {k!r}={spec!r}; "
                    "use strategy='random' or give a discrete choice list."
                )
            grids.append(spec)  # type: ignore[arg-type]
        for i, combo in enumerate(itertools.product(*grids)):
            if i >= max_trials:
                return
            yield Trial(trial_id=i, params=dict(zip(keys, combo, strict=True)))
        return
    if strategy == "random":
        rng = random.Random(seed)
        for i in range(max_trials):
            params = {k: _sample_one(spec, rng) for k, spec in search_space.items()}
            yield Trial(trial_id=i, params=params)
        return
    raise ValueError(f"unknown strategy {strategy!r}; expected 'grid' or 'random'")


def grid_size(search_space: SearchSpace) -> int:
    """Number of points in the full Cartesian grid (categorical fields only)."""
    n = 1
    for spec in search_space.values():
        if _is_continuous(spec):
            raise ValueError("grid_size is undefined for a continuous search space")
        n *= len(spec)  # type: ignore[arg-type]
    return n


def select_best(
    results: Sequence[TrialResult],
    *,
    higher_is_better: bool = True,
) -> TrialResult | None:
    """Return the winning trial, or None if no trial produced a metric.

    Trials with ``metric is None`` (failed/crashed) are skipped. Ties on the
    metric are broken by the lowest ``trial_id`` for determinism.
    """
    scored = [r for r in results if r.metric is not None]
    if not scored:
        return None
    return (max if higher_is_better else min)(
        scored,
        key=lambda r: (r.metric, -r.trial_id if higher_is_better else r.trial_id),
    )


def beats_experiment_best(
    candidate_metric: float | None,
    existing_metrics: Iterable[float | None],
    *,
    higher_is_better: bool = True,
) -> bool:
    """Return True if ``candidate_metric`` STRICTLY beats every existing metric.

    The "challenger registration gate": a freshly retrained winner only earns the
    ``@challenger`` alias when its validation metric strictly improves on the best
    existing registered version / run in the experiment. Pure comparison so the
    notebook (``02b_hpo_sweep.py``) just supplies the numbers.

    Semantics:
      * ``candidate_metric is None`` (the run never produced a metric) -> False;
        you can't promote a result you can't measure.
      * ``None`` entries in ``existing_metrics`` are ignored (failed prior runs).
      * No scored existing metrics (empty, or all None) -> True; nothing to beat,
        so the first measurable version becomes the challenger.
      * Otherwise: strictly greater than the max (``higher_is_better``) or strictly
        less than the min. Ties do NOT pass — an equal challenger does not displace
        the incumbent.
    """
    if candidate_metric is None:
        return False
    scored = [m for m in existing_metrics if m is not None]
    if not scored:
        return True
    if higher_is_better:
        return candidate_metric > max(scored)
    return candidate_metric < min(scored)


__all__ = [
    "FieldSpec",
    "SearchSpace",
    "Trial",
    "TrialResult",
    "beats_experiment_best",
    "grid_size",
    "iter_trials",
    "select_best",
]
