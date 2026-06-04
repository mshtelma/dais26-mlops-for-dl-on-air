"""Tests for the pure HPO sweep helpers (`train/sweep.py`)."""

from __future__ import annotations

import pytest

from dais26_dentex.train.sweep import (
    TrialResult,
    beats_experiment_best,
    grid_size,
    iter_trials,
    select_best,
)


def test_grid_enumerates_cartesian_product() -> None:
    space = {"lr": [1e-3, 1e-4], "backbone_mode": ["frozen", "full"]}
    trials = list(iter_trials(space, strategy="grid", max_trials=99))
    assert len(trials) == 4 == grid_size(space)
    combos = {(t.params["lr"], t.params["backbone_mode"]) for t in trials}
    assert combos == {(1e-3, "frozen"), (1e-3, "full"), (1e-4, "frozen"), (1e-4, "full")}
    # trial_ids are 0..n-1 and unique.
    assert [t.trial_id for t in trials] == [0, 1, 2, 3]


def test_grid_respects_max_trials_cap() -> None:
    space = {"a": [1, 2, 3], "b": [10, 20, 30]}  # full grid = 9
    trials = list(iter_trials(space, strategy="grid", max_trials=4))
    assert len(trials) == 4


def test_grid_rejects_continuous_spec() -> None:
    with pytest.raises(ValueError, match="continuous"):
        list(iter_trials({"lr": ("loguniform", 1e-5, 1e-3)}, strategy="grid", max_trials=3))


def test_random_is_deterministic_for_seed() -> None:
    space = {
        "lr": ("loguniform", 1e-5, 1e-3),
        "box_loss_weight": ("uniform", 0.5, 3.0),
        "backbone_trainable_blocks": ("int", 1, 6),
        "backbone_mode": ["frozen", "lora", "full"],
    }
    a = list(iter_trials(space, strategy="random", max_trials=5, seed=42))
    b = list(iter_trials(space, strategy="random", max_trials=5, seed=42))
    c = list(iter_trials(space, strategy="random", max_trials=5, seed=7))
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.params for t in a] != [t.params for t in c]
    assert len(a) == 5


def test_random_respects_bounds() -> None:
    space = {
        "lr": ("loguniform", 1e-5, 1e-3),
        "n": ("int", 2, 4),
        "u": ("uniform", 0.0, 1.0),
    }
    for t in iter_trials(space, strategy="random", max_trials=50, seed=1):
        assert 1e-5 <= t.params["lr"] <= 1e-3
        assert t.params["n"] in (2, 3, 4)
        assert 0.0 <= t.params["u"] <= 1.0


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        list(iter_trials({"a": [1]}, strategy="bogus", max_trials=1))


def test_max_trials_zero_yields_nothing() -> None:
    assert list(iter_trials({"a": [1]}, strategy="grid", max_trials=0)) == []


def test_select_best_higher_is_better() -> None:
    results = [
        TrialResult(0, {"lr": 1e-3}, metric=0.30),
        TrialResult(1, {"lr": 1e-4}, metric=0.42),
        TrialResult(2, {"lr": 1e-5}, metric=None),  # failed trial skipped
    ]
    best = select_best(results, higher_is_better=True)
    assert best is not None
    assert best.trial_id == 1
    assert best.metric == 0.42


def test_select_best_lower_is_better() -> None:
    results = [
        TrialResult(0, {}, metric=1.5),
        TrialResult(1, {}, metric=0.9),
    ]
    best = select_best(results, higher_is_better=False)
    assert best is not None and best.trial_id == 1


def test_select_best_tie_breaks_on_lowest_trial_id() -> None:
    results = [
        TrialResult(3, {}, metric=0.5),
        TrialResult(1, {}, metric=0.5),
        TrialResult(2, {}, metric=0.5),
    ]
    best = select_best(results, higher_is_better=True)
    assert best is not None and best.trial_id == 1


def test_select_best_all_failed_returns_none() -> None:
    results = [TrialResult(0, {}, metric=None), TrialResult(1, {}, metric=None)]
    assert select_best(results) is None
    assert select_best([]) is None


# --- beats_experiment_best (challenger registration gate) --------------------


def test_beats_experiment_best_no_prior_versions_passes() -> None:
    # First measurable version: nothing to beat -> becomes the challenger.
    assert beats_experiment_best(0.42, []) is True
    assert beats_experiment_best(0.0, [None, None]) is True


def test_beats_experiment_best_strictly_greater_passes() -> None:
    assert beats_experiment_best(0.50, [0.30, 0.49, None]) is True


def test_beats_experiment_best_tie_does_not_pass() -> None:
    # An equal challenger does NOT displace the incumbent.
    assert beats_experiment_best(0.49, [0.30, 0.49]) is False


def test_beats_experiment_best_worse_does_not_pass() -> None:
    assert beats_experiment_best(0.40, [0.55]) is False


def test_beats_experiment_best_none_candidate_never_passes() -> None:
    assert beats_experiment_best(None, []) is False
    assert beats_experiment_best(None, [0.1]) is False


def test_beats_experiment_best_lower_is_better() -> None:
    # e.g. a loss metric: strictly lower than the min existing wins.
    assert beats_experiment_best(0.10, [0.20, 0.15], higher_is_better=False) is True
    assert beats_experiment_best(0.15, [0.20, 0.15], higher_is_better=False) is False
    assert beats_experiment_best(0.30, [0.20], higher_is_better=False) is False
