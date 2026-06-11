"""Tests for `config.campaigns` — typed campaign stages + validation."""

from __future__ import annotations

import pytest

from dais26_dentex.config.campaigns import (
    CAMPAIGN_STAGES,
    SWEEP_DEFAULTS,
    CampaignStage,
    validate_stage,
)
from dais26_dentex.config.trainer_config import TrainerConfig
from dais26_dentex.train.sweep import iter_trials

EXPECTED_STAGES = {
    "smoke",  # lane-plumbing validation stage, not part of the science campaign
    "dinov3_s1", "dinov3_regres", "dinov3_s2", "dinov3_s3", "dinov3_s4",
    "dinov3_res1536", "dinov3_falpha", "dinov3_fusion",
    "cradio_s1", "cradio_long", "cradio_s2", "cradio_s3", "cradio_s4",
    "cradio_giou", "cradio_final", "dinov3_final", "dinov3_final_giou",
}


def test_stage_inventory_is_the_documented_campaign() -> None:
    """The stage chain is a historical record (docs/HPO.md) — renames/removals
    should be deliberate, so pin the inventory."""
    assert set(CAMPAIGN_STAGES) == EXPECTED_STAGES


@pytest.mark.parametrize("name", sorted(CAMPAIGN_STAGES))
def test_every_stage_validates(name: str) -> None:
    validate_stage(CAMPAIGN_STAGES[name])


def test_sweep_defaults_validate() -> None:
    validate_stage(SWEEP_DEFAULTS)


@pytest.mark.parametrize("name", sorted(CAMPAIGN_STAGES))
def test_stage_pinned_plus_trial_builds_a_valid_trainer_config(name: str) -> None:
    """End-to-end: a stage's pinned set merged with its first sampled trial must
    construct a valid TrainerConfig — catches value-level mistakes (bad ranges,
    wrong list shapes) that key validation alone misses."""
    stage = CAMPAIGN_STAGES[name]
    trial = next(iter_trials(dict(stage.search_space), strategy="random", max_trials=1, seed=42))
    params = {
        "catalog": "c",
        "schema": "s",
        "backbone_name": stage.backbone,
        "epochs": stage.trial_epochs,
        **stage.pinned,
        **{k: v for k, v in trial.params.items() if k != "anchor_mode"},
    }
    TrainerConfig.from_dict(params).validate()


def test_validate_stage_rejects_unknown_field() -> None:
    bad = CampaignStage(
        backbone="cradio_v4_so400m",
        trial_epochs=5,
        schedule_epochs=(10,),
        max_trials=1,
        register_winner=False,
        pinned={"learning_rate": 1e-3},  # typo: the field is `lr`
        search_space={"base_seed": [42]},
    )
    with pytest.raises(ValueError, match="learning_rate"):
        validate_stage(bad)


def test_validate_stage_rejects_empty_search_space() -> None:
    bad = CampaignStage(
        backbone="cradio_v4_so400m",
        trial_epochs=5,
        schedule_epochs=(10,),
        max_trials=1,
        register_winner=False,
        search_space={},
    )
    with pytest.raises(ValueError, match="search_space"):
        validate_stage(bad)


def test_validate_stage_allows_anchor_mode_virtual_key() -> None:
    stage = CampaignStage(
        backbone="cradio_v4_so400m",
        trial_epochs=5,
        schedule_epochs=(10,),
        max_trials=2,
        register_winner=False,
        search_space={"anchor_mode": ["default", "calibrated"]},
    )
    validate_stage(stage)  # must not raise


def test_finalize_stages_register() -> None:
    for name in ("dinov3_s4", "cradio_s4", "cradio_final", "dinov3_final", "dinov3_final_giou"):
        assert CAMPAIGN_STAGES[name].register_winner, name
    for name in ("dinov3_s1", "cradio_s1", "cradio_long", "dinov3_fusion"):
        assert not CAMPAIGN_STAGES[name].register_winner, name


def test_smoke_stage_is_cheap_and_side_effect_free() -> None:
    """The smoke stage exists to validate lane plumbing in minutes: it must
    never register/move aliases and must stay tiny."""
    smoke = CAMPAIGN_STAGES["smoke"]
    assert smoke.register_winner is False
    assert smoke.max_trials == 1
    assert smoke.trial_epochs <= 2
    assert max(smoke.schedule_epochs) <= 5
