"""Tests for `config.recipes` — the per-backbone single source of truth."""

from __future__ import annotations

import pytest

from dais26_dentex.config.recipes import (
    DETECTOR_NAMES_BY_BACKBONE,
    RECIPES,
    build_trainer_config,
)
from dais26_dentex.config.trainer_config import ALLOWED_BACKBONES


def test_every_allowed_backbone_has_a_recipe_and_names() -> None:
    assert set(RECIPES) == set(ALLOWED_BACKBONES)
    assert set(DETECTOR_NAMES_BY_BACKBONE) == set(ALLOWED_BACKBONES)


@pytest.mark.parametrize("backbone", sorted(RECIPES))
def test_recipe_builds_valid_config(backbone: str) -> None:
    cfg = build_trainer_config(backbone, catalog="ml_dev", schema="dais26_vfm")
    assert cfg.backbone_name == backbone
    assert cfg.catalog == "ml_dev"
    assert cfg.model_name == DETECTOR_NAMES_BY_BACKBONE[backbone]["model_short"]


@pytest.mark.parametrize("backbone", sorted(RECIPES))
def test_recipes_encode_the_post_fix_detection_recipe(backbone: str) -> None:
    """Every recipe must carry the structural fixes the campaigns proved out
    (per-level stride-scaled anchors + per-class NMS) — the whole point of the
    recipe layer is that quickstarts stop training the legacy config."""
    cfg = build_trainer_config(backbone, catalog="c", schema="s")
    assert cfg.anchor_layout == "per_level"
    assert cfg.nms_per_class is True


def test_amp_policy_per_backbone() -> None:
    """C-RADIO trains fp16; DINOv3 is autocast-unstable and must resolve fp32."""
    cradio = build_trainer_config("cradio_v4_so400m", catalog="c", schema="s")
    dinov3 = build_trainer_config("dinov3_vitl16", catalog="c", schema="s")
    assert cradio.effective_amp_dtype() == "fp16"
    assert dinov3.effective_amp_dtype() == "fp32"


def test_dinov3_recipe_uses_fusion() -> None:
    """dinov3_final's winning lever (capricious-hound-240) is multi-layer fusion."""
    cfg = build_trainer_config("dinov3_vitl16", catalog="c", schema="s")
    assert cfg.fusion_layers == [6, 12, 18, 24]


def test_cradio_recipe_is_the_plain_150ep_winner() -> None:
    """Provenance guard: best C-RADIO is dazzling-mole-850 (cradio_long, 0.5931)
    — plain smooth_l1, NO GIoU, NO Caries oversampling (cradio_final regressed)."""
    cfg = build_trainer_config("cradio_v4_so400m", catalog="c", schema="s")
    assert cfg.epochs == 150
    assert cfg.box_loss_type == "smooth_l1"
    assert cfg.caries_oversample == 1.0
    assert cfg.backbone_mode == "full"


def test_override_precedence_recipe_then_env_then_override() -> None:
    cfg = build_trainer_config(
        "cradio_v4_so400m",
        catalog="c",
        schema="s",
        model_name="custom_model",
        epochs=3,  # demo-time override beats the recipe's 150
    )
    assert cfg.epochs == 3
    assert cfg.model_name == "custom_model"
    # untouched recipe values survive
    assert cfg.lr == pytest.approx(2e-4)


def test_unknown_backbone_raises_with_known_list() -> None:
    with pytest.raises(KeyError, match="cradio_v4_so400m"):
        build_trainer_config("resnet50", catalog="c", schema="s")


def test_invalid_override_fails_validation() -> None:
    with pytest.raises(ValueError, match="epochs"):
        build_trainer_config("cradio_v4_so400m", catalog="c", schema="s", epochs=0)
