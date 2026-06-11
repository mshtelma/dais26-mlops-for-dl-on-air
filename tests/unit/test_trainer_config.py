"""Tests for `dais26_dentex.config.trainer_config.TrainerConfig`.

The point of this dataclass is to be the *one* place a teammate adds a knob.
That role makes it load-bearing, so test the construction + coercion + YAML
round-trip seams hard. Validation should fail loudly with field names.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dais26_dentex.config.trainer_config import (
    ALLOWED_BACKBONES,
    BACKBONE_ALIASES,
    TrainerConfig,
    _coerce_bool,
)

# --- construction --------------------------------------------------------


def test_required_fields_are_catalog_and_schema() -> None:
    cfg = TrainerConfig(catalog="ml_dev", schema="dais26_vfm")
    assert cfg.catalog == "ml_dev"
    assert cfg.schema == "dais26_vfm"
    # Sensible defaults survive
    assert cfg.backbone_name == "cradio_v4_so400m"
    assert cfg.epochs == 10
    assert cfg.use_lora is False


def test_is_frozen() -> None:
    cfg = TrainerConfig(catalog="c", schema="s")
    # `FrozenInstanceError` subclasses `AttributeError`, so this covers both.
    with pytest.raises(AttributeError):
        cfg.epochs = 99  # type: ignore[misc]


def test_with_overrides_returns_new_instance() -> None:
    cfg = TrainerConfig(catalog="c", schema="s")
    cfg2 = cfg.with_overrides(epochs=42, lr=5e-4)
    assert cfg.epochs == 10  # unchanged
    assert cfg2.epochs == 42
    assert cfg2.lr == 5e-4
    assert cfg is not cfg2


# --- from_dict coercion -------------------------------------------------


def test_from_dict_coerces_strings_to_typed_values() -> None:
    """sgcli SHOULD pass typed YAML values, but defensively coerce stringly-
    typed inputs too — that's what the old `_coerce` function did and tests
    here pin the behavior so it doesn't regress."""
    cfg = TrainerConfig.from_dict(
        {
            "catalog": "c",
            "schema": "s",
            "epochs": "20",
            "lr": "0.0005",
            "use_lora": "true",
            "lora_rank": "16",
        }
    )
    assert cfg.epochs == 20
    assert isinstance(cfg.epochs, int)
    assert cfg.lr == pytest.approx(5e-4)
    assert cfg.use_lora is True
    assert cfg.lora_rank == 16


def test_from_dict_handles_typed_values_unchanged() -> None:
    cfg = TrainerConfig.from_dict(
        {
            "catalog": "c",
            "schema": "s",
            "epochs": 5,
            "lr": 1e-4,
            "use_lora": True,
        }
    )
    assert cfg.epochs == 5
    assert cfg.lr == 1e-4
    assert cfg.use_lora is True


def test_from_dict_silently_drops_unknown_keys() -> None:
    """Tolerant of unknown YAML keys (cli.py warns; we just drop)."""
    cfg = TrainerConfig.from_dict(
        {
            "catalog": "c",
            "schema": "s",
            "completely_made_up_key": "ignored",
            "another": 42,
        }
    )
    assert cfg.catalog == "c"


def test_from_dict_resolves_backbone_aliases() -> None:
    cfg = TrainerConfig.from_dict(
        {
            "catalog": "c",
            "schema": "s",
            "backbone_name": "nvidia/C-RADIOv4-SO400M",
        }
    )
    assert cfg.backbone_name == "cradio_v4_so400m"


@pytest.mark.parametrize("alias,canonical", list(BACKBONE_ALIASES.items()))
def test_all_backbone_aliases_resolve(alias: str, canonical: str) -> None:
    """Each entry in BACKBONE_ALIASES should resolve to a canonical name and
    that canonical name should appear in ALLOWED_BACKBONES."""
    cfg = TrainerConfig.from_dict(
        {
            "catalog": "c",
            "schema": "s",
            "backbone_name": alias,
        }
    )
    assert cfg.backbone_name == canonical
    assert canonical in ALLOWED_BACKBONES


def test_from_dict_preserves_none() -> None:
    cfg = TrainerConfig.from_dict(
        {
            "catalog": "c",
            "schema": "s",
            "backbone_revision": None,
            "volume_path": None,
            "num_classes": None,
        }
    )
    assert cfg.backbone_revision is None
    assert cfg.volume_path is None
    assert cfg.num_classes is None


# --- _coerce_bool -------------------------------------------------------


@pytest.mark.parametrize(
    "v,expected",
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
        ("on", True),
        ("off", False),
        ("y", True),
        ("n", False),
        ("", False),
        ("  true  ", True),
    ],
)
def test_coerce_bool_accepted_shapes(v: object, expected: bool) -> None:
    assert _coerce_bool(v) is expected


@pytest.mark.parametrize("v", ["maybe", "definitely", "tru", "ja", object()])
def test_coerce_bool_rejects_garbage(v: object) -> None:
    with pytest.raises(ValueError, match="Cannot coerce"):
        _coerce_bool(v)


# --- from_yaml ----------------------------------------------------------


def test_from_yaml_round_trip(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent("""
        catalog: ml_dev
        schema: dais26_vfm
        backbone_name: nvidia/C-RADIOv4-SO400M
        epochs: 3
        lr: 0.0005
        use_lora: true
        lora_rank: 16
        lora_alpha: 32.0
        img_size: 512
    """).strip()
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text)

    cfg = TrainerConfig.from_yaml(p)
    assert cfg.catalog == "ml_dev"
    assert cfg.schema == "dais26_vfm"
    assert cfg.backbone_name == "cradio_v4_so400m"
    assert cfg.epochs == 3
    assert cfg.lr == pytest.approx(5e-4)
    assert cfg.use_lora is True
    assert cfg.lora_rank == 16
    assert cfg.img_size == 512


def test_from_yaml_rejects_non_mapping_root(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n- c\n")
    with pytest.raises(ValueError, match="did not produce a mapping"):
        TrainerConfig.from_yaml(p)


def test_from_yaml_empty_file_works(tmp_path: Path) -> None:
    """An empty YAML still needs catalog/schema as required positional kwargs;
    we expect a TypeError from the dataclass `__init__`, not a parser
    crash."""
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(TypeError):
        TrainerConfig.from_yaml(p)


# --- validate -----------------------------------------------------------


def test_validate_passes_on_default_config() -> None:
    cfg = TrainerConfig(catalog="c", schema="s")
    cfg.validate()  # no raise


def test_validate_rejects_dotted_catalog() -> None:
    cfg = TrainerConfig(catalog="bad.dotted", schema="s")
    with pytest.raises(ValueError, match="catalog must be"):
        cfg.validate()


def test_validate_rejects_unknown_backbone() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", backbone_name="not_a_real_backbone")
    with pytest.raises(ValueError, match="backbone_name"):
        cfg.validate()


def test_validate_rejects_zero_epochs() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", epochs=0)
    with pytest.raises(ValueError, match="epochs"):
        cfg.validate()


def test_validate_rejects_negative_lr() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", lr=-1e-4)
    with pytest.raises(ValueError, match="lr must be"):
        cfg.validate()


def test_validate_rejects_pct_start_outside_open_interval() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", onecycle_pct_start=0.0)
    with pytest.raises(ValueError, match="onecycle_pct_start"):
        cfg.validate()
    cfg = TrainerConfig(catalog="c", schema="s", onecycle_pct_start=1.0)
    with pytest.raises(ValueError, match="onecycle_pct_start"):
        cfg.validate()


def test_validate_rejects_img_size_not_multiple_of_16() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", img_size=1023)
    with pytest.raises(ValueError, match="img_size"):
        cfg.validate()


def test_validate_collects_multiple_errors() -> None:
    """Validate should report ALL errors in one go so the user can fix them
    in one round-trip — not one-at-a-time."""
    cfg = TrainerConfig(
        catalog="bad.dot",
        schema="s",
        epochs=0,
        lr=-1,
        batch_size=0,
    )
    with pytest.raises(ValueError) as exc_info:
        cfg.validate()
    msg = str(exc_info.value)
    assert "catalog" in msg
    assert "epochs" in msg
    assert "lr" in msg
    assert "batch_size" in msg


def test_validate_lora_only_checked_when_enabled() -> None:
    """If `use_lora=False`, lora_rank=0 shouldn't be a validation error."""
    cfg = TrainerConfig(catalog="c", schema="s", use_lora=False, lora_rank=0)
    cfg.validate()  # no raise

    cfg2 = TrainerConfig(catalog="c", schema="s", use_lora=True, lora_rank=0)
    with pytest.raises(ValueError, match="lora_rank"):
        cfg2.validate()


# --- to_mlflow_params ---------------------------------------------------


def test_to_mlflow_params_stringifies_everything() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", epochs=3, use_lora=True)
    params = cfg.to_mlflow_params()
    assert all(isinstance(v, str) for v in params.values())
    assert params["catalog"] == "c"
    assert params["epochs"] == "3"
    assert params["use_lora"] == "True"


def test_to_mlflow_params_renders_none_as_empty_string() -> None:
    """Empty string is nicer for MLflow run-diff UI than a missing key."""
    cfg = TrainerConfig(catalog="c", schema="s")
    params = cfg.to_mlflow_params()
    assert params["volume_path"] == ""
    assert params["backbone_revision"] == ""


def test_to_mlflow_params_includes_every_field() -> None:
    """Forward-compat: if someone adds a field to TrainerConfig and forgets
    to extend `to_mlflow_params`, this test breaks."""
    import dataclasses

    cfg = TrainerConfig(catalog="c", schema="s")
    params = cfg.to_mlflow_params()
    field_names = {f.name for f in dataclasses.fields(cfg)}
    assert set(params.keys()) == field_names


# --- to_dict ------------------------------------------------------------


def test_to_dict_contains_all_fields() -> None:
    import dataclasses

    cfg = TrainerConfig(catalog="c", schema="s")
    d = cfg.to_dict()
    assert {f.name for f in dataclasses.fields(cfg)} == set(d.keys())


# --- anchor + backbone-mode knobs ---------------------------------------


def test_new_knob_defaults() -> None:
    cfg = TrainerConfig(catalog="c", schema="s")
    assert cfg.anchor_scales is None
    assert cfg.aspect_ratios is None
    assert cfg.backbone_mode == "frozen"
    assert cfg.backbone_lr == pytest.approx(1e-5)
    assert cfg.backbone_trainable_blocks == 0


def test_from_dict_coerces_anchor_lists_and_backbone_fields() -> None:
    cfg = TrainerConfig.from_dict(
        {
            "catalog": "c",
            "schema": "s",
            "anchor_scales": ["16", "32", 64],  # mixed str/int -> ints
            "aspect_ratios": ["0.5", 1, "2.0"],  # -> floats
            "backbone_mode": "partial",
            "backbone_lr": "2e-5",
            "backbone_trainable_blocks": "4",
        }
    )
    assert cfg.anchor_scales == [16, 32, 64]
    assert all(isinstance(s, int) for s in cfg.anchor_scales)
    assert cfg.aspect_ratios == [0.5, 1.0, 2.0]
    assert all(isinstance(r, float) for r in cfg.aspect_ratios)
    assert cfg.backbone_mode == "partial"
    assert cfg.backbone_lr == pytest.approx(2e-5)
    assert cfg.backbone_trainable_blocks == 4


def test_validate_rejects_unknown_backbone_mode() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", backbone_mode="turbo")
    with pytest.raises(ValueError, match="backbone_mode"):
        cfg.validate()


def test_validate_rejects_nonpositive_backbone_lr() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", backbone_lr=0.0)
    with pytest.raises(ValueError, match="backbone_lr"):
        cfg.validate()


def test_validate_partial_requires_trainable_blocks() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", backbone_mode="partial", backbone_trainable_blocks=0)
    with pytest.raises(ValueError, match="partial"):
        cfg.validate()
    # With >=1 blocks it passes.
    TrainerConfig(catalog="c", schema="s", backbone_mode="partial", backbone_trainable_blocks=2).validate()


def test_validate_rejects_bad_anchor_lists() -> None:
    with pytest.raises(ValueError, match="anchor_scales"):
        TrainerConfig(catalog="c", schema="s", anchor_scales=[]).validate()
    with pytest.raises(ValueError, match="anchor_scales"):
        TrainerConfig(catalog="c", schema="s", anchor_scales=[16, -1]).validate()
    with pytest.raises(ValueError, match="aspect_ratios"):
        TrainerConfig(catalog="c", schema="s", aspect_ratios=[0.0, 1.0]).validate()


def test_validate_accepts_valid_anchor_lists() -> None:
    TrainerConfig(
        catalog="c", schema="s", anchor_scales=[16, 32, 64, 128], aspect_ratios=[0.5, 1.0, 2.0]
    ).validate()


def test_effective_backbone_mode_maps_legacy_use_lora() -> None:
    # Legacy: use_lora=True with default frozen mode resolves to "lora".
    cfg = TrainerConfig(catalog="c", schema="s", use_lora=True)
    assert cfg.effective_backbone_mode() == "lora"
    # Explicit backbone_mode wins over use_lora.
    cfg2 = TrainerConfig(catalog="c", schema="s", use_lora=True, backbone_mode="full")
    assert cfg2.effective_backbone_mode() == "full"
    # Plain default stays frozen.
    assert TrainerConfig(catalog="c", schema="s").effective_backbone_mode() == "frozen"


def test_to_mlflow_params_stringifies_new_knobs() -> None:
    cfg = TrainerConfig(
        catalog="c", schema="s", anchor_scales=[16, 32], backbone_mode="full", backbone_lr=1e-5
    )
    params = cfg.to_mlflow_params()
    assert params["anchor_scales"] == "[16, 32]"
    assert params["aspect_ratios"] == ""  # None -> ""
    assert params["backbone_mode"] == "full"
    assert params["backbone_lr"] == str(1e-5)


# --- anchor-layout + NMS knobs ------------------------------------------


def test_anchor_layout_knob_defaults() -> None:
    cfg = TrainerConfig(catalog="c", schema="s")
    assert cfg.anchor_layout == "absolute"
    assert cfg.anchor_base_scale == pytest.approx(4.0)
    assert cfg.anchor_octaves is None
    assert cfg.nms_per_class is False


def test_from_dict_coerces_anchor_layout_knobs() -> None:
    cfg = TrainerConfig.from_dict(
        {
            "catalog": "c",
            "schema": "s",
            "anchor_layout": "per_level",
            "anchor_base_scale": "4.0",
            "anchor_octaves": ["1", "1.26", 1.587],  # -> floats
            "nms_per_class": "true",
        }
    )
    assert cfg.anchor_layout == "per_level"
    assert cfg.anchor_base_scale == pytest.approx(4.0)
    assert cfg.anchor_octaves == [1.0, 1.26, 1.587]
    assert all(isinstance(o, float) for o in cfg.anchor_octaves)
    assert cfg.nms_per_class is True


def test_validate_rejects_bad_anchor_layout() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", anchor_layout="pyramid")
    with pytest.raises(ValueError, match="anchor_layout"):
        cfg.validate()


def test_validate_rejects_nonpositive_anchor_base_scale() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", anchor_base_scale=0.0)
    with pytest.raises(ValueError, match="anchor_base_scale"):
        cfg.validate()


def test_validate_rejects_bad_anchor_octaves() -> None:
    with pytest.raises(ValueError, match="anchor_octaves"):
        TrainerConfig(catalog="c", schema="s", anchor_octaves=[]).validate()
    with pytest.raises(ValueError, match="anchor_octaves"):
        TrainerConfig(catalog="c", schema="s", anchor_octaves=[1.0, -0.5]).validate()


def test_validate_accepts_valid_per_level_config() -> None:
    TrainerConfig(
        catalog="c",
        schema="s",
        anchor_layout="per_level",
        anchor_base_scale=4.0,
        anchor_octaves=[1.0, 1.26, 1.587],
        nms_per_class=True,
    ).validate()


# --- eval-time thresholds + grad-accum + augmentation knobs (push-to-0.60) ---


def test_threshold_and_aug_knob_defaults_are_legacy() -> None:
    """Defaults must reproduce the legacy behavior (no behavior change for
    existing runs)."""
    cfg = TrainerConfig(catalog="c", schema="s")
    assert cfg.score_threshold == pytest.approx(0.05)
    assert cfg.nms_iou_threshold == pytest.approx(0.5)
    assert cfg.max_detections == 100
    assert cfg.grad_accum_steps == 1
    assert cfg.aug_hflip_prob == pytest.approx(0.5)
    assert cfg.aug_jitter_prob == pytest.approx(0.5)
    assert cfg.aug_jitter_scale == pytest.approx(1.0)
    assert cfg.aug_rotation_deg == pytest.approx(0.0)
    assert cfg.aug_multiscale_range is None


def test_from_dict_coerces_threshold_aug_and_grad_accum() -> None:
    cfg = TrainerConfig.from_dict(
        {
            "catalog": "c",
            "schema": "s",
            "score_threshold": "0.01",
            "nms_iou_threshold": "0.6",
            "max_detections": "200",
            "grad_accum_steps": "4",
            "aug_rotation_deg": "7",
            "aug_jitter_scale": "1.5",
            "aug_multiscale_range": ["0.8", 1.0],  # -> floats
        }
    )
    assert cfg.score_threshold == pytest.approx(0.01)
    assert cfg.nms_iou_threshold == pytest.approx(0.6)
    assert cfg.max_detections == 200
    assert isinstance(cfg.max_detections, int)
    assert cfg.grad_accum_steps == 4
    assert cfg.aug_rotation_deg == pytest.approx(7.0)
    assert cfg.aug_jitter_scale == pytest.approx(1.5)
    assert cfg.aug_multiscale_range == [0.8, 1.0]
    assert all(isinstance(x, float) for x in cfg.aug_multiscale_range)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"score_threshold": 1.0}, "score_threshold"),
        ({"score_threshold": -0.1}, "score_threshold"),
        ({"nms_iou_threshold": 0.0}, "nms_iou_threshold"),
        ({"nms_iou_threshold": 1.5}, "nms_iou_threshold"),
        ({"max_detections": 0}, "max_detections"),
        ({"grad_accum_steps": 0}, "grad_accum_steps"),
        ({"aug_hflip_prob": 1.5}, "aug_hflip_prob"),
        ({"aug_jitter_prob": -0.1}, "aug_jitter_prob"),
        ({"aug_jitter_scale": -1.0}, "aug_jitter_scale"),
        ({"aug_rotation_deg": -5.0}, "aug_rotation_deg"),
        ({"aug_multiscale_range": [1.0]}, "aug_multiscale_range"),
        ({"aug_multiscale_range": [0.9, 0.8]}, "aug_multiscale_range"),
        ({"aug_multiscale_range": [0.0, 1.0]}, "aug_multiscale_range"),
        ({"aug_multiscale_range": [0.5, 1.2]}, "aug_multiscale_range"),
    ],
)
def test_validate_rejects_bad_threshold_aug_values(kwargs: dict, match: str) -> None:
    cfg = TrainerConfig(catalog="c", schema="s", **kwargs)
    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_validate_accepts_valid_threshold_aug_config() -> None:
    TrainerConfig(
        catalog="c",
        schema="s",
        score_threshold=0.01,
        nms_iou_threshold=0.6,
        max_detections=300,
        grad_accum_steps=4,
        aug_hflip_prob=0.5,
        aug_jitter_prob=0.5,
        aug_jitter_scale=1.5,
        aug_rotation_deg=7.0,
        aug_multiscale_range=[0.7, 1.0],
    ).validate()


# --- Track B levers: fusion_layers / box_loss_type / caries_oversample ---


def test_defaults_are_legacy_for_track_b() -> None:
    cfg = TrainerConfig(catalog="c", schema="s")
    assert cfg.fusion_layers is None
    assert cfg.box_loss_type == "smooth_l1"
    assert cfg.caries_oversample == pytest.approx(1.0)


def test_from_dict_coerces_track_b_fields() -> None:
    cfg = TrainerConfig.from_dict(
        {
            "catalog": "c",
            "schema": "s",
            "backbone_name": "dinov3_vitl16",
            "fusion_layers": ["6", 12, "18", 24],  # -> ints
            "box_loss_type": "giou",
            "caries_oversample": "2.0",  # -> float
        }
    )
    assert cfg.fusion_layers == [6, 12, 18, 24]
    assert all(isinstance(x, int) for x in cfg.fusion_layers)
    assert cfg.box_loss_type == "giou"
    assert cfg.caries_oversample == pytest.approx(2.0)
    cfg.validate()  # valid combo


def test_validate_fusion_requires_dinov3() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", backbone_name="cradio_v4_so400m", fusion_layers=[6, 12])
    with pytest.raises(ValueError, match="fusion_layers"):
        cfg.validate()


def test_validate_fusion_empty_list_rejected() -> None:
    cfg = TrainerConfig(catalog="c", schema="s", backbone_name="dinov3_vitl16", fusion_layers=[])
    with pytest.raises(ValueError, match="fusion_layers"):
        cfg.validate()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"box_loss_type": "iou"}, "box_loss_type"),
        ({"caries_oversample": 0.5}, "caries_oversample"),
    ],
)
def test_validate_rejects_bad_track_b_values(kwargs: dict, match: str) -> None:
    cfg = TrainerConfig(catalog="c", schema="s", **kwargs)
    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_to_mlflow_params_includes_track_b_fields() -> None:
    cfg = TrainerConfig(
        catalog="c",
        schema="s",
        backbone_name="dinov3_vitl16",
        fusion_layers=[6, 12, 18, 24],
        box_loss_type="giou",
        caries_oversample=2.0,
    )
    params = cfg.to_mlflow_params()
    assert params["box_loss_type"] == "giou"
    assert params["caries_oversample"] == "2.0"
    assert params["fusion_layers"] == "[6, 12, 18, 24]"
