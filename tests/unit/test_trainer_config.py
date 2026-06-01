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


# --- to_kwargs_for_train_detector ---------------------------------------


def test_to_kwargs_for_train_detector_subset_matches_legacy_signature() -> None:
    """The kwargs dict must be exactly the legacy `train_detector` signature
    minus self. If `train_detector` adds a kwarg, this test breaks until we
    extend `to_kwargs_for_train_detector` (intentional — forces visibility)."""
    import inspect

    from dais26_dentex.train.train_detector import train_detector

    legacy_keys = set(inspect.signature(train_detector).parameters.keys())

    cfg = TrainerConfig(catalog="c", schema="s")
    kwargs = cfg.to_kwargs_for_train_detector()
    assert set(kwargs.keys()) == legacy_keys, (
        f"Drift detected:\n"
        f"  Missing from cfg: {legacy_keys - set(kwargs.keys())}\n"
        f"  Extra in cfg:     {set(kwargs.keys()) - legacy_keys}"
    )


def test_to_kwargs_round_trip_through_train_detector_signature() -> None:
    """The kwargs dict can actually be unpacked into `train_detector` (we
    don't call it — that needs the full env — but inspect.bind validates
    the shape)."""
    import inspect

    from dais26_dentex.train.train_detector import train_detector

    cfg = TrainerConfig(catalog="c", schema="s")
    sig = inspect.signature(train_detector)
    sig.bind(**cfg.to_kwargs_for_train_detector())  # raises TypeError on mismatch


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
