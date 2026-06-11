"""Tests for `dais26_dentex.train.cli` — the YAML → TrainerConfig → Trainer dispatch.

The old `_coerce / _INT_KEYS / filter_to_known_kwargs` helpers moved onto
`TrainerConfig.from_dict` (covered in `test_trainer_config.py`); these tests
exercise the cli surface only — argument parsing, env var handling, and the
end-to-end dispatch into `Trainer(cfg).run()`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
import yaml

from dais26_dentex.train import cli


class _FakeTrainer:
    """Captures the `cfg` passed to `Trainer(cfg)` and returns a canned run_id.

    Set `_FakeTrainer.run_id` / `_FakeTrainer.captured` per test. Mirrors the
    real `Trainer(cfg).run()` seam now that cli builds the trainer directly
    instead of round-tripping through `train_detector`.
    """

    run_id: ClassVar[str | None] = "fake_run_id"
    captured: ClassVar[dict[str, object]] = {}

    def __init__(self, cfg: object) -> None:
        type(self).captured = {"cfg": cfg}

    def run(self) -> str | None:
        return type(self).run_id

# --- _resolve_yaml_path --------------------------------------------------


def test_resolve_yaml_path_prefers_args_over_env() -> None:
    with patch.dict(os.environ, {"HYPERPARAMETERS_PATH": "/env/path.yaml"}):
        assert cli._resolve_yaml_path("/args/path.yaml") == "/args/path.yaml"


def test_resolve_yaml_path_falls_back_to_env() -> None:
    with patch.dict(os.environ, {"HYPERPARAMETERS_PATH": "/env/path.yaml"}):
        assert cli._resolve_yaml_path(None) == "/env/path.yaml"


def test_resolve_yaml_path_returns_none_when_neither_set() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert cli._resolve_yaml_path(None) is None


# --- main: error paths ---------------------------------------------------


def test_main_returns_2_when_no_config_anywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPERPARAMETERS_PATH", raising=False)
    rc = cli.main([])
    assert rc == 2


def test_main_returns_2_when_config_path_missing(tmp_path: Path) -> None:
    nonexistent = str(tmp_path / "missing.yaml")
    rc = cli.main(["--config", nonexistent])
    assert rc == 2


# --- main: dry-run -------------------------------------------------------


def test_main_dry_run_prints_resolved_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "catalog": "ml_dev",
                "schema": "dais26_vfm",
                "epochs": 7,
            }
        )
    )
    rc = cli.main(["--config", str(path), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "catalog: ml_dev" in out
    assert "schema: dais26_vfm" in out
    assert "epochs: 7" in out


def test_main_dry_run_does_not_invoke_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({"catalog": "c", "schema": "s"}))

    called = {"yes": False}

    class _ShouldNotRun:
        def __init__(self, cfg: object) -> None:
            called["yes"] = True

        def run(self) -> str:
            return "should_not_happen"

    monkeypatch.setattr(cli, "Trainer", _ShouldNotRun)
    rc = cli.main(["--config", str(path), "--dry-run"])
    assert rc == 0
    assert called["yes"] is False


# --- main: end-to-end dispatch ------------------------------------------


def test_main_dispatches_to_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HYPERPARAMETERS_PATH → TrainerConfig → Trainer(cfg).run().

    The cli builds the trainer from the fully-typed cfg, so the loss/optimizer
    knobs that the old `to_kwargs_for_train_detector` subset dropped now reach
    the Trainer. We assert the cfg carries both legacy and previously-dropped
    knobs.
    """
    path = tmp_path / "p.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "catalog": "c",
                "schema": "s",
                "epochs": "3",  # string — must be coerced to int
                "use_lora": "false",
                "backbone_name": "nvidia/C-RADIOv4-SO400M",
                "box_loss_weight": "2.0",  # previously dropped by the legacy subset
                "backbone_mode": "full",
                "noise_key": "ignored",  # must be filtered out
            }
        )
    )

    _FakeTrainer.run_id = "fake_run_id"
    monkeypatch.setenv("HYPERPARAMETERS_PATH", str(path))
    monkeypatch.setattr(cli, "Trainer", _FakeTrainer)
    rc = cli.main([])
    assert rc == 0
    cfg = _FakeTrainer.captured["cfg"]
    assert cfg.catalog == "c"
    assert cfg.epochs == 3
    assert cfg.use_lora is False
    assert cfg.backbone_name == "cradio_v4_so400m"
    assert cfg.box_loss_weight == 2.0
    assert cfg.backbone_mode == "full"
    assert not hasattr(cfg, "noise_key")


def test_main_validates_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A semantically bad config (e.g. epochs=0) should fail before the
    trainer ever runs — `cfg.validate()` raises and bubbles up."""
    path = tmp_path / "p.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "catalog": "c",
                "schema": "s",
                "epochs": 0,
            }
        )
    )
    called = {"yes": False}

    class _ShouldNotRun:
        def __init__(self, cfg: object) -> None:
            called["yes"] = True

        def run(self) -> str:
            return ""

    monkeypatch.setattr(cli, "Trainer", _ShouldNotRun)
    with pytest.raises(ValueError, match="epochs"):
        cli.main(["--config", str(path)])
    assert called["yes"] is False


def test_main_prints_model_uri_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({"catalog": "c", "schema": "s"}))

    _FakeTrainer.run_id = "abc123"
    monkeypatch.setattr(cli, "Trainer", _FakeTrainer)
    monkeypatch.setattr(cli, "is_rank0", lambda: True)
    rc = cli.main(["--config", str(path)])
    assert rc == 0
    assert "MODEL_URI=abc123" in capsys.readouterr().out


def test_main_does_not_print_uri_on_non_rank0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On non-rank-0 ranks, MODEL_URI should not be printed (only rank 0
    holds the run_id)."""
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({"catalog": "c", "schema": "s"}))

    _FakeTrainer.run_id = "abc123"
    monkeypatch.setattr(cli, "Trainer", _FakeTrainer)
    monkeypatch.setattr(cli, "is_rank0", lambda: False)
    rc = cli.main(["--config", str(path)])
    assert rc == 0
    assert "MODEL_URI" not in capsys.readouterr().out


# --- load_config: recipe resolution ---------------------------------------


def test_load_config_without_recipe_is_legacy_passthrough(tmp_path: Path) -> None:
    p = tmp_path / "hp.yaml"
    p.write_text(yaml.safe_dump({"catalog": "c", "schema": "s", "epochs": 3}))
    cfg = cli.load_config(str(p))
    assert cfg.catalog == "c"
    assert cfg.epochs == 3
    assert cfg.anchor_layout == "absolute"  # legacy default, no recipe applied


def test_load_config_recipe_applies_best_known_then_overrides(tmp_path: Path) -> None:
    p = tmp_path / "hp.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "recipe": "cradio_v4_so400m",
                "catalog": "c",
                "schema": "s",
                "epochs": 2,
            }
        )
    )
    cfg = cli.load_config(str(p))
    assert cfg.backbone_name == "cradio_v4_so400m"
    assert cfg.anchor_layout == "per_level"  # from the recipe
    assert cfg.nms_per_class is True
    assert cfg.epochs == 2  # explicit override wins over the recipe's 150
    assert cfg.model_name == "cradio_detector"  # derived from the recipe


def test_load_config_unknown_recipe_raises(tmp_path: Path) -> None:
    p = tmp_path / "hp.yaml"
    p.write_text(yaml.safe_dump({"recipe": "resnet50", "catalog": "c", "schema": "s"}))
    with pytest.raises(ValueError, match="Unknown recipe"):
        cli.load_config(str(p))


def test_load_config_recipe_requires_catalog_and_schema(tmp_path: Path) -> None:
    p = tmp_path / "hp.yaml"
    p.write_text(yaml.safe_dump({"recipe": "cradio_v4_so400m", "schema": "s"}))
    with pytest.raises(ValueError, match="catalog"):
        cli.load_config(str(p))
