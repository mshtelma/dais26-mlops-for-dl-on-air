"""Tests for `dais26_dentex.train.cli` — the YAML → TrainerConfig → train_detector dispatch.

The old `_coerce / _INT_KEYS / filter_to_known_kwargs` helpers moved onto
`TrainerConfig.from_dict` (covered in `test_trainer_config.py`); these tests
exercise the cli surface only — argument parsing, env var handling, and the
end-to-end dispatch into `train_detector`.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from dais26_dentex.train import cli

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

    def trainer_should_not_be_called(**_: object) -> str:
        called["yes"] = True
        return "should_not_happen"

    monkeypatch.setattr(cli, "train_detector", trainer_should_not_be_called)
    rc = cli.main(["--config", str(path), "--dry-run"])
    assert rc == 0
    assert called["yes"] is False


# --- main: end-to-end dispatch ------------------------------------------


def test_main_dispatches_to_train_detector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HYPERPARAMETERS_PATH → TrainerConfig → train_detector(**cfg)."""
    path = tmp_path / "p.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "catalog": "c",
                "schema": "s",
                "epochs": "3",  # string — must be coerced to int
                "use_lora": "false",
                "backbone_name": "nvidia/C-RADIOv4-SO400M",
                "noise_key": "ignored",  # must be filtered out
            }
        )
    )
    captured: dict[str, object] = {}

    def fake_train_detector(**kwargs: object) -> str:
        captured.update(kwargs)
        return "fake_run_id"

    monkeypatch.setenv("HYPERPARAMETERS_PATH", str(path))
    monkeypatch.setattr(cli, "train_detector", fake_train_detector)
    rc = cli.main([])
    assert rc == 0
    assert captured["catalog"] == "c"
    assert captured["epochs"] == 3
    assert captured["use_lora"] is False
    assert captured["backbone_name"] == "cradio_v4_so400m"
    assert "noise_key" not in captured


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

    def trainer_should_not_be_called(**_: object) -> str:
        called["yes"] = True
        return ""

    monkeypatch.setattr(cli, "train_detector", trainer_should_not_be_called)
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

    monkeypatch.setattr(cli, "train_detector", lambda **_: "abc123")
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

    monkeypatch.setattr(cli, "train_detector", lambda **_: "abc123")
    monkeypatch.setattr(cli, "is_rank0", lambda: False)
    rc = cli.main(["--config", str(path)])
    assert rc == 0
    assert "MODEL_URI" not in capsys.readouterr().out
