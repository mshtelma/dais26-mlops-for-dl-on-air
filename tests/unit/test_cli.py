"""Tests for src.train.cli — HYPERPARAMETERS_PATH round-trip + filtering + coercion."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import yaml

from src.train import cli


def test_coerce_int_keys():
    out = cli._coerce({"epochs": "10", "batch_size": "8", "num_workers": "4"})
    assert out["epochs"] == 10 and isinstance(out["epochs"], int)
    assert out["batch_size"] == 8 and isinstance(out["batch_size"], int)
    assert out["num_workers"] == 4 and isinstance(out["num_workers"], int)


def test_coerce_float_keys():
    out = cli._coerce({"lr": "1e-3", "lora_alpha": "32.0"})
    assert out["lr"] == pytest.approx(1e-3)
    assert out["lora_alpha"] == pytest.approx(32.0)


def test_coerce_bool_keys_truthy_strings():
    out = cli._coerce({"use_lora": "true", "register_model": "yes", "set_candidate_alias": "1"})
    assert out["use_lora"] is True
    assert out["register_model"] is True
    assert out["set_candidate_alias"] is True


def test_coerce_bool_keys_falsy_strings():
    out = cli._coerce({"use_lora": "false", "register_model": "no", "set_candidate_alias": "0"})
    assert out["use_lora"] is False
    assert out["register_model"] is False
    assert out["set_candidate_alias"] is False


def test_coerce_preserves_native_bools():
    out = cli._coerce({"use_lora": True})
    assert out["use_lora"] is True


def test_coerce_skips_none_for_numeric_keys():
    """None for numeric keys should NOT be coerced to int(None) — that would raise."""
    out = cli._coerce({"epochs": None, "backbone_revision": None})
    assert out["epochs"] is None


def test_backbone_alias_resolved():
    out = cli._coerce({"backbone_name": "nvidia/C-RADIOv4-SO400M"})
    assert out["backbone_name"] == "cradio_v4_so400m"


def test_backbone_alias_pass_through_for_literal():
    out = cli._coerce({"backbone_name": "cradio_v4_so400m"})
    assert out["backbone_name"] == "cradio_v4_so400m"


def test_load_params_no_env_var_returns_empty():
    with patch.dict(os.environ, {}, clear=True):
        assert cli.load_params() == {}


def test_load_params_missing_file_exits_2(tmp_path):
    nonexistent = str(tmp_path / "missing.yaml")
    with patch.dict(os.environ, {"HYPERPARAMETERS_PATH": nonexistent}):
        with pytest.raises(SystemExit) as exc:
            cli.load_params()
        assert exc.value.code == 2


def test_load_params_reads_yaml(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump({"epochs": 5, "lr": 1e-4}))
    with patch.dict(os.environ, {"HYPERPARAMETERS_PATH": str(path)}):
        params = cli.load_params()
    assert params == {"epochs": 5, "lr": 1e-4}


def test_filter_to_known_kwargs_drops_unknown():
    out = cli.filter_to_known_kwargs({"epochs": 5, "unknown_key": "foo", "another_unknown": 42})
    assert "epochs" in out
    assert "unknown_key" not in out
    assert "another_unknown" not in out


def test_filter_to_known_kwargs_keeps_all_valid():
    # Use a small valid set
    out = cli.filter_to_known_kwargs(
        {"catalog": "c", "schema": "s", "epochs": 3, "batch_size": 4}
    )
    assert out == {"catalog": "c", "schema": "s", "epochs": 3, "batch_size": 4}


def test_main_dispatches_to_train_detector(tmp_path, monkeypatch):
    """End-to-end: HYPERPARAMETERS_PATH → load → coerce → filter → train_detector(**filtered).

    The fake function must declare the same explicit kwargs as the real one so
    inspect.signature() identifies them as valid (otherwise filter_to_known_kwargs
    drops everything against a `**kwargs`-only signature).
    """
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({
        "catalog": "c", "schema": "s",
        "epochs": "3",                 # string — must be coerced to int
        "use_lora": "false",
        "backbone_name": "nvidia/C-RADIOv4-SO400M",
        "noise_key": "ignored",        # must be filtered out
    }))
    captured = {}

    def fake_train_detector(
        catalog: str = "",
        schema: str = "",
        backbone_name: str = "",
        epochs: int = 0,
        use_lora: bool = False,
    ):
        captured.update(
            catalog=catalog, schema=schema,
            backbone_name=backbone_name, epochs=epochs, use_lora=use_lora,
        )
        return "fake_run_id"

    monkeypatch.setenv("HYPERPARAMETERS_PATH", str(path))
    monkeypatch.setattr(cli, "train_detector", fake_train_detector)
    rc = cli.main()
    assert rc == 0
    assert captured["catalog"] == "c"
    assert captured["epochs"] == 3
    assert captured["use_lora"] is False
    assert captured["backbone_name"] == "cradio_v4_so400m"
    assert "noise_key" not in captured
