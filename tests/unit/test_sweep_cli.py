"""Tests for `train.sweep_cli` — the air/torchrun sweep entrypoint."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from dais26_dentex.config.campaigns import CAMPAIGN_STAGES
from dais26_dentex.config.environments import ENVIRONMENTS, load_environment
from dais26_dentex.train import sweep_cli

REPO = Path(__file__).resolve().parents[2]
SWEEP_WORKLOAD = REPO / "air" / "workload_sweep.yaml"


def _write(tmp_path: Path, payload: dict) -> str:
    p = tmp_path / "hp.yaml"
    p.write_text(yaml.safe_dump(payload))
    return str(p)


def test_load_sweep_inputs_resolves_stage_and_defaults(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "stage": "dinov3_s1",
            "catalog": "c",
            "schema": "s",
            "volume_path": "/Volumes/c/s/raw",
            "experiment_name": "/Users/x/exp",
        },
    )
    spec, base = sweep_cli.load_sweep_inputs(path)
    assert spec.stage_name == "dinov3_s1"
    assert spec.backbone == "dinov3_vitl16"
    assert spec.max_trials == CAMPAIGN_STAGES["dinov3_s1"].max_trials
    # model name derived from the stage's backbone; env values pass through.
    assert base["model_name"] == "dinov3_detector"
    assert base["catalog"] == "c"
    assert "stage" not in base  # consumed, not forwarded into TrainerConfig


def test_load_sweep_inputs_strategy_and_seed_are_consumed(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {"stage": "cradio_s1", "catalog": "c", "schema": "s", "strategy": "grid", "seed": 7},
    )
    spec, base = sweep_cli.load_sweep_inputs(path)
    assert spec.strategy == "grid"
    assert spec.seed == 7
    assert "strategy" not in base and "seed" not in base


def test_load_sweep_inputs_env_resolves_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`env: df1` supplies the UC locations into the base kwargs, so the sweep
    workload need not restate catalog/schema/volume/experiment."""
    for var in ("DAIS26_ENV", "DAIS26_CATALOG", "DAIS26_SCHEMA", "DAIS26_EXPERIMENT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DAIS26_ENV_FILE", "/nonexistent/dais26-no-overlay.yaml")
    path = _write(tmp_path, {"stage": "dinov3_s1", "env": "df1"})
    spec, base = sweep_cli.load_sweep_inputs(path)
    df1 = load_environment("df1")
    assert base["catalog"] == df1.catalog == "main"
    assert base["schema"] == df1.schema == "mshtelma"
    assert base["experiment_name"] == df1.experiment_name
    assert base["volume_path"] == df1.volume_path
    assert "env" not in base  # consumed, not forwarded into TrainerConfig
    assert spec.backbone == "dinov3_vitl16"


def test_load_sweep_inputs_accepts_air_full_spec_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """air writes the FULL workload spec to $HYPERPARAMETERS_PATH with the user
    params NESTED under `parameters:` — regression for the E2E '`stage` is
    required' failure (flat-dict unit fixtures masked it)."""
    for var in ("DAIS26_ENV", "DAIS26_CATALOG", "DAIS26_SCHEMA", "DAIS26_EXPERIMENT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DAIS26_ENV_FILE", "/nonexistent/dais26-no-overlay.yaml")
    path = _write(
        tmp_path,
        {
            "experiment_name": "dais26-mlops-for-dl-on-air",
            "compute": {"num_accelerators": 8},
            "parameters": {"stage": "smoke", "env": "df1"},
        },
    )
    spec, base = sweep_cli.load_sweep_inputs(path)
    assert spec.stage_name == "smoke"
    assert spec.backbone == "cradio_v4_so400m"
    assert base["catalog"] == "main"
    assert "env" not in base


def test_load_sweep_inputs_requires_stage(tmp_path: Path) -> None:
    path = _write(tmp_path, {"catalog": "c", "schema": "s"})
    with pytest.raises(ValueError, match="`stage` is required"):
        sweep_cli.load_sweep_inputs(path)


def test_load_sweep_inputs_rejects_unknown_stage(tmp_path: Path) -> None:
    path = _write(tmp_path, {"stage": "dinov3_s99", "catalog": "c", "schema": "s"})
    with pytest.raises(ValueError, match="Unknown stage"):
        sweep_cli.load_sweep_inputs(path)


def test_main_dry_run_prints_spec_without_mlflow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(
        tmp_path,
        {"stage": "dinov3_s1", "catalog": "c", "schema": "s", "experiment_name": "/Users/x/exp"},
    )
    rc = sweep_cli.main(["--config", path, "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "stage=dinov3_s1" in out
    assert "backbone=dinov3_vitl16" in out


def test_main_clears_ambient_mlflow_run_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sweep owns its parent + per-trial runs, so `air`'s ambient
    MLFLOW_RUN_ID (the workload's own run) must be cleared on every rank before
    any run is created — otherwise each trial's fluent start_run() would attach
    to the workload run. The guard runs before the dry-run return, so --dry-run
    exercises it without MLflow."""
    path = _write(
        tmp_path,
        {"stage": "dinov3_s1", "catalog": "c", "schema": "s", "experiment_name": "/Users/x/exp"},
    )
    monkeypatch.setenv("MLFLOW_RUN_ID", "workload-run-xyz")
    rc = sweep_cli.main(["--config", path, "--dry-run"])
    assert rc == 0
    assert "MLFLOW_RUN_ID" not in os.environ


def test_main_returns_2_when_no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPERPARAMETERS_PATH", raising=False)
    assert sweep_cli.main([]) == 2


# ---------------------------------------------------------------------------
# Workload structural contract
# ---------------------------------------------------------------------------


def test_sweep_workload_targets_the_sweep_cli_module() -> None:
    with SWEEP_WORKLOAD.open() as f:
        workload = yaml.safe_load(f)
    assert "torchrun" in workload["command"]
    assert "-m dais26_dentex.train.sweep_cli" in workload["command"]
    assert workload["compute"] == {"num_accelerators": 8, "accelerator_type": "GPU_8xH100"}
    # A retry would re-run already-finished trials inside the same parent.
    assert workload["max_retries"] == 0


def test_sweep_workload_parameters_are_stage_plus_environment() -> None:
    with SWEEP_WORKLOAD.open() as f:
        params = yaml.safe_load(f)["parameters"]
    assert params["stage"] in CAMPAIGN_STAGES
    assert params["env"] in ENVIRONMENTS
    allowed = {"stage", "env", "model_name", "backbone_revision", "strategy", "seed"}
    assert set(params) <= allowed, f"unexpected sweep workload params: {set(params) - allowed}"
    # UC locations come from the named env, not restated in the workload.
    restated = {"catalog", "schema", "volume_path", "cache_dir", "experiment_name"} & set(params)
    assert not restated, f"sweep workload restates env-derived keys: {restated}"
    # the named env must pin the shared experiment the gates read
    assert load_environment(params["env"]).experiment_name.endswith("dais26_vfm_experiment")
