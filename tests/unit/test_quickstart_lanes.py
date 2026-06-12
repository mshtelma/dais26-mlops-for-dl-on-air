"""Structural tests for the two supported quickstart lanes.

The lanes share one source of hyperparameter truth (`config.recipes`): the
notebook builds via `build_trainer_config(BACKBONE, ...)`, the air workload
names the same recipe in its `parameters:` block and resolves through
`train.cli.load_config`. These tests pin that contract — workload parameters
must stay environment-only (plus deliberate, allowlisted overrides), never a
re-statement of hyperparameters.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import yaml

from dais26_dentex.config.environments import load_environment
from dais26_dentex.config.recipes import DETECTOR_NAMES_BY_BACKBONE, RECIPES
from dais26_dentex.config.trainer_config import TrainerConfig
from dais26_dentex.train import cli


@pytest.fixture(autouse=True)
def _no_ambient_dais26_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`env: df1` must resolve to the committed environment, not a tester's
    ambient DAIS26_* / overlay."""
    for var in (
        "DAIS26_ENV",
        "DAIS26_CATALOG",
        "DAIS26_SCHEMA",
        "DAIS26_EXPERIMENT",
        "DAIS26_VOLUME_PATH",
        "DAIS26_CACHE_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DAIS26_ENV_FILE", "/nonexistent/dais26-no-overlay.yaml")

REPO = Path(__file__).resolve().parents[2]
DAB_TRAIN_JOB = REPO / "resources" / "jobs" / "train_detector.yml"
DAB_TRAIN_NOTEBOOK = REPO / "notebooks" / "02_train_detector_air.py"
AIR_WORKLOADS = {
    "cradio_v4_so400m": REPO / "air" / "workload_train_detector.yaml",
    "dinov3_vitl16": REPO / "air" / "workload_train_detector_dinov3.yaml",
}

# Workload parameters that are legitimately environment/launch concerns rather
# than hyperparameters. Anything else in `parameters:` must be a TrainerConfig
# field deliberately overriding the recipe — and the recipe-critical knobs may
# not be silently overridden at all.
# UC locations now collapse to a single `env:` key (config.environments); the
# remaining non-hyperparameter keys are model identity + launch flags.
ENV_KEYS = {
    "recipe",
    "env",
    "model_name",
    "backbone_revision",
    "register_model",
    "set_candidate_alias",
}
ALLOWED_OVERRIDES = {"epochs", "base_seed", "num_workers", "batch_size"}
RECIPE_CRITICAL = {
    "anchor_layout",
    "nms_per_class",
    "amp_dtype",
    "fusion_layers",
    "box_loss_type",
    "backbone_mode",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data


def _task(tasks: list[dict[str, Any]], key: str) -> dict[str, Any]:
    matches = [task for task in tasks if task.get("task_key") == key]
    assert len(matches) == 1, f"expected exactly one task named {key!r}"
    return matches[0]


# ---------------------------------------------------------------------------
# DAB lane
# ---------------------------------------------------------------------------


def test_dab_quickstart_stops_after_challenger_confirmation() -> None:
    data = _load_yaml(DAB_TRAIN_JOB)
    job = data["targets"]["dev"]["resources"]["jobs"]["train_detector"]
    tasks = job["tasks"]
    task_keys = [task["task_key"] for task in tasks]

    assert task_keys == ["setup", "train", "confirm_challenger"]
    assert "deploy_endpoint" not in task_keys

    train = _task(tasks, "train")
    assert train["notebook_task"]["notebook_path"] == "../../notebooks/02_train_detector_air.py"
    assert train["compute"]["hardware_accelerator"] == "GPU_8xH100"

    confirm = _task(tasks, "confirm_challenger")
    assert confirm["notebook_task"]["notebook_path"] == "../../notebooks/04_deploy_serving.py"
    assert confirm["notebook_task"]["base_parameters"]["deploy_action"] == "register_and_set_candidate"
    assert confirm["max_retries"] == 0


def test_dab_notebook_trains_from_the_shared_recipe_without_torchrun() -> None:
    text = DAB_TRAIN_NOTEBOOK.read_text()

    assert "does **not** use `torchrun`" in text
    assert "from serverless_gpu import distributed" in text
    assert "from dais26_dentex.config.recipes import build_trainer_config" in text
    assert "from dais26_dentex.train.trainer import Trainer" in text
    assert "Trainer(cfg).run()" in text
    # Hand-listed hyperparameters must not creep back into the notebook.
    assert "TRAIN_LR" not in text
    assert "TRAIN_BATCH_SIZE" not in text


# ---------------------------------------------------------------------------
# air lane
# ---------------------------------------------------------------------------


def test_air_quickstart_keeps_torchrun_on_single_node_h100() -> None:
    workload = _load_yaml(AIR_WORKLOADS["cradio_v4_so400m"])

    assert workload["compute"] == {"num_accelerators": 8, "accelerator_type": "GPU_8xH100"}
    assert "torchrun" in workload["command"]
    assert "-m dais26_dentex.train.cli" in workload["command"]
    assert workload["parameters"]["register_model"] is True
    assert workload["parameters"]["set_candidate_alias"] is True


@pytest.mark.parametrize("backbone", sorted(AIR_WORKLOADS))
def test_air_parameters_are_recipe_plus_environment_only(backbone: str) -> None:
    """Every workload parameter is either an environment value, an allowlisted
    deliberate override, or it fails this test — the duplication the old
    value-mirror test merely guarded is now structurally impossible."""
    params = _load_yaml(AIR_WORKLOADS[backbone])["parameters"]

    assert params["recipe"] == backbone
    assert params["recipe"] in RECIPES

    # `recipe` and `env` are resolution keys, not TrainerConfig fields; every
    # other parameter must be a real field (the air CLI would silently drop a typo).
    trainer_fields = {f.name for f in fields(TrainerConfig)}
    unknown = set(params) - trainer_fields - {"recipe", "env"}
    assert not unknown, f"parameters not understood by TrainerConfig: {unknown}"

    overrides = set(params) - ENV_KEYS
    assert overrides <= ALLOWED_OVERRIDES, (
        f"non-allowlisted hyperparameter overrides in {AIR_WORKLOADS[backbone].name}: "
        f"{overrides - ALLOWED_OVERRIDES}"
    )
    assert not (set(params) & RECIPE_CRITICAL), (
        "recipe-critical knobs must come from the recipe, not the workload: "
        f"{set(params) & RECIPE_CRITICAL}"
    )

    # The named env must pin the MLflow experiment so air-trained versions are
    # visible to the sweep / deployment-job best-in-experiment gates.
    assert load_environment(params["env"]).experiment_name.endswith("dais26_vfm_experiment")


@pytest.mark.parametrize("backbone", sorted(AIR_WORKLOADS))
def test_air_parameters_resolve_through_the_shared_recipe(backbone: str, tmp_path: Path) -> None:
    """End-to-end lane equivalence: feeding the workload's parameters block
    through the real cli loader must yield the recipe's hyperparameters with
    only the declared overrides applied."""
    params = _load_yaml(AIR_WORKLOADS[backbone])["parameters"]
    hp = tmp_path / "hp.yaml"
    hp.write_text(yaml.safe_dump(params))

    cfg = cli.load_config(str(hp))
    cfg.validate()

    assert cfg.backbone_name == backbone
    assert cfg.model_name == DETECTOR_NAMES_BY_BACKBONE[backbone]["model_short"]
    assert cfg.anchor_layout == "per_level"
    assert cfg.nms_per_class is True
    assert cfg.epochs == params["epochs"]  # the explicit demo override
    # Recipe values not overridden must come through untouched.
    recipe = RECIPES[backbone]
    assert cfg.lr == pytest.approx(recipe["lr"])
    assert cfg.batch_size == recipe["batch_size"]
    # UC locations come from the named env, not restated in the workload.
    env = load_environment(params["env"])
    assert cfg.catalog == env.catalog
    assert cfg.experiment_name == env.experiment_name
