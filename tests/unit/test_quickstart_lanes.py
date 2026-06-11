"""Structural tests for the two supported quickstart lanes."""

from __future__ import annotations

import ast
import contextlib
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
DAB_CONFIG = REPO / "notebooks" / "00_config.py"
DAB_TRAIN_JOB = REPO / "resources" / "jobs" / "train_detector.yml"
DAB_TRAIN_NOTEBOOK = REPO / "notebooks" / "02_train_detector_air.py"
SGCLI_WORKLOAD = REPO / "sgcli" / "workload_train_detector.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data


def _task(tasks: list[dict[str, Any]], key: str) -> dict[str, Any]:
    matches = [task for task in tasks if task.get("task_key") == key]
    assert len(matches) == 1, f"expected exactly one task named {key!r}"
    return matches[0]


def _literal_assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text())
    values: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            with contextlib.suppress(ValueError):
                values[node.targets[0].id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            with contextlib.suppress(ValueError):
                values[node.target.id] = ast.literal_eval(node.value)
    return values


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


def test_dab_notebook_uses_shared_trainer_without_torchrun() -> None:
    text = DAB_TRAIN_NOTEBOOK.read_text()

    assert "does **not** use `torchrun`" in text
    assert "from serverless_gpu import distributed" in text
    assert "from dais26_dentex.config.trainer_config import TrainerConfig" in text
    assert "from dais26_dentex.train.trainer import Trainer" in text
    assert "Trainer(cfg).run()" in text


def test_sgcli_quickstart_keeps_torchrun_on_single_node_h100() -> None:
    workload = _load_yaml(SGCLI_WORKLOAD)

    assert workload["compute"] == {"gpus": 8, "gpu_type": "h100"}
    assert "torchrun" in workload["command"]
    assert "-m dais26_dentex.train.cli" in workload["command"]
    assert workload["parameters"]["register_model"] is True
    assert workload["parameters"]["set_candidate_alias"] is True


def test_sgcli_default_parameters_match_dab_quickstart_defaults() -> None:
    config = _literal_assignments(DAB_CONFIG)
    params = _load_yaml(SGCLI_WORKLOAD)["parameters"]

    assert params["catalog"] == config["CATALOG"]
    assert params["schema"] == config["SCHEMA"]
    assert params["backbone_name"] == config["BACKBONE"]
    assert params["backbone_revision"] == config["BACKBONE_REVISION"]
    assert params["volume_path"] == f"/Volumes/{config['CATALOG']}/{config['SCHEMA']}/{config['DENTEX_VOLUME']}"
    assert params["cache_dir"] == f"/Volumes/{config['CATALOG']}/{config['SCHEMA']}/{config['MODEL_CACHE_VOLUME']}"
    assert params["epochs"] == config["TRAIN_EPOCHS"]
    assert params["lr"] == config["TRAIN_LR"]
    assert params["batch_size"] == config["TRAIN_BATCH_SIZE"]
    assert params["use_lora"] == config["TRAIN_USE_LORA"]
    assert params["lora_rank"] == config["TRAIN_LORA_RANK"]
    assert params["lora_alpha"] == config["TRAIN_LORA_ALPHA"]
    model_name = config["_DETECTOR_NAMES_BY_BACKBONE"][config["BACKBONE"]]["model_short"]
    assert params["model_name"] == model_name
