"""Schema/structural tests for air/*.yaml + cross-file consistency with pyproject.toml."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKLOAD = REPO / "air" / "workload_train_detector.yaml"
REQS = REPO / "air" / "requirements.yaml"
PYPROJECT = REPO / "pyproject.toml"


def _load(p: Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


def test_workload_yaml_parses():
    d = _load(WORKLOAD)
    assert isinstance(d, dict)


def test_workload_required_keys():
    d = _load(WORKLOAD)
    required = {"experiment_name", "environment", "compute", "command"}
    assert required <= set(d.keys()), f"missing: {required - set(d.keys())}"


def test_workload_compute_h100_or_a10():
    d = _load(WORKLOAD)
    assert d["compute"]["accelerator_type"] in {"GPU_8xH100", "GPU_1xH100", "GPU_1xA10"}
    assert isinstance(d["compute"]["num_accelerators"], int) and d["compute"]["num_accelerators"] >= 1


def test_workload_code_source_root_path_is_repo_root():
    """air resolves snapshot.root_path relative to the WORKLOAD YAML file (which
    lives in air/), so it must be `..` to package the repo root — `.` would
    snapshot only the air/ directory (observed via `air run --dry-run`)."""
    d = _load(WORKLOAD)
    assert d["code_source"]["snapshot"]["root_path"] == ".."


def test_workload_command_does_not_use_no_deps():
    d = _load(WORKLOAD)
    assert "--no-deps" not in d["command"], (
        "pip install -e . --no-deps drops transitives not in requirements.yaml — fragile"
    )


def test_workload_command_uses_torchrun_module_entrypoint():
    d = _load(WORKLOAD)
    assert "torchrun" in d["command"]
    assert "-m dais26_dentex.train.cli" in d["command"]


def test_workload_parameters_use_internal_recipe_literal():
    """The parameters block names a recipe by the internal backbone literal,
    not the HF id — `config.recipes.RECIPES` is keyed by the internal names."""
    d = _load(WORKLOAD)
    assert d["parameters"]["recipe"] in {
        "cradio_v4_so400m",
        "dinov3_vitl16",
        "dinov2_base",
    }


def test_workload_environment_has_dependencies_pointer():
    d = _load(WORKLOAD)
    assert d["environment"]["dependencies"] == "requirements.yaml"


def test_requirements_yaml_parses():
    d = _load(REQS)
    assert isinstance(d, dict)


def test_requirements_uses_base_env_v4():
    d = _load(REQS)
    assert d["version"] == "4"


def test_requirements_lists_dependencies():
    d = _load(REQS)
    deps = d["dependencies"]
    assert isinstance(deps, list) and deps


def test_mlflow_pin_matches_pyproject():
    """requirements.yaml `mlflow>=X` must not conflict with pyproject's pin.

    Bumped to MLflow 3 for deployment jobs + logging metrics to model versions
    (LoggedModel); both files now pin `mlflow>=3.1`.
    """
    req = _load(REQS)
    mlflow_specs = [s for s in req["dependencies"] if isinstance(s, str) and "mlflow" in s]
    assert mlflow_specs, "no mlflow dependency in requirements.yaml"
    pyproject_text = PYPROJECT.read_text()
    # Both should require MLflow 3 (no major-version split between the two files).
    for spec in mlflow_specs:
        assert ">=3.1" in spec, f"mlflow pin {spec!r} drifts from pyproject (>=3.1)"
    assert "mlflow>=3.1" in pyproject_text


def test_requirements_includes_pyyaml():
    d = _load(REQS)
    assert any(re.match(r"(?i)pyyaml", s) for s in d["dependencies"])
