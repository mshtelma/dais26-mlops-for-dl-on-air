"""Schema/structural tests for sgcli/*.yaml + cross-file consistency with pyproject.toml."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKLOAD = REPO / "sgcli" / "workload_train_detector.yaml"
REQS = REPO / "sgcli" / "requirements.yaml"
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
    assert d["compute"]["gpu_type"] in {"h100", "a10"}
    assert isinstance(d["compute"]["gpus"], int) and d["compute"]["gpus"] >= 1


def test_workload_code_source_repo_path_is_parent():
    """repo_path is resolved relative to the YAML location — must be `..` so it points
    at the repo root from sgcli/."""
    d = _load(WORKLOAD)
    assert d["code_source"]["snapshot"]["repo_path"] == ".."


def test_workload_command_does_not_use_no_deps():
    d = _load(WORKLOAD)
    assert "--no-deps" not in d["command"], (
        "pip install -e . --no-deps drops transitives not in requirements.yaml — fragile"
    )


def test_workload_command_uses_torchrun_module_entrypoint():
    d = _load(WORKLOAD)
    assert "torchrun" in d["command"]
    assert "-m dais26_dentex.train.cli" in d["command"]


def test_workload_parameters_use_internal_backbone_literal():
    """The parameters block must use the internal Literal name, not the HF id —
    train_detector's BackboneName Literal does not accept the HF id."""
    d = _load(WORKLOAD)
    assert d["parameters"]["backbone_name"] in {
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
