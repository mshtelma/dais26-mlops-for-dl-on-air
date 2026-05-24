"""Verify rank-0 gating: simulate WORLD_SIZE=2 and assert mlflow.start_run
is called only on RANK=0. We mock all heavy dependencies to keep the test fast.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.train import train_detector as td_mod


@pytest.fixture
def mock_heavy_deps(monkeypatch):
    """Mock all expensive imports inside train_detector so the function body runs cheaply."""

    # Backbone returns a tiny module + a fake BackboneInfo
    fake_bb = MagicMock()
    fake_info = MagicMock()
    fake_info.model_name = "test"
    fake_info.revision = ""
    fake_info.summary_dim = 64
    fake_info.spatial_dim = 64
    fake_info.patch_size = 16

    monkeypatch.setattr(
        "src.models.backbones.load_backbone",
        lambda **kw: (fake_bb, fake_info),
    )

    import torch.nn as nn

    class FakeDetectionModel(nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def to(self, *a, **kw): return self
        def forward_train(self, x): pass

    monkeypatch.setattr("src.models.detection_head.DetectionModel", FakeDetectionModel)
    monkeypatch.setattr("src.models.detection_head.DEFAULT_ANCHOR_SCALES", [1])
    monkeypatch.setattr("src.models.detection_head.DEFAULT_ASPECT_RATIOS", [1.0])
    monkeypatch.setattr("src.models.peft.apply_lora", lambda m, **kw: m)


def test_rank0_starts_mlflow_run(mock_heavy_deps, monkeypatch):
    """Rank-0 should call mlflow.start_run exactly once."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")  # single-process to avoid NCCL setup

    mock_start = MagicMock()
    mock_run_ctx = MagicMock()
    mock_run_ctx.info.run_id = "rank0_run"
    mock_start.return_value.__enter__ = MagicMock(return_value=mock_run_ctx)
    mock_start.return_value.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("mlflow.start_run", mock_start)
    monkeypatch.setattr("mlflow.set_registry_uri", MagicMock())
    monkeypatch.setattr("mlflow.set_experiment", MagicMock())
    monkeypatch.setattr("mlflow.log_params", MagicMock())
    monkeypatch.setattr("mlflow.log_metric", MagicMock())
    monkeypatch.setattr("mlflow.log_param", MagicMock())
    monkeypatch.setattr("mlflow.pyfunc.log_model", MagicMock())

    run_id = td_mod.train_detector(
        catalog="c", schema="s",
        volume_path=None,        # skip dataloaders for unit test
        epochs=0,
        register_model=False,
        set_candidate_alias=False,
    )

    assert run_id == "rank0_run"
    assert mock_start.call_count == 1, "rank-0 should call mlflow.start_run exactly once"


def _stub_distributed(monkeypatch):
    """Stub out torch.distributed + DDP so train_detector's distributed branch
    doesn't need a real process group."""
    import torch.distributed as dist
    monkeypatch.setattr(dist, "init_process_group", MagicMock())
    monkeypatch.setattr(dist, "is_initialized", MagicMock(return_value=False))
    monkeypatch.setattr(dist, "barrier", MagicMock())
    monkeypatch.setattr(dist, "destroy_process_group", MagicMock())
    # Mock DDP wrap to be a pass-through identity function
    import torch.nn as nn
    monkeypatch.setattr(
        nn.parallel, "DistributedDataParallel",
        lambda m, **kw: m,
    )


def test_non_rank0_does_not_start_mlflow_run(mock_heavy_deps, monkeypatch):
    """Non-rank-0 must NOT call mlflow.start_run (would create orphan runs)."""
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29502")
    _stub_distributed(monkeypatch)

    mock_start = MagicMock()
    monkeypatch.setattr("mlflow.start_run", mock_start)
    monkeypatch.setattr("mlflow.set_registry_uri", MagicMock())
    monkeypatch.setattr("mlflow.set_experiment", MagicMock())
    monkeypatch.setattr("mlflow.log_params", MagicMock())
    monkeypatch.setattr("mlflow.log_metric", MagicMock())
    monkeypatch.setattr("mlflow.log_param", MagicMock())
    monkeypatch.setattr("mlflow.pyfunc.log_model", MagicMock())

    run_id = td_mod.train_detector(
        catalog="c", schema="s",
        volume_path=None,
        epochs=0,
        register_model=False,
        set_candidate_alias=False,
    )

    assert run_id is None, "non-rank-0 must return None"
    assert mock_start.call_count == 0, "non-rank-0 must NOT call mlflow.start_run"


def test_set_registry_uri_only_on_rank0(mock_heavy_deps, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    _stub_distributed(monkeypatch)

    set_uri = MagicMock()
    set_exp = MagicMock()
    monkeypatch.setattr("mlflow.set_registry_uri", set_uri)
    monkeypatch.setattr("mlflow.set_experiment", set_exp)
    monkeypatch.setattr("mlflow.start_run", MagicMock())
    monkeypatch.setattr("mlflow.log_params", MagicMock())
    monkeypatch.setattr("mlflow.log_metric", MagicMock())
    monkeypatch.setattr("mlflow.log_param", MagicMock())
    monkeypatch.setattr("mlflow.pyfunc.log_model", MagicMock())

    td_mod.train_detector(
        catalog="c", schema="s", volume_path=None, epochs=0,
        experiment_name="/Shared/exp",
        register_model=False, set_candidate_alias=False,
    )

    # On non-rank-0, these MLflow setup calls must not fire
    assert set_uri.call_count == 0
    assert set_exp.call_count == 0
