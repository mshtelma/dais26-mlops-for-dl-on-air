"""`Trainer._log_metrics_to_logged_model` links the best-epoch val metrics to the
MLflow 3 LoggedModel (model_id) returned by log_model.

The method is exercised in isolation (``Trainer.__new__`` — no GPU / distributed
setup) because it is a pure side-effect helper over ``self._best_*`` state.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from dais26_dentex.train.trainer import Trainer


def _trainer_with_best(metrics: dict[str, float], best: float) -> Trainer:
    t = Trainer.__new__(Trainer)  # bypass heavy __init__ (device / DDP / model build)
    t._best_val_metrics = dict(metrics)
    t._best_metric = best
    return t


def test_links_best_metrics_to_logged_model(monkeypatch):
    log_metrics = MagicMock()
    monkeypatch.setattr("mlflow.log_metrics", log_metrics)

    t = _trainer_with_best({"val/mAP_50": 0.51, "val/mAP_75": 0.30, "val/mAP_50_95": 0.22}, best=0.51)
    t._log_metrics_to_logged_model(SimpleNamespace(model_id="m-123"))

    assert log_metrics.call_count == 1
    args, kwargs = log_metrics.call_args
    assert kwargs["model_id"] == "m-123"
    logged = args[0]
    # All best-epoch val metrics plus the sweep's primary metric land on the model.
    assert logged["val/mAP_50"] == 0.51
    assert logged["val/mAP_75"] == 0.30
    assert logged["val/mAP_50_95"] == 0.22
    assert logged["val/best_mAP_50"] == 0.51


def test_noop_when_model_info_has_no_model_id(monkeypatch):
    log_metrics = MagicMock()
    monkeypatch.setattr("mlflow.log_metrics", log_metrics)

    t = _trainer_with_best({"val/mAP_50": 0.4}, best=0.4)
    # Stubbed log_model returns None (see test_train_smoke); also covers a ModelInfo
    # without a model_id attribute.
    t._log_metrics_to_logged_model(None)
    t._log_metrics_to_logged_model(SimpleNamespace())

    assert log_metrics.call_count == 0


def test_noop_when_no_best_metric_captured(monkeypatch):
    log_metrics = MagicMock()
    monkeypatch.setattr("mlflow.log_metrics", log_metrics)

    # Sentinel -1.0 best (val never ran) and no metric dict -> nothing to link.
    t = _trainer_with_best({}, best=-1.0)
    t._log_metrics_to_logged_model(SimpleNamespace(model_id="m-1"))

    assert log_metrics.call_count == 0


def test_swallows_older_client_without_model_id_kwarg(monkeypatch):
    def _raise_typeerror(*a, **kw):
        raise TypeError("log_metrics() got an unexpected keyword argument 'model_id'")

    monkeypatch.setattr("mlflow.log_metrics", _raise_typeerror)

    t = _trainer_with_best({"val/mAP_50": 0.4}, best=0.4)
    # Must not raise — the run-level metrics already cover the legacy read path.
    t._log_metrics_to_logged_model(SimpleNamespace(model_id="m-9"))
