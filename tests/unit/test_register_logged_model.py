"""Unit tests for ``MlflowReporter.register_logged_model``.

The whole LoggedModel approach (sweep gate reading metrics off the LoggedModel,
cross-schema champion ``copy_model_version``) hinges on registered versions
carrying a non-empty ``model_id``. That only happens when we register FROM the
LoggedModel URI (``models:/<model_id>``) rather than passing
``registered_model_name=`` to ``log_model`` (which strands the version with
``model_id=''``). These tests pin that behavior.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dais26_dentex.platform.mlflow_io import AliasingError, MlflowReporter


def _reporter() -> MlflowReporter:
    # Avoid set_experiment / network: construct without an experiment.
    return MlflowReporter(registry_uri="databricks-uc://UNIT")


def test_registers_from_logged_model_uri_and_sets_alias(monkeypatch):
    captured: dict[str, object] = {}

    def fake_register_model(uri, name):
        captured["uri"] = uri
        captured["name"] = name
        return SimpleNamespace(version="7")

    fake_client = MagicMock()
    monkeypatch.setattr("dais26_dentex.platform.mlflow_io.mlflow.register_model", fake_register_model)
    monkeypatch.setattr(
        "dais26_dentex.platform.mlflow_io.MlflowClient",
        lambda *a, **k: fake_client,
    )

    info = SimpleNamespace(model_id="abc123", model_uri="models:/abc123")
    version = _reporter().register_logged_model(info, "cat.sch.model", alias="candidate")

    assert version == "7"
    # Registered from the LoggedModel URI (preserves model_id), not a run artifact.
    assert captured["uri"] == "models:/abc123"
    assert captured["name"] == "cat.sch.model"
    fake_client.set_registered_model_alias.assert_called_once_with(
        name="cat.sch.model", alias="candidate", version="7"
    )


def test_builds_uri_from_model_id_when_model_uri_missing(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "dais26_dentex.platform.mlflow_io.mlflow.register_model",
        lambda uri, name: captured.update(uri=uri) or SimpleNamespace(version="2"),
    )
    monkeypatch.setattr("dais26_dentex.platform.mlflow_io.MlflowClient", lambda *a, **k: MagicMock())

    info = SimpleNamespace(model_id="zzz")  # no model_uri attribute
    _reporter().register_logged_model(info, "cat.sch.model", alias=None)
    assert captured["uri"] == "models:/zzz"


def test_no_alias_skips_alias_call(monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr(
        "dais26_dentex.platform.mlflow_io.mlflow.register_model",
        lambda uri, name: SimpleNamespace(version="1"),
    )
    monkeypatch.setattr("dais26_dentex.platform.mlflow_io.MlflowClient", lambda *a, **k: fake_client)

    _reporter().register_logged_model(SimpleNamespace(model_id="m", model_uri="models:/m"), "c.s.m", alias=None)
    fake_client.set_registered_model_alias.assert_not_called()


def test_missing_model_id_raises(monkeypatch):
    # A stubbed/classic log_model that produced no LoggedModel must not silently
    # register a model_id-less version.
    monkeypatch.setattr(
        "dais26_dentex.platform.mlflow_io.mlflow.register_model",
        lambda uri, name: pytest.fail("register_model should not be called without a model_id"),
    )
    with pytest.raises(AliasingError, match="no model_id"):
        _reporter().register_logged_model(SimpleNamespace(model_id=""), "c.s.m")


def test_alias_failure_raises_aliasing_error(monkeypatch):
    fake_client = MagicMock()
    fake_client.set_registered_model_alias.side_effect = RuntimeError("boom")
    monkeypatch.setattr(
        "dais26_dentex.platform.mlflow_io.mlflow.register_model",
        lambda uri, name: SimpleNamespace(version="9"),
    )
    monkeypatch.setattr("dais26_dentex.platform.mlflow_io.MlflowClient", lambda *a, **k: fake_client)

    with pytest.raises(AliasingError, match="Failed to set @candidate"):
        _reporter().register_logged_model(
            SimpleNamespace(model_id="m", model_uri="models:/m"), "c.s.m", alias="candidate"
        )
