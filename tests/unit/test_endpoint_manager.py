import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_mlflow(monkeypatch):
    fake_client = MagicMock()
    fake_mv = MagicMock(version="5")
    fake_client.get_model_version_by_alias.return_value = fake_mv

    class FakeMlflowModule:
        def set_registry_uri(self, uri):
            pass

    class FakeTracking:
        class MlflowClient:
            def __init__(self, registry_uri=None):
                self._inner = fake_client

            def get_model_version_by_alias(self, name, alias):
                return fake_client.get_model_version_by_alias(name=name, alias=alias)

            def set_registered_model_alias(self, name, alias, version):
                fake_client.set_registered_model_alias(name=name, alias=alias, version=version)

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflowModule())
    monkeypatch.setitem(sys.modules, "mlflow.tracking", FakeTracking)
    return fake_client


def test_resolve_alias_to_version(mock_mlflow):
    from dais26_dentex.serve.endpoint_manager import resolve_alias_to_version

    v = resolve_alias_to_version("cat", "sch", "mdl", "candidate")
    assert v == "5"
    mock_mlflow.get_model_version_by_alias.assert_called_once_with(name="cat.sch.mdl", alias="candidate")


def test_capture_previous_champion_none(monkeypatch):
    """When no @champion exists, capture returns None."""

    class FakeMlflowModule:
        def set_registry_uri(self, uri):
            pass

    class FakeTracking:
        class MlflowClient:
            def __init__(self, registry_uri=None):
                pass

            def get_model_version_by_alias(self, name, alias):
                raise Exception("Alias does not exist")

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflowModule())
    monkeypatch.setitem(sys.modules, "mlflow.tracking", FakeTracking)
    from dais26_dentex.serve.endpoint_manager import capture_previous_champion

    assert capture_previous_champion("cat", "sch", "mdl") is None


def test_smoke_test_no_predictions(monkeypatch):
    class FakeSdkModule:
        class WorkspaceClient:
            def __init__(self):
                self.serving_endpoints = MagicMock()
                self.serving_endpoints.query.return_value = MagicMock(predictions=None)

    monkeypatch.setitem(sys.modules, "databricks", MagicMock())
    monkeypatch.setitem(sys.modules, "databricks.sdk", FakeSdkModule)
    from dais26_dentex.serve.endpoint_manager import smoke_test_endpoint

    ok, _err = smoke_test_endpoint("ep", b"x")
    assert ok is False
