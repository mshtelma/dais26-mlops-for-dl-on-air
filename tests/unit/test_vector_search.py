"""Tests for serve.vector_search — the VS endpoint/index ensure logic shared by
notebook 04b (and formerly duplicated inline in 04). Driven by a plain fake
WorkspaceClient (same style as test_sweep_runner's fake MLflow client)."""

from __future__ import annotations

import pytest

from dais26_dentex.serve import vector_search as vs


class _Status:
    def __init__(self, ready: bool, indexed: int) -> None:
        self.ready = ready
        self.indexed_row_count = indexed
        self.message = "ok"


class _Col:
    def __init__(self, dim: int) -> None:
        self.embedding_dimension = dim


class _Spec:
    def __init__(self, dim: int) -> None:
        self.embedding_vector_columns = [_Col(dim)]


class _Index:
    def __init__(self, dim: int, status: _Status) -> None:
        self.delta_sync_index_spec = _Spec(dim)
        self.status = status


class _Endpoints:
    def __init__(self, exists: bool = False) -> None:
        self.created: list[str] = []
        self._exists = exists

    def create_endpoint_and_wait(self, name, endpoint_type):
        if self._exists:
            raise RuntimeError("RESOURCE_ALREADY_EXISTS: endpoint already exists")
        self.created.append(name)


class _Indexes:
    """Scripted fake of `w.vector_search_indexes`.

    `exists_dim=None` => create succeeds; an int => the first create raises
    "already exists" and `get_index` reports that dim (until a delete).
    """

    def __init__(self, *, exists_dim: int | None = None, status: _Status | None = None) -> None:
        self.exists_dim = exists_dim
        self._status = status or _Status(True, 0)
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.synced: list[str] = []
        self._deleted = False

    def create_index(self, **kwargs):
        self.created.append(kwargs)
        if self.exists_dim is not None and not self._deleted:
            raise RuntimeError("RESOURCE_ALREADY_EXISTS: index already exists")

    def get_index(self, index_name):
        if self._deleted:
            raise RuntimeError("index does not exist")
        return _Index(self.exists_dim if self.exists_dim is not None else 0, self._status)

    def delete_index(self, index_name):
        self.deleted.append(index_name)
        self._deleted = True

    def sync_index(self, index_name):
        self.synced.append(index_name)


class _W:
    def __init__(self, endpoints: _Endpoints, indexes: _Indexes) -> None:
        self.vector_search_endpoints = endpoints
        self.vector_search_indexes = indexes


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vs.time, "sleep", lambda *_a, **_k: None)


def test_ensure_index_creates_when_absent() -> None:
    idx = _Indexes(exists_dim=None)
    action = vs.ensure_index(
        _W(_Endpoints(), idx), endpoint_name="e", index_name="i", source_table="t", embedding_dim=2304
    )
    assert action == "created"
    assert len(idx.created) == 1 and not idx.deleted and not idx.synced


def test_ensure_index_recreates_on_dim_mismatch() -> None:
    idx = _Indexes(exists_dim=1024)  # live index is 1024-dim; new champion is 2304
    action = vs.ensure_index(
        _W(_Endpoints(), idx), endpoint_name="e", index_name="i", source_table="t", embedding_dim=2304
    )
    assert action == "recreated"
    assert idx.deleted == ["i"]
    assert len(idx.created) == 2  # initial (already-exists) + recreate after delete


def test_ensure_index_syncs_on_matching_dim() -> None:
    idx = _Indexes(exists_dim=2304)
    action = vs.ensure_index(
        _W(_Endpoints(), idx), endpoint_name="e", index_name="i", source_table="t", embedding_dim=2304
    )
    assert action == "synced"
    assert idx.synced == ["i"] and not idx.deleted


def test_ensure_endpoint_idempotent_when_exists() -> None:
    eps = _Endpoints(exists=True)
    vs.ensure_endpoint(_W(eps, _Indexes()), "e")  # must not raise
    assert eps.created == []


def test_wait_until_online_returns_when_ready() -> None:
    idx = _Indexes(exists_dim=2304, status=_Status(True, 50))
    n = vs.wait_until_online(_W(_Endpoints(), idx), index_name="i", expected_rows=50, timeout_s=5, poll_s=0)
    assert n == 50


def test_wait_until_online_times_out() -> None:
    idx = _Indexes(exists_dim=2304, status=_Status(False, 0))
    with pytest.raises(RuntimeError, match="did not reach ONLINE"):
        vs.wait_until_online(_W(_Endpoints(), idx), index_name="i", expected_rows=50, timeout_s=0.02, poll_s=0)


def test_ensure_vector_search_index_end_to_end_create() -> None:
    idx = _Indexes(exists_dim=None, status=_Status(True, 50))
    action = vs.ensure_vector_search_index(
        _W(_Endpoints(), idx), endpoint_name="e", index_name="i", source_table="t",
        embedding_dim=2304, wait_online=True, expected_rows=50, timeout_s=5, poll_s=0,
    )
    assert action == "created"
    assert len(idx.created) == 1
