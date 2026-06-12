"""Vector Search endpoint + DELTA_SYNC index management.

Single source of the create / sync / dimension-aware-recreate logic that
notebook `04b_create_vector_search.py` carried inline (and `04_deploy_serving.py`
had a now-removed copy of). A VS index's embedding dimension is **immutable**, so
when the champion backbone changes between promotions (C-RADIOv4-SO400M summary
= 2304, DINOv3-ViTL16 = 1024, DINOv2-base = 768) the index must be dropped +
recreated, not synced — otherwise dim-mismatched vectors get pushed into a stale
index and similarity search breaks.

Spark stays in the caller: the notebook derives `embedding_dim` / `expected_rows`
from the source Delta table and passes them in, so this module depends only on
the Databricks SDK and is unit-testable with a fake `WorkspaceClient`. SDK
service types are imported inside the functions so importing this module never
requires them at module load.
"""

from __future__ import annotations

import time
from typing import Any

DEFAULT_COLUMNS_TO_SYNC: tuple[str, ...] = ("image_id", "diagnosis", "split")


def _existing_index_dim(w: Any, index_name: str) -> int | None:
    """Embedding dimension baked into the live index, or None if undiscoverable."""
    idx = w.vector_search_indexes.get_index(index_name=index_name)
    spec = getattr(idx, "delta_sync_index_spec", None)
    cols = getattr(spec, "embedding_vector_columns", None) or []
    for col in cols:
        d = getattr(col, "embedding_dimension", None)
        if d:
            return int(d)
    return None


def ensure_endpoint(w: Any, endpoint_name: str) -> None:
    """Create the Vector Search endpoint if absent (idempotent)."""
    from databricks.sdk.service.vectorsearch import EndpointType

    try:
        w.vector_search_endpoints.create_endpoint_and_wait(
            name=endpoint_name, endpoint_type=EndpointType.STANDARD
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise


def ensure_index(
    w: Any,
    *,
    endpoint_name: str,
    index_name: str,
    source_table: str,
    embedding_dim: int,
    primary_key: str = "image_id",
    columns_to_sync: tuple[str, ...] = DEFAULT_COLUMNS_TO_SYNC,
    embedding_column: str = "embedding",
) -> str:
    """Create the DELTA_SYNC index; on conflict drop+recreate (dim changed) or sync.

    Self-managed (precomputed) embeddings: the column already holds the
    ARRAY<FLOAT> vectors, so we use `embedding_vector_columns` (not a source
    column to embed). Returns the action taken: "created" | "recreated" | "synced".
    """
    from databricks.sdk.service.vectorsearch import (
        DeltaSyncVectorIndexSpecRequest,
        EmbeddingVectorColumn,
        PipelineType,
        VectorIndexType,
    )

    def _create() -> None:
        w.vector_search_indexes.create_index(
            name=index_name,
            endpoint_name=endpoint_name,
            primary_key=primary_key,
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=source_table,
                embedding_vector_columns=[
                    EmbeddingVectorColumn(name=embedding_column, embedding_dimension=embedding_dim),
                ],
                pipeline_type=PipelineType.TRIGGERED,
                columns_to_sync=list(columns_to_sync),
            ),
        )

    try:
        _create()
        return "created"
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise

    existing = _existing_index_dim(w, index_name)
    if existing is not None and existing != embedding_dim:
        # Backbone changed since the index was built: the dim is immutable, so the
        # only fix is drop + recreate at the new dimension.
        w.vector_search_indexes.delete_index(index_name=index_name)
        # Wait for the delete to settle so the recreate doesn't race "already exists".
        del_deadline = time.time() + 300
        while time.time() < del_deadline:
            try:
                w.vector_search_indexes.get_index(index_name=index_name)
                time.sleep(5)
            except Exception:
                break
        _create()
        return "recreated"

    w.vector_search_indexes.sync_index(index_name=index_name)
    return "synced"


def wait_until_online(
    w: Any,
    *,
    index_name: str,
    expected_rows: int,
    timeout_s: float = 1800,
    poll_s: float = 20,
) -> int:
    """Poll until the index is ready and `indexed_row_count >= expected_rows`.

    DELTA_SYNC indexes start an initial sync on creation; if the endpoint is
    ready but nothing has indexed after a grace period, kick a sync once (covers
    the case where the initial TRIGGERED sync didn't auto-start). Returns the
    final indexed row count; raises `RuntimeError` on timeout.
    """
    deadline = time.time() + timeout_s
    last_seen = None
    synced_kicked = False
    while time.time() < deadline:
        idx = w.vector_search_indexes.get_index(index_name=index_name)
        status = idx.status
        indexed = getattr(status, "indexed_row_count", None) or 0
        ready = bool(getattr(status, "ready", False))
        seen = (ready, indexed)
        if seen != last_seen:
            print(f"index ready={ready} indexed_rows={indexed} msg={getattr(status, 'message', None)}")
            last_seen = seen
        if ready and indexed >= expected_rows:
            return indexed
        if ready and indexed == 0 and not synced_kicked:
            try:
                w.vector_search_indexes.sync_index(index_name=index_name)
                print("Triggered index sync")
            except Exception as e:
                print(f"sync trigger note: {e}")
            synced_kicked = True
        time.sleep(poll_s)
    raise RuntimeError(
        f"Index {index_name} did not reach ONLINE/fully-synced within {timeout_s:.0f}s; "
        f"last_seen={last_seen}"
    )


def ensure_vector_search_index(
    w: Any,
    *,
    endpoint_name: str,
    index_name: str,
    source_table: str,
    embedding_dim: int,
    primary_key: str = "image_id",
    columns_to_sync: tuple[str, ...] = DEFAULT_COLUMNS_TO_SYNC,
    embedding_column: str = "embedding",
    wait_online: bool = False,
    expected_rows: int | None = None,
    timeout_s: float = 1800,
    poll_s: float = 20,
) -> str:
    """Ensure the endpoint + DELTA_SYNC index exist (dimension-aware) and optionally
    wait for the index to come ONLINE. Returns the index action ("created" /
    "recreated" / "synced"). `expected_rows` is required when `wait_online=True`."""
    ensure_endpoint(w, endpoint_name)
    action = ensure_index(
        w,
        endpoint_name=endpoint_name,
        index_name=index_name,
        source_table=source_table,
        embedding_dim=embedding_dim,
        primary_key=primary_key,
        columns_to_sync=columns_to_sync,
        embedding_column=embedding_column,
    )
    if wait_online:
        wait_until_online(
            w,
            index_name=index_name,
            expected_rows=expected_rows or 0,
            timeout_s=timeout_s,
            poll_s=poll_s,
        )
    return action


__all__ = [
    "DEFAULT_COLUMNS_TO_SYNC",
    "ensure_endpoint",
    "ensure_index",
    "ensure_vector_search_index",
    "wait_until_online",
]
