# Databricks notebook source
# MAGIC %md
# MAGIC # 04b — Create Vector Search endpoint + index (self-contained)
# MAGIC
# MAGIC Creates the Vector Search endpoint (`VS_ENDPOINT_NAME`) and a DELTA_SYNC
# MAGIC index (`VS_INDEX_NAME`) over the precomputed embeddings table, then triggers
# MAGIC a sync and waits for the index to come ONLINE.
# MAGIC
# MAGIC This is the body of the `create_vector_search` branch in
# MAGIC `04_deploy_serving.py`, lifted into its own notebook so it runs
# MAGIC UNCONDITIONALLY (no dependence on the hardcoded `DEPLOY_ACTION`). The
# MAGIC embedding dimension is derived from the source table rather than hardcoded,
# MAGIC so it stays correct regardless of the configured backbone
# MAGIC (C-RADIOv4-SO400M summary = 2304, DINOv3-ViTL16 = 1024).

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    EndpointType,
    PipelineType,
    VectorIndexType,
)

w = WorkspaceClient()

source_table = TRAIN_EMBEDDINGS_TABLE
print(f"VS endpoint : {VS_ENDPOINT_NAME}")
print(f"VS index    : {VS_INDEX_NAME}")
print(f"Source table: {source_table}")

# COMMAND ----------
# ---- Sanity-check the source table + derive the embedding dimension ----
row_count = spark.table(source_table).count()
if row_count == 0:
    raise RuntimeError(
        f"Source table {source_table} is empty — run the precompute_embeddings "
        "job first so the Delta Sync index has rows to index."
    )

embedding_dim = int(
    spark.sql(f"SELECT size(embedding) AS d FROM {source_table} LIMIT 1").collect()[0]["d"]
)
print(f"Source rows : {row_count}")
print(f"Embedding dim (derived from table): {embedding_dim}")

# COMMAND ----------
# ---- Create the Vector Search endpoint (idempotent) ----
try:
    w.vector_search_endpoints.create_endpoint_and_wait(
        name=VS_ENDPOINT_NAME, endpoint_type=EndpointType.STANDARD,
    )
    print(f"Created VS endpoint: {VS_ENDPOINT_NAME}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"VS endpoint {VS_ENDPOINT_NAME} already exists")
    else:
        raise

# COMMAND ----------
# ---- Create the DELTA_SYNC index (idempotent + dimension-aware) ----
# A VS index's embedding_dimension is IMMUTABLE. When the champion's backbone
# changes between promotions the embedding dim changes too (C-RADIOv4-SO400M summary=2304,
# DINOv3-ViTL16=1024, DINOv2-base=768), so an index built for the previous champion
# no longer matches the freshly precomputed vectors. If the existing index's dim
# differs from the source table's, drop + recreate it; otherwise just sync. (Just
# syncing would push dim-mismatched vectors into a stale index and the promotion
# fails — the manual workaround in docs/RUNBOOK.md Step 3, now automated.)
def _create_index(dim: int) -> None:
    w.vector_search_indexes.create_index(
        name=VS_INDEX_NAME,
        endpoint_name=VS_ENDPOINT_NAME,
        primary_key="image_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=source_table,
            # Self-managed (precomputed) embeddings: the column already holds the
            # ARRAY<FLOAT> vectors, so use embedding_vector_columns (not source).
            embedding_vector_columns=[
                EmbeddingVectorColumn(name="embedding", embedding_dimension=dim),
            ],
            pipeline_type=PipelineType.TRIGGERED,
            columns_to_sync=["image_id", "diagnosis", "split"],
        ),
    )


def _existing_index_dim() -> int | None:
    """Embedding dimension baked into the live index, or None if undiscoverable."""
    idx = w.vector_search_indexes.get_index(index_name=VS_INDEX_NAME)
    spec = getattr(idx, "delta_sync_index_spec", None)
    cols = getattr(spec, "embedding_vector_columns", None) or []
    for col in cols:
        d = getattr(col, "embedding_dimension", None)
        if d:
            return int(d)
    return None


print(f"Creating index {VS_INDEX_NAME} from {source_table} (dim={embedding_dim})")
try:
    _create_index(embedding_dim)
    print(f"Created VS index: {VS_INDEX_NAME}")
except Exception as e:
    if "already exists" not in str(e).lower():
        raise
    existing_dim = _existing_index_dim()
    if existing_dim is not None and existing_dim != embedding_dim:
        # Backbone changed since the index was built: the dim is immutable, so the
        # only fix is drop + recreate at the new dimension.
        print(
            f"VS index {VS_INDEX_NAME} exists at dim={existing_dim} but the new "
            f"champion's embeddings are dim={embedding_dim}; dropping + recreating."
        )
        w.vector_search_indexes.delete_index(index_name=VS_INDEX_NAME)
        # Wait for the delete to settle so the recreate doesn't race "already exists".
        del_deadline = time.time() + 300
        while time.time() < del_deadline:
            try:
                w.vector_search_indexes.get_index(index_name=VS_INDEX_NAME)
                time.sleep(5)
            except Exception:
                break
        _create_index(embedding_dim)
        print(f"Recreated VS index {VS_INDEX_NAME} at dim={embedding_dim}")
    else:
        print(
            f"VS index {VS_INDEX_NAME} already exists at matching dim={existing_dim}; "
            "triggering sync"
        )
        w.vector_search_indexes.sync_index(index_name=VS_INDEX_NAME)

# COMMAND ----------
# ---- Wait for the index to come ONLINE and finish its first sync ----
# DELTA_SYNC indexes start an initial sync on creation; poll until the index is
# ready and the indexed row count reflects the source table.
deadline = time.time() + 1800  # 30 min
last_seen = None
synced_kicked = False
while time.time() < deadline:
    idx = w.vector_search_indexes.get_index(index_name=VS_INDEX_NAME)
    status = idx.status
    indexed = getattr(status, "indexed_row_count", None) or 0
    ready = bool(getattr(status, "ready", False))
    msg = getattr(status, "message", None)
    seen = (ready, indexed)
    if seen != last_seen:
        print(f"index ready={ready} indexed_rows={indexed} msg={msg}")
        last_seen = seen
    if ready and indexed >= row_count:
        print(f"Index ONLINE and fully synced: {indexed}/{row_count} rows")
        break
    # If the endpoint is ready but no rows have been indexed after a grace
    # period, kick a sync once (covers the case where the initial TRIGGERED
    # sync didn't auto-start).
    if ready and indexed == 0 and not synced_kicked:
        try:
            w.vector_search_indexes.sync_index(index_name=VS_INDEX_NAME)
            print("Triggered index sync")
        except Exception as e:
            print(f"sync trigger note: {e}")
        synced_kicked = True
    time.sleep(20)
else:
    raise RuntimeError(
        f"Index {VS_INDEX_NAME} did not reach ONLINE/fully-synced within timeout; "
        f"last_seen={last_seen}"
    )

# COMMAND ----------
# ---- Smoke-test query: confirm the index returns neighbors ----
sample = (
    spark.table(source_table)
    .select("image_id", "embedding", "diagnosis")
    .limit(1)
    .toPandas()
)
query_vec = list(map(float, sample.iloc[0]["embedding"]))
res = w.vector_search_indexes.query_index(
    index_name=VS_INDEX_NAME,
    columns=["image_id", "diagnosis", "split"],
    query_vector=query_vec,
    num_results=5,
)
rows = res.result.data_array if hasattr(res, "result") else getattr(res, "data_array", [])
print(f"Sample query returned {len(rows)} neighbors:")
for r in rows:
    print("  ", r)
if not rows:
    raise RuntimeError("Sample similarity query returned no neighbors")
print("VS index create + verify: OK")
