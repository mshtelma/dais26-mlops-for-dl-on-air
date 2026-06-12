# Databricks notebook source
# MAGIC %md
# MAGIC # 04b — Create Vector Search endpoint + index (self-contained)
# MAGIC
# MAGIC Ensures the Vector Search endpoint (`VS_ENDPOINT_NAME`) and a DELTA_SYNC
# MAGIC index (`VS_INDEX_NAME`) over the precomputed embeddings table exist, waits
# MAGIC for the index to come ONLINE, then smoke-tests a similarity query. Runs
# MAGIC UNCONDITIONALLY (the `create_vector_search` task of `deploy_champion_job`).
# MAGIC
# MAGIC The create / **dimension-aware drop-recreate** / sync logic lives in
# MAGIC `dais26_dentex.serve.vector_search.ensure_vector_search_index` (shared +
# MAGIC unit-tested). The embedding dimension is derived from the source table, so
# MAGIC it stays correct across backbones (C-RADIOv4-SO400M summary = 2304,
# MAGIC DINOv3-ViTL16 = 1024) — a champion backbone change drops + recreates the
# MAGIC index at the new dimension rather than syncing mismatched vectors.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
from databricks.sdk import WorkspaceClient

from dais26_dentex.serve.vector_search import ensure_vector_search_index

w = WorkspaceClient()
source_table = TRAIN_EMBEDDINGS_TABLE
print(f"VS endpoint : {VS_ENDPOINT_NAME}")
print(f"VS index    : {VS_INDEX_NAME}")
print(f"Source table: {source_table}")

# COMMAND ----------
# ---- Sanity-check the source table + derive the (immutable) embedding dimension ----
# Spark stays in the notebook; the package helper takes the derived dim + row count.
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
# ---- Ensure endpoint + DELTA_SYNC index (dimension-aware) and wait ONLINE ----
action = ensure_vector_search_index(
    w,
    endpoint_name=VS_ENDPOINT_NAME,
    index_name=VS_INDEX_NAME,
    source_table=source_table,
    embedding_dim=embedding_dim,
    wait_online=True,
    expected_rows=row_count,
)
print(f"Index {VS_INDEX_NAME}: {action}; ONLINE with {row_count} rows")

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
