# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Vector Search Similarity Demo
# MAGIC Query the embeddings index with a held-out val image, display top-10 similar images.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

import numpy as np
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

val_df = (
    spark.table(TRAIN_EMBEDDINGS_TABLE)
    .filter("split = 'val'")
    .limit(SIMILARITY_QUERY_COUNT)
    .toPandas()
)
print(f"Querying with {len(val_df)} val images")

# COMMAND ----------

hits = 0
total = 0
for _, row in val_df.iterrows():
    query_vec = list(map(float, row["embedding"]))
    expected = row["diagnosis"]
    res = w.vector_search_indexes.query_index(
        index_name=VS_INDEX_NAME,
        columns=["image_id", "diagnosis", "split"],
        query_vector=query_vec,
        num_results=10,
    )
    rows = res.result.data_array if hasattr(res, "result") else getattr(res, "data_array", [])
    # Strip the query itself if present
    rows = [r for r in rows if r and r[0] != row["image_id"]][:10]
    same_class = sum(1 for r in rows if r and len(r) > 1 and r[1] == expected)
    hits += same_class
    total += len(rows)

recall = hits / total if total else 0.0
print(f"Same-class hits: {hits} / {total} = {recall:.3f}")
print(f"E7 pass (recall >= 0.80): {recall >= 0.80}")
