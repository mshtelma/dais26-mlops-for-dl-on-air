# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Precompute Embeddings + UMAP
# MAGIC Runs frozen C-RADIOv4 over all DENTEX splits. Writes summary embeddings to a Delta
# MAGIC table as ARRAY<FLOAT> with CDF enabled. Optionally creates / syncs the Vector Search index.

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

dbutils.widgets.text("batch_size", "32")
dbutils.widgets.text("vs_endpoint", "")
dbutils.widgets.text("vs_index", "")

batch_size = int(dbutils.widgets.get("batch_size"))
vs_endpoint = dbutils.widgets.get("vs_endpoint").strip() or None
vs_index = dbutils.widgets.get("vs_index").strip() or VS_INDEX_NAME

# COMMAND ----------

from src.train.precompute_embeddings import precompute_embeddings

n = precompute_embeddings(
    spark=spark,
    catalog=CATALOG,
    schema=SCHEMA,
    volume_path=VOLUME_PATH,
    backbone_name=BACKBONE,  # type: ignore[arg-type]
    backbone_revision=BACKBONE_REVISION,
    cache_dir=CACHE_DIR,
    batch_size=batch_size,
    vector_search_endpoint=vs_endpoint,
    vector_search_index=vs_index,
)
print(f"Wrote {n} embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## UMAP visualization (plotly — replaces FiftyOne per Critic iter 3)

# COMMAND ----------

import plotly.express as px
import numpy as np
import umap

df = spark.table(TRAIN_EMBEDDINGS_TABLE).toPandas()
emb = np.stack(df["embedding"].apply(np.asarray).to_list()).astype(np.float32)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)
coords = reducer.fit_transform(emb)
df["umap_x"] = coords[:, 0]
df["umap_y"] = coords[:, 1]

fig = px.scatter(
    df, x="umap_x", y="umap_y", color="diagnosis",
    symbol="split", hover_data=["image_id", "image_path"],
    title=f"UMAP of {BACKBONE} embeddings ({len(df)} images)",
    height=600,
)
fig.show()
