# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Precompute Embeddings + UMAP
# MAGIC Runs frozen C-RADIOv4 over all DENTEX splits. Writes summary embeddings to a Delta
# MAGIC table as ARRAY<FLOAT> with CDF enabled. Optionally creates / syncs the Vector Search index.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

from dais26_dentex.train.precompute_embeddings import precompute_embeddings

n = precompute_embeddings(
    spark=spark,
    catalog=CATALOG,
    schema=SCHEMA,
    volume_path=VOLUME_PATH,
    backbone_name=BACKBONE,  # type: ignore[arg-type]
    backbone_revision=BACKBONE_REVISION,
    cache_dir=CACHE_DIR,
    batch_size=EMBEDDINGS_BATCH_SIZE,
    vector_search_endpoint=EMBEDDINGS_VS_ENDPOINT,
    vector_search_index=EMBEDDINGS_VS_INDEX,
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
