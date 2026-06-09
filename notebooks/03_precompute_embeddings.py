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
# ---- Resolve the backbone from the LIVE champion (not the static config) ----
# The prod champion is a single backbone-agnostic model; whichever architecture
# currently holds @champion may differ from BACKBONE in 00_config. RegisterChampion
# (notebooks/12) tags the prod version with `source_dev_model`, so we reverse-map
# that to the backbone here and embed with the matching feature extractor. Falls
# back to BACKBONE when there is no champion yet / no tag (e.g. first run, or a
# standalone dev invocation).
from mlflow.tracking import MlflowClient

from dais26_dentex.config.champion import (
    SOURCE_DEV_MODEL_TAG,
    resolve_backbone_from_source_model,
)

_source_dev_model = None
try:
    _champ_mv = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(
        name=CHAMPION_MODEL_NAME, alias="champion"
    )
    _source_dev_model = (_champ_mv.tags or {}).get(SOURCE_DEV_MODEL_TAG)
except Exception as e:
    print(f"No resolvable @champion ({type(e).__name__}: {e}); falling back to BACKBONE={BACKBONE}")

EFFECTIVE_BACKBONE = resolve_backbone_from_source_model(
    _source_dev_model, _DETECTOR_NAMES_BY_BACKBONE, BACKBONE
)
print(
    f"Embeddings backbone = {EFFECTIVE_BACKBONE} "
    f"(champion source_dev_model={_source_dev_model}, config BACKBONE={BACKBONE})"
)

# COMMAND ----------

from dais26_dentex.train.precompute_embeddings import precompute_embeddings

n = precompute_embeddings(
    spark=spark,
    # Write to the CHAMPION catalog/schema, NOT the dev CATALOG/SCHEMA. This
    # notebook only runs as the champion deploy job's refresh step, and the
    # downstream consumers (04b create_vector_search, 05 drift_demo) all read
    # `TRAIN_EMBEDDINGS_TABLE` == `CHAMPION_CATALOG.CHAMPION_SCHEMA.<prefix>train_embeddings`.
    # Writing to the dev schema here made precompute "succeed" while
    # create_vector_search then failed with TABLE_OR_VIEW_NOT_FOUND on the prod
    # table. (Source images are still read from the dev VOLUME_PATH below.)
    catalog=CHAMPION_CATALOG,
    schema=CHAMPION_SCHEMA,
    volume_path=VOLUME_PATH,
    backbone_name=EFFECTIVE_BACKBONE,  # type: ignore[arg-type]
    backbone_revision=BACKBONE_REVISION,
    cache_dir=CACHE_DIR,
    batch_size=EMBEDDINGS_BATCH_SIZE,
    # table_name matches TRAIN_EMBEDDINGS_TABLE's prefix-aware name so the
    # VS index / drift / similarity-search notebooks find the embeddings.
    table_name=f"{TABLE_PREFIX}train_embeddings",
    vector_search_endpoint=EMBEDDINGS_VS_ENDPOINT,
    vector_search_index=EMBEDDINGS_VS_INDEX,
)
print(f"Wrote {n} embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## UMAP visualization (plotly — replaces FiftyOne per Critic iter 3)

# COMMAND ----------

# Non-essential visualization. The embeddings write above is the load-bearing
# step; umap / plotly may be absent on the serverless-GPU base env, so guard the
# viz so a missing optional dep never fails the job after embeddings are written.
try:
    import numpy as np
    import plotly.express as px
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
        title=f"UMAP of {EFFECTIVE_BACKBONE} embeddings ({len(df)} images)",
        height=600,
    )
    fig.show()
except Exception as e:
    print(f"[viz skipped] UMAP/plotly visualization failed (non-fatal): {e}")
