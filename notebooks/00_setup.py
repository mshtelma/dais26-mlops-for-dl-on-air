# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — UC Bootstrap
# MAGIC
# MAGIC Idempotent setup of Unity Catalog catalog, schema, volumes, secret scope, and grants
# MAGIC for the DAIS26 VFM showcase. On first run also stages the DENTEX dataset into the
# MAGIC `dentex_raw` volume in three explicit steps (skipped entirely if
# MAGIC `annotations/train.json` already exists):
# MAGIC
# MAGIC 1. `download_dentex` — HuggingFace `snapshot_download` of `ibrahimhamamci/DENTEX`.
# MAGIC 2. `extract_all_zips` — recursive unpack of every `*.zip` under the volume.
# MAGIC 3. `convert_to_coco` — count-match discovery of per-split source JSONs and
# MAGIC    canonical COCO output at `annotations/{train,val,test}.json`.
# MAGIC
# MAGIC Inference-table grants are deferred to `scripts/grant_inference_table_access.py`
# MAGIC (those tables don't exist until the first endpoint query).

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

print(f"Catalog: {CATALOG}")
print(f"Schema:  {CATALOG}.{SCHEMA}")
print(f"SP app-ID: {SP_APP_ID or '(none)'}")

# COMMAND ----------

# Volumes for the dataset and the model cache
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{DENTEX_VOLUME}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{MODEL_CACHE_VOLUME}")

# COMMAND ----------

# Stage the DENTEX dataset into the volume in three explicit steps
# (idempotent — skipped entirely if annotations/train.json already exists).
from pathlib import Path

from dais26_dentex.data.dentex_loader import (
    convert_to_coco,
    download_dentex,
    extract_all_zips,
    normalize_canonical_annotations,
)

train_ann = Path(VOLUME_PATH) / "annotations" / "train.json"
if train_ann.exists():
    print(f"DENTEX already prepared at {VOLUME_PATH} (annotations/train.json present); skipping download.")
else:
    print(f"[1/3] Downloading DENTEX repo -> {VOLUME_PATH}")
    download_dentex(volume_path=VOLUME_PATH, hf_token=HF_TOKEN)

    print(f"[2/3] Extracting all *.zip archives under {VOLUME_PATH}")
    extracted_dirs = extract_all_zips(VOLUME_PATH)
    print(f"  unpacked {len(extracted_dirs)} archive(s):")
    for d in extracted_dirs:
        print(f"    {d}")

    print("[3/3] Building canonical COCO annotations + image splits")
    mapping = convert_to_coco(VOLUME_PATH)
    for split, path in mapping.items():
        print(f"  {split} -> {path}")

# Self-heal: older versions of the loader wrote canonical files with DENTEX's
# hierarchical category_id_3 instead of the flat category_id that downstream
# code reads. Re-normalize in place; no-op when already normalized.
healed = normalize_canonical_annotations(VOLUME_PATH)
for split, changed in healed.items():
    if changed:
        print(f"  normalized stale annotations in {split}.json (category_id_3 -> category_id)")

# COMMAND ----------

# train_embeddings table — ARRAY<FLOAT> (NOT ARRAY<DOUBLE>) + Change Data Feed
# (Vector Search Delta Sync requires both per Critic iter 3 finding D5/R8)
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TRAIN_EMBEDDINGS_TABLE} (
    image_id   STRING,
    embedding  ARRAY<FLOAT>,
    diagnosis  STRING,
    split      STRING,
    image_path STRING
)
USING DELTA
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# drift_scores table
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {DRIFT_SCORES_TABLE} (
    timestamp     STRING,
    batch_id      STRING,
    knn_distance  DOUBLE,
    mmd_score     DOUBLE,
    n_images      BIGINT,
    alert         BOOLEAN
)
USING DELTA
""")

# COMMAND ----------

# UC grants for the service principal (if provided)
if SP_APP_ID:
    grants = [
        f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{SP_APP_ID}`",
        f"GRANT USE SCHEMA, CREATE TABLE, MODIFY ON SCHEMA {CATALOG}.{SCHEMA} TO `{SP_APP_ID}`",
        f"GRANT CREATE MODEL ON SCHEMA {CATALOG}.{SCHEMA} TO `{SP_APP_ID}`",
        f"GRANT READ VOLUME, WRITE VOLUME ON VOLUME {CATALOG}.{SCHEMA}.{DENTEX_VOLUME} TO `{SP_APP_ID}`",
        f"GRANT READ VOLUME, WRITE VOLUME ON VOLUME {CATALOG}.{SCHEMA}.{MODEL_CACHE_VOLUME} TO `{SP_APP_ID}`",
        f"GRANT SELECT, MODIFY ON TABLE {TRAIN_EMBEDDINGS_TABLE} TO `{SP_APP_ID}`",
        f"GRANT SELECT, MODIFY ON TABLE {DRIFT_SCORES_TABLE} TO `{SP_APP_ID}`",
    ]
    for stmt in grants:
        try:
            spark.sql(stmt)
            print(f"OK: {stmt}")
        except Exception as e:
            print(f"SKIP ({e.__class__.__name__}): {stmt}")
else:
    print("No SP_APP_ID configured in 00_config.py; skipping grants.")

# COMMAND ----------

print("UC bootstrap complete.")
print("Volume paths:")
print(f"  {VOLUME_PATH}")
print(f"  {CACHE_DIR}")
