# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — UC Bootstrap
# MAGIC
# MAGIC Idempotent setup of Unity Catalog catalog, schema, volumes, secret scope, and grants
# MAGIC for the DAIS26 VFM showcase.
# MAGIC
# MAGIC Inference-table grants are deferred to `scripts/grant_inference_table_access.py`
# MAGIC (those tables don't exist until the first endpoint query).

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

dbutils.widgets.text("sp_app_id", "", "Service principal app-ID (UUID); leave blank for dev")
sp_app_id = dbutils.widgets.get("sp_app_id").strip()

print(f"Catalog: {CATALOG}")
print(f"Schema:  {CATALOG}.{SCHEMA}")
print(f"SP app-ID: {sp_app_id or '(none)'}")

# COMMAND ----------

# Catalog (skip if already exists; requires CREATE CATALOG privilege)
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# Volumes for the dataset and the model cache
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{DENTEX_VOLUME}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{MODEL_CACHE_VOLUME}")

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
if sp_app_id:
    grants = [
        f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{sp_app_id}`",
        f"GRANT USE SCHEMA, CREATE TABLE, MODIFY ON SCHEMA {CATALOG}.{SCHEMA} TO `{sp_app_id}`",
        f"GRANT CREATE MODEL ON SCHEMA {CATALOG}.{SCHEMA} TO `{sp_app_id}`",
        f"GRANT READ VOLUME, WRITE VOLUME ON VOLUME {CATALOG}.{SCHEMA}.{DENTEX_VOLUME} TO `{sp_app_id}`",
        f"GRANT READ VOLUME, WRITE VOLUME ON VOLUME {CATALOG}.{SCHEMA}.{MODEL_CACHE_VOLUME} TO `{sp_app_id}`",
        f"GRANT SELECT, MODIFY ON TABLE {TRAIN_EMBEDDINGS_TABLE} TO `{sp_app_id}`",
        f"GRANT SELECT, MODIFY ON TABLE {DRIFT_SCORES_TABLE} TO `{sp_app_id}`",
    ]
    for stmt in grants:
        try:
            spark.sql(stmt)
            print(f"OK: {stmt}")
        except Exception as e:
            print(f"SKIP ({e.__class__.__name__}): {stmt}")
else:
    print("No sp_app_id provided; skipping grants.")

# COMMAND ----------

print("UC bootstrap complete.")
print("Volume paths:")
print(f"  {VOLUME_PATH}")
print(f"  {CACHE_DIR}")
