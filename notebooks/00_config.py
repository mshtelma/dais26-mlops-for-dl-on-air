# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Shared configuration
# MAGIC
# MAGIC `%run ./00_config` from any other notebook to pull in shared UC config,
# MAGIC derived paths, table FQNs, model names, endpoint names, and the MLflow
# MAGIC experiment path. **All values are configured here directly** — edit this
# MAGIC notebook to switch catalogs / schemas / backbones.

# COMMAND ----------
# ---- Core UC config (edit here to switch environments) ----
CATALOG = "ml_dev"
SCHEMA = "dais26_vfm"
BACKBONE = "cradio_v4_so400m"
BACKBONE_REVISION = "main"

# Table-name prefix so multiple DAIS26 projects can share one schema without colliding.
TABLE_PREFIX = "dais26_dentex_"

# COMMAND ----------
# ---- Volume names + derived paths ----
DENTEX_VOLUME = "dentex_raw"
MODEL_CACHE_VOLUME = "model_cache"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{DENTEX_VOLUME}"
CACHE_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{MODEL_CACHE_VOLUME}"

# COMMAND ----------
# ---- UC Delta tables (all prefixed with TABLE_PREFIX) ----
TRAIN_EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_PREFIX}train_embeddings"
DRIFT_SCORES_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_PREFIX}drift_scores"
DETECTOR_INFERENCE_TABLE = f"{TABLE_PREFIX}detector_inference_payload"

# COMMAND ----------
# ---- UC-registered model names ----
DETECTOR_MODEL_SHORT = "cradio_detector"
DETECTOR_LORA_MODEL_SHORT = "cradio_detector_lora"
EMBEDDER_MODEL_SHORT = "cradio_embedder"

DETECTOR_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{DETECTOR_MODEL_SHORT}"
DETECTOR_LORA_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{DETECTOR_LORA_MODEL_SHORT}"
EMBEDDER_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{EMBEDDER_MODEL_SHORT}"

# COMMAND ----------
# ---- Serving + Vector Search names ----
DETECTOR_ENDPOINT_NAME = "dais26-cradio-detector-dev"
EMBEDDER_ENDPOINT_NAME = "dais26-cradio-embedder-dev"

VS_ENDPOINT_NAME = "dais26-vfm-vs"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.{TABLE_PREFIX}embeddings_index"

# COMMAND ----------
# ---- MLflow experiment ----
_current_user = spark.sql("SELECT current_user()").collect()[0][0]
EXPERIMENT_NAME = f"/Users/{_current_user}/dais26-detector"

# COMMAND ----------
# ---- Backbone literals (canonical internal names for src.models.backbones) ----
PRIMARY_BACKBONE = "cradio_v4_so400m"
COMPARISON_BACKBONE = "dinov3_vitl16"
FALLBACK_BACKBONE = "dinov2_base"

# COMMAND ----------
print(f"CATALOG          = {CATALOG}")
print(f"SCHEMA           = {SCHEMA}")
print(f"BACKBONE         = {BACKBONE} @ {BACKBONE_REVISION}")
print(f"VOLUME_PATH      = {VOLUME_PATH}")
print(f"CACHE_DIR        = {CACHE_DIR}")
print(f"EXPERIMENT_NAME  = {EXPERIMENT_NAME}")
print(f"DETECTOR_MODEL   = {DETECTOR_MODEL_NAME}")
print(f"TRAIN_EMB_TABLE  = {TRAIN_EMBEDDINGS_TABLE}")
print(f"VS_INDEX_NAME    = {VS_INDEX_NAME}")
