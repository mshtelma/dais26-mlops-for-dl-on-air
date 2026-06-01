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
CATALOG = "mlops_pj"
SCHEMA = "dais26_vfm"
BACKBONE = "dinov3_vitl16"
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
# ---- UC-registered model names (backbone-aware) ----
# Keyed by the BACKBONE literal so flipping BACKBONE above retargets the
# detector model + endpoint cleanly and a DINOv3 run does NOT overwrite the
# C-RADIO model. EXPERIMENT_NAME stays shared (see below) so all runs land in
# one experiment for side-by-side comparison. The cradio entry preserves the
# historical names for backward compatibility.
_DETECTOR_NAMES_BY_BACKBONE: dict[str, dict[str, str]] = {
    "cradio_v4_so400m": {
        "model_short": "cradio_detector",
        "endpoint": "dais26-cradio-detector-dev",
    },
    "dinov3_vitl16": {
        "model_short": "dinov3_detector",
        "endpoint": "dais26-dinov3-detector-dev",
    },
    "dinov2_base": {
        "model_short": "dinov2_detector",
        "endpoint": "dais26-dinov2-detector-dev",
    },
}
_detector_names = _DETECTOR_NAMES_BY_BACKBONE.get(
    BACKBONE, _DETECTOR_NAMES_BY_BACKBONE["cradio_v4_so400m"]
)

DETECTOR_MODEL_SHORT = _detector_names["model_short"]
# Keep the LoRA short name backbone-aware too (cradio -> "cradio_detector_lora",
# matching the historical value) so 02_train_detector_air.py's use_lora branch
# stays consistent with DETECTOR_MODEL_SHORT.
DETECTOR_LORA_MODEL_SHORT = f"{DETECTOR_MODEL_SHORT}_lora"
EMBEDDER_MODEL_SHORT = "cradio_embedder"

DETECTOR_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{DETECTOR_MODEL_SHORT}"
DETECTOR_LORA_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{DETECTOR_LORA_MODEL_SHORT}"
EMBEDDER_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{EMBEDDER_MODEL_SHORT}"

# COMMAND ----------
# ---- Serving + Vector Search names ----
DETECTOR_ENDPOINT_NAME = _detector_names["endpoint"]
EMBEDDER_ENDPOINT_NAME = "dais26-cradio-embedder-dev"

VS_ENDPOINT_NAME = "dais26-vfm-vs"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.{TABLE_PREFIX}embeddings_index"

# COMMAND ----------
# ---- MLflow experiment ----
# Use the dbutils notebook context (no spark — AIR workers don't have spark, and
# the driver's spark.sql round-trip is unnecessary when dbutils gives us the
# username directly).
_current_user = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)
EXPERIMENT_NAME = f"/Users/{_current_user}/dais26-detector"

# COMMAND ----------
# ---- Backbone literals (canonical internal names for dais26_dentex.models.backbones) ----
PRIMARY_BACKBONE = "cradio_v4_so400m"
COMPARISON_BACKBONE = "dinov3_vitl16"
FALLBACK_BACKBONE = "dinov2_base"

# COMMAND ----------
# ---- Notebook defaults (formerly dbutils widgets) ----
# Edit these to change per-notebook behavior instead of using widget overrides.

# 00_setup
SP_APP_ID: str | None = None        # service principal app-ID (UUID); None skips UC grants
HF_TOKEN: str | None = None         # set only if the DENTEX HF repo is gated

# 01_explore_dentex
EXPLORE_SPLIT = "train"             # train | val | test | drift_synthetic

# 02_train_detector_air
TRAIN_EPOCHS = 10
TRAIN_LR = 1e-3
TRAIN_BATCH_SIZE = 8
TRAIN_USE_LORA = False
TRAIN_LORA_RANK = 8
TRAIN_LORA_ALPHA = 32.0
TRAIN_GPUS = 8                      # passed to @distributed
TRAIN_GPU_TYPE = "h100"             # "h100" | "a10"

# 03_precompute_embeddings
EMBEDDINGS_BATCH_SIZE = 32
# Auto-sync the VS index after writing embeddings. Set BOTH to enable;
# leave None to skip (create the VS index via 04 first, then enable here).
EMBEDDINGS_VS_ENDPOINT: str | None = None
EMBEDDINGS_VS_INDEX: str | None = None

# 04_deploy_serving
# register_and_set_candidate | deploy_and_smoke_test | create_vector_search
DEPLOY_ACTION = "deploy_and_smoke_test"
DEPLOY_WORKLOAD_TYPE = "GPU_SMALL"  # GPU_SMALL | GPU_MEDIUM | GPU_LARGE
DEPLOY_WORKLOAD_SIZE = "Small"      # Small | Medium | Large
DEPLOY_SCALE_TO_ZERO = True

# 05_drift_demo
DRIFT_MODE = "demo"                 # "demo" or "scheduled"
DRIFT_KNN_K = 50
DRIFT_ALERT_THRESHOLD = 2.0

# 06_similarity_search_demo
SIMILARITY_QUERY_COUNT = 50

# 07_latency_benchmark
LATENCY_NUM_REQUESTS = 1000
LATENCY_WARMUP_REQUESTS = 20
LATENCY_PIVOT_THRESHOLD_MS = 150.0

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
