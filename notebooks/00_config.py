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
BACKBONE = "cradio_v4_so400m"
BACKBONE_REVISION = "main"

# ---- Prod / champion schema (Big Book "deploy code" dev/prod asset split) ----
# Dev models (with @challenger) live in CATALOG.SCHEMA; the promote task copies
# the approved version into a SEPARATE prod/broad schema and registers it as
# @champion there (lineage back to the source run is preserved via
# MlflowClient.copy_model_version). Same catalog by default; can become a
# separate catalog later without touching the rest of the code.
CHAMPION_CATALOG = "mlops_pj"
CHAMPION_SCHEMA = "dais26_vfm_prod"

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
# The embedding + monitoring subsystem (reference embeddings, drift scores) and
# the Vector Search index (VS_INDEX_NAME below) live in the PROD / champion schema,
# NOT the dev schema. precompute_embeddings, create_vector_search, and drift_monitor
# are prod-only jobs (run_as the SP), so their artifacts sit alongside the @champion
# model rather than being populated by a dev job and read cross-tier from prod. Keep
# in sync with the targets.prod scoping in
# resources/jobs/{precompute_embeddings,create_vector_search,drift_monitor}.yml.
TRAIN_EMBEDDINGS_TABLE = f"{CHAMPION_CATALOG}.{CHAMPION_SCHEMA}.{TABLE_PREFIX}train_embeddings"
DRIFT_SCORES_TABLE = f"{CHAMPION_CATALOG}.{CHAMPION_SCHEMA}.{TABLE_PREFIX}drift_scores"
DETECTOR_INFERENCE_TABLE = f"{TABLE_PREFIX}detector_inference_payload"

# COMMAND ----------
# ---- UC-registered model names (backbone-aware) ----
# Keyed by the BACKBONE literal so flipping BACKBONE above retargets the
# detector model + endpoint cleanly and a DINOv3 run does NOT overwrite the
# C-RADIO model. EXPERIMENT_NAME stays shared (see below) so all runs land in
# one experiment for side-by-side comparison. The mapping lives in the package
# (`config.recipes`) so the air lane and tests resolve the same identities;
# the leading-underscore alias keeps the historical notebook-local name.
from dais26_dentex.config.recipes import (
    DETECTOR_NAMES_BY_BACKBONE as _DETECTOR_NAMES_BY_BACKBONE,
)

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

# Prod-schema champion model — SINGLE, backbone-AGNOSTIC. Broad/prod deployment
# comes from ONE champion model in ONE schema, regardless of which dev backbone
# (C-RADIOv4 / DINOv3 / ...) won. The dev side keeps backbone-keyed models that
# compete on the val gate; the winner — whatever its architecture — is copied into
# this single model and served. So prod is never two competing architecture-named
# champions. The RegisterChampion task (notebooks/12) does
# copy_model_version(DETECTOR_MODEL_NAME -> CHAMPION_MODEL_NAME) and tags the source
# backbone on the prod version; the deploy_champion task (notebooks/14) serves it.
CHAMPION_MODEL_SHORT = "detector_champion"
CHAMPION_MODEL_NAME = f"{CHAMPION_CATALOG}.{CHAMPION_SCHEMA}.{CHAMPION_MODEL_SHORT}"

# COMMAND ----------
# ---- Serving + Vector Search names ----
# Dev endpoints are backbone-keyed (per-architecture testing). The PROD champion
# endpoint is SINGLE and backbone-agnostic: broad deployment serves whatever
# architecture currently holds @champion from one endpoint (mirrors the single
# CHAMPION_MODEL_NAME). The deploy_champion task (notebooks/14) deploys here.
DETECTOR_ENDPOINT_NAME = _detector_names["endpoint"]
CHAMPION_ENDPOINT_NAME = "dais26-detector-champion"
EMBEDDER_ENDPOINT_NAME = "dais26-cradio-embedder-dev"

VS_ENDPOINT_NAME = "dais26-vfm-vs"
# Index lives in the prod/champion schema alongside the embeddings table it syncs
# from (created by the prod-only create_vector_search job).
VS_INDEX_NAME = f"{CHAMPION_CATALOG}.{CHAMPION_SCHEMA}.{TABLE_PREFIX}embeddings_index"

# COMMAND ----------
# ---- MLflow experiment ----
# Use the dbutils notebook context (no spark — AIR workers don't have spark, and
# the driver's spark.sql round-trip is unnecessary when dbutils gives us the
# username directly).
#
# Point at the BUNDLE-MANAGED experiment (resources/experiments/vfm_experiment.yml,
# name "/Users/${workspace.current_user.userName}/dais26_vfm_experiment") so every
# training run, the HPO sweep, and the lineage-preserving champion copy all share
# one experiment root that the DAB owns. Keeping the literal in sync with the YAML
# means the deployment job's best-in-experiment gate (notebooks/10) reads the same
# experiment the trainer logs to. (Still per-user today; when prod runs as the SP,
# repoint both this and the YAML at a shared project-folder experiment.)
_current_user = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)
EXPERIMENT_NAME = f"/Users/{_current_user}/dais26_vfm_experiment"

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
# Hyperparameters come from the per-backbone recipe in
# `dais26_dentex.config.recipes.RECIPES` (campaign-final best-known config;
# same source the air workloads name via `recipe:`). The constants below are
# the notebook lane's EXPLICIT overrides / launch knobs only.
#
# Demo wall-time override: the recipes' full schedule is 150 epochs
# (dazzling-mole-850 / capricious-hound-240); 50 keeps the quickstart ~2h.
TRAIN_EPOCHS = 50
TRAIN_USE_LORA = False              # stretch path; recipes default to full fine-tune
TRAIN_LORA_RANK = 8
TRAIN_LORA_ALPHA = 32.0
TRAIN_GPUS = 8                      # passed to @distributed
TRAIN_GPU_TYPE = "h100"             # "h100" | "a10"

# 02b_hpo_sweep
# The sweep/campaign configuration lives in the PACKAGE now:
#   dais26_dentex.config.campaigns  — CAMPAIGN_STAGES (the "push to 0.60"
#       stage chain, typed + unit-tested) and SWEEP_DEFAULTS (the legacy
#       post-fix sweep block).
#   dais26_dentex.train.sweep_runner — the orchestration both lanes share.
# Pick a stage here (or pass the `sweep_stage` job parameter, which wins);
# None runs the legacy SWEEP_DEFAULTS sweep. The same stages are launchable
# from a terminal via `air run -f air/workload_sweep.yaml
# --override parameters.stage=<name>`.
SWEEP_STAGE: str | None = None

# COMMAND ----------
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
# A cold GPU serving deploy of the multi-GB detector can take ~1h; wait it out
# (and wait out any in-flight update on a re-run) rather than timing out and
# colliding with the still-rolling update. Keep below the job timeout_seconds.
DEPLOY_TIMEOUT_SECONDS = 5400  # 90 min

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
print(f"CHAMPION_MODEL   = {CHAMPION_MODEL_NAME}")
print(f"CHAMPION_ENDPT   = {CHAMPION_ENDPOINT_NAME}")
print(f"TRAIN_EMB_TABLE  = {TRAIN_EMBEDDINGS_TABLE}")
print(f"VS_INDEX_NAME    = {VS_INDEX_NAME}")
