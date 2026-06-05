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
# TEMP (sweep-derived): winner of hpo-sweep-cradio_v4_so400m is backbone_mode=frozen
# (band 40-50); its val/best_mAP_50 was still setting new highs at the final epoch
# (15) -> "still rising" -> pick the long end of the frozen band. >= SWEEP_TRIAL_EPOCHS.
TRAIN_EPOCHS = 50
# Longer-epoch hedge for the winner retrain (see docs/HPO.md "Next HPO sweep").
# The per-level anchor fix reshapes the matcher/loss landscape, so the prior ~e40
# saturation point may move; the sweep retrains the winner at both TRAIN_EPOCHS and
# TRAIN_EPOCHS_LONG and keeps whichever scores higher on SWEEP_PRIMARY_METRIC. The
# Trainer tracks the best checkpoint, so a longer run that peaks early still reports
# its peak rather than an over-trained final epoch.
TRAIN_EPOCHS_LONG = 100
TRAIN_LR = 1e-3
TRAIN_BATCH_SIZE = 8
TRAIN_USE_LORA = False
TRAIN_LORA_RANK = 8
TRAIN_LORA_ALPHA = 32.0
TRAIN_GPUS = 8                      # passed to @distributed
TRAIN_GPU_TYPE = "h100"             # "h100" | "a10"

# 02b_hpo_sweep
# Sequential @distributed AIR trials, each a nested MLflow run under one parent.
# Keep trials cheap (short epochs, small count) — the winner is retrained at both
# TRAIN_EPOCHS and TRAIN_EPOCHS_LONG, and @candidate points at the better schedule.
# Each trial is a full 8xH100 job, so the orchestrating job needs the 8h timeout
# (see resources/jobs + sgcli).
SWEEP_STRATEGY = "random"            # "grid" | "random"
SWEEP_MAX_TRIALS = 8
SWEEP_TRIAL_EPOCHS = 25              # ~25 epochs/trial; winner retrained at TRAIN_EPOCHS
SWEEP_SEED = 42
SWEEP_PRIMARY_METRIC = "val/best_mAP_50"
SWEEP_REGISTER_WINNER = True         # retrain the winning config full-length + register @candidate

# Post-fix sweep (see docs/HPO.md "Next HPO sweep"). The encoder axis is settled
# (full fine-tune won at 0.335) and the optimizer region is known, so the budget
# is spent on the newly-unlocked per-level anchor geometry. These knobs are PINNED
# on every trial (merged into the base TrainerConfig in 02b_hpo_sweep.py); they are
# NOT swept:
SWEEP_PINNED: dict = {
    "backbone_mode": "full",         # full fine-tune beat lora/frozen (0.335 vs 0.228/0.213)
    "backbone_lr": 1e-5,             # discriminative LR; head lr is swept below
    "onecycle_pct_start": 0.3,       # won previously
    "weight_decay": 1e-2,            # won previously
    "img_size": 1024,
    "anchor_layout": "per_level",    # THE fix: stride-scaled RetinaNet anchors
    "nms_per_class": True,           # per-class batched_nms
}
# Field name -> list of choices. `anchor_base_scale` sets per-level base = stride x
# base_scale; `aspect_ratios` adds an elongated-box option for impacted teeth.
SWEEP_SEARCH_SPACE: dict = {
    "anchor_base_scale": [3.0, 4.0, 5.0],
    "aspect_ratios": [[0.5, 1.0, 2.0], [0.33, 0.5, 1.0, 2.0, 3.0]],
    "lr": [1e-4, 2e-4],
    "box_loss_weight": [1.0, 2.0],
    "focal_gamma": [2.0, 2.5],
}

# COMMAND ----------
# ---- "Push to 0.60" tuning campaign (docs/HPO.md, plan: DINOv3 and C-RADIO to 0.60) ----
# Two sequential single-model campaigns (DINOv3 first, then C-RADIO), each a chain
# of gated stages that close the ~0.08 gap from ~0.52 -> ~0.60 by adding the
# untried high-leverage knobs (resolution, longer schedule, denser anchors,
# stronger augmentation) on top of the settled per-level recipe.
#
# Set SWEEP_STAGE to one of CAMPAIGN_STAGES below and run notebooks/02b_hpo_sweep.py;
# the notebook overrides BACKBONE / model_name / SWEEP_PINNED / SWEEP_SEARCH_SPACE /
# trial-epochs / schedule / register-flag from the selected stage. Leave it None to
# use the legacy SWEEP_* constants above. Run stages in order — the pinned values in
# s2/s3/s4 are SEEDED with the current best and MUST be updated to the prior stage's
# MLflow winner before launching (the chain is inherently sequential).
SWEEP_STAGE: str | None = None

# Standard RetinaNet octave sets used as `anchor_octaves` sweep choices. OCT3 is the
# 3-octave default written long-hand (so it shows up explicitly in the run params);
# OCT4 adds a 4th octave for denser scale coverage of small caries / periapical.
_OCT3 = [2.0**0, 2.0 ** (1.0 / 3.0), 2.0 ** (2.0 / 3.0)]
_OCT4 = [2.0**0, 2.0**0.25, 2.0**0.5, 2.0**0.75]
_AR3 = [0.5, 1.0, 2.0]
_AR5 = [0.33, 0.5, 1.0, 2.0, 3.0]

CAMPAIGN_STAGES: dict[str, dict] = {
    # ===== Campaign 1 — DINOv3 (fp32; fix = regularize THEN extend + raise res) =====
    "dinov3_s1": {  # resolution x schedule
        "backbone": "dinov3_vitl16",
        "trial_epochs": 30,
        "schedule_epochs": [50, 75],   # winner retrained at both; keep better
        "max_trials": 6,
        "register_winner": False,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto",       # -> fp32 for DINOv3
            "batch_size": 2, "grad_accum_steps": 2,   # effective 2*2*8=32 at 1024 AND 1280
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
        },
        "search_space": {
            "img_size": [1024, 1280],
            "lr": [1e-4, 2e-4],
            "onecycle_pct_start": [0.1, 0.3],
        },
    },
    "dinov3_regres": {  # regularize + resolution (overfit fix; see docs/HPO.md "DINOv3 plateau")
        # Diagnosis (intrigued-stork-789): 1024px/no-aug DINOv3 overfits — val
        # mAP flat ~0.50 from e30 while train loss keeps falling to 0.23; the
        # 0.532 @e49 was 50-img val noise. Stage-1 trials never sampled 1280, and
        # GPU mem was only 31% used. Fix = regularization + resolution together.
        "backbone": "dinov3_vitl16",
        "trial_epochs": 10,            # trivial single trial; the 75ep retrain is the real run
        "schedule_epochs": [75],
        "max_trials": 1,
        "register_winner": False,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto",       # -> fp32 for DINOv3
            "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280,          # was never tried at 1024-only Stage 1; mem headroom is huge
            "lr": 2e-4, "onecycle_pct_start": 0.1,   # s1 winner optimizer region
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
            # Regularization (the DINOv3 gap): multi-scale + small rotation + stronger jitter
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
        },
        "search_space": {"base_seed": [42]},   # degenerate 1-trial "sweep" -> 75ep retrain
    },
    "dinov3_s2": {  # anchor + loss (pin s1 winner res/lr/pct_start below!)
        "backbone": "dinov3_vitl16",
        "trial_epochs": 30,
        "schedule_epochs": [75],
        "max_trials": 8,
        "register_winner": False,
        "pinned": {
            "backbone_mode": "full", "weight_decay": 1e-2,
            "anchor_layout": "per_level", "nms_per_class": True, "amp_dtype": "auto",
            "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.1,   # <- regres winner region
            # carry the regularization base forward: anchor/loss sweep on a
            # non-augmented base would just re-overfit (see "DINOv3 plateau" in HPO.md)
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
        },
        "search_space": {
            "anchor_base_scale": [3.0, 4.0],
            "anchor_octaves": [_OCT3, _OCT4],
            "aspect_ratios": [_AR3, _AR5],
            "focal_gamma": [2.0, 2.5],
            "box_loss_weight": [1.0, 2.0],
            "backbone_lr": [1e-5, 2e-5],
        },
    },
    "dinov3_s3": {  # augmentation / regularization (pin s2 winner anchors/loss below!)
        "backbone": "dinov3_vitl16",
        "trial_epochs": 40,
        "schedule_epochs": [75],
        "max_trials": 6,
        "register_winner": False,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "nms_per_class": True, "amp_dtype": "auto",
            "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 1e-4, "onecycle_pct_start": 0.3,
            "anchor_base_scale": 4.0, "focal_gamma": 2.0, "box_loss_weight": 1.0,  # <- from s2 winner
        },
        "search_space": {
            "aug_multiscale_range": [[0.8, 1.0], [0.7, 1.0]],
            "aug_rotation_deg": [0.0, 7.0],
            "aug_jitter_scale": [1.0, 1.5],
        },
    },
    "dinov3_s4": {  # finalize: single combined-best combo, register @candidate
        "backbone": "dinov3_vitl16",
        "trial_epochs": 30,
        "schedule_epochs": [75],
        "max_trials": 1,
        "register_winner": True,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "nms_per_class": True, "amp_dtype": "auto",
            "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 1e-4, "onecycle_pct_start": 0.3,
            "anchor_base_scale": 4.0, "focal_gamma": 2.0, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
        },
        "search_space": {"base_seed": [42]},   # degenerate 1-trial "sweep" -> retrain+register
    },
    "dinov3_res1536": {  # step-change attempt: push resolution past 1280 (see HPO.md "DINOv3 ceiling")
        # DINOv3 capped ~0.535 on every planned knob; the only untried high-leverage
        # lever is resolution beyond 1280. 1536px fp32 needs batch=1 (grad_accum keeps
        # effective batch 32). Longer 100ep schedule since aug controls overfit now.
        "backbone": "dinov3_vitl16",
        "trial_epochs": 10,
        "schedule_epochs": [100],
        "max_trials": 1,
        "register_winner": False,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto",       # -> fp32 for DINOv3
            "batch_size": 1, "grad_accum_steps": 4,   # effective 1*4*8=32 at 1536 fp32
            "img_size": 1536,
            "lr": 2e-4, "onecycle_pct_start": 0.1,
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
        },
        "search_space": {"base_seed": [42]},   # degenerate 1-trial "sweep" -> 100ep retrain
    },
    "dinov3_falpha": {  # the one config knob we skipped for DINOv3: focal_alpha (Caries class)
        # Low odds of breaking the ceiling, but cheap and closes the config search
        # honestly. Pinned on the victorious-goose-410 base (1280 + aug).
        "backbone": "dinov3_vitl16",
        "trial_epochs": 30,
        "schedule_epochs": [75],
        "max_trials": 3,
        "register_winner": False,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.1,
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
        },
        "search_space": {"focal_alpha": [0.25, 0.5, 0.75]},
    },
    "dinov3_fusion": {  # Track B step-change: multi-layer ViT feature fusion (ViT-Det style)
        # The headline architectural lever for DINOv3 — fuse hidden states
        # L6/12/18/24 into the FPN instead of last-layer-only. Pinned on the
        # victorious-goose-410 base so the ONLY change vs that run is fusion;
        # clean attribution of any gain past the ~0.53 ceiling.
        "backbone": "dinov3_vitl16",
        "trial_epochs": 10,
        "schedule_epochs": [75],
        "max_trials": 1,
        "register_winner": False,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.1,
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
            "fusion_layers": [6, 12, 18, 24],
        },
        "search_space": {"base_seed": [42]},
    },
    # ===== Campaign 2 — C-RADIO (fp16; fix = extend schedule + raise resolution) =====
    "cradio_s1": {  # schedule x resolution (+ mild aug: DINOv3 proved no-aug long runs overfit)
        "backbone": "cradio_v4_so400m",
        "trial_epochs": 30,
        "schedule_epochs": [75, 100],
        "max_trials": 8,
        "register_winner": False,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 3.0, "nms_per_class": True,
            "amp_dtype": "auto",       # -> fp16 for C-RADIO
            "batch_size": 4, "grad_accum_steps": 2,   # effective 4*2*8=64 (matches 0.5219 run)
            "focal_gamma": 2.5, "box_loss_weight": 1.0,
            # Mild regularization baked in: DINOv3 showed 75-100ep with NO aug just
            # overfits the 705-img train set. Keep it gentle so the schedule x
            # resolution signal stays readable; s3 explores aug strength further.
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0, "aug_jitter_scale": 1.5,
        },
        "search_space": {
            "img_size": [1024, 1280, 1536],
            "lr": [2e-4, 3e-4],
            "onecycle_pct_start": [0.2, 0.3],
        },
    },
    "cradio_long": {  # pure schedule test: does 150ep beat 100ep? (winner still rising at e100)
        # useful-mare-854 was still climbing at e100 with train loss falling — extend
        # the exact winner config to 150ep to bank the free schedule gain.
        "backbone": "cradio_v4_so400m",
        "trial_epochs": 10,
        "schedule_epochs": [150],
        "max_trials": 1,
        "register_winner": False,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 3.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 4, "grad_accum_steps": 2,
            "img_size": 1024, "lr": 2e-4, "onecycle_pct_start": 0.2,
            "focal_gamma": 2.5, "focal_alpha": 0.25, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0, "aug_jitter_scale": 1.5,
        },
        "search_space": {"base_seed": [42]},
    },
    "cradio_s2": {  # anchor + loss, pinned on s1 winner useful-mare-854 (1024/100ep+aug)
        "backbone": "cradio_v4_so400m",
        "trial_epochs": 30,
        "schedule_epochs": [100],
        "max_trials": 10,
        "register_winner": False,
        "pinned": {
            "backbone_mode": "full", "weight_decay": 1e-2,
            "anchor_layout": "per_level", "nms_per_class": True, "amp_dtype": "auto",
            "batch_size": 4, "grad_accum_steps": 2,
            "img_size": 1024, "lr": 2e-4, "onecycle_pct_start": 0.2,   # <- s1 winner useful-mare-854
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0, "aug_jitter_scale": 1.5,
        },
        "search_space": {
            "anchor_base_scale": [2.0, 2.5, 3.0],
            "anchor_octaves": [_OCT3, _OCT4],
            "aspect_ratios": [_AR3, _AR5],
            "focal_gamma": [2.0, 2.5],
            "box_loss_weight": [1.0, 2.0],
            "focal_alpha": [0.25, 0.5],
            "backbone_lr": [1e-5, 2e-5],
        },
    },
    "cradio_s3": {  # augmentation (pin s2 winner anchors/loss below!)
        "backbone": "cradio_v4_so400m",
        "trial_epochs": 40,
        "schedule_epochs": [100],
        "max_trials": 4,
        "register_winner": False,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "nms_per_class": True, "amp_dtype": "auto",
            "batch_size": 4, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.3,
            "anchor_base_scale": 2.5, "focal_gamma": 2.5, "box_loss_weight": 1.0,  # <- from s2 winner
        },
        "search_space": {
            "aug_multiscale_range": [[0.8, 1.0], [0.7, 1.0]],
            "aug_rotation_deg": [0.0, 5.0],
        },
    },
    "cradio_s4": {  # finalize: single combined-best combo, register @candidate
        "backbone": "cradio_v4_so400m",
        "trial_epochs": 30,
        "schedule_epochs": [100],
        "max_trials": 1,
        "register_winner": True,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "nms_per_class": True, "amp_dtype": "auto",
            "batch_size": 4, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.3,
            "anchor_base_scale": 2.5, "focal_gamma": 2.5, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0,
        },
        "search_space": {"base_seed": [42]},
    },
    "cradio_giou": {  # Track B for C-RADIO: GIoU box loss + Caries oversampling (no fusion)
        # C-RADIO can't fuse (custom HF model), but the backbone-agnostic Track B
        # levers apply: GIoU localization loss + 2x Caries oversampling, pinned on
        # the useful-mare-854 winner. Pushes C-RADIO further while DINOv3 fuses.
        "backbone": "cradio_v4_so400m",
        "trial_epochs": 10,
        "schedule_epochs": [100],
        "max_trials": 1,
        "register_winner": False,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 3.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 4, "grad_accum_steps": 2,
            "img_size": 1024, "lr": 2e-4, "onecycle_pct_start": 0.2,
            "focal_gamma": 2.5, "focal_alpha": 0.25, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0, "aug_jitter_scale": 1.5,
            "box_loss_type": "giou", "caries_oversample": 2.0,
        },
        "search_space": {"base_seed": [42]},
    },
    # ===== Finalize — compound the confirmed winners, register @candidate =====
    # Results so far (val mAP@50): C-RADIO 150ep=0.5931 (schedule is the lever),
    # C-RADIO giou+cov@100ep=0.5674 (50:95 ↑), DINOv3 fusion@75ep=0.5504 (broke
    # the ceiling). Nobody combined the winners — these stages do.
    "cradio_final": {  # C-RADIO: 150ep schedule winner + GIoU + Caries x2, registered
        "backbone": "cradio_v4_so400m",
        "trial_epochs": 10,
        "schedule_epochs": [150],
        "max_trials": 1,
        "register_winner": True,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 3.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 4, "grad_accum_steps": 2,
            "img_size": 1024, "lr": 2e-4, "onecycle_pct_start": 0.2,
            "focal_gamma": 2.5, "focal_alpha": 0.25, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0, "aug_jitter_scale": 1.5,
            "box_loss_type": "giou", "caries_oversample": 2.0,
        },
        "search_space": {"base_seed": [42]},
    },
    "dinov3_final": {  # DINOv3: fusion + 150ep (clean compound of the two confirmed winners)
        # Highest-confidence DINOv3 final: fusion (the only lever that broke the
        # ceiling) + the long schedule that gave C-RADIO +0.028. Keeps smooth_l1
        # (proven for the fusion run) so this is a clean extension, not a gamble.
        "backbone": "dinov3_vitl16",
        "trial_epochs": 10,
        "schedule_epochs": [150],
        "max_trials": 1,
        "register_winner": True,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.1,
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
            "fusion_layers": [6, 12, 18, 24],
        },
        "search_space": {"base_seed": [42]},
    },
    "dinov3_final_giou": {  # DINOv3: fusion + 150ep + GIoU (the extra shot at 0.58+)
        "backbone": "dinov3_vitl16",
        "trial_epochs": 10,
        "schedule_epochs": [150],
        "max_trials": 1,
        "register_winner": True,
        "pinned": {
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.1,
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
            "fusion_layers": [6, 12, 18, 24], "box_loss_type": "giou",
        },
        "search_space": {"base_seed": [42]},
    },
}

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
print(f"CHAMPION_MODEL   = {CHAMPION_MODEL_NAME}")
print(f"CHAMPION_ENDPT   = {CHAMPION_ENDPOINT_NAME}")
print(f"TRAIN_EMB_TABLE  = {TRAIN_EMBEDDINGS_TABLE}")
print(f"VS_INDEX_NAME    = {VS_INDEX_NAME}")
