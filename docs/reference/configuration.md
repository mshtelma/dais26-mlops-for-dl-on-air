# Configuration reference

The config layer is three named axes (recipe / environment / campaign stage) over one schema
(`TrainerConfig`). Conceptual overview: [Named configuration](../lanes/configuration.md). This
page is the field-level reference.

## `TrainerConfig` fields

The frozen dataclass in
[`config/trainer_config.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/config/trainer_config.py).
**Defaults are the *legacy* values** (absolute anchors, class-agnostic NMS, frozen backbone) so
historical runs stay byte-identical; the winning values live in [recipes](#recipes), not here.
`from_dict` coerces types and `validate()` raises on bad combinations.

### Required

| Field | Type | Notes |
|-------|------|-------|
| `catalog` | str | UC catalog (no `.`) |
| `schema` | str | UC schema (no `.`) |

### Backbone

| Field | Default | Notes |
|-------|---------|-------|
| `backbone_name` | `cradio_v4_so400m` | one of `cradio_v4_so400m` / `dinov3_vitl16` / `dinov2_base` (HF ids alias via `BACKBONE_ALIASES`) |
| `backbone_revision` | `None` | pin a 40-char HF SHA for reproducibility |
| `cache_dir` | `None` | UC Volume model cache |
| `fusion_layers` | `None` | multi-layer ViT fusion (e.g. `[6,12,18,24]`); **DINOv3 only** |

### Data & schedule

| Field | Default | Notes |
|-------|---------|-------|
| `img_size` | `1024` | multiple of 16 |
| `epochs` | `10` | recipes set 150; quickstart overrides to 50 |
| `lr` | `1e-3` | head LR (recipes use `2e-4`) |
| `weight_decay` | `1e-4` | recipes use `1e-2` |
| `grad_clip_norm` | `1.0` | |
| `onecycle_pct_start` | `0.1` | OneCycle warmup fraction |
| `batch_size` | `8` | per-GPU |
| `num_workers` | `4` | |
| `grad_accum_steps` | `1` | effective batch = `batch_size × grad_accum_steps × world_size` |
| `base_seed` | `42` | |

### Augmentation

| Field | Default | Notes |
|-------|---------|-------|
| `aug_hflip_prob` | `0.5` | |
| `aug_jitter_prob` | `0.5` | colour jitter probability |
| `aug_jitter_scale` | `1.0` | multiplier on jitter magnitudes |
| `aug_rotation_deg` | `0.0` | ±deg (X-rays ~axis-aligned; keep ≤10) |
| `aug_multiscale_range` | `None` | `[lo,hi]` in (0,1]; down-scale + pad to `img_size` |
| `caries_oversample` | `1.0` | replicate Caries-bearing images (hard class) ≥1.0 |

### Mixed precision

| Field | Default | Notes |
|-------|---------|-------|
| `amp_dtype` | `auto` | `auto`→ fp32 for DINOv3 (NaNs under fp16 **and** bf16), fp16 otherwise; or `fp16`/`bf16`/`fp32` |
| `flat_loss_patience` | `0` | abort if `train/loss` hasn't dropped after N epochs (0 = off) |

### Loss

| Field | Default | Notes |
|-------|---------|-------|
| `focal_alpha` | `0.25` | |
| `focal_gamma` | `2.0` | recipes use 2.0–2.5 |
| `box_loss_weight` | `1.0` | |
| `box_loss_type` | `smooth_l1` | or `giou` (scale-invariant; better 50:95) |

### Detection / anchors

| Field | Default | Notes |
|-------|---------|-------|
| `num_classes` | `None` | defers to `len(get_label_map())` (4) |
| `anchor_layout` | `absolute` | `per_level` = RetinaNet stride-scaled (recipes use this) |
| `anchor_scales` | `None` | absolute mode only |
| `aspect_ratios` | `None` | defaults `[0.5,1,2]` |
| `anchor_base_scale` | `4.0` | per_level base (recipes: C-RADIO 3.0, DINOv3 4.0) |
| `anchor_octaves` | `None` | defaults `{2^0, 2^(1/3), 2^(2/3)}` |
| `nms_per_class` | `False` | recipes set `True` (`batched_nms` by label) |
| `score_threshold` | `0.05` | eval/serve; gridable post-hoc |
| `nms_iou_threshold` | `0.5` | eval/serve |
| `max_detections` | `100` | eval/serve |

### Backbone adaptation, PEFT, DDP, MLflow

| Field | Default | Notes |
|-------|---------|-------|
| `backbone_mode` | `frozen` | `frozen`/`lora`/`full`/`partial` (recipes: `full`) |
| `backbone_lr` | `1e-5` | discriminative LR; keep 10–100× below `lr` |
| `backbone_trainable_blocks` | `0` | for `partial` |
| `use_lora` / `lora_rank` / `lora_alpha` | `False` / `8` / `32.0` | legacy LoRA path (maps to `backbone_mode=lora`) |
| `ddp_find_unused` | `True` | `False` only for `full` (see [`#ddp-trainable-only`](../RUNBOOK.md#ddp-trainable-only)) |
| `barrier_timeout_seconds` | `600.0` | `safe_barrier` deadline → `BarrierTimeoutError` |
| `experiment_name` | `None` | from the env |
| `model_name` | `cradio_detector` | dev model short name |
| `register_model` / `set_candidate_alias` | `True` / `True` | register + set `@challenger` |
| `resume_from_checkpoint` | `None` | |

!!! note "`effective_backbone_mode()` / `effective_amp_dtype()`"
    `use_lora=True` maps to `backbone_mode="lora"` only when mode is still the `frozen` default. For
    DINOv3, `amp_dtype="auto"` resolves to **fp32** (fp16 and bf16 both NaN its RoPE/LayerScale
    encoder — see [HPO → DINOv3 A/B](../HPO.md)).

## Recipes

Best-known overrides per backbone, in
[`config/recipes.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/config/recipes.py).
`build_trainer_config(backbone, *, catalog, schema, …, **overrides)` merges **recipe → env kwargs
→ overrides** (last wins).

| Backbone | Provenance (best run) | val mAP@50 | Notable settings |
|----------|----------------------|-----------|------------------|
| `cradio_v4_so400m` | `dazzling-mole-850` (150ep) | **0.5931** | full FT, per_level/base_scale 3.0, lr 2e-4, amp→fp16, bs 4×accum 2, 1024px, smooth_l1 |
| `dinov3_vitl16` | `capricious-hound-240` v7 (fusion×150ep) | **0.5738** | full FT, fusion [6,12,18,24], base_scale 4.0, amp→fp32, bs 2×accum 2, **1280px** |
| `dinov2_base` | emergency fallback (untuned) | — | frozen head, per_level, amp→fp16, bs 8, 1024px, 50ep |

`DETECTOR_NAMES_BY_BACKBONE` maps each backbone to its dev model short name + dev endpoint
(`cradio_v4_so400m → cradio_detector / dais26-cradio-detector-dev`).

## Environments

`EnvSpec` in
[`config/environments.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/config/environments.py).
Required per entry: `catalog`, `schema`, `experiment_name`. Derived: `champion_catalog`
(=catalog), `champion_schema` (=`<schema>_prod`), `volume_path`, `cache_dir`.

| Env | catalog | schema | champion_schema |
|-----|---------|--------|-----------------|
| `df1` (default) | `main` | `mshtelma` | `mshtelma` |
| `prod` | `mlops_pj` | `dais26_vfm` | `dais26_vfm_prod` |

Precedence + overlay + `$DAIS26_*` vars: [Per-user environment overrides](../scenarios/env-overrides.md).

## `00_config.py` notebook knobs

Not hyperparameters — environment selection + per-notebook launch knobs:

| Knob | Default | Used by |
|------|---------|---------|
| `ENV` | `df1` | selects the named environment |
| `BACKBONE` / `BACKBONE_REVISION` | `cradio_v4_so400m` / `main` | backbone + revision |
| `TABLE_PREFIX` | `dais26_dentex_` | table/index prefix |
| `TRAIN_EPOCHS` | `50` | demo override of the recipe's 150 |
| `TRAIN_GPUS` / `TRAIN_GPU_TYPE` | `8` / `h100` | `@distributed` |
| `TRAIN_USE_LORA` / `TRAIN_LORA_RANK` / `TRAIN_LORA_ALPHA` | `False` / `8` / `32.0` | stretch LoRA path |
| `SWEEP_STAGE` | `None` | campaign stage (also `sweep_stage` widget) |
| `EMBEDDINGS_BATCH_SIZE` | `32` | embeddings |
| `EMBEDDINGS_VS_ENDPOINT` / `EMBEDDINGS_VS_INDEX` | `None` | auto-sync VS after embeddings |
| `DEPLOY_ACTION` | `deploy_and_smoke_test` | `04_deploy_serving` mode |
| `DEPLOY_WORKLOAD_TYPE` / `DEPLOY_WORKLOAD_SIZE` / `DEPLOY_SCALE_TO_ZERO` / `DEPLOY_TIMEOUT_SECONDS` | `GPU_SMALL` / `Small` / `True` / `5400` | endpoint sizing |
| `DRIFT_MODE` / `DRIFT_KNN_K` / `DRIFT_ALERT_THRESHOLD` | `demo` / `50` / `2.0` | drift |
| `EXPLORE_SPLIT` | `train` | `01_explore_dentex` |
| `SIMILARITY_QUERY_COUNT` | `50` | `06_similarity_search_demo` |
| `LATENCY_NUM_REQUESTS` / `LATENCY_WARMUP_REQUESTS` / `LATENCY_PIVOT_THRESHOLD_MS` | `1000` / `20` / `150.0` | `07_latency_benchmark` |
| `SP_APP_ID` / `HF_TOKEN` | `None` / `None` | `00_setup` grants / gated backbone |

## Key constants (`config/constants.py`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `ARTIFACT_FORMAT_VERSION` | `2` | single `manifest.json` (v1 sidecars no longer loadable) |
| `ALIAS_CANDIDATE` | `challenger` | dev alias (name kept; value moved from `candidate`) |
| `ALIAS_CHAMPION` | `champion` | prod live alias |
| `ALIAS_CHAMPION_CANDIDATE` | `champion_candidate` | prod staging alias |
| `FPN_LEVELS` | P3–P6 | FPN feature-pyramid levels |
| `HF_ENV_*` | — | HuggingFace env-var names set by `platform.hf_env` |

## Serving dependencies (`[tool.dais26.serving-deps]`)

The single source of truth for the pyfunc's `pip_requirements`, read at log-time by
`serving_pip_requirements`. `torch==2.6.0` / `torchvision==0.21.0` (cu124) and
`transformers==4.56.2` are **exact pins** — see
[torch/torchvision cu124 pin](../RUNBOOK.md#torch-cu124-pin) and
[`#pip-requirements-rationale`](../RUNBOOK.md#pip-requirements-rationale).
