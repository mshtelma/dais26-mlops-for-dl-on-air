# Architecture

## System overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Unity Catalog: ml.dais26_vfm                                           │
│                                                                          │
│  Volumes                          Delta Tables                           │
│  ┌──────────────┐                 ┌──────────────────────────────────┐   │
│  │ dentex_raw   │──── loader ────>│ dentex_train / val / test        │   │
│  │ (705/50/250  │                 │ (image paths + COCO annotations) │   │
│  │  X-ray imgs) │                 └──────────────────────────────────┘   │
│  │              │                                                         │
│  │ model_cache  │──┐              ┌──────────────────────────────────┐   │
│  │ (C-RADIOv4   │  │              │ train_embeddings                 │   │
│  │  DINOv2 bkp) │  │              │ ARRAY<FLOAT> dim=1152, CDF=on   │   │
│  └──────────────┘  │              └──────────────┬───────────────────┘   │
│                     │                            │                        │
└─────────────────────┼────────────────────────────┼────────────────────────┘
                      │                            │
          ┌───────────┘                   Vector Search Sync
          │ frozen backbone                        │
          ▼                                        ▼
┌─────────────────┐                    ┌─────────────────────┐
│  AIR H100       │                    │  Mosaic AI          │
│  DAB Job:       │                    │  Vector Search      │
│  train_detector │                    │  embeddings_index   │
│                 │                    │  dim=1152, HNSW+L2  │
│  Task 1: setup  │                    └─────────────────────┘
│  Task 2: train  │
│  Task 3: reg    │──── @candidate ──>┌──────────────────────────┐
│  Task 4: deploy │                   │  UC Model Registry       │
└─────────────────┘                   │  ml.dais26_vfm.          │
                                      │  cradio_detector         │
┌─────────────────┐                   │  @candidate / @champion  │
│  AIR H100       │                   └──────────┬───────────────┘
│  DAB Job:       │──── summary ──>              │ numeric version
│  precompute_    │     dim=1152                 ▼
│  embeddings     │              ┌───────────────────────────────────────┐
└─────────────────┘              │  Mosaic AI Model Serving              │
                                 │  dais26-cradio-detector-{target}      │
┌─────────────────┐              │  GPU_SMALL, scale_to_zero=false(prod) │
│  A10G           │              │                                        │
│  DAB Job:       │              │  AI Gateway inference table:           │
│  drift_monitor  │◄─── reads ───│  ml.dais26_vfm.detector_inference_*  │
│  (scheduled     │  inference   │  (STRING request/response, TIMESTAMP) │
│   hourly)       │  table       └───────────────────────────────────────┘
│                 │
│  re-embeds via  │──── writes ──>┌──────────────────────────────┐
│  summary dim    │               │  drift_scores Delta table    │
│  1152           │               │  knn_distance, mmd_score,    │
└─────────────────┘               │  alert BOOLEAN               │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │  Lakehouse Monitoring        │
                                  │  (alerting layer on          │
                                  │   drift_scores table)        │
                                  └──────────────────────────────┘
```

---

## C-RADIOv4 BackboneInfo contract

**This is load-bearing. Every downstream module depends on it.**

NVIDIA C-RADIOv4-SO400M returns a **tuple of two tensors**, not a single output:

```python
summary, spatial_features = backbone(images)
# images: (B, 3, H, W)  e.g. (B, 3, 1024, 1024)
```

| Output | Shape | Dim | Used by |
|--------|-------|-----|---------|
| `summary` | `(B, 1152)` | 1152 | Embeddings, drift, Vector Search, EmbedderPyfunc |
| `spatial_features` | `(B, T, 1536)` | 1536 | FPN adapter, RetinaNet head, DetectorPyfunc |

Where `T = (H / patch_size) * (W / patch_size)`. For 1024×1024 input with `patch_size=16`: `T = 64*64 = 4096`.

**Critical facts:**
- `summary` dim (1152) and `spatial_features` dim (1536) are **different**. Never interchange them.
- `summary` is a separately pooled global feature — it is **not** a CLS token from the patch sequence.
- FPN `in_channels` must be **1536** (spatial dim).
- Vector Search index `embedding_dimension` must be **1152** (summary dim).
- Drift KNN reference must be built from **1152**-dim vectors.

### BackboneInfo dataclass

```python
@dataclass
class BackboneInfo:
    summary_dim: int        # C-RADIOv4: 1152  |  DINOv2-base: 768
    spatial_dim: int        # C-RADIOv4: 1536  |  DINOv2-base: 768
    spatial_scale: int      # patch stride (16 for both)
    has_separate_summary: bool  # True for C-RADIOv4; False for DINOv2 (CLS token)
    patch_size: int
    model_name: str
    revision: str
```

All dimension-dependent code parameterizes on `backbone_info.summary_dim` and
`backbone_info.spatial_dim`. Hardcoding 1152 or 1536 anywhere outside of `backbones.py` is a bug.

### Cascade through codebase

| Module | Uses |
|--------|------|
| `src/models/backbones.py` | Defines BackboneInfo; returns `(summary, spatial_features)` |
| `src/models/adapters.py` | `FPNAdapter(in_channels=backbone_info.spatial_dim)` → 1536 |
| `src/serve/detector_pyfunc.py` | Uses `spatial_features` for detection |
| `src/serve/embedder_pyfunc.py` | Uses `summary`; output dim = `backbone_info.summary_dim` |
| `src/train/precompute_embeddings.py` | Writes `ARRAY<FLOAT>` of length `backbone_info.summary_dim` |
| `src/drift/embeddings.py` | Extracts `summary`, L2-normalizes |
| Vector Search index | `embedding_dimension = backbone_info.summary_dim` |
| Drift reference | Built from `backbone_info.summary_dim`-dim vectors |

### DINOv2-base fallback contract (different dimensions)

| Output | Shape | Extraction |
|--------|-------|-----------|
| `summary` (CLS token) | `(B, 768)` | `output[:, 0, :]` |
| `spatial_features` | `(B, T, 768)` | `output[:, 1:, :]` |

The fallback requires rebuilding all dimension-dependent artifacts. See [RUNBOOK.md](RUNBOOK.md#dinov2-fallback).

---

## Two-phase deployment

### Why not YAML-deployed endpoints

`databricks bundle deploy` cannot create a serving endpoint that references a model alias before any
model version is registered. Attempting `entity_version: "@champion"` in a YAML resource on first
deploy fails because `@champion` does not exist yet.

The solution: endpoints are **SDK-driven**, created by the `deploy_endpoint` task inside the
`train_detector` job, gated on training completion.

### Phase 1: `databricks bundle deploy -t dev`

Deploys:
- UC catalog, schema, volumes (`dentex_raw`, `model_cache`)
- MLflow experiment
- Job definitions (train_detector, precompute_embeddings, drift_monitor)
- Secret scope `dais26-secrets`

Does **not** deploy:
- Serving endpoints (no `resources/serving/*.yml`; intentionally excluded from `databricks.yml`)
- Model versions (created by training job)
- Vector Search indexes (created by embedding job)

### Phase 2: `databricks bundle run train_detector -t dev`

```
1. setup         (notebooks/00_setup.py)
   UC bootstrap: CREATE IF NOT EXISTS for all tables and volumes.
   UC grants for service principal. train_embeddings table with CDF enabled.
   |
   v
2. train         (notebooks/02_train_detector_air.py)
   Train FPN + RetinaNet head on DENTEX (10 epochs, frozen backbone).
   Log to MLflow with signature + input_example.
   |
   v
3. register_and_alias  (notebooks/04_deploy_serving.py, action=register_and_set_candidate)
   Register model version N in UC.
   Set @candidate alias on version N.
   |
   v
4. deploy_endpoint     (notebooks/04_deploy_serving.py, action=deploy_and_smoke_test)
   Resolve @candidate -> numeric version N via MlflowClient.get_model_version_by_alias()
   Create/update endpoint with numeric version N (never an alias string)
   ai_gateway is a top-level sibling of config (NOT nested under config)
   Wait for READY state (600s timeout, 15s poll)
   Smoke test: 1 image -> 200 OK with detections
   On SUCCESS: promote @candidate -> @champion
   On FAILURE: leave @candidate, do NOT touch @champion
```

### Phase 3: `databricks bundle run precompute_embeddings -t dev`

```
5. precompute    (notebooks/03_precompute_embeddings.py)
   C-RADIOv4 forward pass over all 1005 DENTEX images
   Extract summary (dim 1152), L2-normalize
   Write to train_embeddings: ARRAY<FLOAT>, CDF enabled
   Create/sync Vector Search index (embedding_dimension=1152)
```

**Key invariant:** Endpoints are never created before a model version exists.

---

## @candidate to @champion promotion flow

```
Training job task 3: register_and_alias
  │
  │ mlflow.pyfunc.log_model(registered_model_name=...)
  │ → Model version N created in UC
  │
  │ MlflowClient.set_registered_model_alias(alias="candidate", version=N)
  │ → @candidate = version N
  │
  ▼
Training job task 4: deploy_endpoint
  │
  │ MlflowClient.get_model_version_by_alias(alias="candidate")
  │ → candidate_version = "N"   (numeric string)
  │
  │ WorkspaceClient.serving_endpoints.create_and_wait(  ← new endpoint
  │     name="dais26-cradio-detector-dev",
  │     config=EndpointCoreConfigInput(
  │         served_entities=[ServedEntityInput(
  │             entity_name="ml.dais26_vfm.cradio_detector",
  │             entity_version="N",          ← numeric, never "@champion"
  │             workload_type="GPU_SMALL",
  │             scale_to_zero_enabled=True,  ← dev only
  │         )]
  │     )
  │ )
  │   OR for an existing endpoint:
  │ WorkspaceClient.serving_endpoints.update_config_and_wait(...)
  │
  │ Poll endpoint.state.ready == "READY"
  │
  │ WorkspaceClient.serving_endpoints.query(
  │     name="dais26-cradio-detector-dev",
  │     dataframe_split={"columns": ["image"], "data": [[img_b64]]}
  │ )
  │ → assert predictions is not None
  │
  │ ON SUCCESS:
  │ MlflowClient.set_registered_model_alias(alias="champion", version=N)
  │ → @champion = version N
  │
  │ ON FAILURE:
  │ @candidate stays on version N
  │ @champion untouched (previous version still serves)
  ▼
  Done
```

### SDK method note

`WorkspaceClient.serving_endpoints.create_or_update()` does not exist in the Databricks SDK.
Use the correct methods:
- **New endpoint**: `serving_endpoints.create_and_wait(name, config, ai_gateway, tags)`
- **Existing endpoint**: `serving_endpoints.update_config_and_wait(name, served_entities)`

### ai_gateway placement

`ai_gateway` is a **top-level argument** of `create_and_wait`, not nested under `config`:

```python
# CORRECT
w.serving_endpoints.create_and_wait(
    name=endpoint_name,
    config=EndpointCoreConfigInput(...),
    ai_gateway=AiGatewayConfig(           # top-level sibling of config
        inference_table_config=AiGatewayInferenceTableConfig(
            catalog_name=catalog,
            schema_name=schema,
            table_name_prefix="detector_inference",
            enabled=True,
        )
    ),
)

# WRONG — ai_gateway nested under config causes silent failure
w.serving_endpoints.create_and_wait(
    name=endpoint_name,
    config=EndpointCoreConfigInput(
        ai_gateway=...,   # ← wrong location
        ...
    ),
)
```

---

## Reference endpoint configuration (documentation only)

This YAML shows the target endpoint state. It is **not** in `resources/` and is **not** deployed by
`databricks bundle deploy`.

```yaml
# docs/ARCHITECTURE.md reference — SDK equivalent configuration
# Deployed programmatically by notebooks/04_deploy_serving.py
name: "dais26-cradio-detector-{target}"
config:
  served_entities:
    - name: "detector"
      entity_name: "{catalog}.dais26_vfm.cradio_detector"
      entity_version: "<numeric_version>"   # resolved from @champion via SDK
      workload_size: "Small"
      workload_type: "GPU_SMALL"
      scale_to_zero_enabled: false           # false for prod; true for dev
ai_gateway:                                  # top-level sibling of config
  inference_table_config:
    catalog_name: "{catalog}"
    schema_name: "dais26_vfm"
    table_name_prefix: "detector_inference"
    enabled: true
tags:
  - key: project
    value: dais26-vfm
  - key: component
    value: detection
```

---

## Detection pipeline: FPN + RetinaNet

```
Input image (B, 3, 1024, 1024)
    │
    ▼ C-RADIOv4-SO400M (frozen, 412M params)
    ├── summary         (B, 1152)  → used by EmbedderPyfunc, drift, VS
    └── spatial_features (B, 4096, 1536)  → used by FPN below
        │
        ▼ FPNAdapter (in_channels=1536, out_channels=256) — ~2.4M params
        │  1. Reshape tokens to (B, 1536, 64, 64) spatial grid
        │  2. 1×1 conv: 1536 → 256
        │  3. Bilinear 2× upsample → P3 (B, 256, 128, 128)
        │  4. Identity → P4 (B, 256, 64, 64)
        │  5. Stride-2 conv → P5 (B, 256, 32, 32)
        │  6. Stride-2 conv → P6 (B, 256, 16, 16)
        │
        ▼ RetinaNetHead — ~2.8M params
           4 conv layers per subnet (cls + reg), 9 anchors/location
           Focal loss (alpha=0.25, gamma=2.0) + Smooth L1
           NMS threshold=0.5, score threshold=0.05, max_dets=100
           │
           ▼
           {'boxes': [[x1,y1,x2,y2],...], 'scores': [...], 'labels': [...]}
```

Anchor scales tuned for DENTEX: `[16, 32, 64, 128]` px (smaller than COCO defaults).
Ratios: `[0.5, 1.0, 2.0]`. Classes: Caries, Deep Caries, Periapical Lesion, Impacted.

---

## Drift monitoring architecture

The drift monitor tracks **detector** traffic, not embedder traffic, because the detector is the
production decision-maker.

```
AI Gateway inference table
ml.dais26_vfm.detector_inference_*
  request STRING (JSON),  response STRING,  request_time TIMESTAMP
     │
     │ inference_table_reader.py
     │ Parses STRING request column (NOT typed JSON)
     │ Handles: dataframe_split, dataframe_records formats
     │ Skips: NULL rows (>1 MiB payload cap)
     ▼
  Raw images (base64 decoded)
     │
     │ drift/embeddings.py
     │ Frozen C-RADIOv4 backbone, summary output only
     │ L2-normalize → (N, 1152) float32 array
     ▼
  New embeddings
     │
     │ drift/reference.py
     │ Compare against train_embeddings table (705 train images)
     │ KNN distance (k=50), MMD score
     ▼
  drift_scores Delta table
  knn_distance DOUBLE, mmd_score DOUBLE, alert BOOLEAN
     │
     ▼
  Lakehouse Monitoring (alerting layer)
  Tracks drift_scores table; raises alert if knn_distance > 2.0× baseline
```

Zero added latency to detection requests — drift computation runs on a separate hourly job.

---

## Serving alternatives comparison

| Capability | Mosaic AI Model Serving | Triton Inference Server | BentoML / KServe |
|------------|------------------------|------------------------|-----------------|
| UC model lineage | Yes, via MLflow UC registry | No | No |
| AI Gateway inference tables | Yes | No | No |
| GPU auto-provisioning | Yes (`workload_type=GPU_SMALL`) | Manual container | Manual |
| Scale to zero | Yes | Manual | Varies |
| Databricks-native auth | Yes (PAT / OAuth M2M) | Custom | Custom |
| Alias-based promotion | Yes (`@champion`, `@candidate`) | No | No |
| Custom container required | No | Yes | Yes |
| Audience reproducibility | `databricks bundle run` | Multi-step container build | External toolchain |

Triton, BentoML, and KServe are listed for the comparison slide only. Mosaic AI Model Serving is
the only path supported by this repo.

---

## Code module dependency graph

```
src/
├── data/
│   ├── dentex_loader.py      (huggingface_hub → UC Volume)
│   └── transforms.py         (Albumentations, C-RADIOv4 normalization stats)
│
├── models/
│   ├── backbones.py          (BackboneInfo dataclass — single source of truth)
│   │     ↑ consumed by everything below
│   ├── adapters.py           (FPNAdapter; in_channels=backbone_info.spatial_dim)
│   ├── detection_head.py     (RetinaNetHead; input from FPNAdapter)
│   └── peft.py               (STRETCH: LoRA on backbone QKV+proj)
│
├── train/
│   ├── train_detector.py     (uses backbones + adapters + detection_head + eval)
│   └── precompute_embeddings.py (uses backbones; writes summary to Delta)
│
├── eval/
│   └── coco_metrics.py       (pycocotools wrapper)
│
├── drift/
│   ├── embeddings.py         (uses backbones; extracts summary, L2-normalize)
│   ├── reference.py          (KNN/MMD fitting over summary embeddings)
│   ├── monitor.py            (orchestrates: reader → embeddings → reference → scores)
│   └── inference_table_reader.py (parses AI Gateway STRING request column)
│
└── serve/
    ├── detector_pyfunc.py    (uses backbones + adapters + detection_head)
    ├── embedder_pyfunc.py    (uses backbones; returns summary dim=backbone_info.summary_dim)
    └── endpoint_manager.py   (SDK: create_and_wait, update_config_and_wait, smoke_test, promote)
```

---

## Unity Catalog resource map

| Resource type | Full name | Notes |
|---------------|-----------|-------|
| Schema | `ml.dais26_vfm` (prod) / `ml_dev.dais26_vfm` (dev) | Created by `00_setup.py` |
| Volume | `…/dentex_raw` | Raw DENTEX images + COCO JSON |
| Volume | `…/model_cache` | Pinned C-RADIOv4 weights + pre-baked DINOv2 fallback |
| Delta table | `…/train_embeddings` | `ARRAY<FLOAT>` dim=1152, CDF=true |
| Delta table | `…/drift_scores` | Hourly drift job output |
| Delta table | `…/detector_inference_*` | Auto-created by AI Gateway on first request |
| Registered model | `…/cradio_detector` | Aliases: `@champion`, `@candidate`, `@demo-frozen` |
| Registered model | `…/cradio_embedder` | Alias: `@champion` (STRETCH) |
| MLflow experiment | `/Users/<user>/dais26_vfm_experiment` | All training runs |
| VS index | `…/embeddings_index` | HNSW+L2, dim=1152, Delta Sync |
| Secret scope | `dais26-secrets` | Key: `hf-token` (DINOv3 only) |
