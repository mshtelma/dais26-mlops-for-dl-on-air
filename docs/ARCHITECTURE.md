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
| `spatial_features` | `(B, T, 1152)` | 1152 | FPN adapter, RetinaNet head, DetectorPyfunc |

Where `T = (H / patch_size) * (W / patch_size)`. For 1024×1024 input with `patch_size=16`: `T = 64*64 = 4096`.

**Critical facts:**
- `summary` and `spatial_features` are **distinct outputs** — never interchange them. They share the SO400M ViT hidden dim (1152) but `summary` is a separately pooled global feature, not a CLS token from the patch sequence.
- FPN `in_channels` must be **1152** (spatial dim).
- Vector Search index `embedding_dimension` must be **1152** (summary dim).
- Drift KNN reference must be built from **1152**-dim vectors.

### BackboneInfo dataclass

```python
@dataclass
class BackboneInfo:
    summary_dim: int        # C-RADIOv4: 1152  |  DINOv2-base: 768
    spatial_dim: int        # C-RADIOv4-SO400M: 1152  |  DINOv2-base: 768
    spatial_scale: int      # patch stride (16 for both)
    has_separate_summary: bool  # True for C-RADIOv4; False for DINOv2 (CLS token)
    patch_size: int
    model_name: str
    revision: str
```

All dimension-dependent code parameterizes on `backbone_info.summary_dim` and
`backbone_info.spatial_dim`. Hardcoding 1152 or 768 anywhere outside of `backbones.py` is a bug.

### Cascade through codebase

| Module | Uses |
|--------|------|
| `src/dais26_dentex/models/backbones.py` | Defines BackboneInfo; returns `(summary, spatial_features)` |
| `src/dais26_dentex/models/adapters.py` | `FPNAdapter(in_channels=backbone_info.spatial_dim)` → 1152 |
| `src/dais26_dentex/models/builder.py` | `build_detector(...)` — wraps backbone load in `rank0_first` to dodge the cold-cache HF race |
| `src/dais26_dentex/models/targets.py` | Anchor + target encoding; `FPNLevel` enum from `config/constants.py` |
| `src/dais26_dentex/serve/detector_pyfunc.py` | Uses `spatial_features` for detection (NMS + decode in `serve/postprocess.py`) |
| `src/dais26_dentex/serve/embedder_pyfunc.py` | Uses `summary`; output dim = `backbone_info.summary_dim` |
| `src/dais26_dentex/drift/embeddings.py` | Extracts `summary`, L2-normalizes |
| `notebooks/03_precompute_embeddings.py` | Writes `ARRAY<FLOAT>` of length `backbone_info.summary_dim` |
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
1. setup         (notebooks/00_setup.py, serverless notebook task)
   UC bootstrap: CREATE IF NOT EXISTS for all tables and volumes.
   UC grants for service principal. train_embeddings table with CDF enabled.
   |
   v
2. train         (notebooks/02_train_detector_air.py, serverless notebook task)
   Notebook calls serverless_gpu.@distributed → H100 pool.
   Inside the worker: configure_hf_env(...) → train_detector(...) →
     Trainer.run(): build_detector (rank0_first), DDP wrap (find_unused_parameters=True),
     _epoch_loop, _validate, _save_and_register (rank-0 only):
       MlflowReporter.log_pyfunc(
           python_model=serve/detector_model_script.py,   # models-from-code, NOT a pickled instance
           code_paths=[<dais26_dentex pkg dir>],           # bundle source for serving import
           pip_requirements=serving_pip_requirements())
       MlflowReporter.set_candidate_alias(@candidate)
   Returns run_id on rank 0; None on other ranks.
   |
   v
3. deploy_endpoint     (notebooks/04_deploy_serving.py, switches on DEPLOY_ACTION)
   action=register_and_set_candidate: verify @candidate exists, exit
   action=deploy_and_smoke_test:
     dais26_dentex.serve.endpoint_manager.deploy_and_smoke_test(...)
       Resolve @candidate -> numeric version N via MlflowClient.get_model_version_by_alias()
       Create/update endpoint with numeric version N (never an alias string)
       ai_gateway is a top-level sibling of config (NOT nested under config)
       Wait for READY state (900s timeout, 15s poll)
       Smoke test: 1 image -> 200 OK with detections
       On SUCCESS: promote @candidate -> @champion (capture previous_champion for rollback)
       On FAILURE: leave @candidate, do NOT touch @champion
```

The training core (`Trainer`) is identical across launch paths — `notebook @distributed`
or `sgcli` / `torchrun`. The CLI entry (`train.cli:main`) reads `$HYPERPARAMETERS_PATH`
or `--config`, builds the `TrainerConfig`, and dispatches to the same `Trainer.run()`.

### Phase 3: `databricks bundle run precompute_embeddings -t dev`

```
5. precompute    (notebooks/03_precompute_embeddings.py)
   Frozen-backbone forward pass over all 1005 DENTEX images
   Extract summary (C-RADIOv4=1152, DINOv3=1024), L2-normalize
   Write to <prefix>train_embeddings: ARRAY<FLOAT>, CDF enabled
   (VS index auto-synced only if EMBEDDINGS_VS_* are set in 00_config)
```

### Phase 3b: `databricks bundle run create_vector_search -t dev`

```
6. create_vector_search   (notebooks/04b_create_vector_search.py, no GPU)
   Create VS endpoint (dais26-vfm-vs) + DELTA_SYNC index (idempotent)
   embedding_dimension DERIVED from size(embedding) on the source table
     → correct for any backbone, no hardcoded dim
   Trigger sync → poll until ONLINE & fully synced → smoke-test similarity query
```

The `create_vector_search` branch in `04_deploy_serving.py` is kept in sync with this notebook;
`04b` is the always-on, `DEPLOY_ACTION`-independent version run by the job.

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

### Models-from-code serving load path

The detector is logged with `mlflow.pyfunc.log_model(python_model="…/detector_model_script.py", code_paths=[…])`
— a **script path**, not a `DetectorPyfunc()` instance. Two failure modes drove this:

- **`ModuleNotFoundError: transformers_modules`** — pickling the instance at log time captured a
  reference to the HuggingFace *dynamic* backbone class (created at runtime by `trust_remote_code=True`),
  which lives in the `transformers_modules.*` package. The serving container has no such package, so
  unpickling `python_model.pkl` failed. Models-from-code re-executes the script at load time and builds
  a fresh `DetectorPyfunc`; the backbone is materialized inside `load_context` and never serialized.
- **`ModuleNotFoundError: dais26_dentex`** — the pyfunc class lives in a locally-installed package
  (not on PyPI), so MLflow cannot pin it in `requirements.txt`. `code_paths` bundles the package into
  the model's `code/` dir; MLflow prepends it to `sys.path` at load time.

At load time the backbone is read strictly offline (`local_files_only=True`, offline HF env) from the
`model_cache` artifact bundled with the model — the serving container has no egress. `torch.compile`
is intentionally **disabled** at serving (the DINOv3 modeling stack raises `NameError: torch` under
TorchDynamo, and CUDA-graph `reduce-overhead` needs static shapes the variable-size image path can't
guarantee). See [RUNBOOK.md#models-from-code](RUNBOOK.md#models-from-code).

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
    └── spatial_features (B, 4096, 1152)  → used by FPN below
        │
        ▼ FPNAdapter (in_channels=1152, out_channels=256) — ~2.4M params
        │  1. Reshape tokens to (B, 1152, 64, 64) spatial grid
        │  2. 1×1 conv: 1152 → 256
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
src/dais26_dentex/
├── config/
│   ├── constants.py          (ARTIFACT_FORMAT_VERSION=2, MANIFEST_FILE,
│   │                          ALIAS_CANDIDATE/CHAMPION, FPNLevel StrEnum,
│   │                          HF env var name constants)
│   ├── manifest.py           (Manifest v2 dataclass: BackboneSpec + DetectorSpec
│   │                          → single manifest.json, version-first key order)
│   └── trainer_config.py     (TrainerConfig frozen dataclass; from_yaml/from_dict
│                              feeds notebook @distributed AND sgcli/torchrun)
│
├── data/
│   ├── dataset.py            (PyTorch Dataset over COCO JSON + UC Volume images)
│   ├── dentex_loader.py      (huggingface_hub → UC Volume; module-level
│   │                          `setdefault` for HF transfer/xet env vars)
│   └── transforms.py         (Albumentations, C-RADIOv4 normalization stats)
│
├── distributed/
│   ├── primitives.py         (setup_distributed, safe_barrier, seed_per_rank,
│   │                          world_size, is_rank0, unwrap_model;
│   │                          BarrierTimeoutError surfaces dead-rank deadlocks)
│   └── barrier_dance.py      (rank0_first contextmanager — sequence-matched
│                              NCCL barriers; used by models/builder.py)
│
├── eval/
│   └── coco_metrics.py       (pycocotools wrapper)
│
├── models/
│   ├── backbones.py          (BackboneInfo dataclass — single source of truth;
│   │                          C-RADIOv4 trust_remote_code dep guard)
│   │     ↑ consumed by everything below
│   ├── adapters.py           (FPNAdapter; in_channels=backbone_info.spatial_dim)
│   ├── builder.py            (build_detector wrapped in rank0_first)
│   ├── detection_head.py     (RetinaNetHead; input from FPNAdapter)
│   ├── targets.py            (anchor generator + target encoding; FPNLevel)
│   └── peft.py               (STRETCH: LoRA on backbone QKV+proj)
│
├── platform/
│   ├── hf_env.py             (configure_hf_env: HF_HOME, TRANSFORMERS_CACHE,
│   │                          HF_HUB_ENABLE_HF_TRANSFER=0, HF_HUB_DISABLE_XET=1
│   │                          — must be called BEFORE any HF import)
│   ├── mlflow_io.py          (MlflowReporter; serving_pip_requirements reads
│   │                          [tool.dais26.serving-deps] from pyproject.toml
│   │                          shipped inside the wheel; AliasingError;
│   │                          _log_model_artifact_kwarg picks name=/artifact_path=)
│   └── uc.py                 (UCName fqn, VolumePath child(); regex-validated
│                              identifiers — no inline f"{cat}.{sch}.{name}")
│
├── drift/
│   ├── embeddings.py         (uses backbones; extracts summary, L2-normalize)
│   ├── reference.py          (KNN/MMD fitting over summary embeddings)
│   ├── monitor.py            (orchestrates: reader → embeddings → reference → scores)
│   └── inference_table_reader.py (parses AI Gateway STRING request column)
│
├── serve/
│   ├── detector_pyfunc.py    (uses backbones + adapters + detection_head;
│   │                          loads Manifest v2; raises IncompatibleArtifactError
│   │                          on v1 artifacts; offline backbone load, no torch.compile)
│   ├── detector_model_script.py (models-from-code loader: set_model(DetectorPyfunc());
│   │                          logged as python_model instead of a pickled instance)
│   ├── embedder_pyfunc.py    (uses backbones; returns summary dim=backbone_info.summary_dim)
│   ├── postprocess.py        (NMS + decode + label remap — split out of pyfunc
│   │                          for unit-testability)
│   └── endpoint_manager.py   (SDK: create_and_wait, update_config_and_wait, smoke_test, promote)
│
└── train/
    ├── trainer.py            (Trainer class — owns DDP wrap, _epoch_loop,
    │                          _validate, _save_and_register; rank-0-only
    │                          MlflowReporter + UC registration)
    ├── train_detector.py     (thin shim: builds TrainerConfig, calls Trainer(cfg).run())
    ├── losses.py             (focal + smooth-L1)
    └── cli.py                (sgcli/torchrun entrypoint — reads
                               $HYPERPARAMETERS_PATH or --config; prints
                               MODEL_URI=<run_id> on rank 0)
```

### Cross-cutting hardening anchors

| Anchor | Module | Why |
|---|---|---|
| Manifest v2 | `config/manifest.py` | One `manifest.json` (version first → `head -1` triages a model) replaces v1's three sidecar JSONs. `load_manifest` raises `IncompatibleArtifactError` on v1 with a one-shot migration hint. |
| TrainerConfig | `config/trainer_config.py` | Frozen dataclass; `from_dict` / `from_yaml` / `validate`; same instance feeds notebook `@distributed` and sgcli's torchrun. |
| `safe_barrier` | `distributed/primitives.py` | Bounded `wait()` over `dist.barrier(async_op=True)` — surfaces `BarrierTimeoutError` instead of hanging on NCCL when a peer rank crashed. |
| `rank0_first` | `distributed/barrier_dance.py` | Non-rank-0 hits its barrier first, rank 0 does the work and then hits its barrier; trailing symmetric barrier. Pattern fixes the cold-cache HF download race. |
| `configure_hf_env` | `platform/hf_env.py` | One canonical site for `HF_HUB_ENABLE_HF_TRANSFER=0` + `HF_HUB_DISABLE_XET=1` — UC Volume FUSE rejects parallel chunked writes. |
| `serving_pip_requirements` | `platform/mlflow_io.py` ↔ `pyproject.toml::[tool.dais26.serving-deps]` | One edit to add a runtime dep. The wheel ships `pyproject.toml` as `dais26_dentex/_pyproject.toml` (hatchling `force-include`) so `importlib.resources` resolves it inside AIR's ephemeral env. CI guards via `assert_serving_reqs_match_pyproject`. |
| `_log_model_artifact_kwarg` | `platform/mlflow_io.py` | `name=` vs `artifact_path=` resolved once at import via `inspect.signature` — replaces the per-call `try/except TypeError`. |
| models-from-code + `code_paths` | `serve/detector_model_script.py` + `platform/mlflow_io.py::_default_code_paths` | Detector logged as a script (not a pickled instance) with the package source bundled. Fixes `ModuleNotFoundError: transformers_modules` (dynamic `trust_remote_code` class) and `ModuleNotFoundError: dais26_dentex` at serving. |
| `UCName` / `VolumePath` | `platform/uc.py` | Regex-validated identifiers; replaces inline `f"{catalog}.{schema}.{name}"` so dotted-catalog typos fail fast. |

---

## Unity Catalog resource map

Names are config-driven from `notebooks/00_config.py` (`CATALOG`, `SCHEMA`, `TABLE_PREFIX`, and the
backbone-keyed model/endpoint names). Current defaults: catalog `mlops_pj`, schema `dais26_vfm`,
table/index prefix `dais26_dentex_`, backbone `dinov3_vitl16`. The table below uses the legacy
`ml`/`cradio_detector` names for illustration.

| Resource type | Full name | Notes |
|---------------|-----------|-------|
| Schema | `<catalog>.dais26_vfm` | Created by `00_setup.py` |
| Volume | `…/dentex_raw` | Raw DENTEX images + COCO JSON |
| Volume | `…/model_cache` | Pinned C-RADIOv4 weights + pre-baked DINOv2 fallback |
| Delta table | `…/train_embeddings` | `ARRAY<FLOAT>` dim=1152, CDF=true |
| Delta table | `…/drift_scores` | Hourly drift job output |
| Delta table | `…/detector_inference_*` | Auto-created by AI Gateway on first request |
| Registered model | `…/cradio_detector` (or `…/dinov3_detector`) | Backbone-keyed name; aliases: `@champion`, `@candidate`, `@demo-frozen` |
| Registered model | `…/cradio_embedder` | Alias: `@champion` (STRETCH) |
| MLflow experiment | `/Users/<user>/dais26_vfm_experiment` | All training runs |
| VS index | `…/embeddings_index` | DELTA_SYNC, dim derived from the embeddings table (1152 / 1024) |
| Secret scope | `dais26-secrets` | Key: `hf-token` (DINOv3 only) |
