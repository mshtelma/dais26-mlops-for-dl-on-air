# DAIS26 MLOps for AI Runtime with Vision Foundation Models

**One frozen backbone, three jobs: detection head, drift sensor, embedding service.**

Databricks-native showcase for the Data + AI Summit 2026 (June 15-18, San Francisco).
Backbone is selectable in `notebooks/00_config.py` (`BACKBONE`): NVIDIA C-RADIOv4-SO400M
(ungated, commercial-OK, `summary`/`spatial` dim 1152) or Meta DINOv3-ViTL16 (gated,
comparison, dim 1024). DINOv2-base (dim 768) is the emergency fallback. All dimension-dependent
code parameterizes on `BackboneInfo` — nothing downstream hardcodes a dimension.
Dataset: DENTEX dental X-rays (CC-BY-NC-SA 4.0, research/demo only).
Compute: Databricks AI Runtime (AIR) / Serverless GPU only — no traditional ML clusters anywhere.
Training launch: notebook `@distributed` or `sgcli` (terminal). Both share one training core.
Serving: Mosaic AI Model Serving GPU endpoints, SDK-driven.

![CI](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/actions/workflows/ci.yml/badge.svg)

## What this repo demonstrates

| Job | Backbone output used | Purpose |
|-----|----------------------|---------|
| `train_detector` | `spatial_features` | FPN + RetinaNet detection head on DENTEX |
| `precompute_embeddings` | `summary` | Delta table + Mosaic AI Vector Search index |
| `drift_monitor` | `summary` | KNN-distance drift detection on detector traffic |

Feature dim is the backbone's `summary`/`spatial` dim (C-RADIOv4-SO400M: 1152, DINOv3-ViTL16: 1024)
and flows from `BackboneInfo` — the Vector Search index dimension is even derived from the
embeddings table at index-creation time, so switching `BACKBONE` needs no doc/code edits downstream.

One frozen backbone artifact in a UC Volume. Three downstream consumers. Zero backbone gradients.

## Quick start

```bash
git clone <repo> && cd dais26-mlops-for-dl-on-air
pip install uv && uv pip install -e ".[dev]"
databricks auth login --host <DATABRICKS_HOST>
uv build                                       # ships pyproject.toml inside the wheel for serving-deps lookup
databricks bundle deploy -t dev               # deploys UC + jobs (NOT endpoints)
databricks bundle run train_detector -t dev   # trains + @candidate + endpoint via SDK + @champion
databricks bundle run precompute_embeddings -t dev
```

Full 11-step sequence with smoke tests and prerequisites: [docs/README.md](docs/README.md)

## Two launch paths for training — **AIR-only** (no traditional ML clusters)

All training runs on Databricks AI Runtime / Serverless GPU Compute. The DAB job
itself uses **serverless notebook tasks**; the training notebook dispatches the
actual GPU work via `serverless_gpu.@distributed` to the H100 pool. A second
launch surface — `sgcli` — submits the same training core directly from a terminal.

### Path A — Notebook (`@distributed`)

Run `notebooks/02_train_detector_air.py` interactively, or let the DAB job
trigger it. The decorator dispatches to the serverless GPU pool:

```python
from serverless_gpu import distributed
from dais26_dentex.train.train_detector import train_detector

@distributed(gpus=8, gpu_type="h100")
def run_train():
    return train_detector(catalog=..., schema=..., epochs=10, ...)

results = run_train.distributed()
run_id  = next((r for r in results if r), None)   # rank-0 only returns a value
```

### Path B — sgcli (terminal)

```bash
# One-time:
uv tool install --python 3.12 /path/to/databricks_serverless_gpu_cli-<v>.whl
databricks auth login --host <DATABRICKS_HOST>

# Launch:
sgcli run -f sgcli/workload_train_detector.yaml --watch -p dev
sgcli get runs --limit 10 -p dev
sgcli get logs <run-id> --rank 0 -p dev
```

Both paths share **one core** — `src/dais26_dentex/train/train_detector.py` — which is
distributed-aware (DistributedDataParallel with `find_unused_parameters=True`
for the frozen backbone, rank-0-only MLflow + UC registration). See
[sgcli/README.md](sgcli/README.md) for the terminal flow.

## Deployment model

Two phases — `bundle deploy` handles infrastructure; endpoints are SDK-driven.

```
bundle deploy -t dev
  |-- UC catalog / schema / volumes / secret scope
  |-- MLflow experiment
  |-- Job definitions (train_detector, precompute_embeddings, drift_monitor)
  `-- (NO serving endpoints)

bundle run train_detector -t dev
  setup --> train --> register @candidate --> deploy endpoint (SDK) --> smoke test --> @champion
```

Endpoints are never created before a model version exists. The detector pyfunc is logged via
**MLflow models-from-code** (`serve/detector_model_script.py`) with the package source bundled via
`code_paths` — pickling the instance captured a `trust_remote_code` `transformers_modules.*`
backbone reference the serving container cannot import. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Three standalone jobs let you re-run a single phase without the full `train_detector` chain:

| Job | Notebook | Purpose |
|-----|----------|---------|
| `deploy_endpoint` | `04_deploy_serving.py` | (Re)deploy `@candidate` → endpoint → smoke test → `@champion` |
| `create_vector_search` | `04b_create_vector_search.py` | Create VS endpoint + DELTA_SYNC index over the embeddings table, sync, smoke-test query |
| `precompute_embeddings` | `03_precompute_embeddings.py` | Write the `summary`-embedding Delta table (prerequisite for `create_vector_search`) |

## Repo structure

```
dais26-mlops-for-dl-on-air/
|-- databricks.yml          # DAB root manifest (UC + job resources; no endpoint YAML)
|-- pyproject.toml          # Python deps + [tool.dais26.serving-deps] SoT for pyfunc
|-- Makefile                # Developer shortcuts
|-- src/
|   `-- dais26_dentex/      # Importable Python package (src layout)
|       |-- config/         # constants, manifest v2, TrainerConfig dataclass
|       |-- data/           # dentex_loader.py, dataset.py, transforms.py
|       |-- distributed/    # primitives (setup/safe_barrier/seed), barrier_dance (rank0_first)
|       |-- drift/          # embeddings.py, reference.py, monitor.py, inference_table_reader.py
|       |-- eval/           # coco_metrics.py
|       |-- models/         # backbones (BackboneInfo), adapters (FPN), detection_head, builder, targets
|       |-- platform/       # hf_env, mlflow_io (MlflowReporter, serving_pip_requirements), uc (UCName)
|       |-- serve/          # detector_pyfunc.py, detector_model_script.py (models-from-code loader), embedder_pyfunc.py, postprocess.py, endpoint_manager.py
|       `-- train/          # trainer.py (Trainer class), losses, train_detector (thin shim), cli (sgcli entry)
|-- notebooks/              # 00_config / 00_setup .. 04b_create_vector_search .. 07_latency_benchmark, 08_backbone_comparison, 09_eval_comparison; widget-free, params via 00_config.py
|-- resources/              # DAB resource YAML (jobs/, experiments/; NO serving/)
|-- sgcli/                  # Serverless GPU CLI workload (terminal launch path)
|-- scripts/                # discover_air_runtime.py, warmup_endpoints.py, pin_model_cache.py, ...
|-- tests/                  # unit/ and integration/ pytest suites
`-- docs/                   # ARCHITECTURE, RUNBOOK, BENCHMARKS, TALK
```

## Engineering anchors (Phase 4 hardening)

| Concern | Where it lives | Why |
|---|---|---|
| Artifact contract | `config/manifest.py` (v2 `manifest.json`) | Single file replaces v1's three sidecar JSONs; `version` first → `head -1` triages a model. |
| Trainer hyperparameters | `config/trainer_config.py` (`TrainerConfig`) | Single dataclass; same instance feeds the notebook `@distributed` path and the sgcli/torchrun YAML. |
| Distributed primitives | `distributed/primitives.py` + `distributed/barrier_dance.py` | `safe_barrier` surfaces dead-rank deadlocks as `BarrierTimeoutError` instead of hanging on NCCL. `rank0_first` is sequence-matched and avoids the cold-cache HF download race. |
| HF env hardening | `platform/hf_env.py::configure_hf_env` | One canonical site for `HF_HUB_ENABLE_HF_TRANSFER=0` + `HF_HUB_DISABLE_XET=1` (UC Volume FUSE rejects parallel chunked writes). |
| Pyfunc serving deps | `pyproject.toml::[tool.dais26.serving-deps]` ↔ `platform/mlflow_io.py::serving_pip_requirements` | One edit to add a runtime dep; the wheel ships `pyproject.toml` as `dais26_dentex/_pyproject.toml` so the lookup works in AIR's ephemeral env. CI guards via `assert_serving_reqs_match_pyproject`. `torch`/`torchvision` are pinned to the cu124 build (`2.6.0` / `0.21.0`) so GPU_SMALL (T4, driver CUDA 12.4) does not silently fall back to CPU. |
| Pyfunc serving load path | `serve/detector_model_script.py` + `platform/mlflow_io.py::_default_code_paths` | Detector logged via **models-from-code** (script, not pickled instance) with the package bundled via `code_paths`. Avoids `ModuleNotFoundError: transformers_modules` (dynamic `trust_remote_code` class) and `ModuleNotFoundError: dais26_dentex` at serving. Backbone loads strictly offline (`local_files_only`) from the bundled HF cache; `torch.compile` is disabled at serving. |
| MLflow API drift | `platform/mlflow_io.py::_log_model_artifact_kwarg` | `name=` vs `artifact_path=` resolved once at import via `inspect.signature`. |
| UC identifiers | `platform/uc.py::UCName`, `VolumePath` | Stop hand-rolling `f"{catalog}.{schema}.{name}"`; UC ident regex catches dotted-catalog typos. |
| Notebook params | `notebooks/00_config.py` | All params live there; **no `dbutils.widgets`** and no DAB `base_parameters`. |

Full rationale (race traces, alternatives considered, follow-ups) in
[docs/RUNBOOK.md#engineering-rationale](docs/RUNBOOK.md#engineering-rationale).

## Licenses

| Asset | License | Notes |
|-------|---------|-------|
| Code in this repo | Apache-2.0 | See [LICENSE](LICENSE) |
| DENTEX dataset | CC-BY-NC-SA 4.0 | Research and demo only. No commercial use. |
| C-RADIOv4-SO400M weights | NVIDIA Open Model License | Commercial use permitted. Ungated on HuggingFace. |
| DINOv3 weights | Custom `dinov3-license` (gated) | Comparison backbone (`BACKBONE=dinov3_vitl16`). Requires HF token approval (`dais26-secrets/hf-token`). |

**No trained model weights are stored in this repository.** Weights are downloaded at runtime and
cached in a UC Volume by `scripts/pin_model_cache.py`.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/README.md](docs/README.md) | 5-minute quickstart, full 11-step CLI sequence, prerequisites, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System diagram, BackboneInfo contract, two-phase deploy, @candidate→@champion, Mosaic AI comparison |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Pre-demo D-1 checklist, rollback procedure, DINOv2 fallback, service principal creation |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Latency and accuracy numbers (populated Phase 4) |
| [docs/TALK.md](docs/TALK.md) | 45-minute talk outline with timing marks and slide-to-demo mapping |

## Developer commands

```bash
make install          # uv pip install -e ".[dev]"
make test             # pytest tests/unit/
make build            # uv build (produces dist/*.whl)
make bundle-validate  # databricks bundle validate -t dev
make bundle-deploy-dev
make bundle-run-train
make discover-air     # Day 1 discovery gate
make warmup           # pre-warm endpoints before demo
make help             # full command list
```

## Timeline

| Milestone | Date |
|-----------|------|
| Phase 1: Foundation + Discovery | May 25-27 |
| Phase 2: Training + Detection + Endpoint | May 28-29 |
| Phase 3: Embeddings + Vector Search + Drift | Jun 1-3 |
| Phase 4: Integration + Polish | Jun 4-5 |
| Code freeze | Jun 8 (EOD) |
| Phase 5: Talk prep | Jun 9-12 |
| Demo rehearsal | Jun 13-14 |
| DAIS26 talk | Jun 15, 2026 |
