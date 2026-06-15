# DAIS26 MLOps for AI Runtime with Vision Foundation Models

**One frozen backbone, three jobs: detection head, drift sensor, embedding service.**

Databricks-native showcase for the Data + AI Summit 2026 (June 15-18, San Francisco).
Backbone is selectable in `notebooks/00_config.py` (`BACKBONE`): NVIDIA C-RADIOv4-SO400M
(ungated, commercial-OK, `spatial` dim 1152 / `summary` dim 2304) or Meta DINOv3-ViTL16 (gated,
comparison, dim 1024). DINOv2-base (dim 768) is the emergency fallback. All dimension-dependent
code parameterizes on `BackboneInfo` — nothing downstream hardcodes a dimension.
Dataset: DENTEX dental X-rays (CC-BY-NC-SA 4.0, research/demo only).
Compute: Databricks AI Runtime (AIR) / Serverless GPU only — no traditional ML clusters anywhere.
Training launch: notebook `@distributed` or the AIR CLI `air` (terminal). Both share one training core.
Serving: Mosaic AI Model Serving GPU endpoints, SDK-driven.

![CI](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/actions/workflows/ci.yml/badge.svg)

📖 **Full documentation site: <https://mshtelma.github.io/dais26-mlops-for-dl-on-air/>** — ultra-detailed
guides for every lane (DAB + air CLI) and the whole MLOps lifecycle. Best place to start.

## What this repo demonstrates

| Job | Backbone output used | Purpose |
|-----|----------------------|---------|
| `train_detector` | `spatial_features` | FPN + RetinaNet detection head on DENTEX |
| `precompute_embeddings` | `summary` | Delta table + Mosaic AI Vector Search index |
| `drift_monitor` | `summary` | KNN-distance drift detection on detector traffic |

Feature dim flows from `BackboneInfo`: detection uses `spatial` (C-RADIOv4-SO400M: 1152,
DINOv3-ViTL16: 1024); embeddings/VS/drift use `summary` (C-RADIOv4-SO400M: 2304 = 2×1152,
DINOv3-ViTL16: 1024). The Vector Search index dimension is even derived from the
embeddings table at index-creation time, so switching `BACKBONE` needs no doc/code edits downstream.

One frozen backbone artifact in a UC Volume. Three downstream consumers. Zero backbone gradients.

## Quick start

```bash
git clone <repo> && cd dais26-mlops-for-dl-on-air
pip install uv && uv pip install -e ".[dev]"
databricks auth login --host <DATABRICKS_HOST>
uv build                                       # ships pyproject.toml inside the wheel for serving-deps lookup
databricks bundle deploy -t dev               # deploys UC + jobs (NOT endpoints)
databricks bundle run train_detector -t dev   # DAB quickstart: train + register @challenger only
```

Terminal-first training alternative:

```bash
air run -f air/workload_train_detector.yaml --watch -p df1
```

Both quickstarts stop after the challenger model version is registered. Endpoint
deployment, human approval, champion promotion, embeddings, Vector Search, and
drift are separate operator lanes. Full prerequisites and lane details:
[docs/README.md](docs/README.md).

## Two launch paths for training — **AIR-only** (no traditional ML clusters)

Both quickstarts train on one 8xH100 AIR machine and register `@challenger`.
They differ only in launch mechanics:

- **DAB quickstart**: `databricks bundle run train_detector -t dev` runs the
  notebook job on `GPU_8xH100`. The notebook uses the local
  `serverless_gpu.@distributed` helper and does **not** use `torchrun`.
- **AIR CLI quickstart**: `air run -f air/workload_train_detector.yaml --watch -p df1`
  submits from a terminal and uses `torchrun`.

### Path A — Notebook (`@distributed`)

Run `notebooks/02_train_detector_air.py` interactively, or let the DAB job
trigger it. The decorator uses the task's eight local H100s:

```python
from serverless_gpu import distributed
from dais26_dentex.config.recipes import build_trainer_config
from dais26_dentex.train.trainer import Trainer

@distributed(gpus=8, gpu_type="h100")
def run_train():
    # campaign-final recipe + environment + explicit demo-time overrides
    cfg = build_trainer_config(BACKBONE, catalog=..., schema=..., epochs=TRAIN_EPOCHS)
    return Trainer(cfg).run()

results = run_train.distributed()
run_id  = next((r for r in results if r), None)   # rank-0 only returns a value
```

### Path B — air (terminal)

```bash
# One-time:
pip install databricks-air                      # AIR CLI (Beta)
databricks auth login --host <DATABRICKS_HOST> --profile df1

# Launch:
air run -f air/workload_train_detector.yaml --watch -p df1
air list runs --limit 10 -p df1
air logs <run-id> -p df1
```

Both paths share **one core and two named-config sources**: hyperparameters from
the per-backbone recipe in `config/recipes.py` and UC locations from the named
environment in `config/environments.py` — the workload YAML just names both
(`recipe: cradio_v4_so400m`, `env: df1`), the notebook resolves the identical
pair, and execution is `src/dais26_dentex/train/trainer.py::Trainer`, distributed-aware
(DDP, rank-0-only MLflow + UC registration) under both `@distributed` and
`torchrun`. Both lanes log to the same MLflow experiment, so the promotion
gates treat their runs identically. See [air/README.md](air/README.md).

### HP sweeps run on both paths too

The HPO campaign stages (`config/campaigns.py`) execute through one
`SweepRunner` (`train/sweep_runner.py`) from either lane:

```bash
databricks bundle run campaign_sweep -t dev -- --params sweep_stage=cradio_s2   # DAB
air run -f air/workload_sweep.yaml -p df1 --override parameters.stage=cradio_s2   # terminal
```

## Deployment model

Two phases — `bundle deploy` handles infrastructure; endpoints are SDK-driven.

```
bundle deploy -t dev
  |-- UC catalog / schema / volumes / secret scope
  |-- MLflow experiment
  |-- Dev job definitions (train_detector, campaign_sweep, eval_comparison, eval_threshold_grid)
  `-- (NO serving endpoints)

bundle deploy -t prod
  |-- Champion schema (dais26_vfm_prod) + champion models + deployment job
  `-- Prod-only embedding/monitoring jobs (precompute_embeddings, create_vector_search, drift_monitor)

bundle run train_detector -t dev
  setup --> train --> register @challenger --> confirm @challenger

operator promotion lane
  new @challenger --> eval --> approval --> prod champion copy --> deploy + smoke --> @champion
```

Endpoints are never created before a model version exists. The detector pyfunc is logged via
**MLflow models-from-code** (`serve/detector_model_script.py`) with the package source bundled via
`code_paths` — pickling the instance captured a `trust_remote_code` `transformers_modules.*`
backbone reference the serving container cannot import. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

These standalone jobs let you re-run a single phase (or run the governed promotion path) without the full `train_detector` chain:

| Job | Target | Notebook | Purpose |
|-----|--------|----------|---------|
| `deploy_endpoint` | dev + prod | `04_deploy_serving.py` | (Re)deploy `@challenger` → endpoint → smoke test → `@champion` |
| `create_vector_search` | **prod** | `04b_create_vector_search.py` | Create VS endpoint + DELTA_SYNC index over the prod-schema embeddings table, sync, smoke-test query |
| `precompute_embeddings` | **prod** | `03_precompute_embeddings.py` | Write the `summary`-embedding Delta table in the prod (champion) schema (prerequisite for `create_vector_search`) |
| `deploy_job_detector` | dev + prod | `10`–`14` | **Primary** promotion path (MLflow 3 deployment job): eval on test → approval → cross-schema champion promote + deploy |
| `connect_deployment_job` | dev + prod | `13_connect_deployment_job.py` | One-time wiring: set `deployment_job_id` on the dev detector models so a new `@challenger` auto-triggers `deploy_job_detector` |

The embedding/monitoring subsystem (`precompute_embeddings`, `create_vector_search`,
`drift_monitor`) is **prod-only**: its tables and VS index live in the champion
schema (`dais26_vfm_prod`), so run those with `-t prod`.

## Repo structure

```
dais26-mlops-for-dl-on-air/
|-- databricks.yml          # DAB root manifest (UC + job resources; no endpoint YAML)
|-- pyproject.toml          # Python deps + [tool.dais26.serving-deps] SoT for pyfunc
|-- Makefile                # Developer shortcuts
|-- src/
|   `-- dais26_dentex/      # Importable Python package (src layout)
|       |-- config/         # constants, manifest v2, TrainerConfig schema, recipes (per-backbone best-known), campaigns (HPO stages)
|       |-- data/           # dentex_loader.py, dataset.py, transforms.py
|       |-- distributed/    # primitives (setup/safe_barrier/seed), barrier_dance (rank0_first)
|       |-- drift/          # embeddings.py, reference.py, monitor.py, inference_table_reader.py
|       |-- eval/           # coco_metrics.py
|       |-- models/         # backbones (BackboneInfo), adapters (FPN), detection_head, builder, targets
|       |-- platform/       # hf_env, mlflow_io (MlflowReporter, serving_pip_requirements), uc (UCName)
|       |-- serve/          # detector_pyfunc.py, detector_model_script.py (models-from-code loader), embedder_pyfunc.py, postprocess.py, endpoint_manager.py
|       `-- train/          # trainer.py (Trainer class), losses, sweep + sweep_runner (HPO brain), cli + sweep_cli (air entries)
|-- notebooks/              # 00_config / 00_setup .. 09_eval_comparison / 09b_eval_threshold_grid / 10_deploy_eval_task / 11_deploy_approval_task / 12_promote_task / 13_connect_deployment_job / 14_champion_deploy; env selected via 00_config.py (`ENV`), UC locations via config/environments.py, hyperparameters via config/recipes.py
|-- resources/              # DAB resource YAML (jobs/, experiments/; NO serving/)
|-- air/                    # AIR CLI workloads (terminal train + sweep lanes)
|-- scripts/                # discover_air_runtime.py, warmup_endpoints.py, pin_model_cache.py, ...
|-- tests/                  # unit/ and integration/ pytest suites
`-- docs/                   # ARCHITECTURE, RUNBOOK, BENCHMARKS, TALK, HPO
```

## Engineering anchors (Phase 4 hardening)

| Concern | Where it lives | Why |
|---|---|---|
| Artifact contract | `config/manifest.py` (v2 `manifest.json`) | Single file replaces v1's three sidecar JSONs; `version` first → `head -1` triages a model. |
| Trainer hyperparameters | `config/trainer_config.py` (`TrainerConfig`) | Single dataclass schema; same instance feeds the notebook `@distributed` path and the air/torchrun YAML. Defaults stay legacy-compatible — best-known VALUES live in recipes. |
| Per-backbone recipes | `config/recipes.py` (`RECIPES`, `build_trainer_config`) | One source of campaign-final hyperparameters; the notebook builds from it and the air workloads name it (`recipe:`), so the lanes cannot drift. |
| HPO campaign | `config/campaigns.py` (`CAMPAIGN_STAGES`) + `train/sweep_runner.py` (`SweepRunner`) | Typed, validated stages + one sweep brain (parent run, trials, retrains, challenger gate) behind two launchers: notebook 02b (`@distributed`) and `train/sweep_cli.py` (torchrun via `air/workload_sweep.yaml`). |
| Distributed primitives | `distributed/primitives.py` + `distributed/barrier_dance.py` | `safe_barrier` surfaces dead-rank deadlocks as `BarrierTimeoutError` instead of hanging on NCCL. `rank0_first` is sequence-matched and avoids the cold-cache HF download race. |
| HF env hardening | `platform/hf_env.py::configure_hf_env` | One canonical site for `HF_HUB_ENABLE_HF_TRANSFER=0` + `HF_HUB_DISABLE_XET=1` (UC Volume FUSE rejects parallel chunked writes). |
| Pyfunc serving deps | `pyproject.toml::[tool.dais26.serving-deps]` ↔ `platform/mlflow_io.py::serving_pip_requirements` | One edit to add a runtime dep; the wheel ships `pyproject.toml` as `dais26_dentex/_pyproject.toml` so the lookup works in AIR's ephemeral env. CI guards via `assert_serving_reqs_match_pyproject`. `torch`/`torchvision` are pinned to the cu124 build (`2.6.0` / `0.21.0`) so GPU_SMALL (T4, driver CUDA 12.4) does not silently fall back to CPU. |
| Pyfunc serving load path | `serve/detector_model_script.py` + `platform/mlflow_io.py::_default_code_paths` | Detector logged via **models-from-code** (script, not pickled instance) with the package bundled via `code_paths`. Avoids `ModuleNotFoundError: transformers_modules` (dynamic `trust_remote_code` class) and `ModuleNotFoundError: dais26_dentex` at serving. Backbone loads strictly offline (`local_files_only`) from the bundled HF cache; `torch.compile` is disabled at serving. |
| MLflow API drift | `platform/mlflow_io.py::_log_model_artifact_kwarg` | `name=` vs `artifact_path=` resolved once at import via `inspect.signature`. |
| UC identifiers | `platform/uc.py::UCName`, `VolumePath` | Stop hand-rolling `f"{catalog}.{schema}.{name}"`; UC ident regex catches dotted-catalog typos. |
| Notebook params | `notebooks/00_config.py` | Selects a named environment (`ENV`) — UC locations resolve from `config/environments.py`; hyperparameters from `config/recipes.py`; demo-time knobs (TRAIN_EPOCHS etc.) live here. Two deliberate job-parameter exceptions — `sweep_stage` (campaign_sweep) and `deploy_action` (confirm_challenger) — ride DAB `base_parameters` into notebook widgets so one job definition serves every stage/mode. |

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

**Everything below is published as a navigable site: <https://mshtelma.github.io/dais26-mlops-for-dl-on-air/>**
(MkDocs + Material, built from `docs/`). The Markdown sources:

| Doc | Contents |
|-----|----------|
| [docs/](docs/README.md) → [site Get Started](https://mshtelma.github.io/dais26-mlops-for-dl-on-air/getting-started/overview/) | DAB + AIR CLI quickstarts, prerequisites, operator lanes, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System diagram, BackboneInfo contract, two-phase deploy, @challenger→@champion, Mosaic AI comparison |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Pre-demo D-1 checklist, rollback procedure, DINOv2 fallback, service principal creation |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Latency and accuracy numbers (populated Phase 4) |
| [docs/HPO.md](docs/HPO.md) | Detector HPO log: the push-to-0.60 mAP campaign, architectural fixes, and best-run record |
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
