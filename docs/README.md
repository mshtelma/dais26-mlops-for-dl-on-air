# Quickstart

This guide walks you through deploying the DAIS26 VFM showcase from a clean workspace in under 60 minutes.

## Prerequisites

Before running the 11-step sequence, verify the following are available in your Databricks workspace:

| Requirement | Check |
|-------------|-------|
| Unity Catalog enabled | Workspace settings → Unity Catalog |
| AI Runtime access with H100 quota | `databricks clusters spark-versions \| grep -i ai-runtime` |
| Mosaic AI Model Serving GPU enabled | Workspace settings → Model Serving |
| Mosaic AI Vector Search enabled | Workspace settings → Vector Search |
| Databricks CLI v0.230+ | `databricks version` |
| Service principal (prod target only) | See [RUNBOOK.md](RUNBOOK.md#service-principal-creation) |

HuggingFace account is **not** required for the default C-RADIOv4 path (ungated). A HF token is only
needed if you activate the DINOv3 comparison path.

---

## 11-Step Deployment Sequence (E14)

### Step 1 — Clone and enter the repo

```bash
git clone <repo-url>
cd dais26-mlops-for-dl-on-air
```

### Step 2 — Install Python dependencies

```bash
pip install uv
uv pip install -e ".[dev]"
```

This installs all runtime and dev dependencies (torch, transformers, mlflow, databricks-sdk, etc.)
in editable mode so changes to `src/dais26_dentex/` are immediately reflected. Notebook params live
in `notebooks/00_config.py` — there are no `dbutils.widgets` and no DAB `base_parameters`; edit the
config file before launching.

**All UC names are config-driven from `notebooks/00_config.py`.** The current defaults are:

| Config knob | Default | Drives |
|-------------|---------|--------|
| `CATALOG` | `mlops_pj` | catalog for all schemas/tables/models |
| `SCHEMA` | `dais26_vfm` | schema |
| `TABLE_PREFIX` | `dais26_dentex_` | table/index prefix so multiple projects share one schema (e.g. `dais26_dentex_train_embeddings`) |
| `BACKBONE` | `dinov3_vitl16` | backbone + the backbone-keyed model/endpoint names (`dinov3_detector`, `dais26-dinov3-detector-dev`) |

Switch `BACKBONE` to `cradio_v4_so400m` for the ungated C-RADIOv4 path. The command examples below
use the legacy `ml_dev` / `cradio_detector` names; substitute your configured values.

### Step 3 — Authenticate with Databricks

```bash
databricks auth login --host <DATABRICKS_HOST>
```

Replace `<DATABRICKS_HOST>` with your workspace URL (e.g., `https://adb-1234567890.azuredatabricks.net`).
This writes credentials to `~/.databrickscfg`. Alternatively, export `DATABRICKS_HOST` and
`DATABRICKS_TOKEN` environment variables.

### Step 4 — Discover AIR runtime values (Day 1 gate)

```bash
python scripts/discover_air_runtime.py
```

This script lists available AI Runtime spark versions and node types, writes discovered values to
`.air-discovery.json`, and prints the values to substitute into `databricks.yml`.

After running, update the two DAB variables in `databricks.yml`:

```yaml
variables:
  air_spark_version:
    default: "<value from discover_air_runtime.py>"   # e.g. "ai-runtime-16.4.x-gpu-scala2.12"
  air_node_type_id:
    default: "<value from discover_air_runtime.py>"   # e.g. "Standard_NC24ads_A100_v4"
```

Until these are set, `databricks bundle validate` will fail with a `TODO_DISCOVER_DAY1` error.

Also pin the C-RADIOv4 commit SHA at this point:

```bash
# Find the current HEAD SHA on HuggingFace
python -c "
from huggingface_hub import HfApi
api = HfApi()
commits = api.list_repo_commits('nvidia/C-RADIOv4-SO400M')
print(commits[0].commit_id)
"
# Then set in databricks.yml:
#   cradio_commit_sha:
#     default: "<sha>"
```

### Step 5 — Build the Python wheel

```bash
uv build
```

Produces `dist/dais26_dentex-0.1.0-py3-none-any.whl` (exact name may vary).
The wheel is attached to all job tasks via the DAB `libraries` block.

The build copies `pyproject.toml` into the wheel as `dais26_dentex/_pyproject.toml`
(via hatchling `force-include`). At log-time, `platform.mlflow_io.serving_pip_requirements`
reads `[tool.dais26.serving-deps]` from this packaged copy — necessary because the AIR
ephemeral env installs the package into a site-packages whose ancestors do not contain
`pyproject.toml`. See [RUNBOOK.md#pip-requirements-rationale](RUNBOOK.md#pip-requirements-rationale).

Verify the table is present in the wheel:

```bash
ls dist/*.whl
python -m zipfile -l dist/dais26_dentex-0.1.0-py3-none-any.whl | grep _pyproject.toml
# → dais26_dentex/_pyproject.toml ...
```

### Step 6 — Deploy infrastructure (Phase 1)

```bash
databricks bundle deploy -t dev
```

This deploys **only** UC resources and job definitions. It does **not** deploy serving endpoints.

What gets created:
- UC schema `ml_dev.dais26_vfm` with volumes `dentex_raw` and `model_cache`
- MLflow experiment `/Users/<you>/dais26_vfm_experiment`
- Job definitions: `train_detector`, `precompute_embeddings`, `drift_monitor`
- Secret scope `dais26-secrets` (for optional DINOv3 path)

### Step 7 — Run the training job (Phase 2)

```bash
databricks bundle run train_detector -t dev
```

This job executes three serverless notebook tasks in sequence:

```
setup (00_setup.py)
  --> train (02_train_detector_air.py)
      --> deploy_endpoint (04_deploy_serving.py)
```

The `train` task is a serverless notebook that calls `serverless_gpu.@distributed`
to dispatch the actual GPU work to the H100 pool — the training core lives in
`src/dais26_dentex/train/trainer.py::Trainer` (rank-0-only MLflow log_model and
UC registration with `@candidate` alias). The `deploy_endpoint` task switches on
`DEPLOY_ACTION` from `notebooks/00_config.py`:

1. `register_and_set_candidate` — verifies the trained model is registered + has `@candidate`
2. `deploy_and_smoke_test` — does the real work:
   - Resolves the `@candidate` alias to a numeric model version
   - Creates the endpoint `dais26-cradio-detector-dev` via the Databricks SDK
   - Waits up to 900s for the endpoint to reach `READY` state
   - Runs a smoke test (1 sample image, expects 200 OK with detections)
   - Promotes `@candidate` to `@champion` on success

Endpoint orchestration (create / update / wait / smoke / promote) is centralized
in `src/dais26_dentex/serve/endpoint_manager.py::deploy_and_smoke_test`.

### Step 8 — Wait for the training job to complete

Expected wall time: **20-30 minutes** on a single H100 (10 epochs default).

Monitor progress in the Databricks Jobs UI or via:

```bash
# Stream logs (job run ID printed by the previous command)
databricks jobs get-run <run-id>
```

To run a faster 1-epoch validation gate instead of 10 epochs:

```bash
databricks bundle run train_detector -t dev --params train_epochs=1
```

### Step 9 — Precompute embeddings (Phase 3)

```bash
databricks bundle run precompute_embeddings -t dev
```

This runs `03_precompute_embeddings.py` on serverless GPU, which:
- Computes the backbone `summary` embeddings (C-RADIOv4: dim 1152, DINOv3: dim 1024) for all 1005 DENTEX images
- Writes to `<catalog>.<schema>.<prefix>train_embeddings` as `ARRAY<FLOAT>` with Change Data Feed enabled

Wait for completion (~15-20 minutes). The Vector Search index is **not** created here unless
`EMBEDDINGS_VS_ENDPOINT` and `EMBEDDINGS_VS_INDEX` are both set in `00_config.py`; otherwise create
it explicitly in the next step.

### Step 9b — Create the Vector Search endpoint + index

```bash
databricks bundle run create_vector_search -t dev
```

This runs `04b_create_vector_search.py` (no GPU), which idempotently creates the VS endpoint
(`dais26-vfm-vs`) and a DELTA_SYNC index over the embeddings table, triggers a sync, waits for the
index to come `ONLINE`, and runs a smoke-test similarity query. The embedding dimension is **derived
from the source table** (not hardcoded), so it stays correct for any backbone. The job fails fast
if the embeddings table from step 9 is empty.

### Step 10 — Smoke test the detector endpoint

```bash
export DATABRICKS_HOST=<your-workspace-url>
export DATABRICKS_TOKEN=<your-pat>

curl -X POST \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["image"], "data": [["<base64_encoded_image>"]]}}' \
  "https://$DATABRICKS_HOST/serving-endpoints/dais26-cradio-detector-dev/invocations"
```

Replace `<base64_encoded_image>` with a base64-encoded PNG or JPEG. To encode a test image:

```bash
base64 -i /path/to/test_image.png | tr -d '\n'
```

Expected response:

```json
{
  "predictions": [
    {
      "boxes": [[x1, y1, x2, y2], ...],
      "scores": [0.87, ...],
      "labels": ["Caries", ...],
      "num_detections": 3
    }
  ]
}
```

### Step 11 — Query Vector Search

Run this in a Databricks notebook or via the SDK locally:

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

results = w.vector_search_indexes.query_index(
    index_name="ml_dev.dais26_vfm.embeddings_index",
    columns=["image_id", "diagnosis"],
    query_vector=[0.0] * 1152,   # backbone summary_dim: C-RADIOv4=1152, DINOv3-ViTL16=1024
    num_results=10,
)

for row in results.result.data_array:
    print(row)
```

Expected: 10 results with `image_id` and `diagnosis` columns.

---

## Deploying to production

```bash
# Create service principal first (see RUNBOOK.md)
databricks bundle deploy -t prod
databricks bundle run train_detector -t prod
databricks bundle run precompute_embeddings -t prod
```

The `prod` target sets `scale_to_zero: false` (minimum 1 replica always warm) and uses a service
principal for `run_as`. See [RUNBOOK.md](RUNBOOK.md#service-principal-creation) for SP setup.

---

## Non-AIR fallback

If AIR is unavailable in your region, use the `dev_non_air` target which substitutes standard DBR ML
GPU runtimes:

```bash
databricks bundle deploy -t dev_non_air
databricks bundle run train_detector -t dev_non_air
```

Node type defaults: AWS `g5.12xlarge` (4x A10G), Azure `Standard_NC24ads_A100_v4`, GCP `a2-highgpu-1g`.

---

## Common troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `bundle validate` fails with `TODO_DISCOVER_DAY1` | AIR runtime values not set | Run step 4 and update `databricks.yml` variables |
| Training job fails in `setup` task | UC catalog or schema missing | Verify UC is enabled; check `var.catalog` value |
| Endpoint stuck in `PENDING` for >10 min | Smoke test failure or GPU capacity issue | Check `deploy_endpoint` task logs; look for `smoke test` error; verify GPU_SMALL quota |
| Vector Search index stuck syncing | CDF not enabled on source table | Run `DESCRIBE EXTENDED ml_dev.dais26_vfm.train_embeddings` and verify `delta.enableChangeDataFeed = true` |
| `dist/*.whl` not found during bundle deploy | Step 5 skipped | Run `uv build` before `databricks bundle deploy` |
| `FileNotFoundError: Could not locate pyproject.toml` at log-time | Stale wheel built before the `force-include` block was added | Re-run `uv build`; verify with `python -m zipfile -l dist/*.whl \| grep _pyproject.toml` |
| `ModuleNotFoundError: timm` / `einops` / `open_clip` at serving | Runtime dep missing from `[tool.dais26.serving-deps].detector` | Add to that table in `pyproject.toml`; `assert_serving_reqs_match_pyproject` is the CI guard |
| `trust_remote_code` error loading C-RADIOv4 | Transformers version mismatch | Pin `transformers>=4.48.0` in your cluster; check pyproject.toml |
| Endpoint `DEPLOYMENT_FAILED` / `ModuleNotFoundError: transformers_modules` at model-server load | Model logged as a pickled pyfunc instance captured the dynamic `trust_remote_code` backbone class | Already fixed — the trainer logs via **models-from-code** (`serve/detector_model_script.py`). Re-train (or re-log) against current code; do not pickle a `DetectorPyfunc()` instance |
| `ModuleNotFoundError: dais26_dentex` at serving | Package source not bundled with the model | `MlflowReporter.log_pyfunc` passes `code_paths=[<dais26_dentex dir>]` by default; verify the model's `code/` dir contains the package |
| Endpoint serves on CPU (0% GPU util, slow) on GPU_SMALL | Unpinned `torch` resolved to a cu126/cu128 wheel the T4 driver (CUDA 12.4) can't init → `torch.cuda.is_available()` is False | Keep `torch==2.6.0` / `torchvision==0.21.0` (cu124) pinned in `[tool.dais26.serving-deps]` |
| C-RADIOv4/DINOv3 backbone tries to reach huggingface.co at serving and fails to start | Online HF load in an egress-less serving container | Serving forces `local_files_only` + offline HF env from the bundled `model_cache` artifact (handled in `detector_pyfunc.load_context`) |
| DINOv3 backbone download 401/403 during training | Gated repo, missing HF token | Put the token in secret `dais26-secrets/hf-token`; `02_train_detector_air.py` reads it on the driver and forwards `HF_TOKEN` into the `@distributed` worker |
| HF download fails with `os error 5` / `os error 95` on AIR | `HF_HUB_ENABLE_HF_TRANSFER=1` or `hf-xet` writing to UC Volume FUSE | Set `HF_HUB_ENABLE_HF_TRANSFER=0` and `HF_HUB_DISABLE_XET=1` *before* importing `dais26_dentex` (use `platform.hf_env.configure_hf_env`) — see [RUNBOOK.md#hf-transfer-fuse-incompat](RUNBOOK.md#hf-transfer-fuse-incompat) |
| Cold-cache HF download deadlock on multi-rank run | Naive `barrier()` doesn't fix the from_pretrained race | Use `distributed.barrier_dance.rank0_first` (already wired in `models/builder.py`) — see [RUNBOOK.md#hf-cache-race](RUNBOOK.md#hf-cache-race) |
| `BarrierTimeoutError` from `safe_barrier` | A rank crashed earlier; NCCL would have hung silently | Inspect ranks' logs in order; the bounded wait surfaces the dead-rank instead of hanging |
| `IncompatibleArtifactError: artifact_format_version=1` at load | Loading a v1 artifact (sidecar JSONs) with the v2 manifest loader | Re-train against the current code; v1→v2 migration is one-shot, not auto-converted |
| `@champion` alias not set after training | Smoke test failed; `@candidate` left in place | Check `deploy_endpoint` task logs; re-run or promote manually |
