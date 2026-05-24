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

This installs all runtime and dev dependencies (torch, transformers, mlflow, databricks-sdk, albumentations, etc.)
in editable mode so changes to `src/` are immediately reflected.

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

Produces `dist/dais26_mlops_for_dl_on_air-0.1.0-py3-none-any.whl` (exact name may vary).
The wheel is attached to all job tasks via the DAB `libraries` block.

Verify:

```bash
ls dist/*.whl
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

This job executes four tasks in sequence:

```
setup (00_setup.py)
  --> train (02_train_detector_air.py)
      --> register_and_alias (04_deploy_serving.py, action=register_and_set_candidate)
          --> deploy_endpoint (04_deploy_serving.py, action=deploy_and_smoke_test)
```

The `deploy_endpoint` task:
1. Resolves the `@candidate` alias to a numeric model version
2. Creates the endpoint `dais26-cradio-detector-dev` via the Databricks SDK
3. Waits up to 600s for the endpoint to reach `READY` state
4. Runs a smoke test (1 sample image, expects 200 OK with detections)
5. Promotes `@candidate` to `@champion` on success

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

This runs `03_precompute_embeddings.py` on an AIR H100 cluster, which:
- Computes C-RADIOv4 `summary` embeddings (dim 1152) for all 1005 DENTEX images
- Writes to `ml_dev.dais26_vfm.train_embeddings` as `ARRAY<FLOAT>` with Change Data Feed enabled
- Creates the Mosaic AI Vector Search index `ml_dev.dais26_vfm.embeddings_index`

Wait for completion (~15-20 minutes).

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
    query_vector=[0.0] * 1152,   # backbone_info.summary_dim for C-RADIOv4
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
| `trust_remote_code` error loading C-RADIOv4 | Transformers version mismatch | Pin `transformers>=4.48.0` in your cluster; check pyproject.toml |
| `@champion` alias not set after training | Smoke test failed; `@candidate` left in place | Check `deploy_endpoint` task logs; re-run or promote manually |
