# Quickstart

This guide gets a clean workspace to a registered `@challenger` detector model.
It intentionally stops before endpoint deployment, champion promotion, embeddings,
Vector Search, and drift. Those are operator lanes after the quickstarts are
green.

## Prerequisites

Before running either quickstart, verify the following are available in your Databricks workspace:

| Requirement | Check |
|-------------|-------|
| Unity Catalog enabled | Workspace settings → Unity Catalog |
| AI Runtime access with single-node 8xH100 quota | Databricks account / workspace quota |
| Databricks CLI v0.230+ | `databricks version` |
| `sgcli` private-preview wheel | Required only for the SGCLI quickstart |

HuggingFace account is **not** required for the default C-RADIOv4 path (ungated). A HF token is only
needed if you activate the DINOv3 comparison path.

---

## Shared Setup

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
| `BACKBONE` | `cradio_v4_so400m` | backbone + the backbone-keyed model/endpoint names (`cradio_detector`, `dais26-cradio-detector-dev`) |

Switch `BACKBONE` to `dinov3_vitl16` for the gated DINOv3 comparison path. The command examples below
use the legacy `ml_dev` / `cradio_detector` names; substitute your configured values.

### Step 3 — Authenticate with Databricks

```bash
databricks auth login --host <DATABRICKS_HOST>
```

Replace `<DATABRICKS_HOST>` with your workspace URL (e.g., `https://adb-1234567890.azuredatabricks.net`).
This writes credentials to `~/.databrickscfg`. Alternatively, export `DATABRICKS_HOST` and
`DATABRICKS_TOKEN` environment variables.

### Step 4 — Build the Python wheel

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

### Step 5 — Deploy dev infrastructure

```bash
databricks bundle deploy -t dev
```

This deploys **only** UC resources and job definitions. It does **not** deploy serving endpoints.

What gets created:
- UC schema `ml_dev.dais26_vfm` with volumes `dentex_raw` and `model_cache`
- MLflow experiment `/Users/<you>/dais26_vfm_experiment`
- Dev job definitions: `train_detector`, `campaign_sweep`, `eval_comparison`, `eval_threshold_grid`
- Secret scope `dais26-secrets` (for optional DINOv3 path)

> The embedding/monitoring jobs (`precompute_embeddings`, `create_vector_search`,
> `drift_monitor`) and the champion schema/models are **prod-only** — they deploy
> under `databricks bundle deploy -t prod` because their tables and VS index live in
> the champion schema (`dais26_vfm_prod`).

---

## DAB Quickstart

Run the Databricks Asset Bundle training job:

```bash
databricks bundle run train_detector -t dev
```

This job executes three notebook tasks:

```
setup (00_setup.py)
  --> train (02_train_detector_air.py)
      --> confirm_challenger (04_deploy_serving.py)
```

The `train` task runs on one `GPU_8xH100` AIR notebook environment. It uses
`serverless_gpu.@distributed` inside the notebook and does **not** use
`torchrun`. The `confirm_challenger` task only resolves `@challenger`; it does
not create/update a serving endpoint and does not set `@champion`.

Expected wall time depends on `TRAIN_EPOCHS` in `notebooks/00_config.py`.

Monitor progress in the Databricks Jobs UI or via:

```bash
# Stream logs (job run ID printed by the previous command)
databricks jobs get-run <run-id>
```

Success criteria:

- the job reaches `TERMINATED/SUCCESS`
- `confirm_challenger` prints `@challenger -> version <n>`
- the MLflow run contains the registered detector model artifacts

---

## SGCLI Quickstart

From the repo root:

```bash
sgcli run -f sgcli/workload_train_detector.yaml --watch -p dev
```

This submits the same training config to one 8xH100 machine through SGCLI and
launches the package CLI with `torchrun`. The package CLI reads the SGCLI-written
`$HYPERPARAMETERS_PATH`, constructs `TrainerConfig`, and runs the same
`Trainer` core as the DAB notebook quickstart.

Inspect a run:

```bash
sgcli get runs --limit 10 -p dev
sgcli get logs <run-id> --rank 0 -p dev
```

Success criteria are the same as DAB: rank 0 logs the MLflow run, registers the
detector model, and sets `@challenger`.

---

## Next Lanes

### Hyperparameter sweep (Phase 2b)

If the single training run plateaus (the DENTEX detector has historically hit a ~3% mAP@50
ceiling), run the HPO sweep, which tunes the detector head **and** fine-tunes the
C-RADIO / DINOv3 backbone:

```bash
# 1. (recommended) audit the architecture first - anchors, positive ratio, NMS, delta clamp
# Open notebooks/02a_arch_probe.py and run all.

# 2. launch the sweep (single driver; pick a stage)
databricks bundle run campaign_sweep -t dev -- --params sweep_stage=dinov3_s1
```

> See [HPO.md](HPO.md) for the full push-to-0.60 mAP campaign - the architectural fixes
> (per-level anchors, per-class NMS), the winning runs, and the stage-by-stage sweep record.

The sweep runs as a parent MLflow run with one nested child run per trial, sharing the same
`Trainer` core as the quickstart lanes. It explores learning rates, `backbone_mode`
(`frozen`/`lora`/`partial`/`full`), unfreeze depth, anchor geometry, and head
regularization. All sweep parameters are config-driven from the `SWEEP_*` block in
`notebooks/00_config.py`:

| Config knob | Default | Drives |
|-------------|---------|--------|
| `SWEEP_STRATEGY` | `random` | `random` sampling or full `grid` |
| `SWEEP_MAX_TRIALS` | `8` | trial budget |
| `SWEEP_TRIAL_EPOCHS` | `25` | epochs per trial (shorter than final retrain) |
| `SWEEP_PRIMARY_METRIC` | `val/best_mAP_50` | metric `select_best` ranks on |
| `SWEEP_REGISTER_WINNER` | `True` | re-train winner for full `TRAIN_EPOCHS` -> `@challenger` |
| `SWEEP_SEARCH_SPACE` | see config | per-knob value lists (incl. `backbone_mode`, `anchor_mode`) |

Expected wall time: up to **48 hours** on `GPU_8xH100` (the job carries a 48-hour timeout).
The winning trial is re-trained at full epochs, registered to UC, and aliased `@challenger`
only when it clears the best-in-experiment gate; the trailing `confirm_challenger` task then
asserts the alias landed. Promotion to `@champion` happens via the `deploy_job_detector`
deployment job (eval -> approval -> cross-schema promote).

### Embeddings and Vector Search Lane

```bash
databricks bundle run deploy_champion_job -t prod --only precompute_embeddings
```

`precompute_embeddings` is a task inside the prod `deploy_champion_job` (not a standalone job).
This runs `03_precompute_embeddings.py` on serverless GPU, which:
- Computes the backbone `summary` embeddings (C-RADIOv4: dim 2304, DINOv3: dim 1024) for all 1005 DENTEX images
- Writes to `<catalog>.<schema>.<prefix>train_embeddings` as `ARRAY<FLOAT>` with Change Data Feed enabled

Wait for completion (~15-20 minutes). The Vector Search index is **not** created here unless
`EMBEDDINGS_VS_ENDPOINT` and `EMBEDDINGS_VS_INDEX` are both set in `00_config.py`; otherwise create
it explicitly in the next step.

### Create the Vector Search endpoint + index

```bash
databricks bundle run deploy_champion_job -t prod --only create_vector_search
```

`create_vector_search` is also a task inside `deploy_champion_job`.
This runs `04b_create_vector_search.py` (no GPU), which idempotently creates the VS endpoint
(`dais26-vfm-vs`) and a DELTA_SYNC index over the embeddings table, triggers a sync, waits for the
index to come `ONLINE`, and runs a smoke-test similarity query. The embedding dimension is **derived
from the source table** (not hardcoded), so it stays correct for any backbone. The job fails fast
if the embeddings table is empty.

### Serving Smoke Test Lane

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

### Vector Search Query Lane

Run this in a Databricks notebook or via the SDK locally:

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

results = w.vector_search_indexes.query_index(
    index_name="ml_dev.dais26_vfm.embeddings_index",
    columns=["image_id", "diagnosis"],
    query_vector=[0.0] * 2304,   # backbone summary_dim: C-RADIOv4=2304, DINOv3-ViTL16=1024
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
databricks bundle run connect_deployment_job -t dev
databricks bundle run connect_deployment_job -t prod
```

The `prod` target uses a service principal for `run_as` and owns the champion
schema/model resources. The normal production flow is triggered by a new
`@challenger` version on the dev model: evaluation -> approval -> prod champion
copy -> champion deployment job. See [RUNBOOK.md](RUNBOOK.md#service-principal-creation)
for SP setup.

---

## Common troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Training job fails in `setup` task | UC catalog or schema missing | Verify UC is enabled; check `var.catalog` value |
| `confirm_challenger` fails | Training did not register or alias the model version | Check the `train` task logs and MLflow registration output |
| Endpoint stuck in `PENDING` for >10 min | Smoke test failure or GPU capacity issue in the operator deploy lane | Check `deploy_endpoint` or `deploy_champion` task logs; look for `smoke test` error; verify GPU_SMALL quota |
| Vector Search index stuck syncing | CDF not enabled on source table | Run `DESCRIBE EXTENDED ml_dev.dais26_vfm.train_embeddings` and verify `delta.enableChangeDataFeed = true` |
| `dist/*.whl` not found during bundle deploy | Build step skipped | Run `uv build` before `databricks bundle deploy` |
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
| `@champion` alias not set after champion deploy | Smoke test failed; `@challenger` or `@champion_candidate` remains staged | Check deployment-job logs; re-run or promote manually |
