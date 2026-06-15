# Quickstart — DAB lane

Goal: from a clean workspace to a registered **`@challenger`** detector model using Databricks
Asset Bundles. This intentionally **stops before** endpoint deployment, champion promotion,
embeddings, Vector Search, and drift — those are operator lanes covered in the
[MLOps Lifecycle](../lifecycle/overview.md).

Complete [Install & authenticate](installation.md) first.

## Step 1 — Deploy dev infrastructure

```bash
databricks bundle deploy -t dev
```

Phase 1 deploys **only** UC resources and job definitions — **no serving endpoints**. What gets
created on `-t dev`:

- UC schema (e.g. `main.mshtelma`) with volumes `dentex_raw` and `model_cache`
- the bundle-managed MLflow experiment
- dev job definitions: `train_detector`, `campaign_sweep`, `eval_comparison`,
  `eval_threshold_grid`, plus the deployment-job wiring jobs
- secret scope `dais26-secrets` (for the optional DINOv3 path)

!!! note "Prod-only assets deploy under `-t prod`"
    The embedding/monitoring jobs (`precompute_embeddings`, `create_vector_search`,
    `drift_monitor`) and the champion schema/model are **prod-only** — they live in the champion
    schema and deploy with `databricks bundle deploy -t prod`. See
    [Production deployment](../scenarios/production-deploy.md).

## Step 2 — Run the training job

```bash
databricks bundle run train_detector -t dev
```

The job runs three notebook tasks in sequence:

```
setup (00_setup.py)
  └─> train (02_train_detector_air.py)   ← GPU_8xH100, serverless_gpu.@distributed
        └─> confirm_challenger (04_deploy_serving.py, deploy_action=register_and_set_candidate)
```

- The **`train`** task runs on one `GPU_8xH100` AIR notebook environment, using the local
  `serverless_gpu.@distributed` helper. It does **not** use `torchrun`.
- The **`confirm_challenger`** task only resolves `@challenger` and fails loudly if training did
  not register a usable version. It does **not** create an endpoint or set `@champion`.

Hyperparameters come from the per-backbone **recipe**; the demo wall-time knob `TRAIN_EPOCHS = 50`
(in `notebooks/00_config.py`) overrides the recipe's full 150-epoch schedule to keep the
quickstart ≈2h. The job carries an 8-hour timeout.

## Step 3 — Monitor

Watch the run in the Jobs UI, or:

```bash
databricks bundle run train_detector -t dev   # prints the run URL + run-id
databricks jobs get-run <run-id>               # poll status
```

## Success criteria

- [x] the job reaches `TERMINATED / SUCCESS`
- [x] `confirm_challenger` prints `@challenger -> version <n>`
- [x] the MLflow run contains the registered detector model artifacts

The dev detector model (e.g. `main.mshtelma.cradio_detector`) now has `@challenger` set on the
new version.

## What's next

| Next step | Page |
|---|---|
| Same thing from a terminal | [Quickstart — air CLI lane](quickstart-air.md) |
| Tune past the single-run plateau | [HPO sweep](../lifecycle/hpo-sweep.md) |
| Evaluate, approve, promote to `@champion` | [Evaluate → approve → promote](../lifecycle/evaluate-approve-promote.md) |
| Deploy a serving endpoint | [Serve & AI Gateway](../lifecycle/serve.md) |
| Full job/task details | [Jobs catalog](../reference/jobs.md) |

If anything failed, see [Troubleshooting](../reference/troubleshooting.md).
