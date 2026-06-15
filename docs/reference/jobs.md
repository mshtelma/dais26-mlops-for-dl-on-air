# Jobs catalog

Every job is defined in `resources/jobs/*.yml` and deployed by `databricks bundle deploy`. All
compute is **AIR / Serverless** — there are no `job_clusters` anywhere. The wheel is attached per
serverless environment (`dependencies: ["../../dist/*.whl"]`), since serverless tasks reject
task-level `libraries`.

## Quick map

| Job | Target | Trigger | Compute (per task) | Timeout |
|-----|--------|---------|--------------------|---------|
| `train_detector` | dev | manual / quickstart | GPU_8xH100 (train); CPU (setup/confirm) | 8h |
| `campaign_sweep` | dev | manual (`sweep_stage`) | GPU_8xH100 (sweep) | 48h |
| `eval_comparison` | dev | manual | GPU_1xA10 | 2h |
| `eval_threshold_grid` | dev | manual | GPU_1xA10 | 4h |
| `deploy_job_detector` | dev | new `@challenger` version | GPU_1xA10 (eval); CPU (approve/register) | 2h |
| `deploy_champion_job` | prod | new `detector_champion` version | CPU (deploy/VS); GPU_1xA10 (embed/drift) | 3h |
| `connect_deployment_job` | dev + prod | manual (one-shot wiring) | CPU | 10m |
| `deploy_endpoint` | dev/prod | manual (break-glass) | CPU driver | ~55m |
| `drift_monitor` | **prod** | hourly cron (**PAUSED**) | CPU | 30m |

## `train_detector` (dev)

Quickstart training. `max_retries: 1` on setup/train; `queue.enabled: true`;
`performance_target: PERFORMANCE_OPTIMIZED`.

```
setup (00_setup.py, default env)
  └─> train (02_train_detector_air.py, GPU_8xH100, databricks_ai_v5)
        └─> confirm_challenger (04_deploy_serving.py, base_parameters: deploy_action=register_and_set_candidate)
```

## `campaign_sweep` (dev)

The "push to 0.60" stage driver. Job parameter `sweep_stage` (default `dinov3_s1`) → notebook
widget. `max_concurrent_runs: 5` (stages share the GPU pool); `timeout_seconds: 172800` (48h).

```
setup → sweep (02b_hpo_sweep.py, GPU_8xH100, base_parameters: sweep_stage={{job.parameters.sweep_stage}})
      → confirm_challenger (04_deploy_serving.py, register_and_set_candidate)
```

## `deploy_job_detector` (dev) — deployment job, CHALLENGER side

MLflow 3 deployment job. Triggered by a new `@challenger` version (via `deployment_job_id`).
`max_concurrent_runs: 1`; **no retries on any task** (fail fast). Job params `model_name` +
`model_version` injected by the framework. Emails on success/failure; emails the reviewer when
`Approval_Check` starts.

```
Evaluation (10_deploy_eval_task.py, GPU_1xA10)   gate: challenger beats @champion on ≥2/3 metrics
  └─> Approval_Check (11_deploy_approval_task.py, CPU)   UC tag Approval_Check=Approved (Approve button)
        └─> RegisterChampion (12_promote_task.py, CPU)   copy dev→prod, set @champion_candidate
```

Creating the new `detector_champion` version triggers `deploy_champion_job`.

## `deploy_champion_job` (prod) — deployment job, CHAMPION side

Triggered by a new `detector_champion` version (the RegisterChampion copy). `max_concurrent_runs:
1`; no retries; `timeout_seconds: 10800` (3h — a cold GPU serving deploy alone can take ~1h).

```
deploy_champion (14_champion_deploy.py, CPU)         deploy @champion_candidate + smoke → flip @champion on pass
  └─> precompute_embeddings (03, GPU_1xA10)
        └─> create_vector_search (04b, CPU)
              └─> drift_baseline (05, GPU_1xA10)
```

## `eval_comparison` (dev)

Re-score every registered detector through the serving pyfunc on the same held-out split (COCO
mAP) — apples-to-apples vs train-time metrics. GPU_1xA10, `timeout_seconds: 7200` (loading two ViT
backbones eats ~an hour before the small eval starts).

## `eval_threshold_grid` (dev)

Free, no-retrain decode/NMS threshold grid (`09b`) over the registered detectors; reports best
free thresholds + Caries AP@50 baseline. GPU_1xA10, `timeout_seconds: 14400` (~36 forward-only
1280px passes). HPO "Push to 0.60" **Stage 0**.

## `connect_deployment_job` (dev + prod)

One-shot post-deploy wiring (`13`). Target-aware: on `-t dev` connects `deploy_job_detector` →
dev detector models; on `-t prod` connects `deploy_champion_job` → `detector_champion`. Job params
`bundle_target`, `challenger_deployment_job_id`
(`${resources.jobs.deploy_job_detector.id}`), `champion_deployment_job_id`
(`${resources.jobs.deploy_champion_job.id}`). Run after every `bundle deploy`.

## `deploy_endpoint` (dev/prod) — break-glass

Manual endpoint (re)deploy via `04_deploy_serving.py` (`DEPLOY_ACTION` in `00_config`). The
**primary** path is `deploy_job_detector`; use this only for manual redeploys. CPU driver
(endpoint uses GPU serving compute), `timeout_seconds: 3300`.

## `drift_monitor` (prod) — PAUSED cron

Scheduled embedding-drift monitor (`05`). `schedule: 0 0 * * * ?` UTC, `pause_status: PAUSED`
(unpause post-demo). CPU serverless. Prod-only because the drift table lives in the champion
schema.

## Other bundle resources

- **Experiment** — `resources/experiments/vfm_experiment.yml`: the bundle-managed MLflow
  experiment all training/sweep/eval runs log to (the env's `experiment_name`).
- **Registered model** — `resources/registered_models/detector_models_champion.yml`: the prod
  `detector_champion`, declared **prod-only**, referencing the `dais26_vfm_prod` schema for
  create-ordering. (Dev detector models are **not** bundle-managed — see [DAB lane](../lanes/dab.md).)

How a release flows across the two deployment jobs:
[Evaluate → approve → promote](../lifecycle/evaluate-approve-promote.md) and
[Serve & AI Gateway](../lifecycle/serve.md).
