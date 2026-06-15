# DAB (Asset Bundle) lane

The Databricks Asset Bundle is the declarative, reproducible lane. It deploys Unity Catalog
objects, the MLflow experiment, and **job definitions** — then you `bundle run` jobs to do work.
It is the lane the entire [MLOps lifecycle](../lifecycle/overview.md) and the
[deployment jobs](../lifecycle/evaluate-approve-promote.md) are wired around.

## The bundle root

[`databricks.yml`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/databricks.yml)
defines:

- `bundle.name: dais26-vfm`
- a `dais26_vfm_wheel` artifact built by `uv build`
- `include:` globs for `resources/jobs/*.yml`, `resources/experiments/*.yml`,
  `resources/registered_models/*.yml`
- **No `resources/serving/*.yml`** — serving endpoints are SDK-driven (see
  [two-phase deploy](../ARCHITECTURE.md#two-phase-deployment))

```yaml title="databricks.yml (excerpt)"
include:
  - "resources/jobs/*.yml"
  - "resources/experiments/*.yml"
  - "resources/registered_models/*.yml"
  # NOTE: resources/serving/*.yml is intentionally NOT included.
```

## Targets

| Target | Mode | `run_as` | What it owns |
|--------|------|----------|--------------|
| `dev` (default) | `development` | deploying user | dev schema (data-laden, **not** bundle-managed), dev jobs |
| `prod` | `production` | `${var.sp_app_id}` (service principal) | champion schema (Terraform-managed), champion model + prod-only jobs |

`mode: development` prefixes most resource names with `dev_<user>_` — which is exactly why the
**dev detector models** are *not* bundle-managed (they'd be prefixed and clash with the literal
`00_config` name) and why the **dev schema** is created by `00_setup.py`, not Terraform. The prod
champion schema/model are declared **prod-only** so prod mode (no prefix) resolves them to the
literal names. See [Production deployment](../scenarios/production-deploy.md).

## Two-phase deploy

```bash
databricks bundle deploy -t dev    # Phase 1: UC + jobs + experiment (NO endpoints)
databricks bundle run train_detector -t dev   # Phase 2: train → @challenger
```

`bundle deploy -t dev` creates the dev schema/volumes, the experiment, the secret scope, and the
dev job definitions. `bundle deploy -t prod` additionally creates the champion schema + champion
model + the prod-only embedding/monitoring jobs. Endpoints are **never** created by `deploy` —
they're deployed by SDK inside the deployment jobs after a version exists.

## Jobs you run

| Job | Target | Purpose |
|-----|--------|---------|
| `train_detector` | dev | Quickstart: train + register `@challenger` (setup → train → confirm) |
| `campaign_sweep` | dev | "Push to 0.60" HPO stage driver (parametrized by `sweep_stage`) |
| `eval_comparison` | dev | Re-score registered detectors through the serving pyfunc (val + test) |
| `eval_threshold_grid` | dev | Free, no-retrain decode/NMS threshold grid |
| `deploy_job_detector` | dev | **Deployment job (challenger side)**: eval → approval → register champion |
| `deploy_champion_job` | prod | **Deployment job (champion side)**: deploy + smoke + flip `@champion` → embeddings → VS → drift |
| `connect_deployment_job` | dev + prod | One-shot wiring of `deployment_job_id` on the trigger models |
| `deploy_endpoint` | dev/prod | Break-glass manual endpoint (re)deploy |
| `drift_monitor` | prod | Scheduled (paused) hourly drift cron |

Full task graphs, compute, timeouts, and triggers: **[Jobs catalog](../reference/jobs.md)**.

## Passing parameters to a job

Most config is read from `notebooks/00_config.py` via `%run ./00_config`. Two deliberate
exceptions ride DAB `base_parameters` into notebook widgets so one job definition serves every
mode:

```bash
# HPO stage selector:
databricks bundle run campaign_sweep -t dev -- --params sweep_stage=cradio_s2

# deploy-action switch (used by the confirm_challenger task / break-glass deploy):
#   register_and_set_candidate | deploy_and_smoke_test | create_vector_search
```

The deployment jobs (`deploy_job_detector`, `deploy_champion_job`) instead receive
`model_name` + `model_version` automatically — injected by the MLflow 3 deployment-job framework
when a new model version triggers the run.

## Run a single task of a job

```bash
databricks bundle run deploy_champion_job -t prod --only precompute_embeddings
```

## Validate / deploy locally

```bash
make bundle-validate        # databricks bundle validate -t dev
make bundle-deploy-dev      # databricks bundle deploy -t dev
make bundle-run-train       # databricks bundle run train_detector -t dev
./scripts/deploy_bundle.sh -t dev   # deploy + connect_deployment_job (the CI equivalent)
```

!!! warning "CI cannot reach the workspace"
    `databricks bundle validate`/`deploy` from GitHub-hosted runners are blocked by the
    workspace IP access list (403). Run them locally or from a self-hosted / allowlisted runner.
    See [CI/CD](../scenarios/cicd.md).

## How the DAB lane resolves config

`notebooks/00_config.py` is the single place to switch the target: `ENV = "df1"` calls
`load_environment(ENV)` and derives every catalog/schema/volume/experiment value. Hyperparameters
come from the per-backbone recipe via `build_trainer_config(BACKBONE, …)`. Both are the same
sources the air lane names. See **[Named configuration](configuration.md)**.

Continue to the **[air CLI lane](air.md)** or **[Named configuration](configuration.md)**.
