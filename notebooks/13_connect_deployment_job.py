# Databricks notebook source
# MAGIC %md
# MAGIC # 13 — Connect the deployment jobs to their registered models (target-aware)
# MAGIC
# MAGIC One-shot post-deploy wiring step. After `databricks bundle deploy`, this notebook
# MAGIC calls `MlflowClient.update_registered_model(name, deployment_job_id=...)` so new
# MAGIC model versions auto-trigger the right MLflow 3 deployment job. It is **target-aware**
# MAGIC (driven by the `bundle_target` job parameter), because the dev and prod sides own
# MAGIC different models and jobs:
# MAGIC
# MAGIC - **`-t dev`** → connect the CHALLENGER job (`deploy_job_detector`, id from
# MAGIC   `challenger_deployment_job_id`) to **both dev-schema detector models**
# MAGIC   (`cradio_detector`, `dinov3_detector`). A new `@challenger` version then triggers
# MAGIC   Evaluation → Approval → RegisterChampion.
# MAGIC - **`-t prod`** → connect the CHAMPION job (`deploy_champion_job`, id from
# MAGIC   `champion_deployment_job_id`) to the **single prod `detector_champion` model**.
# MAGIC   The new champion version that RegisterChampion creates (the cross-schema copy)
# MAGIC   then triggers deploy + smoke-test + `@champion` flip + embeddings/VS/drift refresh.
# MAGIC
# MAGIC The champion job is connected to `detector_champion` (which it never writes new
# MAGIC versions to — it only sets aliases + deploys), so there is no trigger loop. This
# MAGIC replaces the abandoned `MODEL_ALIAS_SET` trigger (unsupported by the Terraform
# MAGIC provider) with the GA model-**version** deployment trigger.
# MAGIC
# MAGIC No GPU; serverless CPU.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("bundle_target", "")
dbutils.widgets.text("challenger_deployment_job_id", "")
dbutils.widgets.text("champion_deployment_job_id", "")
BUNDLE_TARGET = dbutils.widgets.get("bundle_target").strip()
CHALLENGER_JOB_ID = dbutils.widgets.get("challenger_deployment_job_id").strip()
CHAMPION_JOB_ID = dbutils.widgets.get("champion_deployment_job_id").strip()
if not BUNDLE_TARGET:
    raise ValueError(
        "bundle_target job parameter is required (wired from ${bundle.target} in the connect job)."
    )

# Target-aware association: prod owns the single champion model + champion job; dev
# owns the backbone-keyed detector models + challenger job. Build the
# (registered_model -> deployment_job_id) pairs accordingly.
if BUNDLE_TARGET == "prod":
    if not CHAMPION_JOB_ID:
        raise ValueError(
            "champion_deployment_job_id is required on -t prod "
            "(wired from ${resources.jobs.deploy_champion_job.id})."
        )
    # The new champion version that RegisterChampion creates triggers this job.
    CONNECTIONS = [(CHAMPION_MODEL_NAME, CHAMPION_JOB_ID)]
else:
    if not CHALLENGER_JOB_ID:
        raise ValueError(
            "challenger_deployment_job_id is required on -t dev "
            "(wired from ${resources.jobs.deploy_job_detector.id})."
        )
    # Dev-schema detector models (skip the dinov2 fallback); a new @challenger
    # version on either triggers the challenger job.
    CONNECTIONS = [
        (f"{CATALOG}.{SCHEMA}.{names['model_short']}", CHALLENGER_JOB_ID)
        for backbone, names in _DETECTOR_NAMES_BY_BACKBONE.items()
        if backbone in ("cradio_v4_so400m", "dinov3_vitl16")
    ]

print(f"Target={BUNDLE_TARGET}; connecting: {[(m, j) for m, j in CONNECTIONS]}")

client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------
# update_registered_model(deployment_job_id=...) requires the caller to OWN / have
# MANAGE on the model. Let failures raise so a mis-wired connection is loud (the old
# silent SKIP hid the prod SP lacking MANAGE on the dev models). We still report each
# success explicitly.
for full_model, job_id in CONNECTIONS:
    client.update_registered_model(name=full_model, deployment_job_id=job_id)
    refreshed = client.get_registered_model(full_model).deployment_job_id
    if str(refreshed) != str(job_id):
        raise RuntimeError(
            f"Set deployment_job_id={job_id} on {full_model} but read back {refreshed!r}."
        )
    print(f"OK: {full_model} -> deployment_job_id={job_id}")

dbutils.notebook.exit("ok")
