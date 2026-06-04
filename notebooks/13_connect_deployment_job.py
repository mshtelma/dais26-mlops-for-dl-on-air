# Databricks notebook source
# MAGIC %md
# MAGIC # 13 — Connect the deployment job to the dev detector models
# MAGIC
# MAGIC One-shot post-deploy wiring step. After `databricks bundle deploy` creates
# MAGIC `deploy_job_detector`, this notebook resolves that job's id (passed in as the
# MAGIC `deployment_job_id` job parameter via `${resources.jobs.deploy_job_detector.id}`)
# MAGIC and calls `MlflowClient.update_registered_model(name, deployment_job_id=...)`
# MAGIC for **both** dev-schema detector models (`cradio_detector`, `dinov3_detector`).
# MAGIC
# MAGIC Once connected, registering a new `@challenger` version on either dev model
# MAGIC auto-triggers the deployment job (Evaluation -> Approval -> Promote) with that
# MAGIC version's `model_name` + `model_version`. One parameterized job, both models.
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

dbutils.widgets.text("deployment_job_id", "")
DEPLOYMENT_JOB_ID = dbutils.widgets.get("deployment_job_id").strip()
if not DEPLOYMENT_JOB_ID:
    raise ValueError(
        "deployment_job_id job parameter is required "
        "(wired from ${resources.jobs.deploy_job_detector.id} in the connect job)."
    )

# Dev-schema detector models to connect (skip the dinov2 fallback). Full UC names
# in CATALOG.SCHEMA from 00_config.
DEV_MODELS = [
    f"{CATALOG}.{SCHEMA}.{names['model_short']}"
    for backbone, names in _DETECTOR_NAMES_BY_BACKBONE.items()
    if backbone in ("cradio_v4_so400m", "dinov3_vitl16")
]
print(f"Connecting deployment job {DEPLOYMENT_JOB_ID} to: {DEV_MODELS}")

client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------
for full_model in DEV_MODELS:
    try:
        client.update_registered_model(name=full_model, deployment_job_id=DEPLOYMENT_JOB_ID)
        print(f"OK: {full_model} -> deployment_job_id={DEPLOYMENT_JOB_ID}")
    except Exception as e:
        print(f"SKIP ({type(e).__name__}): {full_model}: {e}")

dbutils.notebook.exit("ok")
