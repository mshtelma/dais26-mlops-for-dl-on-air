# Databricks notebook source
# MAGIC %md
# MAGIC # 11 — Deployment job: Approval task
# MAGIC
# MAGIC Second task of `deploy_job_detector` (runs after Evaluation passes). This is
# MAGIC the human-in-the-loop gate. It passes **only** when the UC model version
# MAGIC carries the tag `Approval_Check = Approved`; otherwise it raises, failing the
# MAGIC task. The job sets this task's `max_retries: 0`, so the first run fails fast
# MAGIC and waits for a human (the service principal later) to set the tag and
# MAGIC re-run / repair the task.
# MAGIC
# MAGIC To approve, set the tag on the evaluated version, e.g.:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.tracking import MlflowClient
# MAGIC c = MlflowClient(registry_uri="databricks-uc")
# MAGIC c.set_model_version_tag(name=MODEL_NAME, version=MODEL_VERSION,
# MAGIC                         key="Approval_Check", value="Approved")
# MAGIC ```
# MAGIC
# MAGIC then repair-run this task. No GPU needed (serverless CPU).

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

APPROVAL_TAG = "Approval_Check"
APPROVED_VALUE = "Approved"

dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_version", "")
MODEL_NAME = dbutils.widgets.get("model_name").strip()
MODEL_VERSION = dbutils.widgets.get("model_version").strip()
if not MODEL_NAME or not MODEL_VERSION:
    raise ValueError(
        "model_name and model_version job parameters are required "
        f"(got model_name={MODEL_NAME!r}, model_version={MODEL_VERSION!r})."
    )

client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------
mv = client.get_model_version(name=MODEL_NAME, version=MODEL_VERSION)
tags = dict(getattr(mv, "tags", {}) or {})
status = tags.get(APPROVAL_TAG)
print(f"{MODEL_NAME} v{MODEL_VERSION} tags: {tags}")

if status != APPROVED_VALUE:
    raise RuntimeError(
        f"Approval gate not satisfied: UC tag '{APPROVAL_TAG}' = {status!r} "
        f"(expected {APPROVED_VALUE!r}) on {MODEL_NAME} v{MODEL_VERSION}. "
        f"Set the tag (see this notebook's header) and repair-run this task."
    )

print(f"Approval gate PASSED: {APPROVAL_TAG} = {APPROVED_VALUE} on {MODEL_NAME} v{MODEL_VERSION}.")
dbutils.notebook.exit("ok")
