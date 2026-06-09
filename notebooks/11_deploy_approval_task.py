# Databricks notebook source
# MAGIC %md
# MAGIC # 11 — Deployment job: Approval task
# MAGIC
# MAGIC Second task of `deploy_job_detector` (runs after Evaluation passes). This is
# MAGIC the human-in-the-loop gate. It passes **only** when the UC model version
# MAGIC carries the tag `Approval_Check = Approved`; otherwise it raises, failing the
# MAGIC task. The job runs this task with no retries, so the first run fails fast and
# MAGIC waits for a human to approve and repair-run.
# MAGIC
# MAGIC **The tag key is the approval task's name.** Databricks treats a task whose
# MAGIC name starts with `approval` (case-insensitive) as the deployment-job approval
# MAGIC gate and shows an **Approve** button on the model version page. Clicking it
# MAGIC auto-repairs the run and writes a UC tag whose **key == the task name** with
# MAGIC value `Approved`. Our task is named **`Approval_Check`**, so the UI writes
# MAGIC `Approval_Check=Approved` — which is exactly what `APPROVAL_TAG` below checks.
# MAGIC Keep the task name (resources/jobs/deploy_job_detector.yml) and `APPROVAL_TAG`
# MAGIC in sync, or the UI Approve button will write a tag this gate never reads.
# MAGIC
# MAGIC To approve, click **Approve** on the model version page, or set the tag
# MAGIC manually and repair-run this task:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.tracking import MlflowClient
# MAGIC c = MlflowClient(registry_uri="databricks-uc")
# MAGIC c.set_model_version_tag(name=MODEL_NAME, version=MODEL_VERSION,
# MAGIC                         key="Approval_Check", value="Approved")
# MAGIC ```
# MAGIC
# MAGIC No GPU needed (serverless CPU).

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
