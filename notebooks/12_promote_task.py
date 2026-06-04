# Databricks notebook source
# MAGIC %md
# MAGIC # 12 — Deployment job: Promote task (cross-schema, lineage-preserving)
# MAGIC
# MAGIC Final task of `deploy_job_detector` (runs after Evaluation + Approval pass).
# MAGIC It promotes the approved dev `@challenger` version into the separate prod /
# MAGIC champion schema with full lineage, then points serving at it:
# MAGIC
# MAGIC 1. `copy_model_version("models:/{model_name}/{model_version}",
# MAGIC    CHAMPION_FULL)` — creates a new version in the prod-schema registered
# MAGIC    model as a shallow copy whose version **points back to the source run**,
# MAGIC    so lineage to the training experiment is preserved (MLflow >= 2.8,
# MAGIC    UC -> UC same metastore).
# MAGIC 2. `deploy_and_smoke_test(..., model_version=<new prod version>,
# MAGIC    promote_on_success=True)` — deploys that explicit prod version to the
# MAGIC    endpoint, smoke-tests it, and only THEN flips `@champion` on the prod
# MAGIC    model. On smoke-test failure `@champion` is left untouched (the prior
# MAGIC    champion keeps serving).
# MAGIC
# MAGIC No GPU on the driver (the endpoint uses GPU serving compute); serverless CPU.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import mlflow
from mlflow.tracking import MlflowClient

from dais26_dentex.serve.endpoint_manager import deploy_and_smoke_test

mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_version", "")
MODEL_NAME = dbutils.widgets.get("model_name").strip()
MODEL_VERSION = dbutils.widgets.get("model_version").strip()
if not MODEL_NAME or not MODEL_VERSION:
    raise ValueError(
        "model_name and model_version job parameters are required "
        f"(got model_name={MODEL_NAME!r}, model_version={MODEL_VERSION!r})."
    )

# Prod champion model = same short (last) name in the prod/champion schema; the
# serving endpoint is the per-backbone one mapped in 00_config.
_short = MODEL_NAME.split(".")[-1]
CHAMPION_FULL = f"{CHAMPION_CATALOG}.{CHAMPION_SCHEMA}.{_short}"
_endpoint = next(
    (names["endpoint"] for names in _DETECTOR_NAMES_BY_BACKBONE.values() if names["model_short"] == _short),
    DETECTOR_ENDPOINT_NAME,
)
print(f"Promoting {MODEL_NAME} v{MODEL_VERSION} -> {CHAMPION_FULL}; endpoint {_endpoint}")

client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------
# ---- 1. Lineage-preserving cross-schema copy (dev -> prod/champion schema) ----
copied = client.copy_model_version(
    src_model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}",
    dst_name=CHAMPION_FULL,
)
prod_version = str(copied.version)
print(f"Copied to {CHAMPION_FULL} v{prod_version} (lineage to source run preserved).")

# COMMAND ----------
# ---- 2. Deploy that prod version + smoke test; flip @champion only on success ----
result = deploy_and_smoke_test(
    endpoint_name=_endpoint,
    catalog=CHAMPION_CATALOG,
    schema=CHAMPION_SCHEMA,
    model_name=_short,
    model_version=prod_version,
    workload_size=DEPLOY_WORKLOAD_SIZE,
    workload_type=DEPLOY_WORKLOAD_TYPE,
    scale_to_zero=DEPLOY_SCALE_TO_ZERO,
    promote_on_success=True,
)
print(result)

if not (result.smoke_test_passed and result.promoted_to_champion):
    raise RuntimeError(
        f"Promotion failed: smoke_passed={result.smoke_test_passed} "
        f"promoted={result.promoted_to_champion} state={result.state} error={result.error}. "
        f"@champion on {CHAMPION_FULL} left untouched (previous champion still serves)."
    )

print(
    f"Promoted {CHAMPION_FULL} v{prod_version} to @champion and deployed to {_endpoint} "
    f"(previous champion: {result.previous_champion})."
)
dbutils.jobs.taskValues.set(key="champion_model", value=CHAMPION_FULL)
dbutils.jobs.taskValues.set(key="champion_version", value=prod_version)
dbutils.notebook.exit("ok")
