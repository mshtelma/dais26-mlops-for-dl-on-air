# Databricks notebook source
# MAGIC %md
# MAGIC # 12 — Deployment job: RegisterChampion task (cross-schema, lineage-preserving)
# MAGIC
# MAGIC RegisterChampion task of `deploy_job_detector` (the CHALLENGER-side job; runs
# MAGIC after Evaluation + Approval pass). It **registers** the approved dev `@challenger`
# MAGIC version as the prod champion *candidate* — it does NOT deploy the endpoint or flip
# MAGIC `@champion`. Endpoint deploy + smoke test + the `@champion` flip + the
# MAGIC embeddings/Vector-Search/drift refresh are owned by the prod CHAMPION-side job
# MAGIC (`deploy_champion_job`, notebooks/14), which is auto-triggered by the new
# MAGIC `detector_champion` **version** this task creates (the supported MLflow 3
# MAGIC model-version deployment trigger; connect_deployment_job wires
# MAGIC detector_champion.deployment_job_id on -t prod).
# MAGIC
# MAGIC 1. `copy_model_version("models:/{model_name}/{model_version}",
# MAGIC    CHAMPION_FULL)` — creates a new version in the prod-schema registered
# MAGIC    model as a shallow copy whose version **points back to the source run**,
# MAGIC    so lineage to the training experiment is preserved (MLflow >= 2.8,
# MAGIC    UC -> UC same metastore). This version creation is what triggers
# MAGIC    `deploy_champion_job`. If the source dev version predates MLflow 3
# MAGIC    LoggedModels (empty `model_id`), `copy_model_version` can't copy it, so we
# MAGIC    fall back to registering a new champion version from the source run's
# MAGIC    artifact (`runs:/{run_id}/{artifact}`) — lineage is still preserved via the
# MAGIC    run_id.
# MAGIC 2. `set_registered_model_alias(CHAMPION_FULL, "champion_candidate", <new prod
# MAGIC    version>)` — `deploy_champion_job` resolves this alias, deploys the candidate,
# MAGIC    smoke-tests it, and flips `@champion` only on success, so the prior champion
# MAGIC    keeps serving until a new one is verified live.
# MAGIC
# MAGIC No GPU on the driver; serverless CPU.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from dais26_dentex.config.constants import ALIAS_CHAMPION_CANDIDATE

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

# SINGLE backbone-agnostic prod champion model (CHAMPION_MODEL_NAME from 00_config).
# The dev winner of ANY architecture is copied into this one model, so prod is never
# two competing architecture-named champions. Capture the source dev short name so we
# can tag which backbone the promoted version came from.
CHAMPION_FULL = CHAMPION_MODEL_NAME
_source_short = MODEL_NAME.split(".")[-1]
print(f"Registering {MODEL_NAME} v{MODEL_VERSION} -> {CHAMPION_FULL} as @{ALIAS_CHAMPION_CANDIDATE}")

client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------
# ---- 1. Lineage-preserving cross-schema copy (dev -> single prod/champion model) ----
# Prefer copy_model_version (UC -> UC, same metastore). MLflow 3's UC copy requires the
# source version to carry a non-empty model_id (its LoggedModel link). Versions created
# the classic way (mlflow.register_model from a run artifact) have model_id='' and fail
# the copy with "model_id must be a non-empty string". Fall back to register_model from
# the source run artifact, which creates a fresh champion version still tied to the
# source run_id (lineage preserved) and is what triggers deploy_champion_job.
try:
    copied = client.copy_model_version(
        src_model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}",
        dst_name=CHAMPION_FULL,
    )
    prod_version = str(copied.version)
    print(f"Copied to {CHAMPION_FULL} v{prod_version} via copy_model_version (LoggedModel lineage).")
except MlflowException as e:
    if "model_id" not in str(e):
        raise
    src_mv = client.get_model_version(name=MODEL_NAME, version=MODEL_VERSION)
    if not src_mv.run_id:
        raise RuntimeError(
            f"Cannot register champion: {MODEL_NAME} v{MODEL_VERSION} has neither a model_id "
            f"(copy_model_version failed: {e}) nor a run_id to register from."
        ) from e
    # Artifact path is the leaf of the version's source (typically 'model').
    artifact_path = (src_mv.source or "").rstrip("/").split("/")[-1] or "model"
    run_uri = f"runs:/{src_mv.run_id}/{artifact_path}"
    print(
        f"copy_model_version unsupported for this source (no model_id); registering "
        f"{run_uri} -> {CHAMPION_FULL} instead."
    )
    registered = mlflow.register_model(model_uri=run_uri, name=CHAMPION_FULL)
    prod_version = str(registered.version)
    print(f"Registered {CHAMPION_FULL} v{prod_version} from {run_uri} (lineage via run_id).")

# Record the source dev model on the prod version so operators (and the downstream
# embeddings/Vector-Search refresh) can tell which architecture this champion uses.
client.set_model_version_tag(
    name=CHAMPION_FULL,
    version=prod_version,
    key="source_dev_model",
    value=MODEL_NAME,
)

# COMMAND ----------
# ---- 2. Stage @champion_candidate for the prod deploy_champion_job ----
# The copy above already created a new detector_champion version, which triggers
# deploy_champion_job (notebooks/14_champion_deploy.py). That job resolves
# @champion_candidate, deploys + smoke-tests the endpoint, flips @champion only on
# success, then refreshes embeddings / Vector Search / drift. We deliberately do NOT
# touch @champion here, so the previous champion keeps serving until the new candidate
# is verified live. (Set the alias before the trigger job's cold start finishes.)
client.set_registered_model_alias(
    name=CHAMPION_FULL,
    alias=ALIAS_CHAMPION_CANDIDATE,
    version=prod_version,
)
print(
    f"Set @{ALIAS_CHAMPION_CANDIDATE} -> {CHAMPION_FULL} v{prod_version}. "
    f"The prod deploy_champion_job (triggered by this new version) will deploy + "
    f"smoke-test it and flip @champion on success."
)
dbutils.jobs.taskValues.set(key="champion_model", value=CHAMPION_FULL)
dbutils.jobs.taskValues.set(key="champion_candidate_version", value=prod_version)
dbutils.notebook.exit("ok")
