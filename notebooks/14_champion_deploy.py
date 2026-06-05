# Databricks notebook source
# MAGIC %md
# MAGIC # 14 — Champion deploy task (prod)
# MAGIC
# MAGIC First task of the prod CHAMPION-side job `deploy_champion_job`
# MAGIC (`resources/jobs/deploy_champion_job.yml`). That job is an MLflow 3 deployment job
# MAGIC connected to the prod `detector_champion` model, so it is **auto-triggered by the
# MAGIC new champion version** that the challenger-side RegisterChampion task (notebooks/12)
# MAGIC creates when it copies an approved dev winner into the prod schema.
# MAGIC
# MAGIC (Champion deploy was briefly designed as a `MODEL_ALIAS_SET`-triggered job, then
# MAGIC folded into the challenger job — but model/alias triggers are Private Preview and
# MAGIC unsupported by the Terraform provider. The model-**version** deployment trigger IS
# MAGIC supported, so the champion side is its own job again, triggered on the champion copy.)
# MAGIC
# MAGIC This task resolves the single backbone-agnostic champion model + endpoint from
# MAGIC `00_config` and deploys the `@champion_candidate` version:
# MAGIC
# MAGIC 1. `deploy_and_smoke_test(..., candidate_alias="champion_candidate",
# MAGIC    promote_on_success=True)` — resolves `@champion_candidate` -> numeric
# MAGIC    version, creates/updates the serving endpoint to serve it, smoke-tests it,
# MAGIC    and flips `@champion` to that version **only on success**. On failure
# MAGIC    `@champion` is left untouched, so the previous champion keeps serving.
# MAGIC 2. On success the downstream tasks (precompute_embeddings -> create_vector_search
# MAGIC    -> drift_baseline) refresh the reference embeddings, the Vector Search index,
# MAGIC    and the drift baseline for the new champion.
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

from dais26_dentex.config.constants import ALIAS_CHAMPION_CANDIDATE
from dais26_dentex.serve.endpoint_manager import deploy_and_smoke_test

mlflow.set_registry_uri("databricks-uc")

# SINGLE backbone-agnostic prod champion model + its single prod champion endpoint
# (from 00_config). Broad deployment serves whatever architecture holds @champion
# from one model + one endpoint — never two competing architecture-named champions.
_short = CHAMPION_MODEL_SHORT
_endpoint = CHAMPION_ENDPOINT_NAME
print(
    f"Deploying @{ALIAS_CHAMPION_CANDIDATE} of {CHAMPION_MODEL_NAME} -> endpoint {_endpoint}"
)

# COMMAND ----------
# ---- Deploy the candidate, smoke-test it, flip @champion only on success ----
result = deploy_and_smoke_test(
    endpoint_name=_endpoint,
    catalog=CHAMPION_CATALOG,
    schema=CHAMPION_SCHEMA,
    model_name=_short,
    candidate_alias=ALIAS_CHAMPION_CANDIDATE,
    workload_size=DEPLOY_WORKLOAD_SIZE,
    workload_type=DEPLOY_WORKLOAD_TYPE,
    scale_to_zero=DEPLOY_SCALE_TO_ZERO,
    promote_on_success=True,
)
print(result)

if not (result.smoke_test_passed and result.promoted_to_champion):
    raise RuntimeError(
        f"Champion deploy failed: smoke_passed={result.smoke_test_passed} "
        f"promoted={result.promoted_to_champion} state={result.state} error={result.error}. "
        f"@champion on {CHAMPION_MODEL_NAME} left untouched (previous champion still serves)."
    )

print(
    f"Deployed {CHAMPION_MODEL_NAME} v{result.deployed_version} to {_endpoint} and flipped "
    f"@champion (previous champion: {result.previous_champion}). Downstream refresh follows."
)
dbutils.jobs.taskValues.set(key="champion_model", value=CHAMPION_MODEL_NAME)
dbutils.jobs.taskValues.set(key="champion_version", value=result.deployed_version)
dbutils.notebook.exit("ok")
