# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Deploy Serving (SDK-driven)
# MAGIC Two modes:
# MAGIC - `register_and_set_candidate`: verifies the trained model is registered + has `@challenger`
# MAGIC - `deploy_and_smoke_test`: resolves `@challenger` to numeric, deploys endpoint via SDK,
# MAGIC   runs smoke test, promotes to `@champion`. Uses create_and_wait / update_config_and_wait.
# MAGIC
# MAGIC This is the break-glass / manual deploy path; the primary promotion route is
# MAGIC the `deploy_job_detector` deployment job (eval -> approval -> cross-schema promote).
# MAGIC Vector Search index creation lives in `04b_create_vector_search.py`.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
# Optional per-run override of DEPLOY_ACTION. Jobs can pass it as a notebook
# parameter (e.g. campaign_sweep's `confirm_challenger` task sets
# deploy_action=register_and_set_candidate to just verify the @challenger alias
# without deploying). Falls back to the 00_config default when absent/empty.
dbutils.widgets.text("deploy_action", "")
_deploy_action_override = dbutils.widgets.get("deploy_action").strip()
if _deploy_action_override:
    DEPLOY_ACTION = _deploy_action_override

# COMMAND ----------

full_model = f"{CATALOG}.{SCHEMA}.{DETECTOR_MODEL_SHORT}"

print(f"Action: {DEPLOY_ACTION}")
print(f"Model: {full_model}")
print(f"Endpoint: {DETECTOR_ENDPOINT_NAME}")

# COMMAND ----------

if DEPLOY_ACTION == "register_and_set_candidate":
    # The training task already registers + sets @challenger; verify it.
    import mlflow
    from mlflow.tracking import MlflowClient

    from dais26_dentex.config.constants import ALIAS_CANDIDATE

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient(registry_uri="databricks-uc")
    try:
        mv = client.get_model_version_by_alias(name=full_model, alias=ALIAS_CANDIDATE)
        print(f"@{ALIAS_CANDIDATE} -> version {mv.version}")
    except Exception as e:
        raise RuntimeError(
            f"No @{ALIAS_CANDIDATE} alias on {full_model}; training task may have failed: {e}"
        ) from e
    dbutils.notebook.exit("ok")

# COMMAND ----------

if DEPLOY_ACTION == "deploy_and_smoke_test":
    from dais26_dentex.serve.endpoint_manager import deploy_and_smoke_test

    result = deploy_and_smoke_test(
        endpoint_name=DETECTOR_ENDPOINT_NAME,
        catalog=CATALOG,
        schema=SCHEMA,
        model_name=DETECTOR_MODEL_SHORT,
        workload_type=DEPLOY_WORKLOAD_TYPE,
        workload_size=DEPLOY_WORKLOAD_SIZE,
        scale_to_zero=DEPLOY_SCALE_TO_ZERO,
        ai_gateway_enabled=True,
        inference_table_prefix=f"{DETECTOR_MODEL_SHORT}_inference",
        promote_on_success=True,
        # GPU serving endpoints can take 20-40 min to cold-start on first deploy
        # (container build + model download + GPU attach); 15 min was too short.
        timeout_seconds=2400,
    )
    print(result)
    if not result.smoke_test_passed:
        raise RuntimeError(f"Smoke test failed: {result.error}")
    print(f"Promoted to @champion: {result.promoted_to_champion}")
    print(f"Previous champion (capture for rollback): {result.previous_champion}")

# COMMAND ----------
# The former `create_vector_search` action moved to its own always-on notebook,
# notebooks/04b_create_vector_search.py (wired into deploy_champion_job's
# create_vector_search task). The shared, dimension-aware create/sync logic is
# dais26_dentex.serve.vector_search.ensure_vector_search_index.
