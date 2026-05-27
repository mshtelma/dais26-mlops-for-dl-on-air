# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Deploy Serving (SDK-driven)
# MAGIC Two modes:
# MAGIC - `register_and_set_candidate`: verifies the trained model is registered + has `@candidate`
# MAGIC - `deploy_and_smoke_test`: resolves `@candidate` to numeric, deploys endpoint via SDK,
# MAGIC   runs smoke test, promotes to `@champion`. Uses create_and_wait / update_config_and_wait.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

full_model = f"{CATALOG}.{SCHEMA}.{DETECTOR_MODEL_SHORT}"

print(f"Action: {DEPLOY_ACTION}")
print(f"Model: {full_model}")
print(f"Endpoint: {DETECTOR_ENDPOINT_NAME}")

# COMMAND ----------

if DEPLOY_ACTION == "register_and_set_candidate":
    # The training task already registers + sets @candidate; verify it.
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient(registry_uri="databricks-uc")
    try:
        mv = client.get_model_version_by_alias(name=full_model, alias="candidate")
        print(f"@candidate -> version {mv.version}")
    except Exception as e:
        raise RuntimeError(
            f"No @candidate alias on {full_model}; training task may have failed: {e}"
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
        timeout_seconds=900,
    )
    print(result)
    if not result.smoke_test_passed:
        raise RuntimeError(f"Smoke test failed: {result.error}")
    print(f"Promoted to @champion: {result.promoted_to_champion}")
    print(f"Previous champion (capture for rollback): {result.previous_champion}")

# COMMAND ----------

if DEPLOY_ACTION == "create_vector_search":
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    # Create endpoint (if not exists)
    try:
        w.vector_search_endpoints.create_endpoint_and_wait(
            name=VS_ENDPOINT_NAME, endpoint_type="STANDARD",
        )
        print(f"Created VS endpoint: {VS_ENDPOINT_NAME}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"VS endpoint {VS_ENDPOINT_NAME} already exists")
        else:
            raise

    # Create Delta Sync index with precomputed embeddings (dim=1152 for C-RADIOv4)
    source_table = TRAIN_EMBEDDINGS_TABLE
    print(f"Creating index {VS_INDEX_NAME} from {source_table}")
    try:
        w.vector_search_indexes.create_index(
            name=VS_INDEX_NAME,
            endpoint_name=VS_ENDPOINT_NAME,
            primary_key="image_id",
            index_type="DELTA_SYNC",
            delta_sync_index_spec={
                "source_table": source_table,
                "embedding_vector_column": "embedding",
                "embedding_dimension": 1152,
                "pipeline_type": "TRIGGERED",
                "columns_to_sync": ["image_id", "diagnosis", "split"],
            },
        )
        print(f"Created VS index: {VS_INDEX_NAME}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"VS index {VS_INDEX_NAME} already exists; syncing")
            w.vector_search_indexes.sync_index(index_name=VS_INDEX_NAME)
        else:
            raise
