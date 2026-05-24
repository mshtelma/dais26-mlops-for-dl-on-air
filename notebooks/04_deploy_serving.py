# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Deploy Serving (SDK-driven)
# MAGIC Two modes:
# MAGIC - `register_and_set_candidate`: verifies the trained model is registered + has `@candidate`
# MAGIC - `deploy_and_smoke_test`: resolves `@candidate` to numeric, deploys endpoint via SDK,
# MAGIC   runs smoke test, promotes to `@champion`. Uses create_and_wait / update_config_and_wait.

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

dbutils.widgets.dropdown(
    "action", "deploy_and_smoke_test",
    ["register_and_set_candidate", "deploy_and_smoke_test", "create_vector_search"],
)
dbutils.widgets.text("model_name", DETECTOR_MODEL_SHORT)
dbutils.widgets.text("endpoint_name", DETECTOR_ENDPOINT_NAME)
dbutils.widgets.dropdown("workload_type", "GPU_SMALL", ["GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE"])
dbutils.widgets.dropdown("workload_size", "Small", ["Small", "Medium", "Large"])
dbutils.widgets.dropdown("scale_to_zero", "true", ["true", "false"])
dbutils.widgets.text("vs_endpoint_name", VS_ENDPOINT_NAME)
dbutils.widgets.text("vs_index_name", "")

action = dbutils.widgets.get("action")
model_name = dbutils.widgets.get("model_name")
endpoint_name = dbutils.widgets.get("endpoint_name")
workload_type = dbutils.widgets.get("workload_type")
workload_size = dbutils.widgets.get("workload_size")
scale_to_zero = dbutils.widgets.get("scale_to_zero") == "true"
vs_endpoint_name = dbutils.widgets.get("vs_endpoint_name")
vs_index_name = dbutils.widgets.get("vs_index_name").strip() or VS_INDEX_NAME
full_model = f"{CATALOG}.{SCHEMA}.{model_name}"

print(f"Action: {action}")
print(f"Model: {full_model}")
print(f"Endpoint: {endpoint_name}")

# COMMAND ----------

if action == "register_and_set_candidate":
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

if action == "deploy_and_smoke_test":
    from src.serve.endpoint_manager import deploy_and_smoke_test

    result = deploy_and_smoke_test(
        endpoint_name=endpoint_name,
        catalog=CATALOG,
        schema=SCHEMA,
        model_name=model_name,
        workload_type=workload_type,
        workload_size=workload_size,
        scale_to_zero=scale_to_zero,
        ai_gateway_enabled=True,
        inference_table_prefix=f"{model_name}_inference",
        promote_on_success=True,
        timeout_seconds=900,
    )
    print(result)
    if not result.smoke_test_passed:
        raise RuntimeError(f"Smoke test failed: {result.error}")
    print(f"Promoted to @champion: {result.promoted_to_champion}")
    print(f"Previous champion (capture for rollback): {result.previous_champion}")

# COMMAND ----------

if action == "create_vector_search":
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    # Create endpoint (if not exists)
    try:
        w.vector_search_endpoints.create_endpoint_and_wait(
            name=vs_endpoint_name, endpoint_type="STANDARD",
        )
        print(f"Created VS endpoint: {vs_endpoint_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"VS endpoint {vs_endpoint_name} already exists")
        else:
            raise

    # Create Delta Sync index with precomputed embeddings (dim=1152 for C-RADIOv4)
    source_table = TRAIN_EMBEDDINGS_TABLE
    print(f"Creating index {vs_index_name} from {source_table}")
    try:
        w.vector_search_indexes.create_index(
            name=vs_index_name,
            endpoint_name=vs_endpoint_name,
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
        print(f"Created VS index: {vs_index_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"VS index {vs_index_name} already exists; syncing")
            w.vector_search_indexes.sync_index(index_name=vs_index_name)
        else:
            raise
