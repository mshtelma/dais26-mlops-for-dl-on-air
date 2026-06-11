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

if DEPLOY_ACTION == "create_vector_search":
    # NOTE: a self-contained, always-on version of this logic lives in
    # notebooks/04b_create_vector_search.py (run via the create_vector_search
    # job). This branch is kept in sync with it.
    import time

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.vectorsearch import (
        DeltaSyncVectorIndexSpecRequest,
        EmbeddingVectorColumn,
        EndpointType,
        PipelineType,
        VectorIndexType,
    )

    w = WorkspaceClient()

    # Derive the embedding dimension from the source table so this stays correct
    # regardless of backbone (C-RADIOv4-SO400M summary=2304, DINOv3-ViTL16=1024).
    source_table = TRAIN_EMBEDDINGS_TABLE
    embedding_dim = int(
        spark.sql(f"SELECT size(embedding) AS d FROM {source_table} LIMIT 1").collect()[0]["d"]
    )

    # Create endpoint (if not exists)
    try:
        w.vector_search_endpoints.create_endpoint_and_wait(
            name=VS_ENDPOINT_NAME, endpoint_type=EndpointType.STANDARD,
        )
        print(f"Created VS endpoint: {VS_ENDPOINT_NAME}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"VS endpoint {VS_ENDPOINT_NAME} already exists")
        else:
            raise

    # Create Delta Sync index with precomputed (self-managed) embeddings. The
    # embedding_dimension is IMMUTABLE, so a backbone change between champions
    # (C-RADIOv4 summary=2304, DINOv3=1024, DINOv2=768) requires a drop + recreate, not a
    # sync — otherwise dim-mismatched vectors get synced into the stale index.
    def _create_index(dim: int) -> None:
        w.vector_search_indexes.create_index(
            name=VS_INDEX_NAME,
            endpoint_name=VS_ENDPOINT_NAME,
            primary_key="image_id",
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=source_table,
                embedding_vector_columns=[
                    EmbeddingVectorColumn(name="embedding", embedding_dimension=dim),
                ],
                pipeline_type=PipelineType.TRIGGERED,
                columns_to_sync=["image_id", "diagnosis", "split"],
            ),
        )

    def _existing_index_dim() -> int | None:
        """Embedding dimension baked into the live index, or None if undiscoverable."""
        idx = w.vector_search_indexes.get_index(index_name=VS_INDEX_NAME)
        spec = getattr(idx, "delta_sync_index_spec", None)
        cols = getattr(spec, "embedding_vector_columns", None) or []
        for col in cols:
            d = getattr(col, "embedding_dimension", None)
            if d:
                return int(d)
        return None

    print(f"Creating index {VS_INDEX_NAME} from {source_table} (dim={embedding_dim})")
    try:
        _create_index(embedding_dim)
        print(f"Created VS index: {VS_INDEX_NAME}")
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise
        existing_dim = _existing_index_dim()
        if existing_dim is not None and existing_dim != embedding_dim:
            print(
                f"VS index {VS_INDEX_NAME} exists at dim={existing_dim} but the new "
                f"champion's embeddings are dim={embedding_dim}; dropping + recreating."
            )
            w.vector_search_indexes.delete_index(index_name=VS_INDEX_NAME)
            del_deadline = time.time() + 300
            while time.time() < del_deadline:
                try:
                    w.vector_search_indexes.get_index(index_name=VS_INDEX_NAME)
                    time.sleep(5)
                except Exception:
                    break
            _create_index(embedding_dim)
            print(f"Recreated VS index {VS_INDEX_NAME} at dim={embedding_dim}")
        else:
            print(f"VS index {VS_INDEX_NAME} already exists at matching dim={existing_dim}; syncing")
            w.vector_search_indexes.sync_index(index_name=VS_INDEX_NAME)
