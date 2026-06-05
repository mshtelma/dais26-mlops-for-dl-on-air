from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from datetime import timedelta

from dais26_dentex.config.constants import ALIAS_CANDIDATE

logger = logging.getLogger(__name__)


@dataclass
class EndpointDeployResult:
    endpoint_name: str
    deployed_version: str
    previous_champion: str | None
    smoke_test_passed: bool
    promoted_to_champion: bool
    state: str
    error: str | None = None


def resolve_alias_to_version(
    catalog: str,
    schema: str,
    model_name: str,
    alias: str,
) -> str:
    """Resolve a UC model alias (e.g., 'candidate', 'champion') to a numeric version string.

    Sets MLflow registry URI to databricks-uc first.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient(registry_uri="databricks-uc")
    full_name = f"{catalog}.{schema}.{model_name}"
    try:
        mv = client.get_model_version_by_alias(name=full_name, alias=alias)
    except Exception as e:
        raise RuntimeError(f"Failed to resolve @{alias} for {full_name}: {e}") from e
    return str(mv.version)


def capture_previous_champion(
    catalog: str,
    schema: str,
    model_name: str,
) -> str | None:
    """Return the version currently aliased as @champion, or None if no @champion yet."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient(registry_uri="databricks-uc")
    full_name = f"{catalog}.{schema}.{model_name}"
    try:
        mv = client.get_model_version_by_alias(name=full_name, alias="champion")
        return str(mv.version)
    except Exception:
        return None


def smoke_test_endpoint(
    endpoint_name: str,
    image_bytes: bytes,
) -> tuple[bool, str | None]:
    """Send a single base64-encoded image to the endpoint, return (success, error_msg)."""
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import DataframeSplitInput

        w = WorkspaceClient()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        response = w.serving_endpoints.query(
            name=endpoint_name,
            dataframe_split=DataframeSplitInput(columns=["image"], data=[[b64]]),
        )
        # Response shape varies by endpoint; just verify we got predictions back
        preds = getattr(response, "predictions", None)
        if preds is None:
            return False, "No predictions in response"
        return True, None
    except Exception as e:
        return False, f"Smoke test exception: {e}"


def deploy_and_smoke_test(
    endpoint_name: str,
    catalog: str,
    schema: str,
    model_name: str,
    candidate_alias: str = ALIAS_CANDIDATE,
    workload_size: str = "Small",
    workload_type: str = "GPU_SMALL",
    scale_to_zero: bool = True,
    ai_gateway_enabled: bool = True,
    inference_table_prefix: str | None = None,
    smoke_image_bytes: bytes | None = None,
    timeout_seconds: int = 600,
    promote_on_success: bool = True,
    model_version: str | None = None,
) -> EndpointDeployResult:
    """Resolve a version, deploy/update endpoint, smoke-test, and optionally
    promote to @champion.

    When ``model_version`` is given it is deployed directly (the cross-schema
    promote task already knows the prod version it copied); otherwise the
    ``candidate_alias`` (``@challenger`` by default) is resolved to a numeric
    version. ``promote_on_success=True`` sets ``@champion`` on
    ``{catalog}.{schema}.{model_name}`` after a passing smoke test, so the
    promote task can deploy the prod-schema version AND flip its champion alias
    in one call (and leave @champion untouched on failure).

    Uses Databricks SDK's create_and_wait / update_config_and_wait pattern.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        AiGatewayConfig,
        AiGatewayInferenceTableConfig,
        EndpointCoreConfigInput,
        EndpointTag,
        ServedEntityInput,
        ServingModelWorkloadType,
    )

    w = WorkspaceClient()

    # 1. Resolve the version to deploy: an explicit numeric version (prod
    #    promote path) wins; otherwise resolve the dev @challenger alias.
    full_model = f"{catalog}.{schema}.{model_name}"
    if model_version is not None:
        version = str(model_version)
        logger.info("Deploying explicit version %s for %s", version, full_model)
    else:
        version = resolve_alias_to_version(catalog, schema, model_name, candidate_alias)
        logger.info("Resolved @%s -> version %s for %s", candidate_alias, version, full_model)

    # 2. Capture previous @champion for potential rollback
    previous = capture_previous_champion(catalog, schema, model_name)
    logger.info("Previous @champion: %s", previous)

    # 3. Build served entity referencing NUMERIC version
    workload_type_enum = ServingModelWorkloadType(workload_type) if isinstance(workload_type, str) else workload_type
    served_entity = ServedEntityInput(
        name=model_name,
        entity_name=full_model,
        entity_version=version,
        workload_size=workload_size,
        workload_type=workload_type_enum,
        scale_to_zero_enabled=scale_to_zero,
    )
    config = EndpointCoreConfigInput(name=endpoint_name, served_entities=[served_entity])

    # 4. AI Gateway top-level (NOT nested under config)
    ai_gateway = None
    if ai_gateway_enabled:
        prefix = inference_table_prefix or f"{model_name}_inference"
        ai_gateway = AiGatewayConfig(
            inference_table_config=AiGatewayInferenceTableConfig(
                catalog_name=catalog,
                schema_name=schema,
                table_name_prefix=prefix,
                enabled=True,
            ),
        )

    tags = [EndpointTag(key="project", value="dais26-vfm")]

    # 5. Create or update endpoint
    try:
        existing = w.serving_endpoints.get(name=endpoint_name)
        logger.info("Endpoint %s exists (state=%s); updating config", endpoint_name, existing.state)
        w.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=[served_entity],
            timeout=timedelta(seconds=timeout_seconds),
        )
        # AI Gateway config updates go through put_ai_gateway after the endpoint exists
        if ai_gateway is not None:
            w.serving_endpoints.put_ai_gateway(
                name=endpoint_name,
                inference_table_config=ai_gateway.inference_table_config,
            )
    except Exception as e:
        msg = str(e).lower()
        if "does not exist" in msg or "not found" in msg or "resourcenotfound" in msg:
            logger.info("Endpoint %s does not exist; creating", endpoint_name)
            w.serving_endpoints.create_and_wait(
                name=endpoint_name,
                config=config,
                ai_gateway=ai_gateway,
                tags=tags,
                timeout=timedelta(seconds=timeout_seconds),
            )
        else:
            return EndpointDeployResult(
                endpoint_name=endpoint_name,
                deployed_version=version,
                previous_champion=previous,
                smoke_test_passed=False,
                promoted_to_champion=False,
                state="ERROR",
                error=str(e),
            )

    # 6. Poll until READY (create_and_wait/update_config_and_wait should already block,
    #    but double-check)
    deadline = time.time() + timeout_seconds
    state = "UNKNOWN"
    while time.time() < deadline:
        ep = w.serving_endpoints.get(name=endpoint_name)
        state = str(getattr(ep.state, "ready", getattr(ep, "state", "UNKNOWN")))
        if "READY" in state.upper():
            break
        if "FAILED" in state.upper():
            return EndpointDeployResult(
                endpoint_name=endpoint_name,
                deployed_version=version,
                previous_champion=previous,
                smoke_test_passed=False,
                promoted_to_champion=False,
                state=state,
                error=f"Endpoint entered FAILED state: {state}",
            )
        time.sleep(10)
    else:
        return EndpointDeployResult(
            endpoint_name=endpoint_name,
            deployed_version=version,
            previous_champion=previous,
            smoke_test_passed=False,
            promoted_to_champion=False,
            state=state,
            error=f"Timeout waiting for READY after {timeout_seconds}s; last state: {state}",
        )

    # 7. Smoke test (with provided image bytes; if None, use a synthetic 224x224 black PNG)
    if smoke_image_bytes is None:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (224, 224), (0, 0, 0)).save(buf, format="PNG")
        smoke_image_bytes = buf.getvalue()

    ok, err = smoke_test_endpoint(endpoint_name, smoke_image_bytes)
    if not ok:
        logger.error("Smoke test failed for %s v%s: %s", endpoint_name, version, err)
        return EndpointDeployResult(
            endpoint_name=endpoint_name,
            deployed_version=version,
            previous_champion=previous,
            smoke_test_passed=False,
            promoted_to_champion=False,
            state=state,
            error=err,
        )

    # 8. Promote @candidate -> @champion (atomic alias overwrite via MLflow)
    promoted = False
    if promote_on_success:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_registry_uri("databricks-uc")
            client = MlflowClient(registry_uri="databricks-uc")
            client.set_registered_model_alias(name=full_model, alias="champion", version=version)
            promoted = True
            logger.info("Promoted version %s to @champion (previous: %s)", version, previous)
        except Exception as e:
            logger.error("Failed to promote @champion: %s", e)
            return EndpointDeployResult(
                endpoint_name=endpoint_name,
                deployed_version=version,
                previous_champion=previous,
                smoke_test_passed=True,
                promoted_to_champion=False,
                state=state,
                error=f"Smoke OK but promotion failed: {e}",
            )

    return EndpointDeployResult(
        endpoint_name=endpoint_name,
        deployed_version=version,
        previous_champion=previous,
        smoke_test_passed=True,
        promoted_to_champion=promoted,
        state=state,
    )
