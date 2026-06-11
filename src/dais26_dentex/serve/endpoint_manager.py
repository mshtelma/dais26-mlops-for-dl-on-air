from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

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


def archive_orphaned_inference_table(
    spark: Any,
    catalog: str,
    schema: str,
    prefix: str,
) -> str | None:
    """Rename a leftover AI Gateway payload table so a fresh endpoint can recreate it.

    AI Gateway ALWAYS auto-creates ``{prefix}_payload`` on endpoint create/update and
    refuses to reuse a pre-existing table ("Table ... already exists. Please specify a
    different table prefix."). When a prior endpoint was torn down, its payload table is
    orphaned (no writer) but still occupies the canonical name, hard-failing the next
    create. This renames the orphan to ``{prefix}_payload_archived_<utc_ts>`` — preserving
    every row — and frees the canonical name. Returns the archived table name, or None if
    there was nothing to archive.

    The caller MUST ensure the endpoint does not currently exist before calling this:
    renaming a *live* inference table corrupts logging (per Databricks AI Gateway docs).
    """
    payload = f"{catalog}.{schema}.{prefix}_payload"
    try:
        if not spark.catalog.tableExists(payload):
            return None
    except Exception as e:  # noqa: BLE001 — a lookup failure just means "nothing to archive"
        logger.warning("Could not check for orphaned inference table %s: %s", payload, e)
        return None
    ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    archived = f"{catalog}.{schema}.{prefix}_payload_archived_{ts}"
    logger.info("Archiving orphaned inference table %s -> %s", payload, archived)
    spark.sql(f"ALTER TABLE {payload} RENAME TO {archived}")
    return archived


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
    timeout_seconds: int = 5400,
    promote_on_success: bool = True,
    model_version: str | None = None,
    spark: Any = None,
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

    # 4. AI Gateway top-level (NOT nested under config). The payload-logging table
    #    is canonically named {prefix}_payload.
    prefix = inference_table_prefix or f"{model_name}_inference"
    ai_gateway = None
    if ai_gateway_enabled:
        ai_gateway = AiGatewayConfig(
            inference_table_config=AiGatewayInferenceTableConfig(
                catalog_name=catalog,
                schema_name=schema,
                table_name_prefix=prefix,
                enabled=True,
            ),
        )

    tags = [EndpointTag(key="project", value="dais26-vfm")]

    # 5. Create or update endpoint.
    #    Serving endpoints reject a config update while a prior update is still in
    #    flight ("served entities are currently being updated"). A first-time GPU
    #    deploy can stay IN_PROGRESS for many minutes, so a re-run or an overlapping
    #    deploy_champion trigger would hit that transient and hard-fail. Wait for any
    #    in-flight update to settle first, and treat the transient as retryable rather
    #    than fatal so the champion deploy is idempotent across re-runs.
    def _endpoint_exists() -> bool:
        try:
            w.serving_endpoints.get(name=endpoint_name)
            return True
        except Exception as e:
            msg = str(e).lower()
            if "does not exist" in msg or "not found" in msg or "resourcenotfound" in msg:
                return False
            raise

    def _wait_until_not_updating(timeout: int) -> None:
        """Block until the endpoint is not mid config-update.

        A cold GPU deploy of a multi-GB model can stay IN_PROGRESS for ~1 hour,
        so this must be allowed to wait that long (``timeout`` ~= 90 min) — a
        short wait here is exactly what caused re-runs to fall through and
        collide with the still-in-flight update ("currently being updated").
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            ep = w.serving_endpoints.get(name=endpoint_name)
            cu = str(getattr(ep.state, "config_update", "") or "").upper()
            # IN_PROGRESS is the only "actively updating" config_update state; every
            # other state (NOT_UPDATING, UPDATE_FAILED, UPDATE_CANCELED) means it has
            # settled and we can proceed. Do NOT substring-match "UPDATING" here:
            # "UPDATING" is a substring of "NOT_UPDATING", so that check treated the
            # idle endpoint as still-updating and looped for the full `timeout`
            # (~90 min), never reaching the config update.
            if "IN_PROGRESS" not in cu:
                return
            logger.info("Endpoint %s config_update=%s; waiting to settle", endpoint_name, cu)
            time.sleep(30)

    def _current_ready_version() -> str | None:
        """Return the active served version iff the endpoint is READY on it.

        Lets a re-run / repair be idempotent: if a prior (possibly hour-long)
        update already brought the endpoint up on the target version, we skip
        issuing another update and go straight to the smoke test instead of
        colliding with / needlessly restarting the rollout.
        """
        try:
            ep = w.serving_endpoints.get(name=endpoint_name)
        except Exception:
            return None
        if "READY" not in str(getattr(ep.state, "ready", "") or "").upper():
            return None
        cfg = getattr(ep, "config", None)
        for se in (getattr(cfg, "served_entities", None) or []) if cfg else []:
            # Match on the FULL model identity, not just the version number: the
            # endpoint may currently serve a different model (e.g. the dev
            # `dinov3_detector` v9) whose version int could coincide with the prod
            # target version and cause us to wrongly skip the update.
            if getattr(se, "entity_name", None) != full_model:
                continue
            v = getattr(se, "entity_version", None)
            if v is not None:
                return str(v)
        return None

    def _is_transient_update(m: str) -> bool:
        return (
            "currently being updated" in m
            or "update is no longer in progress" in m
            or "try again" in m
        )

    try:
        if not _endpoint_exists():
            logger.info("Endpoint %s does not exist; creating", endpoint_name)
            # The endpoint is absent, so any existing {prefix}_payload table is an
            # orphan from a previously torn-down endpoint. AI Gateway always auto-
            # creates that table on endpoint create and won't reuse an existing one,
            # so archive (rename) the orphan first to avoid the "table already exists /
            # specify a different table prefix" failure — without losing its rows.
            # Only safe because no endpoint is writing to it right now.
            if ai_gateway is not None and spark is not None:
                archive_orphaned_inference_table(spark, catalog, schema, prefix)
            w.serving_endpoints.create_and_wait(
                name=endpoint_name,
                config=config,
                ai_gateway=ai_gateway,
                tags=tags,
                timeout=timedelta(seconds=timeout_seconds),
            )
        else:
            # Wait out any in-flight update (a cold GPU deploy can take ~1h), then
            # be idempotent: if the endpoint already serves the target version and is
            # READY, skip the update entirely. Otherwise issue the update, retrying the
            # "currently being updated" transient (each retry re-waits for it to settle).
            _wait_until_not_updating(timeout_seconds)
            if _current_ready_version() == version:
                logger.info(
                    "Endpoint %s already serving v%s and READY; skipping update",
                    endpoint_name,
                    version,
                )
            else:
                last_err: Exception | None = None
                for attempt in range(5):
                    _wait_until_not_updating(timeout_seconds)
                    try:
                        logger.info(
                            "Updating config for %s -> v%s (attempt %d)",
                            endpoint_name,
                            version,
                            attempt + 1,
                        )
                        w.serving_endpoints.update_config_and_wait(
                            name=endpoint_name,
                            served_entities=[served_entity],
                            timeout=timedelta(seconds=timeout_seconds),
                        )
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        if not _is_transient_update(str(e).lower()):
                            raise
                        logger.info(
                            "Transient endpoint-update conflict (%s); waiting to retry",
                            str(e)[:120],
                        )
                        time.sleep(30)
                if last_err is not None:
                    raise last_err
            # AI Gateway config updates go through put_ai_gateway after the endpoint exists
            if ai_gateway is not None:
                w.serving_endpoints.put_ai_gateway(
                    name=endpoint_name,
                    inference_table_config=ai_gateway.inference_table_config,
                )
    except Exception as e:
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

    # 8. Promote @challenger -> @champion (atomic alias overwrite via MLflow)
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
