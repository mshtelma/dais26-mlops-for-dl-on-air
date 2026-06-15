# 7 · Rollback

When a newly promoted `@champion` is producing bad results, revert to the previous version. The
two-alias safety (`@champion_candidate` → `@champion`) means the prior champion kept serving until
the candidate passed its smoke test — but if a *passed* champion later proves bad in production,
roll back manually.

!!! info "Where `@champion` lives"
    `@champion` is on the **prod** model `CHAMPION_CATALOG.CHAMPION_SCHEMA.detector_champion`. The
    examples below use placeholder names — substitute your env's champion model and endpoint
    (`dais26-detector-champion`).

## Step 1 — Find the previous champion version

Before each promotion, `deploy_champion`/`deploy_endpoint` logs the previous champion to the task
output:

```
Previous @champion was version 2. Promoting version 3.
```

Or list versions and their aliases:

```python
from mlflow.tracking import MlflowClient
c = MlflowClient(registry_uri="databricks-uc")
for v in c.search_model_versions("name='<champion_catalog>.<champion_schema>.detector_champion'"):
    print(f"version={v.version}, aliases={v.aliases}, status={v.status}")
```

## Step 2 — Point `@champion` back

```python
from mlflow.tracking import MlflowClient
c = MlflowClient(registry_uri="databricks-uc")
PREVIOUS_VERSION = "2"
c.set_registered_model_alias(
    name="<champion_catalog>.<champion_schema>.detector_champion",
    alias="champion", version=PREVIOUS_VERSION)
print(f"@champion reset to version {PREVIOUS_VERSION}")
```

## Step 3 — Re-deploy the endpoint with the rollback version

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput
w = WorkspaceClient()
PREVIOUS_VERSION = "2"
w.serving_endpoints.update_config_and_wait(
    name="dais26-detector-champion",
    served_entities=[ServedEntityInput(
        name="detector",
        entity_name="<champion_catalog>.<champion_schema>.detector_champion",
        entity_version=PREVIOUS_VERSION,
        workload_size="Small", workload_type="GPU_SMALL",
        scale_to_zero_enabled=False)])
print("Rollback complete.")
```

`update_config_and_wait` is a **zero-downtime** rolling update — the old served entity keeps
serving until the rollback version is READY.

## Step 4 — Verify

```bash
databricks serving-endpoints get dais26-detector-champion \
  | jq '.config.served_entities[0].entity_version'
# → "2"
```

Then smoke-test the endpoint ([Serving smoke test](../scenarios/serving-smoke-test.md)) and, if
the rolled-back architecture differs, refresh the downstream artifacts
([Embeddings → VS → drift](embeddings-vector-search-drift.md)).

## Related failure modes

- **Champion deploy left `@champion` unset** — the smoke test failed, so `@challenger` /
  `@champion_candidate` remains staged and the prior champion still serves. Check the
  `deploy_champion` task logs; re-run or promote manually.
- **Auto-`@candidate`/`@challenger` aliases pointed at a sub-best run** — happened during the
  HPO campaign (`register_winner=True` aliased whatever registered last). Register the intended
  run and set the alias explicitly. See [HPO campaign log → champion registration](../HPO.md).
- **Registered model serves garbage** (e.g. a serialization break) — re-register from a known-good
  run; don't trust a served number you didn't re-eval through the pyfunc. See
  [Operations & runbook](../RUNBOOK.md).

Full operational procedures (pre-demo checklist, switch-to-video, DINOv2 fallback) live in
**[Operations & runbook](../RUNBOOK.md)**.
