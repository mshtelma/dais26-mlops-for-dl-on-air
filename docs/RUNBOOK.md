# Runbook

Operational procedures for the DAIS26 VFM showcase: pre-demo preparation, rollback, DINOv2 fallback,
and service principal setup.

---

## Pre-demo D-1 checklist

Run this checklist the day before the talk (Sunday June 14). Estimated time: 45 minutes.

### 30 minutes before talk (D-0 morning)

- [ ] Run `python scripts/warmup_endpoints.py` — sends 5 sample requests to each endpoint to
      prevent cold-start latency during the demo

- [ ] Verify endpoint is `READY`:

  ```bash
  databricks serving-endpoints get dais26-cradio-detector-prod | jq .state
  # Expected: {"ready": "READY", "config_update": "NOT_UPDATING"}
  ```

- [ ] Start the latency probe loop on your talk laptop (runs every 60s):

  ```bash
  bash scripts/latency_probe.sh
  ```

  The probe will log probe failures. Two consecutive failures (60s apart) is the trigger to
  switch to the backup video for that demo segment.

- [ ] Verify `@champion` alias resolves:

  ```python
  from mlflow.tracking import MlflowClient
  client = MlflowClient(registry_uri="databricks-uc")
  mv = client.get_model_version_by_alias("ml.dais26_vfm.cradio_detector", "champion")
  print(f"@champion = version {mv.version}")
  ```

### D-1 (full day)

- [ ] Load backup videos in OBS: 6 segments, ~90 seconds each

  | Segment | Demo covered |
  |---------|-------------|
  | `seg1_hook.mp4` | DENTEX X-ray + model finding caries |
  | `seg2_detection.mp4` | Training run (live epochs 1-2 + pre-baked 3-10) |
  | `seg3_similarity.mp4` | Vector Search top-10 results |
  | `seg4_drift.mp4` | Synthetic drift KNN distance chart |
  | `seg5_serving.mp4` | curl to detector endpoint + response |
  | `seg6_lineage.mp4` | UC lineage diagram |

- [ ] Verify notebook cells have cached outputs committed (so "Run All" shows results even without
      compute — key fallback if cluster fails to start)

- [ ] Run full 45-minute rehearsal end-to-end against the prod workspace

- [ ] Confirm Vector Search index is synced:

  ```bash
  databricks vector-search indexes get ml.dais26_vfm.embeddings_index | jq .status
  # Expected: {"detailed_state": "ONLINE", "ready": true}
  ```

- [ ] Confirm drift_scores table has recent rows:

  ```sql
  SELECT * FROM ml.dais26_vfm.drift_scores ORDER BY timestamp DESC LIMIT 3
  ```

- [ ] Tag the release:

  ```bash
  git tag v1.0.0-dais26
  git push origin v1.0.0-dais26
  ```

---

## Switch-to-video procedure

**Trigger:** 2 consecutive failed latency probes (60 seconds apart).

**Action:**
1. Announce to audience: "I'm going to switch to a recorded segment for this section so we keep to time."
2. In OBS: switch to the pre-loaded backup scene for the current demo segment.
3. Play the 90-second backup video.
4. Continue the talk narrative verbally — the content is identical to what the live demo would show.
5. Do not attempt to recover the live demo mid-talk. Continue with video for remaining segments if
   the endpoint is still failing.

**After the talk:** diagnose endpoint state via the Mosaic AI serving metrics dashboard.

---

## Rollback procedure

Use this when a newly promoted `@champion` version is producing bad results and you need to revert
to the previous version.

### Step 1 — Find the previous champion version

Before any promotion, the training job logs the previous champion version to the job run output.
Check the `deploy_endpoint` task logs for a line like:
```
Previous @champion was version 2. Promoting version 3.
```

Alternatively, list model versions:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions("name='ml.dais26_vfm.cradio_detector'")
for v in versions:
    print(f"version={v.version}, aliases={v.aliases}, status={v.status}")
```

### Step 2 — Set `@champion` back to the previous version

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")

PREVIOUS_VERSION = "2"  # captured from task logs or search above

client.set_registered_model_alias(
    name="ml.dais26_vfm.cradio_detector",
    alias="champion",
    version=PREVIOUS_VERSION,
)
print(f"@champion reset to version {PREVIOUS_VERSION}")
```

### Step 3 — Re-deploy the endpoint with the rollback version

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()

PREVIOUS_VERSION = "2"  # same value as above

w.serving_endpoints.update_config_and_wait(
    name="dais26-cradio-detector-prod",
    served_entities=[
        ServedEntityInput(
            name="detector",
            entity_name="ml.dais26_vfm.cradio_detector",
            entity_version=PREVIOUS_VERSION,
            workload_size="Small",
            workload_type="GPU_SMALL",
            scale_to_zero_enabled=False,
        )
    ],
)
print("Rollback complete. Endpoint updated to previous version.")
```

### Step 4 — Verify

```bash
databricks serving-endpoints get dais26-cradio-detector-prod | jq '.config.served_entities[0].entity_version'
# Should print: "2"
```

---

## DINOv2 fallback 6-step runbook

Use this if C-RADIOv4 has a breaking change or is yanked from HuggingFace before the conference.

**Important:** DINOv2-base is NOT a drop-in swap. It has different dimensions:
- `summary_dim = 768` (vs 1152 for C-RADIOv4)
- `spatial_dim = 768` (vs 1536 for C-RADIOv4)

All downstream artifacts must be rebuilt. Budget approximately 2 hours.

**Pre-baked shortcut:** Phase 1 Day 2-3 runs `scripts/pin_model_cache.py`, which generates a DINOv2
fallback head checkpoint stored in the `model_cache` UC Volume. If this checkpoint exists, skip
step 1 (saves ~15 minutes).

### Step 1 — Retrain detection head with DINOv2 backbone

```bash
databricks bundle run train_detector -t dev \
  --params train_epochs=10,backbone=dinov2_base
```

If the pre-baked fallback checkpoint exists in the Volume, the training script loads it directly and
skips training. Check the task log for:
```
Found pre-baked DINOv2 fallback checkpoint at /Volumes/.../model_cache/dinov2_fallback_head.pt
Loading pre-baked checkpoint. Skipping training.
```

### Step 2 — Recompute embeddings at dim=768

```bash
databricks bundle run precompute_embeddings -t dev \
  --params backbone=dinov2_base
```

This rewrites `ml_dev.dais26_vfm.train_embeddings` with 768-dim `ARRAY<FLOAT>` vectors.

### Step 3 — Recreate the Vector Search index with `embedding_dimension=768`

The existing index at dim=1152 must be dropped and recreated:

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import DeltaSyncVectorIndexSpecRequest

w = WorkspaceClient()
catalog = "ml_dev"

# Drop old index
try:
    w.vector_search_indexes.delete_index(
        index_name=f"{catalog}.dais26_vfm.embeddings_index"
    )
    print("Old index dropped.")
except Exception as e:
    print(f"Drop failed (may not exist): {e}")

# Recreate with dim=768
w.vector_search_indexes.create_index(
    name=f"{catalog}.dais26_vfm.embeddings_index",
    endpoint_name="dais26-vfm-vs-endpoint",
    primary_key="image_id",
    index_type="DELTA_SYNC",
    delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
        source_table=f"{catalog}.dais26_vfm.train_embeddings",
        embedding_vector_column="embedding",
        embedding_dimension=768,            # DINOv2-base summary_dim
        pipeline_type="TRIGGERED",
        columns_to_sync=["image_id", "diagnosis", "split"],
    ),
)
print("New index created at dim=768.")
```

Alternatively, use the "fallback" widget in `notebooks/04_deploy_serving.py`.

### Step 4 — Regenerate drift reference distribution

The drift monitor reads `backbone_info.summary_dim` from `BackboneInfo` automatically. With
`backbone=dinov2_base`, the reference will be built from 768-dim embeddings. No manual change needed
beyond running the monitor once with the new backbone parameter:

```bash
databricks bundle run drift_monitor -t dev \
  --params backbone=dinov2_base
```

### Step 5 — Deploy endpoint with the new DINOv2-trained model version

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
mv = client.get_model_version_by_alias("ml_dev.dais26_vfm.cradio_detector", "candidate")
dinov2_version = mv.version

w = WorkspaceClient()
w.serving_endpoints.update_config_and_wait(
    name="dais26-cradio-detector-dev",
    served_entities=[
        ServedEntityInput(
            name="detector",
            entity_name="ml_dev.dais26_vfm.cradio_detector",
            entity_version=dinov2_version,
            workload_size="Small",
            workload_type="GPU_SMALL",
            scale_to_zero_enabled=True,
        )
    ],
)
```

### Step 6 — Update `@champion` alias to the DINOv2 version

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(
    name="ml_dev.dais26_vfm.cradio_detector",
    alias="champion",
    version=dinov2_version,
)
print(f"@champion updated to DINOv2 version {dinov2_version}.")
```

**Talk narrative adjustment:** Update the "BackboneInfo contract" slide to show `summary_dim=768`
instead of 1152. The three-jobs story is identical; only the dimensions change.

---

## Service principal creation

Required for the `prod` target (`run_as` in `databricks.yml`).

### Step 1 — Create the service principal

```bash
SP_RESPONSE=$(databricks service-principals create \
  --display-name dais26-vfm-sp \
  --output JSON)

SP_APP_ID=$(echo "$SP_RESPONSE" | jq -r '.applicationId')
echo "Service Principal Application ID: $SP_APP_ID"
# Save this value — you will need it in the next steps.
```

The application ID is a UUID (e.g., `a1b2c3d4-e5f6-7890-abcd-ef1234567890`). It is **not** the
display name `dais26-vfm-sp`. The DAB `run_as.service_principal_name` field requires the
application ID.

### Step 2 — Set the DAB variable

Either export an environment variable:

```bash
export DATABRICKS_SP_APP_ID="$SP_APP_ID"
```

Or set the default in `databricks.yml`:

```yaml
variables:
  sp_app_id:
    default: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"   # your SP app ID
```

Or pass it at deploy time:

```bash
databricks bundle deploy -t prod --var sp_app_id="$SP_APP_ID"
```

### Step 3 — Grant UC privileges

The setup notebook (`notebooks/00_setup.py`) runs these grants automatically during the `setup` task.
To run them manually:

```bash
databricks sql execute --statement "
GRANT USE CATALOG ON CATALOG ml TO \`$SP_APP_ID\`;
GRANT USE SCHEMA ON SCHEMA ml.dais26_vfm TO \`$SP_APP_ID\`;
GRANT CREATE TABLE ON SCHEMA ml.dais26_vfm TO \`$SP_APP_ID\`;
GRANT MODIFY, SELECT ON SCHEMA ml.dais26_vfm TO \`$SP_APP_ID\`;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME ml.dais26_vfm.dentex_raw TO \`$SP_APP_ID\`;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME ml.dais26_vfm.model_cache TO \`$SP_APP_ID\`;
GRANT CREATE MODEL ON SCHEMA ml.dais26_vfm TO \`$SP_APP_ID\`;
GRANT EXECUTE ON MODEL ml.dais26_vfm.cradio_detector TO \`$SP_APP_ID\`;
"
```

### Step 4 — Grant inference table access (deferred)

The AI Gateway auto-creates the inference table on the first endpoint request. Run this script
**after** the first inference request has flowed through:

```bash
python scripts/grant_inference_table_access.py \
  --catalog ml \
  --schema dais26_vfm \
  --sp-app-id "$SP_APP_ID"
```

This grants `SELECT` on `ml.dais26_vfm.detector_inference_<suffix>`. The table does not exist until
after the first request, so the grant cannot be applied earlier.

### Azure note

On Azure Databricks, service principals are backed by Microsoft Entra ID (formerly Azure AD).
Options:
1. Pre-create the service principal in Entra ID, then import by display name in Databricks:
   `databricks service-principals get-by-display-name dais26-vfm-sp`
2. Use the Databricks UI (Settings → Service Principals → Add service principal) and copy the
   application ID displayed there.

The `applicationId` used in DAB `run_as` must match the Entra App ID, not the Databricks internal ID.

---

## Inference table access setup

After the detection endpoint serves its first request, grant the service principal access to the
auto-created inference table:

```bash
# Find the actual table name (suffix added by AI Gateway)
databricks tables list --catalog ml --schema dais26_vfm \
  | grep detector_inference

# Grant access
databricks sql execute --statement "
GRANT SELECT ON TABLE ml.dais26_vfm.detector_inference_<suffix> TO \`$SP_APP_ID\`
"
```

Or use the helper script which automates the discovery:

```bash
python scripts/grant_inference_table_access.py
```

---

## GPU memory validation

After the detection endpoint is deployed (Phase 2), validate GPU memory utilization:

```bash
python scripts/probe_endpoint_gpu.py
```

This sends 5 warm-up requests and checks the endpoint metrics dashboard.

**Acceptance criterion:** GPU memory utilization at idle <= 85% on `GPU_SMALL`.

If idle utilization exceeds 85%, escalate the workload type:

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput

w = WorkspaceClient()
w.serving_endpoints.update_config_and_wait(
    name="dais26-cradio-detector-prod",
    served_entities=[
        ServedEntityInput(
            name="detector",
            entity_name="ml.dais26_vfm.cradio_detector",
            entity_version="<current_champion_version>",
            workload_size="Small",
            workload_type="GPU_MEDIUM",   # escalated from GPU_SMALL
            scale_to_zero_enabled=False,
        )
    ],
)
```

Then re-run `scripts/probe_endpoint_gpu.py` and update `docs/BENCHMARKS.md` with the new baseline.
