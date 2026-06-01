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
- `summary_dim = 768` (vs 1152 for C-RADIOv4-SO400M)
- `spatial_dim = 768` (vs 1152 for C-RADIOv4-SO400M)

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

---

## Engineering rationale

This section is the long-form home for "why we did it this way" notes that
otherwise bloat source-file comments. Code references here use stable anchors
(`#hf-cache-race`, `#hf-transfer-fuse-incompat`, `#ddp-trainable-only`,
`#pip-requirements-rationale`) so a one-line pointer in code keeps working as
the surrounding text evolves.

### HF cache race {#hf-cache-race}

**Symptom (legacy).** Cold-cache multi-rank runs of `AutoModel.from_pretrained`
race on the trust_remote_code modules cache and on the weight shards. Survivor
ranks see partially-written files and crash with `FileNotFoundError` —
sometimes silently corrupting the cache for the next run.

**Why naive `barrier()` doesn't fix it.** The obvious shape

```python
if is_rank0():
    do_side_effect()
barrier()   # other ranks waited
use_side_effect()
```

still races: every rank then re-enters the same `from_pretrained` call,
which itself does a fresh cache lookup + (sometimes) a partial download. The
"use" step IS the race.

**Fix.** Sequence-matched NCCL barriers — non-rank-0 hits its barrier *first*
and waits, rank 0 performs the work and then hits its barrier. NCCL pairs
calls by call-order, so the pairing is deterministic regardless of how many
collectives the body itself does. Codified in
`src/dais26_dentex/distributed/barrier_dance.py::rank0_first`. Used at one
site today (`models/builder.py::build_detector`) and reusable for cache
pre-warm, dataset stage-in, registration locks.

### HF transfer / UC Volume FUSE incompatibility {#hf-transfer-fuse-incompat}

**Symptom.** With `HF_HUB_ENABLE_HF_TRANSFER=1` and `cache_dir` pointing at a
UC Volume FUSE mount, `huggingface_hub` downloads fail with
`Io: Input/output error (os error 5) (no permits available)` or
`Io: Operation not supported (os error 95)`. The `hf-xet` backend hits the
same failure shape on FUSE.

**Cause.** UC Volume FUSE rejects the parallel chunked writers used by both
`hf_transfer` (Rust binary) and `hf-xet` (CAS service client). FUSE supports
sequential writes only.

**Fix.** Force `HF_HUB_ENABLE_HF_TRANSFER=0` and `HF_HUB_DISABLE_XET=1` before
any HF library imports — both libraries read these once at constants-module
import time, so setting them later is a no-op. Single canonical site:
`src/dais26_dentex/platform/hf_env.py::configure_hf_env`. The `data/dentex_loader.py`
module-level `setdefault` block handles dataset downloads where
`configure_hf_env` is not in the call path.

In notebooks, the env vars must be set inside the `@distributed` worker body,
*before* `from dais26_dentex...` because cloudpickle resolves free variables
on the worker eagerly. See `notebooks/02_train_detector_air.py` for the
canonical pattern.

### DDP trainable-only {#ddp-trainable-only}

**Current.** `TrainerConfig.ddp_find_unused = True` because the backbone is
frozen and DDP's reducer needs to know not to wait on grads from the frozen
subtree. The cost is a per-iteration reducer scan over the entire backbone
(inexpensive for a single-GPU smoke run, measurable on 8×H100).

**Cleaner shape (deferred past Phase 5).** Pass only `requires_grad=True`
parameters into the optimizer *and* hide the frozen subtree from DDP by
wrapping only the head + FPN + (optional) LoRA parameters in DDP. With the
frozen backbone outside DDP, `find_unused_parameters` flips to `False` and
the per-iter reducer scan goes away.

**Why deferred.** This is structural surgery: it changes the model-build
shape, the state-dict layout (head/FPN are no longer accessed via `model.module.`),
and the `DistributedDataParallel(...)` construction site. It needs an A/B run
on 8 H100s before defaulting on. The current `find_unused_parameters=True`
is correct, just not optimal — there is no correctness regression to chase.

Tracked as a follow-up; flip the default by setting `ddp_find_unused: bool = False`
once the trainable-only DDP wrap is in place and the 1-epoch val/loss matches
within 5%.

### `pip_requirements` source of truth {#pip-requirements-rationale}

**Symptom (legacy).** The training pipeline hardcoded a `pip_requirements`
list inside `train_detector.py`. The serving wheel's `pyproject.toml` had a
parallel list. Adding a runtime dep (e.g. `timm` for C-RADIOv4
trust_remote_code) required editing both — and the most common failure mode
was forgetting the second site, which surfaced as a serving-endpoint
deploy-time `ImportError`.

**Fix.** Single source of truth in `pyproject.toml`:

```toml
[tool.dais26.serving-deps]
detector = ["torch", "torchvision", "transformers", ...]
```

`platform/mlflow_io.py::serving_pip_requirements` reads this table at log
time. CI guards it via `assert_serving_reqs_match_pyproject` — a non-empty
list of strings is the syntactic contract; semantic completeness is verified
by the CI smoke `mlflow models predict` against the logged pyfunc.

**Wheel-bundled `pyproject.toml`.** The historical `_find_pyproject` walked
up from `__file__` looking for `pyproject.toml`, which works against the
source tree (notebooks, pytest) but fails inside AIR's ephemeral env where
the package is installed under
`/local_disk0/.ephemeral_nfs/envs/.../site-packages/dais26_dentex/...` —
none of those parents contain a `pyproject.toml`, so log-time raises
`FileNotFoundError: Could not locate pyproject.toml`. The fix ships
`pyproject.toml` inside the wheel as `dais26_dentex/_pyproject.toml`
(hatchling `force-include`) and `_find_pyproject` consults
`importlib.resources` first, falling back to the source-tree walk for
editable installs. Verify with:

```bash
python -m zipfile -l dist/*.whl | grep _pyproject.toml
# → dais26_dentex/_pyproject.toml ...
```

**MLflow API drift.** The `name=` / `artifact_path=` rename across MLflow
minor versions is detected once at import via `inspect.signature` rather than
paid per-call as a `try/except TypeError`. See `_log_model_artifact_kwarg()`
in the same module.

### sgcli launch troubleshooting {#sgcli-launch}

The terminal launch path (`sgcli/workload_train_detector.yaml`) snapshots
the repo, runs `pip install .`, then `torchrun --nproc_per_node=$GPUS_PER_NODE
-m dais26_dentex.train.cli`. The `cli` entrypoint reads
`$HYPERPARAMETERS_PATH` (or `--config`), builds a `TrainerConfig`, and
dispatches to `Trainer.run()`. Common failures:

| Symptom | Cause | Fix |
|---|---|---|
| `MODEL_URI=` missing from rank 0 stdout | Training succeeded on non-rank-0 ranks, rank 0 crashed in `_save_and_register` | Inspect rank-0 logs (`sgcli get logs <run-id> --rank 0`); MlflowReporter raises typed `AliasingError` instead of swallowing |
| `ModuleNotFoundError: serverless_gpu` in cli flow | Don't need it — the cli is the torchrun path, not the `@distributed` path | Confirm you're running `sgcli`/`torchrun`, not the notebook; `serverless_gpu` is only the notebook decorator |
| Hyperparameters from yaml ignored | yaml top-level shape mismatch | The sgcli yaml has `env_variables:` and `parameters:` as siblings — `parameters` lands at `$HYPERPARAMETERS_PATH` as JSON; `TrainerConfig.from_dict` validates fields and raises with the field-by-field error list |
| `experiment_name` from yaml clashes with `EXPERIMENT_NAME` from `notebooks/00_config.py` | Both surfaces define their own naming — intentional | Keep them independent; they target different runs (sgcli vs. notebook) |
