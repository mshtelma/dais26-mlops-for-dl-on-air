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

## Deployment job (primary promotion path)

The MLflow 3 deployment job `deploy_job_detector` (`resources/jobs/deploy_job_detector.yml`)
is the primary path from a trained model to a served champion. The standalone
`deploy_endpoint` job (notebook `04_deploy_serving.py`) is now the break-glass / manual
redeploy path only.

### Dev / prod asset split (two schemas)

Following the Big Book "deploy code" pattern, dev and prod registered models live in
separate UC schemas (`notebooks/00_config.py`):

- **Dev** — `CATALOG.SCHEMA` (`mlops_pj.dais26_vfm`). Training / the HPO sweep register
  new versions here and set the dev alias `@challenger` (constant `ALIAS_CANDIDATE`,
  value `challenger`).
- **Prod / champion** — `CHAMPION_CATALOG.CHAMPION_SCHEMA` (`mlops_pj.dais26_vfm_prod`).
  The promote task copies the approved dev version here (lineage preserved) and sets
  `@champion`. `notebooks/00_setup.py` creates this schema + the SP grants.

### How a release flows

1. **Trigger.** The HPO sweep (`02b`) sets `@challenger` on a dev detector model **only
   when** the retrained winner's `val/best_mAP_50` strictly beats the experiment's prior
   best (the challenger registration gate, pure `sweep.beats_experiment_best`). A new
   `@challenger` version auto-triggers `deploy_job_detector` (wired by
   `connect_deployment_job` / `notebooks/13`, which calls
   `update_registered_model(deployment_job_id=...)` for both dev detectors).
2. **Evaluation** (`notebooks/10`, GPU `GPU_1xA10` / `databricks_ai_v5`): re-scores the
   triggered version on the DENTEX **test** split via the shared `eval.runner`, logs
   `test/*` metrics to the model version, and gates on `mAP@50 ≥ 0.58 AND
   Caries AP@50 ≥ 0.30` AND best-in-experiment (≥ current prod `@champion` re-scored on
   test, and ≥ every prior evaluated version). Fails the task otherwise.
3. **Approval** (`notebooks/11`, CPU, `max_retries: 0`): passes only when the UC tag
   `Approval_Check = Approved` is set on the version. To approve:
   ```python
   from mlflow.tracking import MlflowClient
   c = MlflowClient(registry_uri="databricks-uc")
   c.set_model_version_tag(name=MODEL_NAME, version=MODEL_VERSION,
                           key="Approval_Check", value="Approved")
   ```
   then repair-run the Approval task.
4. **Promote** (`notebooks/12`, CPU): `copy_model_version` from dev →
   `CHAMPION_CATALOG.CHAMPION_SCHEMA` (lineage to the source run preserved), then
   `deploy_and_smoke_test(..., model_version=<new prod version>, promote_on_success=True)`
   deploys that explicit prod version and flips `@champion` **only** on a passing smoke
   test. On failure `@champion` is left untouched (previous champion keeps serving).

### One-time wiring after deploy

```bash
databricks bundle deploy -t <target>
databricks bundle run connect_deployment_job -t <target>   # wires deployment_job_id to both dev models
```

Re-run `connect_deployment_job` if `deploy_job_detector` is ever recreated (its id changes).

---

## Rollback procedure

Use this when a newly promoted `@champion` version is producing bad results and you need to revert
to the previous version. (`@champion` now lives on the **prod** model
`mlops_pj.dais26_vfm_prod.<backbone>_detector`; the examples below use the dev-schema
names but apply the same way to the prod champion model.)

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
GRANT APPLY TAG ON SCHEMA ml.dais26_vfm TO \`$SP_APP_ID\`;
GRANT EXECUTE ON MODEL ml.dais26_vfm.cradio_detector TO \`$SP_APP_ID\`;
-- Prod / champion schema (deployment-job promote target): the SP copies the
-- approved dev version here and sets @champion, so it needs USE/CREATE MODEL/
-- APPLY TAG on the prod schema and EXECUTE on the dev models (copy source).
GRANT USE CATALOG ON CATALOG ml TO \`$SP_APP_ID\`;
GRANT USE SCHEMA, CREATE MODEL, APPLY TAG ON SCHEMA ml.dais26_vfm_prod TO \`$SP_APP_ID\`;
"
```

> These grants are applied automatically by `notebooks/00_setup.py` (which uses the
> `CHAMPION_CATALOG.CHAMPION_SCHEMA` config values); the SQL above is the manual
> equivalent for reference.

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

### CI/CD via OAuth M2M (service principal client id + secret) {#cicd-oauth-m2m}

All three workflows authenticate to Databricks via **OAuth machine-to-machine
(M2M)** using the SP's client id + an OAuth secret. The secret mints short-lived
OAuth tokens (no long-lived PAT). This path needs only **SP-create + OAuth-secret**
rights — **no account admin / workload-identity-federation policy** (that route is
account-admin-only; use it instead if you have account-admin access).

- `.github/workflows/deploy.yml` — deploys the bundle + runs `connect_deployment_job`
  (environments `dev`/`prod`).
- `.github/workflows/ci.yml` (`dab-validate`) and `weekly_air_check.yml` (`check-air`)
  — read-only checks; both use a shared `ci` environment.

One-time setup:

1. **Generate an OAuth secret on the SP.** Either the UI (Settings → Identity and
   access → Service principals → `dais26-vfm-sp` → Secrets → Generate secret) or the
   CLI:

   ```bash
   # SP_NUMERIC_ID is the SP's Databricks id (not the application UUID).
   databricks service-principal-secrets create <SP_NUMERIC_ID>
   # → returns { "secret": "dose...", "secret_hash": "..." }  — copy `secret` now;
   #   it is shown only once.
   ```

   The SP **application UUID** is the value for `DATABRICKS_CLIENT_ID`; the returned
   `secret` is `DATABRICKS_CLIENT_SECRET`.

2. **Create the `dev`, `prod`, and `ci` GitHub Environments** (repo Settings →
   Environments). Add a **required reviewer** on `prod` to gate production deploys;
   leave `dev` and `ci` without reviewers (they run unattended / on a schedule).
   Per environment, set:
   - **Variable** `DATABRICKS_HOST` — the workspace URL for that environment.
   - **Variable** `DATABRICKS_CLIENT_ID` — the SP application UUID (also reused for
     `sp_app_id` in `run_as`; the workflow passes it through automatically).
   - **Secret** `DATABRICKS_CLIENT_SECRET` — the OAuth secret from step 1.

3. The workflows set `DATABRICKS_AUTH_TYPE: oauth-m2m`; the Databricks CLI exchanges
   the client id + secret for short-lived OAuth tokens. Trigger manually (choose the
   target) or let a push to `main` auto-deploy `dev`.

> Rotate the OAuth secret periodically: generate a new one (an SP can hold multiple),
> update `DATABRICKS_CLIENT_SECRET` in each environment, then delete the old secret
> with `databricks service-principal-secrets delete <SP_NUMERIC_ID> <SECRET_ID>`.
> If you later obtain account-admin access, prefer migrating to OIDC token federation
> (secret-free) — the workflows only need `DATABRICKS_AUTH_TYPE` flipped to
> `github-oidc` + `permissions: id-token: write`.

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

### Models-from-code serving load path {#models-from-code}

**Symptom 1 — `ModuleNotFoundError: transformers_modules`.** The endpoint
deploy fails at model-server startup ("model server failed to load"). The
detector was logged by pickling a `DetectorPyfunc()` instance in the training
process; that pickle captured a reference to the HuggingFace *dynamic* backbone
class. `trust_remote_code=True` (nvidia/C-RADIOv4-SO400M, facebook/dinov3-vitl16)
generates that class at runtime inside the `transformers_modules.*` package,
which does not exist in the serving container — so unpickling `python_model.pkl`
raises `ModuleNotFoundError: No module named 'transformers_modules'`.

**Symptom 2 — `ModuleNotFoundError: dais26_dentex`.** The pyfunc class lives in
a locally-installed package (not on PyPI). MLflow cannot pin it in
`requirements.txt`, so the serving container can't import the model class.

**Fix.** Log the detector via **models-from-code** *and* bundle the package source:

```python
reporter.log_pyfunc(
    python_model=".../serve/detector_model_script.py",  # a SCRIPT, not an instance
    code_paths=[<dir of installed dais26_dentex>],       # default in log_pyfunc
    ...
)
```

`detector_model_script.py` is a 3-line module whose body is
`set_model(DetectorPyfunc())`. MLflow stores the script as the model definition
and re-executes it at load time (building a fresh `DetectorPyfunc`), so no
HF dynamic class is ever serialized. `code_paths` copies the package into the
model's `code/` dir and MLflow prepends it to `sys.path` at load — `import
dais26_dentex` then works with no pip install. The trainer wires both at the
single `_save_and_register` log site (`train/trainer.py`); there is no separate
re-log step.

**Offline backbone load.** The serving container has no egress. `load_context`
forces `local_files_only=True` and the offline HF env (`HF_HUB_OFFLINE`) and
reads the backbone from the `model_cache` artifact bundled with the model.
Without this, `from_pretrained` tries to reach huggingface.co and the model
server fails to start.

**No `torch.compile` at serving.** `_maybe_compile` is a no-op. Wrapping the
full `DetectionModel` makes TorchDynamo trace *into* the backbone on the first
`predict`; the DINOv3 stack routes through transformers'
`output_capturing.py`, whose module namespace does not bind `torch`, so the
Dynamo-transformed frame raises `NameError: name 'torch' is not defined` during
inference (the model still *loads* fine — `load_context` runs eagerly). CUDA
graphs (`reduce-overhead`) also need static shapes the variable-size image path
can't guarantee. The marginal latency win isn't worth the serving fragility.

### torch/torchvision cu124 pin {#torch-cu124-pin}

**Symptom.** The detector endpoint deploys and serves, but inference is slow and
the GPU dashboard shows ~0% GPU utilization — the model silently ran on CPU.
`torch.cuda.is_available()` returns `False` inside the container.

**Cause.** GPU_SMALL serving nodes (NVIDIA T4) ship a driver that reports CUDA
12.4. An unpinned `torch` in `[tool.dais26.serving-deps]` resolves to the newest
PyPI wheel (cu126/cu128), whose CUDA runtime the 12.4 driver cannot initialize —
torch falls back to CPU instead of erroring.

**Fix.** Pin `torch==2.6.0` and `torchvision==0.21.0` in
`[tool.dais26.serving-deps].detector`; those versions default to the cu124 build
on PyPI and initialize cleanly on the 12.4 driver. Re-validate with
`scripts/probe_endpoint_gpu.py` if you bump the versions.

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

### Hyperparameter sweep + backbone fine-tuning {#hpo-sweep}

The `hpo_sweep` job (`notebooks/02b_hpo_sweep.py`) tunes the detector head and
fine-tunes the C-RADIO / DINOv3 backbone. It is config-driven from the `SWEEP_*`
block in `notebooks/00_config.py` and shares the `Trainer` core with the single-run
path, so anything below applies to both `train_detector` and `hpo_sweep`.

Backbone fine-tuning is controlled by three `TrainerConfig` knobs (settable in
`SWEEP_SEARCH_SPACE`, the sgcli yaml, or `00_config.py`):

| Knob | Values | Effect |
|------|--------|--------|
| `backbone_mode` | `frozen` (default) / `lora` / `partial` / `full` | how much of the encoder receives gradients; resolved via `cfg.effective_backbone_mode()` (legacy `use_lora=True` still maps to `lora`) |
| `backbone_trainable_blocks` | int ≥ 1 | for `partial`: how many trailing transformer blocks `peft.unfreeze_last_blocks` unfreezes |
| `backbone_lr` | float > 0 | discriminative LR for the backbone param group (head/FPN keep `lr`); both feed a multi-`max_lr` `OneCycleLR` |

| Symptom | Cause | Fix |
|---|---|---|
| Sweep / fine-tune run OOMs on the H100 pool | `backbone_mode=full` doubles activations vs frozen | Drop to `partial` with a small `backbone_trainable_blocks`, or lower `batch_size` in the trial space |
| Loss diverges immediately when fine-tuning the backbone | `backbone_lr` too high → catastrophic forgetting of the VFM | Keep `backbone_lr` ≈ 1e-5 (10–100× below the head `lr`); the discriminative param groups exist for exactly this |
| DDP `find_unused_parameters` error | `backbone_mode=full` expects every param to get a grad | Trainer sets `find_unused_parameters=False` only for `full`; for `frozen`/`lora`/`partial` it stays `True` |
| Fine-tuned weights don't survive serving | backbone weights stripped at load for frozen/LoRA artifacts | The manifest records `backbone.trained_mode`; `detector_pyfunc` keeps the full backbone state only when it is `full`/`partial`. Re-train so the manifest is written by current code |
| Sweep job times out | default job timeout too short for multi-trial + winner re-train | `hpo_sweep` (and `train_detector`) carry an 8-hour timeout (`timeout_seconds: 28800`); sgcli workloads use `timeout_minutes: 480` |
| Anchor changes have no effect | `anchor_scales`/`aspect_ratios` left unset | `build_detector` only overrides the FPN defaults when both are set; the sweep's `anchor_mode` maps presets onto these fields |
