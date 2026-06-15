# Production deployment

Going to `prod` adds a **service principal** (for `run_as`), the **champion schema/model**, the
**prod-only embedding/monitoring jobs**, and the **deployment-job wiring**. The dev quickstarts
need none of this.

## The prod target

```yaml title="databricks.yml (prod target)"
targets:
  prod:
    mode: production
    run_as:
      service_principal_name: "${var.sp_app_id}"
    resources:
      schemas:
        dais26_vfm_prod: { catalog_name: mlops_pj, name: dais26_vfm_prod }
```

`mode: production` applies no name prefix, so the champion schema/model resolve to their literal
names. The champion registered model is declared **prod-only** in
`resources/registered_models/detector_models_champion.yml` and references the schema above for
create-ordering.

## Step 1 — Create the service principal

```bash
SP_RESPONSE=$(databricks service-principals create --display-name dais26-vfm-sp --output JSON)
SP_APP_ID=$(echo "$SP_RESPONSE" | jq -r '.applicationId')   # a UUID, NOT the display name
echo "$SP_APP_ID"
```

The DAB `run_as.service_principal_name` requires the **application ID** (UUID). On Azure, the SP
is backed by Microsoft Entra ID — pre-create it there or in the UI and use its Entra App ID. Full
detail (incl. Azure notes): [Operations & runbook → service principal creation](../RUNBOOK.md#service-principal-creation).

## Step 2 — Set the SP variable

```bash
export DATABRICKS_SP_APP_ID="$SP_APP_ID"
# or set variables.sp_app_id.default in databricks.yml
# or pass at deploy: databricks bundle deploy -t prod --var sp_app_id="$SP_APP_ID"
```

## Step 3 — Bind the prod schema (first time on an existing workspace)

If `dais26_vfm_prod` already exists outside Terraform state, bind it once so `bundle deploy -t
prod` doesn't error that the schema already exists:

```bash
databricks bundle deployment bind dais26_vfm_prod mlops_pj.dais26_vfm_prod \
  -t prod --profile <profile> --auto-approve
```

## Step 4 — Deploy + wire

```bash
databricks bundle deploy -t prod
databricks bundle run connect_deployment_job -t dev      # challenger job → dev models
databricks bundle run connect_deployment_job -t prod     # champion job → detector_champion
```

`scripts/deploy_bundle.sh -t prod` runs the deploy + connect in one shot (and idempotently ensures
the dev + champion schemas exist). Re-run `connect_deployment_job` if a deployment job is recreated
(its id changes). See [Evaluate → approve → promote](../lifecycle/evaluate-approve-promote.md).

!!! note "What `-t prod` deploys that `-t dev` doesn't"
    The champion schema + champion model + deployment job, and the prod-only embedding/monitoring
    jobs (`precompute_embeddings`/`create_vector_search` as champion-job tasks, plus the
    `drift_monitor` paused cron). The dev schema is shared, data-laden infra and is **never**
    bundle-managed.

## Step 5 — Grant inference-table access (deferred)

The AI Gateway auto-creates the inference table on the **first** endpoint request, so this grant
can only run afterward:

```bash
python scripts/grant_inference_table_access.py --catalog mlops_pj --schema dais26_vfm --sp-app-id "$SP_APP_ID"
```

## Step 6 — The release flow

A new `@challenger` (from training or the HPO sweep) now auto-triggers the governed path:
eval → approval → cross-schema champion copy → champion deploy + smoke → `@champion` flip →
embeddings/VS/drift refresh. Walk it in
[Evaluate → approve → promote](../lifecycle/evaluate-approve-promote.md) and
[Serve & AI Gateway](../lifecycle/serve.md).

## CI/CD

Deploy from CI via OAuth M2M (no long-lived PAT). The deploy workflow is currently blocked by the
workspace IP access list on GitHub-hosted runners — deploy from an allowlisted/self-hosted runner
or locally with `scripts/deploy_bundle.sh`. See [CI/CD](cicd.md).

## Pre-demo / operational procedures

Pre-demo D-1 checklist, endpoint warmup, latency probe, switch-to-video, GPU-memory validation,
and rollback all live in **[Operations & runbook](../RUNBOOK.md)**.
