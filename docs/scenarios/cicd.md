# CI/CD (GitHub Actions)

Three workflows live in `.github/workflows/`:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | PR + push to `main` | lint (ruff) + unit tests; build the wheel + import-smoke it in a clean venv |
| `deploy.yml` | push to `main` + manual dispatch | `bundle deploy` + `connect_deployment_job` (OAuth M2M) |
| `docs.yml` | push to `main` (docs paths) + manual | build this MkDocs site `--strict` and publish to GitHub Pages |

## CI — `ci.yml`

Two jobs on `ubuntu-latest`, Python 3.12:

- **lint-and-test**: `uv pip install -e ".[dev]" --system` → `ruff check src/ tests/ scripts/`
  → `pytest tests/unit/`.
- **build-and-import**: `uv build` → verify `dist/*.whl` → install the wheel in a fresh venv and
  `import dais26_dentex`.

No workspace access required, so CI runs green on GitHub-hosted runners.

## Deploy — `deploy.yml` (OAuth M2M)

Authenticates via **OAuth machine-to-machine** using the service principal's client id + an OAuth
secret (short-lived tokens, no long-lived PAT). Needs only SP-create + OAuth-secret rights — **no
account admin / federation policy**.

It binds to a GitHub **Environment** (`dev`/`prod`) so host/credentials are scoped per environment
and `prod` can require a manual reviewer.

```yaml title="deploy.yml (core)"
env:
  DATABRICKS_AUTH_TYPE: oauth-m2m
  DATABRICKS_HOST: ${{ vars.DATABRICKS_HOST }}
  DATABRICKS_CLIENT_ID: ${{ vars.DATABRICKS_CLIENT_ID }}
  DATABRICKS_CLIENT_SECRET: ${{ secrets.DATABRICKS_CLIENT_SECRET }}
steps:
  - run: databricks bundle deploy -t "$TARGET" --var="sp_app_id=${DATABRICKS_CLIENT_ID}"
  - run: databricks bundle run connect_deployment_job -t "$TARGET" --var="sp_app_id=${DATABRICKS_CLIENT_ID}"
```

### One-time setup

1. **Generate an OAuth secret on the SP** (UI: Settings → Identity and access → Service
   principals → `dais26-vfm-sp` → Secrets → Generate secret; or CLI
   `databricks service-principal-secrets create <SP_NUMERIC_ID>` — copy the `secret`, shown once).
   The SP **application UUID** is `DATABRICKS_CLIENT_ID`; the returned `secret` is
   `DATABRICKS_CLIENT_SECRET`.
2. **Create `dev`, `prod`, `ci` GitHub Environments** (repo Settings → Environments). Add a
   required reviewer on `prod`. Per environment set: var `DATABRICKS_HOST`, var
   `DATABRICKS_CLIENT_ID` (the SP app UUID, reused for `run_as`), secret
   `DATABRICKS_CLIENT_SECRET`.
3. Trigger manually (choose the target) or let a push to `main` auto-deploy `dev`.

Rotate the OAuth secret periodically; if you later get account-admin access, prefer migrating to
OIDC token federation (secret-free — flip `DATABRICKS_AUTH_TYPE: github-oidc` +
`permissions: id-token: write`). Full setup:
[Operations & runbook → CI/CD via OAuth M2M](../RUNBOOK.md#cicd-oauth-m2m).

!!! warning "The workspace IP access list blocks GitHub-hosted runners"
    The `dev` workspace enforces an IP access list that returns 403 for GitHub-hosted runner IPs.
    So `deploy.yml` (and `databricks bundle validate`) currently **cannot reach the workspace from
    CI**. Until CI runs from an allowlisted/self-hosted runner, deploy from a machine inside the
    allowed network:

    ```bash
    ./scripts/deploy_bundle.sh -t dev     # or -t prod
    ```

    This is the manual equivalent of the workflow: it ensures the UC schemas exist, then runs
    `bundle deploy` + `bundle run connect_deployment_job`. (`databricks bundle validate` is
    deliberately **not** run in CI for the same reason; run it locally before pushing.)

## Docs — `docs.yml`

Builds this site with `mkdocs build --strict` (fails on any broken internal link/anchor) and
deploys to GitHub Pages. Independent of Databricks, so the IP access list does not affect it.
**One-time:** repo Settings → Pages → Source = "GitHub Actions". See the repo's
`.github/workflows/docs.yml`.
