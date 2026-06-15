# Install & authenticate

These steps are shared by both lanes. Budget ~5 minutes on a clean machine.

## 1. Clone

```bash
git clone https://github.com/mshtelma/dais26-mlops-for-dl-on-air.git
cd dais26-mlops-for-dl-on-air
```

## 2. Install Python dependencies

```bash
pip install uv
uv pip install -e ".[dev]"
```

This installs all runtime + dev dependencies (torch, transformers, mlflow, databricks-sdk,
pycocotools, …) in editable mode, so edits to `src/dais26_dentex/` are picked up immediately.
`make install` is the shortcut.

!!! info "What `[dev]` and `[air]` extras add"
    `[dev]` adds pytest, ruff, pyright. `[air]` adds `serverless_gpu` (the notebook
    `@distributed` helper). The AIR CLI itself (`pip install databricks-air`) is separate — see
    [Quickstart — air CLI lane](quickstart-air.md).

## 3. Authenticate to Databricks

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com
```

This writes credentials to `~/.databrickscfg`. For the **air** lane, give the profile a name so
`air -p <profile>` can select it:

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile df1
```

Alternatively export `DATABRICKS_HOST` + `DATABRICKS_TOKEN`. For CI, use OAuth M2M — see
[CI/CD](../scenarios/cicd.md).

## 4. Build the wheel

```bash
uv build      # or: make build
```

Produces `dist/dais26_dentex-0.1.0-py3-none-any.whl`. The DAB attaches this wheel to every job
task; the air lane `pip install .`s the snapshot on the pod.

!!! warning "The wheel ships `pyproject.toml` — verify it"
    The build copies `pyproject.toml` into the wheel as `dais26_dentex/_pyproject.toml` (via
    hatchling `force-include`). At model log-time,
    `platform.mlflow_io.serving_pip_requirements` reads `[tool.dais26.serving-deps]` from that
    packaged copy — necessary because AIR's ephemeral env installs into a `site-packages` whose
    ancestors contain no `pyproject.toml`. Confirm it landed:

    ```bash
    python -m zipfile -l dist/*.whl | grep _pyproject.toml
    # → dais26_dentex/_pyproject.toml ...
    ```

    A stale wheel built before that block raises `FileNotFoundError: Could not locate
    pyproject.toml` at log-time. Re-run `uv build`. See
    [pip_requirements source of truth](../RUNBOOK.md#pip-requirements-rationale).

## 5. Pick your target (optional)

The default environment is `df1` (→ catalog `main`, schema `mshtelma`). To target elsewhere
without editing tracked files, set `$DAIS26_ENV`, drop an `environments.local.yaml` overlay, or
export `$DAIS26_CATALOG` / `$DAIS26_SCHEMA`. See
[Per-user environment overrides](../scenarios/env-overrides.md).

## Verify your setup

```bash
make test                          # unit tests (no workspace needed)
databricks current-user me         # confirms auth
make discover                      # Day-1 AIR runtime discovery gate
```

You're ready. Pick a lane:

- **[Quickstart — DAB lane](quickstart-dab.md)**
- **[Quickstart — air CLI lane](quickstart-air.md)**
