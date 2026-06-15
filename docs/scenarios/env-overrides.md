# Per-user environment overrides

You rarely need to edit `config/environments.py`. To point a run at your own catalog/schema
without committing anything, use one of three escape hatches — all honored by **both** lanes.

## Resolution precedence (highest wins)

1. explicit keyword overrides to `load_environment(...)`
2. **`DAIS26_*` environment variables** (CI / one-offs)
3. an optional per-user **`environments.local.yaml`** overlay
4. the committed named entry in `ENVIRONMENTS` (`df1`, `prod`)

So a `DAIS26_*` var beats the overlay, which beats the committed env.

## Option A — `$DAIS26_*` environment variables

| Variable | Sets |
|----------|------|
| `DAIS26_ENV` | which named env to start from (default `df1`) |
| `DAIS26_CATALOG` | `catalog` |
| `DAIS26_SCHEMA` | `schema` |
| `DAIS26_EXPERIMENT` | `experiment_name` |
| `DAIS26_VOLUME_PATH` / `DAIS26_CACHE_DIR` | volume / cache paths |
| `DAIS26_CHAMPION_CATALOG` / `DAIS26_CHAMPION_SCHEMA` | champion location |
| `DAIS26_ENV_FILE` | explicit path to an overlay file |

```bash
# Local / notebook cluster env:
export DAIS26_CATALOG=my_catalog DAIS26_SCHEMA=my_sandbox

# air lane (env vars on the pod):
air run -f air/workload_train_detector.yaml \
  --override env_variables.DAIS26_SCHEMA=my_sandbox --watch -p df1
```

## Option B — `environments.local.yaml` overlay

Drop an `environments.local.yaml` at the repo root (the loader also honors `$DAIS26_ENV_FILE`, and
searches up from the CWD to the project root):

```yaml title="environments.local.yaml"
catalog: my_catalog
schema: my_sandbox
experiment_name: /Users/me@example.com/dais26_vfm_experiment
```

!!! info "Why it's deliberately NOT git-ignored"
    air's working-tree snapshot and the notebooks' `%pip install ..` reinstall both carry this
    file to the remote pod/cluster, so your local edits reach **both** lanes with no commit. A
    `.env` name would *not* work — the repo git-ignores `.env*`/`*.local`, and air respects
    `.gitignore`, so the file would never reach the pod. Hence the `.yaml` name. Pinned-commit
    reproducible runs intentionally see only the committed `ENVIRONMENTS`, never an uncommitted
    overlay.

## Option C — single-value `--override` (air)

```bash
air run -f air/workload_train_detector.yaml --override parameters.schema=my_sandbox --watch -p df1
```

`parameters.schema` lands in `$HYPERPARAMETERS_PATH` and overrides just that field of the resolved
env.

## Add a permanent named environment

For a target you'll reuse, add an entry to `ENVIRONMENTS` in
[`config/environments.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/config/environments.py)
(only `catalog`/`schema`/`experiment_name` are required; the rest derive):

```python
ENVIRONMENTS = {
    "df1": {...}, "prod": {...},
    "my_env": {
        "catalog": "my_catalog",
        "schema": "my_sandbox",
        "experiment_name": "/Users/me@example.com/dais26_vfm_experiment",
    },
}
```

Then `ENV = "my_env"` (notebook) or `env: my_env` / `--override parameters.env=my_env` (air). See
[Named configuration](../lanes/configuration.md).

!!! warning "Never put secrets in environments"
    `EnvSpec` holds non-secret locations only. The HuggingFace token flows through Databricks
    secret scopes / the air `secrets:` block.
