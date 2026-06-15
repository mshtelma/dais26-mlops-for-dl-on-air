# Prerequisites

Verify the following before running either quickstart.

## Workspace & access

| Requirement | How to check |
|-------------|--------------|
| **Unity Catalog** enabled | Workspace settings → Unity Catalog |
| **AI Runtime / Serverless GPU** with single-node **8×H100** quota | Databricks account / workspace quota; confirm with [`make discover`](../reference/scripts.md) |
| A catalog + schema you can create tables/volumes/models in | The default env `df1` targets `main.mshtelma`; change it in [`config/environments.py`](../lanes/configuration.md) |

!!! note "AIR-only, by design"
    All training, embedding, and drift compute runs on Databricks AI Runtime / Serverless GPU.
    There are **no traditional ML clusters** defined anywhere in the bundle. The detector serving
    endpoint runs on Mosaic AI Model Serving GPU compute (`GPU_SMALL`/`GPU_MEDIUM`).

## Tooling

| Tool | Version | Needed for | Install |
|------|---------|-----------|---------|
| **Databricks CLI** | v0.230+ | both lanes, all `databricks bundle`/`serving-endpoints`/`secrets` commands | [docs](https://docs.databricks.com/dev-tools/cli/install.html) — check `databricks version` |
| **Python** | 3.12+ | the package, tests, local tooling | `python3 --version` |
| **uv** | latest | install + build the wheel | `pip install uv` |
| **AIR CLI (`air`)** | Beta | **only** the air CLI lane | `pip install databricks-air` |

The AIR CLI is in **Beta**; `air -h` and `air config -h` are the always-current command / YAML
reference.

## Backbone access

- **C-RADIOv4-SO400M** (the default `BACKBONE`) is **ungated** under the NVIDIA Open Model
  License — **no HuggingFace account or token is required.**
- **DINOv3** is **gated**. You only need a HuggingFace token (stored in the
  `dais26-secrets/hf-token` secret) if you switch to the DINOv3 comparison path. See
  [Switch backbone](../scenarios/switch-backbone.md).

## Production-only prerequisites

The `prod` target and the governed promotion path additionally need a **service principal** (for
`run_as`) and UC grants. These are not required for the dev quickstarts. See
[Production deployment](../scenarios/production-deploy.md) and
[Operations & runbook → service principal creation](../RUNBOOK.md#service-principal-creation).

Next: **[Install & authenticate](installation.md)**.
