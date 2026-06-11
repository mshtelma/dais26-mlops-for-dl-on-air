# sgcli launch surface

This directory holds the [Databricks Serverless GPU CLI (`sgcli`)](http://go/sgc/sgcli)
artifacts for running the detector training workload from a terminal — no notebook
required, no traditional ML cluster involved.

> `sgcli` is in **private preview (v ≤ 0.0.7)**. The wheel is delivered by your
> Solutions Architect. See the internal onboarding doc for the latest install link.

## Files

| File | Purpose |
|---|---|
| `workload_train_detector.yaml` | C-RADIOv4 training (`recipe: cradio_v4_so400m`). |
| `workload_train_detector_dinov3.yaml` | DINOv3 training (`recipe: dinov3_vitl16`, gated — needs `dais26-secrets/hf-token`). |
| `workload_sweep.yaml` | HPO campaign stage via `train.sweep_cli` (same `SweepRunner` as notebook 02b). |
| `requirements.yaml` | Python deps not in the AIR base env v4 image. |

Hyperparameters are NOT listed in the workload YAMLs: each training workload
names a **recipe** (`dais26_dentex.config.recipes.RECIPES` — the campaign-final
best-known config, the same source the notebook lane builds from) and the sweep
workload names a **stage** (`dais26_dentex.config.campaigns.CAMPAIGN_STAGES`).
The `parameters:` blocks carry only environment values (catalog/schema/paths/
experiment) plus explicit, deliberate overrides such as the demo-time `epochs`.

## One-time setup

```bash
# Install uv (Rust-based Python tool installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install sgcli into an isolated env on your PATH
uv tool install --python 3.12 /path/to/databricks_serverless_gpu_cli-<version>-py3-none-any.whl

# Authenticate to your workspace (creates ~/.databrickscfg)
databricks auth login --host https://<your-workspace>.cloud.databricks.com

# Optionally pin a profile for sgcli:
export DATABRICKS_CONFIG_PROFILE=dev
```

## Launch training

From the repo root:

```bash
sgcli run -f sgcli/workload_train_detector.yaml --watch -p dev
```

The workload:

1. Snapshots the entire repo (`code_source.snapshot.repo_path: .`) to the GPU pod, including
   uncommitted edits (`allow_uncommitted: true`). For a reproducible release run, flip
   `allow_uncommitted: false` and pin `git_commit:` instead — sgcli treats them as mutually
   exclusive.
2. Installs deps from `requirements.yaml` (AIR base env v4 supplies torch, mlflow, etc.).
3. `pip install .` so `dais26_dentex.train.cli` is importable (no `-e`; snapshot is read-only).
4. Launches `torchrun --nproc_per_node=8 -m dais26_dentex.train.cli` across the 8 H100s of one node.
5. The package CLI reads `$HYPERPARAMETERS_PATH`, resolves the named `recipe:` from
   `config.recipes.RECIPES` (best-known campaign-final hyperparameters), applies the
   YAML's explicit overrides, builds `TrainerConfig`, and runs `Trainer`.
6. Rank 0 logs the MLflow run **into the shared `dais26_vfm_experiment`**
   (`parameters.experiment_name`), registers the model in UC, and sets `@challenger` —
   exactly what the notebook quickstart produces.

## Launch a hyperparameter sweep

The same `SweepRunner` that powers the `campaign_sweep` DAB job runs from the
terminal — one torchrun allocation executes a whole campaign stage (sequential
trials + winner retrains + the `@challenger` best-in-experiment gate):

```bash
sgcli run -f sgcli/workload_sweep.yaml --watch -p dev \
  --override parameters.stage=cradio_s2

sgcli get logs <run-id> --rank 0 -p dev   # rank 0 prints SWEEP_* summary lines
```

Stages live in `dais26_dentex.config.campaigns.CAMPAIGN_STAGES` (the "push to
0.60" chain, docs/HPO.md). For DINOv3 stages, uncomment the `HF_TOKEN` secret in
`workload_sweep.yaml`.

## Override at submit time

```bash
# Full 150-epoch recipe schedule instead of the 50-epoch demo override:
sgcli run -f sgcli/workload_train_detector.yaml \
  --override parameters.epochs=150 \
  -p dev \
  --watch
```

## Inspect a run

```bash
sgcli get runs --limit 10 -p dev
sgcli get status <run-id>  -p dev
sgcli get logs   <run-id> --rank 0 -p dev
sgcli get logs   <run-id> --debug  -p dev    # all nodes, filtered for errors
sgcli cancel     <run-id>  -p dev
```

## Notes

- `compute.gpus: 16` (or 24 / 32) automatically becomes multi-node — the `command:` arithmetic
  computes `NNODES = WORLD_SIZE / GPUS_PER_NODE` correctly.
- The notebook entrypoint (`notebooks/02_train_detector_air.py`) and this sgcli workload
  share the same `TrainerConfig` + `Trainer` core. The notebook path uses the local
  `serverless_gpu.@distributed` helper and does not use `torchrun`; this SGCLI path uses
  `torchrun`.
- All quickstart training is **AIR-only** on single-node 8xH100 by default — no traditional
  ML clusters.
