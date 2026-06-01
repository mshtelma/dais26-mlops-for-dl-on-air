# sgcli launch surface

This directory holds the [Databricks Serverless GPU CLI (`sgcli`)](http://go/sgc/sgcli)
artifacts for running the detector training workload from a terminal — no notebook
required, no traditional ML cluster involved.

> `sgcli` is in **private preview (v ≤ 0.0.7)**. The wheel is delivered by your
> Solutions Architect. See the internal onboarding doc for the latest install link.

## Files

| File | Purpose |
|---|---|
| `workload_train_detector.yaml` | Workload spec (compute, snapshot, parameters, command). |
| `requirements.yaml` | Python deps not in the AIR base env v4 image. |

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

1. Snapshots the entire repo (`code_source.snapshot.repo_path: ..`) to the GPU pod, including
   uncommitted edits (`allow_uncommitted: true`). For a reproducible release run, flip
   `allow_uncommitted: false` and pin `git_commit:` instead — sgcli treats them as mutually
   exclusive.
2. Installs deps from `requirements.yaml` (AIR base env v4 supplies torch, mlflow, etc.).
3. `pip install .` so `dais26_dentex.train.cli` is importable (no `-e`; snapshot is read-only).
4. Launches `torchrun --nproc_per_node=8 -m dais26_dentex.train.cli` across the 8 H100s of one node.
5. Each rank runs `train_detector(...)` with kwargs read from `$HYPERPARAMETERS_PATH`.
6. Rank 0 logs the MLflow run, registers the model in UC, and sets `@candidate`.

## Override at submit time

```bash
# Switch to LoRA, 4 epochs, override compute:
sgcli run -f sgcli/workload_train_detector.yaml \
  --override compute.gpus=8 \
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
  share the same `src/dais26_dentex/train/train_detector.py` core. They differ only in how the
  `torch.distributed` ranks are launched.
- All training is **AIR-only** (Serverless GPU Compute) — no traditional ML clusters.
