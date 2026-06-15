# air CLI lane

The [Databricks **AIR CLI**](https://preview.docs.databricks.com/aws/en/machine-learning/ai-runtime/cli)
(`air`) runs detector training and HPO sweeps from a terminal — no notebook, no traditional ML
cluster. `air` submits a YAML workload to Serverless GPU and (with `--watch`) streams the logs
back. It launches the **same `Trainer` core** as the [DAB lane](dab.md), under `torchrun`.

!!! info "Beta"
    The AIR CLI is in Beta: `pip install databricks-air`. `air -h` and `air config -h` are the
    always-current command / YAML-field reference; this page captures the project-specific usage.

## Workload files (`air/`)

| File | Purpose |
|------|---------|
| `workload_train_detector.yaml` | C-RADIOv4 training (`recipe: cradio_v4_so400m`) |
| `workload_train_detector_dinov3.yaml` | DINOv3 training (`recipe: dinov3_vitl16`, gated — needs `dais26-secrets/hf-token`, already wired) |
| `workload_sweep.yaml` | HPO campaign stage via `train.sweep_cli` (same `SweepRunner` as notebook 02b) |
| `requirements.yaml` | Python deps not in the AIR base env v4 image |

Nothing is hand-mirrored: each training workload names a **recipe**, the sweep workload names a
**stage**, and all name an **environment** — the same sources the notebook lane resolves. See
[Named configuration](configuration.md).

## One-time setup

```bash
pip install databricks-air                 # or: uv tool install databricks-air
databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile df1
air --version
```

All commands take `-p df1` (or set `DATABRICKS_CONFIG_PROFILE=df1`).

## Launch training

```bash
air run -f air/workload_train_detector.yaml --watch -p df1
```

The workload (see [Quickstart — air CLI lane](../getting-started/quickstart-air.md) for the
step-by-step): snapshots the repo → installs `requirements.yaml` → `pip install .` →
`torchrun … -m dais26_dentex.train.cli` across 8 H100s → the CLI resolves `recipe`/`env`, builds
`TrainerConfig`, runs `Trainer`, and rank 0 registers `@challenger` in the shared experiment.

```yaml title="workload_train_detector.yaml (key blocks)"
compute:
  num_accelerators: 8              # 1 node (GPU_8xH100 = 8/node); 16 for 2-node scaling
  accelerator_type: GPU_8xH100
timeout_minutes: 480               # 8h: full backbone fine-tune + longer schedules
code_source:
  type: snapshot
  snapshot:
    root_path: ..                  # repo root (relative to this YAML's dir)
parameters:                        # written to $HYPERPARAMETERS_PATH
  recipe: cradio_v4_so400m
  env: df1
  backbone_revision: main
  register_model: true
  set_candidate_alias: true
  epochs: 50                       # demo override of the recipe's 150-epoch schedule
command: |-
  set -euxo pipefail
  cd "$CODE_SOURCE_PATH"
  pip install .
  torchrun --nnodes=$NUM_NODES --nproc_per_node=$LOCAL_WORLD_SIZE \
    --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m dais26_dentex.train.cli
```

`air` exports the rendezvous env vars (`$NUM_NODES`, `$LOCAL_WORLD_SIZE`, `$NODE_RANK`,
`$MASTER_ADDR`, `$MASTER_PORT`), so the same `command` scales to multi-node unchanged.

## Launch an HPO sweep

One `torchrun` allocation runs a whole campaign stage (sequential trials + winner retrains + the
`@challenger` best-in-experiment gate):

```bash
air run -f air/workload_sweep.yaml --watch -p df1 --override parameters.stage=cradio_s2
```

Stages live in `config.campaigns.CAMPAIGN_STAGES` (the "push to 0.60" chain, see
[HPO campaign log](../HPO.md)); the cheap `smoke` stage (1 trial × 1 epoch) validates the lane.
For DINOv3 stages, uncomment the `HF_TOKEN` secret block in `workload_sweep.yaml`. Full sweep
walkthrough: [HPO sweep](../lifecycle/hpo-sweep.md).

## Override at submit time

`--override` takes space-separated `dotted.key=value` pairs. `parameters.*` keys land in
`$HYPERPARAMETERS_PATH`; top-level keys edit the workload spec.

```bash
# Full schedule instead of the demo override:
air run -f air/workload_train_detector.yaml --override parameters.epochs=150 --watch -p df1

# Switch the whole target (catalog/schema/experiment):
air run -f air/workload_train_detector.yaml --override parameters.env=prod --watch -p df1

# Override one location of the named env:
air run -f air/workload_train_detector.yaml --override parameters.schema=my_sandbox --watch -p df1

# Scale to 2 nodes (16 H100s) + bigger budget:
air run -f air/workload_train_detector.yaml \
  --override compute.num_accelerators=16 timeout_minutes=720 --watch -p df1
```

You can also drop a (non-git-ignored) `environments.local.yaml` at the repo root — air's snapshot
carries it to the pod. See [Per-user environment overrides](../scenarios/env-overrides.md).

## Inspect / manage runs

```bash
air list runs --limit 10 -p df1
air get run <run-id> -p df1
air logs <run-id> -p df1                  # stream (running) or last 10k lines (completed)
air logs <run-id> --node 1 -p df1         # a specific node (0..num_nodes-1; default 0)
air logs <run-id> --download-to ./logs -p df1
air cancel <run-id> -p df1
air cancel --all -p df1
```

## DINOv3 (gated backbone)

`workload_train_detector_dinov3.yaml` already activates the `secrets:` block;
`workload_sweep.yaml` has it commented for DINOv3 stages. `load_backbone` reads
`os.environ["HF_TOKEN"]`. Create the scope + secret once:

```bash
databricks secrets create-scope dais26-secrets
databricks secrets put-secret dais26-secrets hf-token
```

See [Switch backbone](../scenarios/switch-backbone.md).

## Notes & gotchas

- **AIR base env v4** already ships torch/torchvision/transformers/mlflow; `requirements.yaml`
  adds only what's missing.
- **`HF_HUB_ENABLE_HF_TRANSFER=0`** is set at the workload `env_variables` level *before* Python
  imports `huggingface_hub` — the parallel chunked downloader fails on UC Volume FUSE. See
  [HF transfer / FUSE incompatibility](../RUNBOOK.md#hf-transfer-fuse-incompat).
- **`MODEL_URI=` missing from rank-0 stdout** → training crashed in `_save_and_register` on rank 0;
  inspect `air logs <run-id>`. More AIR-launch failure modes:
  [Troubleshooting](../reference/troubleshooting.md).
- The notebook lane uses `serverless_gpu.@distributed` (no `torchrun`); this lane uses `torchrun`.
  `serverless_gpu` is **not** needed in the CLI flow.
