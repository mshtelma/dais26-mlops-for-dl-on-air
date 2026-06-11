# air launch surface (Databricks AIR CLI)

This directory holds the [Databricks **AIR CLI**](https://preview.docs.databricks.com/pr-2058861/aws/en/machine-learning/ai-runtime/cli)
(`air`) workloads for running the detector training and HPO sweep from a terminal
— no notebook required, no traditional ML cluster involved. `air` submits a YAML
spec to Databricks Serverless GPU and (with `--watch`) streams the logs back.

> The AIR CLI is in **Beta**. Install it with `pip install databricks-air`.
> `air -h` and `air config -h` are the always-current command / YAML-field
> reference; this README captures the project-specific usage.

## Files

| File | Purpose |
|---|---|
| `workload_train_detector.yaml` | C-RADIOv4 training (`recipe: cradio_v4_so400m`). |
| `workload_train_detector_dinov3.yaml` | DINOv3 training (`recipe: dinov3_vitl16`, gated — needs the `dais26-secrets/hf-token` secret, already wired). |
| `workload_sweep.yaml` | HPO campaign stage via `train.sweep_cli` (same `SweepRunner` as notebook 02b). |
| `requirements.yaml` | Python deps not in the AIR base env v4 image. |

Hyperparameters are **not** listed in the workload YAMLs: each training workload
names a **recipe** (`dais26_dentex.config.recipes.RECIPES` — the campaign-final
best-known config, the same source the notebook lane builds from) and the sweep
workload names a **stage** (`dais26_dentex.config.campaigns.CAMPAIGN_STAGES`).
The `parameters:` blocks carry only environment values (catalog / schema / paths /
experiment) plus explicit, deliberate overrides such as the demo-time `epochs`.

## One-time setup

```bash
# Install the AIR CLI (Beta). Isolate it with uv if you prefer:
pip install databricks-air            # or: uv tool install databricks-air

# Authenticate to your workspace; this writes the `df1` profile to ~/.databrickscfg
databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile df1

air --version
```

All `air` commands below take `-p df1` to select that profile (or set
`DATABRICKS_CONFIG_PROFILE=df1`).

## Launch training

From anywhere (paths inside the YAML resolve relative to the YAML file):

```bash
air run -f air/workload_train_detector.yaml --watch -p df1
```

The workload:

1. **Snapshots** the repo to the GPU pod. `code_source.snapshot.root_path: ..`
   resolves to the repo root (relative to the YAML's own directory — `.` would
   snapshot only `air/`). The working tree ships as-is by default (staged +
   unstaged edits, `.gitignore` respected) so local iteration needs no commit;
   for a reproducible release run, pin `git: { commit: <40-char SHA> }` instead.
2. **Installs** `requirements.yaml` (AIR base env v4 already supplies torch,
   torchvision, transformers, mlflow, …).
3. `cd "$CODE_SOURCE_PATH" && pip install .` so `dais26_dentex` is importable
   (no `-e`; the snapshot is read-only).
4. Launches `torchrun --nnodes=$NUM_NODES --nproc_per_node=$LOCAL_WORLD_SIZE
   --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT
   -m dais26_dentex.train.cli` across the 8 H100s of one node — `air` exports the
   rendezvous env vars, so the same `command` scales to multi-node unchanged.
5. The CLI reads `$HYPERPARAMETERS_PATH` (the `parameters:` block `air` wrote),
   resolves the named `recipe:` from `config.recipes.RECIPES`, applies the YAML's
   explicit overrides, builds + validates a `TrainerConfig`, and runs `Trainer`.
6. The CLI clears the ambient `MLFLOW_RUN_ID` (`air` sets it for the workload's
   **own** run) so rank 0 logs the training run into the shared
   `dais26_vfm_experiment` named by `parameters.experiment_name`, registers the
   model in UC, and sets `@challenger` — exactly what the notebook quickstart
   produces.

## Launch a hyperparameter sweep

The same `SweepRunner` that powers the `campaign_sweep` DAB job runs from the
terminal — one torchrun allocation executes a whole campaign stage (sequential
trials + winner retrains + the `@challenger` best-in-experiment gate):

```bash
air run -f air/workload_sweep.yaml --watch -p df1 \
  --override parameters.stage=cradio_s2

# rank 0 (master = node 0) prints SWEEP_* summary lines:
air logs <run-id> -p df1
```

Stages live in `dais26_dentex.config.campaigns.CAMPAIGN_STAGES` (the "push to
0.60" chain, docs/HPO.md); the cheap `smoke` stage (1 trial × 1 epoch) is there
for lane validation. For DINOv3 stages, uncomment the `HF_TOKEN` secret in
`workload_sweep.yaml`.

## Override at submit time

`--override` takes space-separated `dotted.key=value` pairs. `parameters.*` keys
land in `$HYPERPARAMETERS_PATH`; top-level keys edit the workload spec.

```bash
# Full 150-epoch recipe schedule instead of the 50-epoch demo override:
air run -f air/workload_train_detector.yaml \
  --override parameters.epochs=150 \
  --watch -p df1

# Scale to 2 nodes (16 H100s) and raise the wall-clock budget:
air run -f air/workload_train_detector.yaml \
  --override compute.num_accelerators=16 timeout_minutes=720 \
  --watch -p df1

# Point a run at your own experiment:
air run -f air/workload_train_detector.yaml \
  --override parameters.experiment_name=/Users/<you>/dais26_vfm_experiment \
  --watch -p df1
```

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

## Notes

- **Two MLflow surfaces, deliberately distinct.** The workload's top-level
  `experiment_name` / `mlflow_run_name` track the `air` run itself;
  `parameters.experiment_name` aligns the **training** run into the shared
  `dais26_vfm_experiment` the sweep / deployment-job gates read.
- **Multi-node** is just a bigger `compute.num_accelerators` (16 / 24 / 32) —
  `air` spreads it across nodes and the `command`'s `$NUM_NODES` /
  `$LOCAL_WORLD_SIZE` / `$NODE_RANK` arithmetic feeds torchrun correctly.
- The notebook entrypoint (`notebooks/02_train_detector_air.py`) and this air
  workload share the same `TrainerConfig` + `Trainer` core. The notebook path
  uses the local `serverless_gpu.@distributed` helper and does **not** use
  `torchrun`; this air path uses `torchrun`.
- All quickstart training is **AIR-only** on single-node 8xH100 by default — no
  traditional ML clusters.
- YAML field reference: `air config -h` (and `air config.<field> -h`). Command
  reference: `air <command> -h`.
