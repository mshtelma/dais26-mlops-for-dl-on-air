# Quickstart — air CLI lane

Goal: the same train-to-**`@challenger`** result as the [DAB quickstart](quickstart-dab.md), but
launched from a terminal with the Databricks **AIR CLI** — no notebook, no `bundle deploy`
required for training itself.

Complete [Install & authenticate](installation.md) first, including a **named profile**
(`--profile df1`).

## Step 1 — Install the AIR CLI (one-time)

```bash
pip install databricks-air            # or: uv tool install databricks-air
air --version
```

The AIR CLI is in **Beta**. Authenticate the profile the workloads use:

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile df1
```

All `air` commands below take `-p df1` (or set `DATABRICKS_CONFIG_PROFILE=df1`).

## Step 2 — Launch training

From the repo root:

```bash
air run -f air/workload_train_detector.yaml --watch -p df1
```

What the workload does:

1. **Snapshots** the repo to the GPU pod (`code_source.snapshot.root_path: ..` resolves to the
   repo root). The working tree ships as-is — staged + unstaged edits, `.gitignore` respected —
   so local iteration needs no commit. For a reproducible run, pin `git: { commit: <40-char SHA> }`.
2. **Installs** `air/requirements.yaml` (the AIR base env v4 already supplies torch, torchvision,
   transformers, mlflow, …).
3. `cd "$CODE_SOURCE_PATH" && pip install .` so `dais26_dentex` is importable (no `-e`; the
   snapshot is read-only).
4. Launches `torchrun … -m dais26_dentex.train.cli` across the 8 H100s of one node.
5. The CLI reads `$HYPERPARAMETERS_PATH` (the `parameters:` block), resolves the named
   `recipe: cradio_v4_so400m` and `env: df1`, builds + validates a `TrainerConfig`, and runs the
   same `Trainer` core as the DAB lane — logging into the same shared MLflow experiment,
   registering the model, and setting `@challenger`.

The `parameters:` block carries only names + deliberate overrides (note the demo `epochs: 50`):

```yaml
parameters:
  recipe: cradio_v4_so400m   # config.recipes literal, NOT the HF id
  env: df1                   # config.environments entry
  register_model: true
  set_candidate_alias: true
  epochs: 50                 # demo override of the recipe's 150-epoch schedule
```

## Step 3 — Inspect the run

```bash
air list runs --limit 10 -p df1
air get run <run-id> -p df1
air logs <run-id> -p df1                  # stream (running) or last 10k lines (completed)
air logs <run-id> --download-to ./logs -p df1
air cancel <run-id> -p df1
```

## Success criteria

Identical to the DAB lane: rank 0 logs the MLflow run, registers the detector model, and sets
`@challenger`. Both lanes write to the same experiment, so the promotion gates treat their runs
identically.

## Common submit-time overrides

```bash
# Full 150-epoch recipe schedule instead of the demo 50:
air run -f air/workload_train_detector.yaml --override parameters.epochs=150 --watch -p df1

# Point at a different environment (catalog/schema/experiment all switch):
air run -f air/workload_train_detector.yaml --override parameters.env=prod --watch -p df1

# Override a single location of the named env:
air run -f air/workload_train_detector.yaml --override parameters.schema=my_sandbox --watch -p df1

# Scale to 2 nodes (16 H100s) and raise the wall-clock budget:
air run -f air/workload_train_detector.yaml \
  --override compute.num_accelerators=16 timeout_minutes=720 --watch -p df1
```

## What's next

| Next step | Page |
|---|---|
| The DAB equivalent | [Quickstart — DAB lane](quickstart-dab.md) |
| Full air lane reference (multi-node, run mgmt, DINOv3) | [air CLI lane](../lanes/air.md) |
| HPO sweeps from the terminal | [HPO sweep](../lifecycle/hpo-sweep.md) |
| Promote to `@champion` | [Evaluate → approve → promote](../lifecycle/evaluate-approve-promote.md) |

If anything failed, see [Troubleshooting](../reference/troubleshooting.md) (the AIR CLI section).
