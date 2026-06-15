# 2 · Train & register @challenger

Train the detection head (and optionally fine-tune the backbone) on DENTEX, then register a UC
model version and set the **`@challenger`** alias. Both lanes run the same `Trainer` core and log
to the same MLflow experiment.

## Launch

=== "DAB"

    ```bash
    databricks bundle run train_detector -t dev
    ```

    Runs `setup → train → confirm_challenger`. The `train` task runs
    `notebooks/02_train_detector_air.py` on one `GPU_8xH100` task via
    `serverless_gpu.@distributed` (no `torchrun`). The `confirm_challenger` task
    (`04_deploy_serving.py`, `deploy_action=register_and_set_candidate`) just resolves
    `@challenger` and fails loudly if absent.

=== "air CLI"

    ```bash
    air run -f air/workload_train_detector.yaml --watch -p df1
    ```

    Snapshots the repo → `pip install .` → `torchrun … -m dais26_dentex.train.cli` across 8 H100s.
    The CLI resolves `recipe`/`env`, builds `TrainerConfig`, and runs the same `Trainer`.

## What the notebook `@distributed` path looks like

```python title="notebooks/02_train_detector_air.py (shape)"
from serverless_gpu import distributed
from dais26_dentex.config.recipes import build_trainer_config
from dais26_dentex.train.trainer import Trainer

@distributed(gpus=TRAIN_GPUS, gpu_type=TRAIN_GPU_TYPE)   # 8, "h100"
def run_train():
    # in-worker: re-set HF env + MLflow experiment — AIR workers DON'T inherit driver env
    cfg = build_trainer_config(BACKBONE, catalog=CATALOG, schema=SCHEMA, epochs=TRAIN_EPOCHS, ...)
    return Trainer(cfg).run()

results = run_train.distributed()
run_id  = next((r for r in results if r), None)   # rank 0 only returns a value
```

!!! danger "AIR workers do not inherit driver env — set it inside the worker"
    `cloudpickle` resolves free variables on the worker eagerly, so the `@distributed` body must
    re-set `HF_HUB_ENABLE_HF_TRANSFER=0` (+ `HF_HUB_DISABLE_XET=1`) and
    `MLFLOW_EXPERIMENT_NAME` **before** `from dais26_dentex …`. See
    [HF transfer / FUSE incompatibility](../RUNBOOK.md#hf-transfer-fuse-incompat) and the
    canonical pattern in `02_train_detector_air.py`.

## Inside `Trainer.run()`

1. `build_detector(...)` — wrapped in `rank0_first` to dodge the cold-cache HuggingFace download
   race (see [HF cache race](../RUNBOOK.md#hf-cache-race)).
2. DDP wrap (`find_unused_parameters=True` for frozen/lora/partial — the frozen subtree has no
   grads; `False` only for `full`).
3. `_epoch_loop` → `_validate` (COCO mAP on val), tracking the best checkpoint.
4. `_save_and_register` (**rank 0 only**):
    - log the pyfunc via **models-from-code** (`serve/detector_model_script.py`) + `code_paths`,
      with `pip_requirements = serving_pip_requirements()` (from `[tool.dais26.serving-deps]`);
    - register the version **from the LoggedModel URI** (`mlflow.register_model("models:/<id>")`)
      so lineage survives the later cross-schema champion copy;
    - set the `@challenger` alias;
    - re-log the best-epoch `val/*` metrics against the LoggedModel so they render on the Models tab.

All other ranks return `None`. See [Models-from-code serving load path](../RUNBOOK.md#models-from-code).

## The recipe and the demo override

Hyperparameters come from the per-backbone recipe (`config/recipes.py`). The C-RADIOv4 recipe is
the `dazzling-mole-850` config (full fine-tune, per-level anchors, 150 epochs, val mAP@50 0.5931).
For the quickstart, `TRAIN_EPOCHS = 50` (notebook) / `epochs: 50` (air) overrides the 150-epoch
schedule to keep wall time ≈2h.

```bash
# Run the full recipe schedule instead of the demo override:
# DAB:  set TRAIN_EPOCHS = 150 in notebooks/00_config.py
# air:
air run -f air/workload_train_detector.yaml --override parameters.epochs=150 --watch -p df1
```

To fine-tune the backbone differently, set `backbone_mode` (`frozen`/`lora`/`partial`/`full`) and
`backbone_lr` — see [Configuration reference](../reference/configuration.md) and
[HPO sweep](hpo-sweep.md).

## Verify

=== "DAB"

    ```bash
    databricks jobs get-run <run-id>      # TERMINATED / SUCCESS
    ```

    `confirm_challenger` prints `@challenger -> version <n>`.

=== "air CLI"

    ```bash
    air logs <run-id> -p df1 | grep -E "MODEL_URI|challenger"
    ```

    Rank 0 prints the registered model URI and sets `@challenger`.

Confirm the alias programmatically:

```python
from mlflow.tracking import MlflowClient
c = MlflowClient(registry_uri="databricks-uc")
mv = c.get_model_version_by_alias("main.mshtelma.cradio_detector", "challenger")
print(f"@challenger = version {mv.version}")
```

## What a new `@challenger` triggers

If the deployment job is wired (`connect_deployment_job`), registering a new `@challenger` version
**auto-triggers** `deploy_job_detector` (Evaluation → Approval → RegisterChampion). See
[Evaluate → approve → promote](evaluate-approve-promote.md). Without the wiring, training just
leaves `@challenger` set for you to promote manually.

Next: **[HPO sweep](hpo-sweep.md)** or **[Evaluate → approve → promote](evaluate-approve-promote.md)**.
