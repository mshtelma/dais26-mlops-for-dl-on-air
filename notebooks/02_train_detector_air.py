# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Train Detector on AIR (multi-GPU via `@distributed`)
# MAGIC
# MAGIC Fine-tunes the FPN + RetinaNet head over a **frozen C-RADIOv4-SO400M** backbone on DENTEX.
# MAGIC The notebook driver runs on serverless compute; the `@distributed` decorator dispatches
# MAGIC the actual training to the H100 serverless GPU pool. No traditional ML cluster involved.
# MAGIC
# MAGIC Logs to MLflow, registers the model in UC, sets `@candidate` alias (NOT `@champion` —
# MAGIC promotion happens after smoke test in the `deploy_endpoint` task).

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
# Hyperparameters come from notebooks/00_config.py (TRAIN_* constants).
# Module-level constants pulled in via `%run ./00_config` capture cleanly into
# the @distributed closure, so dbutils / spark are NOT needed on workers.
model_name = DETECTOR_LORA_MODEL_SHORT if TRAIN_USE_LORA else DETECTOR_MODEL_SHORT

# COMMAND ----------
from serverless_gpu import distributed


@distributed(gpus=TRAIN_GPUS, gpu_type=TRAIN_GPU_TYPE)
def run_train():
    # Body runs in the serverless GPU pool, not on the driver — no spark / dbutils.
    # HF env must be set BEFORE any HF import on this worker (constants-module
    # import locks the values). The `train_detector` import below is intentionally
    # deferred for the same reason. See docs/RUNBOOK.md#hf-transfer-fuse-incompat.
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    # MLflow auto-resolves the experiment from MLFLOW_EXPERIMENT_NAME if no
    # explicit set_experiment call has run yet. Set inside the worker because
    # AIR workers don't inherit the driver's process env.
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME

    from dais26_dentex.train.train_detector import train_detector

    return train_detector(
        catalog=CATALOG,
        schema=SCHEMA,
        backbone_name=BACKBONE,                                        # type: ignore[arg-type]
        backbone_revision=BACKBONE_REVISION,
        volume_path=VOLUME_PATH,
        cache_dir=CACHE_DIR,
        epochs=TRAIN_EPOCHS,
        lr=TRAIN_LR,
        batch_size=TRAIN_BATCH_SIZE,
        use_lora=TRAIN_USE_LORA,
        lora_rank=TRAIN_LORA_RANK,
        lora_alpha=TRAIN_LORA_ALPHA,
        model_name=model_name,
        experiment_name=EXPERIMENT_NAME,
        register_model=True,
        set_candidate_alias=True,
    )


# Set MLFLOW_EXPERIMENT_NAME on the driver BEFORE .distributed() — both for any
# driver-side mlflow ops (today there are none, but cheap insurance) and so the
# value is in place at closure-capture time. The same line is repeated inside
# run_train() because AIR workers run in fresh processes that don't inherit
# driver env.
import os
os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME

results = run_train.distributed()        # returns list of per-rank return values
# Only rank 0 returns the run_id; other ranks return None.
run_id = next((r for r in results if r), None)
print(f"Training run id: {run_id}")

# COMMAND ----------
dbutils.jobs.taskValues.set(key="run_id", value=run_id)
