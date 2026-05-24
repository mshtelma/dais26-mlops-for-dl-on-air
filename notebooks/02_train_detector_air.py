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
# MAGIC %pip install --quiet serverless_gpu pyyaml
# MAGIC %pip install --quiet /Workspace/Users/$(whoami)/dais26/dist/*.whl
# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
dbutils.widgets.text("epochs", "10")
dbutils.widgets.text("lr", "1e-3")
dbutils.widgets.text("batch_size", "8")
dbutils.widgets.dropdown("use_lora", "false", ["true", "false"])
dbutils.widgets.text("lora_rank", "8")
dbutils.widgets.text("lora_alpha", "32")
dbutils.widgets.text("gpus", "8")
dbutils.widgets.dropdown("gpu_type", "h100", ["h100", "a10"])

# COMMAND ----------
# Resolve driver-side values. dbutils / spark are NOT available inside the
# @distributed-wrapped function — anything that depends on them must be
# captured into a closure variable here. The shared config constants
# (CATALOG, SCHEMA, BACKBONE, BACKBONE_REVISION, VOLUME_PATH, CACHE_DIR,
# EXPERIMENT_NAME) come from `%run ./00_config` above; they are
# module-level constants and capture cleanly into the closure.
epochs = int(dbutils.widgets.get("epochs"))
lr = float(dbutils.widgets.get("lr"))
batch_size = int(dbutils.widgets.get("batch_size"))
use_lora = dbutils.widgets.get("use_lora") == "true"
lora_rank = int(dbutils.widgets.get("lora_rank"))
lora_alpha = float(dbutils.widgets.get("lora_alpha"))
gpus = int(dbutils.widgets.get("gpus"))
gpu_type = dbutils.widgets.get("gpu_type")

model_name = DETECTOR_LORA_MODEL_SHORT if use_lora else DETECTOR_MODEL_SHORT

# COMMAND ----------
from serverless_gpu import distributed

from src.train.train_detector import train_detector


@distributed(gpus=gpus, gpu_type=gpu_type)
def run_train():
    # IMPORTANT: this function body runs in the serverless GPU pool, not on the driver.
    # No spark / dbutils / driver-side imports allowed here.
    return train_detector(
        catalog=CATALOG,
        schema=SCHEMA,
        backbone_name=BACKBONE,                                        # type: ignore[arg-type]
        backbone_revision=BACKBONE_REVISION,
        volume_path=VOLUME_PATH,
        cache_dir=CACHE_DIR,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        model_name=model_name,
        experiment_name=EXPERIMENT_NAME,
        register_model=True,
        set_candidate_alias=True,
    )


results = run_train.distributed()        # returns list of per-rank return values
# Only rank 0 returns the run_id; other ranks return None.
run_id = next((r for r in results if r), None)
print(f"Training run id: {run_id}")

# COMMAND ----------
dbutils.jobs.taskValues.set(key="run_id", value=run_id)
