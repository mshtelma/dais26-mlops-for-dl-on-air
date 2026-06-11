# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Train Detector on AIR (multi-GPU via `@distributed`)
# MAGIC
# MAGIC Trains the FPN + RetinaNet detector over the configured backbone on DENTEX,
# MAGIC using the backbone's **campaign-final recipe** from
# MAGIC `dais26_dentex.config.recipes` (the same recipe the air lane names in its
# MAGIC workload YAML). This is the DAB quickstart launcher: the job task runs on one
# MAGIC `GPU_8xH100` AIR notebook environment and uses the local
# MAGIC `serverless_gpu.@distributed` helper to use the task's eight H100s. This path
# MAGIC does **not** use `torchrun`.
# MAGIC
# MAGIC Logs to MLflow, registers the model in UC, and sets `@challenger`. Endpoint
# MAGIC deploy, smoke test, and `@champion` promotion live in separate operator lanes.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
# Hyperparameters come from the per-backbone recipe in
# `dais26_dentex.config.recipes` — the same source the air lane resolves via
# its `recipe:` parameter — so both quickstarts train the best-known
# (campaign-final) config. Only environment values (UC names, experiment) and
# the explicit demo-time overrides below come from 00_config. Module-level
# constants pulled in via `%run ./00_config` capture cleanly into the
# @distributed closure, so dbutils / spark are NOT needed on workers.
model_name = DETECTOR_LORA_MODEL_SHORT if TRAIN_USE_LORA else DETECTOR_MODEL_SHORT

# Explicit, visible overrides on the recipe. TRAIN_EPOCHS keeps the quickstart
# inside demo wall-time (the recipe's full schedule is 150 epochs); the LoRA
# block is the stretch path (recipes default to the campaign-winning full
# fine-tune).
train_overrides: dict = {"epochs": TRAIN_EPOCHS}
if TRAIN_USE_LORA:
    train_overrides.update(
        backbone_mode="lora",
        use_lora=True,
        lora_rank=TRAIN_LORA_RANK,
        lora_alpha=TRAIN_LORA_ALPHA,
    )

# HF token for gated backbones (e.g. DINOv3). Read on the driver from the
# secret scope and capture as a plain-string local so it flows into the
# @distributed closure (AIR workers don't inherit driver env or dbutils).
# Guarded so the C-RADIO path still works when the scope/secret is absent.
try:
    hf_token = dbutils.secrets.get("dais26-secrets", "hf-token")
except Exception:
    hf_token = ""

# COMMAND ----------
from serverless_gpu import distributed


@distributed(gpus=TRAIN_GPUS, gpu_type=TRAIN_GPU_TYPE)
def run_train():
    # Body runs on the GPU workers, not on the notebook driver - no spark / dbutils.
    # HF env must be set BEFORE any HF import on this worker. The trainer imports
    # below are intentionally deferred for the same reason. See
    # docs/RUNBOOK.md#hf-transfer-fuse-incompat.
    import os

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    # Gated-model auth for the backbone download. `load_backbone` reads
    # os.environ.get("HF_TOKEN"); set it here (BEFORE the deferred
    # trainer import) from the closure-captured driver value. No-op for
    # the C-RADIO path when the secret is absent (empty string).
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    # MLflow auto-resolves the experiment from MLFLOW_EXPERIMENT_NAME if no
    # explicit set_experiment call has run yet. Set inside the worker because
    # AIR workers don't inherit the driver's process env.
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME

    from dais26_dentex.config.recipes import build_trainer_config
    from dais26_dentex.train.trainer import Trainer

    cfg = build_trainer_config(
        BACKBONE,
        catalog=CATALOG,
        schema=SCHEMA,
        backbone_revision=BACKBONE_REVISION,
        volume_path=VOLUME_PATH,
        cache_dir=CACHE_DIR,
        model_name=model_name,
        experiment_name=EXPERIMENT_NAME,
        register_model=True,
        set_candidate_alias=True,
        **train_overrides,
    )
    return Trainer(cfg).run()


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
