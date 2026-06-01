# Databricks notebook source
# MAGIC %md
# MAGIC # 02b — Detector HPO sweep (sequential `@distributed` AIR trials)
# MAGIC
# MAGIC Runs a hyperparameter sweep over the detector recipe to lift it off the
# MAGIC ~3% mAP@50 plateau. Each trial is a full 8xH100 AIR training job launched via
# MAGIC `@distributed`; trials run **sequentially** (the GPU pool runs one job at a
# MAGIC time) and are logged as **nested MLflow runs** under a single parent sweep run.
# MAGIC
# MAGIC Search strategy, trial count, per-trial epochs, and the search space all come
# MAGIC from `SWEEP_*` in `00_config.py`. The pure trial-enumeration + winner-selection
# MAGIC logic lives in `dais26_dentex.train.sweep` (unit-tested); this notebook owns the
# MAGIC side effects (dispatch, MLflow nesting, winner registration).
# MAGIC
# MAGIC Flow:
# MAGIC 1. (optional) calibrate anchors once on the driver from the train split.
# MAGIC 2. start a parent run; for each trial build a `TrainerConfig` (00_config defaults
# MAGIC    + trial override), train with `register_model=False`, tag the run as nested.
# MAGIC 3. pick the winner by `SWEEP_PRIMARY_METRIC`; retrain it at `TRAIN_EPOCHS` with
# MAGIC    `register_model=True` + `@candidate` (only the winner is registered).
# MAGIC
# MAGIC Requires the orchestrating job to carry the 8h timeout (see `resources/jobs`).

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import os

import mlflow
from mlflow.tracking import MlflowClient

from dais26_dentex.train.sweep import TrialResult, iter_trials, select_best

# HF token for gated backbones (DINOv3); empty string is a no-op for C-RADIO.
try:
    hf_token = dbutils.secrets.get("dais26-secrets", "hf-token")
except Exception:
    hf_token = ""

model_name = DETECTOR_MODEL_SHORT
os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME
client = MlflowClient()

# COMMAND ----------
# ---- Optional: calibrate anchors once on the driver (for the "calibrated" toggle) ----
# calibrate_anchors / calibrate_aspect_ratios read the canonical COCO train split at
# {VOLUME_PATH}/annotations/train.json (written by 01) and fit anchor geometry to the
# DENTEX box distribution. Computed here so each trial just receives a plain list.
CALIBRATED_SCALES = None
CALIBRATED_RATIOS = None
if any("calibrated" in str(v) for v in SWEEP_SEARCH_SPACE.get("anchor_mode", [])):
    from dais26_dentex.models.detection_head import calibrate_anchors, calibrate_aspect_ratios

    _ann = f"{VOLUME_PATH}/annotations/train.json"
    CALIBRATED_SCALES = calibrate_anchors(_ann)
    CALIBRATED_RATIOS = calibrate_aspect_ratios(_ann)
    print(f"Calibrated anchors: scales={CALIBRATED_SCALES} ratios={CALIBRATED_RATIOS}")


def _resolve_overrides(params: dict) -> dict:
    """Map a trial's sampled params onto TrainerConfig kwargs.

    `anchor_mode` is translated into concrete `anchor_scales` / `aspect_ratios`
    (None keeps the module defaults); every other key is a TrainerConfig field
    passed through unchanged.
    """
    override = dict(params)
    mode = override.pop("anchor_mode", "default")
    if mode == "calibrated" and CALIBRATED_SCALES is not None:
        override["anchor_scales"] = list(CALIBRATED_SCALES)
        override["aspect_ratios"] = list(CALIBRATED_RATIOS)
    return override


# COMMAND ----------
def _train_config_kwargs(override: dict, *, epochs: int, register: bool) -> dict:
    """Base TrainerConfig kwargs (00_config defaults) merged with a trial override."""
    base = dict(
        catalog=CATALOG,
        schema=SCHEMA,
        backbone_name=BACKBONE,
        backbone_revision=BACKBONE_REVISION,
        volume_path=VOLUME_PATH,
        cache_dir=CACHE_DIR,
        epochs=epochs,
        batch_size=TRAIN_BATCH_SIZE,
        experiment_name=EXPERIMENT_NAME,
        model_name=model_name,
        register_model=register,
        set_candidate_alias=register,
    )
    return {**base, **override}


def _run_distributed_training(cfg_kwargs: dict) -> str | None:
    """Launch one @distributed AIR training job; return rank-0 run_id."""
    from serverless_gpu import distributed

    @distributed(gpus=TRAIN_GPUS, gpu_type=TRAIN_GPU_TYPE)
    def _run():
        import os as _os

        _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        _os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
        if hf_token:
            _os.environ["HF_TOKEN"] = hf_token
        _os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME

        from dais26_dentex.config.trainer_config import TrainerConfig
        from dais26_dentex.train.trainer import Trainer

        cfg = TrainerConfig(**cfg_kwargs)
        return Trainer(cfg).run()

    results = _run.distributed()
    return next((r for r in results if r), None)


# COMMAND ----------
# ---- Run the sweep: parent run + sequential nested trials ----
trials = list(
    iter_trials(
        SWEEP_SEARCH_SPACE,
        strategy=SWEEP_STRATEGY,
        max_trials=SWEEP_MAX_TRIALS,
        seed=SWEEP_SEED,
    )
)
print(f"Planned {len(trials)} trials ({SWEEP_STRATEGY}, seed={SWEEP_SEED}).")

results: list[TrialResult] = []
with mlflow.start_run(run_name=f"hpo-sweep-{BACKBONE}") as parent:
    parent_run_id = parent.info.run_id
    mlflow.log_params(
        {
            "sweep_strategy": SWEEP_STRATEGY,
            "sweep_max_trials": SWEEP_MAX_TRIALS,
            "sweep_trial_epochs": SWEEP_TRIAL_EPOCHS,
            "sweep_primary_metric": SWEEP_PRIMARY_METRIC,
            "sweep_backbone": BACKBONE,
        }
    )

    for trial in trials:
        override = _resolve_overrides(trial.params)
        print(f"\n=== Trial {trial.trial_id}/{len(trials) - 1}: {override} ===")
        cfg_kwargs = _train_config_kwargs(override, epochs=SWEEP_TRIAL_EPOCHS, register=False)
        run_id = _run_distributed_training(cfg_kwargs)

        metric = None
        if run_id:
            # Nest the trial's training run under the parent for the UI, and read
            # back the best metric the Trainer logged.
            client.set_tag(run_id, "mlflow.parentRunId", parent_run_id)
            client.set_tag(run_id, "sweep_trial_id", str(trial.trial_id))
            metric = client.get_run(run_id).data.metrics.get(SWEEP_PRIMARY_METRIC)
        results.append(TrialResult(trial_id=trial.trial_id, params=override, metric=metric, run_id=run_id))
        print(f"Trial {trial.trial_id}: run_id={run_id} {SWEEP_PRIMARY_METRIC}={metric}")

    # ---- Rank + pick winner ----
    ranked = sorted(results, key=lambda r: (r.metric is not None, r.metric or -1.0), reverse=True)
    print("\n=== Sweep results (best first) ===")
    for r in ranked:
        print(f"trial {r.trial_id:>2}  {SWEEP_PRIMARY_METRIC}={r.metric}  {r.params}")

    best = select_best(results, higher_is_better=True)
    if best is None:
        print("No trial produced a metric — nothing to register.")
    else:
        mlflow.log_params({"winner_trial_id": best.trial_id, "winner_run_id": best.run_id or ""})
        mlflow.log_metric(f"winner_{SWEEP_PRIMARY_METRIC.replace('/', '_')}", best.metric or 0.0)
        print(f"\nWinner: trial {best.trial_id} ({SWEEP_PRIMARY_METRIC}={best.metric}) params={best.params}")

# COMMAND ----------
# ---- Retrain the winner at full length and register @candidate (only the winner) ----
if best is not None and SWEEP_REGISTER_WINNER:
    print(f"Retraining winner at {TRAIN_EPOCHS} epochs and registering as {model_name}...")
    winner_kwargs = _train_config_kwargs(best.params, epochs=TRAIN_EPOCHS, register=True)
    winner_run_id = _run_distributed_training(winner_kwargs)
    print(f"Winner registered. run_id={winner_run_id}")
    dbutils.jobs.taskValues.set(key="run_id", value=winner_run_id)
else:
    dbutils.jobs.taskValues.set(key="run_id", value=(best.run_id if best else None))
