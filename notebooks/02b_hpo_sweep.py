# Databricks notebook source
# MAGIC %md
# MAGIC # 02b — Detector HPO sweep (sequential `@distributed` AIR trials)
# MAGIC
# MAGIC Thin launcher over `dais26_dentex.train.sweep_runner.SweepRunner` — the same
# MAGIC sweep brain the terminal lane runs via `air/workload_sweep.yaml`. This
# MAGIC notebook owns only the lane-specific part: each trial is dispatched as one
# MAGIC `serverless_gpu.@distributed` 8xH100 job.
# MAGIC
# MAGIC Stage selection: the `sweep_stage` job parameter (set by
# MAGIC `resources/jobs/campaign_sweep.yml`) or `SWEEP_STAGE` in `00_config.py` picks
# MAGIC a stage from `dais26_dentex.config.campaigns.CAMPAIGN_STAGES`; `None` runs the
# MAGIC legacy post-fix sweep (`SWEEP_DEFAULTS`). The runner handles trial
# MAGIC enumeration, MLflow parent/child nesting, the winner's schedule retrains, and
# MAGIC the `@challenger` best-in-experiment registration gate. See docs/HPO.md.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

from dais26_dentex.config.campaigns import CAMPAIGN_STAGES, SWEEP_DEFAULTS
from dais26_dentex.config.recipes import DETECTOR_NAMES_BY_BACKBONE
from dais26_dentex.train.sweep_runner import SweepRunner, SweepSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# HF token for gated backbones (DINOv3); empty string is a no-op for C-RADIO.
try:
    hf_token = dbutils.secrets.get("dais26-secrets", "hf-token")
except Exception:
    hf_token = ""

os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------
# ---- Resolve the active stage: job parameter > 00_config.SWEEP_STAGE > legacy ----
try:
    dbutils.widgets.text("sweep_stage", "")
    _stage_param = dbutils.widgets.get("sweep_stage").strip()
    if _stage_param:
        SWEEP_STAGE = _stage_param
except Exception:
    pass

if SWEEP_STAGE:
    if SWEEP_STAGE not in CAMPAIGN_STAGES:
        raise ValueError(
            f"SWEEP_STAGE={SWEEP_STAGE!r} not in CAMPAIGN_STAGES ({sorted(CAMPAIGN_STAGES)}). "
            "Set it to None for the legacy sweep."
        )
    spec = SweepSpec.from_stage(SWEEP_STAGE, CAMPAIGN_STAGES[SWEEP_STAGE])
else:
    spec = SweepSpec.from_stage("legacy", SWEEP_DEFAULTS)

model_name = DETECTOR_NAMES_BY_BACKBONE[spec.backbone]["model_short"]
print(
    f"Sweep '{spec.stage_name}': backbone={spec.backbone} model={model_name} "
    f"trial_epochs={spec.trial_epochs} max_trials={spec.max_trials} "
    f"schedule={spec.schedule_epochs} register_winner={spec.register_winner}"
)
print(f"  pinned       = {dict(spec.pinned)}")
print(f"  search_space = {dict(spec.search_space)}")

# COMMAND ----------
# ---- Optional: calibrate anchors once on the driver (for the "calibrated" toggle) ----
# calibrate_anchors / calibrate_aspect_ratios read the canonical COCO train split at
# {VOLUME_PATH}/annotations/train.json (written by 01) and fit anchor geometry to the
# DENTEX box distribution. Computed here so each trial just receives a plain list.
CALIBRATED_SCALES = None
CALIBRATED_RATIOS = None
if any("calibrated" in str(v) for v in spec.search_space.get("anchor_mode", [])):
    from dais26_dentex.models.detection_head import calibrate_anchors, calibrate_aspect_ratios

    _ann = f"{VOLUME_PATH}/annotations/train.json"
    CALIBRATED_SCALES = calibrate_anchors(_ann)
    CALIBRATED_RATIOS = calibrate_aspect_ratios(_ann)
    print(f"Calibrated anchors: scales={CALIBRATED_SCALES} ratios={CALIBRATED_RATIOS}")


def resolve_overrides(params: dict) -> dict:
    """Map a trial's sampled params onto TrainerConfig kwargs (`anchor_mode` ->
    concrete anchor_scales / aspect_ratios; everything else passes through)."""
    override = dict(params)
    mode = override.pop("anchor_mode", "default")
    if mode == "calibrated" and CALIBRATED_SCALES is not None:
        override["anchor_scales"] = list(CALIBRATED_SCALES)
        override["aspect_ratios"] = list(CALIBRATED_RATIOS)
    return override


# COMMAND ----------
def launch(cfg_kwargs: dict) -> str | None:
    """Lane-specific trial executor: one @distributed AIR job per config."""
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

        cfg = TrainerConfig.from_dict(cfg_kwargs)
        return Trainer(cfg).run()

    results = _run.distributed()
    return next((r for r in results if r), None)


# COMMAND ----------
# ---- Run the sweep ----
runner = SweepRunner(
    spec,
    base_config_kwargs={
        "catalog": CATALOG,
        "schema": SCHEMA,
        "backbone_revision": BACKBONE_REVISION,
        "volume_path": VOLUME_PATH,
        "cache_dir": CACHE_DIR,
        "experiment_name": EXPERIMENT_NAME,
        "model_name": model_name,
    },
    launch=launch,
    client=client,
    model_fqn=f"{CATALOG}.{SCHEMA}.{model_name}",
    resolve_overrides=resolve_overrides,
)
outcome = runner.run()

print(f"\nParent run: {outcome.parent_run_id}")
if outcome.winner is not None:
    print(f"Winner: trial {outcome.winner.trial_id} ({spec.primary_metric}={outcome.winner.metric})")
print(
    f"Best full-length retrain: run {outcome.retrain_run_id} "
    f"{spec.primary_metric}={outcome.retrain_metric} version={outcome.registered_version} "
    f"challenger_set={outcome.challenger_set}"
)

# COMMAND ----------
dbutils.jobs.taskValues.set(
    key="run_id",
    value=outcome.retrain_run_id or (outcome.winner.run_id if outcome.winner else None),
)
