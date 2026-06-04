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
# MAGIC 3. pick the winner by `SWEEP_PRIMARY_METRIC`; retrain it at both `TRAIN_EPOCHS`
# MAGIC    and `TRAIN_EPOCHS_LONG` with `register_model=True`, then point `@challenger`
# MAGIC    at whichever schedule scored higher — but only when it beats the prior best
# MAGIC    in the experiment (the challenger registration gate; only the winner config
# MAGIC    is registered).
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

from dais26_dentex.config.constants import ALIAS_CANDIDATE
from dais26_dentex.train.sweep import (
    TrialResult,
    beats_experiment_best,
    iter_trials,
    select_best,
)

# HF token for gated backbones (DINOv3); empty string is a no-op for C-RADIO.
try:
    hf_token = dbutils.secrets.get("dais26-secrets", "hf-token")
except Exception:
    hf_token = ""

model_name = DETECTOR_MODEL_SHORT
os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME
client = MlflowClient()

# ---- Resolve the active campaign stage (the "push to 0.60" chain) ----
# When SWEEP_STAGE names a stage in CAMPAIGN_STAGES (00_config), override the
# legacy SWEEP_* globals with that stage's backbone / pinned recipe / search
# space / schedule / register-flag. Leaving SWEEP_STAGE=None preserves the
# original post-fix-sweep behavior. See docs/HPO.md "Push to 0.60".
#
# A `sweep_stage` job parameter / notebook widget overrides the 00_config value,
# so the one campaign_sweep job (resources/jobs/campaign_sweep.yml) can launch
# any stage by name (e.g. dinov3_s1) without editing 00_config.
try:
    dbutils.widgets.text("sweep_stage", "")
    _stage_param = dbutils.widgets.get("sweep_stage").strip()
    if _stage_param:
        SWEEP_STAGE = _stage_param
except Exception:
    pass

SCHEDULE_EPOCHS: list[int] = [TRAIN_EPOCHS, TRAIN_EPOCHS_LONG]
_REGISTER_WINNER: bool = SWEEP_REGISTER_WINNER
if SWEEP_STAGE and SWEEP_STAGE not in CAMPAIGN_STAGES:
    raise ValueError(
        f"SWEEP_STAGE={SWEEP_STAGE!r} not in CAMPAIGN_STAGES "
        f"({sorted(CAMPAIGN_STAGES)}). Set it to None for the legacy sweep."
    )
_stage = CAMPAIGN_STAGES.get(SWEEP_STAGE) if SWEEP_STAGE else None
if _stage is not None:
    BACKBONE = _stage["backbone"]
    model_name = _DETECTOR_NAMES_BY_BACKBONE[BACKBONE]["model_short"]
    SWEEP_PINNED = _stage["pinned"]
    SWEEP_SEARCH_SPACE = _stage["search_space"]
    SWEEP_TRIAL_EPOCHS = _stage["trial_epochs"]
    SWEEP_MAX_TRIALS = _stage.get("max_trials", SWEEP_MAX_TRIALS)
    SCHEDULE_EPOCHS = list(_stage["schedule_epochs"])
    _REGISTER_WINNER = bool(_stage.get("register_winner", True))
    print(
        f"Campaign stage '{SWEEP_STAGE}': backbone={BACKBONE} model={model_name} "
        f"trial_epochs={SWEEP_TRIAL_EPOCHS} max_trials={SWEEP_MAX_TRIALS} "
        f"schedule={SCHEDULE_EPOCHS} register_winner={_REGISTER_WINNER}"
    )
    print(f"  pinned       = {SWEEP_PINNED}")
    print(f"  search_space = {SWEEP_SEARCH_SPACE}")
else:
    print("No SWEEP_STAGE set — using legacy SWEEP_* constants from 00_config.")

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
    """Base TrainerConfig kwargs (00_config defaults + SWEEP_PINNED) merged with a
    trial override. SWEEP_PINNED carries the knobs held fixed across the sweep
    (full fine-tune, per-level anchors, per-class NMS, etc.); the trial override
    (swept fields) wins on any collision, though by design they do not overlap."""
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
    return {**base, **SWEEP_PINNED, **override}


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
# ---- Retrain the winner; confirm the schedule (TRAIN_EPOCHS vs TRAIN_EPOCHS_LONG) ----
# The per-level anchor fix reshapes the matcher/loss landscape, so the prior ~e40
# saturation point may move. Retrain the winning config at both the short and long
# schedules and keep whichever scores higher on SWEEP_PRIMARY_METRIC. Both runs
# register a UC version (the Trainer sets @challenger to itself and logs the version
# as the `registered_version` param); after both finish we move @challenger onto the
# better run's version IF it clears the best-in-experiment gate (below). The Trainer
# tracks the best checkpoint, so a 100-epoch run
# that peaks early still registers its peak rather than an over-trained final epoch.
# Retrain when registering (legacy + finalize stages) OR when a measure-only
# stage is active (s1-s3 need the full-length metric for their gate).
_DO_RETRAIN = best is not None and (_REGISTER_WINNER or _stage is not None)
if _DO_RETRAIN:
    full_model = f"{CATALOG}.{SCHEMA}.{model_name}"

    def _retrain_winner(epochs: int, register: bool) -> tuple[str | None, float | None, str | None]:
        verb = "registering as" if register else "measuring (no register)"
        print(f"Retraining winner at {epochs} epochs, {verb} {full_model}...")
        kwargs = _train_config_kwargs(best.params, epochs=epochs, register=register)
        rid = _run_distributed_training(kwargs)
        metric = version = None
        if rid:
            run_data = client.get_run(rid).data
            metric = run_data.metrics.get(SWEEP_PRIMARY_METRIC)
            version = run_data.params.get("registered_version")
        print(f"  epochs={epochs}: run_id={rid} {SWEEP_PRIMARY_METRIC}={metric} version={version}")
        return rid, metric, version

    # Retrain the winner at every schedule in SCHEDULE_EPOCHS (the stage's
    # schedule arm, or [TRAIN_EPOCHS, TRAIN_EPOCHS_LONG] for the legacy sweep).
    # Stages s1-s3 set register_winner=False — they retrain only to MEASURE the
    # full-length metric for gating; only the finalize stage (s4) registers.
    schedule_runs = [_retrain_winner(ep, _REGISTER_WINNER) for ep in SCHEDULE_EPOCHS]
    # Pick the better schedule by the primary metric (None-safe: trained-with-metric
    # beats no-metric, then higher metric wins).
    winner_rid, winner_metric, winner_version = max(
        schedule_runs, key=lambda rmv: (rmv[1] is not None, rmv[1] or -1.0)
    )
    print(f"\nBest full-length retrain: run {winner_rid} {SWEEP_PRIMARY_METRIC}={winner_metric}")
    if _REGISTER_WINNER and winner_version is not None:
        # ---- Challenger registration gate (best-in-experiment) ----
        # The retrains already registered a UC version AND set @challenger onto the
        # last-trained schedule. Only KEEP @challenger on the new winner when its
        # validation SWEEP_PRIMARY_METRIC strictly beats every PRIOR registered
        # version's (the experiment bar); otherwise restore @challenger to the prior
        # best version so a regression never becomes the challenger that triggers the
        # deployment job. Comparison is the pure `beats_experiment_best` (unit-tested).
        prior: list[tuple[str, float]] = []
        for mv in client.search_model_versions(f"name='{full_model}'"):
            if str(mv.version) == str(winner_version) or not mv.run_id:
                continue
            try:
                m = client.get_run(mv.run_id).data.metrics.get(SWEEP_PRIMARY_METRIC)
            except Exception:
                m = None
            if m is not None:
                prior.append((str(mv.version), float(m)))

        if beats_experiment_best(winner_metric, [m for _, m in prior], higher_is_better=True):
            client.set_registered_model_alias(full_model, ALIAS_CANDIDATE, winner_version)
            bar = max((m for _, m in prior), default=None)
            print(
                f"@{ALIAS_CANDIDATE} -> version {winner_version} (run {winner_rid}); "
                f"{SWEEP_PRIMARY_METRIC}={winner_metric} beats prior best {bar}."
            )
        elif prior:
            best_prior_version = max(prior, key=lambda vm: vm[1])[0]
            client.set_registered_model_alias(full_model, ALIAS_CANDIDATE, best_prior_version)
            print(
                f"Gate NOT passed: winner {SWEEP_PRIMARY_METRIC}={winner_metric} does not beat "
                f"prior best {max(m for _, m in prior)}; restored @{ALIAS_CANDIDATE} -> "
                f"version {best_prior_version}. New version {winner_version} registered but not challenger."
            )
        else:
            client.set_registered_model_alias(full_model, ALIAS_CANDIDATE, winner_version)
            print(f"@{ALIAS_CANDIDATE} -> version {winner_version} (run {winner_rid}); first measurable version.")
    elif _REGISTER_WINNER:
        print(f"No registered_version on the winning run ({winner_rid}); alias left as-is.")
    else:
        print(f"Stage register_winner=False — measured full-length metric only; @{ALIAS_CANDIDATE} unchanged.")
    dbutils.jobs.taskValues.set(key="run_id", value=winner_rid)
else:
    dbutils.jobs.taskValues.set(key="run_id", value=(best.run_id if best else None))
