# Databricks notebook source
# MAGIC %md
# MAGIC # 02c — DINOv3 anchor-fix A/B (controlled)
# MAGIC
# MAGIC Measures whether the proposed detection-head changes (per-level anchors +
# MAGIC per-class NMS + encode/decode clamp symmetry) actually lift mAP **on the
# MAGIC DINOv3 backbone specifically**. Every tuned result in `docs/HPO.md` (incl.
# MAGIC the 0.335 best) is on C-RADIOv4; DINOv3 only appears once, in the broken
# MAGIC phase (0.027). This notebook turns "directionally expected" into a measured
# MAGIC number.
# MAGIC
# MAGIC It is a **controlled A/B**, not a sweep: two runs identical except the change
# MAGIC bundle, same `base_seed`, `register_model=False`, so the `val/best_mAP_50`
# MAGIC delta is attributable to the bundle.
# MAGIC
# MAGIC * **Arm A (baseline):** `anchor_layout=absolute`, class-agnostic NMS — the
# MAGIC   legacy geometry.
# MAGIC * **Arm B (treatment):** `anchor_layout=per_level` (`base_scale=4.0`),
# MAGIC   per-class `batched_nms`.
# MAGIC
# MAGIC Pinned recipe (settled on the C-RADIO sweep): `backbone_mode=full`,
# MAGIC `backbone_lr=1e-5`, `lr=1e-4`, `weight_decay=1e-2`, `onecycle_pct_start=0.3`,
# MAGIC `img_size=1024`.
# MAGIC
# MAGIC The backbone is pinned to `COMPARISON_BACKBONE` (`dinov3_vitl16`) **locally**
# MAGIC — this notebook does NOT flip the global `BACKBONE` in `00_config.py`.
# MAGIC
# MAGIC Requires a **GPU** notebook (the Phase-1 probe loads DINOv3 onto CUDA) and an
# MAGIC 8h job timeout (each arm is a full 8xH100 fine-tune; see `resources/jobs`).

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import os

import mlflow
import torch
from mlflow.tracking import MlflowClient

# HF token for the gated DINOv3 backbone (same secret the sweep / sgcli use).
try:
    hf_token = dbutils.secrets.get("dais26-secrets", "hf-token")
except Exception:
    hf_token = ""
    print("WARNING: dais26-secrets/hf-token not found — gated DINOv3 download will fail.")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME
client = MlflowClient()

# A/B targets the DINOv3 backbone regardless of the global BACKBONE literal.
AB_BACKBONE = COMPARISON_BACKBONE  # "dinov3_vitl16"
AB_MODEL_SHORT = _DETECTOR_NAMES_BY_BACKBONE[AB_BACKBONE]["model_short"]  # "dinov3_detector"
AB_EPOCHS = TRAIN_EPOCHS  # 50; full-length so the A/B reflects the shippable recipe
AB_SEED = SWEEP_SEED  # 42; identical across both arms

# Settled recipe (held fixed across both arms); only the change bundle differs.
SHARED_RECIPE: dict = {
    "backbone_mode": "full",
    "backbone_lr": 1e-5,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "onecycle_pct_start": 0.3,
    "img_size": 1024,
    "base_seed": AB_SEED,
    # Precision is backbone-aware: "auto" gives DINOv3 fp32. Both fp16 AND bf16
    # NaN this DINOv3 detector stack (fp16 -> flat-loss collapse via GradScaler
    # skips; bf16 -> nan forward at step 0). Only fp32 trains (smoke: loss
    # 1.39->0.77, val mAP@50 0->0.22 in 3 ep; see docs/HPO.md "DINOv3 A/B").
    "amp_dtype": "auto",
    # fp32 activations are ~2x bf16 (which already sat at 68% of an 80GB H100 at
    # batch 8), so halve the per-GPU batch to stay off the OOM line. The
    # baseline/treatment arms share this, so the A/B stays controlled.
    "batch_size": 4,
    # The flat-loss guard aborts a dead arm after 8 epochs instead of burning 50.
    "flat_loss_patience": 8,
}

# The two arms. Only these keys differ — everything else comes from SHARED_RECIPE.
ARMS: dict[str, dict] = {
    "absolute_baseline": {
        "anchor_layout": "absolute",
        "nms_per_class": False,
    },
    "per_level_treatment": {
        "anchor_layout": "per_level",
        "anchor_base_scale": 4.0,
        "nms_per_class": True,
    },
}

# Acceptance bar (docs/BENCHMARKS.md). Treatment MUST beat the baseline arm; the
# must-ship target is the absolute bar.
MUST_SHIP_MAP50 = 0.45

print(f"A/B backbone   = {AB_BACKBONE} ({AB_MODEL_SHORT})")
print(f"A/B epochs     = {AB_EPOCHS}  seed = {AB_SEED}")
print(f"arms           = {list(ARMS)}")


# COMMAND ----------
def _arm_config_kwargs(arm: dict, *, epochs: int, register: bool) -> dict:
    """TrainerConfig kwargs for one arm: 00_config UC objects + SHARED_RECIPE + arm."""
    base = dict(
        catalog=CATALOG,
        schema=SCHEMA,
        backbone_name=AB_BACKBONE,
        backbone_revision=BACKBONE_REVISION,
        volume_path=VOLUME_PATH,
        cache_dir=CACHE_DIR,
        epochs=epochs,
        batch_size=TRAIN_BATCH_SIZE,
        experiment_name=EXPERIMENT_NAME,
        model_name=AB_MODEL_SHORT,
        register_model=register,
        set_candidate_alias=register,
    )
    return {**base, **SHARED_RECIPE, **arm}


# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 1 — real-DINOv3 arch probe (compatibility + mechanism)
# MAGIC
# MAGIC Builds each arm's detector on the **real DINOv3 encoder** and runs one
# MAGIC forward + anchor-match pass over a real DENTEX val batch. This proves the
# MAGIC anchor changes are geometrically compatible with DINOv3 (64x64 token grid at
# MAGIC 1024px/patch16) and shows the mechanism — positives should move off P3 onto
# MAGIC P4 in the treatment arm — *before* burning GPU hours on the full fine-tunes.

# COMMAND ----------
from dais26_dentex.config.trainer_config import TrainerConfig
from dais26_dentex.data.dataset import DENTEXDetectionDataset, detection_collate
from dais26_dentex.data.transforms import get_val_transforms
from dais26_dentex.models.arch_probe import probe_detection_model, render_report
from dais26_dentex.models.builder import build_detector, resolve_num_classes

PROBE_IMG_SIZE = SHARED_RECIPE["img_size"]
PROBE_BATCH = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"probe device = {device}")

# One real val batch, shared across both arms (same boxes → comparable positives).
_ds = DENTEXDetectionDataset(
    volume_path=VOLUME_PATH, split="val", transforms=get_val_transforms(PROBE_IMG_SIZE)
)
_loader = torch.utils.data.DataLoader(
    _ds, batch_size=PROBE_BATCH, shuffle=False, collate_fn=detection_collate
)
_probe_images, _probe_targets = next(iter(_loader))
_probe_images = _probe_images.to(device)
print(f"probe batch: images={tuple(_probe_images.shape)} "
      f"gts={[int(t['labels'].numel()) for t in _probe_targets]}")

probe_reports: dict[str, dict] = {}
for arm_name, arm in ARMS.items():
    cfg = TrainerConfig.from_dict(_arm_config_kwargs(arm, epochs=AB_EPOCHS, register=False))
    model, _info = build_detector(cfg, device=device)
    model.eval()
    num_classes = resolve_num_classes(cfg)
    report = probe_detection_model(model, _probe_images, _probe_targets, num_classes=num_classes)
    probe_reports[arm_name] = report
    print(f"\n=== Phase-1 probe — {arm_name} (layout={arm['anchor_layout']}) ===")
    print(render_report(report))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n--- Phase-1 summary (real DINOv3 backbone) ---")
for arm_name, rep in probe_reports.items():
    anchors = rep["anchors"]  # nested anchor report (anchors_per_cell/total_anchors/...)
    print(
        f"{arm_name:>20}: anchors/cell={anchors['anchors_per_cell']} "
        f"total_anchors={anchors['total_anchors']} "
        f"all_scales_every_level={anchors['all_scales_every_level']} "
        f"positives_per_level={rep['positives_per_level']} "
        f"nms={rep['nms_mode']}"
    )


# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 2 — controlled A/B training
# MAGIC
# MAGIC Each arm is one `@distributed` 8xH100 AIR run, nested under a single A/B
# MAGIC parent run, `register_model=False`. Both log `val/best_mAP_50` via the same
# MAGIC `Trainer` eval code, so the delta is attributable to the change bundle.

# COMMAND ----------
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

        from dais26_dentex.config.trainer_config import TrainerConfig as _TC
        from dais26_dentex.train.trainer import Trainer

        cfg = _TC(**cfg_kwargs)
        return Trainer(cfg).run()

    results = _run.distributed()
    return next((r for r in results if r), None)


# COMMAND ----------
# Create the parent run and log its params BEFORE launching any arm, then close
# it. The @distributed worker calls mlflow.start_run() on the GPU side, which
# ends the driver's fluent run almost immediately — so logging the verdict to
# the *fluent* parent after the arms run is lost. Instead we capture
# parent_run_id and write the verdict via MlflowClient (which logs to a run by
# id regardless of whether it's the active run).
with mlflow.start_run(run_name=f"anchor-ab-{AB_BACKBONE}") as parent:
    parent_run_id = parent.info.run_id
    mlflow.log_params(
        {
            "ab_backbone": AB_BACKBONE,
            "ab_epochs": AB_EPOCHS,
            "ab_seed": AB_SEED,
            "ab_must_ship_mAP_50": MUST_SHIP_MAP50,
            "ab_amp_dtype": SHARED_RECIPE["amp_dtype"],
        }
    )

arm_results: dict[str, dict] = {}
for arm_name, arm in ARMS.items():
    print(f"\n=== Arm '{arm_name}': {arm} ===")
    cfg_kwargs = _arm_config_kwargs(arm, epochs=AB_EPOCHS, register=False)
    run_id = _run_distributed_training(cfg_kwargs)

    metric = None
    if run_id:
        client.set_tag(run_id, "mlflow.parentRunId", parent_run_id)
        client.set_tag(run_id, "anchor_ab_arm", arm_name)
        metric = client.get_run(run_id).data.metrics.get("val/best_mAP_50")
    arm_results[arm_name] = {"run_id": run_id, "val/best_mAP_50": metric}
    print(f"Arm '{arm_name}': run_id={run_id} val/best_mAP_50={metric}")

# ---- Decision ---- (logged to the parent via client, not the closed fluent run)
base = arm_results["absolute_baseline"]["val/best_mAP_50"]
treat = arm_results["per_level_treatment"]["val/best_mAP_50"]
delta = None
if base is not None and treat is not None:
    delta = treat - base
    client.log_metric(parent_run_id, "ab_baseline_mAP_50", base)
    client.log_metric(parent_run_id, "ab_treatment_mAP_50", treat)
    client.log_metric(parent_run_id, "ab_delta_mAP_50", delta)
print("\n=== A/B result ===")
print(f"  baseline  (absolute):  {base}")
print(f"  treatment (per_level): {treat}")
print(f"  delta:                 {delta}")


# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 3 — verdict
# MAGIC
# MAGIC Treatment is an uplift iff it beats the baseline arm; the must-ship bar
# MAGIC (`docs/BENCHMARKS.md`) is `mAP@50 >= 0.45`. For the per-class breakdown
# MAGIC (`Caries AP@50 >= 0.30`) and the apples-to-apples re-eval through the serving
# MAGIC pyfunc, register the treatment arm (next cell) and run
# MAGIC `notebooks/09_eval_comparison.py`.

# COMMAND ----------
if delta is None:
    print("INCONCLUSIVE — one or both arms produced no metric (check the run logs).")
else:
    uplift = delta > 0
    ship = treat is not None and treat >= MUST_SHIP_MAP50
    print(f"Uplift vs baseline:  {'YES' if uplift else 'NO'} ({delta:+.4f} mAP@50)")
    print(f"Clears must-ship bar ({MUST_SHIP_MAP50}): {'YES' if ship else 'NO'}")
    if uplift and ship:
        print("VERDICT: per-level anchors + per-class NMS deliver the expected uplift on DINOv3.")
    elif uplift:
        print("VERDICT: uplift confirmed but below the must-ship bar — tune via the 02b sweep.")
    else:
        print("VERDICT: no uplift on DINOv3 — investigate (the C-RADIO win may not transfer).")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optional — register the treatment arm as `@candidate`
# MAGIC
# MAGIC Set `REGISTER_TREATMENT = True` to retrain the treatment arm with
# MAGIC `register_model=True`, registering it as `dinov3_detector@candidate` so
# MAGIC `09_eval_comparison.py` can re-evaluate it through the serving pyfunc.

# COMMAND ----------
REGISTER_TREATMENT = False

if REGISTER_TREATMENT:
    print(f"Retraining treatment arm and registering as {AB_MODEL_SHORT}@candidate...")
    winner_kwargs = _arm_config_kwargs(ARMS["per_level_treatment"], epochs=AB_EPOCHS, register=True)
    winner_run_id = _run_distributed_training(winner_kwargs)
    print(f"Registered. run_id={winner_run_id}")
    dbutils.jobs.taskValues.set(key="run_id", value=winner_run_id)
