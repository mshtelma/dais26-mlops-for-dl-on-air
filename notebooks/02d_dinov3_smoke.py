# Databricks notebook source
# MAGIC %md
# MAGIC # 02d — DINOv3 precision/normalization smoke test
# MAGIC
# MAGIC Cheap go/no-go gate before the full A/B. The DINOv3 A/B collapsed (0.0 mAP,
# MAGIC dead-flat loss) because the trainer ran DINOv3 under **fp16 autocast** (which
# MAGIC NaNs the RoPE/LayerScale encoder, so the GradScaler skipped every step) and
# MAGIC fed it **CLIP-normalized** inputs (DINOv3 expects ImageNet). The fix makes
# MAGIC precision + normalization backbone-aware (`amp_dtype="auto"` -> bf16 for
# MAGIC DINOv3; `BackboneInfo.image_mean/std` -> ImageNet). See docs/HPO.md.
# MAGIC
# MAGIC This runs a short (`SMOKE_EPOCHS`) DINOv3 fine-tune with the post-fix
# MAGIC `per_level` treatment recipe and asserts:
# MAGIC   1. `train/loss` actually **decreases** (loss not flat -> optimizer stepping),
# MAGIC   2. `val/mAP_50 > 0` (the model is learning something),
# MAGIC   3. `train/grad_norm` is finite and `train/amp_scale == 1.0` (bf16, no scaler).
# MAGIC
# MAGIC The `flat_loss_patience` guard aborts fast if the loss is still flat, so a
# MAGIC regressed fix fails in minutes, not a full schedule. If bf16 mAP looks weak,
# MAGIC set `AMP_DTYPE = "fp32"` and re-run to check the research caveat that DINOv3
# MAGIC bf16 can trail fp32.
# MAGIC
# MAGIC Requires a **GPU** job (8xH100) and the `dais26-secrets/hf-token` secret.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import os

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

# ---- Smoke knobs ----
SMOKE_BACKBONE = COMPARISON_BACKBONE  # "dinov3_vitl16"
SMOKE_MODEL_SHORT = _DETECTOR_NAMES_BY_BACKBONE[SMOKE_BACKBONE]["model_short"]
SMOKE_EPOCHS = 3
# bf16 ("auto") was tried first and NaN'd the DINOv3 forward at step 0 (cls+box
# loss = nan, amp_scale=1.0) — i.e. even bf16 autocast destabilizes the encoder
# here, not just fp16. fp32 (autocast disabled) is the stable escape hatch the
# research/plan calls for. See docs/HPO.md "DINOv3 A/B".
AMP_DTYPE = "fp32"
# fp32 activations are ~2x bf16 (which already used 68% of an 80GB H100 at
# batch_size=8), so halve the per-GPU batch to stay off the OOM line for the
# smoke. The full A/B picks its own batch.
SMOKE_BATCH_SIZE = 4
FLAT_LOSS_PATIENCE = 2        # abort if train/loss hasn't moved after 2 epochs

# Post-fix treatment recipe (mirrors the A/B treatment arm + the sgcli workload).
SMOKE_CFG: dict = {
    "catalog": CATALOG,
    "schema": SCHEMA,
    "backbone_name": SMOKE_BACKBONE,
    "backbone_revision": BACKBONE_REVISION,
    "volume_path": VOLUME_PATH,
    "cache_dir": CACHE_DIR,
    "epochs": SMOKE_EPOCHS,
    "batch_size": SMOKE_BATCH_SIZE,
    "backbone_mode": "full",
    "backbone_lr": 1e-5,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "onecycle_pct_start": 0.3,
    "img_size": 1024,
    "base_seed": SWEEP_SEED,
    "anchor_layout": "per_level",
    "anchor_base_scale": 4.0,
    "nms_per_class": True,
    "amp_dtype": AMP_DTYPE,
    "flat_loss_patience": FLAT_LOSS_PATIENCE,
    "experiment_name": EXPERIMENT_NAME,
    "model_name": SMOKE_MODEL_SHORT,
    "register_model": False,
    "set_candidate_alias": False,
}

print(f"smoke backbone = {SMOKE_BACKBONE} ({SMOKE_MODEL_SHORT})")
print(f"epochs={SMOKE_EPOCHS}  amp_dtype={AMP_DTYPE}  flat_loss_patience={FLAT_LOSS_PATIENCE}")


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
run_id = _run_distributed_training(SMOKE_CFG)
print(f"smoke run_id = {run_id}")


# COMMAND ----------
# MAGIC %md
# MAGIC ## Verdict — did DINOv3 actually train?

# COMMAND ----------
def _history(run_id: str, key: str) -> list[tuple[int, float]]:
    if not run_id:
        return []
    pts = client.get_metric_history(run_id, key)
    return [(m.step, m.value) for m in sorted(pts, key=lambda m: m.step)]


loss_hist = _history(run_id, "train/loss")
grad_hist = _history(run_id, "train/grad_norm")
scale_hist = _history(run_id, "train/amp_scale")
map_hist = _history(run_id, "val/mAP_50")

print("epoch | train/loss | grad_norm | amp_scale | val/mAP_50")
for i in range(len(loss_hist)):
    step, loss = loss_hist[i]
    gn = next((v for s, v in grad_hist if s == step), float("nan"))
    sc = next((v for s, v in scale_hist if s == step), float("nan"))
    mp = next((v for s, v in map_hist if s == step), float("nan"))
    print(f"{step:>5} | {loss:>10.4f} | {gn:>9.3f} | {sc:>9.1f} | {mp:>10.4f}")

# Pass criteria.
losses = [v for _, v in loss_hist]
maps = [v for _, v in map_hist]
loss_decreased = len(losses) >= 2 and min(losses[1:]) < losses[0] - 1e-3
best_map = max(maps) if maps else 0.0
map_positive = best_map > 0.0

print("\n=== SMOKE VERDICT ===")
print(f"train/loss decreased:  {'YES' if loss_decreased else 'NO'} "
      f"(first={losses[0]:.4f} best={min(losses):.4f})" if losses else "NO loss history")
print(f"val/mAP_50 > 0:        {'YES' if map_positive else 'NO'} (best={best_map:.4f})")
if loss_decreased and map_positive:
    print("PASS — DINOv3 is learning. Proceed to the full A/B (02c).")
else:
    print("FAIL — DINOv3 still not training. Inspect grad_norm/amp_scale above; "
          "try AMP_DTYPE='fp32' or re-check the normalization fix before the A/B.")
