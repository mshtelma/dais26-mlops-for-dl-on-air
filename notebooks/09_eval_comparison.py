# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Eval comparison (C-RADIOv4 vs DINOv3), apples-to-apples
# MAGIC
# MAGIC Rather than ranking backbones by the `val/mAP_*` metrics the trainer logs,
# MAGIC this notebook does the stronger thing: it **re-evaluates each
# MAGIC registered detector from scratch on the same held-out split**, through the
# MAGIC real serving pyfunc (`DetectorPyfunc`), and scores it with the same
# MAGIC `eval.coco_metrics.evaluate_coco` the trainer uses. That makes the
# MAGIC comparison independent of what (if anything) got logged at train time and
# MAGIC guarantees both backbones are measured on identical data with identical
# MAGIC eval code + box post-processing.
# MAGIC
# MAGIC Why the serving pyfunc and not the raw `DetectionModel`: the pyfunc resizes
# MAGIC the input to `input_size` and **rescales predicted boxes back to original
# MAGIC image pixels**, which is the coordinate frame the COCO ground-truth JSON is
# MAGIC in. Evaluating the bare model at 1024px would mismatch the GT frame.
# MAGIC
# MAGIC Requires a **GPU** notebook (the ViT backbones load onto CUDA). Models are
# MAGIC loaded and freed one at a time so two large backbones never sit in VRAM
# MAGIC together.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import gc

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch

from dais26_dentex.eval.runner import build_name_to_category_id, score_model_on_split

mlflow.set_registry_uri("databricks-uc")

# Splits to evaluate on. `val` (50 imgs) is the selection surface the trainer
# validated on; `test` (250 imgs) is the larger held-out generalization /
# publication surface. Both are held out of training. The shared eval path lives
# in `dais26_dentex.eval.runner` (also used by the deployment-job eval task,
# notebooks/10) so this comparison and the promotion gate score identically.
EVAL_SPLITS = ["val", "test"]

# Per-image prediction chunk — the pyfunc forwards one image at a time
# internally, so this only bounds how many rows we build into a DataFrame at
# once (keeps memory flat + gives progress output).
PREDICT_CHUNK = 16

# Alias to evaluate, in preference order. The deployment job promotes the dev
# @challenger -> prod @champion after eval + approval; fall back to @challenger
# so a backbone that's trained but not yet promoted still participates.
ALIAS_PREFERENCE = ("champion", "challenger")

# Backbones to compare. Keys are the `params.backbone_name` literal; `model_short`
# mirrors the per-backbone registered-model name derived in 00_config.py.
COMPARE_BACKBONES: dict[str, dict[str, str]] = {
    "cradio_v4_so400m": {"model_short": "cradio_detector"},
    "dinov3_vitl16": {"model_short": "dinov3_detector"},
}

# Inverse of the canonical label map: predicted class *names* (what the pyfunc
# returns) -> integer category_id (what COCO scoring expects).
NAME_TO_ID = build_name_to_category_id()

print(f"EVAL_SPLITS = {EVAL_SPLITS}")
print(f"Catalog/schema = {CATALOG}.{SCHEMA}")

# COMMAND ----------
# ---- Per-backbone, per-split evaluation ----


def _load_detector(model_short: str):
    """Load a registered detector pyfunc by alias preference, or (None, None)."""
    full = f"{CATALOG}.{SCHEMA}.{model_short}"
    for alias in ALIAS_PREFERENCE:
        uri = f"models:/{full}@{alias}"
        try:
            model = mlflow.pyfunc.load_model(uri)
            return model, uri
        except Exception as e:
            print(f"  {uri}: unavailable ({type(e).__name__})")
    return None, None


# results[split][backbone] = metrics dict from score_model_on_split.
results: dict[str, dict[str, dict]] = {split: {} for split in EVAL_SPLITS}
for backbone, cfg in COMPARE_BACKBONES.items():
    print(f"\n=== {backbone} ({cfg['model_short']}) ===")
    model, uri = _load_detector(cfg["model_short"])
    if model is None:
        print(f"  no registered model for {backbone}; skipping")
        continue
    print(f"  loaded {uri}")

    for split in EVAL_SPLITS:
        print(f"  scoring on '{split}'...")
        metrics = score_model_on_split(
            model, VOLUME_PATH, split, name_to_id=NAME_TO_ID, predict_chunk=PREDICT_CHUNK
        )
        metrics["_uri"] = uri
        results[split][backbone] = metrics

    # Free VRAM before loading the next backbone.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if not any(results[split] for split in EVAL_SPLITS):
    raise RuntimeError(
        "No detectors could be loaded. Train + register at least one backbone "
        "(02_train_detector_air.py / the sgcli workloads) first."
    )

# COMMAND ----------
# ---- Side-by-side comparison tables (one per split) ----
OVERALL_COLS = ["mAP_50", "mAP_50_95", "mAP_75", "AR_1", "AR_10", "AR_100"]

for split in EVAL_SPLITS:
    split_results = results[split]
    if not split_results:
        continue
    comparison_df = pd.DataFrame(
        {
            backbone: {
                **{c: round(m[c], 4) for c in OVERALL_COLS},
                "num_predictions": m["num_predictions"],
            }
            for backbone, m in split_results.items()
        }
    ).T
    comparison_df.index.name = "backbone"
    print(f"\n=== Eval comparison on '{split}' split ===")
    display(comparison_df)

    per_class_df = pd.DataFrame(
        {backbone: m["per_class_AP50"] for backbone, m in split_results.items()}
    ).T.round(4)
    per_class_df.index.name = "backbone"
    print(f"=== Per-class AP@50 ('{split}') ===")
    display(per_class_df)

# COMMAND ----------
# ---- Bar chart of mAP_50 per split + winner ----
for split in EVAL_SPLITS:
    split_results = results[split]
    if not split_results:
        continue
    chart = pd.Series({b: float(m["mAP_50"]) for b, m in split_results.items()})
    fig, ax = plt.subplots(figsize=(6, 4))
    chart.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"][: len(chart)])
    ax.set_ylabel("mAP_50")
    ax.set_title(f"Detector mAP_50 by backbone (re-eval on '{split}')")
    ax.set_ylim(0, max(0.01, chart.max() * 1.15))
    for i, v in enumerate(chart.values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    winner = chart.idxmax()
    print(f"\nWINNER (highest mAP_50 on '{split}'): {winner} = {chart.max():.4f}")
    if len(chart) > 1:
        runner_up = chart.drop(winner).idxmax()
        delta = chart[winner] - chart[runner_up]
        print(f"  beats {runner_up} ({chart[runner_up]:.4f}) by {delta:+.4f} mAP_50")
