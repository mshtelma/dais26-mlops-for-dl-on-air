# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Eval comparison (C-RADIOv4 vs DINOv3), apples-to-apples
# MAGIC
# MAGIC `08_backbone_comparison.py` ranks backbones by the `val/mAP_*` metrics the
# MAGIC trainer logs. This notebook does the stronger thing: it **re-evaluates each
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
import base64
import gc
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch

from dais26_dentex.data.dentex_loader import get_label_map, load_canonical_split
from dais26_dentex.eval.coco_metrics import evaluate_coco, format_predictions_for_coco

mlflow.set_registry_uri("databricks-uc")

# Split to evaluate on. `val` mirrors what the trainer validated on (50 imgs);
# switch to `test` (250 imgs) for the larger held-out generalization measure.
# Both are held out of training, so either is a fair comparison surface.
EVAL_SPLIT = "val"

# Per-image prediction chunk — the pyfunc forwards one image at a time
# internally, so this only bounds how many rows we build into a DataFrame at
# once (keeps memory flat + gives progress output).
PREDICT_CHUNK = 16

# Alias to evaluate, in preference order. The deploy task promotes @candidate ->
# @champion after a passing smoke test; fall back to @candidate so a backbone
# that's trained but not yet deployed still participates.
ALIAS_PREFERENCE = ("champion", "candidate")

# Backbones to compare. Keys are the `params.backbone_name` literal; `model_short`
# mirrors the per-backbone registered-model name derived in 00_config.py.
COMPARE_BACKBONES: dict[str, dict[str, str]] = {
    "cradio_v4_so400m": {"model_short": "cradio_detector"},
    "dinov3_vitl16": {"model_short": "dinov3_detector"},
}

# Inverse of the canonical label map: predicted class *names* (what the pyfunc
# returns) -> integer category_id (what COCO scoring expects).
LABEL_MAP = get_label_map()  # {0: "Caries", ...}
NAME_TO_ID = {v: k for k, v in LABEL_MAP.items()}

print(f"EVAL_SPLIT = {EVAL_SPLIT}")
print(f"Catalog/schema = {CATALOG}.{SCHEMA}")

# COMMAND ----------
# ---- Materialize a normalized COCO ground-truth file for pycocotools ----
# `evaluate_coco` reads the GT path directly via pycocotools.COCO, so we write a
# fresh JSON from `load_canonical_split` (which normalizes DENTEX's hierarchical
# category_id_3 -> our flat category_id in memory). We also backfill `area` /
# `iscrowd` if missing, since COCOeval's area-range buckets need them.
coco_gt = load_canonical_split(VOLUME_PATH, EVAL_SPLIT)
for ann in coco_gt["annotations"]:
    if "area" not in ann:
        x, y, w, h = ann["bbox"]
        ann["area"] = float(w) * float(h)
    ann.setdefault("iscrowd", 0)

_gt_tmp = tempfile.NamedTemporaryFile("w", suffix=f"_{EVAL_SPLIT}_gt.json", delete=False)
json.dump(coco_gt, _gt_tmp)
_gt_tmp.close()
GT_PATH = _gt_tmp.name

images_dir = Path(VOLUME_PATH) / "images" / EVAL_SPLIT
print(
    f"GT: {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations "
    f"-> {GT_PATH}"
)
print(f"Images dir: {images_dir}")

# COMMAND ----------
# ---- Per-backbone evaluation ----


def _load_detector(model_short: str):
    """Load a registered detector pyfunc by alias preference, or (None, None)."""
    full = f"{CATALOG}.{SCHEMA}.{model_short}"
    for alias in ALIAS_PREFERENCE:
        uri = f"models:/{full}@{alias}"
        try:
            model = mlflow.pyfunc.load_model(uri)
            return model, uri
        except Exception as e:  # noqa: BLE001 — any resolve/load failure -> try next alias
            print(f"  {uri}: unavailable ({type(e).__name__})")
    return None, None


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _predict_split(model) -> list[dict]:
    """Run the pyfunc over every image in the split; return COCO model_output.

    Each element: {image_id, boxes (N,4 xyxy px), scores (N,), labels (N, int)}.
    Predicted label *names* are mapped back to integer category_ids.
    """
    items = [(img["id"], images_dir / img["file_name"]) for img in coco_gt["images"]]
    model_output: list[dict] = []
    for start in range(0, len(items), PREDICT_CHUNK):
        chunk = items[start : start + PREDICT_CHUNK]
        df_in = pd.DataFrame({"image": [_b64(p) for _, p in chunk]})
        preds = model.predict(df_in).reset_index(drop=True)
        for (image_id, _), (_, row) in zip(chunk, preds.iterrows(), strict=True):
            labels = [NAME_TO_ID.get(str(name), int(name)) for name in row["labels"]]
            model_output.append(
                {
                    "image_id": int(image_id),
                    "boxes": torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4),
                    "scores": torch.tensor(row["scores"], dtype=torch.float32).reshape(-1),
                    "labels": torch.tensor(labels, dtype=torch.long).reshape(-1),
                }
            )
        print(f"    predicted {min(start + PREDICT_CHUNK, len(items))}/{len(items)}")
    return model_output


results: dict[str, dict] = {}
for backbone, cfg in COMPARE_BACKBONES.items():
    print(f"\n=== {backbone} ({cfg['model_short']}) ===")
    model, uri = _load_detector(cfg["model_short"])
    if model is None:
        print(f"  no registered model for {backbone}; skipping")
        continue
    print(f"  loaded {uri}")

    model_output = _predict_split(model)
    coco_preds = format_predictions_for_coco(model_output)
    metrics = evaluate_coco(coco_preds, GT_PATH)
    metrics["_uri"] = uri
    metrics["_num_predictions"] = len(coco_preds)
    results[backbone] = metrics

    # Free VRAM before loading the next backbone.
    del model, model_output, coco_preds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if not results:
    raise RuntimeError(
        "No detectors could be loaded. Train + register at least one backbone "
        "(02_train_detector_air.py / the sgcli workloads) first."
    )

# COMMAND ----------
# ---- Side-by-side comparison table ----
OVERALL_COLS = ["mAP_50", "mAP_50_95", "mAP_75", "AR_1", "AR_10", "AR_100"]

comparison_df = pd.DataFrame(
    {
        backbone: {
            **{c: round(m[c], 4) for c in OVERALL_COLS},
            "num_predictions": m["_num_predictions"],
        }
        for backbone, m in results.items()
    }
).T
comparison_df.index.name = "backbone"
print(f"\n=== Eval comparison on '{EVAL_SPLIT}' split ({len(coco_gt['images'])} images) ===")
display(comparison_df)  # noqa: F821  (Databricks builtin)

# COMMAND ----------
# ---- Per-class AP50 breakdown ----
per_class_df = pd.DataFrame(
    {backbone: m["per_class_AP50"] for backbone, m in results.items()}
).T.round(4)
per_class_df.index.name = "backbone"
print("\n=== Per-class AP@50 ===")
display(per_class_df)  # noqa: F821  (Databricks builtin)

# COMMAND ----------
# ---- Bar chart of mAP_50 + winner ----
chart = comparison_df["mAP_50"].astype(float)
fig, ax = plt.subplots(figsize=(6, 4))
chart.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"][: len(chart)])
ax.set_ylabel("mAP_50")
ax.set_title(f"Detector mAP_50 by backbone (re-eval on '{EVAL_SPLIT}')")
ax.set_ylim(0, max(0.01, chart.max() * 1.15))
for i, v in enumerate(chart.values):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

winner = chart.idxmax()
print(f"\nWINNER (highest mAP_50 on '{EVAL_SPLIT}'): {winner} = {chart.max():.4f}")
if len(chart) > 1:
    runner_up = chart.drop(winner).idxmax()
    delta = chart[winner] - chart[runner_up]
    print(f"  beats {runner_up} ({chart[runner_up]:.4f}) by {delta:+.4f} mAP_50")

# COMMAND ----------
# Clean up the temp GT file.
Path(GT_PATH).unlink(missing_ok=True)
