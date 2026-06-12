# Databricks notebook source
# MAGIC %md
# MAGIC # 09b — Free eval-threshold grid (no retraining)
# MAGIC
# MAGIC The decode/NMS thresholds (`score_threshold`, `nms_iou_threshold`,
# MAGIC `max_detections`) are **inference-time** knobs the `DetectionModel` reads as
# MAGIC plain attributes at forward time, so we can sweep them on an **already-trained,
# MAGIC registered** detector without spending a GPU-hour on retraining — the cheapest
# MAGIC lever in the "push to 0.60" plan (docs/HPO.md). Run it first to bank any free
# MAGIC mAP and record the per-class Caries AP@50 the campaign gates on.
# MAGIC
# MAGIC For each registered backbone (`@candidate` preferred, `@champion` fallback) it
# MAGIC loads the serving pyfunc once, then per grid point mutates the inner
# MAGIC `DetectionModel`'s thresholds and re-scores the held-out split through the SAME
# MAGIC `eval.runner.score_model_on_split` the comparison + promotion gate use (09 / 10),
# MAGIC then prints the best (score, nms_iou, max_det) by mAP@50 plus the Caries AP@50
# MAGIC there and at the as-registered defaults.
# MAGIC
# MAGIC Requires a **GPU** notebook. The winning thresholds fold into the finalize-stage
# MAGIC `TrainerConfig` (now config fields) so the re-registered model reproduces them.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import gc
import json
import tempfile
from itertools import product

import mlflow
import pandas as pd
import torch

from dais26_dentex.data.dentex_loader import get_label_map
from dais26_dentex.eval.runner import (
    build_name_to_category_id,
    inner_detection_model,
    load_detector_by_alias,
    score_model_on_split,
)

mlflow.set_registry_uri("databricks-uc")

EVAL_SPLIT = "val"  # held out of training; mirrors trainer validation
PREDICT_CHUNK = 16
# Grid the CANDIDATE first: freshly-registered tuning winners live at @candidate;
# @champion is the post-tuning end-state (often stale). Champion is only a fallback.
ALIAS_PREFERENCE = ("candidate", "champion")

# Backbones to grid. Mirrors 09_eval_comparison.py.
COMPARE_BACKBONES: dict[str, dict[str, str]] = {
    "cradio_v4_so400m": {"model_short": "cradio_detector"},
    "dinov3_vitl16": {"model_short": "dinov3_detector"},
}

# The free grid. Smaller score_threshold recovers low-confidence true positives
# (helps AR / small-lesion recall); nms_iou_threshold trades duplicate suppression
# vs recall of overlapping lesions; max_detections rarely binds but is cheap to check.
SCORE_THRESHOLDS = [0.01, 0.05, 0.10]
NMS_IOU_THRESHOLDS = [0.40, 0.50, 0.60]
MAX_DETECTIONS = [100, 300]

NAME_TO_ID = build_name_to_category_id()  # predicted class name -> COCO category_id
CARIES_NAME = get_label_map().get(0, "Caries")

print(f"EVAL_SPLIT = {EVAL_SPLIT}")
print(f"Catalog/schema = {CATALOG}.{SCHEMA}")
print(f"Grid: {len(SCORE_THRESHOLDS)}x{len(NMS_IOU_THRESHOLDS)}x{len(MAX_DETECTIONS)} = "
      f"{len(SCORE_THRESHOLDS) * len(NMS_IOU_THRESHOLDS) * len(MAX_DETECTIONS)} combos/backbone")


def _score_grid_point(loaded) -> dict:
    """Score the loaded detector on EVAL_SPLIT at its CURRENT thresholds, via the
    shared eval path. (Re-materializes GT + re-encodes per call — negligible next
    to the GPU forward on the 50-image val split.)"""
    m = score_model_on_split(
        loaded, VOLUME_PATH, EVAL_SPLIT,
        name_to_id=NAME_TO_ID, predict_chunk=PREDICT_CHUNK, verbose=False,
    )
    return {
        "mAP_50": float(m["mAP_50"]),
        "caries_AP50": float(m.get("per_class_AP50", {}).get(CARIES_NAME, float("nan"))),
        "n_pred": int(m["num_predictions"]),
    }


# COMMAND ----------
# ---- Grid each backbone ----
grid_rows: list[dict] = []
best_per_backbone: dict[str, dict] = {}

for backbone, cfg in COMPARE_BACKBONES.items():
    print(f"\n=== {backbone} ({cfg['model_short']}) ===")
    loaded, uri = load_detector_by_alias(f"{CATALOG}.{SCHEMA}.{cfg['model_short']}", ALIAS_PREFERENCE)
    if loaded is None:
        print(f"  no registered model for {backbone}; skipping")
        continue
    dm = inner_detection_model(loaded)
    base = (float(dm.score_threshold), float(dm.nms_iou_threshold), int(dm.max_detections))
    print(f"  loaded {uri}; as-registered thresholds score={base[0]} nms={base[1]} max_det={base[2]}")

    rows: list[dict] = []
    for s, n, d in product(SCORE_THRESHOLDS, NMS_IOU_THRESHOLDS, MAX_DETECTIONS):
        dm.score_threshold = s
        dm.nms_iou_threshold = n
        dm.max_detections = d
        res = _score_grid_point(loaded)
        row = {"backbone": backbone, "score": s, "nms_iou": n, "max_det": d, **res}
        rows.append(row)
        grid_rows.append(row)
        print(f"    score={s:<4} nms={n:<4} max_det={d:<4} "
              f"mAP50={res['mAP_50']:.4f} caries_AP50={res['caries_AP50']:.4f}")

    rows_df = pd.DataFrame(rows).sort_values("mAP_50", ascending=False).reset_index(drop=True)
    best = rows_df.iloc[0].to_dict()
    dflt = next((r for r in rows if (r["score"], r["nms_iou"], r["max_det"]) == base), None)
    best_per_backbone[backbone] = {"best": best, "default": dflt, "uri": uri}
    print(f"  BEST mAP50={best['mAP_50']:.4f} @ score={best['score']} nms={best['nms_iou']} "
          f"max_det={best['max_det']} (caries_AP50={best['caries_AP50']:.4f})")
    if dflt is not None:
        print(f"  default mAP50={dflt['mAP_50']:.4f} -> free gain {best['mAP_50'] - dflt['mAP_50']:+.4f}")

    del loaded, dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if not grid_rows:
    raise RuntimeError("No detectors could be loaded. Train + register at least one backbone first.")

# COMMAND ----------
# ---- Summary tables ----
grid_df = pd.DataFrame(grid_rows)
print(f"\n=== Full threshold grid on '{EVAL_SPLIT}' ===")
display(grid_df.sort_values(["backbone", "mAP_50"], ascending=[True, False]))

summary = []
for backbone, info in best_per_backbone.items():
    b, d = info["best"], info["default"]
    summary.append(
        {
            "backbone": backbone,
            "best_mAP50": round(b["mAP_50"], 4),
            "best_caries_AP50": round(b["caries_AP50"], 4),
            "best_score": b["score"],
            "best_nms_iou": b["nms_iou"],
            "best_max_det": b["max_det"],
            "default_mAP50": round(d["mAP_50"], 4) if d else None,
            "free_gain": round(b["mAP_50"] - d["mAP_50"], 4) if d else None,
        }
    )
summary_df = pd.DataFrame(summary).set_index("backbone")
print("\n=== Best free thresholds per backbone (fold these into the finalize-stage config) ===")
display(summary_df)

# COMMAND ----------
# ---- Persist results (so they survive past the live cell outputs) ----
# The job-run export strips cell stdout, so log a compact, queryable record to
# MLflow AND return it as the notebook exit value for `jobs get-run-output`.
_payload = {"split": EVAL_SPLIT, "grid": grid_rows, "summary": summary}
try:
    mlflow.set_experiment(EXPERIMENT_NAME)  # from 00_config
except Exception as _e:
    print(f"set_experiment skipped: {_e}")
with mlflow.start_run(run_name="eval-threshold-grid"):
    for _bk, _info in best_per_backbone.items():
        _b, _d = _info["best"], _info["default"]
        mlflow.log_params({
            f"{_bk}.best_score": _b["score"],
            f"{_bk}.best_nms_iou": _b["nms_iou"],
            f"{_bk}.best_max_det": _b["max_det"],
        })
        mlflow.log_metrics({
            f"{_bk}.best_mAP50": float(_b["mAP_50"]),
            f"{_bk}.best_caries_AP50": float(_b["caries_AP50"]),
            f"{_bk}.default_mAP50": float(_d["mAP_50"]) if _d else float("nan"),
            f"{_bk}.free_gain": float(_b["mAP_50"] - _d["mAP_50"]) if _d else float("nan"),
        })
    with tempfile.NamedTemporaryFile("w", suffix="_threshold_grid.json", delete=False) as _f:
        json.dump(_payload, _f, indent=2)
        _grid_json = _f.name
    mlflow.log_artifact(_grid_json, artifact_path="threshold_grid")

# COMMAND ----------
dbutils.notebook.exit(json.dumps(_payload))
