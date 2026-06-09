# Databricks notebook source
# MAGIC %md
# MAGIC # 09b — Free eval-threshold grid (no retraining)
# MAGIC
# MAGIC The decode/NMS thresholds (`score_threshold`, `nms_iou_threshold`,
# MAGIC `max_detections`) are **inference-time** knobs: `DetectionModel` reads them
# MAGIC as plain attributes at forward time (see `models/detection_head.py`), so we
# MAGIC can sweep them on an **already-trained, registered** detector without
# MAGIC spending a single GPU-hour on retraining. This is the cheapest lever in the
# MAGIC "push to 0.60" plan (docs/HPO.md) — run it first to bank any free mAP and to
# MAGIC record the per-class **Caries AP@50** baseline that the campaign gates on.
# MAGIC
# MAGIC For each registered backbone (`@candidate` preferred, `@champion` fallback —
# MAGIC champion is the post-tuning end-state; we grade the in-tuning candidate) this notebook:
# MAGIC   1. loads the serving pyfunc once,
# MAGIC   2. mutates the inner `DetectionModel`'s thresholds across a grid,
# MAGIC   3. re-evaluates the held-out split with the same `evaluate_coco` the
# MAGIC      trainer uses, and
# MAGIC   4. prints the best (score, nms_iou, max_det) by mAP@50 plus the Caries
# MAGIC      AP@50 at that point and at the as-registered defaults.
# MAGIC
# MAGIC Requires a **GPU** notebook. The winning thresholds are then folded into the
# MAGIC finalize-stage `TrainerConfig` (now config fields) so the re-registered model
# MAGIC reproduces them at serve time.

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
from itertools import product
from pathlib import Path

import mlflow
import pandas as pd
import torch

from dais26_dentex.data.dentex_loader import get_label_map, load_canonical_split
from dais26_dentex.eval.coco_metrics import evaluate_coco, format_predictions_for_coco

mlflow.set_registry_uri("databricks-uc")

EVAL_SPLIT = "val"          # held out of training; mirrors trainer validation
PREDICT_CHUNK = 16
# Grid the CANDIDATE first: during the push-to-0.60 tuning the freshly-registered
# tuning winners live at @candidate, while @champion is the *post-tuning* end-state
# (often stale/old or, for DINOv3, the known-broken v2). We want to grade what's
# being tuned, so candidate takes precedence and champion is only a fallback.
ALIAS_PREFERENCE = ("candidate", "champion")

# Backbones to grid. Mirrors 09_eval_comparison.py.
COMPARE_BACKBONES: dict[str, dict[str, str]] = {
    "cradio_v4_so400m": {"model_short": "cradio_detector"},
    "dinov3_vitl16": {"model_short": "dinov3_detector"},
}

# The free grid. Smaller `score_threshold` recovers low-confidence true positives
# (helps AR / small-lesion recall); `nms_iou_threshold` trades duplicate
# suppression vs. recall of overlapping lesions; `max_detections` rarely binds on
# dental panoramics but is cheap to check.
SCORE_THRESHOLDS = [0.01, 0.05, 0.10]
NMS_IOU_THRESHOLDS = [0.40, 0.50, 0.60]
MAX_DETECTIONS = [100, 300]

LABEL_MAP = get_label_map()  # {0: "Caries", ...}
NAME_TO_ID = {v: k for k, v in LABEL_MAP.items()}
CARIES_NAME = LABEL_MAP.get(0, "Caries")


def _to_category_id(name: object) -> int:
    key = str(name)
    return NAME_TO_ID[key] if key in NAME_TO_ID else int(key)


print(f"EVAL_SPLIT = {EVAL_SPLIT}")
print(f"Catalog/schema = {CATALOG}.{SCHEMA}")
print(f"Grid: {len(SCORE_THRESHOLDS)}x{len(NMS_IOU_THRESHOLDS)}x{len(MAX_DETECTIONS)} = "
      f"{len(SCORE_THRESHOLDS) * len(NMS_IOU_THRESHOLDS) * len(MAX_DETECTIONS)} combos/backbone")

# COMMAND ----------
# ---- Materialize a normalized COCO ground-truth file (same as 09) ----
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
print(f"GT: {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations -> {GT_PATH}")

# COMMAND ----------
# ---- Helpers ----


def _load_detector(model_short: str):
    full = f"{CATALOG}.{SCHEMA}.{model_short}"
    for alias in ALIAS_PREFERENCE:
        uri = f"models:/{full}@{alias}"
        try:
            return mlflow.pyfunc.load_model(uri), uri
        except Exception as e:  # noqa: BLE001
            print(f"  {uri}: unavailable ({type(e).__name__})")
    return None, None


def _inner_detection_model(loaded):
    """Reach the DetectionModel inside a loaded serving pyfunc.

    Uses the public `unwrap_python_model()` when available (recent MLflow),
    falling back to the private impl path. Returns the torch `DetectionModel`
    whose `score_threshold`/`nms_iou_threshold`/`max_detections` we mutate.
    """
    pyfunc_model = None
    if hasattr(loaded, "unwrap_python_model"):
        try:
            pyfunc_model = loaded.unwrap_python_model()
        except Exception:  # noqa: BLE001
            pyfunc_model = None
    if pyfunc_model is None:
        pyfunc_model = loaded._model_impl.python_model  # noqa: SLF001
    return pyfunc_model.model


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


_ITEMS = [(img["id"], images_dir / img["file_name"]) for img in coco_gt["images"]]
# Pre-encode once; reused across every grid point so the grid cost is pure forward.
_ENCODED = [(image_id, _b64(p)) for image_id, p in _ITEMS]


def _predict_split(loaded) -> list[dict]:
    model_output: list[dict] = []
    for start in range(0, len(_ENCODED), PREDICT_CHUNK):
        chunk = _ENCODED[start : start + PREDICT_CHUNK]
        df_in = pd.DataFrame({"image": [b64 for _, b64 in chunk]})
        preds = loaded.predict(df_in).reset_index(drop=True)
        for (image_id, _), (_, row) in zip(chunk, preds.iterrows(), strict=True):
            labels = [_to_category_id(name) for name in row["labels"]]
            model_output.append(
                {
                    "image_id": int(image_id),
                    "boxes": torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4),
                    "scores": torch.tensor(row["scores"], dtype=torch.float32).reshape(-1),
                    "labels": torch.tensor(labels, dtype=torch.long).reshape(-1),
                }
            )
    return model_output


def _score(loaded) -> dict:
    out = _predict_split(loaded)
    coco_preds = format_predictions_for_coco(out)
    if not coco_preds:
        return {"mAP_50": 0.0, "caries_AP50": 0.0, "n_pred": 0}
    m = evaluate_coco(coco_preds, GT_PATH)
    return {
        "mAP_50": float(m["mAP_50"]),
        "caries_AP50": float(m.get("per_class_AP50", {}).get(CARIES_NAME, float("nan"))),
        "n_pred": len(coco_preds),
    }


# COMMAND ----------
# ---- Grid each backbone ----
grid_rows: list[dict] = []
best_per_backbone: dict[str, dict] = {}

for backbone, cfg in COMPARE_BACKBONES.items():
    print(f"\n=== {backbone} ({cfg['model_short']}) ===")
    loaded, uri = _load_detector(cfg["model_short"])
    if loaded is None:
        print(f"  no registered model for {backbone}; skipping")
        continue
    dm = _inner_detection_model(loaded)
    base = (float(dm.score_threshold), float(dm.nms_iou_threshold), int(dm.max_detections))
    print(f"  loaded {uri}; as-registered thresholds score={base[0]} nms={base[1]} max_det={base[2]}")

    rows: list[dict] = []
    for s, n, d in product(SCORE_THRESHOLDS, NMS_IOU_THRESHOLDS, MAX_DETECTIONS):
        dm.score_threshold = s
        dm.nms_iou_threshold = n
        dm.max_detections = d
        res = _score(loaded)
        row = {"backbone": backbone, "score": s, "nms_iou": n, "max_det": d, **res}
        rows.append(row)
        grid_rows.append(row)
        print(f"    score={s:<4} nms={n:<4} max_det={d:<4} "
              f"mAP50={res['mAP_50']:.4f} caries_AP50={res['caries_AP50']:.4f}")

    rows_df = pd.DataFrame(rows).sort_values("mAP_50", ascending=False).reset_index(drop=True)
    best = rows_df.iloc[0].to_dict()
    # Default (as-registered) row for the free-gain delta.
    dflt = next(
        (r for r in rows if (r["score"], r["nms_iou"], r["max_det"]) == base),
        None,
    )
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
display(grid_df.sort_values(["backbone", "mAP_50"], ascending=[True, False]))  # noqa: F821

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
display(summary_df)  # noqa: F821

# COMMAND ----------
# ---- Persist results (so they survive past the live cell outputs) ----
# The job-run export strips cell stdout, so log a compact, queryable record to
# MLflow AND return it as the notebook exit value for `jobs get-run-output`.
_payload = {
    "split": EVAL_SPLIT,
    "grid": grid_rows,
    "summary": summary,
}
try:
    mlflow.set_experiment(EXPERIMENT_NAME)  # from 00_config
except Exception as _e:  # noqa: BLE001
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
Path(GT_PATH).unlink(missing_ok=True)
dbutils.notebook.exit(json.dumps(_payload))  # noqa: F821
