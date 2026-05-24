from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def format_predictions_for_coco(
    model_output: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert model output to COCO-format prediction list.

    Each item in model_output: {'image_id': int, 'boxes': Tensor (N,4) xyxy pixel,
                                'scores': Tensor (N,), 'labels': Tensor (N,)}
    Returns list of dicts: [{'image_id': int, 'bbox': [x,y,w,h], 'score': float,
                             'category_id': int}, ...]
    """
    coco_preds: list[dict[str, Any]] = []
    for item in model_output:
        image_id = int(item["image_id"])
        boxes = item["boxes"].cpu().numpy() if hasattr(item["boxes"], "cpu") else np.asarray(item["boxes"])
        scores = item["scores"].cpu().numpy() if hasattr(item["scores"], "cpu") else np.asarray(item["scores"])
        labels = item["labels"].cpu().numpy() if hasattr(item["labels"], "cpu") else np.asarray(item["labels"])
        for box, score, label in zip(boxes, scores, labels, strict=True):
            x1, y1, x2, y2 = box
            coco_preds.append(
                {
                    "image_id": image_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                    "category_id": int(label),
                }
            )
    return coco_preds


def evaluate_coco(
    predictions: list[dict[str, Any]],
    ground_truth_path: str,
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Run COCO evaluation. Returns mAP_50, mAP_50_95, mAP_75, AR_1, AR_10, AR_100,
    per_class_AP50 dict.

    predictions: list from format_predictions_for_coco.
    ground_truth_path: path to COCO annotation JSON.
    """
    coco_gt = COCO(ground_truth_path)
    if not predictions:
        return {
            "mAP_50_95": 0.0,
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "AR_1": 0.0,
            "AR_10": 0.0,
            "AR_100": 0.0,
            "per_class_AP50": {},
        }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(predictions, f)
        pred_path = f.name

    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats  # length 12
    out: dict[str, Any] = {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
    }

    # Per-class AP50: filter eval per category
    per_class: dict[str, float] = {}
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    for cat in cats:
        cat_id = cat["id"]
        coco_eval_c = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval_c.params.catIds = [cat_id]
        coco_eval_c.evaluate()
        coco_eval_c.accumulate()
        # AP50 = precisions at IoU=0.5, averaged over recall + area
        precisions = coco_eval_c.eval["precision"][0]  # IoU=0.5 first index
        valid = precisions[precisions > -1]
        ap50 = float(np.mean(valid)) if len(valid) > 0 else 0.0
        per_class[cat["name"]] = ap50

    out["per_class_AP50"] = per_class
    Path(pred_path).unlink(missing_ok=True)
    return out
