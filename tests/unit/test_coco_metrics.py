from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dais26_dentex.eval.coco_metrics import evaluate_coco, format_predictions_for_coco


@pytest.fixture()
def coco_gt_file(tmp_path: Path) -> str:
    """Minimal COCO annotation JSON with 1 image, 1 category, 1 annotation."""
    gt = {
        "images": [{"id": 1, "width": 640, "height": 480, "file_name": "img1.jpg"}],
        "categories": [{"id": 1, "name": "cat", "supercategory": "animal"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 100.0, 80.0],  # x,y,w,h
                "area": 8000.0,
                "iscrowd": 0,
            }
        ],
    }
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt))
    return str(gt_path)


def test_format_predictions_for_coco_basic() -> None:
    model_output = [
        {
            "image_id": 1,
            "boxes": np.array([[10.0, 20.0, 110.0, 100.0]]),
            "scores": np.array([0.9]),
            "labels": np.array([1]),
        }
    ]
    preds = format_predictions_for_coco(model_output)
    assert len(preds) == 1
    assert preds[0]["image_id"] == 1
    assert preds[0]["category_id"] == 1
    assert preds[0]["score"] == pytest.approx(0.9)
    # xyxy -> xywh: x2-x1=100, y2-y1=80
    assert preds[0]["bbox"] == pytest.approx([10.0, 20.0, 100.0, 80.0])


def test_evaluate_coco_returns_expected_keys(coco_gt_file: str) -> None:
    model_output = [
        {
            "image_id": 1,
            "boxes": np.array([[10.0, 20.0, 110.0, 100.0]]),
            "scores": np.array([0.9]),
            "labels": np.array([1]),
        }
    ]
    preds = format_predictions_for_coco(model_output)
    results = evaluate_coco(preds, coco_gt_file)

    for key in ("mAP_50", "mAP_50_95", "mAP_75", "AR_1", "AR_10", "AR_100"):
        assert key in results, f"Missing key: {key}"
        assert isinstance(results[key], float)

    assert "per_class_AP50" in results
    assert isinstance(results["per_class_AP50"], dict)
    assert "cat" in results["per_class_AP50"]


def test_evaluate_coco_no_predictions_returns_zeros(coco_gt_file: str) -> None:
    results = evaluate_coco([], coco_gt_file)
    assert results["mAP_50"] == 0.0
    assert results["mAP_50_95"] == 0.0
    assert results["per_class_AP50"] == {}
