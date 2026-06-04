"""Shared detector evaluation helpers (DRY across notebooks 09 / 10).

Re-evaluates a registered detector pyfunc from scratch on a held-out DENTEX
split through the real serving `DetectorPyfunc` and scores it with the same
`eval.coco_metrics.evaluate_coco` the trainer uses, so the comparison is
independent of what got logged at train time and identical across notebooks.

Split of responsibilities:
  * `build_name_to_category_id` / `to_category_id` — pure label-name -> COCO
    integer category_id mapping (the pyfunc returns class *names*).
  * `materialize_gt` — load the canonical split, backfill `area`/`iscrowd`, and
    write a normalized COCO GT JSON pycocotools can read.
  * `model_output_row` — pure per-image conversion of a pyfunc prediction row
    into the COCO `model_output` dict (kept separate so it is unit-testable
    without IO / a GPU / pycocotools).
  * `predict_split` — run the pyfunc over every image in the split.
  * `score_model_on_split` — the full path: materialize GT -> predict -> COCO
    score. Returns the `evaluate_coco` metrics dict plus `num_predictions`.

Only `materialize_gt` / `score_model_on_split` touch the filesystem; the row
conversion + name mapping are pure so the scoring shape can be tested directly.
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from dais26_dentex.data.dentex_loader import get_label_map, load_canonical_split
from dais26_dentex.eval.coco_metrics import evaluate_coco, format_predictions_for_coco


def build_name_to_category_id(label_map: dict[int, str] | None = None) -> dict[str, int]:
    """Return the inverse of the canonical label map: class name -> category_id.

    Defaults to the DENTEX `get_label_map()` ({0: "Caries", ...}).
    """
    lm = label_map if label_map is not None else get_label_map()
    return {v: k for k, v in lm.items()}


def to_category_id(name: object, name_to_id: dict[str, int]) -> int:
    """Map a predicted class *name* -> integer category_id.

    The pyfunc returns class names (e.g. "Caries"); COCO scoring wants the
    integer id. Fall back to int() only for the (defensive) case of an already
    numeric label — done as an explicit branch, NOT a dict-get default, since
    the default expression `int(name)` would be evaluated eagerly even for known
    names and raise on "Caries".
    """
    key = str(name)
    if key in name_to_id:
        return name_to_id[key]
    return int(key)


def materialize_gt(volume_path: str, split: str) -> tuple[dict[str, Any], str]:
    """Load the canonical COCO ground-truth for `split` and write a temp JSON.

    `load_canonical_split` normalizes DENTEX's hierarchical category_id_3 -> our
    flat category_id in memory. We backfill `area` / `iscrowd` (COCOeval's
    area-range buckets need them) and dump to a temp file that `evaluate_coco`
    reads via pycocotools. Returns (gt_dict, gt_path); the caller owns deleting
    the path (or use `score_model_on_split`, which cleans up).
    """
    coco_gt = load_canonical_split(volume_path, split)
    for ann in coco_gt["annotations"]:
        if "area" not in ann:
            _x, _y, w, h = ann["bbox"]
            ann["area"] = float(w) * float(h)
        ann.setdefault("iscrowd", 0)

    # delete=False on purpose: pycocotools.COCO reads the path after we close it;
    # the caller (or score_model_on_split) unlinks it. A `with` block would delete
    # it too early, so SIM115 doesn't apply here.
    gt_tmp = tempfile.NamedTemporaryFile("w", suffix=f"_{split}_gt.json", delete=False)  # noqa: SIM115
    json.dump(coco_gt, gt_tmp)
    gt_tmp.close()
    return coco_gt, gt_tmp.name


def model_output_row(image_id: int, row: Any, name_to_id: dict[str, int]) -> dict[str, Any]:
    """Convert one pyfunc prediction row into a COCO `model_output` dict.

    `row` is a mapping with `boxes` (N,4 xyxy px), `scores` (N,), `labels` (N
    class names). Predicted label names are mapped back to integer category_ids.
    Pure: no IO, no model — directly unit-testable for the scoring shape.
    """
    labels = [to_category_id(name, name_to_id) for name in row["labels"]]
    return {
        "image_id": int(image_id),
        "boxes": torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(row["scores"], dtype=torch.float32).reshape(-1),
        "labels": torch.tensor(labels, dtype=torch.long).reshape(-1),
    }


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def predict_split(
    model: Any,
    images_dir: str | Path,
    coco_gt: dict[str, Any],
    name_to_id: dict[str, int],
    *,
    predict_chunk: int = 16,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run the pyfunc over every image in the split; return COCO model_output.

    Each element: {image_id, boxes (N,4 xyxy px), scores (N,), labels (N, int)}.
    The pyfunc forwards one image at a time internally, so `predict_chunk` only
    bounds how many rows are built into a DataFrame at once (keeps memory flat).
    """
    import pandas as pd

    images_dir = Path(images_dir)
    items = [(img["id"], images_dir / img["file_name"]) for img in coco_gt["images"]]
    model_output: list[dict[str, Any]] = []
    for start in range(0, len(items), predict_chunk):
        chunk = items[start : start + predict_chunk]
        df_in = pd.DataFrame({"image": [_b64(p) for _, p in chunk]})
        preds = model.predict(df_in).reset_index(drop=True)
        for (image_id, _), (_, row) in zip(chunk, preds.iterrows(), strict=True):
            model_output.append(model_output_row(image_id, row, name_to_id))
        if verbose:
            print(f"    predicted {min(start + predict_chunk, len(items))}/{len(items)}")
    return model_output


def score_model_on_split(
    model: Any,
    volume_path: str,
    split: str,
    *,
    name_to_id: dict[str, int] | None = None,
    predict_chunk: int = 16,
    verbose: bool = True,
) -> dict[str, Any]:
    """Materialize GT, run the pyfunc, and COCO-score it on `split`.

    Returns the `evaluate_coco` metrics dict (`mAP_50`, `mAP_50_95`, `mAP_75`,
    `AR_*`, `per_class_AP50`) plus `num_predictions`. The temp GT file is always
    cleaned up.
    """
    mapping = name_to_id if name_to_id is not None else build_name_to_category_id()
    coco_gt, gt_path = materialize_gt(volume_path, split)
    images_dir = Path(volume_path) / "images" / split
    try:
        model_output = predict_split(
            model, images_dir, coco_gt, mapping, predict_chunk=predict_chunk, verbose=verbose
        )
        coco_preds = format_predictions_for_coco(model_output)
        metrics = evaluate_coco(coco_preds, gt_path)
        metrics["num_predictions"] = len(coco_preds)
        return metrics
    finally:
        Path(gt_path).unlink(missing_ok=True)


__all__ = [
    "build_name_to_category_id",
    "materialize_gt",
    "model_output_row",
    "predict_split",
    "score_model_on_split",
    "to_category_id",
]
