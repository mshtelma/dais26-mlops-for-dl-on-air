"""Detection postprocessing: anchor decode + per-class score filter + NMS.

Pulled out of `DetectionModel.forward` so the model returns raw
logits/regression/anchors only — pyfunc / serving / eval code shares one
postprocess implementation. Easier to unit-test, easier to override score
thresholds at inference, and removes the inline `from torchvision.ops
import nms` from the hot path.

Inputs are batched the same way `DetectionModel.forward_train` returns
them: `cls_logits` (B, N, C), `box_pred` (B, N, 4), `anchors` (N, 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.ops import nms

from dais26_dentex.models.detection_head import decode_boxes


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """Score / NMS / cap knobs used at inference.

    Defaults match the legacy `DetectionModel.__init__` so behavior is
    unchanged when an artifact doesn't override them.
    """

    score_threshold: float = 0.05
    nms_iou_threshold: float = 0.5
    max_detections: int = 100


def postprocess_detections(
    cls_logits: torch.Tensor,
    box_pred: torch.Tensor,
    anchors: torch.Tensor,
    cfg: PostprocessConfig | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Decode boxes, filter by score, run per-image class-agnostic NMS.

    Args:
        cls_logits: (B, N, C) raw logits from the head.
        box_pred:   (B, N, 4) (dx, dy, dw, dh) deltas relative to ``anchors``.
        anchors:    (N, 4) xyxy anchors shared across the batch.
        cfg: thresholds; defaults to legacy behavior when omitted.

    Returns:
        ``{"boxes": [Tensor], "scores": [Tensor], "labels": [Tensor]}``
        with one entry per batch image. Empty tensors when nothing passes
        ``score_threshold``.
    """
    cfg = cfg or PostprocessConfig()
    b = cls_logits.shape[0]
    scores_full = torch.sigmoid(cls_logits)
    device = cls_logits.device

    boxes_out: list[torch.Tensor] = []
    scores_out: list[torch.Tensor] = []
    labels_out: list[torch.Tensor] = []

    for i in range(b):
        decoded = decode_boxes(box_pred[i], anchors)
        n, c = scores_full[i].shape
        flat_scores = scores_full[i].reshape(-1)
        flat_labels = torch.arange(c, device=device).repeat(n)
        flat_boxes = decoded.repeat_interleave(c, dim=0)

        keep = flat_scores > cfg.score_threshold
        flat_scores = flat_scores[keep]
        flat_labels = flat_labels[keep]
        flat_boxes = flat_boxes[keep]

        if flat_boxes.numel() == 0:
            boxes_out.append(flat_boxes)
            scores_out.append(flat_scores)
            labels_out.append(flat_labels)
            continue

        keep_idx = nms(flat_boxes, flat_scores, cfg.nms_iou_threshold)[: cfg.max_detections]
        boxes_out.append(flat_boxes[keep_idx])
        scores_out.append(flat_scores[keep_idx])
        labels_out.append(flat_labels[keep_idx])

    return {"boxes": boxes_out, "scores": scores_out, "labels": labels_out}


__all__ = ["PostprocessConfig", "postprocess_detections"]
