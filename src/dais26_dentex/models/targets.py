"""IoU-based anchor → ground-truth matcher and box encoding.

Replaces the smoke-test `_build_targets` that lived in `train_detector.py` —
that stub deterministically marked the first N anchors positive for the gt
class regardless of geometry, so box-regression weights drifted unsupervised
and inference returned random boxes from a randomly-initialized regression
subnet.

Standard RetinaNet matcher:
    * IoU(anchor, gt) >= fg_iou           → positive (matched)
    * IoU(anchor, gt) <  bg_iou           → negative (background)
    * bg_iou <= IoU(anchor, gt) < fg_iou  → ignored (no contribution)
    * for every gt, force its best-IoU anchor to be positive — guarantees
      every gt object has at least one matched anchor even when all IoUs
      are below `fg_iou` (e.g. very small gt boxes vs. large anchor scales).

Box-target parameterization mirrors `decode_boxes` in detection_head.py
exactly so encode/decode are inverses.
"""

from __future__ import annotations

import torch
from torchvision.ops import box_iou


def coco_xywh_to_xyxy(boxes_xywh: torch.Tensor) -> torch.Tensor:
    """COCO ``[x, y, w, h]`` → ``[x1, y1, x2, y2]`` corner format."""
    if boxes_xywh.numel() == 0:
        return boxes_xywh.reshape(0, 4)
    x, y, w, h = boxes_xywh.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def encode_boxes_xyxy_to_deltas(
    gt_boxes_xyxy: torch.Tensor,
    anchors_xyxy: torch.Tensor,
) -> torch.Tensor:
    """Encode ``gt_boxes`` as ``(dx, dy, dw, dh)`` deltas relative to the
    matched anchors.

    Inverse of `decode_boxes` in detection_head.py — keep these in sync.
    Both shapes are ``(N, 4)`` with one anchor per gt box.
    """
    a_x1, a_y1, a_x2, a_y2 = anchors_xyxy.unbind(-1)
    a_w = (a_x2 - a_x1).clamp(min=1.0)
    a_h = (a_y2 - a_y1).clamp(min=1.0)
    a_cx = a_x1 + a_w / 2
    a_cy = a_y1 + a_h / 2

    g_x1, g_y1, g_x2, g_y2 = gt_boxes_xyxy.unbind(-1)
    g_w = (g_x2 - g_x1).clamp(min=1.0)
    g_h = (g_y2 - g_y1).clamp(min=1.0)
    g_cx = g_x1 + g_w / 2
    g_cy = g_y1 + g_h / 2

    dx = (g_cx - a_cx) / a_w
    dy = (g_cy - a_cy) / a_h
    dw = torch.log(g_w / a_w)
    dh = torch.log(g_h / a_h)
    return torch.stack([dx, dy, dw, dh], dim=-1)


def match_anchors_to_targets(
    anchors_xyxy: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    fg_iou: float = 0.5,
    bg_iou: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match anchors to ground-truth boxes via IoU.

    Args:
        anchors_xyxy: ``(N, 4)`` anchors in xyxy.
        gt_boxes_xyxy: ``(G, 4)`` ground-truth boxes in xyxy.
        gt_labels: ``(G,)`` integer class labels in ``[0, num_classes)``.
        num_classes: number of classes (size of cls_target last dim).
        fg_iou: IoU threshold above which an anchor is positive.
        bg_iou: IoU threshold below which an anchor is negative.

    Returns:
        cls_target: ``(N, num_classes)`` one-hot at the matched class for
            positive anchors; all-zero for negative + ignored.
        box_target: ``(N, 4)`` ``(dx, dy, dw, dh)`` deltas; values at non-
            positive rows are placeholder zeros (masked by `fg_mask`).
        fg_mask: ``(N,)`` bool — True where the anchor is matched.
        ignore_mask: ``(N,)`` bool — True where the anchor falls in the
            ``[bg_iou, fg_iou)`` ignore zone.
    """
    n_anchors = anchors_xyxy.shape[0]
    device = anchors_xyxy.device
    cls_target = torch.zeros(n_anchors, num_classes, device=device)
    box_target = torch.zeros(n_anchors, 4, device=device)
    fg_mask = torch.zeros(n_anchors, dtype=torch.bool, device=device)
    ignore_mask = torch.zeros(n_anchors, dtype=torch.bool, device=device)

    if gt_boxes_xyxy.shape[0] == 0:
        return cls_target, box_target, fg_mask, ignore_mask

    iou = box_iou(anchors_xyxy, gt_boxes_xyxy)  # (N, G)
    best_iou_per_anchor, best_gt_per_anchor = iou.max(dim=1)
    _, best_anchor_per_gt = iou.max(dim=0)

    fg_mask = best_iou_per_anchor >= fg_iou
    ignore_mask = (best_iou_per_anchor >= bg_iou) & ~fg_mask

    # Force each gt's best anchor to be positive — keeps every gt visible to
    # the loss even when the best IoU is below `fg_iou`.
    fg_mask[best_anchor_per_gt] = True
    ignore_mask[best_anchor_per_gt] = False

    if fg_mask.any():
        matched_gt = best_gt_per_anchor[fg_mask]
        labels = gt_labels[matched_gt].to(torch.long).clamp(0, num_classes - 1)
        positive_idx = fg_mask.nonzero(as_tuple=False).squeeze(-1)
        cls_target[positive_idx, labels] = 1.0

        matched_boxes = gt_boxes_xyxy[matched_gt]
        matched_anchors = anchors_xyxy[fg_mask]
        box_target[fg_mask] = encode_boxes_xyxy_to_deltas(matched_boxes, matched_anchors)

    return cls_target, box_target, fg_mask, ignore_mask


def build_targets_for_batch(
    anchors_xyxy: torch.Tensor,
    targets_per_image: list[dict[str, torch.Tensor]],
    num_classes: int,
    fg_iou: float = 0.5,
    bg_iou: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-image matcher → batched tensors.

    ``targets_per_image`` is the per-image list emitted by `detection_collate`;
    each entry has ``boxes`` (G, 4) in COCO xywh and ``labels`` (G,) long.
    """
    device = anchors_xyxy.device
    cls_targets, box_targets, fg_masks, ignore_masks = [], [], [], []
    for t in targets_per_image:
        boxes_xywh = t["boxes"].to(device)
        labels = t["labels"].to(device)
        boxes_xyxy = coco_xywh_to_xyxy(boxes_xywh)
        cls_t, box_t, fg, ignore = match_anchors_to_targets(
            anchors_xyxy,
            boxes_xyxy,
            labels,
            num_classes,
            fg_iou,
            bg_iou,
        )
        cls_targets.append(cls_t)
        box_targets.append(box_t)
        fg_masks.append(fg)
        ignore_masks.append(ignore)
    return (
        torch.stack(cls_targets),
        torch.stack(box_targets),
        torch.stack(fg_masks),
        torch.stack(ignore_masks),
    )


__all__ = [
    "build_targets_for_batch",
    "coco_xywh_to_xyxy",
    "encode_boxes_xyxy_to_deltas",
    "match_anchors_to_targets",
]
