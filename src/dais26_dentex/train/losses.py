"""Detection loss = focal classification + smooth-L1 box regression.

Replaces the cls-only loss in the old `train_detector.py` (which discarded
``_box_reg`` and ``_anchors``). Box weights drifted unsupervised → inference
returned random boxes from a randomly-initialized regression subnet.

The classification loss is unchanged (focal loss, sigmoid). The box loss is
smooth-L1 on encoded ``(dx, dy, dw, dh)`` deltas, masked by the ``fg_mask``
returned by `models.targets.build_targets_for_batch` and normalized by the
positive-anchor count.

Returns a dict instead of a single scalar so the trainer can log
``train/cls_loss`` and ``train/box_loss`` separately — the breakdown matters
when one head plateaus while the other still improves.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def focal_classification_loss(
    cls_logits: torch.Tensor,
    cls_targets: torch.Tensor,
    fg_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss with explicit ignore handling.

    Args:
        cls_logits: ``(B, N, C)`` raw logits.
        cls_targets: ``(B, N, C)`` one-hot targets.
        fg_mask: ``(B, N)`` True where anchor is positive.
        ignore_mask: ``(B, N)`` True where the anchor is in the ignore zone
            (between bg_iou and fg_iou). These contribute nothing to either
            term.
        alpha: focal balancing factor (default 0.25).
        gamma: focal modulating factor (default 2.0).

    Returns: scalar loss = sum(focal terms over fg + bg) / max(num_positives, 1).
    """
    p = torch.sigmoid(cls_logits)
    ce = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, reduction="none")
    p_t = p * cls_targets + (1 - p) * (1 - cls_targets)
    alpha_t = alpha * cls_targets + (1 - alpha) * (1 - cls_targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce  # (B, N, C)

    # Drop the ignored anchors entirely from the sum.
    keep = (~ignore_mask).unsqueeze(-1)  # (B, N, 1)
    loss = loss * keep
    num_positives = max(int(fg_mask.sum().item()), 1)
    return loss.sum() / num_positives


def smooth_l1_box_loss(
    box_pred: torch.Tensor,
    box_targets: torch.Tensor,
    fg_mask: torch.Tensor,
    beta: float = 1.0 / 9.0,
) -> torch.Tensor:
    """Smooth-L1 (Huber) loss over the matched anchors only.

    Args:
        box_pred: ``(B, N, 4)`` predicted ``(dx, dy, dw, dh)`` deltas.
        box_targets: ``(B, N, 4)`` target deltas (same encoding).
        fg_mask: ``(B, N)`` True for matched anchors.
        beta: smooth-L1 transition point (RetinaNet uses 1/9).

    Returns: scalar loss normalized by positive count. Returns 0 if no
    positives — the trainer handles this gracefully (no division-by-zero).
    """
    num_positives = max(int(fg_mask.sum().item()), 1)
    if not fg_mask.any():
        return box_pred.sum() * 0.0  # preserve the graph; value is 0

    pred = box_pred[fg_mask]  # (P, 4)
    target = box_targets[fg_mask]  # (P, 4)
    return F.smooth_l1_loss(pred, target, beta=beta, reduction="sum") / num_positives


def giou_box_loss(
    box_pred: torch.Tensor,
    box_targets: torch.Tensor,
    anchors: torch.Tensor,
    fg_mask: torch.Tensor,
) -> torch.Tensor:
    """Generalized-IoU box loss over matched anchors only.

    Unlike :func:`smooth_l1_box_loss` (which penalizes the *encoded delta*
    error and is scale-dependent), GIoU is computed on the *decoded* xyxy boxes
    and is scale-invariant — it directly optimizes overlap, which usually lifts
    localization (mAP@50:95) and small-box recall.

    Args:
        box_pred: ``(B, N, 4)`` predicted ``(dx, dy, dw, dh)`` deltas.
        box_targets: ``(B, N, 4)`` target deltas (same encoding).
        anchors: ``(N, 4)`` xyxy anchors shared across the batch.
        fg_mask: ``(B, N)`` True for matched anchors.

    Returns: scalar loss = sum(1 - GIoU over positives) / max(num_positives, 1).
    Returns a graph-preserving 0 when there are no positives.
    """
    from dais26_dentex.models.detection_head import decode_boxes

    num_positives = max(int(fg_mask.sum().item()), 1)
    if not fg_mask.any():
        return box_pred.sum() * 0.0

    b, n, _ = box_pred.shape
    anchors_b = anchors.unsqueeze(0).expand(b, n, 4)
    pa = anchors_b[fg_mask].float()  # (P, 4)
    # Decode in fp32 for numerical stability (exp in decode + IoU divisions).
    pred_xyxy = decode_boxes(box_pred[fg_mask].float(), pa)
    tgt_xyxy = decode_boxes(box_targets[fg_mask].float(), pa)
    giou = _giou_elementwise(pred_xyxy, tgt_xyxy)  # (P,)
    return (1.0 - giou).sum() / num_positives


def _giou_elementwise(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Element-wise generalized IoU between two ``(P, 4)`` xyxy box tensors.

    Pairwise (not the NxM matrix) so it stays O(P) for the matched anchors.
    """
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    lt = torch.maximum(a[:, :2], b[:, :2])
    rb = torch.minimum(a[:, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=eps)

    # Smallest enclosing box.
    lt_c = torch.minimum(a[:, :2], b[:, :2])
    rb_c = torch.maximum(a[:, 2:], b[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, 0] * wh_c[:, 1]
    return iou - (area_c - union) / area_c.clamp(min=eps)


def detection_loss(
    cls_logits: torch.Tensor,
    box_pred: torch.Tensor,
    cls_targets: torch.Tensor,
    box_targets: torch.Tensor,
    fg_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
    *,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    box_weight: float = 1.0,
    box_loss_type: str = "smooth_l1",
    anchors: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Combined detection loss.

    Returns ``{'loss', 'cls_loss', 'box_loss', 'num_positives'}``. The trainer
    backprops ``loss``; the components are for MLflow logging.

    ``box_loss_type`` selects the regression term: ``"smooth_l1"`` (legacy
    delta Huber, default) or ``"giou"`` (1 - GIoU on decoded boxes, which
    requires ``anchors`` to decode).
    """
    cls_loss = focal_classification_loss(
        cls_logits,
        cls_targets,
        fg_mask,
        ignore_mask,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )
    if box_loss_type == "giou":
        if anchors is None:
            raise ValueError("box_loss_type='giou' requires `anchors` to decode boxes")
        box_loss = giou_box_loss(box_pred, box_targets, anchors, fg_mask)
    else:
        box_loss = smooth_l1_box_loss(box_pred, box_targets, fg_mask)
    total = cls_loss + box_weight * box_loss
    return {
        "loss": total,
        "cls_loss": cls_loss.detach(),
        "box_loss": box_loss.detach(),
        "num_positives": fg_mask.sum().detach(),
    }


__all__ = [
    "detection_loss",
    "focal_classification_loss",
    "giou_box_loss",
    "smooth_l1_box_loss",
]
