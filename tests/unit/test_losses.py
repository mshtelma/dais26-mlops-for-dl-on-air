"""Tests for the detection loss (focal cls + smooth-L1 box).

Replaces the cls-only `focal_loss` that the legacy `train_detector.py` used.
The two correctness traps the test must catch:

  (1) `_box_reg` was discarded → the new `detection_loss` must include a
      non-zero box_loss when there are positives.
  (2) The ignore-zone anchors must NOT contribute to either term.
"""

from __future__ import annotations

import torch

import pytest

from dais26_dentex.train.losses import (
    _giou_elementwise,
    detection_loss,
    focal_classification_loss,
    giou_box_loss,
    smooth_l1_box_loss,
)


def test_focal_loss_ignores_ignore_mask() -> None:
    """Anchors flagged in `ignore_mask` must contribute zero to the loss."""
    n, c = 4, 3
    cls_logits = torch.randn(1, n, c, requires_grad=True)
    cls_targets = torch.zeros(1, n, c)
    fg_mask = torch.zeros(1, n, dtype=torch.bool)
    fg_mask[0, 0] = True
    cls_targets[0, 0, 1] = 1.0  # the one positive anchor
    ignore_mask = torch.zeros(1, n, dtype=torch.bool)
    ignore_mask[0, 1] = True
    ignore_mask[0, 2] = True

    loss_with_ignore = focal_classification_loss(
        cls_logits,
        cls_targets,
        fg_mask,
        ignore_mask,
        alpha=0.25,
        gamma=2.0,
    )

    # Now mutate the *ignored* logits drastically; the loss must not change.
    perturbed = cls_logits.detach().clone()
    perturbed[0, 1] += 5.0
    perturbed[0, 2] -= 5.0
    perturbed.requires_grad_(True)
    loss_perturbed = focal_classification_loss(
        perturbed,
        cls_targets,
        fg_mask,
        ignore_mask,
        alpha=0.25,
        gamma=2.0,
    )
    assert torch.allclose(loss_with_ignore, loss_perturbed, atol=1e-6), "ignore_mask anchors leaked into focal loss"


def test_focal_loss_normalized_by_positives() -> None:
    """Loss scales with 1 / max(num_positives, 1)."""
    cls_logits = torch.zeros(1, 2, 2)  # logits=0 → p=0.5
    cls_targets = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    fg_a = torch.tensor([[True, False]])  # 1 positive
    fg_b = torch.tensor([[True, True]])  # 2 positives
    ignore = torch.zeros(1, 2, dtype=torch.bool)

    loss_a = focal_classification_loss(cls_logits, cls_targets, fg_a, ignore)
    loss_b = focal_classification_loss(cls_logits, cls_targets, fg_b, ignore)
    # With identical logits/targets, doubling positives must halve per-positive loss.
    assert loss_a.item() > 0
    assert loss_b.item() > 0
    assert loss_a.item() > loss_b.item()


def test_smooth_l1_zero_when_no_positives() -> None:
    """No positives → 0 loss, but the gradient graph is preserved."""
    box_pred = torch.randn(1, 5, 4, requires_grad=True)
    box_targets = torch.zeros(1, 5, 4)
    fg_mask = torch.zeros(1, 5, dtype=torch.bool)
    loss = smooth_l1_box_loss(box_pred, box_targets, fg_mask)
    assert loss.item() == 0.0
    assert loss.requires_grad  # graph preserved


def test_smooth_l1_only_uses_fg_anchors() -> None:
    """Non-fg anchors must not affect the loss value."""
    box_pred = torch.zeros(1, 4, 4, requires_grad=True)
    box_targets = torch.zeros(1, 4, 4)
    box_targets[0, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    fg_mask = torch.tensor([[True, False, False, False]])
    loss_a = smooth_l1_box_loss(box_pred, box_targets, fg_mask)

    # Change only the non-fg targets — loss must not change.
    box_targets_b = box_targets.clone()
    box_targets_b[0, 1] = torch.tensor([10.0, 10.0, 10.0, 10.0])
    box_targets_b[0, 2] = torch.tensor([-7.0, -7.0, -7.0, -7.0])
    loss_b = smooth_l1_box_loss(box_pred, box_targets_b, fg_mask)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)


def test_detection_loss_returns_dict_and_finite() -> None:
    """End-to-end: combined loss is finite, non-negative, and reports components."""
    cls_logits = torch.randn(2, 6, 4)
    box_pred = torch.randn(2, 6, 4)
    cls_targets = torch.zeros(2, 6, 4)
    cls_targets[0, 0, 1] = 1.0
    box_targets = torch.zeros(2, 6, 4)
    box_targets[0, 0] = torch.tensor([0.1, -0.2, 0.05, 0.0])
    fg_mask = torch.zeros(2, 6, dtype=torch.bool)
    fg_mask[0, 0] = True
    ignore_mask = torch.zeros(2, 6, dtype=torch.bool)
    ignore_mask[0, 3] = True

    out = detection_loss(
        cls_logits=cls_logits,
        box_pred=box_pred,
        cls_targets=cls_targets,
        box_targets=box_targets,
        fg_mask=fg_mask,
        ignore_mask=ignore_mask,
    )
    assert set(out.keys()) >= {"loss", "cls_loss", "box_loss", "num_positives"}
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() >= 0.0
    assert out["cls_loss"].item() >= 0.0
    assert out["box_loss"].item() >= 0.0
    assert int(out["num_positives"].item()) == 1


def test_detection_loss_box_term_nonzero_with_positives() -> None:
    """Catches B1: when positives exist, box_loss must be > 0 unless prediction is exact."""
    # logits/targets matched (cls_loss minimal); box_pred deliberately wrong.
    cls_logits = torch.full((1, 1, 2), -10.0)  # near-zero p
    cls_targets = torch.zeros(1, 1, 2)
    box_pred = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
    box_targets = torch.zeros(1, 1, 4)
    fg_mask = torch.tensor([[True]])
    ignore_mask = torch.zeros(1, 1, dtype=torch.bool)
    out = detection_loss(
        cls_logits=cls_logits,
        box_pred=box_pred,
        cls_targets=cls_targets,
        box_targets=box_targets,
        fg_mask=fg_mask,
        ignore_mask=ignore_mask,
    )
    assert out["box_loss"].item() > 0.0


def test_detection_loss_weighting() -> None:
    """`box_weight` scales the box term linearly."""
    cls_logits = torch.zeros(1, 1, 2)
    cls_targets = torch.zeros(1, 1, 2)
    box_pred = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
    box_targets = torch.zeros(1, 1, 4)
    fg_mask = torch.tensor([[True]])
    ignore_mask = torch.zeros(1, 1, dtype=torch.bool)

    out_1 = detection_loss(
        cls_logits=cls_logits,
        box_pred=box_pred,
        cls_targets=cls_targets,
        box_targets=box_targets,
        fg_mask=fg_mask,
        ignore_mask=ignore_mask,
        box_weight=1.0,
    )
    out_2 = detection_loss(
        cls_logits=cls_logits,
        box_pred=box_pred,
        cls_targets=cls_targets,
        box_targets=box_targets,
        fg_mask=fg_mask,
        ignore_mask=ignore_mask,
        box_weight=2.0,
    )
    delta = out_2["loss"].item() - out_1["loss"].item()
    assert abs(delta - out_1["box_loss"].item()) < 1e-6


# --- GIoU box loss (Track B) ----------------------------------------------


def test_giou_elementwise_identical_boxes_is_one() -> None:
    """GIoU of a box with itself is exactly 1.0."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 7.0, 9.0]])
    giou = _giou_elementwise(boxes, boxes)
    assert torch.allclose(giou, torch.ones(2), atol=1e-5)


def test_giou_elementwise_disjoint_boxes_is_negative() -> None:
    """Far-apart boxes: IoU=0 and the enclosing-box penalty drives GIoU negative."""
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[100.0, 100.0, 101.0, 101.0]])
    giou = _giou_elementwise(a, b)
    assert giou.item() < 0.0


def test_giou_box_loss_zero_when_prediction_exact() -> None:
    """Zero deltas decode to the anchor; identical pred/target → ~0 loss."""
    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]])
    box_pred = torch.zeros(1, 2, 4)
    box_targets = torch.zeros(1, 2, 4)
    fg_mask = torch.tensor([[True, True]])
    loss = giou_box_loss(box_pred, box_targets, anchors, fg_mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_giou_box_loss_zero_when_no_positives_keeps_graph() -> None:
    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    box_pred = torch.randn(1, 1, 4, requires_grad=True)
    box_targets = torch.zeros(1, 1, 4)
    fg_mask = torch.zeros(1, 1, dtype=torch.bool)
    loss = giou_box_loss(box_pred, box_targets, anchors, fg_mask)
    assert loss.item() == 0.0
    assert loss.requires_grad


def test_detection_loss_giou_requires_anchors() -> None:
    """box_loss_type='giou' without anchors is a hard error, not a silent fallback."""
    cls_logits = torch.zeros(1, 1, 2)
    cls_targets = torch.zeros(1, 1, 2)
    box_pred = torch.zeros(1, 1, 4)
    box_targets = torch.zeros(1, 1, 4)
    fg_mask = torch.tensor([[True]])
    ignore_mask = torch.zeros(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires `anchors`"):
        detection_loss(
            cls_logits=cls_logits,
            box_pred=box_pred,
            cls_targets=cls_targets,
            box_targets=box_targets,
            fg_mask=fg_mask,
            ignore_mask=ignore_mask,
            box_loss_type="giou",
        )


def test_detection_loss_giou_runs_with_anchors() -> None:
    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    cls_logits = torch.full((1, 1, 2), -10.0)
    cls_targets = torch.zeros(1, 1, 2)
    box_pred = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])  # decode to a shifted/scaled box
    box_targets = torch.zeros(1, 1, 4)
    fg_mask = torch.tensor([[True]])
    ignore_mask = torch.zeros(1, 1, dtype=torch.bool)
    out = detection_loss(
        cls_logits=cls_logits,
        box_pred=box_pred,
        cls_targets=cls_targets,
        box_targets=box_targets,
        fg_mask=fg_mask,
        ignore_mask=ignore_mask,
        box_loss_type="giou",
        anchors=anchors,
    )
    assert torch.isfinite(out["loss"])
    assert out["box_loss"].item() > 0.0
