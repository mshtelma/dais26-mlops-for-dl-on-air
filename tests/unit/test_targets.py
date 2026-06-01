"""Tests for IoU-based anchor → ground-truth matcher.

Replaces the smoke-test `_build_targets` stub. The matcher is the most
correctness-critical piece of Phase 3 -- if it returns wrong masks, the
loss trains on garbage signal even with the loss function fixed.
"""

from __future__ import annotations

import torch

from dais26_dentex.models.targets import (
    build_targets_for_batch,
    coco_xywh_to_xyxy,
    encode_boxes_xyxy_to_deltas,
    match_anchors_to_targets,
)


def test_coco_xywh_to_xyxy_basic() -> None:
    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    out = coco_xywh_to_xyxy(boxes)
    assert torch.allclose(out, torch.tensor([[10.0, 20.0, 40.0, 60.0]]))


def test_coco_xywh_to_xyxy_empty() -> None:
    out = coco_xywh_to_xyxy(torch.zeros(0, 4))
    assert out.shape == (0, 4)


def test_encode_decode_inverse() -> None:
    """encode_boxes_xyxy_to_deltas is the inverse of detection_head.decode_boxes."""
    from dais26_dentex.models.detection_head import decode_boxes

    anchors = torch.tensor(
        [
            [0.0, 0.0, 100.0, 100.0],
            [50.0, 50.0, 150.0, 150.0],
        ]
    )
    gt_boxes = torch.tensor(
        [
            [10.0, 20.0, 90.0, 80.0],
            [60.0, 70.0, 140.0, 120.0],
        ]
    )
    deltas = encode_boxes_xyxy_to_deltas(gt_boxes, anchors)
    decoded = decode_boxes(deltas, anchors)
    assert torch.allclose(decoded, gt_boxes, atol=1e-4)


def test_match_anchors_no_gt() -> None:
    anchors = torch.zeros(10, 4)
    gt_boxes = torch.zeros(0, 4)
    gt_labels = torch.zeros(0, dtype=torch.long)
    cls_t, box_t, fg, ignore = match_anchors_to_targets(anchors, gt_boxes, gt_labels, num_classes=4)
    assert cls_t.shape == (10, 4)
    assert box_t.shape == (10, 4)
    assert not fg.any()
    assert not ignore.any()


def test_match_anchors_high_iou_positive() -> None:
    """An anchor that perfectly overlaps a gt box should be positive."""
    anchors = torch.tensor(
        [
            [0.0, 0.0, 100.0, 100.0],  # exact match
            [200.0, 200.0, 300.0, 300.0],  # no overlap
        ]
    )
    gt_boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    gt_labels = torch.tensor([2], dtype=torch.long)
    cls_t, _box_t, fg, ignore = match_anchors_to_targets(
        anchors,
        gt_boxes,
        gt_labels,
        num_classes=4,
        fg_iou=0.5,
        bg_iou=0.4,
    )
    assert fg[0].item() is True
    assert fg[1].item() is False
    assert not ignore.any()
    assert cls_t[0, 2].item() == 1.0
    assert cls_t[0, [0, 1, 3]].sum().item() == 0.0


def test_match_anchors_ignore_zone() -> None:
    """Anchors with IoU in [bg_iou, fg_iou) should be ignored."""
    # Anchor partially overlaps gt -> IoU somewhere in mid range
    anchors = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    gt_boxes = torch.tensor([[50.0, 50.0, 150.0, 150.0]])  # IoU = 1/7 ~ 0.14
    gt_labels = torch.tensor([0], dtype=torch.long)
    # Force-best-anchor-per-gt makes this fg even at low IoU. Test the
    # ignore zone with a setup where the only anchor is the forced-best
    # but a SECOND anchor sits in the [bg_iou, fg_iou) band.
    anchors = torch.tensor(
        [
            [0.0, 0.0, 100.0, 100.0],  # exact match -> fg
            [40.0, 40.0, 110.0, 110.0],  # IoU ~ 0.36 -> ignore band
            [500.0, 500.0, 600.0, 600.0],  # no overlap -> bg
        ]
    )
    gt_boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    cls_t, _box_t, fg, ignore = match_anchors_to_targets(
        anchors,
        gt_boxes,
        gt_labels,
        num_classes=4,
        fg_iou=0.5,
        bg_iou=0.3,
    )
    assert fg[0].item() is True
    assert ignore[1].item() is True  # ignore band
    assert fg[1].item() is False
    assert fg[2].item() is False
    assert ignore[2].item() is False  # below bg_iou is negative
    assert cls_t[1].sum().item() == 0.0  # ignore band has no class signal


def test_force_best_anchor_per_gt() -> None:
    """Even when no anchor reaches fg_iou, every gt's best anchor must be positive."""
    # Tiny gt + big anchors -> all IoUs below fg threshold
    anchors = torch.tensor(
        [
            [0.0, 0.0, 1000.0, 1000.0],
            [0.0, 0.0, 800.0, 800.0],
        ]
    )
    gt_boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])  # tiny
    gt_labels = torch.tensor([1], dtype=torch.long)
    _cls_t, _box_t, fg, _ignore = match_anchors_to_targets(
        anchors,
        gt_boxes,
        gt_labels,
        num_classes=4,
        fg_iou=0.5,
        bg_iou=0.4,
    )
    # The smaller anchor has higher IoU -> picked as best for the gt.
    assert fg[1].item() is True
    assert fg[0].item() is False


def test_build_targets_for_batch_shapes() -> None:
    anchors = torch.tensor(
        [
            [0.0, 0.0, 100.0, 100.0],
            [200.0, 0.0, 300.0, 100.0],
        ]
    )
    targets_per_image = [
        # image 0: one gt that exactly matches anchor 0
        {
            "boxes": torch.tensor([[0.0, 0.0, 100.0, 100.0]]),  # COCO xywh
            "labels": torch.tensor([0], dtype=torch.long),
        },
        # image 1: no gt
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)},
    ]
    cls_t, box_t, fg, ignore = build_targets_for_batch(anchors, targets_per_image, num_classes=4)
    assert cls_t.shape == (2, 2, 4)
    assert box_t.shape == (2, 2, 4)
    assert fg.shape == (2, 2)
    assert ignore.shape == (2, 2)
    # Image 0 anchor 0 is fg (perfect overlap).
    assert fg[0, 0].item() is True
    # Image 1 has no fg.
    assert not fg[1].any().item()


def test_num_classes_parameterized() -> None:
    """Verify the matcher works for arbitrary num_classes (not hardcoded 4)."""
    for nc in (1, 4, 10):
        anchors = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
        gt_boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
        # label is num_classes - 1 (last valid class)
        gt_labels = torch.tensor([nc - 1], dtype=torch.long)
        cls_t, _box_t, fg, _ig = match_anchors_to_targets(
            anchors,
            gt_boxes,
            gt_labels,
            num_classes=nc,
            fg_iou=0.5,
            bg_iou=0.4,
        )
        assert cls_t.shape == (1, nc)
        assert fg[0].item() is True
        assert cls_t[0, nc - 1].item() == 1.0
