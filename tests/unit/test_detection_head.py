import json
from pathlib import Path

import torch
import torch.nn as nn

from src.models.detection_head import (
    DEFAULT_ANCHOR_SCALES,
    AnchorGenerator,
    DetectionModel,
    RetinaNetHead,
    calibrate_anchors,
    calibrate_aspect_ratios,
    decode_boxes,
    focal_loss,
)


def test_retinanet_head_output_shapes():
    head = RetinaNetHead(in_channels=256, num_classes=4, num_anchors=12, num_convs=2)
    head.eval()
    features = {
        "p3": torch.randn(2, 256, 128, 128),
        "p4": torch.randn(2, 256, 64, 64),
        "p5": torch.randn(2, 256, 32, 32),
        "p6": torch.randn(2, 256, 16, 16),
    }
    with torch.no_grad():
        cls, box = head(features)
    # Total anchors per spatial loc = 12; total cells = 128*128 + 64*64 + 32*32 + 16*16 = 21,840
    total = (128 * 128 + 64 * 64 + 32 * 32 + 16 * 16) * 12
    assert cls.shape == (2, total, 4)
    assert box.shape == (2, total, 4)


def test_retinanet_head_param_count_reasonable():
    head = RetinaNetHead(in_channels=256, num_classes=4, num_anchors=12, num_convs=4)
    total = sum(p.numel() for p in head.parameters())
    assert 500_000 < total < 10_000_000, f"head param count out of range: {total}"


def test_anchor_generator_count_and_shape():
    gen = AnchorGenerator(scales=[16, 32, 64, 128], aspect_ratios=[0.5, 1.0, 2.0])
    features = {
        "p3": torch.zeros(1, 256, 8, 8),
        "p4": torch.zeros(1, 256, 4, 4),
        "p5": torch.zeros(1, 256, 2, 2),
        "p6": torch.zeros(1, 256, 1, 1),
    }
    anchors = gen(features)
    expected = (8 * 8 + 4 * 4 + 2 * 2 + 1 * 1) * (4 * 3)
    assert anchors.shape == (expected, 4)
    # All anchors should have x2 > x1, y2 > y1
    assert (anchors[:, 2] > anchors[:, 0]).all()
    assert (anchors[:, 3] > anchors[:, 1]).all()


def test_focal_loss_returns_scalar():
    logits = torch.randn(2, 100, 4)
    targets = torch.zeros(2, 100, 4)
    targets[:, :10, 0] = 1.0  # 10 positive samples
    loss = focal_loss(logits, targets)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_decode_boxes_identity_with_zero_deltas():
    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    deltas = torch.zeros(2, 4)
    decoded = decode_boxes(deltas, anchors)
    assert torch.allclose(decoded, anchors, atol=1e-5)


def test_calibrate_anchors(tmp_path: Path):
    coco = {
        "images": [{"id": 1, "file_name": "x.png", "width": 1024, "height": 1024}],
        "annotations": [
            {"id": i, "image_id": 1, "bbox": [0, 0, 20, 20], "category_id": 0, "area": 400, "iscrowd": 0}
            for i in range(5)
        ] + [
            {"id": 5 + i, "image_id": 1, "bbox": [0, 0, 100, 100], "category_id": 1, "area": 10000, "iscrowd": 0}
            for i in range(5)
        ],
        "categories": [{"id": 0, "name": "small"}, {"id": 1, "name": "large"}],
    }
    p = tmp_path / "ann.json"
    p.write_text(json.dumps(coco))
    scales = calibrate_anchors(str(p))
    assert len(scales) == 4
    assert scales == sorted(scales)
    # Smallest should reflect the 20px bboxes (sqrt(400)=20, /2=10)
    assert scales[0] <= 25


def test_calibrate_anchors_empty(tmp_path: Path):
    p = tmp_path / "ann.json"
    p.write_text(json.dumps({"annotations": [], "images": [], "categories": []}))
    assert calibrate_anchors(str(p)) == DEFAULT_ANCHOR_SCALES


def test_calibrate_aspect_ratios(tmp_path: Path):
    coco = {
        "images": [{"id": 1}],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [0, 0, 10, 20], "category_id": 0, "area": 200, "iscrowd": 0},
            {"id": 2, "image_id": 1, "bbox": [0, 0, 20, 10], "category_id": 0, "area": 200, "iscrowd": 0},
            {"id": 3, "image_id": 1, "bbox": [0, 0, 20, 20], "category_id": 0, "area": 400, "iscrowd": 0},
        ],
    }
    p = tmp_path / "ann.json"
    p.write_text(json.dumps(coco))
    ratios = calibrate_aspect_ratios(str(p))
    assert len(ratios) == 3
    assert all(0.25 <= r <= 4.0 for r in ratios)


def test_detection_model_inference():
    """End-to-end forward with a fake backbone returning (summary, spatial)."""
    class FakeBackbone(nn.Module):
        def __init__(self, spatial_dim: int = 1536):
            super().__init__()
            self.spatial_dim = spatial_dim

        def forward(self, x: torch.Tensor):
            b, _, h, w = x.shape
            ph = h // 16
            pw = w // 16
            return (
                torch.randn(b, 1152),                  # summary
                torch.randn(b, ph * pw, self.spatial_dim),  # spatial
            )

    bb = FakeBackbone(spatial_dim=1536)
    model = DetectionModel(
        backbone=bb,
        spatial_dim=1536,
        num_classes=4,
        scales=[16, 32, 64, 128],
        aspect_ratios=[0.5, 1.0, 2.0],
        patch_size=16,
    )
    model.eval()
    images = torch.randn(2, 3, 256, 256)  # smaller for fast test
    with torch.no_grad():
        out = model(images)
    assert set(out.keys()) == {"boxes", "scores", "labels"}
    assert len(out["boxes"]) == 2
    assert len(out["scores"]) == 2
    assert len(out["labels"]) == 2
    for box, score, label in zip(out["boxes"], out["scores"], out["labels"], strict=True):
        assert box.ndim == 2 and box.shape[1] == 4
        assert score.ndim == 1
        assert label.ndim == 1
        assert box.shape[0] == score.shape[0] == label.shape[0]


def test_detection_model_forward_train():
    class FakeBackbone(nn.Module):
        def forward(self, x: torch.Tensor):
            b, _, h, w = x.shape
            ph = h // 16
            pw = w // 16
            return (torch.randn(b, 1152), torch.randn(b, ph * pw, 1536))

    model = DetectionModel(backbone=FakeBackbone(), spatial_dim=1536)
    model.eval()
    images = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        cls_logits, box_reg, anchors = model.forward_train(images)
    assert cls_logits.ndim == 3
    assert box_reg.ndim == 3
    assert anchors.ndim == 2
    assert anchors.shape[1] == 4
    assert cls_logits.shape[1] == box_reg.shape[1] == anchors.shape[0]
