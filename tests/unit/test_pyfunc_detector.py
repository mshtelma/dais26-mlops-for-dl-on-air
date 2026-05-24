import base64
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image


def _make_png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _FakeBackbone(nn.Module):
    """Stand-in for C-RADIOv4 returning (summary, spatial)."""

    def __init__(self, summary_dim: int = 1152, spatial_dim: int = 1536, patch_size: int = 16):
        super().__init__()
        self.summary_dim = summary_dim
        self.spatial_dim = spatial_dim
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        ph, pw = h // self.patch_size, w // self.patch_size
        return torch.randn(b, self.summary_dim), torch.randn(b, ph * pw, self.spatial_dim)


@pytest.fixture
def detector_artifacts(tmp_path: Path, monkeypatch) -> dict[str, str]:
    """Build a minimal artifacts bundle for DetectorPyfunc.load_context."""
    from src.models.detection_head import DetectionModel

    model = DetectionModel(
        backbone=_FakeBackbone(),
        spatial_dim=1536,
        num_classes=4,
        scales=[16, 32, 64, 128],
        aspect_ratios=[0.5, 1.0, 2.0],
        patch_size=16,
    )
    state_path = tmp_path / "model_state.pt"
    torch.save(model.state_dict(), state_path)

    backbone_config = tmp_path / "backbone_config.json"
    backbone_config.write_text(json.dumps({
        "name": "fake_backbone",
        "revision": None,
        "summary_dim": 1152,
        "spatial_dim": 1536,
        "patch_size": 16,
    }))

    detection_config = tmp_path / "detection_config.json"
    detection_config.write_text(json.dumps({
        "num_classes": 4,
        "scales": [16, 32, 64, 128],
        "aspect_ratios": [0.5, 1.0, 2.0],
        "score_threshold": 0.05,
        "nms_iou_threshold": 0.5,
        "max_detections": 100,
        "input_size": 64,
    }))

    label_map = tmp_path / "label_map.json"
    label_map.write_text(json.dumps({
        "0": "Caries",
        "1": "Deep Caries",
        "2": "Periapical Lesion",
        "3": "Impacted",
    }))

    # Monkey-patch load_backbone to return our fake
    from src.models.backbones import BackboneInfo

    def fake_load(name, revision=None, cache_dir=None, device="cpu"):
        return _FakeBackbone(), BackboneInfo(
            name=name, summary_dim=1152, spatial_dim=1536,
            patch_size=16, model_name="fake", revision=revision,
        )

    monkeypatch.setattr("src.models.backbones.load_backbone", fake_load)

    return {
        "model_state": str(state_path),
        "backbone_config": str(backbone_config),
        "detection_config": str(detection_config),
        "label_map": str(label_map),
    }


def test_detector_predict_output_schema(detector_artifacts):
    from src.serve.detector_pyfunc import DetectorPyfunc

    pyfunc = DetectorPyfunc()
    ctx = MagicMock()
    ctx.artifacts = detector_artifacts
    pyfunc.load_context(ctx)

    df = pd.DataFrame({"image": [_make_png_b64(), _make_png_b64()]})
    out = pyfunc.predict(ctx, df)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {"boxes", "scores", "labels", "num_detections"}
    assert len(out) == 2
    for _, row in out.iterrows():
        assert isinstance(row["boxes"], list)
        assert isinstance(row["scores"], list)
        assert isinstance(row["labels"], list)
        assert isinstance(row["num_detections"], int)
        # If detections exist, labels should be class names
        for lbl in row["labels"]:
            assert lbl in {"Caries", "Deep Caries", "Periapical Lesion", "Impacted"}


def test_detector_missing_image_column(detector_artifacts):
    from src.serve.detector_pyfunc import DetectorPyfunc

    pyfunc = DetectorPyfunc()
    ctx = MagicMock()
    ctx.artifacts = detector_artifacts
    pyfunc.load_context(ctx)
    with pytest.raises(ValueError, match="must have 'image' column"):
        pyfunc.predict(ctx, pd.DataFrame({"foo": ["bar"]}))


def test_build_signature_and_example():
    from src.serve.detector_pyfunc import build_signature_and_example

    sig, example = build_signature_and_example()
    assert sig is not None
    assert isinstance(example, pd.DataFrame)
    assert "image" in example.columns
