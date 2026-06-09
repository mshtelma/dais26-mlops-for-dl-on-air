import base64
import io
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

    def __init__(self, summary_dim: int = 1152, spatial_dim: int = 1152, patch_size: int = 16):
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
    """Build a minimal v2 artifacts bundle for DetectorPyfunc.load_context.

    v2 collapses the old backbone_config/detection_config/label_map trio into a
    single typed ``manifest.json`` (see config/manifest.py)."""
    from dais26_dentex.config.manifest import BackboneSpec, DetectorSpec, Manifest
    from dais26_dentex.models.detection_head import DetectionModel

    model = DetectionModel(
        backbone=_FakeBackbone(),
        spatial_dim=1152,
        num_classes=4,
        scales=[16, 32, 64, 128],
        aspect_ratios=[0.5, 1.0, 2.0],
        patch_size=16,
    )
    state_path = tmp_path / "model_state.pt"
    torch.save(model.state_dict(), state_path)

    manifest_path = tmp_path / "manifest.json"
    Manifest(
        backbone=BackboneSpec(
            name="fake_backbone",
            revision=None,
            summary_dim=1152,
            spatial_dim=1152,
            patch_size=16,
        ),
        detector=DetectorSpec(
            num_classes=4,
            scales=[16, 32, 64, 128],
            aspect_ratios=[0.5, 1.0, 2.0],
            score_threshold=0.05,
            nms_iou_threshold=0.5,
            max_detections=100,
            input_size=64,
        ),
        label_map={
            "0": "Caries",
            "1": "Deep Caries",
            "2": "Periapical Lesion",
            "3": "Impacted",
        },
    ).write(manifest_path)

    # Monkey-patch load_backbone to return our fake. `**kwargs` absorbs the
    # serving-only knobs (local_files_only, fusion_layers) the builder passes.
    from dais26_dentex.models.backbones import BackboneInfo

    def fake_load(name, revision=None, cache_dir=None, device="cpu", **kwargs):
        return _FakeBackbone(), BackboneInfo(
            name=name,
            summary_dim=1152,
            spatial_dim=1152,
            patch_size=16,
            model_name="fake",
            revision=revision,
        )

    monkeypatch.setattr("dais26_dentex.models.backbones.load_backbone", fake_load)

    return {
        "model_state": str(state_path),
        "manifest": str(manifest_path),
    }


def test_detector_predict_output_schema(detector_artifacts):
    from dais26_dentex.serve.detector_pyfunc import DetectorPyfunc

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
    from dais26_dentex.serve.detector_pyfunc import DetectorPyfunc

    pyfunc = DetectorPyfunc()
    ctx = MagicMock()
    ctx.artifacts = detector_artifacts
    pyfunc.load_context(ctx)
    with pytest.raises(ValueError, match="must have 'image' column"):
        pyfunc.predict(ctx, pd.DataFrame({"foo": ["bar"]}))


def test_letterbox_decode_and_inverse_roundtrip():
    """Serving preprocessing must match training (`transforms._resize_and_pad`):
    aspect-preserving longest-side resize + bottom-right zero-pad, and the box
    inverse must be a single uniform scale. The old anisotropic squash + per-axis
    inverse silently wrecked served mAP on non-square images; this guards it.
    """
    from dais26_dentex.serve.detector_pyfunc import _decode_b64_image, _predict_batch

    # Wide non-square image: 200 wide x 100 tall (DENTEX panoramics are ~2:1).
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), (255, 0, 0)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    input_size = 64
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]
    tensor, orig = _decode_b64_image(b64, input_size, mean, std)

    assert orig == (100, 200)  # (H, W)
    assert tuple(tensor.shape) == (3, input_size, input_size)
    # Longest side 200 -> 64, so content is 64w x 32h in the top-left; the
    # bottom 32 rows are zero-pad (would be a full square under the old squash).
    assert torch.allclose(tensor[:, 32:, :], torch.zeros(3, 32, input_size))
    assert tensor[:, :32, :].abs().sum() > 0

    # A prediction covering the full content box [0,0,64,32] must map back to the
    # full original image [0,0,200,100]. The old per-axis inverse would have
    # produced [0,0,200,50] (half height) — this asserts the uniform inverse.
    class _FixedModel(nn.Module):
        def forward(self, batch):
            return {
                "boxes": [torch.tensor([[0.0, 0.0, 64.0, 32.0]])],
                "scores": [torch.tensor([0.9])],
                "labels": [torch.tensor([0])],
            }

    out = _predict_batch(
        model=_FixedModel(),
        device="cpu",
        label_map={0: "Caries"},
        input_size=input_size,
        mean=mean,
        std=std,
        model_input=pd.DataFrame({"image": [b64]}),
    )
    assert out.iloc[0]["boxes"][0] == pytest.approx([0.0, 0.0, 200.0, 100.0], abs=1e-3)


def test_build_signature_and_example():
    from dais26_dentex.serve.detector_pyfunc import build_signature_and_example

    sig, example = build_signature_and_example()
    assert sig is not None
    assert isinstance(example, pd.DataFrame)
    assert "image" in example.columns
