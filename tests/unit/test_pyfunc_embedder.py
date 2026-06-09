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
    Image.new("RGB", (32, 32), (200, 100, 50)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _FakeBackbone(nn.Module):
    def __init__(self, summary_dim: int = 1152):
        super().__init__()
        self.summary_dim = summary_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        return torch.randn(b, self.summary_dim) * 5.0, torch.zeros(b, 4, 1152)


@pytest.fixture
def embedder_artifacts(tmp_path: Path, monkeypatch) -> dict[str, str]:
    backbone_config = tmp_path / "backbone_config.json"
    backbone_config.write_text(
        json.dumps(
            {
                "name": "fake_backbone",
                "revision": None,
                "summary_dim": 1152,
                "spatial_dim": 1152,
                "patch_size": 16,
            }
        )
    )
    from dais26_dentex.models.backbones import BackboneInfo

    def fake_load(name, revision=None, cache_dir=None, device="cpu", local_files_only=False):
        return _FakeBackbone(), BackboneInfo(
            name=name,
            summary_dim=1152,
            spatial_dim=1152,
            patch_size=16,
            model_name="fake",
            revision=revision,
        )

    monkeypatch.setattr("dais26_dentex.models.backbones.load_backbone", fake_load)
    return {"backbone_config": str(backbone_config)}


def test_embedder_output_dim_and_norm(embedder_artifacts):
    from dais26_dentex.serve.embedder_pyfunc import EmbedderPyfunc

    pyfunc = EmbedderPyfunc()
    ctx = MagicMock()
    ctx.artifacts = embedder_artifacts
    pyfunc.load_context(ctx)

    df = pd.DataFrame({"image": [_make_png_b64()]})
    out = pyfunc.predict(ctx, df)
    assert "embedding" in out.columns
    emb = out["embedding"].iloc[0]
    assert len(emb) == 1152
    # L2-normalized
    norm = sum(x * x for x in emb) ** 0.5
    assert abs(norm - 1.0) < 1e-4, f"embedding not L2-normalized: norm={norm}"


def test_embedder_batch(embedder_artifacts):
    from dais26_dentex.serve.embedder_pyfunc import EmbedderPyfunc

    pyfunc = EmbedderPyfunc()
    ctx = MagicMock()
    ctx.artifacts = embedder_artifacts
    pyfunc.load_context(ctx)
    df = pd.DataFrame({"image": [_make_png_b64() for _ in range(3)]})
    out = pyfunc.predict(ctx, df)
    assert len(out) == 3


def test_embedder_signature():
    from dais26_dentex.serve.embedder_pyfunc import build_embedder_signature_and_example

    sig, example = build_embedder_signature_and_example(summary_dim=1152)
    assert sig is not None
    assert "image" in example.columns
    # Fallback case
    sig768, _ = build_embedder_signature_and_example(summary_dim=768)
    assert sig768 is not None
