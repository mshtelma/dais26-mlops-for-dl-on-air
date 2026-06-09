"""Unit tests for the multi-layer ViT feature fusion combiner (Track B).

`_LayerFusion` is the only fusion piece testable without a real HF backbone
download — the DINOv3 wrapper's `output_hidden_states` path needs the model.
"""

from __future__ import annotations

import pytest
import torch

from dais26_dentex.models.backbones import _LayerFusion


def test_fusion_preserves_shape():
    fusion = _LayerFusion(num_layers=3, dim=8)
    maps = [torch.randn(2, 5, 8) for _ in range(3)]
    out = fusion(maps)
    assert out.shape == (2, 5, 8)


def test_fusion_uniform_at_init_is_mean_of_layernormed():
    """Weight inits to zeros → softmax uniform → output is the plain mean of the
    per-layer LayerNorm'd maps (the ViT-Det equal-average hypothesis)."""
    torch.manual_seed(0)
    fusion = _LayerFusion(num_layers=4, dim=6)
    fusion.eval()
    maps = [torch.randn(2, 3, 6) for _ in range(4)]
    expected = torch.stack(
        [norm(m) for norm, m in zip(fusion.norms, maps, strict=True)], dim=0
    ).mean(dim=0)
    out = fusion(maps)
    assert torch.allclose(out, expected, atol=1e-5)


def test_fusion_weight_is_learnable_and_sized():
    fusion = _LayerFusion(num_layers=4, dim=6)
    assert fusion.weight.requires_grad
    assert fusion.weight.shape == (4,)
    # All norms are trainable too.
    assert all(p.requires_grad for p in fusion.norms.parameters())


def test_fusion_wrong_map_count_raises():
    fusion = _LayerFusion(num_layers=2, dim=4)
    with pytest.raises(ValueError, match="expected 2 maps"):
        fusion([torch.randn(1, 3, 4)])


def test_fusion_backprops_to_weight():
    fusion = _LayerFusion(num_layers=3, dim=4)
    maps = [torch.randn(1, 2, 4) for _ in range(3)]
    fusion(maps).sum().backward()
    assert fusion.weight.grad is not None
    assert fusion.weight.grad.shape == (3,)
