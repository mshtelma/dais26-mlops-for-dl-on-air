import pytest
import torch

from dais26_dentex.models.adapters import FPNAdapter


@pytest.mark.parametrize("in_channels", [1152, 768])
def test_fpn_output_shapes(in_channels: int):
    """Verify FPN produces 4 levels at expected spatial scales."""
    adapter = FPNAdapter(in_channels=in_channels, out_channels=256)
    adapter.eval()
    # Simulate 1024x1024 input with patch_size=16 -> spatial_shape (64, 64), T=4096
    batch, h, w = 2, 64, 64
    tokens = torch.randn(batch, h * w, in_channels)
    with torch.no_grad():
        out = adapter(tokens, spatial_shape=(h, w))
    assert set(out.keys()) == {"p3", "p4", "p5", "p6"}
    assert out["p3"].shape == (batch, 256, 128, 128)
    assert out["p4"].shape == (batch, 256, 64, 64)
    assert out["p5"].shape == (batch, 256, 32, 32)
    assert out["p6"].shape == (batch, 256, 16, 16)


def test_fpn_shape_mismatch_raises():
    adapter = FPNAdapter(in_channels=1152)
    tokens = torch.randn(1, 100, 1152)  # T=100 but spatial_shape says 64*64=4096
    with pytest.raises(ValueError, match="spatial_shape"):
        adapter(tokens, spatial_shape=(64, 64))


def test_fpn_channel_mismatch_raises():
    adapter = FPNAdapter(in_channels=1152)
    tokens = torch.randn(1, 4096, 768)  # wrong C
    with pytest.raises(ValueError, match="in_channels"):
        adapter(tokens, spatial_shape=(64, 64))


def test_fpn_param_count_reasonable():
    """FPN should be lightweight (~2-4M params)."""
    adapter = FPNAdapter(in_channels=1152, out_channels=256)
    total = sum(p.numel() for p in adapter.parameters())
    assert total < 5_000_000, f"FPN too large: {total} params"
    assert total > 500_000, f"FPN suspiciously small: {total} params"


def test_fpn_groupnorm_used_not_batchnorm():
    """Verify GroupNorm (not BatchNorm) for batch-size-1 inference safety."""
    adapter = FPNAdapter(in_channels=1152)
    has_bn = any(isinstance(m, torch.nn.BatchNorm1d | torch.nn.BatchNorm2d) for m in adapter.modules())
    has_gn = any(isinstance(m, torch.nn.GroupNorm) for m in adapter.modules())
    assert not has_bn, "FPN must not use BatchNorm"
    assert has_gn, "FPN must use GroupNorm"


def test_fpn_works_with_batch_1():
    """Ensure batch=1 works (BatchNorm would fail in eval mode without running stats)."""
    adapter = FPNAdapter(in_channels=1152)
    adapter.eval()
    tokens = torch.randn(1, 4096, 1152)
    with torch.no_grad():
        out = adapter(tokens, spatial_shape=(64, 64))
    assert out["p4"].shape == (1, 256, 64, 64)
