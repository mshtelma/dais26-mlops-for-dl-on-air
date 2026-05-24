import pytest
import torch
import torch.nn as nn

from src.models.backbones import BackboneInfo, CRadioWrapper, DinoV2Wrapper


def test_backbone_info_dataclass():
    info = BackboneInfo(
        name='cradio_v4_so400m', summary_dim=1152, spatial_dim=1536,
        patch_size=16, model_name='nvidia/C-RADIOv4-SO400M', revision=None,
    )
    assert info.summary_dim == 1152
    assert info.spatial_dim == 1536
    # frozen dataclass
    with pytest.raises(AttributeError):
        info.summary_dim = 999  # type: ignore


def test_cradio_wrapper_tuple_output():
    class FakeCradio(nn.Module):
        def forward(self, x):
            return (torch.randn(x.shape[0], 1152), torch.randn(x.shape[0], 64, 1536))

    wrapper = CRadioWrapper(FakeCradio())
    summary, spatial = wrapper(torch.randn(2, 3, 224, 224))
    assert summary.shape == (2, 1152)
    assert spatial.shape == (2, 64, 1536)


def test_cradio_wrapper_object_output():
    class FakeOut:
        def __init__(self, s, f):
            self.summary = s
            self.features = f

    class FakeCradio(nn.Module):
        def forward(self, x):
            return FakeOut(torch.randn(x.shape[0], 1152), torch.randn(x.shape[0], 64, 1536))

    wrapper = CRadioWrapper(FakeCradio())
    summary, spatial = wrapper(torch.randn(2, 3, 224, 224))
    assert summary.shape == (2, 1152)
    assert spatial.shape == (2, 64, 1536)


def test_dinov2_wrapper_dict_output():
    class FakeDinov2(nn.Module):
        def forward_features(self, x):
            b = x.shape[0]
            return {
                'x_norm_clstoken': torch.randn(b, 768),
                'x_norm_patchtokens': torch.randn(b, 256, 768),
            }

    wrapper = DinoV2Wrapper(FakeDinov2())
    summary, spatial = wrapper(torch.randn(2, 3, 224, 224))
    assert summary.shape == (2, 768)
    assert spatial.shape == (2, 256, 768)


def test_dinov2_wrapper_tensor_output():
    class FakeDinov2(nn.Module):
        def __call__(self, x):
            b = x.shape[0]
            return torch.randn(b, 257, 768)  # T+1 = 257 (CLS + 256 patches)

    wrapper = DinoV2Wrapper(FakeDinov2())
    summary, spatial = wrapper(torch.randn(2, 3, 224, 224))
    assert summary.shape == (2, 768)
    assert spatial.shape == (2, 256, 768)


def test_load_backbone_dinov2_via_hub(monkeypatch):
    """Mock torch.hub.load to avoid actual download."""
    class FakeDinov2(nn.Module):
        def forward_features(self, x):
            b = x.shape[0]
            return {
                'x_norm_clstoken': torch.randn(b, 768),
                'x_norm_patchtokens': torch.randn(b, 256, 768),
            }

    def fake_hub_load(*args, **kwargs):
        return FakeDinov2()

    monkeypatch.setattr(torch.hub, 'load', fake_hub_load)
    from src.models.backbones import load_backbone
    backbone, info = load_backbone('dinov2_base', device='cpu')
    assert info.summary_dim == 768
    assert info.spatial_dim == 768
    assert info.patch_size == 14
    # Backbone is frozen and eval
    assert not backbone.training
    for p in backbone.parameters():
        assert not p.requires_grad
