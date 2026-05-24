from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

BackboneName = Literal['cradio_v4_so400m', 'dinov3_vitl16', 'dinov2_base']


@dataclass(frozen=True)
class BackboneInfo:
    """Single source of truth for backbone output dimensions.

    Different backbones return different output structures:
    - C-RADIOv4: tuple (summary, spatial_features), summary_dim != spatial_dim
    - DINOv2: single tensor (B, T, D) where summary is just the CLS token at index 0; summary_dim == spatial_dim
    Downstream consumers (FPN, embedder, drift, VS index) must parameterize on these dims.
    """

    name: str
    summary_dim: int        # C-RADIOv4: 1152; DINOv2-base: 768
    spatial_dim: int        # C-RADIOv4: 1536; DINOv2-base: 768
    patch_size: int         # C-RADIOv4: 16; DINOv2-base: 14
    model_name: str         # HuggingFace ID or torch.hub identifier
    revision: str | None    # Pinned commit SHA (None for torch.hub)


class CRadioWrapper(nn.Module):
    """Wraps C-RADIOv4 to ensure consistent (summary, spatial_features) tuple output."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(images)
        # C-RADIOv4 returns either a tuple (summary, features) directly or a RadioOutput-like object
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1]
        # Fallback: try attribute access
        if hasattr(out, 'summary') and hasattr(out, 'features'):
            return out.summary, out.features
        raise RuntimeError(f'Unexpected C-RADIOv4 output type: {type(out)}')


class DinoV2Wrapper(nn.Module):
    """Wraps DINOv2 to provide consistent (summary, spatial_features) tuple output.

    DINOv2 forward returns dict with keys including 'x_norm_clstoken' and 'x_norm_patchtokens'.
    Or for torch.hub version: returns Tensor (B, T+1, D) where index 0 is CLS.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model.forward_features(images) if hasattr(self.model, 'forward_features') else self.model(images)
        if isinstance(out, dict):
            summary = out['x_norm_clstoken']            # (B, D)
            spatial = out['x_norm_patchtokens']         # (B, T, D)
            return summary, spatial
        if isinstance(out, torch.Tensor) and out.ndim == 3:
            # (B, T+1, D) with CLS at index 0
            summary = out[:, 0, :]                      # (B, D)
            spatial = out[:, 1:, :]                     # (B, T, D)
            return summary, spatial
        raise RuntimeError(f'Unexpected DINOv2 output type: {type(out)}')


def load_backbone(
    name: BackboneName,
    revision: str | None = None,
    cache_dir: str | None = None,
    device: str = 'cuda',
) -> tuple[nn.Module, BackboneInfo]:
    """Load a vision backbone, return wrapped model + dimension info.

    cache_dir is honored via HF_HOME env override (so HF caches go to UC Volume).
    Returned backbone is frozen (requires_grad_=False) and in eval() mode.
    """
    if cache_dir is not None:
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir

    if name == 'cradio_v4_so400m':
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            'nvidia/C-RADIOv4-SO400M',
            trust_remote_code=True,
            revision=revision,
            cache_dir=cache_dir,
        )
        wrapped = CRadioWrapper(model)
        info = BackboneInfo(
            name='cradio_v4_so400m',
            summary_dim=1152,
            spatial_dim=1536,
            patch_size=16,
            model_name='nvidia/C-RADIOv4-SO400M',
            revision=revision,
        )
    elif name == 'dinov3_vitl16':
        from transformers import AutoModel

        token = os.environ.get('HF_TOKEN')
        model = AutoModel.from_pretrained(
            'facebook/dinov3-vitl16-pretrain-lvd1689m',
            token=token,
            revision=revision,
            cache_dir=cache_dir,
        )
        wrapped = DinoV2Wrapper(model)  # DINOv3 output structure similar to DINOv2 for our purposes
        info = BackboneInfo(
            name='dinov3_vitl16',
            summary_dim=1024,
            spatial_dim=1024,
            patch_size=16,
            model_name='facebook/dinov3-vitl16-pretrain-lvd1689m',
            revision=revision,
        )
    elif name == 'dinov2_base':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)
        wrapped = DinoV2Wrapper(model)
        info = BackboneInfo(
            name='dinov2_base',
            summary_dim=768,
            spatial_dim=768,
            patch_size=14,
            model_name='facebookresearch/dinov2/dinov2_vitb14',
            revision=None,
        )
    else:
        raise ValueError(f'Unknown backbone: {name}')

    wrapped.requires_grad_(False)
    wrapped.eval()
    wrapped.to(device)
    return wrapped, info
