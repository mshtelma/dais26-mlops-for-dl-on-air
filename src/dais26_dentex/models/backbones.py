from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

from dais26_dentex.data.transforms import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD

BackboneName = Literal["cradio_v4_so400m", "dinov3_vitl16", "dinov2_base"]


@dataclass(frozen=True)
class BackboneInfo:
    """Single source of truth for backbone output dimensions + preprocessing.

    Different backbones return different output structures:
    - C-RADIOv4: tuple (summary, spatial_features), summary_dim != spatial_dim
      (summary is 2304 = 2x the 1152 ViT hidden dim — RADIO concatenates two
      pooled reps; spatial_features keep the 1152 hidden dim).
    - DINOv2: single tensor (B, T, D) where summary is just the CLS token at index 0; summary_dim == spatial_dim
    Downstream consumers (FPN, embedder, drift, VS index) must parameterize on these dims.

    ``image_mean``/``image_std`` are the input normalisation stats the backbone
    was pretrained with: CLIP for C-RADIO, ImageNet for DINOv2/DINOv3. Training,
    eval, and serving must all use the SAME stats — feeding CLIP-normalised
    inputs to DINOv3 makes them OOD and caps mAP (see docs/HPO.md "DINOv3 A/B").
    Defaults to CLIP so legacy constructors (and pre-existing manifests) are
    unchanged.
    """

    name: str
    # C-RADIOv4-SO400M: 2304 — NVIDIA RADIO's `summary` concatenates two pooled
    # representations (CLS + pooled), so it is 2x the ViT hidden dim (2*1152).
    # DINOv2-base / DINOv3: 768 / 1024 (CLS token, == spatial_dim).
    summary_dim: int
    spatial_dim: int  # C-RADIOv4-SO400M: 1152 (SO400M ViT hidden dim); DINOv2-base: 768
    patch_size: int  # C-RADIOv4: 16; DINOv2-base: 14
    model_name: str  # HuggingFace ID or torch.hub identifier
    revision: str | None  # Pinned commit SHA (None for torch.hub)
    image_mean: list[float] = field(default_factory=lambda: list(CLIP_MEAN))
    image_std: list[float] = field(default_factory=lambda: list(CLIP_STD))


class _LayerFusion(nn.Module):
    """Learnable softmax-weighted fusion of several ViT layers' patch tokens.

    Given a list of ``num_layers`` token maps each ``(B, T, D)`` (one per
    selected hidden-state depth), apply a per-layer LayerNorm and combine them
    with a learnable softmax weight into a single ``(B, T, D)`` map. The weight
    is initialised to zeros so the softmax starts uniform — i.e. an equal
    average of the chosen depths, the ViT-Det multi-scale-feature hypothesis —
    then learns to re-weight depths during training.
    """

    def __init__(self, num_layers: int, dim: int) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.weight = nn.Parameter(torch.zeros(num_layers))

    def forward(self, maps: list[torch.Tensor]) -> torch.Tensor:
        if len(maps) != len(self.norms):
            raise ValueError(f"_LayerFusion expected {len(self.norms)} maps, got {len(maps)}")
        stacked = torch.stack(
            [norm(m) for norm, m in zip(self.norms, maps, strict=True)], dim=0
        )  # (L,B,T,D)
        w = torch.softmax(self.weight, dim=0).view(-1, 1, 1, 1).to(stacked.dtype)
        return (w * stacked).sum(dim=0)  # (B,T,D)


def _assert_cradio_runtime_deps() -> None:
    """Pre-flight check for nvidia/C-RADIOv4-SO400M trust_remote_code deps.

    The HF custom modeling code (hf_model.py, radio_model.py, open_clip_adaptor.py,
    extra_timm_models.py, vit_patch_generator.py, dual_hybrid_vit.py) imports timm,
    einops, and open_clip at load time. Without them AutoModel.from_pretrained
    raises a deep-stack ImportError; this guard surfaces a single actionable line
    instead. PyPI dist name `open_clip_torch` maps to import name `open_clip` —
    keep the names below as IMPORT names, not PyPI names.

    Aligned with pyproject.toml dependencies.
    """
    missing: list[str] = []
    for mod in ("timm", "einops", "open_clip"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError(
            f"C-RADIOv4 trust_remote_code requires {missing}. "
            "These should be installed via the package wheel (pyproject deps). "
            "If you are in a notebook, re-run `%pip install --quiet ..` and "
            "`dbutils.library.restartPython()`."
        )


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
        if hasattr(out, "summary") and hasattr(out, "features"):
            return out.summary, out.features
        raise RuntimeError(f"Unexpected C-RADIOv4 output type: {type(out)}")


class DinoV2Wrapper(nn.Module):
    """Wraps DINOv2 to provide consistent (summary, spatial_features) tuple output.

    DINOv2 forward returns dict with keys including 'x_norm_clstoken' and 'x_norm_patchtokens'.
    Or for torch.hub version: returns Tensor (B, T+1, D) where index 0 is CLS.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model.forward_features(images) if hasattr(self.model, "forward_features") else self.model(images)
        if isinstance(out, dict):
            summary = out["x_norm_clstoken"]  # (B, D)
            spatial = out["x_norm_patchtokens"]  # (B, T, D)
            return summary, spatial
        if isinstance(out, torch.Tensor) and out.ndim == 3:
            # (B, T+1, D) with CLS at index 0
            summary = out[:, 0, :]  # (B, D)
            spatial = out[:, 1:, :]  # (B, T, D)
            return summary, spatial
        raise RuntimeError(f"Unexpected DINOv2 output type: {type(out)}")


class DinoV3Wrapper(nn.Module):
    """Wraps HuggingFace DINOv3 to provide consistent (summary, spatial_features) output.

    HF `AutoModel` for `facebook/dinov3-vitl16-pretrain-lvd1689m` returns a
    `BaseModelOutputWithPooling` (a `ModelOutput` dict subclass) whose
    `.last_hidden_state` has shape (B, 1 + num_register_tokens + T, D), with the
    prefix laid out as [CLS, register tokens...]. This differs from DINOv2's
    `x_norm_clstoken` / `x_norm_patchtokens` dict keys, so the DINOv2 wrapper
    cannot be reused here.

    summary = last_hidden_state[:, 0, :]                   (CLS token)
    spatial = last_hidden_state[:, 1 + num_register_tokens:, :]  (patch tokens only)

    Stripping CLS + register tokens is load-bearing: at 1024px / patch16 the
    spatial output must be exactly 64*64 = 4096 patch tokens so `FPNAdapter`
    and `DetectionModel.forward` (ph,pw = H//16, W//16) pass the token-count
    assertion. `num_register_tokens` is read from `model.config` rather than
    hardcoded so a config change doesn't silently misalign the grid.
    """

    def __init__(self, model: nn.Module, fusion_layers: list[int] | None = None) -> None:
        super().__init__()
        self.model = model
        # 1 CLS token + N register tokens precede the patch tokens.
        self.num_prefix = 1 + int(getattr(model.config, "num_register_tokens", 0))
        # Optional multi-layer feature fusion (ViT-Det style). When set, the
        # spatial map fed to the FPN is a learnable weighted blend of these
        # hidden-state depths instead of the last layer only. `fusion` is a
        # trainable head (see builder/trainer param-group routing).
        self.fusion_layers = list(fusion_layers) if fusion_layers else None
        self.fusion: _LayerFusion | None = None
        if self.fusion_layers:
            dim = int(model.config.hidden_size)
            self.fusion = _LayerFusion(len(self.fusion_layers), dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_layers and self.fusion is not None:
            out = self.model(images, output_hidden_states=True)
            hidden = out.hidden_states  # tuple len (num_layers + 1), each (B, 1+R+T, D)
            n_hidden = len(hidden)
            maps: list[torch.Tensor] = []
            for idx in self.fusion_layers:
                if not (-n_hidden <= idx < n_hidden):
                    raise IndexError(
                        f"fusion_layers index {idx} out of range for {n_hidden} hidden states"
                    )
                maps.append(hidden[idx][:, self.num_prefix:, :])  # (B, T, D) patch tokens only
            summary = out.last_hidden_state[:, 0, :]  # CLS -> (B, D)
            spatial = self.fusion(maps)  # (B, T, D)
            return summary, spatial
        out = self.model(images)
        last = out.last_hidden_state  # (B, 1 + R + T, D)
        summary = last[:, 0, :]  # CLS -> (B, D)
        spatial = last[:, self.num_prefix:, :]  # drop CLS + R registers -> (B, T, D)
        return summary, spatial


def load_backbone(
    name: BackboneName,
    revision: str | None = None,
    cache_dir: str | None = None,
    device: str = "cuda",
    local_files_only: bool = False,
    freeze: bool = True,
    fusion_layers: list[int] | None = None,
) -> tuple[nn.Module, BackboneInfo]:
    """Load a vision backbone, return wrapped model + dimension info.

    cache_dir is honored via HF_HOME env override (so HF caches go to UC Volume).

    By default the backbone is frozen (``requires_grad_=False``) and in
    ``eval()`` mode — the historical head-only training path. Pass
    ``freeze=False`` to fine-tune the whole encoder: weights keep
    ``requires_grad=True`` and the module is left in ``train()`` mode so dropout
    / stochastic-depth behave correctly. Partial unfreezing (last-N blocks) is
    layered on top by the caller (``builder.build_detector`` ->
    ``peft.unfreeze_last_blocks``) starting from a frozen load.

    local_files_only forces a strictly-offline load from the cache (set at
    serving time, where the HF cache is bundled into the model artifact and the
    container has no egress). It also flips the HF offline env flags.

    ``fusion_layers`` (DINOv3 only) enables multi-layer ViT feature fusion: the
    spatial map handed to the FPN becomes a learnable weighted blend of those
    hidden-state depths instead of the last layer alone. Its small combiner is
    forced trainable even under ``freeze=True`` so it can learn.
    """
    from dais26_dentex.platform.hf_env import configure_hf_env

    configure_hf_env(cache_dir, offline=local_files_only)

    if name == "cradio_v4_so400m":
        _assert_cradio_runtime_deps()
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "nvidia/C-RADIOv4-SO400M",
            trust_remote_code=True,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        wrapped = CRadioWrapper(model)
        info = BackboneInfo(
            name="cradio_v4_so400m",
            # RADIO `summary` = concat of two pooled reps = 2 x 1152 hidden dim.
            summary_dim=2304,
            spatial_dim=1152,
            patch_size=16,
            model_name="nvidia/C-RADIOv4-SO400M",
            revision=revision,
            image_mean=list(CLIP_MEAN),
            image_std=list(CLIP_STD),
        )
    elif name == "dinov3_vitl16":
        from transformers import AutoModel

        token = os.environ.get("HF_TOKEN")
        model = AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            token=token,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        # HF BaseModelOutputWithPooling: strip CLS + register tokens (+ optional fusion)
        wrapped = DinoV3Wrapper(model, fusion_layers=fusion_layers)
        info = BackboneInfo(
            name="dinov3_vitl16",
            summary_dim=1024,
            spatial_dim=1024,
            patch_size=16,
            model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
            revision=revision,
            image_mean=list(IMAGENET_MEAN),  # HF DINOv3ViTImageProcessor default
            image_std=list(IMAGENET_STD),
        )
    elif name == "dinov2_base":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
        wrapped = DinoV2Wrapper(model)
        info = BackboneInfo(
            name="dinov2_base",
            summary_dim=768,
            spatial_dim=768,
            patch_size=14,
            model_name="facebookresearch/dinov2/dinov2_vitb14",
            revision=None,
            image_mean=list(IMAGENET_MEAN),  # DINOv2 also pretrained with ImageNet stats
            image_std=list(IMAGENET_STD),
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")

    if fusion_layers and name != "dinov3_vitl16":
        raise ValueError(f"fusion_layers is only supported for dinov3_vitl16, not {name!r}")

    if freeze:
        wrapped.requires_grad_(False)
        wrapped.eval()
    else:
        # Full fine-tune: keep grads on and stay in train() mode so dropout /
        # stochastic depth in the encoder behave as during pretraining.
        wrapped.requires_grad_(True)
        wrapped.train()
    # The fusion combiner is a learnable head, not pretrained weights — keep it
    # trainable (and in train() mode) regardless of the encoder freeze policy.
    fusion = getattr(wrapped, "fusion", None)
    if fusion is not None:
        fusion.requires_grad_(True)
        fusion.train()
    wrapped.to(device)
    return wrapped, info
