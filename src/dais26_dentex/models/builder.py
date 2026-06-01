"""Single source of truth for assembling a `DetectionModel`.

Consolidates the `load_backbone + apply_lora + DetectionModel(...)` recipe
that used to be inlined at `train_detector.py:130-150` AND open-coded again
at `detector_pyfunc.py:71-89`. Two copies = two ways the train-time and
serve-time stacks can drift (e.g. the trainer adds LoRA but the pyfunc
loader can't, or the trainer changes anchor scales but the pyfunc keeps
the old defaults).

`build_detector(cfg)` returns the same `(model, info)` pair from both call
sites. The trainer wraps it in DDP; the pyfunc loader loads the head/FPN
state into it. Backbone weights still come from HF on the serve side --
only head/FPN state is persisted.
"""

from __future__ import annotations

import logging
from typing import cast

import torch

from dais26_dentex.config.trainer_config import TrainerConfig
from dais26_dentex.data.dentex_loader import get_label_map
from dais26_dentex.distributed.barrier_dance import rank0_first
from dais26_dentex.models.backbones import BackboneInfo, BackboneName, load_backbone
from dais26_dentex.models.detection_head import (
    DEFAULT_ANCHOR_SCALES,
    DEFAULT_ASPECT_RATIOS,
    DetectionModel,
)
from dais26_dentex.models.peft import apply_lora

logger = logging.getLogger(__name__)


def resolve_num_classes(cfg: TrainerConfig) -> int:
    """`cfg.num_classes` if set, else `len(get_label_map())`.

    Centralizing this here prevents the old hardcoded `num_classes=4` from
    leaking back in. The label map is the source of truth -- if the dataset
    grows a class, only `dentex_loader.LABEL_MAP` changes.
    """
    if cfg.num_classes is not None:
        return cfg.num_classes
    return len(get_label_map())


def build_detector(
    cfg: TrainerConfig,
    *,
    device: torch.device | str,
    apply_peft: bool | None = None,
) -> tuple[DetectionModel, BackboneInfo]:
    """Build a `DetectionModel` from a `TrainerConfig`.

    Args:
        cfg: TrainerConfig instance. Drives backbone choice, anchor params,
            score/NMS thresholds, LoRA injection, num_classes resolution.
        device: target device. Backbone weights load directly here.
        apply_peft: override `cfg.use_lora`. The pyfunc loader passes
            `False` because LoRA is merged into the backbone before the
            head/FPN state is saved (see `merge_lora_for_serving`); merging
            again at load time double-applies the deltas. Trainer leaves it
            `None` so `cfg.use_lora` wins.

    Returns:
        `(detection_model, backbone_info)`. Caller is responsible for any
        DDP wrap, optimizer construction, or `state_dict` load.
    """
    use_lora_effective = cfg.use_lora if apply_peft is None else apply_peft

    # rank0_first guards the cold-cache multi-rank race; degrades to a plain
    # yield when world_size <= 1. See docs/RUNBOOK.md#hf-cache-race.
    # `cfg.validate()` already enforces `backbone_name in ALLOWED_BACKBONES`,
    # so the cast is safe; pyright just can't narrow `str` alone.
    backbone_name = cast(BackboneName, cfg.backbone_name)
    with rank0_first():
        backbone, info = load_backbone(
            name=backbone_name,
            revision=cfg.backbone_revision,
            cache_dir=cfg.cache_dir,
            device=str(device),
        )

    if use_lora_effective:
        backbone = apply_lora(backbone, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
        logger.info("LoRA injected (rank=%d, alpha=%.1f)", cfg.lora_rank, cfg.lora_alpha)

    num_classes = resolve_num_classes(cfg)
    model = DetectionModel(
        backbone=backbone,
        spatial_dim=info.spatial_dim,
        num_classes=num_classes,
        scales=DEFAULT_ANCHOR_SCALES,
        aspect_ratios=DEFAULT_ASPECT_RATIOS,
        patch_size=info.patch_size,
    ).to(device)

    return model, info


__all__ = [
    "build_detector",
    "resolve_num_classes",
]
