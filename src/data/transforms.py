"""Albumentations-based transform pipelines for DENTEX detection."""

from __future__ import annotations

import albumentations as A  # noqa: N812
from albumentations.pytorch import ToTensorV2

# CLIP ViT normalisation constants (used by C-RADIOv4 / SAM family)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def get_train_transforms(img_size: int = 1024) -> A.Compose:
    """Return training augmentation pipeline.

    Resizes to ``img_size`` via longest-side scaling + zero-padding,
    applies horizontal flip and colour jitter, then normalises and
    converts to a ``torch.Tensor`` in (C, H, W) order.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_visibility=0.3),
    )


def get_val_transforms(img_size: int = 1024) -> A.Compose:
    """Return validation / inference transform pipeline (no augmentation)."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0),
            A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
    )


def get_cradio_preprocessor(revision: str | None = None):
    """Return the C-RADIOv4-SO400M CLIP image processor from HuggingFace.

    Args:
        revision: Optional git revision / tag to pin the processor version.
    """
    from transformers import CLIPImageProcessor

    return CLIPImageProcessor.from_pretrained(
        "nvidia/C-RADIOv4-SO400M",
        revision=revision,
        trust_remote_code=True,
    )
