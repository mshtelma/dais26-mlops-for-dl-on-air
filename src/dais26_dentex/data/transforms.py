"""torchvision.transforms.v2-based transform pipelines for DENTEX detection.

Replaces the previous ``albumentations`` pipeline; functionally equivalent for
this project (longest-edge resize + bottom-right zero-pad + optional flip /
colour-jitter + CLIP normalisation), but drops the ``albucore → stringzilla``
dependency chain that fails to install on our wheel-only test platform.

The returned object is a *callable* that accepts the same dict-keyed
``(image, bboxes, class_labels)`` shape consumers (e.g. ``DENTEXDetectionDataset``)
already pass — no call-site changes needed.

Bounding boxes flow through the pipeline as ``tv_tensors.BoundingBoxes`` so the
resize / pad / flip ops auto-update their coordinates; we hand back plain
``list[list[float]]`` to keep the dataset's existing tensor-conversion path
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F  # noqa: N812

# CLIP ViT normalisation constants (used by C-RADIOv4 / SAM family). Single
# source of truth — `embeddings.py` and `detector_pyfunc.py` import from here.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _resize_and_pad(
    image: torch.Tensor,
    boxes: tv_tensors.BoundingBoxes,
    target_size: int,
) -> tuple[torch.Tensor, tv_tensors.BoundingBoxes]:
    """Longest-side resize to ``target_size`` then bottom-right zero-pad to a
    ``(target_size, target_size)`` square. Equivalent to albumentations'
    ``LongestMaxSize + PadIfNeeded(border_mode=0, value=0)``.
    """
    h, w = image.shape[-2:]
    scale = target_size / max(h, w)
    new_h = round(h * scale)
    new_w = round(w * scale)
    image = F.resize(image, [new_h, new_w], antialias=True)
    boxes = F.resize(boxes, [new_h, new_w])
    pad = [0, 0, target_size - new_w, target_size - new_h]  # left, top, right, bottom
    image = F.pad(image, pad, fill=0)
    boxes = F.pad(boxes, pad, fill=0)
    return image, boxes


def _make_pipeline(*, img_size: int, train: bool) -> Callable[..., dict[str, Any]]:
    color_jitter = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05) if train else None

    def _apply(
        *,
        image: np.ndarray,
        bboxes: list[list[float]],
        class_labels: list[int],
    ) -> dict[str, Any]:
        img_t = F.to_image(image)  # HWC uint8 numpy → CHW uint8 tensor
        h, w = img_t.shape[-2:]
        boxes_tensor = (
            torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
            if len(bboxes) > 0
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        boxes = tv_tensors.BoundingBoxes(
            boxes_tensor,
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=(h, w),
        )
        labels = torch.as_tensor(class_labels, dtype=torch.long)

        img_t, boxes = _resize_and_pad(img_t, boxes, img_size)

        if train:
            if torch.rand(()).item() < 0.5:
                img_t = F.horizontal_flip(img_t)
                boxes = F.horizontal_flip(boxes)
            if torch.rand(()).item() < 0.5 and color_jitter is not None:
                img_t = color_jitter(img_t)

        img_t = F.to_dtype(img_t, dtype=torch.float32, scale=True)
        img_t = F.normalize(img_t, mean=CLIP_MEAN, std=CLIP_STD)

        out_boxes = boxes.tolist() if boxes.numel() > 0 else []
        return {
            "image": img_t,
            "bboxes": out_boxes,
            "class_labels": labels.tolist(),
        }

    return _apply


def get_train_transforms(img_size: int = 1024) -> Callable[..., dict[str, Any]]:
    """Return training augmentation callable.

    Resizes to ``img_size`` via longest-side scaling + zero-padding, applies
    optional horizontal flip and colour jitter (each at p=0.5), then normalises
    and converts to a ``torch.Tensor`` in (C, H, W) order.
    """
    return _make_pipeline(img_size=img_size, train=True)


def get_val_transforms(img_size: int = 1024) -> Callable[..., dict[str, Any]]:
    """Return validation / inference transform callable (no augmentation)."""
    return _make_pipeline(img_size=img_size, train=False)


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
