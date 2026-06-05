"""torchvision.transforms.v2-based transform pipelines for DENTEX detection.

Replaces the previous ``albumentations`` pipeline; functionally equivalent for
this project (longest-edge resize + bottom-right zero-pad + optional flip /
colour-jitter + per-backbone normalisation), but drops the ``albucore →
stringzilla`` dependency chain that fails to install on our wheel-only test
platform.

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

# ImageNet normalisation constants (used by DINOv2 / DINOv3 — their HF image
# processors default to these; feeding CLIP stats makes DINOv3 inputs OOD and
# caps mAP, see docs/HPO.md "DINOv3 A/B"). The per-backbone choice lives on
# `BackboneInfo.image_mean/std`; these are the values it points at.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _resize_and_pad(
    image: torch.Tensor,
    boxes: tv_tensors.BoundingBoxes,
    canvas_size: int,
    resize_to: int | None = None,
) -> tuple[torch.Tensor, tv_tensors.BoundingBoxes]:
    """Longest-side resize then bottom-right zero-pad to a square canvas.

    The longest side is resized to ``resize_to`` (defaults to ``canvas_size``)
    and the result is zero-padded bottom-right to ``(canvas_size, canvas_size)``.
    With ``resize_to < canvas_size`` this is a *multi-scale* down-scale that keeps
    the output tensor a fixed ``canvas_size`` square (so batching is unaffected)
    while shrinking the content. Equivalent to albumentations'
    ``LongestMaxSize + PadIfNeeded(border_mode=0, value=0)`` when
    ``resize_to == canvas_size``.
    """
    target = resize_to if resize_to is not None else canvas_size
    h, w = image.shape[-2:]
    scale = target / max(h, w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    image = F.resize(image, [new_h, new_w], antialias=True)
    boxes = F.resize(boxes, [new_h, new_w])
    pad = [0, 0, canvas_size - new_w, canvas_size - new_h]  # left, top, right, bottom
    image = F.pad(image, pad, fill=0)
    boxes = F.pad(boxes, pad, fill=0)
    return image, boxes


def _make_pipeline(
    *,
    img_size: int,
    train: bool,
    mean: list[float],
    std: list[float],
    hflip_prob: float = 0.5,
    jitter_prob: float = 0.5,
    jitter_scale: float = 1.0,
    rotation_deg: float = 0.0,
    multiscale_range: list[float] | None = None,
) -> Callable[..., dict[str, Any]]:
    # Base jitter magnitudes scaled by `jitter_scale` (clamp hue to its valid
    # [0, 0.5] range so a large scale doesn't raise inside ColorJitter).
    jitter = max(0.0, jitter_scale)
    color_jitter = (
        v2.ColorJitter(
            brightness=0.2 * jitter,
            contrast=0.2 * jitter,
            saturation=0.2 * jitter,
            hue=min(0.5, 0.05 * jitter),
        )
        if train and jitter > 0.0
        else None
    )

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

        # Multi-scale: jitter the longest-side resize target within the range,
        # then pad back to the fixed `img_size` canvas (train only).
        resize_to: int | None = None
        if train and multiscale_range is not None:
            lo, hi = multiscale_range
            factor = lo + (hi - lo) * torch.rand(()).item()
            resize_to = max(16, round(img_size * factor))
        img_t, boxes = _resize_and_pad(img_t, boxes, img_size, resize_to=resize_to)

        if train:
            if hflip_prob > 0.0 and torch.rand(()).item() < hflip_prob:
                img_t = F.horizontal_flip(img_t)
                boxes = F.horizontal_flip(boxes)
            if rotation_deg > 0.0:
                angle = float((torch.rand(()).item() * 2.0 - 1.0) * rotation_deg)
                img_t = F.rotate(img_t, angle, expand=False, fill=0)
                boxes = F.rotate(boxes, angle, expand=False)
            if color_jitter is not None and torch.rand(()).item() < jitter_prob:
                img_t = color_jitter(img_t)

        img_t = F.to_dtype(img_t, dtype=torch.float32, scale=True)
        img_t = F.normalize(img_t, mean=mean, std=std)

        out_boxes = boxes.tolist() if boxes.numel() > 0 else []
        return {
            "image": img_t,
            "bboxes": out_boxes,
            "class_labels": labels.tolist(),
        }

    return _apply


def get_train_transforms(
    img_size: int = 1024,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    *,
    hflip_prob: float = 0.5,
    jitter_prob: float = 0.5,
    jitter_scale: float = 1.0,
    rotation_deg: float = 0.0,
    multiscale_range: list[float] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Return training augmentation callable.

    Resizes to ``img_size`` via longest-side scaling + zero-padding, applies
    optional multi-scale jitter, horizontal flip, small rotation, and colour
    jitter, then normalises and converts to a ``torch.Tensor`` in (C, H, W)
    order. The keyword defaults reproduce the legacy pipeline (hflip p=0.5 +
    mild jitter p=0.5, no rotation, no multi-scale); the tuning campaigns pass
    stronger values from ``TrainerConfig.aug_*`` (see docs/HPO.md).

    ``mean``/``std`` default to CLIP stats (C-RADIO) for back-compat; pass the
    backbone's ``BackboneInfo.image_mean/std`` (ImageNet for DINOv2/v3) so the
    encoder sees in-distribution inputs.
    """
    return _make_pipeline(
        img_size=img_size,
        train=True,
        mean=mean if mean is not None else CLIP_MEAN,
        std=std if std is not None else CLIP_STD,
        hflip_prob=hflip_prob,
        jitter_prob=jitter_prob,
        jitter_scale=jitter_scale,
        rotation_deg=rotation_deg,
        multiscale_range=multiscale_range,
    )


def get_val_transforms(
    img_size: int = 1024,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Return validation / inference transform callable (no augmentation).

    ``mean``/``std`` default to CLIP stats; pass the backbone's norm to match
    training (see ``get_train_transforms``).
    """
    return _make_pipeline(
        img_size=img_size,
        train=False,
        mean=mean if mean is not None else CLIP_MEAN,
        std=std if std is not None else CLIP_STD,
    )
