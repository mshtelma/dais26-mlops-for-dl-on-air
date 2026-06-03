from __future__ import annotations

import contextlib
import json
import math
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

DEFAULT_ANCHOR_SCALES: list[int] = [16, 32, 64, 128]
DEFAULT_ASPECT_RATIOS: list[float] = [0.5, 1.0, 2.0]
DEFAULT_NUM_CLASSES: int = 4

# Per-level anchor layout defaults (standard RetinaNet octave scales). In
# `per_level` mode each FPN level's base anchor size is `stride * base_scale`,
# then multiplied by these octaves and the aspect ratios → 3*3=9 anchors/cell,
# uniform across levels but correctly sized per stride. See
# `arch_probe.KNOWN_ISSUES` (the MAJOR anchor-over-generation finding).
DEFAULT_ANCHOR_OCTAVES: list[float] = [2.0**0, 2.0 ** (1.0 / 3.0), 2.0 ** (2.0 / 3.0)]
DEFAULT_ANCHOR_BASE_SCALE: float = 4.0
ANCHOR_LAYOUTS: tuple[str, ...] = ("absolute", "per_level")

# Shared bound for the box-regression log-space width/height delta. `decode_boxes`
# clamps `exp(dw/dh)` at this value; `targets.encode_boxes_xyxy_to_deltas` clamps
# the encoded target to match so a large-gt/small-anchor pair stays reachable
# (encode/decode symmetry — see the MINOR clamp issue in arch_probe).
BOX_DELTA_LOG_CLAMP: float = 4.0


def _init_classification_bias(num_anchors: int, num_classes: int, prior: float = 0.01) -> torch.Tensor:
    """RetinaNet classification subnet bias init for class imbalance (focal loss prior)."""
    bias_value = -math.log((1 - prior) / prior)
    return torch.full((num_anchors * num_classes,), bias_value)


class RetinaNetHead(nn.Module):
    """RetinaNet classification + regression head shared across FPN levels.

    Architecture per level:
        - num_convs x (3x3 conv + GroupNorm + GELU) shared subnet for classification
        - num_convs x (3x3 conv + GroupNorm + GELU) shared subnet for regression
        - Classification output: (B, H*W*A, num_classes), sigmoid applied during loss / decoding
        - Regression output: (B, H*W*A, 4) in (dx, dy, dw, dh) deltas

    A (num_anchors) = scales x aspect_ratios; default 4 scales x 3 ratios but the head only
    needs num_anchors, not the actual scales (those live in the AnchorGenerator).
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = DEFAULT_NUM_CLASSES,
        num_anchors: int = 12,  # 4 scales x 3 ratios
        num_convs: int = 4,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        def _subnet() -> nn.Sequential:
            layers: list[nn.Module] = []
            for _ in range(num_convs):
                layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
                layers.append(nn.GroupNorm(num_groups, in_channels))
                layers.append(nn.GELU())
            return nn.Sequential(*layers)

        self.cls_subnet = _subnet()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        self.box_subnet = _subnet()
        self.box_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        # Focal-loss prior init for classification bias
        with torch.no_grad():
            self.cls_logits.bias.data = _init_classification_bias(num_anchors, num_classes)

    def forward(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        features: {'p3': (B, C, H3, W3), 'p4': ..., 'p5': ..., 'p6': ...}
        Returns:
            cls_logits: (B, total_anchors, num_classes)
            box_regression: (B, total_anchors, 4)
        """
        all_cls: list[torch.Tensor] = []
        all_box: list[torch.Tensor] = []
        # Iterate in fixed order
        for key in ("p3", "p4", "p5", "p6"):
            if key not in features:
                continue
            x = features[key]
            b = x.shape[0]
            # Classification path
            c = self.cls_logits(self.cls_subnet(x))  # (B, A*C, H, W)
            c = c.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)
            all_cls.append(c)
            # Regression path
            r = self.box_pred(self.box_subnet(x))  # (B, A*4, H, W)
            r = r.permute(0, 2, 3, 1).reshape(b, -1, 4)
            all_box.append(r)
        cls_out = torch.cat(all_cls, dim=1) if all_cls else torch.empty(0)
        box_out = torch.cat(all_box, dim=1) if all_box else torch.empty(0)
        return cls_out, box_out


class AnchorGenerator(nn.Module):
    """Generates anchor boxes per FPN level.

    Each FPN level has a stride: p3=8, p4=16, p5=32, p6=64. For each grid cell,
    `num_anchors_per_cell` anchors are centered on the cell. Two layouts:

    - ``absolute`` (legacy): the same ``scales x aspect_ratios`` sizes are placed
      at *every* level. This over-generates geometrically useless anchors (a
      128px anchor on stride-8 p3, a 16px anchor on stride-64 p6) and was the
      prime suspect for the mAP plateau (see ``arch_probe.KNOWN_ISSUES``).
    - ``per_level`` (standard RetinaNet): each level's base size is
      ``stride * base_scale``, multiplied by ``octaves`` and ``aspect_ratios``.
      The per-cell count stays uniform (so the shared head is valid) but the
      absolute sizes scale with the level — p3 gets small anchors, p6 large.
    """

    LEVEL_STRIDES: ClassVar[dict[str, int]] = {"p3": 8, "p4": 16, "p5": 32, "p6": 64}

    def __init__(
        self,
        scales: list[int] | None = None,
        aspect_ratios: list[float] | None = None,
        *,
        layout: str = "absolute",
        base_scale: float = DEFAULT_ANCHOR_BASE_SCALE,
        octaves: list[float] | None = None,
    ) -> None:
        super().__init__()
        if layout not in ANCHOR_LAYOUTS:
            raise ValueError(f"anchor layout {layout!r} not in {ANCHOR_LAYOUTS}")
        self.layout = layout
        self.scales = scales if scales is not None else DEFAULT_ANCHOR_SCALES
        self.aspect_ratios = aspect_ratios if aspect_ratios is not None else DEFAULT_ASPECT_RATIOS
        self.base_scale = float(base_scale)
        self.octaves = list(octaves) if octaves is not None else list(DEFAULT_ANCHOR_OCTAVES)

    @property
    def num_anchors_per_cell(self) -> int:
        """Anchors generated per grid cell — uniform across levels in both layouts."""
        if self.layout == "per_level":
            return len(self.octaves) * len(self.aspect_ratios)
        return len(self.scales) * len(self.aspect_ratios)

    def _sizes_for_stride(self, stride: int) -> list[float]:
        """Base anchor sizes (pre aspect-ratio) for one level's stride."""
        if self.layout == "per_level":
            base = stride * self.base_scale
            return [base * o for o in self.octaves]
        return [float(s) for s in self.scales]

    def _anchors_for_level(self, stride: int, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
        """Generate (grid_h * grid_w * A, 4) anchors in xyxy pixel coords."""
        ys = torch.arange(grid_h, device=device) * stride + stride / 2
        xs = torch.arange(grid_w, device=device) * stride + stride / 2
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (H*W, 2)

        anchor_sizes: list[tuple[float, float]] = []
        for s in self._sizes_for_stride(stride):
            for r in self.aspect_ratios:
                w = float(s) * math.sqrt(r)
                h = float(s) / math.sqrt(r)
                anchor_sizes.append((w, h))
        sizes = torch.tensor(anchor_sizes, device=device)  # (A, 2)
        a = sizes.shape[0]

        # Broadcast: (H*W, A, 4)
        centers_exp = centers.unsqueeze(1).expand(-1, a, -1)  # (H*W, A, 2)
        sizes_exp = sizes.unsqueeze(0).expand(centers.shape[0], -1, -1)  # (H*W, A, 2)
        x1 = centers_exp[..., 0] - sizes_exp[..., 0] / 2
        y1 = centers_exp[..., 1] - sizes_exp[..., 1] / 2
        x2 = centers_exp[..., 0] + sizes_exp[..., 0] / 2
        y2 = centers_exp[..., 1] + sizes_exp[..., 1] / 2
        return torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns (total_anchors, 4) in xyxy pixel coords on the feature device."""
        device = next(iter(features.values())).device
        all_anchors: list[torch.Tensor] = []
        for key in ("p3", "p4", "p5", "p6"):
            if key not in features:
                continue
            _, _, h, w = features[key].shape
            stride = self.LEVEL_STRIDES[key]
            all_anchors.append(self._anchors_for_level(stride, h, w, device))
        return torch.cat(all_anchors, dim=0)


def focal_loss(
    cls_logits: torch.Tensor,
    cls_targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss for classification.

    cls_logits: (B, N, C) raw logits.
    cls_targets: (B, N, C) one-hot targets in {0, 1}.
    Returns: scalar loss (mean over positives + small negative weight).
    """
    p = torch.sigmoid(cls_logits)
    ce = functional.binary_cross_entropy_with_logits(cls_logits, cls_targets, reduction="none")
    p_t = p * cls_targets + (1 - p) * (1 - cls_targets)
    alpha_t = alpha * cls_targets + (1 - alpha) * (1 - cls_targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    return loss.sum() / max(cls_targets.sum().item(), 1.0)


def decode_boxes(box_pred: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Decode regression deltas relative to anchors.

    box_pred: (N, 4) deltas (dx, dy, dw, dh).
    anchors: (N, 4) xyxy.
    Returns: (N, 4) xyxy decoded boxes.
    """
    a_x1, a_y1, a_x2, a_y2 = anchors.unbind(-1)
    a_w = a_x2 - a_x1
    a_h = a_y2 - a_y1
    a_cx = a_x1 + a_w / 2
    a_cy = a_y1 + a_h / 2

    dx, dy, dw, dh = box_pred.unbind(-1)
    cx = a_cx + dx * a_w
    cy = a_cy + dy * a_h
    w = a_w * torch.exp(dw.clamp(max=BOX_DELTA_LOG_CLAMP))
    h = a_h * torch.exp(dh.clamp(max=BOX_DELTA_LOG_CLAMP))
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


class DetectionModel(nn.Module):
    """Full detection model: frozen backbone + FPN adapter + RetinaNet head + anchors.

    Forward returns COCO-style dict with keys 'boxes', 'scores', 'labels' AFTER NMS.
    Loss-computing forward (train mode) returns (cls_logits, box_regression, anchors).
    """

    def __init__(
        self,
        backbone: nn.Module,
        spatial_dim: int,
        num_classes: int = DEFAULT_NUM_CLASSES,
        scales: list[int] | None = None,
        aspect_ratios: list[float] | None = None,
        patch_size: int = 16,
        nms_iou_threshold: float = 0.5,
        score_threshold: float = 0.05,
        max_detections: int = 100,
        *,
        anchor_layout: str = "absolute",
        anchor_base_scale: float = DEFAULT_ANCHOR_BASE_SCALE,
        anchor_octaves: list[float] | None = None,
        nms_per_class: bool = False,
    ) -> None:
        super().__init__()
        from dais26_dentex.models.adapters import FPNAdapter

        self.backbone = backbone
        self.fpn = FPNAdapter(in_channels=spatial_dim, out_channels=256)
        scales_eff = scales if scales is not None else DEFAULT_ANCHOR_SCALES
        ratios_eff = aspect_ratios if aspect_ratios is not None else DEFAULT_ASPECT_RATIOS
        # Build the anchor generator first so the head's per-cell anchor count is
        # derived from the chosen layout (per_level → octaves x ratios, absolute
        # → scales x ratios) instead of assuming the absolute formula.
        self.anchor_gen = AnchorGenerator(
            scales=scales_eff,
            aspect_ratios=ratios_eff,
            layout=anchor_layout,
            base_scale=anchor_base_scale,
            octaves=anchor_octaves,
        )
        num_anchors = self.anchor_gen.num_anchors_per_cell
        self.head = RetinaNetHead(in_channels=256, num_classes=num_classes, num_anchors=num_anchors)
        self.patch_size = patch_size
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        # When True, NMS is run per predicted class (torchvision.batched_nms) so a
        # lesion box co-located inside its tooth box is not suppressed by a
        # higher-scoring cross-class detection. See the MEDIUM NMS issue.
        self.nms_per_class = nms_per_class
        # Whether the backbone participates in autograd. The builder sets the
        # backbone's requires_grad (frozen / lora / partial / full) BEFORE
        # constructing this model, so we can snapshot it once here. When the
        # backbone is trainable, `forward_train` must NOT wrap it in
        # torch.no_grad() or gradients never reach the encoder (the bug that
        # silently no-op'd the old LoRA path).
        self.backbone_frozen: bool = not any(p.requires_grad for p in backbone.parameters())

    def forward(self, images: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Inference forward. images: (B, 3, H, W).

        Returns dict {'boxes': List[Tensor], 'scores': List[Tensor], 'labels': List[Tensor]}
        (one entry per batch image; lists of length B).
        """
        from torchvision.ops import batched_nms, nms

        b, _, h, w = images.shape
        ph = h // self.patch_size
        pw = w // self.patch_size

        # Backbone forward: (summary, spatial_features)
        with torch.no_grad():
            _, spatial = self.backbone(images)
        # FPN
        fmap = self.fpn(spatial, spatial_shape=(ph, pw))
        # Head
        cls_logits, box_reg = self.head(fmap)
        anchors = self.anchor_gen(fmap)

        boxes_out: list[torch.Tensor] = []
        scores_out: list[torch.Tensor] = []
        labels_out: list[torch.Tensor] = []
        scores = torch.sigmoid(cls_logits)
        for i in range(b):
            decoded = decode_boxes(box_reg[i], anchors)
            # Flatten over (anchor, class)
            n, c = scores[i].shape
            flat_scores = scores[i].reshape(-1)
            flat_labels = torch.arange(c, device=images.device).repeat(n)
            flat_boxes = decoded.repeat_interleave(c, dim=0)
            keep = flat_scores > self.score_threshold
            flat_scores = flat_scores[keep]
            flat_labels = flat_labels[keep]
            flat_boxes = flat_boxes[keep]
            if flat_boxes.numel() == 0:
                boxes_out.append(flat_boxes)
                scores_out.append(flat_scores)
                labels_out.append(flat_labels)
                continue
            # Both nms and batched_nms return indices in decreasing score order,
            # so a plain truncation yields the top-`max_detections` detections.
            if self.nms_per_class:
                keep_idx = batched_nms(flat_boxes, flat_scores, flat_labels, self.nms_iou_threshold)
            else:
                keep_idx = nms(flat_boxes, flat_scores, self.nms_iou_threshold)
            keep_idx = keep_idx[: self.max_detections]
            boxes_out.append(flat_boxes[keep_idx])
            scores_out.append(flat_scores[keep_idx])
            labels_out.append(flat_labels[keep_idx])
        return {"boxes": boxes_out, "scores": scores_out, "labels": labels_out}

    def forward_train(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training-mode forward returns raw logits + anchors for loss computation."""
        _, _, h, w = images.shape
        ph = h // self.patch_size
        pw = w // self.patch_size
        # Only suppress backbone grads when the encoder is frozen; otherwise the
        # encoder is being fine-tuned (full/partial/lora) and needs autograd.
        backbone_ctx = torch.no_grad() if self.backbone_frozen else contextlib.nullcontext()
        with backbone_ctx:
            _, spatial = self.backbone(images)
        fmap = self.fpn(spatial, spatial_shape=(ph, pw))
        cls_logits, box_reg = self.head(fmap)
        anchors = self.anchor_gen(fmap)
        return cls_logits, box_reg, anchors

    @classmethod
    def from_mlflow(cls, model_uri: str) -> DetectionModel:
        """Load a serialized DetectionModel from MLflow UC registry.

        Note: this returns the inner DetectionModel; the MLflow pyfunc wrapper lives in
        src/dais26_dentex/serve/detector_pyfunc.py.
        """
        import mlflow

        loaded = mlflow.pytorch.load_model(model_uri)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded


def calibrate_anchors(
    coco_annotations_path: str,
    target_levels: list[float] | None = None,
) -> list[int]:
    """Data-driven anchor scale calibration from DENTEX bbox size distribution.

    Reads COCO annotations, computes sqrt(w*h) for each bbox, returns 4 anchor scales near:
      [p10/2, p50, p90, p90*2]
    Default target levels (the percentiles to base scales on): [10, 50, 90].

    Returns: sorted list of 4 integer anchor scales.
    """
    target_levels = target_levels or [10, 50, 90]
    with open(coco_annotations_path) as f:
        coco = json.load(f)
    sizes: list[float] = []
    for ann in coco.get("annotations", []):
        w, h = ann["bbox"][2], ann["bbox"][3]
        if w > 0 and h > 0:
            sizes.append(math.sqrt(w * h))
    if not sizes:
        return DEFAULT_ANCHOR_SCALES

    p = np.percentile(np.asarray(sizes), target_levels)
    # 4 scales: p10/2 (small), p50, p90, p90*2 (large)
    raw = [p[0] / 2, p[1], p[2], p[2] * 2]
    # Round to int and ensure strictly increasing
    rounded = sorted({max(4, round(x)) for x in raw})
    # Pad if dedup collapsed
    while len(rounded) < 4:
        rounded.append(rounded[-1] * 2)
    return rounded[:4]


def calibrate_aspect_ratios(coco_annotations_path: str) -> list[float]:
    """Optional: data-driven aspect ratio buckets.

    Returns aspect ratios near 25/50/75 percentile of bbox w/h ratios, clamped to [0.25, 4.0].
    """
    with open(coco_annotations_path) as f:
        coco = json.load(f)
    ratios: list[float] = []
    for ann in coco.get("annotations", []):
        w, h = ann["bbox"][2], ann["bbox"][3]
        if w > 0 and h > 0:
            ratios.append(w / h)
    if not ratios:
        return DEFAULT_ASPECT_RATIOS
    p = np.percentile(np.asarray(ratios), [25, 50, 75])
    rounded = sorted({float(max(0.25, min(4.0, round(r, 2)))) for r in p.tolist()})
    while len(rounded) < 3:
        rounded.append(1.0)
    return rounded[:3]
