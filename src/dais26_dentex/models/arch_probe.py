"""Architecture-consistency probe + the up-front issue register.

Two things live here:

1. ``KNOWN_ISSUES`` — a hand-curated register of the architectural
   inconsistencies found during the HPO/backbone-customization review. These
   are *static* findings (code-shape problems), each with a severity, the
   source location, why it matters, and the intended fix. They are surfaced
   verbatim by ``notebooks/02a_arch_probe.py`` so the issues are flagged before
   any sweep burns GPU hours chasing a ceiling the architecture imposes.

2. A *live* probe (``probe_detection_model``) that builds one forward+match
   pass and reports the runtime facts behind those issues: per-FPN-level anchor
   counts, how many of a batch's matched positives land on each level, the
   fraction of box-regression targets that exceed the decoder's ``exp`` clamp,
   and the token/grid alignment. The probe takes an already-constructed
   ``DetectionModel`` so it can run against either the real backbone (notebook,
   CUDA) or a tiny fake backbone (unit test, CPU).

Nothing here mutates the model — it is a read-only diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from dais26_dentex.models.detection_head import AnchorGenerator
from dais26_dentex.models.targets import build_targets_for_batch

Severity = Literal["MAJOR", "MEDIUM", "MINOR"]

# Fixed FPN level order — anchors are concatenated in this order by
# ``AnchorGenerator.forward`` and ``RetinaNetHead.forward``, so per-level
# slicing of a flat (N,) anchor/positive tensor must use the same order.
LEVEL_ORDER: tuple[str, ...] = ("p3", "p4", "p5", "p6")


@dataclass(frozen=True, slots=True)
class ArchIssue:
    """One flagged architectural inconsistency."""

    severity: Severity
    title: str
    location: str
    detail: str
    fix: str


# Curated register. Keep ordered by severity (MAJOR first) for rendering.
KNOWN_ISSUES: tuple[ArchIssue, ...] = (
    ArchIssue(
        severity="MAJOR",
        title="Every anchor scale is generated at every FPN level",
        location="models/detection_head.py: AnchorGenerator._anchors_for_level / forward",
        detail=(
            "AnchorGenerator places all len(scales) x len(aspect_ratios) anchors "
            "(default 4x3=12) at EVERY level p3..p6. So p6 (stride 64) gets a 16px "
            "anchor and p3 (stride 8) gets a 128px anchor. Standard FPN assigns one "
            "base size per level (size ~ scale*stride). The current scheme floods the "
            "matcher with geometrically useless anchors, dilutes the positive ratio, "
            "and is a prime suspect for the ~3% mAP@50 ceiling seen across both "
            "backbones."
        ),
        fix=(
            "Assign one base scale per level (e.g. base = 4*stride) with a small set of "
            "size multipliers + the aspect ratios, instead of the full scale list at "
            "every level."
        ),
    ),
    ArchIssue(
        severity="MEDIUM",
        title="Class-agnostic NMS can suppress co-located cross-class detections",
        location="models/detection_head.py: DetectionModel.forward (single nms call)",
        detail=(
            "Inference flattens (anchor, class) and runs one nms() across all classes. "
            "For DENTEX a lesion sits inside its tooth box; a single class-agnostic NMS "
            "can drop the overlapping cross-class detection and cap recall."
        ),
        fix="Use torchvision.ops.batched_nms keyed by predicted label (per-class NMS).",
    ),
    ArchIssue(
        severity="MEDIUM",
        title="Backbone is hard-gated against training (blocks fine-tuning AND LoRA)",
        location="models/backbones.py: load_backbone (requires_grad_(False)+eval); "
        "models/detection_head.py: forward_train/forward wrap backbone in torch.no_grad()",
        detail=(
            "The backbone is frozen at load and the train-time forward wraps it in "
            "torch.no_grad(). The no_grad context blocks gradients even when LoRA params "
            "have requires_grad=True, so the existing use_lora path may be a silent "
            "no-op and full fine-tuning is impossible without surgery."
        ),
        fix=(
            "Gate the no_grad on a 'backbone frozen?' check; load_backbone takes a freeze "
            "flag; builder branches on backbone_mode (frozen/lora/full/partial)."
        ),
    ),
    ArchIssue(
        severity="MEDIUM",
        title="Single optimizer param group applies head LR to the backbone",
        location="train/trainer.py: AdamW(trainable, lr=cfg.lr)",
        detail=(
            "All trainable params share one LR. Once the backbone is trainable it would "
            "get the head LR (1e-3), which destroys pretrained features (catastrophic "
            "forgetting)."
        ),
        fix="Discriminative LRs: backbone param group at cfg.backbone_lr (~1e-5), head/FPN at cfg.lr.",
    ),
    ArchIssue(
        severity="MINOR",
        title="Backbone output dims are hardcoded, not read from model config",
        location="models/backbones.py: BackboneInfo literals (e.g. C-RADIO 1152)",
        detail=(
            "summary_dim/spatial_dim are literals per backbone. A model/config change "
            "silently mis-sizes the FPN; caught only by the FPNAdapter shape assert at "
            "runtime."
        ),
        fix="Read hidden size from model.config where available; keep literals as fallback.",
    ),
    ArchIssue(
        severity="MINOR",
        title="encode/decode delta clamp asymmetry",
        location="models/targets.py: encode_boxes_xyxy_to_deltas vs "
        "models/detection_head.py: decode_boxes (exp clamp max=4.0)",
        detail=(
            "Box targets encode log(gw/aw) with no clamp, but decode clamps exp(dw/dh) "
            "at 4.0. For large-gt/small-anchor pairs the target delta is unreachable by "
            "the decoder, injecting an irreducible localization error."
        ),
        fix="Clamp encode deltas to match the decode bound, or widen/remove the decode clamp.",
    ),
)


def _grid_shapes(fmap: dict[str, torch.Tensor]) -> dict[str, tuple[int, int]]:
    """Per-level (H, W) of the FPN feature maps, in canonical level order."""
    return {k: (int(fmap[k].shape[-2]), int(fmap[k].shape[-1])) for k in LEVEL_ORDER if k in fmap}


def level_anchor_report(
    scales: list[int],
    aspect_ratios: list[float],
    grid_shapes: dict[str, tuple[int, int]],
    *,
    layout: str = "absolute",
    anchors_per_cell: int | None = None,
) -> dict[str, Any]:
    """Per-level anchor counts + the over-generation flag.

    ``layout`` is the :class:`AnchorGenerator` layout. In ``absolute`` mode the
    same scale list lands on every level (the over-generation smell); in
    ``per_level`` mode each level is sized by its stride so the smell is gone.
    ``anchors_per_cell`` overrides the count (the generator computes it from
    octaves x ratios in per_level mode); when omitted it falls back to the
    legacy ``len(scales) * len(aspect_ratios)``.

    Returns ``{levels: {p3: {grid, anchors, stride}, ...}, anchors_per_cell,
    total_anchors, all_scales_every_level, layout}``.
    """
    apc = anchors_per_cell if anchors_per_cell is not None else len(scales) * len(aspect_ratios)
    levels: dict[str, dict[str, Any]] = {}
    total = 0
    for key, (h, w) in grid_shapes.items():
        n = h * w * apc
        total += n
        levels[key] = {
            "grid": (h, w),
            "stride": AnchorGenerator.LEVEL_STRIDES.get(key),
            "anchors": n,
        }
    return {
        "levels": levels,
        "anchors_per_cell": apc,
        "total_anchors": total,
        # The smell only exists in the legacy absolute layout: the same full
        # scale list lands on every level. Resolved by the per_level layout.
        "all_scales_every_level": layout == "absolute" and len(scales) > 1 and len(grid_shapes) > 1,
        "layout": layout,
        "scales": list(scales),
        "aspect_ratios": list(aspect_ratios),
    }


def positive_level_distribution(
    grid_shapes: dict[str, tuple[int, int]],
    anchors_per_cell: int,
    fg_mask: torch.Tensor,
) -> dict[str, int]:
    """Count matched-positive anchors that fall on each FPN level.

    ``fg_mask`` is (B, N) or (N,); anchors are concatenated in ``LEVEL_ORDER``.
    """
    flat = fg_mask.reshape(-1, fg_mask.shape[-1]) if fg_mask.ndim == 2 else fg_mask.unsqueeze(0)
    per_level: dict[str, int] = {}
    start = 0
    for key, (h, w) in grid_shapes.items():
        n = h * w * anchors_per_cell
        per_level[key] = int(flat[:, start : start + n].sum().item())
        start += n
    return per_level


def delta_overflow_fraction(
    box_targets: torch.Tensor,
    fg_mask: torch.Tensor,
    clamp: float = 4.0,
) -> float:
    """Fraction of matched positives whose dw/dh target exceeds the decode clamp."""
    if box_targets.ndim == 2:
        box_targets = box_targets.unsqueeze(0)
        fg_mask = fg_mask.unsqueeze(0)
    fg = fg_mask.bool()
    if int(fg.sum().item()) == 0:
        return 0.0
    deltas = box_targets[fg]  # (P, 4) = (dx, dy, dw, dh)
    over = (deltas[:, 2:].abs() > clamp).any(dim=1)
    return float(over.float().mean().item())


@torch.no_grad()
def probe_detection_model(
    model: Any,
    images: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    *,
    decode_clamp: float = 4.0,
) -> dict[str, Any]:
    """Run one forward + match pass and return a structured consistency report.

    Args:
        model: a built ``DetectionModel`` (real or fake backbone).
        images: (B, 3, H, W) batch on the model's device.
        targets: per-image dicts with ``boxes`` (COCO xywh) + ``labels``.
        num_classes: detector class count.
        decode_clamp: the exp clamp used by ``decode_boxes`` (for overflow stat).
    """
    device = images.device
    _, _, h, w = images.shape
    ph, pw = h // model.patch_size, w // model.patch_size

    _, spatial = model.backbone(images)
    fmap = model.fpn(spatial, spatial_shape=(ph, pw))
    anchors = model.anchor_gen(fmap)

    grid_shapes = _grid_shapes(fmap)
    ag = model.anchor_gen
    anchor_rep = level_anchor_report(
        ag.scales,
        ag.aspect_ratios,
        grid_shapes,
        layout=getattr(ag, "layout", "absolute"),
        anchors_per_cell=getattr(ag, "num_anchors_per_cell", None),
    )

    _cls_t, box_t, fg_mask, _ignore = build_targets_for_batch(
        anchors.to(device), [{k: v.to(device) for k, v in t.items()} for t in targets], num_classes
    )

    total_positives = int(fg_mask.sum().item())
    return {
        "image_size": (int(h), int(w)),
        "patch_size": int(model.patch_size),
        "token_grid": (int(ph), int(pw)),
        "grid_shapes": {k: list(v) for k, v in grid_shapes.items()},
        "anchors": anchor_rep,
        "anchors_emitted": int(anchors.shape[0]),
        "head_anchor_count_matches": int(anchors.shape[0]) == anchor_rep["total_anchors"],
        "positives_total": total_positives,
        "positives_per_level": positive_level_distribution(grid_shapes, anchor_rep["anchors_per_cell"], fg_mask),
        "positive_fraction": (total_positives / max(anchors.shape[0] * images.shape[0], 1)),
        "delta_overflow_fraction": delta_overflow_fraction(box_t, fg_mask, clamp=decode_clamp),
        "nms_mode": (
            "per-class (torchvision.batched_nms keyed by label)"
            if getattr(model, "nms_per_class", False)
            else "class-agnostic (single nms across all classes)"
        ),
    }


def render_report(report: dict[str, Any], issues: tuple[ArchIssue, ...] = KNOWN_ISSUES) -> str:
    """Human-readable rendering of the live probe + the static issue register."""
    lines: list[str] = []
    lines.append("=== Live architecture probe ===")
    lines.append(f"image_size={report['image_size']} patch_size={report['patch_size']} token_grid={report['token_grid']}")
    a = report["anchors"]
    lines.append(
        f"anchors: {report['anchors_emitted']} emitted, {a['anchors_per_cell']}/cell, "
        f"all-scales-every-level={a['all_scales_every_level']} "
        f"(head_count_matches={report['head_anchor_count_matches']})"
    )
    for key, lv in a["levels"].items():
        pos = report["positives_per_level"].get(key, 0)
        lines.append(f"  {key}: grid={lv['grid']} stride={lv['stride']} anchors={lv['anchors']} positives={pos}")
    lines.append(
        f"positives_total={report['positives_total']} "
        f"positive_fraction={report['positive_fraction']:.6f} "
        f"delta_overflow_fraction={report['delta_overflow_fraction']:.4f}"
    )
    lines.append(f"nms_mode={report['nms_mode']}")
    lines.append("")
    lines.append("=== Flagged architectural issues (up front) ===")
    for i in issues:
        lines.append(f"[{i.severity}] {i.title}")
        lines.append(f"    where: {i.location}")
        lines.append(f"    why:   {i.detail}")
        lines.append(f"    fix:   {i.fix}")
    return "\n".join(lines)


__all__ = [
    "KNOWN_ISSUES",
    "LEVEL_ORDER",
    "ArchIssue",
    "Severity",
    "delta_overflow_fraction",
    "level_anchor_report",
    "positive_level_distribution",
    "probe_detection_model",
    "render_report",
]
