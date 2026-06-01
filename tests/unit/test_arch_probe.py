"""Tests for the architecture-consistency probe.

Uses a tiny fake backbone (no real ViT) so the probe runs on CPU in CI. The
probe is read-only; these tests pin the runtime facts it reports + the static
issue register.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dais26_dentex.models.arch_probe import (
    KNOWN_ISSUES,
    LEVEL_ORDER,
    delta_overflow_fraction,
    level_anchor_report,
    positive_level_distribution,
    probe_detection_model,
    render_report,
)
from dais26_dentex.models.detection_head import DetectionModel


class _FakeBackbone(nn.Module):
    """Returns (summary, spatial) shaped like a patch-16 ViT."""

    def __init__(self, spatial_dim: int = 64) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        ph, pw = h // 16, w // 16
        return torch.randn(b, self.spatial_dim), torch.randn(b, ph * pw, self.spatial_dim)


def _build_model(spatial_dim: int = 64) -> DetectionModel:
    return DetectionModel(
        backbone=_FakeBackbone(spatial_dim),
        spatial_dim=spatial_dim,
        num_classes=4,
        scales=[16, 32, 64, 128],
        aspect_ratios=[0.5, 1.0, 2.0],
        patch_size=16,
    )


def test_known_issues_register_is_well_formed() -> None:
    assert len(KNOWN_ISSUES) >= 4
    sev = {"MAJOR", "MEDIUM", "MINOR"}
    assert all(i.severity in sev for i in KNOWN_ISSUES)
    # The anchor over-generation issue is the headline MAJOR finding.
    assert any(i.severity == "MAJOR" and "every" in i.title.lower() for i in KNOWN_ISSUES)
    assert all(i.location and i.detail and i.fix for i in KNOWN_ISSUES)


def test_level_anchor_report_flags_over_generation() -> None:
    grids = {"p3": (32, 32), "p4": (16, 16), "p5": (8, 8), "p6": (4, 4)}
    rep = level_anchor_report([16, 32, 64, 128], [0.5, 1.0, 2.0], grids)
    assert rep["anchors_per_cell"] == 12
    assert rep["all_scales_every_level"] is True
    expected = (32 * 32 + 16 * 16 + 8 * 8 + 4 * 4) * 12
    assert rep["total_anchors"] == expected
    assert set(rep["levels"]) == set(grids)


def test_positive_level_distribution_sums_to_total() -> None:
    grids = {"p3": (2, 2), "p4": (1, 1)}
    apc = 3
    n = (2 * 2 + 1 * 1) * apc  # 15
    fg = torch.zeros(2, n, dtype=torch.bool)
    fg[0, 0] = True  # p3
    fg[0, 1] = True  # p3
    fg[1, 13] = True  # p4 (last 3 entries are p4)
    dist = positive_level_distribution(grids, apc, fg)
    assert dist["p3"] == 2
    assert dist["p4"] == 1
    assert sum(dist.values()) == int(fg.sum().item())


def test_delta_overflow_fraction() -> None:
    box_t = torch.zeros(1, 3, 4)
    fg = torch.zeros(1, 3, dtype=torch.bool)
    fg[0, :2] = True
    box_t[0, 0, 2] = 9.0  # dw overflow on a positive
    box_t[0, 1, 3] = 0.5  # within clamp
    # 1 of 2 positives overflows.
    assert delta_overflow_fraction(box_t, fg, clamp=4.0) == 0.5
    # No positives -> 0.0.
    assert delta_overflow_fraction(box_t, torch.zeros(1, 3, dtype=torch.bool)) == 0.0


def test_probe_detection_model_end_to_end() -> None:
    model = _build_model()
    model.eval()
    images = torch.randn(2, 3, 256, 256)
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 40.0, 30.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)},
    ]
    report = probe_detection_model(model, images, targets, num_classes=4)

    # Token grid for 256px / patch16.
    assert report["token_grid"] == (16, 16)
    # Head emits exactly as many anchors as the level report counts.
    assert report["head_anchor_count_matches"] is True
    assert report["anchors_emitted"] == report["anchors"]["total_anchors"]
    # The single GT must be matched (force-best-anchor-per-gt guarantees >=1).
    assert report["positives_total"] >= 1
    assert sum(report["positives_per_level"].values()) == report["positives_total"]
    assert 0.0 <= report["delta_overflow_fraction"] <= 1.0
    assert "class-agnostic" in report["nms_mode"]

    text = render_report(report)
    assert "Live architecture probe" in text
    assert "Flagged architectural issues" in text


def test_level_order_constant() -> None:
    assert LEVEL_ORDER == ("p3", "p4", "p5", "p6")
