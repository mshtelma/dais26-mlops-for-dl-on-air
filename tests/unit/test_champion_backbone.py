"""Tests for `dais26_dentex.config.champion.resolve_backbone_from_source_model`.

The single backbone-agnostic prod champion can hold any architecture, so the
embeddings refresh must reverse-map the `source_dev_model` tag back to the
backbone that produced it. This is the seam that keeps the feature extractor in
sync with the live champion, so test the match / fallback paths hard.
"""

from __future__ import annotations

import pytest

from dais26_dentex.config.champion import resolve_backbone_from_source_model

# Mirrors the shape of `_DETECTOR_NAMES_BY_BACKBONE` in notebooks/00_config.py.
NAMES_BY_BACKBONE = {
    "cradio_v4_so400m": {"model_short": "cradio_detector", "endpoint": "dais26-cradio-detector-dev"},
    "dinov3_vitl16": {"model_short": "dinov3_detector", "endpoint": "dais26-dinov3-detector-dev"},
    "dinov2_base": {"model_short": "dinov2_detector", "endpoint": "dais26-dinov2-detector-dev"},
}
DEFAULT = "cradio_v4_so400m"


@pytest.mark.parametrize(
    ("source_dev_model", "expected"),
    [
        ("mlops_pj.dais26_vfm.cradio_detector", "cradio_v4_so400m"),
        ("mlops_pj.dais26_vfm.dinov3_detector", "dinov3_vitl16"),
        ("mlops_pj.dais26_vfm.dinov2_detector", "dinov2_base"),
        # Bare short name (no catalog.schema) still resolves.
        ("dinov3_detector", "dinov3_vitl16"),
    ],
)
def test_resolves_known_backbone(source_dev_model: str, expected: str) -> None:
    assert resolve_backbone_from_source_model(source_dev_model, NAMES_BY_BACKBONE, DEFAULT) == expected


@pytest.mark.parametrize("source_dev_model", [None, "", "mlops_pj.dais26_vfm.unknown_detector"])
def test_falls_back_to_default(source_dev_model: str | None) -> None:
    # No champion yet, missing tag, or an unrecognized model → static config default.
    assert resolve_backbone_from_source_model(source_dev_model, NAMES_BY_BACKBONE, DEFAULT) == DEFAULT


def test_fallback_default_is_returned_verbatim() -> None:
    # The default need not be a key in the map; it is returned as-is.
    assert resolve_backbone_from_source_model(None, NAMES_BY_BACKBONE, "dinov3_vitl16") == "dinov3_vitl16"
