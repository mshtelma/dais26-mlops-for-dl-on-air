"""Tests for the v2 artifact contract: `Manifest` ↔ `load_manifest`.

The v2 cut-over collapses three v1 sidecar JSONs into a single
`manifest.json`. These tests pin the producer ↔ consumer round-trip and the
version-mismatch failure mode so a future schema bump cannot silently
load against an incompatible loader.

Also covers `serving_pip_requirements` — the second producer-side
single-source-of-truth introduced in Phase 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dais26_dentex.config.constants import ARTIFACT_FORMAT_VERSION
from dais26_dentex.config.manifest import (
    BackboneSpec,
    DetectorSpec,
    IncompatibleArtifactError,
    Manifest,
    load_manifest,
)
from dais26_dentex.platform.mlflow_io import (
    assert_serving_reqs_match_pyproject,
    serving_pip_requirements,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_manifest() -> Manifest:
    """Realistic v2 manifest matching what the trainer writes today."""
    return Manifest(
        backbone=BackboneSpec(
            name="cradio_v4_so400m",
            revision="abc123",
            summary_dim=1152,
            spatial_dim=1152,
            patch_size=16,
        ),
        detector=DetectorSpec(
            num_classes=4,
            scales=[32, 64, 128, 256],
            aspect_ratios=[0.5, 1.0, 2.0],
            score_threshold=0.05,
            nms_iou_threshold=0.5,
            max_detections=100,
            input_size=1024,
        ),
        label_map={"0": "Caries", "1": "Periapical", "2": "Deep Caries", "3": "Impacted"},
        trainer={"epochs": 10, "lr": 1e-3, "best_epoch": 7, "best_val_mAP_50": 0.42},
    )


# ----------------------------------------------------------------------
# Round-trip
# ----------------------------------------------------------------------


def test_manifest_roundtrip_preserves_all_fields(tmp_path: Path) -> None:
    """`Manifest.write` then `load_manifest` returns an equivalent object."""
    src = _make_manifest()
    path = tmp_path / "manifest.json"
    src.write(path)

    loaded = load_manifest(path)

    assert loaded.version == ARTIFACT_FORMAT_VERSION
    assert loaded.backbone == src.backbone
    assert loaded.detector == src.detector
    assert loaded.label_map == src.label_map
    assert loaded.trainer == src.trainer


def test_manifest_json_has_version_first(tmp_path: Path) -> None:
    """`head -1` should be enough to triage an artifact in production —
    keys must be ordered with `version` first."""
    path = tmp_path / "manifest.json"
    _make_manifest().write(path)

    text = path.read_text()
    # The first non-`{` key in the JSON object is `version`.
    obj = json.loads(text)
    assert next(iter(obj.keys())) == "version"


# ----------------------------------------------------------------------
# Version mismatch
# ----------------------------------------------------------------------


def test_load_v1_manifest_raises_with_migration_hint(tmp_path: Path) -> None:
    """v1-shaped manifest → typed error mentioning DetectorPyfuncV1."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 1, "anything": "else"}))

    with pytest.raises(IncompatibleArtifactError) as exc_info:
        load_manifest(path)

    err = exc_info.value
    assert err.found == 1
    assert err.expected == ARTIFACT_FORMAT_VERSION
    assert "DetectorPyfuncV1" in str(err)


def test_load_missing_version_raises(tmp_path: Path) -> None:
    """A bare JSON object with no `version` is treated as incompatible."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"backbone": {}, "detector": {}}))

    with pytest.raises(IncompatibleArtifactError) as exc_info:
        load_manifest(path)

    assert exc_info.value.found is None


def test_load_future_version_raises_without_v1_hint(tmp_path: Path) -> None:
    """A v3 manifest must not advertise the v1 migration path."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 99}))

    with pytest.raises(IncompatibleArtifactError) as exc_info:
        load_manifest(path)

    err = exc_info.value
    assert err.found == 99
    assert "DetectorPyfuncV1" not in str(err)


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------


def test_load_missing_backbone_raises(tmp_path: Path) -> None:
    """Required field missing → `ValueError` with the field name."""
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": ARTIFACT_FORMAT_VERSION,
                "detector": {
                    "num_classes": 4,
                    "scales": [32],
                    "aspect_ratios": [1.0],
                },
                "label_map": {},
            }
        )
    )

    with pytest.raises(ValueError, match="missing required field"):
        load_manifest(path)


def test_load_non_dict_label_map_raises(tmp_path: Path) -> None:
    """`label_map` must be a JSON object, not a list."""
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": ARTIFACT_FORMAT_VERSION,
                "backbone": {
                    "name": "x",
                    "revision": None,
                    "summary_dim": 1,
                    "spatial_dim": 1,
                    "patch_size": 1,
                },
                "detector": {
                    "num_classes": 1,
                    "scales": [1],
                    "aspect_ratios": [1.0],
                },
                "label_map": ["wrong", "shape"],
            }
        )
    )

    with pytest.raises(ValueError, match="label_map must be a dict"):
        load_manifest(path)


def test_load_non_object_root_raises(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([1, 2, 3]))

    with pytest.raises(ValueError, match="must be a JSON object"):
        load_manifest(path)


def test_detector_spec_defaults_match_postprocess() -> None:
    """`DetectorSpec` defaults must match `PostprocessConfig` defaults so an
    artifact missing optional fields still loads with the legacy behavior."""
    from dais26_dentex.serve.postprocess import PostprocessConfig

    d = DetectorSpec(num_classes=4, scales=[32], aspect_ratios=[1.0])
    p = PostprocessConfig()
    assert d.score_threshold == p.score_threshold
    assert d.nms_iou_threshold == p.nms_iou_threshold
    assert d.max_detections == p.max_detections


# ----------------------------------------------------------------------
# Serving pip_requirements (single source of truth)
# ----------------------------------------------------------------------


def test_serving_pip_requirements_returns_non_empty_list() -> None:
    deps = serving_pip_requirements()
    assert isinstance(deps, list)
    assert len(deps) > 0
    assert all(isinstance(d, str) for d in deps)


def test_serving_pip_requirements_includes_runtime_essentials() -> None:
    """Sanity-check: the deps actually used at predict-time must be present.
    If this list shrinks the pyfunc fails to import at serve-time."""
    deps = serving_pip_requirements()
    names = {d.split(">")[0].split("=")[0].split("<")[0].strip().lower() for d in deps}
    for required in {"torch", "mlflow", "pillow", "transformers"}:
        assert required in names, f"serving deps missing {required}; got {sorted(names)}"


def test_serving_pip_requirements_unknown_profile_raises() -> None:
    with pytest.raises(KeyError, match="does not define profile"):
        serving_pip_requirements(profile="nonexistent")


def test_assert_serving_reqs_match_pyproject_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """CI guard runs without raising on the configured profile."""
    assert_serving_reqs_match_pyproject()
    captured = capsys.readouterr()
    assert "serving-deps:detector" in captured.err
