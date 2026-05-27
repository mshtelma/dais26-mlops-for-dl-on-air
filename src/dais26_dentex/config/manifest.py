"""Typed v2 artifact manifest.

A single ``manifest.json`` replaces the v1 trio of sidecar JSONs
(``backbone_config.json``, ``detection_config.json``, ``label_map.json``).
The advantage is one file to read at load time and one schema to verify;
producer and consumer cannot drift on filename casing, missing keys, or
"which sidecar held this knob."

The v2 contract:

    {
      "version": 2,
      "backbone": {"name": ..., "revision": ..., "summary_dim": ...,
                   "spatial_dim": ..., "patch_size": ...},
      "detector": {"num_classes": ..., "scales": [...],
                   "aspect_ratios": [...], "score_threshold": ...,
                   "nms_iou_threshold": ..., "max_detections": ...,
                   "input_size": ...},
      "label_map": {"0": "Caries", ...},
      "trainer": {"epochs": ..., "lr": ..., ...}   # for provenance only
    }

`load_manifest(path)` raises `IncompatibleArtifactError` on a missing or
mismatched ``version`` field — the loader bumps cleanly when v3 lands.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dais26_dentex.config.constants import ARTIFACT_FORMAT_VERSION


class IncompatibleArtifactError(ValueError):
    """Raised when a manifest's ``version`` is not the one we can load.

    Carries enough context for the operator to know what's needed: the
    artifact's claimed version, the version we expected, and a hint about
    the v1 → v2 migration (``DetectorPyfuncV1``).
    """

    def __init__(self, found: int | None, expected: int, *, hint: str = "") -> None:
        msg = f"Artifact format version {found!r} is not compatible with this loader (expected {expected})."
        if hint:
            msg = f"{msg} {hint}"
        super().__init__(msg)
        self.found = found
        self.expected = expected


@dataclass(frozen=True, slots=True)
class BackboneSpec:
    name: str
    revision: str | None
    summary_dim: int
    spatial_dim: int
    patch_size: int


@dataclass(frozen=True, slots=True)
class DetectorSpec:
    num_classes: int
    scales: list[int]
    aspect_ratios: list[float]
    score_threshold: float = 0.05
    nms_iou_threshold: float = 0.5
    max_detections: int = 100
    input_size: int = 1024


@dataclass(frozen=True, slots=True)
class Manifest:
    """v2 model manifest. Pyfunc reads this from `manifest.json`."""

    backbone: BackboneSpec
    detector: DetectorSpec
    label_map: dict[str, str]
    # Free-form provenance dict; producer logs the trainer params here so a
    # downstream debugger can see "what was the lr / which seed?" without
    # cross-referencing the MLflow run. Excluded from the schema check.
    trainer: dict[str, Any] = field(default_factory=dict)
    version: int = ARTIFACT_FORMAT_VERSION

    def to_json(self) -> str:
        d = asdict(self)
        # Stable ordering: version first so a `head -1` is enough for triage.
        return json.dumps(
            {
                "version": d["version"],
                "backbone": d["backbone"],
                "detector": d["detector"],
                "label_map": d["label_map"],
                "trainer": d["trainer"],
            },
            indent=2,
            sort_keys=False,
        )

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())


def load_manifest(path: str | Path) -> Manifest:
    """Read and validate a v2 manifest from disk.

    Raises:
        IncompatibleArtifactError: ``version`` missing or != current.
        ValueError: required field missing or wrong type.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"manifest.json must be a JSON object; got {type(raw).__name__}")

    found_raw = raw.get("version")
    found: int | None = int(found_raw) if isinstance(found_raw, (int, str)) and str(found_raw).isdigit() else None
    if found != ARTIFACT_FORMAT_VERSION:
        hint = (
            "v1 artifacts are still loadable via "
            "`dais26_dentex.serve.detector_pyfunc.DetectorPyfuncV1`; re-train "
            "to produce v2."
            if found in (1, None)
            else ""
        )
        raise IncompatibleArtifactError(found, ARTIFACT_FORMAT_VERSION, hint=hint)

    try:
        backbone = BackboneSpec(**raw["backbone"])
        detector = DetectorSpec(**raw["detector"])
    except (KeyError, TypeError) as e:
        raise ValueError(f"manifest.json missing required field: {e}") from e

    label_map = raw.get("label_map", {})
    if not isinstance(label_map, dict):
        raise ValueError(f"label_map must be a dict; got {type(label_map).__name__}")

    return Manifest(
        backbone=backbone,
        detector=detector,
        label_map={str(k): str(v) for k, v in label_map.items()},
        trainer=raw.get("trainer", {}) or {},
        version=ARTIFACT_FORMAT_VERSION,  # equals `found` per the check above
    )


__all__ = [
    "BackboneSpec",
    "DetectorSpec",
    "IncompatibleArtifactError",
    "Manifest",
    "load_manifest",
]
