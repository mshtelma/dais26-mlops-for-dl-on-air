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

# CLIP normalisation stats — the default for pre-existing manifests written
# before `image_mean/std` were recorded (every such artifact was C-RADIO, which
# uses CLIP norm). Duplicated here (not imported from data.transforms) so the
# lightweight config package doesn't pull in torch/torchvision at import time.
_DEFAULT_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
_DEFAULT_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


class IncompatibleArtifactError(ValueError):
    """Raised when a manifest's ``version`` is not the one we can load.

    Carries enough context for the operator to know what's needed: the
    artifact's claimed version, the version we expected, and a hint to
    re-train when an older (v1) artifact is encountered.
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
    # How the backbone was trained: frozen | lora | full | partial. Drives
    # whether the serving loader must restore fine-tuned backbone weights from
    # the saved state dict (full/partial) instead of re-fetching the pretrained
    # encoder from HF. Defaults to "frozen" so pre-existing manifests load.
    trained_mode: str = "frozen"
    # Input normalisation stats the model was trained with (CLIP for C-RADIO,
    # ImageNet for DINOv2/v3). Serving MUST normalise with these exact values or
    # the encoder sees OOD inputs. Defaults to CLIP so pre-v.next manifests
    # (all C-RADIO) keep reproducing their training-time preprocessing.
    image_mean: list[float] = field(default_factory=lambda: list(_DEFAULT_IMAGE_MEAN))
    image_std: list[float] = field(default_factory=lambda: list(_DEFAULT_IMAGE_STD))
    # Multi-layer ViT feature-fusion depths (DINOv3 only). `None` (default) =
    # last-layer-only, so pre-existing manifests load unchanged. When set, the
    # serving loader must rebuild the fusion combiner so the saved
    # `backbone.fusion.*` weights have a home to load into.
    fusion_layers: list[int] | None = None


@dataclass(frozen=True, slots=True)
class DetectorSpec:
    num_classes: int
    scales: list[int]
    aspect_ratios: list[float]
    score_threshold: float = 0.05
    nms_iou_threshold: float = 0.5
    max_detections: int = 100
    input_size: int = 1024
    # Anchor layout + NMS mode. Defaulted so pre-existing v2 manifests (written
    # before these knobs) still load and reproduce the legacy absolute-layout,
    # class-agnostic-NMS model. `anchor_octaves=None` defers to the module
    # default octave set when rebuilding the AnchorGenerator at serve/eval time.
    anchor_layout: str = "absolute"
    anchor_base_scale: float = 4.0
    anchor_octaves: list[float] | None = None
    nms_per_class: bool = False


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
    found: int | None = int(found_raw) if isinstance(found_raw, int | str) and str(found_raw).isdigit() else None
    if found != ARTIFACT_FORMAT_VERSION:
        hint = (
            "Pre-manifest-v2 (v1) artifacts are no longer loadable; re-train to produce v2."
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
