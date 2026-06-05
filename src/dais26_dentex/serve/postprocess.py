"""Detection postprocessing config.

`PostprocessConfig` carries the score / NMS / cap knobs used at inference.
Defaults match the legacy `DetectionModel.__init__` so behavior is unchanged
when an artifact doesn't override them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """Score / NMS / cap knobs used at inference.

    Defaults match the legacy `DetectionModel.__init__` so behavior is
    unchanged when an artifact doesn't override them.
    """

    score_threshold: float = 0.05
    nms_iou_threshold: float = 0.5
    max_detections: int = 100


__all__ = ["PostprocessConfig"]
