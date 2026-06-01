"""Configuration primitives for dais26_dentex.

`constants` holds pure literals (alias names, FPN levels, artifact filenames,
schema version). `trainer_config` holds the runtime-tunable knobs.
"""

from dais26_dentex.config.trainer_config import (
    ALLOWED_BACKBONES,
    BACKBONE_ALIASES,
    TrainerConfig,
)

__all__ = [
    "ALLOWED_BACKBONES",
    "BACKBONE_ALIASES",
    "TrainerConfig",
]
