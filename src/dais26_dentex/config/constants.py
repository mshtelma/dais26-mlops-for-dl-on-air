"""Pure constants — no logic, no runtime branching.

If a value is "this string identifies an artifact, alias, or FPN level", put it
here. If it's "this default knob a user might tune at runtime", put it on
`TrainerConfig` / `DetectorConfig` instead.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

# --- Artifact contract ---------------------------------------------------
# Bumped whenever the on-disk artifact layout (filenames, manifest schema,
# expected keys) changes in a way the old `DetectorPyfunc` cannot read.
# v2 introduces a single `manifest.json` replacing the 3 v1 sidecars
# (backbone_config / detection_config / label_map). v1 artifacts are still
# loadable via `DetectorPyfuncV1` for one release.
ARTIFACT_FORMAT_VERSION: Final[int] = 2

# Artifact filenames inside the MLflow model dir (kept v1-compatible until
# Phase 4 ships v2).
MODEL_STATE_FILE: Final[str] = "model_state.pt"
BACKBONE_CONFIG_FILE: Final[str] = "backbone_config.json"
DETECTION_CONFIG_FILE: Final[str] = "detection_config.json"
LABEL_MAP_FILE: Final[str] = "label_map.json"
MODEL_CACHE_DIR: Final[str] = "model_cache"
MANIFEST_FILE: Final[str] = "manifest.json"  # v2 only

# --- UC Model Registry aliases ------------------------------------------
# `@challenger` is set on the dev-schema model by training / the HPO sweep
# (Big Book "deploy code" terminology); registering a new @challenger version
# auto-triggers the deployment job. `@champion` is set on the separate prod
# schema model by the promote task after eval + approval pass (see
# docs/RUNBOOK.md#deployment-job). The constant name keeps the historical
# `ALIAS_CANDIDATE` identifier so existing call sites (set_candidate_alias,
# the `candidate_alias` kwargs) stay stable; only the alias VALUE moved to
# `challenger`.
ALIAS_CANDIDATE: Final[str] = "challenger"
ALIAS_CHAMPION: Final[str] = "champion"


# --- FPN levels ----------------------------------------------------------
class FPNLevel(StrEnum):
    """Feature-pyramid level identifiers used by `models.detection_head` and
    `models.adapters`. String-valued so JSON sidecars and log lines stay
    human-readable.
    """

    P3 = "p3"
    P4 = "p4"
    P5 = "p5"
    P6 = "p6"


FPN_LEVELS: Final[tuple[FPNLevel, ...]] = (
    FPNLevel.P3,
    FPNLevel.P4,
    FPNLevel.P5,
    FPNLevel.P6,
)


# --- HuggingFace env knobs ----------------------------------------------
# Single source of truth for the env-var names used by `platform.hf_env`.
# Keep keys identical to upstream so `os.environ.get(...)` in third-party
# code reads the same values we set.
HF_ENV_HOME: Final[str] = "HF_HOME"
HF_ENV_TRANSFORMERS_CACHE: Final[str] = "TRANSFORMERS_CACHE"
HF_ENV_ENABLE_HF_TRANSFER: Final[str] = "HF_HUB_ENABLE_HF_TRANSFER"
HF_ENV_DOWNLOAD_TIMEOUT: Final[str] = "HF_HUB_DOWNLOAD_TIMEOUT"
HF_ENV_TOKEN: Final[str] = "HF_TOKEN"
