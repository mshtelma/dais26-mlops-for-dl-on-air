"""Tests for `dais26_dentex.config.constants`.

Constants modules deserve tests precisely because they are the load-bearing
strings the rest of the codebase keys off of. A typo here silently breaks
artifact loading or alias-based promotion. These tests are cheap insurance.
"""

from __future__ import annotations

from dais26_dentex.config import constants as C  # noqa: N812


def test_artifact_format_version_is_int() -> None:
    """The pyfunc loader compares this to a literal `int`, not a string."""
    assert isinstance(C.ARTIFACT_FORMAT_VERSION, int)
    assert C.ARTIFACT_FORMAT_VERSION >= 1


def test_artifact_filenames_are_distinct() -> None:
    """If two artifact files end up with the same filename one will overwrite
    the other on `mlflow.log_artifacts`. Catch this here."""
    files = [
        C.MODEL_STATE_FILE,
        C.BACKBONE_CONFIG_FILE,
        C.DETECTION_CONFIG_FILE,
        C.LABEL_MAP_FILE,
        C.MANIFEST_FILE,
    ]
    assert len(set(files)) == len(files)


def test_aliases_are_distinct() -> None:
    assert C.ALIAS_CANDIDATE != C.ALIAS_CHAMPION


def test_fpn_levels_are_unique_and_ordered() -> None:
    """`FPN_LEVELS` is iterated in this order in the FPN builder; the order
    matters for output-channel ordering."""
    values = [lvl.value for lvl in C.FPN_LEVELS]
    assert values == ["p3", "p4", "p5", "p6"]
    assert len(set(values)) == len(values)


def test_fpn_levels_string_compare() -> None:
    """`FPNLevel(str, Enum)` lets us compare to plain strings without
    `.value` — used in JSON sidecar comparisons."""
    assert C.FPNLevel.P3 == "p3"
    assert C.FPNLevel.P4 == "p4"


def test_hf_env_keys_match_upstream_names() -> None:
    """We are NOT free to rename these — they must match what the HF libs
    read at import time."""
    assert C.HF_ENV_HOME == "HF_HOME"
    assert C.HF_ENV_TRANSFORMERS_CACHE == "TRANSFORMERS_CACHE"
    assert C.HF_ENV_ENABLE_HF_TRANSFER == "HF_HUB_ENABLE_HF_TRANSFER"
    assert C.HF_ENV_DOWNLOAD_TIMEOUT == "HF_HUB_DOWNLOAD_TIMEOUT"
    assert C.HF_ENV_TOKEN == "HF_TOKEN"


def test_model_cache_dir_is_relative() -> None:
    """`MODEL_CACHE_DIR` is appended under the artifact root; it must NOT
    be an absolute path or it would escape the MLflow artifact dir."""
    assert not C.MODEL_CACHE_DIR.startswith("/")
