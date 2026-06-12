"""Tests for config.champion — champion→backbone routing.

`resolve_backbone_from_source_model` is pure; `resolve_effective_backbone` reads
the live champion's `source_dev_model` tag via the UC registry (monkeypatched
here) and is the helper notebooks 03 / 05 call.
"""

from __future__ import annotations

import pytest

from dais26_dentex.config import champion

NAMES = {
    "cradio_v4_so400m": {"model_short": "cradio_detector"},
    "dinov3_vitl16": {"model_short": "dinov3_detector"},
}


# --- pure: resolve_backbone_from_source_model ------------------------------


def test_resolve_backbone_from_source_model_maps_short() -> None:
    assert (
        champion.resolve_backbone_from_source_model("main.s.dinov3_detector", NAMES, "cradio_v4_so400m")
        == "dinov3_vitl16"
    )


def test_resolve_backbone_from_source_model_falls_back() -> None:
    assert champion.resolve_backbone_from_source_model(None, NAMES, "cradio_v4_so400m") == "cradio_v4_so400m"
    assert (
        champion.resolve_backbone_from_source_model("main.s.unknown_detector", NAMES, "cradio_v4_so400m")
        == "cradio_v4_so400m"
    )


# --- impure: resolve_effective_backbone (registry lookup) ------------------


class _MV:
    def __init__(self, tags: dict[str, str]) -> None:
        self.tags = tags


def _patch_client(monkeypatch: pytest.MonkeyPatch, *, mv: _MV | None = None, raises: bool = False) -> None:
    import mlflow.tracking

    class _Client:
        def __init__(self, registry_uri: str | None = None) -> None:
            pass

        def get_model_version_by_alias(self, name: str, alias: str):
            if raises:
                raise RuntimeError("no champion")
            return mv

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", _Client)


def test_resolve_effective_backbone_reads_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, mv=_MV({"source_dev_model": "main.s.dinov3_detector"}))
    bb, src = champion.resolve_effective_backbone("main.s.detector_champion", NAMES, "cradio_v4_so400m")
    assert bb == "dinov3_vitl16"
    assert src == "main.s.dinov3_detector"


def test_resolve_effective_backbone_no_champion_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, raises=True)
    bb, src = champion.resolve_effective_backbone("main.s.detector_champion", NAMES, "cradio_v4_so400m")
    assert bb == "cradio_v4_so400m"
    assert src is None


def test_resolve_effective_backbone_missing_tag_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, mv=_MV({}))  # champion exists but no source_dev_model tag
    bb, src = champion.resolve_effective_backbone("main.s.detector_champion", NAMES, "cradio_v4_so400m")
    assert bb == "cradio_v4_so400m"
    assert src is None
