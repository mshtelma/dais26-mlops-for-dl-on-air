"""Tests for the pure parts of the shared eval runner (`eval/runner.py`).

Only the IO-free helpers are exercised here: the label-name -> category_id
mapping and the per-image pyfunc-row -> COCO ``model_output`` conversion. The
filesystem/pycocotools paths (`materialize_gt`, `score_model_on_split`) need the
DENTEX volume + a GPU model and are covered by the notebooks.
"""

from __future__ import annotations

import pytest
import torch

from dais26_dentex.eval.runner import (
    build_name_to_category_id,
    inner_detection_model,
    load_detector_by_alias,
    model_output_row,
    to_category_id,
)

LABEL_MAP = {0: "Caries", 1: "Deep Caries", 2: "Impacted", 3: "Periapical Lesion"}


def test_build_name_to_category_id_inverts_label_map() -> None:
    name_to_id = build_name_to_category_id(LABEL_MAP)
    assert name_to_id == {"Caries": 0, "Deep Caries": 1, "Impacted": 2, "Periapical Lesion": 3}


def test_to_category_id_maps_known_name() -> None:
    name_to_id = build_name_to_category_id(LABEL_MAP)
    assert to_category_id("Caries", name_to_id) == 0
    assert to_category_id("Impacted", name_to_id) == 2


def test_to_category_id_passes_through_numeric_string() -> None:
    # Defensive branch: an already-numeric label falls back to int().
    assert to_category_id("2", {"Caries": 0}) == 2
    assert to_category_id(3, {"Caries": 0}) == 3


def test_to_category_id_unknown_name_raises() -> None:
    with pytest.raises(ValueError):
        to_category_id("NotAClass", {"Caries": 0})


def test_model_output_row_shapes_and_label_mapping() -> None:
    name_to_id = build_name_to_category_id(LABEL_MAP)
    row = {
        "boxes": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        "scores": [0.9, 0.5],
        "labels": ["Caries", "Impacted"],
    }
    out = model_output_row(image_id=7, row=row, name_to_id=name_to_id)

    assert out["image_id"] == 7
    assert out["boxes"].shape == (2, 4)
    assert out["scores"].shape == (2,)
    assert out["labels"].shape == (2,)
    assert out["boxes"].dtype == torch.float32
    assert out["labels"].dtype == torch.long
    assert out["labels"].tolist() == [0, 2]
    assert out["scores"].tolist() == pytest.approx([0.9, 0.5])


def test_model_output_row_empty_predictions() -> None:
    name_to_id = build_name_to_category_id(LABEL_MAP)
    out = model_output_row(image_id=1, row={"boxes": [], "scores": [], "labels": []}, name_to_id=name_to_id)
    assert out["boxes"].shape == (0, 4)
    assert out["scores"].shape == (0,)
    assert out["labels"].shape == (0,)


# --- load_detector_by_alias (shared by 09 / 09b) ---------------------------


def test_load_detector_by_alias_prefers_first_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow.pyfunc  # force the submodule object the function resolves

    tried: list[str] = []

    def _fake_load(uri: str):
        tried.append(uri)
        if uri.endswith("@champion"):
            return f"MODEL[{uri}]"
        raise RuntimeError("not found")

    monkeypatch.setattr(mlflow.pyfunc, "load_model", _fake_load)
    model, uri = load_detector_by_alias("c.s.m", ("candidate", "champion"))
    assert model == "MODEL[models:/c.s.m@champion]"
    assert uri == "models:/c.s.m@champion"
    assert tried == ["models:/c.s.m@candidate", "models:/c.s.m@champion"]  # candidate tried first


def test_load_detector_by_alias_none_when_all_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow.pyfunc

    def _always_fail(uri: str):
        raise RuntimeError("nope")

    monkeypatch.setattr(mlflow.pyfunc, "load_model", _always_fail)
    assert load_detector_by_alias("c.s.m", ("candidate", "champion")) == (None, None)


# --- inner_detection_model (the threshold-grid mutation seam) ---------------


class _PyModel:
    def __init__(self, model: str) -> None:
        self.model = model


def test_inner_detection_model_via_unwrap() -> None:
    class _Loaded:
        def unwrap_python_model(self):
            return _PyModel("DM")

    assert inner_detection_model(_Loaded()) == "DM"


def test_inner_detection_model_falls_back_to_impl() -> None:
    class _Impl:
        python_model = _PyModel("DM2")

    class _Loaded:  # no unwrap_python_model attribute
        _model_impl = _Impl()

    assert inner_detection_model(_Loaded()) == "DM2"


def test_inner_detection_model_unwrap_raises_falls_back() -> None:
    class _Impl:
        python_model = _PyModel("DM3")

    class _Loaded:
        _model_impl = _Impl()

        def unwrap_python_model(self):
            raise RuntimeError("old mlflow")

    assert inner_detection_model(_Loaded()) == "DM3"
