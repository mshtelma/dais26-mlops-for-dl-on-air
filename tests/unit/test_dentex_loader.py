"""Unit tests for src/data/dentex_loader.py."""

from __future__ import annotations

import json

import pytest

from src.data.dentex_loader import convert_to_coco, download_dentex, get_label_map

# ---------------------------------------------------------------------------
# get_label_map
# ---------------------------------------------------------------------------

def test_get_label_map_keys():
    lm = get_label_map()
    assert set(lm.keys()) == {0, 1, 2, 3}


def test_get_label_map_values():
    lm = get_label_map()
    assert lm[0] == "Caries"
    assert lm[1] == "Deep Caries"
    assert lm[2] == "Periapical Lesion"
    assert lm[3] == "Impacted"


def test_get_label_map_returns_copy():
    lm1 = get_label_map()
    lm2 = get_label_map()
    lm1[99] = "poison"
    assert 99 not in lm2


# ---------------------------------------------------------------------------
# download_dentex - monkeypatched
# ---------------------------------------------------------------------------

def test_download_dentex_calls_snapshot_download(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    download_dentex(str(tmp_path), hf_token="tok123")

    assert len(calls) == 1
    assert calls[0]["repo_id"] == "ibrahimhamamci/DENTEX"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["local_dir"] == str(tmp_path)
    assert calls[0]["token"] == "tok123"


def test_download_dentex_no_token(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    download_dentex(str(tmp_path))

    assert calls[0]["token"] is None


def test_download_dentex_creates_image_subdirs(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: None)

    download_dentex(str(tmp_path))

    for split in ("train", "val", "test"):
        assert (tmp_path / "images" / split).is_dir()


# ---------------------------------------------------------------------------
# convert_to_coco - ValueError on wrong split sizes
# ---------------------------------------------------------------------------

def test_convert_to_coco_raises_on_missing_source(tmp_path):
    """No source JSONs raises ValueError."""
    with pytest.raises(ValueError, match="No source JSON found"):
        convert_to_coco(str(tmp_path))


def test_convert_to_coco_raises_on_wrong_count(tmp_path):
    """Source JSON present but wrong image count raises ValueError."""
    src_json = tmp_path / "train_annotations.json"
    src_json.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.png", "width": 16, "height": 16}],
        "annotations": [],
    }))
    with pytest.raises(ValueError, match="expected 705 images, found 1"):
        convert_to_coco(str(tmp_path))
