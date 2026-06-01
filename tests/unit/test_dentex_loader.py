"""Unit tests for src/dais26_dentex/data/dentex_loader.py."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from dais26_dentex.data.dentex_loader import (
    DENTEX_CAT3_TO_OURS,
    _normalize_annotations,
    aggregate_per_image_jsons,
    convert_to_coco,
    download_dentex,
    extract_all_zips,
    find_coco_json_for_split,
    find_per_image_label_dir,
    get_label_map,
    load_canonical_split,
    normalize_canonical_annotations,
)

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
# download_dentex - monkeypatched (thin snapshot_download wrapper)
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


def test_download_dentex_creates_root_only(tmp_path, monkeypatch):
    """After refactor: download_dentex creates the volume root and does NOT
    pre-create images/{split} subdirs (extraction is a separate step)."""
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: None)

    new_root = tmp_path / "fresh_volume"
    download_dentex(str(new_root))

    assert new_root.is_dir()
    assert not (new_root / "images").exists()


# ---------------------------------------------------------------------------
# extract_all_zips
# ---------------------------------------------------------------------------


def test_extract_all_zips_unpacks_nested_zip(tmp_path):
    """A zip in a subdirectory should be discovered and unpacked in-place."""
    nested = tmp_path / "DENTEX"
    nested.mkdir()
    zip_path = nested / "training_data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("inner/foo.txt", b"hello")
        zf.writestr("inner/bar.png", b"\x89PNG\r\n\x1a\n")

    extracted = extract_all_zips(str(tmp_path))

    out_dir = nested / "training_data"
    assert out_dir in extracted
    assert (out_dir / "inner" / "foo.txt").read_bytes() == b"hello"
    assert (out_dir / "inner" / "bar.png").exists()


def test_extract_all_zips_is_idempotent(tmp_path):
    """Re-running extract_all_zips must not overwrite or re-copy existing files."""
    zip_path = tmp_path / "a.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("note.txt", b"original")

    extract_all_zips(str(tmp_path))
    target = tmp_path / "a" / "note.txt"
    target.write_bytes(b"user-edited")

    extract_all_zips(str(tmp_path))

    assert target.read_bytes() == b"user-edited"


# ---------------------------------------------------------------------------
# find_coco_json_for_split
# ---------------------------------------------------------------------------


def _coco_payload(n_images: int, n_annotations: int = 1) -> dict:
    return {
        "images": [{"id": i, "file_name": f"img_{i}.png", "width": 16, "height": 16} for i in range(n_images)],
        "annotations": [
            {"id": j, "image_id": j % max(n_images, 1), "category_id": 0, "bbox": [0, 0, 1, 1]}
            for j in range(n_annotations)
        ],
    }


def test_find_coco_json_for_split_picks_by_count(tmp_path):
    (tmp_path / "small.json").write_text(json.dumps(_coco_payload(1)))
    (tmp_path / "big.json").write_text(json.dumps(_coco_payload(705)))

    found = find_coco_json_for_split(str(tmp_path), 705)

    assert found is not None
    assert found.name == "big.json"


def test_find_coco_json_for_split_returns_none_when_no_match(tmp_path):
    (tmp_path / "small.json").write_text(json.dumps(_coco_payload(1)))

    assert find_coco_json_for_split(str(tmp_path), 705) is None


def test_find_coco_json_for_split_ignores_annotations_dir(tmp_path):
    """A canonical output already written to volume/annotations/ must not be
    re-picked as a source on subsequent runs."""
    own_dir = tmp_path / "annotations"
    own_dir.mkdir()
    (own_dir / "train.json").write_text(json.dumps(_coco_payload(705)))

    assert find_coco_json_for_split(str(tmp_path), 705) is None


def test_find_coco_json_for_split_skips_empty_annotations(tmp_path):
    """A JSON with the right image count but empty annotations is the
    HF unlabeled index — skip it."""
    (tmp_path / "unlabeled.json").write_text(json.dumps(_coco_payload(705, n_annotations=0)))

    assert find_coco_json_for_split(str(tmp_path), 705) is None


def test_find_coco_json_for_split_discovers_nested(tmp_path):
    """rglob — a JSON several levels deep must still be found."""
    deep = tmp_path / "DENTEX" / "training_data" / "quadrant_enumeration_disease"
    deep.mkdir(parents=True)
    (deep / "train.json").write_text(json.dumps(_coco_payload(705)))

    found = find_coco_json_for_split(str(tmp_path), 705)
    assert found == deep / "train.json"


# ---------------------------------------------------------------------------
# convert_to_coco - failure modes
# ---------------------------------------------------------------------------


def test_convert_to_coco_raises_on_missing_source(tmp_path):
    """No JSONs anywhere -> ValueError mentioning No source JSON found."""
    with pytest.raises(ValueError, match="No source JSON found"):
        convert_to_coco(str(tmp_path))


def test_convert_to_coco_raises_on_wrong_count(tmp_path):
    """JSON exists but doesn't match any split count -> same ValueError path."""
    (tmp_path / "tiny.json").write_text(json.dumps(_coco_payload(1)))
    with pytest.raises(ValueError, match="No source JSON found"):
        convert_to_coco(str(tmp_path))


# ---------------------------------------------------------------------------
# convert_to_coco - success path (single-split via monkeypatched SPLIT_SIZES)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _normalize_annotations
# ---------------------------------------------------------------------------


def test_normalize_annotations_renames_and_remaps_category_id_3():
    """category_id_3 (DENTEX) -> category_id (ours), with the ordering remap."""
    raw = [
        {
            "id": 0,
            "image_id": 0,
            "bbox": [0, 0, 1, 1],
            "area": 1,
            "iscrowd": 0,
            "segmentation": [[0, 0, 1, 1]],
            "category_id_1": 1,
            "category_id_2": 5,
            "category_id_3": 1,
        },
        {"id": 1, "image_id": 0, "bbox": [1, 1, 2, 2], "category_id_1": 2, "category_id_2": 6, "category_id_3": 3},
    ]
    out = _normalize_annotations(raw)
    assert out[0]["category_id"] == DENTEX_CAT3_TO_OURS[1]  # Caries -> 0
    assert out[1]["category_id"] == DENTEX_CAT3_TO_OURS[3]  # Deep Caries -> 1
    for ann in out:
        assert "category_id_1" not in ann
        assert "category_id_2" not in ann
        assert "category_id_3" not in ann
    assert out[0]["bbox"] == [0, 0, 1, 1]
    assert out[0]["segmentation"] == [[0, 0, 1, 1]]


def test_normalize_annotations_passes_through_existing_category_id():
    """When the source already has category_id (test fixtures, our own output),
    leave it alone."""
    raw = [{"id": 0, "image_id": 0, "bbox": [0, 0, 1, 1], "category_id": 2}]
    out = _normalize_annotations(raw)
    assert out == raw  # equal content, fresh list
    assert out is not raw


def test_normalize_annotations_idempotent():
    """Running twice doesn't change the result."""
    raw = [{"id": 0, "image_id": 0, "bbox": [0, 0, 1, 1], "category_id_3": 2, "category_id_1": 1, "category_id_2": 5}]
    once = _normalize_annotations(raw)
    twice = _normalize_annotations(once)
    assert once == twice


# ---------------------------------------------------------------------------
# find_per_image_label_dir
# ---------------------------------------------------------------------------


def test_find_per_image_label_dir_picks_directory_by_count(tmp_path):
    label = tmp_path / "DENTEX" / "test_data" / "disease" / "label"
    label.mkdir(parents=True)
    for i in range(3):
        (label / f"test_{i}.json").write_text("{}")

    assert find_per_image_label_dir(str(tmp_path), 3) == label
    assert find_per_image_label_dir(str(tmp_path), 4) is None


def test_find_per_image_label_dir_skips_annotations_dir(tmp_path):
    own = tmp_path / "annotations"
    own.mkdir()
    for i in range(3):
        (own / f"x_{i}.json").write_text("{}")

    assert find_per_image_label_dir(str(tmp_path), 3) is None


# ---------------------------------------------------------------------------
# aggregate_per_image_jsons
# ---------------------------------------------------------------------------


def _per_image_payload(image_id_hint: int, category_id_3: int) -> dict:
    return {
        "images": [
            {
                "id": image_id_hint,
                "file_name": f"test_{image_id_hint}.png",
                "width": 16,
                "height": 16,
            }
        ],
        "annotations": [
            {
                "id": image_id_hint,
                "image_id": image_id_hint,
                "bbox": [0, 0, 1, 1],
                "area": 1,
                "iscrowd": 0,
                "category_id_1": 0,
                "category_id_2": 0,
                "category_id_3": category_id_3,
            }
        ],
    }


def test_aggregate_per_image_jsons_builds_coco(tmp_path):
    label = tmp_path / "label"
    label.mkdir()
    for i, cat3 in enumerate([1, 3, 0]):  # Caries, Deep Caries, Impacted (DENTEX ids)
        (label / f"test_{i}.json").write_text(json.dumps(_per_image_payload(i, cat3)))

    coco = aggregate_per_image_jsons(label, expected_count=3)

    assert coco is not None
    assert len(coco["images"]) == 3
    assert [img["id"] for img in coco["images"]] == [0, 1, 2]
    assert len(coco["annotations"]) == 3
    assert [ann["id"] for ann in coco["annotations"]] == [0, 1, 2]
    assert [ann["image_id"] for ann in coco["annotations"]] == [0, 1, 2]
    # Remapped: DENTEX 1->0, 3->1, 0->3
    assert [ann["category_id"] for ann in coco["annotations"]] == [0, 1, 3]
    for ann in coco["annotations"]:
        assert "category_id_3" not in ann
    assert coco["categories"][0]["name"] == "Caries"


def test_aggregate_per_image_jsons_synthesizes_filename_when_missing(tmp_path):
    label = tmp_path / "label"
    label.mkdir()
    (label / "test_0.json").write_text(json.dumps({"annotations": []}))
    (label / "test_1.json").write_text(json.dumps({"annotations": []}))

    coco = aggregate_per_image_jsons(label, expected_count=2)
    assert coco is not None
    file_names = sorted(img["file_name"] for img in coco["images"])
    assert file_names == ["test_0.png", "test_1.png"]


def test_aggregate_per_image_jsons_returns_none_on_count_mismatch(tmp_path):
    label = tmp_path / "label"
    label.mkdir()
    (label / "test_0.json").write_text(json.dumps(_per_image_payload(0, 1)))
    (label / "test_1.json").write_text(json.dumps(_per_image_payload(1, 2)))

    assert aggregate_per_image_jsons(label, expected_count=3) is None


def test_aggregate_per_image_jsons_natural_sort(tmp_path):
    """test_2 must precede test_10 (numeric ordering, not lex)."""
    label = tmp_path / "label"
    label.mkdir()
    for i in [0, 1, 2, 10]:
        (label / f"test_{i}.json").write_text(json.dumps(_per_image_payload(i, 1)))

    coco = aggregate_per_image_jsons(label, expected_count=4)
    assert coco is not None
    file_names = [img["file_name"] for img in coco["images"]]
    assert file_names == ["test_0.png", "test_1.png", "test_2.png", "test_10.png"]


# ---------------------------------------------------------------------------
# convert_to_coco - per-image aggregation path
# ---------------------------------------------------------------------------


def test_convert_to_coco_uses_per_image_aggregation_for_test(tmp_path, monkeypatch):
    """When no bundled JSON has the expected image count, fall back to
    per-image aggregation. Output annotations must be normalized."""
    monkeypatch.setattr(
        "dais26_dentex.data.dentex_loader.SPLIT_SIZES",
        {"test": 2},
    )

    label = tmp_path / "DENTEX" / "test_data" / "disease" / "label"
    label.mkdir(parents=True)
    for i, cat3 in enumerate([1, 0]):  # Caries, Impacted (DENTEX)
        (label / f"test_{i}.json").write_text(json.dumps(_per_image_payload(i, cat3)))
    xrays = tmp_path / "DENTEX" / "test_data" / "disease" / "xrays"
    xrays.mkdir()
    (xrays / "test_0.png").write_bytes(b"\x89PNG-0")
    (xrays / "test_1.png").write_bytes(b"\x89PNG-1")

    result = convert_to_coco(str(tmp_path))

    out_path = Path(result["test"])
    assert out_path == tmp_path / "annotations" / "test.json"
    with open(out_path) as f:
        data = json.load(f)
    assert len(data["images"]) == 2
    assert len(data["annotations"]) == 2
    # Normalized: category_id present, category_id_3 absent
    for ann in data["annotations"]:
        assert "category_id" in ann
        assert "category_id_3" not in ann
    assert data["annotations"][0]["category_id"] == 0  # DENTEX 1 -> 0
    assert data["annotations"][1]["category_id"] == 3  # DENTEX 0 -> 3
    # Images copied into canonical location
    assert (tmp_path / "images" / "test" / "test_0.png").read_bytes() == b"\x89PNG-0"
    assert (tmp_path / "images" / "test" / "test_1.png").read_bytes() == b"\x89PNG-1"


def test_convert_to_coco_normalizes_bundled_source(tmp_path, monkeypatch):
    """When the bundled (count-match) path is taken, annotations carrying the
    DENTEX hierarchical fields still get normalized before write."""
    monkeypatch.setattr(
        "dais26_dentex.data.dentex_loader.SPLIT_SIZES",
        {"train": 1},
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "src.json").write_text(
        json.dumps(
            {
                "images": [{"id": 0, "file_name": "a.png", "width": 16, "height": 16}],
                "annotations": [
                    {
                        "id": 0,
                        "image_id": 0,
                        "bbox": [0, 0, 1, 1],
                        "category_id_1": 1,
                        "category_id_2": 2,
                        "category_id_3": 2,
                    }
                ],
            }
        )
    )
    (raw_dir / "a.png").write_bytes(b"\x89PNG")

    convert_to_coco(str(tmp_path))

    with open(tmp_path / "annotations" / "train.json") as f:
        data = json.load(f)
    ann = data["annotations"][0]
    assert ann["category_id"] == 2  # DENTEX 2 (Periapical Lesion) -> 2
    assert "category_id_3" not in ann


# ---------------------------------------------------------------------------
# normalize_canonical_annotations (self-heal pass)
# ---------------------------------------------------------------------------


def _write_canonical(annotations_dir: Path, split: str, annotations: list[dict]) -> Path:
    annotations_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "info": {"description": f"DENTEX {split}"},
        "licenses": [],
        "categories": [{"id": 0, "name": "Caries"}],
        "images": [{"id": 0, "file_name": "a.png", "width": 16, "height": 16}],
        "annotations": annotations,
    }
    path = annotations_dir / f"{split}.json"
    path.write_text(json.dumps(payload))
    return path


def test_normalize_canonical_annotations_heals_stale_split(tmp_path):
    """A pre-existing split JSON with hierarchical category_id_3 (no category_id)
    must be rewritten in place with normalized annotations."""
    ann_dir = tmp_path / "annotations"
    stale = [
        {
            "id": 0,
            "image_id": 0,
            "bbox": [0, 0, 1, 1],
            "category_id_1": 1,
            "category_id_2": 5,
            "category_id_3": 1,
        }
    ]
    train_path = _write_canonical(ann_dir, "train", stale)

    result = normalize_canonical_annotations(str(tmp_path))

    assert result["train"] is True
    healed = json.loads(train_path.read_text())
    ann = healed["annotations"][0]
    assert ann["category_id"] == DENTEX_CAT3_TO_OURS[1]  # 0
    assert "category_id_1" not in ann
    assert "category_id_2" not in ann
    assert "category_id_3" not in ann


def test_normalize_canonical_annotations_noop_on_already_normalized(tmp_path):
    """When annotations already carry category_id, no rewrite should occur and
    the helper must report changed=False."""
    ann_dir = tmp_path / "annotations"
    good = [{"id": 0, "image_id": 0, "bbox": [0, 0, 1, 1], "category_id": 2}]
    path = _write_canonical(ann_dir, "val", good)
    mtime_before = path.stat().st_mtime_ns
    bytes_before = path.read_bytes()

    result = normalize_canonical_annotations(str(tmp_path))

    assert result["val"] is False
    assert path.stat().st_mtime_ns == mtime_before
    assert path.read_bytes() == bytes_before


def test_normalize_canonical_annotations_skips_missing_files(tmp_path):
    """No annotations dir / no split files at all -> empty mapping, no error."""
    result = normalize_canonical_annotations(str(tmp_path))
    assert result == {}


def test_normalize_canonical_annotations_heals_multiple_splits(tmp_path):
    """Stale train + val + already-good test -> heals first two, leaves test."""
    ann_dir = tmp_path / "annotations"
    stale_ann = [{"id": 0, "image_id": 0, "bbox": [0, 0, 1, 1], "category_id_3": 1}]
    good_ann = [{"id": 0, "image_id": 0, "bbox": [0, 0, 1, 1], "category_id": 1}]
    _write_canonical(ann_dir, "train", stale_ann)
    _write_canonical(ann_dir, "val", stale_ann)
    _write_canonical(ann_dir, "test", good_ann)

    result = normalize_canonical_annotations(str(tmp_path))

    assert result == {"train": True, "val": True, "test": False}
    for split in ("train", "val"):
        data = json.loads((ann_dir / f"{split}.json").read_text())
        assert data["annotations"][0]["category_id"] == 0
        assert "category_id_3" not in data["annotations"][0]


# ---------------------------------------------------------------------------
# load_canonical_split (read-time normalization)
# ---------------------------------------------------------------------------


def test_load_canonical_split_returns_normalized_coco(tmp_path):
    """A canonical file written with hierarchical category_id_3 only must be
    returned with a flat category_id (remapped via DENTEX_CAT3_TO_OURS) and
    no hierarchical fields. The on-disk file is NOT modified."""
    ann_dir = tmp_path / "annotations"
    stale = [
        {
            "id": 0,
            "image_id": 0,
            "bbox": [0, 0, 1, 1],
            "category_id_1": 1,
            "category_id_2": 5,
            "category_id_3": 1,
        }
    ]
    path = _write_canonical(ann_dir, "train", stale)
    bytes_before = path.read_bytes()

    coco = load_canonical_split(str(tmp_path), "train")

    ann = coco["annotations"][0]
    assert ann["category_id"] == DENTEX_CAT3_TO_OURS[1]  # 0
    assert "category_id_1" not in ann
    assert "category_id_2" not in ann
    assert "category_id_3" not in ann
    assert path.read_bytes() == bytes_before  # read-only


def test_load_canonical_split_passes_through_already_normalized(tmp_path):
    """Already-flat annotations must round-trip unchanged in content, and the
    on-disk file must not be touched (mtime preserved)."""
    ann_dir = tmp_path / "annotations"
    good = [{"id": 0, "image_id": 0, "bbox": [0, 0, 1, 1], "category_id": 2}]
    path = _write_canonical(ann_dir, "val", good)
    mtime_before = path.stat().st_mtime_ns
    bytes_before = path.read_bytes()

    coco = load_canonical_split(str(tmp_path), "val")

    assert coco["annotations"][0]["category_id"] == 2
    assert path.stat().st_mtime_ns == mtime_before
    assert path.read_bytes() == bytes_before


def test_load_canonical_split_raises_file_not_found_for_missing_split(tmp_path):
    """A missing split JSON should surface as FileNotFoundError so callers see
    a clear failure instead of silently getting an empty COCO."""
    with pytest.raises(FileNotFoundError):
        load_canonical_split(str(tmp_path), "train")


def test_convert_to_coco_writes_canonical_output(tmp_path, monkeypatch):
    """End-to-end: source JSON + images on disk -> canonical COCO + image copy."""
    monkeypatch.setattr(
        "dais26_dentex.data.dentex_loader.SPLIT_SIZES",
        {"train": 2},
    )

    raw_dir = tmp_path / "DENTEX" / "training_data" / "quadrant_enumeration_disease"
    raw_dir.mkdir(parents=True)
    src = raw_dir / "train_quadrant_enumeration_disease.json"
    src.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 0, "file_name": "a.png", "width": 16, "height": 16},
                    {"id": 1, "file_name": "b.png", "width": 16, "height": 16},
                ],
                "annotations": [
                    {"id": 0, "image_id": 0, "category_id": 0, "bbox": [0, 0, 1, 1]},
                ],
            }
        )
    )
    xrays = raw_dir / "xrays"
    xrays.mkdir()
    (xrays / "a.png").write_bytes(b"\x89PNG-a")
    (xrays / "b.png").write_bytes(b"\x89PNG-b")

    result = convert_to_coco(str(tmp_path))

    out_path = Path(result["train"])
    assert out_path == tmp_path / "annotations" / "train.json"
    with open(out_path) as f:
        data = json.load(f)
    assert len(data["images"]) == 2
    assert data["categories"][0]["name"] == "Caries"
    assert (tmp_path / "images" / "train" / "a.png").read_bytes() == b"\x89PNG-a"
    assert (tmp_path / "images" / "train" / "b.png").read_bytes() == b"\x89PNG-b"
