"""Unit tests for src/dais26_dentex/data/dataset.py using the dentex_mini fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "dentex_mini"


@pytest.fixture(scope="module")
def train_dataset():
    from dais26_dentex.data.dataset import DENTEXDetectionDataset

    return DENTEXDetectionDataset(str(FIXTURE_DIR), split="train")


@pytest.fixture(scope="module")
def val_dataset():
    from dais26_dentex.data.dataset import DENTEXDetectionDataset

    return DENTEXDetectionDataset(str(FIXTURE_DIR), split="val")


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


def test_train_len(train_dataset):
    assert len(train_dataset) == 2


def test_val_len(val_dataset):
    assert len(val_dataset) == 1


# ---------------------------------------------------------------------------
# __getitem__ structure
# ---------------------------------------------------------------------------


def test_getitem_returns_tuple(train_dataset):
    item = train_dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 2


def test_getitem_image_is_ndarray(train_dataset):
    img, _ = train_dataset[0]
    assert isinstance(img, np.ndarray)


def test_getitem_image_rgb(train_dataset):
    img, _ = train_dataset[0]
    assert img.ndim == 3
    assert img.shape[2] == 3


def test_getitem_target_keys(train_dataset):
    _, target = train_dataset[0]
    assert "boxes" in target
    assert "labels" in target
    assert "image_id" in target


def test_getitem_boxes_dtype(train_dataset):
    _, target = train_dataset[0]
    assert target["boxes"].dtype == torch.float32


def test_getitem_boxes_shape(train_dataset):
    _, target = train_dataset[0]
    # img_0001 has 2 annotations
    assert target["boxes"].shape == (2, 4)


def test_getitem_labels_dtype(train_dataset):
    _, target = train_dataset[0]
    assert target["labels"].dtype == torch.long


def test_getitem_image_id_tensor(train_dataset):
    _, target = train_dataset[0]
    assert isinstance(target["image_id"], torch.Tensor)


# ---------------------------------------------------------------------------
# Second item (img_0002 - 1 annotation)
# ---------------------------------------------------------------------------


def test_getitem_second_item_boxes(train_dataset):
    _, target = train_dataset[1]
    assert target["boxes"].shape == (1, 4)


# ---------------------------------------------------------------------------
# detection_collate
# ---------------------------------------------------------------------------


def test_detection_collate():
    from dais26_dentex.data.dataset import detection_collate

    img1 = torch.zeros(3, 16, 16)
    img2 = torch.zeros(3, 16, 16)
    t1 = {"boxes": torch.zeros(2, 4), "labels": torch.zeros(2, dtype=torch.long), "image_id": torch.tensor(1)}
    t2 = {"boxes": torch.zeros(1, 4), "labels": torch.zeros(1, dtype=torch.long), "image_id": torch.tensor(2)}

    images, targets = detection_collate([(img1, t1), (img2, t2)])

    assert images.shape == (2, 3, 16, 16)
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (2, 4)


# ---------------------------------------------------------------------------
# Caries oversampling (Track B): per_image_label_sets + index expansion
# ---------------------------------------------------------------------------


def test_per_image_label_sets(train_dataset):
    sets = train_dataset.per_image_label_sets()
    assert len(sets) == len(train_dataset)
    assert all(isinstance(s, set) for s in sets)


def test_oversample_identity_when_factor_one():
    from dais26_dentex.data.dataset import build_caries_oversampled_indices

    label_sets = [{0}, {1}, {0, 2}]
    assert build_caries_oversampled_indices(label_sets, 1.0, positive_class=0) == [0, 1, 2]


def test_oversample_integer_factor_doubles_positives():
    from dais26_dentex.data.dataset import build_caries_oversampled_indices

    # indices 0 and 2 contain Caries (class 0); index 1 does not.
    label_sets = [{0}, {1}, {0, 2}]
    expanded = build_caries_oversampled_indices(label_sets, 2.0, positive_class=0)
    # Every image once + one extra copy of each positive.
    assert sorted(expanded) == [0, 0, 1, 2, 2]
    assert expanded.count(0) == 2
    assert expanded.count(2) == 2
    assert expanded.count(1) == 1


def test_oversample_fractional_factor_partial_extra():
    from dais26_dentex.data.dataset import build_caries_oversampled_indices

    # 4 positives, factor 1.5 → round(0.5*4)=2 extra copies (first two positives).
    label_sets = [{0}, {0}, {0}, {0}, {1}]
    expanded = build_caries_oversampled_indices(label_sets, 1.5, positive_class=0)
    assert len(expanded) == 5 + 2
    assert expanded[:5] == [0, 1, 2, 3, 4]
    assert expanded[5:] == [0, 1]


def test_oversample_deterministic():
    from dais26_dentex.data.dataset import build_caries_oversampled_indices

    label_sets = [{0}, {1}, {0, 2}, {3}, {0}]
    a = build_caries_oversampled_indices(label_sets, 2.0)
    b = build_caries_oversampled_indices(label_sets, 2.0)
    assert a == b  # no RNG → every DDP rank agrees


def test_index_remap_dataset_maps_and_lengths():
    from dais26_dentex.data.dataset import IndexRemapDataset

    base = ["a", "b", "c"]
    remap = IndexRemapDataset(base, [0, 0, 2])
    assert len(remap) == 3
    assert [remap[i] for i in range(len(remap))] == ["a", "a", "c"]
