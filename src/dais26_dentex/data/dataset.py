"""PyTorch Dataset for DENTEX detection task."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dentex_loader import load_canonical_split


class DENTEXDetectionDataset(Dataset):
    """COCO-style detection dataset for DENTEX dental X-ray images.

    Args:
        volume_path: Root path of the DENTEX Unity Catalog Volume mount
            (e.g. ``/Volumes/<catalog>/dais26_vfm/dentex_raw``).
        split: One of ``train``, ``val``, ``test``, or ``drift_synthetic``.
        transforms: Optional callable accepting keyword args ``image``,
            ``bboxes`` (COCO ``[x, y, w, h]``), ``class_labels`` and returning
            a dict with the transformed ``image`` (CHW tensor), ``bboxes``
            (list of ``[x, y, w, h]``), and ``class_labels`` (list of int).
            See ``dais26_dentex.data.transforms``.
    """

    def __init__(self, volume_path: str, split: str, transforms=None) -> None:
        self.images_dir = Path(volume_path) / "images" / split
        coco = load_canonical_split(volume_path, split)
        self.images: list[dict] = coco["images"]  # [{id, file_name, width, height}, ...]
        self.ann_by_img: dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            self.ann_by_img[ann["image_id"]].append(ann)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def per_image_label_sets(self) -> list[set[int]]:
        """Return the set of category ids present in each image, by dataset index.

        Cheap (annotation-only; no image decode) so a sampler can be built from
        it. Used for class-balanced oversampling (see
        :func:`build_caries_oversampled_indices`).
        """
        return [{ann["category_id"] for ann in self.ann_by_img[img["id"]]} for img in self.images]

    def __getitem__(self, idx: int):
        info = self.images[idx]
        img = np.array(Image.open(self.images_dir / info["file_name"]).convert("RGB"))
        anns = self.ann_by_img[info["id"]]
        bboxes = [ann["bbox"] for ann in anns]  # COCO [x, y, w, h]
        labels = [ann["category_id"] for ann in anns]

        if self.transforms is not None:
            out = self.transforms(image=img, bboxes=bboxes, class_labels=labels)
            img, bboxes, labels = out["image"], out["bboxes"], out["class_labels"]

        target = {
            "boxes": torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(info["id"]),
        }
        return img, target


class IndexRemapDataset(Dataset):
    """Wrap a dataset, exposing it through a (possibly repeated) index list.

    ``IndexRemapDataset(base, [0, 0, 1, 2, 2, 2])`` makes ``base[0]`` appear
    twice and ``base[2]`` thrice per epoch. This is the DDP-safe way to
    oversample: a longer flat index list works transparently with
    ``DistributedSampler`` (which only knows ``len`` and integer indices),
    unlike ``WeightedRandomSampler`` which has no distributed variant.
    """

    def __init__(self, base: Dataset, indices: list[int]) -> None:
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[self.indices[i]]


def build_caries_oversampled_indices(
    label_sets: list[set[int]],
    factor: float,
    positive_class: int = 0,
) -> list[int]:
    """Build an expanded index list that oversamples ``positive_class`` images.

    Every image appears at least once. Images containing ``positive_class``
    (Caries id 0) appear ``floor(factor)`` times, plus one extra copy for the
    first ``round(frac * n_pos)`` positives (in index order) where
    ``frac = factor - floor(factor)``. Deterministic — no RNG — so every DDP
    rank constructs an identical list. ``factor <= 1.0`` returns the identity
    index list (legacy behavior).
    """
    base = list(range(len(label_sets)))
    if factor <= 1.0:
        return base
    positives = [i for i, labs in enumerate(label_sets) if positive_class in labs]
    full = int(factor)
    frac = factor - full
    expanded = list(base)
    for _ in range(full - 1):  # `base` already contributes one copy
        expanded.extend(positives)
    extra = int(round(frac * len(positives)))
    expanded.extend(positives[:extra])
    return expanded


def detection_collate(batch):
    """Collate function for DataLoader.

    Stacks images (assumes same H x W after padding) and keeps targets as a
    list of dicts so variable numbers of boxes per image are handled cleanly.
    """
    images = torch.stack([item[0] for item in batch], 0)
    targets = [item[1] for item in batch]
    return images, targets
