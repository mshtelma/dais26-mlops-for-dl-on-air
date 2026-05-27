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


def detection_collate(batch):
    """Collate function for DataLoader.

    Stacks images (assumes same H x W after padding) and keeps targets as a
    list of dicts so variable numbers of boxes per image are handled cleanly.
    """
    images = torch.stack([item[0] for item in batch], 0)
    targets = [item[1] for item in batch]
    return images, targets
