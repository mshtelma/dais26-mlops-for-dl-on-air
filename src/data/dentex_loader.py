"""DENTEX dataset loader: download from HuggingFace Hub, convert to COCO format, create drift splits."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

LABEL_MAP: dict[int, str] = {
    0: "Caries",
    1: "Deep Caries",
    2: "Periapical Lesion",
    3: "Impacted",
}

SPLIT_SIZES: dict[str, int] = {"train": 705, "val": 50, "test": 250}

COCO_CATEGORIES = [
    {"id": 0, "name": "Caries", "supercategory": "tooth"},
    {"id": 1, "name": "Deep Caries", "supercategory": "tooth"},
    {"id": 2, "name": "Periapical Lesion", "supercategory": "tooth"},
    {"id": 3, "name": "Impacted", "supercategory": "tooth"},
]


def download_dentex(volume_path: str, hf_token: str | None = None) -> None:
    """Download DENTEX dataset from HuggingFace Hub and unpack zip archives.

    Idempotent: skips files that already exist.
    Unpacks per-split zip archives into volume_path/images/{train,val,test}/.
    """
    import huggingface_hub

    root = Path(volume_path)
    root.mkdir(parents=True, exist_ok=True)

    huggingface_hub.snapshot_download(
        repo_id="ibrahimhamamci/DENTEX",
        repo_type="dataset",
        local_dir=str(root),
        token=hf_token,
    )

    images_root = root / "images"
    for split in ("train", "val", "test"):
        out_dir = images_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for zip_path in root.glob(f"*{split}*.zip"):
            _unpack_zip_idempotent(zip_path, out_dir)


def _unpack_zip_idempotent(zip_path: Path, out_dir: Path) -> None:
    """Unpack a zip archive, skipping members that already exist at destination."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            dest = out_dir / Path(member.filename).name
            if dest.exists():
                continue
            with zf.open(member) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def convert_to_coco(volume_path: str) -> dict[str, str]:
    """Convert DENTEX HuggingFace JSON annotations to canonical COCO format.

    Expects disease_quadrant subset with 705/50/250 train/val/test images.
    Raises ValueError if expected split sizes are not found.
    Returns mapping of split name to output annotation file path.
    """
    root = Path(volume_path)
    annotations_dir = root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, str] = {}

    for split, expected_count in SPLIT_SIZES.items():
        candidate_jsons = list((root / split).glob("*.json")) if (root / split).exists() else []
        candidate_jsons += list(root.glob(f"*{split}*.json"))
        seen: set[Path] = set()
        hf_jsons: list[Path] = []
        for p in candidate_jsons:
            if p not in seen:
                seen.add(p)
                hf_jsons.append(p)

        if not hf_jsons:
            raise ValueError(
                f"No source JSON found for split '{split}' under {root}. "
                "Run download_dentex() first."
            )

        images: list[dict] = []
        annotations: list[dict] = []
        ann_id_offset = 0

        for hf_json in hf_jsons:
            with open(hf_json) as f:
                raw = json.load(f)
            imgs = raw.get("images", [])
            anns = raw.get("annotations", [])
            images.extend(imgs)
            for ann in anns:
                ann = dict(ann)
                ann["id"] = ann.get("id", 0) + ann_id_offset
                annotations.append(ann)
            ann_id_offset += len(anns)

        if len(images) != expected_count:
            raise ValueError(
                f"Split '{split}': expected {expected_count} images, found {len(images)}. "
                "Ensure the disease_quadrant subset is present."
            )

        coco_out = {
            "info": {"description": f"DENTEX {split}", "version": "1.0"},
            "licenses": [],
            "categories": COCO_CATEGORIES,
            "images": images,
            "annotations": annotations,
        }

        out_path = annotations_dir / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(coco_out, f)

        output_paths[split] = str(out_path)

    return output_paths


def create_drift_split(
    volume_path: str,
    split: str = "test",
    contrast_factor: float = 0.5,
    gamma: float = 2.0,
    output_suffix: str = "drift_synthetic",
) -> str:
    """Apply contrast + gamma shift to images and save as a new split.

    Applies torchvision.transforms.functional.adjust_contrast then adjust_gamma.
    Copies COCO annotation file unchanged (bboxes are unaffected by pixel transforms).
    Returns path to new image directory.
    """
    import torchvision.transforms.functional as tvf
    from PIL import Image

    root = Path(volume_path)
    src_dir = root / "images" / split
    dst_dir = root / "images" / output_suffix
    dst_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(src_dir.glob("*.png")):
        dst_path = dst_dir / img_path.name
        if dst_path.exists():
            continue
        pil_img = Image.open(img_path).convert("RGB")
        tensor = tvf.to_tensor(pil_img)
        tensor = tvf.adjust_contrast(tensor, contrast_factor)
        tensor = tvf.adjust_gamma(tensor, gamma)
        out_img = tvf.to_pil_image(tensor)
        out_img.save(dst_path, format="PNG")

    src_ann = root / "annotations" / f"{split}.json"
    dst_ann = root / "annotations" / f"{output_suffix}.json"
    if src_ann.exists() and not dst_ann.exists():
        shutil.copy2(src_ann, dst_ann)

    return str(dst_dir)


def get_label_map() -> dict[int, str]:
    """Return the DENTEX disease category id to name mapping."""
    return dict(LABEL_MAP)
