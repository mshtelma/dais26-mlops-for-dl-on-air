"""DENTEX dataset loader: download from HuggingFace Hub, convert to COCO format, create drift splits."""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from pathlib import Path

# Disable XET (parallel chunked writer) — incompatible with UC Volume FUSE.
# Must be set before any `huggingface_hub` import. See
# docs/RUNBOOK.md#hf-transfer-fuse-incompat.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

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

# DENTEX hierarchical annotations use a 3-tier scheme: category_id_1 (quadrant),
# category_id_2 (tooth enumeration), category_id_3 (disease). Its disease IDs
# are {0:Impacted, 1:Caries, 2:Periapical Lesion, 3:Deep Caries}, while our
# canonical LABEL_MAP / COCO_CATEGORIES order is
# {0:Caries, 1:Deep Caries, 2:Periapical Lesion, 3:Impacted}.
# This dict maps DENTEX category_id_3 -> our category_id.
DENTEX_CAT3_TO_OURS: dict[int, int] = {0: 3, 1: 0, 2: 2, 3: 1}


def _normalize_annotations(annotations: list[dict]) -> list[dict]:
    """Return a copy of ``annotations`` with disease-level ``category_id`` set
    from DENTEX's ``category_id_3`` (remapped via :data:`DENTEX_CAT3_TO_OURS`)
    and the hierarchical fields stripped.

    Idempotent: if an annotation already has ``category_id`` and no
    ``category_id_3``, the input is passed through unchanged. All other COCO
    fields (``id``, ``image_id``, ``bbox``, ``area``, ``iscrowd``,
    ``segmentation``, …) are preserved.
    """
    hierarchical = {"category_id_1", "category_id_2", "category_id_3"}
    out: list[dict] = []
    for ann in annotations:
        new = {k: v for k, v in ann.items() if k not in hierarchical}
        if "category_id_3" in ann:
            new["category_id"] = DENTEX_CAT3_TO_OURS[ann["category_id_3"]]
        out.append(new)
    return out


def _natural_sort_key(path: Path) -> tuple:
    """Sort key that orders ``test_2.json`` before ``test_10.json``."""
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.stem))


def download_dentex(volume_path: str, hf_token: str | None = None) -> None:
    """Download the DENTEX repo from HuggingFace Hub into ``volume_path``.

    Thin wrapper around ``huggingface_hub.snapshot_download`` — idempotent at
    the file level (existing files are skipped). Does NOT extract any zips;
    call :func:`extract_all_zips` next.

    The XET backend is disabled at module import time (see the module-level
    ``HF_HUB_DISABLE_XET`` block) so this falls back to the classic HTTP
    downloader, which writes files sequentially and works on UC Volume FUSE
    mounts.
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


def extract_all_zips(volume_path: str) -> list[Path]:
    """Recursively unpack every ``*.zip`` found under ``volume_path``.

    Each zip is extracted into a sibling directory named after the zip (no
    extension), preserving the archive's internal structure. Idempotent at
    the member level: a file that already exists at the destination is
    skipped, never overwritten.

    Returns the list of extraction target directories (one per zip found),
    sorted alphabetically.
    """
    root = Path(volume_path)
    extracted: list[Path] = []
    for zip_path in sorted(root.rglob("*.zip")):
        out_dir = zip_path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                dest = out_dir / member.filename
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        extracted.append(out_dir)
    return extracted


def find_coco_json_for_split(volume_path: str, expected_count: int) -> Path | None:
    """Locate a COCO-format JSON under ``volume_path`` whose ``images`` list has
    exactly ``expected_count`` entries and a non-empty ``annotations`` list.

    Files already under ``{volume_path}/annotations/`` are skipped so reruns do
    not re-source their own output. Returns the first match in sorted order,
    or ``None`` when nothing matches.
    """
    root = Path(volume_path)
    own_dir = (root / "annotations").resolve()
    for path in sorted(root.rglob("*.json")):
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if own_dir in resolved.parents or resolved == own_dir:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        images = data.get("images")
        annotations = data.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            continue
        if len(images) == expected_count and annotations:
            return path
    return None


def find_per_image_label_dir(volume_path: str, expected_count: int) -> Path | None:
    """Locate a directory under ``volume_path`` containing exactly
    ``expected_count`` immediate ``*.json`` files (the DENTEX per-image-label
    format, e.g. ``test_data/disease/label/test_0.json`` ... ``test_249.json``).

    Directories whose JSONs live under ``{volume_path}/annotations/`` are
    skipped. Returns the first match in sorted-path order, or ``None``.
    """
    root = Path(volume_path)
    own_dir = (root / "annotations").resolve()
    by_parent: dict[Path, int] = {}
    for p in root.rglob("*.json"):
        try:
            resolved = p.resolve()
        except OSError:
            continue
        if own_dir in resolved.parents or resolved == own_dir:
            continue
        by_parent[p.parent] = by_parent.get(p.parent, 0) + 1
    for parent in sorted(by_parent.keys()):
        if by_parent[parent] == expected_count:
            return parent
    return None


def aggregate_per_image_jsons(label_dir: Path, expected_count: int) -> dict | None:
    """Aggregate per-image JSON annotation files in ``label_dir`` into a single
    COCO-format dict, or return ``None`` if the file count does not match
    ``expected_count`` (don't aggregate a partial set).

    Each per-image file is expected to contain at minimum an ``annotations``
    array (DENTEX hierarchical schema). If it also has an ``images`` array, the
    first entry there provides image metadata; otherwise an image record is
    synthesized from the JSON's stem (``test_0.json`` -> ``test_0.png``).

    Output guarantees:
      - Image ``id`` is renumbered 0..N-1 in natural-sort order of the source files.
      - Annotation ``id`` is monotonic and each annotation's ``image_id`` matches
        its parent image's new id.
      - Annotations are normalized via :func:`_normalize_annotations` (DENTEX
        ``category_id_3`` remapped to our canonical ``category_id``).
    """
    files = sorted(label_dir.glob("*.json"), key=_natural_sort_key)
    if len(files) != expected_count:
        return None

    images: list[dict] = []
    annotations: list[dict] = []
    next_ann_id = 0

    for i, jf in enumerate(files):
        try:
            with open(jf) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None

        raw_images = data.get("images")
        img: dict = (
            dict(raw_images[0])
            if isinstance(raw_images, list) and raw_images and isinstance(raw_images[0], dict)
            else {"file_name": jf.stem + ".png"}
        )
        img["id"] = i
        images.append(img)

        raw_anns = data.get("annotations") or []
        for ann in _normalize_annotations(raw_anns):
            ann = dict(ann)
            ann["id"] = next_ann_id
            ann["image_id"] = i
            annotations.append(ann)
            next_ann_id += 1

    return {
        "info": {"description": "DENTEX (aggregated from per-image JSONs)", "version": "1.0"},
        "licenses": [],
        "categories": COCO_CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


def _build_image_index(volume_path: Path) -> dict[str, Path]:
    """Return ``{basename: full_path}`` for every image under ``volume_path``.

    When the same basename appears in multiple subdirectories, prefer paths
    containing ``quadrant_enumeration_disease`` (the fully-labeled DENTEX
    subset) so the picked file matches our COCO annotations.
    """
    own_images = (volume_path / "images").resolve()
    candidates: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        candidates.extend(volume_path.rglob(pattern))

    def priority(p: Path) -> tuple[int, str]:
        marker = "quadrant_enumeration_disease"
        return (0 if marker in str(p).lower() else 1, str(p))

    candidates.sort(key=priority)

    index: dict[str, Path] = {}
    for p in candidates:
        try:
            if own_images in p.resolve().parents:
                continue
        except OSError:
            pass
        index.setdefault(p.name, p)
    return index


def convert_to_coco(volume_path: str) -> dict[str, str]:
    """Convert discovered DENTEX source JSONs into canonical COCO files.

    For each split in :data:`SPLIT_SIZES`:

    - locate the source JSON via :func:`find_coco_json_for_split` (count match),
    - copy referenced images into ``{volume_path}/images/{split}/`` (idempotent),
    - write canonical COCO to ``{volume_path}/annotations/{split}.json``.

    Returns a mapping of split name to written annotation path. Raises
    :class:`ValueError` if a split's source JSON cannot be found or a
    referenced image cannot be resolved under ``volume_path``.
    """
    root = Path(volume_path)
    annotations_dir = root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    image_index: dict[str, Path] | None = None
    output_paths: dict[str, str] = {}

    for split, expected_count in SPLIT_SIZES.items():
        src_json = find_coco_json_for_split(volume_path, expected_count)
        if src_json is not None:
            with open(src_json) as f:
                raw = json.load(f)
            provenance = str(src_json)
        else:
            label_dir = find_per_image_label_dir(volume_path, expected_count)
            raw = aggregate_per_image_jsons(label_dir, expected_count) if label_dir is not None else None
            if raw is None:
                scanned = "\n  ".join(str(p) for p in sorted(root.rglob("*.json")))
                raise ValueError(
                    f"No source JSON found for split '{split}' "
                    f"(expected {expected_count} images) under {root}. "
                    f"Scanned JSONs:\n  {scanned or '(none)'}"
                )
            provenance = f"{label_dir} (aggregated {expected_count} per-image files)"

        images = list(raw["images"])
        annotations = _normalize_annotations(list(raw["annotations"]))

        if image_index is None:
            image_index = _build_image_index(root)

        images_out_dir = root / "images" / split
        images_out_dir.mkdir(parents=True, exist_ok=True)

        for img in images:
            file_name = Path(img["file_name"]).name
            img["file_name"] = file_name
            dest = images_out_dir / file_name
            if dest.exists():
                continue
            src = image_index.get(file_name)
            if src is None:
                raise ValueError(
                    f"Could not locate image '{file_name}' for split '{split}' "
                    f"under {root}. Extraction may be incomplete."
                )
            shutil.copy2(src, dest)

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
        print(
            f"[convert_to_coco] {split}: matched {provenance} "
            f"({len(images)} images, {len(annotations)} annotations) -> {out_path}"
        )

    return output_paths


def normalize_canonical_annotations(volume_path: str) -> dict[str, bool]:
    """Self-heal pass on existing ``{volume_path}/annotations/{split}.json`` files.

    For each split in :data:`SPLIT_SIZES`, load the canonical COCO JSON (if it
    exists), run its ``annotations`` list through :func:`_normalize_annotations`,
    and rewrite the file only when the result differs from the input. Returns
    ``{split: changed}`` for every split file found on disk; splits whose JSON
    is missing are omitted (the caller decides whether absence is an error).

    Use case: an older version of the loader (pre-normalization) wrote canonical
    files with the DENTEX hierarchical fields intact, so on-disk annotations
    carry ``category_id_3`` instead of the flat ``category_id`` that downstream
    code (``DENTEXDetectionDataset``, exploration notebooks) reads. Re-running
    ``convert_to_coco`` is skipped by the existence gate in ``00_setup.py``;
    this helper heals the stale files in place without re-downloading or
    re-extracting.
    """
    annotations_dir = Path(volume_path) / "annotations"
    result: dict[str, bool] = {}
    for split in SPLIT_SIZES:
        path = annotations_dir / f"{split}.json"
        if not path.exists():
            continue
        with open(path) as f:
            coco = json.load(f)
        original = coco.get("annotations") or []
        normalized = _normalize_annotations(original)
        if normalized != original:
            coco["annotations"] = normalized
            with open(path, "w") as f:
                json.dump(coco, f)
            result[split] = True
        else:
            result[split] = False
    return result


def load_canonical_split(volume_path: str, split: str) -> dict:
    """Load ``{volume_path}/annotations/{split}.json`` and return a COCO dict
    with ``annotations`` normalized via :func:`_normalize_annotations`.

    Use this from any consumer (notebooks, :class:`DENTEXDetectionDataset`,
    embedding precompute, …) instead of a raw ``json.load``. It guarantees a
    flat ``category_id`` is present on every annotation even when the on-disk
    file was written by an older version of the loader that preserved DENTEX's
    hierarchical ``category_id_3`` schema. The on-disk file is NOT modified —
    healing happens in memory. For one-shot in-place healing, see
    :func:`normalize_canonical_annotations`.
    """
    path = Path(volume_path) / "annotations" / f"{split}.json"
    with open(path) as f:
        coco = json.load(f)
    coco["annotations"] = _normalize_annotations(coco.get("annotations") or [])
    return coco


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
