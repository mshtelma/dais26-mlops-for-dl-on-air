from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _load_image_tensor(
    path: str,
    size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> torch.Tensor:
    """Load image, resize, normalize, return (3, size, size) tensor.

    ``mean``/``std`` default to CLIP stats; pass the backbone's
    ``BackboneInfo.image_mean/std`` so DINOv2/v3 embeddings use ImageNet norm.
    """
    from dais26_dentex.data.transforms import CLIP_MEAN, CLIP_STD

    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean_a = np.array(mean if mean is not None else CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std_a = np.array(std if std is not None else CLIP_STD, dtype=np.float32).reshape(3, 1, 1)
    arr = arr.transpose(2, 0, 1)
    arr = (arr - mean_a) / std_a
    return torch.from_numpy(arr)


def precompute_embeddings(
    spark,
    catalog: str,
    schema: str,
    volume_path: str,
    backbone_name: Literal["cradio_v4_so400m", "dinov3_vitl16", "dinov2_base"] = "cradio_v4_so400m",
    backbone_revision: str | None = None,
    cache_dir: str | None = None,
    batch_size: int = 32,
    splits: list[str] | None = None,
    table_name: str = "train_embeddings",
    image_size: int = 224,
    vector_search_endpoint: str | None = None,
    vector_search_index: str | None = None,
) -> int:
    """Run frozen backbone over DENTEX, write summary embeddings to Delta + optionally trigger VS sync.

    Delta schema (NOTE: ARRAY<FLOAT>, not ARRAY<DOUBLE>):
        image_id     STRING
        embedding    ARRAY<FLOAT>
        diagnosis    STRING (or null if image has no annotations)
        split        STRING
        image_path   STRING

    Table is created with delta.enableChangeDataFeed=true (required by Vector Search Delta Sync).

    Returns: total number of embeddings written.
    """
    from pyspark.sql import Row
    from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

    from dais26_dentex.data.dentex_loader import get_label_map, load_canonical_split
    from dais26_dentex.models.backbones import load_backbone

    splits = splits or ["train", "val", "test"]
    label_map = get_label_map()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, info = load_backbone(
        name=backbone_name,
        revision=backbone_revision,
        cache_dir=cache_dir,
        device=device,
    )
    backbone.eval()
    embedding_dim = info.summary_dim
    logger.info("Backbone %s loaded; summary_dim=%d", info.model_name, embedding_dim)

    # Schema (ARRAY<FLOAT> required for Vector Search)
    schema_struct = StructType(
        [
            StructField("image_id", StringType(), nullable=False),
            StructField("embedding", ArrayType(FloatType()), nullable=False),
            StructField("diagnosis", StringType(), nullable=True),
            StructField("split", StringType(), nullable=False),
            StructField("image_path", StringType(), nullable=False),
        ]
    )

    full_table = f"{catalog}.{schema}.{table_name}"

    # Create the table with CDF enabled if not exists (idempotent)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {full_table} (
            image_id STRING,
            embedding ARRAY<FLOAT>,
            diagnosis STRING,
            split STRING,
            image_path STRING
        )
        USING DELTA
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    total = 0
    for split in splits:
        ann_path = Path(volume_path) / "annotations" / f"{split}.json"
        images_dir = Path(volume_path) / "images" / split
        if not ann_path.exists():
            logger.warning("Annotation file missing: %s", ann_path)
            continue
        coco = load_canonical_split(volume_path, split)
        img_by_id = {img["id"]: img for img in coco.get("images", [])}
        anns_by_img: dict[int, list] = {}
        for ann in coco.get("annotations", []):
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        ids: list[int] = []
        paths: list[str] = []
        diagnoses: list[str | None] = []
        for img_id, img_info in img_by_id.items():
            path = str(images_dir / img_info["file_name"])
            anns = anns_by_img.get(img_id, [])
            diag = label_map.get(anns[0]["category_id"]) if anns else None
            ids.append(img_id)
            paths.append(path)
            diagnoses.append(diag)

        # Batched forward
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            tensors = torch.stack(
                [_load_image_tensor(p, image_size, info.image_mean, info.image_std) for p in batch_paths]
            ).to(device)
            with torch.no_grad():
                summary, _ = backbone(tensors)
                # L2-normalize
                summary = summary / (summary.norm(dim=-1, keepdim=True) + 1e-12)
            all_embeddings.append(summary.cpu().numpy().astype(np.float32))
        if not all_embeddings:
            continue
        embeddings_arr = np.concatenate(all_embeddings, axis=0)

        # Build Spark rows
        rows = [
            Row(
                # Split-scope the id: raw COCO image ids restart per split, so
                # they collide across train/val/test. image_id is the Vector
                # Search primary key, so colliding ids silently dedupe the index
                # (e.g. 1005 rows -> 706). Prefix with split for global uniqueness.
                image_id=f"{split}_{ids[i]}",
                embedding=embeddings_arr[i].tolist(),
                diagnosis=diagnoses[i],
                split=split,
                image_path=paths[i],
            )
            for i in range(len(ids))
        ]
        df = spark.createDataFrame(rows, schema=schema_struct)
        # Overwrite only this split partition
        df.write.mode("overwrite").option("replaceWhere", f"split = '{split}'").saveAsTable(full_table)
        total += len(rows)
        logger.info("Wrote %d embeddings for split=%s", len(rows), split)

    # Trigger Vector Search index sync if configured
    if vector_search_endpoint is not None and vector_search_index is not None:
        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient()
            w.vector_search_indexes.sync_index(index_name=vector_search_index)
            logger.info("Triggered VS index sync: %s", vector_search_index)
            # Best-effort wait
            time.sleep(5)
        except Exception as e:
            logger.error("Failed to sync VS index: %s", e)

    return total
