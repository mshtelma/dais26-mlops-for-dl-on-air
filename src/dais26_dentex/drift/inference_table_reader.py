from __future__ import annotations

import base64
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_request_payload(request_json: str | None) -> list[bytes]:
    """Parse an AI Gateway request JSON string and extract base64 image bytes.

    Handles both 'dataframe_split' and 'dataframe_records' formats.
    Returns list of image bytes (one per row in the request). Returns [] on any error.

    Args:
        request_json: STRING column from AI Gateway inference table; may be None
                      (payload was > 1 MiB and dropped).
    """
    if request_json is None:
        return []
    try:
        req = json.loads(request_json)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Malformed JSON in request: {e}")
        return []

    images: list[bytes] = []
    try:
        if "dataframe_split" in req:
            cols = req["dataframe_split"].get("columns", [])
            data = req["dataframe_split"].get("data", [])
            if "image" in cols:
                img_idx = cols.index("image")
                for row in data:
                    if row and len(row) > img_idx:
                        try:
                            images.append(base64.b64decode(row[img_idx]))
                        except (base64.binascii.Error, ValueError) as e:
                            logger.warning(f"Invalid base64 in dataframe_split row: {e}")
        elif "dataframe_records" in req:
            for rec in req["dataframe_records"]:
                if "image" in rec:
                    try:
                        images.append(base64.b64decode(rec["image"]))
                    except (base64.binascii.Error, ValueError) as e:
                        logger.warning(f"Invalid base64 in dataframe_records rec: {e}")
        elif "inputs" in req:  # Some endpoints use 'inputs' shape
            for inp in req["inputs"]:
                if isinstance(inp, dict) and "image" in inp:
                    try:
                        images.append(base64.b64decode(inp["image"]))
                    except (base64.binascii.Error, ValueError) as e:
                        logger.warning(f"Invalid base64 in inputs rec: {e}")
        else:
            logger.warning(f"Unknown request schema: keys={list(req.keys())}")
    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Error extracting image from request: {e}")

    return images


def read_recent_inference_images(
    spark: Any,
    catalog: str,
    schema: str,
    table_name: str,
    lookback_hours: int = 1,
    max_rows: int = 1000,
) -> list[bytes]:
    """Read recent base64-decoded images from an AI Gateway inference table.

    Filters request IS NOT NULL (skips > 1 MiB cap rows).
    Filters request_time within lookback window.
    Returns list of image bytes.
    """
    from pyspark.sql import functions as F  # noqa: N812

    df = (
        spark.table(f"{catalog}.{schema}.{table_name}")
        .filter(F.col("request_time") >= F.expr(f"current_timestamp() - interval {lookback_hours} hours"))
        .filter(F.col("request").isNotNull())
        .limit(max_rows)
        .select("request")
    )

    images: list[bytes] = []
    for row in df.collect():
        images.extend(parse_request_payload(row["request"]))
    return images
