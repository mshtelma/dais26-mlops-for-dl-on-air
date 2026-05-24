"""Pre-warm Mosaic AI Model Serving endpoints before a live demo.

Sends N small synthetic image requests to the detector endpoint to keep it READY.

Usage:
    python scripts/warmup_endpoints.py \\
        --endpoint dais26-cradio-detector-prod \\
        --requests 5
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
import time
from pathlib import Path


def _make_sample_image_b64(size: int = 224) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), (128, 128, 128)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Serving endpoint name")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--image-path", default=None,
                        help="Optional path to a real PNG/JPEG to use instead of synthetic")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("warmup")

    if args.image_path:
        b64 = base64.b64encode(Path(args.image_path).read_bytes()).decode("ascii")
    else:
        b64 = _make_sample_image_b64()

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import DataframeSplitInput

    w = WorkspaceClient()
    successes = 0
    failures = 0
    for i in range(args.requests):
        try:
            t0 = time.time()
            resp = w.serving_endpoints.query(
                name=args.endpoint,
                dataframe_split=DataframeSplitInput(columns=["image"], data=[[b64]]),
            )
            dt_ms = (time.time() - t0) * 1000
            preds = getattr(resp, "predictions", None)
            log.info("warmup #%d ok  %.0fms  predictions=%s", i + 1, dt_ms,
                     "ok" if preds is not None else "empty")
            successes += 1
        except Exception as e:
            log.error("warmup #%d failed: %s", i + 1, e)
            failures += 1
        if i + 1 < args.requests:
            time.sleep(args.interval_seconds)

    log.info("Done: %d ok / %d failed", successes, failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
