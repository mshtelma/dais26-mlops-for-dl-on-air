"""Download DENTEX dataset from HuggingFace and stage in a UC Volume.

Usage:
    python scripts/download_dentex.py \\
        --volume-path /Volumes/<catalog>/dais26_vfm/dentex_raw \\
        [--hf-token <token>] \\
        [--create-drift-split]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-path", required=True, help="Target UC Volume path")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--create-drift-split", action="store_true",
                        help="Also generate synthetic drift split via contrast+gamma shift")
    parser.add_argument("--contrast-factor", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("download_dentex")

    from src.data.dentex_loader import convert_to_coco, create_drift_split, download_dentex

    log.info("Downloading DENTEX -> %s", args.volume_path)
    download_dentex(volume_path=args.volume_path, hf_token=args.hf_token)
    log.info("Converting to COCO format")
    mapping = convert_to_coco(args.volume_path)
    for split, path in mapping.items():
        log.info("  %s -> %s", split, path)

    if args.create_drift_split:
        log.info("Creating synthetic drift split (contrast=%.2f, gamma=%.2f)",
                 args.contrast_factor, args.gamma)
        out = create_drift_split(
            volume_path=args.volume_path, split="test",
            contrast_factor=args.contrast_factor, gamma=args.gamma,
            output_suffix="drift_synthetic",
        )
        log.info("Drift split written to %s", out)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
