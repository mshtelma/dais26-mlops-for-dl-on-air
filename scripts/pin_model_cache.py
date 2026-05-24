"""Cache vision backbone weights and DINOv2 fallback head into a UC Volume.

Usage:
    python scripts/pin_model_cache.py \\
        --cache-dir /Volumes/<catalog>/dais26_vfm/model_cache \\
        --cradio-revision <sha> \\
        [--bake-dinov2-fallback]

Per Architect condition #1: --bake-dinov2-fallback runs ONE DINOv2 training pass so
the fallback runbook can skip re-training during a crisis (saves ~15 min).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True,
                        help="UC Volume path to use as HF_HOME / TRANSFORMERS_CACHE")
    parser.add_argument("--cradio-revision", default="main",
                        help="HuggingFace revision (commit SHA) for C-RADIOv4")
    parser.add_argument("--include-dinov3", action="store_true",
                        help="Also cache DINOv3 (requires HF token, gated)")
    parser.add_argument("--bake-dinov2-fallback", action="store_true",
                        help="Train + save DINOv2 fallback head checkpoint into the cache")
    parser.add_argument("--catalog", default="ml_dev",
                        help="UC catalog for the dinov2 fallback training run (only with --bake)")
    parser.add_argument("--schema", default="dais26_vfm")
    parser.add_argument("--volume-path", default=None,
                        help="DENTEX volume path; only needed with --bake-dinov2-fallback")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("pin_cache")

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    os.makedirs(args.cache_dir, exist_ok=True)

    log.info("Caching C-RADIOv4-SO400M (revision=%s)", args.cradio_revision)
    from transformers import AutoModel

    AutoModel.from_pretrained(
        "nvidia/C-RADIOv4-SO400M",
        trust_remote_code=True,
        revision=args.cradio_revision,
        cache_dir=args.cache_dir,
    )

    if args.include_dinov3:
        token = os.environ.get("HF_TOKEN")
        if not token:
            log.warning("--include-dinov3 set but HF_TOKEN not present; skipping")
        else:
            log.info("Caching DINOv3-ViT-L/16")
            AutoModel.from_pretrained(
                "facebook/dinov3-vitl16-pretrain-lvd1689m",
                token=token,
                cache_dir=args.cache_dir,
            )

    log.info("Caching DINOv2-base (always; needed for fallback)")
    import torch

    torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)

    if args.bake_dinov2_fallback:
        if not args.volume_path:
            log.error("--volume-path required for --bake-dinov2-fallback")
            return 1
        log.info("Baking DINOv2 fallback head checkpoint (1 epoch) ...")
        from src.train.train_detector import train_detector

        train_detector(
            catalog=args.catalog,
            schema=args.schema,
            backbone_name="dinov2_base",
            volume_path=args.volume_path,
            cache_dir=args.cache_dir,
            epochs=1,
            batch_size=2,
            num_workers=0,
            model_name="cradio_detector_dinov2_fallback",
            set_candidate_alias=False,
        )
        log.info("DINOv2 fallback head trained + cached")

    log.info("Cache populated at %s", args.cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
