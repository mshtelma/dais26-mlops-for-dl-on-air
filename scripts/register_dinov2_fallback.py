"""DINOv2-base fallback runbook (emergency procedure when C-RADIOv4 fails).

6 steps:
    1. Re-train detector head on top of DINOv2 (dim=768, NOT 1152)
    2. Recompute embeddings with new dim
    3. Recreate Vector Search index with embedding_dimension=768
    4. Regenerate drift reference
    5. Deploy endpoint serving the DINOv2 model
    6. Update @champion alias

If --use-baked-head is passed, skip step 1 (assumes pin_model_cache.py was run with
--bake-dinov2-fallback during Phase 1).

Usage:
    python scripts/register_dinov2_fallback.py \\
        --catalog ml --schema dais26_vfm \\
        --volume-path /Volumes/ml/dais26_vfm/dentex_raw \\
        --endpoint-name dais26-cradio-detector-prod \\
        --vs-endpoint dais26-vfm-vs \\
        --vs-index ml.dais26_vfm.embeddings_index_dinov2
"""
from __future__ import annotations

import argparse
import logging
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--volume-path", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--vs-endpoint", required=True)
    parser.add_argument("--vs-index", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--use-baked-head", action="store_true",
                        help="Skip step 1 by using the pre-baked DINOv2 head from Phase 1")
    parser.add_argument("--workload-type", default="GPU_SMALL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("dinov2_fallback")

    # Step 1: Re-train head on DINOv2 (unless --use-baked-head)
    if not args.use_baked_head:
        log.info("Step 1/6: Training detector head on DINOv2-base (dim=768) ...")
        from dais26_dentex.train.train_detector import train_detector

        train_detector(
            catalog=args.catalog,
            schema=args.schema,
            backbone_name="dinov2_base",
            volume_path=args.volume_path,
            cache_dir=args.cache_dir,
            epochs=10,
            model_name="cradio_detector",  # share model name; alias differentiates versions
            set_candidate_alias=True,
        )
    else:
        log.info("Step 1/6: SKIPPED (--use-baked-head; using Phase 1 pre-baked checkpoint)")

    # Step 2: Recompute embeddings with summary_dim=768
    log.info("Step 2/6: Recomputing embeddings with DINOv2 (dim=768)")
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    from dais26_dentex.train.precompute_embeddings import precompute_embeddings

    precompute_embeddings(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
        volume_path=args.volume_path,
        backbone_name="dinov2_base",
        cache_dir=args.cache_dir,
        table_name="train_embeddings_dinov2",
    )

    # Step 3: Recreate VS index with dim=768
    log.info("Step 3/6: Creating Vector Search index with embedding_dimension=768")
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    try:
        w.vector_search_indexes.create_index(
            name=args.vs_index,
            endpoint_name=args.vs_endpoint,
            primary_key="image_id",
            index_type="DELTA_SYNC",
            delta_sync_index_spec={
                "source_table": f"{args.catalog}.{args.schema}.train_embeddings_dinov2",
                "embedding_vector_column": "embedding",
                "embedding_dimension": 768,
                "pipeline_type": "TRIGGERED",
                "columns_to_sync": ["image_id", "diagnosis", "split"],
            },
        )
    except Exception as e:
        log.warning("Index create failed (may exist already): %s", e)

    # Step 4: Drift reference is regenerated automatically by run_drift_monitor on next run
    log.info("Step 4/6: Drift reference will regenerate from Step 2 embeddings on next monitor run")

    # Step 5: Deploy endpoint serving the DINOv2 model + Step 6: promote @champion
    log.info("Step 5+6/6: Deploying endpoint + promoting to @champion")
    from dais26_dentex.serve.endpoint_manager import deploy_and_smoke_test

    result = deploy_and_smoke_test(
        endpoint_name=args.endpoint_name,
        catalog=args.catalog,
        schema=args.schema,
        model_name="cradio_detector",
        workload_type=args.workload_type,
        promote_on_success=True,
    )
    log.info("Endpoint result: %s", result)
    return 0 if result.promoted_to_champion else 1


if __name__ == "__main__":
    sys.exit(main())
