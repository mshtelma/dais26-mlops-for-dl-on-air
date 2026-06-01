"""Grant SELECT on the auto-created AI Gateway inference tables to a service principal.

Runs AFTER the first endpoint deploy + first inference request (so the table exists).

Usage:
    python scripts/grant_inference_table_access.py \\
        --catalog ml --schema dais26_vfm \\
        --table-prefix detector_inference \\
        --principal <sp_app_id>
"""
from __future__ import annotations

import argparse
import logging
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--table-prefix", required=True,
                        help="Prefix used in ai_gateway.inference_table_config (e.g., detector_inference)")
    parser.add_argument("--principal", required=True,
                        help="Service principal application ID (UUID) or display name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("grant_inference")

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    # AI Gateway inference table is named <prefix>_payload (request/response) and <prefix>_logs.
    # Grant on both (the SDK names vary; we try common patterns and ignore non-existent).
    candidate_tables = [
        f"{args.catalog}.{args.schema}.{args.table_prefix}_payload",
        f"{args.catalog}.{args.schema}.{args.table_prefix}_logs",
    ]
    for tbl in candidate_tables:
        try:
            stmt = f"GRANT SELECT ON TABLE {tbl} TO `{args.principal}`"
            log.info("Running: %s", stmt)
            w.statement_execution.execute_statement(
                statement=stmt,
                wait_timeout="30s",
                warehouse_id="auto",  # User must set DATABRICKS_WAREHOUSE_ID env var
            )
        except Exception as e:
            log.warning("Skipping %s: %s", tbl, e)
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
