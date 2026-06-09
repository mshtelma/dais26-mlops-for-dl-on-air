"""Probe a serving endpoint's GPU memory utilization.

Uses Databricks SDK's serving_endpoints.export_metrics() API (Prometheus format) to fetch
GPU memory metrics. Recommends escalation to GPU_MEDIUM if idle utilization > 85%.

Usage:
    python scripts/probe_endpoint_gpu.py --endpoint dais26-cradio-detector-dev
"""
from __future__ import annotations

import argparse
import logging
import re
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--idle-threshold", type=float, default=85.0,
                        help="Percent GPU memory at which to recommend escalation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("probe_gpu")

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    try:
        text = w.serving_endpoints.export_metrics(name=args.endpoint).contents
    except Exception as e:
        log.error("export_metrics failed (%s). Falling back to a manual check via the Mosaic AI dashboard.", e)
        log.info("Check https://<workspace>/serving-endpoints/%s/metrics manually.", args.endpoint)
        return 2

    if hasattr(text, "decode"):
        text = text.decode("utf-8", errors="ignore")
    if isinstance(text, bytes | bytearray):
        text = text.decode("utf-8", errors="ignore")

    # Look for GPU memory utilization in Prometheus output
    gpu_mem_match = re.search(r"gpu_mem_(?:utilization|used)_percent\s+([0-9.]+)", str(text))
    gpu_util_match = re.search(r"gpu_utilization_percent\s+([0-9.]+)", str(text))

    if gpu_mem_match:
        gpu_mem = float(gpu_mem_match.group(1))
        log.info("GPU memory utilization: %.1f%%", gpu_mem)
        if gpu_mem > args.idle_threshold:
            log.warning("ESCALATE: GPU memory %.1f%% > threshold %.1f%%. "
                        "Recommend upgrading endpoint workload_type from GPU_SMALL to GPU_MEDIUM.",
                        gpu_mem, args.idle_threshold)
            return 1
    else:
        log.warning("Could not find GPU memory metric in Prometheus output. "
                    "Available metric names matching 'gpu': %s",
                    [m for m in re.findall(r"^gpu_\w+", str(text), re.MULTILINE)][:10])

    if gpu_util_match:
        log.info("GPU utilization: %.1f%%", float(gpu_util_match.group(1)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
