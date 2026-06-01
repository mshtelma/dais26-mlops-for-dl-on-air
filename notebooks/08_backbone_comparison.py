# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Backbone comparison (C-RADIOv4 vs DINOv3)
# MAGIC
# MAGIC Pulls the best detector run per backbone from the **shared** MLflow experiment
# MAGIC (`EXPERIMENT_NAME`), builds a side-by-side table of COCO mAP + model stats, renders
# MAGIC a bar chart of `val/mAP_50`, and declares the winner. Because both `train_detector`
# MAGIC runs log `params.backbone_name` and the same `val/mAP_*` metrics into one experiment,
# MAGIC they are directly comparable.
# MAGIC
# MAGIC The final (OPTIONAL) cell measures serving latency by querying each backbone's
# MAGIC endpoint; it skips gracefully when an endpoint is absent.

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import matplotlib.pyplot as plt
import pandas as pd
from mlflow.tracking import MlflowClient

# Backbones to compare. Keys are the `params.backbone_name` values the trainer logs;
# values mirror the per-backbone model/endpoint naming derived in 00_config.py.
COMPARE_BACKBONES: dict[str, dict[str, str]] = {
    "cradio_v4_so400m": {
        "model_short": "cradio_detector",
        "endpoint": "dais26-cradio-detector-dev",
    },
    "dinov3_vitl16": {
        "model_short": "dinov3_detector",
        "endpoint": "dais26-dinov3-detector-dev",
    },
}

# Prefer the monotonic best metric the trainer logs (`val/best_mAP_50`); fall back to
# the last-epoch `val/mAP_50` for older runs that predate the best-metric logging.
PRIMARY_METRIC = "val/best_mAP_50"
FALLBACK_METRIC = "val/mAP_50"

client = MlflowClient()

# COMMAND ----------
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(
        f"No MLflow experiment at {EXPERIMENT_NAME!r}. Train at least one backbone first "
        "(02_train_detector_air.py or the sgcli workloads)."
    )
print(f"Experiment: {EXPERIMENT_NAME} (id={experiment.experiment_id})")


def _best_run(backbone_name: str):
    """Return the highest-scoring finished run for a backbone, or None if absent.

    Ranks by PRIMARY_METRIC, falling back to FALLBACK_METRIC. Ordering is done in
    Python (not MLflow `order_by`) so the `/` in the metric key needs no quoting and
    runs missing the primary metric still participate via the fallback.
    """
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.backbone_name = '{backbone_name}'",
        max_results=1000,
    )
    if not runs:
        return None

    def _score(run) -> float:
        m = run.data.metrics
        if PRIMARY_METRIC in m:
            return m[PRIMARY_METRIC]
        return m.get(FALLBACK_METRIC, float("-inf"))

    return max(runs, key=_score)


best_runs = {bn: _best_run(bn) for bn in COMPARE_BACKBONES}
for bn, run in best_runs.items():
    if run is None:
        print(f"  {bn}: no runs found")
    else:
        print(f"  {bn}: best run {run.info.run_id} ({PRIMARY_METRIC}/{FALLBACK_METRIC} ranked)")

# COMMAND ----------
# ---- Side-by-side comparison table ----


def _row(backbone_name: str, run) -> dict[str, object]:
    if run is None:
        return {"backbone": backbone_name, "run_id": None}
    m = run.data.metrics
    p = run.data.params
    # Train wall-clock from the run lifecycle (no explicit duration metric is logged).
    start, end = run.info.start_time, run.info.end_time
    duration_min = round((end - start) / 1000 / 60, 1) if (start and end) else None
    return {
        "backbone": backbone_name,
        "run_id": run.info.run_id,
        "val/mAP_50": m.get("val/mAP_50"),
        "val/mAP_50_95": m.get("val/mAP_50_95"),
        "val/mAP_75": m.get("val/mAP_75"),
        "val/best_mAP_50": m.get("val/best_mAP_50"),
        "trainable_params": int(p["trainable_params"]) if "trainable_params" in p else None,
        "summary_dim": int(p["summary_dim"]) if "summary_dim" in p else None,
        "spatial_dim": int(p["spatial_dim"]) if "spatial_dim" in p else None,
        "patch_size": int(p["patch_size"]) if "patch_size" in p else None,
        "epochs": int(p["epochs"]) if "epochs" in p else None,
        "train_duration_min": duration_min,
    }


comparison_df = pd.DataFrame([_row(bn, run) for bn, run in best_runs.items()]).set_index("backbone")
print("\n=== Backbone comparison ===")
display(comparison_df)  # noqa: F821  (Databricks builtin)

# COMMAND ----------
# ---- Bar chart of val/mAP_50 per backbone + winner ----
chart_df = comparison_df["val/mAP_50"].dropna()
if chart_df.empty:
    print("No val/mAP_50 metrics available yet — train both backbones first.")
else:
    fig, ax = plt.subplots(figsize=(6, 4))
    chart_df.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"][: len(chart_df)])
    ax.set_ylabel("val/mAP_50")
    ax.set_title("Detector val/mAP_50 by backbone")
    ax.set_ylim(0, max(0.01, chart_df.max() * 1.15))
    for i, v in enumerate(chart_df.values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()

    winner = chart_df.idxmax()
    print(f"\nWINNER (highest val/mAP_50): {winner} = {chart_df.max():.4f}")
    if len(chart_df) > 1:
        runner_up = chart_df.drop(winner).idxmax()
        delta = chart_df[winner] - chart_df[runner_up]
        print(f"  beats {runner_up} ({chart_df[runner_up]:.4f}) by {delta:+.4f} mAP_50")

# COMMAND ----------
# MAGIC %md
# MAGIC ## OPTIONAL — serving latency per backbone
# MAGIC Queries each backbone's serving endpoint with a single 1024px image and reports a
# MAGIC quick p50/p95 (reuses the query pattern from `07_latency_benchmark.py`). Safe to run
# MAGIC even if one/both endpoints aren't deployed — missing endpoints are skipped.

# COMMAND ----------
import base64
import io
import time

import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import DataframeSplitInput
from PIL import Image

# Smaller request count than the full 07 benchmark — this is a comparison smoke, not a
# load test. Bump LATENCY_* in 00_config.py if you want the heavier run.
LATENCY_SMOKE_REQUESTS = 50
LATENCY_SMOKE_WARMUP = 5

w = WorkspaceClient()

buf = io.BytesIO()
Image.new("RGB", (1024, 1024), (128, 128, 128)).save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode("ascii")
payload = DataframeSplitInput(columns=["image"], data=[[b64]])


def _measure_endpoint(endpoint_name: str) -> dict[str, float] | None:
    """Return {p50, p95, mean} ms for an endpoint, or None if it can't be queried."""
    try:
        w.serving_endpoints.get(name=endpoint_name)
    except Exception as e:
        print(f"  {endpoint_name}: not found / unavailable ({type(e).__name__}); skipping")
        return None

    for _ in range(LATENCY_SMOKE_WARMUP):
        try:
            w.serving_endpoints.query(name=endpoint_name, dataframe_split=payload)
        except Exception:
            pass

    latencies: list[float] = []
    for _ in range(LATENCY_SMOKE_REQUESTS):
        t0 = time.perf_counter()
        try:
            w.serving_endpoints.query(name=endpoint_name, dataframe_split=payload)
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception:
            pass
    if not latencies:
        print(f"  {endpoint_name}: 0 successful queries; skipping")
        return None
    arr = np.array(latencies)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "mean_ms": round(float(arr.mean()), 1),
        "n": len(arr),
    }


latency_rows: list[dict[str, object]] = []
for bn, cfg in COMPARE_BACKBONES.items():
    print(f"Measuring {cfg['endpoint']} ...")
    stats = _measure_endpoint(cfg["endpoint"])
    if stats is not None:
        latency_rows.append({"backbone": bn, "endpoint": cfg["endpoint"], **stats})

if latency_rows:
    latency_df = pd.DataFrame(latency_rows).set_index("backbone")
    print("\n=== Serving latency (batch=1, 1024px) ===")
    display(latency_df)  # noqa: F821  (Databricks builtin)
else:
    print("\nNo endpoints available to benchmark — deploy them via 04_deploy_serving.py first.")
