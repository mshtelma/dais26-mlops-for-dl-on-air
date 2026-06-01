# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Latency Benchmark
# MAGIC Fire 1000 requests at batch=1 against a warmed endpoint; report p50/p95/p99 and apply
# MAGIC the pivot ladder (if p99 > 150ms, recommend GPU_MEDIUM).

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

import base64
import io
import time
import numpy as np
from PIL import Image
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import DataframeSplitInput

w = WorkspaceClient()

# Generate a small test image
buf = io.BytesIO()
Image.new("RGB", (1024, 1024), (128, 128, 128)).save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode("ascii")
payload = DataframeSplitInput(columns=["image"], data=[[b64]])

# Warm-up
print(f"Warming up with {LATENCY_WARMUP_REQUESTS} requests...")
for _ in range(LATENCY_WARMUP_REQUESTS):
    try:
        w.serving_endpoints.query(name=DETECTOR_ENDPOINT_NAME, dataframe_split=payload)
    except Exception:
        pass

# Measurement
print(f"Measuring {LATENCY_NUM_REQUESTS} requests...")
latencies_ms: list[float] = []
errors = 0
for i in range(LATENCY_NUM_REQUESTS):
    t0 = time.perf_counter()
    try:
        w.serving_endpoints.query(name=DETECTOR_ENDPOINT_NAME, dataframe_split=payload)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    except Exception:
        errors += 1
    if (i + 1) % 100 == 0:
        print(f"  done {i + 1}/{LATENCY_NUM_REQUESTS}; errors so far: {errors}")

arr = np.array(latencies_ms)
print(f"\nResults (n={len(arr)} successful, {errors} errors):")
print(f"  p50 = {np.percentile(arr, 50):.1f} ms")
print(f"  p95 = {np.percentile(arr, 95):.1f} ms")
print(f"  p99 = {np.percentile(arr, 99):.1f} ms")
print(f"  mean = {arr.mean():.1f} ms")
print(f"  max = {arr.max():.1f} ms")

p99 = float(np.percentile(arr, 99))
if p99 > LATENCY_PIVOT_THRESHOLD_MS:
    print(f"\nPIVOT RECOMMENDED: p99 {p99:.1f}ms > threshold {LATENCY_PIVOT_THRESHOLD_MS}ms")
    print("Suggested ladder:")
    print(" 1. Reduce input resolution 1024 -> 768")
    print(" 2. Upgrade workload_type GPU_SMALL -> GPU_MEDIUM")
    print(" 3. FP16-only weights")
