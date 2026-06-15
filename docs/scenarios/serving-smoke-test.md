# Serving smoke test (curl)

Send one image to the live detector endpoint and read back detections — and confirm AI Gateway
logged the request to its inference table.

## Prerequisites

A deployed, READY endpoint (`dais26-detector-champion` for prod, or a dev endpoint like
`dais26-cradio-detector-dev`). See [Serve & AI Gateway](../lifecycle/serve.md).

```bash
databricks serving-endpoints get dais26-detector-champion | jq .state
# Expected: {"ready": "READY", "config_update": "NOT_UPDATING"}
```

## Encode an image

```bash
IMG_B64=$(base64 -i /path/to/xray.png | tr -d '\n')
```

## Invoke

```bash
export DATABRICKS_HOST=<your-workspace-url>
export DATABRICKS_TOKEN=<your-pat>

curl -X POST \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["image"], "data": [["'"$IMG_B64"'"]]}}' \
  "https://$DATABRICKS_HOST/serving-endpoints/dais26-detector-champion/invocations"
```

Expected response:

```json
{
  "predictions": [
    {
      "boxes": [[x1, y1, x2, y2], ...],
      "scores": [0.87, ...],
      "labels": ["Caries", "Deep Caries", "Periapical Lesion", "Impacted", ...],
      "num_detections": 7
    }
  ]
}
```

## Confirm AI Gateway logged it

Every request flows through AI Gateway into a Delta table — no client instrumentation. Right
after the curl:

```sql
SELECT request_time, request, response
FROM <catalog>.<schema>.dais26_dentex_detector_inference_payload
ORDER BY request_time DESC
LIMIT 5;
```

That row was written by AI Gateway; it's the same table the [drift
monitor](drift-monitoring.md) reads. The table is auto-created on the **first** request, so grant
the prod SP `SELECT` afterward:

```bash
python scripts/grant_inference_table_access.py    # discovers the suffixed table name + grants
```

## Warm up before a demo

Cold GPU endpoints have high first-request latency. Pre-warm:

```bash
make warmup        # python scripts/warmup_endpoints.py — sends 5 sample requests per endpoint
```

The talk-day latency probe (`bash scripts/latency_probe.sh`) loops every 60s; two consecutive
failures is the switch-to-video trigger. See [Operations & runbook](../RUNBOOK.md).

## Latency & GPU checks

```bash
python scripts/probe_endpoint_gpu.py     # idle GPU util should be <= 85% on GPU_SMALL
```

Open `notebooks/07_latency_benchmark.py` for the p50/p95/p99 protocol; results populate
[Benchmarks](../BENCHMARKS.md). If p99 > 150 ms, follow the pivot ladder (768px → GPU_MEDIUM →
GPU_LARGE → FP16-only).

!!! warning "Train/serve preprocessing must match"
    An earlier serving bug squashed non-square X-rays anisotropically while training letterboxed
    them — served mAP@50 collapsed to 0.176 (vs 0.519 after the fix). The pyfunc now letterboxes
    exactly like training. If you see good train metrics but bad served detections on 2:1
    panoramics, suspect preprocessing. See [HPO campaign log → serving re-eval](../HPO.md).
