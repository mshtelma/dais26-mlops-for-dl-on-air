# Benchmarks

Performance numbers from Phase 4 validation runs. To be filled after `notebooks/07_latency_benchmark.py`
completes on the production workspace.

See the benchmark protocol in [ARCHITECTURE.md](ARCHITECTURE.md#e3e4-latency-benchmark-protocol) and
acceptance criteria E1, E3, E13, E15 in the plan.

---

## Detection endpoint latency

Benchmark protocol: 50 warm-up requests, then 1000 requests at batch=1, warmed endpoint (scale_to_zero=false).
Input: 1024×1024 dental X-ray. Measured end-to-end (client → invocations → client).

| Config | Input size | p50 (ms) | p95 (ms) | p99 (ms) |
|--------|-----------|---------|---------|---------|
| GPU_SMALL, batch=1 | 1024×1024 | TBD | TBD | TBD |
| GPU_SMALL, batch=1 | 768×768 | TBD | TBD | TBD |
| GPU_SMALL, batch=8 | 1024×1024 | TBD | TBD | TBD |
| GPU_MEDIUM, batch=1 | 1024×1024 | TBD | TBD | TBD |

**Pivot ladder** (applied if p99 > 150 ms at GPU_SMALL, 1024×1024):
1. Reduce input resolution to 768×768
2. Upgrade `workload_size` from Small to Medium
3. Upgrade to Large
4. Switch to FP16-only weights (disable `torch.compile` if it adds overhead)

Final workload configuration used for demo: TBD

---

## Training time (E13)

Target: full 10-epoch frozen training completes in ≤20 minutes on a single H100.

| Run | Backbone | Epochs | LoRA | H100 wall time | mAP@50 |
|-----|----------|--------|------|----------------|--------|
| Baseline frozen | C-RADIOv4-SO400M | 10 | No | TBD | TBD |
| LoRA (STRETCH) | C-RADIOv4-SO400M | 10 | rank=8, alpha=32 | TBD | TBD |
| DINOv2 fallback | DINOv2-base | 10 | No | TBD | TBD |

---

## Detection accuracy (E1)

Target: frozen-backbone [email protected] on DENTEX val set ≥ 0.45.

Per-class AP@50 is produced by `eval.coco_metrics.evaluate_coco` (`per_class_AP50`)
and surfaced by `notebooks/09_eval_comparison.py` / `09b_eval_threshold_grid.py`.
The "push to 0.60" campaign rows track the per-level + resolution/schedule/anchor/
augmentation tuning (see [HPO.md](HPO.md#push-to-060--two-sequential-single-model-campaigns)).

| Model | Backbone | mAP@50 | mAP@50:95 | Caries AP@50 | Deep Caries AP@50 | Periapical AP@50 | Impacted AP@50 |
|-------|----------|--------|-----------|-------------|------------------|-----------------|---------------|
| Frozen head | C-RADIOv4-SO400M | TBD | TBD | TBD | TBD | TBD | TBD |
| LoRA rank=8 (STRETCH) | C-RADIOv4-SO400M | TBD | TBD | TBD | TBD | TBD | TBD |
| Per-level (baseline) | C-RADIOv4-SO400M | 0.5219 | TBD | 0.2102 (09b) | TBD | TBD | TBD |
| Per-level (baseline) | DINOv3-ViTL16 | 0.5181 | 0.285 | TBD (broken reg.) | TBD | TBD | TBD |
| Campaign best (`useful-mare-854`, 1024/100ep+aug) | C-RADIOv4-SO400M | 0.5648 | 0.291 | TBD (re-eval) | TBD | TBD | TBD |
| Campaign best (`victorious-goose-410`, 1280+aug) | DINOv3-ViTL16 | 0.5355 | 0.316 | TBD (re-eval) | TBD | TBD | TBD |
| Campaign candidate | C-RADIOv4-SO400M | TBD | TBD | TBD | TBD | TBD | TBD |
| Campaign candidate | DINOv3-ViTL16 | TBD | TBD | TBD | TBD | TBD | TBD |

Acceptance thresholds:
- mAP@50 ≥ 0.45 (frozen, MUST-SHIP)
- mAP@50 ≥ 0.55 (LoRA, STRETCH)
- Caries AP@50 ≥ 0.30 (anchor calibration validation, per C5b protocol)
- **mAP@50 ≥ 0.58 AND Caries AP@50 ≥ 0.30 — per-backbone gate for the 0.60 campaign**
  (finalize stage `<backbone>_detector@candidate`; target 0.60)

---

## Drift detection (E5, E6)

Target: drift score ratio ≥ 2.0× (synthetic-shifted vs. clean).

| Batch type | KNN distance (k=50) | MMD score | Ratio vs. clean | Bootstrap 95% CI |
|-----------|--------------------|-----------|-----------------|--------------------|
| Clean val (50 images) | TBD | TBD | 1.0× (baseline) | — |
| Synthetic shift (contrast=0.5, gamma=2.0) | TBD | TBD | TBD | TBD |

Bootstrap protocol: 1000 iterations, resample shifted batch with replacement, compute KNN distance each
iteration, report 2.5th and 97.5th percentiles. Pass if lower bound > reference mean (95% CI excludes 0).

---

## Vector Search recall (E7)

Target: top-10 recall ≥ 0.80 same-class on 50 val images against 705 train images.

| Index config | Query corpus | Recall@10 (same-class) | Mean query latency |
|-------------|-------------|----------------------|-------------------|
| HNSW+L2, dim=1152 | 50 val images | TBD | TBD |

Recall definition: for each of the 50 val queries, count same-diagnosis results in top-10.
Recall = (total same-class hits) / (50 × 10).

---

## GPU memory utilization (E15)

Target: ≤ 85% idle utilization on GPU_SMALL.

| Workload type | Model | Idle utilization | Peak (inference) |
|---------------|-------|-----------------|-----------------|
| GPU_SMALL | C-RADIOv4-SO400M + head | TBD | TBD |
| GPU_MEDIUM (if escalated) | C-RADIOv4-SO400M + head | TBD | TBD |

Measurement: `scripts/probe_endpoint_gpu.py` after 5 warm-up requests, monitoring via Mosaic AI
serving metrics dashboard.

---

## Embedding precompute throughput

| Dataset | Images | Backbone | Batch size | Total time | Images/sec |
|---------|--------|----------|-----------|-----------|-----------|
| DENTEX (train+val+test) | 1005 | C-RADIOv4-SO400M | 32 | TBD | TBD |

---

## How to reproduce these numbers

```bash
# 1. Deploy and train (full 10 epochs)
databricks bundle deploy -t prod
databricks bundle run train_detector -t prod

# 2. Run latency benchmark
# Open notebooks/07_latency_benchmark.py on the prod workspace and run all cells.
# Output: p50/p95/p99 table written to this file.

# 3. Run drift benchmark
databricks bundle run drift_monitor -t prod
# Query: SELECT * FROM ml.dais26_vfm.drift_scores ORDER BY timestamp DESC LIMIT 10

# 4. Run Vector Search recall benchmark
# Open notebooks/06_similarity_search_demo.py and run the recall evaluation cell.

# 5. GPU memory check
python scripts/probe_endpoint_gpu.py
```

All numbers in this file come from the prod workspace (`catalog=ml`). Dev numbers (`catalog=ml_dev`)
will differ slightly due to scale_to_zero behavior affecting cold-start measurements.
