# HPO log — DENTEX detector

Running log of the detector's hyperparameter-optimization journey on the DENTEX
diagnosis task (4 classes: Caries, Deep Caries, Periapical Lesion, Impacted; 705/50/250
train/val/test). Metric is COCO `val/mAP_50` on the 50-image val split, as logged by the
`Trainer` and ranked by the sweep (`SWEEP_PRIMARY_METRIC = val/best_mAP_50`).

Experiment: `/Users/puneet.jain@databricks.com/dais26-detector` (`experiment_id 1010776927889350`).

## TL;DR

- We went from a broken **~0.03 mAP@50** (random-box ceiling) to **0.335 mAP@50** by fixing
  the training recipe and **fully fine-tuning** the C-RADIOv4 backbone.
- That recipe has **saturated** (peaks ~epoch 40, then plateaus). The optimizer/encoder axis
  is exhausted.
- The next lever is **architectural** and still untested: the anchor generator emits every
  scale at every FPN level (a known bug), which starves the positive ratio. Literature shows a
  correctly-wired RetinaNet reaches **~0.60 AP50** on DENTEX, so there is ~0.27 mAP of headroom.

## Where mAP has moved

| Phase | Recipe | Best mAP@50 | Notes |
|-------|--------|-------------|-------|
| Initial (broken) | head-only-ish, default abs anchors, `lr=1e-3`, 10ep | **0.00 – 0.03** | smoke/initial runs (`burly-frog-514`=0.00, `blushing-bird-639`=0.031, dinov3 `wise-yak-538`=0.027) — the documented "~3% ceiling" |
| Sweep, frozen | `frozen`, `lr=1e-4`, 25ep | 0.21 | `masked-goat-391`=0.213, `mysterious-shoat-260`=0.210 |
| Sweep, LoRA | `lora`, `lr=1e-4`, 25ep | 0.23 | `indecisive-chimp-452`=0.228 |
| Sweep, full FT | `full`, `lr=1e-4`, 10/15/25/50ep | **0.335** | monotone in epochs: 0.214 → 0.239 → 0.262 → **0.335** (`traveling-rook-459`) |
| Calibrated abs anchors | `[83,215,273,546]`, high `lr` | 0.04 – 0.19 | large absolute anchors at every level *hurt* (`puzzled-shoat-871`=0.048, `selective-bug-480`=0.190 only at lower lr) |

### Current best run — `traveling-rook-459` (mAP@50 = 0.335)

```
backbone_name      = cradio_v4_so400m
backbone_mode      = full         # full fine-tune (NOT frozen)
lr                 = 1e-4
backbone_lr        = 1e-5         # discriminative LR
epochs             = 50
box_loss_weight    = 2.0
focal_gamma        = 2.5
weight_decay       = 1e-2
onecycle_pct_start = 0.3
img_size           = 1024
anchor_scales      = (default)    # absolute [16,32,64,128] at every level
```

Per-epoch `val/mAP_50` trajectory (abridged): 0.00 (e0-1) → 0.13 (e7) → 0.23 (e15) →
0.26 (e24) → 0.31 (e30) → **0.335 (e40 peak)** → 0.297 (e49). **Saturates / slightly
overfits after ~epoch 40** — more epochs will not close the gap.

## What we learned

1. **Full fine-tune > LoRA > frozen** for this dataset (0.335 vs 0.228 vs 0.213). The stale
   note in `notebooks/00_config.py` claiming `frozen` won is incorrect — `full` is the winner.
2. **Discriminative LR matters**: head `lr=1e-4` with backbone `backbone_lr=1e-5` is stable;
   `lr=5e-4` collapsed once anchors were perturbed.
3. **Calibrating *absolute* anchor scales does not fix the geometry.** `calibrate_anchors`
   produced `[83,215,273,546]` (skewed large), which — still applied at *every* FPN level —
   starved the small-lesion (P3) levels and hurt mAP. The problem is the per-level *assignment*,
   not the scale *values*.
4. **The recipe has saturated**; remaining gains are architectural.

## Root cause of the remaining gap (untested)

`AnchorGenerator` emits all `scales x aspect_ratios` (default 4x3=12) at **every** FPN level
P3–P6 (`src/dais26_dentex/models/detection_head.py`). So P3 (stride 8) gets 128px anchors and
P6 (stride 64) gets 16px anchors — most anchors are geometrically useless, the IoU matcher's
positive fraction collapses, focal loss is starved. Flagged as the MAJOR issue in
`src/dais26_dentex/models/arch_probe.py::KNOWN_ISSUES` and quantified live by
`notebooks/02a_arch_probe.py` (`all_scales_every_level=True`, tiny `positive_fraction`).

External cross-reference (DENTEX papers, arXiv:2305.19112; RetinaNet/Focal-Loss reviews):
RetinaNet baseline on DENTEX diagnosis ≈ **0.604 AP50**; top challenge teams ~0.68. RetinaNet
best practice = **9 anchors/level (3 octave scales {2^0, 2^(1/3), 2^(2/3)} x 3 ratios)** with
the **base area scaled per pyramid level**, not absolute sizes reused everywhere.

## Soon-to-be fixes (planned)

1. **Per-level anchor sizing** (MAJOR): anchor size = `stride x base_scale x octave_multiplier x ratio`,
   A=9/cell uniform across levels; keep the absolute mode behind a flag.
2. **Per-class NMS** (MEDIUM): `torchvision.ops.batched_nms` keyed by label so a lesion box
   inside a tooth box is not suppressed.
3. **Encode/decode clamp symmetry** (MINOR): clamp encoded `dw/dh` to the decode `exp` bound (4.0)
   so large-gt/small-anchor targets are reachable.
4. Config + manifest + serve plumbing so the anchor geometry is reproduced at eval/serve time.

See the full step-by-step in the plan; arch details in
[ARCHITECTURE.md](ARCHITECTURE.md) and the issue register in `models/arch_probe.py`.

## Next HPO sweep (post-fix design)

Encoder axis is settled and the optimizer region is known, so the next sweep spends its budget
on the newly-unlocked anchor geometry.

**Pinned** (from prior sweep): `backbone_mode=full`, `backbone_lr=1e-5`, `onecycle_pct_start=0.3`,
`weight_decay=1e-2`, `img_size=1024`, `nms_per_class=True`, `anchor_layout=per_level`.

**Swept:**

| Field | Choices | Rationale |
|-------|---------|-----------|
| `anchor_base_scale` | `3.0, 4.0, 5.0` | per-level base = stride x base_scale |
| `aspect_ratios` | `[0.5,1,2]`, `[0.33,0.5,1,2,3]` | impacted teeth are elongated |
| `lr` | `1e-4, 2e-4` | 1e-4 won; high lr collapsed with anchor changes |
| `box_loss_weight` | `1.0, 2.0` | 2.0 won previously |
| `focal_gamma` | `2.0, 2.5` | 2.5 won previously |

**Budget:** `strategy=random`, `max_trials=8–10`, `trial_epochs=25`, winner `TRAIN_EPOCHS=50`,
primary metric `val/best_mAP_50`. Runs on `GPU_8xH100` with the 8h job timeout.

**Acceptance:** beat **0.335** → **≥ 0.45** (MUST-SHIP, [BENCHMARKS.md](BENCHMARKS.md)) →
**~0.60** stretch (RetinaNet-parity). `Caries AP@50 ≥ 0.30`.

## How to reproduce the numbers above

```python
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("databricks")
c = MlflowClient()
runs = c.search_runs(
    ["1010776927889350"],
    filter_string="metrics.`val/best_mAP_50` > 0",
    order_by=["metrics.`val/best_mAP_50` DESC"],
    max_results=10,
)
for r in runs:
    print(r.info.run_name, r.data.metrics["val/best_mAP_50"], r.data.params.get("backbone_mode"))
```
