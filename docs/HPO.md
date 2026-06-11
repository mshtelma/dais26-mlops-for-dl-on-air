# HPO log — DENTEX detector

> Stage definitions now live as typed package data in
> `dais26_dentex.config.campaigns.CAMPAIGN_STAGES` (validated by unit tests);
> the best-known final recipes are `dais26_dentex.config.recipes.RECIPES`.
> Launch any stage via the DAB `campaign_sweep` job or
> `sgcli/workload_sweep.yaml` — both run `train.sweep_runner.SweepRunner`.

Running log of the detector's hyperparameter-optimization journey on the DENTEX
diagnosis task (4 classes: Caries, Deep Caries, Periapical Lesion, Impacted; 705/50/250
train/val/test). Metric is COCO `val/mAP_50` on the 50-image val split, as logged by the
`Trainer` and ranked by the sweep (`SWEEP_PRIMARY_METRIC = val/best_mAP_50`).

Experiment: `/Users/puneet.jain@databricks.com/dais26-detector` (`experiment_id 1010776927889350`).

## TL;DR

- We went from a broken **~0.03 mAP@50** (random-box ceiling) to **0.335 mAP@50** by fixing
  the training recipe and **fully fine-tuning** the C-RADIOv4 backbone.
- That recipe **saturated** (peaks ~epoch 40, then plateaus): the optimizer/encoder axis was
  exhausted. The next lever was **architectural** — the anchor generator emitted every scale at
  every FPN level (a known bug), starving the positive ratio.
- **The architectural fix landed and won: `upbeat-mink-783` = 0.5219 mAP@50** (per-level anchors
  + per-class NMS, full fine-tune, 50ep). That is **+0.187 over the prior best**, clears the
  **0.45 must-ship** bar, and reaches ~87% of the **0.60** RetinaNet-parity stretch. Registered as
  `mlops_pj.dais26_vfm.cradio_detector` **v10**, aliased `@candidate`.
- **Push-to-0.60 campaign status (current best):** **C-RADIO 0.5931** (`dazzling-mole-850`,
  plain 150ep) and **DINOv3 0.5738** (`capricious-hound-240`, multi-layer fusion + 150ep).
  C-RADIO's lever is the **schedule** (150ep = +0.028); DINOv3's was **architectural** —
  multi-layer ViT feature fusion broke a hard ~0.535 ceiling that no knob could, and
  compounding fusion with the long schedule took it to **0.574** (+0.056 over its 0.518
  baseline). Round-4 finalize (compounding) is **done**: the fusion×schedule compound *won* for
  DINOv3, but GIoU+oversampling *regressed* C-RADIO at 150ep — so the best C-RADIO remains the
  plain `dazzling-mole-850`. Both still ~0.01–0.04 short of 0.60. Full detail below in "Round 3
  returns" / "Round 4 returns" and the results-tracking table.

## Where mAP has moved

| Phase | Recipe | Best mAP@50 | Notes |
|-------|--------|-------------|-------|
| Initial (broken) | head-only-ish, default abs anchors, `lr=1e-3`, 10ep | **0.00 – 0.03** | smoke/initial runs (`burly-frog-514`=0.00, `blushing-bird-639`=0.031, dinov3 `wise-yak-538`=0.027) — the documented "~3% ceiling" |
| Sweep, frozen | `frozen`, `lr=1e-4`, 25ep | 0.21 | `masked-goat-391`=0.213, `mysterious-shoat-260`=0.210 |
| Sweep, LoRA | `lora`, `lr=1e-4`, 25ep | 0.23 | `indecisive-chimp-452`=0.228 |
| Sweep, full FT | `full`, `lr=1e-4`, 10/15/25/50ep | **0.335** | monotone in epochs: 0.214 → 0.239 → 0.262 → **0.335** (`traveling-rook-459`) |
| Calibrated abs anchors | `[83,215,273,546]`, high `lr` | 0.04 – 0.19 | large absolute anchors at every level *hurt* (`puzzled-shoat-871`=0.048, `selective-bug-480`=0.190 only at lower lr) |
| **Post-fix sweep** (`per_level` anchors + per-class NMS) | `base_scale=3.0`, `ar=[0.5,1,2]`, `lr=2e-4`, full FT | **0.522** | even a 25ep trial beat the old best (`treasured-hog-714`=0.412); 50ep retrain **`upbeat-mink-783`=0.5219** (v10, `@candidate`) |

### Current best run — `upbeat-mink-783` (mAP@50 = 0.5219)

Winner of the post-fix sweep `hpo-sweep-cradio_v4_so400m` (ran 06-01 22:22 → 06-02 01:55).
Same encoder recipe as the prior best; the deltas are the **architectural fix** (per-level
anchors + per-class NMS) plus the swept geometry/lr.

```
backbone_name      = cradio_v4_so400m
backbone_mode      = full         # full fine-tune (NOT frozen)
lr                 = 2e-4         # swept; 2e-4 > 1e-4 once anchors were fixed
backbone_lr        = 1e-5         # discriminative LR
epochs             = 50
box_loss_weight    = 2.0
focal_gamma        = 2.5
weight_decay       = 1e-2
onecycle_pct_start = 0.3
img_size           = 1024
anchor_layout      = per_level    # NEW: size = stride x base_scale x octaves x ratio
anchor_base_scale  = 3.0          # swept; 3.0 > 4.0/5.0
aspect_ratios      = [0.5,1,2]    # swept; 3 ratios > the wider 5-ratio set
nms_per_class      = True         # NEW: batched_nms keyed by label
```

Registered as `mlops_pj.dais26_vfm.cradio_detector` **v10**, aliased `@candidate`.

### Prior best (pre-fix) — `traveling-rook-459` (mAP@50 = 0.335)

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
anchor_scales      = (default)    # absolute [16,32,64,128] at every level (the bug)
```

Per-epoch `val/mAP_50` trajectory (abridged): 0.00 (e0-1) → 0.13 (e7) → 0.23 (e15) →
0.26 (e24) → 0.31 (e30) → **0.335 (e40 peak)** → 0.297 (e49). **Saturated / slightly
overfit after ~epoch 40** with the buggy absolute anchors — more epochs did not close the
gap; the **anchor geometry** did.

### Training schedule vs literature

Our `epochs=50` is an **internal, ViT-fine-tune choice**, not paper-derived. The DENTEX
baseline paper (arXiv:2303.06500, Table 2) measures detector training in **iterations, not
epochs**:

| Model | Backbone | Schedule | LR |
|-------|----------|----------|----|
| RetinaNet (the 0.604 AP50 baseline) | ResNet-101 | **40,000 iters** | 0.01 |
| Faster R-CNN | ResNet-101 | 40,000 iters | 0.02 |
| DiffusionDet / their method | FPN-Swin | 40,000 iters | 2.5e-5 |
| DETR | ResNet-50 | **300 epochs** | 1e-4 |

(batch 16, single A6000, random crops.) At 705 train images / batch 16 (~44 iters/epoch),
40k iters is **~900 epoch-equivalents** (loose, since they train on random crops) — so the
paper trains **far longer than 50**, and none of the detectors are quoted at 50 epochs.

Why we still use 50: those baselines train a detector head on top of an **ImageNet ResNet at
high LR (0.01)**, which needs a very long schedule to converge. We instead **fine-tune a strong
pretrained ViT (C-RADIOv4) at low LR (1e-4 head / 1e-5 backbone)**, which converges in far fewer
epochs — empirically peaking ~e40 (above). So 50 is justified for *our* architecture, not by
matching the paper's iteration budget.

**Caveat:** the per-level anchor fix changes the matcher/loss landscape (positives move P3→P4),
so the ~e40 saturation point may shift. The schedule should be **re-confirmed post-fix** rather
than assumed — see the longer-epoch arm under "Next HPO sweep".

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

## Root cause of the remaining gap (confirmed + fixed)

> **Outcome:** fixing this lifted mAP@50 from 0.335 → **0.522** (`upbeat-mink-783`). The
> diagnosis below is what the fix addressed.


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

## Phase 0 — measured arch-probe baseline

Ran `probe_detection_model` on the **real DENTEX val batch** (first 4 images,
23 GT boxes, boxes scaled to 1024px exactly as `get_val_transforms` does). The
probe's metrics depend only on the anchor grid + GT box sizes (not backbone
weights), so this is faithful on CPU. Captures the *before* reference and the
*after* of the implemented fix on the same batch.

| Metric | BEFORE (`absolute`, class-agnostic NMS) | AFTER (`per_level`, per-class NMS) |
|--------|------------------------------------------|-------------------------------------|
| total anchors | 261,120 | 195,840 |
| anchors/cell | 12 | 9 |
| `all_scales_every_level` | **True** (MAJOR smell) | **False** (resolved) |
| positives total | 810 | 588 |
| positive fraction | 0.0776% | 0.0751% |
| positives per level | p3=611, p4=159, p5=34, p6=6 | p3=120, **p4=458**, p5=10, p6=0 |
| `delta_overflow_fraction` | 0.0 | 0.0 |
| NMS mode | class-agnostic | per-class (`batched_nms`) |

**Read:** the headline change is the **positive distribution**, not the raw
fraction. Before, 75% of positives pile onto P3 (stride-8, finest grid) because
that level happens to host the small anchors that best-match objects — the
matcher is forced onto the wrong pyramid level. After per-level sizing, 78% of
positives move to **P4 (stride-16)**, the level whose base size (`16 x 4 = 64px`,
x octaves) actually matches DENTEX tooth/lesion sizes at 1024px. The head no
longer learns from geometrically-misassigned anchors, and 65k fewer anchors are
generated. `positive_fraction` looks similar only because force-best-anchor-per-gt
guarantees a match regardless of layout; the true mAP impact is measured by the
Phase 5 training A/B. `delta_overflow=0.0` on this batch means the clamp-symmetry
fix is a safety net here (no large-gt/small-anchor extreme in the first 4 images)
rather than a mover of this metric.

Reproduce: pull `val.json` from the UC Volume and run the probe locally (CPU,
fake backbone, 64x64 grid), or run `notebooks/02a_arch_probe.py` on a GPU node
for the full real-backbone pass.

## Fixes (shipped — drove 0.335 → 0.522)

1. **Per-level anchor sizing** (MAJOR): anchor size = `stride x base_scale x octave_multiplier x ratio`,
   A=9/cell uniform across levels; keep the absolute mode behind a flag.
2. **Per-class NMS** (MEDIUM): `torchvision.ops.batched_nms` keyed by label so a lesion box
   inside a tooth box is not suppressed.
3. **Encode/decode clamp symmetry** (MINOR): clamp encoded `dw/dh` to the decode `exp` bound (4.0)
   so large-gt/small-anchor targets are reachable.
4. Config + manifest + serve plumbing so the anchor geometry is reproduced at eval/serve time.

See the full step-by-step in the plan; arch details in
[ARCHITECTURE.md](ARCHITECTURE.md) and the issue register in `models/arch_probe.py`.

## Post-fix sweep — RESULT (`hpo-sweep-cradio_v4_so400m`)

Encoder axis was settled and the optimizer region known, so this sweep spent its budget on the
newly-unlocked anchor geometry. **It worked.**

**Pinned:** `backbone_mode=full`, `backbone_lr=1e-5`, `onecycle_pct_start=0.3`,
`weight_decay=1e-2`, `img_size=1024`, `nms_per_class=True`, `anchor_layout=per_level`.

**Swept + results** (8 trials @ 25ep, `strategy=random`, seed=42, `GPU_8xH100`):

| Trial | mAP@50 | `anchor_base_scale` | `aspect_ratios` | `lr` |
|-------|--------|---------------------|-----------------|------|
| **`treasured-hog-714` (winner cfg)** | **0.4121** | 3.0 | `[0.5,1,2]` | 2e-4 |
| `grandiose-snail-270` | 0.3951 | 3.0 | `[0.5,1,2]` | 1e-4 |
| `secretive-fly-507` | 0.3775 | 5.0 | `[0.5,1,2]` | 1e-4 |
| `nimble-newt-955` | 0.3763 | 3.0 | `[0.5,1,2]` | 1e-4 |
| `omniscient-calf-908` | 0.3660 | 4.0 | `[0.33,0.5,1,2,3]` | 1e-4 |
| `entertaining-rat-187` | 0.3481 | 5.0 | `[0.5,1,2]` | 1e-4 |
| `merciful-trout-363` | 0.3451 | 4.0 | `[0.33,0.5,1,2,3]` | 1e-4 |
| `melodic-shad-574` | 0.3429 | 4.0 | `[0.33,0.5,1,2,3]` | 1e-4 |

Signal: **smaller `base_scale=3.0` + the standard 3 aspect ratios** beat the wider 5-ratio set;
`lr=2e-4` edged `1e-4`. Even the worst trial (0.343) is within noise of the *old* 50ep best.

**Winner retrain** (50ep): **`upbeat-mink-783` = 0.5219**, registered v10 `@candidate`. (This run
used the single-schedule retrain; the 50-vs-100ep schedule arm below lands on the next sweep.)

**Longer-epoch arm (schedule hedge).** Trials stay cheap at `trial_epochs=25` for *ranking only*.
Because the per-level anchor fix reshapes the loss landscape, the prior ~e40 saturation point
may move, so the winner is retrained at **both 50 and `TRAIN_EPOCHS_LONG=100` epochs** and we keep
the better by `val/best_mAP_50`. This is safe against overfitting: the `Trainer` already tracks the
best checkpoint, so a 100-epoch run that peaks earlier still reports (and registers) its peak rather
than the over-trained final epoch. Only one extra full-length run is added — the cheap trials are
unchanged. (Historical wiring: `TRAIN_EPOCHS_LONG` in the then-monolithic `00_config.py`; today the
schedule arm is `schedule_epochs` on the stage in `config/campaigns.py`, executed by
`train/sweep_runner.py`. Applied to the next sweep, not the in-flight run.)

**Acceptance:** beat **0.335** ✅ → **≥ 0.45** MUST-SHIP ✅ (**0.519** re-eval, [BENCHMARKS.md](BENCHMARKS.md))
→ **~0.60** stretch (RetinaNet-parity) — *remaining headroom ~0.08*. Per-class confirmed via
[09_eval_comparison.py](../notebooks/09_eval_comparison.py) (see "Serving re-eval" below): all four
classes lifted, but **`Caries AP@50 = 0.205` still misses the `≥ 0.30` sub-bar** — the open next lever.

## Serving re-eval + a preprocessing bug it caught (`09_eval_comparison.py`)

Re-evaluating the registered winner through the **serving pyfunc** (`DetectorPyfunc`) on the val
split — apples-to-apples, independent of train-time logged metrics — first surfaced a **serving
bug**, then (once fixed) confirmed the 0.52.

**The bug (train/serve preprocessing mismatch).** Training resizes with an aspect-preserving
**letterbox** (`data/transforms.py::_resize_and_pad`: longest-side resize + bottom-right zero-pad),
but the pyfunc decoded with an **anisotropic squash** (`img.resize((input_size, input_size))`) and
inverted predicted boxes with per-axis factors. On DENTEX's ~2:1 panoramics the served model saw a
horizontally stretched image and boxes mapped back at the wrong scale, so IoU collapsed for the
dominant (large) boxes. The smoking gun was the per-area split: **medium AP 0.90 vs large AP 0.05**.

**The fix** (`serve/detector_pyfunc.py`, guarded by `test_letterbox_decode_and_inverse_roundtrip`):
decode now letterboxes exactly like training, and the box inverse is a **single uniform scale**
(`max(orig)/input_size`) + clip to original bounds. No re-registration was needed — the
models-from-code script imports `DetectorPyfunc` from the wheel, so rebuilding/deploying the wheel
propagated the fix to v10 in place.

**Re-eval result** (val, 50 imgs, `cradio_detector@champion` = v10):

| Metric | Before (squash bug) | After (letterbox fix) | Train-time |
|--------|---------------------|-----------------------|------------|
| mAP@50 | 0.176 | **0.519** | 0.522 |
| mAP@.50:.95 | 0.051 | 0.299 | 0.308 |
| mAP@75 | 0.018 | 0.327 | 0.341 |
| AP (large area) | 0.048 | **0.299** | — |
| AR@100 | ~0.18 | 0.649 | — |

**Per-class AP@50:**

| Class | Before (squash) | After (letterbox) |
|-------|-----------------|-------------------|
| Caries | 0.051 | **0.205** ⚠️ (< 0.30) |
| Deep Caries | 0.111 | **0.500** |
| Periapical Lesion | 0.007 | **0.429** |
| Impacted | 0.494 | **0.657** |

**Takeaways:** (1) the 0.52 is real — the serving path now reproduces it (0.519 vs 0.522). (2) Any
endpoint serving this pyfunc was mis-detecting on non-square images before this fix. (3) `Caries`
(smallest/subtlest class) is the remaining weak spot at 0.205 — the natural next lever is a
Caries-targeted tweak (finer low-level anchors / class-balanced focal weighting). (4) DINOv3
re-eval is ~0.02 (broken — see the DINOv3 RCA below).

## DINOv3 A/B — does the fix transfer off C-RADIO?

Every number above is on **C-RADIOv4**. The detection-head fix (per-level anchors
+ per-class NMS + clamp symmetry) is backbone-agnostic and plumbed end-to-end, so
it is *compatible* with DINOv3 — but the uplift on DINOv3 was untested (DINOv3
appears only once, in the broken phase: `wise-yak-538`=0.027). DINOv3 is a
single-scale ViT (patch16): the FPN synthesizes P3 (a bilinear upsample) and only
P4 carries native stride-16 resolution, so concentrating positives on P4 — exactly
what `per_level` does — is expected to help DINOv3 *at least* as much as C-RADIO.

### Detour: DINOv3 didn't train at all (precision + normalization bug)

The first A/B attempt (job `843732378571873`) was **inconclusive**: both arms hit
**0.0 mAP with dead-flat loss across all 50 epochs**. Two root causes, both now
fixed (plan: *Fix DINOv3 training collapse*):

1. **Precision (decisive).** The trainer ran DINOv3 under **fp16 autocast +
   GradScaler**. DINOv3's RoPE/LayerScale encoder NaNs in fp16, so the scaler
   skipped *every* `optimizer.step()` → flat loss. The plan expected **bf16** to
   fix it, but an empirical smoke showed **bf16 also NaNs the forward at step 0**
   (`cls_loss`/`box_loss`=nan, `amp_scale`=1.0) for this detector stack. Only
   **fp32** (autocast disabled) trains. Precision is now backbone-aware via
   `cfg.amp_dtype` (`auto` → fp32 for DINOv3, fp16 for C-RADIO); a per-epoch
   `train/grad_norm` + `train/amp_scale` log and a `flat_loss_patience` guard
   fail a dead run fast.
2. **Normalization (accuracy cap).** Inputs were hardcoded to **CLIP** mean/std for
   every backbone; DINOv3 expects **ImageNet**. Norm is now carried on
   `BackboneInfo.image_mean/std` (CLIP for C-RADIO, ImageNet for DINOv2/v3),
   recorded in the manifest, and used everywhere (train, eval, serve, embeddings).

**Smoke (3 epochs, fp32, `per_level` treatment, batch 4):** `train/loss`
**1.39 → 0.80 → 0.77**, `val/mAP_50` **0.07 → 0.22 → 0.20**, finite `grad_norm`.
The collapse was a precision/normalization bug, **not the DINOv3 backbone** — with
the fix it learns. The full A/B below runs in **fp32 at batch 4** (fp32 activations
are ~2× bf16, which already sat at ~68% of an 80 GB H100 at batch 8).

To turn that into a measured number, a **controlled A/B** on `dinov3_vitl16`
(pinning the backbone locally; it does NOT flip the global `BACKBONE`) was run.
Two runs identical except the change bundle, same
`base_seed=42`, `register_model=False`, full-length (`TRAIN_EPOCHS`), on the settled
recipe (`backbone_mode=full`, `backbone_lr=1e-5`, `lr=1e-4`, `weight_decay=1e-2`,
`onecycle_pct_start=0.3`, `img_size=1024`):

- **Arm A (baseline):** `anchor_layout=absolute`, class-agnostic NMS.
- **Arm B (treatment):** `anchor_layout=per_level` (`anchor_base_scale=4.0`), per-class `batched_nms`.

Phase 1 of that A/B first runs `probe_detection_model` on the **real DINOv3
encoder** for each arm (confirms the 64x64 token grid aligns and shows positives
moving off P3 onto P4); Phase 2 dispatches the two `@distributed` 8xH100 arms and
reads back `val/best_mAP_50`; Phase 3 prints the verdict. The decision rule:
treatment must beat the baseline arm and clear the must-ship bar (`mAP@50 ≥ 0.45`,
`Caries AP@50 ≥ 0.30`); register the treatment arm and run
[09_eval_comparison.py](../notebooks/09_eval_comparison.py) for the apples-to-apples
per-class re-eval through the serving pyfunc.

Phase-1 probe rows below are the **real-DINOv3-encoder** numbers (1024px, 64×64
token grid); the mAP rows are from the full A/B (re-run in fp32 after the fix).

| Metric | Arm A — baseline (`absolute`, class-agnostic NMS) | Arm B — treatment (`per_level`, per-class NMS) |
|--------|---------------------------------------------------|------------------------------------------------|
| anchors/cell | 12 | 9 |
| `all_scales_every_level` (Phase-1 probe) | True | False |
| positives_per_level (Phase-1 probe) | P3=611, P4=161, P5=34, P6=6 (total 812) | P3=120, **P4=460**, P5=10, P6=0 (total 590) |
| `val/best_mAP_50` | **0.383** (`flawless-hare-837`, fp32) | **0.518** (`chill-robin-965`, fp32) |
| served mAP@50 (pyfunc re-eval) | _not re-eval'd_ | **0.506** (\|Δ\| vs trainer 0.012) |
| `Caries AP@50` (pyfunc re-eval) | _not re-eval'd_ | **0.195** |

The Phase-1 probe confirms the mechanism: the treatment moves the positive mass
**off P3 (611→120) onto P4 (161→460)** — the only level that carries DINOv3's
native stride-16 resolution — exactly the redistribution that lifted C-RADIO.

**Verdict: per-level anchors win on DINOv3 — `val/best_mAP_50` 0.383 → 0.518
(Δ = +0.136, +35% relative).** The full A/B ran in fp32 (parent run
`anchor-ab-dinov3_vitl16`: `ab_baseline_mAP_50=0.383`, `ab_treatment_mAP_50=0.518`,
`ab_delta_mAP_50=0.136`). Both arms trained cleanly (final `grad_norm` ~7,
`amp_scale`=1) — versus **0.0 mAP / flat loss ~2.4** for the pre-fix collapsed runs
(`wise-squirrel-746`/`defiant-sponge-982`). The anchor fix is now validated on
**both** backbones at ~0.52 mAP@50 (C-RADIO 0.335→0.522, DINOv3 0.383→0.518),
confirming it as a backbone-agnostic win.

### Serving re-eval — pyfunc behaves correctly

The DINOv3 treatment arm (`chill-robin-965`) was registered as
`dinov3_detector@candidate` (from its run artifact, no retrain) and the C-RADIO
`per_level` winner aliased `cradio_detector@candidate` (v10), then both were
re-scored on the **same `val` split through the real `DetectorPyfunc`**
(`load_context` + `predict`: manifest-driven normalization, letterbox, box
rescale). Served mAP@50 matches the trainer's `val/best_mAP_50` within tolerance,
so serving normalization/post-processing is correct:

| Backbone (`per_level`) | served mAP@50 | trainer `val/best_mAP_50` | \|Δ\| | verdict |
|---|---|---|---|---|
| C-RADIOv4 | 0.519 | 0.522 | 0.003 | PASS |
| DINOv3 | 0.506 | 0.518 | 0.012 | PASS |

Per-class AP@50 (pyfunc, `val`): **C-RADIO** Caries 0.205 / Deep Caries 0.501 /
Periapical 0.429 / Impacted 0.657; **DINOv3** Caries 0.195 / Deep Caries 0.427 /
Periapical 0.364 / Impacted 0.691. Caries (~0.20 both) remains the hard class,
below the `≥0.30` must-ship bar — the next accuracy lever, independent of the
(now-validated) anchor + serving fixes.

> Note: `09_eval_comparison.py` loads each detector via `mlflow.pyfunc.load_model`,
> which downloads the **entire bundled model artifact (~5 GB HF cache + 1.24 GB
> state) per backbone** before printing anything (~82 min, looks hung). The numbers
> above were produced by loading the backbone from the shared `model_cache` volume
> instead (~13 min, instrumented). Worth not bundling the full cache into every
> registered model, or having 09 read the backbone from the volume.

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

## Push to 0.60 — two sequential single-model campaigns

Both backbones sit at ~0.52 `val/best_mAP_50` after the per-level anchor fix. The
goal is **0.60 on each model individually (no ensemble)** — a ~0.08 lift on 705
train images. The credible path is the set of levers we have **never** swept:
input resolution, schedule length, anchor density, and augmentation strength.
DINOv3 and C-RADIO are tuned in **separate sequential campaigns** because their
failure modes differ: C-RADIO is *schedule-starved* (final-epoch mAP == best at
e50 → still rising), DINOv3 is *regularization-starved* (best at ~e35, then
overfits).

### Stage 0 plumbing (landed)

New `TrainerConfig` fields, all defaulted to the legacy behavior so existing runs
are byte-identical:

- **Eval-time thresholds** `score_threshold` (0.05), `nms_iou_threshold` (0.5),
  `max_detections` (100) — previously hardcoded `DetectionModel` args. Threaded
  through `models/builder.py` and written into the manifest's `DetectorSpec`, so
  serve + eval reproduce the chosen values. Gridable for free post-hoc (no
  retrain) — see below.
- **Augmentation** `aug_hflip_prob`, `aug_jitter_prob`, `aug_jitter_scale`,
  `aug_rotation_deg`, `aug_multiscale_range` — wired into `data/transforms.py`
  (`get_train_transforms`) and the trainer's train loader. Multi-scale jitters
  the longest-side resize within `[lo,hi]` then pads back to `img_size` so tensor
  shapes stay uniform for batching; rotation updates boxes via tv_tensors.
- **`grad_accum_steps`** — accumulate grads over N micro-batches before one
  optimizer step (the OneCycle scheduler now steps once per *optimizer* step).
  Lets fp32 DINOv3 keep a large effective batch at 1280px with `batch_size=2`.

Campaign stages live in `CAMPAIGN_STAGES` (`notebooks/00_config.py`); the 02b
sweep harness honors a `sweep_stage` widget / job parameter. New launchable jobs:
`resources/jobs/campaign_sweep.yml` (parametrized by `sweep_stage`,
`max_concurrent_runs: 2` so two stages can share the GPU pool) and
`resources/jobs/eval_threshold_grid.yml` (the free grid, single A10,
`timeout_seconds: 14400` — 36 forward-only val passes run ~95–120 min).

The free grid (`09b_eval_threshold_grid.py`) **persists** its result: best
thresholds + Caries AP@50 per backbone are logged to an MLflow run
(`eval-threshold-grid`) and the full grid is returned via `dbutils.notebook.exit`
(readable with `jobs get-run-output`). The job-run HTML export strips cell stdout,
so without this the numbers are unrecoverable — an early run was lost this way.

### Runbook (per stage)

```bash
databricks bundle deploy -t dev            # ship code + new jobs + wheel

# Stage 0: bank free threshold gains + record Caries AP@50 baseline (cheap, A10)
databricks bundle run eval_threshold_grid -t dev

# Each campaign stage (run in order; ~hours each on 8xH100):
databricks bundle run campaign_sweep -t dev -- --params sweep_stage=dinov3_s1
#   → read the winner from MLflow (parent run "hpo-sweep-dinov3_vitl16"), then
#     UPDATE the next stage's pinned img_size/lr/pct_start (s2) or anchors/loss
#     (s3) in CAMPAIGN_STAGES to that winner before launching it.
databricks bundle run campaign_sweep -t dev -- --params sweep_stage=dinov3_s2
databricks bundle run campaign_sweep -t dev -- --params sweep_stage=dinov3_s3
databricks bundle run campaign_sweep -t dev -- --params sweep_stage=dinov3_s4   # registers @candidate
databricks bundle run eval_comparison -t dev                                    # per-class re-eval
# repeat with cradio_s1 … cradio_s4
```

Stages **s1–s3** retrain the per-stage winner full-length only to **measure** the
gate metric (`register_winner=False`); only the finalize stage **s4** registers
`<backbone>_detector@candidate`. The pinned values in s2/s3/s4 are **seeded with
the current best and MUST be updated to the prior stage's MLflow winner** before
launch — the chain is inherently sequential.

### Gate (before promoting / moving to the next backbone)

Per backbone, the finalized model must clear **both**: `val/mAP@50 ≥ 0.58` AND
`Caries AP@50 ≥ 0.30` (the hard class, per the C5b protocol). The 0.58 interim
gate leaves headroom toward the 0.60 target; if a backbone tops out ~0.56–0.57
after all stages, push resolution to 1536 / a longer schedule rather than
ensembling (the no-ensemble constraint).

### Results tracking (fill as stages complete)

| Stage | Backbone | What swept | Winner run | `val/best_mAP_50` | Caries AP@50 | Verdict |
|-------|----------|------------|-----------|-------------------|--------------|---------|
| baseline | C-RADIO | — | `upbeat-mink-783` | 0.5219 | _pending (09)_ | reference |
| baseline | DINOv3 | — | `chill-robin-965` | 0.5181 | _pending (09)_ | reference |
| Stage 0 grid | C-RADIO (registered) | score/nms/max_det | `09b` best score=0.01/nms=0.4/md=300 | 0.5239 (vs 0.519 default) | **0.2102** | free +0.005; Caries < 0.30 gate |
| Stage 0 grid | DINOv3 (registered) | score/nms/max_det | `09b` | **0.0266** | 0.0697 | ⚠ **registered serving model is BROKEN** (re-register needed) |
| dinov3_s1 | DINOv3 | img_size × lr × pct_start × {50,75}ep | `intrigued-stork-789` (75ep), `rumbling-boar-852` (50ep) | 0.53 (noisy peak; ~0.50 stable) | _pending_ | **overfit — 1024/no-aug caps ~0.50; resolution never sampled** |
| dinov3_regres | DINOv3 | 1280px + multi-scale/rotation/jitter, 75ep | `victorious-goose-410` | 0.5355 (50:95=**0.317**) | _pending_ | **overfit fixed — curve healthy, but ceiling ~0.53; need anchor/loss next** |
| dinov3_s2 | DINOv3 | base_scale, octaves, AR, fg, box, blr @1280+aug | `serious-kit-244` (fg=2.5, 3-oct, AR[.5,1,2]) | 0.5256 (50:95=0.301) | _pending_ | **no gain — below the regres base (0.5355); anchor/loss lever tapped** |
| dinov3_s3 | DINOv3 | multiscale, rotation, jitter | _superseded by regres_ | — | — | folded into `dinov3_regres` |
| dinov3_res1536 | DINOv3 | **step-change**: 1536px + aug, 100ep, b1×accum4 | `vaunted-crane-68` | 0.5326 (50:95=**0.3238**) | _pending_ | **1536 ≈ 1280; no mAP@50 gain — ceiling confirmed ~0.53** |
| dinov3_falpha | DINOv3 | focal_alpha on `victorious-goose-410` base | `casual-jay-300` (75ep) | 0.5163 | _pending_ | **no gain — below fusion + below regres base** |
| **dinov3_fusion** | DINOv3 | **multi-layer ViT fusion** (L6/12/18/24) @1280+aug, 75ep | **`delicate-stork-19`** | **0.5504** (50:95=**0.321**) | _pending_ | **best DINOv3 — broke the ~0.535 ceiling (+0.015); architectural lever, not a knob** |
| **dinov3_final** | DINOv3 | fusion + **150ep** (smooth_l1), register | **`capricious-hound-240`** (v7) | **0.5738** (50:95=0.333) | _pending_ | **best DINOv3 — fusion × long schedule compounded (+0.023 over fusion@75ep)** |
| dinov3_final_giou | DINOv3 | fusion + 150ep + **GIoU**, register | `rebellious-gnu-395` (v8, `@candidate`) | 0.5704 (50:95=**0.340**) | _pending_ | GIoU −0.003 mAP@50 vs smooth_l1 but **best 50:95**; localization-leaning |
| cradio_s1 | C-RADIO | {75,100}ep × img_size × lr × pct_start (+mild aug) | `useful-mare-854` (1024px, 100ep, lr2e-4, pct0.2) | 0.5648 (50:95=0.291) | _pending_ | breakout +0.043 over baseline; still rising at e100 → schedule/aug is the lever |
| cradio_s2 | C-RADIO | base_scale, octaves, AR, fg, box, fa, blr | `polite-frog-337` (100ep) | 0.5489 | _pending_ | **no gain — below the `useful-mare-854` base; anchor/loss tapped for C-RADIO too** |
| **cradio_long** | C-RADIO | **150ep** schedule on the `useful-mare-854` recipe | **`dazzling-mole-850`** | **0.5931** (50:95=0.304) | _pending_ | **best overall — +0.028 over s1; schedule is C-RADIO's dominant lever** |
| cradio_giou | C-RADIO | **GIoU** box loss + **Caries ×2** oversample, 100ep | `amazing-moth-867` | 0.5674 (50:95=0.299) | _pending_ | +0.003 mAP@50 over s1; localization lever, additive |
| cradio_final | C-RADIO | 150ep + GIoU + Caries×2, register | `resilient-moth-415` (v11, `@candidate`) | 0.5697 (50:95=0.288) | _pending_ | **regressed −0.023 vs plain 150ep — GIoU+oversample *hurt* at 150ep; best C-RADIO stays `dazzling-mole-850` 0.5931** |

### Live status & deviations from the linear plan

The campaign did not run as a strict s1→s2→s3→s4 chain; the data forced two
adaptations (both detailed below):

- **`dinov3_regres`** replaced a clean s1→s2 hop: Stage-1 overfit at 1024/no-aug,
  so resolution (1280) + the planned s3 augmentation were pulled forward together.
- **`dinov3_s2`** (anchor/loss on the 1280+aug base) gave no gain → DINOv3 declared
  capped ~0.535 (see "DINOv3 ceiling" below).
- **Round 2 (done):** `dinov3_res1536` confirmed the DINOv3 ceiling (0.5326);
  `cradio_s1` broke C-RADIO out to **0.5648** (+0.043); `09b` banked thresholds +
  the 0.21 Caries baseline and exposed the broken registered DINOv3.
- **Round 3 (done):** ran a config-track (Track A) and an architectural-track
  (Track B) in parallel — see "Round 3 returns" below. Headlines: **C-RADIO 150ep
  `dazzling-mole-850` = 0.5931** (new best overall) and **DINOv3 fusion
  `delicate-stork-19` = 0.5504** (broke the ~0.535 DINOv3 ceiling, the only lever
  that did). Anchor/loss and focal_alpha gave nothing on either backbone.
- **Round 4 (done):** three finalize runs. **DINOv3 fusion×150ep `capricious-hound-240`
  = 0.5738** (new best DINOv3, +0.023). C-RADIO's GIoU+oversample compound *regressed*
  at 150ep (`resilient-moth-415` 0.5697 < plain-150ep 0.5931), so **best C-RADIO stays
  `dazzling-mole-850` 0.5931**. See "Round 4 returns".
- **Next (open):** champion promotion needs manual fix — the auto-`@candidate` aliases
  point at sub-best runs (see "Champion registration" below). Register
  `dazzling-mole-850` → `cradio_detector@champion`; set `dinov3_detector@champion` → v7
  (`capricious-hound-240`); then run `09b` on the champions (Caries AP@50 gate + free
  decode/NMS). C-RADIO is ~0.007 from 0.60 (push 175–200ep / 1280px); DINOv3 ~0.026
  (fusion is working — more schedule or a wider fusion set).

Note: the **deck** (`docs/TALK.md`) intentionally does **not** cover this campaign —
it is the C-RADIO "one frozen backbone, three jobs" narrative. This `HPO.md` is the
sole record of the push-to-0.60 build-out.

### DINOv3 plateau (Stage 1) — diagnosis & fix

The DINOv3 Stage-1 winner retrained at 75 epochs (`intrigued-stork-789`: 1024px,
`lr=2e-4`, `pct_start=0.1`, **no augmentation**) plateaued. The full trajectory:

| phase | val/mAP@50 | val/mAP@50:95 | train/loss | grad_norm |
|------|-----------|---------------|-----------|-----------|
| epoch ~30 | ~0.50 | ~0.28 | 0.39 | ~3 |
| epoch 49 (peak) | 0.532 | 0.317 | 0.32 | ~6 |
| epochs 50–69 | 0.47–0.51 (flat) | 0.29–0.30 (flat) | 0.27 → 0.23 | ~3–4 |

**Verdict: overfitting, not an optimization failure.**

- **Train loss falls monotonically (0.39 → 0.23) while val mAP is flat from epoch
  ~30** — the classic memorization signature on the 705-image train set.
- `grad_norm` ~3–4 and `amp_scale`=1 throughout → fp32 is stable, optimization is
  healthy. It's a generalization wall, not instability (contrast the pre-fix
  flat-loss collapse runs).
- The `best_mAP_50=0.532` @ epoch 49 is a **single-epoch spike on the 50-image
  val set** (neighbours ~0.49). Honest stable level ≈ 0.50; schedule extension
  50→75 added only ~+0.01 over `rumbling-boar-852` (0.521).

**Two compounding causes (both predicted by the plan):**

1. **Zero regularization.** The Stage-1 arm had `aug_multiscale_range` /
   `aug_rotation_deg` / `aug_jitter_scale` all off. A 300M-param full fine-tune on
   705 images with no augmentation saturates by ~epoch 30; the extra 45 epochs are
   pure overfit. (Plan: "schedule only helps DINOv3 *once overfit is controlled*".)
2. **Resolution was never exercised.** Every *finished* Stage-1 trial was 1024px —
   the `random` sweep never sampled 1280 in its 6 trials (no 1280 OOM'd either),
   while the run used only **31% of GPU memory (26.7/80 GB)**. The highest-value
   lever for small caries/periapical lesions sat unused.

**Fix applied:** cancelled `intrigued-stork-789` and launched the `dinov3_regres`
stage — **regularization + resolution together**: `img_size=1280`,
`aug_multiscale_range=[0.7,1.0]`, `aug_rotation_deg=7`, `aug_jitter_scale=1.5`, 75
epochs (s1 optimizer region). This brings the plan's Stage-3 augmentation forward
and combines it with the Stage-1 resolution lever the random sweep skipped, rather
than running the Stage-2 anchor sweep on a base that's still overfitting. The
50-image val makes single-epoch mAP noisy, so judge on the stable band, not peaks.

**Outcome (`victorious-goose-410`, 1280px + aug, 75ep): overfit fixed, but not a breakthrough.**
The curve shape is now healthy — val mAP@50 *rises monotonically to e66 (0.535) and holds
0.530 to e74* (no late collapse), and **val mAP@50:95 climbs to 0.317 and holds** (best DINOv3
yet, vs ~0.29 flat before). Crucially train/loss only falls to **0.318** (vs 0.23 unregularized)
— the train/val gap is small, confirming augmentation closed the overfit. But headline
**mAP@50 ≈ 0.535** is only ~+0.015 over the `chill-robin-965` baseline (0.518) and still ~0.045
short of the 0.58 gate. **Takeaway:** resolution + augmentation exhaust the "regularization /
schedule" lever for DINOv3 at a genuine ~0.53 ceiling; the binding constraint is now recall /
anchor geometry, so Stage 2 (anchor + loss) should run **pinned at 1280px + aug** (not 1024).
Free decode/NMS grid (09b) not yet banked — likely +0.01–0.02 on top.

### DINOv3 ceiling — all planned levers exhausted (~0.535)

Stage 2 (anchor + loss sweep, 8 trials @30ep pinned on the 1280+aug base, winner
retrained 75ep) produced `serious-kit-244` = **0.5256** (50:95 0.301) — *below* the
`victorious-goose-410` base (0.5355 / 0.316). The "winning" combo differed only by
`focal_gamma 2.5` (vs 2.0) + explicit 3-octave anchors, and the 30ep trial ranking
is dominated by 50-image val noise, so the sweep effectively found nothing better
than the base. The full DINOv3 ladder:

| step | run | mAP@50 | 50:95 | Δ |
|------|-----|--------|-------|---|
| baseline (per-level) | `chill-robin-965` | 0.5181 | 0.285 | — |
| + schedule 50ep | `rumbling-boar-852` | 0.5212 | 0.294 | +0.003 |
| + schedule 75ep (no aug) | `intrigued-stork-789` | 0.532* | 0.295 | overfit (peak is noise) |
| **+ 1280px + aug (best)** | **`victorious-goose-410`** | **0.5355** | **0.316** | **+0.014** |
| + anchor/loss sweep | `serious-kit-244` | 0.5256 | 0.301 | −0.010 |

\* single-epoch spike on the 50-img val; stable band ≈0.50.

**Conclusion:** every planned lever — schedule, resolution, augmentation, anchor
geometry, focal/box loss — has now been tried. Resolution+augmentation delivered
essentially all the gain (+0.014 over baseline to **0.5355**); schedule, anchors and
loss are tapped out. DINOv3-ViTL16 on this 705-image set with the current detection
head is **capped at ~0.53–0.54 mAP@50**, ~0.045 short of the 0.58 gate and ~0.065
short of 0.60. Reaching 0.60 on DINOv3 would require a step-change (e.g. 1536px,
larger backbone/head capacity, or more labelled data), not more of these knobs.
**Best DINOv3 model = `victorious-goose-410` (0.5355).** The 1536px step-change
(`vaunted-crane-68`, 100ep) confirmed it: **0.5326** mAP@50 (no gain vs 1280) though
50:95 nudged to a DINOv3-best 0.3238. Resolution past 1280 buys localization, not
detection rate. DINOv3 is done at ~0.53.

### C-RADIO breakout — schedule + augmentation is the lever (0.5648)

Unlike DINOv3, **C-RADIO responded strongly**. Stage 1 (schedule × resolution +
mild aug) winner `useful-mare-854` — **1024px**, 100ep, lr 2e-4, pct_start 0.2,
base_scale 3.0, fg 2.5, aug `[0.8,1.0]`/rot5/jit1.5 — hit **0.5648** mAP@50
(50:95 0.291), **+0.043 over the 0.5219 baseline**.

Two things stand out:

1. **Resolution did *not* win — schedule did.** The winner is 1024px, not 1280/1536.
   C-RADIO's gain came from the longer 100ep schedule with mild augmentation, the
   opposite of DINOv3 (where resolution mattered and schedule didn't). The 75ep
   retrain (`crawling-wolf-195`) only hit 0.4915, so the last 25 epochs are worth
   ~+0.07 here.
2. **Still rising at e100.** val mAP@50 climbs 0.521 (e72) → 0.549 (e99, peak 0.5648
   @ e90) and train/loss is still falling (0.243) with a small train/val gap — aug
   is holding overfit off. **C-RADIO is not saturated at 100ep**, so a longer
   schedule (150ep) and the untouched anchor/loss + aug-strength levers have a
   credible path to the 0.58 gate.

**Best C-RADIO model = `useful-mare-854` (0.5648), not yet registered.** Caries
AP@50 baseline (from 09b on the *old* registered model) is **0.21** — the hard
class and the binding half of the gate; Stage 2's focal_alpha / anchor density
sweep targets exactly this.

### Round 3 returns — Track A (config) + Track B (architecture), in parallel

With DINOv3 capped on every *knob* and C-RADIO still rising, Round 3 split into two
tracks run concurrently (`max_concurrent_runs` raised to 5 on `campaign_sweep`):

- **Track A — config-only levers:** `cradio_long` (push the schedule to 150ep),
  `cradio_s2` (anchor/loss sweep), `dinov3_falpha` (focal_alpha for the 0.21 Caries
  class), all pinned on each backbone's best Round-2 recipe.
- **Track B — architectural levers (new code, config-gated, default-off):**
  `dinov3_fusion` (learnable softmax fusion of ViT hidden states L6/12/18/24 into the
  FPN, vs last-layer-only) and `cradio_giou` (GIoU box loss + 2× Caries oversampling).
  C-RADIO can't fuse (custom HF model), so it gets the backbone-agnostic Track-B levers.

| Track | Stage | Run | mAP@50 | 50:95 | Δ vs prior best | Verdict |
|-------|-------|-----|--------|-------|-----------------|---------|
| A | `cradio_long` (150ep) | `dazzling-mole-850` | **0.5931** | 0.304 | **+0.028** (C-RADIO) | **best overall; schedule is the dominant C-RADIO lever** |
| A | `cradio_s2` (anchor/loss) | `polite-frog-337` | 0.5489 | — | −0.016 | no gain (below s1 base) |
| A | `dinov3_falpha` | `casual-jay-300` | 0.5163 | 0.302 | −0.034 | no gain (below fusion + regres base) |
| B | `dinov3_fusion` (75ep) | `delicate-stork-19` | **0.5504** | **0.321** | **+0.015** (DINOv3) | **best DINOv3; broke the ~0.535 ceiling** |
| B | `cradio_giou` (100ep) | `amazing-moth-867` | 0.5674 | 0.299 | +0.003 vs s1 | small but additive localization gain |

**What this proves:**

1. **The DINOv3 ceiling was architectural, not a tuning failure.** Every *knob*
   (schedule, resolution, augmentation, anchors, focal/box loss, focal_alpha) topped
   out at ~0.535. The one lever that moved it — **multi-layer feature fusion** — is an
   architecture change. Single-layer ViT features were the binding constraint; fusing
   L6/12/18/24 lifts mAP@50 +0.015 and mAP@50:95 to a DINOv3-best **0.321** in only 75ep.
2. **C-RADIO's dominant lever is the schedule.** 150ep alone took it to **0.5931**
   (+0.028), the closest any single model has come to 0.60. GIoU + Caries oversampling
   add a smaller, orthogonal +0.003 on localization.
3. **Anchor/loss is fully tapped on both backbones** — `cradio_s2` and `dinov3_falpha`
   both regressed. No further budget goes there.

### Round 4 returns — compound the confirmed winners (done)

Nobody had combined the levers that *individually* won, so three finalize runs (all
150ep, `register_winner=True`) compounded them:

| Stage | Recipe | Run (version) | mAP@50 | 50:95 | Δ | Verdict |
|-------|--------|---------------|--------|-------|---|---------|
| `dinov3_final` | fusion + 150ep (smooth_l1) | `capricious-hound-240` (v7) | **0.5738** | 0.333 | **+0.023** vs fusion@75ep | **WIN — best DINOv3; fusion × schedule compound** |
| `dinov3_final_giou` | fusion + 150ep + GIoU | `rebellious-gnu-395` (v8) | 0.5704 | **0.340** | −0.003 vs smooth_l1 | GIoU trades a hair of mAP@50 for the **best 50:95** |
| `cradio_final` | 150ep + GIoU + Caries×2 | `resilient-moth-415` (v11) | 0.5697 | 0.288 | **−0.023** vs plain 150ep | **REGRESSED — GIoU+oversample hurt once the schedule was long** |

**Findings:**

1. **DINOv3 compounded cleanly.** The two confirmed wins *stack*: fusion (75ep → 0.5504)
   × long schedule (→ **0.5738**). Net DINOv3 journey is **0.518 → 0.574 (+0.056, +11%
   relative)**, with mAP@50:95 up to 0.333 (0.340 with GIoU) — fully past the old ~0.535
   ceiling. DINOv3 now sits ~0.026 short of 0.60.
2. **GIoU is a localization, not a detection, lever.** On DINOv3 it lifts 50:95 (0.333 →
   0.340) while shaving mAP@50 (−0.003) — useful if box tightness matters, neutral for the
   headline metric.
3. **The C-RADIO compound *failed*.** GIoU + Caries×2 *helped* at 100ep (+0.003 vs s1) but
   *hurt* at 150ep (0.5697 vs the plain-150ep 0.5931, −0.023). The extra regularization the
   oversampling/GIoU impose collides with the long schedule's need to keep fitting. **Best
   C-RADIO stays the plain `dazzling-mole-850` (0.5931).** C-RADIO's path to 0.60 is *more
   schedule / resolution*, not more loss-side levers.

### ⚠ Champion registration — the auto-`@candidate` aliases point at the wrong runs

`register_winner=True` set `@candidate` on whatever registered **last**, which is *not* the
best model on either backbone:

| Model | `@champion` (now) | `@candidate` (now) | **Best run** | Action needed |
|-------|-------------------|--------------------|--------------|---------------|
| `cradio_detector` | v10 `upbeat-mink-783` (0.5219) | v11 `resilient-moth-415` (0.5697) | **`dazzling-mole-850` 0.5931 — UNREGISTERED** | register `dazzling-mole-850` → `@champion` |
| `dinov3_detector` | v2 (old, **broken**) | v8 `rebellious-gnu-395` (0.5704) | **v7 `capricious-hound-240` 0.5738 (registered)** | set `@champion` → v7 |

So promoting champions is **not** "alias the candidate": for C-RADIO the best model isn't
registered at all (it was a Round-3 `register_winner=False` run), and for DINOv3 the best is
v7, not the v8 the candidate alias landed on. Promotion plan: register `dazzling-mole-850`
from its run as a new `cradio_detector` version and alias it `@champion`; set
`dinov3_detector@champion` → v7. Then run `09b` on the two champions for the Caries AP@50 gate
+ free decode/NMS gains (this also retires the broken-DINOv3 issue below).

### ⚠ Registered DINOv3 serving model is broken

The 09b grid scored the **registered** `dinov3_detector` at **0.027** mAP@50 (vs
0.53 for the training runs) across every threshold — i.e. the deployed/registered
DINOv3 pyfunc produces garbage, a registration/serialization problem independent of
training. C-RADIO's registered model is fine (0.519). Action: re-register a good
DINOv3 (e.g. `victorious-goose-410`) before trusting any served DINOv3 number.
