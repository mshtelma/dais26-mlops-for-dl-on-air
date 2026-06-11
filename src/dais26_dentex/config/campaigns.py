"""HPO campaign definitions — the "push to 0.60" stage chain, as package data.

Moved verbatim from `notebooks/00_config.py` so the campaign is (a) importable
by every launch surface (notebook 02b AND the sgcli/torchrun sweep CLI) and
(b) validated by unit tests instead of failing at GPU time on a typo'd field.

The stages are a HISTORICAL RECORD of the campaign chain (docs/HPO.md): pinned
values were seeded from each prior stage's MLflow winner at the time the stage
ran. Do not retro-edit them to match `config.recipes.RECIPES` — recipes encode
the forward-looking best-known config; stages encode how we got there.

Stage semantics (consumed by `train.sweep_runner.SweepRunner`):
  * `pinned`       — TrainerConfig overrides held fixed for every trial.
  * `search_space` — field -> spec for `train.sweep.iter_trials`.
  * `trial_epochs` — cheap per-trial schedule.
  * `schedule_epochs` — the winner is retrained at each of these; the better
    run (by the primary metric) is kept.
  * `register_winner` — False = measure-only stage (gating data for the next
    stage); True = finalize stage that registers + gates `@challenger`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from dais26_dentex.config.trainer_config import TrainerConfig

# Standard RetinaNet octave sets used as `anchor_octaves` sweep choices. OCT3 is
# the 3-octave default written long-hand (so it shows up explicitly in the run
# params); OCT4 adds a 4th octave for denser scale coverage of small caries /
# periapical lesions.
_OCT3 = [2.0**0, 2.0 ** (1.0 / 3.0), 2.0 ** (2.0 / 3.0)]
_OCT4 = [2.0**0, 2.0**0.25, 2.0**0.5, 2.0**0.75]
_AR3 = [0.5, 1.0, 2.0]
_AR5 = [0.33, 0.5, 1.0, 2.0, 3.0]

# Keys legal in `pinned` / `search_space` beyond literal TrainerConfig fields.
# `anchor_mode` is a launcher-level toggle ("default" | "calibrated") resolved
# into concrete anchor_scales/aspect_ratios by the sweep driver.
_VIRTUAL_KEYS: frozenset[str] = frozenset({"anchor_mode"})


@dataclass(frozen=True, slots=True)
class CampaignStage:
    """One gated stage of a tuning campaign."""

    backbone: str
    trial_epochs: int
    schedule_epochs: tuple[int, ...]
    max_trials: int
    register_winner: bool
    pinned: Mapping[str, Any] = field(default_factory=dict)
    search_space: Mapping[str, Any] = field(default_factory=dict)


def validate_stage(stage: CampaignStage) -> None:
    """Raise `ValueError` if a stage references unknown TrainerConfig fields
    or carries inconsistent shape (empty schedule, non-positive budgets)."""
    errs: list[str] = []
    valid = {f.name for f in fields(TrainerConfig)} | _VIRTUAL_KEYS
    for kind, mapping in (("pinned", stage.pinned), ("search_space", stage.search_space)):
        for key in mapping:
            if key not in valid:
                errs.append(f"{kind} key {key!r} is not a TrainerConfig field")
    if stage.trial_epochs < 1:
        errs.append(f"trial_epochs must be >= 1, got {stage.trial_epochs}")
    if stage.max_trials < 1:
        errs.append(f"max_trials must be >= 1, got {stage.max_trials}")
    if not stage.schedule_epochs or any(e < 1 for e in stage.schedule_epochs):
        errs.append(f"schedule_epochs must be non-empty positive ints, got {stage.schedule_epochs}")
    if not stage.search_space:
        errs.append("search_space must be non-empty (use {'base_seed': [42]} for a 1-trial stage)")
    if errs:
        raise ValueError("CampaignStage validation failed:\n  - " + "\n  - ".join(errs))


# Legacy post-fix sweep defaults (the pre-campaign `SWEEP_*` block): the encoder
# axis was settled (full fine-tune won at 0.335) and the budget went to the
# newly-unlocked per-level anchor geometry.
SWEEP_DEFAULTS = CampaignStage(
    backbone="cradio_v4_so400m",
    trial_epochs=25,
    schedule_epochs=(50, 100),  # TRAIN_EPOCHS / TRAIN_EPOCHS_LONG at the time
    max_trials=8,
    register_winner=True,
    pinned={
        "backbone_mode": "full",  # full fine-tune beat lora/frozen (0.335 vs 0.228/0.213)
        "backbone_lr": 1e-5,  # discriminative LR; head lr is swept below
        "onecycle_pct_start": 0.3,  # won previously
        "weight_decay": 1e-2,  # won previously
        "img_size": 1024,
        "anchor_layout": "per_level",  # THE fix: stride-scaled RetinaNet anchors
        "nms_per_class": True,  # per-class batched_nms
    },
    search_space={
        "anchor_base_scale": [3.0, 4.0, 5.0],
        "aspect_ratios": [_AR3, _AR5],
        "lr": [1e-4, 2e-4],
        "box_loss_weight": [1.0, 2.0],
        "focal_gamma": [2.0, 2.5],
    },
)

# ---- "Push to 0.60" tuning campaign (docs/HPO.md) -------------------------
# Two sequential single-model campaigns (DINOv3 first, then C-RADIO), each a
# chain of gated stages. Run stages in order — the pinned values in s2/s3/s4
# were SEEDED with the then-current best and updated to the prior stage's
# MLflow winner before launching (the chain is inherently sequential).
CAMPAIGN_STAGES: dict[str, CampaignStage] = {
    # ===== Campaign 1 — DINOv3 (fp32; fix = regularize THEN extend + raise res) =====
    "dinov3_s1": CampaignStage(  # resolution x schedule
        backbone="dinov3_vitl16",
        trial_epochs=30,
        schedule_epochs=(50, 75),  # winner retrained at both; keep better
        max_trials=6,
        register_winner=False,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto",  # -> fp32 for DINOv3
            "batch_size": 2, "grad_accum_steps": 2,  # effective 2*2*8=32 at 1024 AND 1280
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
        },
        search_space={
            "img_size": [1024, 1280],
            "lr": [1e-4, 2e-4],
            "onecycle_pct_start": [0.1, 0.3],
        },
    ),
    "dinov3_regres": CampaignStage(  # regularize + resolution (overfit fix; HPO.md "DINOv3 plateau")
        # Diagnosis (intrigued-stork-789): 1024px/no-aug DINOv3 overfits — val
        # mAP flat ~0.50 from e30 while train loss keeps falling to 0.23; the
        # 0.532 @e49 was 50-img val noise. Stage-1 trials never sampled 1280,
        # and GPU mem was only 31% used. Fix = regularization + resolution.
        backbone="dinov3_vitl16",
        trial_epochs=10,  # trivial single trial; the 75ep retrain is the real run
        schedule_epochs=(75,),
        max_trials=1,
        register_winner=False,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto",
            "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280,  # never tried at 1024-only Stage 1; mem headroom is huge
            "lr": 2e-4, "onecycle_pct_start": 0.1,  # s1 winner optimizer region
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
            # Regularization (the DINOv3 gap): multi-scale + small rotation + stronger jitter
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
        },
        search_space={"base_seed": [42]},  # degenerate 1-trial "sweep" -> 75ep retrain
    ),
    "dinov3_s2": CampaignStage(  # anchor + loss (pinned on the regres winner)
        backbone="dinov3_vitl16",
        trial_epochs=30,
        schedule_epochs=(75,),
        max_trials=8,
        register_winner=False,
        pinned={
            "backbone_mode": "full", "weight_decay": 1e-2,
            "anchor_layout": "per_level", "nms_per_class": True, "amp_dtype": "auto",
            "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.1,  # <- regres winner region
            # carry the regularization base forward: anchor/loss sweep on a
            # non-augmented base would just re-overfit ("DINOv3 plateau" in HPO.md)
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
        },
        search_space={
            "anchor_base_scale": [3.0, 4.0],
            "anchor_octaves": [_OCT3, _OCT4],
            "aspect_ratios": [_AR3, _AR5],
            "focal_gamma": [2.0, 2.5],
            "box_loss_weight": [1.0, 2.0],
            "backbone_lr": [1e-5, 2e-5],
        },
    ),
    "dinov3_s3": CampaignStage(  # augmentation / regularization (pinned on s2 winner)
        backbone="dinov3_vitl16",
        trial_epochs=40,
        schedule_epochs=(75,),
        max_trials=6,
        register_winner=False,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "nms_per_class": True, "amp_dtype": "auto",
            "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 1e-4, "onecycle_pct_start": 0.3,
            "anchor_base_scale": 4.0, "focal_gamma": 2.0, "box_loss_weight": 1.0,  # <- s2 winner
        },
        search_space={
            "aug_multiscale_range": [[0.8, 1.0], [0.7, 1.0]],
            "aug_rotation_deg": [0.0, 7.0],
            "aug_jitter_scale": [1.0, 1.5],
        },
    ),
    "dinov3_s4": CampaignStage(  # finalize: single combined-best combo, register @challenger
        backbone="dinov3_vitl16",
        trial_epochs=30,
        schedule_epochs=(75,),
        max_trials=1,
        register_winner=True,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "nms_per_class": True, "amp_dtype": "auto",
            "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 1e-4, "onecycle_pct_start": 0.3,
            "anchor_base_scale": 4.0, "focal_gamma": 2.0, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
        },
        search_space={"base_seed": [42]},  # degenerate 1-trial "sweep" -> retrain+register
    ),
    "dinov3_res1536": CampaignStage(  # step-change attempt: resolution past 1280 (HPO.md "DINOv3 ceiling")
        # DINOv3 capped ~0.535 on every planned knob; the only untried
        # high-leverage lever was resolution beyond 1280. 1536px fp32 needs
        # batch=1 (grad_accum keeps effective batch 32). Longer 100ep schedule
        # since aug controls overfit now.
        backbone="dinov3_vitl16",
        trial_epochs=10,
        schedule_epochs=(100,),
        max_trials=1,
        register_winner=False,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto",
            "batch_size": 1, "grad_accum_steps": 4,  # effective 1*4*8=32 at 1536 fp32
            "img_size": 1536,
            "lr": 2e-4, "onecycle_pct_start": 0.1,
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
        },
        search_space={"base_seed": [42]},  # degenerate 1-trial "sweep" -> 100ep retrain
    ),
    "dinov3_falpha": CampaignStage(  # the one config knob skipped for DINOv3: focal_alpha (Caries)
        # Low odds of breaking the ceiling, but cheap and closes the config
        # search honestly. Pinned on the victorious-goose-410 base (1280 + aug).
        backbone="dinov3_vitl16",
        trial_epochs=30,
        schedule_epochs=(75,),
        max_trials=3,
        register_winner=False,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.1,
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
        },
        search_space={"focal_alpha": [0.25, 0.5, 0.75]},
    ),
    "dinov3_fusion": CampaignStage(  # Track B step-change: multi-layer ViT feature fusion
        # The headline architectural lever for DINOv3 — fuse hidden states
        # L6/12/18/24 into the FPN instead of last-layer-only. Pinned on the
        # victorious-goose-410 base so the ONLY change vs that run is fusion;
        # clean attribution of any gain past the ~0.53 ceiling.
        backbone="dinov3_vitl16",
        trial_epochs=10,
        schedule_epochs=(75,),
        max_trials=1,
        register_winner=False,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.1,
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
            "fusion_layers": [6, 12, 18, 24],
        },
        search_space={"base_seed": [42]},
    ),
    # ===== Campaign 2 — C-RADIO (fp16; fix = extend schedule + raise resolution) =====
    "cradio_s1": CampaignStage(  # schedule x resolution (+ mild aug)
        backbone="cradio_v4_so400m",
        trial_epochs=30,
        schedule_epochs=(75, 100),
        max_trials=8,
        register_winner=False,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 3.0, "nms_per_class": True,
            "amp_dtype": "auto",  # -> fp16 for C-RADIO
            "batch_size": 4, "grad_accum_steps": 2,  # effective 4*2*8=64 (matches 0.5219 run)
            "focal_gamma": 2.5, "box_loss_weight": 1.0,
            # Mild regularization baked in: DINOv3 showed 75-100ep with NO aug
            # just overfits the 705-img train set. Keep it gentle so the
            # schedule x resolution signal stays readable.
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0, "aug_jitter_scale": 1.5,
        },
        search_space={
            "img_size": [1024, 1280, 1536],
            "lr": [2e-4, 3e-4],
            "onecycle_pct_start": [0.2, 0.3],
        },
    ),
    "cradio_long": CampaignStage(  # pure schedule test: 150ep on the exact s1 winner config
        # useful-mare-854 was still climbing at e100 with train loss falling —
        # extend the exact winner config to 150ep to bank the free schedule
        # gain. Winner dazzling-mole-850 (0.5931) — the best overall run; this
        # pinned set seeds `config.recipes.RECIPES["cradio_v4_so400m"]`.
        backbone="cradio_v4_so400m",
        trial_epochs=10,
        schedule_epochs=(150,),
        max_trials=1,
        register_winner=False,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 3.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 4, "grad_accum_steps": 2,
            "img_size": 1024, "lr": 2e-4, "onecycle_pct_start": 0.2,
            "focal_gamma": 2.5, "focal_alpha": 0.25, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0, "aug_jitter_scale": 1.5,
        },
        search_space={"base_seed": [42]},
    ),
    "cradio_s2": CampaignStage(  # anchor + loss, pinned on s1 winner useful-mare-854
        backbone="cradio_v4_so400m",
        trial_epochs=30,
        schedule_epochs=(100,),
        max_trials=10,
        register_winner=False,
        pinned={
            "backbone_mode": "full", "weight_decay": 1e-2,
            "anchor_layout": "per_level", "nms_per_class": True, "amp_dtype": "auto",
            "batch_size": 4, "grad_accum_steps": 2,
            "img_size": 1024, "lr": 2e-4, "onecycle_pct_start": 0.2,  # <- s1 winner
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0, "aug_jitter_scale": 1.5,
        },
        search_space={
            "anchor_base_scale": [2.0, 2.5, 3.0],
            "anchor_octaves": [_OCT3, _OCT4],
            "aspect_ratios": [_AR3, _AR5],
            "focal_gamma": [2.0, 2.5],
            "box_loss_weight": [1.0, 2.0],
            "focal_alpha": [0.25, 0.5],
            "backbone_lr": [1e-5, 2e-5],
        },
    ),
    "cradio_s3": CampaignStage(  # augmentation (pinned on s2 winner)
        backbone="cradio_v4_so400m",
        trial_epochs=40,
        schedule_epochs=(100,),
        max_trials=4,
        register_winner=False,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "nms_per_class": True, "amp_dtype": "auto",
            "batch_size": 4, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.3,
            "anchor_base_scale": 2.5, "focal_gamma": 2.5, "box_loss_weight": 1.0,  # <- s2 winner
        },
        search_space={
            "aug_multiscale_range": [[0.8, 1.0], [0.7, 1.0]],
            "aug_rotation_deg": [0.0, 5.0],
        },
    ),
    "cradio_s4": CampaignStage(  # finalize: single combined-best combo, register @challenger
        backbone="cradio_v4_so400m",
        trial_epochs=30,
        schedule_epochs=(100,),
        max_trials=1,
        register_winner=True,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "nms_per_class": True, "amp_dtype": "auto",
            "batch_size": 4, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.3,
            "anchor_base_scale": 2.5, "focal_gamma": 2.5, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0,
        },
        search_space={"base_seed": [42]},
    ),
    "cradio_giou": CampaignStage(  # Track B for C-RADIO: GIoU box loss + Caries oversampling
        # C-RADIO can't fuse (custom HF model), but the backbone-agnostic
        # Track B levers apply: GIoU localization loss + 2x Caries
        # oversampling, pinned on the useful-mare-854 winner.
        backbone="cradio_v4_so400m",
        trial_epochs=10,
        schedule_epochs=(100,),
        max_trials=1,
        register_winner=False,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 3.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 4, "grad_accum_steps": 2,
            "img_size": 1024, "lr": 2e-4, "onecycle_pct_start": 0.2,
            "focal_gamma": 2.5, "focal_alpha": 0.25, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0, "aug_jitter_scale": 1.5,
            "box_loss_type": "giou", "caries_oversample": 2.0,
        },
        search_space={"base_seed": [42]},
    ),
    # ===== Finalize — compound the confirmed winners, register @challenger =====
    # Round-4 outcomes (docs/HPO.md): DINOv3 compounded cleanly
    # (capricious-hound-240 0.5738); C-RADIO's GIoU+oversample REGRESSED at
    # 150ep (resilient-moth-415 0.5697 < plain-150ep dazzling-mole-850 0.5931).
    "cradio_final": CampaignStage(  # C-RADIO: 150ep + GIoU + Caries x2, registered
        backbone="cradio_v4_so400m",
        trial_epochs=10,
        schedule_epochs=(150,),
        max_trials=1,
        register_winner=True,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 3.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 4, "grad_accum_steps": 2,
            "img_size": 1024, "lr": 2e-4, "onecycle_pct_start": 0.2,
            "focal_gamma": 2.5, "focal_alpha": 0.25, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0, "aug_jitter_scale": 1.5,
            "box_loss_type": "giou", "caries_oversample": 2.0,
        },
        search_space={"base_seed": [42]},
    ),
    "dinov3_final": CampaignStage(  # DINOv3: fusion + 150ep (clean compound of the two wins)
        # Winner capricious-hound-240 (0.5738) — best DINOv3; this pinned set
        # seeds `config.recipes.RECIPES["dinov3_vitl16"]`.
        backbone="dinov3_vitl16",
        trial_epochs=10,
        schedule_epochs=(150,),
        max_trials=1,
        register_winner=True,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.1,
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
            "fusion_layers": [6, 12, 18, 24],
        },
        search_space={"base_seed": [42]},
    ),
    "dinov3_final_giou": CampaignStage(  # DINOv3: fusion + 150ep + GIoU (the extra shot at 0.58+)
        backbone="dinov3_vitl16",
        trial_epochs=10,
        schedule_epochs=(150,),
        max_trials=1,
        register_winner=True,
        pinned={
            "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
            "anchor_layout": "per_level", "anchor_base_scale": 4.0, "nms_per_class": True,
            "amp_dtype": "auto", "batch_size": 2, "grad_accum_steps": 2,
            "img_size": 1280, "lr": 2e-4, "onecycle_pct_start": 0.1,
            "focal_gamma": 2.0, "box_loss_weight": 1.0,
            "aug_multiscale_range": [0.7, 1.0], "aug_rotation_deg": 7.0, "aug_jitter_scale": 1.5,
            "fusion_layers": [6, 12, 18, 24], "box_loss_type": "giou",
        },
        search_space={"base_seed": [42]},
    ),
}


__all__ = [
    "CAMPAIGN_STAGES",
    "SWEEP_DEFAULTS",
    "CampaignStage",
    "validate_stage",
]
