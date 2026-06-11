"""Per-backbone training recipes — the single source of truth for both launch lanes.

A *recipe* is the best-known set of `TrainerConfig` overrides for one backbone,
as proven by the "push to 0.60" tuning campaigns (`config.campaigns`,
docs/HPO.md). Both launch surfaces consume the same dict:

* notebook lane: `02_train_detector_air.py` calls `build_trainer_config(BACKBONE, ...)`
  inside the `@distributed` closure;
* sgcli lane: `train/cli.py` resolves a `recipe:` key in the workload
  `parameters:` block and merges the YAML's remaining keys on top.

`TrainerConfig` field defaults stay frozen at the *legacy* values (absolute
anchors, class-agnostic NMS) so historical runs remain byte-identical; the
post-fix recipe lives HERE, not in the dataclass defaults. Recipes deliberately
contain only hyperparameters — UC locations (catalog/schema/volumes), MLflow
experiment, and model identity arrive as explicit arguments because they are
environment, not science.

Values are dicts (not a parallel dataclass) on purpose: validity is enforced by
building a real `TrainerConfig` from each recipe in unit tests, so there is no
second schema to keep in sync.
"""

from __future__ import annotations

from typing import Any

from dais26_dentex.config.trainer_config import TrainerConfig

# UC model + dev-endpoint names keyed by the internal backbone literal. Moved
# here from notebooks/00_config.py so the sgcli lane and tests can resolve the
# same identity mapping the notebooks use. The cradio entry preserves the
# historical names for backward compatibility.
DETECTOR_NAMES_BY_BACKBONE: dict[str, dict[str, str]] = {
    "cradio_v4_so400m": {
        "model_short": "cradio_detector",
        "endpoint": "dais26-cradio-detector-dev",
    },
    "dinov3_vitl16": {
        "model_short": "dinov3_detector",
        "endpoint": "dais26-dinov3-detector-dev",
    },
    "dinov2_base": {
        "model_short": "dinov2_detector",
        "endpoint": "dais26-dinov2-detector-dev",
    },
}

# Best-known training recipes per backbone. Provenance: the winning MLflow run
# of the corresponding campaign stage (docs/HPO.md "Round 3/4 returns").
RECIPES: dict[str, dict[str, Any]] = {
    # `cradio_long` winner `dazzling-mole-850` — val mAP@50 0.5931 (50:95 0.304),
    # the best overall run. NOTE: the nominal finalize stage (`cradio_final`,
    # +GIoU +Caries x2) REGRESSED to 0.5697 at 150ep, so the recipe is the plain
    # `useful-mare-854` config on the 150-epoch schedule — smooth_l1, no
    # oversampling (docs/HPO.md "Round 4 returns").
    "cradio_v4_so400m": {
        "backbone_mode": "full",
        "backbone_lr": 1e-5,
        "weight_decay": 1e-2,
        "anchor_layout": "per_level",
        "anchor_base_scale": 3.0,
        "nms_per_class": True,
        "amp_dtype": "auto",  # -> fp16 (C-RADIO's stable path)
        "batch_size": 4,
        "grad_accum_steps": 2,  # effective 4*2*8 = 64 on one 8xH100 node
        "img_size": 1024,
        "lr": 2e-4,
        "onecycle_pct_start": 0.2,
        "focal_gamma": 2.5,
        "focal_alpha": 0.25,
        "box_loss_weight": 1.0,
        "box_loss_type": "smooth_l1",
        "aug_multiscale_range": [0.8, 1.0],
        "aug_rotation_deg": 5.0,
        "aug_jitter_scale": 1.5,
        "epochs": 150,
    },
    # `dinov3_final` winner `capricious-hound-240` (v7) — val mAP@50 0.5738
    # (50:95 0.333), best DINOv3: multi-layer fusion x 150ep compound. smooth_l1
    # (the GIoU variant traded -0.003 mAP@50 for 50:95; we keep the mAP@50
    # winner — the campaign gate metric). fp32 via amp "auto" (fp16/bf16 NaN).
    "dinov3_vitl16": {
        "backbone_mode": "full",
        "backbone_lr": 1e-5,
        "weight_decay": 1e-2,
        "anchor_layout": "per_level",
        "anchor_base_scale": 4.0,
        "nms_per_class": True,
        "amp_dtype": "auto",  # -> fp32 for DINOv3 (autocast-unstable)
        "batch_size": 2,
        "grad_accum_steps": 2,  # effective 2*2*8 = 32 at 1280px fp32
        "img_size": 1280,
        "lr": 2e-4,
        "onecycle_pct_start": 0.1,
        "focal_gamma": 2.0,
        "box_loss_weight": 1.0,
        "box_loss_type": "smooth_l1",
        "aug_multiscale_range": [0.7, 1.0],
        "aug_rotation_deg": 7.0,
        "aug_jitter_scale": 1.5,
        "fusion_layers": [6, 12, 18, 24],
        "epochs": 150,
    },
    # Emergency fallback — never campaign-tuned. Keeps the cheap frozen-head
    # path with the structural fixes (per-level anchors, per-class NMS) that
    # are backbone-agnostic wins.
    "dinov2_base": {
        "backbone_mode": "frozen",
        "anchor_layout": "per_level",
        "anchor_base_scale": 4.0,
        "nms_per_class": True,
        "amp_dtype": "auto",  # -> fp16
        "batch_size": 8,
        "img_size": 1024,
        "lr": 1e-3,
        "epochs": 50,
    },
}


def build_trainer_config(
    backbone: str,
    *,
    catalog: str,
    schema: str,
    volume_path: str | None = None,
    cache_dir: str | None = None,
    experiment_name: str | None = None,
    model_name: str | None = None,
    backbone_revision: str | None = None,
    **overrides: Any,
) -> TrainerConfig:
    """Build a validated `TrainerConfig` from a backbone's recipe.

    Precedence (last wins): recipe -> environment kwargs -> explicit overrides.
    `model_name` defaults to the backbone's registered short name from
    `DETECTOR_NAMES_BY_BACKBONE`. Raises `KeyError` with the known backbones on
    an unknown `backbone`, and whatever `TrainerConfig.validate` raises on a
    bad override combination.
    """
    if backbone not in RECIPES:
        raise KeyError(f"No recipe for backbone {backbone!r}; known: {sorted(RECIPES)}")
    if model_name is None:
        model_name = DETECTOR_NAMES_BY_BACKBONE[backbone]["model_short"]
    params: dict[str, Any] = {
        **RECIPES[backbone],
        "backbone_name": backbone,
        "catalog": catalog,
        "schema": schema,
        "volume_path": volume_path,
        "cache_dir": cache_dir,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "backbone_revision": backbone_revision,
        **overrides,
    }
    cfg = TrainerConfig.from_dict(params)
    cfg.validate()
    return cfg


__all__ = [
    "DETECTOR_NAMES_BY_BACKBONE",
    "RECIPES",
    "build_trainer_config",
]
