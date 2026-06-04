"""TrainerConfig — single source of truth for training hyperparameters.

Replaces the kwarg proliferation on `train_detector(...)` and the parallel
`_INT_KEYS / _FLOAT_KEYS / _BOOL_KEYS` coercion lists in `train/cli.py`. All
runtime knobs the user might tune live here; load-bearing string identifiers
(artifact filenames, alias names, FPN levels) live in `config.constants`.

Pure dataclass + manual validate() — no Pydantic dependency. The whole point
of this module is to be the *one* place a teammate goes to add a knob, so
keep it small and read-able.

Usage:

    cfg = TrainerConfig.from_yaml("sgcli/workload_train_detector.yaml")
    cfg = TrainerConfig.from_dict(yaml.safe_load(open(path)))
    cfg = TrainerConfig(catalog="ml_dev", schema="dais26_vfm", ...)

    cfg.validate()                        # raises ValueError on bad combos
    mlflow.log_params(cfg.to_mlflow_params())
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

import yaml

# HF model IDs / friendly names → internal `load_backbone` literal. Lives
# here (not in cli.py) because callers other than the YAML cli.py path may
# want the same alias resolution (e.g., notebook drivers).
BACKBONE_ALIASES: dict[str, str] = {
    "nvidia/C-RADIOv4-SO400M": "cradio_v4_so400m",
    "nvidia/C-RADIOv4-H": "cradio_v4_so400m",  # not yet wired up; alias to SO400M for now
    "facebook/dinov3-vitl16": "dinov3_vitl16",
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "dinov3_vitl16",
    "facebook/dinov2-base": "dinov2_base",
}

# Allowed values for `backbone_name` — kept as a tuple so `validate()` can
# emit a clear error and tests can iterate it.
ALLOWED_BACKBONES: tuple[str, ...] = (
    "cradio_v4_so400m",
    "dinov3_vitl16",
    "dinov2_base",
)

# How much of the backbone is trainable:
#   frozen  — backbone weights fixed (the historical default; head/FPN only).
#   lora    — inject LoRA adapters into attention (cfg.lora_rank/alpha).
#   full    — fine-tune the entire backbone end-to-end.
#   partial — unfreeze only the last `backbone_trainable_blocks` transformer blocks.
ALLOWED_BACKBONE_MODES: tuple[str, ...] = ("frozen", "lora", "full", "partial")

# Anchor generation layout (mirrors detection_head.ANCHOR_LAYOUTS; duplicated here
# so config validation doesn't import the heavy models package):
#   absolute  — same scale list at every FPN level (legacy; over-generates).
#   per_level — base size = stride * base_scale, x octaves x ratios (RetinaNet).
ALLOWED_ANCHOR_LAYOUTS: tuple[str, ...] = ("absolute", "per_level")

# Autocast precision for CUDA training:
#   auto — bf16 for DINOv3 (NaNs under fp16), fp16 for everything else.
#   fp16 — half precision + GradScaler (C-RADIO's proven, stable path).
#   bf16 — bfloat16 autocast, no GradScaler (wide dynamic range; DINOv3-safe).
#   fp32 — autocast disabled (slowest, most accurate; bf16 fallback for DINOv3).
ALLOWED_AMP_DTYPES: tuple[str, ...] = ("auto", "fp16", "bf16", "fp32")

# Box-regression loss:
#   smooth_l1 — Huber on encoded (dx,dy,dw,dh) deltas (legacy RetinaNet default).
#   giou      — 1 - GIoU on *decoded* boxes (scale-invariant; better localization).
ALLOWED_BOX_LOSS_TYPES: tuple[str, ...] = ("smooth_l1", "giou")


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """All hyperparameters and runtime knobs for `train_detector`.

    Frozen so accidental mutation in long-running notebooks is caught early.
    `slots=True` keeps the object cheap; we'll log many of these to MLflow.
    """

    # --- Required: UC location -------------------------------------------
    catalog: str
    schema: str

    # --- Backbone --------------------------------------------------------
    backbone_name: str = "cradio_v4_so400m"
    backbone_revision: str | None = None
    cache_dir: str | None = None
    # Multi-layer ViT feature fusion: hidden-state layer indices to fuse into the
    # single spatial map the FPN consumes. `None` (default) = legacy last-layer-
    # only behavior. e.g. [6, 12, 18, 24] fuses 4 depths via a learnable softmax
    # weighted sum (ViT-Det style), which usually beats last-layer-only for
    # detection on ViT backbones. Requires an HF backbone exposing
    # `output_hidden_states` (dinov3_vitl16 / dinov2_base); see
    # `models.backbones` and docs/HPO.md "Multi-layer fusion".
    fusion_layers: list[int] | None = None

    # --- Data ------------------------------------------------------------
    volume_path: str | None = None
    img_size: int = 1024

    # --- Optimizer / schedule -------------------------------------------
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    onecycle_pct_start: float = 0.1
    batch_size: int = 8
    num_workers: int = 4
    # Gradient accumulation: accumulate grads over this many micro-batches before
    # one optimizer step, so high-resolution fp32 runs (e.g. DINOv3 @ 1280px) can
    # keep a large *effective* batch while fitting a tiny per-GPU `batch_size` in
    # memory. The OneCycle scheduler steps once per optimizer step, so the
    # effective batch is `batch_size * grad_accum_steps * world_size`. `1` (the
    # default) is the legacy no-accumulation path.
    grad_accum_steps: int = 1

    # --- Augmentation ----------------------------------------------------
    # Training-time augmentation strength. Defaults reproduce the legacy pipeline
    # (hflip p=0.5 + mild colour-jitter p=0.5, no rotation, no multi-scale) so
    # existing runs are byte-identical; the tuning campaigns dial these up to
    # regularize the 705-image train set (see docs/HPO.md). Val/inference never
    # augment regardless of these values.
    aug_hflip_prob: float = 0.5
    aug_jitter_prob: float = 0.5
    # Multiplier on the base colour-jitter magnitudes (brightness/contrast/
    # saturation 0.2, hue 0.05). 1.0 = legacy; >1 = stronger photometric jitter.
    aug_jitter_scale: float = 1.0
    # Max absolute rotation in degrees (random in [-deg, +deg], applied to image
    # and boxes via tv_tensors). 0.0 = disabled (legacy). Dental X-rays are
    # roughly axis-aligned, so keep this small (<=10).
    aug_rotation_deg: float = 0.0
    # Multi-scale training: per-sample, the longest-side resize target is scaled
    # by a random factor in this [lo, hi] range (then zero-padded back to
    # `img_size` so tensors stay uniform for batching). `None` = disabled
    # (legacy). Both bounds must be in (0, 1] — we only down-scale to avoid
    # cropping content out of the fixed `img_size` canvas.
    aug_multiscale_range: list[float] | None = None
    # Class-balanced oversampling: replicate training images that contain at least
    # one box of the hard class (Caries, label id 0) by this integer-ish factor
    # via a WeightedRandomSampler. 1.0 (default) = legacy uniform sampling. e.g.
    # 2.0 makes Caries-bearing images ~2x as likely per epoch, directly targeting
    # the binding Caries AP@50 gate. Applied train-only; val is untouched.
    caries_oversample: float = 1.0

    # --- Mixed precision -------------------------------------------------
    # Autocast dtype for CUDA training. "auto" resolves to bf16 for DINOv3
    # (which NaNs under fp16 — see docs/HPO.md "DINOv3 A/B") and fp16 otherwise
    # (C-RADIO trains stably in fp16). The GradScaler is only engaged for fp16;
    # bf16/fp32 need no loss scaling. See `effective_amp_dtype`.
    amp_dtype: str = "auto"
    # Fail-fast guard: abort training if `train/loss` has not decreased at all
    # after this many epochs (0 = disabled). Catches dead runs (e.g. AMP
    # instability skipping every optimizer step) before they burn the full
    # schedule.
    flat_loss_patience: int = 0

    # --- Loss weights ----------------------------------------------------
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    box_loss_weight: float = 1.0
    # Box-regression loss function. "smooth_l1" (default) is the legacy delta
    # Huber loss; "giou" computes 1 - generalized-IoU on decoded boxes, which is
    # scale-invariant and usually improves localization (mAP@50:95) and small-box
    # recall. See `train.losses.giou_box_loss`.
    box_loss_type: str = "smooth_l1"

    # --- Detection -------------------------------------------------------
    # `num_classes=None` defers to `len(get_label_map())` at trainer build time.
    num_classes: int | None = None
    # Anchor geometry. `None` defers to `detection_head.DEFAULT_ANCHOR_SCALES` /
    # `DEFAULT_ASPECT_RATIOS` so existing runs are unchanged; set explicitly
    # (e.g. from `calibrate_anchors`) to retarget anchors to the DENTEX box
    # distribution. The values used are recorded in the manifest so serve/eval
    # reconstruct the identical anchor set.
    anchor_scales: list[int] | None = None
    aspect_ratios: list[float] | None = None
    # Anchor layout. `absolute` (default) keeps the legacy behavior so existing
    # runs are byte-identical. `per_level` switches to stride-scaled RetinaNet
    # anchors: each level's base size is `stride * anchor_base_scale`, multiplied
    # by `anchor_octaves` and `aspect_ratios`. `anchor_scales` is ignored in
    # per_level mode. `anchor_octaves=None` defers to the module default
    # ({2^0, 2^(1/3), 2^(2/3)}). All three are recorded in the manifest so
    # serve/eval rebuild the identical anchor set.
    anchor_layout: str = "absolute"
    anchor_base_scale: float = 4.0
    anchor_octaves: list[float] | None = None
    # Run NMS per predicted class (torchvision.batched_nms) instead of one
    # class-agnostic pass, so a lesion box inside its tooth box is not
    # suppressed by a higher-scoring cross-class detection.
    nms_per_class: bool = False
    # Decode/NMS thresholds applied at eval + serve time. These were previously
    # hardcoded as `DetectionModel` defaults; surfacing them lets the campaign
    # grid them post-hoc on a trained checkpoint (a free, no-retrain lever) and
    # records the chosen values in the manifest so serve/eval reproduce them.
    score_threshold: float = 0.05
    nms_iou_threshold: float = 0.5
    max_detections: int = 100

    # --- Backbone adaptation --------------------------------------------
    # `backbone_mode` is the modern knob (frozen|lora|full|partial). `use_lora`
    # is retained for back-compat: when it's True and `backbone_mode` is left at
    # the "frozen" default, the effective mode resolves to "lora"
    # (see `effective_backbone_mode`).
    backbone_mode: str = "frozen"
    # Discriminative LR for the (un)frozen backbone params. Much smaller than the
    # head LR (`lr`) to avoid catastrophic forgetting during fine-tuning.
    backbone_lr: float = 1e-5
    # For `partial` mode: number of trailing transformer blocks to unfreeze.
    backbone_trainable_blocks: int = 0

    # --- PEFT ------------------------------------------------------------
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 32.0

    # --- DDP -------------------------------------------------------------
    # `find_unused_parameters=True` is correct here because the backbone is
    # frozen — DDP's reducer needs to know not to wait on grads from the
    # frozen subtree. Routing only-trainable params through DDP would let
    # us flip this to False, but that's structural surgery deferred past
    # Phase 5 (see docs/RUNBOOK.md#ddp-trainable-only).
    ddp_find_unused: bool = True

    # `safe_barrier` deadline before we surface a `BarrierTimeoutError`
    # instead of hanging until the NCCL job-level timeout. Default is the
    # same 10 minutes as the legacy implicit deadline; bump for very large
    # checkpoint waits, drop for fast-fail in CI.
    barrier_timeout_seconds: float = 600.0

    # --- MLflow / UC -----------------------------------------------------
    experiment_name: str | None = None
    model_name: str = "cradio_detector"
    register_model: bool = True
    set_candidate_alias: bool = True

    # --- Resume / checkpointing -----------------------------------------
    resume_from_checkpoint: str | None = None

    # --- Misc ------------------------------------------------------------
    base_seed: int = 42

    # --- Class-level: type maps for from_dict coercion -------------------
    # Tracked here so callers can introspect ("which fields are floats?") and
    # so the coercion logic stays a single dispatch instead of three parallel
    # constants like the old cli.py had.
    _INT_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "epochs",
            "batch_size",
            "num_workers",
            "grad_accum_steps",
            "lora_rank",
            "img_size",
            "num_classes",
            "max_detections",
            "base_seed",
            "backbone_trainable_blocks",
            "flat_loss_patience",
        }
    )
    _FLOAT_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "lr",
            "weight_decay",
            "grad_clip_norm",
            "onecycle_pct_start",
            "focal_alpha",
            "focal_gamma",
            "box_loss_weight",
            "lora_alpha",
            "barrier_timeout_seconds",
            "backbone_lr",
            "anchor_base_scale",
            "score_threshold",
            "nms_iou_threshold",
            "aug_hflip_prob",
            "aug_jitter_prob",
            "aug_jitter_scale",
            "aug_rotation_deg",
            "caries_oversample",
        }
    )
    _BOOL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "use_lora",
            "register_model",
            "set_candidate_alias",
            "ddp_find_unused",
            "nms_per_class",
        }
    )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> TrainerConfig:
        """Build a config from a (possibly stringly-typed) dict.

        Coerces YAML/CLI-shaped values into the typed fields. Tolerant of
        unknown keys — they're dropped with a warning by `cli.py`, but here
        we silently ignore so subclasses / tests don't have to mirror every
        new field.
        """
        valid = {f.name for f in fields(cls)}
        cleaned: dict[str, Any] = {}
        for k, v in params.items():
            if k not in valid:
                continue
            if v is None:
                cleaned[k] = None
                continue
            if k in cls._INT_FIELDS:
                cleaned[k] = int(v)
            elif k in cls._FLOAT_FIELDS:
                cleaned[k] = float(v)
            elif k in cls._BOOL_FIELDS:
                cleaned[k] = _coerce_bool(v)
            elif k in ("anchor_scales", "fusion_layers"):
                cleaned[k] = [int(x) for x in v]
            elif k in ("aspect_ratios", "anchor_octaves", "aug_multiscale_range"):
                cleaned[k] = [float(x) for x in v]
            else:
                cleaned[k] = v
        # Backbone alias resolution.
        bn = cleaned.get("backbone_name")
        if isinstance(bn, str) and bn in BACKBONE_ALIASES:
            cleaned["backbone_name"] = BACKBONE_ALIASES[bn]
        return cls(**cleaned)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainerConfig:
        """Read a YAML file and dispatch to `from_dict`."""
        p = Path(path)
        with p.open() as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML at {p} did not produce a mapping; got {type(data).__name__}")
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_mlflow_params(self) -> dict[str, str]:
        """Stringify everything for `mlflow.log_params`. None → "" so the
        param appears in the run (it's nicer for diffs than a missing key).
        """
        out: dict[str, str] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = "" if v is None else str(v)
        return out

    def to_kwargs_for_train_detector(self) -> dict[str, Any]:
        """Subset of fields that the legacy `train_detector(...)` signature
        accepts. Used during Phase 2/3 cohabitation while the legacy call
        path still exists; Phase 3 collapses this to `vars(self)`.
        """
        # The legacy signature is what `train/cli.py` used to filter to;
        # we re-create that filter here once instead of duplicating it.
        legacy_keys = {
            "catalog",
            "schema",
            "backbone_name",
            "backbone_revision",
            "volume_path",
            "cache_dir",
            "epochs",
            "lr",
            "batch_size",
            "num_workers",
            "use_lora",
            "lora_rank",
            "lora_alpha",
            "experiment_name",
            "model_name",
            "register_model",
            "set_candidate_alias",
            "img_size",
            "base_seed",
        }
        return {k: getattr(self, k) for k in legacy_keys}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise `ValueError` for any internally-inconsistent combination.

        Type errors are caught by the dataclass itself; this catches
        *semantic* errors (negative epochs, lr outside reasonable range,
        unknown backbone). Called explicitly — kept out of `__post_init__`
        so `from_dict` can preserve "raw → coerced → validated" as three
        clear stages.
        """
        errs: list[str] = []
        if not self.catalog or "." in self.catalog:
            errs.append(f"catalog must be a non-empty UC identifier, got {self.catalog!r}")
        if not self.schema or "." in self.schema:
            errs.append(f"schema must be a non-empty UC identifier, got {self.schema!r}")
        if not self.model_name:
            errs.append("model_name must be non-empty")
        if self.backbone_name not in ALLOWED_BACKBONES:
            errs.append(
                f"backbone_name {self.backbone_name!r} not in {ALLOWED_BACKBONES}; "
                "did you forget to add it to BACKBONE_ALIASES?"
            )
        if self.epochs < 1:
            errs.append(f"epochs must be >= 1, got {self.epochs}")
        if self.lr <= 0:
            errs.append(f"lr must be > 0, got {self.lr}")
        if self.batch_size < 1:
            errs.append(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            errs.append(f"num_workers must be >= 0, got {self.num_workers}")
        if not (0.0 < self.onecycle_pct_start < 1.0):
            errs.append(f"onecycle_pct_start must be in (0, 1), got {self.onecycle_pct_start}")
        if self.weight_decay < 0:
            errs.append(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.grad_clip_norm <= 0:
            errs.append(f"grad_clip_norm must be > 0, got {self.grad_clip_norm}")
        if self.img_size < 32 or self.img_size % 16 != 0:
            errs.append(f"img_size must be a multiple of 16 and >= 32, got {self.img_size}")
        if self.use_lora and self.lora_rank < 1:
            errs.append(f"lora_rank must be >= 1 when use_lora=True, got {self.lora_rank}")
        if self.use_lora and self.lora_alpha <= 0:
            errs.append(f"lora_alpha must be > 0 when use_lora=True, got {self.lora_alpha}")
        if self.backbone_mode not in ALLOWED_BACKBONE_MODES:
            errs.append(f"backbone_mode {self.backbone_mode!r} not in {ALLOWED_BACKBONE_MODES}")
        if self.backbone_lr <= 0:
            errs.append(f"backbone_lr must be > 0, got {self.backbone_lr}")
        if self.backbone_trainable_blocks < 0:
            errs.append(f"backbone_trainable_blocks must be >= 0, got {self.backbone_trainable_blocks}")
        if self.backbone_mode == "partial" and self.backbone_trainable_blocks < 1:
            errs.append("backbone_mode='partial' requires backbone_trainable_blocks >= 1")
        if self.anchor_scales is not None and (
            len(self.anchor_scales) == 0 or any(s <= 0 for s in self.anchor_scales)
        ):
            errs.append(f"anchor_scales must be a non-empty list of positive ints, got {self.anchor_scales}")
        if self.aspect_ratios is not None and (
            len(self.aspect_ratios) == 0 or any(r <= 0 for r in self.aspect_ratios)
        ):
            errs.append(f"aspect_ratios must be a non-empty list of positive floats, got {self.aspect_ratios}")
        if self.anchor_layout not in ALLOWED_ANCHOR_LAYOUTS:
            errs.append(f"anchor_layout {self.anchor_layout!r} not in {ALLOWED_ANCHOR_LAYOUTS}")
        if self.anchor_base_scale <= 0:
            errs.append(f"anchor_base_scale must be > 0, got {self.anchor_base_scale}")
        if self.anchor_octaves is not None and (
            len(self.anchor_octaves) == 0 or any(o <= 0 for o in self.anchor_octaves)
        ):
            errs.append(f"anchor_octaves must be a non-empty list of positive floats, got {self.anchor_octaves}")
        if self.num_classes is not None and self.num_classes < 1:
            errs.append(f"num_classes must be >= 1 (or None to derive), got {self.num_classes}")
        if self.barrier_timeout_seconds <= 0:
            errs.append(f"barrier_timeout_seconds must be > 0, got {self.barrier_timeout_seconds}")
        if self.amp_dtype not in ALLOWED_AMP_DTYPES:
            errs.append(f"amp_dtype {self.amp_dtype!r} not in {ALLOWED_AMP_DTYPES}")
        if self.flat_loss_patience < 0:
            errs.append(f"flat_loss_patience must be >= 0, got {self.flat_loss_patience}")
        if self.grad_accum_steps < 1:
            errs.append(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if not (0.0 <= self.score_threshold < 1.0):
            errs.append(f"score_threshold must be in [0, 1), got {self.score_threshold}")
        if not (0.0 < self.nms_iou_threshold <= 1.0):
            errs.append(f"nms_iou_threshold must be in (0, 1], got {self.nms_iou_threshold}")
        if self.max_detections < 1:
            errs.append(f"max_detections must be >= 1, got {self.max_detections}")
        if not (0.0 <= self.aug_hflip_prob <= 1.0):
            errs.append(f"aug_hflip_prob must be in [0, 1], got {self.aug_hflip_prob}")
        if not (0.0 <= self.aug_jitter_prob <= 1.0):
            errs.append(f"aug_jitter_prob must be in [0, 1], got {self.aug_jitter_prob}")
        if self.aug_jitter_scale < 0:
            errs.append(f"aug_jitter_scale must be >= 0, got {self.aug_jitter_scale}")
        if self.aug_rotation_deg < 0:
            errs.append(f"aug_rotation_deg must be >= 0, got {self.aug_rotation_deg}")
        if self.aug_multiscale_range is not None:
            r = self.aug_multiscale_range
            if len(r) != 2 or not (0.0 < r[0] <= r[1] <= 1.0):
                errs.append(
                    "aug_multiscale_range must be [lo, hi] with 0 < lo <= hi <= 1, "
                    f"got {self.aug_multiscale_range}"
                )
        if self.box_loss_type not in ALLOWED_BOX_LOSS_TYPES:
            errs.append(f"box_loss_type {self.box_loss_type!r} not in {ALLOWED_BOX_LOSS_TYPES}")
        if self.caries_oversample < 1.0:
            errs.append(f"caries_oversample must be >= 1.0, got {self.caries_oversample}")
        if self.fusion_layers is not None:
            if len(self.fusion_layers) == 0:
                errs.append("fusion_layers must be a non-empty list of ints (or None)")
            if self.backbone_name != "dinov3_vitl16":
                errs.append(
                    "fusion_layers is currently only supported for dinov3_vitl16 "
                    f"(HF output_hidden_states), not {self.backbone_name!r}"
                )
        if errs:
            raise ValueError("TrainerConfig validation failed:\n  - " + "\n  - ".join(errs))

    def with_overrides(self, **overrides: Any) -> TrainerConfig:
        """Return a new TrainerConfig with the given fields replaced.

        Useful for tests and notebook drivers that want to load a base YAML
        and tweak one or two fields without re-typing the whole thing.
        """
        return dataclasses.replace(self, **overrides)

    def effective_backbone_mode(self) -> str:
        """Resolve the backbone-adaptation mode, honoring the legacy `use_lora`.

        `backbone_mode` is authoritative. The legacy `use_lora=True` flag maps
        onto `"lora"` only when `backbone_mode` is still at its `"frozen"`
        default, so old callers that set `use_lora=True` keep working while a
        caller that explicitly sets `backbone_mode` wins.
        """
        if self.backbone_mode == "frozen" and self.use_lora:
            return "lora"
        return self.backbone_mode

    def effective_amp_dtype(self) -> str:
        """Resolve the autocast precision, honoring the "auto" backbone policy.

        "auto" maps DINOv3 to fp32 and everything else to fp16 (C-RADIO's stable,
        proven path). DINOv3 is autocast-unstable here: fp16 NaNs the
        RoPE/LayerScale encoder (GradScaler then skips every step → dead-flat
        loss), and an empirical smoke (docs/HPO.md "DINOv3 A/B") showed **bf16
        also NaNs the forward at step 0** for this detector stack — only fp32
        (autocast disabled) trains (loss 1.39→0.77, val mAP@50 0→0.22 in 3 ep).
        An explicit `amp_dtype` always wins, so bf16 stays available to retest.
        """
        if self.amp_dtype != "auto":
            return self.amp_dtype
        if self.backbone_name == "dinov3_vitl16":
            return "fp32"
        return "fp16"


# Module-level so it's testable. Accepts the YAML scalar shapes plus
# common stringly-typed inputs ("True"/"true"/"1"/"yes"). Anything else is
# the user fat-fingering — fail fast.
def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "on", "y"}:
            return True
        if s in {"false", "0", "no", "off", "n", ""}:
            return False
    raise ValueError(f"Cannot coerce {v!r} to bool")


# Re-export `field` so callers building configs programmatically don't have
# to also import `dataclasses`.
__all__ = [
    "ALLOWED_AMP_DTYPES",
    "ALLOWED_BACKBONES",
    "BACKBONE_ALIASES",
    "TrainerConfig",
    "field",
]
