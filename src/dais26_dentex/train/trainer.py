"""`Trainer` class — owns the train + validate + checkpoint + register loop.

The single training core behind every launch surface: the notebook
`@distributed` closure (02 / 02b), the sgcli/torchrun CLIs (`train.cli`,
`train.sweep_cli`), and tests all construct a `TrainerConfig` and call
`Trainer(cfg).run()`.

Key correctness fixes vs. the legacy procedural implementation:
  * `_build_targets` smoke stub → IoU-based matcher (`models.targets`)
  * cls-only loss → `detection_loss` (focal cls + smooth-L1 box)
  * hardcoded `num_classes=4` → `resolve_num_classes(cfg)`
  * `_val_loader` discarded → real validation each epoch with COCO mAP
  * "last epoch" save → "best epoch by val/map50" save
  * swallowed `set_candidate_alias` failure → typed exception surfaced
  * unconditional pre-save `barrier()` → `safe_barrier` with bounded
    async-wait — surfaces dead-rank deadlocks as a typed
    `BarrierTimeoutError` instead of hanging until the NCCL job-level
    timeout.

The class is single-responsibility per phase: `__init__` builds, `run()`
trains. State that needs to outlive a method (best-state-dict, run id) is
on `self`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import tempfile
from pathlib import Path
from typing import Any, cast

import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from dais26_dentex.config.constants import ALIAS_CANDIDATE, MANIFEST_FILE, MODEL_STATE_FILE
from dais26_dentex.config.manifest import BackboneSpec, DetectorSpec, Manifest
from dais26_dentex.config.trainer_config import TrainerConfig
from dais26_dentex.data.dataset import (
    DENTEXDetectionDataset,
    IndexRemapDataset,
    build_caries_oversampled_indices,
    detection_collate,
)
from dais26_dentex.data.dentex_loader import get_label_map
from dais26_dentex.data.transforms import get_train_transforms, get_val_transforms
from dais26_dentex.distributed.primitives import (
    is_distributed,
    is_rank0,
    maybe_distributed_sampler,
    safe_barrier,
    seed_per_rank,
    setup_distributed,
    teardown_distributed,
    unwrap_model,
    world_size,
)
from dais26_dentex.eval.coco_metrics import evaluate_coco, format_predictions_for_coco
from dais26_dentex.models.builder import build_detector, resolve_num_classes
from dais26_dentex.models.detection_head import (
    DEFAULT_ANCHOR_SCALES,
    DEFAULT_ASPECT_RATIOS,
    DetectionModel,
)
from dais26_dentex.models.targets import build_targets_for_batch
from dais26_dentex.platform.mlflow_io import MlflowReporter
from dais26_dentex.platform.uc import UCName
from dais26_dentex.serve.detector_pyfunc import build_signature_and_example
from dais26_dentex.train.losses import detection_loss

logger = logging.getLogger(__name__)


class Trainer:
    """Owns one `train_detector` invocation end-to-end.

    Construction is cheap (no side effects beyond device init); `run()` is
    where MLflow / DDP / training happen. Splitting them lets tests
    instantiate without a full training environment.
    """

    def __init__(self, cfg: TrainerConfig, *, manage_process_group: bool = True) -> None:
        cfg.validate()
        self.cfg = cfg
        # Launcher plumbing, deliberately NOT a TrainerConfig field (it would
        # leak into to_mlflow_params and the reproducibility surface). True =
        # this Trainer owns the PG lifecycle (single-run launches: notebook
        # @distributed dispatch, one-shot torchrun). False = an outer driver
        # (the in-job torchrun sweep) init'ed the PG once and reuses it across
        # sequential Trainer runs — destroying it between trials would force a
        # risky NCCL re-init per trial.
        self._manage_pg = manage_process_group
        self.device: torch.device = setup_distributed()
        seed_per_rank(cfg.base_seed)
        logger.info("Trainer device=%s world_size=%d", self.device, world_size())

        self.num_classes = resolve_num_classes(cfg)
        self.run_id: str | None = None

        # Best-state tracking (rank-0 only writes; broadcast not needed
        # because only rank 0 ever serializes).
        self._best_metric: float = -1.0
        self._best_state_dict: dict[str, torch.Tensor] | None = None
        self._best_epoch: int = -1
        # Full val/* metric dict at the best epoch — logged against the
        # LoggedModel (model_id) at save time so the metrics surface on the
        # experiment Models tab and the sweep's best-in-experiment gate can
        # read them off the LoggedModel rather than the run.
        self._best_val_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Build helpers (called once from run())
    # ------------------------------------------------------------------

    def _build_model(self) -> tuple[nn.Module, Any]:
        model, info = build_detector(self.cfg, device=self.device)
        if is_distributed():
            # When the entire backbone is trainable (full fine-tune) every param
            # gets a grad, so `find_unused_parameters` can drop to False for
            # speed. Frozen / lora / partial keep the configured value because a
            # subtree is non-trainable.
            find_unused = self.cfg.ddp_find_unused and self.cfg.effective_backbone_mode() != "full"
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                find_unused_parameters=find_unused,
                broadcast_buffers=False,
            )
        return model, info

    def _configure_amp(self) -> None:
        """Resolve `cfg.amp_dtype` into the autocast enable flag + torch dtype.

        Sets ``self._amp_eff`` (the resolved "fp16"|"bf16"|"fp32" string),
        ``self._amp_enabled`` (autocast on?), and ``self._amp_dtype`` (the torch
        dtype, or None for fp32/CPU). On CPU we always disable autocast.
        """
        eff = self.cfg.effective_amp_dtype()
        self._amp_eff: str = eff
        if self.device.type != "cuda" or eff == "fp32":
            self._amp_enabled = False
            self._amp_dtype: torch.dtype | None = None
        elif eff == "bf16":
            self._amp_enabled = True
            self._amp_dtype = torch.bfloat16
        else:  # fp16
            self._amp_enabled = True
            self._amp_dtype = torch.float16
        logger.info("AMP: amp_dtype=%s -> effective=%s (autocast=%s)", self.cfg.amp_dtype, eff, self._amp_enabled)

    @staticmethod
    def _split_param_groups(
        model: nn.Module,
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        """Split trainable params into (backbone, head/FPN) groups by name.

        DDP prefixes names with ``module.``; the backbone subtree is the only
        one whose qualified name contains ``backbone``, so a substring test is
        robust to the wrap. Frozen params are excluded.

        The multi-layer fusion combiner (``...backbone.fusion.*``) is a small
        learnable head bolted onto the encoder, not pretrained weights — it is
        routed to the *head* LR group so it can actually learn, instead of
        crawling at the tiny discriminative ``backbone_lr``.
        """
        backbone_params: list[torch.nn.Parameter] = []
        head_params: list[torch.nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_backbone = "backbone" in name and "fusion" not in name
            (backbone_params if is_backbone else head_params).append(p)
        return backbone_params, head_params

    def _build_loaders(
        self,
        info: Any = None,
    ) -> tuple[DataLoader | None, DataLoader | None, Any]:
        cfg = self.cfg
        if cfg.volume_path is None:
            return None, None, None
        # Normalise with the backbone's own pretraining stats (CLIP for C-RADIO,
        # ImageNet for DINOv2/v3). `info=None` (defensive) falls back to the
        # transform defaults (CLIP).
        mean = getattr(info, "image_mean", None)
        std = getattr(info, "image_std", None)
        train_ds = DENTEXDetectionDataset(
            volume_path=cfg.volume_path,
            split="train",
            transforms=get_train_transforms(
                cfg.img_size,
                mean=mean,
                std=std,
                hflip_prob=cfg.aug_hflip_prob,
                jitter_prob=cfg.aug_jitter_prob,
                jitter_scale=cfg.aug_jitter_scale,
                rotation_deg=cfg.aug_rotation_deg,
                multiscale_range=cfg.aug_multiscale_range,
            ),
        )
        val_ds = DENTEXDetectionDataset(
            volume_path=cfg.volume_path,
            split="val",
            transforms=get_val_transforms(cfg.img_size, mean=mean, std=std),
        )
        # Class-balanced oversampling of Caries-bearing images (id 0): replicate
        # their indices so the binding Caries AP@50 gate sees more positives per
        # epoch. Index replication (not WeightedRandomSampler) keeps it DDP-safe
        # — DistributedSampler just shards a longer flat index list. Val is never
        # oversampled.
        train_dataset: object = train_ds
        if cfg.caries_oversample > 1.0:
            expanded = build_caries_oversampled_indices(
                train_ds.per_image_label_sets(), cfg.caries_oversample, positive_class=0
            )
            train_dataset = IndexRemapDataset(train_ds, expanded)
            logger.info(
                "Caries oversampling x%.2f: train samples %d -> %d",
                cfg.caries_oversample,
                len(train_ds),
                len(expanded),
            )
        train_sampler = maybe_distributed_sampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            collate_fn=detection_collate,
            persistent_workers=(cfg.num_workers > 0),
            pin_memory=torch.cuda.is_available(),
        )
        # Validation runs rank-0 only — simpler than DistributedSampler +
        # all_reduce of mAP, and the val set is small.
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=detection_collate,
            persistent_workers=(cfg.num_workers > 0),
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, val_loader, train_sampler

    # ------------------------------------------------------------------
    # Inner loops
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.amp.GradScaler,
        trainable: list[torch.nn.Parameter],
    ) -> dict[str, float]:
        cfg = self.cfg
        model.train()
        bare = cast(DetectionModel, unwrap_model(model))

        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_box = 0.0
        epoch_grad = 0.0
        n_batches = 0
        n_opt_steps = 0

        # Gradient accumulation: backward every micro-batch (grads sum into
        # .grad), but only step/zero on accumulation boundaries. The loss is
        # scaled by 1/accum so the accumulated gradient equals the mean over the
        # effective batch, matching a single large-batch step. The scheduler
        # steps once per optimizer step (its total step budget is sized in run()
        # from ceil(len(loader)/accum)).
        accum = max(1, cfg.grad_accum_steps)
        n_micro = len(loader)
        optimizer.zero_grad()
        for micro_idx, (images, targets) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            is_boundary = ((micro_idx + 1) % accum == 0) or (micro_idx + 1 == n_micro)
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self._amp_enabled,
                dtype=self._amp_dtype,
            ):
                cls_logits, box_pred, anchors = bare.forward_train(images)
                cls_t, box_t, fg_mask, ignore_mask = build_targets_for_batch(
                    anchors,
                    targets,
                    num_classes=self.num_classes,
                )
                losses = detection_loss(
                    cls_logits=cls_logits,
                    box_pred=box_pred,
                    cls_targets=cls_t,
                    box_targets=box_t,
                    fg_mask=fg_mask,
                    ignore_mask=ignore_mask,
                    focal_alpha=cfg.focal_alpha,
                    focal_gamma=cfg.focal_gamma,
                    box_weight=cfg.box_loss_weight,
                    box_loss_type=cfg.box_loss_type,
                    anchors=anchors,
                )
                loss = losses["loss"]

            scaler.scale(loss / accum).backward()

            if is_boundary:
                # unscale_ is a no-op when the scaler is disabled (bf16/fp32), so
                # the clip + grad-norm readout is identical across precisions.
                # Under fp16 an inf grad makes total_norm inf and scaler.step
                # skips the update — the dead-flat-loss signature we log via
                # grad_norm/amp_scale below.
                scaler.unscale_(optimizer)
                total_norm = nn.utils.clip_grad_norm_(trainable, max_norm=cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                epoch_grad += float(total_norm)
                n_opt_steps += 1

            epoch_loss += float(loss.item())
            epoch_cls += float(losses["cls_loss"].item())
            epoch_box += float(losses["box_loss"].item())
            n_batches += 1

        # grad_norm + amp_scale are diagnostics: a flat loss with finite grad_norm
        # near 0 or a collapsing amp_scale both point at AMP/precision trouble.
        amp_scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0

        # All-reduce so the logged metric is the global mean across ranks. Loss
        # terms average over micro-batches; grad_norm averages over optimizer
        # steps (they differ when grad_accum_steps > 1).
        if is_distributed():
            t = torch.tensor(
                [epoch_loss, epoch_cls, epoch_box, epoch_grad, float(n_batches), float(n_opt_steps)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            denom = max(t[4].item(), 1.0)
            grad_denom = max(t[5].item(), 1.0)
            return {
                "train/loss": float(t[0].item() / denom),
                "train/cls_loss": float(t[1].item() / denom),
                "train/box_loss": float(t[2].item() / denom),
                "train/grad_norm": float(t[3].item() / grad_denom),
                "train/amp_scale": amp_scale,
            }
        denom = max(n_batches, 1)
        grad_denom = max(n_opt_steps, 1)
        return {
            "train/loss": epoch_loss / denom,
            "train/cls_loss": epoch_cls / denom,
            "train/box_loss": epoch_box / denom,
            "train/grad_norm": epoch_grad / grad_denom,
            "train/amp_scale": amp_scale,
        }

    @torch.no_grad()
    def _validate_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> dict[str, float]:
        """Rank-0 only validation. Returns COCO-style metrics dict.

        Other ranks return an empty dict (caller logs only on rank 0). Keeps
        the impl simple without needing all_reduce on per-class APs.
        """
        if not is_rank0():
            return {}
        cfg = self.cfg
        model.eval()
        bare = cast(DetectionModel, unwrap_model(model))

        # Source the COCO ground-truth file from the canonical-split path
        # written by `convert_to_coco` and read by `dentex_loader.load_canonical_split`
        # (i.e. `{volume_path}/annotations/{split}.json` — NOT `canonical/`, which
        # never existed; the old path silently skipped val mAP every epoch).
        gt_path = Path(cfg.volume_path or ".") / "annotations" / "val.json"
        if not gt_path.exists():
            logger.warning("Val GT not found at %s; skipping mAP.", gt_path)
            return {}

        # The val transform (`get_val_transforms`) does a uniform longest-side
        # resize to `img_size` + bottom-right zero-pad, so the model emits boxes
        # in that padded `img_size` coordinate space. COCO GT in `val.json` is
        # in ORIGINAL image pixel coords, so predictions must be rescaled back by
        # `max(H, W) / img_size` per image before eval — otherwise IoU(pred, gt)
        # ~= 0 and every mAP is pinned at 0.0 even as train loss drops.
        with open(gt_path) as f:
            gt_coco = json.load(f)
        scale_by_id = {
            int(im["id"]): max(float(im["width"]), float(im["height"])) / float(cfg.img_size)
            for im in gt_coco.get("images", [])
        }

        preds: list[dict[str, Any]] = []
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self._amp_enabled,
                dtype=self._amp_dtype,
            ):
                out = bare(images)
            for i, t in enumerate(targets):
                image_id = int(t["image_id"].item()) if hasattr(t["image_id"], "item") else int(t["image_id"])
                factor = scale_by_id.get(image_id, 1.0)
                preds.append(
                    {
                        "image_id": image_id,
                        "boxes": out["boxes"][i].cpu() * factor,
                        "scores": out["scores"][i].cpu(),
                        "labels": out["labels"][i].cpu(),
                    }
                )

        coco_preds = format_predictions_for_coco(preds)
        metrics = evaluate_coco(coco_preds, str(gt_path))
        return {
            "val/mAP_50_95": metrics["mAP_50_95"],
            "val/mAP_50": metrics["mAP_50"],
            "val/mAP_75": metrics["mAP_75"],
        }

    # ------------------------------------------------------------------
    # Save + register
    # ------------------------------------------------------------------

    def _build_manifest(self, info: Any) -> Manifest:
        """Compose the v2 ``Manifest`` from this run's config + backbone info."""
        cfg = self.cfg
        # Record the anchor geometry ACTUALLY used (cfg overrides → defaults),
        # not the module defaults — otherwise tuned anchors won't be
        # reconstructed at serve/eval time and box decoding mismatches the GT.
        scales = cfg.anchor_scales if cfg.anchor_scales is not None else list(DEFAULT_ANCHOR_SCALES)
        aspect_ratios = cfg.aspect_ratios if cfg.aspect_ratios is not None else list(DEFAULT_ASPECT_RATIOS)
        mode = cfg.effective_backbone_mode()
        return Manifest(
            backbone=BackboneSpec(
                name=cfg.backbone_name,
                revision=cfg.backbone_revision,
                summary_dim=info.summary_dim,
                spatial_dim=info.spatial_dim,
                patch_size=info.patch_size,
                trained_mode=mode,
                image_mean=list(info.image_mean),
                image_std=list(info.image_std),
                fusion_layers=list(cfg.fusion_layers) if cfg.fusion_layers is not None else None,
            ),
            detector=DetectorSpec(
                num_classes=self.num_classes,
                scales=list(scales),
                aspect_ratios=list(aspect_ratios),
                score_threshold=cfg.score_threshold,
                nms_iou_threshold=cfg.nms_iou_threshold,
                max_detections=cfg.max_detections,
                input_size=cfg.img_size,
                anchor_layout=cfg.anchor_layout,
                anchor_base_scale=cfg.anchor_base_scale,
                anchor_octaves=list(cfg.anchor_octaves) if cfg.anchor_octaves is not None else None,
                nms_per_class=cfg.nms_per_class,
            ),
            label_map={str(k): v for k, v in get_label_map().items()},
            # Provenance only — drift-debug breadcrumb. See manifest.py.
            trainer={
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "backbone_lr": cfg.backbone_lr,
                "weight_decay": cfg.weight_decay,
                "batch_size": cfg.batch_size,
                "img_size": cfg.img_size,
                "backbone_mode": mode,
                "best_epoch": self._best_epoch,
                "best_val_mAP_50": self._best_metric,
            },
        )

    def _save_and_register(self, info: Any) -> None:
        """Rank-0 only: write v2 artifacts, log pyfunc, set @challenger alias."""
        cfg = self.cfg
        if self._best_state_dict is None:
            logger.warning("No best state captured; skipping save.")
            return

        reporter = MlflowReporter(experiment_name=cfg.experiment_name)
        full_model = UCName(cfg.catalog, cfg.schema, cfg.model_name).fqn

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            state_path = td_path / MODEL_STATE_FILE
            torch.save(self._best_state_dict, state_path)

            manifest_path = td_path / MANIFEST_FILE
            self._build_manifest(info).write(manifest_path)

            artifacts = {
                "model_state": str(state_path),
                "manifest": str(manifest_path),
            }
            if cfg.cache_dir is not None:
                artifacts["model_cache"] = cfg.cache_dir

            signature, example = build_signature_and_example()
            # Models-from-code: pass the loader SCRIPT path (not a DetectorPyfunc()
            # instance). Pickling the instance captured an HF `transformers_modules.*`
            # backbone reference that the serving container can't import. The script
            # is re-executed at load time instead. See detector_model_script.py.
            from dais26_dentex.serve import detector_pyfunc as _dp

            model_script = str(Path(_dp.__file__).with_name("detector_model_script.py"))
            # Log UNREGISTERED on purpose: registering happens below from the
            # LoggedModel URI (models:/<model_id>) so the UC version keeps its
            # model_id link. Passing registered_model_name here would register
            # from the run artifact and strand the version with model_id='' —
            # see MlflowReporter.register_logged_model.
            model_info = reporter.log_pyfunc(
                python_model=model_script,
                artifacts=artifacts,
                signature=signature,
                input_example=example,
                registered_model_name=None,
            )

        # MLflow 3: attach the best-epoch val metrics to the LoggedModel created
        # by log_model so they show on the experiment's Models tab and the sweep
        # gate can query them off the LoggedModel. Best-effort: a stubbed/older
        # client (no model_id, or log_metrics without model_id=) is a no-op.
        self._log_metrics_to_logged_model(model_info)

        if cfg.register_model:
            version = reporter.register_logged_model(
                model_info,
                full_model,
                alias=ALIAS_CANDIDATE if cfg.set_candidate_alias else None,
            )
            mlflow.log_param("registered_version", version)

    def _log_metrics_to_logged_model(self, model_info: Any) -> None:
        """Link the best-epoch val metrics to the LoggedModel (rank-0 only).

        ``mlflow.pyfunc.log_model`` returns a ``ModelInfo`` carrying the MLflow 3
        ``model_id`` of the LoggedModel it created. We re-log the best-epoch
        ``val/*`` metrics — plus ``val/best_mAP_50`` (the sweep's primary metric,
        ``SWEEP_PRIMARY_METRIC``) — against that ``model_id`` so they render on the
        experiment's Models tab and the best-in-experiment gate in
        ``02b_hpo_sweep.py`` can read them straight off the LoggedModel.

        Best-effort by design: a stubbed client (model_info=None / no model_id) or
        a pre-3.x ``log_metrics`` without a ``model_id`` kwarg is logged and
        swallowed — the run-level metrics already cover the legacy read path.
        """
        model_id = getattr(model_info, "model_id", None)
        if not model_id:
            return
        metrics = dict(self._best_val_metrics)
        if self._best_metric >= 0.0:
            metrics["val/best_mAP_50"] = self._best_metric
        if not metrics:
            return
        try:
            mlflow.log_metrics(metrics, model_id=model_id)
            logger.info("Logged %d best-epoch metric(s) to LoggedModel %s", len(metrics), model_id)
        except TypeError:
            logger.info("mlflow.log_metrics has no model_id kwarg; metrics live on the run only.")
        except Exception as e:
            logger.warning("Could not log metrics to LoggedModel %s: %s", model_id, e)

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run(self) -> str | None:
        """Train + validate + save. Returns MLflow run_id on rank 0."""
        cfg = self.cfg

        if is_rank0():
            mlflow.set_registry_uri("databricks-uc")
            if cfg.experiment_name:
                mlflow.set_experiment(cfg.experiment_name)

        model, info = self._build_model()
        train_loader, val_loader, train_sampler = self._build_loaders(info)

        backbone_params, head_params = self._split_param_groups(model)
        trainable = head_params + backbone_params
        n_trainable = sum(p.numel() for p in trainable)
        logger.info(
            "Trainable params: %d (head/FPN=%d, backbone=%d, mode=%s)",
            n_trainable,
            sum(p.numel() for p in head_params),
            sum(p.numel() for p in backbone_params),
            cfg.effective_backbone_mode(),
        )

        # Discriminative LRs: head/FPN at cfg.lr, backbone at the much smaller
        # cfg.backbone_lr so fine-tuning doesn't wipe pretrained features. The
        # OneCycle max_lr list must line up with the param-group order below.
        param_groups: list[dict[str, Any]] = []
        max_lrs: list[float] = []
        if head_params:
            param_groups.append({"params": head_params, "lr": cfg.lr})
            max_lrs.append(cfg.lr)
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": cfg.backbone_lr})
            max_lrs.append(cfg.backbone_lr)
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self._onecycle_max_lr: list[float] = max_lrs

        # Resolve autocast precision. DINOv3 NaNs under fp16 (RoPE + LayerScale),
        # so "auto" gives it bf16 while C-RADIO keeps its proven fp16 path. The
        # GradScaler is meaningful ONLY for fp16; for bf16/fp32 a disabled scaler
        # passes through `scale/unscale_/step/update` so the loop body is shared.
        self._configure_amp()
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.device.type == "cuda" and self._amp_eff == "fp16")
        )

        mlflow_ctx = mlflow.start_run() if is_rank0() else contextlib.nullcontext()
        try:
            with mlflow_ctx as run:
                if is_rank0() and run is not None:
                    self.run_id = run.info.run_id
                    params = cfg.to_mlflow_params()
                    params.update(
                        {
                            "summary_dim": str(info.summary_dim),
                            "spatial_dim": str(info.spatial_dim),
                            "patch_size": str(info.patch_size),
                            "trainable_params": str(n_trainable),
                            "device": str(self.device),
                            "world_size": str(world_size()),
                            "num_classes": str(self.num_classes),
                            "amp_dtype_effective": self._amp_eff,
                        }
                    )
                    mlflow.log_params(params)
                    self._log_dataset_lineage()

                if train_loader is not None:
                    # One scheduler step per optimizer step. With gradient
                    # accumulation that is ceil(micro_batches / grad_accum_steps).
                    accum = max(1, cfg.grad_accum_steps)
                    steps_per_epoch = max(math.ceil(len(train_loader) / accum), 1)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=self._onecycle_max_lr,
                        epochs=cfg.epochs,
                        steps_per_epoch=steps_per_epoch,
                        pct_start=cfg.onecycle_pct_start,
                    )
                    self._epoch_loop(
                        model,
                        train_loader,
                        val_loader,
                        train_sampler,
                        optimizer,
                        scheduler,
                        scaler,
                        trainable,
                    )

                self._safe_barrier_or_skip()

                if is_rank0():
                    # If no training ran (config-only smoke), persist the
                    # initialized weights as the "best" so the artifact
                    # contract is still satisfied.
                    if self._best_state_dict is None:
                        self._best_state_dict = {
                            k: v.detach().cpu().clone() for k, v in unwrap_model(model).state_dict().items()
                        }
                    self._save_and_register(info)
        finally:
            if self._manage_pg:
                teardown_distributed()
        return self.run_id

    def _epoch_loop(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        train_sampler: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.amp.GradScaler,
        trainable: list[torch.nn.Parameter],
    ) -> None:
        cfg = self.cfg
        first_loss: float | None = None
        best_loss: float = float("inf")
        for epoch in range(cfg.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = self._train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                trainable,
            )
            if is_rank0():
                for name, val in train_metrics.items():
                    mlflow.log_metric(name, val, step=epoch)
                logger.info(
                    "Epoch %d %s",
                    epoch,
                    " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()),
                )

            # Fail-fast guard. train_metrics is identical across ranks (all-reduce),
            # so every rank raises together at the same epoch — no barrier skew.
            loss_now = train_metrics.get("train/loss", float("inf"))
            if first_loss is None:
                first_loss = loss_now
            best_loss = min(best_loss, loss_now)
            self._check_flat_loss(epoch, first_loss, best_loss)

            if val_loader is not None:
                val_metrics = self._validate_one_epoch(model, val_loader)
                if is_rank0():
                    for name, val in val_metrics.items():
                        mlflow.log_metric(name, val, step=epoch)
                    map50 = val_metrics.get("val/mAP_50", 0.0)
                    if val_metrics and map50 > self._best_metric:
                        self._best_metric = map50
                        self._best_epoch = epoch
                        self._best_val_metrics = dict(val_metrics)
                        self._best_state_dict = {
                            k: v.detach().cpu().clone() for k, v in unwrap_model(model).state_dict().items()
                        }
                        mlflow.log_metric("val/best_mAP_50", map50, step=epoch)
                        logger.info("Epoch %d new best val/mAP_50=%.4f", epoch, map50)

        if is_rank0() and self._best_state_dict is None:
            # Nothing improved (possible if val never ran, e.g. tiny CPU
            # smoke). Fall back to last-epoch weights so save still works.
            self._best_state_dict = {k: v.detach().cpu().clone() for k, v in unwrap_model(model).state_dict().items()}
            self._best_epoch = cfg.epochs - 1

    def _check_flat_loss(self, epoch: int, first_loss: float, best_loss: float) -> None:
        """Abort a dead run before it burns the full schedule.

        Fires when `flat_loss_patience > 0` and, after that many epochs, the
        best `train/loss` seen has not improved on the first epoch's loss (within
        a 0.1% relative tolerance). A perfectly flat loss is the signature of
        AMP/precision instability skipping every optimizer step (see docs/HPO.md
        "DINOv3 A/B"); failing fast here saves the rest of the GPU budget.
        """
        patience = self.cfg.flat_loss_patience
        if patience <= 0 or (epoch + 1) < patience:
            return
        if best_loss >= first_loss * (1.0 - 1e-3):
            raise RuntimeError(
                f"Flat-loss guard tripped after {epoch + 1} epochs: train/loss has not "
                f"decreased (first={first_loss:.6f}, best={best_loss:.6f}). This is the "
                "AMP/precision-instability signature (e.g. fp16 on DINOv3 NaNing and the "
                "GradScaler skipping every step). Check train/grad_norm and train/amp_scale; "
                "try amp_dtype=bf16 or fp32. Aborting to save GPU budget."
            )

    def _log_dataset_lineage(self) -> None:
        """Log the DENTEX train split as an MLflow dataset input (rank-0 only).

        Surfaces the training data on the model version's Lineage tab. The source
        is the canonical COCO annotation file in the UC Volume; the digest comes
        from a small per-split summary frame. Best-effort: any failure (offline
        unit env, missing annotations) is logged and swallowed so it never breaks
        a training run.
        """
        cfg = self.cfg
        if cfg.volume_path is None:
            return
        try:
            import pandas as pd

            from dais26_dentex.data.dentex_loader import load_canonical_split

            train = load_canonical_split(cfg.volume_path, "train")
            summary = pd.DataFrame(
                {
                    "split": ["train"],
                    "num_images": [len(train.get("images", []))],
                    "num_annotations": [len(train.get("annotations", []))],
                }
            )
            dataset = mlflow.data.from_pandas(
                summary,
                source=f"{cfg.volume_path}/annotations/train.json",
                name="DENTEX-train",
            )
            mlflow.log_input(dataset, context="training")
            logger.info("Logged DENTEX-train dataset lineage (%s images)", summary["num_images"][0])
        except Exception as e:
            logger.warning("Dataset lineage logging skipped: %s", e)

    def _safe_barrier_or_skip(self) -> None:
        """Pre-save sync that surfaces dead-rank deadlocks as typed errors.

        Wraps `safe_barrier` so the upgrade from the legacy unconditional
        `barrier()` (which would hang if a peer died) is a one-liner here
        and the timeout flows from `TrainerConfig`.
        """
        if is_distributed():
            safe_barrier(self.cfg.barrier_timeout_seconds)


__all__ = ["Trainer"]
