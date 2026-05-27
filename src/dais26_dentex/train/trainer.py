"""`Trainer` class — owns the train + validate + checkpoint + register loop.

Replaces the 360-line procedural body of `train_detector(...)`. The thin
orchestrator at `train_detector.py` now constructs a `TrainerConfig` and
delegates here.

Key correctness fixes vs. the legacy:
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
import logging
import tempfile
from pathlib import Path
from typing import Any, cast

import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from dais26_dentex.config.constants import MANIFEST_FILE, MODEL_STATE_FILE
from dais26_dentex.config.manifest import BackboneSpec, DetectorSpec, Manifest
from dais26_dentex.config.trainer_config import TrainerConfig
from dais26_dentex.data.dataset import DENTEXDetectionDataset, detection_collate
from dais26_dentex.data.dentex_loader import get_label_map
from dais26_dentex.data.transforms import get_train_transforms, get_val_transforms
from dais26_dentex.distributed.primitives import (
    BarrierTimeoutError,
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
from dais26_dentex.platform.mlflow_io import AliasingError, MlflowReporter
from dais26_dentex.platform.uc import UCName
from dais26_dentex.serve.detector_pyfunc import DetectorPyfunc, build_signature_and_example
from dais26_dentex.train.losses import detection_loss

logger = logging.getLogger(__name__)


# Re-exported for callers that import these symbols from this module.
# Note: the canonical `__all__` is at module bottom — this earlier one is
# a documentation breadcrumb only and is overwritten before import-time
# settles.


class Trainer:
    """Owns one `train_detector` invocation end-to-end.

    Construction is cheap (no side effects beyond device init); `run()` is
    where MLflow / DDP / training happen. Splitting them lets tests
    instantiate without a full training environment.
    """

    def __init__(self, cfg: TrainerConfig) -> None:
        cfg.validate()
        self.cfg = cfg
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

    # ------------------------------------------------------------------
    # Build helpers (called once from run())
    # ------------------------------------------------------------------

    def _build_model(self) -> tuple[nn.Module, Any]:
        model, info = build_detector(self.cfg, device=self.device)
        if is_distributed():
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                find_unused_parameters=self.cfg.ddp_find_unused,
                broadcast_buffers=False,
            )
        return model, info

    def _build_loaders(
        self,
    ) -> tuple[DataLoader | None, DataLoader | None, Any]:
        cfg = self.cfg
        if cfg.volume_path is None:
            return None, None, None
        train_ds = DENTEXDetectionDataset(
            volume_path=cfg.volume_path,
            split="train",
            transforms=get_train_transforms(cfg.img_size),
        )
        val_ds = DENTEXDetectionDataset(
            volume_path=cfg.volume_path,
            split="val",
            transforms=get_val_transforms(cfg.img_size),
        )
        train_sampler = maybe_distributed_sampler(train_ds, shuffle=True)
        train_loader = DataLoader(
            train_ds,
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
        n_batches = 0

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
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
                )
                loss = losses["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, max_norm=cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += float(loss.item())
            epoch_cls += float(losses["cls_loss"].item())
            epoch_box += float(losses["box_loss"].item())
            n_batches += 1

        # All-reduce so the logged metric is the global mean across ranks.
        if is_distributed():
            t = torch.tensor(
                [epoch_loss, epoch_cls, epoch_box, float(n_batches)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            denom = max(t[3].item(), 1.0)
            return {
                "train/loss": float(t[0].item() / denom),
                "train/cls_loss": float(t[1].item() / denom),
                "train/box_loss": float(t[2].item() / denom),
            }
        denom = max(n_batches, 1)
        return {
            "train/loss": epoch_loss / denom,
            "train/cls_loss": epoch_cls / denom,
            "train/box_loss": epoch_box / denom,
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
        # used by `dentex_loader.load_canonical_split`.
        gt_path = Path(cfg.volume_path or ".") / "canonical" / "val.json"
        if not gt_path.exists():
            logger.warning("Val GT not found at %s; skipping mAP.", gt_path)
            return {}

        preds: list[dict[str, Any]] = []
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=(self.device.type == "cuda"),
            ):
                out = bare(images)
            for i, t in enumerate(targets):
                preds.append(
                    {
                        "image_id": int(t["image_id"].item()) if hasattr(t["image_id"], "item") else int(t["image_id"]),
                        "boxes": out["boxes"][i].cpu(),
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
        return Manifest(
            backbone=BackboneSpec(
                name=cfg.backbone_name,
                revision=cfg.backbone_revision,
                summary_dim=info.summary_dim,
                spatial_dim=info.spatial_dim,
                patch_size=info.patch_size,
            ),
            detector=DetectorSpec(
                num_classes=self.num_classes,
                scales=list(DEFAULT_ANCHOR_SCALES),
                aspect_ratios=list(DEFAULT_ASPECT_RATIOS),
                input_size=cfg.img_size,
            ),
            label_map={str(k): v for k, v in get_label_map().items()},
            # Provenance only — drift-debug breadcrumb. See manifest.py.
            trainer={
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "batch_size": cfg.batch_size,
                "img_size": cfg.img_size,
                "best_epoch": self._best_epoch,
                "best_val_mAP_50": self._best_metric,
            },
        )

    def _save_and_register(self, info: Any) -> None:
        """Rank-0 only: write v2 artifacts, log pyfunc, set @candidate alias."""
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
            reporter.log_pyfunc(
                python_model=DetectorPyfunc(),
                artifacts=artifacts,
                signature=signature,
                input_example=example,
                registered_model_name=full_model if cfg.register_model else None,
            )

        if cfg.register_model and cfg.set_candidate_alias:
            assert self.run_id is not None, "set_candidate_alias requires an active MLflow run"
            version = reporter.set_candidate_alias(full_model=full_model, run_id=self.run_id)
            mlflow.log_param("registered_version", version)

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
        train_loader, val_loader, train_sampler = self._build_loaders()

        trainable = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)
        logger.info("Trainable params: %d", n_trainable)

        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

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
                        }
                    )
                    mlflow.log_params(params)

                if train_loader is not None:
                    steps_per_epoch = max(len(train_loader), 1)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=cfg.lr,
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

            if val_loader is not None:
                val_metrics = self._validate_one_epoch(model, val_loader)
                if is_rank0():
                    for name, val in val_metrics.items():
                        mlflow.log_metric(name, val, step=epoch)
                    map50 = val_metrics.get("val/mAP_50", 0.0)
                    if val_metrics and map50 > self._best_metric:
                        self._best_metric = map50
                        self._best_epoch = epoch
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

    def _safe_barrier_or_skip(self) -> None:
        """Pre-save sync that surfaces dead-rank deadlocks as typed errors.

        Wraps `safe_barrier` so the upgrade from the legacy unconditional
        `barrier()` (which would hang if a peer died) is a one-liner here
        and the timeout flows from `TrainerConfig`.
        """
        if is_distributed():
            safe_barrier(self.cfg.barrier_timeout_seconds)


__all__ = ["AliasingError", "BarrierTimeoutError", "Trainer"]
