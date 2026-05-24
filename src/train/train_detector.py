from __future__ import annotations

import contextlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Literal

import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

from src.train.distributed_utils import (
    barrier,
    is_distributed,
    is_rank0,
    maybe_distributed_sampler,
    seed_per_rank,
    setup_distributed,
    teardown_distributed,
    unwrap_model,
    world_size,
)

logger = logging.getLogger(__name__)

BackboneName = Literal["cradio_v4_so400m", "dinov3_vitl16", "dinov2_base"]


def _build_targets(
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    total_anchors: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cls_target, box_target) for a batch.

    Simplified target assignment for smoke testing: marks one anchor per ground-truth box
    as positive for the gt class. In production, this would use IoU-based assignment.
    Returns: cls_target (B, N, C) one-hot, box_target (B, N, 4) zeros (placeholder).
    """
    batch_size = len(targets)
    cls_target = torch.zeros(batch_size, total_anchors, num_classes, device=device)
    box_target = torch.zeros(batch_size, total_anchors, 4, device=device)
    for i, t in enumerate(targets):
        if t["labels"].numel() == 0:
            continue
        # Distribute gt labels across the first few anchors deterministically
        for j, lbl in enumerate(t["labels"].tolist()):
            if j >= total_anchors:
                break
            cls_target[i, j, int(lbl) % num_classes] = 1.0
    return cls_target, box_target


def train_detector(
    catalog: str,
    schema: str,
    backbone_name: BackboneName = "cradio_v4_so400m",
    backbone_revision: str | None = None,
    volume_path: str | None = None,
    cache_dir: str | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 8,
    num_workers: int = 4,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 32.0,
    experiment_name: str | None = None,
    model_name: str = "cradio_detector",
    register_model: bool = True,
    set_candidate_alias: bool = True,
    img_size: int = 1024,
    base_seed: int = 42,
) -> str | None:
    """Train the detection head on DENTEX. Distributed-aware (DDP).

    Pipeline:
        1. setup_distributed() — init NCCL when WORLD_SIZE>1
        2. mlflow.set_registry_uri('databricks-uc'), set_experiment (rank-0 only)
        3. Load frozen backbone via src.models.backbones.load_backbone
        4. Build DetectionModel (FPN + RetinaNet head)
        5. Optionally apply LoRA to backbone (use_lora=True)
        6. Wrap with DistributedDataParallel when WORLD_SIZE>1
        7. Train with AdamW + OneCycleLR + AMP
        8. all_reduce per-epoch loss; rank-0 logs to MLflow
        9. Rank-0 only: save model_state, configs, label_map as artifacts
       10. Rank-0 only: log pyfunc with signature + input_example, register in UC
       11. Rank-0 only: set @candidate alias (NOT @champion -- that happens after smoke test)
       12. teardown_distributed()

    Returns: MLflow run_id on rank 0, None on other ranks.
    """
    from src.data.dataset import DENTEXDetectionDataset, detection_collate
    from src.data.dentex_loader import get_label_map
    from src.data.transforms import get_train_transforms, get_val_transforms
    from src.models.backbones import load_backbone
    from src.models.detection_head import DEFAULT_ANCHOR_SCALES, DEFAULT_ASPECT_RATIOS, DetectionModel
    from src.models.peft import apply_lora
    from src.serve.detector_pyfunc import DetectorPyfunc, build_signature_and_example

    device = setup_distributed()
    seed_per_rank(base_seed)
    logger.info("Training on device=%s world_size=%d", device, world_size())

    if is_rank0():
        mlflow.set_registry_uri("databricks-uc")
        if experiment_name:
            mlflow.set_experiment(experiment_name)

    backbone, info = load_backbone(
        name=backbone_name,
        revision=backbone_revision,
        cache_dir=cache_dir,
        device=str(device),
    )

    if use_lora:
        backbone = apply_lora(backbone, rank=lora_rank, alpha=lora_alpha)
        logger.info("LoRA injected (rank=%d, alpha=%.1f)", lora_rank, lora_alpha)

    model = DetectionModel(
        backbone=backbone,
        spatial_dim=info.spatial_dim,
        num_classes=4,
        scales=DEFAULT_ANCHOR_SCALES,
        aspect_ratios=DEFAULT_ASPECT_RATIOS,
        patch_size=info.patch_size,
    ).to(device)

    # DDP wrap. find_unused_parameters=True is REQUIRED for the frozen backbone —
    # without it DDP's reducer waits forever for grads from backbone params.
    # broadcast_buffers=False — backbone is in eval(); its LayerNorm stats never change.
    if is_distributed():
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    # Rebuild trainable list from the (possibly wrapped) model so the optimizer
    # tracks the exact Parameter objects DDP will sync.
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info("Trainable params: %d", n_trainable)

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Build dataloaders (skip when volume_path is None to allow smoke-test config-only mode)
    train_loader = None
    _val_loader = None
    train_sampler = None
    if volume_path is not None:
        train_ds = DENTEXDetectionDataset(
            volume_path=volume_path, split="train", transforms=get_train_transforms(img_size),
        )
        val_ds = DENTEXDetectionDataset(
            volume_path=volume_path, split="val", transforms=get_val_transforms(img_size),
        )
        train_sampler = maybe_distributed_sampler(train_ds, shuffle=True)
        # Validation runs rank-0 only (simpler than DistributedSampler + all_reduce; sufficient here)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=detection_collate,
            persistent_workers=(num_workers > 0),
            pin_memory=torch.cuda.is_available(),
        )
        _val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=detection_collate,
            persistent_workers=(num_workers > 0),
            pin_memory=torch.cuda.is_available(),
        )

    # Rank-0 owns the MLflow run; other ranks use a no-op context.
    mlflow_ctx = mlflow.start_run() if is_rank0() else contextlib.nullcontext()
    run_id: str | None = None

    with mlflow_ctx as run:
        if is_rank0() and run is not None:
            run_id = run.info.run_id
            mlflow.log_params({
                "backbone": info.model_name,
                "backbone_revision": info.revision or "",
                "summary_dim": info.summary_dim,
                "spatial_dim": info.spatial_dim,
                "patch_size": info.patch_size,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "use_lora": use_lora,
                "lora_rank": lora_rank if use_lora else 0,
                "lora_alpha": lora_alpha if use_lora else 0,
                "img_size": img_size,
                "trainable_params": n_trainable,
                "device": str(device),
                "world_size": world_size(),
            })

        if train_loader is not None:
            from src.models.detection_head import focal_loss

            steps_per_epoch = max(len(train_loader), 1)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=0.1,
            )

            for epoch in range(epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                model.train()
                epoch_loss = 0.0
                n_batches = 0
                for images, targets in train_loader:
                    images = images.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        cls_logits, _box_reg, _anchors = unwrap_model(model).forward_train(images)
                        cls_t, _ = _build_targets(
                            targets, num_classes=4, total_anchors=cls_logits.shape[1], device=device,
                        )
                        loss = focal_loss(cls_logits, cls_t)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    epoch_loss += float(loss.item())
                    n_batches += 1

                # Reduce per-epoch loss across ranks so logged metric reflects the global mean.
                if is_distributed():
                    t = torch.tensor([epoch_loss, float(n_batches)], device=device, dtype=torch.float64)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    global_loss = (t[0] / max(t[1].item(), 1.0)).item()
                else:
                    global_loss = epoch_loss / max(n_batches, 1)

                if is_rank0():
                    mlflow.log_metric("train/loss", global_loss, step=epoch)
                    logger.info("Epoch %d train loss=%.4f (global mean)", epoch, global_loss)

        # Synchronize before save so rank-0 has a consistent view of the model state.
        barrier()

        if is_rank0():
            # Save state + configs to a tempdir for MLflow logging.
            with tempfile.TemporaryDirectory() as td:
                state_path = Path(td) / "model_state.pt"
                # unwrap_model strips the DDP "module." prefix so the saved state_dict
                # loads cleanly into the bare DetectionModel at serving time.
                torch.save(unwrap_model(model).state_dict(), state_path)

                backbone_cfg = Path(td) / "backbone_config.json"
                backbone_cfg.write_text(json.dumps({
                    "name": backbone_name, "revision": backbone_revision,
                    "summary_dim": info.summary_dim, "spatial_dim": info.spatial_dim,
                    "patch_size": info.patch_size,
                }))

                detection_cfg = Path(td) / "detection_config.json"
                detection_cfg.write_text(json.dumps({
                    "num_classes": 4,
                    "scales": DEFAULT_ANCHOR_SCALES,
                    "aspect_ratios": DEFAULT_ASPECT_RATIOS,
                    "score_threshold": 0.05,
                    "nms_iou_threshold": 0.5,
                    "max_detections": 100,
                    "input_size": img_size,
                }))

                label_map_path = Path(td) / "label_map.json"
                label_map_path.write_text(json.dumps({str(k): v for k, v in get_label_map().items()}))

                artifacts = {
                    "model_state": str(state_path),
                    "backbone_config": str(backbone_cfg),
                    "detection_config": str(detection_cfg),
                    "label_map": str(label_map_path),
                }
                if cache_dir is not None:
                    artifacts["model_cache"] = cache_dir

                signature, example = build_signature_and_example()

                full_model = f"{catalog}.{schema}.{model_name}"
                log_kwargs = {
                    "python_model": DetectorPyfunc(),
                    "artifacts": artifacts,
                    "signature": signature,
                    "input_example": example,
                    "pip_requirements": [
                        "torch", "torchvision", "transformers", "mlflow",
                        "Pillow", "numpy", "pandas",
                    ],
                }
                if register_model:
                    log_kwargs["registered_model_name"] = full_model

                try:
                    mlflow.pyfunc.log_model("model", **log_kwargs)
                except TypeError:
                    # Older MLflow API uses `artifact_path` kwarg
                    mlflow.pyfunc.log_model(artifact_path="model", **log_kwargs)

            if register_model and set_candidate_alias:
                try:
                    client = MlflowClient(registry_uri="databricks-uc")
                    versions = client.search_model_versions(f"name='{full_model}'")
                    versions_for_run = [v for v in versions if v.run_id == run_id]
                    if versions_for_run:
                        latest = max(versions_for_run, key=lambda v: int(v.version))
                        client.set_registered_model_alias(
                            name=full_model, alias="candidate", version=latest.version,
                        )
                        mlflow.log_param("registered_version", latest.version)
                        logger.info(
                            "Set @candidate alias on %s version %s", full_model, latest.version,
                        )
                except Exception as e:
                    logger.error("Failed to set @candidate alias: %s", e)

    teardown_distributed()
    return run_id
