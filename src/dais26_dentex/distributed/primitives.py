"""Distributed-training primitives for AIR / serverless GPU / torchrun.

Single source of truth shared by:
  * the `serverless_gpu.@distributed`-wrapped notebook entrypoint
  * the `sgcli`/`torchrun -m dais26_dentex.train.cli` entrypoint

Safe to call when WORLD_SIZE=1 (everything degrades to a no-op).
"""

from __future__ import annotations

import logging
import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return world_size() > 1


def is_rank0() -> bool:
    return global_rank() == 0


def setup_distributed(timeout_minutes: int = 30) -> torch.device:
    """Initialize the process group when WORLD_SIZE > 1 and not yet initialized.

    Returns the torch.device for this rank (cuda:{local_rank} or cpu).
    Idempotent: callable from both the @distributed notebook path (where
    `serverless_gpu` may already have set env vars) and the torchrun path.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
        device = torch.device(f"cuda:{local_rank()}")
    else:
        device = torch.device("cpu")

    if is_distributed() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=timeout_minutes),
        )
        logger.info(
            "Initialized %s process group: rank=%d/%d local_rank=%d device=%s",
            backend,
            global_rank(),
            world_size(),
            local_rank(),
            device,
        )
    return device


def teardown_distributed() -> None:
    """Barrier + destroy process group when initialized. No-op otherwise."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    """Cross-rank barrier when distributed. No-op otherwise."""
    if dist.is_initialized():
        dist.barrier()


class BarrierTimeoutError(RuntimeError):
    """Raised when `safe_barrier` did not complete within its deadline.

    Typical cause: a peer rank crashed before the pre-save sync, leaving the
    survivors blocked in `dist.barrier()`. Surfacing this as a typed error
    means the job exits with a clear message instead of hanging until the
    NCCL job-level timeout (the legacy failure mode).
    """


def safe_barrier(timeout_seconds: float = 600.0) -> None:
    """Pre-save sync that surfaces dead-rank deadlocks as typed errors.

    Implemented via async barrier + bounded `wait()`: when a peer dies, the
    work-handle's wait raises (NCCL `RuntimeError` or a generic timeout),
    and we re-raise as `BarrierTimeoutError` with context. Single-rank /
    not-initialized: no-op (matches `barrier()` semantics).

    Replaces the unconditional `dist.barrier()` previously used pre-save,
    which would hang the surviving ranks indefinitely if rank N crashed
    mid-epoch.
    """
    if not dist.is_initialized():
        return
    try:
        work = dist.barrier(async_op=True)
        if work is None:
            # Some backends return None even with async_op=True (gloo edge
            # cases). Fall back to the synchronous barrier — best we can do.
            dist.barrier()
            return
        work.wait(timeout=timedelta(seconds=timeout_seconds))
    except (RuntimeError, TimeoutError) as e:
        raise BarrierTimeoutError(
            f"safe_barrier did not complete within {timeout_seconds:.0f}s; "
            f"a peer rank likely crashed before the pre-save sync. "
            f"Underlying error: {e}"
        ) from e


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip a DistributedDataParallel / DataParallel wrapper if present."""
    return getattr(model, "module", model)


def maybe_distributed_sampler(dataset, shuffle: bool):
    """Return a DistributedSampler when distributed; else None (DataLoader uses native shuffle)."""
    from torch.utils.data import DistributedSampler

    if is_distributed():
        return DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
    return None


def seed_per_rank(base_seed: int = 42) -> int:
    """Deterministic seed that diverges per rank (different augmentations across replicas)."""
    seed = base_seed + global_rank()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
