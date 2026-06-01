"""DDP smoke test using gloo backend on CPU.

Spawns 2 worker processes via torch.multiprocessing and verifies:
  - setup_distributed initializes the group
  - DistributedDataParallel wraps a model with a frozen sub-module without hanging
  - unwrap_model(ddp_model).state_dict() keys do NOT have a "module." prefix
  - teardown_distributed cleans up
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from dais26_dentex.train.distributed_utils import (
    is_distributed,
    setup_distributed,
    teardown_distributed,
    unwrap_model,
)


class TinyHead(nn.Module):
    """Frozen "backbone" + trainable "head" — same shape as the real DetectionModel."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        # Freeze backbone — this is the case that requires find_unused_parameters=True
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


def _worker(rank: int, world_size: int, tmpfile: str):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"

    device = setup_distributed()
    assert is_distributed()
    assert dist.is_initialized()

    model = TinyHead().to(device)
    ddp = nn.parallel.DistributedDataParallel(
        model,
        device_ids=None,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    x = torch.randn(8, 4, device=device)
    y = torch.randint(0, 2, (8,), device=device)
    opt = torch.optim.SGD([p for p in ddp.parameters() if p.requires_grad], lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    # One backward — this is the step that deadlocks without find_unused_parameters=True
    out = ddp(x)
    loss = loss_fn(out, y)
    loss.backward()
    opt.step()

    # Verify unwrap_model strips the DDP "module." prefix
    raw = unwrap_model(ddp)
    keys = list(raw.state_dict().keys())
    assert all(not k.startswith("module.") for k in keys), keys

    if rank == 0:
        # Write a sentinel so the parent can confirm success
        with open(tmpfile, "w") as f:
            f.write("OK")

    teardown_distributed()


@pytest.mark.skipif(sys.platform == "win32", reason="mp.spawn flaky on win32")
def test_ddp_two_rank_gloo_smoke(tmp_path):
    """Two-rank DDP forward+backward+step on CPU using gloo. Validates the
    frozen-backbone + find_unused_parameters=True pattern doesn't deadlock."""
    sentinel = tmp_path / "ok.txt"
    mp.spawn(_worker, args=(2, str(sentinel)), nprocs=2, join=True)
    assert sentinel.exists()
    assert sentinel.read_text() == "OK"
