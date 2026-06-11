"""Tests for dais26_dentex.distributed primitives — env-var detection + unwrap_model."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from dais26_dentex import distributed as du


def test_world_size_default():
    with patch.dict(os.environ, {}, clear=True):
        assert du.world_size() == 1


def test_world_size_from_env():
    with patch.dict(os.environ, {"WORLD_SIZE": "8"}, clear=True):
        assert du.world_size() == 8


def test_global_rank_default():
    with patch.dict(os.environ, {}, clear=True):
        assert du.global_rank() == 0


def test_global_rank_from_env():
    with patch.dict(os.environ, {"RANK": "3"}, clear=True):
        assert du.global_rank() == 3


def test_local_rank_default():
    with patch.dict(os.environ, {}, clear=True):
        assert du.local_rank() == 0


def test_local_rank_from_env():
    with patch.dict(os.environ, {"LOCAL_RANK": "2"}, clear=True):
        assert du.local_rank() == 2


def test_is_distributed_single():
    with patch.dict(os.environ, {}, clear=True):
        assert du.is_distributed() is False


def test_is_distributed_multi():
    with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=True):
        assert du.is_distributed() is True


def test_is_rank0_true():
    with patch.dict(os.environ, {"RANK": "0"}, clear=True):
        assert du.is_rank0() is True


def test_is_rank0_false():
    with patch.dict(os.environ, {"RANK": "1"}, clear=True):
        assert du.is_rank0() is False


def test_unwrap_model_no_wrapper():
    m = nn.Linear(4, 4)
    assert du.unwrap_model(m) is m


def test_unwrap_model_with_module_attr():
    """unwrap_model returns .module when present (mimics DDP wrapper)."""
    inner = nn.Linear(4, 4)

    class FakeDDP:
        def __init__(self, mod):
            self.module = mod

    wrapped = FakeDDP(inner)
    assert du.unwrap_model(wrapped) is inner


def test_seed_per_rank_is_deterministic_per_rank():
    with patch.dict(os.environ, {"RANK": "0"}, clear=True):
        s0 = du.seed_per_rank(42)
    with patch.dict(os.environ, {"RANK": "1"}, clear=True):
        s1 = du.seed_per_rank(42)
    assert s0 == 42 and s1 == 43


def test_maybe_distributed_sampler_returns_none_when_not_distributed():
    ds = [(torch.zeros(3), 0) for _ in range(10)]
    with patch.dict(os.environ, {}, clear=True):
        assert du.maybe_distributed_sampler(ds, shuffle=True) is None


def test_maybe_distributed_sampler_returns_sampler_when_distributed(monkeypatch):
    """DistributedSampler.__init__ asks dist for world_size/rank — mock those
    so the test doesn't need an actual process group."""
    import torch.distributed as dist

    ds = [(torch.zeros(3), 0) for _ in range(10)]
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}, clear=True):
        sampler = du.maybe_distributed_sampler(ds, shuffle=True)
    from torch.utils.data import DistributedSampler

    assert isinstance(sampler, DistributedSampler)


def test_setup_distributed_returns_cpu_device_without_cuda():
    """Without CUDA + without WORLD_SIZE, returns cpu device, doesn't init NCCL."""
    if torch.cuda.is_available():
        pytest.skip("CUDA available — test for CPU-only path")
    with patch.dict(os.environ, {}, clear=True):
        device = du.setup_distributed()
    assert device.type == "cpu"
    import torch.distributed as dist

    assert not dist.is_initialized()


def test_barrier_no_op_when_not_initialized():
    """barrier() must be a no-op when no process group is initialized."""
    import torch.distributed as dist

    assert not dist.is_initialized()
    du.barrier()  # should not raise


def test_teardown_no_op_when_not_initialized():
    """teardown_distributed() must be a no-op when no process group is initialized."""
    import torch.distributed as dist

    assert not dist.is_initialized()
    du.teardown_distributed()  # should not raise


def test_broadcast_object_identity_when_not_distributed():
    """No process group -> broadcast_object is identity (the notebook-driver
    path of the sweep command loop)."""
    payload = ("trial", 3, {"lr": 1e-4})
    assert du.broadcast_object(payload) is payload
    assert du.broadcast_object(None) is None
