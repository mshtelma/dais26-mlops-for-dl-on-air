"""Tests for `safe_barrier` — the pre-save sync that prevents the legacy
hang-until-NCCL-timeout failure mode when a peer rank dies mid-epoch.

We can't spin up a real process group in unit tests, so the strategy is:
mock `torch.distributed` at the module surface and assert `safe_barrier`
threads `timeout_seconds` through to `Work.wait` and rebadges the
underlying error class as `BarrierTimeoutError`.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from dais26_dentex.distributed.primitives import (
    BarrierTimeoutError,
    safe_barrier,
)

# ----------------------------------------------------------------------
# Single-rank / not-initialized
# ----------------------------------------------------------------------


def test_safe_barrier_noop_when_not_initialized() -> None:
    """Single-rank / pre-init: no `dist.barrier` call, no error."""
    with patch("dais26_dentex.distributed.primitives.dist") as mock_dist:
        mock_dist.is_initialized.return_value = False

        safe_barrier(timeout_seconds=10.0)

        mock_dist.barrier.assert_not_called()


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_safe_barrier_happy_path_threads_timeout() -> None:
    """Distributed mode: async barrier issued, `wait` called with the
    requested deadline as a `timedelta`."""
    with patch("dais26_dentex.distributed.primitives.dist") as mock_dist:
        mock_dist.is_initialized.return_value = True
        work = MagicMock()
        mock_dist.barrier.return_value = work

        safe_barrier(timeout_seconds=42.5)

        mock_dist.barrier.assert_called_once_with(async_op=True)
        work.wait.assert_called_once()
        kwargs = work.wait.call_args.kwargs
        assert isinstance(kwargs["timeout"], timedelta)
        assert kwargs["timeout"].total_seconds() == pytest.approx(42.5)


# ----------------------------------------------------------------------
# Backend returned None (gloo edge case): fall back to sync barrier.
# ----------------------------------------------------------------------


def test_safe_barrier_falls_back_to_sync_when_work_is_none() -> None:
    """Some backends return None from `barrier(async_op=True)`; we issue a
    plain blocking `dist.barrier()` so the sync still happens."""
    with patch("dais26_dentex.distributed.primitives.dist") as mock_dist:
        mock_dist.is_initialized.return_value = True
        # First call: async — returns None.
        # Second call: the fallback sync barrier.
        mock_dist.barrier.side_effect = [None, None]

        safe_barrier(timeout_seconds=10.0)

        assert mock_dist.barrier.call_count == 2
        # Second call has no kwargs (synchronous).
        assert mock_dist.barrier.call_args_list[1].kwargs == {}


# ----------------------------------------------------------------------
# Failure modes — both `RuntimeError` (NCCL) and `TimeoutError` rebadged.
# ----------------------------------------------------------------------


def test_safe_barrier_runtime_error_becomes_barrier_timeout_error() -> None:
    """NCCL surfaces a peer-rank crash via RuntimeError on `wait`; we
    rebadge as the typed error so callers can `except BarrierTimeoutError`."""
    with patch("dais26_dentex.distributed.primitives.dist") as mock_dist:
        mock_dist.is_initialized.return_value = True
        work = MagicMock()
        work.wait.side_effect = RuntimeError("NCCL: rank 1 disconnected")
        mock_dist.barrier.return_value = work

        with pytest.raises(BarrierTimeoutError) as exc_info:
            safe_barrier(timeout_seconds=5.0)

        msg = str(exc_info.value)
        assert "5" in msg
        assert "peer rank" in msg.lower()
        # Underlying is preserved on `__cause__`.
        assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_safe_barrier_timeout_error_becomes_barrier_timeout_error() -> None:
    """Generic `TimeoutError` from `wait()` is also rebadged."""
    with patch("dais26_dentex.distributed.primitives.dist") as mock_dist:
        mock_dist.is_initialized.return_value = True
        work = MagicMock()
        work.wait.side_effect = TimeoutError("deadline exceeded")
        mock_dist.barrier.return_value = work

        with pytest.raises(BarrierTimeoutError):
            safe_barrier(timeout_seconds=1.0)


# ----------------------------------------------------------------------
# Symbol surface
# ----------------------------------------------------------------------


def test_barrier_timeout_error_is_runtime_error_subclass() -> None:
    """Callers should be able to `except RuntimeError` to catch this — keeps
    the broader sweep at the top of `cli.py` working without a code change."""
    assert issubclass(BarrierTimeoutError, RuntimeError)


def test_safe_barrier_exported_from_distributed_package() -> None:
    """The package-level re-export is the public API — pin it."""
    from dais26_dentex import distributed as d

    assert d.safe_barrier is safe_barrier
    assert d.BarrierTimeoutError is BarrierTimeoutError
