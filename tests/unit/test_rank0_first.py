"""Tests for `dais26_dentex.distributed.rank0_first`.

Single-process tests only — covers the non-distributed degraded path and
the call-ordering invariants when `is_distributed()` / `is_rank0()` are
stubbed. Multi-process barrier semantics are exercised in
`test_ddp_smoke.py`.

Note: the function `rank0_first` lives in the submodule
`dais26_dentex.distributed.barrier_dance`. The package `__init__.py`
re-exports it at top-level (`from dais26_dentex.distributed import
rank0_first`), but we patch the submodule directly so name resolution
is unambiguous.
"""

from __future__ import annotations

import pytest

import dais26_dentex.distributed.barrier_dance as _r0f_mod
from dais26_dentex.distributed import rank0_first


def test_no_op_in_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """When `is_distributed()` is False, the body runs and no barrier is
    issued. This is the local-dev / single-rank path."""
    barrier_calls: list[int] = []

    monkeypatch.setattr(_r0f_mod, "is_distributed", lambda: False)
    monkeypatch.setattr(_r0f_mod, "barrier", lambda: barrier_calls.append(1))

    ran = False
    with rank0_first():
        ran = True

    assert ran is True
    assert barrier_calls == []


def test_rank0_path_runs_body_then_barriers(monkeypatch: pytest.MonkeyPatch) -> None:
    """On rank 0: yield, then issue barriers (the side effect is in-flight
    inside the `with`, the barriers are released afterwards so non-rank-0
    can proceed)."""
    events: list[str] = []

    monkeypatch.setattr(_r0f_mod, "is_distributed", lambda: True)
    monkeypatch.setattr(_r0f_mod, "is_rank0", lambda: True)
    monkeypatch.setattr(_r0f_mod, "barrier", lambda: events.append("barrier"))

    with rank0_first():
        events.append("body")

    assert events[0] == "body"
    assert all(e == "barrier" for e in events[1:])
    assert events.count("barrier") >= 1


def test_non_rank0_waits_then_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """On non-rank-0: barrier (wait for rank 0), then yield, then trailing
    barrier."""
    events: list[str] = []

    monkeypatch.setattr(_r0f_mod, "is_distributed", lambda: True)
    monkeypatch.setattr(_r0f_mod, "is_rank0", lambda: False)
    monkeypatch.setattr(_r0f_mod, "barrier", lambda: events.append("barrier"))

    with rank0_first():
        events.append("body")

    assert events[0] == "barrier", "non-rank-0 must wait before entering body"
    assert "body" in events
    assert events[-1] == "barrier", "trailing barrier closes the dance"


def test_body_exception_still_releases_barrier(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the body raises on rank 0, the trailing barriers still fire — so
    other ranks aren't deadlocked. The exception itself is re-raised."""
    events: list[str] = []

    monkeypatch.setattr(_r0f_mod, "is_distributed", lambda: True)
    monkeypatch.setattr(_r0f_mod, "is_rank0", lambda: True)
    monkeypatch.setattr(_r0f_mod, "barrier", lambda: events.append("barrier"))

    class _BoomError(RuntimeError):
        pass

    with pytest.raises(_BoomError), rank0_first():
        raise _BoomError("body died")

    # Exception path still released the barriers.
    assert events.count("barrier") >= 1


def test_body_exception_on_non_rank0_still_trailing_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric exception-safety on the non-rank-0 path."""
    events: list[str] = []

    monkeypatch.setattr(_r0f_mod, "is_distributed", lambda: True)
    monkeypatch.setattr(_r0f_mod, "is_rank0", lambda: False)
    monkeypatch.setattr(_r0f_mod, "barrier", lambda: events.append("barrier"))

    class _BoomError(RuntimeError):
        pass

    with pytest.raises(_BoomError), rank0_first():
        events.append("body-pre-raise")
        raise _BoomError("non-rank-0 body died")

    # Should have: leading barrier, body-pre-raise, trailing barrier.
    assert events[0] == "barrier"
    assert "body-pre-raise" in events
    assert events[-1] == "barrier"
