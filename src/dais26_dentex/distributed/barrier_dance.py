"""`rank0_first` — sequence-matched barrier dance.

Rank 0 performs a cache-warming side effect; other ranks wait, then proceed
against the warm cache. Pattern + failure-mode rationale:
docs/RUNBOOK.md#hf-cache-race.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator

from dais26_dentex.distributed.primitives import barrier, is_distributed, is_rank0


@contextlib.contextmanager
def rank0_first() -> Generator[None, None, None]:
    """Run the body on rank 0 first; other ranks wait for rank 0 to finish.

    Single-rank / non-distributed runs degrade to a plain context manager
    (no `dist.*` calls).

    Usage::

        with rank0_first():
            backbone, info = load_backbone(...)

    Both ranks see `backbone` populated against a warm cache afterwards.
    """
    if not is_distributed():
        yield
        return

    if not is_rank0():
        # Non-rank-0: wait for rank 0 to finish, then proceed.
        barrier()
        try:
            yield
        finally:
            # Symmetric trailing barrier so all ranks exit together; matches
            # the `finally` barrier on rank 0 below.
            barrier()
        return

    # Rank 0: do the work, then release the others.
    try:
        yield
    finally:
        barrier()
        # Trailing barrier so the rank-0 process exits the context at the
        # same point as the others.
        barrier()
