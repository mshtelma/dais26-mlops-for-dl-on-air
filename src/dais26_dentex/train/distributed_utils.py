"""Legacy import path — re-exports from `dais26_dentex.distributed`.

The primitives moved to `dais26_dentex.distributed.primitives` in Phase 1
of the refactor. This module remains so existing call sites (notebooks,
serialized configs, third-party imports) keep working. Will be removed one
release after the new path ships.
"""

from __future__ import annotations

import warnings

from dais26_dentex.distributed.primitives import (
    barrier,
    global_rank,
    is_distributed,
    is_rank0,
    local_rank,
    maybe_distributed_sampler,
    seed_per_rank,
    setup_distributed,
    teardown_distributed,
    unwrap_model,
    world_size,
)

__all__ = [
    "barrier",
    "global_rank",
    "is_distributed",
    "is_rank0",
    "local_rank",
    "maybe_distributed_sampler",
    "seed_per_rank",
    "setup_distributed",
    "teardown_distributed",
    "unwrap_model",
    "world_size",
]

warnings.warn(
    "dais26_dentex.train.distributed_utils is deprecated; import from dais26_dentex.distributed instead.",
    DeprecationWarning,
    stacklevel=2,
)
