"""Distributed-training primitives.

Re-exports from `primitives` so callers write `from dais26_dentex.distributed
import is_rank0, rank0_first`.
"""

from dais26_dentex.distributed.barrier_dance import rank0_first
from dais26_dentex.distributed.primitives import (
    BarrierTimeoutError,
    barrier,
    global_rank,
    is_distributed,
    is_rank0,
    local_rank,
    maybe_distributed_sampler,
    safe_barrier,
    seed_per_rank,
    setup_distributed,
    teardown_distributed,
    unwrap_model,
    world_size,
)

__all__ = [
    "BarrierTimeoutError",
    "barrier",
    "global_rank",
    "is_distributed",
    "is_rank0",
    "local_rank",
    "maybe_distributed_sampler",
    "rank0_first",
    "safe_barrier",
    "seed_per_rank",
    "setup_distributed",
    "teardown_distributed",
    "unwrap_model",
    "world_size",
]
