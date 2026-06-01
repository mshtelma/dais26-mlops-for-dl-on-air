"""Single canonical HuggingFace env setup.

Replaces the inline `os.environ[...] = ...` blocks scattered across
`models/backbones.py`, `serve/detector_pyfunc.py`, and
`scripts/pin_model_cache.py`. Each call site historically had a slightly
different list of env vars; pin_model_cache was missing
`HF_HUB_ENABLE_HF_TRANSFER=0`, which would FUSE-fail on a cold cache run
because UC Volume FUSE rejects the parallel chunked downloader.

This module is the sole writer of those env vars.
"""

from __future__ import annotations

import logging
import os

from dais26_dentex.config.constants import (
    HF_ENV_DOWNLOAD_TIMEOUT,
    HF_ENV_ENABLE_HF_TRANSFER,
    HF_ENV_HOME,
    HF_ENV_TRANSFORMERS_CACHE,
)

logger = logging.getLogger(__name__)


def configure_hf_env(
    cache_dir: str | os.PathLike[str] | None,
    *,
    allow_transfer: bool = False,
    download_timeout: int | None = 600,
) -> None:
    """Set the HuggingFace environment variables we standardize on.

    Args:
        cache_dir: Path to use for both `HF_HOME` and `TRANSFORMERS_CACHE`.
            When None, the existing env values are left untouched (no clear).
        allow_transfer: When False (default), force `HF_HUB_ENABLE_HF_TRANSFER=0`.
            UC Volume FUSE rejects the parallel chunked downloader with
            `Io: Input/output error (os error 5)` / `Io: Operation not
            supported (os error 95)`. The streaming fallback works on FUSE.
            See docs/RUNBOOK.md#hf-transfer-fuse-incompat.
        download_timeout: Seconds passed to `HF_HUB_DOWNLOAD_TIMEOUT`. None
            leaves whatever was there.

    Idempotent. Safe to call from any rank in a distributed run; the values
    are process-local.
    """
    if cache_dir is not None:
        cache_str = os.fspath(cache_dir)
        os.environ[HF_ENV_HOME] = cache_str
        os.environ[HF_ENV_TRANSFORMERS_CACHE] = cache_str
        logger.debug("HF cache dirs set to %s", cache_str)

    # Only set when we want to enforce; do not clobber a deliberately-set "1".
    if not allow_transfer:
        os.environ[HF_ENV_ENABLE_HF_TRANSFER] = "0"

    if download_timeout is not None:
        os.environ.setdefault(HF_ENV_DOWNLOAD_TIMEOUT, str(download_timeout))
