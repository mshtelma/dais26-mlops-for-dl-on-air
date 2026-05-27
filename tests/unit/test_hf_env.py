"""Tests for `dais26_dentex.platform.hf_env.configure_hf_env`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dais26_dentex.config.constants import (
    HF_ENV_DOWNLOAD_TIMEOUT,
    HF_ENV_ENABLE_HF_TRANSFER,
    HF_ENV_HOME,
    HF_ENV_TRANSFORMERS_CACHE,
)
from dais26_dentex.platform.hf_env import configure_hf_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the HF env vars before each test so we observe what configure_hf_env sets."""
    for var in (HF_ENV_HOME, HF_ENV_TRANSFORMERS_CACHE, HF_ENV_ENABLE_HF_TRANSFER, HF_ENV_DOWNLOAD_TIMEOUT):
        monkeypatch.delenv(var, raising=False)


def test_sets_cache_dirs_from_string() -> None:
    configure_hf_env("/tmp/hf-cache")
    assert os.environ[HF_ENV_HOME] == "/tmp/hf-cache"
    assert os.environ[HF_ENV_TRANSFORMERS_CACHE] == "/tmp/hf-cache"


def test_sets_cache_dirs_from_pathlike() -> None:
    configure_hf_env(Path("/tmp/hf-cache"))
    assert os.environ[HF_ENV_HOME] == "/tmp/hf-cache"
    assert os.environ[HF_ENV_TRANSFORMERS_CACHE] == "/tmp/hf-cache"


def test_none_cache_dir_leaves_existing_values_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HF_ENV_HOME, "/preset/home")
    monkeypatch.setenv(HF_ENV_TRANSFORMERS_CACHE, "/preset/cache")
    configure_hf_env(None)
    assert os.environ[HF_ENV_HOME] == "/preset/home"
    assert os.environ[HF_ENV_TRANSFORMERS_CACHE] == "/preset/cache"


def test_disables_hf_transfer_by_default() -> None:
    configure_hf_env("/tmp/x")
    # Critical: UC Volume FUSE rejects the parallel chunked downloader.
    assert os.environ[HF_ENV_ENABLE_HF_TRANSFER] == "0"


def test_can_opt_into_hf_transfer() -> None:
    configure_hf_env("/tmp/x", allow_transfer=True)
    # When the caller knows the cache dir is NOT FUSE-backed, we don't force "0".
    assert HF_ENV_ENABLE_HF_TRANSFER not in os.environ


def test_download_timeout_set_when_unset() -> None:
    configure_hf_env("/tmp/x", download_timeout=600)
    assert os.environ[HF_ENV_DOWNLOAD_TIMEOUT] == "600"


def test_download_timeout_does_not_clobber_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HF_ENV_DOWNLOAD_TIMEOUT, "1200")
    configure_hf_env("/tmp/x", download_timeout=600)
    # Existing user override wins.
    assert os.environ[HF_ENV_DOWNLOAD_TIMEOUT] == "1200"


def test_idempotent_repeated_calls() -> None:
    configure_hf_env("/tmp/x")
    configure_hf_env("/tmp/x")
    assert os.environ[HF_ENV_HOME] == "/tmp/x"
