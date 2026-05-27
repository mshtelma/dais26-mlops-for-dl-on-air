"""Smoke tests for the C-RADIOv4 trust_remote_code dep guard.

The guard exists to turn opaque HF deep-stack ImportErrors into a single
actionable line when timm/einops/open_clip are missing. These tests verify:
1. Happy path — no-op when all deps are importable.
2. Missing-dep path — raises ImportError naming the missing module.
"""

import builtins

import pytest


def test_assert_cradio_runtime_deps_passes_when_all_present():
    """Happy path: skip if deps aren't installed in this env (CI dev/test envs
    may not install the full GPU stack). When they ARE present, the guard must
    be a no-op."""
    pytest.importorskip("timm")
    pytest.importorskip("einops")
    pytest.importorskip("open_clip")

    from dais26_dentex.models.backbones import _assert_cradio_runtime_deps

    _assert_cradio_runtime_deps()


def test_assert_cradio_runtime_deps_raises_with_missing_module(monkeypatch):
    """If einops is missing the guard raises ImportError naming it — not a deep
    HF stack. Uses the IMPORT name (einops), not a PyPI dist name."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "einops":
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from dais26_dentex.models.backbones import _assert_cradio_runtime_deps

    with pytest.raises(ImportError, match="einops"):
        _assert_cradio_runtime_deps()


def test_assert_cradio_runtime_deps_uses_import_names_not_pypi_names(monkeypatch):
    """Guard must check 'open_clip' (import name), not 'open_clip_torch' (PyPI).
    Regression guard against F9 in the plan."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        # `open_clip` is the import name; if guard checked `open_clip_torch` it would
        # never raise here because `open_clip_torch` import is never attempted by the guard.
        if name == "open_clip":
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from dais26_dentex.models.backbones import _assert_cradio_runtime_deps

    with pytest.raises(ImportError, match="open_clip"):
        _assert_cradio_runtime_deps()
