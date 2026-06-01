"""Tests for `dais26_dentex.platform.uc` — UCName + VolumePath."""

from __future__ import annotations

import pytest

from dais26_dentex.config.constants import ALIAS_CANDIDATE, ALIAS_CHAMPION
from dais26_dentex.platform.uc import UCName, VolumePath

# ---------- UCName ----------


def test_ucname_fqn() -> None:
    name = UCName("ml_dev", "dais26_vfm", "cradio_detector")
    assert name.fqn == "ml_dev.dais26_vfm.cradio_detector"
    assert str(name) == "ml_dev.dais26_vfm.cradio_detector"


def test_ucname_is_frozen() -> None:
    name = UCName("a", "b", "c")
    # `FrozenInstanceError` subclasses `AttributeError`, so this covers both.
    with pytest.raises(AttributeError):
        name.catalog = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        "with space",
        "dotted.id",
        "1leadingdigit",
        "",
        "with/slash",
        "with;semi",
    ],
)
def test_ucname_rejects_bad_idents(bad: str) -> None:
    with pytest.raises(ValueError, match="Invalid UC"):
        UCName(bad, "schema", "name")
    with pytest.raises(ValueError, match="Invalid UC"):
        UCName("catalog", bad, "name")
    with pytest.raises(ValueError, match="Invalid UC"):
        UCName("catalog", "schema", bad)


@pytest.mark.parametrize("ok", ["abc", "abc_def", "abc-def", "_priv", "a1", "A1B-C_2"])
def test_ucname_accepts_good_idents(ok: str) -> None:
    UCName(ok, ok, ok)  # no raise


def test_ucname_equality_and_hash() -> None:
    a = UCName("c", "s", "m")
    b = UCName("c", "s", "m")
    assert a == b
    assert hash(a) == hash(b)
    # Hashable → usable as dict key
    assert {a: 1}[b] == 1


# ---------- VolumePath ----------


def test_volumepath_root() -> None:
    vp = VolumePath("main", "mshtelma", "dentex_raw")
    assert vp.root == "/Volumes/main/mshtelma/dentex_raw"
    assert vp.path == "/Volumes/main/mshtelma/dentex_raw"
    assert str(vp) == "/Volumes/main/mshtelma/dentex_raw"


def test_volumepath_with_subpath() -> None:
    vp = VolumePath("main", "mshtelma", "dentex_raw", "extracted/train")
    assert vp.path == "/Volumes/main/mshtelma/dentex_raw/extracted/train"


def test_volumepath_strips_leading_slash_in_subpath() -> None:
    vp = VolumePath("main", "mshtelma", "dentex_raw", "/extracted/train")
    # The literal subpath survives, but `path` strips the leading slash so we
    # never produce `/Volumes/.../volume//extracted/...`.
    assert vp.path == "/Volumes/main/mshtelma/dentex_raw/extracted/train"


def test_volumepath_child_appends() -> None:
    vp = VolumePath("main", "mshtelma", "dentex_raw")
    sub = vp.child("extracted", "train")
    assert sub.path == "/Volumes/main/mshtelma/dentex_raw/extracted/train"
    # Original is unchanged (frozen).
    assert vp.subpath == ""


def test_volumepath_child_chains() -> None:
    vp = VolumePath("c", "s", "v").child("a").child("b", "c")
    assert vp.path == "/Volumes/c/s/v/a/b/c"


def test_volumepath_child_drops_empty_and_slashes() -> None:
    vp = VolumePath("c", "s", "v").child("/a/", "", "/b")
    assert vp.path == "/Volumes/c/s/v/a/b"


def test_volumepath_validates_root_idents() -> None:
    with pytest.raises(ValueError, match="Invalid UC"):
        VolumePath("bad name", "s", "v")
    with pytest.raises(ValueError, match="Invalid UC"):
        VolumePath("c", "s", "with/slash")


def test_volumepath_subpath_is_not_validated_as_uc_ident() -> None:
    """UC accepts arbitrary file names inside a volume — periods, dashes,
    file extensions are all fine."""
    vp = VolumePath("c", "s", "v", "model.pt")
    assert vp.path.endswith("/model.pt")


# ---------- aliases (sanity / single-source-of-truth assertion) ----------


def test_alias_constants_are_stable() -> None:
    """Loose strings used to live inline (`"candidate"`, `"champion"`); the
    refactor consolidates them. If a teammate accidentally edits the values
    in `config.constants`, this test fails — and the registry-aliasing
    contract is exactly the kind of invariant we want a green test to
    guard."""
    assert ALIAS_CANDIDATE == "candidate"
    assert ALIAS_CHAMPION == "champion"
