"""Tests for `config.environments` — the named-environment resolver shared by
both launch lanes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dais26_dentex.config.environments import (
    DEFAULT_ENV,
    ENVIRONMENTS,
    EnvSpec,
    load_environment,
)

_DAIS26_VARS = (
    "DAIS26_ENV",
    "DAIS26_ENV_FILE",
    "DAIS26_CATALOG",
    "DAIS26_SCHEMA",
    "DAIS26_EXPERIMENT",
    "DAIS26_VOLUME_PATH",
    "DAIS26_CACHE_DIR",
    "DAIS26_CHAMPION_CATALOG",
    "DAIS26_CHAMPION_SCHEMA",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a clean slate, and point the overlay file at a
    guaranteed-missing path so a stray repo-root environments.local.yaml can't
    leak in. Tests that exercise the overlay override DAIS26_ENV_FILE."""
    for var in _DAIS26_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DAIS26_ENV_FILE", "/nonexistent/dais26-no-overlay.yaml")


# --- named environments ---------------------------------------------------


def test_default_env_is_df1_main_mshtelma() -> None:
    env = load_environment()
    assert DEFAULT_ENV == "df1"
    assert env.catalog == "main"
    assert env.schema == "mshtelma"
    # volumes derive from catalog+schema
    assert env.volume_path == "/Volumes/main/mshtelma/dentex_raw"
    assert env.cache_dir == "/Volumes/main/mshtelma/model_cache"
    # champion: same catalog; df1 keeps champion in the same schema (main has no
    # CREATE SCHEMA for a separate mshtelma_prod), as a distinct model.
    assert env.champion_catalog == "main"
    assert env.champion_schema == "mshtelma"
    assert env.experiment_name.endswith("dais26_vfm_experiment")


def test_named_env_prod() -> None:
    env = load_environment("prod")
    assert env.catalog == "mlops_pj"
    assert env.schema == "dais26_vfm"
    assert env.champion_schema == "dais26_vfm_prod"


def test_unknown_env_raises() -> None:
    with pytest.raises(ValueError, match="Unknown environment"):
        load_environment("does_not_exist")


def test_dais26_env_var_selects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAIS26_ENV", "prod")
    assert load_environment().catalog == "mlops_pj"


def test_every_named_env_builds_a_valid_spec() -> None:
    for name in ENVIRONMENTS:
        env = load_environment(name)
        assert env.volume_path.startswith(f"/Volumes/{env.catalog}/{env.schema}/")


# --- as_training_kwargs ----------------------------------------------------


def test_as_training_kwargs_shape() -> None:
    kw = load_environment("df1").as_training_kwargs()
    assert set(kw) == {"catalog", "schema", "volume_path", "cache_dir", "experiment_name"}
    assert kw["catalog"] == "main"


# --- overrides: env vars > overlay > named env ----------------------------


def test_env_var_overrides_named_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAIS26_SCHEMA", "my_sandbox")
    env = load_environment("df1")
    assert env.schema == "my_sandbox"
    # volume path re-derives from the overridden schema
    assert env.volume_path == "/Volumes/main/my_sandbox/dentex_raw"


def test_overlay_file_overrides_named_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    overlay = tmp_path / "environments.local.yaml"
    overlay.write_text(yaml.safe_dump({"schema": "alice_dev", "experiment_name": "/Users/alice/exp"}))
    monkeypatch.setenv("DAIS26_ENV_FILE", str(overlay))
    env = load_environment("df1")
    assert env.schema == "alice_dev"
    assert env.experiment_name == "/Users/alice/exp"
    assert env.catalog == "main"  # untouched keys fall through to the named env


def test_env_var_beats_overlay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    overlay = tmp_path / "environments.local.yaml"
    overlay.write_text(yaml.safe_dump({"schema": "from_overlay"}))
    monkeypatch.setenv("DAIS26_ENV_FILE", str(overlay))
    monkeypatch.setenv("DAIS26_SCHEMA", "from_env_var")
    assert load_environment("df1").schema == "from_env_var"


def test_explicit_override_wins_and_skips_none() -> None:
    env = load_environment("df1", schema="explicit", experiment_name=None)
    assert env.schema == "explicit"
    # experiment_name=None must NOT clobber the named env's value
    assert env.experiment_name.endswith("dais26_vfm_experiment")


def test_overlay_with_unknown_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    overlay = tmp_path / "environments.local.yaml"
    overlay.write_text(yaml.safe_dump({"catlog": "typo"}))  # misspelled
    monkeypatch.setenv("DAIS26_ENV_FILE", str(overlay))
    with pytest.raises(ValueError, match="Unknown environment field"):
        load_environment("df1")


def test_explicit_volume_path_override_skips_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAIS26_VOLUME_PATH", "/Volumes/other/place/raw")
    assert load_environment("df1").volume_path == "/Volumes/other/place/raw"


# --- EnvSpec validation ----------------------------------------------------


def test_envspec_rejects_dotted_catalog() -> None:
    with pytest.raises(ValueError, match="catalog must be"):
        EnvSpec(catalog="a.b", schema="s")


def test_envspec_explicit_champion_catalog_preserved() -> None:
    spec = EnvSpec(catalog="main", schema="dev", champion_catalog="prod_cat")
    assert spec.champion_catalog == "prod_cat"
    assert spec.champion_schema == "dev_prod"  # still derived
