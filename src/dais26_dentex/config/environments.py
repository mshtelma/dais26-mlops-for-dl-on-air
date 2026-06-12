"""Named deployment environments — the single source of UC-location truth.

Where `config.recipes` answers "which *hyperparameters*?", this module answers
"which *workspace / catalog / schema / experiment*?" — the second axis of the
DAB-vs-air harmonization. Both launch lanes select an environment **by name**
and resolve the same `EnvSpec`, so a target switch is one token, not five
hand-mirrored keys:

* notebook / DAB lane: `00_config.py` does `env = load_environment(ENV)` and
  reads `env.catalog` / `env.schema` / `env.volume_path` / ...;
* air lane: the workload `parameters:` carry `env: df1` and `train/cli.py`
  (and `sweep_cli.py`) resolve it exactly like they resolve `recipe:`.

Resolution precedence (highest wins):

1. explicit keyword overrides passed to `load_environment(...)`;
2. `DAIS26_*` environment variables (CI / one-offs) — see `_ENV_VAR_MAP`;
3. an optional, per-user `environments.local.yaml` overlay (see below);
4. the committed named environment from `ENVIRONMENTS`.

`volume_path` / `cache_dir` / `champion_*` derive from `catalog` + `schema`, so
a named environment usually only states `catalog`, `schema`, and the MLflow
`experiment_name`. **Secrets never live here** — the HF token still flows
through Databricks secret scopes / the air `secrets:` block.

Per-user override without committing
------------------------------------
Drop an `environments.local.yaml` at the repo root (the loader also honors an
explicit `$DAIS26_ENV_FILE` path and searches up from the CWD to the project
root). It is **deliberately not git-ignored** — air's working-tree snapshot and
the notebooks' `%pip install ..` reinstall both carry it to the remote
pod/cluster, so your local edits reach both lanes with no commit. (A `.env`
name would NOT work: this repo git-ignores `.env*`/`*.local`, and air respects
`.gitignore` — hence the `.yaml` name.) Pinned-commit reproducible runs
intentionally see only the committed `ENVIRONMENTS`, never an uncommitted
overlay.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default volume names; overridable per environment / overlay.
DENTEX_VOLUME = "dentex_raw"
MODEL_CACHE_VOLUME = "model_cache"

# The committed, reviewed targets — the "one obvious place". Each entry needs
# only catalog/schema/experiment_name; everything else derives. Keep these to
# NON-SECRET location values.
ENVIRONMENTS: dict[str, dict[str, Any]] = {
    # The workspace this repo's E2E gates run in (df1 profile). `main` is the
    # general-purpose catalog there; `mshtelma` is the author's namespace.
    "df1": {
        "catalog": "main",
        "schema": "mshtelma",
        "experiment_name": "/Users/michael.shtelma@databricks.com/dais26_vfm_experiment",
        "champion_schema": "mshtelma_prod",
    },
    # The talk's nominal project workspace.
    "prod": {
        "catalog": "mlops_pj",
        "schema": "dais26_vfm",
        "experiment_name": "/Users/michael.shtelma@databricks.com/dais26_vfm_experiment",
        "champion_schema": "dais26_vfm_prod",
    },
}

# Selected when neither an explicit name nor $DAIS26_ENV is given.
DEFAULT_ENV = "df1"

# Per-user file overlay; see the module docstring. Intentionally not in
# .gitignore so it rides air's working-tree snapshot.
OVERLAY_FILENAME = "environments.local.yaml"

# $DAIS26_* env var -> EnvSpec field. The dotenv-style escape hatch that works
# on both lanes (notebook cluster env; `air run --override env_variables.*`).
_ENV_VAR_MAP: dict[str, str] = {
    "DAIS26_CATALOG": "catalog",
    "DAIS26_SCHEMA": "schema",
    "DAIS26_EXPERIMENT": "experiment_name",
    "DAIS26_VOLUME_PATH": "volume_path",
    "DAIS26_CACHE_DIR": "cache_dir",
    "DAIS26_CHAMPION_CATALOG": "champion_catalog",
    "DAIS26_CHAMPION_SCHEMA": "champion_schema",
}


@dataclass(frozen=True)
class EnvSpec:
    """Resolved UC locations for one environment.

    Frozen; `champion_*`, `volume_path`, and `cache_dir` derive from
    `catalog`+`schema` when left as `None`.
    """

    catalog: str
    schema: str
    experiment_name: str | None = None
    champion_catalog: str | None = None
    champion_schema: str | None = None
    dentex_volume: str = DENTEX_VOLUME
    model_cache_volume: str = MODEL_CACHE_VOLUME
    volume_path: str | None = None
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        for fld in ("catalog", "schema"):
            val = getattr(self, fld)
            if not val or "." in val:
                raise ValueError(
                    f"EnvSpec.{fld} must be a non-empty UC identifier without '.', got {val!r}"
                )
        if self.champion_catalog is None:
            object.__setattr__(self, "champion_catalog", self.catalog)
        if self.champion_schema is None:
            object.__setattr__(self, "champion_schema", f"{self.schema}_prod")
        if self.volume_path is None:
            object.__setattr__(
                self, "volume_path", f"/Volumes/{self.catalog}/{self.schema}/{self.dentex_volume}"
            )
        if self.cache_dir is None:
            object.__setattr__(
                self, "cache_dir", f"/Volumes/{self.catalog}/{self.schema}/{self.model_cache_volume}"
            )

    def as_training_kwargs(self) -> dict[str, Any]:
        """The subset both `build_trainer_config` and `TrainerConfig` consume."""
        return {
            "catalog": self.catalog,
            "schema": self.schema,
            "volume_path": self.volume_path,
            "cache_dir": self.cache_dir,
            "experiment_name": self.experiment_name,
        }


def _overlay_path() -> Path | None:
    """Locate the per-user overlay: `$DAIS26_ENV_FILE`, else the first
    `environments.local.yaml` found from the CWD up to the project root."""
    explicit = os.environ.get("DAIS26_ENV_FILE")
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    start = Path.cwd()
    for d in (start, *start.parents):
        cand = d / OVERLAY_FILENAME
        if cand.is_file():
            return cand
        if (d / "pyproject.toml").is_file():
            break  # reached the project root; stop ascending
    return None


def _read_overlay() -> dict[str, Any]:
    p = _overlay_path()
    if p is None:
        return {}
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a YAML mapping; got {type(data).__name__}")
    logger.info("Applied per-user environment overlay from %s", p)
    return data


def _env_var_overrides() -> dict[str, str]:
    return {field: os.environ[var] for var, field in _ENV_VAR_MAP.items() if os.environ.get(var)}


def load_environment(name: str | None = None, **overrides: Any) -> EnvSpec:
    """Resolve a named environment into a validated `EnvSpec`.

    `name` defaults to `$DAIS26_ENV` then `DEFAULT_ENV`. Layers the overlay
    file and `DAIS26_*` env vars on top of the named entry (see module
    docstring for precedence). `overrides` (skipping `None` values) win over
    everything. Raises `ValueError` on an unknown environment or an
    unrecognized field anywhere in the merge.
    """
    name = name or os.environ.get("DAIS26_ENV") or DEFAULT_ENV
    if name not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment {name!r}; known: {sorted(ENVIRONMENTS)}. Add it to "
            "ENVIRONMENTS, or override values via environments.local.yaml / DAIS26_* env vars."
        )
    merged: dict[str, Any] = dict(ENVIRONMENTS[name])
    merged.update(_read_overlay())
    merged.update(_env_var_overrides())
    merged.update({k: v for k, v in overrides.items() if v is not None})

    valid = {f.name for f in fields(EnvSpec)}
    unknown = set(merged) - valid
    if unknown:
        raise ValueError(
            f"Unknown environment field(s) {sorted(unknown)}; valid fields: {sorted(valid)}"
        )
    return EnvSpec(**merged)


__all__ = [
    "DEFAULT_ENV",
    "ENVIRONMENTS",
    "OVERLAY_FILENAME",
    "EnvSpec",
    "load_environment",
]
