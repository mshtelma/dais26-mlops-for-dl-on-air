"""Unity Catalog identifiers — one definition, no inline f-strings.

`UCName` and `VolumePath` produce the canonical string forms; alias
constants live in `config.constants`. Anywhere we used to write
`f"{catalog}.{schema}.{model_name}"` should now build a `UCName`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Permissive UC identifier match. UC actually allows quoted identifiers with
# spaces, but the FE workflows in this repo never use them — keep the test
# strict so a typo (e.g. dotted catalog) fails fast.
_UC_IDENT_RE: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def _validate_ident(name: str, role: str) -> str:
    if not _UC_IDENT_RE.fullmatch(name):
        raise ValueError(f"Invalid UC {role} identifier: {name!r}. Expected [A-Za-z_][A-Za-z0-9_-]*.")
    return name


@dataclass(frozen=True, slots=True)
class UCName:
    """Three-level UC name (catalog.schema.name).

    Used for registered models, tables, and volumes. The `fqn` property is
    safe to interpolate into SQL identifiers and into MLflow's
    `registered_model_name` argument.
    """

    catalog: str
    schema: str
    name: str

    def __post_init__(self) -> None:
        _validate_ident(self.catalog, "catalog")
        _validate_ident(self.schema, "schema")
        _validate_ident(self.name, "name")

    @property
    def fqn(self) -> str:
        """`catalog.schema.name` — what UC and MLflow expect everywhere."""
        return f"{self.catalog}.{self.schema}.{self.name}"

    def __str__(self) -> str:
        return self.fqn


@dataclass(frozen=True, slots=True)
class VolumePath:
    """A `/Volumes/<catalog>/<schema>/<volume>` path.

    The trailing path under the volume root is appended via `child(...)` so
    callers don't hand-roll string concatenation. Only the volume root is
    validated; subpaths are stored verbatim (UC accepts arbitrary file
    names inside a volume).
    """

    catalog: str
    schema: str
    volume: str
    subpath: str = ""

    def __post_init__(self) -> None:
        _validate_ident(self.catalog, "catalog")
        _validate_ident(self.schema, "schema")
        _validate_ident(self.volume, "volume")

    @property
    def root(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    @property
    def path(self) -> str:
        if not self.subpath:
            return self.root
        return f"{self.root}/{self.subpath.lstrip('/')}"

    def child(self, *parts: str) -> VolumePath:
        """Return a new `VolumePath` with the given path components joined
        beneath the current `subpath`.
        """
        joined = "/".join(p.strip("/") for p in parts if p)
        new_sub = f"{self.subpath}/{joined}".lstrip("/") if self.subpath else joined
        return VolumePath(self.catalog, self.schema, self.volume, new_sub)

    def __str__(self) -> str:
        return self.path
