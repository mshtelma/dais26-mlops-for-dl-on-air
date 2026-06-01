"""MLflow producer-side utilities — single home for `log_model`,
`set_alias`, and the `pip_requirements` source-of-truth read.

What this replaces:
  * The `try/except TypeError` for the `name=` vs `artifact_path=` MLflow
    API drift — detected once at import.
  * The hardcoded ``pip_requirements`` list in ``trainer.py`` — sourced
    from ``pyproject.toml [tool.dais26.serving-deps].detector`` so adding
    a runtime dep is a one-place edit.
  * The bare ``except Exception`` around alias-setting — surfaces a typed
    ``AliasingError`` instead.

Rank-awareness lives at the call site (``Trainer`` already gates on
``is_rank0()``); this module is rank-agnostic.
"""

from __future__ import annotations

import inspect
import logging
import sys
import tomllib
from functools import cache
from pathlib import Path
from typing import Any, Final

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from dais26_dentex.config.constants import ALIAS_CANDIDATE

logger = logging.getLogger(__name__)


class AliasingError(RuntimeError):
    """Raised when registry alias assignment fails after a successful log.

    Surfacing this means downstream gates (smoke test → ``@champion``
    promotion) can't silently operate on an un-aliased version.
    """


# ----------------------------------------------------------------------
# pip_requirements source-of-truth
# ----------------------------------------------------------------------

_PYPROJECT_NAME: Final[str] = "pyproject.toml"
_SERVING_TABLE: Final[str] = "tool.dais26.serving-deps"


def _find_pyproject() -> Path:
    """Locate ``pyproject.toml`` — packaged resource first, then source-tree walk.

    In a wheel install (AIR / ephemeral envs) the package's ancestors do
    not contain ``pyproject.toml``, so the historical walk-up fails. The
    wheel ships ``pyproject.toml`` as ``dais26_dentex/_pyproject.toml``
    via hatchling ``force-include`` (see pyproject.toml), and we read it
    via ``importlib.resources``. The walk-up branch stays for editable
    installs and pytest runs against the source tree.
    """
    try:
        from importlib.resources import files

        ref = files("dais26_dentex") / "_pyproject.toml"
        if ref.is_file():
            return Path(str(ref))
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        candidate = parent / _PYPROJECT_NAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {_PYPROJECT_NAME} from {here} and "
        f"dais26_dentex/_pyproject.toml is not packaged; serving-deps "
        f"table cannot be resolved."
    )


@cache
def serving_pip_requirements(profile: str = "detector") -> list[str]:
    """Read ``[tool.dais26.serving-deps].<profile>`` from pyproject.toml.

    Cached because pyproject.toml is read-only at runtime and the same
    list is requested multiple times by tests + the trainer.
    """
    path = _find_pyproject()
    data = tomllib.loads(path.read_text())
    table = (data.get("tool", {}) or {}).get("dais26", {}).get("serving-deps", {})
    if profile not in table:
        raise KeyError(
            f"[{_SERVING_TABLE}] does not define profile '{profile}' in {path}; available: {sorted(table.keys())}"
        )
    deps = table[profile]
    if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
        raise TypeError(f"[{_SERVING_TABLE}.{profile}] must be a list[str]; got {type(deps).__name__}")
    return list(deps)


# ----------------------------------------------------------------------
# log_model API drift detection
# ----------------------------------------------------------------------


@cache
def _log_model_artifact_kwarg() -> str:
    """Return ``"name"`` or ``"artifact_path"`` based on the installed MLflow.

    MLflow renamed the positional argument across minor versions. We pick
    once at import time instead of paying a try/except per call, and we
    do it via signature inspection so the result is correct regardless of
    install order. Falls back to ``"name"`` when inspection fails.
    """
    try:
        sig = inspect.signature(mlflow.pyfunc.log_model)
        if "name" in sig.parameters:
            return "name"
        if "artifact_path" in sig.parameters:
            return "artifact_path"
    except (TypeError, ValueError):
        pass
    return "name"


# ----------------------------------------------------------------------
# Reporter
# ----------------------------------------------------------------------


class MlflowReporter:
    """Centralizes MLflow producer-side calls used by the trainer.

    Single instance per run. Caller decides on rank-awareness; this class
    does no rank gating itself so it stays unit-testable in-process.
    """

    REGISTRY_URI: Final[str] = "databricks-uc"

    def __init__(self, *, experiment_name: str | None = None, registry_uri: str | None = None) -> None:
        self.experiment_name = experiment_name
        self.registry_uri = registry_uri or self.REGISTRY_URI

    def configure(self) -> None:
        """Set the registry URI + experiment. Idempotent."""
        mlflow.set_registry_uri(self.registry_uri)
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

    def log_pyfunc(
        self,
        *,
        python_model: Any,
        artifacts: dict[str, str],
        signature: Any,
        input_example: Any,
        registered_model_name: str | None = None,
        pip_requirements: list[str] | None = None,
        artifact_path: str = "model",
    ) -> Any:
        """Wrap ``mlflow.pyfunc.log_model`` with a single call signature.

        ``pip_requirements`` defaults to the ``detector`` profile from
        ``[tool.dais26.serving-deps]``.
        """
        kwargs: dict[str, Any] = {
            _log_model_artifact_kwarg(): artifact_path,
            "python_model": python_model,
            "artifacts": artifacts,
            "signature": signature,
            "input_example": input_example,
            "pip_requirements": pip_requirements or serving_pip_requirements(),
        }
        if registered_model_name is not None:
            kwargs["registered_model_name"] = registered_model_name
        return mlflow.pyfunc.log_model(**kwargs)

    def set_candidate_alias(
        self,
        *,
        full_model: str,
        run_id: str,
        alias: str = ALIAS_CANDIDATE,
    ) -> str:
        """Set ``@<alias>`` on the version registered for ``run_id``.

        Returns the version string. Raises ``AliasingError`` instead of
        swallowing — the legacy bare-except hid silent gate failures.
        """
        try:
            client = MlflowClient(registry_uri=self.registry_uri)
            versions = client.search_model_versions(f"name='{full_model}'")
            for_run = [v for v in versions if v.run_id == run_id]
            if not for_run:
                raise AliasingError(f"No registered version found for run_id={run_id} on {full_model}")
            latest = max(for_run, key=lambda v: int(v.version))
            client.set_registered_model_alias(
                name=full_model,
                alias=alias,
                version=latest.version,
            )
            logger.info("Set @%s alias on %s v%s", alias, full_model, latest.version)
            return str(latest.version)
        except AliasingError:
            raise
        except Exception as e:
            raise AliasingError(f"Failed to set @{alias} alias on {full_model}: {e}") from e


def assert_serving_reqs_match_pyproject(profile: str = "detector") -> None:
    """CI guard: ensure the serving-deps table is syntactically valid + non-empty.

    Doesn't try to verify dep names against pyproject's main dependency
    table (the serving subset is intentionally narrower). Just confirms
    the table parses, the profile exists, and the list is well-typed.
    """
    deps = serving_pip_requirements(profile)
    if not deps:
        raise AssertionError(
            f"[{_SERVING_TABLE}.{profile}] resolved to an empty list; serving will fail to find runtime deps."
        )
    print(
        f"[serving-deps:{profile}] {len(deps)} packages OK ({', '.join(deps)})",
        file=sys.stderr,
    )


__all__ = [
    "AliasingError",
    "MlflowReporter",
    "assert_serving_reqs_match_pyproject",
    "serving_pip_requirements",
]
