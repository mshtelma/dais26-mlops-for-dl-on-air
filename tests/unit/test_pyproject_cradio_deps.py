"""Cross-check: pyproject.toml declares the C-RADIOv4 trust_remote_code deps.

If a contributor removes timm/einops/open_clip_torch from pyproject without
updating backbones._assert_cradio_runtime_deps, runtime breaks silently on
AIR (and serving endpoints fail on first inference because pip_requirements
in train_detector.py also references these). This test catches the drift.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _dep_names() -> set[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    return {d.split(">=")[0].split("==")[0].split("<")[0].split(";")[0].strip().lower() for d in deps}


def test_pyproject_declares_cradio_trust_remote_code_deps():
    """timm, einops, open_clip_torch must appear in [project] dependencies.
    See src/dais26_dentex/models/backbones.py::_assert_cradio_runtime_deps."""
    dep_names = _dep_names()
    for required in ("timm", "einops", "open_clip_torch"):
        assert required in dep_names, (
            f"{required!r} missing from pyproject.toml [project] dependencies — "
            f"required by nvidia/C-RADIOv4-SO400M trust_remote_code. "
            f"See backbones.py::_assert_cradio_runtime_deps and "
            f"train_detector.py pip_requirements."
        )
