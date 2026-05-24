"""sgcli / torchrun entrypoint for `train_detector`.

Reads `$HYPERPARAMETERS_PATH` (a YAML file that sgcli writes from the workload's
`parameters:` block) and dispatches to `train_detector(**filtered_kwargs)`.

Tolerant of unknown YAML keys (logs and skips). Explicit about type coercion.
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
from typing import Any

import yaml

from src.train.distributed_utils import is_rank0
from src.train.train_detector import train_detector

logger = logging.getLogger(__name__)

# HF model IDs / friendly names → internal `load_backbone` literal.
_BACKBONE_ALIASES = {
    "nvidia/C-RADIOv4-SO400M": "cradio_v4_so400m",
    "nvidia/C-RADIOv4-H":      "cradio_v4_so400m",  # not yet wired up; alias to SO400M for now
    "facebook/dinov3-vitl16":  "dinov3_vitl16",
    "facebook/dinov2-base":    "dinov2_base",
}

_INT_KEYS   = {"epochs", "batch_size", "num_workers", "lora_rank", "img_size"}
_FLOAT_KEYS = {"lr", "lora_alpha"}
_BOOL_KEYS  = {"use_lora", "register_model", "set_candidate_alias"}


def _coerce(params: dict[str, Any]) -> dict[str, Any]:
    """Defensive type coercion in case sgcli passes strings (it usually preserves YAML types)."""
    out = dict(params)
    for k in _INT_KEYS:
        if k in out and out[k] is not None:
            out[k] = int(out[k])
    for k in _FLOAT_KEYS:
        if k in out and out[k] is not None:
            out[k] = float(out[k])
    for k in _BOOL_KEYS:
        v = out.get(k)
        if isinstance(v, str):
            out[k] = v.strip().lower() in {"true", "1", "yes"}
    if "backbone_name" in out and out["backbone_name"] in _BACKBONE_ALIASES:
        out["backbone_name"] = _BACKBONE_ALIASES[out["backbone_name"]]
    return out


def load_params() -> dict[str, Any]:
    """Load YAML from $HYPERPARAMETERS_PATH. Returns {} when env var unset.

    Raises SystemExit(2) when the env var is set but the file is missing — fail-fast.
    """
    path = os.environ.get("HYPERPARAMETERS_PATH")
    if not path:
        return {}
    if not os.path.exists(path):
        print(
            f"FATAL: HYPERPARAMETERS_PATH={path} does not exist",
            file=sys.stderr,
        )
        raise SystemExit(2)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def filter_to_known_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    valid = set(inspect.signature(train_detector).parameters.keys())
    unknown = set(params) - valid
    if unknown:
        logger.warning("Ignoring unknown training parameters: %s", sorted(unknown))
    return {k: v for k, v in params.items() if k in valid}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    params = _coerce(load_params())
    filtered = filter_to_known_kwargs(params)
    run_id = train_detector(**filtered)
    if is_rank0() and run_id:
        print(f"MODEL_URI={run_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
