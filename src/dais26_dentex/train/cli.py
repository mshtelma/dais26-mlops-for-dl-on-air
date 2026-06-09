"""sgcli / torchrun entrypoint for `train_detector`.

Reads `$HYPERPARAMETERS_PATH` (a YAML file that sgcli writes from the workload's
`parameters:` block), builds a `TrainerConfig`, and runs `Trainer(cfg).run()`
directly.

We deliberately do NOT round-trip through `train_detector(**cfg.to_kwargs_for_train_detector())`:
that helper forwards only the legacy kwarg subset, which silently dropped the
loss/optimizer/anchor/backbone knobs (`weight_decay`, `onecycle_pct_start`,
`grad_clip_norm`, `focal_*`, `box_loss_weight`, `anchor_scales`, `aspect_ratios`,
`backbone_mode`, `backbone_lr`). Constructing the `Trainer` from the fully-typed
`cfg` honors every field the YAML sets.

The old hand-rolled `_INT_KEYS / _FLOAT_KEYS / _BOOL_KEYS / filter_to_known_kwargs /
_coerce` lives on the dataclass now (see `config.trainer_config.TrainerConfig.from_dict`),
so adding a new knob is a one-line change to `TrainerConfig` instead of editing
two coercion lists plus the YAML.

Flags:
    --config <path>   Override `$HYPERPARAMETERS_PATH`. Useful for local dry-runs.
    --dry-run         Print the resolved `TrainerConfig` and exit 0 without training.
    --log-level       Logging level (default INFO).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dais26_dentex.config.trainer_config import TrainerConfig
from dais26_dentex.distributed import is_rank0
from dais26_dentex.train.trainer import Trainer

logger = logging.getLogger(__name__)


def _resolve_yaml_path(args_config: str | None) -> str | None:
    """Pick the YAML path: --config > $HYPERPARAMETERS_PATH > None."""
    if args_config:
        return args_config
    return os.environ.get("HYPERPARAMETERS_PATH") or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dais26-train")
    parser.add_argument("--config", default=None, help="YAML config path (overrides $HYPERPARAMETERS_PATH)")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved TrainerConfig and exit")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    yaml_path = _resolve_yaml_path(args.config)
    if yaml_path is None:
        logger.error("No config path provided. Use --config or set $HYPERPARAMETERS_PATH.")
        return 2
    if not os.path.exists(yaml_path):
        logger.error("Config file does not exist: %s", yaml_path)
        return 2

    cfg = TrainerConfig.from_yaml(yaml_path)
    cfg.validate()

    if args.dry_run:
        for k, v in cfg.to_dict().items():
            print(f"{k}: {v}")
        return 0

    run_id = Trainer(cfg).run()
    if is_rank0() and run_id:
        print(f"MODEL_URI={run_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
