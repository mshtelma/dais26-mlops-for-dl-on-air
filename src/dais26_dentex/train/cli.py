"""air / torchrun entrypoint for `train_detector`.

Reads `$HYPERPARAMETERS_PATH` (a YAML file the AIR CLI writes from the workload's
`parameters:` block), builds a `TrainerConfig`, and runs `Trainer(cfg).run()`
directly.

The YAML may name an `env:` (a `config.environments` entry — catalog / schema /
volume_path / cache_dir / experiment_name) and a `recipe:` (a backbone literal
from `config.recipes.RECIPES` — the best-known hyperparameters). Each expands
into defaults that the remaining YAML keys override. This is what keeps the air
workload `parameters:` block down to two names plus a deliberate override or
two, instead of a hand-maintained mirror of the notebook constants; both launch
lanes resolve the same env + recipe.

Coercion of stringly-typed YAML values lives on the dataclass
(`TrainerConfig.from_dict`), so adding a new knob is a one-line change to
`TrainerConfig`.

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

import yaml

from dais26_dentex.config.environments import load_environment
from dais26_dentex.config.recipes import RECIPES, build_trainer_config
from dais26_dentex.config.trainer_config import TrainerConfig
from dais26_dentex.distributed import is_rank0
from dais26_dentex.train.trainer import Trainer

logger = logging.getLogger(__name__)


def load_config(yaml_path: str) -> TrainerConfig:
    """Build a TrainerConfig from a (air-written) parameters YAML.

    Two optional resolution keys, each mirroring the notebook lane: `env:`
    names a `config.environments` entry (catalog / schema / volume_path /
    cache_dir / experiment_name), and `recipe:` names a `config.recipes` entry
    (hyperparameters). Each expands into defaults that the YAML's remaining
    keys override (explicit always wins). Without either, the YAML must carry a
    full config (legacy behavior).
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML at {yaml_path} did not produce a mapping; got {type(raw).__name__}")
    recipe = raw.pop("recipe", None)
    env_name = raw.pop("env", None)
    if env_name is not None:
        # Environment locations are defaults; explicit YAML keys still win.
        raw = {**load_environment(env_name).as_training_kwargs(), **raw}
    if recipe is None:
        return TrainerConfig.from_dict(raw)
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe {recipe!r}; known: {sorted(RECIPES)}")
    try:
        catalog = raw.pop("catalog")
        schema = raw.pop("schema")
    except KeyError as e:
        raise ValueError(
            f"recipe-based config requires {e.args[0]!r} (set it directly or via env:)"
        ) from e
    return build_trainer_config(recipe, catalog=catalog, schema=schema, **raw)


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

    cfg = load_config(yaml_path)
    cfg.validate()
    if cfg.experiment_name is None:
        logger.warning(
            "experiment_name is not set: the training run will land in the pod's "
            "ambient/default MLflow experiment (or the AIR workload's own run via "
            "MLFLOW_RUN_ID), invisible to the sweep and deployment-job "
            "best-in-experiment gates. Set parameters.experiment_name in the air "
            "workload (distinct from the workload's own top-level experiment_name)."
        )
    elif os.environ.pop("MLFLOW_RUN_ID", None):
        # The AIR CLI exports MLFLOW_RUN_ID for the workload's OWN MLflow run;
        # left in place, mlflow.start_run() inside the Trainer would attach to
        # that run instead of creating a fresh one in the configured
        # experiment. Clear it so the training run lands where the gates look.
        logger.info(
            "Cleared ambient MLFLOW_RUN_ID (AIR workload run); training run will "
            "be created in %s.",
            cfg.experiment_name,
        )

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
