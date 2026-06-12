"""air / torchrun entrypoint for the HPO sweep — terminal twin of 02b.

Runs the SAME `SweepRunner` brain as `notebooks/02b_hpo_sweep.py`, but inside
one torchrun allocation: trials execute sequentially in-process on all ranks
(each is a full DDP training run), instead of one `@distributed` dispatch per
trial. Rank 0 orchestrates; workers follow the runner's broadcast command loop.

Reads `$HYPERPARAMETERS_PATH` (the air workload `parameters:` block), which
must carry:

    stage: dinov3_s1            # a config.campaigns.CAMPAIGN_STAGES name
    env: df1                    # a config.environments entry (catalog / schema /
                                # volume_path / cache_dir / experiment_name)

Optional keys: `model_name` (defaults to the stage backbone's registered
name), `backbone_revision`, `strategy`, `seed`. Any env-derived location can
also be set explicitly to override the named environment.

The process group is initialized ONCE here and shared by every trial
(`Trainer(cfg, manage_process_group=False)`); destroying/re-initializing NCCL
per trial would be needless risk.

Flags mirror `train.cli`: --config, --dry-run, --log-level.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import yaml

from dais26_dentex.config.campaigns import CAMPAIGN_STAGES
from dais26_dentex.config.environments import load_environment
from dais26_dentex.config.recipes import DETECTOR_NAMES_BY_BACKBONE
from dais26_dentex.config.trainer_config import TrainerConfig
from dais26_dentex.distributed import is_rank0, setup_distributed, teardown_distributed
from dais26_dentex.train.sweep_runner import SweepRunner, SweepSpec
from dais26_dentex.train.trainer import Trainer

logger = logging.getLogger(__name__)


def load_sweep_inputs(yaml_path: str) -> tuple[SweepSpec, dict[str, Any]]:
    """Parse the workload parameters into (SweepSpec, base TrainerConfig kwargs)."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML at {yaml_path} did not produce a mapping; got {type(raw).__name__}")

    stage_name = raw.pop("stage", None)
    if not stage_name:
        raise ValueError(f"`stage` is required; known stages: {sorted(CAMPAIGN_STAGES)}")
    if stage_name not in CAMPAIGN_STAGES:
        raise ValueError(f"Unknown stage {stage_name!r}; known: {sorted(CAMPAIGN_STAGES)}")

    env_name = raw.pop("env", None)
    if env_name is not None:
        # Environment locations are defaults; explicit YAML keys still win.
        raw = {**load_environment(env_name).as_training_kwargs(), **raw}

    spec = SweepSpec.from_stage(
        stage_name,
        CAMPAIGN_STAGES[stage_name],
        strategy=raw.pop("strategy", "random"),
        seed=int(raw.pop("seed", 42)),
    )
    base = dict(raw)
    base.setdefault("model_name", DETECTOR_NAMES_BY_BACKBONE[spec.backbone]["model_short"])
    if not base.get("experiment_name"):
        logger.warning(
            "experiment_name is not set: sweep + trial runs will land in the pod's "
            "ambient/default MLflow experiment, invisible to the deployment-job "
            "best-in-experiment gate. Set parameters.experiment_name in the workload."
        )
    return spec, base


def _launch(cfg_kwargs: dict[str, Any]) -> str | None:
    """In-process trial executor: every torchrun rank trains; the shared PG is
    owned by main(), not the per-trial Trainer."""
    cfg = TrainerConfig.from_dict(cfg_kwargs)
    return Trainer(cfg, manage_process_group=False).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dais26-sweep")
    parser.add_argument("--config", default=None, help="YAML config path (overrides $HYPERPARAMETERS_PATH)")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved sweep spec and exit")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    yaml_path = args.config or os.environ.get("HYPERPARAMETERS_PATH") or None
    if yaml_path is None:
        logger.error("No config path provided. Use --config or set $HYPERPARAMETERS_PATH.")
        return 2
    if not os.path.exists(yaml_path):
        logger.error("Config file does not exist: %s", yaml_path)
        return 2

    spec, base = load_sweep_inputs(yaml_path)

    # The AIR CLI exports MLFLOW_RUN_ID for the workload's OWN MLflow run; the
    # sweep manages its own parent + per-trial runs, so a leaked ambient run id
    # would make every trial's fluent mlflow.start_run() attach to the same
    # workload run. Clear it on every rank — before the dry-run return so the
    # printed env is faithful, and before the mlflow import below.
    if os.environ.pop("MLFLOW_RUN_ID", None):
        logger.info("Cleared ambient MLFLOW_RUN_ID (AIR workload run); sweep manages its own runs.")

    if args.dry_run:
        print(f"stage={spec.stage_name} backbone={spec.backbone}")
        print(f"trial_epochs={spec.trial_epochs} max_trials={spec.max_trials}")
        print(f"schedule_epochs={spec.schedule_epochs} register_winner={spec.register_winner}")
        print(f"pinned={dict(spec.pinned)}")
        print(f"search_space={dict(spec.search_space)}")
        print(f"base={base}")
        return 0

    # Deferred: mlflow import is heavy and unnecessary for --dry-run.
    import mlflow
    from mlflow.tracking import MlflowClient

    if is_rank0():
        # UC 3-part model names need the UC registry; the experiment env var
        # backs up an unset cfg.experiment_name exactly like the notebook lane.
        mlflow.set_registry_uri("databricks-uc")
        if base.get("experiment_name"):
            os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", str(base["experiment_name"]))

    setup_distributed()
    try:
        runner = SweepRunner(
            spec,
            base_config_kwargs=base,
            launch=_launch,
            client=MlflowClient(),
            model_fqn=f"{base['catalog']}.{base['schema']}.{base['model_name']}",
        )
        outcome = runner.run()
    finally:
        teardown_distributed()

    if is_rank0():
        winner_metric = outcome.winner.metric if outcome.winner else None
        print(f"SWEEP_PARENT_RUN={outcome.parent_run_id}", flush=True)
        print(f"SWEEP_WINNER_METRIC={winner_metric}", flush=True)
        print(f"SWEEP_RETRAIN_RUN={outcome.retrain_run_id}", flush=True)
        print(f"SWEEP_REGISTERED_VERSION={outcome.registered_version}", flush=True)
        print(f"SWEEP_CHALLENGER_SET={outcome.challenger_set}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
