"""Thin orchestrator over `Trainer` — preserved for notebook compatibility.

`notebooks/02_train_detector_air.py` calls `train_detector(...)` directly via
`@distributed`, so the kwarg-based signature stays. The body just builds a
`TrainerConfig` and delegates.

The 360-line procedural body has moved to `train.trainer.Trainer` (which
also fixes the cls-only loss, the smoke-test target stub, the hardcoded
`num_classes=4`, and the discarded validation loader).

For sgcli runs, `train.cli` parses YAML directly into a `TrainerConfig` and
invokes `Trainer(cfg).run()` -- it doesn't go through this function.
"""

from __future__ import annotations

import logging
from typing import Literal

from dais26_dentex.config.trainer_config import TrainerConfig
from dais26_dentex.train.trainer import Trainer

logger = logging.getLogger(__name__)

BackboneName = Literal["cradio_v4_so400m", "dinov3_vitl16", "dinov2_base"]


def train_detector(
    catalog: str,
    schema: str,
    backbone_name: BackboneName = "cradio_v4_so400m",
    backbone_revision: str | None = None,
    volume_path: str | None = None,
    cache_dir: str | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 8,
    num_workers: int = 4,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 32.0,
    experiment_name: str | None = None,
    model_name: str = "cradio_detector",
    register_model: bool = True,
    set_candidate_alias: bool = True,
    img_size: int = 1024,
    base_seed: int = 42,
) -> str | None:
    """Build TrainerConfig from kwargs and delegate to `Trainer.run()`.

    Returns: MLflow run_id on rank 0, None on other ranks.
    """
    cfg = TrainerConfig(
        catalog=catalog,
        schema=schema,
        backbone_name=backbone_name,
        backbone_revision=backbone_revision,
        volume_path=volume_path,
        cache_dir=cache_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        num_workers=num_workers,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        experiment_name=experiment_name,
        model_name=model_name,
        register_model=register_model,
        set_candidate_alias=set_candidate_alias,
        img_size=img_size,
        base_seed=base_seed,
    )
    return Trainer(cfg).run()


__all__ = ["BackboneName", "train_detector"]
