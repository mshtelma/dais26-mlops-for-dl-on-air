"""Champion-backbone routing helpers.

The prod champion is a SINGLE, backbone-agnostic registered model
(`CHAMPION_MODEL_NAME`): the approved dev winner of any architecture is copied
into it and tagged with `source_dev_model` (the dev model it came from). The
embeddings / Vector-Search / drift refresh that runs after a champion deploy must
use the SAME backbone the live champion was trained with — not a static config
default — otherwise it would embed images with the wrong feature extractor.

This module is pure (no MLflow / Spark imports) so it stays unit-testable; the
notebook reads the `source_dev_model` tag off the `@champion` version and passes
it here together with the backbone→short-name map from `00_config`.
"""

from __future__ import annotations

from collections.abc import Mapping

SOURCE_DEV_MODEL_TAG = "source_dev_model"


def resolve_backbone_from_source_model(
    source_dev_model: str | None,
    names_by_backbone: Mapping[str, Mapping[str, str]],
    default_backbone: str,
) -> str:
    """Reverse-map a dev model name to the backbone that produced it.

    Args:
        source_dev_model: Fully-qualified dev model the champion was copied from
            (e.g. ``mlops_pj.dais26_vfm.dinov3_detector``), as recorded in the
            ``source_dev_model`` tag. ``None``/empty when there is no champion yet
            or the tag is missing.
        names_by_backbone: The ``{backbone: {"model_short": ..., ...}}`` map from
            ``notebooks/00_config.py``.
        default_backbone: Backbone to fall back to when the source model is
            unknown or absent (the static ``BACKBONE`` config value).

    Returns:
        The backbone key whose ``model_short`` matches the source model's short
        (last dot-segment) name, else ``default_backbone``.
    """
    if not source_dev_model:
        return default_backbone
    short = source_dev_model.rsplit(".", 1)[-1]
    for backbone, names in names_by_backbone.items():
        if names.get("model_short") == short:
            return backbone
    return default_backbone
