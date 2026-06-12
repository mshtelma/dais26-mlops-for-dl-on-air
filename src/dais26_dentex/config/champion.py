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


def resolve_effective_backbone(
    champion_model_name: str,
    names_by_backbone: Mapping[str, Mapping[str, str]],
    default_backbone: str,
    *,
    alias: str = "champion",
    registry_uri: str = "databricks-uc",
) -> tuple[str, str | None]:
    """Resolve the LIVE champion's backbone from the registry (impure companion).

    Reads the ``source_dev_model`` tag off the ``@{alias}`` version of
    ``champion_model_name`` via the UC registry and reverse-maps it with
    `resolve_backbone_from_source_model`. Returns ``(effective_backbone,
    source_dev_model)`` — falling back to ``default_backbone`` (and ``None``)
    when there is no champion yet / the tag is missing (first run, or a
    standalone dev invocation). `mlflow` is imported lazily so the module's pure
    helpers stay import-light + unit-testable.

    Used by notebooks 03 (precompute embeddings) and 05 (drift) so the refresh
    embeds images with the SAME feature extractor the champion was trained with
    (the prod champion is backbone-agnostic, so the static config ``BACKBONE``
    may not match the live champion's architecture).
    """
    from mlflow.tracking import MlflowClient

    source_dev_model: str | None = None
    try:
        mv = MlflowClient(registry_uri=registry_uri).get_model_version_by_alias(
            name=champion_model_name, alias=alias
        )
        source_dev_model = (mv.tags or {}).get(SOURCE_DEV_MODEL_TAG)
    except Exception as e:
        print(
            f"No resolvable @{alias} on {champion_model_name} "
            f"({type(e).__name__}: {e}); falling back to {default_backbone}"
        )
    effective = resolve_backbone_from_source_model(
        source_dev_model, names_by_backbone, default_backbone
    )
    return effective, source_dev_model
