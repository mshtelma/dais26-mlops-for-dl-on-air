"""MLflow pyfunc wrappers for the detection model.

Two classes are exported:

* ``DetectorPyfunc`` — current (v2) consumer. Reads a single
  ``manifest.json`` artifact (see ``config/manifest.py``).
* ``DetectorPyfuncV1`` — v1 reader retained for one release so an
  artifact logged before the v2 cut-over still loads. Will be removed
  next minor.

Both classes share the same ``predict`` shape so a v1 endpoint URL can
be swapped to a re-trained v2 artifact without changing the caller.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, ClassVar

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image

from dais26_dentex.config.constants import (
    BACKBONE_CONFIG_FILE,
    DETECTION_CONFIG_FILE,
    LABEL_MAP_FILE,
    MANIFEST_FILE,
    MODEL_STATE_FILE,
)
from dais26_dentex.config.manifest import load_manifest
from dais26_dentex.data.transforms import CLIP_MEAN as _CLIP_MEAN_SRC
from dais26_dentex.data.transforms import CLIP_STD as _CLIP_STD_SRC
from dais26_dentex.platform.hf_env import configure_hf_env
from dais26_dentex.serve.postprocess import PostprocessConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def _decode_b64_image(
    b64_str: str, input_size: int, mean: list[float], std: list[float]
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Decode base64 → normalized (3, H, W) tensor + original (H, W) size.

    Pulled out as a free function so both v1 and v2 wrappers reuse the
    exact same normalization. CLIP_MEAN/STD live in ``data/transforms``
    as the single source of truth (see Phase 1).
    """
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_size = (img.height, img.width)
    img = img.resize((input_size, input_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean_a = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std_a = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = (arr - mean_a) / std_a
    return torch.from_numpy(arr), orig_size


def _build_detection_model(
    *,
    backbone_name: str,
    backbone_revision: str | None,
    cache_dir: str | None,
    device: str,
    num_classes: int,
    scales: list[int],
    aspect_ratios: list[float],
    nms_iou_threshold: float,
    score_threshold: float,
    max_detections: int,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    """Build a `DetectionModel` and return ``(model, info)``.

    Same builder both v1 and v2 wrappers use. Done inline (not via
    ``models.builder.build_detector``) because the trainer's builder
    expects a ``TrainerConfig`` shape and the pyfunc has artifact-shaped
    inputs instead.

    ``local_files_only`` is set at serving time to force a strictly-offline
    backbone load from the bundled HF cache (no egress in the serving box).
    """
    from dais26_dentex.models.backbones import load_backbone
    from dais26_dentex.models.detection_head import DetectionModel

    backbone, info = load_backbone(
        name=backbone_name,
        revision=backbone_revision,
        cache_dir=cache_dir,
        device=device,
        local_files_only=local_files_only,
    )
    model = DetectionModel(
        backbone=backbone,
        spatial_dim=info.spatial_dim,
        num_classes=num_classes,
        scales=scales,
        aspect_ratios=aspect_ratios,
        patch_size=info.patch_size,
        nms_iou_threshold=nms_iou_threshold,
        score_threshold=score_threshold,
        max_detections=max_detections,
    ).to(device)
    return model, info


def _load_state_into(model: Any, state_path: str, device: str) -> None:
    """Load the FPN+head state dict; backbone weights come from HF."""
    state = torch.load(state_path, map_location=device, weights_only=True)
    head_state = {k: v for k, v in state.items() if not k.startswith("backbone.")}
    missing = model.load_state_dict(head_state, strict=False)
    logger.info(
        "Loaded head/FPN state. Missing: %s, Unexpected: %s",
        len(missing.missing_keys),
        len(missing.unexpected_keys),
    )


def _maybe_compile(model: Any, device: str) -> Any:
    """Return the model uncompiled at serving time.

    ``torch.compile`` is intentionally NOT applied here. The frozen HF
    transformer backbones (``facebook/dinov3-vitl16``, ``nvidia/C-RADIOv4``)
    run fine eagerly — training + validation forward them in eager mode — but
    wrapping the full ``DetectionModel`` with ``torch.compile`` makes
    TorchDynamo trace *into* the backbone at the first ``predict`` call. The
    DINOv3 modeling stack routes through transformers' output-capturing
    decorator (``transformers/utils/output_capturing.py``), whose module
    namespace does not bind ``torch``; the Dynamo-transformed frame then raises
    ``NameError: name 'torch' is not defined`` during inference (the model
    still *loads* fine, since ``load_context`` runs eagerly). ``reduce-overhead``
    (CUDA graphs) also needs static shapes, which the variable-size image input
    path does not guarantee. The marginal latency win is not worth the serving
    fragility, so we serve the eager module.
    """
    return model


def _predict_batch(
    *,
    model: Any,
    device: str,
    label_map: dict[int, str],
    input_size: int,
    mean: list[float],
    std: list[float],
    model_input: pd.DataFrame,
) -> pd.DataFrame:
    """Per-image decode + forward + scaling. Used by both v1 and v2."""
    if "image" not in model_input.columns:
        raise ValueError(f"model_input must have 'image' column; got {list(model_input.columns)}")

    rows: list[dict[str, Any]] = []
    for b64 in model_input["image"].astype(str).tolist():
        tensor, orig = _decode_b64_image(b64, input_size, mean, std)
        batch = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(batch)
        boxes_t = out["boxes"][0].cpu()
        scores_t = out["scores"][0].cpu()
        labels_t = out["labels"][0].cpu()
        sx = orig[1] / input_size
        sy = orig[0] / input_size
        scaled = boxes_t.clone()
        scaled[:, [0, 2]] *= sx
        scaled[:, [1, 3]] *= sy
        rows.append(
            {
                "boxes": scaled.tolist(),
                "scores": scores_t.tolist(),
                "labels": [label_map.get(int(lbl), str(int(lbl))) for lbl in labels_t.tolist()],
                "num_detections": int(scaled.shape[0]),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# v2 (current)
# ----------------------------------------------------------------------


class DetectorPyfunc(mlflow.pyfunc.PythonModel):
    """v2 detector pyfunc.

    Input schema: DataFrame with column ``image`` (base64-encoded PNG/JPEG).
    Output schema: DataFrame with columns:
        - ``boxes``:  list[list[float]]  each [x1, y1, x2, y2] pixel coords
        - ``scores``: list[float]
        - ``labels``: list[str]          class names from the manifest's label_map
        - ``num_detections``: int

    Single artifact contract:
        - ``manifest`` (json) — see ``config.manifest.Manifest``
        - ``model_state`` (.pt) — DetectionModel state_dict
        - ``model_cache`` (optional) — UC Volume path for HF cache
    """

    DEFAULT_INPUT_SIZE: int = 1024
    CLIP_MEAN: ClassVar[list[float]] = _CLIP_MEAN_SRC
    CLIP_STD: ClassVar[list[float]] = _CLIP_STD_SRC

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        artifacts = context.artifacts
        if MANIFEST_FILE.split(".")[0] not in artifacts and "manifest" not in artifacts:
            raise FileNotFoundError(
                f"Expected '{MANIFEST_FILE}' artifact (key 'manifest'); "
                f"got keys: {sorted(artifacts.keys())}. "
                f"For v1 artifacts use DetectorPyfuncV1."
            )
        manifest_path = artifacts.get("manifest") or artifacts[MANIFEST_FILE.split(".")[0]]
        manifest = load_manifest(manifest_path)

        self.label_map: dict[int, str] = {int(k): v for k, v in manifest.label_map.items()}
        self.input_size: int = manifest.detector.input_size

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = artifacts.get("model_cache")
        # Serving bundles the HF cache as an artifact and the container has no
        # egress — force a strictly-offline load so trust_remote_code does not
        # try to reach huggingface.co (which fails model-server startup).
        offline = cache_dir is not None
        configure_hf_env(cache_dir, offline=offline)

        model, _info = _build_detection_model(
            backbone_name=manifest.backbone.name,
            backbone_revision=manifest.backbone.revision,
            cache_dir=cache_dir,
            device=device,
            num_classes=manifest.detector.num_classes,
            scales=manifest.detector.scales,
            aspect_ratios=manifest.detector.aspect_ratios,
            nms_iou_threshold=manifest.detector.nms_iou_threshold,
            score_threshold=manifest.detector.score_threshold,
            max_detections=manifest.detector.max_detections,
            local_files_only=offline,
        )
        _load_state_into(model, artifacts["model_state"], device)
        model.eval()
        self.model = _maybe_compile(model, device)
        self.device = device
        # Stash for downstream callers that want to override at predict-time.
        self.postprocess_cfg = PostprocessConfig(
            score_threshold=manifest.detector.score_threshold,
            nms_iou_threshold=manifest.detector.nms_iou_threshold,
            max_detections=manifest.detector.max_detections,
        )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        return _predict_batch(
            model=self.model,
            device=self.device,
            label_map=self.label_map,
            input_size=self.input_size,
            mean=self.CLIP_MEAN,
            std=self.CLIP_STD,
            model_input=model_input,
        )


# ----------------------------------------------------------------------
# v1 (deprecated, kept one release)
# ----------------------------------------------------------------------


class DetectorPyfuncV1(mlflow.pyfunc.PythonModel):
    """v1 reader — three-sidecar JSON layout (backbone_config /
    detection_config / label_map).

    Retained for one release so an artifact registered before the
    manifest-v2 cut-over still loads. Re-train with the current code to
    move to v2 (a typed manifest + one-file contract). Will be removed
    in the next minor release.
    """

    DEFAULT_INPUT_SIZE: int = 1024
    CLIP_MEAN: ClassVar[list[float]] = _CLIP_MEAN_SRC
    CLIP_STD: ClassVar[list[float]] = _CLIP_STD_SRC

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        import json

        artifacts = context.artifacts
        with open(artifacts[BACKBONE_CONFIG_FILE.split(".")[0]]) as f:
            backbone_config = json.load(f)
        with open(artifacts[DETECTION_CONFIG_FILE.split(".")[0]]) as f:
            detection_config = json.load(f)
        with open(artifacts[LABEL_MAP_FILE.split(".")[0]]) as f:
            label_map = json.load(f)

        self.label_map: dict[int, str] = {int(k): v for k, v in label_map.items()}
        self.input_size = int(detection_config.get("input_size", self.DEFAULT_INPUT_SIZE))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = artifacts.get("model_cache")
        offline = cache_dir is not None
        configure_hf_env(cache_dir, offline=offline)

        model, _info = _build_detection_model(
            backbone_name=backbone_config["name"],
            backbone_revision=backbone_config.get("revision"),
            cache_dir=cache_dir,
            device=device,
            num_classes=detection_config["num_classes"],
            scales=detection_config["scales"],
            aspect_ratios=detection_config["aspect_ratios"],
            nms_iou_threshold=detection_config["nms_iou_threshold"],
            score_threshold=detection_config["score_threshold"],
            max_detections=detection_config["max_detections"],
            local_files_only=offline,
        )
        _load_state_into(model, artifacts[MODEL_STATE_FILE.split(".")[0]], device)
        model.eval()
        self.model = _maybe_compile(model, device)
        self.device = device

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        return _predict_batch(
            model=self.model,
            device=self.device,
            label_map=self.label_map,
            input_size=self.input_size,
            mean=self.CLIP_MEAN,
            std=self.CLIP_STD,
            model_input=model_input,
        )


# ----------------------------------------------------------------------
# Signature/example helper (used by the trainer + tests)
# ----------------------------------------------------------------------


def build_signature_and_example() -> tuple[Any, pd.DataFrame]:
    """Construct an MLflow signature and input_example for UC registration."""
    from mlflow.models import infer_signature

    buf = io.BytesIO()
    Image.new("RGB", (3, 3), (0, 0, 0)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    example = pd.DataFrame({"image": [b64]})
    output = pd.DataFrame(
        [
            {
                "boxes": [[0.0, 0.0, 1.0, 1.0]],
                "scores": [0.5],
                "labels": ["Caries"],
                "num_detections": 1,
            }
        ]
    )
    signature = infer_signature(example, output)
    return signature, example


__all__ = [
    "DetectorPyfunc",
    "DetectorPyfuncV1",
    "build_signature_and_example",
]
