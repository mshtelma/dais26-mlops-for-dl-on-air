from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, ClassVar

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.transforms import CLIP_MEAN as _CLIP_MEAN_SRC
from src.data.transforms import CLIP_STD as _CLIP_STD_SRC

logger = logging.getLogger(__name__)


class DetectorPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc for object detection.

    Input schema: DataFrame with column 'image' (base64-encoded PNG/JPEG string).
    Output schema: DataFrame with columns:
        - 'boxes':  list[list[float]]   each [x1, y1, x2, y2] pixel coords
        - 'scores': list[float]
        - 'labels': list[str]           class names from label_map
        - 'num_detections': int

    Configuration artifacts (loaded by load_context):
        - 'model_state' (PyTorch .pt file): DetectionModel state_dict
        - 'label_map' (json): {category_id: name}
        - 'backbone_config' (json): {name, revision, summary_dim, spatial_dim, patch_size}
        - 'detection_config' (json): {scales, aspect_ratios, num_classes, score_threshold,
                                       nms_iou_threshold, max_detections, input_size}
    """

    DEFAULT_INPUT_SIZE: int = 1024
    # Source of truth: src/data/transforms.py. Aliased here for backward-compatible attribute access.
    CLIP_MEAN: ClassVar[list[float]] = _CLIP_MEAN_SRC
    CLIP_STD: ClassVar[list[float]] = _CLIP_STD_SRC

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load backbone + FPN + head weights, move to GPU, compile.

        Reads artifacts from the MLflow context. The model is built from configuration JSON
        so that the same pyfunc supports C-RADIOv4 (summary_dim=1152, spatial_dim=1536) AND
        the DINOv2 fallback (summary_dim=768, spatial_dim=768).
        """
        import json

        from src.models.backbones import load_backbone
        from src.models.detection_head import DetectionModel

        artifacts = context.artifacts
        with open(artifacts["backbone_config"]) as f:
            backbone_config = json.load(f)
        with open(artifacts["detection_config"]) as f:
            detection_config = json.load(f)
        with open(artifacts["label_map"]) as f:
            label_map = json.load(f)
        self.label_map = {int(k): v for k, v in label_map.items()}
        self.input_size = detection_config.get("input_size", self.DEFAULT_INPUT_SIZE)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Honor model_cache if provided
        cache_dir = artifacts.get("model_cache")
        if cache_dir is not None:
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir

        backbone, info = load_backbone(
            name=backbone_config["name"],
            revision=backbone_config.get("revision"),
            cache_dir=cache_dir,
            device=device,
        )
        # Build DetectionModel and load weights
        self.model = DetectionModel(
            backbone=backbone,
            spatial_dim=info.spatial_dim,
            num_classes=detection_config["num_classes"],
            scales=detection_config["scales"],
            aspect_ratios=detection_config["aspect_ratios"],
            patch_size=info.patch_size,
            nms_iou_threshold=detection_config["nms_iou_threshold"],
            score_threshold=detection_config["score_threshold"],
            max_detections=detection_config["max_detections"],
        ).to(device)
        state = torch.load(artifacts["model_state"], map_location=device, weights_only=True)
        # Backbone weights come from HF; only load FPN + head state
        head_state = {k: v for k, v in state.items() if not k.startswith("backbone.")}
        missing = self.model.load_state_dict(head_state, strict=False)
        logger.info("Loaded head/FPN state. Missing: %s, Unexpected: %s",
                    len(missing.missing_keys), len(missing.unexpected_keys))
        self.model.eval()
        # torch.compile for latency (skip on CPU to keep tests fast)
        if device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                logger.warning("torch.compile failed (%s); continuing uncompiled", e)
        self.device = device

    def _decode_image(self, b64_str: str) -> tuple[torch.Tensor, tuple[int, int]]:
        """Decode base64 -> normalized (3, H, W) tensor + original (H, W) size."""
        raw = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        orig_size = (img.height, img.width)
        img = img.resize((self.input_size, self.input_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array(self.CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(self.CLIP_STD, dtype=np.float32).reshape(3, 1, 1)
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        arr = (arr - mean) / std
        return torch.from_numpy(arr), orig_size

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Run detection on base64-encoded images."""
        if "image" not in model_input.columns:
            raise ValueError(f"model_input must have 'image' column; got {list(model_input.columns)}")

        rows = []
        for b64 in model_input["image"].astype(str).tolist():
            tensor, orig = self._decode_image(b64)
            batch = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(batch)
            boxes_t = out["boxes"][0].cpu()
            scores_t = out["scores"][0].cpu()
            labels_t = out["labels"][0].cpu()
            # Scale boxes from input_size back to original
            sx = orig[1] / self.input_size
            sy = orig[0] / self.input_size
            scaled = boxes_t.clone()
            scaled[:, [0, 2]] *= sx
            scaled[:, [1, 3]] *= sy
            rows.append({
                "boxes": scaled.tolist(),
                "scores": scores_t.tolist(),
                "labels": [self.label_map.get(int(lbl), str(int(lbl))) for lbl in labels_t.tolist()],
                "num_detections": int(scaled.shape[0]),
            })
        return pd.DataFrame(rows)


def build_signature_and_example() -> tuple[Any, pd.DataFrame]:
    """Construct an MLflow signature and input_example for UC registration.

    UC model registration requires a signature; this helper centralizes its creation
    so the training script and unit tests share the same definition.
    """
    from mlflow.models import infer_signature

    # Tiny black PNG (3x3) for the example
    buf = io.BytesIO()
    Image.new("RGB", (3, 3), (0, 0, 0)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    example = pd.DataFrame({"image": [b64]})
    output = pd.DataFrame([{
        "boxes": [[0.0, 0.0, 1.0, 1.0]],
        "scores": [0.5],
        "labels": ["Caries"],
        "num_detections": 1,
    }])
    signature = infer_signature(example, output)
    return signature, example
