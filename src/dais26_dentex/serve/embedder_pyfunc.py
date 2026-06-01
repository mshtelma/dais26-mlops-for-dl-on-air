from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, ClassVar

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image

from dais26_dentex.data.transforms import CLIP_MEAN as _CLIP_MEAN_SRC
from dais26_dentex.data.transforms import CLIP_STD as _CLIP_STD_SRC
from dais26_dentex.platform.hf_env import configure_hf_env

logger = logging.getLogger(__name__)


class EmbedderPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc for CLS-summary embedding extraction.

    Input schema: DataFrame with column 'image' (base64-encoded PNG/JPEG string).
    Output schema: DataFrame with column 'embedding' (list[float], length backbone.summary_dim).
    For C-RADIOv4-SO400M: length 1152. For DINOv2-base fallback: length 768.

    Configuration artifacts:
        - 'backbone_config' (json): {name, revision, summary_dim, spatial_dim, patch_size}
        - 'embedder_config' (json): {input_size}
    """

    DEFAULT_INPUT_SIZE: int = 224
    # Source of truth: src/dais26_dentex/data/transforms.py. Aliased here for backward-compatible attribute access.
    CLIP_MEAN: ClassVar[list[float]] = _CLIP_MEAN_SRC
    CLIP_STD: ClassVar[list[float]] = _CLIP_STD_SRC

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from dais26_dentex.models.backbones import load_backbone

        artifacts = context.artifacts
        with open(artifacts["backbone_config"]) as f:
            backbone_config = json.load(f)
        embedder_config = {"input_size": self.DEFAULT_INPUT_SIZE}
        cfg_path = artifacts.get("embedder_config")
        if cfg_path is not None:
            with open(cfg_path) as f:
                embedder_config.update(json.load(f))
        self.input_size = embedder_config["input_size"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = artifacts.get("model_cache")
        configure_hf_env(cache_dir)
        self.backbone, self.info = load_backbone(
            name=backbone_config["name"],
            revision=backbone_config.get("revision"),
            cache_dir=cache_dir,
            device=device,
        )
        self.backbone.eval()
        if device == "cuda":
            try:
                self.backbone = torch.compile(self.backbone, mode="reduce-overhead")
            except Exception as e:
                logger.warning("torch.compile failed (%s); continuing uncompiled", e)
        self.device = device

    def _decode_image(self, b64_str: str) -> torch.Tensor:
        raw = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((self.input_size, self.input_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array(self.CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(self.CLIP_STD, dtype=np.float32).reshape(3, 1, 1)
        arr = arr.transpose(2, 0, 1)
        arr = (arr - mean) / std
        return torch.from_numpy(arr)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if "image" not in model_input.columns:
            raise ValueError(f"model_input must have 'image' column; got {list(model_input.columns)}")
        tensors = torch.stack([self._decode_image(b) for b in model_input["image"].astype(str).tolist()])
        tensors = tensors.to(self.device)
        with torch.no_grad():
            summary, _ = self.backbone(tensors)
            summary = summary / (summary.norm(dim=-1, keepdim=True) + 1e-12)
        return pd.DataFrame({"embedding": [row.cpu().tolist() for row in summary]})


def build_embedder_signature_and_example(summary_dim: int = 1152) -> tuple[Any, pd.DataFrame]:
    """Construct an MLflow signature for the embedder.

    summary_dim defaults to 1152 (C-RADIOv4); pass 768 for the DINOv2 fallback path.
    """
    from mlflow.models import infer_signature

    buf = io.BytesIO()
    Image.new("RGB", (3, 3), (0, 0, 0)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    example = pd.DataFrame({"image": [b64]})
    output = pd.DataFrame({"embedding": [[0.0] * summary_dim]})
    signature = infer_signature(example, output)
    return signature, example
