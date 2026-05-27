from __future__ import annotations

import io
import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _bytes_to_tensor(image_bytes: bytes, target_size: int = 224) -> torch.Tensor:
    """Decode bytes -> PIL -> normalized tensor (3, H, W). Uses CLIP normalization."""
    from dais26_dentex.data.transforms import CLIP_MEAN, CLIP_STD

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((target_size, target_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(CLIP_STD, dtype=np.float32).reshape(3, 1, 1)
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = (arr - mean) / std
    return torch.from_numpy(arr)


def compute_embeddings(
    backbone: torch.nn.Module,
    images: list[bytes] | torch.Tensor,
    batch_size: int = 32,
    device: str = "cuda",
    image_size: int = 224,
) -> np.ndarray:
    """Compute L2-normalized summary embeddings for a batch of images.

    Args:
        backbone: wrapped backbone returning (summary, spatial_features) tuple.
                  See src/dais26_dentex/models/backbones.py for the contract.
        images: list of base64-decoded image bytes OR a preprocessed (N, 3, H, W) Tensor.
        batch_size: forward-pass batch size.
        device: 'cuda' or 'cpu'.
        image_size: resize target. C-RADIOv4 accepts variable sizes; 224 is fastest for drift.

    Returns:
        np.ndarray of shape (N, summary_dim) with L2-normalized rows.
    """
    backbone.eval()
    backbone.to(device)
    if isinstance(images, torch.Tensor):
        tensors = images
    else:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        tensors = torch.stack([_bytes_to_tensor(b, image_size) for b in images])

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, tensors.shape[0], batch_size):
            batch = tensors[start : start + batch_size].to(device)
            summary, _ = backbone(batch)  # (B, summary_dim), (B, T, spatial_dim)
            # L2-normalize
            normed = summary / (summary.norm(dim=-1, keepdim=True) + 1e-12)
            embeddings.append(normed.cpu().numpy())

    return np.concatenate(embeddings, axis=0)
