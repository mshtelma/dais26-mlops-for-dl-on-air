from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors

Method = Literal['knn', 'mmd', 'energy']


@dataclass
class ReferenceDistribution:
    """Fitted reference distribution for drift detection.

    Attributes:
        method: 'knn', 'mmd', or 'energy'.
        embeddings: (N_ref, D) reference embeddings (L2-normalized).
        knn_index: sklearn NearestNeighbors index (for method='knn').
        params: method-specific parameters (k, bandwidth, etc.).
    """
    method: Method
    embeddings: np.ndarray
    knn_index: NearestNeighbors | None = None
    params: dict[str, Any] = field(default_factory=dict)


def _median_pairwise_distance(x: np.ndarray, sample_size: int = 500) -> float:
    """Median heuristic for MMD Gaussian kernel bandwidth."""
    n = x.shape[0]
    idx = np.random.default_rng(42).choice(n, size=min(n, sample_size), replace=False)
    sub = x[idx]
    sq = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
    triu = sq[np.triu_indices_from(sq, k=1)]
    return float(np.sqrt(np.median(triu)))


def fit_reference(
    embeddings: np.ndarray,
    method: Method = 'knn',
    k: int = 50,
) -> ReferenceDistribution:
    """Fit a reference distribution from training embeddings.

    Args:
        embeddings: (N, D) L2-normalized training embeddings.
        method: 'knn', 'mmd', or 'energy'.
        k: number of neighbors for KNN method.

    Returns:
        ReferenceDistribution.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"embeddings must be 2-D non-empty, got shape {embeddings.shape}")

    if method == 'knn':
        # Use cosine via L2 on normalized vectors equivalently
        n_neighbors = min(k, embeddings.shape[0])
        idx = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        idx.fit(embeddings)
        return ReferenceDistribution(
            method='knn',
            embeddings=embeddings,
            knn_index=idx,
            params={'k': n_neighbors},
        )
    if method == 'mmd':
        bw = _median_pairwise_distance(embeddings)
        return ReferenceDistribution(
            method='mmd',
            embeddings=embeddings,
            params={'bandwidth': bw},
        )
    if method == 'energy':
        return ReferenceDistribution(method='energy', embeddings=embeddings)

    raise ValueError(f"Unknown method: {method}")
