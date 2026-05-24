from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from scipy.spatial.distance import cdist

from src.drift.reference import ReferenceDistribution, fit_reference

logger = logging.getLogger(__name__)


def _gaussian_kernel(x: np.ndarray, y: np.ndarray, bw: float) -> np.ndarray:
    sq = cdist(x, y, 'sqeuclidean')
    return np.exp(-sq / (2 * bw**2))


def score_drift(
    incoming: np.ndarray,
    reference: ReferenceDistribution,
    method: Literal['knn', 'mmd', 'energy'] | None = None,
) -> float:
    """Compute a scalar drift score (higher = more drift).

    - KNN: mean cosine distance from each incoming embedding to its k-th nearest neighbor in reference.
    - MMD: unbiased MMD^2 between incoming and reference with Gaussian kernel (median bandwidth from reference).
    - Energy: mean pairwise L2 between (incoming, ref) minus half of within-ref and within-incoming.
    """
    if incoming.ndim != 2 or incoming.shape[0] == 0:
        raise ValueError(f"incoming must be 2-D non-empty, got {incoming.shape}")
    m = method or reference.method

    if m == 'knn':
        if reference.knn_index is None:
            raise ValueError("Reference has no fitted KNN index")
        dists, _ = reference.knn_index.kneighbors(incoming)
        # mean over rows of the largest-K distance per row
        return float(dists[:, -1].mean())

    if m == 'mmd':
        bw = reference.params.get('bandwidth', 1.0)
        kxx = _gaussian_kernel(incoming, incoming, bw)
        kyy = _gaussian_kernel(reference.embeddings, reference.embeddings, bw)
        kxy = _gaussian_kernel(incoming, reference.embeddings, bw)
        n, mref = incoming.shape[0], reference.embeddings.shape[0]
        # Unbiased estimator (zero-out diag for self terms)
        kxx_off = (kxx.sum() - np.trace(kxx)) / max(n * (n - 1), 1)
        kyy_off = (kyy.sum() - np.trace(kyy)) / max(mref * (mref - 1), 1)
        kxy_mean = kxy.mean()
        return float(kxx_off + kyy_off - 2 * kxy_mean)

    if m == 'energy':
        ref = reference.embeddings
        d_xy = cdist(incoming, ref, 'euclidean').mean()
        d_xx = cdist(incoming, incoming, 'euclidean').mean()
        d_yy = cdist(ref, ref, 'euclidean').mean()
        return float(2 * d_xy - d_xx - d_yy)

    raise ValueError(f"Unknown method: {m}")


def bootstrap_drift_ci(
    incoming: np.ndarray,
    reference: ReferenceDistribution,
    n_iterations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap 95% CI for the drift score.

    For each iteration: resample `incoming` with replacement, compute drift, collect.
    Returns: {'mean': float, 'p2_5': float, 'p97_5': float}.
    """
    rng = np.random.default_rng(seed)
    n = incoming.shape[0]
    scores: list[float] = []
    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(score_drift(incoming[idx], reference))
    arr = np.asarray(scores)
    return {
        'mean': float(arr.mean()),
        'p2_5': float(np.percentile(arr, 2.5)),
        'p97_5': float(np.percentile(arr, 97.5)),
    }


def run_drift_monitor(
    spark: Any,
    backbone: Any,
    catalog: str,
    schema: str,
    reference_table: str = 'train_embeddings',
    inference_table: str = 'detector_inference_payload',
    drift_scores_table: str = 'drift_scores',
    k: int = 50,
    alert_threshold: float = 2.0,
    lookback_hours: int = 1,
) -> dict[str, Any]:
    """Production drift monitor.

    Pipeline:
      1. Read last hour of detector inference images (base64) from AI Gateway inference table.
      2. Re-embed via backbone (CLS-summary).
      3. Load reference embeddings from Delta and fit KNN reference.
      4. Compute KNN distance + MMD score.
      5. Compare to alert_threshold (multiplier vs reference self-distance).
      6. Append row to drift_scores Delta table.

    Returns: {'batch_id', 'knn_distance', 'mmd_score', 'n_images', 'alert', 'timestamp'}.
    """
    from src.drift.embeddings import compute_embeddings
    from src.drift.inference_table_reader import read_recent_inference_images

    images = read_recent_inference_images(
        spark,
        catalog,
        schema,
        inference_table,
        lookback_hours=lookback_hours,
    )
    if not images:
        logger.info("No inference images in lookback window; skipping drift compute.")
        return {
            'batch_id': str(uuid.uuid4()),
            'knn_distance': 0.0,
            'mmd_score': 0.0,
            'n_images': 0,
            'alert': False,
            'timestamp': datetime.now(UTC).isoformat(),
        }

    incoming = compute_embeddings(backbone, images)
    ref_df = spark.table(f"{catalog}.{schema}.{reference_table}").select("embedding").toPandas()
    ref_arr = np.stack(ref_df['embedding'].apply(np.asarray).to_list()).astype(np.float32)
    reference = fit_reference(ref_arr, method='knn', k=k)

    knn = score_drift(incoming, reference, method='knn')
    mmd = score_drift(incoming, reference, method='mmd')
    alert = knn > alert_threshold

    result: dict[str, Any] = {
        'batch_id': str(uuid.uuid4()),
        'knn_distance': float(knn),
        'mmd_score': float(mmd),
        'n_images': len(images),
        'alert': bool(alert),
        'timestamp': datetime.now(UTC).isoformat(),
    }
    # Append to Delta
    try:
        from pyspark.sql import Row

        df = spark.createDataFrame([Row(**result)])
        df.write.mode('append').saveAsTable(f"{catalog}.{schema}.{drift_scores_table}")
    except Exception as e:
        logger.error(f"Failed to write drift_scores: {e}")
    return result
