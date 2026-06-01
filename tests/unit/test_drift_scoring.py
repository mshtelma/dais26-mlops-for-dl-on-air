import numpy as np
import pytest

from dais26_dentex.drift.monitor import bootstrap_drift_ci, score_drift
from dais26_dentex.drift.reference import fit_reference


@pytest.fixture
def synth_ref():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (200, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x


def test_knn_identical_distribution_low(synth_ref):
    ref = fit_reference(synth_ref, method="knn", k=10)
    # Same distribution -> low drift
    rng = np.random.default_rng(1)
    same = rng.normal(0, 1, (50, 16)).astype(np.float32)
    same /= np.linalg.norm(same, axis=1, keepdims=True) + 1e-12
    score_same = score_drift(same, ref)
    # Shifted distribution -> high drift (moderate shift keeps directional spread)
    shifted = rng.normal(0.5, 1, (50, 16)).astype(np.float32)
    shifted /= np.linalg.norm(shifted, axis=1, keepdims=True) + 1e-12
    score_shifted = score_drift(shifted, ref)
    assert score_shifted > score_same, f"Shifted ({score_shifted}) should exceed same ({score_same})"


def test_mmd_identical_vs_shifted(synth_ref):
    ref = fit_reference(synth_ref, method="mmd")
    rng = np.random.default_rng(2)
    same = rng.normal(0, 1, (50, 16)).astype(np.float32)
    same /= np.linalg.norm(same, axis=1, keepdims=True) + 1e-12
    shifted = rng.normal(2, 1, (50, 16)).astype(np.float32)
    shifted /= np.linalg.norm(shifted, axis=1, keepdims=True) + 1e-12
    s_same = score_drift(same, ref)
    s_shift = score_drift(shifted, ref)
    assert s_shift > s_same


def test_energy_identical_vs_shifted(synth_ref):
    ref = fit_reference(synth_ref, method="energy")
    rng = np.random.default_rng(3)
    same = rng.normal(0, 1, (50, 16)).astype(np.float32)
    same /= np.linalg.norm(same, axis=1, keepdims=True) + 1e-12
    shifted = rng.normal(2, 1, (50, 16)).astype(np.float32)
    shifted /= np.linalg.norm(shifted, axis=1, keepdims=True) + 1e-12
    assert score_drift(shifted, ref, method="energy") > score_drift(same, ref, method="energy")


def test_bootstrap_ci_shape(synth_ref):
    ref = fit_reference(synth_ref, method="knn", k=5)
    incoming = synth_ref[:30] + 0.1
    incoming /= np.linalg.norm(incoming, axis=1, keepdims=True) + 1e-12
    ci = bootstrap_drift_ci(incoming, ref, n_iterations=50)
    assert {"mean", "p2_5", "p97_5"} <= set(ci.keys())
    assert ci["p2_5"] <= ci["mean"] <= ci["p97_5"]


def test_fit_reference_empty_raises():
    with pytest.raises(ValueError):
        fit_reference(np.zeros((0, 16)))


def test_score_drift_empty_raises(synth_ref):
    ref = fit_reference(synth_ref, method="knn", k=5)
    with pytest.raises(ValueError):
        score_drift(np.zeros((0, 16)), ref)
