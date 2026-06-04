from pathlib import Path

import torch
import torch.nn as nn


class _FakeBackbone(nn.Module):
    def __init__(self, summary_dim: int = 1152, spatial_dim: int = 1152):
        super().__init__()
        self.summary_dim = summary_dim
        self.spatial_dim = spatial_dim

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        ph, pw = h // 16, w // 16
        return torch.randn(b, self.summary_dim), torch.randn(b, ph * pw, self.spatial_dim)


def test_train_no_data_runs_config_only(monkeypatch, tmp_path: Path):
    """When volume_path is None, the loop should still set up backbone/model and log to MLflow
    (but skip dataloaders)."""
    from dais26_dentex.models.backbones import BackboneInfo

    def fake_load(name, revision=None, cache_dir=None, device="cpu", **kwargs):
        return _FakeBackbone(), BackboneInfo(
            name=name,
            summary_dim=1152,
            spatial_dim=1152,
            patch_size=16,
            model_name="fake",
            revision=revision,
        )

    monkeypatch.setattr("dais26_dentex.models.backbones.load_backbone", fake_load)

    # Local MLflow tracking
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    monkeypatch.setattr("mlflow.set_registry_uri", lambda uri: None)

    # Stub log_model + alias steps so we don't need UC
    import mlflow.pyfunc as pyf

    monkeypatch.setattr(pyf, "log_model", lambda *a, **kw: None)
    import mlflow.tracking as tracking

    class FakeClient:
        def __init__(self, registry_uri=None):
            pass

        def search_model_versions(self, q):
            return []

        def set_registered_model_alias(self, **kw):
            pass

    monkeypatch.setattr(tracking, "MlflowClient", FakeClient)

    from dais26_dentex.train.train_detector import train_detector

    run_id = train_detector(
        catalog="dev_cat",
        schema="dev_schema",
        volume_path=None,  # no data
        epochs=1,
        batch_size=1,
        num_workers=0,
        register_model=False,
        set_candidate_alias=False,
    )
    assert isinstance(run_id, str) and len(run_id) > 0


def test_precompute_image_loader(tmp_path: Path):
    """The image loader should produce a normalized (3, S, S) tensor."""
    from PIL import Image

    from dais26_dentex.train.precompute_embeddings import _load_image_tensor

    p = tmp_path / "x.png"
    Image.new("RGB", (50, 80), (200, 100, 50)).save(p)
    t = _load_image_tensor(str(p), size=64)
    assert t.shape == (3, 64, 64)
    assert t.dtype == torch.float32
