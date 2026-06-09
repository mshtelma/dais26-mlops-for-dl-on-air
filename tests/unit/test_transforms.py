"""Unit tests for src/dais26_dentex/data/transforms.py."""

from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# get_train_transforms
# ---------------------------------------------------------------------------


def test_train_transforms_output_shape():
    from dais26_dentex.data.transforms import get_train_transforms

    tfm = get_train_transforms(img_size=1024)
    img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    bboxes = [[10, 10, 200, 200], [300, 300, 100, 100]]
    class_labels = [0, 1]

    out = tfm(image=img, bboxes=bboxes, class_labels=class_labels)

    assert out["image"].shape == (3, 1024, 1024)


def test_train_transforms_output_dtype():
    from dais26_dentex.data.transforms import get_train_transforms

    tfm = get_train_transforms(img_size=1024)
    img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    bboxes = [[10, 10, 200, 200], [300, 300, 100, 100]]
    class_labels = [0, 2]

    out = tfm(image=img, bboxes=bboxes, class_labels=class_labels)

    assert out["image"].dtype == torch.float32


def test_train_transforms_bboxes_nonempty():
    from dais26_dentex.data.transforms import get_train_transforms

    # Use a large bbox so it is not dropped by min_visibility filter
    tfm = get_train_transforms(img_size=1024)
    img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    bboxes = [[10, 10, 500, 500]]
    class_labels = [3]

    out = tfm(image=img, bboxes=bboxes, class_labels=class_labels)

    assert len(out["bboxes"]) > 0


# ---------------------------------------------------------------------------
# get_val_transforms
# ---------------------------------------------------------------------------


def test_val_transforms_output_shape():
    from dais26_dentex.data.transforms import get_val_transforms

    tfm = get_val_transforms(img_size=1024)
    img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    bboxes = [[50, 50, 300, 300]]
    class_labels = [1]

    out = tfm(image=img, bboxes=bboxes, class_labels=class_labels)

    assert out["image"].shape == (3, 1024, 1024)
    assert out["image"].dtype == torch.float32


def test_val_transforms_deterministic():
    """Val transforms have no randomness - same input gives same output."""
    from dais26_dentex.data.transforms import get_val_transforms

    tfm = get_val_transforms(img_size=256)
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    bboxes = [[10, 10, 50, 50]]
    class_labels = [0]

    out1 = tfm(image=img, bboxes=bboxes, class_labels=class_labels)
    out2 = tfm(image=img, bboxes=bboxes, class_labels=class_labels)

    assert torch.allclose(out1["image"], out2["image"])


# ---------------------------------------------------------------------------
# get_train_transforms - smaller img_size for speed
# ---------------------------------------------------------------------------


def test_train_transforms_resize():
    from dais26_dentex.data.transforms import get_train_transforms

    tfm = get_train_transforms(img_size=64)
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    bboxes = [[5, 5, 50, 50]]
    class_labels = [2]

    out = tfm(image=img, bboxes=bboxes, class_labels=class_labels)

    assert out["image"].shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# augmentation knobs (push-to-0.60): multi-scale + rotation must keep the
# output a fixed `img_size` square so batching is unaffected.
# ---------------------------------------------------------------------------


def test_train_transforms_multiscale_keeps_canvas_shape():
    from dais26_dentex.data.transforms import get_train_transforms

    tfm = get_train_transforms(img_size=128, multiscale_range=[0.5, 1.0])
    img = np.random.randint(0, 256, (256, 200, 3), dtype=np.uint8)
    bboxes = [[10, 10, 80, 80]]
    class_labels = [1]
    for _ in range(8):  # several draws to exercise the random scale
        out = tfm(image=img, bboxes=bboxes, class_labels=class_labels)
        assert out["image"].shape == (3, 128, 128)


def test_train_transforms_rotation_keeps_canvas_shape():
    from dais26_dentex.data.transforms import get_train_transforms

    tfm = get_train_transforms(img_size=128, rotation_deg=10.0)
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    bboxes = [[20, 20, 60, 60]]
    class_labels = [0]
    out = tfm(image=img, bboxes=bboxes, class_labels=class_labels)
    assert out["image"].shape == (3, 128, 128)


def test_train_transforms_jitter_scale_zero_disables_jitter():
    """jitter_scale=0 -> no ColorJitter; with hflip disabled and no rotation the
    pipeline is deterministic geometry + normalize, so two calls match."""
    from dais26_dentex.data.transforms import get_train_transforms

    tfm = get_train_transforms(img_size=64, hflip_prob=0.0, jitter_scale=0.0)
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    bboxes = [[5, 5, 40, 40]]
    class_labels = [2]
    out1 = tfm(image=img, bboxes=bboxes, class_labels=class_labels)
    out2 = tfm(image=img, bboxes=bboxes, class_labels=class_labels)
    assert torch.allclose(out1["image"], out2["image"])
