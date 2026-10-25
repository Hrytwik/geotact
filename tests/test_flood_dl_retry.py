"""T6k: flood deep-learning retry building blocks (torch; small synthetic data)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from affine import Affine

from src import flood_dl_retry as fr


def test_30m_cells_map_to_the_10m_pixel_at_their_centre():
    tf30 = Affine(30, 0, 1000, 0, -30, 5000); tf10 = Affine(10, 0, 1000, 0, -10, 5000)
    r10, c10 = fr.cells_30m_to_10m(np.array([0, 2]), np.array([0, 1]), tf30, tf10)
    assert r10.tolist() == [1, 7] and c10.tolist() == [1, 4]                              # centre of a 3 x 3 block


class TinyEncoder(torch.nn.Module):
    """Stand-in for the U-Net encoder: returns a list of feature maps, the last with 4 channels."""

    def __init__(self, c_in=10):
        super().__init__()
        self.conv = torch.nn.Conv2d(c_in, 4, 3, stride=4, padding=1)

    def forward(self, x):
        return [x, self.conv(x)]


def test_encoder_features_have_one_row_per_sample_and_work_at_the_image_edge():
    X = np.random.default_rng(0).normal(size=(10, 50, 60)).astype("float32")
    F = fr.encoder_features(TinyEncoder(), X, np.array([0, 25, 49]), np.array([0, 30, 59]), size=16, batch=2)
    assert F.shape == (3, 4) and np.isfinite(F).all()


def test_extra_positives_from_the_held_out_fold_are_never_used_in_training():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 2)); y = (rng.random(100) < 0.3).astype(int); folds = np.arange(100) % 5
    xe = rng.normal(size=(20, 2)) + 100; fe = np.arange(20) % 5                            # extra positives, far away, tagged by fold
    seen = []

    def train(X, Y, seed=0):
        seen.append(X)
        return lambda Z: np.zeros(len(Z))

    fr.cv_with_extra(x, y, folds, xe, fe, train, predict=lambda m, Z: m(Z), seeds=(0,))
    for k, X in zip(range(5), seen):
        extra_used = X[X[:, 0] > 50]
        held = xe[fe == k]
        assert not any(np.allclose(r, h) for r in extra_used for h in held)
        assert len(extra_used) == (fe != k).sum()


def test_subsampling_keeps_every_negative_and_the_asked_share_of_positives():
    y = np.array([1] * 40 + [0] * 60)
    idx = fr.subsample_positives(y, np.arange(100), 0.25, np.random.default_rng(0))
    assert (y[idx] == 0).sum() == 60 and (y[idx] == 1).sum() == 10


def test_cnn_estimator_crops_the_centre_for_smaller_patches():
    x = np.zeros((3, 2, 33, 33), "float32"); x[:, :, 16, 16] = 1.0
    assert fr.crop_centre(x, 17).shape == (3, 2, 17, 17) and fr.crop_centre(x, 17)[0, 0, 8, 8] == 1.0
    assert fr.crop_centre(x, 33) is x
