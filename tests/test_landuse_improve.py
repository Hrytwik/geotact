"""T4c: probability prediction, ensemble, spatial folds for points, correction-model cross-validation (small synthetic data)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from src import landuse as lu
from src import landuse_improve as li


class Fixed(torch.nn.Module):
    """1x1 'model' with fixed weights so its output is known."""

    def __init__(self, c_in=3, k=6):
        super().__init__()
        self.conv = torch.nn.Conv2d(c_in, k, 1)
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(k, c_in, 1, 1, generator=g)); self.conv.bias.zero_()

    def forward(self, x):
        return self.conv(x)


def test_probabilities_sum_to_one_and_their_argmax_is_the_class_map():
    X = np.random.default_rng(0).normal(size=(3, 70, 90)).astype("float32")
    m = Fixed()
    P = lu.predict_probs(m, X, tile=64, margin=8)
    assert P.shape == (6, 70, 90)
    assert np.allclose(P.astype("float32").sum(axis=0), 1.0, atol=1e-2)
    assert np.array_equal(P.argmax(axis=0), lu.predict_map(m, X, tile=64, margin=8))


def test_ensemble_averages_probabilities_before_taking_the_class():
    a = np.zeros((6, 1, 2), "float32"); b = np.zeros((6, 1, 2), "float32")
    a[0, 0, 0], a[1, 0, 0] = 0.6, 0.4; b[0, 0, 0], b[1, 0, 0] = 0.0, 1.0              # mean: 0.3 vs 0.7 -> class 1
    a[2, 0, 1] = 1.0; b[2, 0, 1] = 1.0
    cls, mean = li.ensemble([a, b])
    assert cls.tolist() == [[1, 2]] and mean[1, 0, 0] == pytest.approx(0.7)


def test_point_folds_keep_points_of_the_same_block_together():
    x = np.array([100.0, 900.0, 5100.0, 5200.0, 12000.0]); y = np.array([100.0, 300.0, 100.0, 3000.0, 100.0])
    f = li.point_folds(x, y, block_m=4000, k=5, seed=0)
    assert f[0] == f[1] and f[2] == f[3] and len(f) == 5


def test_cross_validated_correction_never_trains_on_the_held_out_block_and_learns_a_clean_signal():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 3, n)
    F = rng.normal(size=(n, 5)); F[np.arange(n), y] += 3.0
    folds = rng.integers(0, 5, n)
    seen = []

    class Spy:
        def __init__(self):
            self.m = li.correction_model()

        def fit(self, X, yy):
            seen.append(set(np.round(X[:, -1]).astype(int)))
            self.m.fit(X[:, :-1], yy)
            return self

        def predict(self, X):
            return self.m.predict(X[:, :-1])

    Fi = np.column_stack([F, np.arange(n)])
    oof = li.cv_predict(Fi, y, folds, Spy)
    for k, s in zip(np.unique(folds), seen):
        assert not (s & set(np.flatnonzero(folds == k)))
    assert (oof == y).mean() > 0.9


def test_stacking_inputs_appends_ndvi_seasons_and_fills_nan_with_the_season_mean():
    X = np.ones((10, 4, 4), "float32"); nd = np.full((8, 4, 4), 0.5, "float32"); nd[0, 0, 0] = np.nan; nd[0, 1, 1] = 0.8
    Z = li.stack_inputs(X, nd)
    assert Z.shape == (18, 4, 4) and np.isfinite(Z[10:]).all()
    assert Z[10, 0, 0] == pytest.approx(np.nanmean(nd[0]))                              # 0 is a real NDVI value, so not 0


def test_features_are_identical_in_layout_for_points_and_pixels():
    probs = np.array([[0.1, 0.5], [0.9, 0.5]] + [[0.0, 0.0]] * 4, "float32")             # 6 classes x 2 samples
    esri = np.array([1, 4]); wc = np.array([1, 255]); nd = np.array([[0.3, 0.4]] * 8, "float32"); built = np.array([0.0, 1.0])
    F = li.features_from(probs, esri, wc, nd, built)
    assert F.shape == (2, 6 + 6 + 6 + 8 + 1)
    assert F[0, 6 + 1] == 1 and F[1, 6 + 4] == 1 and F[1, 12:18].sum() == 0                # one-hot; unusable WorldCover -> all zero
    assert F[1, -1] == 1.0
