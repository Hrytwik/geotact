"""T6g baselines: features, single-layer scores and the spatial CV wrapper (synthetic data; no torch)."""
import numpy as np
import pytest

from src import flood_cnn_baselines as bl

NAMES = ["slope_deg", "hand", "twi"]


def test_local_means_ignore_nan_and_match_a_hand_computed_window():
    a = np.arange(100, dtype="float32").reshape(10, 10)
    a[5, 5] = np.nan
    m = bl.local_means(a[None], size=3)[0]
    win = a[0:3, 0:3]                                                     # reflect padding is irrelevant at (1, 1)
    assert m[1, 1] == pytest.approx(np.nanmean(win))
    assert m[5, 5] == pytest.approx(np.nanmean(a[4:7, 4:7]))              # the NaN cell itself still gets its neighbours' mean
    assert np.isfinite(m).all()


def test_local_means_is_nan_where_the_whole_window_is_nan():
    a = np.full((1, 9, 9), np.nan, dtype="float32")
    assert np.isnan(bl.local_means(a, size=3)).all()


def test_features_are_centre_values_then_window_means_with_matching_names():
    stack = np.random.default_rng(0).normal(size=(3, 20, 20)).astype("float32")
    rows, cols = np.array([4, 10]), np.array([5, 12])
    X, names = bl.features(stack, rows, cols, NAMES, size=3)
    assert X.shape == (2, 6)
    assert names == NAMES + [f"{n}_m3" for n in NAMES]
    assert np.allclose(X[:, :3], stack[:, rows, cols].T)
    assert X[0, 3 + 1] == pytest.approx(stack[1, 3:6, 4:7].mean(), abs=1e-5)


def test_single_layer_scores_flip_sign_so_that_higher_means_more_flood_prone():
    stack = np.zeros((3, 4, 4), dtype="float32")
    stack[1] = [[0, 1, 2, 3]] * 4                                         # "hand": low = flood-prone
    stack[2] = [[0, 1, 2, 3]] * 4                                         # "twi": high = flood-prone
    rows, cols = np.array([0, 0]), np.array([0, 3])
    s = bl.single_layer_scores(stack, rows, cols, NAMES, {"hand": -1, "twi": +1})
    assert s["hand"][0] > s["hand"][1]
    assert s["twi"][0] < s["twi"][1]


def test_single_layer_scores_fill_nan_with_the_median_so_evaluation_can_run():
    stack = np.zeros((1, 4, 4), dtype="float32")
    stack[0] = np.arange(16).reshape(4, 4)
    stack[0, 0, 0] = np.nan
    s = bl.single_layer_scores(stack, np.array([0, 1, 2]), np.array([0, 1, 2]), ["hand"], {"hand": -1})
    assert np.isfinite(s["hand"]).all()


def test_cv_scores_every_sample_exactly_once_out_of_fold_and_learns_a_clean_signal():
    rng = np.random.default_rng(1)
    n = 600
    y = (rng.random(n) < 0.2).astype(int)
    X = rng.normal(size=(n, 4))
    X[:, 0] += 2.0 * y
    folds = rng.integers(0, 5, size=n)
    oof = bl.cv_scores(X, y, folds, bl.logistic_factory())
    assert np.isfinite(oof).all()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, oof) > 0.85


def test_logistic_factory_survives_nan_features():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    X[::7, 1] = np.nan
    model = bl.logistic_factory()()
    model.fit(X, y)
    assert np.isfinite(model.predict_proba(X)[:, 1]).all()


def test_cv_never_trains_on_the_held_out_fold():
    class Spy:
        seen = []

        def fit(self, X, y):
            Spy.seen.append(set(np.round(X[:, 0]).astype(int)))
            return self

        def predict_proba(self, X):
            return np.column_stack([1 - np.zeros(len(X)), np.zeros(len(X))])

    X = np.arange(50, dtype=float)[:, None]
    folds = np.arange(50) % 5
    bl.cv_scores(X, np.zeros(50, int), folds, Spy)
    for k, seen in enumerate(Spy.seen):
        assert not any(i % 5 == k for i in seen)
