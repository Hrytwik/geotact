"""T6d: susceptibility model machinery (spatial folds, label aggregation, cross-validation, single-feature baselines).
Unit tests use logistic regression: LightGBM and PyTorch must never share a process (libomp), and pytest loads torch."""
import numpy as np
import pytest
from affine import Affine
from sklearn.linear_model import LogisticRegression

from src import flood_model as fm


# ---------------------------------------------------------------- spatial folds
def test_block_folds_keep_each_block_in_one_fold_and_are_balanced():
    folds = fm.block_folds((200, 160), block=40, k=5, seed=0)
    assert folds.shape == (200, 160) and set(np.unique(folds)) == {0, 1, 2, 3, 4}
    for r in range(0, 200, 40):
        for c in range(0, 160, 40):
            assert len(np.unique(folds[r:r + 40, c:c + 40])) == 1                     # a block never straddles two folds
    counts = np.bincount(folds.ravel())
    assert counts.max() / counts.min() < 2.0
    assert np.array_equal(folds, fm.block_folds((200, 160), 40, 5, 0))                # deterministic
    assert not np.array_equal(folds, fm.block_folds((200, 160), 40, 5, 1))


# ---------------------------------------------------------------- aggregation to the 30 m grid
def test_to_grid_averages_ten_metre_pixels_into_thirty_metre_cells():
    src_tf = Affine(10, 0, 0, 0, -10, 60)
    dst_tf = Affine(30, 0, 0, 0, -30, 60)
    a = np.zeros((6, 6), "float32")
    a[0:3, 0:3] = 1.0                                   # fills cell (0, 0)
    a[3:6, 3:6] = np.tile([1, 0, 0], (3, 1))            # a third of cell (1, 1)
    out = fm.to_grid(a, src_tf, dst_tf, (2, 2), "EPSG:32645")
    assert out == pytest.approx(np.array([[1.0, 0.0], [0.0, 1 / 3]]), abs=1e-5)


def test_cell_labels_follow_the_registered_rule():
    """Rule: eligible if >= 50 percent of the cell's 10 m pixels are open; positive if >= 30 percent of those open
    pixels have F >= 0.12 (pos_frac is the share of ALL pixels that are open and positive)."""
    open_frac = np.array([[1.0, 0.4], [0.9, 0.5]])
    pos_frac = np.array([[0.40, 0.40], [0.10, 0.20]])
    y, eligible = fm.cell_labels(open_frac, pos_frac, min_open=0.5, min_pos=0.30)
    assert eligible.tolist() == [[True, False], [True, True]]
    assert y.tolist() == [[1, 0], [0, 1]]             # 0.40/1.0 yes; ineligible -> 0; 0.10/0.9 = 0.11 no; 0.20/0.5 = 0.40 yes


# ---------------------------------------------------------------- cross-validation
def _toy(seed=0, n=4000, signal=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 1.0).astype(int) if signal else (rng.random(n) < 0.1).astype(int)
    folds = rng.integers(0, 5, size=n)
    return X, y, folds


def test_spatial_cv_finds_a_real_signal_and_none_in_noise():
    make = lambda: LogisticRegression(max_iter=200)
    X, y, f = _toy(signal=True)
    res = fm.spatial_cv(X, y, f, make)
    assert res["auc"] > 0.85 and len(res["fold_auc"]) == 5
    X, y, f = _toy(signal=False)
    assert abs(fm.spatial_cv(X, y, f, make)["auc"] - 0.5) < 0.06                     # labels unrelated to features: no leakage


def test_out_of_fold_predictions_never_come_from_a_model_that_saw_the_cell():
    seen = []

    class Spy(LogisticRegression):
        def fit(self, X, y, **kw):
            seen.append(len(y)); return super().fit(X, y, **kw)
    X, y, f = _toy()
    res = fm.spatial_cv(X, y, f, lambda: Spy(max_iter=100))
    assert all(s == (f != k).sum() for s, k in zip(seen, range(5)))                  # each model trained on the other four folds
    assert np.isfinite(res["oof"]).all()


def test_single_feature_baselines_use_the_physical_direction():
    rng = np.random.default_rng(0)
    hand = rng.uniform(0, 20, 5000); twi = rng.uniform(4, 16, 5000)
    y = (rng.random(5000) < 1 / (1 + np.exp((hand - 4) / 2))).astype(int)            # flooding likelier at LOW height above drainage
    X = np.column_stack([hand, twi]); names = ["hand", "twi"]
    auc = fm.single_feature_aucs(X, y, names, {"hand": -1, "twi": +1})
    assert auc["hand"] > 0.8 and abs(auc["twi"] - 0.5) < 0.06


def test_auc_within_groups_is_reported_separately():
    X, y, f = _toy()
    g = (np.arange(len(y)) % 2).astype(bool)
    oof = fm.spatial_cv(X, y, f, lambda: LogisticRegression(max_iter=200))["oof"]
    out = fm.auc_within(oof, y, {"even": ~g, "odd": g})
    assert set(out) == {"even", "odd"} and all(0.8 < v < 1.0 for v in out.values())
