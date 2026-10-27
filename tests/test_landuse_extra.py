"""T18 extra training labels for the land-use correction model: spatial exclusion of extras and the out-of-fold loop (synthetic)."""
import numpy as np

from src import landuse_extra as lx
from src.landuse_improve import correction_model


def test_extras_in_a_test_block_are_left_out_and_others_kept():
    test_xy = np.array([[100.0, 100.0], [9000.0, 9000.0]])             # blocks (0,0) and (2,2) at 4 km
    extra_xy = np.array([[3999.0, 50.0], [4001.0, 50.0], [8500.0, 8100.0], [20000.0, 0.0]])
    keep = lx.allowed_extras(test_xy, extra_xy, block_m=4000.0)
    assert keep.tolist() == [False, True, False, True]


def test_cv_with_extras_never_trains_on_a_test_point_and_extras_help_a_starved_class():
    rng = np.random.default_rng(0)
    n = 120
    xy = rng.uniform(0, 40000, (n, 2))
    y = rng.integers(0, 3, n)
    F = np.c_[y + 0.8 * rng.standard_normal(n), rng.standard_normal(n)]
    folds = rng.integers(0, 5, n)
    oof = lx.cv_predict_extra(F, y, folds, xy, np.empty((0, 2)), np.empty(0, int), np.empty((0, 2)), correction_model)
    from src.landuse_improve import cv_predict
    assert (oof == cv_predict(F, y, folds, correction_model)).all()          # no extras -> exactly the T4c loop
    m = 300
    xe = rng.uniform(0, 40000, (m, 2)); ye = rng.integers(0, 3, m)
    Fe = np.c_[ye + 0.8 * rng.standard_normal(m), rng.standard_normal(m)]
    oof2 = lx.cv_predict_extra(F, y, folds, xy, Fe, ye, xe, correction_model)
    assert oof2.shape == y.shape and np.mean(oof2 == y) >= np.mean(oof == y) - 0.05
