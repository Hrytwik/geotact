"""T13 land-use accuracy push: embedding de-quantisation, spectral indices, nested selection (never sees the held-out fold, picks the
informative candidate), and the active-learning point picker (synthetic data, no torch)."""
import numpy as np
import pytest

from src import landuse_push as lp


def test_dequantise_follows_the_published_mapping_and_masks_no_data():
    v = np.array([-128, -127, -64, 0, 64, 127], dtype=np.int8)
    d = lp.dequantise(v)
    assert np.isnan(d[0])
    assert d[1:] == pytest.approx([-(127 / 127.5) ** 2, -(64 / 127.5) ** 2, 0.0, (64 / 127.5) ** 2, (127 / 127.5) ** 2])


def test_spectral_indices_use_the_right_bands():
    # band order B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
    b = np.array([0.05, 0.08, 0.06, 0.1, 0.2, 0.25, 0.30, 0.31, 0.15, 0.10], dtype="float32")[:, None]
    nd, mndwi, ndbi = lp.indices(b)[:, 0]
    assert nd == pytest.approx((0.30 - 0.06) / (0.30 + 0.06))
    assert mndwi == pytest.approx((0.08 - 0.15) / (0.08 + 0.15))
    assert ndbi == pytest.approx((0.15 - 0.30) / (0.15 + 0.30))


def toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    good = rng.normal(size=(n, 4)); good[np.arange(n), y] += 2.5
    noise = rng.normal(size=(n, 4))
    strata = rng.integers(0, 3, n)
    W = {0: 0.5, 1: 0.3, 2: 0.2}
    folds = np.arange(n) % 5
    return y, {"good": good, "noise": noise}, strata, W, folds


def test_nested_selection_picks_the_informative_candidate_and_never_fits_on_the_held_out_fold():
    y, sets, strata, W, folds = toy()
    seen = []

    def spy_logistic(C):
        m = lp.make_learner("logistic", dict(C=C))

        class Spy:
            def fit(self, X, yy):
                seen.append(len(yy)); m.fit(X, yy); return self

            def predict(self, X):
                return m.predict(X)
        return Spy()

    cands = [("good", "logistic"), ("noise", "logistic")]
    res = lp.nested_select(sets, cands, y, strata, W, folds, grids={"logistic": [dict(C=1.0)]}, make=lambda name, p: spy_logistic(p["C"]))
    assert all(c[0] == "good" for c in res["chosen"])
    assert lp.stratified(strata, res["oof"] == y, W) > 0.8
    assert max(seen) <= (folds != 0).sum()                                                   # no fit ever saw all 300 points


def test_per_candidate_scores_are_reported_for_every_candidate():
    y, sets, strata, W, folds = toy(seed=1)
    res = lp.nested_select(sets, [("good", "logistic"), ("noise", "logistic")], y, strata, W, folds, grids={"logistic": [dict(C=0.1), dict(C=1.0)]})
    assert set(res["per_candidate"]) == {"good|logistic", "noise|logistic"}
    assert res["per_candidate"]["good|logistic"]["accuracy"] > res["per_candidate"]["noise|logistic"]["accuracy"] + 0.3


def test_uncertain_points_are_spread_out_and_keep_away_from_the_accuracy_sample():
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, 400, 3000), rng.integers(0, 400, 3000)
    margin = rng.random(3000)
    avoid = np.array([[200, 200]])
    pick = lp.pick_uncertain(rows, cols, margin, n=20, min_px=20, avoid=avoid)
    assert len(pick) == 20
    P = np.column_stack([rows[pick], cols[pick]])
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]) + np.eye(20) * 1e9
    assert d.min() >= 20 and np.hypot(P[:, 0] - 200, P[:, 1] - 200).min() >= 20
    assert np.median(margin[pick]) < np.median(margin)                                      # the least sure come first


def test_margin_is_the_gap_between_the_two_most_probable_classes():
    P = np.array([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2], [1.0, 0.0, 0.0]])
    assert lp.margin_from_proba(P) == pytest.approx([0.5, 0.0, 1.0])
