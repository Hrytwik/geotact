"""T17 vegetation-loss correction model: design weights, F1 cut-off, spatial folds, out-of-fold voting, scoring and the paired
bootstrap (synthetic data)."""
import numpy as np
import pytest

from src import vegetation_v2 as v2

W = {"loss": 0.1, "no_loss": 0.9}


def test_design_weights_are_stratum_weight_over_sample_size():
    w = v2.design_weights(np.array(["loss"] * 4 + ["no_loss"] * 6), W)
    assert np.allclose(w[:4], 0.025) and np.allclose(w[4:], 0.15) and np.isclose(w.sum(), 1.0)


def test_f1_cutoff_separates_a_clean_example_and_is_weighted():
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    y = np.array([1, 1, 1, 0, 0, 0])
    c = v2.f1_cutoff(p, y, np.ones(6))
    assert 0.2 <= c < 0.7 and ((p >= c) == y).all()
    # a heavy weight on the one negative scored 0.7 pushes the cut above it: losing 1 weighted positive beats a heavy false positive
    p2 = np.array([0.9, 0.7, 0.4, 0.1])
    y2 = np.array([1, 0, 1, 0])
    c2 = v2.f1_cutoff(p2, y2, np.array([1.0, 50.0, 1.0, 1.0]))
    assert c2 > 0.7


def test_spatial_folds_keep_a_block_together_and_use_every_fold():
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, 2000, 300), rng.integers(0, 2000, 300)
    f = v2.spatial_folds(rows, cols, k=5, seed=3, block_px=200)
    blocks = (rows // 200) * 1000 + cols // 200
    for b in np.unique(blocks):
        assert len(set(f[blocks == b])) == 1
    assert set(f) == set(range(5))
    assert not np.array_equal(f, v2.spatial_folds(rows, cols, k=5, seed=4, block_px=200))


def test_oof_votes_recover_a_planted_signal_and_do_not_invent_one_from_noise():
    rng = np.random.default_rng(1)
    n = 200
    strata = np.array(["loss"] * 100 + ["no_loss"] * 100)
    y = np.r_[rng.random(100) < 0.7, rng.random(100) < 0.1].astype(int)
    rows, cols = rng.integers(0, 3000, n), rng.integers(0, 3000, n)
    sig = np.c_[y + 0.3 * rng.standard_normal(n), rng.standard_normal(n)]
    noise = rng.standard_normal((n, 2))
    pred_sig = v2.oof_vote(sig, y, strata, W, rows, cols, reps=5, seed=0)
    pred_noise = v2.oof_vote(noise, y, strata, W, rows, cols, reps=5, seed=0)
    f_sig = v2.score_prediction(pred_sig, y, strata, W)["f1"]
    f_noise = v2.score_prediction(pred_noise, y, strata, W)["f1"]
    assert f_sig > 0.8 and f_noise < 0.6 and f_sig > f_noise + 0.3


def test_score_prediction_matches_the_estimators_and_returns_nan_when_nothing_is_predicted():
    strata = np.array(["loss"] * 4 + ["no_loss"] * 4)
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    pred = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s = v2.score_prediction(pred, y, strata, W)
    # user's accuracy: P(loss | predicted) = (0.1*0.5) / (0.1*0.75 + 0.9*0) = 2/3 ; producer's: 0.05 / (0.05 + 0.9*0.25)
    assert np.isclose(s["ua"], 0.05 / 0.075) and np.isclose(s["pa"], 0.05 / (0.05 + 0.225))
    assert np.isclose(s["f1"], 2 * s["ua"] * s["pa"] / (s["ua"] + s["pa"]))
    assert np.isnan(v2.score_prediction(np.zeros(8, int), y, strata, W)["f1"])


def test_paired_bootstrap_is_zero_for_identical_predictions_and_positive_for_a_better_one():
    rng = np.random.default_rng(2)
    strata = np.array(["loss"] * 60 + ["no_loss"] * 60)
    y = np.r_[(rng.random(60) < 0.6), (rng.random(60) < 0.15)].astype(int)
    good = y.copy()
    bad = np.zeros_like(y); bad[:60] = 1
    same = v2.paired_bootstrap(bad, bad, y, strata, W, n_boot=200, seed=0)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0
    better = v2.paired_bootstrap(good, bad, y, strata, W, n_boot=300, seed=0)
    assert better["diff"] > 0.2 and better["lo"] > 0


def test_neighbourhood_mean_ignores_nan_and_nonveg_probability_sums_the_right_bands():
    a = np.array([[1.0, np.nan, 3.0], [1.0, 1.0, 1.0], [np.nan, 1.0, 1.0]], "float32")
    m = v2.nan_mean3(a)
    assert np.isclose(m[1, 1], 9 / 7) and np.isclose(m[0, 0], 1.0) and np.isclose(m[0, 2], 5 / 3)   # NaN cells are left out of the mean
    probs = np.zeros((6, 1, 1), "uint8"); probs[0] = 20; probs[4] = 40; probs[5] = 20; probs[1] = 20      # percent: water, built, bare, tree
    assert np.isclose(v2.nonveg_probability(probs)[0, 0], 0.8)
