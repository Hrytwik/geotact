"""T6g: evaluation shared by the CNN and the baselines (pooled out-of-fold AUC / AP with bootstrap intervals)."""
import numpy as np
import pytest

from src import flood_cnn_eval as ev


def toy(n_pos=100, n_bg=2000, n_nf=25, seed=0, shift=1.5):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n_pos), np.zeros(n_bg + n_nf)].astype(int)
    kind = np.array(["flood"] * n_pos + ["background"] * n_bg + ["nonflood"] * n_nf)
    score = rng.normal(size=len(y)) + shift * y
    return score, y, kind


def test_pooled_auc_matches_the_closed_form_for_two_gaussians():
    score, y, kind = toy(n_pos=3000, n_bg=3000, n_nf=0, shift=1.5)
    r = ev.evaluate(score, y, kind, n_boot=0)
    assert r["auc"] == pytest.approx(0.855, abs=0.015)                    # Phi(1.5 / sqrt 2) = 0.855


def test_nonflood_auc_uses_only_the_field_non_flood_points_as_negatives():
    score, y, kind = toy(n_pos=200, n_bg=500, n_nf=60, shift=0.0)
    score[kind == "nonflood"] -= 5.0                                      # field non-flood points are easy negatives
    r = ev.evaluate(score, y, kind, n_boot=0)
    assert r["auc_nonflood"] > 0.99 and abs(r["auc"] - 0.5) < 0.08        # pooled with background it stays near chance


def test_bootstrap_interval_brackets_the_estimate_and_narrows_with_more_data():
    s1, y1, k1 = toy(n_pos=50, n_bg=500, n_nf=10)
    s2, y2, k2 = toy(n_pos=800, n_bg=800, n_nf=10)
    a, b = ev.evaluate(s1, y1, k1, n_boot=400, seed=0), ev.evaluate(s2, y2, k2, n_boot=400, seed=0)
    for r in (a, b):
        assert r["auc_ci"][0] < r["auc"] < r["auc_ci"][1]
    assert (b["auc_ci"][1] - b["auc_ci"][0]) < (a["auc_ci"][1] - a["auc_ci"][0])


def test_no_signal_gives_an_interval_containing_one_half():
    s, y, k = toy(shift=0.0, seed=3)
    r = ev.evaluate(s, y, k, n_boot=500, seed=0)
    assert r["auc_ci"][0] < 0.5 < r["auc_ci"][1]


def test_nan_scores_are_rejected_not_silently_dropped():
    s, y, k = toy()
    s[3] = np.nan
    with pytest.raises(ValueError):
        ev.evaluate(s, y, k)


def test_average_precision_reflects_the_low_prevalence():
    s, y, k = toy(n_pos=100, n_bg=9900, n_nf=0, shift=0.0)
    r = ev.evaluate(s, y, k, n_boot=0)
    assert r["ap"] == pytest.approx(100 / 10000, abs=0.01)                # chance AP equals prevalence
