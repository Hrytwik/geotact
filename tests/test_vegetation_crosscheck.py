"""T5b: U-Net tree loss against NDVI change (synthetic arrays)."""
import numpy as np
import pandas as pd
import pytest

from src import vegetation_crosscheck as vc

TREE, BUILT, CROP = 1, 4, 2


def maps():
    """Six pixels in a row: converted / seed-disagree / stable tree / stable built / tree->crop / built->tree."""
    a23s0 = np.array([[TREE, TREE, TREE, BUILT, TREE, BUILT]])
    a23s1 = np.array([[TREE, BUILT, TREE, BUILT, TREE, BUILT]])
    a26s0 = np.array([[BUILT, BUILT, TREE, BUILT, CROP, TREE]])
    a26s1 = np.array([[BUILT, BUILT, TREE, BUILT, CROP, TREE]])
    return a23s0, a23s1, a26s0, a26s1


def test_conversion_masks_apply_the_both_seed_rule():
    study = np.ones((1, 6), bool)
    m = vc.conversion_masks(*maps(), study)
    assert list(np.flatnonzero(m["converted"])) == [0]
    assert list(np.flatnonzero(m["stable_tree"])) == [2]
    assert list(np.flatnonzero(m["stable_built"])) == [3]
    assert list(np.flatnonzero(m["tree_to_other"])) == [4]
    assert list(np.flatnonzero(m["built_to_tree"])) == [5]
    assert list(np.flatnonzero(m["converted_seed0"])) == [0, 1]                          # seed 0 alone calls pixel 1 a conversion
    assert list(np.flatnonzero(m["converted_seed1"])) == [0]                              # seed 1 calls pixel 1 built-up already in 2023
    assert list(np.flatnonzero(m["disagree_2023"])) == [1]                              # tree in one seed, built in the other
    assert not m["disagree_2026"].any()


def test_masks_respect_the_study_area():
    study = np.ones((1, 6), bool); study[0, 0] = False
    assert not vc.conversion_masks(*maps(), study)["converted"].any()


def test_ndvi_change_is_late_pair_minus_early_pair():
    stack = np.zeros((8, 1, 1))
    stack[3], stack[4], stack[6], stack[7] = 0.5, 0.7, 0.2, 0.4
    assert vc.ndvi_change(stack)[0, 0] == pytest.approx(0.3 - 0.6)


def test_prob_less_matches_a_hand_count_with_ties_counted_half():
    assert vc.prob_less(np.array([1.0, 2.0]), np.array([2.0, 3.0])) == pytest.approx((1 + 1 + 0.5 + 1) / 4)   # (1<2,1<3,2=2,2<3)
    x = np.random.default_rng(0).normal(size=500); y = np.random.default_rng(1).normal(size=500)
    assert vc.prob_less(x, y) == pytest.approx(0.5, abs=0.06)


def test_block_bootstrap_interval_contains_the_point_and_detects_a_clear_shift():
    rng = np.random.default_rng(0)
    xb = rng.integers(0, 20, 600); yb = rng.integers(0, 20, 3000)
    x = rng.normal(-0.1, 0.05, 600); y = rng.normal(0.0, 0.05, 3000)
    est, (lo, hi) = vc.block_bootstrap_prob_less(x, xb, y, yb, n_boot=200, seed=0)
    assert lo <= est <= hi and lo > 0.5 and est > 0.9
    same_x = rng.normal(0, 0.05, 600)
    est2, (lo2, hi2) = vc.block_bootstrap_prob_less(same_x, xb, y, yb, n_boot=200, seed=0)
    assert lo2 < 0.5 < hi2


def test_ward_table_computes_loss_fraction_and_mean_change_over_2023_trees():
    wards = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
    tree23 = np.ones((2, 4), bool)
    conv = np.array([[True, False, False, False], [False, False, False, False]])
    change = np.array([[-0.2, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 0.1]])
    t = vc.ward_table(wards, tree23, conv, change, px_ha=1.0, min_ha=3.0)
    r1, r2 = t.set_index("ward").loc[1], t.set_index("ward").loc[2]
    assert r1.tree_ha == 4 and r1.converted_ha == 1 and r1.loss_frac == pytest.approx(0.25)
    assert r1.ndvi_change_mean == pytest.approx(-0.05) and r2.loss_frac == 0 and r2.ndvi_change_mean == pytest.approx(0.1)
    assert len(vc.ward_table(wards, tree23, conv, change, px_ha=1.0, min_ha=5.0)) == 0   # wards under the tree-area floor are dropped


def test_spearman_bootstrap_recovers_a_negative_relationship():
    rng = np.random.default_rng(0)
    a = rng.random(60); b = -a + rng.normal(0, 0.15, 60)
    rho, (lo, hi) = vc.spearman_bootstrap(a, b, n_boot=300, seed=0)
    assert rho < -0.7 and hi < 0 and lo <= rho <= hi


def test_decision_rule_follows_the_registered_thresholds():
    ok_pix = dict(prob=0.7, ci=(0.62, 0.78), median_diff=-0.03)
    weak_pix = dict(prob=0.7, ci=(0.45, 0.78), median_diff=-0.03)
    small_pix = dict(prob=0.7, ci=(0.62, 0.78), median_diff=-0.01)
    ok_ward = dict(rho=-0.5, ci=(-0.7, -0.2)); bad_ward = dict(rho=-0.5, ci=(-0.7, 0.1))
    assert vc.decide(ok_pix, ok_ward).startswith("corroborated")
    assert vc.decide(ok_pix, bad_ward).startswith("partly")
    assert vc.decide(weak_pix, ok_ward).startswith("unconfirmed")
    assert vc.decide(small_pix, bad_ward).startswith("unconfirmed")
