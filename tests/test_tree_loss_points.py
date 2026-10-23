"""Tree-loss check points: selection, spacing, blind sheet and scoring (synthetic)."""
import numpy as np
import pandas as pd
import pytest

from src import tree_loss_points as tl


def test_core_mask_requires_enough_same_group_neighbours():
    m = np.zeros((7, 7), bool); m[1:6, 1:6] = True
    c = tl.core_mask(m, min_same=8)
    assert c[3, 3] and not c[1, 1] and c.sum() == 9                                      # only the inner 3 x 3 has all 8 neighbours
    assert tl.core_mask(m, min_same=3).sum() > c.sum()
    assert not tl.core_mask(m, 0)[0, 0]                                                  # never outside the group


def test_pick_points_keeps_the_minimum_spacing_and_is_deterministic():
    cand = np.ones((60, 60), bool)
    a = tl.pick_points(cand, 12, np.random.default_rng(1), min_dist_px=10)
    b = tl.pick_points(cand, 12, np.random.default_rng(1), min_dist_px=10)
    assert np.array_equal(a, b) and len(a) == 12
    d = np.hypot(a[:, None, 0] - a[None, :, 0], a[:, None, 1] - a[None, :, 1]); np.fill_diagonal(d, 99)
    assert d.min() >= 10


def test_pick_points_returns_fewer_when_the_candidates_run_out():
    cand = np.zeros((40, 40), bool); cand[5, 5] = cand[5, 6] = cand[30, 30] = True
    assert len(tl.pick_points(cand, 10, np.random.default_rng(0), 10)) == 2


def test_choose_relaxes_the_core_rule_until_enough_points_are_found():
    g = np.zeros((80, 80), bool)
    g[10, 10] = g[10, 30] = g[10, 50] = g[10, 70] = True                                 # isolated pixels: no neighbours at all
    got = tl.choose({"converted": g}, {"converted": 3}, np.random.default_rng(0))
    assert len(got) == 3 and all(p["group"] == "converted" and p["relax"] == 0 for p in got)
    solid = np.zeros((80, 80), bool); solid[10:40, 10:40] = True
    got2 = tl.choose({"stable_tree": solid}, {"stable_tree": 3}, np.random.default_rng(0))
    assert all(p["relax"] == 6 for p in got2)                                            # strictest threshold was enough


def test_wilson_interval_matches_the_textbook_value():
    lo, hi = tl.wilson(20, 25)
    assert lo == pytest.approx(0.6086, abs=0.003) and hi == pytest.approx(0.9114, abs=0.003)
    assert tl.wilson(0, 0) == (float("nan"), float("nan")) or all(np.isnan(tl.wilson(0, 0)))


def test_score_reports_precision_for_converted_and_agreement_for_controls():
    key = pd.DataFrame(dict(id=[f"T{i:02d}" for i in range(1, 9)], group=["converted"] * 4 + ["stable_tree"] * 2 + ["stable_built"] * 2))
    sheet = pd.DataFrame(dict(id=key.id,
        trees_in_2023=["Yes", "Yes", "No", "Unsure", "Yes", "Yes", "No", "No"],
        now_covers=["Built-up", "Bare ground", "Built-up", "Built-up", "Trees", "Built-up", "Built-up", "Built-up"]))
    r = tl.score(sheet, key)
    c = r["converted"]
    assert c["n"] == 4 and c["tree_to_built_lower"] == pytest.approx(1 / 4) and c["tree_to_built_upper"] == pytest.approx(1 / 3)
    assert c["tree_loss_any_lower"] == pytest.approx(2 / 4) and c["unsure"] == 1
    assert r["stable_tree"]["agree"] == 1 and r["stable_tree"]["n"] == 2
    assert r["stable_built"]["agree"] == 2 and r["stable_built"]["n"] == 2
