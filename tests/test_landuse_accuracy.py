"""T7: stratified accuracy estimators checked against hand-worked examples (no maps needed)."""
import numpy as np
import pandas as pd
import pytest

from src import landuse_accuracy as la


def test_overall_accuracy_is_the_weighted_mean_of_stratum_accuracies_with_the_textbook_variance():
    strata = np.array(["A"] * 4 + ["B"] * 5)
    correct = np.array([1, 1, 1, 0] + [1, 0, 0, 0, 0], dtype=float)                       # A: 0.75, B: 0.20
    W = {"A": 0.6, "B": 0.4}
    acc, se = la.stratified_accuracy(strata, correct, W)
    assert acc == pytest.approx(0.6 * 0.75 + 0.4 * 0.2)
    var = 0.6 ** 2 * (0.25 * 0.75 * 4 / 3) / 4 + 0.4 ** 2 * (0.2 * 0.8 * 5 / 4) / 5          # s^2 = p(1-p) n/(n-1)
    assert se == pytest.approx(np.sqrt(var))


def test_a_perfect_map_has_accuracy_one_and_zero_standard_error():
    strata = np.array(["A"] * 5 + ["B"] * 5)
    acc, se = la.stratified_accuracy(strata, np.ones(10), {"A": 0.3, "B": 0.7})
    assert acc == 1.0 and se == 0.0


def test_ratio_estimate_reduces_to_the_stratum_proportion_when_the_strata_are_the_map_classes():
    # one stratum only mapped as class 1: user's accuracy of class 1 = share correct in that stratum
    strata = np.array(["S1"] * 6 + ["S2"] * 6)
    mapped = np.array([1] * 6 + [2] * 6); ref = np.array([1, 1, 1, 1, 2, 2] + [2] * 6)
    W = {"S1": 0.5, "S2": 0.5}
    ua = la.users_accuracy(strata, mapped, ref, W, [1, 2])
    assert ua[1][0] == pytest.approx(4 / 6) and ua[2][0] == pytest.approx(1.0)


def test_users_and_producers_accuracy_use_different_denominators():
    strata = np.array(["S"] * 10)
    mapped = np.array([1] * 6 + [2] * 4); ref = np.array([1, 1, 1, 1, 2, 2, 1, 2, 2, 2])
    W = {"S": 1.0}
    ua = la.users_accuracy(strata, mapped, ref, W, [1, 2]); pa = la.producers_accuracy(strata, mapped, ref, W, [1, 2])
    assert ua[1][0] == pytest.approx(4 / 6)                                              # mapped 1: 6, right: 4
    assert pa[1][0] == pytest.approx(4 / 5)                                              # truly 1: 5, found: 4
    assert ua[2][0] == pytest.approx(3 / 4) and pa[2][0] == pytest.approx(3 / 5)


def test_area_confusion_sums_to_one_and_has_the_stratum_weights_as_row_margins_for_stratum_defined_maps():
    strata = np.array(["A"] * 4 + ["B"] * 4)
    mapped = np.array(["A"] * 4 + ["B"] * 4); ref = np.array(["A", "A", "A", "B", "B", "B", "A", "B"])
    W = {"A": 0.7, "B": 0.3}
    p = la.area_confusion(strata, mapped, ref, W, ["A", "B"])
    assert p.values.sum() == pytest.approx(1.0)
    assert p.loc["A", "A"] == pytest.approx(0.7 * 3 / 4) and p.loc["A", "B"] == pytest.approx(0.7 * 1 / 4)
    assert p.loc["B"].sum() == pytest.approx(0.3)


def test_paired_difference_detects_a_better_map_and_is_signed_a_minus_b():
    rng = np.random.default_rng(0)
    strata = np.array(["A"] * 100 + ["B"] * 100)
    ca = (rng.random(200) < 0.9).astype(float); cb = (rng.random(200) < 0.6).astype(float)
    W = {"A": 0.5, "B": 0.5}
    r = la.paired_difference(strata, ca, cb, W, n_boot=500, seed=0)
    assert r["diff"] > 0.2 and r["ci"][0] > 0 and r["p"] < 0.01
    same = la.paired_difference(strata, ca, ca, W, n_boot=100, seed=0)
    assert same["diff"] == 0 and same["ci"] == [0.0, 0.0]


def test_label_normalisation_fixes_case_and_rejects_unknown_classes():
    s = pd.Series(["water", "Water", "Trees", "built-up", "Grass/Scrub"])
    out = la.normalise_labels(s)
    assert out.tolist() == ["Water", "Water", "Trees", "Built-up", "Grass/Scrub"]
    with pytest.raises(ValueError):
        la.normalise_labels(pd.Series(["Trees", "Forest"]))
    conf = la.normalise_confidence(pd.Series(["sure", "Sure", "Unsure", "unsure"]))
    assert conf.tolist() == ["Sure", "Sure", "Unsure", "Unsure"]
