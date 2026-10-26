"""T15 vegetation-loss map check: frozen Youden threshold, loss map inside the vegetated population, stratified blind sample and the
design-based scoring (synthetic data)."""
import numpy as np
import pandas as pd
import pytest

from src import vegetation_check as vc


def test_youden_threshold_separates_a_clean_example_and_is_on_the_change_scale():
    change = np.array([-0.30, -0.25, -0.20, -0.02, 0.00, 0.03, -0.22, 0.01])
    truth = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    tau = vc.youden_threshold(change, truth)
    assert -0.20 <= tau < -0.02                                                               # loss = change <= tau
    assert ((change <= tau).astype(int) == truth).all()


def test_loss_map_only_inside_the_vegetated_population():
    change = np.array([[-0.3, -0.3], [0.1, np.nan]])
    veg = np.array([[True, False], [True, True]])
    loss, pop = vc.loss_map(change, veg, tau=-0.1)
    assert loss.tolist() == [[True, False], [False, False]] and pop.tolist() == [[True, False], [True, False]]   # NaN change is out


def test_sample_draws_the_planned_counts_apart_from_each_other_and_from_old_points():
    rng = np.random.default_rng(0)
    loss = np.zeros((300, 300), bool); loss[50:150, 50:150] = rng.random((100, 100)) < 0.5
    pop = np.ones((300, 300), bool)
    avoid = np.array([[100, 100], [250, 250]])
    pts = vc.draw_sample(loss, pop, {"loss": 20, "no_loss": 20}, min_px=10, avoid=avoid, seed=1)
    assert pts.stratum.value_counts().to_dict() == {"loss": 20, "no_loss": 20}
    P = pts[["row", "col"]].to_numpy()
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]) + np.eye(len(P)) * 1e9
    assert d.min() >= 10 and min(np.hypot(*(P - a).T).min() for a in avoid) >= 10
    lp, nl = pts[pts.stratum == "loss"], pts[pts.stratum == "no_loss"]
    assert loss[lp.row, lp.col].all() and not loss[nl.row, nl.col].any()                     # each point sits in its own stratum
    assert pts.equals(vc.draw_sample(loss, pop, {"loss": 20, "no_loss": 20}, min_px=10, avoid=avoid, seed=1))


def test_truth_counts_only_permanent_cover_loss_and_drops_unsure():
    sheet = pd.DataFrame(dict(id=["V1", "V2", "V3", "V4", "V5"], vegetation_before=["Yes", "Yes", "No", "Yes", "Unsure"],
                              now_covers=["Built-up", "Cropland", "Bare ground", "Unsure", "Built-up"]))
    t = vc.truth(sheet)
    assert t.set_index("id").loss.to_dict() == {"V1": 1, "V2": 0, "V3": 0}


def test_design_based_scores_match_a_hand_worked_example():
    key = pd.DataFrame(dict(id=[f"V{i}" for i in range(8)], stratum=["loss"] * 4 + ["no_loss"] * 4))
    truth = pd.DataFrame(dict(id=key.id, loss=[1, 1, 1, 0, 0, 0, 1, 0]))
    W = {"loss": 0.1, "no_loss": 0.9}
    r = vc.score(truth, key, W)
    assert r["users_accuracy_loss"][0] == pytest.approx(0.75)                                 # 3 of 4 flagged are real
    true_loss_share = 0.1 * 0.75 + 0.9 * 0.25
    assert r["loss_area_share"][0] == pytest.approx(true_loss_share)
    assert r["producers_accuracy_loss"][0] == pytest.approx(0.1 * 0.75 / true_loss_share)
    assert r["overall"][0] == pytest.approx(0.1 * 0.75 + 0.9 * 0.75)
