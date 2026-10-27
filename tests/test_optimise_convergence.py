"""T19 convergence judgement and the union hypervolume (synthetic)."""
import numpy as np

from src import optimise_convergence as oc


def test_union_hypervolume_is_at_least_every_member():
    rng = np.random.default_rng(0)
    a, b = rng.random((30, 3)), rng.random((30, 3))
    lo, hi = np.zeros(3), np.ones(3)
    u = oc.front_hv([a, b], lo, hi)
    assert u >= oc.front_hv([a], lo, hi) - 1e-12 and u >= oc.front_hv([b], lo, hi) - 1e-12


def test_judge_applies_the_registered_rule():
    hv = {300: dict(union=0.90, seeds=[0.80, 0.85, 0.88]), 1000: dict(union=0.97, seeds=[0.95, 0.955, 0.96]),
          3000: dict(union=0.972, seeds=[0.966, 0.968, 0.97]), 10000: dict(union=0.973, seeds=[0.97, 0.971, 0.972])}
    j = oc.judge(hv, (300, 1000, 3000, 10000))
    assert not j[300]["converged"]                    # gains 7.8% and spreads 9%
    assert j[1000]["converged"]                       # gain to 3000 is 0.2% (< 0.5%) and spread 1.05% (< 2%)
    assert j[3000]["converged"] and j["shortest_converged"] == 1000
    assert not j["first_adequate"]                    # 0.90 < 0.99 * 0.973
    assert not j[10000]["converged"]                  # nothing longer to compare with


def test_judge_marks_the_first_setting_adequate_when_close_to_the_longest():
    hv = {g: dict(union=u, seeds=[u, u, u]) for g, u in ((300, 0.968), (1000, 0.97), (3000, 0.971), (10000, 0.971))}
    assert oc.judge(hv, (300, 1000, 3000, 10000))["first_adequate"]


def test_archive_update_keeps_only_unique_nondominated_points():
    from src import optimise as op
    X0, F0 = np.empty((0, 2), int), np.empty((0, 2))
    X1 = np.array([[1, 0], [0, 1], [1, 1]]); F1 = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])     # third is dominated
    X, F = op.archive_update(X0, F0, X1, F1)
    assert sorted(map(tuple, X)) == [(0, 1), (1, 0)]
    X2 = np.array([[1, 0], [2, 2]]); F2 = np.array([[1.0, 2.0], [0.5, 0.5]])                       # duplicate, then one that dominates all
    X, F = op.archive_update(X, F, X2, F2)
    assert X.tolist() == [[2, 2]] and F.tolist() == [[0.5, 0.5]]


def test_archive_search_returns_at_least_as_much_hypervolume_as_the_final_population():
    from src import optimise as op
    rng = np.random.default_rng(0)
    n = 40
    allowed = np.ones((n, 4), bool)
    area = np.c_[np.zeros(n), rng.uniform(0.5, 1.5, (n, 3))]
    d = op.SiteData(allowed, area, -rng.random((n, 4)) * [0, 1, 0.2, 0.3], rng.normal(0, 1, (n, 4)) * [0, -1, 1, 0],
                    rng.random((n, 4)) * [0, 0.5, 0.2, 1], 10.0, 0.0, 0.0, d_heat=-rng.random((n, 4)) * [0, 0.1, 0.3, 0.2])
    Xp, Fp = op.run_nsga(d, 20, 40, 0)
    Xa, Fa = op.run_nsga(d, 20, 40, 0, archive_every=5)
    U = np.vstack([Fp, Fa]); lo, hi = U.min(axis=0), U.max(axis=0)
    assert len(Xa) >= len(Xp) * 0.5 and op.hypervolume(Fa, lo, hi) >= op.hypervolume(Fp[op.pareto_mask(Fp)], lo, hi) - 1e-9
    assert (op.area_used(d, Xa) <= d.budget + 1e-9).all()


def test_headlines_count_all_four_and_find_the_knee():
    import pandas as pd
    b = dict(runoff_m3=100.0, food_gj=50.0)
    R = pd.DataFrame(dict(runoff_m3=[90.0, 99.0, 101.0], food_gj=[49.0, 51.0, 60.0], eca_gain_ha=[5.0, 1.0, 0.0],
                          heat_c=[-0.01, -0.001, 0.0], wetland_ha=[1, 2, 3], garden_ha=[0, 0, 0], corridor_ha=[0, 0, 0]))
    h = oc.headlines(R, b)
    assert h["better_all_four"] == 1 and np.isclose(h["min_runoff_pct"], -10.0) and np.isclose(h["max_food_gj"], 10.0)
    assert np.isclose(h["best_all4_runoff_pct"], -1.0) and h["knee"]["wetland_ha"] in (1, 2, 3)
