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
