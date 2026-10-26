"""T10: NSGA-II site selection on toy problems with a brute-force exact Pareto front."""
import itertools

import numpy as np
import pytest

from src import optimise as op


def toy(n=7, seed=0):
    """n blocks, options 0 (none), 1 wetland, 2 garden, 3 corridor; random additive effects; some options not allowed."""
    rng = np.random.default_rng(seed)
    allowed = np.ones((n, 4), bool)
    allowed[:, 1:] = rng.random((n, 3)) < 0.7
    area = np.zeros((n, 4)); area[:, 1:] = rng.uniform(0.5, 1.0, (n, 3))
    d_runoff = np.zeros((n, 4)); d_runoff[:, 1] = -rng.uniform(5, 10, n); d_runoff[:, 2] = -rng.uniform(0, 1, n); d_runoff[:, 3] = -rng.uniform(1, 3, n)
    d_food = np.zeros((n, 4)); d_food[:, 1] = -rng.uniform(0, 1, n); d_food[:, 2] = rng.uniform(0.5, 1, n)
    d_conn = np.zeros((n, 4)); d_conn[:, 1] = rng.uniform(0, 1, n); d_conn[:, 3] = rng.uniform(1, 2, n)
    return op.SiteData(allowed=allowed, area=area, d_runoff=d_runoff, d_food=d_food, d_conn=d_conn, budget=3.0,
                       base_runoff=100.0, base_food=10.0)


def brute_force(data):
    n = data.allowed.shape[0]
    X, F = [], []
    for x in itertools.product(range(4), repeat=n):
        x = np.array(x)
        if not data.allowed[np.arange(n), x].all() or data.area[np.arange(n), x].sum() > data.budget:
            continue
        X.append(x); F.append(op.objectives(data, x[None])[0])
    F = np.array(F)
    return F[op.pareto_mask(F)]


def test_objectives_are_sums_of_the_chosen_options():
    d = toy()
    x = np.zeros((1, 7), int); x[0, 0] = 1
    if not d.allowed[0, 1]:
        d.allowed[0, 1] = True
    f = op.objectives(d, x)[0]
    assert f[0] == pytest.approx(100.0 + d.d_runoff[0, 1]) and f[1] == pytest.approx(-(10.0 + d.d_food[0, 1]))
    assert f[2] == pytest.approx(-d.d_conn[0, 1])


def test_pareto_mask_keeps_only_non_dominated_points():
    F = np.array([[1, 2, 3], [2, 2, 3], [1, 1, 4], [0, 5, 5]], float)
    assert op.pareto_mask(F).tolist() == [True, False, True, True]


def test_repair_enforces_allowed_options_and_the_budget():
    d = toy()
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, (50, 7))
    R = op.repair(d, X, rng)
    n = np.arange(7)
    for x in R:
        assert d.allowed[n, x].all() and d.area[n, x].sum() <= d.budget + 1e-9


def test_nsga2_recovers_the_brute_force_front_and_respects_the_constraints():
    d = toy()
    exact = brute_force(d)
    X, F = op.run_nsga(d, pop=60, gens=120, seed=0)
    n = np.arange(7)
    assert all(d.allowed[n, x].all() and d.area[n, x].sum() <= d.budget + 1e-9 for x in X)
    lo, hi = np.vstack([exact, F]).min(axis=0), np.vstack([exact, F]).max(axis=0)
    hv_exact = op.hypervolume(exact, lo, hi); hv = op.hypervolume(F, lo, hi)
    assert hv >= 0.98 * hv_exact


def test_same_seed_gives_the_same_front():
    d = toy(seed=1)
    _, F1 = op.run_nsga(d, pop=30, gens=20, seed=3)
    _, F2 = op.run_nsga(d, pop=30, gens=20, seed=3)
    assert np.allclose(np.sort(F1, axis=0), np.sort(F2, axis=0))


# ---------------------------------------------------------------- T14: optional fourth objective (heat), three-objective path unchanged
def with_heat(d, seed=5):
    rng = np.random.default_rng(seed)
    n = d.allowed.shape[0]
    h = np.zeros((n, 4)); h[:, 1] = -rng.uniform(0.0, 1.0, n); h[:, 2] = -rng.uniform(0.0, 0.3, n); h[:, 3] = -rng.uniform(0.2, 1.5, n)
    import dataclasses
    return dataclasses.replace(d, d_heat=h)


def test_three_objective_problems_are_unchanged_and_heat_adds_a_fourth_column():
    d = toy()
    x = np.zeros((1, 7), int); x[0, 1] = 3
    d.allowed[1, 3] = True
    assert op.objectives(d, x).shape == (1, 3)
    dh = with_heat(d)
    f = op.objectives(dh, x)[0]
    assert f.shape == (4,) and f[3] == pytest.approx(dh.d_heat[1, 3]) and f[:3] == pytest.approx(op.objectives(d, x)[0])


def test_nsga2_with_heat_recovers_the_four_objective_brute_force_front():
    d = with_heat(toy())
    exact = brute_force(d)
    assert exact.shape[1] == 4
    X, F = op.run_nsga(d, pop=80, gens=150, seed=0)
    n = np.arange(7)
    assert F.shape[1] == 4 and all(d.allowed[n, x].all() and d.area[n, x].sum() <= d.budget + 1e-9 for x in X)
    lo, hi = np.vstack([exact, F]).min(axis=0), np.vstack([exact, F]).max(axis=0)
    assert op.hypervolume(F, lo, hi) >= 0.95 * op.hypervolume(exact, lo, hi)
