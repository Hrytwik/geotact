"""T14 heat model: multi-scale share features, edit-class mapping, exact additive exposure change (synthetic)."""
import numpy as np
import pytest

from src import heat


def toy_map(seed=0, shape=(60, 70)):
    return np.random.default_rng(seed).integers(0, 6, shape).astype(np.uint8)


def test_edited_classes_are_read_as_their_closest_land_cover():
    m = np.array([[6, 7, 8, 4]], dtype=np.uint8)                                           # garden, wetland, corridor, built
    assert heat.as_cover(m).tolist() == [[3, 0, 1, 4]]


def test_feature_stack_has_three_scales_of_five_shares_plus_elevation():
    F = heat.features(toy_map(), elevation=np.zeros((60, 70), "float32"))
    assert F.shape == (16, 60, 70) and np.isfinite(F).all()
    assert heat.FEATURES[:5] == ["water_s0", "trees_s0", "crop_s0", "grass_s0", "bare_s0"] and heat.FEATURES[-1] == "elevation"


def test_exposure_change_from_the_fast_linear_path_equals_a_full_recompute():
    rng = np.random.default_rng(1)
    base = toy_map(1)
    beta = rng.normal(0, 1, 16); beta[-1] = 0.0
    w = rng.random(base.shape).astype("float32")
    edited = base.copy(); edited[10:20, 30:45] = 8; edited[40:44, 5:9] = 7                   # a corridor and a wetland
    full = heat.exposure_change(edited, base, beta, w)
    G = heat.exposure_kernels(w)
    fast = heat.exposure_change_fast(edited, base, beta, G, w.sum())
    assert fast == pytest.approx(full, rel=1e-4, abs=1e-6)


def test_fast_changes_add_up_across_separate_edits():
    rng = np.random.default_rng(2)
    base = toy_map(2); beta = rng.normal(0, 1, 16); w = rng.random(base.shape).astype("float32")
    G = heat.exposure_kernels(w)
    a = base.copy(); a[5:10, 5:10] = 8
    b = base.copy(); b[40:50, 50:60] = 7
    ab = a.copy(); ab[40:50, 50:60] = 7
    da, db, dab = (heat.exposure_change_fast(m, base, beta, G, w.sum()) for m in (a, b, ab))
    assert dab == pytest.approx(da + db, rel=1e-6, abs=1e-9)


def test_spatial_cv_recovers_a_planted_linear_effect():
    rng = np.random.default_rng(3)
    m = toy_map(3, (90, 90))
    F = heat.features(m, elevation=rng.normal(0, 1, (90, 90)).astype("float32"))
    beta = np.zeros(16); beta[1] = -3.0; beta[16 - 1] = 0.5                                 # trees cool, elevation a little
    y = np.tensordot(beta, F, axes=1) + rng.normal(0, 0.05, (90, 90))
    r = heat.spatial_cv(F, y, np.ones((90, 90), bool), block_px=30, step=3)
    assert r["r2"] > 0.8


def test_sign_bounds_follow_the_physical_constraints():
    lo, hi = heat.sign_bounds()
    for i, n in enumerate(heat.FEATURES):
        if n.startswith(("water", "trees", "grass")):
            assert hi[i] == 0.0 and lo[i] == -np.inf
        elif n.startswith("bare"):
            assert lo[i] == 0.0 and hi[i] == np.inf
        else:
            assert lo[i] == -np.inf and hi[i] == np.inf


def test_bounded_fit_never_returns_a_forbidden_sign_even_when_the_data_ask_for_it():
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] - 1.0 * X[:, 2] + rng.normal(0, 0.1, n)
    m = heat.BoundedOLS([-np.inf, -np.inf, 0.0], [np.inf, 0.0, np.inf]).fit(X, y)           # true x2 < 0 and x1 > 0 are both forbidden
    assert m.coef_[0] == pytest.approx(2.0, abs=0.05) and m.coef_[1] <= 1e-9 and m.coef_[2] >= -1e-9
    free = heat.BoundedOLS([-np.inf] * 3, [np.inf] * 3).fit(X, y)
    assert free.coef_ == pytest.approx([2.0, 0.5, -1.0], abs=0.05) and free.intercept_ == pytest.approx(0.0, abs=0.02)


def test_with_constrained_signs_turning_bare_ground_into_a_cooling_class_never_raises_heat_exposure():
    """Provable under the constraints: bare ground has a coefficient >= 0 and water / trees / grass have coefficients <= 0, so every
    term of the change is <= 0. (Farmland, whose coefficient is free, or grass to water, carry no such guarantee.)"""
    rng = np.random.default_rng(1)
    base = toy_map(4, (80, 90)); w = rng.random(base.shape).astype("float32")
    lo, hi = heat.sign_bounds()
    beta = np.where(lo == 0.0, rng.uniform(0, 2, len(lo)), np.where(hi == 0.0, -rng.uniform(0, 2, len(lo)), rng.normal(0, 1, len(lo))))
    beta[-1] = 0.0                                                                           # elevation is never edited
    G = heat.exposure_kernels(w)
    b = base.copy(); b[20:60, 20:70] = 5                                                     # a bare-ground patch
    for cls in (7, 8, 6):                                                                    # wetland, corridor, garden
        e = b.copy(); e[30:50, 30:60] = cls
        assert heat.exposure_change_fast(e, b, beta, G, w.sum()) <= 1e-9
