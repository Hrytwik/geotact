"""T6i: nested tuning, soft vote and stacking (no torch; synthetic data and spy estimators)."""
import numpy as np
import pytest

from src import flood_tuning as tu


class Pick:
    """Estimator whose quality depends on a parameter: which=0 uses the signal column, which=1 the noise column."""

    def __init__(self, which=0):
        self.which = which

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        s = 1 / (1 + np.exp(-X[:, 1 + self.which]))
        return np.column_stack([1 - s, s])


def toy(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    sig = rng.normal(size=n) + 2.0 * y
    noise = rng.normal(size=n)
    X = np.column_stack([np.arange(n), sig, noise])                       # column 0 = sample index (for the spies)
    return X, y, np.arange(n) % 5


def test_grid_is_the_product_of_the_options():
    g = tu.grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(g) == 6 and {"a": 2, "b": "z"} in g


def test_percentile_map_is_monotone_bounded_and_saturates_outside_the_training_range():
    f = tu.percentile_map(np.array([1.0, 2.0, 3.0, 4.0]))
    v = f(np.array([-5.0, 1.0, 2.5, 4.0, 99.0]))
    assert (np.diff(v) >= 0).all() and v.min() >= 0 and v.max() <= 1
    assert v[0] < v[2] < v[-1] and v[-1] > 0.85 and v[0] < 0.15


def test_nested_selection_picks_the_better_parameter_in_every_outer_fold():
    X, y, folds = toy()
    out = tu.nested_family(X, y, folds, lambda p: Pick(**p), tu.grid({"which": [1, 0]}))
    assert [c["which"] for c in out["chosen"]] == [0] * 5
    assert out["outer"].shape == (500,) and np.isfinite(out["outer"]).all()


def test_selection_for_an_outer_fold_never_fits_on_that_folds_samples():
    X, y, folds = toy(n=250)
    calls = []

    class Spy(Pick):
        def fit(self, X, y):
            calls.append(set(X[:, 0].astype(int)))
            return self

    g = tu.grid({"which": [0, 1]})
    tu.nested_family(X, y, folds, lambda p: Spy(**p), g)
    per_outer = len(g) * 4 + 1                                              # inner fits (4 inner folds per config) + the outer refit
    assert len(calls) == 5 * per_outer
    for k in range(5):
        block = calls[k * per_outer:(k + 1) * per_outer]
        held_out = set(np.flatnonzero(folds == k))
        assert all(not (c & held_out) for c in block)


def test_inner_out_of_fold_scores_cover_every_training_sample_once():
    X, y, folds = toy()
    out = tu.nested_family(X, y, folds, lambda p: Pick(**p), tu.grid({"which": [0]}))
    for k in range(5):
        assert out["inner_oof"][k].shape == ((folds != k).sum(),) and np.isfinite(out["inner_oof"][k]).all()


def test_soft_vote_ignores_each_models_scale():
    X, y, folds = toy()
    a = tu.nested_family(X, y, folds, lambda p: Pick(**p), tu.grid({"which": [0]}))
    b = tu.nested_family(X, y, folds, lambda p: Pick(**p), tu.grid({"which": [1]}))
    v1 = tu.soft_vote({"a": a, "b": b}, folds)
    big = dict(b, outer=b["outer"] * 1000 + 7, inner_oof=[o * 1000 + 7 for o in b["inner_oof"]])
    v2 = tu.soft_vote({"a": a, "b": big}, folds)
    assert np.allclose(v1, v2, atol=1e-9)


def test_stacking_uses_only_training_information_and_beats_a_noise_model():
    from sklearn.metrics import roc_auc_score
    X, y, folds = toy(seed=3)
    a = tu.nested_family(X, y, folds, lambda p: Pick(**p), tu.grid({"which": [0]}))
    b = tu.nested_family(X, y, folds, lambda p: Pick(**p), tu.grid({"which": [1]}))
    s = tu.stack({"a": a, "b": b}, y, folds)
    assert np.isfinite(s).all()
    assert roc_auc_score(y, s) > roc_auc_score(y, b["outer"]) + 0.2
    assert roc_auc_score(y, s) > 0.85
