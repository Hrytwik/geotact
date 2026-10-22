"""T6j Part B: urban sample assembly, regimes and tasks (no torch; synthetic world)."""
import numpy as np
import pandas as pd
import pytest

from src import flood_tuning as tu
from src import flood_urban as ur


def world(seed=0):
    """A small 'city': survey flood/non-flood/background samples, some built-up, plus urban cells."""
    rng = np.random.default_rng(seed)
    n_bg = 400
    rows = np.r_[rng.integers(0, 100, 30), rng.integers(0, 100, 5), rng.integers(0, 100, n_bg)]
    cols = np.r_[rng.integers(0, 100, 30), rng.integers(0, 100, 5), rng.integers(0, 100, n_bg)]
    kind = np.array(["flood"] * 30 + ["nonflood"] * 5 + ["background"] * n_bg)
    folds_map = (np.arange(100)[:, None] // 20 + 0 * np.arange(100)[None, :]) % 5
    built = rng.random((100, 100))
    return rows, cols, kind, folds_map, built


def test_urban_cells_keep_only_usable_rows_and_optionally_drop_the_doubtful_ones(tmp_path):
    csv = tmp_path / "u.csv"
    pd.DataFrame(dict(id=["A", "B", "C", "D"], lat=[20.30, 20.31, 20.32, 20.33], lon=[85.80, 85.81, 85.82, 85.83],
                      use_in_training=[True, True, False, True], doubtful=[False, True, False, False])).to_csv(csv, index=False)
    ident = lambda lon, lat: (lat * 1000 % 100, lon * 1000 % 100)                 # stand-in projection (row, col) in a toy grid
    cells = ur.urban_cells(csv, ident, drop_doubtful=False)
    assert len(cells) == 3
    cells2 = ur.urban_cells(csv, ident, drop_doubtful=True)
    assert len(cells2) == 2


def test_assemble_adds_urban_cells_gives_them_folds_and_drops_background_next_to_them():
    rows, cols, kind, fmap, built = world()
    base = dict(rows=rows, cols=cols, kind=kind, fold=fmap[rows, cols])
    urb = np.array([[rows[40], cols[40]], [10, 10]])                                   # first sits ON a background cell
    s = ur.assemble(base, urb, fmap, built, min_dist_cells=2)
    assert (s["kind"] == "urban").sum() == 2
    assert not ((s["rows"] == rows[40]) & (s["cols"] == cols[40]) & (s["kind"] == "background")).any()
    d = np.hypot(s["rows"][s["kind"] == "background"][:, None] - urb[:, 0][None], s["cols"][s["kind"] == "background"][:, None] - urb[:, 1][None])
    assert d.min() >= 2
    assert (s["fold"][s["kind"] == "urban"] == fmap[urb[:, 0], urb[:, 1]]).all()
    assert s["builtup"].dtype == bool and len(s["builtup"]) == len(s["kind"])


def test_nan_builtup_share_counts_as_not_built_up():
    rows, cols, kind, fmap, built = world()
    built = built.copy(); built[rows[0], cols[0]] = np.nan
    base = dict(rows=rows, cols=cols, kind=kind, fold=fmap[rows, cols])
    s = ur.assemble(base, np.zeros((0, 2), int), fmap, built)
    assert not s["builtup"][0]


def test_regimes_use_the_registered_positives_and_negatives():
    kind = np.array(["flood", "flood", "urban", "urban", "nonflood", "nonflood", "background", "background", "background"])
    builtup = np.array([True, False, True, True, True, False, True, False, False])
    m, y = ur.regime("S_full", kind, builtup)
    assert m.sum() == 2 + 2 + 3 and y[m].sum() == 2                                    # flood vs nonflood + all background
    m, y = ur.regime("S", kind, builtup)
    assert set(np.flatnonzero(m)) == {0, 1, 4, 6} and y[m].sum() == 2                    # flood vs built-up negatives only
    m, y = ur.regime("U", kind, builtup)
    assert set(np.flatnonzero(m)) == {2, 3, 4, 6} and y[m].sum() == 2
    m, y = ur.regime("S+U", kind, builtup)
    assert set(np.flatnonzero(m)) == {0, 1, 2, 3, 4, 6} and y[m].sum() == 4
    with pytest.raises(ValueError):
        ur.regime("nope", kind, builtup)


def test_tasks_pit_the_right_positives_against_the_right_negatives():
    kind = np.array(["flood", "urban", "urban", "nonflood", "nonflood", "background", "background", "background"])
    builtup = np.array([False, True, True, True, False, True, False, True])
    t = ur.task("urban", kind, builtup)
    assert set(np.flatnonzero(t["mask"])) == {1, 2, 3, 5, 7} and t["y"][t["mask"]].sum() == 2
    assert list(t["kind"][t["mask"]]).count("flood") == 2 and list(t["kind"][t["mask"]]).count("nonflood") == 1
    s = ur.task("survey", kind, builtup)
    assert set(np.flatnonzero(s["mask"])) == {0, 3, 4, 5, 6, 7} and s["y"][s["mask"]].sum() == 1


def test_nested_family_can_also_score_samples_outside_the_regime_training_set():
    rng = np.random.default_rng(0)
    n = 300
    y = (rng.random(n) < 0.3).astype(int)
    sig = rng.normal(size=n) + 2 * y
    X = np.column_stack([np.arange(n), sig, rng.normal(size=n)])
    folds = np.arange(n) % 5
    # evaluation set = a different, larger sample set with its own folds (as for regime U scoring all samples)
    Xe = np.vstack([X, X[:50] + 0.0])
    fe = np.r_[folds, folds[:50]]

    class Pick:
        def __init__(self, which=0): self.which = which
        def fit(self, X, y): return self
        def predict_proba(self, X):
            s = 1 / (1 + np.exp(-X[:, 1 + self.which])); return np.column_stack([1 - s, s])

    out = tu.nested_family(X, y, folds, lambda p: Pick(**p), tu.grid({"which": [0]}), X_eval=Xe, folds_eval=fe)
    assert out["outer_eval"].shape == (len(Xe),) and np.isfinite(out["outer_eval"]).all()
    assert np.allclose(out["outer_eval"][:n], out["outer"])


def test_soft_vote_eval_ignores_each_models_scale():
    rng = np.random.default_rng(1)
    n = 200
    y = (rng.random(n) < 0.3).astype(int)
    X = np.column_stack([np.arange(n), rng.normal(size=n) + 2 * y, rng.normal(size=n) + y])
    folds = np.arange(n) % 5

    class Pick:
        def __init__(self, which=0): self.which = which
        def fit(self, X, y): return self
        def predict_proba(self, X):
            s = 1 / (1 + np.exp(-X[:, 1 + self.which])); return np.column_stack([1 - s, s])

    a = tu.nested_family(X, y, folds, lambda p: Pick(**p), tu.grid({"which": [0]}), X_eval=X, folds_eval=folds)
    b = tu.nested_family(X, y, folds, lambda p: Pick(**p), tu.grid({"which": [1]}), X_eval=X, folds_eval=folds)
    v1 = tu.soft_vote_eval({"a": a, "b": b}, folds)
    big = dict(b, outer_eval=b["outer_eval"] * 1000 + 3, inner_oof=[o * 1000 + 3 for o in b["inner_oof"]])
    assert np.allclose(v1, tu.soft_vote_eval({"a": a, "b": big}, folds), atol=1e-9)
