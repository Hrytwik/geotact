"""T17: correction model for the T15 vegetation-loss map (protocol in docs/build_plan.md). A small design-weighted logistic regression
on six per-pixel features, trained on the 100 T15 labels and scored out of fold (spatial blocks, repeated, majority vote) with the
T15 stratified estimators and a paired stratified bootstrap against the frozen map."""
import json

import numpy as np
import pandas as pd

from src import landuse_accuracy as la

BLOCK_PX = 200                                   # 2 km blocks on the 10 m grid
WATER, BUILT, BARE = 0, 4, 5                     # band indices of the ensemble probability rasters (CLASSES order)
FEATURES = ["ndvi_change", "ndvi_before", "ndvi_now", "ndvi_change_3x3", "nonveg_2026", "nonveg_2023"]


def design_weights(strata, W):
    strata = np.asarray(strata)
    n = {h: int((strata == h).sum()) for h in np.unique(strata)}
    return np.array([W[h] / n[h] for h in strata])


def f1_cutoff(p, y, w):
    """Probability cut-off (predict loss when p >= cut) maximising the weighted F1 = 2TP / (2TP + FP + FN); ties take the middle of
    the best plateau, and the cut sits midway between two observed values."""
    p, y, w = np.asarray(p, float), np.asarray(y, int), np.asarray(w, float)
    u = np.unique(p)
    cuts = np.r_[(u[:-1] + u[1:]) / 2, u[-1] + 1e-9] if len(u) > 1 else u
    f = []
    for c in cuts:
        pr = p >= c
        tp, fp, fn = w[pr & (y == 1)].sum(), w[pr & (y == 0)].sum(), w[~pr & (y == 1)].sum()
        f.append(2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0)
    f = np.array(f)
    best = np.flatnonzero(np.isclose(f, f.max()))
    return float(cuts[best[len(best) // 2]])


def spatial_folds(rows, cols, k, seed, block_px=BLOCK_PX):
    """Fold id per point; whole blocks go to one fold, blocks are dealt to the folds at random."""
    blocks = (np.asarray(rows) // block_px) * 100000 + np.asarray(cols) // block_px
    ub = np.unique(blocks)
    rng = np.random.default_rng(seed)
    assign = dict(zip(rng.permutation(ub), np.arange(len(ub)) % k))
    return np.array([assign[b] for b in blocks])


def _fit_predict(Xtr, ytr, wtr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=1.0, max_iter=1000)
    sw = wtr / wtr.mean()
    m.fit(sc.transform(Xtr), ytr, sample_weight=sw)
    cut = f1_cutoff(m.predict_proba(sc.transform(Xtr))[:, 1], ytr, wtr)
    return m.predict_proba(sc.transform(Xte))[:, 1] >= cut


def oof_vote(X, y, strata, W, rows, cols, reps=20, k=5, seed=0):
    """Majority vote (>= half of the repetitions) of the out-of-fold predictions; every cut-off is chosen on the training fold only."""
    X, y = np.asarray(X, float), np.asarray(y, int)
    w = design_weights(strata, W)
    votes = np.zeros(len(y))
    for r in range(reps):
        f = spatial_folds(rows, cols, k, seed * 1000 + r)
        for j in range(k):
            te = f == j
            if te.sum() == 0 or len(np.unique(y[~te])) < 2:
                continue
            votes[te] += _fit_predict(X[~te], y[~te], w[~te], X[te])
    return (votes >= reps / 2).astype(int)


def oof_repetitions(X, y, strata, W, rows, cols, reps=20, k=5, seed=0):
    """F1 of each single repetition (the spread caused by how the blocks fell into folds)."""
    out = []
    for r in range(reps):
        out.append(score_prediction(oof_vote(X, y, strata, W, rows, cols, reps=1, k=k, seed=seed * 1000 + r + 7), y, strata, W))
    return out


def score_prediction(pred, y, strata, W):
    pred, ref = np.asarray(pred, bool), np.asarray(y, bool)
    ua = la.ratio_estimate(strata, pred & ref, pred, W)[0]
    pa = la.ratio_estimate(strata, pred & ref, ref, W)[0]
    f1 = 2 * ua * pa / (ua + pa) if np.isfinite(ua) and np.isfinite(pa) and ua + pa > 0 else float("nan")
    return dict(ua=float(ua), pa=float(pa), f1=float(f1))


def paired_bootstrap(new, old, y, strata, W, n_boot=2000, seed=0):
    """Stratified bootstrap (resample points inside each stratum) of F1(new) - F1(old), and intervals for the new map's accuracies."""
    new, old, y, strata = np.asarray(new), np.asarray(old), np.asarray(y), np.asarray(strata)
    rng = np.random.default_rng(seed)
    idx_by = {h: np.flatnonzero(strata == h) for h in np.unique(strata)}
    d, ua, pa, f1 = [], [], [], []
    for _ in range(n_boot):
        ix = np.concatenate([rng.choice(v, len(v)) for v in idx_by.values()])
        a, b = score_prediction(new[ix], y[ix], strata[ix], W), score_prediction(old[ix], y[ix], strata[ix], W)
        d.append(a["f1"] - b["f1"]); ua.append(a["ua"]); pa.append(a["pa"]); f1.append(a["f1"])
    q = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]
    s0, s1 = score_prediction(new, y, strata, W), score_prediction(old, y, strata, W)
    return dict(diff=float(s0["f1"] - s1["f1"]), lo=q(d)[0], hi=q(d)[1], ua_ci=q(ua), pa_ci=q(pa), f1_ci=q(f1), n_nan=int(np.isnan(d).sum()))


# ---------------------------------------------------------------- features
def nan_mean3(a):
    """3x3 mean that ignores NaN (a pixel with no valid neighbour stays NaN)."""
    from scipy import ndimage as ndi
    a = np.asarray(a, "float64")
    ok = np.isfinite(a)
    s = ndi.uniform_filter(np.where(ok, a, 0.0), 3, mode="constant") * 9
    c = ndi.uniform_filter(ok.astype("float64"), 3, mode="constant") * 9
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0.5, s / c, np.nan)


def nonveg_probability(probs):
    """Probability (0-1) of Water + Built-up + Bare ground from an ensemble probability stack stored as percent."""
    p = np.asarray(probs, "float32")
    return (p[WATER] + p[BUILT] + p[BARE]) / 100.0


def feature_arrays():
    """The six features as full-grid float32 arrays (read once; ~0.3 GB peak)."""
    import rasterio

    from src import landuse as lu
    from src.config import OUTPUTS
    stack = lu.ndvi_seasons()
    before = np.nanmean(stack[[3, 4]], axis=0)
    now = np.nanmean(stack[[6, 7]], axis=0)
    del stack
    change = (now - before).astype("float32")
    out = [change, before.astype("float32"), now.astype("float32"), nan_mean3(change).astype("float32")]
    for name in ("probs_2025_26_ens.tif", "probs_2023_ens.tif"):
        with rasterio.open(OUTPUTS / "landuse" / name) as r:
            out.append(nonveg_probability(r.read()).astype("float32"))
    return out


def design_matrix(arrays, rows, cols):
    return np.column_stack([a[rows, cols] for a in arrays]).astype(float)


# ---------------------------------------------------------------- real run
def run(reps=20, n_boot=2000, seed=0):
    from src.config import OUTPUTS, ROOT
    from src.vegetation_check import truth
    out = ROOT / "validation" / "vegetation_check"
    info = json.loads((OUTPUTS / "vegetation" / "vegetation_check.json").read_text())
    W = info["weights"]
    t = truth(pd.read_excel(out / "vegetation_labels.xlsx", sheet_name="labels"))
    key = pd.read_csv(out / "vegetation_key_private_do_not_open.csv")
    d = key.merge(t, on="id").reset_index(drop=True)
    arrays = feature_arrays()
    rows, cols = d.row.values, d.col.values
    X = design_matrix(arrays, rows, cols)
    ok = np.isfinite(X).all(axis=1)
    d, X, rows, cols = d[ok].reset_index(drop=True), X[ok], rows[ok], cols[ok]
    y, strata = d.loss.values.astype(int), d.stratum.values
    old = (strata == "loss").astype(int)                                  # the frozen T15 map's class at each sampled pixel
    res = dict(n_points=int(len(d)), n_dropped_nan=int((~ok).sum()), n_loss=int(y.sum()), features=FEATURES)
    res["old_map"] = score_prediction(old, y, strata, W)
    for name, cols_ in (("six_features", list(range(6))), ("threshold_only", [0])):
        pred = oof_vote(X[:, cols_], y, strata, W, rows, cols, reps=reps, seed=seed)
        reps_f1 = [r["f1"] for r in oof_repetitions(X[:, cols_], y, strata, W, rows, cols, reps=reps, seed=seed)]
        res[name] = dict(score=score_prediction(pred, y, strata, W), vs_old=paired_bootstrap(pred, old, y, strata, W, n_boot, seed),
                         repetition_f1_range=[float(np.nanmin(reps_f1)), float(np.nanmedian(reps_f1)), float(np.nanmax(reps_f1))],
                         n_predicted_loss=int(pred.sum()))
    m = res["six_features"]["vs_old"]
    res["adopt"] = bool(m["lo"] > 0 and m["diff"] >= 0.10)
    res["claim_rule_met"] = bool(res["six_features"]["score"]["ua"] >= 0.70 and res["six_features"]["score"]["pa"] >= 0.50)
    (OUTPUTS / "vegetation" / "vegetation_v2_cv.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=float))
