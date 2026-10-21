"""T6g: evaluation shared by the CNN and the baselines. numpy + scikit-learn only (no torch, no LightGBM)."""
import numpy as np


def evaluate(score, y, kind, n_boot=1000, seed=0):
    """Pooled AUC and average precision of `score` for flood (y=1) versus everything else, plus AUC against the field non-flood
    points alone; bootstrap 95 percent intervals resample samples within each kind."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    score, y, kind = np.asarray(score, float), np.asarray(y), np.asarray(kind)
    if not np.isfinite(score).all():
        raise ValueError("scores contain NaN or inf; fix them before evaluating")
    nf = (kind == "flood") | (kind == "nonflood")

    def point(s, yy, kk):
        nfm = (kk == "flood") | (kk == "nonflood")
        return roc_auc_score(yy, s), average_precision_score(yy, s), roc_auc_score(yy[nfm], s[nfm])

    auc, ap, auc_nf = point(score, y, kind)
    out = dict(auc=float(auc), ap=float(ap), auc_nonflood=float(auc_nf), n_flood=int((y == 1).sum()),
               n_background=int((kind == "background").sum()), n_nonflood=int((kind == "nonflood").sum()))
    if n_boot:
        rng = np.random.default_rng(seed)
        groups = [np.flatnonzero(kind == k) for k in np.unique(kind)]
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
            boots.append(point(score[idx], y[idx], kind[idx]))
        b = np.array(boots)
        for j, name in enumerate(("auc", "ap", "auc_nonflood")):
            out[f"{name}_ci"] = [float(np.percentile(b[:, j], 2.5)), float(np.percentile(b[:, j], 97.5))]
    return out


def paired_auc_diff(score_a, score_b, y, kind, n_boot=1000, seed=0):
    """Pooled AUC of model A minus model B, with a paired bootstrap 95 percent interval: both models are rescored on the SAME
    resampled samples (within kind), so the interval reflects the difference and not each model's own spread."""
    from sklearn.metrics import roc_auc_score
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    y, kind = np.asarray(y), np.asarray(kind)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("scores contain NaN or inf; fix them before evaluating")
    diff = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(kind == k) for k in np.unique(kind)]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return dict(diff=diff, ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if n_boot else [diff, diff])
