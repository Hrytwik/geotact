"""T18: the T4c land-use correction model retrained with the T15 vegetation-check points as extra TRAINING labels (protocol in
docs/results_log.md). Scored only on the 300 designed labels, with the T4c folds; extras inside a test fold's 4 km blocks are left out."""
import json

import numpy as np
import pandas as pd

BLOCK_M = 4000.0


def allowed_extras(test_xy, extra_xy, block_m=BLOCK_M):
    """True for extra points whose block holds no test point."""
    tb = {tuple(b) for b in np.floor(np.asarray(test_xy) / block_m).astype(int)}
    eb = np.floor(np.asarray(extra_xy) / block_m).astype(int)
    return np.array([tuple(b) not in tb for b in eb], bool)


def cv_predict_extra(F, y, folds, xy, Fx, yx, xyx, make):
    """T4c out-of-fold loop with the allowed extra rows appended to every training fold."""
    F, y, folds = np.asarray(F), np.asarray(y), np.asarray(folds)
    Fx, yx = np.asarray(Fx).reshape(len(yx), -1) if len(yx) else np.empty((0, F.shape[1])), np.asarray(yx, int)
    oof = np.empty(len(y), dtype=y.dtype)
    for k in np.unique(folds):
        te = folds == k
        keep = allowed_extras(np.asarray(xy)[te], xyx) if len(yx) else np.zeros(0, bool)
        m = make()
        m.fit(np.vstack([F[~te], Fx[keep]]), np.r_[y[~te], yx[keep]])
        oof[te] = m.predict(F[te])
    return oof


def extra_points():
    """T15 rows with confidence Sure and a six-class now_covers, with their 10 m pixel and UTM centre."""
    from src.config import ROOT
    from src import landuse_accuracy as la
    out = ROOT / "validation" / "vegetation_check"
    s = pd.read_excel(out / "vegetation_labels.xlsx", sheet_name="labels")
    key = pd.read_csv(out / "vegetation_key_private_do_not_open.csv")
    s = s[s.confidence.astype(str).str.strip().str.lower() == "sure"]
    lk = {c.lower(): c for c in la.CLASSES}
    s = s.assign(cls=s.now_covers.astype(str).str.strip().str.lower().map(lk)).dropna(subset=["cls"])
    return s[["id", "cls"]].merge(key[["id", "row", "col"]], on="id")


def run(n_boot=2000):
    from src import landuse_accuracy as la
    from src import landuse_improve as li
    from src.config import OUTPUTS
    L = li._layers()
    F, y, folds = li.correction_training(L)
    d, W = la.load_inputs()
    xy = np.c_[d.x_utm.values, d.y_utm.values]
    e = extra_points()
    rr, cc = e.row.values, e.col.values
    Fx = li.features_from(L["probs"][:, rr, cc], L["esri"][rr, cc], L["wc"][rr, cc], L["nd"][:, rr, cc], L["built"][rr, cc])
    yx = np.array([la.CLASSES.index(c) for c in e.cls])
    xyx = np.array([L["tf"] * (c + 0.5, r + 0.5) for r, c in zip(rr, cc)])
    base = li.cv_predict(F, y, folds, li.correction_model)
    new = cv_predict_extra(F, y, folds, xy, Fx, yx, xyx, li.correction_model)
    ref, s = d.ref.values, d.stratum.values
    to_cls = lambda a: np.array([la.CLASSES[k] for k in a])
    res = dict(n_extra=int(len(e)), extra_classes=e.cls.value_counts().to_dict(),
               extras_used_per_fold={int(k): int(allowed_extras(xy[folds == k], xyx).sum()) for k in np.unique(folds)})
    for name, oof in (("t4c", base), ("t18", new)):
        c = to_cls(oof)
        acc, se = la.stratified_accuracy(s, c == ref, W)
        res[name] = dict(overall=acc, ci=[acc - 1.96 * se, acc + 1.96 * se],
                         users=la.users_accuracy(s, c, ref, W, la.CLASSES), producers=la.producers_accuracy(s, c, ref, W, la.CLASSES))
    res["paired"] = la.paired_difference(s, to_cls(new) == ref, to_cls(base) == ref, W, n_boot)
    res["replace"] = bool(res["paired"]["diff"] >= 0.03 and res["paired"]["ci"][0] > 0)
    (OUTPUTS / "landuse" / "t18_scores.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=float))
