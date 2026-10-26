"""T4c: attempts to raise land-use accuracy (protocol in docs/build_plan.md): an ensemble of the four U-Nets, a multi-season
U-Net (2023 bands plus 8 NDVI seasons) and a correction model trained on the 300 hand labels with spatial cross-validation."""
import numpy as np


def ensemble(prob_list):
    """Mean of the models' class probabilities and the most probable class."""
    mean = np.mean([np.asarray(p, dtype="float32") for p in prob_list], axis=0)
    return mean.argmax(axis=0).astype("uint8"), mean


def stack_inputs(X, ndvi):
    """2023 bands followed by the NDVI seasons; missing NDVI is filled with that season's mean (0 is a real NDVI value)."""
    nd = np.asarray(ndvi, dtype="float32").copy()
    for b in range(nd.shape[0]):
        m = np.nanmean(nd[b])
        nd[b] = np.where(np.isfinite(nd[b]), nd[b], m if np.isfinite(m) else 0.0)
    return np.concatenate([np.asarray(X, dtype="float32"), nd], axis=0)


def point_folds(x, y, block_m=4000.0, k=5, seed=0):
    """Spatial folds for sample points: whole square blocks of block_m metres go to one fold."""
    keys = np.column_stack([np.floor(np.asarray(x) / block_m), np.floor(np.asarray(y) / block_m)]).astype(int)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    fold_of_block = np.random.default_rng(seed).permutation(len(uniq)) % k
    return fold_of_block[inv.ravel()]


def correction_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))


def cv_predict(F, y, folds, make):
    """Out-of-fold class predictions: each fold predicted by a model that never saw it."""
    F, y, folds = np.asarray(F), np.asarray(y), np.asarray(folds)
    oof = np.empty(len(y), dtype=y.dtype)
    for k in np.unique(folds):
        te = folds == k
        m = make()
        m.fit(F[~te], y[~te])
        oof[te] = m.predict(F[te])
    return oof


def features_from(probs, esri_cls, wc_cls, ndvi, built_share):
    """Per-sample features (columns): 6 ensemble class probabilities, Esri class one-hot (6), WorldCover class one-hot (6, all zero
    when unusable), 8 NDVI seasons, built-up share within 300 m. Inputs are (k, n) for probs/ndvi and (n,) for the rest."""
    probs, ndvi = np.asarray(probs, dtype="float32"), np.asarray(ndvi, dtype="float32")
    oh = lambda c: (np.asarray(c)[:, None] == np.arange(6)[None, :]).astype("float32")
    return np.hstack([probs.T, oh(esri_cls), oh(wc_cls), ndvi.T, np.asarray(built_share, dtype="float32")[:, None]])


# ---------------------------------------------------------------- real runs
ENSEMBLE_TAGS = ("seed0", "seed1", "cons_seed0", "cons_seed1")


def model_probs(tag, X10, extra="none", device=None):
    import torch

    from src import landuse as lu
    from src.config import HOME_OUTPUTS
    home = HOME_OUTPUTS / "landuse"                                   # trained models always come from Bhubaneswar
    st = np.load(home / lu.stats_path(tag, extra).name)
    Xs = lu.standardise(lu.add_extra(X10, extra), st["mean"], st["std"])
    model = lu.build_model(in_ch=Xs.shape[0])
    model.load_state_dict(torch.load(home / f"unet_{tag}.pt", map_location="cpu"))
    return lu.predict_probs(model, Xs, device=lu._device(device))


def _write(path, arr, like, dtype):
    import rasterio
    with rasterio.open(like) as r:
        meta = r.meta.copy()
    meta.update(count=1 if arr.ndim == 2 else arr.shape[0], dtype=dtype, compress="deflate", nodata=None)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr.astype(dtype), 1) if arr.ndim == 2 else dst.write(arr.astype(dtype))


def run_ensemble():
    from src import grid
    from src import landuse as lu
    from src.config import RAW
    like = lu.OUT / "landuse_2023_seed0.tif"
    for comp, yr in (("s2_2023", "2023"), ("s2_2025_26_dry", "2025_26")):
        X10 = (grid.read_aligned(RAW / f"{comp}.tif") / 10000.0).astype("float32")
        probs = [model_probs(t, X10) for t in ENSEMBLE_TAGS]
        cls, mean = ensemble(probs)
        _write(lu.OUT / f"landuse_{yr}_ens.tif", cls, like, "uint8")
        _write(lu.OUT / f"probs_{yr}_ens.tif", np.round(mean * 100), like, "uint8")
        agree = [float((p.argmax(0) == cls).mean()) for p in probs]
        print(f"ensemble {yr}: written; share of pixels where each member agrees with the ensemble {np.round(agree, 3).tolist()}", flush=True)


def score(candidates, n_boot=2000):
    """T7 estimator on the 300 hand labels for each candidate (name -> class array at the 300 points), plus the paired difference
    to the original seed-0 2023 map."""
    from src import landuse_accuracy as la
    d, W = la.load_inputs()
    ref, s = d.ref.values, d.stratum.values
    orig = la.sample_map(la_out() / "landuse_2023_seed0.tif", d.x_utm.values, d.y_utm.values)
    res = {}
    for name, cls in candidates.items():
        acc, se = la.stratified_accuracy(s, cls == ref, W)
        res[name] = dict(overall=acc, ci=[acc - 1.96 * se, acc + 1.96 * se], vs_original=la.paired_difference(s, cls == ref, orig == ref, W, n_boot),
                         users={c: v for c, v in la.users_accuracy(s, cls, ref, W, la.CLASSES).items()},
                         producers={c: v for c, v in la.producers_accuracy(s, cls, ref, W, la.CLASSES).items()})
    return res


def _layers(probs_path=None, ens_path=None):
    """Full-grid inputs of the correction model (same definitions at points and pixels). Defaults: the 2023 ensemble (T4c); the
    live twin passes the ensemble of a newer composite."""
    import rasterio
    from scipy import ndimage as ndi

    from src import grid
    from src import landuse as lu
    from src.config import RAW
    with rasterio.open(probs_path or lu.OUT / "probs_2023_ens.tif") as r:
        probs = r.read().astype("float32") / 100.0
        tf = r.transform
    esri = lu.map_labels(grid.read_aligned(RAW / "io_lulc_2023.tif", resampling="nearest", dtype="uint8"))
    wc = lu.wc_labels(grid.read_aligned(RAW / "worldcover_2021.tif", resampling="nearest", dtype="uint8"))
    nd = stack_inputs(np.zeros((0,) + esri.shape, "float32"), lu.ndvi_seasons())
    ens = rasterio.open(ens_path or lu.OUT / "landuse_2023_ens.tif").read(1)
    built = ndi.uniform_filter((ens == 4).astype("float32"), size=31, mode="nearest")
    return dict(probs=probs, esri=esri, wc=wc, nd=nd, built=built, tf=tf)


def correction_training(L):
    """Training table of the correction model: features at the 300 hand-labelled points, labels and the 4 km spatial folds."""
    import rasterio

    from src import landuse_accuracy as la
    d, W = la.load_inputs()
    rr, cc = rasterio.transform.rowcol(L["tf"], d.x_utm.values, d.y_utm.values)
    rr, cc = np.asarray(rr), np.asarray(cc)
    F = features_from(L["probs"][:, rr, cc], L["esri"][rr, cc], L["wc"][rr, cc], L["nd"][:, rr, cc], L["built"][rr, cc])
    y = np.array([la.CLASSES.index(c) for c in d.ref])
    return F, y, point_folds(d.x_utm.values, d.y_utm.values, 4000.0, 5, 0)


def save_correction_model(L=None):
    """Fit the T4c correction on all 300 points and save it, WITHOUT rewriting any map (live twin and transfer apply it)."""
    import pickle

    from src import landuse as lu
    F, y, _ = correction_training(L or _layers())
    model = correction_model().fit(F, y)
    with open(lu.OUT / "correction_model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    return model


def run_correction():
    from src import landuse_accuracy as la
    L = _layers()
    F, y, folds = correction_training(L)
    oof = cv_predict(F, y, folds, correction_model)
    print(f"correction: points per fold {np.bincount(folds).tolist()}; out-of-fold plain agreement {np.mean(oof == y):.3f}", flush=True)
    _apply(save_correction_model(L), L)
    return np.array([la.CLASSES[k] for k in oof])


def _apply(model, L, out_path=None):
    from src import landuse as lu
    H, Wd = L["esri"].shape
    out = np.zeros((H, Wd), dtype="uint8")
    for r0 in range(0, H, 200):
        r1 = min(r0 + 200, H)
        sl = (slice(r0, r1), slice(None))
        Fc = features_from(L["probs"][:, r0:r1].reshape(6, -1), L["esri"][sl].ravel(), L["wc"][sl].ravel(),
                           L["nd"][:, r0:r1].reshape(8, -1), L["built"][sl].ravel())
        out[r0:r1] = model.predict(Fc).reshape(r1 - r0, Wd)
    _write(out_path or lu.OUT / "landuse_2023_corr.tif", out, lu.OUT / "landuse_2023_seed0.tif", "uint8")


def transfer_city():
    """Another city (GEOTACT_CITY): per-model maps from the Bhubaneswar U-Nets, their ensemble, then the saved correction model."""
    import pickle

    from src import grid
    from src import landuse as lu
    from src.config import HOME_OUTPUTS, RAW
    lu.OUT.mkdir(parents=True, exist_ok=True)
    like = RAW / "s2_2023.tif"
    for comp, yr in (("s2_2023", "2023"), ("s2_2025_26_dry", "2025_26")):
        X10 = (grid.read_aligned(RAW / f"{comp}.tif") / 10000.0).astype("float32")
        probs = []
        for tg in ENSEMBLE_TAGS:
            P = model_probs(tg, X10)
            _write(lu.OUT / f"landuse_{yr}_{tg}.tif", P.argmax(0), like, "uint8")
            probs.append(P)
        cls, mean = ensemble(probs)
        _write(lu.OUT / f"landuse_{yr}_ens.tif", cls, like, "uint8")
        _write(lu.OUT / f"probs_{yr}_ens.tif", np.round(mean * 100), like, "uint8")
        print(f"transfer {yr}: ensemble written", flush=True)
    with open(HOME_OUTPUTS / "landuse" / "correction_model.pkl", "rb") as fh:
        model = pickle.load(fh)
    _apply(model, _layers())
    print("transfer: corrected map written", flush=True)


def score_round(extra_maps=()):
    import json

    from src import landuse_accuracy as la
    d, W = la.load_inputs()
    pts = lambda f: la.sample_map(la_out() / f, d.x_utm.values, d.y_utm.values)
    cands = {"original_2023_seed0": pts("landuse_2023_seed0.tif"), "ensemble_2023": pts("landuse_2023_ens.tif"),
             "ensemble_2025_26": pts("landuse_2025_26_ens.tif"), "correction_2023_cv": run_correction()}
    for name, f in extra_maps:
        cands[name] = pts(f)
    res = score(cands)
    f = la_out() / "improve_scores.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(res)
    f.write_text(json.dumps(old, indent=1, default=float))
    for k, v in res.items():
        dv = v["vs_original"]
        print(f"{k:22s} {100 * v['overall']:.1f}% [{100 * v['ci'][0]:.1f}, {100 * v['ci'][1]:.1f}]  vs original {100 * dv['diff']:+.1f} "
              f"[{100 * dv['ci'][0]:+.1f}, {100 * dv['ci'][1]:+.1f}] p {dv['p']:.3f}", flush=True)
    return res


def la_out():
    from src.config import OUTPUTS
    return OUTPUTS / "landuse"


if __name__ == "__main__":
    import sys

    from src.wardmap import memory_guard
    memory_guard(4.5)
    if "--transfer" in sys.argv:
        transfer_city()
    elif "--save-model" in sys.argv:
        save_correction_model()
        print("correction model saved (no map rewritten)", flush=True)
    elif "--ensemble" in sys.argv:
        run_ensemble()
    elif "--score" in sys.argv:
        extra = [("multiseason_2023_seed0", "landuse_2023_ms_seed0.tif"), ("multiseason_2023_seed1", "landuse_2023_ms_seed1.tif")] \
            if "--with-ms" in sys.argv else []
        score_round(extra)
