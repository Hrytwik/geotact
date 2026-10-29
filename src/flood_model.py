"""T6d: monsoon surface-water susceptibility model on terrain and hydrology features (protocol in docs/results_log.md).

Training unit: 30 m cells on the DEM grid. Label: monsoon surface-water frequency (NOT flood extent: the radar
label follows the paddy calendar, not the rain). Built-up share is deliberately not a feature. LightGBM runs only in
run(); LightGBM and PyTorch must never share a process (libomp), so nothing here imports torch.
"""
import json

import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.config import OUTPUTS, RAW

OUT = OUTPUTS / "flood"
TAU = 0.12                       # F >= 3 of 25 scenes
MIN_OPEN, MIN_POS = 0.5, 0.30    # cell eligibility and positive share (registered)
BLOCK = 67                       # 67 cells x 30 m = 2 km spatial blocks
SIGNS = {"elevation": -1, "slope_deg": -1, "laplacian": +1, "log10_acc_m2": +1, "twi": +1, "hand": -1,
         "dist_drain": -1, "dist_river": -1, "dist_water": -1, "dist_channel": -1}   # physical direction of 'more water'


# ---------------------------------------------------------------- grids and labels
def to_grid(arr, src_transform, dst_transform, dst_shape, crs, resampling=Resampling.average):
    """Resample an array onto another grid (default: area average) with the same CRS."""
    out = np.full(dst_shape, np.nan, dtype="float32")
    reproject(np.asarray(arr, dtype="float32"), out, src_transform=src_transform, src_crs=crs, dst_transform=dst_transform,
              dst_crs=crs, resampling=resampling)
    return out


def cell_labels(open_frac, pos_frac, min_open=MIN_OPEN, min_pos=MIN_POS):
    open_frac, pos_frac = np.asarray(open_frac, float), np.asarray(pos_frac, float)
    eligible = open_frac >= min_open
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(open_frac > 0, pos_frac / open_frac, 0.0)
    return (eligible & (share >= min_pos)).astype(int), eligible


def block_folds(shape, block=BLOCK, k=5, seed=0):
    """Fold id per cell; whole square blocks are assigned to folds at random, so neighbours share a fold."""
    H, W = shape
    nby, nbx = -(-H // block), -(-W // block)
    ids = np.random.default_rng(seed).permutation(nby * nbx) % k
    rows = (np.arange(H) // block)[:, None]
    cols = (np.arange(W) // block)[None, :]
    return ids.reshape(nby, nbx)[rows, cols]


# ---------------------------------------------------------------- evaluation
def spatial_cv(X, y, folds, make_model, min_per_class=5):
    """Leave-one-fold-out over spatial folds. Returns out-of-fold probabilities, pooled and per-fold AUC / AP."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    oof = np.full(len(y), np.nan)
    fold_auc, fold_ap = [], []
    for k in np.unique(folds):
        te = folds == k
        model = make_model()
        model.fit(X[~te], y[~te])
        oof[te] = model.predict_proba(X[te])[:, 1]
        if min(y[te].sum(), (1 - y[te]).sum()) >= min_per_class:
            fold_auc.append(float(roc_auc_score(y[te], oof[te])))
            fold_ap.append(float(average_precision_score(y[te], oof[te])))
    return dict(oof=oof, auc=float(roc_auc_score(y, oof)), ap=float(average_precision_score(y, oof)),
                fold_auc=fold_auc, fold_ap=fold_ap)


def single_feature_aucs(X, y, names, signs=SIGNS):
    from sklearn.metrics import roc_auc_score
    out = {}
    for j, n in enumerate(names):
        v = signs.get(n, 1) * X[:, j]
        ok = np.isfinite(v)
        out[n] = float(roc_auc_score(y[ok], v[ok]))
    return out


def auc_within(score, y, groups):
    from sklearn.metrics import roc_auc_score
    return {name: float(roc_auc_score(y[m], score[m])) for name, m in groups.items() if 0 < y[m].sum() < m.sum()}


# ---------------------------------------------------------------- real run (own process: LightGBM, no torch)
def lgbm_factory(seed=0):
    def make():
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=50, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4, verbose=-1, random_state=seed)
    return make


def build_dataset(terrain_stack=None, names=None):
    """Cells on the DEM grid: features, label, eligibility, spatial folds, cropland share."""
    import rasterio
    from scipy.ndimage import binary_dilation

    from src import grid
    with rasterio.open(OUTPUTS / "terrain" / "terrain_features.tif") as r:
        stack, names0, tf, crs = r.read(), list(r.descriptions), r.transform, r.crs
    if terrain_stack is not None:
        stack, names0 = terrain_stack, names
    shape30 = stack.shape[1:]
    F = rasterio.open(OUT / "flood_freq.tif").read(1)
    wc = grid.read_aligned(RAW / "worldcover_2021.tif", resampling="nearest", dtype="uint8")
    unet = rasterio.open(OUTPUTS / "landuse" / "landuse_2025_26_seed0.tif").read(1)
    built = binary_dilation((wc == 50) | (unet == 4), iterations=2)       # radar is blind in built-up: excluded from training
    water = (wc == 80) | (unet == 0)
    open10 = ~built & ~water & np.isfinite(F)
    pos10 = open10 & (F >= TAU - 1e-6)
    g10 = grid.load_grid()
    open_frac = to_grid(open10, g10.transform, tf, shape30, crs)
    pos_frac = to_grid(pos10, g10.transform, tf, shape30, crs)
    crop_frac = to_grid(open10 & (wc == 40), g10.transform, tf, shape30, crs)
    y, eligible = cell_labels(np.nan_to_num(open_frac), np.nan_to_num(pos_frac))
    X = stack.reshape(len(names0), -1).T.astype("float64")
    eligible = eligible.ravel() & np.isfinite(X[:, names0.index("elevation")])
    with np.errstate(invalid="ignore", divide="ignore"):
        is_crop = (np.nan_to_num(crop_frac) / np.maximum(np.nan_to_num(open_frac), 1e-6) >= 0.5).ravel()
    return dict(X=X, names=names0, y=y.ravel(), eligible=eligible, folds=block_folds(shape30).ravel(), is_crop=is_crop,
                shape=shape30, transform=tf, crs=crs)


def run(seed=0, sensitivity=(100, 300, 1000)):
    import rasterio

    from src import terrain as tr
    OUT.mkdir(parents=True, exist_ok=True)
    d = build_dataset()
    el = d["eligible"]
    X, y, folds, names = d["X"][el], d["y"][el], d["folds"][el], d["names"]
    print(f"eligible cells {el.sum()} | positives {y.sum()} ({100 * y.mean():.2f}%) | cropland share of positives "
          f"{100 * d['is_crop'][el][y == 1].mean():.1f}%", flush=True)
    make = lgbm_factory(seed)
    cv = spatial_cv(X, y, folds, make)
    base = single_feature_aucs(X, y, names)
    bmc = tr.real_inputs()["bmc"].ravel()[el]
    within = auc_within(cv["oof"], y, {"cropland": d["is_crop"][el], "non_cropland": ~d["is_crop"][el],
                                       "inside_BMC": bmc, "outside_BMC": ~bmc})
    prev_bmc = dict(inside=float(y[bmc].mean()), outside=float(y[~bmc].mean()), cells_inside=int(bmc.sum()), pos_inside=int(y[bmc].sum()))
    print("label prevalence inside / outside BMC:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in prev_bmc.items()}, flush=True)
    print(f"spatial 5-fold CV: AUC {cv['auc']:.3f} (folds {np.round(cv['fold_auc'], 3).tolist()}) | AP {cv['ap']:.3f} "
          f"(prevalence {y.mean():.3f})", flush=True)
    print("single-feature AUC (physical direction):", {k: round(v, 3) for k, v in sorted(base.items(), key=lambda kv: -kv[1])}, flush=True)
    print("AUC within cropland / non-cropland cells:", {k: round(v, 3) for k, v in within.items()}, flush=True)
    sens = {}
    for T in sensitivity:
        out, dd, _ = tr.compute_real(T)
        ds = build_dataset(np.stack([out[b] for b in tr.BANDS]).astype("float32"), list(tr.BANDS))
        e2 = ds["eligible"]
        r = spatial_cv(ds["X"][e2], ds["y"][e2], ds["folds"][e2], make)
        sens[T] = dict(auc=r["auc"], ap=r["ap"], n_pos=int(ds["y"][e2].sum()))
        print(f"  stream threshold {T:5d} cells: CV AUC {r['auc']:.3f} AP {r['ap']:.3f}", flush=True)
    model = make()
    model.fit(X, y)
    imp = dict(zip(names, model.booster_.feature_importance("gain").round(1).tolist()))
    print("feature importance (gain):", dict(sorted(imp.items(), key=lambda kv: -kv[1])), flush=True)
    prob = np.full(d["X"].shape[0], np.nan, dtype="float32")
    ok = np.isfinite(d["X"][:, names.index("elevation")])
    prob[ok] = model.predict_proba(d["X"][ok])[:, 1]
    oof_full = np.full(d["X"].shape[0], np.nan, dtype="float32")
    oof_full[np.flatnonzero(el)] = cv["oof"]
    h, w = d["shape"]
    for name, arr in [("susceptibility_30m", prob), ("oof_30m", oof_full)]:
        with rasterio.open(OUT / f"{name}.tif", "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=d["crs"],
                           transform=d["transform"], nodata=np.nan, compress="deflate") as dst:
            dst.write(arr.reshape(h, w), 1)
    res = dict(n_cells=int(el.sum()), n_pos=int(y.sum()), prevalence=float(y.mean()), cv_auc=cv["auc"], cv_ap=cv["ap"],
               fold_auc=cv["fold_auc"], fold_ap=cv["fold_ap"], single_feature_auc=base, auc_within=within, prevalence_bmc=prev_bmc,
               stream_threshold_sensitivity=sens, importance_gain=imp, seed=seed, tau=TAU, features=names)
    (OUT / "metrics_model.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    from src.wardmap import memory_guard
    memory_guard(5.0)
    run()
