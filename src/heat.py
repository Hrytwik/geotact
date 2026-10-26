"""T14 heat: summer land-surface temperature explained by land-cover shares at three scales (OLS, spatial CV), and residents' heat
exposure (population-weighted mean surface temperature). The model is linear in shares and shares are linear filters of the map,
so an edit's exposure change is exact and additive (exposure_change_fast). No torch."""
import json

import numpy as np

COVER_OF = np.arange(256, dtype=np.uint8)
COVER_OF[6], COVER_OF[7], COVER_OF[8] = 3, 0, 1          # garden -> grass/scrub, restored wetland -> water, tree corridor -> trees
CLS = (0, 1, 2, 3, 5)                                       # water, trees, cropland, grass/scrub, bare (built-up = reference)
NAMES = ("water", "trees", "crop", "grass", "bare")
SCALES_PX = (1.5, 15.0, 50.0)                               # radii in 10 m pixels: the 30 m cell, 150 m, 500 m
FEATURES = [f"{n}_s{i}" for i in range(len(SCALES_PX)) for n in NAMES] + ["elevation"]


def as_cover(m):
    return COVER_OF[np.asarray(m, dtype=np.uint8)]


def features(classmap, elevation):
    from src.biodiversity import masked_disk_mean
    c = as_cover(classmap)
    out = [masked_disk_mean((c == k).astype("float32"), r) for r in SCALES_PX for k in CLS]
    return np.stack(out + [np.asarray(elevation, "float32")])


def predict(F, beta, intercept=0.0):
    return np.tensordot(np.asarray(beta, float), F, axes=1) + intercept


def exposure_change(edited, base, beta, w, elevation=None):
    """Full recompute: population-weighted mean of the predicted temperature change between two maps."""
    e = np.zeros(np.shape(base), "float32") if elevation is None else elevation
    d = predict(features(edited, e), beta) - predict(features(base, e), beta)
    return float((np.asarray(w) * d).sum() / np.asarray(w).sum())


def exposure_kernels(w):
    """G_s = disk-filter of (w / pixel count of the disk), per scale: the weight each pixel's cover change carries in the exposure."""
    from scipy.signal import fftconvolve

    from src.biodiversity import _disk
    w = np.asarray(w, "float32")
    out = []
    for r in SCALES_PX:
        K = _disk(r)
        cnt = fftconvolve(np.ones(w.shape, "float32"), K, mode="same")
        out.append(fftconvolve(w / np.maximum(cnt, 1e-6), K, mode="same").astype("float32"))
    return out


def exposure_change_fast(edited, base, beta, G, wsum):
    ce, cb = as_cover(edited), as_cover(base)
    diff = ce != cb
    total = 0.0
    for s, Gs in enumerate(G):
        g = Gs[diff]
        for j, k in enumerate(CLS):
            d = (ce[diff] == k).astype("float32") - (cb[diff] == k).astype("float32")
            total += float(beta[s * len(CLS) + j]) * float((d * g).sum())
    return total / float(wsum)


def sign_bounds(names=None):
    """Physical sign constraints relative to built-up: water, trees, grass/scrub cool (<= 0), bare ground warms (>= 0); the rest free."""
    names = FEATURES if names is None else names
    lo = [-np.inf] * len(names)
    hi = [np.inf] * len(names)
    for i, n in enumerate(names):
        if n.startswith(("water", "trees", "grass")):
            hi[i] = 0.0
        elif n.startswith("bare"):
            lo[i] = 0.0
    return np.array(lo), np.array(hi)


class BoundedOLS:
    """Least squares with box bounds on the coefficients (scipy bvls) and a free intercept; same interface as LinearRegression."""

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def fit(self, X, y):
        from scipy.optimize import lsq_linear
        X = np.asarray(X, float)
        mu, ym = X.mean(axis=0), float(np.mean(y))
        r = lsq_linear(X - mu, np.asarray(y, float) - ym, bounds=(self.lo, self.hi), method="bvls")
        self.coef_ = r.x
        self.intercept_ = ym - float(mu @ r.x)
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


def spatial_cv(F, y, mask, block_px=200, step=3, k=5, seed=0, bounds=None):
    """Pooled out-of-fold R2 on every step-th pixel inside the mask; folds = square blocks of block_px pixels. bounds=(lo, hi) per
    feature gives the sign-constrained fit, otherwise plain OLS."""
    from sklearn.linear_model import LinearRegression
    make = (lambda: LinearRegression()) if bounds is None else (lambda: BoundedOLS(*bounds))

    from src.landuse_improve import point_folds
    rr, cc = np.mgrid[step // 2:F.shape[1]:step, step // 2:F.shape[2]:step]
    rr, cc = rr.ravel(), cc.ravel()
    X, yy = F[:, rr, cc].T, np.asarray(y)[rr, cc]
    ok = np.asarray(mask)[rr, cc] & np.isfinite(X).all(axis=1) & np.isfinite(yy)
    X, yy, rr, cc = X[ok], yy[ok], rr[ok], cc[ok]
    folds = point_folds(cc.astype(float), rr.astype(float), float(block_px), k, seed)
    oof = np.empty(len(yy))
    for f in np.unique(folds):
        te = folds == f
        oof[te] = make().fit(X[~te], yy[~te]).predict(X[te])
    m = make().fit(X, yy)
    return dict(r2=float(1 - ((yy - oof) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()), n=int(len(yy)),
                beta=m.coef_.tolist(), intercept=float(m.intercept_))


# ---------------------------------------------------------------- real data
def inputs():
    from src import grid
    from src.config import RAW
    lst = np.nanmean([grid.read_aligned(RAW / f"lst_summer_{y}.tif") for y in ("2025", "2026")], axis=0)
    elev = grid.read_aligned(RAW / "copdem30.tif")
    pop = grid.read_aligned(RAW / "ghsl" / "GHS_POP_E2020_GLOBE_R2023A_54009_100_R7_C27" / "GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R7_C27.tif",
                            resampling="average")
    return lst, elev, np.nan_to_num(pop, nan=0.0).clip(min=0)


def run(map_path):
    import rasterio

    from src import grid
    from src.config import OUTPUTS
    out = OUTPUTS / "heat"
    out.mkdir(exist_ok=True)
    cm = rasterio.open(map_path).read(1)
    lst, elev, pop = inputs()
    mask = grid.study_mask()
    F = features(cm, elev)
    free = spatial_cv(F, lst, mask)                                          # unconstrained, reported only
    full = spatial_cv(F, lst, mask, bounds=sign_bounds())                    # T14 amendment 1: the model that is used
    onehot = np.stack([(as_cover(cm) == k).astype("float32") for k in CLS] + [elev.astype("float32")])
    base = spatial_cv(onehot, lst, mask)
    w = np.where(mask, pop, 0.0).astype("float32")
    res = dict(map=str(map_path), cv_r2=full["r2"], cv_r2_unconstrained=free["r2"], unconstrained_beta=dict(zip(FEATURES, free["beta"])), cv_r2_cell_class_only=base["r2"], n_cells=full["n"], beta=dict(zip(FEATURES, full["beta"])),
               intercept=full["intercept"], adopted=bool(full["r2"] >= 0.30),
               observed_exposure_c=float((w * np.nan_to_num(lst)).sum() / w[np.isfinite(lst)].sum()), population_weight_sum=float(w.sum()))
    (out / "heat_model.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "beta"}), "\n", {k: round(v, 3) for k, v in res["beta"].items()}, flush=True)
    return res


if __name__ == "__main__":
    import sys

    from src.wardmap import memory_guard
    memory_guard(4.0)
    run(sys.argv[1])
