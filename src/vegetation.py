"""T5: vegetation health over time. Index-based (NDVI), not deep learning.

Eight November-March Sentinel-2 composites (2018-19 .. 2025-26) give one NDVI layer per season.
Unchanged land (same Esri class 2017-2023, not water) is used to estimate and remove a common
year-level shift (atmosphere / sensor / season; it moves unchanged land by +-0.05 NDVI between years),
then a per-pixel Theil-Sen slope with an exact Mann-Kendall p-value and a latest-year anomaly are computed.
"""
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import rasterio

from src import grid
from src.config import OUTPUTS, RAW

SEASONS = ["2018_19", "2019_20", "2020_21", "2021_22", "2022_23", "2023_24", "2024_25", "2025_26"]
YEARS = [int(s[:4]) for s in SEASONS]            # season start year
IO_YEARS = list(range(2017, 2024))
IO_WATER, IO_TREES, IO_CROPS, IO_BUILT, IO_RANGE = 1, 2, 5, 7, 11
OUT = OUTPUTS / "vegetation"


# ---------------------------------------------------------------- NDVI
def ndvi(red, nir):
    """NDVI as float32; NaN where either band is 0 (a composite stores 0 for 'no clear day')."""
    red = np.asarray(red, dtype="float32")
    nir = np.asarray(nir, dtype="float32")
    valid = (red > 0) & (nir > 0)
    out = np.full(red.shape, np.nan, dtype="float32")
    out[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
    return out


# ---------------------------------------------------------------- exact Mann-Kendall
@lru_cache(maxsize=None)
def mk_null(n):
    """Exact null distribution of Kendall's S for n untied observations.

    S = n(n-1)/2 - 2 * (number of inversions); inversion counts are the Mahonian numbers.
    Returns (s ascending, probability)."""
    counts = np.array([1.0])
    for k in range(2, n + 1):
        counts = np.convolve(counts, np.ones(k))
    m = n * (n - 1) // 2
    s = m - 2 * np.arange(len(counts))
    return s[::-1].astype(int), (counts / counts.sum())[::-1]


@lru_cache(maxsize=None)
def _p_table(n):
    """Two-sided exact p for every possible S, indexed by (S + n(n-1)/2) // 2."""
    s, p = mk_null(n)
    return np.array([p[np.abs(s) >= abs(v)].sum() for v in s])


def sen_mk(stack, min_valid=6, block_rows=100):
    """Per-pixel Theil-Sen slope, Kendall S, exact two-sided p, valid count. stack is (T, H, W), NaN = missing.

    Pixels with fewer than min_valid valid years get NaN slope/S/p. Slope is per time step (per year)."""
    T, H, W = stack.shape
    pairs = [(i, j) for i in range(T) for j in range(i + 1, T)]
    slope = np.full((H, W), np.nan)
    S = np.full((H, W), np.nan)
    p = np.full((H, W), np.nan)
    n_valid = np.isfinite(stack).sum(0).astype("int16")
    for r0 in range(0, H, block_rows):
        x = stack[:, r0:r0 + block_rows, :].astype("float64")
        s_blk = np.zeros(x.shape[1:])
        sl = np.full((len(pairs),) + x.shape[1:], np.nan)
        for k, (i, j) in enumerate(pairs):
            d = x[j] - x[i]
            sl[k] = d / (j - i)
            s_blk += np.where(np.isfinite(d), np.sign(d), 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(sl, axis=0)
        ok = n_valid[r0:r0 + block_rows] >= min_valid
        slope[r0:r0 + block_rows] = np.where(ok, med, np.nan)
        S[r0:r0 + block_rows] = np.where(ok, s_blk, np.nan)
    for n in np.unique(n_valid[n_valid >= min_valid]):
        sel = n_valid == n
        m = int(n) * (int(n) - 1) // 2
        p[sel] = _p_table(int(n))[((S[sel] + m) // 2).astype(int)]
    return slope, S, p, n_valid


# ---------------------------------------------------------------- year normalisation
def estimate_year_offsets(stack, stable):
    """Common additive year shift: median over unchanged pixels of (NDVI_year - pixel mean over years),
    centred so the overall NDVI level is kept."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pixel_mean = np.nanmean(stack, axis=0)
        off = np.array([np.nanmedian(stack[y][stable] - pixel_mean[stable]) for y in range(stack.shape[0])])
    return off - off.mean()


def normalise(stack, offsets):
    return (stack - np.asarray(offsets, dtype=stack.dtype)[:, None, None]).astype(stack.dtype)


# ---------------------------------------------------------------- anomaly
def anomaly(stack, floor=0.02, min_prior=5):
    """z-score of the latest year against the previous years; the std is floored to avoid blow-ups."""
    last, prior = stack[-1], stack[:-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(prior, axis=0)
        sd = np.maximum(np.nanstd(prior, axis=0, ddof=1), floor)
    z = (last - mean) / sd
    return np.where(np.isfinite(prior).sum(0) >= min_prior, z, np.nan).astype("float32")


# ---------------------------------------------------------------- data loading
def load_composites(step=1):
    """(raw NDVI stack, red-band medians inside the study area); every `step`-th pixel if step > 1."""
    m = grid.study_mask()
    nd, red_med = [], []
    for s in SEASONS:
        a = grid.read_aligned(RAW / f"s2_ndvi_{s}.tif")            # bands: B04, B08
        v = ndvi(a[0], a[1])
        v[~m] = np.nan
        r = a[0][m]
        red_med.append(float(np.median(r[r > 0])))
        nd.append(v[::step, ::step])
    return np.stack(nd), red_med


def land_cover_masks(step=1):
    """Esri annual classes 2017-2023 -> masks used for normalisation and for checks."""
    io = np.stack([grid.read_aligned(RAW / f"io_lulc_{y}.tif", resampling="nearest", dtype="uint8")[::step, ::step]
                   for y in IO_YEARS])
    m = grid.study_mask()[::step, ::step]
    unchanged = (io == io[0]).all(0)
    stable = unchanged & (io[0] != IO_WATER) & m
    to_built = (io[3:] == IO_BUILT).all(0) & m                              # built in every year 2020-2023
    converted_trees = (io[0] == IO_TREES) & to_built
    converted_crops = (io[0] == IO_CROPS) & to_built
    converted_range = (io[0] == IO_RANGE) & to_built
    stable_built = unchanged & (io[0] == IO_BUILT) & m
    stable_trees = unchanged & (io[0] == IO_TREES) & m
    return dict(stable=stable, converted_trees=converted_trees, converted_crops=converted_crops,
                converted_range=converted_range, stable_built=stable_built, stable_trees=stable_trees)


def load_real_sample(step=4):
    raw, red_med = load_composites(step)
    masks = land_cover_masks(step)
    off = estimate_year_offsets(raw, masks["stable"])
    return dict(raw=raw, norm=normalise(raw, off), offsets=off, red_median=red_med, **masks)


# ---------------------------------------------------------------- full run
def run():
    OUT.mkdir(parents=True, exist_ok=True)
    g = grid.load_grid()
    raw, red_med = load_composites(1)
    masks = land_cover_masks(1)
    off = estimate_year_offsets(raw, masks["stable"])
    norm = normalise(raw, off)
    slope, S, p, n = sen_mk(norm)
    z = anomaly(norm)
    prof = dict(driver="GTiff", height=g.shape[0], width=g.shape[1], crs=g.crs, transform=g.transform,
                compress="deflate", tiled=True, nodata=np.nan, dtype="float32")
    with rasterio.open(OUT / "ndvi_norm_stack.tif", "w", **{**prof, "count": len(SEASONS)}) as dst:
        dst.write(norm)
        dst.descriptions = tuple(SEASONS)
    for name, arr in [("ndvi_slope", slope), ("ndvi_trend_p", p), ("ndvi_anomaly_z", z)]:
        with rasterio.open(OUT / f"{name}.tif", "w", **{**prof, "count": 1}) as dst:
            dst.write(arr.astype("float32"), 1)
    pd.DataFrame(dict(season=SEASONS, offset=off, red_median=red_med,
                      raw_city_median=np.nanmedian(raw.reshape(len(SEASONS), -1), axis=1),
                      norm_city_median=np.nanmedian(norm.reshape(len(SEASONS), -1), axis=1))).to_csv(
        OUT / "year_offsets.csv", index=False)
    summaries(norm, slope, p, z, masks)
    return dict(offsets=off, slope=slope, p=p, z=z)


def summaries(norm, slope, p, z, masks):
    m = grid.study_mask()
    wards = grid.ward_raster()
    wc = grid.read_aligned(RAW / "worldcover_2021.tif", resampling="nearest", dtype="uint8")
    vegetated = np.isin(wc, [10, 20, 30, 40]) & m                        # tree, shrub, grass, crop (2021)
    sig_down = (p < 0.05) & (slope < 0)
    rows = []
    for w in range(1, 68):
        sel = (wards == w) & vegetated
        if sel.sum() < 50:
            rows.append(dict(ward=w, vegetated_ha=sel.sum() / 100))
            continue
        row = dict(ward=w, vegetated_ha=round(sel.sum() / 100, 1), median_slope_per_yr=float(np.nanmedian(slope[sel])),
                   pct_sig_decline=float(100 * np.nanmean(sig_down[sel])), mean_anomaly_z=float(np.nanmean(z[sel])))
        for i, s in enumerate(SEASONS):
            row[f"ndvi_{s}"] = float(np.nanmean(norm[i][sel]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "ward_summary.csv", index=False)
    classes = {"trees": 10, "shrub": 20, "grass": 30, "crops": 40, "built": 50, "bare": 60}
    rows = [dict(cls=k, px=int(((wc == c) & m).sum()), median_slope_per_yr=float(np.nanmedian(slope[(wc == c) & m])),
                 pct_sig_decline=float(100 * np.nanmean(sig_down[(wc == c) & m]))) for k, c in classes.items()]
    pd.DataFrame(rows).to_csv(OUT / "landcover_summary.csv", index=False)
    groups = {"trees -> built (2017 trees, built 2020-23)": "converted_trees", "crops -> built": "converted_crops",
              "rangeland -> built": "converted_range", "stable built": "stable_built", "stable trees": "stable_trees"}
    first, last = np.nanmean(norm[:2], axis=0), np.nanmean(norm[-2:], axis=0)
    rows = []
    for name, key in groups.items():
        g = masks[key] & np.isfinite(slope)
        rows.append(dict(group=name, px=int(g.sum()), median_slope_per_yr=float(np.median(slope[g])),
                         pct_sig_decline=float(100 * np.mean(sig_down[g])), ndvi_first2=float(np.nanmean(first[g])),
                         ndvi_last2=float(np.nanmean(last[g]))))
    pd.DataFrame(rows).to_csv(OUT / "validation_conversion.csv", index=False)


if __name__ == "__main__":
    from src.wardmap import memory_guard
    memory_guard(4.0)
    run()
    print(pd.read_csv(OUT / "validation_conversion.csv").to_string(index=False))
