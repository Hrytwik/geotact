"""T6e: ONE-SHOT test of the frozen susceptibility model against the sanitation-plan waterlogging hotspots.

Protocol (docs/build_plan.md, registered before any model existed): for each of the 16 hotspots take the maximum
susceptibility percentile within 150 m of its point(s); percentiles are ranks over all BMC cells except mapped water.
AMENDED 2026-10-21, BEFORE the real test was run: the registered aggregator was the MAXIMUM percentile in the 150 m disk, but a 150 m
disk holds ~81 cells, so on a random score that statistic averages 98.6 and the >= 70 bar is vacuous. The aggregator is now the
MEAN percentile within the disk (random score -> about 50); the bar (>= 70), p < 0.05, the null and the baselines are unchanged.
Statistic = mean over the 16. Null = 5000 random sets with the same number of points per hotspot and the same
aggregation. PASS if the model's mean percentile >= 70 and one-sided permutation p < 0.05. Baselines use the same
procedure on single layers (low elevation, low HAND, high TWI, near drain). Failure is reported, not repaired.
"""
import json

import numpy as np
from scipy.ndimage import convolve
from scipy.stats import rankdata

from src.config import INTERIM, OUTPUTS

OUT = OUTPUTS / "flood"
RADIUS_M = 150.0
PASS_MEAN_PERCENTILE, PASS_P = 70.0, 0.05


def percentile_raster(score, ref):
    """Percentile rank (0-100) of each reference cell among the finite reference cells; NaN elsewhere."""
    score = np.asarray(score, dtype=float)
    valid = np.asarray(ref, bool) & np.isfinite(score)
    out = np.full(score.shape, np.nan)
    out[valid] = rankdata(score[valid], method="average") / valid.sum() * 100.0
    return out


def disk_mean(p, radius_cells):
    """Mean of the finite values of p over a disk of the given radius (in cells) around every cell; NaN if the disk is empty."""
    r = int(np.floor(radius_cells))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    footprint = ((x ** 2 + y ** 2) <= radius_cells ** 2).astype(float)
    finite = np.isfinite(p)
    total = convolve(np.where(finite, p, 0.0), footprint, mode="constant", cval=0.0)
    count = convolve(finite.astype(float), footprint, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0.5, total / count, np.nan)


def hotspot_values(dm, points_by_hotspot):
    return np.array([np.nanmax([dm[r, c] for r, c in pts]) for pts in points_by_hotspot], dtype=float)


def null_means(dm, ref, counts, n_perm=5000, seed=0):
    """Mean percentile of n_perm random hotspot sets; hotspot i receives counts[i] random reference cells (max over them)."""
    rng = np.random.default_rng(seed)
    cells = np.flatnonzero(np.asarray(ref, bool).ravel() & np.isfinite(dm.ravel()))
    counts = np.asarray(counts)
    idx = rng.integers(len(cells), size=(n_perm, counts.sum()))
    vals = dm.ravel()[cells[idx]]
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    return np.maximum.reduceat(vals, starts, axis=1).mean(axis=1)


def permutation_test(values, dm, ref, counts, n_perm=5000, seed=0):
    obs = float(np.mean(values))
    null = null_means(dm, ref, counts, n_perm, seed)
    return dict(mean_percentile=obs, p=float((1 + (null >= obs).sum()) / (n_perm + 1)), null_mean=float(null.mean()),
                null_sd=float(null.std()), n_perm=int(n_perm))


def bootstrap_ci(values, n=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def load_hotspot_cells():
    """Hotspot points as (row, col) cells on the 30 m DEM grid, grouped by hotspot WL1..WL16."""
    import geopandas as gpd
    import rasterio
    from rasterio.transform import rowcol
    with rasterio.open(OUTPUTS / "terrain" / "terrain_features.tif") as r:
        tf, shape = r.transform, (r.height, r.width)
    g = gpd.read_file(INTERIM / "hotspots.gpkg")
    names = sorted(g.hotspot.unique(), key=lambda s: int(s[2:]))
    pts = []
    for n in names:
        sub = g[g.hotspot == n]
        rows, cols = rowcol(tf, sub.geometry.x.values, sub.geometry.y.values)
        pts.append([(int(r), int(c)) for r, c in zip(rows, cols)])
    return dict(names=names, points_by_hotspot=pts, shape=shape, transform=tf)


def run(n_perm=5000, seed=0):
    import rasterio

    from src import terrain as tr
    OUT.mkdir(parents=True, exist_ok=True)
    h = load_hotspot_cells()
    d = tr.real_inputs()
    with rasterio.open(OUTPUTS / "terrain" / "terrain_features.tif") as r:
        terr = {n: r.read(i + 1) for i, n in enumerate(r.descriptions)}
    sus = rasterio.open(OUT / "susceptibility_30m.tif").read(1)
    radius_cells = RADIUS_M / abs(h["transform"].a)
    scores = {"model": sus, "low_elevation": -terr["elevation"], "low_HAND": -terr["hand"], "high_TWI": terr["twi"],
              "near_drain": -terr["dist_drain"]}
    counts = [len(p) for p in h["points_by_hotspot"]]
    res, table = {}, {"hotspot": h["names"]}
    for name, score in scores.items():
        ref = d["bmc"] & ~d["water"] & np.isfinite(score)
        dm = disk_mean(percentile_raster(score, ref), radius_cells)
        v = hotspot_values(dm, h["points_by_hotspot"])
        res[name] = dict(**permutation_test(v, dm, ref, counts, n_perm, seed), ci95=bootstrap_ci(v, seed=seed),
                         median_percentile=float(np.median(v)), share_top20pct=float((v >= 80).mean()), n_hotspots=len(v))
        table[name] = v.round(1).tolist()
        print(f"{name:14s} mean percentile {res[name]['mean_percentile']:6.1f}  95% CI [{res[name]['ci95'][0]:5.1f}, "
              f"{res[name]['ci95'][1]:5.1f}]  null {res[name]['null_mean']:5.1f}  p = {res[name]['p']:.4f}  "
              f"hotspots in top 20%: {int((np.array(table[name]) >= 80).sum())}/16", flush=True)
    m = res["model"]
    res["PASS"] = bool(m["mean_percentile"] >= PASS_MEAN_PERCENTILE and m["p"] < PASS_P)
    res["criterion"] = f"model mean percentile >= {PASS_MEAN_PERCENTILE} and p < {PASS_P}"
    best_base = max((k for k in scores if k != "model"), key=lambda k: res[k]["mean_percentile"])
    res["best_baseline"] = best_base
    diff = np.array(table["model"]) - np.array(table[best_base])
    res["model_minus_best_baseline"] = dict(mean=float(diff.mean()), ci95=bootstrap_ci(diff, seed=seed))
    import pandas as pd
    pd.DataFrame(table).to_csv(OUT / "hotspot_test_table.csv", index=False)
    (OUT / "hotspot_test.json").write_text(json.dumps(res, indent=1))
    print("PASS" if res["PASS"] else "FAIL", "|", res["criterion"], "| best baseline:", best_base,
          "| model minus best baseline:", round(res["model_minus_best_baseline"]["mean"], 1), flush=True)
    return res


if __name__ == "__main__":
    run()
