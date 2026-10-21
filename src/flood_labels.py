"""T6c: flood evidence per radar scene -> per-pixel flood frequency (protocol in docs/build_plan.md).

Change detection is always against the SAME orbit's dry-season scenes (two tracks: relative orbits 85 and 121).
Pure numpy/pandas apart from run(); nothing here imports torch."""
import json

import numpy as np
import pandas as pd

from src import flood_sar as fs
from src.flood_sar import flood_frequency  # noqa: F401  (stack version, re-exported)
from src.config import INTERIM, OUTPUTS, RAW

OUT = OUTPUTS / "flood"
MIN_BALANCED_ACCURACY = 0.80          # protocol: a scene whose water/land separation is worse than this is excluded
MIN_COVER = 0.95
MIN_VALID = 15                        # scenes that must have observed a pixel for its frequency to be defined


def scene_ok(balanced_accuracy):
    return balanced_accuracy >= MIN_BALANCED_ACCURACY


def group_by_orbit(df, min_cover=MIN_COVER):
    """{relative orbit: {"dry": [days], "monsoon": [days]}} for scenes that cover the study area."""
    d = df[df.cover >= min_cover].drop_duplicates(["kind", "day"])
    return {int(rel): {k: sorted(g[g.kind == k].day.tolist()) for k in ("dry", "monsoon")} for rel, g in d.groupby("rel")}


def flood_masks_for_orbit(dry_db, dry_thresholds, mon_db, mon_thresholds, slope=None, hand=None):
    """Flood masks for the monsoon scenes of one orbit, using that orbit's dry scenes as the reference.
    dry_db and mon_db are (n, H, W) stacks of filtered dB. Returns (list of masks, dry reference, dark frequency)."""
    ref = fs.dry_reference(dry_db)
    dark = fs.dark_frequency(dry_db, dry_thresholds)
    masks = [fs.flood_mask(m, t, ref, dark, slope=slope, hand=hand) for m, t in zip(mon_db, mon_thresholds)]
    return masks, ref, dark


def frequency_from_counts(flood_count, valid_count, min_valid=MIN_VALID):
    n = np.asarray(valid_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(n >= min_valid, np.asarray(flood_count) / np.maximum(n, 1), np.nan)
    return f.astype("float32"), n.astype("int16")


# ---------------------------------------------------------------- real run
def _load_db(kind, day, grid, size=3):
    return fs.speckle_filter(fs.to_db(grid.read_aligned(RAW / "s1" / f"s1_{kind}_{day}.tif", band=1)), size)


def run(min_valid=MIN_VALID):
    from src import grid
    OUT.mkdir(parents=True, exist_ok=True)
    g = group_by_orbit(pd.read_csv(INTERIM / "s1_orbits.csv"))
    wc = grid.read_aligned(RAW / "worldcover_2021.tif", resampling="nearest", dtype="uint8")
    water, land = wc == 80, np.isin(wc, [10, 20, 30, 40])
    terr = OUTPUTS / "terrain" / "terrain_features.tif"
    slope = grid.read_aligned(terr, band=2)
    hand = np.nan_to_num(grid.read_aligned(terr, band=6), nan=0.0)      # no drainage path found: not excluded
    bmc, built = grid.study_mask(), wc == 50
    shape = grid.load_grid().shape
    k_count, n_count = np.zeros(shape, np.int16), np.zeros(shape, np.int16)
    rows, packed, days = [], [], []
    for rel, d in sorted(g.items()):
        dry, dthr = [], []
        for day in d["dry"]:
            x = _load_db("dry", day, grid)
            t, b = fs.calibrate_threshold(x, water, land)
            rows.append(dict(kind="dry", orbit=rel, day=day, threshold_db=t, balanced_accuracy=b, used=scene_ok(b)))
            if scene_ok(b):
                dry.append(x); dthr.append(t)
        mon, mthr, mdays = [], [], []
        for day in d["monsoon"]:
            x = _load_db("monsoon", day, grid)
            t, b = fs.calibrate_threshold(x, water, land)
            ok = scene_ok(b)
            rows.append(dict(kind="monsoon", orbit=rel, day=day, threshold_db=t, balanced_accuracy=b, used=ok))
            if ok:
                mon.append(x); mthr.append(t); mdays.append(day)
        masks, ref, dark = flood_masks_for_orbit(np.stack(dry), dthr, np.stack(mon), mthr, slope=slope, hand=hand)
        for x, m, day in zip(mon, masks, mdays):
            valid = np.isfinite(x) & np.isfinite(ref)
            k_count += (m & valid)
            n_count += valid
            open_land = bmc & ~built & valid
            r = next(r for r in rows if r["kind"] == "monsoon" and r["day"] == day)
            r["flood_pct_bmc_open"] = float(100 * m[open_land].mean())
            packed.append(np.packbits(m)); days.append(f"{rel}:{day}")
        del dry, mon
    f, n = frequency_from_counts(k_count, n_count, min_valid)
    import rasterio
    gd = grid.load_grid()
    for name, arr, dt in [("flood_freq", f, "float32"), ("n_valid", n, "int16")]:
        with rasterio.open(OUT / f"{name}.tif", "w", driver="GTiff", height=shape[0], width=shape[1], count=1, dtype=dt,
                           crs=gd.crs, transform=gd.transform, nodata=np.nan if dt == "float32" else -1, compress="deflate") as dst:
            dst.write(arr, 1)
    np.savez_compressed(OUT / "flood_masks_packed.npz", masks=np.stack(packed), days=np.array(days), shape=np.array(shape))
    pd.DataFrame(rows).to_csv(OUT / "scene_table.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(dict(scenes_used=len(days), min_valid=min_valid, min_balanced_accuracy=MIN_BALANCED_ACCURACY)))
    return f, n


def rain_diagnostic():
    """Does flagged surface water follow the rain or the calendar? Per land-cover class: mean flagged area in June-August
    vs September-October scenes, and Spearman correlation with antecedent ERA5 rain (3 and 7 days) and with day of year."""
    from scipy.stats import spearmanr

    from src import grid
    z = np.load(OUT / "flood_masks_packed.npz")
    H, W = (int(v) for v in z["shape"])
    days = [d.split(":")[1] for d in z["days"]]
    wc = grid.read_aligned(RAW / "worldcover_2021.tif", resampling="nearest", dtype="uint8")
    bmc = grid.study_mask()
    cls = {"trees": wc == 10, "shrub": wc == 20, "grass": wc == 30, "cropland": wc == 40, "bare": wc == 60}
    rain = pd.read_csv(RAW / "rain_era5_daily.csv", parse_dates=["time"]).groupby("time").precipitation_sum.mean()
    rows = []
    for i in sorted(range(len(days)), key=lambda i: days[i]):
        m = np.unpackbits(z["masks"][i])[:H * W].reshape(H, W).astype(bool)
        d = pd.Timestamp(days[i])
        row = dict(day=days[i], rain3d=float(rain[d - pd.Timedelta(days=3):d].sum()),
                   rain7d=float(rain[d - pd.Timedelta(days=7):d].sum()), doy=d.dayofyear)
        for k, v in cls.items():
            row[k] = float(100 * m[bmc & v].mean())
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "scene_flag_by_class.csv", index=False)
    early, late = t[t.day < "2025-09-01"], t[t.day >= "2025-09-01"]
    out = []
    for k in cls:
        r3, r7, rc = spearmanr(t.rain3d, t[k]), spearmanr(t.rain7d, t[k]), spearmanr(t.doy, t[k])
        out.append(dict(cls=k, jun_aug_pct=early[k].mean(), sep_oct_pct=late[k].mean(), ratio=early[k].mean() / late[k].mean(),
                        rho_rain3d=r3.statistic, p_rain3d=r3.pvalue, rho_rain7d=r7.statistic, p_rain7d=r7.pvalue,
                        rho_calendar=rc.statistic, p_calendar=rc.pvalue))
    out = pd.DataFrame(out)
    out.to_csv(OUT / "rain_calendar_diagnostic.csv", index=False)
    return out


if __name__ == "__main__":
    from src.wardmap import memory_guard
    memory_guard(5.0)
    run()
