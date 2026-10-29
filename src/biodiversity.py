"""T12b: bias-aware bird-habitat layer (protocol in docs/results_log.md). Units are birding visits (place + day); the response is whether
a visit's list holds a guild (AVONET habitat), effort is the number of other species on the list (List Length Analysis), habitat is
the land-use mix within 250 m. Comparing visits with visits removes most of the where-birders-go bias. No torch."""
import json

import numpy as np
import pandas as pd

GUILDS = {"wetland": {"Wetland", "Riverine", "Coastal"}, "tree": {"Forest", "Woodland"}}
RADIUS_M, PX_M = 250.0, 10.0
COVARIATES = ["share_water", "share_trees", "share_crop", "share_grass", "share_built", "share_bare", "log_dist_water", "ndvi"]


# ---------------------------------------------------------------- visits
def make_visits(rec, guild_of, decimals=4):
    """rec: species, lat, lon, date. One row per visit (rounded coordinates + date) with list length, guild flags and the effort
    of each guild (log1p of species on the list that are not in that guild)."""
    r = rec.dropna(subset=["species"]).copy()
    r["lat"], r["lon"] = r.lat.round(decimals), r.lon.round(decimals)
    r["guild"] = r.species.map(guild_of)
    out = []
    for (lat, lon, date), g in r.groupby(["lat", "lon", "date"], sort=False):
        sp = g.drop_duplicates("species")
        row = dict(lat=lat, lon=lon, date=date, n_species=len(sp))
        for name in GUILDS:
            inside = (sp.guild == name).sum()
            row[f"has_{name}"] = int(inside > 0)
            row[f"effort_{name}"] = float(np.log1p(len(sp) - inside))
        out.append(row)
    return pd.DataFrame(out)


def resolve_names(species, known, synonyms):
    """Species name -> AVONET name: exact match, else the first synonym that AVONET knows, else None."""
    res = {}
    for s in species:
        if s in known:
            res[s] = s
        else:
            res[s] = next((x for x in synonyms(s) if x in known), None)
    return res


def _get_json(url, params=None, attempts=4):
    import time

    import requests
    for a in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if a == attempts - 1:
                raise
            print(f"retry {url} after {type(e).__name__}", flush=True)
            time.sleep(10 * (a + 1))


def gbif_synonyms(name):
    """Accepted name and synonyms of a name in the GBIF backbone (canonical names)."""
    m = _get_json("https://api.gbif.org/v1/species/match", dict(name=name, kingdom="Animalia"))
    keys = {m.get("usageKey"), m.get("acceptedUsageKey")} - {None}
    names = []
    for k in keys:
        names.append(_get_json(f"https://api.gbif.org/v1/species/{k}").get("canonicalName"))
        res = _get_json(f"https://api.gbif.org/v1/species/{k}/synonyms", dict(limit=200)).get("results", [])
        names += [x.get("canonicalName") for x in res]
    return [n for n in names if n and n != name]


# ---------------------------------------------------------------- covariates
def _disk(radius_px):
    r = int(np.floor(radius_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ((yy ** 2 + xx ** 2) <= radius_px ** 2).astype("float32")


def masked_disk_mean(a, radius_px):
    """Mean of the finite values within a disk around every pixel; NaN where none."""
    from scipy.signal import fftconvolve
    a = np.asarray(a, "float32")
    fin = np.isfinite(a)
    k = _disk(radius_px)
    num = fftconvolve(np.where(fin, a, 0.0), k, mode="same")
    den = fftconvolve(fin.astype("float32"), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan).astype("float32")


def class_shares(classmap, radius_px, n=6):
    """Share of each class within the disk (pixels outside the image do not count; unusable codes are ignored)."""
    m = np.asarray(classmap)
    valid = m < n
    out = []
    for k in range(n):
        a = np.where(valid, (m == k).astype("float32"), np.nan)
        out.append(masked_disk_mean(a, radius_px))
    S = np.stack(out)
    return np.clip(S, 0.0, 1.0)


def covariate_stack(classmap, water, ndvi, radius_m=RADIUS_M, px_m=PX_M):
    from scipy.ndimage import distance_transform_edt
    rp = radius_m / px_m
    S = class_shares(classmap, rp)
    w = np.asarray(water, bool) | (np.asarray(classmap) == 0)
    dist = np.log1p(distance_transform_edt(~w) * px_m) if w.any() else np.full(w.shape, np.log1p(1e5))
    nd = masked_disk_mean(ndvi, rp)
    return np.concatenate([S, dist[None].astype("float32"), nd[None]])


# ---------------------------------------------------------------- evaluation
def decide(auc, auc_ci, diff_ci, bar=0.70):
    return bool(auc >= bar and diff_ci[0] > 0)


def location_bootstrap(score_a, score_b, y, loc, n_boot=1000, seed=0):
    """AUC of A with a 95% interval and the paired AUC difference A - B, resampling whole locations."""
    from sklearn.metrics import roc_auc_score
    a, b, y, loc = (np.asarray(v) for v in (score_a, score_b, y, loc))
    ids = np.unique(loc)
    rows = {k: np.flatnonzero(loc == k) for k in ids}
    rng = np.random.default_rng(seed)
    aucs, diffs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rows[k] for k in rng.choice(ids, len(ids), replace=True)])
        if 0 < y[idx].sum() < len(idx):
            aa = roc_auc_score(y[idx], a[idx])
            aucs.append(aa); diffs.append(aa - roc_auc_score(y[idx], b[idx]))
    return dict(auc=float(roc_auc_score(y, a)), auc_ci=[float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
                diff=float(roc_auc_score(y, a) - roc_auc_score(y, b)), diff_ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])


def logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def forest(seed=0):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=4, random_state=seed)


# ---------------------------------------------------------------- real run
def _water_mask(shape, transform):
    import geopandas as gpd
    from rasterio import features

    from src.config import CRS, INTERIM
    geoms = []
    for layer in ("wb", "river"):
        try:
            geoms += [g for g in gpd.read_file(INTERIM / "wardmap.gpkg", layer=layer).to_crs(CRS).geometry if g is not None and not g.is_empty]
        except Exception as e:                                                       # reported, not hidden
            print(f"water layer {layer} not used: {e}", flush=True)
    if not geoms:
        return np.zeros(shape, bool)
    return features.rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, dtype="uint8").astype(bool)


def stack_for_map(classmap, transform):
    import rasterio

    from src.config import OUTPUTS
    with rasterio.open(OUTPUTS / "vegetation" / "ndvi_norm_stack.tif") as r:
        ndvi = r.read(8).astype("float32")                                           # season 2025-26
    return covariate_stack(classmap, _water_mask(classmap.shape, transform), ndvi)


def guild_table():
    """AVONET habitat for every GBIF bird species (exact name, else GBIF synonyms); cached in data/interim."""
    import glob

    from src.config import INTERIM, RAW
    cache = INTERIM / "avonet_guilds.json"
    if cache.exists():
        return json.loads(cache.read_text())
    xl = RAW / "external" / "avonet" / "AVONET_Supplementary_dataset_1.xlsx"
    hab = {}
    for sheet, col in (("AVONET2_eBird", "Species2"), ("AVONET1_BirdLife", "Species1"), ("AVONET3_BirdTree", "Species3")):
        av = pd.read_excel(xl, sheet_name=sheet, usecols=[col, "Habitat"]).dropna(subset=[col])
        for s, h in zip(av[col], av.Habitat):
            hab.setdefault(s, h)                                                     # eBird taxonomy first, then BirdLife, BirdTree
    g = pd.read_csv(sorted(glob.glob(str(RAW / "gbif" / "*.csv")))[0], sep="\t", usecols=["class", "species"], low_memory=False)
    species = sorted(g[g["class"] == "Aves"].species.dropna().unique())

    def synonyms(name):
        try:
            return gbif_synonyms(name)
        except Exception as e:                                                       # an unreachable lookup leaves the name unmatched
            print(f"synonyms of {name} not resolved: {type(e).__name__}", flush=True)
            return []
    names = resolve_names(species, set(hab), synonyms)
    out = {s: dict(avonet=n, habitat=hab.get(n) if n else None,
                   guild=next((k for k, v in GUILDS.items() if n and hab.get(n) in v), None)) for s, n in names.items()}
    cache.write_text(json.dumps(out, indent=1))
    return out


def run(n_boot=1000, seed=0):
    import glob
    import pickle

    import rasterio
    from pyproj import Transformer

    from src import flood_cnn_baselines as bl
    from src import landuse_improve as li
    from src.config import CRS, OUTPUTS, RAW
    out = OUTPUTS / "biodiversity"
    out.mkdir(exist_ok=True)
    gt = guild_table()
    unmatched = sorted(s for s, v in gt.items() if v["avonet"] is None)
    g = pd.read_csv(sorted(glob.glob(str(RAW / "gbif" / "*.csv")))[0], sep="\t", low_memory=False,
                    usecols=["class", "species", "decimalLatitude", "decimalLongitude", "eventDate", "year"])
    g = g[(g["class"] == "Aves") & (g.year >= 2018)].rename(columns=dict(decimalLatitude="lat", decimalLongitude="lon"))
    g["date"] = g.eventDate.astype(str).str[:10]
    v = make_visits(g, {s: d["guild"] for s, d in gt.items()})
    with rasterio.open(OUTPUTS / "landuse" / "landuse_2023_corr.tif") as r:
        cm, tf = r.read(1), r.transform
    x, y = Transformer.from_crs(4326, CRS, always_xy=True).transform(v.lon.values, v.lat.values)
    rows, cols = rasterio.transform.rowcol(tf, x, y)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < cm.shape[0]) & (cols >= 0) & (cols < cm.shape[1])
    v, rows, cols, x, y = v[inside].reset_index(drop=True), rows[inside], cols[inside], np.asarray(x)[inside], np.asarray(y)[inside]
    S = stack_for_map(cm, tf)
    H = S[:, rows, cols].T
    ok = np.isfinite(H).all(axis=1)
    v, H, x, y = v[ok].reset_index(drop=True), H[ok], x[ok], y[ok]
    loc = pd.factorize(v.lat.astype(str) + "," + v.lon.astype(str))[0]
    folds = li.point_folds(x, y, 2000.0, 5, seed)
    res = dict(n_visits=int(len(v)), n_locations=int(loc.max() + 1), unmatched_species=unmatched,
               guild_species={k: sorted(s for s, d in gt.items() if d["guild"] == k) for k in GUILDS}, models={})
    lo, hi = np.nanmin(H, axis=0), np.nanmax(H, axis=0)
    for name in GUILDS:
        yv = v[f"has_{name}"].values
        eff = v[f"effort_{name}"].values[:, None]
        X = np.hstack([H, eff])
        sc = {"logistic": bl.cv_scores(X, yv, folds, logistic), "forest": bl.cv_scores(X, yv, folds, lambda: forest(seed)),
              "effort_only": bl.cv_scores(eff, yv, folds, logistic)}
        r = {k: location_bootstrap(s, sc["effort_only"], yv, loc, n_boot, seed) for k, s in sc.items() if k != "effort_only"}
        r["effort_only_auc"] = location_bootstrap(sc["effort_only"], sc["effort_only"], yv, loc, 50, seed)["auc"]
        p = r["logistic"]
        r["adopted"] = decide(p["auc"], p["auc_ci"], p["diff_ci"])
        r["prevalence"] = float(yv.mean())
        m = logistic().fit(X, yv)
        coef = dict(zip(COVARIATES + ["effort"], m[-1].coef_[0].round(3).tolist()))
        r["standardised_coefficients"] = coef
        res["models"][name] = r
        print(f"{name}: visits {len(yv)} prevalence {yv.mean():.2f} | logistic AUC {p['auc']:.3f} [{p['auc_ci'][0]:.3f}, {p['auc_ci'][1]:.3f}] "
              f"gain over effort-only {p['diff']:+.3f} [{p['diff_ci'][0]:+.3f}, {p['diff_ci'][1]:+.3f}] | forest AUC {r['forest']['auc']:.3f} "
              f"| effort-only {r['effort_only_auc']:.3f} | adopted: {r['adopted']} | coef {coef}", flush=True)
        if r["adopted"]:
            with open(out / f"model_{name}.pkl", "wb") as fh:
                pickle.dump(dict(model=m, effort_median=float(np.median(eff)), lo=lo, hi=hi), fh)
            suit, extrap = suitability(m, S, float(np.median(eff)), lo, hi)
            r["extrapolated_pixel_share"] = extrap
            meta = rasterio.open(OUTPUTS / "landuse" / "landuse_2023_corr.tif").meta.copy()
            meta.update(dtype="float32", nodata=np.nan, compress="deflate")
            with rasterio.open(out / f"suit_{name}.tif", "w", **meta) as dst:
                dst.write(suit, 1)
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def suitability(model, S, effort, lo, hi, chunk=500_000):
    """Predicted probability that a visit of the given effort records the guild, per pixel; share of pixels outside the visited
    covariate range (extrapolation)."""
    C, Hh, Ww = S.shape
    F = S.reshape(C, -1).T
    out = np.full(F.shape[0], np.nan, dtype="float32")
    ok = np.isfinite(F).all(axis=1)
    idx = np.flatnonzero(ok)
    for s in range(0, len(idx), chunk):
        i = idx[s:s + chunk]
        out[i] = model.predict_proba(np.hstack([F[i], np.full((len(i), 1), effort)]))[:, 1]
    extrap = float((((F[ok] < lo) | (F[ok] > hi)).any(axis=1)).mean())
    return out.reshape(Hh, Ww), extrap


def node_suitability(classmap, transform, pid, n):
    """Mean suitability per twin node for each adopted guild, recomputed on the given map (empty dict if no guild was adopted)."""
    import pickle

    from src import twin as tw
    from src.config import HOME_OUTPUTS
    models = {k: HOME_OUTPUTS / "biodiversity" / f"model_{k}.pkl" for k in GUILDS}
    models = {k: p for k, p in models.items() if p.exists()}
    if not models:
        return {}
    S = stack_for_map(classmap, transform)
    out = {}
    for k, p in models.items():
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        suit, _ = suitability(d["model"], S, d["effort_median"], d["lo"], d["hi"])
        out[k] = tw.zonal_mean(pid, suit, n)
    return out


if __name__ == "__main__":
    from src.wardmap import memory_guard
    memory_guard(4.0)
    run()
