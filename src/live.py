"""T12: live twin (protocol in docs/results_log.md). Each version = newest clear Sentinel-2 window -> land-use product (no retraining)
-> T8 twin -> change report against the previous version -> rainfall-forecast runoff outlook; versions are append-only.

Usage: python -m src.live --tag 2023 --composite 2023 --window 2023-01-01 2023-12-31
       python -m src.live --auto            (finds the newest clear window that is not a version yet)"""
import datetime as dt
import json

import numpy as np
import pandas as pd

from src import scenarios as sc

CLASSES = ["Water", "Trees", "Cropland", "Grass/Scrub", "Built-up", "Bare ground"]
CITY_CENTRE = (20.27, 85.82)                 # lat, lon (rain forecast point)
SEASON_GAP_DAYS = 60                         # median acquisition days further apart than this: 'different season'
PRODUCT = "T4c: 4 U-Net ensemble + logistic correction (2023-ensemble features, 300 hand labels)"   # recorded in every version


# ---------------------------------------------------------------- registry (append-only)
def load_registry(path):
    from pathlib import Path
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []


def register(path, entry):
    from pathlib import Path
    reg = load_registry(path)
    if any(v["tag"] == entry["tag"] for v in reg):
        raise FileExistsError(f"version {entry['tag']!r} already exists: versions are never overwritten")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(reg + [entry], indent=1, default=str))


def previous_tag(path, tag):
    tags = [v["tag"] for v in load_registry(path)]
    i = tags.index(tag) if tag in tags else len(tags)
    return tags[i - 1] if i > 0 else None


# ---------------------------------------------------------------- change report
def transitions(old, new, mask, px_ha=0.01):
    """Area (ha) moving from each old class (rows) to each new class (columns) inside the mask."""
    m = np.asarray(mask, bool)
    o, n = np.asarray(old)[m].astype(int), np.asarray(new)[m].astype(int)
    ok = (o < 6) & (n < 6)
    c = np.bincount(o[ok] * 6 + n[ok], minlength=36).reshape(6, 6) * px_ha
    return pd.DataFrame(c, index=CLASSES, columns=CLASSES)


def patch_change(old_pid, old_types, new_pid, new_types, min_share=0.10):
    """New patch: under min_share of it overlaps old patches of its type. Lost: the mirror. Persisting: new id -> old id of the
    same type with the largest overlap."""
    o, n = np.asarray(old_pid).ravel(), np.asarray(new_pid).ravel()
    both = (o > 0) & (n > 0)
    pairs, cnt = np.unique(np.column_stack([o[both], n[both]]), axis=0, return_counts=True)
    same = [(int(a), int(b), int(k)) for (a, b), k in zip(pairs, cnt) if old_types.get(int(a)) == new_types.get(int(b))]
    size_n = {int(k): int(v) for k, v in zip(*np.unique(n[n > 0], return_counts=True))}
    size_o = {int(k): int(v) for k, v in zip(*np.unique(o[o > 0], return_counts=True))}
    ov_n, ov_o, best = {}, {}, {}
    for a, b, k in same:
        ov_n[b] = ov_n.get(b, 0) + k
        ov_o[a] = ov_o.get(a, 0) + k
        if k > best.get(b, (None, -1))[1]:
            best[b] = (a, k)
    new = sorted(b for b in new_types if ov_n.get(b, 0) < min_share * size_n.get(b, 1))
    lost = sorted(a for a in old_types if ov_o.get(a, 0) < min_share * size_o.get(a, 1))
    persisting = {b: best[b][0] for b in sorted(new_types) if b not in new and b in best}
    return dict(new=new, lost=lost, persisting=persisting)


# ---------------------------------------------------------------- rain outlook
def fetch_forecast(lat=CITY_CENTRE[0], lon=CITY_CENTRE[1], days=7):
    import requests
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60,
                     params=dict(latitude=lat, longitude=lon, daily="precipitation_sum", timezone="Asia/Kolkata", forecast_days=days))
    r.raise_for_status()
    d = r.json()["daily"]
    return dict(time=d["time"], precipitation_sum=[float(v or 0.0) for v in d["precipitation_sum"]],
                fetched_at=dt.datetime.now().isoformat(timespec="minutes"), source="Open-Meteo forecast API", lat=lat, lon=lon)


def rain_outlook(classmap, mask, wards, forecast=None, hsg="C", pixel_m2=100.0, top=5):
    """SCS runoff for the wettest forecast day: city volume and the wards with the deepest runoff on this version's map."""
    fc = forecast or fetch_forecast()
    k = int(np.argmax(fc["precipitation_sum"]))
    P = float(fc["precipitation_sum"][k])
    cn = sc.CN[hsg]
    lut = np.full(256, np.nan)
    for c, v in cn.items():
        lut[c] = sc.scs_runoff_mm(P, v)
    m = np.asarray(mask, bool)
    q = lut[np.asarray(classmap, dtype=np.uint8)]
    w = np.asarray(wards)
    rows = []
    for wd in np.unique(w[m & (w > 0)]):
        sel = m & (w == wd) & np.isfinite(q)
        if sel.any():
            rows.append(dict(ward=int(wd), runoff_mm=float(q[sel].mean()), runoff_m3=float(q[sel].sum() / 1000.0 * pixel_m2)))
    rows.sort(key=lambda r: (-r["runoff_mm"], r["ward"]))
    return dict(wettest_day=fc["time"][k], rain_mm=P, soil_group=hsg, city_runoff_m3=sc.runoff_volume_m3(classmap, m, P, cn, pixel_m2=pixel_m2),
                top_wards=rows[:top], forecast={kk: fc[kk] for kk in fc})


# ---------------------------------------------------------------- automatic window
def latest_window(days, today, lookback_days=60, min_days=3, max_cloud=20.0, last_clear_registered=None):
    """(start, today): start = the first clear day within the lookback; None if too few clear days or none newer than the last
    registered version's newest clear day. days: (date, cloud %) per scene; a date's cloud is its clearest scene."""
    best = {}
    for d, c in days:
        best[d] = min(c, best.get(d, 101.0))
    clear = sorted(d for d, c in best.items() if c < max_cloud and today - dt.timedelta(days=lookback_days) <= d <= today)
    if len(clear) < min_days:
        return None
    if last_clear_registered is not None and clear[-1] <= last_clear_registered:
        return None
    return clear[0], today


def scene_dates(scene_ids):
    """Acquisition dates from Sentinel-2 item ids (S2X_MSIL2A_YYYYMMDDT...)."""
    return sorted(dt.datetime.strptime(s.split("_")[2][:8], "%Y%m%d").date() for s in scene_ids)


def season_label(dates_a, dates_b):
    """'same season' if the median day-of-year of the two scene sets is within SEASON_GAP_DAYS (circular), else 'different season'."""
    def med(ds):
        return float(np.median([d.timetuple().tm_yday for d in ds]))
    g = abs(med(dates_a) - med(dates_b))
    g = min(g, 365 - g)
    return ("same season" if g <= SEASON_GAP_DAYS else "different season: differences mix season and land change"), g


# ---------------------------------------------------------------- real pipeline
def classify(composite_path, out_dir):
    """Land-use product on a composite, no retraining: 4 U-Nets averaged (stored as whole percent, as in T4c), then the saved
    correction model on the T4c features of this composite."""
    import pickle

    from src import grid
    from src import landuse as lu
    from src import landuse_improve as li
    from src.config import HOME_OUTPUTS
    out_dir.mkdir(parents=True, exist_ok=True)
    like = HOME_OUTPUTS / "landuse" / "landuse_2023_seed0.tif"
    X10 = (grid.read_aligned(composite_path) / 10000.0).astype("float32")
    cls, mean = li.ensemble([li.model_probs(t, X10) for t in li.ENSEMBLE_TAGS])
    del X10
    li._write(out_dir / "ens.tif", cls, like, "uint8")
    li._write(out_dir / "probs_ens.tif", np.round(mean * 100), like, "uint8")
    del mean
    with open(HOME_OUTPUTS / "landuse" / "correction_model.pkl", "rb") as fh:
        model = pickle.load(fh)
    li._apply(model, li._layers(out_dir / "probs_ens.tif", out_dir / "ens.tif"), out_dir / "landuse.tif")
    return out_dir / "landuse.tif"


def compare_versions(old_dir, new_dir, mask):
    import rasterio
    rd = lambda d, f: rasterio.open(d / f).read(1)
    t = transitions(rd(old_dir, "landuse.tif"), rd(new_dir, "landuse.tif"), mask)
    no, nn = pd.read_csv(old_dir / "nodes.csv"), pd.read_csv(new_dir / "nodes.csv")
    pc = patch_change(rd(old_dir, "patch_ids.tif"), dict(zip(no.id, no.type)), rd(new_dir, "patch_ids.tif"), dict(zip(nn.id, nn.type)))
    so, sn = (json.loads((d / "summary.json").read_text()) for d in (old_dir, new_dir))
    eca = lambda s: float(np.sqrt(s["pc"]["500"]) * s["study_area_ha"])
    by = lambda n: {k: dict(n=int((n.type == k).sum()), ha=round(float(n.area_ha[n.type == k].sum()), 1)) for k in ("water", "green", "farm")}
    area = lambda n, ids: round(float(n.set_index("id").area_ha.reindex(ids).sum()), 1)
    net = (t.sum(axis=0) - t.sum(axis=1)).round(1)
    return dict(class_net_change_ha=net.to_dict(), transitions_ha=t.round(1).to_dict(), nodes_before=by(no), nodes_after=by(nn),
                patches_new=len(pc["new"]), patches_new_ha=area(nn, pc["new"]), patches_lost=len(pc["lost"]), patches_lost_ha=area(no, pc["lost"]),
                patches_persisting=len(pc["persisting"]), eca_ha_before=round(eca(so), 1), eca_ha_after=round(eca(sn), 1))


def build_version(tag, composite, window, note="", forecast=None):
    import rasterio

    from src import grid
    from src import twin as tw
    from src.config import OUTPUTS, RAW
    root = OUTPUTS / "twin_live"
    reg = root / "versions.json"
    if any(v["tag"] == tag for v in load_registry(reg)):
        raise FileExistsError(f"version {tag!r} already exists: versions are never overwritten")
    out = root / tag
    scenes = json.loads((RAW / f"s2_{composite}_scenes.json").read_text())
    print(f"[{tag}] classifying s2_{composite}.tif ({len(scenes)} scenes)", flush=True)
    lu_path = classify(RAW / f"s2_{composite}.tif", out)
    print(f"[{tag}] building the twin", flush=True)
    nodes, E, summary = tw.build(map_path=lu_path, out_dir=out, acc_version="v2")
    mask = grid.study_mask()
    cm = rasterio.open(lu_path).read(1)
    shares = {CLASSES[k]: round(100 * float((cm[mask] == k).mean()), 1) for k in range(6)}
    rain = rain_outlook(cm, mask, grid.ward_raster(), forecast)
    prev = previous_tag(reg, tag)
    change, season = None, None
    if prev:
        print(f"[{tag}] change report against {prev}", flush=True)
        change = compare_versions(root / prev, out, mask)
        pv = next(v for v in load_registry(reg) if v["tag"] == prev)
        season = season_label(scene_dates(pv["scenes"]), scene_dates(scenes))
    entry = dict(tag=tag, window=list(window), composite=f"s2_{composite}.tif", scenes=scenes, scene_dates=[str(d) for d in scene_dates(scenes)],
                 created=dt.datetime.now().isoformat(timespec="minutes"), land_use_product=PRODUCT, class_share_pct=shares, nodes=summary["nodes"], edges=summary["edges"],
                 eca_ha=round(float(np.sqrt(summary["pc"]["500"]) * summary["study_area_ha"]), 1), previous=prev,
                 season_vs_previous=season[0] if season else None, season_gap_days=season[1] if season else None,
                 change_vs_previous=change, rain_outlook=rain, note=note)
    (out / "version.json").write_text(json.dumps(entry, indent=1, default=str))
    register(reg, entry)
    print(json.dumps({k: entry[k] for k in ("tag", "scene_dates", "class_share_pct", "nodes", "eca_ha", "previous", "season_vs_previous")}, default=str), flush=True)
    if change:
        print("change:", json.dumps({k: change[k] for k in ("class_net_change_ha", "nodes_before", "nodes_after", "patches_new", "patches_new_ha",
                                                           "patches_lost", "patches_lost_ha", "eca_ha_before", "eca_ha_after")}), flush=True)
    print("rain outlook:", json.dumps({k: rain[k] for k in ("wettest_day", "rain_mm", "city_runoff_m3", "top_wards")}), flush=True)
    return entry


def auto(today=None, lookback_days=60, min_days=3, max_cloud=20.0, n_best=6):
    """Find the newest clear window that is not yet a version; reuse a downloaded composite with exactly those scenes, else download."""
    from src import download as dl
    from src.config import OUTPUTS, RAW
    today = today or dt.date.today()
    reg = load_registry(OUTPUTS / "twin_live" / "versions.json")
    last = max((max(dt.date.fromisoformat(d) for d in v["scene_dates"]) for v in reg), default=None)
    items = dl.search("sentinel-2-l2a", f"{today - dt.timedelta(days=lookback_days)}/{today}", {"eo:cloud_cover": {"lt": 100}})
    w = latest_window([(i.datetime.date(), i.properties["eo:cloud_cover"]) for i in items], today, lookback_days, min_days, max_cloud, last)
    if w is None:
        print("auto: no new clear window since the last version; nothing to do", flush=True)
        return None
    chosen, _ = dl.select_items(f"{w[0]}/{w[1]}", max_cloud, n_best)
    ids = sorted(i.id for i in chosen)
    name = next((p.name[3:-12] for p in sorted(RAW.glob("s2_*_scenes.json")) if sorted(json.loads(p.read_text())) == ids), None)
    print(f"auto: window {w[0]} to {w[1]}, {len(ids)} scenes; " + (f"reusing downloaded composite s2_{name}.tif" if name else "downloading"), flush=True)
    if name is None:
        name = f"live_{w[1]:%Y_%m_%d}"
        dl.s2_composite(name, f"{w[0]}/{w[1]}", max_cloud=max_cloud, n_best=n_best)
    return build_version(str(w[1]), name, w, note="found by the automatic window search")


if __name__ == "__main__":
    import argparse

    from src.wardmap import memory_guard
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag"); ap.add_argument("--composite"); ap.add_argument("--window", nargs=2); ap.add_argument("--note", default="")
    ap.add_argument("--auto", action="store_true")
    a = ap.parse_args()
    memory_guard(4.5)
    auto() if a.auto else build_version(a.tag, a.composite, a.window, a.note)
