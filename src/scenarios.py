"""T9: scenario engine for the twin (design registered in docs/results_log.md): wetland restoration, community gardens and tree
corridors as edits to the land-use map, scored on runoff (SDG 11 proxy), food-potential area (SDG 2 proxy), public open space
(SDG 11.7.1 proxy) and habitat connectivity. Uncalibrated screening tools, not forecasts."""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from src import twin as tw

WATER, TREES, CROP, GRASS, BUILT, BARE, GARDEN, WETLAND, CORRIDOR = range(9)
# USDA TR-55 curve numbers (checked 2026-10-25 against the TR-55 tables reproduced on Wikipedia 'Runoff curve number')
CN = {
    "B": {WATER: 98, WETLAND: 98, TREES: 55, CORRIDOR: 55, CROP: 78, GRASS: 69, BUILT: 92, BARE: 86, GARDEN: 61},
    "C": {WATER: 98, WETLAND: 98, TREES: 70, CORRIDOR: 70, CROP: 85, GRASS: 79, BUILT: 94, BARE: 91, GARDEN: 74},
    "D": {WATER: 98, WETLAND: 98, TREES: 77, CORRIDOR: 77, CROP: 89, GRASS: 84, BUILT: 95, BARE: 94, GARDEN: 80},
}
TWIN_GROUPS = {"water": [WATER, WETLAND], "green": [TREES, GRASS, GARDEN, CORRIDOR], "farm": [CROP]}
NEVER_EDIT = [WATER, TREES, BUILT]


def scs_runoff_mm(P, cn):
    """SCS curve-number direct runoff (mm) for rainfall P (mm): Q = (P - 0.2S)^2 / (P + 0.8S), S = 25400/CN - 254."""
    S = 25400.0 / np.asarray(cn, dtype=float) - 254.0
    Ia = 0.2 * S
    q = np.where(P > Ia, (P - Ia) ** 2 / np.maximum(P - Ia + S, 1e-12), 0.0)
    return float(q) if np.ndim(q) == 0 else q


def runoff_volume_m3(classmap, mask, P, cn, wetland_storage_m=0.3, pixel_m2=100.0):
    m = np.asarray(classmap)[np.asarray(mask, bool)]
    counts = np.bincount(m.ravel(), minlength=16)
    vol = sum(counts[k] * scs_runoff_mm(P, v) / 1000.0 * pixel_m2 for k, v in cn.items())
    vol -= wetland_storage_m * counts[WETLAND] * pixel_m2
    return float(max(vol, 0.0))


def food_index_ha(classmap, mask, pixel_ha=0.01):
    return float((np.isin(classmap, [CROP, GARDEN]) & np.asarray(mask, bool)).sum() * pixel_ha)


def open_space_share(classmap, mask, osm_open, built_px):
    space = (np.asarray(osm_open, bool) | np.isin(classmap, [GARDEN, CORRIDOR])) & np.asarray(mask, bool)
    return float(space.sum() / built_px)


def block_ids(shape, size=10):
    r, c = np.indices(shape)
    return (r // size) * (-(-shape[1] // size)) + (c // size)


def select_blocks(cand, blocks, score, budget_px, min_frac=0.5, exclude=frozenset()):
    """Blocks in descending score whose candidate share is at least min_frac, added until their candidate cells reach budget_px."""
    blocks = np.asarray(blocks)
    n = int(blocks.max()) + 1
    size = np.bincount(blocks.ravel(), minlength=n)
    cnt = np.bincount(blocks[np.asarray(cand, bool)], minlength=n)
    elig = [b for b in score if cnt[b] >= min_frac * size[b] and cnt[b] > 0 and b not in exclude]
    chosen, total = [], 0
    for b in sorted(elig, key=lambda k: -score[k]):
        if total >= budget_px:
            break
        chosen.append(int(b)); total += int(cnt[b])
    return chosen


def apply_edit(classmap, cand, blocks, chosen, new_class):
    out = np.array(classmap, copy=True)
    sel = np.asarray(cand, bool) & np.isin(blocks, chosen) & ~np.isin(out, NEVER_EDIT)
    out[sel] = new_class
    return out


def candidates(classmap, mask, flood_pct, near_water, built_share, riparian):
    m, mask = np.asarray(classmap), np.asarray(mask, bool)
    with np.errstate(invalid="ignore"):
        hot = np.nan_to_num(flood_pct, nan=-1.0) >= 80
        dense = np.nan_to_num(built_share, nan=0.0) >= 0.5
    return {"wetland": np.isin(m, [GRASS, BARE, CROP]) & hot & np.asarray(near_water, bool) & mask,
            "garden": np.isin(m, [GRASS, BARE]) & dense & mask,
            "corridor": np.isin(m, [GRASS, BARE, CROP]) & np.asarray(riparian, bool) & mask}


def two_nearest_gap(pid, ids, max_px):
    """Per cell, distance to the nearest plus the second-nearest DIFFERENT patch (pixels); inf where fewer than two lie within max_px."""
    pid = np.asarray(pid)
    best1 = np.full(pid.shape, np.inf); best2 = np.full(pid.shape, np.inf)
    for k in ids:
        rr, cc = np.nonzero(pid == k)
        if not len(rr):
            continue
        r0, r1 = max(rr.min() - max_px, 0), min(rr.max() + max_px + 1, pid.shape[0])
        c0, c1 = max(cc.min() - max_px, 0), min(cc.max() + max_px + 1, pid.shape[1])
        d = ndi.distance_transform_edt(pid[r0:r1, c0:c1] != k)
        d = np.where(d <= max_px, d, np.inf)
        b1, b2 = best1[r0:r1, c0:c1], best2[r0:r1, c0:c1]
        nb2 = np.where(d < b1, b1, np.minimum(b2, d))
        nb1 = np.minimum(b1, d)
        best1[r0:r1, c0:c1], best2[r0:r1, c0:c1] = nb1, nb2
    return np.where(np.isfinite(best2), best1 + best2, np.inf)


def metrics(classmap, mask, osm_open, built_px, P=100.0, hsg="C", d50=500.0, max_gap=1500.0, min_px=100):
    pid, nodes, _ = tw.label_patches(classmap, mask, TWIN_GROUPS, min_px)
    hab = nodes[nodes.type.isin(["water", "green"])]
    area = dict(zip(hab.id, hab.n_px / 100.0))
    gaps = tw.patch_gaps(pid, list(area), max_gap_m=max_gap) if len(area) > 1 else {}
    A = float(np.asarray(mask, bool).sum() / 100.0)
    pc = tw.pc_index(area, [(i, j, tw.connectivity_weight(g, d50)) for (i, j), g in gaps.items()], A) if area else 0.0
    return dict(runoff_m3=runoff_volume_m3(classmap, mask, P, CN[hsg]), food_ha=food_index_ha(classmap, mask),
                open_space_share=open_space_share(classmap, mask, osm_open, built_px), pc=float(pc), eca_ha=float(np.sqrt(pc) * A),
                habitat_nodes=int(len(hab)))


# ---------------------------------------------------------------- real run (T9)
OSM_OPEN = {("leisure", "park"), ("leisure", "garden"), ("leisure", "recreation_ground"), ("leisure", "playground"),
            ("leisure", "pitch"), ("leisure", "nature_reserve"), ("landuse", "grass"), ("landuse", "recreation_ground")}


def osm_open_polygons(path):
    """Closed OSM ways tagged as public open space (Overpass 'out geom'); relations are skipped and counted."""
    import json

    from shapely.geometry import Polygon
    js = json.loads(open(path).read())
    polys, skipped = [], 0
    for e in js["elements"]:
        tags = e.get("tags", {})
        if not any((k, tags.get(k)) in OSM_OPEN for k in ("leisure", "landuse")):
            continue
        if e["type"] != "way":
            skipped += 1
            continue
        pts = [(p["lon"], p["lat"]) for p in e.get("geometry", [])]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            polys.append(Polygon(pts))
    return polys, skipped


def inputs(map_name="landuse_2023_seed0.tif"):
    import rasterio
    from rasterio import features

    from src import grid
    from src.config import CRS, INTERIM, OUTPUTS, RAW
    from src.terrain import to_utm
    with rasterio.open(OUTPUTS / "landuse" / map_name) as r:
        cm, tf, crs = r.read(1), r.transform, r.crs
    mask = grid.study_mask()
    with rasterio.open(OUTPUTS / "terrain" / "terrain_features.tif") as r:
        tf30 = r.transform
    fc = OUTPUTS / "flood_cnn"
    survey = tw._city_percentile(tw.survey_soft_vote(fc))
    flood10 = tw._sample_30m(survey, tf30, tf, cm.shape)
    import geopandas as gpd
    geoms = []
    for layer in ("drain_line", "canal", "river", "drain_poly"):
        g = gpd.read_file(INTERIM / "wardmap.gpkg", layer=layer).to_crs(CRS)
        geoms += [x for x in g.geometry if x is not None and not x.is_empty]
    drains = features.rasterize([(x, 1) for x in geoms], out_shape=cm.shape, transform=tf, all_touched=True, dtype="uint8").astype(bool)
    near_water = ndi.distance_transform_edt(~((cm == WATER) | drains)) <= 20
    riparian = ndi.distance_transform_edt(~drains) <= 3
    built_share = ndi.uniform_filter((cm == BUILT).astype("float32"), size=31, mode="nearest")
    polys, skipped = osm_open_polygons(RAW / "osm_raw.json")
    osm = features.rasterize([(x, 1) for x in to_utm(polys, CRS)], out_shape=cm.shape, transform=tf, dtype="uint8").astype(bool) if polys else np.zeros(cm.shape, bool)
    return dict(cm=cm, tf=tf, crs=crs, mask=mask, flood=flood10, near_water=near_water, riparian=riparian, built_share=built_share,
                osm=osm & mask, osm_polygons=len(polys), osm_relations_skipped=skipped)


def _block_mean(values, cand, blocks, n):
    s = np.bincount(blocks[cand], weights=np.nan_to_num(values[cand], nan=0.0), minlength=n)
    c = np.bincount(blocks[cand], minlength=n)
    return {int(b): float(s[b] / c[b]) for b in np.flatnonzero(c)}


def build_scenarios(d, twin_pid, habitat_ids, budget_px=5000):
    cm, mask = d["cm"], d["mask"]
    cand = candidates(cm, mask, d["flood"], d["near_water"], d["built_share"], d["riparian"])
    blocks = block_ids(cm.shape, 10)
    n = int(blocks.max()) + 1
    gap = two_nearest_gap(twin_pid, habitat_ids, max_px=150)
    score = {"wetland": _block_mean(d["flood"], cand["wetland"], blocks, n),
             "garden": _block_mean(d["built_share"], cand["garden"], blocks, n),
             "corridor": {b: -v for b, v in _block_mean(np.where(np.isfinite(gap), gap, 1e6), cand["corridor"], blocks, n).items()}}
    new = {"wetland": WETLAND, "garden": GARDEN, "corridor": CORRIDOR}
    maps, chosen = {"baseline": cm}, {}
    for k in ("wetland", "garden", "corridor"):
        chosen[k] = select_blocks(cand[k], blocks, score[k], budget_px)
        maps[k] = apply_edit(cm, cand[k], blocks, chosen[k], new[k])
    comb, used = cm.copy(), set()
    for k in ("wetland", "garden", "corridor"):
        ch = select_blocks(cand[k], blocks, score[k], budget_px, exclude=used)
        comb = apply_edit(comb, cand[k] & (comb == cm), blocks, ch, new[k])
        used |= set(ch); chosen[f"combined_{k}"] = ch
    maps["combined"] = comb
    table = pd.DataFrame({"block": np.arange(n)})
    for k in cand:
        table[f"cand_{k}_px"] = np.bincount(blocks[cand[k]], minlength=n)
        table[f"score_{k}"] = table.block.map(score[k])
    table = table[(table.filter(like="cand_").sum(axis=1) > 0)]
    return maps, chosen, cand, blocks, table


def run(map_name="landuse_2023_seed0.tif", write=True, suffix="", acc_version="v1", twin_suffix=None, energy=None, heat=None):
    """twin_suffix: which twin supplies the patches (default: the same suffix). energy / heat (T14): add food energy (GJ) and the heat-exposure change."""
    import json

    import rasterio

    from src.config import OUTPUTS
    out = OUTPUTS / f"scenarios{suffix}"
    out.mkdir(exist_ok=True)
    d = inputs(map_name)
    ts = suffix if twin_suffix is None else twin_suffix
    tw_pid = rasterio.open(OUTPUTS / f"twin{ts}" / "patch_ids.tif").read(1)
    nodes = pd.read_csv(OUTPUTS / f"twin{ts}" / "nodes.csv")
    hab = nodes[nodes.type.isin(["green", "water"])].id.tolist()
    if not suffix and map_name != "landuse_2023_seed0.tif":                    # sensitivity: rebuild the patches of this map
        tw_pid, tbl, _ = tw.label_patches(d["cm"], d["mask"], tw.GROUPS, 100)
        hab = tbl[tbl.type.isin(["green", "water"])].id.tolist()
    maps, chosen, cand, blocks, table = build_scenarios(d, tw_pid, hab)
    built_px = int(((d["cm"] == BUILT) & d["mask"]).sum())
    from src.landuse import CLASSES
    rel = tw.class_accuracy(acc_version)
    res = {}
    base_runoff = {(P, h): runoff_volume_m3(maps["baseline"], d["mask"], P, CN[h]) for P in (50.0, 100.0, 150.0) for h in "BCD"}
    for name, m in maps.items():
        r = metrics(m, d["mask"], d["osm"], built_px)
        changed = (m != d["cm"]) & d["mask"]
        orig = d["cm"][changed]
        r["converted_ha"] = float(changed.sum() / 100.0)
        r["converted_from_ha"] = {CLASSES[k]: float((orig == k).sum() / 100.0) for k in np.unique(orig)}
        r["reliability_weighted_ha"] = float(sum(rel[int(k)] for k in orig) / 100.0) if len(orig) else 0.0
        if energy is not None:
            from src import optimise as op
            r["food_gj"] = op.food_gj(m, d["mask"], energy)
        if heat is not None:
            from src import optimise as op
            r["heat_c"] = op.ht_exact(m, d["cm"], heat)
        r["runoff_change_pct"] = {f"P{int(P)}_{h}": 100.0 * (runoff_volume_m3(m, d["mask"], P, CN[h]) - b) / b for (P, h), b in base_runoff.items()}
        res[name] = r
    summary = dict(map=map_name, candidate_ha={k: float(v.sum() / 100.0) for k, v in cand.items()}, chosen_blocks={k: len(v) for k, v in chosen.items()},
                   osm_open_space_ha=float(d["osm"].sum() / 100.0), osm_polygons=d["osm_polygons"], osm_relations_skipped=d["osm_relations_skipped"],
                   baseline_built_ha=built_px / 100.0, scenarios=res)
    if write:
        meta = dict(driver="GTiff", height=d["cm"].shape[0], width=d["cm"].shape[1], count=1, dtype="uint8", crs=d["crs"], transform=d["tf"], compress="deflate")
        for name, m in maps.items():
            if name != "baseline":
                with rasterio.open(out / f"{name}.tif", "w", **meta) as dst:
                    dst.write(m.astype("uint8"), 1)
        table.to_csv(out / "candidate_blocks.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
        rows = [dict(scenario=k, converted_ha=v["converted_ha"], reliability_weighted_ha=round(v["reliability_weighted_ha"], 1),
                     runoff_m3=round(v["runoff_m3"]), runoff_change_pct_P100_C=round(v["runoff_change_pct"]["P100_C"], 3),
                     food_ha=round(v["food_ha"], 1), open_space_share=round(v["open_space_share"], 4), pc=v["pc"], eca_ha=round(v["eca_ha"], 1),
                     **({} if energy is None else dict(food_gj=round(v["food_gj"], 1))), **({} if heat is None else dict(heat_exposure_change_c=round(v["heat_c"], 5))))
                for k, v in res.items()]
        pd.DataFrame(rows).to_csv(f"docs/scenario_metrics{suffix}.csv", index=False)
    return summary


if __name__ == "__main__":
    import json
    import sys

    from src.wardmap import memory_guard
    memory_guard(4.0)
    if "--v3" in sys.argv:                                   # T14: food energy and heat added, patches of the existing v2 twin (same map)
        from src import optimise as op
        energy, heat = op.v3_inputs()
        s = run(sys.argv[sys.argv.index("--v3") + 1], suffix="_v3", acc_version="v2", twin_suffix="_v2", energy=energy, heat=heat)
    elif "--v2" in sys.argv:
        s = run("landuse_2023_corr.tif", suffix="_v2", acc_version="v2")
    elif "--sensitivity" in sys.argv:
        s = run("landuse_2023_cons_seed0.tif", write=False)
    else:
        s = run()
    print(json.dumps({k: s[k] for k in ("map", "candidate_ha", "chosen_blocks", "osm_open_space_ha", "osm_polygons", "osm_relations_skipped", "baseline_built_ha")}, default=float))
    for k, v in s["scenarios"].items():
        print(f"{k:9s} conv {v['converted_ha']:6.1f} ha (reliab.-weighted {v['reliability_weighted_ha']:5.1f})  runoff {v['runoff_m3']:,.0f} m3 "
              f"({v['runoff_change_pct']['P100_C']:+.2f}%)  food {v['food_ha']:7.1f} ha  open space {100 * v['open_space_share']:.2f}%  "
              f"ECA {v['eca_ha']:.0f} ha  from {v['converted_from_ha']}")
