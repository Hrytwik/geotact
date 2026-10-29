"""T8: graph twin of Bhubaneswar's water, green (habitat) and farm patches (design registered in docs/results_log.md).

Nodes are connected patches of the land-use map; attributes carry their history, greenness, flood exposure, land-use reliability
and biodiversity records; edges link touching patches, habitat patches within dispersal reach, and patches along the same drain."""
import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

GROUPS = {"water": [0], "green": [1, 3], "farm": [2]}
EIGHT = np.ones((3, 3), dtype=int)


def label_patches(classmap, study, groups=GROUPS, min_px=100):
    """Patch-id raster (0 = no node), a table (id, type, n_px) and the pixel total of patches below min_px per type."""
    classmap, study = np.asarray(classmap), np.asarray(study, bool)
    pid = np.zeros(classmap.shape, dtype=np.int32)
    rows, small, nxt = [], {}, 1
    for typ, classes in groups.items():
        mask = np.isin(classmap, classes) & study
        lab, n = ndi.label(mask, structure=EIGHT)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = np.flatnonzero(sizes >= min_px); keep = keep[keep > 0]
        small[typ] = int(sizes[1:][sizes[1:] < min_px].sum())
        lut = np.zeros(n + 1, dtype=np.int32)
        lut[keep] = np.arange(nxt, nxt + len(keep))
        pid[mask] = lut[lab[mask]]
        rows += [dict(id=int(lut[k]), type=typ, n_px=int(sizes[k])) for k in keep]
        nxt += len(keep)
    return pid, pd.DataFrame(rows, columns=["id", "type", "n_px"]), small


def zonal_mean(pid, values, n):
    """Mean of the finite values in each patch; index = patch id (0 unused); NaN for patches without finite values."""
    pid, values = np.asarray(pid), np.asarray(values, dtype=float)
    ok = (pid > 0) & np.isfinite(values)
    s = np.bincount(pid[ok], weights=values[ok], minlength=n + 1)
    c = np.bincount(pid[ok], minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def reliability(pid, classmap, class_accuracy, n):
    """Area-weighted user's accuracy (from the hand-label check) of the classes inside each patch."""
    lut = np.full(256, np.nan)
    for k, v in class_accuracy.items():
        lut[k] = v
    return zonal_mean(pid, lut[np.asarray(classmap, dtype=np.uint8)], n)


def _boundaries(pid):
    p = np.asarray(pid)
    edge = (ndi.minimum_filter(p, size=3, mode="nearest") != p) | (ndi.maximum_filter(p, size=3, mode="nearest") != p)
    return (p > 0) & edge


def patch_gaps(pid, ids, max_gap_m, pixel_m=10.0):
    """Edge-to-edge gap in metres for every pair of the given patches closer than max_gap_m; keys (i, j) with i < j.
    The gap between two pixel squares whose centres are d pixels apart is (d - 1) x pixel size (0 when touching)."""
    from scipy.spatial import cKDTree
    pid = np.asarray(pid)
    b = _boundaries(pid)
    rr, cc = np.nonzero(b & np.isin(pid, list(ids)))
    owner = pid[rr, cc]
    order = np.argsort(owner, kind="stable")
    rr, cc, owner = rr[order], cc[order], owner[order]
    uid, start = np.unique(owner, return_index=True)
    pts = {int(i): np.column_stack([rr[s:e], cc[s:e]]) for i, s, e in zip(uid, start, list(start[1:]) + [len(owner)])}
    reach = max_gap_m / pixel_m + 1.0
    keys = sorted(pts)
    box = np.array([[pts[k][:, 0].min(), pts[k][:, 0].max(), pts[k][:, 1].min(), pts[k][:, 1].max()] for k in keys], dtype=float)
    trees, out = {}, {}
    for a in range(len(keys)):
        near = np.flatnonzero((box[a + 1:, 0] <= box[a, 1] + reach) & (box[a + 1:, 1] >= box[a, 0] - reach) &
                              (box[a + 1:, 2] <= box[a, 3] + reach) & (box[a + 1:, 3] >= box[a, 2] - reach)) + a + 1
        for bi in near:
            i, j = keys[a], keys[bi]
            small, big = (i, j) if len(pts[i]) <= len(pts[j]) else (j, i)
            if big not in trees:
                trees[big] = cKDTree(pts[big])
            d, _ = trees[big].query(pts[small], k=1, distance_upper_bound=reach)
            dmin = float(np.min(d))
            if np.isfinite(dmin):
                gap = max(dmin - 1.0, 0.0) * pixel_m
                if gap <= max_gap_m:
                    out[(min(i, j), max(i, j))] = gap
    return out


def connectivity_weight(gap_m, d50_m=500.0):
    """Dispersal probability between two habitat patches: 1 when touching, 0.5 at d50_m, decaying exponentially."""
    return float(math.exp(-math.log(2) * gap_m / d50_m))


def _pc_parts(areas, edges):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, shortest_path
    ids = sorted(areas)
    pos = {k: i for i, k in enumerate(ids)}
    a = np.array([areas[k] for k in ids], dtype=float)
    if edges:
        i = np.array([pos[e[0]] for e in edges]); j = np.array([pos[e[1]] for e in edges])
        w = np.array([max(-math.log(e[2]), 1e-12) for e in edges])            # -log p; tiny positive keeps p = 1 edges present
        g = coo_matrix((w, (i, j)), shape=(len(ids), len(ids))).tocsr()
    else:
        g = coo_matrix((len(ids), len(ids))).tocsr()
    return ids, pos, a, g, connected_components, shortest_path


def pc_index(areas, edges, total_area):
    """Probability of connectivity (Saura & Pascual-Hortal 2007) with the best-product path probability between patches."""
    ids, pos, a, g, cc, sp = _pc_parts(areas, edges)
    n_comp, comp = cc(g, directed=False)
    num = 0.0
    for c in range(n_comp):
        idx = np.flatnonzero(comp == c)
        if len(idx) == 1:
            num += a[idx[0]] ** 2
            continue
        d = sp(g[idx][:, idx], directed=False)
        num += float(a[idx] @ np.exp(-d) @ a[idx])
    return num / total_area ** 2


def dpc(areas, edges, total_area, nodes=None):
    """Percentage drop of PC when each node (default: all) is removed; only the node's own component is recomputed."""
    ids, pos, a, g, cc, sp = _pc_parts(areas, edges)
    n_comp, comp = cc(g, directed=False)
    comp_num = {}
    for c in range(n_comp):
        idx = np.flatnonzero(comp == c)
        comp_num[c] = float(a[idx] @ np.exp(-sp(g[idx][:, idx], directed=False)) @ a[idx]) if len(idx) > 1 else a[idx[0]] ** 2
    total = sum(comp_num.values())
    out = {}
    for k in (nodes if nodes is not None else ids):
        c = comp[pos[k]]
        idx = np.flatnonzero(comp == c)
        rest = idx[idx != pos[k]]
        new_c = 0.0
        if len(rest):
            sub = g[rest][:, rest]
            new_c = float(a[rest] @ np.exp(-sp(sub, directed=False)) @ a[rest])
        out[k] = 100.0 * (comp_num[c] - new_c) / total
    return out


def nearest_patch(pid):
    """For every pixel, the id of the nearest patch pixel and the distance to it (in pixels)."""
    pid = np.asarray(pid)
    dist, (ir, ic) = ndi.distance_transform_edt(pid == 0, return_indices=True)
    return pid[ir, ic], dist


def assign_points(pid, rows, cols, max_px, near=None):
    """Each point goes to the nearest patch within max_px pixels (0 = none); each point is counted once."""
    near_id, dist = near if near is not None else nearest_patch(pid)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < pid.shape[0]) & (cols >= 0) & (cols < pid.shape[1])
    out = np.zeros(len(rows), dtype=int)
    r, c = rows[inside], cols[inside]
    out[inside] = np.where(dist[r, c] <= max_px, near_id[r, c], 0)
    return out


def drainage_edges(pid, drains, elev, max_px, near=None):
    """Directed downhill links: patches within max_px of the same connected drain network, chained by mean elevation."""
    near_id, dist = near if near is not None else nearest_patch(pid)
    lab, n = ndi.label(np.asarray(drains, bool), structure=EIGHT)
    sel = (lab > 0) & (dist <= max_px) & (near_id > 0)
    pairs = pd.DataFrame(dict(comp=lab[sel], node=near_id[sel])).drop_duplicates()
    out, seen = [], set()
    for _, grp in pairs.groupby("comp"):
        nodes = sorted(set(grp.node), key=lambda k: (-elev.get(k, -np.inf), k))
        for hi, lo in zip(nodes[:-1], nodes[1:]):
            if (hi, lo) not in seen:
                seen.add((hi, lo)); out.append((int(hi), int(lo)))
    return out


# ---------------------------------------------------------------- real build (T8)
def class_accuracy(version="v1"):
    """User's accuracy per class index from the hand-label check: v1 = original U-Net map (T7), v2 = corrected map (T4c, cross-validated)."""
    import json

    from src.config import HOME_OUTPUTS
    from src.landuse import CLASSES
    if version == "v1":
        users = json.loads((HOME_OUTPUTS / "landuse" / "accuracy.json").read_text())["all_300"]["unet_2023_seed0"]["users"]
    elif version == "v2":
        users = json.loads((HOME_OUTPUTS / "landuse" / "improve_scores.json").read_text())["correction_2023_cv"]["users"]
    else:
        raise ValueError(version)
    return {i: users[c][0] for i, c in enumerate(CLASSES)}


def _sample_30m(arr30, tf30, tf10, shape10):
    """Value of a 30 m raster at the centre of every 10 m pixel (NaN outside)."""
    rows, cols = np.indices(shape10)
    x = tf10.c + (cols + 0.5) * tf10.a
    y = tf10.f + (rows + 0.5) * tf10.e
    r30 = np.floor((y - tf30.f) / tf30.e).astype(int)
    c30 = np.floor((x - tf30.c) / tf30.a).astype(int)
    ok = (r30 >= 0) & (r30 < arr30.shape[0]) & (c30 >= 0) & (c30 < arr30.shape[1])
    out = np.full(shape10, np.nan, dtype="float32")
    out[ok] = arr30[r30[ok], c30[ok]]
    return out


FLOOD_MODELS = ("logistic", "forest", "lightgbm", "mlp")


def survey_soft_vote(fc_dir):
    """Mean city percentile of the field-survey flood maps that exist (all four for Bhubaneswar, the transferred ones elsewhere)."""
    maps = [_city_percentile(np.load(fc_dir / f"r3_scores_{m}.npy")) for m in FLOOD_MODELS if (fc_dir / f"r3_scores_{m}.npy").exists()]
    if not maps:
        raise FileNotFoundError(f"no field-survey flood maps in {fc_dir}")
    return np.nanmean(maps, axis=0)


def _city_percentile(arr):
    from scipy.stats import rankdata
    out = np.full(arr.shape, np.nan, dtype="float32")
    ok = np.isfinite(arr)
    out[ok] = rankdata(arr[ok]) / ok.sum() * 100.0
    return out


def build(map_name="landuse_2023_seed0.tif", min_px=100, d50=500.0, max_gap=1500.0, write=True, suffix="", acc_version="v1",
          map_path=None, out_dir=None):
    """map_path / out_dir override the default locations (the live twin builds dated versions elsewhere)."""
    import json

    import geopandas as gpd
    import rasterio
    from rasterio import features
    from shapely.geometry import shape as shp

    from src import grid
    from src import landuse as lu
    from src.config import CRS, INTERIM, OUTPUTS, RAW
    from src.vegetation_crosscheck import conversion_masks
    out_dir = out_dir or OUTPUTS / f"twin{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(map_path or OUTPUTS / "landuse" / map_name) as r:
        cm, tf10, crs = r.read(1), r.transform, r.crs
    study = grid.study_mask()
    pid, nodes, small = label_patches(cm, study, GROUPS, min_px)
    n = int(nodes.id.max())
    nodes["area_ha"] = nodes.n_px / 100.0
    rr, cc = np.indices(pid.shape)
    sel = pid > 0
    cnt = np.bincount(pid[sel], minlength=n + 1)
    cx = np.bincount(pid[sel], weights=cc[sel], minlength=n + 1) / np.maximum(cnt, 1)
    cy = np.bincount(pid[sel], weights=rr[sel], minlength=n + 1) / np.maximum(cnt, 1)
    nodes["x"] = [tf10.c + (cx[i] + 0.5) * tf10.a for i in nodes.id]
    nodes["y"] = [tf10.f + (cy[i] + 0.5) * tf10.e for i in nodes.id]
    wards = grid.ward_raster()
    wd = pd.DataFrame(dict(p=pid[sel], w=wards[sel]))
    nodes["ward"] = nodes.id.map(wd[wd.w > 0].groupby("p").w.agg(lambda s: int(s.mode().iloc[0])))
    tree = zonal_mean(pid, (cm == 1).astype(float), n); grass = zonal_mean(pid, (cm == 3).astype(float), n)
    nodes["tree_share"] = [tree[i] if t == "green" else np.nan for i, t in zip(nodes.id, nodes.type)]
    nodes["grass_share"] = [grass[i] if t == "green" else np.nan for i, t in zip(nodes.id, nodes.type)]
    class_acc = {i: v for i, v in class_accuracy(acc_version).items() if i in (0, 1, 2, 3)}
    rel = reliability(pid, cm, class_acc, n)
    nodes["reliability"] = [rel[i] for i in nodes.id]
    group_classes = {"water": [0], "green": [1, 3], "farm": [2]}
    for yr in range(2017, 2024):
        esri = lu.map_labels(grid.read_aligned(RAW / f"io_lulc_{yr}.tif", resampling="nearest", dtype="uint8"))
        shares = {t: zonal_mean(pid, np.isin(esri, cl).astype(float), n) for t, cl in group_classes.items()}
        nodes[f"hist_{yr}"] = [shares[t][i] for i, t in zip(nodes.id, nodes.type)]
    with rasterio.open(OUTPUTS / "vegetation" / "ndvi_norm_stack.tif") as r:
        for b, desc in enumerate(r.descriptions, 1):
            z = zonal_mean(pid, r.read(b), n)
            nodes[f"ndvi_{desc}"] = [z[i] for i in nodes.id]
    with rasterio.open(OUTPUTS / "terrain" / "terrain_features.tif") as r:
        tf30 = r.transform
        elev30 = r.read(list(r.descriptions).index("elevation") + 1)
    fc = OUTPUTS / "flood_cnn"
    survey = _city_percentile(survey_soft_vote(fc))
    up = fc / "r3_scores_urban_logistic.npy"                               # absent for a transfer city
    urban = _city_percentile(np.load(up)) if up.exists() else np.full(survey.shape, np.nan, dtype="float32")
    for name, arr in (("flood_survey_pct", survey), ("flood_urban_pct", urban), ("elevation_m", elev30)):
        z = zonal_mean(pid, _sample_30m(arr, tf30, tf10, pid.shape), n)
        nodes[name] = [z[i] for i in nodes.id]
    near = nearest_patch(pid)
    rd = lambda f: rasterio.open(OUTPUTS / "landuse" / f).read(1)
    conv = conversion_masks(rd("landuse_2023_seed0.tif"), rd("landuse_2023_seed1.tif"), rd("landuse_2025_26_seed0.tif"),
                            rd("landuse_2025_26_seed1.tif"), study)["converted"]
    hit = conv & (near[1] <= 10)
    tl = np.bincount(near[0][hit], minlength=n + 1) / 100.0
    nodes["treeloss_flag_ha_100m"] = [tl[i] for i in nodes.id]
    # GBIF: each record to the nearest node within 100 m (observer-biased; descriptive only)
    from pyproj import Transformer
    gbf = sorted((RAW / "gbif").glob("*.csv")) if (RAW / "gbif").exists() else []
    gb = pd.read_csv(gbf[0], sep="\t", low_memory=False,
                     usecols=["gbifID", "species", "class", "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters", "year"]) if gbf else \
        pd.DataFrame(columns=["gbifID", "species", "class", "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters", "year"])
    gb = gb[gb.coordinateUncertaintyInMeters.isna() | (gb.coordinateUncertaintyInMeters <= 1000)]
    x, y = Transformer.from_crs(4326, CRS, always_xy=True).transform(gb.decimalLongitude.values, gb.decimalLatitude.values)
    gr = np.floor((np.asarray(y) - tf10.f) / tf10.e).astype(int); gcn = np.floor((np.asarray(x) - tf10.c) / tf10.a).astype(int)
    gb["node"] = assign_points(pid, gr, gcn, 10, near)
    g_in = gb[gb.node > 0]
    rec = g_in.groupby("node").size(); spp = g_in.groupby("node").species.nunique()
    nodes["gbif_records"] = nodes.id.map(rec).fillna(0).astype(int)
    nodes["gbif_species"] = nodes.id.map(spp).fillna(0).astype(int)
    # T12b: bias-aware bird-habitat suitability (adopted guilds only), recomputed on this map
    from src import biodiversity as bd
    for k, z in bd.node_suitability(cm, tf10, pid, n).items():
        nodes[f"bird_{k}_suit"] = [z[i] for i in nodes.id]
    # edges
    types = dict(zip(nodes.id, nodes.type))
    adj = patch_gaps(pid, list(nodes.id), max_gap_m=10.0)
    edges = [dict(u=i, v=j, kind="adjacent", gap_m=g, weight=1.0) for (i, j), g in adj.items() if types[i] != types[j]]
    hab = nodes[nodes.type.isin(["green", "water"])].id.tolist()
    gaps = patch_gaps(pid, hab, max_gap_m=max_gap)
    edges += [dict(u=i, v=j, kind="connectivity", gap_m=g, weight=connectivity_weight(g, d50)) for (i, j), g in gaps.items()]
    geoms = []
    for layer in ("drain_line", "canal", "river", "drain_poly"):
        try:
            g = gpd.read_file(INTERIM / "wardmap.gpkg", layer=layer).to_crs(CRS)
            geoms += [geom for geom in g.geometry if geom is not None and not geom.is_empty]
        except Exception as e:                                                   # a missing layer is reported, not hidden
            print(f"drain layer {layer} not used: {e}", flush=True)
    drains = features.rasterize([(gm, 1) for gm in geoms], out_shape=pid.shape, transform=tf10, all_touched=True, dtype="uint8").astype(bool)
    elev = dict(zip(nodes.id, nodes.elevation_m))
    dr = drainage_edges(pid, drains, elev, max_px=5, near=near)
    edges += [dict(u=i, v=j, kind="drainage", gap_m=np.nan, weight=1.0) for i, j in dr]
    E = pd.DataFrame(edges)
    # connectivity metrics
    area = dict(zip(nodes.id, nodes.area_ha))
    A = float(study.sum() / 100.0)
    hab_area = {k: area[k] for k in hab}
    pc = {}
    for d in (250.0, 500.0, 1000.0):
        e = [(i, j, connectivity_weight(g, d)) for (i, j), g in gaps.items()]
        pc[int(d)] = pc_index(hab_area, e, A)
    e500 = [(i, j, connectivity_weight(g, d50)) for (i, j), g in gaps.items()]
    imp = dpc(hab_area, e500, A)
    nodes["dPC"] = nodes.id.map(imp)
    summary = dict(base_map=map_name, min_patch_ha=min_px / 100.0, study_area_ha=A,
                   nodes={t: dict(n=int((nodes.type == t).sum()), area_ha=float(nodes.area_ha[nodes.type == t].sum())) for t in GROUPS},
                   small_patch_ha={k: v / 100.0 for k, v in small.items()},
                   edges=E.kind.value_counts().to_dict(), pc={str(k): v for k, v in pc.items()}, d50_m=d50, max_gap_m=max_gap,
                   gbif_records_assigned=int(len(g_in)), gbif_records_total=int(len(gb)), drain_pixels=int(drains.sum()),
                   class_accuracy_used=class_acc)
    if write:
        meta = dict(driver="GTiff", height=pid.shape[0], width=pid.shape[1], count=1, dtype="int32", crs=crs, transform=tf10, compress="deflate", nodata=0)
        with rasterio.open(out_dir / "patch_ids.tif", "w", **meta) as dst:
            dst.write(pid, 1)
        polys = [(int(v), shp(g)) for g, v in features.shapes(pid, mask=pid > 0, transform=tf10)]
        pg = gpd.GeoDataFrame(dict(id=[p[0] for p in polys]), geometry=[p[1] for p in polys], crs=crs).dissolve(by="id").reset_index()
        gpd.GeoDataFrame(pg.merge(nodes, on="id"), geometry="geometry", crs=crs).to_file(out_dir / "nodes.gpkg", driver="GPKG")
        E.to_csv(out_dir / "edges.csv", index=False)
        nodes.to_csv(out_dir / "nodes.csv", index=False)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return nodes, E, summary


if __name__ == "__main__":
    import sys

    from src.wardmap import memory_guard
    memory_guard(4.0)
    if "--v2" in sys.argv:
        import json
        nodes, E, s = build("landuse_2023_corr.tif", suffix="_v2", acc_version="v2")
        print(json.dumps({k: s[k] for k in ("nodes", "small_patch_ha", "edges", "pc", "gbif_records_assigned")}, indent=1, default=float))
    elif "--sensitivity" in sys.argv:
        _, E2, s2 = build("landuse_2023_cons_seed0.tif", write=False)
        print("SENSITIVITY (consensus map):", s2["nodes"], s2["edges"], {k: round(v, 5) for k, v in s2["pc"].items()})
    else:
        import json
        nodes, E, s = build()
        print(json.dumps({k: s[k] for k in ("nodes", "small_patch_ha", "edges", "pc", "gbif_records_assigned", "gbif_records_total",
                                            "drain_pixels")}, indent=1, default=float))
