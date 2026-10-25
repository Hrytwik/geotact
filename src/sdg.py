"""T9b: official-style SDG indicators (protocol in docs/build_plan.md): 11.3.1 land consumption vs population growth (GHSL),
11.7.1 share of the urbanised area that is public space (streets + open space), 15.1.1 forest share of land area."""
import numpy as np
from scipy import ndimage as ndi

ROAD_WIDTH_M = {"motorway": 20, "trunk": 20, "primary": 14, "secondary": 10, "tertiary": 8, "residential": 6,
                "unclassified": 6, "living_street": 6, "service": 6}


def lcrpgr(urb1, urb2, pop1, pop2, years):
    lcr = np.log(urb2 / urb1) / years
    pgr = np.log(pop2 / pop1) / years
    return dict(lcr=float(lcr), pgr=float(pgr), ratio=float(lcr / pgr))


def road_width(tag):
    if tag is None:
        return 3
    return ROAD_WIDTH_M.get(str(tag).replace("_link", ""), 3)


def street_mask(lines_widths, shape, transform):
    """Union of the roads buffered by half their full width, rasterised (each cell counted once however many roads cover it)."""
    from rasterio import features
    shapes = [(ln.buffer(w / 2.0, cap_style=2), 1) for ln, w in lines_widths if ln is not None and not ln.is_empty]
    if not shapes:
        return np.zeros(shape, bool)
    return features.rasterize(shapes, out_shape=shape, transform=transform, dtype="uint8").astype(bool)


def open_public_space_share(osm_open, street, built, mask, extra_open=None):
    space = (np.asarray(osm_open, bool) | np.asarray(street, bool))
    if extra_open is not None:
        space |= np.asarray(extra_open, bool)
    mask = np.asarray(mask, bool)
    urban = (np.asarray(built, bool) | space) & mask
    return float((space & mask).sum() / urban.sum())


def forest_share(classmap, mask, min_px=50, tree=1, water=(0, 7)):
    """Tree-cover patches of at least min_px cells (8-connected) as a share of the land area (mask minus water)."""
    m, mask = np.asarray(classmap), np.asarray(mask, bool)
    lab, n = ndi.label((m == tree) & mask, structure=np.ones((3, 3), int))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    ids = np.flatnonzero(sizes >= min_px)
    big = np.isin(lab, ids[ids > 0]) & (lab > 0)                          # label 0 is background, never a patch
    land = mask & ~np.isin(m, water)
    return float(big.sum() / land.sum())


# ---------------------------------------------------------------- real run (T9b)
def _ghsl_sum(prod, epoch, geom_moll):
    import rasterio
    from rasterio.mask import mask as rmask
    from src.config import RAW
    name = f"GHS_{prod}_E{epoch}_GLOBE_R2023A_54009_100"
    with rasterio.open(RAW / "ghsl" / f"{name}_R7_C27" / f"{name}_V1_0_R7_C27.tif") as src:
        arr, _ = rmask(src, [geom_moll], crop=True, filled=True, nodata=0)
        nod = src.nodata
    a = arr[0].astype("float64")
    if nod is not None:
        a[a == nod] = 0
    return float(a[a > 0].sum())


def _streets(tf, shape):
    import json

    import geopandas as gpd
    from shapely.geometry import LineString

    from src.config import CRS, RAW
    js = json.load(open(RAW / "osm_highways.json"))
    lines, widths = [], []
    for e in js["elements"]:
        pts = [(p["lon"], p["lat"]) for p in e.get("geometry", [])]
        if len(pts) >= 2:
            lines.append(LineString(pts)); widths.append(road_width(e.get("tags", {}).get("highway")))
    utm = gpd.GeoSeries(lines, crs=4326).to_crs(CRS)
    return street_mask(list(zip(utm, widths)), shape, tf), len(lines)


def run():
    import json

    import geopandas as gpd
    import rasterio

    from src import grid
    from src import landuse as lu
    from src.config import INTERIM, OUTPUTS, RAW
    out = OUTPUTS / "sdg"
    out.mkdir(exist_ok=True)
    res = {}
    # 11.3.1 with GHSL
    city = gpd.read_file(INTERIM / "wardmap.gpkg", layer="municipal_poly")
    moll = city.to_crs("ESRI:54009").geometry.union_all()
    g = {(p, e): _ghsl_sum(p, e, moll) for p in ("POP", "BUILT_S") for e in (2015, 2020, 2025)}
    res["ghsl_totals"] = {f"{p}_{e}": v for (p, e), v in g.items()}
    res["11.3.1_ghsl_2015_2020"] = lcrpgr(g[("BUILT_S", 2015)], g[("BUILT_S", 2020)], g[("POP", 2015)], g[("POP", 2020)], 5)
    res["11.3.1_ghsl_2020_2025_projection"] = lcrpgr(g[("BUILT_S", 2020)], g[("BUILT_S", 2025)], g[("POP", 2020)], g[("POP", 2025)], 5)
    mask = grid.study_mask()
    esri = {y: lu.map_labels(grid.read_aligned(RAW / f"io_lulc_{y}.tif", resampling="nearest", dtype="uint8")) for y in (2017, 2023)}
    urb = {y: float(((esri[y] == 4) & mask).sum() / 100.0) for y in esri}
    pgr1 = res["11.3.1_ghsl_2015_2020"]["pgr"]; pgr2 = res["11.3.1_ghsl_2020_2025_projection"]["pgr"]
    pop17 = g[("POP", 2015)] * np.exp(pgr1 * 2); pop23 = g[("POP", 2020)] * np.exp(pgr2 * 3)
    res["11.3.1_esri_2017_2023_sensitivity"] = dict(**lcrpgr(urb[2017], urb[2023], pop17, pop23, 6), built_ha_2017=urb[2017], built_ha_2023=urb[2023],
                                                   pop_2017_interp=float(pop17), pop_2023_interp=float(pop23))
    # 11.7.1
    with rasterio.open(lu.OUT / "landuse_2023_corr.tif") as r:
        v2, tf = r.read(1), r.transform
    v1 = rasterio.open(lu.OUT / "landuse_2023_seed0.tif").read(1)
    street, n_lines = _streets(tf, v2.shape)
    from src import scenarios as sc
    osm = sc.inputs("landuse_2023_corr.tif")["osm"]
    def s1171(cmap, extra=None):
        return open_public_space_share(osm, street, cmap == 4, mask, extra)
    comb = rasterio.open(OUTPUTS / "scenarios_v2" / "combined.tif").read(1)
    knee = rasterio.open(OUTPUTS / "optimise_v2" / "solution_knee.tif").read(1)
    res["11.7.1"] = dict(osm_ways_roads=n_lines, street_ha=float((street & mask).sum() / 100.0), osm_open_ha=float((osm & mask).sum() / 100.0),
                         baseline_v2=s1171(v2), combined_scenario_v2=s1171(v2, np.isin(comb, [6, 8])), balanced_plan_v2=s1171(v2, np.isin(knee, [6, 8])),
                         baseline_v1_original_map=s1171(v1))
    # 15.1.1
    wc = lu.wc_labels(grid.read_aligned(RAW / "worldcover_2021.tif", resampling="nearest", dtype="uint8"))
    res["15.1.1"] = dict(corrected_map_v2=forest_share(v2, mask), original_map_v1=forest_share(v1, mask), worldcover_2021=forest_share(wc, mask),
                         combined_scenario_v2=forest_share(np.where(comb == 8, 1, comb), mask))
    (out / "indicators.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    import json

    from src.wardmap import memory_guard
    memory_guard(4.0)
    print(json.dumps(run(), indent=1, default=float))
