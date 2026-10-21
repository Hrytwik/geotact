"""T6b: terrain and hydrology features at the DEM's native 30 m (Copernicus GLO-30).

The DEM is a surface model (buildings and trees included), so local depressions inside the dense city are
unreliable; the features describe the broad drainage setting. Pure numpy/scipy/pysheds: nothing here imports torch.
"""
import json

import numpy as np
from rasterio.features import rasterize as _rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, Polygon

# Compatibility shim: pysheds 0.5 calls np.in1d, which NumPy 2 removed. in1d(a, b) is isin(a, b) flattened, so the
# alias is restored with exactly that definition (checked by tests/test_terrain.py; pinning NumPy < 2 would risk the
# rest of the environment).
if not hasattr(np, "in1d"):
    def _in1d(ar1, ar2, assume_unique=False, invert=False, **kwargs):
        return np.isin(ar1, ar2, assume_unique=assume_unique, invert=invert).ravel()

    np.in1d = _in1d

WATERWAY_LINES = {"river", "stream", "canal", "drain", "ditch"}
WATER_POLYGONS = {"water", "wetland"}


# ---------------------------------------------------------------- geometry helpers
def rasterize(geoms, transform, shape, all_touched=True):
    """Boolean raster that is True where any geometry (line or polygon) touches a cell."""
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, dtype=bool)
    out = _rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, fill=0,
                     all_touched=all_touched, dtype="uint8")
    return out.astype(bool)


def distance_to(mask, res):
    """Euclidean distance in metres from every cell to the nearest True cell (inf if the mask is empty)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return distance_transform_edt(~mask, sampling=res)


def osm_water(path):
    """Waterway lines and water/wetland polygons (lon/lat) from an Overpass 'out geom' dump. Only ways are read;
    multipolygon relations are counted and skipped."""
    js = json.loads(open(path).read())
    lines, polygons, skipped = [], [], 0
    for e in js["elements"]:
        tags = e.get("tags", {})
        if e["type"] == "relation":
            skipped += 1 if (tags.get("natural") in WATER_POLYGONS) else 0
            continue
        pts = [(p["lon"], p["lat"]) for p in e.get("geometry", [])]
        if len(pts) < 2:
            continue
        if tags.get("waterway") in WATERWAY_LINES:
            lines.append(LineString(pts))
        elif tags.get("natural") in WATER_POLYGONS:
            if len(pts) >= 4 and pts[0] == pts[-1]:
                polygons.append(Polygon(pts))
            else:
                lines.append(LineString(pts))
    return dict(lines=lines, polygons=polygons, relations_skipped=skipped)


def to_utm(geoms, crs="EPSG:32645"):
    import geopandas as gpd
    return list(gpd.GeoSeries(list(geoms), crs="EPSG:4326").to_crs(crs)) if len(geoms) else []


# ---------------------------------------------------------------- features
def terrain_features(dem, transform, crs, channels=None, acc_threshold=None):
    """Elevation, slope, Laplacian, flow accumulation, TWI, HAND and distance to channels.

    channels: boolean raster of known channels (mapped drains, canals, rivers). acc_threshold: additionally treat cells
    whose flow accumulation (in cells) reaches this value as channels. With neither, HAND is NaN."""
    from pysheds.grid import Grid
    from pysheds.sview import Raster, ViewFinder

    dem = np.asarray(dem, dtype="float64")
    res = abs(transform.a)
    dzdy, dzdx = np.gradient(dem, res)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    p = np.pad(dem, 1, mode="edge")
    lap = (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * dem) / res ** 2

    vf = ViewFinder(affine=transform, shape=dem.shape, nodata=np.nan, crs=crs)
    g = Grid(viewfinder=vf)
    raw = Raster(dem, viewfinder=vf)
    filled = g.fill_depressions(g.fill_pits(raw))
    fdir = g.flowdir(g.resolve_flats(filled))
    acc = np.asarray(g.accumulation(fdir), dtype="float64")

    tan_beta = np.tan(np.radians(np.maximum(slope, 0.1)))           # floor avoids infinite TWI on flat ground
    twi = np.log(acc * res / tan_beta)

    chan = np.zeros(dem.shape, dtype=bool) if channels is None else np.asarray(channels, dtype=bool).copy()
    if acc_threshold is not None:
        chan |= acc >= acc_threshold
    if chan.any():
        vf_mask = ViewFinder(affine=transform, shape=dem.shape, nodata=False, crs=crs)   # boolean raster: NaN is not representable
        hand = np.asarray(g.compute_hand(fdir, filled, Raster(chan, viewfinder=vf_mask)), dtype="float64")
        # HAND cannot be negative; pysheds works in float32, so a subtraction of two nearly equal elevations can come out
        # as -1e-4 m. Clip that round-off, but fail loudly if anything is meaningfully negative.
        if np.nanmin(hand) < -0.01:
            raise ValueError(f"HAND has a real negative value ({np.nanmin(hand):.4f} m): the drainage routing is inconsistent")
        hand = np.where(hand < 0, 0.0, hand)
        dist = distance_to(chan, res)
    else:
        hand = np.full(dem.shape, np.nan)
        dist = np.full(dem.shape, np.inf)
    return dict(elevation=dem, slope_deg=slope, laplacian=lap, acc_cells=acc, twi=twi, hand=hand,
                dist_channel=dist, channels=chan)


# ---------------------------------------------------------------- real study area
ACC_THRESHOLD = 300          # contributing cells (0.27 sq km); decision recorded in docs/build_plan.md
BANDS = ["elevation", "slope_deg", "laplacian", "log10_acc_m2", "twi", "hand", "dist_drain", "dist_river",
         "dist_water", "dist_channel"]


def real_inputs():
    """DEM (NaN no-data), its geobox, and boolean rasters of mapped channels/rivers/water bodies, plus the BMC mask."""
    import geopandas as gpd
    import rasterio
    from src.config import CRS, INTERIM, RAW
    with rasterio.open(RAW / "copdem30.tif") as r:
        dem = r.read(1).astype("float64")
        dem[dem <= -9990] = np.nan
        tf, crs = r.transform, r.crs
    g = INTERIM / "wardmap.gpkg"
    lay = lambda n: list(gpd.read_file(g, layer=n).to_crs(CRS).geometry)
    osm = osm_water(RAW / "osm_raw.json")
    shape = dem.shape
    drains = rasterize(lay("drain_line") + lay("canal") + lay("drain_poly") + to_utm(osm["lines"]), tf, shape)
    river = rasterize(lay("river"), tf, shape)
    water = rasterize(lay("wb") + lay("river") + to_utm(osm["polygons"]), tf, shape)
    city = gpd.read_file(g, layer="municipal_poly").to_crs(CRS)
    bmc = rasterize(list(city.geometry), tf, shape, all_touched=False)
    return dict(dem=dem, transform=tf, crs=crs, drains=drains, river=river, water=water, bmc=bmc)


def compute_real(acc_threshold=ACC_THRESHOLD):
    d = real_inputs()
    res = abs(d["transform"].a)
    f = terrain_features(d["dem"], d["transform"], d["crs"], channels=d["drains"] | d["river"], acc_threshold=acc_threshold)
    derived = f["acc_cells"] >= acc_threshold
    out = dict(elevation=f["elevation"], slope_deg=f["slope_deg"], laplacian=f["laplacian"],
               log10_acc_m2=np.log10(f["acc_cells"] * res ** 2), twi=f["twi"], hand=f["hand"],
               dist_drain=distance_to(d["drains"] | derived, res), dist_river=distance_to(d["river"], res),
               dist_water=distance_to(d["water"], res), dist_channel=f["dist_channel"])
    return out, d, derived


def build(acc_threshold=ACC_THRESHOLD):
    """Compute and write outputs/terrain/terrain_features.tif (band names in the file descriptions)."""
    import rasterio
    from src.config import OUTPUTS
    out, d, derived = compute_real(acc_threshold)
    odir = OUTPUTS / "terrain"
    odir.mkdir(parents=True, exist_ok=True)
    stack = np.stack([out[b] for b in BANDS]).astype("float32")
    stack[~np.isfinite(stack)] = np.nan
    h, w = stack.shape[1:]
    with rasterio.open(odir / "terrain_features.tif", "w", driver="GTiff", height=h, width=w, count=len(BANDS),
                       dtype="float32", crs=d["crs"], transform=d["transform"], nodata=np.nan, compress="deflate") as dst:
        dst.write(stack)
        dst.descriptions = tuple(BANDS)
    with rasterio.open(odir / "channels.tif", "w", driver="GTiff", height=h, width=w, count=3, dtype="uint8",
                       crs=d["crs"], transform=d["transform"], compress="deflate") as dst:
        dst.write(np.stack([d["drains"], d["river"], derived]).astype("uint8"))
        dst.descriptions = ("mapped_drains", "river", f"derived_acc_ge_{acc_threshold}")
    return out, d


if __name__ == "__main__":
    build()
