"""Download every raw dataset for the Bhubaneswar study area.

Usage: python -m src.download <dataset> [<dataset> ...]   (or "all")
Each dataset is skipped if its output already exists, so re-runs are cheap.
"""
import json
import os
import sys
import time
import warnings

# Streaming Cloud-Optimised GeoTIFFs: fail fast and retry instead of hanging on a stalled read.
os.environ.update(GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3",
                  GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIPLEX="YES",
                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.vrt,.gpkg", VSI_CACHE="TRUE")   # .vrt/.gpkg: AlphaEarth (aef)

import numpy as np
import pandas as pd
import planetary_computer as pc
import requests
import rioxarray  # noqa: F401  (registers .rio accessor)
import dask
import xarray as xr
from odc.stac import load as stac_load
from pystac_client import Client

from src.config import BBOX, CRS, PC_STAC, RAW, RES_10, RES_30, S2_BANDS, S2_SCL_KEEP

warnings.filterwarnings("ignore", category=RuntimeWarning)
dask.config.set(num_workers=2)  # 8 GB RAM machine
UA = {"User-Agent": "geotact-research/0.1" + (f" ({os.environ['GEOTACT_CONTACT']})" if os.environ.get("GEOTACT_CONTACT") else "")}  # contact for API operators


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def catalog():
    return Client.open(PC_STAC, modifier=pc.sign_inplace)


def search(collection, datetime=None, query=None):
    kw = dict(collections=[collection], bbox=BBOX)
    if datetime:
        kw["datetime"] = datetime
    if query:
        kw["query"] = query
    return list(catalog().search(**kw).items())


def save_tif(da, path, nodata=None):
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    da.rio.to_raster(path, compress="deflate", tiled=True)
    log(f"saved {path.name} {tuple(da.shape)} {da.dtype}")


def write_items(items, path):
    """Record exactly which scenes went into a product (provenance for the chapter)."""
    path.write_text(json.dumps(sorted(i.id for i in items), indent=1))


# ---------------------------------------------------------------- Sentinel-2

def select_items(datetime, max_cloud=20, n_best=8):
    """Scenes a composite uses: items under max_cloud, restricted to the n_best clearest acquisition days (slow link, ~1 MB/s)."""
    items = search("sentinel-2-l2a", datetime, {"eo:cloud_cover": {"lt": max_cloud}})
    days = sorted({i.datetime.date() for i in items},
                  key=lambda d: min(i.properties["eo:cloud_cover"] for i in items if i.datetime.date() == d))
    keep = set(days[:n_best])
    return [i for i in items if i.datetime.date() in keep], keep


def s2_composite(name, datetime, max_cloud=20, bands=S2_BANDS, n_best=8):
    out = RAW / f"s2_{name}.tif"
    if out.exists():
        return log(f"skip {out.name}")
    items, keep = select_items(datetime, max_cloud, n_best)
    log(f"S2 {name}: {len(items)} items on {len(keep)} clearest days")
    write_items(items, RAW / f"s2_{name}_scenes.json")

    # Stage 1: download each acquisition day to disk (resumable; a failed read costs one day).
    cache = RAW / "s2_cache" / name
    cache.mkdir(parents=True, exist_ok=True)
    day_files = []
    for day in sorted(keep):
        f = cache / f"{day}.tif"
        if not f.exists():
            di = [i for i in items if i.datetime.date() == day]
            da = load_with_retry(lambda: stac_load(
                di, bands=bands + ["SCL"], bbox=BBOX, crs=CRS, resolution=RES_10, groupby="solar_day",
                dtype="uint16", nodata=0).to_array("band").isel(time=0), f"S2 {name} {day}")
            if da is None:
                continue
            save_tif(da.assign_coords(band=bands + ["SCL"]), f, nodata=0)
        day_files.append(f)
    if len(day_files) < max(3, len(keep) // 2):
        raise RuntimeError(f"S2 {name}: only {len(day_files)}/{len(keep)} days downloaded")

    # Stage 2: cloud-masked median, one band at a time (fits in 8 GB RAM).
    # Harmonise: from processing baseline 04.00 (Jan 2022) L2A values carry BOA_ADD_OFFSET=-1000,
    # i.e. DN = 10000*reflectance + 1000. Remove it per day so every year shares one scale.
    offset = {}
    for f in day_files:
        day = f.stem
        pb = max(float(i.properties.get("s2:processing_baseline", "0")) for i in items
                 if str(i.datetime.date()) == day)
        offset[f] = 1000.0 if pb >= 4.0 else 0.0
    log(f"S2 {name}: offset-corrected days {sum(v > 0 for v in offset.values())}/{len(day_files)}")
    import rasterio
    with rasterio.open(day_files[0]) as r0:
        profile, shape = r0.profile, r0.shape
    clear = np.zeros((len(day_files),) + shape, bool)
    for k, f in enumerate(day_files):
        with rasterio.open(f) as r:
            clear[k] = np.isin(r.read(len(bands) + 1), S2_SCL_KEEP)
    med = np.zeros((len(bands),) + shape, np.uint16)
    for b in range(len(bands)):
        stack = np.full((len(day_files),) + shape, np.nan, np.float32)
        for k, f in enumerate(day_files):
            with rasterio.open(f) as r:
                v = r.read(b + 1).astype(np.float32)
            stack[k] = np.where(clear[k] & (v > 0), np.maximum(v - offset[f], 1.0), np.nan)
        med[b] = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0).astype(np.uint16)
    profile.update(count=len(bands), dtype="uint16", nodata=0, compress="deflate", tiled=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(med)
        dst.descriptions = tuple(bands)
    profile.update(count=1, dtype="uint8", nodata=None)
    with rasterio.open(RAW / f"s2_{name}_clearcount.tif", "w", **profile) as dst:
        dst.write(clear.sum(0).astype(np.uint8), 1)
    log(f"saved {out.name} from {len(day_files)} days; pixels with 0 clear days: {(clear.sum(0) == 0).mean():.2%}")


def load_with_retry(fn, label, attempts=4):
    for a in range(attempts):
        try:
            return fn()
        except Exception as e:
            log(f"{label}: attempt {a + 1} failed: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(15 * (a + 1))
    log(f"{label}: giving up after {attempts} attempts")
    return None


def sentinel2():
    # Target year for mapping (dry season, least cloud).
    s2_composite("2025_26_dry", "2025-11-01/2026-03-31")
    # Label-matched years: WorldCover 2021 and Esri/IO annual LULC 2023 were built from these years.
    s2_composite("2021", "2021-01-01/2021-12-31", max_cloud=10, n_best=10)
    s2_composite("2023", "2023-01-01/2023-12-31", max_cloud=10, n_best=10)


def ndvi_years():
    """Dry-season (Nov-Mar) red + NIR composites per year for the vegetation-health trend."""
    for y in range(2018, 2026):
        s2_composite(f"ndvi_{y}_{(y + 1) % 100:02d}", f"{y}-11-01/{y + 1}-03-31",
                     max_cloud=20, bands=["B04", "B08"], n_best=5)


# ---------------------------------------------------------------- Land cover

def landcover():
    wc = search("esa-worldcover")
    for year in (2020, 2021):
        out = RAW / f"worldcover_{year}.tif"
        if out.exists():
            log(f"skip {out.name}"); continue
        items = [i for i in wc if f"_{year}_" in i.id]
        log(f"WorldCover {year}: {[i.id for i in items]}")
        ds = stac_load(items, bands=["map"], bbox=BBOX, crs=CRS, resolution=RES_10,
                       resampling="nearest", dtype="uint8", nodata=0, groupby="id")
        save_tif(ds["map"].max("time"), out, nodata=0)

    items = search("io-lulc-annual-v02")
    for year in sorted({i.datetime.year if i.datetime else int(i.properties["start_datetime"][:4]) for i in items}):
        out = RAW / f"io_lulc_{year}.tif"
        if out.exists():
            log(f"skip {out.name}"); continue
        yi = [i for i in items if (i.properties.get("start_datetime") or "")[:4] == str(year)]
        ds = stac_load(yi, bands=["data"], bbox=BBOX, crs=CRS, resolution=RES_10,
                       resampling="nearest", dtype="uint8", nodata=0, groupby="id")
        da = ds["data"].max("time") if "time" in ds.dims else ds["data"]
        save_tif(da, out, nodata=0)


# ---------------------------------------------------------------- Elevation

def dem():
    out = RAW / "copdem30.tif"
    if out.exists():
        return log(f"skip {out.name}")
    items = search("cop-dem-glo-30")
    ds = stac_load(items, bands=["data"], bbox=BBOX, crs=CRS, resolution=RES_30,
                   resampling="bilinear", dtype="float32", nodata=-9999)
    save_tif(ds["data"].max("time"), out, nodata=-9999)


# ---------------------------------------------------------------- Sentinel-1 (flood)

def sentinel1():
    """Monsoon 2025 scenes (flood evidence) + dry-season 2025 reference (permanent water)."""
    folder = RAW / "s1"
    folder.mkdir(exist_ok=True)
    for tag, dt in [("monsoon", "2025-06-01/2025-10-31"), ("dry", "2025-01-01/2025-04-30")]:
        items = search("sentinel-1-rtc", dt)
        log(f"S1 {tag}: {len(items)} items")
        write_items(items, folder / f"{tag}_scenes.json")
        days = sorted({i.datetime.date() for i in items})
        for day in days:
            out = folder / f"s1_{tag}_{day}.tif"
            if out.exists():
                continue
            di = [i for i in items if i.datetime.date() == day]
            da = load_with_retry(lambda: stac_load(
                di, bands=["vv", "vh"], bbox=BBOX, crs=CRS, resolution=RES_10, groupby="solar_day",
                dtype="float32", nodata=np.nan)[["vv", "vh"]].to_array("band").isel(time=0), f"S1 {day}")
            if da is None:
                continue
            da = da.assign_coords(band=["vv", "vh"])
            if float(np.isfinite(da.values).mean()) < 0.5:
                log(f"S1 {day}: <50% coverage, kept anyway")
            save_tif(da, out, nodata=np.nan)


# ---------------------------------------------------------------- Landsat surface temperature

def landsat_lst():
    for tag, dt in [("2025", "2025-03-01/2025-05-31"), ("2026", "2026-03-01/2026-05-31")]:
        out = RAW / f"lst_summer_{tag}.tif"
        if out.exists():
            log(f"skip {out.name}"); continue
        items = search("landsat-c2-l2", dt, {"eo:cloud_cover": {"lt": 30},
                                              "platform": {"in": ["landsat-8", "landsat-9"]}})
        log(f"Landsat {tag}: {len(items)} scenes")
        write_items(items, RAW / f"lst_summer_{tag}_scenes.json")
        ds = stac_load(items, bands=["lwir11", "qa_pixel"], bbox=BBOX, crs=CRS, resolution=RES_30,
                       groupby="solar_day", chunks={"x": 2048, "y": 2048}, dtype="uint16", nodata=0)
        clear = (ds["qa_pixel"] & (1 << 6)) > 0          # QA bit 6 = clear
        st = ds["lwir11"].where(clear & (ds["lwir11"] > 0)) * 0.00341802 + 149.0 - 273.15
        save_tif(st.median("time").astype("float32").compute(), out, nodata=np.nan)


# ---------------------------------------------------------------- Rainfall (ERA5 via Open-Meteo)

def rainfall():
    out = RAW / "rain_era5_daily.csv"
    if out.exists():
        return log(f"skip {out.name}")
    rows = []
    for lat in (20.20, 20.35):
        for lon in (85.75, 85.90):
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", headers=UA, timeout=120, params=dict(
                latitude=lat, longitude=lon, start_date="2019-01-01", end_date="2026-09-30",
                daily="precipitation_sum,temperature_2m_max,temperature_2m_min", timezone="Asia/Kolkata"))
            r.raise_for_status()
            d = pd.DataFrame(r.json()["daily"]).assign(lat=lat, lon=lon)
            rows.append(d)
    df = pd.concat(rows)
    df.to_csv(out, index=False)
    log(f"saved {out.name} rows={len(df)}")


# ---------------------------------------------------------------- GBIF

GBIF_FIELDS = ["key", "datasetKey", "basisOfRecord", "kingdom", "phylum", "class", "order", "family",
               "species", "scientificName", "decimalLatitude", "decimalLongitude",
               "coordinateUncertaintyInMeters", "year", "month", "eventDate", "iucnRedListCategory"]


def gbif_query(params):
    base = dict(decimalLatitude=f"{BBOX[1]},{BBOX[3]}", decimalLongitude=f"{BBOX[0]},{BBOX[2]}",
                hasCoordinate="true", hasGeospatialIssue="false")
    base.update(params)
    url = "https://api.gbif.org/v1/occurrence/search"
    for attempt in range(6):
        try:
            n = requests.get(url, params={**base, "limit": 0}, headers=UA, timeout=120).json()["count"]
            return url, base, n
        except Exception as e:
            log(f"GBIF count retry {params}: {e}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"GBIF count failed for {params}")


def gbif():
    out = RAW / "gbif_occurrences.csv"
    if out.exists():
        return log(f"skip {out.name}")
    slices = [{"year": "1700,1999"}] + [{"year": str(y)} for y in range(2000, 2027)]
    recs = []
    for s in slices:
        url, base, n = gbif_query(s)
        sub = [s] if n <= 99_000 else [{**s, "month": str(m)} for m in range(1, 13)]
        for q in sub:
            url, base, n = gbif_query(q)
            off = 0
            while off < n:
                for attempt in range(5):
                    try:
                        r = requests.get(url, params={**base, "limit": 300, "offset": off}, headers=UA, timeout=120)
                        r.raise_for_status(); break
                    except Exception as e:
                        log(f"GBIF retry {q} off={off}: {e}"); time.sleep(5 * (attempt + 1))
                res = r.json()["results"]
                if not res:
                    break
                recs += [{k: x.get(k) for k in GBIF_FIELDS} for x in res]
                off += 300
        log(f"GBIF {s}: {n} (total so far {len(recs)})")
    df = pd.DataFrame(recs).drop_duplicates("key")
    df.to_csv(out, index=False)
    log(f"saved {out.name} rows={len(df)}")


# ---------------------------------------------------------------- OpenStreetMap

OSM_QUERY = """[out:json][timeout:300];
(
  nwr["natural"~"^(water|wetland|wood|scrub|grassland)$"]({s},{w},{n},{e});
  nwr["waterway"]({s},{w},{n},{e});
  nwr["landuse"~"^(farmland|orchard|meadow|grass|forest|recreation_ground|allotments|reservoir|basin|village_green|plant_nursery)$"]({s},{w},{n},{e});
  nwr["leisure"~"^(park|garden|nature_reserve)$"]({s},{w},{n},{e});
  nwr["boundary"="administrative"]["admin_level"~"^(8|9|10)$"]({s},{w},{n},{e});
);
out geom;"""

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]


def osm():
    out = RAW / "osm_raw.json"
    if out.exists():
        return log(f"skip {out.name}")
    w, s, e, n = BBOX
    q = OSM_QUERY.format(s=s, w=w, n=n, e=e)
    for attempt in range(3):
        for ep in OVERPASS:
            try:
                r = requests.post(ep, data={"data": q}, headers=UA, timeout=400)
                r.raise_for_status()
                js = r.json()
                out.write_text(json.dumps(js))
                return log(f"saved {out.name} elements={len(js['elements'])} via {ep}")
            except Exception as ex:
                log(f"OSM {ep} failed: {str(ex)[:100]}")
        time.sleep(60)
    log("OSM: all endpoints failed")


def roads():
    """OpenStreetMap highway lines for the study box (needed by SDG 11.7.1)."""
    out = RAW / "osm_highways.json"
    if out.exists():
        return log(f"skip {out.name}")
    w, s, e, n = BBOX
    q = f'[out:json][timeout:300];way["highway"]({s},{w},{n},{e});out geom;'
    for ep in OVERPASS:
        try:
            r = requests.post(ep, data={"data": q}, headers=UA, timeout=400)
            r.raise_for_status()
            js = r.json()
            out.write_text(json.dumps(js))
            return log(f"saved {out.name} ways={len(js['elements'])} via {ep}")
        except Exception as ex:
            log(f"roads {ep} failed: {str(ex)[:100]}")
    log("roads: all endpoints failed")


AEF_INDEX = "/vsicurl/https://data.source.coop/tge-labs/aef/v1/annual/aef_index.gpkg"


def aef(years=(2025,), block=256):
    """AlphaEarth Foundations annual satellite embedding (64 int8 channels, de-quantise with sign(v) * (v / 127.5)^2, -128 = no data;
    CC-BY 4.0, 'produced by Google and Google DeepMind'), read from the public Source Cooperative copy through its VRT (which fixes
    the bottom-up row order) for exactly our 10 m grid window. Raw int8 values are stored."""
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window

    from src import grid
    g = grid.load_grid()
    idx = gpd.read_file(AEF_INDEX, bbox=BBOX)
    for year in years:
        out = RAW / f"aef_{year}.tif"
        if out.exists():
            log(f"skip {out.name}"); continue
        row = idx[(idx.year == year) & (idx["crs"] == CRS)]          # the column, not GeoDataFrame.crs (the frame's own CRS)
        x0, y0 = g.transform.c, g.transform.f
        x1, y1 = x0 + g.shape[1] * g.res, y0 - g.shape[0] * g.res
        row = row[(row.utm_west <= x0) & (row.utm_east >= x1) & (row.utm_north >= y0) & (row.utm_south <= y1)]
        if len(row) != 1:
            raise RuntimeError(f"AEF {year}: {len(row)} files cover the whole grid; a mosaic would be needed")
        url = "/vsicurl/https://data.source.coop/" + row.path.iloc[0].split("opendata.source.coop/")[1].replace(".tiff", ".vrt")
        # the VRT names its source as /vsis3/ on a public bucket: anonymous access (checked: VRT read == flipped direct read)
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", AWS_REGION="us-west-2"), rasterio.open(url) as src:
            c0, r0 = (x0 - src.transform.c) / src.transform.a, (src.transform.f - y0) / -src.transform.e
            assert float(c0).is_integer() and float(r0).is_integer() and src.transform.a == g.res, "AEF grid not aligned with ours"
            prof = dict(driver="GTiff", height=g.shape[0], width=g.shape[1], count=64, dtype="int8", nodata=-128, crs=g.crs,
                        transform=g.transform, compress="deflate", tiled=True)
            with rasterio.open(out.with_suffix(".part.tif"), "w", **prof) as dst:
                for r in range(0, g.shape[0], block):
                    h = min(block, g.shape[0] - r)
                    w = Window(int(c0), int(r0) + r, g.shape[1], h)
                    data = load_with_retry(lambda: src.read(window=w), f"AEF {year} rows {r}")
                    if data is None:
                        raise RuntimeError(f"AEF {year}: rows {r} failed")
                    dst.write(data, window=Window(0, r, g.shape[1], h))
                dst.descriptions = tuple(f"A{k:02d}" for k in range(64))
        out.with_suffix(".part.tif").rename(out)
        log(f"saved {out.name} from {row.path.iloc[0]}")


GLANCE_URL = "https://nccd.gov.in/uploads/Agricultural_Statisitcs_at_a_Glance_2024_25_11da3cd37d.pdf"


def glance():
    """Ministry of Agriculture & Farmers Welfare, 'Agricultural Statistics at a Glance 2024-25' (state-wise yields; T14 food factors)."""
    out = RAW / "external" / "moa" / "Agricultural_Statistics_at_a_Glance_2024_25.pdf"
    if out.exists():
        return log(f"skip {out.name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(GLANCE_URL, timeout=1800, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    out.write_bytes(r.content)
    log(f"saved {out.name} {len(r.content)} bytes")


DATASETS = dict(dem=dem, landcover=landcover, sentinel2=sentinel2, ndvi=ndvi_years, sentinel1=sentinel1,
                lst=landsat_lst, rain=rainfall, gbif=gbif, osm=osm, roads=roads, aef=aef, glance=glance)

if __name__ == "__main__":
    from src.wardmap import memory_guard
    memory_guard(3.0)
    names = list(DATASETS) if sys.argv[1:] == ["all"] else sys.argv[1:]
    for nm in names:
        log(f"=== {nm}")
        try:
            DATASETS[nm]()
        except Exception as ex:
            log(f"FAILED {nm}: {type(ex).__name__}: {ex}")
    log("done")
