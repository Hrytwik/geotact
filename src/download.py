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
                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff", VSI_CACHE="TRUE")

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
UA = {"User-Agent": "geotact-research/0.1 ([contact email removed])"}


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

def s2_composite(name, datetime, max_cloud=20, bands=S2_BANDS, n_best=8):
    out = RAW / f"s2_{name}.tif"
    if out.exists():
        return log(f"skip {out.name}")
    items = search("sentinel-2-l2a", datetime, {"eo:cloud_cover": {"lt": max_cloud}})
    # Slow link (~1 MB/s): keep only the clearest acquisition days for the median composite.
    days = sorted({i.datetime.date() for i in items},
                  key=lambda d: min(i.properties["eo:cloud_cover"] for i in items if i.datetime.date() == d))
    keep = set(days[:n_best])
    items = [i for i in items if i.datetime.date() in keep]
    log(f"S2 {name}: {len(items)} items on {len(keep)} clearest days")
    write_items(items, RAW / f"s2_{name}_scenes.json")
    ds = stac_load(items, bands=bands + ["SCL"], bbox=BBOX, crs=CRS, resolution=RES_10,
                   groupby="solar_day", chunks={"x": 512, "y": 512}, dtype="uint16", nodata=0)
    clear = ds["SCL"].isin(S2_SCL_KEEP)
    stack = ds[bands].to_array("band").astype("float32").where(clear)
    stack = stack.where(stack > 0)
    med = stack.median("time", skipna=True)
    valid = clear.sum("time")
    from dask.diagnostics import ProgressBar
    with ProgressBar(dt=60):
        med = med.fillna(0).astype("uint16").compute()
    med = med.assign_coords(band=bands)
    save_tif(med, out, nodata=0)
    save_tif(valid.astype("uint8").compute(), RAW / f"s2_{name}_clearcount.tif")


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
            ds = stac_load(di, bands=["vv", "vh"], bbox=BBOX, crs=CRS, resolution=RES_10,
                           groupby="solar_day", dtype="float32", nodata=np.nan,
                           chunks={"x": 2048, "y": 2048})
            da = ds[["vv", "vh"]].to_array("band").isel(time=0).compute()
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


DATASETS = dict(dem=dem, landcover=landcover, sentinel2=sentinel2, ndvi=ndvi_years, sentinel1=sentinel1,
                lst=landsat_lst, rain=rainfall, gbif=gbif, osm=osm)

if __name__ == "__main__":
    names = list(DATASETS) if sys.argv[1:] == ["all"] else sys.argv[1:]
    for nm in names:
        log(f"=== {nm}")
        try:
            DATASETS[nm]()
        except Exception as ex:
            log(f"FAILED {nm}: {type(ex).__name__}: {ex}")
    log("done")
