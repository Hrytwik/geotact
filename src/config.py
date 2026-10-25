"""Shared study-area and grid settings used by every pipeline step. The city is a setting (environment variable GEOTACT_CITY);
Bhubaneswar is the default. Trained models always live under HOME_OUTPUTS (the Bhubaneswar outputs)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITY = os.environ.get("GEOTACT_CITY", "bhubaneswar").strip().lower()
HOME_OUTPUTS = ROOT / "outputs"

if CITY == "bhubaneswar":
    RAW = ROOT / "data" / "raw"
    INTERIM = ROOT / "data" / "interim"
    OUTPUTS = HOME_OUTPUTS
    # Bhubaneswar Municipal Corporation extent (from Ward_Map_1.pdf GeoPDF /GPTS:
    # lat 20.17715-20.38916, lon 85.72931-85.90413) plus ~2 km buffer.
    BBOX = (85.70, 20.15, 85.93, 20.41)  # (min_lon, min_lat, max_lon, max_lat)
elif CITY == "cuttack":
    RAW = ROOT / "data" / "cuttack" / "raw"
    INTERIM = ROOT / "data" / "cuttack" / "interim"
    OUTPUTS = ROOT / "outputs_cuttack"
    BBOX = (85.76, 20.40, 85.98, 20.56)  # Cuttack Municipal Corporation area plus a buffer (T11)
else:
    raise ValueError(f"unknown GEOTACT_CITY {CITY!r}")

# UTM zone 45N covers 84-90 E; all rasters share this CRS and anchor.
CRS = "EPSG:32645"
RES_10 = 10
RES_30 = 30

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# Sentinel-2 scene classification: keep vegetation(4), bare(5), water(6), unclassified(7);
# drop no-data, saturated, dark, cloud shadow, clouds, cirrus, snow.
S2_SCL_KEEP = [4, 5, 6, 7]

for d in (RAW, INTERIM, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)
