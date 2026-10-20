"""Shared study-area and grid settings used by every pipeline step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
OUTPUTS = ROOT / "outputs"

# Bhubaneswar Municipal Corporation extent (from Ward_Map_1.pdf GeoPDF /GPTS:
# lat 20.17715-20.38916, lon 85.72931-85.90413) plus ~2 km buffer.
BBOX = (85.70, 20.15, 85.93, 20.41)  # (min_lon, min_lat, max_lon, max_lat)

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
