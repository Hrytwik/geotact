"""Generate the stratified random sample of points for hand-labelled accuracy assessment.

Design (Olofsson et al. 2014 good practice; Stehman 2014 estimator):
- Strata come from Esri/IO LULC 2023, collapsed to the chapter's 6 classes.
- Allocation: 40 points per stratum + remaining 60 proportional to stratum area,
  so rare classes (water, bare) still get enough points.
- The labeller never sees the stratum (blind labelling); strata are kept in a separate file.
"""
import numpy as np
import pandas as pd
import rasterio
import simplekml
from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation
from pyproj import Transformer

from src.config import CRS, RAW, ROOT

CLASSES = ["Water", "Trees", "Cropland", "Grass/Scrub", "Built-up", "Bare ground"]
# Esri/IO LULC codes -> chapter classes (4 = flooded vegetation goes to Water).
IO_TO_CLASS = {1: 0, 4: 0, 2: 1, 5: 2, 11: 3, 7: 4, 8: 5}
N_TOTAL, N_MIN, SEED = 300, 40, 2026

OUT = ROOT / "validation"


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    with rasterio.open(RAW / "io_lulc_2023.tif") as r:
        io = r.read(1)
        transform = r.transform
    strata = np.full(io.shape, 255, np.uint8)
    for code, cls in IO_TO_CLASS.items():
        strata[io == code] = cls

    counts = np.array([(strata == k).sum() for k in range(len(CLASSES))])
    weights = counts / counts.sum()
    extra = N_TOTAL - N_MIN * len(CLASSES)
    alloc = N_MIN + np.floor(extra * weights).astype(int)
    alloc[np.argmax(weights)] += N_TOTAL - alloc.sum()

    to_wgs = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    rows = []
    for k, n in enumerate(alloc):
        rr, cc = np.nonzero(strata == k)
        pick = rng.choice(len(rr), size=n, replace=False)
        for i in pick:
            x, y = rasterio.transform.xy(transform, rr[i], cc[i])  # pixel centre
            lon, lat = to_wgs.transform(x, y)
            rows.append(dict(stratum=CLASSES[k], x_utm=x, y_utm=y, lat=round(lat, 6), lon=round(lon, 6)))

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "id", [f"P{i + 1:03d}" for i in range(len(df))])

    # Private file: strata + weights needed later for the stratified accuracy estimator.
    df.to_csv(OUT / "strata_private_do_not_open.csv", index=False)
    pd.DataFrame(dict(cls=CLASSES, pixels=counts, weight=weights, n=alloc)).to_csv(
        OUT / "strata_weights.csv", index=False)

    write_kml(df)
    write_sheet(df)
    print(pd.DataFrame(dict(cls=CLASSES, area_pct=(100 * weights).round(1), points=alloc)).to_string(index=False))


def write_kml(df):
    kml = simplekml.Kml(name="Bhubaneswar validation points")
    half = 5.0  # 10 m pixel -> 5 m each side
    to_wgs = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    for p in df.itertuples():
        pt = kml.newpoint(name=p.id, coords=[(p.lon, p.lat)])
        pt.style.iconstyle.scale = 0.6
        sq = [to_wgs.transform(p.x_utm + dx, p.y_utm + dy)
              for dx, dy in [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]]
        pol = kml.newpolygon(name=f"{p.id} pixel", outerboundaryis=sq)
        pol.style.polystyle.fill = 0
        pol.style.linestyle.color = simplekml.Color.yellow
        pol.style.linestyle.width = 2
    kml.save(str(OUT / "validation_points.kml"))


def write_sheet(df):
    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    headers = ["id", "lat", "lon", "label", "confidence", "imagery_date", "notes"]
    ws.append(headers)
    for p in df.itertuples():
        ws.append([p.id, p.lat, p.lon, None, None, None, None])
    dv_label = DataValidation(type="list", formula1='"' + ",".join(CLASSES) + '"', allow_blank=True)
    dv_conf = DataValidation(type="list", formula1='"Sure,Unsure"', allow_blank=True)
    ws.add_data_validation(dv_label)
    ws.add_data_validation(dv_conf)
    dv_label.add(f"D2:D{len(df) + 1}")
    dv_conf.add(f"E2:E{len(df) + 1}")
    for col, w in zip("ABCDEFG", [8, 11, 11, 14, 12, 14, 30]):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"

    guide = wb.create_sheet("how_to_label")
    for line in [
        "1. Open validation_points.kml in Google Earth Pro (File > Open).",
        "2. Double-click a point (P001...) in the left panel to fly to it.",
        "3. Look ONLY inside the small yellow square (10 m x 10 m).",
        "4. Pick what covers MOST of that square, using the most recent imagery.",
        "5. In the 'labels' sheet choose the label and confidence from the dropdowns.",
        "6. Write the imagery date (shown at the bottom of Google Earth) in imagery_date.",
        "",
        "Classes:",
        "Water - rivers, ponds, lakes, flooded areas, wetlands with open water",
        "Trees - tree canopy, orchards, plantations, forest",
        "Cropland - fields (planted or ploughed), paddy",
        "Grass/Scrub - grass, bushes, open green land, sports fields with grass",
        "Built-up - buildings, roads, paved areas, construction",
        "Bare ground - exposed soil, sand, rock, quarries, river sand bars",
        "",
        "If you can't decide, pick the best guess and mark 'Unsure'. Don't skip points.",
    ]:
        guide.append([line])
    guide.column_dimensions["A"].width = 90
    wb.save(OUT / "validation_labels.xlsx")


if __name__ == "__main__":
    main()
