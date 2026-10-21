"""Digitise the field-survey flood inventory from Figure 3 of Panigrahi, Pal and Sharma (2026), AIMS Geosciences 12(1):127-157
(open access). The authors did not publish coordinates; the figure is a longitude/latitude map of 130 field-surveyed
locations (June-September 2023): green = non-flood points, blue stars / black circles / red triangles = flooding of
0.75 / 0.85 / 1.2 m. Positions are digitised from the figure, so each carries roughly +-30-50 m of error, and the
counts inside overlapping clusters are approximate. Cite the paper for the data and say it was digitised.
Pure numpy/scikit-image: nothing here imports torch."""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import match_template, peak_local_max
from skimage.morphology import binary_opening, disk, skeletonize

from src.config import INTERIM, RAW

PDF = RAW / "external" / "panigrahi2026_geosci-12-01-005.pdf"
FIG_PAGE = 7                                                  # PDF page 8
LON_TICKS = [85 + 45 / 60, 85 + 47.5 / 60, 85 + 50 / 60, 85 + 52.5 / 60]              # 85d45'E ... 85d52'30"E
LAT_TICKS = [20 + 22.5 / 60, 20 + 20 / 60, 20 + 17.5 / 60, 20 + 15 / 60, 20 + 12.5 / 60]   # 20d22'30"N ... 20d12'30"N
# isolated example markers (pixel x, y) used as templates; read off the figure
SEEDS = {"star": [(546, 357), (503, 375), (558, 493)], "black": [(897, 916), (1058, 823), (590, 1090)],
         "red": [(1277, 283), (1205, 462), (1137, 812)]}
COLOURS = {"green": ((80, 255, 0), 40), "star": ((46, 102, 197), 45), "black": ((0, 0, 2), 40), "red": ((240, 18, 17), 60)}
CLASSES = {"green": "non_flood", "star": "flood_0.75m", "black": "flood_0.85m", "red": "flood_1.2m"}


# ---------------------------------------------------------------- graticule and georeference
def _runs(idx):
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for v in idx[1:]:
        if v != p + 1:
            out.append((s + p) / 2.0)
            s = v
        p = v
    out.append((s + p) / 2.0)
    return out


def find_ticks(gray, n_x, n_y, dark_level=110):
    """Pixel positions of the graticule ticks drawn just outside the map frame (x ticks above the top frame, y ticks left of
    the left frame). Raises if the expected number of ticks is not found."""
    dark = np.asarray(gray) < dark_level
    H, W = dark.shape
    rows = np.where(dark.sum(1) > 0.6 * W)[0]
    cols = np.where(dark.sum(0) > 0.6 * H)[0]
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    xs = _runs(np.where(dark[top - 10:top - 1, left:right + 1].sum(0) >= 5)[0] + left)
    ys = _runs(np.where(dark[top:bottom + 1, left - 10:left - 1].sum(1) >= 5)[0] + top)
    if len(xs) != n_x or len(ys) != n_y:
        raise ValueError(f"expected {n_x} x ticks and {n_y} y ticks, found {len(xs)} and {len(ys)}")
    return xs, ys


def fit_axis(pixels, values):
    """Least-squares line pixel = a * value + b; returns (a, b)."""
    return np.polyfit(np.asarray(values, float), np.asarray(pixels, float), 1)


def pixel_to_lonlat(x, y, ax, ay):
    return (np.asarray(x, float) - ax[1]) / ax[0], (np.asarray(y, float) - ay[1]) / ay[0]


# ---------------------------------------------------------------- marker detection
def learn_template(rgb, colour, tol, seeds, radius):
    """Mean (2*radius+1)^2 crop around isolated example markers, centred on the marker colour's centre of mass."""
    fm = np.abs(rgb - np.array(colour)).max(2) <= tol
    crops = []
    for sx, sy in seeds:
        win = fm[sy - 14:sy + 15, sx - 14:sx + 15]
        lab, n = ndi.label(win)
        a = np.bincount(lab.ravel())[1:]
        cy, cx = ndi.center_of_mass(lab == (np.argmax(a) + 1))
        cx, cy = int(round(sx - 14 + cx)), int(round(sy - 14 + cy))
        crops.append(rgb[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1, :])
    return np.mean(crops, axis=0)


def match_peaks(rgb, template, threshold, min_dist, valid=None):
    """(x, y) pixel positions where the template correlates at least `threshold` with the image."""
    cc = np.mean([match_template(rgb[:, :, k], template[:, :, k], pad_input=True) for k in range(3)], axis=0)
    if valid is not None:
        cc[~valid] = -1
    pk = peak_local_max(cc, min_distance=min_dist, threshold_abs=threshold, exclude_border=False)
    return pk[:, ::-1].astype(float) if len(pk) else np.zeros((0, 2))


def cluster_points(mask, single_area, step, min_ratio=1.5, open_radius=3):
    """Sample points along merged blobs of overlapping markers: blobs larger than min_ratio single markers are thinned to a
    skeleton and represented by about (skeleton length / step) k-means centres. Isolated markers return nothing."""
    from sklearn.cluster import KMeans
    m = binary_opening(mask, disk(open_radius))
    lab, n = ndi.label(m)
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() <= min_ratio * single_area:
            continue
        ys, xs = np.nonzero(skeletonize(comp))
        pts = np.column_stack([xs, ys]).astype(float)
        k = int(max(2, min(len(pts), round(len(pts) / step))))
        out.append(KMeans(n_clusters=k, n_init=3, random_state=0).fit(pts).cluster_centers_)
    return np.vstack(out) if out else np.zeros((0, 2))


def dedupe(points, tol):
    """Greedy merge: keep a point only if it is farther than tol from every point already kept."""
    kept = []
    for p in np.asarray(points, float):
        if all(np.hypot(*(p - q)) > tol for q in kept):
            kept.append(p)
    return np.array(kept) if kept else np.zeros((0, 2))


# ---------------------------------------------------------------- the real figure
def extract_figure(pdf=PDF, page=FIG_PAGE):
    import io

    import pymupdf
    doc = pymupdf.open(pdf)
    imgs = doc[page].get_images(full=True)
    xref = max(imgs, key=lambda i: i[2] * i[3])[0]
    return Image.open(io.BytesIO(doc.extract_image(xref)["image"])).convert("RGB")


def digitise(img=None):
    import pandas as pd
    img = extract_figure() if img is None else img
    rgb = np.array(img).astype(float)
    H, W, _ = rgb.shape
    gray = np.array(img.convert("L"))
    tx, ty = find_ticks(gray, len(LON_TICKS), len(LAT_TICKS))
    ax, ay = fit_axis(tx, LON_TICKS), fit_axis(ty, LAT_TICKS)
    valid = np.ones((H, W), bool)
    valid[:60, :] = False                                                  # titles above the frame
    valid[1330:, 1000:] = False                                            # legend block
    valid[1700:, :] = False                                                # scale bar and axis labels
    pts = {}
    # green: plain components
    g = np.abs(rgb - np.array(COLOURS["green"][0])).max(2) <= COLOURS["green"][1]
    lab, n = ndi.label(binary_opening(g & valid, disk(1)))
    pts["green"] = np.array(ndi.center_of_mass(g & valid, lab, range(1, n + 1)))[:, ::-1]
    # stars: template matching only (the stream network has the same blue)
    star_T = learn_template(rgb, *COLOURS["star"], SEEDS["star"], 10)
    pts["star"] = match_peaks(rgb, star_T, 0.6, 7, valid)
    for name, thr, md, step, area_cut in (("black", 0.7, 8, 10, 500), ("red", 0.6, 6, 7, 90)):
        colour, tol = COLOURS[name]
        T = learn_template(rgb, colour, tol, SEEDS[name], 12 if name == "black" else 10)
        pk = match_peaks(rgb, T, thr, md, valid)
        pk = pk[[np.abs(rgb[int(y), int(x)] - np.array(colour)).max() <= tol + 20 for x, y in pk]] if len(pk) else pk
        mask = (np.abs(rgb - np.array(colour)).max(2) <= tol) & valid
        lab, n = ndi.label(binary_opening(mask, disk(2 if name == "black" else 1)))
        a = np.array(ndi.sum(mask, lab, range(1, n + 1)))
        single = float(np.median(a[a < area_cut]))
        cl = cluster_points(mask, single, step, open_radius=3 if name == "black" else 1)
        pts[name] = dedupe(np.vstack([pk, cl]), 6.0)
    rows = []
    for name, p in pts.items():
        lon, lat = pixel_to_lonlat(p[:, 0], p[:, 1], ax, ay)
        rows += [dict(cls=CLASSES[name], x_px=float(x), y_px=float(y), lon=float(lo), lat=float(la))
                 for (x, y), lo, la in zip(p, lon, lat)]
    return pd.DataFrame(rows), dict(ax=ax, ay=ay, ticks_x=tx, ticks_y=ty)


def outline_offset(img, ax, ay, n=3000):
    """Median distance (pixels) between our reconstructed BMC outline, projected into the figure, and the dark outline drawn
    in the figure. A registration check that is independent of the graticule fit."""
    import geopandas as gpd

    from src.config import CRS
    city = gpd.read_file(INTERIM / "wardmap.gpkg", layer="municipal_poly").to_crs("EPSG:4326").geometry.iloc[0]
    b = city.exterior
    pts = np.array([b.interpolate(d).coords[0] for d in np.linspace(0, b.length, n)])
    px, py = np.polyval(ax, pts[:, 0]), np.polyval(ay, pts[:, 1])
    dist = ndi.distance_transform_edt(~(np.array(img.convert("RGB")).astype(int).sum(2) < 250))
    ok = (px >= 0) & (px < dist.shape[1] - 1) & (py >= 0) & (py < dist.shape[0] - 1)
    return float(np.median(dist[py[ok].astype(int), px[ok].astype(int)]))


def save(df):
    import geopandas as gpd

    from src.config import CRS
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326").to_crs(CRS)
    g.to_file(INTERIM / "panigrahi2026_fig3_points.gpkg", driver="GPKG")
    df.to_csv(INTERIM / "panigrahi2026_fig3_points.csv", index=False)
    return g


if __name__ == "__main__":
    d, info = digitise()
    save(d)
    print(d.groupby("cls").size().to_string(), "| total", len(d))
