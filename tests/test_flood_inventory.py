"""Digitising the field-survey flood inventory from Figure 3 of Panigrahi, Pal and Sharma (2026): synthetic checks first,
then the real figure."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from src import flood_inventory as fi


def canvas(w=300, h=260):
    return Image.new("RGB", (w, h), (255, 255, 255))


# ---------------------------------------------------------------- graticule
def test_ticks_are_found_outside_the_map_frame_and_give_an_exact_affine():
    im = canvas(400, 300)
    d = ImageDraw.Draw(im)
    d.rectangle([40, 30, 360, 270], outline=(0, 0, 0), width=3)                      # map frame
    xs, ys = [60, 140, 220, 300], [50, 100, 150, 200]                                  # evenly spaced, as on a real graticule
    for x in xs:
        d.line([(x, 20), (x, 28)], fill=(0, 0, 0), width=2)                          # ticks above the top frame
    for y in ys:
        d.line([(30, y), (38, y)], fill=(0, 0, 0), width=2)                          # ticks left of the left frame
    tx, ty = fi.find_ticks(np.array(im.convert("L")), n_x=4, n_y=4)
    assert tx == pytest.approx(xs, abs=1.0) and ty == pytest.approx(ys, abs=1.0)
    ax, ay = fi.fit_axis(tx, [10.0, 10.5, 11.0, 11.5]), fi.fit_axis(ty, [20.0, 19.5, 19.0, 18.5])
    lon, lat = fi.pixel_to_lonlat(np.array([140.0]), np.array([100.0]), ax, ay)
    assert lon[0] == pytest.approx(10.5, abs=0.01) and lat[0] == pytest.approx(19.5, abs=0.01)


# ---------------------------------------------------------------- template matching
def draw_circles(im, centres, r=9, fill=(0, 0, 2)):
    d = ImageDraw.Draw(im)
    for x, y in centres:
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
    return im


def test_template_peaks_recover_planted_markers_and_ignore_thin_lines():
    centres = [(50, 60), (150, 70), (80, 180), (220, 200)]
    im = draw_circles(canvas(), centres)
    ImageDraw.Draw(im).line([(0, 120), (300, 120)], fill=(0, 0, 0), width=2)            # a thin line must not match
    rgb = np.array(im).astype(float)
    T = fi.learn_template(rgb, (0, 0, 2), 40, seeds=[centres[0], centres[1]], radius=12)
    pk = fi.match_peaks(rgb, T, threshold=0.7, min_dist=8)
    assert len(pk) == 4
    for (cx, cy) in centres:
        assert np.min(np.hypot(pk[:, 0] - cx, pk[:, 1] - cy)) <= 1.5                   # peaks are (x, y)


# ---------------------------------------------------------------- merged clusters
def test_a_chain_of_overlapping_circles_is_sampled_along_its_length():
    im = canvas(320, 320)
    chain = [(40 + 6 * k, 40 + 6 * k) for k in range(40)]                              # 40 circles overlapping heavily
    draw_circles(im, chain, r=9)
    mask = np.array(im).sum(2) <= 60
    pts = fi.cluster_points(mask, single_area=float(np.pi * 81), step=10, min_ratio=1.5)
    assert 15 <= len(pts) <= 40
    # the chain runs along y = x from (40, 40) to (274, 274): distance to that LINE, not to the 8.5 px-spaced circle centres
    assert all(abs(x - y) / np.sqrt(2) <= 3 and 35 <= x <= 279 for x, y in pts)
    along = np.linspace(40, 274, 48)
    assert max(np.min(np.hypot(pts[:, 0] - a, pts[:, 1] - a)) for a in along) <= 12     # no stretch of the chain is left uncovered


def test_isolated_blobs_are_not_treated_as_clusters():
    im = draw_circles(canvas(), [(60, 60), (200, 150)], r=9)
    mask = np.array(im).sum(2) <= 60
    assert len(fi.cluster_points(mask, single_area=float(np.pi * 81), step=10, min_ratio=1.5)) == 0


def test_dedupe_merges_points_closer_than_the_tolerance():
    pts = np.array([[10.0, 10.0], [12.0, 11.0], [100.0, 100.0], [40.0, 10.0]])
    out = fi.dedupe(pts, tol=5.0)
    assert len(out) == 3 and any(np.hypot(*(p - [100, 100])) < 1e-9 for p in out)


# ---------------------------------------------------------------- the real figure (guards fixed after inspecting the overlay)
@pytest.fixture(scope="module")
def real():
    img = fi.extract_figure()
    df, info = fi.digitise(img)
    return img, df, info


def test_real_graticule_fit_is_exact_and_the_map_is_plain_lonlat(real):
    img, df, info = real
    assert info["ax"][0] == pytest.approx(8412.0, rel=0.01) and -info["ay"][0] == pytest.approx(8405.0, rel=0.01)   # equal px per degree
    assert np.abs(np.polyval(info["ax"], fi.LON_TICKS) - info["ticks_x"]).max() < 0.5
    assert np.abs(np.polyval(info["ay"], fi.LAT_TICKS) - info["ticks_y"]).max() < 0.5


def test_real_registration_agrees_with_our_own_bmc_outline(real):
    img, df, info = real
    assert fi.outline_offset(img, info["ax"], info["ay"]) <= 3.0                      # about 1.4 px (18 m) observed


def test_real_counts_match_the_paper(real):
    """The abstract speaks of 130 flood event locations; the figure also shows non-flood control points."""
    img, df, info = real
    c = df.groupby("cls").size()
    assert c["non_flood"] == 25
    assert 38 <= c["flood_0.75m"] <= 46
    assert 120 <= c[["flood_0.75m", "flood_0.85m", "flood_1.2m"]].sum() <= 140


def test_real_points_lie_in_bhubaneswar_and_are_not_duplicates(real):
    import geopandas as gpd

    from src.config import CRS, INTERIM
    img, df, info = real
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326").to_crs(CRS)
    city = gpd.read_file(INTERIM / "wardmap.gpkg", layer="municipal_poly").to_crs(CRS).geometry.iloc[0]
    assert g.within(city.buffer(150)).all()
    for cls, sub in df.groupby("cls"):
        p = sub[["x_px", "y_px"]].to_numpy()
        d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1]) + np.eye(len(p)) * 1e9
        assert d.min() > 5.0, cls
