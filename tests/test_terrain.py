"""T6b: terrain and hydrology features at the DEM's native 30 m."""
import json
import math

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, Polygon

from src import terrain as tr

RES = 30.0
TF = Affine(RES, 0, 500000.0, 0, -RES, 2250000.0)
CRS = "EPSG:32645"


def plane(n=20, rise_per_cell=0.3):
    c = np.arange(n)[None, :].repeat(n, 0).astype("float64")
    return rise_per_cell * c                                           # z grows eastward


# ---------------------------------------------------------------- slope and curvature
def test_slope_of_a_tilted_plane_is_the_analytic_value():
    f = tr.terrain_features(plane(), TF, CRS)
    expected = math.degrees(math.atan(0.3 / RES))
    assert f["slope_deg"][2:-2, 2:-2] == pytest.approx(expected, abs=1e-6)


def test_laplacian_of_a_bowl_is_four_times_the_quadratic_coefficient():
    n = 21
    y, x = np.mgrid[0:n, 0:n] * RES
    bowl = 0.001 * ((x - 300) ** 2 + (y - 300) ** 2)
    f = tr.terrain_features(bowl, TF, CRS)
    assert f["laplacian"][3:-3, 3:-3] == pytest.approx(4 * 0.001, rel=1e-6)


# ---------------------------------------------------------------- flow accumulation
def test_flow_accumulation_on_a_plane_that_drains_east():
    n = 20
    dem = 0.3 * (n - np.arange(n))[None, :].repeat(n, 0).astype("float64")      # high in the west, low in the east
    f = tr.terrain_features(dem, TF, CRS)
    assert (f["acc_cells"][:, -1] == n).all()                           # every row drains along itself to the east edge
    assert f["acc_cells"][5, 0] == 1


# ---------------------------------------------------------------- HAND and TWI in a V valley
def v_valley(n=21, r0=10):
    r = np.arange(n)[:, None].repeat(n, 1).astype("float64")
    c = np.arange(n)[None, :].repeat(n, 0).astype("float64")
    return 0.5 * np.abs(r - r0) + 0.01 * (n - c), r0


def test_hand_equals_height_above_the_channel_it_drains_to():
    dem, r0 = v_valley()
    channel = np.zeros(dem.shape, bool)
    channel[r0, :] = True
    f = tr.terrain_features(dem, TF, CRS, channels=channel)
    rows = np.arange(dem.shape[0])
    expected = (0.5 * np.abs(rows - r0))[:, None].repeat(dem.shape[1], 1)
    inner = (slice(1, -1), slice(1, -1))
    assert f["hand"][inner] == pytest.approx(expected[inner], abs=1e-6)       # analytic: 0.5 m per row away from the channel
    assert (f["hand"][r0, 1:-1] == 0).all()


def test_hand_is_undefined_on_the_outer_border_ring():
    """pysheds leaves the 1-cell border of the grid undefined (no complete drainage path on the grid). The study area
    must therefore sit inside the tile; test_real_* asserts the BMC has no NaN."""
    dem, r0 = v_valley()
    channel = np.zeros(dem.shape, bool)
    channel[r0, :] = True
    h = tr.terrain_features(dem, TF, CRS, channels=channel)["hand"]
    border = np.ones(h.shape, bool); border[1:-1, 1:-1] = False
    assert np.isnan(h[border]).all() and np.isfinite(h[~border]).all()


def test_twi_is_higher_on_the_valley_floor_than_on_the_ridge():
    dem, r0 = v_valley()
    f = tr.terrain_features(dem, TF, CRS, channels=np.zeros(dem.shape, bool))
    assert f["twi"][r0 + 1, 10] > f["twi"][0, 10] + 1.0                 # near the axis vs the edge (ridge)
    assert np.isfinite(f["twi"]).all()


def test_flat_ground_does_not_produce_infinite_twi():
    f = tr.terrain_features(np.full((15, 15), 10.0), TF, CRS)
    assert np.isfinite(f["twi"]).all() and np.isfinite(f["slope_deg"]).all()


# ---------------------------------------------------------------- distances and rasterising
def test_distance_to_a_mask_is_in_metres():
    m = np.zeros((9, 9), bool)
    m[:, 4] = True
    d = tr.distance_to(m, RES)
    assert d[3, 0] == pytest.approx(4 * RES) and d[3, 4] == 0 and d[0, 8] == pytest.approx(4 * RES)


def test_rasterize_line_and_polygon_onto_the_grid():
    line = LineString([(500000 + 15, 2250000 - 15), (500000 + 15 + 5 * RES, 2250000 - 15)])        # along row 0
    poly = Polygon([(500000 + 2 * RES, 2250000 - 4 * RES), (500000 + 5 * RES, 2250000 - 4 * RES),
                    (500000 + 5 * RES, 2250000 - 7 * RES), (500000 + 2 * RES, 2250000 - 7 * RES)])
    m = tr.rasterize([line, poly], TF, (10, 10))
    assert m[0, :6].all() and not m[0, 6:].any()
    assert m[4:7, 2:5].all() and not m[8, :].any()
    assert tr.rasterize([], TF, (10, 10)).sum() == 0


def test_osm_parser_returns_lines_and_polygons_by_kind(tmp_path):
    js = {"elements": [
        {"type": "way", "tags": {"waterway": "drain"}, "geometry": [{"lat": 20.2, "lon": 85.8}, {"lat": 20.21, "lon": 85.81}]},
        {"type": "way", "tags": {"natural": "water"}, "geometry": [{"lat": 20.2, "lon": 85.8}, {"lat": 20.2, "lon": 85.81},
                                                                      {"lat": 20.21, "lon": 85.81}, {"lat": 20.2, "lon": 85.8}]},
        {"type": "way", "tags": {"highway": "residential"}, "geometry": [{"lat": 20.2, "lon": 85.8}, {"lat": 20.21, "lon": 85.8}]}]}
    p = tmp_path / "osm.json"; p.write_text(json.dumps(js))
    out = tr.osm_water(p)
    assert len(out["lines"]) == 1 and len(out["polygons"]) == 1         # roads ignored
    assert out["lines"][0].geom_type == "LineString" and out["polygons"][0].geom_type == "Polygon"


def test_numpy2_compat_shim_matches_isin():
    a = np.array([[1, 5, 9], [2, 5, 7]])
    b = np.array([5, 7])
    assert np.array_equal(np.in1d(a.ravel(), b), np.isin(a, b).ravel())
    assert np.array_equal(np.in1d(a.ravel(), b, invert=True), ~np.isin(a, b).ravel())


# ---------------------------------------------------------------- real study area (thresholds fixed before reading the numbers)
@pytest.fixture(scope="module")
def real():
    return tr.compute_real()


def test_real_bmc_sits_well_inside_the_tile(real):
    out, d, derived = real
    rows, cols = np.where(d["bmc"])
    margin = min(rows.min(), d["bmc"].shape[0] - 1 - rows.max(), cols.min(), d["bmc"].shape[1] - 1 - cols.max())
    assert margin >= 50                                                   # the undefined border ring never touches the city


def test_real_features_are_defined_everywhere_in_the_city(real):
    out, d, _ = real
    b = d["bmc"]
    for k, v in out.items():
        frac = np.isfinite(v[b]).mean()
        assert frac >= (0.995 if k == "hand" else 1.0), (k, frac)        # HAND may be NaN where no drainage path is found


def test_real_value_ranges_are_physical(real):
    out, d, _ = real
    b = d["bmc"]
    assert 0 <= np.nanmin(out["slope_deg"][b]) and np.nanmax(out["slope_deg"][b]) < 60
    assert np.nanmedian(out["slope_deg"][b]) < 3                          # flat coastal-plain city
    h = out["hand"][b]
    assert np.nanmin(h) >= 0 and np.nanmax(h) < 150                        # float32 round-off (-9e-5 m) is clipped in terrain_features
    assert 5 < np.nanmin(out["elevation"][b]) and np.nanmax(out["elevation"][b]) < 120


def test_real_distances_are_zero_on_the_mapped_layers(real):
    out, d, derived = real
    assert (out["dist_drain"][d["drains"]] == 0).all()
    assert (out["dist_river"][d["river"]] == 0).all()
    assert (out["dist_water"][d["water"]] == 0).all()
    assert (out["dist_drain"][derived] == 0).all()


def test_real_derived_streams_follow_the_mapped_channels(real):
    """A surface-model DEM cannot see underground drains, so only partial agreement is expected (38 percent at the
    chosen threshold); a guard against a flipped or misaligned grid."""
    from scipy.ndimage import binary_dilation
    out, d, derived = real
    known = (d["drains"] | d["river"]) & d["bmc"]
    near = binary_dilation(derived, iterations=1)
    assert (near & known).sum() / known.sum() >= 0.25
