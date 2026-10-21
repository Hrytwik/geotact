"""T0: shared grid, study mask, ward raster, aligned reads."""
import glob

import numpy as np
import pytest
import rasterio

from src import grid
from src.config import RAW


def test_grid_matches_sentinel2_composite():
    g = grid.load_grid()
    with rasterio.open(RAW / "s2_2023.tif") as r:
        assert g.shape == (r.height, r.width) == (2895, 2424)
        assert g.transform == r.transform
        assert str(g.crs) == "EPSG:32645"
    assert g.res == 10


def test_all_ten_metre_rasters_share_the_grid():
    g = grid.load_grid()
    files = (glob.glob(str(RAW / "s2_*.tif")) + glob.glob(str(RAW / "worldcover_*.tif"))
             + glob.glob(str(RAW / "io_lulc_*.tif")) + glob.glob(str(RAW / "s1" / "*.tif")))
    assert len(files) > 40
    for f in files:
        with rasterio.open(f) as r:
            assert r.transform == g.transform, f
            assert (r.height, r.width) == g.shape, f


def test_study_mask_area_is_the_bmc_boundary():
    m = grid.study_mask()
    assert m.dtype == bool and m.shape == grid.load_grid().shape
    area_km2 = m.sum() * 100 / 1e6
    assert area_km2 == pytest.approx(166.5, rel=0.01)


def test_ward_raster_has_all_67_wards_inside_the_mask():
    w, m = grid.ward_raster(), grid.study_mask()
    ids = set(np.unique(w)) - {0}
    assert ids == set(range(1, 68))
    assert (w[~m] > 0).sum() / (w > 0).sum() < 0.005   # ward polygons overshoot the boundary by < 0.5 percent
    assert (w[m] == 0).sum() / m.sum() < 0.005         # almost every mask pixel belongs to a ward


def test_read_aligned_dem_has_grid_shape_and_valid_heights():
    dem = grid.read_aligned(RAW / "copdem30.tif", resampling="bilinear")
    assert dem.shape == grid.load_grid().shape
    assert np.isfinite(dem).all()
    assert dem.min() > 3 and dem.max() < 180           # Copernicus GLO-30 over Bhubaneswar: about 4.6 to 177 m


def test_read_aligned_categorical_uses_nearest_and_keeps_codes():
    wc = grid.read_aligned(RAW / "worldcover_2021.tif", resampling="nearest")
    assert set(np.unique(wc)) <= {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}


def test_read_aligned_on_a_grid_file_is_identity():
    a = grid.read_aligned(RAW / "s2_2023.tif", band=1, resampling="nearest")
    with rasterio.open(RAW / "s2_2023.tif") as r:
        b = r.read(1)
    assert np.array_equal(a, b)
