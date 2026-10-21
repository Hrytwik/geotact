"""Shared analysis grid, study mask and ward raster (task T0).

Every raster step reads and writes on the Sentinel-2 composite grid (EPSG:32645, 10 m), so layers
from different sources can be stacked pixel for pixel.
"""
from functools import lru_cache
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import reproject

from src.config import INTERIM, RAW

REFERENCE = RAW / "s2_2023.tif"
WARDMAP = INTERIM / "wardmap.gpkg"


class Grid(NamedTuple):
    transform: Affine
    shape: tuple
    crs: object
    res: float


@lru_cache(maxsize=1)
def load_grid() -> Grid:
    with rasterio.open(REFERENCE) as r:
        return Grid(r.transform, (r.height, r.width), r.crs, abs(r.transform.a))


@lru_cache(maxsize=1)
def study_mask() -> np.ndarray:
    """True inside the Bhubaneswar Municipal Corporation boundary."""
    g = load_grid()
    city = gpd.read_file(WARDMAP, layer="municipal_poly").to_crs(g.crs)
    return rasterize(city.geometry, out_shape=g.shape, transform=g.transform, fill=0, default_value=1,
                     dtype="uint8").astype(bool)


@lru_cache(maxsize=1)
def ward_raster() -> np.ndarray:
    """int16 raster of ward numbers 1-67 (0 outside every ward)."""
    g = load_grid()
    wards = gpd.read_file(WARDMAP, layer="wards").to_crs(g.crs)
    shapes = list(zip(wards.geometry, wards.ward_no.astype(int)))
    return rasterize(shapes, out_shape=g.shape, transform=g.transform, fill=0, dtype="int16")


def read_aligned(path, band=None, resampling="bilinear", dtype="float32") -> np.ndarray:
    """Read a raster onto the analysis grid.

    band=None returns (count, rows, cols) for multi-band files and (rows, cols) for single-band ones;
    band=k returns that band. Nodata becomes NaN for float output. Use resampling="nearest" for
    categorical layers.
    """
    g = load_grid()
    method = Resampling[resampling]
    with rasterio.open(path) as src:
        bands = [band] if band else list(range(1, src.count + 1))
        out = np.full((len(bands),) + g.shape, np.nan if np.dtype(dtype).kind == "f" else 0, dtype=dtype)
        for k, b in enumerate(bands):
            reproject(source=rasterio.band(src, b), destination=out[k], src_transform=src.transform,
                      src_crs=src.crs, src_nodata=src.nodata, dst_transform=g.transform, dst_crs=g.crs,
                      dst_nodata=np.nan if np.dtype(dtype).kind == "f" else None, resampling=method)
    return out[0] if len(bands) == 1 else out
