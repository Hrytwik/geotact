"""T6f: samples and input channels for the deep flood model trained on field-surveyed flood locations (protocol registered in
docs/build_plan.md). 30 m cells on the DEM grid. Absolute elevation is NOT a channel (it carried regional position, not
flooding, in T6d); local relief is. No torch and no LightGBM here: the CNN and the LightGBM baseline run in separate processes."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage as ndi

from src import flood_model as fm
from src.config import CRS, INTERIM, OUTPUTS

CHANNELS = ["slope_deg", "laplacian", "log10_acc_m2", "twi", "hand", "log_dist_drain", "log_dist_river", "log_dist_water",
            "local_relief", "share_built", "share_trees", "share_crop", "share_grass", "share_water", "share_bare", "ndvi"]
LANDUSE = {"share_water": 0, "share_trees": 1, "share_crop": 2, "share_grass": 3, "share_built": 4, "share_bare": 5}
PATCH = 33
N_BACKGROUND = 10000
BLOCK = fm.BLOCK


def local_relief(elevation, size=PATCH):
    """Elevation minus its mean over a size x size window: negative in dips, about zero on planes and far from them."""
    e = np.asarray(elevation, dtype="float64")
    return e - ndi.uniform_filter(e, size=size, mode="reflect")


class PatchSource:
    """size x size patches around cells, reflect-padded at the tile edge (NaNs are left for the caller to fill)."""

    def __init__(self, stack, size=PATCH):
        self.size, self.half = size, size // 2
        padded = np.pad(np.asarray(stack, dtype="float32"), ((0, 0), (self.half, self.half), (self.half, self.half)), mode="reflect")
        self.windows = sliding_window_view(padded, (size, size), axis=(1, 2))        # (C, H, W, size, size), a view

    def get(self, rows, cols):
        return np.ascontiguousarray(self.windows[:, np.asarray(rows), np.asarray(cols)].transpose(1, 0, 2, 3))


def sample_background(reference, positive_cells, n, seed=0, min_dist_cells=2):
    """n unique cells (row, col) drawn uniformly from `reference`, at least min_dist_cells (Euclidean, in cells) from every
    positive cell."""
    pos = np.zeros(reference.shape, dtype=bool)
    for r, c in positive_cells:
        pos[r, c] = True
    ok = np.asarray(reference, bool) & (ndi.distance_transform_edt(~pos) >= min_dist_cells)
    flat = np.flatnonzero(ok.ravel())
    if len(flat) < n:
        raise ValueError(f"only {len(flat)} eligible background cells for {n} requested")
    pick = np.random.default_rng(seed).choice(flat, size=n, replace=False)
    return np.column_stack(np.unravel_index(pick, reference.shape))


def assemble(cells, shape, block=BLOCK, k=5, seed=0):
    """Concatenate flood / non-flood / background cells with labels and their spatial-block fold."""
    rows, cols, kind, y = [], [], [], []
    for name, label in (("flood", 1), ("nonflood", 0), ("background", 0)):
        a = np.asarray(cells[name]).reshape(-1, 2)
        rows += a[:, 0].tolist(); cols += a[:, 1].tolist(); kind += [name] * len(a); y += [label] * len(a)
    rows, cols = np.array(rows), np.array(cols)
    folds = fm.block_folds(shape, block, k, seed)
    return dict(rows=rows, cols=cols, kind=np.array(kind), y=np.array(y), fold=folds[rows, cols])


# ---------------------------------------------------------------- real data
def mean_to_grid(arr, src_tf, dst_tf, dst_shape, crs, min_valid=0.2):
    """Area mean of the finite values only; NaN where fewer than min_valid of the cell's source pixels are finite."""
    fin = np.isfinite(arr)
    num = fm.to_grid(np.where(fin, arr, 0.0), src_tf, dst_tf, dst_shape, crs)
    den = fm.to_grid(fin.astype("float32"), src_tf, dst_tf, dst_shape, crs)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den >= min_valid, num / den, np.nan).astype("float32")


def load_channels():
    import rasterio

    from src import grid
    g10 = grid.load_grid()
    with rasterio.open(OUTPUTS / "terrain" / "terrain_features.tif") as r:
        terr = {n: r.read(i + 1) for i, n in enumerate(r.descriptions)}
        tf, crs, shape = r.transform, r.crs, (r.height, r.width)
    ch = {"slope_deg": terr["slope_deg"], "laplacian": terr["laplacian"], "log10_acc_m2": terr["log10_acc_m2"], "twi": terr["twi"],
          "hand": terr["hand"], "log_dist_drain": np.log1p(terr["dist_drain"]), "log_dist_river": np.log1p(terr["dist_river"]),
          "log_dist_water": np.log1p(terr["dist_water"]), "local_relief": local_relief(terr["elevation"])}
    lu = rasterio.open(OUTPUTS / "landuse" / "landuse_2025_26_seed0.tif").read(1)
    for name, k in LANDUSE.items():
        ch[name] = fm.to_grid((lu == k).astype("float32"), g10.transform, tf, shape, crs)
    ndvi = rasterio.open(OUTPUTS / "vegetation" / "ndvi_norm_stack.tif").read(8)       # season 2025-26
    ch["ndvi"] = mean_to_grid(ndvi, g10.transform, tf, shape, crs)
    stack = np.stack([np.asarray(ch[c], dtype="float32") for c in CHANNELS])
    return stack, tf, crs, shape


def digitised_cells(tf):
    import geopandas as gpd
    from rasterio.transform import rowcol
    g = gpd.read_file(INTERIM / "panigrahi2026_fig3_points.gpkg")
    out = {}
    for name, sel in (("flood", g.cls.str.startswith("flood")), ("nonflood", g.cls == "non_flood")):
        sub = g[sel]
        r, c = rowcol(tf, sub.geometry.x.values, sub.geometry.y.values)
        out[name] = np.unique(np.column_stack([r, c]), axis=0)                      # unique 30 m cells
    return out


def build(seed=0):
    from src import terrain as tr
    stack, tf, crs, shape = load_channels()
    d = tr.real_inputs()
    bmc, water = d["bmc"], d["water"]
    cells = digitised_cells(tf)
    cells["background"] = sample_background(bmc & ~water, cells["flood"], N_BACKGROUND, seed)
    return dict(stack=stack, samples=assemble(cells, shape), shape=shape, bmc=bmc, water=water, transform=tf, crs=crs)
