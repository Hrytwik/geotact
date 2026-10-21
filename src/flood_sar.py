"""T6a: Sentinel-1 flood evidence (see the T6 protocol in docs/build_plan.md).

Per scene: backscatter in dB, 3x3 median speckle filter, a scene-specific water threshold calibrated on
permanent-water versus vegetated-land reference pixels, and change detection against the dry-season median of the
SAME orbit. Pure numpy/scipy: nothing here imports torch.
"""
import numpy as np
from scipy.ndimage import median_filter

CHANGE_DB = -3.0        # conventional change threshold (literature support to be verified before citing)
DARK_MAX = 0.5          # water-like in >= 50 percent of the orbit's dry scenes -> persistent, not flooding
SLOPE_MAX = 5.0         # degrees
HAND_MAX = 20.0         # metres above the nearest drainage


def to_db(x):
    """Linear backscatter -> dB (float32). Zero, negative and NaN inputs become NaN."""
    x = np.asarray(x, dtype="float32")
    out = np.full(x.shape, np.nan, dtype="float32")
    ok = x > 0
    out[ok] = 10.0 * np.log10(x[ok])
    return out


def speckle_filter(db, size=3):
    """Median filter on dB; no-data pixels stay NaN and do not contaminate their neighbours."""
    db = np.asarray(db, dtype="float32")
    nan = np.isnan(db)
    fill = np.nanmedian(db) if (~nan).any() else 0.0
    out = median_filter(np.where(nan, fill, db).astype("float32"), size=size, mode="reflect")
    out[nan] = np.nan
    return out


def calibrate_threshold(db, water, land, grid=None):
    """Water threshold (dB) maximising Youden's J between reference water and reference land pixels.

    Returns (threshold, balanced accuracy). When several thresholds tie, the middle of the plateau is used."""
    grid = np.arange(-25.0, 0.0, 0.25) if grid is None else np.asarray(grid)
    w = np.sort(db[np.asarray(water, bool) & np.isfinite(db)])
    l = np.sort(db[np.asarray(land, bool) & np.isfinite(db)])
    if w.size == 0 or l.size == 0:
        raise ValueError("calibration needs observed pixels in both the water and the land reference sets")
    tpr = np.searchsorted(w, grid, side="left") / w.size            # share of water pixels below t
    tnr = 1.0 - np.searchsorted(l, grid, side="left") / l.size      # share of land pixels at or above t
    j = tpr + tnr
    best = np.flatnonzero(np.isclose(j, j.max()))
    k = best[len(best) // 2]
    return float(grid[k]), float(j[k] / 2.0)


def dry_reference(scenes):
    """Per-pixel median dB over the dry-season scenes of one orbit; scenes is (n, H, W) with NaN = no data."""
    return np.nanmedian(np.asarray(scenes, dtype="float32"), axis=0).astype("float32")


def dark_frequency(scenes, thresholds):
    """Share of observing dry scenes in which a pixel is darker than that scene's own water threshold."""
    scenes = np.asarray(scenes, dtype="float32")
    t = np.asarray(thresholds, dtype="float32")[:, None, None]
    seen = np.isfinite(scenes).sum(0)
    dark = (scenes < t).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(seen > 0, dark / seen, np.nan).astype("float32")


def flood_mask(db, threshold, reference, dark_freq, slope=None, hand=None, change_db=CHANGE_DB,
               dark_max=DARK_MAX, slope_max=SLOPE_MAX, hand_max=HAND_MAX):
    """True where open-area inundation is indicated. Every criterion must hold:
    dark enough, darker than the same orbit's dry reference by change_db, not persistently dark,
    and (when given) gentle slope and low height above drainage."""
    with np.errstate(invalid="ignore"):
        m = np.isfinite(db) & np.isfinite(reference)
        m &= db < threshold
        m &= (db - reference) <= change_db
        m &= dark_freq < dark_max
        if slope is not None:
            m &= slope < slope_max
        if hand is not None:
            m &= hand < hand_max
    return m


def flood_frequency(flood, valid, min_valid=10):
    """(frequency, valid-scene count) per pixel over (n, H, W) stacks; NaN where fewer than min_valid scenes saw it."""
    n = np.asarray(valid, bool).sum(0)
    k = (np.asarray(flood, bool) & np.asarray(valid, bool)).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(n >= min_valid, k / np.maximum(n, 1), np.nan)
    return f.astype("float32"), n.astype("int16")
