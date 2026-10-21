"""T6a: Sentinel-1 flood evidence (dB, speckle filter, threshold calibration, change detection, frequency)."""
import numpy as np
import pytest

from src import flood_sar as fs


# ---------------------------------------------------------------- dB
def test_to_db_values_and_invalid_inputs():
    x = np.array([1.0, 0.1, 0.01, 0.0, -0.5, np.nan], dtype="float32")
    out = fs.to_db(x)
    assert out[:3] == pytest.approx([0.0, -10.0, -20.0], abs=1e-5)
    assert np.isnan(out[3:]).all()                         # zero, negative and NaN backscatter are invalid
    assert out.dtype == np.float32


# ---------------------------------------------------------------- speckle filter
def test_speckle_filter_removes_isolated_pixels_but_keeps_real_patches():
    x = np.full((20, 20), -8.0, dtype="float32")
    x[5, 5] = -25.0                                        # single dark speckle pixel
    x[10:15, 10:15] = -20.0                                # a real 5x5 dark patch (pond)
    y = fs.speckle_filter(x, size=3)
    assert y[5, 5] == pytest.approx(-8.0)
    assert (y[11:14, 11:14] == -20.0).all()
    assert y.shape == x.shape and y.dtype == np.float32


def test_speckle_filter_keeps_nan_where_there_is_no_data():
    x = np.full((10, 10), -8.0, dtype="float32")
    x[0:2, 0:2] = np.nan
    y = fs.speckle_filter(x, size=3)
    assert np.isnan(y[0:2, 0:2]).all() and np.isfinite(y[3:, 3:]).all()
    assert y[5, 5] == pytest.approx(-8.0)                  # NaN neighbours do not contaminate the rest


# ---------------------------------------------------------------- threshold calibration
def test_calibrated_threshold_separates_two_known_classes():
    rng = np.random.default_rng(0)
    water = rng.normal(-18, 2, 20000)
    land = rng.normal(-8, 2, 20000)
    db = np.concatenate([water, land]).astype("float32")
    wmask = np.r_[np.ones(20000, bool), np.zeros(20000, bool)]
    t, bacc = fs.calibrate_threshold(db, wmask, ~wmask)
    assert t == pytest.approx(-13.0, abs=0.75)
    assert bacc > 0.97


def test_calibration_with_overlapping_classes_reports_lower_accuracy():
    rng = np.random.default_rng(1)
    db = np.concatenate([rng.normal(-12, 3, 5000), rng.normal(-9, 3, 5000)]).astype("float32")
    w = np.r_[np.ones(5000, bool), np.zeros(5000, bool)]
    _, bacc = fs.calibrate_threshold(db, w, ~w)
    assert 0.55 < bacc < 0.8


def test_calibration_needs_both_classes_and_ignores_nan():
    db = np.array([-20, -19, -5, np.nan], dtype="float32")
    with pytest.raises(ValueError):
        fs.calibrate_threshold(db, np.array([1, 1, 0, 0], bool), np.zeros(4, bool))
    t, b = fs.calibrate_threshold(db, np.array([1, 1, 0, 0], bool), np.array([0, 0, 1, 1], bool))
    assert -19 < t < -5 and b == 1.0                       # the NaN land pixel is ignored


# ---------------------------------------------------------------- dry reference
def test_dry_reference_and_dark_frequency():
    scenes = np.array([[[-8.0, -18.0]], [[-9.0, -17.0]], [[-7.0, -8.0]], [[-8.5, np.nan]]], dtype="float32")   # (4, 1, 2)
    ref = fs.dry_reference(scenes)
    assert ref[0, 0] == pytest.approx(-8.25) and ref[0, 1] == pytest.approx(-17.0)
    thresholds = [-12.0, -12.0, -12.0, -12.0]
    freq = fs.dark_frequency(scenes, thresholds)
    assert freq[0, 0] == pytest.approx(0.0)
    assert freq[0, 1] == pytest.approx(2 / 3)              # dark in 2 of the 3 scenes that observed the pixel


# ---------------------------------------------------------------- flood mask
def _case(**over):
    base = dict(db=-16.0, ref=-8.0, dark=0.0, slope=1.0, hand=2.0)
    base.update(over)
    return base


def _flood(case, t=-12.0):
    a = lambda k: np.array([[case[k]]], dtype="float32")
    return bool(fs.flood_mask(a("db"), t, a("ref"), a("dark"), slope=a("slope"), hand=a("hand"))[0, 0])


def test_flood_mask_needs_every_criterion():
    assert _flood(_case()) is True
    assert _flood(_case(db=-10.0)) is False                # not dark enough in absolute terms
    assert _flood(_case(db=-13.0, ref=-14.0)) is False     # dark, but no drop versus the dry reference
    assert _flood(_case(dark=0.6)) is False                # persistently dark (runway, asphalt, permanent water)
    assert _flood(_case(slope=8.0)) is False               # steep: radar shadow, not flooding
    assert _flood(_case(hand=30.0)) is False               # high above the nearest drainage
    assert _flood(_case(db=np.nan)) is False               # no observation


def test_flood_mask_change_threshold_boundary_and_optional_terrain():
    a = lambda v: np.array([[v]], dtype="float32")
    assert fs.flood_mask(a(-11.5), -10.0, a(-8.5), a(0.0))[0, 0]               # drop exactly -3 dB counts
    assert not fs.flood_mask(a(-11.4), -10.0, a(-8.5), a(0.0))[0, 0]           # drop of -2.9 dB does not
    assert fs.flood_mask(a(-16), -12.0, a(-8), a(0.0), slope=None, hand=None)[0, 0]   # terrain masks are optional


# ---------------------------------------------------------------- flood frequency
def test_flood_frequency_counts_only_valid_scenes():
    flood = np.array([[[1, 0, 1]], [[1, 0, 0]], [[0, 0, 1]], [[1, 0, 1]]], dtype=bool)        # (4 scenes, 1, 3)
    valid = np.array([[[1, 1, 1]], [[1, 1, 1]], [[1, 1, 1]], [[1, 0, 0]]], dtype=bool)
    f, n = fs.flood_frequency(flood, valid, min_valid=2)
    assert f[0, 0] == pytest.approx(3 / 4) and f[0, 1] == pytest.approx(0.0) and f[0, 2] == pytest.approx(2 / 3)
    assert list(n[0]) == [4, 3, 3]
    f2, _ = fs.flood_frequency(flood, valid, min_valid=4)
    assert np.isfinite(f2[0, 0]) and np.isnan(f2[0, 1]) and np.isnan(f2[0, 2])               # too few valid scenes -> NaN


# ---------------------------------------------------------------- real scene
def test_a_real_monsoon_scene_passes_the_protocol_quality_check():
    from src import grid
    from src.config import RAW
    wc = grid.read_aligned(RAW / "worldcover_2021.tif", resampling="nearest", dtype="uint8")
    db = fs.speckle_filter(fs.to_db(grid.read_aligned(RAW / "s1" / "s1_monsoon_2025-07-12.tif", band=1)), 3)
    t, bacc = fs.calibrate_threshold(db, wc == 80, np.isin(wc, [10, 20, 30, 40]))
    assert -15 < t < -9 and bacc >= 0.80                    # protocol rule: scenes below 0.80 are excluded
