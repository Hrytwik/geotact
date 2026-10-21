"""T5: vegetation health over time (index-based NDVI trend, year-normalised)."""
import numpy as np
import pytest
from scipy import stats

from src import vegetation as veg

YEARS = 8
T = np.arange(YEARS, dtype=float)


# ---------------------------------------------------------------- unit: NDVI
def test_ndvi_values_and_invalid_pixels():
    red = np.array([1000, 0, 500, 2000], dtype="float32")
    nir = np.array([3000, 0, 500, 0], dtype="float32")
    out = veg.ndvi(red, nir)
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(out[1])             # nodata (both bands 0)
    assert out[2] == pytest.approx(0.0)
    assert np.isnan(out[3])             # a zero band is invalid in a composite (0 = no clear day)
    assert np.nanmin(out) >= -1 and np.nanmax(out) <= 1


# ---------------------------------------------------------------- unit: exact Mann-Kendall
def test_mk_null_distribution_is_exact():
    for n in (6, 7, 8):
        s, p = veg.mk_null(n)
        assert p.sum() == pytest.approx(1.0)
        assert (s * p).sum() == pytest.approx(0.0, abs=1e-12)          # symmetric about 0
    s, p = veg.mk_null(8)
    assert p[s == 28][0] == pytest.approx(1 / 40320)                    # one monotone ordering out of 8!


def test_sen_mk_on_known_series():
    line = 0.3 - 0.02 * T                                               # exact decline 0.02 per year
    stack = np.stack([line, 0.4 + 0 * T, 0.1 + 0.02 * T], axis=1)[:, :, None]   # (8, 3, 1)
    slope, S, p, n = veg.sen_mk(stack)
    assert slope[:, 0] == pytest.approx([-0.02, 0.0, 0.02], abs=1e-9)
    assert list(S[:, 0]) == [-28, 0, 28]
    assert p[0, 0] == pytest.approx(2 / 40320) and p[2, 0] == pytest.approx(2 / 40320)
    assert p[1, 0] == pytest.approx(1.0)
    assert (n == 8).all()


def test_sen_mk_with_missing_years_and_minimum_valid():
    x = (0.3 - 0.02 * T).astype("float32")
    two_missing = x.copy(); two_missing[[1, 5]] = np.nan
    three_missing = x.copy(); three_missing[[0, 3, 6]] = np.nan         # only 5 valid
    stack = np.stack([two_missing, three_missing], axis=1)[:, :, None]
    slope, S, p, n = veg.sen_mk(stack, min_valid=6)
    assert n[0, 0] == 6 and slope[0, 0] == pytest.approx(-0.02, abs=1e-6) and S[0, 0] == -15
    assert p[0, 0] == pytest.approx(2 / 720)                            # n=6: 2 of 6! orderings are extreme
    assert np.isnan(slope[1, 0]) and np.isnan(p[1, 0]) and n[1, 0] == 5


def test_sen_mk_matches_scipy_on_random_series():
    rng = np.random.default_rng(0)
    series = rng.normal(size=(60, YEARS))
    stack = series.T.reshape(YEARS, 60, 1).astype("float64")
    slope, S, p, n = veg.sen_mk(stack)
    for i in range(60):
        ts = stats.theilslopes(series[i], T)[0]
        kt = stats.kendalltau(T, series[i])
        assert slope[i, 0] == pytest.approx(ts, abs=1e-9)
        assert p[i, 0] == pytest.approx(kt.pvalue, abs=1e-9)           # exact two-sided p, no ties
        assert S[i, 0] == round(kt.statistic * 28)


def test_sen_mk_blocks_do_not_change_the_answer():
    rng = np.random.default_rng(1)
    stack = rng.normal(size=(YEARS, 37, 11)).astype("float32")
    stack[rng.random(stack.shape) < 0.08] = np.nan
    a = veg.sen_mk(stack, block_rows=5)
    b = veg.sen_mk(stack, block_rows=1000)
    for x, y in zip(a, b):
        assert np.array_equal(x, y, equal_nan=True)


# ---------------------------------------------------------------- unit: year normalisation
def test_year_offsets_are_recovered_and_removed():
    rng = np.random.default_rng(2)
    base = rng.uniform(0.1, 0.7, size=(40, 40))
    true = np.array([0.0, 0.05, -0.04, 0.03, -0.06, 0.02, 0.07, -0.07])
    trend = np.zeros((40, 40)); trend[:, 20:] = -0.02                    # right half: real decline
    stack = base[None] + true[:, None, None] + trend[None] * T[:, None, None]
    stable = np.zeros((40, 40), bool); stable[:, :20] = True             # only the unchanged half is used
    off = veg.estimate_year_offsets(stack.astype("float32"), stable)
    assert off == pytest.approx(true - true.mean(), abs=1e-5)           # offsets are centred (overall level kept)
    norm = veg.normalise(stack.astype("float32"), off)
    slope = veg.sen_mk(norm)[0]
    assert np.abs(slope[:, :20]).max() < 1e-5                            # stable half: no trend left
    assert slope[:, 20:] == pytest.approx(-0.02, abs=1e-5)               # changed half: real trend preserved


# ---------------------------------------------------------------- unit: anomaly
def test_anomaly_uses_previous_years_and_a_floor():
    prior = np.full(7, 0.5); prior_noisy = 0.5 + 0.1 * np.array([1, -1, 1, -1, 1, -1, 0.0])
    last = 0.3
    stack = np.stack([np.append(prior, last), np.append(prior_noisy, last)], axis=1)[:, :, None].astype("float32")
    z = veg.anomaly(stack, floor=0.02, min_prior=5)
    assert z[0, 0] == pytest.approx((0.3 - 0.5) / 0.02, abs=1e-4)       # sd 0 -> floor, no division blow-up
    sd = np.std(prior_noisy, ddof=1)
    assert z[1, 0] == pytest.approx((0.3 - prior_noisy.mean()) / sd, abs=1e-4)
    sparse = np.append(np.full(7, np.nan), 0.3).astype("float32").reshape(8, 1, 1)
    assert np.isnan(veg.anomaly(sparse)[0, 0])                           # too few prior years -> undefined


# ---------------------------------------------------------------- real data
@pytest.fixture(scope="module")
def real():
    return veg.load_real_sample(step=4)                                  # every 4th pixel inside the study area


def test_composites_are_on_the_grid_with_plausible_red(real):
    assert real["raw"].shape[0] == YEARS
    red_med = real["red_median"]
    assert len(red_med) == YEARS and min(red_med) > 700 and max(red_med) < 1100   # a missed +1000 offset would fail this


def test_normalisation_removes_the_common_year_effect(real):
    before = np.nanmedian(real["raw"].reshape(YEARS, -1), axis=1)
    after = np.nanmedian(real["norm"].reshape(YEARS, -1), axis=1)
    assert before.std() > 0.03                                           # the raw year effect is real (0.30-0.42)
    assert after.std() < 0.012                                           # and is gone afterwards
    off2 = veg.estimate_year_offsets(real["norm"], real["stable"])
    assert np.abs(off2).max() < 1e-3                                     # idempotent: nothing left to remove


def test_trend_detects_tree_canopy_loss_but_is_not_a_per_pixel_detector(real):
    """Tree pixels converted to built-up (Esri 2017 trees, built every year 2020-23) must trend down more than
    pixels that were built-up throughout. NOTE: the original test used ALL vegetation -> built and FAILED:
    dry-season (Nov-Mar) cropland and scrub are already low-NDVI, so their conversion leaves almost no signal
    (crops 0.388 -> 0.372). The trend is only claimed for tree canopy loss, and only as a population-level shift."""
    slope, _, p, _ = veg.sen_mk(real["norm"])
    conv, sb = real["converted_trees"], real["stable_built"]
    assert conv.sum() > 300 and sb.sum() > 5000
    a = slope[conv][np.isfinite(slope[conv])]
    b = slope[sb][np.isfinite(slope[sb])]
    assert np.median(a) < np.median(b) - 0.003                           # about -0.0057 vs -0.0006 per year
    u = stats.mannwhitneyu(a, b, alternative="less")
    assert u.pvalue < 1e-6
    assert u.statistic / (a.size * b.size) < 0.45                        # P(converted slope > stable slope) well below 0.5


def test_unchanged_land_has_few_spurious_significant_declines(real):
    slope, _, p, _ = veg.sen_mk(real["norm"])
    sig_down = (p < 0.05) & (slope < 0)
    for key, limit in [("stable_built", 0.12), ("stable_trees", 0.06)]:
        sel = real[key] & np.isfinite(slope)
        assert sig_down[sel].mean() < limit, key                         # observed about 6.7 percent and 1.8 percent
