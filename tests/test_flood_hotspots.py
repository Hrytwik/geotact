"""T6e: one-shot hotspot test machinery (percentile ranks, 150 m disk maximum, multi-point hotspots, permutation test)."""
import numpy as np
import pytest

from src import flood_hotspots as fh


def test_percentile_raster_ranks_reference_cells_only_and_averages_ties():
    score = np.array([[1.0, 2.0, 2.0], [4.0, 3.0, 9.0]])
    ref = np.array([[True, True, True], [True, True, False]])        # the 9.0 cell is outside the reference area
    p = fh.percentile_raster(score, ref)
    assert np.isnan(p[1, 2])
    assert p[0, 0] == pytest.approx(100 * 1 / 5) and p[1, 0] == pytest.approx(100 * 5 / 5)
    assert p[0, 1] == p[0, 2] == pytest.approx(100 * 2.5 / 5)        # tie at ranks 2 and 3


def test_disk_mean_matches_a_brute_force_average():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 100, (30, 30)); p[rng.random((30, 30)) < 0.2] = np.nan
    dm = fh.disk_mean(p, radius_cells=5)
    for (r, c) in [(15, 15), (3, 4), (28, 2), (0, 0)]:
        vals = [p[rr, cc] for rr in range(30) for cc in range(30)
                if (rr - r) ** 2 + (cc - c) ** 2 <= 25 and np.isfinite(p[rr, cc])]
        assert dm[r, c] == pytest.approx(np.mean(vals))


def test_disk_radius_is_inclusive_at_exactly_five_cells():
    p = np.zeros((21, 21)); p[10, 10] = 81.0
    dm = fh.disk_mean(p, 5)
    n = int(((np.mgrid[-5:6, -5:6] ** 2).sum(0) <= 25).sum())          # cells in the disk (81 for radius 5)
    assert dm[10, 15] == pytest.approx(81.0 / n) and dm[13, 14] == pytest.approx(81.0 / n)     # 5 and 3-4-5 away: inside
    assert dm[10, 16] == 0.0 and dm[15, 13] == 0.0                                           # 6 and sqrt(34) away: outside


def test_a_random_score_gives_a_mean_percentile_near_fifty_not_near_one_hundred():
    """The registered 'maximum within 150 m' statistic averaged 98.6 on noise, making the >= 70 bar vacuous.
    The disk MEAN must sit near 50 on noise so that 70 means something."""
    ref = np.ones((200, 200), bool)
    score = np.random.default_rng(3).normal(size=(200, 200))
    dm = fh.disk_mean(fh.percentile_raster(score, ref), 5)
    assert abs(np.nanmean(dm) - 50) < 1.0
    null = fh.null_means(dm, ref, [1] * 16, n_perm=2000, seed=0)
    assert abs(null.mean() - 50) < 1.0 and null.std() < 5


def test_hotspot_value_is_the_maximum_over_its_points():
    dm = np.zeros((10, 10)); dm[2, 2] = 40; dm[7, 7] = 85
    v = fh.hotspot_values(dm, [[(2, 2)], [(2, 2), (7, 7)], [(5, 5)]])
    assert v.tolist() == [40.0, 85.0, 0.0]


def test_permutation_test_is_calibrated_on_noise_and_detects_a_planted_signal():
    rng = np.random.default_rng(1)
    H = W = 60
    ref = np.ones((H, W), bool)
    pts = [[(int(rng.integers(8, 52)), int(rng.integers(8, 52)))] for _ in range(16)]
    p_noise = []
    for trial in range(120):                                           # pure noise: p-values must not pile up near 0
        score = np.random.default_rng(100 + trial).normal(size=(H, W))
        dm = fh.disk_mean(fh.percentile_raster(score, ref), 5)
        res = fh.permutation_test(fh.hotspot_values(dm, pts), dm, ref, [len(x) for x in pts], n_perm=300, seed=trial)
        p_noise.append(res["p"])
    assert 0.0 <= np.mean(np.array(p_noise) < 0.05) < 0.14             # about 5 percent expected
    score = np.random.default_rng(5).normal(size=(H, W))
    for hp in pts:
        r, c = hp[0]; score[max(r - 5, 0):r + 6, max(c - 5, 0):c + 6] += 3.0     # raise the whole neighbourhood of each hotspot
    dm = fh.disk_mean(fh.percentile_raster(score, ref), 5)
    res = fh.permutation_test(fh.hotspot_values(dm, pts), dm, ref, [1] * 16, n_perm=2000, seed=0)
    assert res["mean_percentile"] > 80 and res["p"] < 0.01


def test_null_uses_the_same_number_of_points_per_hotspot():
    """A hotspot with 3 points gets a max over 3 locations; the null must give it 3 random locations too."""
    ref = np.ones((60, 60), bool)
    score = np.random.default_rng(0).normal(size=(60, 60))
    dm = fh.disk_mean(fh.percentile_raster(score, ref), 5)
    one = fh.null_means(dm, ref, [1] * 8, n_perm=3000, seed=0)
    three = fh.null_means(dm, ref, [3] * 8, n_perm=3000, seed=0)
    assert three.mean() > one.mean() + 1.5                              # more points -> larger expected maximum


def test_bootstrap_interval_brackets_the_mean():
    v = np.array([30, 50, 70, 90, 95, 20, 60, 80.0])
    lo, hi = fh.bootstrap_ci(v, n=4000, seed=0)
    assert lo < v.mean() < hi


def test_real_hotspots_load_as_16_hotspots_with_19_points_inside_the_grid():
    h = fh.load_hotspot_cells()
    assert len(h["points_by_hotspot"]) == 16 and sum(len(x) for x in h["points_by_hotspot"]) == 19
    assert h["names"][0] == "WL1" and h["names"][-1] == "WL16"
    H, W = h["shape"]
    assert all(0 <= r < H and 0 <= c < W for pts in h["points_by_hotspot"] for r, c in pts)
