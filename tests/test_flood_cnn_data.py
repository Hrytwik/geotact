"""T6f: samples and channels for the deep flood model trained on field-surveyed locations."""
import numpy as np
import pytest

from src import flood_cnn_data as fd


def test_absolute_elevation_is_not_a_channel_but_local_relief_is():
    assert "elevation" not in fd.CHANNELS and "local_relief" in fd.CHANNELS
    assert len(fd.CHANNELS) == len(set(fd.CHANNELS)) == 16


def test_local_relief_is_zero_on_a_plane_and_signed_in_a_dip():
    plane = np.add.outer(np.arange(80) * 0.5, np.arange(80) * 0.2)
    rel = fd.local_relief(plane, size=33)
    assert np.abs(rel[20:-20, 20:-20]).max() < 1e-6                       # a tilted plane has no local relief away from the edges
    bowl = np.full((80, 80), 10.0); bowl[38:42, 38:42] = 5.0
    r2 = fd.local_relief(bowl, size=33)
    assert r2[40, 40] < -3 and r2[10, 10] > -0.5                          # a dip is negative relief; far cells are about zero


def test_patches_are_centred_on_the_requested_cell_and_pad_at_the_edges():
    C, H, W = 3, 40, 50
    stack = np.random.default_rng(0).normal(size=(C, H, W)).astype("float32")
    src = fd.PatchSource(stack, size=33)
    rows, cols = np.array([20, 0, 39]), np.array([25, 0, 49])
    P = src.get(rows, cols)
    assert P.shape == (3, C, 33, 33)
    for i, (r, c) in enumerate(zip(rows, cols)):
        assert np.array_equal(P[i, :, 16, 16], stack[:, r, c])           # centre pixel is the cell itself, also at the corners
    assert np.array_equal(src.get(np.array([20]), np.array([25]))[0], P[0])         # batch size does not change a patch
    assert np.array_equal(P[0, :, 0:5, 0:5], stack[:, 4:9, 9:14])


def test_background_cells_respect_every_exclusion_and_are_reproducible():
    H, W = 120, 120
    ref = np.zeros((H, W), bool); ref[10:110, 10:110] = True
    water = np.zeros((H, W), bool); water[40:60, 40:60] = True
    pos = [(30, 30), (70, 80), (95, 20)]
    a = fd.sample_background(ref & ~water, pos, n=800, seed=0, min_dist_cells=2)
    b = fd.sample_background(ref & ~water, pos, n=800, seed=0, min_dist_cells=2)
    assert np.array_equal(a, b) and len(a) == 800 and len({tuple(x) for x in a}) == 800
    assert (ref & ~water)[a[:, 0], a[:, 1]].all()
    for (r, c) in pos:
        assert np.hypot(a[:, 0] - r, a[:, 1] - c).min() >= 2.0           # nowhere within 2 cells (60 m) of a flood point
    assert not np.array_equal(a, fd.sample_background(ref & ~water, pos, n=800, seed=1))


def test_samples_carry_their_own_spatial_fold():
    H, W = 140, 150
    cells = {"flood": np.array([[20, 20], [100, 120], [70, 40]]), "nonflood": np.array([[50, 90]]),
             "background": np.array([[10, 10], [130, 140], [64, 70]])}
    s = fd.assemble(cells, (H, W), block=40, k=5, seed=0)
    assert set(s["kind"]) == {"flood", "nonflood", "background"} and len(s["y"]) == 7
    assert s["y"].tolist() == [1, 1, 1, 0, 0, 0, 0]
    from src import flood_model as fm
    folds = fm.block_folds((H, W), 40, 5, 0)
    assert all(s["fold"][i] == folds[s["rows"][i], s["cols"][i]] for i in range(7))


# ---------------------------------------------------------------- real data
@pytest.fixture(scope="module")
def real():
    return fd.build()


def test_real_samples_have_the_registered_composition(real):
    s = real["samples"]
    kinds = list(s["kind"])
    assert 100 <= kinds.count("flood") <= 129                              # unique 30 m cells among 129 digitised points
    assert kinds.count("nonflood") == 25 and kinds.count("background") == 10000
    folds = np.asarray(s["fold"])
    per_fold = [int(((folds == k) & (np.asarray(s["y"]) == 1)).sum()) for k in range(5)]
    assert min(per_fold) >= 5 and sum(per_fold) == kinds.count("flood")      # observed 42/16/32/7/32: uneven because points cluster, hence POOLED AUC
    r, c = np.asarray(s["rows"]), np.asarray(s["cols"])
    H, W = real["shape"]
    assert r.min() >= 0 and c.min() >= 0 and r.max() < H and c.max() < W


def test_real_channels_exist_and_are_finite_where_it_matters(real):
    st = real["stack"]
    assert st.shape[0] == 16 and st.shape[1:] == real["shape"]
    bmc = real["bmc"]
    for i, name in enumerate(fd.CHANNELS):
        frac = np.isfinite(st[i][bmc]).mean()
        assert frac >= (0.995 if name in ("hand", "ndvi") else 0.999), (name, frac)


def test_real_background_is_inside_the_city_and_not_in_mapped_water(real):
    s = real["samples"]
    idx = [i for i, k in enumerate(s["kind"]) if k == "background"]
    r, c = np.asarray(s["rows"])[idx], np.asarray(s["cols"])[idx]
    assert real["bmc"][r, c].all() and not real["water"][r, c].any()
