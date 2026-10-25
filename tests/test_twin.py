"""T8 graph twin: building blocks on small synthetic maps (no real data)."""
import math

import numpy as np
import pytest

from src import twin as tw

GROUPS = {"water": [0], "green": [1, 3], "farm": [2]}
BUILT = 4


def toy_map():
    m = np.full((20, 20), BUILT, dtype=np.uint8)
    m[1:5, 1:5] = 1          # green A, 16 px
    m[1:5, 5:7] = 3          # grass touching A -> same green patch (24 px)
    m[10:13, 10:13] = 0      # water, 9 px
    m[10:13, 13:16] = 2      # farm touching the water, 9 px
    m[18, 18] = 1            # tiny green, 1 px (below the minimum)
    return m


def test_patches_merge_trees_and_grass_and_drop_small_ones():
    pid, table, small = tw.label_patches(toy_map(), np.ones((20, 20), bool), GROUPS, min_px=4)
    assert sorted(table.type) == ["farm", "green", "water"]
    assert table.set_index("type").n_px.to_dict() == {"green": 24, "water": 9, "farm": 9}
    assert small["green"] == 1 and pid[18, 18] == 0
    assert set(np.unique(pid)) == {0} | set(table.id)
    for r in table.itertuples():
        assert (pid == r.id).sum() == r.n_px


def test_patches_stay_inside_the_study_mask():
    study = np.ones((20, 20), bool); study[:, :3] = False
    pid, table, _ = tw.label_patches(toy_map(), study, GROUPS, min_px=4)
    assert table.set_index("type").n_px["green"] == 24 - 8 and not pid[:, :3].any()


def test_zonal_mean_ignores_nan_and_returns_one_value_per_patch():
    pid = np.array([[1, 1, 2], [0, 2, 2]])
    v = np.array([[1.0, np.nan, 3.0], [9.0, 5.0, 7.0]])
    out = tw.zonal_mean(pid, v, n=2)
    assert out[1] == pytest.approx(1.0) and out[2] == pytest.approx(5.0)


def test_gaps_between_patches_are_measured_edge_to_edge():
    pid = np.zeros((10, 20), int)
    pid[2:5, 2:5] = 1
    pid[2:5, 5:8] = 2            # touching patch 1
    pid[2:5, 9:12] = 3           # one empty column away from patch 2
    gaps = tw.patch_gaps(pid, [1, 2, 3], max_gap_m=100, pixel_m=10)
    assert gaps[(1, 2)] == 0.0 and gaps[(2, 3)] == pytest.approx(10.0)
    assert gaps[(1, 3)] == pytest.approx(40.0)
    assert all(i < j for i, j in gaps)


def test_gaps_respect_the_maximum_distance():
    pid = np.zeros((10, 60), int); pid[2:5, 2:5] = 1; pid[2:5, 50:53] = 2
    assert tw.patch_gaps(pid, [1, 2], max_gap_m=100, pixel_m=10) == {}


def test_connectivity_weight_halves_at_the_half_distance_and_stays_in_range():
    assert tw.connectivity_weight(0.0, 500) == 1.0
    assert tw.connectivity_weight(500.0, 500) == pytest.approx(0.5)
    assert 0 < tw.connectivity_weight(1500.0, 500) <= 1


def test_pc_index_matches_a_hand_worked_three_node_chain():
    # areas 1, 2, 3; edges 1-2 p=0.5, 2-3 p=0.5; best 1-3 path = 0.25; A = 10
    areas = {1: 1.0, 2: 2.0, 3: 3.0}
    edges = [(1, 2, 0.5), (2, 3, 0.5)]
    pc = tw.pc_index(areas, edges, total_area=10.0)
    num = (1 + 4 + 9) + 2 * (1 * 2 * 0.5 + 2 * 3 * 0.5 + 1 * 3 * 0.25)
    assert pc == pytest.approx(num / 100.0)


def test_dpc_is_largest_for_the_connector_node():
    areas = {1: 1.0, 2: 1.0, 3: 1.0}
    edges = [(1, 2, 0.9), (2, 3, 0.9)]
    d = tw.dpc(areas, edges, total_area=3.0)
    assert d[2] > d[1] and d[1] == pytest.approx(d[3]) and all(v > 0 for v in d.values())


def test_points_are_assigned_once_to_the_nearest_patch_within_reach():
    pid = np.zeros((20, 20), int); pid[2:5, 2:5] = 1; pid[12:15, 12:15] = 2
    rows = np.array([3, 6, 13, 19]); cols = np.array([3, 3, 16, 0])
    got = tw.assign_points(pid, rows, cols, max_px=3)
    assert got.tolist() == [1, 1, 2, 0]                                                  # last one is too far


def test_drainage_links_chain_patches_downhill_along_a_shared_channel():
    pid = np.zeros((10, 30), int)
    pid[1:3, 2:5] = 1; pid[1:3, 12:15] = 2; pid[1:3, 22:25] = 3
    drains = np.zeros((10, 30), bool); drains[4, :] = True                               # one channel under all three
    elev = {1: 30.0, 2: 10.0, 3: 20.0}
    e = tw.drainage_edges(pid, drains, elev, max_px=3)
    assert e == [(1, 3), (3, 2)]                                                         # 30 m -> 20 m -> 10 m


def test_reliability_is_the_area_weighted_class_accuracy():
    m = np.array([[1, 1, 3, 3]], dtype=np.uint8); pid = np.ones((1, 4), int)
    r = tw.reliability(pid, m, {1: 0.9, 3: 0.1}, n=1)
    assert r[1] == pytest.approx(0.5)
