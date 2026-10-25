"""T9b: SDG 11.3.1, 11.7.1 and 15.1.1 building blocks against hand-worked examples."""
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString

from src import sdg


def test_lcrpgr_matches_a_hand_worked_example():
    r = sdg.lcrpgr(urb1=100.0, urb2=121.0, pop1=1000.0, pop2=1100.0, years=5)
    assert r["lcr"] == pytest.approx(np.log(1.21) / 5) and r["pgr"] == pytest.approx(np.log(1.1) / 5)
    assert r["ratio"] == pytest.approx(2.0)                                              # ln 1.21 = 2 ln 1.1


def test_road_widths_follow_the_registered_classes():
    assert sdg.road_width("trunk") == 20 and sdg.road_width("primary_link") == 14 and sdg.road_width("residential") == 6
    assert sdg.road_width("footway") == 3 and sdg.road_width(None) == 3


def test_overlapping_streets_are_counted_once():
    tf = Affine(1, 0, 0, 0, -1, 100)
    a = LineString([(10, 50), (90, 50)]); b = LineString([(50, 10), (50, 90)])
    one = sdg.street_mask([(a, 10)], (100, 100), tf).sum(); both = sdg.street_mask([(a, 10), (b, 10)], (100, 100), tf).sum()
    assert both < 2 * one and both > one


def test_open_space_share_uses_the_union_of_streets_and_open_space_over_the_urbanised_area():
    built = np.zeros((10, 10), bool); built[:, :6] = True
    street = np.zeros_like(built); street[:, 5:7] = True                                   # one column overlaps built-up
    osm = np.zeros_like(built); osm[0, 9] = True
    s = sdg.open_public_space_share(osm, street, built, np.ones_like(built, bool), extra_open=None)
    assert s == pytest.approx((20 + 1) / (60 + 10 + 1))                                   # (street U osm) / (built U street U osm)


def test_forest_share_counts_only_large_tree_patches_and_excludes_water_from_land():
    m = np.full((20, 20), 4, dtype=np.uint8)
    m[:10, :10] = 1                                                                       # 100 px of trees = 1 ha
    m[15, 15] = 1                                                                         # 1 px of trees: too small
    m[:, 19] = 0                                                                          # water column
    share = sdg.forest_share(m, np.ones_like(m, bool), min_px=50)
    assert share == pytest.approx(100 / (400 - 20))
