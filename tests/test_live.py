"""T12 live twin: version registry, change report, rain outlook, automatic window search (synthetic), and the reproduction check
on the real 2023 run (skipped until that run exists)."""
import datetime as dt
import json

import numpy as np
import pytest

from src import live
from src import scenarios as sc

W, T, C, G, B, X = 0, 1, 2, 3, 4, 5


# ---------------------------------------------------------------- registry
def test_registry_appends_versions_and_refuses_a_duplicate_tag(tmp_path):
    reg = tmp_path / "versions.json"
    live.register(reg, dict(tag="a", window=["2023-01-01", "2023-12-31"]))
    live.register(reg, dict(tag="b", window=["2026-03-01", "2026-05-31"]))
    with pytest.raises(FileExistsError):
        live.register(reg, dict(tag="a", window=["x", "y"]))
    got = json.loads(reg.read_text())
    assert [v["tag"] for v in got] == ["a", "b"] and got[0]["window"] == ["2023-01-01", "2023-12-31"]   # first entry untouched
    assert live.previous_tag(reg, "b") == "a" and live.previous_tag(reg, "a") is None


# ---------------------------------------------------------------- change report
def test_transition_areas_count_each_pixel_once_inside_the_mask():
    old = np.array([[T, T, B], [C, C, W]], dtype=np.uint8)
    new = np.array([[T, B, B], [C, G, W]], dtype=np.uint8)
    mask = np.ones_like(old, bool); mask[1, 2] = False
    t = live.transitions(old, new, mask, px_ha=1.0)
    assert t.loc["Trees", "Built-up"] == 1 and t.loc["Trees", "Trees"] == 1 and t.loc["Cropland", "Grass/Scrub"] == 1
    assert t.values.sum() == 5 and t.loc["Water"].sum() == 0                                  # the masked water pixel is not counted


def test_patches_are_classified_new_lost_or_persisting_by_same_type_overlap():
    old = np.zeros((10, 10), int); old[0:4, 0:4] = 1; old[6:9, 6:9] = 2                       # 1 green, 2 water
    new = np.zeros((10, 10), int); new[0:4, 0:3] = 7; new[6:9, 0:3] = 8                       # 7 green (most of old 1), 8 green (new place)
    r = live.patch_change(old, {1: "green", 2: "water"}, new, {7: "green", 8: "green"}, min_share=0.10)
    assert r["new"] == [8] and r["lost"] == [2] and r["persisting"] == {7: 1}


def test_a_new_patch_of_another_type_on_old_ground_is_still_new():
    old = np.zeros((6, 6), int); old[:3, :3] = 1
    new = np.zeros((6, 6), int); new[:3, :3] = 5
    r = live.patch_change(old, {1: "farm"}, new, {5: "green"})
    assert r["new"] == [5] and r["lost"] == [1] and r["persisting"] == {}


# ---------------------------------------------------------------- rain outlook
def test_rain_outlook_uses_the_wettest_forecast_day_and_ranks_wards_by_runoff_depth():
    m = np.array([[B, B, T, T]], dtype=np.uint8)
    wards = np.array([[1, 1, 2, 2]])
    mask = np.ones_like(m, bool)
    fc = dict(time=["2026-10-26", "2026-10-27", "2026-10-28"], precipitation_sum=[1.0, 46.3, 19.0])
    r = live.rain_outlook(m, mask, wards, forecast=fc, pixel_m2=100.0)
    assert r["wettest_day"] == "2026-10-27" and r["rain_mm"] == pytest.approx(46.3)
    assert r["city_runoff_m3"] == pytest.approx(sc.runoff_volume_m3(m, mask, 46.3, sc.CN["C"], pixel_m2=100.0))
    assert [w["ward"] for w in r["top_wards"]] == [1, 2]                                       # built-up ward sheds more water
    assert r["top_wards"][0]["runoff_mm"] == pytest.approx(sc.scs_runoff_mm(46.3, sc.CN["C"][B]))


def test_a_dry_forecast_gives_zero_runoff_not_an_error():
    m = np.full((2, 2), B, np.uint8)
    r = live.rain_outlook(m, np.ones((2, 2), bool), np.ones((2, 2), int), forecast=dict(time=["d1"], precipitation_sum=[0.0]))
    assert r["city_runoff_m3"] == 0.0 and r["top_wards"][0]["runoff_mm"] == 0.0


# ---------------------------------------------------------------- automatic window
def test_latest_window_needs_enough_clear_days_and_something_new():
    days = [(dt.date(2026, 9, 18), 17.2), (dt.date(2026, 9, 28), 8.8), (dt.date(2026, 10, 13), 24.7), (dt.date(2026, 10, 23), 1.3)]
    today = dt.date(2026, 10, 25)
    w = live.latest_window(days, today, lookback_days=60, min_days=3, max_cloud=20.0)
    assert w == (dt.date(2026, 9, 18), today)                                                 # starts at the first clear day in reach
    assert live.latest_window(days, today, min_days=4) is None                                 # only 3 clear days (24.7% is too cloudy)
    assert live.latest_window(days, today, last_clear_registered=dt.date(2026, 10, 23)) is None   # nothing new since the last version
    assert live.latest_window(days, today, lookback_days=10, min_days=1) == (dt.date(2026, 10, 23), today)


# ---------------------------------------------------------------- real run (skipped until produced)
def test_the_live_path_reproduces_the_2023_land_use_product_pixel_for_pixel():
    import rasterio

    from src.config import OUTPUTS
    f = OUTPUTS / "twin_live" / "2023" / "landuse.tif"
    if not f.exists():
        pytest.skip("live 2023 version not built yet")
    a = rasterio.open(f).read(1)
    b = rasterio.open(OUTPUTS / "landuse" / "landuse_2023_corr.tif").read(1)
    assert a.shape == b.shape and int((a != b).sum()) == 0
