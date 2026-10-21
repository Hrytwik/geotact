"""T6c: per-orbit flood masks and flood frequency from the radar scenes (synthetic scenes with known answers)."""
import numpy as np
import pandas as pd
import pytest

from src import flood_labels as fl


def scene(base=-8.0, runway=-20.0, pond=-20.0, field=None):
    x = np.full((20, 20), base, dtype="float32")
    x[0:4, :] = runway                 # runway / asphalt: dark in every season
    x[8:12, 8:12] = pond               # permanent pond
    if field is not None:
        x[14:18, 2:6] = field          # arable field that floods in the monsoon
    return x


def test_only_the_flooded_field_is_flagged_and_permanent_dark_surfaces_are_not():
    dry = np.stack([scene(), scene(), scene()])
    mon = np.stack([scene(field=-18.0), scene(field=-8.0)])
    masks, ref, dark = fl.flood_masks_for_orbit(dry, [-12.0] * 3, mon, [-12.0, -12.0])
    m0, m1 = masks
    assert m0[14:18, 2:6].all()                          # flooded field flagged
    assert m0.sum() == 16                                # nothing else: runway and pond are persistently dark
    assert not m1.any()                                  # a dry monsoon scene flags nothing
    assert dark[0:4, :].min() == 1.0 and dark[8:12, 8:12].min() == 1.0 and dark[14:18, 2:6].max() == 0.0


def test_a_moderate_drop_without_enough_change_is_not_flooding():
    dry = np.stack([scene(base=-11.0)] * 3)               # a naturally darker surface
    mon = np.stack([scene(base=-11.0, field=-13.0)])      # dark enough (< -12) but only 2 dB lower than its dry reference
    (m,), _, _ = fl.flood_masks_for_orbit(dry, [-12.0] * 3, mon, [-12.0])
    assert not m[14:18, 2:6].any()


def test_terrain_masks_remove_steep_and_high_cells():
    dry = np.stack([scene()] * 3)
    mon = np.stack([scene(field=-18.0)])
    slope = np.zeros((20, 20), "float32"); slope[14:16, 2:6] = 9.0
    hand = np.zeros((20, 20), "float32"); hand[16:18, 2:6] = 40.0
    (m,), _, _ = fl.flood_masks_for_orbit(dry, [-12.0] * 3, mon, [-12.0], slope=slope, hand=hand)
    assert not m[14:18, 2:6].any()                       # top two rows too steep, bottom two rows too high above drainage


def test_orbits_are_never_mixed():
    """A descending-orbit scene sees the same ground 3 dB darker. Using the ascending reference would flag it all."""
    df = pd.DataFrame(dict(kind=["dry", "dry", "monsoon", "dry", "dry", "monsoon"], rel=[85, 85, 85, 121, 121, 121],
                           day=["a", "b", "c", "d", "e", "f"], cover=1.0))
    g = fl.group_by_orbit(df)
    assert set(g) == {85, 121}
    assert g[85]["dry"] == ["a", "b"] and g[85]["monsoon"] == ["c"]
    assert g[121]["dry"] == ["d", "e"] and g[121]["monsoon"] == ["f"]
    asc = np.stack([scene(base=-8.0)] * 3)
    desc = np.stack([scene(base=-11.0)] * 3)
    mon_desc = np.stack([scene(base=-11.0)])
    wrong, _, _ = fl.flood_masks_for_orbit(asc, [-12.0] * 3, mon_desc, [-9.0])        # asc reference: a false flood everywhere
    right, _, _ = fl.flood_masks_for_orbit(desc, [-12.0] * 3, mon_desc, [-9.0])
    assert wrong[0][5:8, 12:18].all() and not right[0].any()


def test_scene_quality_rule_matches_the_protocol():
    assert fl.scene_ok(0.80) and fl.scene_ok(0.93) and not fl.scene_ok(0.799)


def test_frequency_over_scenes_uses_valid_observations_only():
    dry = np.stack([scene()] * 3)
    mon = np.stack([scene(field=-18.0), scene(field=-18.0), scene(field=-8.0), scene(field=np.nan)])
    masks, ref, _ = fl.flood_masks_for_orbit(dry, [-12.0] * 3, mon, [-12.0] * 4)
    valid = np.stack([np.isfinite(m) & np.isfinite(ref) for m in mon])
    f, n = fl.flood_frequency(np.stack(masks), valid, min_valid=3)
    assert f[15, 3] == pytest.approx(2 / 3) and n[15, 3] == 3          # the NaN scene is not counted as 'not flooded'
    assert f[5, 15] == 0.0 and n[5, 15] == 4


def test_frequency_from_running_counts_equals_the_stack_version():
    rng = np.random.default_rng(0)
    flood = rng.random((9, 6, 7)) < 0.3
    valid = rng.random((9, 6, 7)) < 0.8
    f1, n1 = fl.frequency_from_counts((flood & valid).sum(0), valid.sum(0), min_valid=5)
    from src import flood_sar as fs
    f2, n2 = fs.flood_frequency(flood, valid, min_valid=5)
    assert np.array_equal(f1, f2, equal_nan=True) and np.array_equal(n1, n2)
