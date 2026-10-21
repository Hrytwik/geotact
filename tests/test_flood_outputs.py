"""T6c outputs on the real scenes. Bounds are generous sanity guards fixed before modelling, not tuning targets."""
import numpy as np
import pandas as pd
import pytest
import rasterio

from src import flood_labels as fl
from src import grid
from src.config import OUTPUTS, RAW

OUT = OUTPUTS / "flood"
pytestmark = pytest.mark.skipif(not (OUT / "flood_freq.tif").exists(), reason="T6c outputs not produced yet")


def test_every_scene_was_screened_and_all_orbits_are_present():
    t = pd.read_csv(OUT / "scene_table.csv")
    assert len(t) == 43 and set(t.orbit) == {85, 121}
    assert (t[t.used].balanced_accuracy >= fl.MIN_BALANCED_ACCURACY).all()
    assert (t[t.kind == "monsoon"].used).sum() >= 20


def test_per_scene_flagged_area_is_small_and_plausible():
    m = pd.read_csv(OUT / "scene_table.csv").query("kind == 'monsoon' and used")
    assert m.flood_pct_bmc_open.between(0.1, 5.0).all()                      # radar-visible surface water, not a city under water


def test_flood_frequency_map_is_defined_and_sparse_in_open_land():
    f = rasterio.open(OUT / "flood_freq.tif").read(1)
    n = rasterio.open(OUT / "n_valid.tif").read(1)
    wc = grid.read_aligned(RAW / "worldcover_2021.tif", resampling="nearest", dtype="uint8")
    open_ = grid.study_mask() & (wc != 50) & (wc != 80)
    assert np.isfinite(f[open_]).mean() > 0.99 and np.median(n[open_]) >= 20
    pos = (f[open_] >= 0.12 - 1e-6).mean()
    assert 0.005 < pos < 0.10                                                  # about 2.4 percent of open pixels flagged in >= 3 of 25 scenes


def test_the_label_is_calendar_driven_not_rain_driven_in_cropland():
    """Pins the finding that decides how the label is described: cropland flags fall with the calendar (paddy water) and do
    not respond to antecedent rain. If this ever changes, the chapter wording must be revisited."""
    d = fl.rain_diagnostic().set_index("cls")
    c = d.loc["cropland"]
    assert c.rho_calendar < -0.4 and c.p_calendar < 0.01
    assert abs(c.rho_rain3d) < 0.4 and abs(c.rho_rain7d) < 0.4
    assert c.ratio > 2.5                                                       # June-August flagged area much larger than Sep-Oct
