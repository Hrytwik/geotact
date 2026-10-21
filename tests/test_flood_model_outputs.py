"""T6d outputs. Guards were fixed before the hotspot test (T6e) and are sanity bounds, not tuning targets."""
import json

import numpy as np
import pytest
import rasterio

from src import terrain as tr
from src.config import OUTPUTS

OUT = OUTPUTS / "flood"
pytestmark = pytest.mark.skipif(not (OUT / "metrics_model.json").exists(), reason="T6d outputs not produced yet")


@pytest.fixture(scope="module")
def m():
    return json.loads((OUT / "metrics_model.json").read_text())


def test_cross_validated_skill_is_real_and_beats_every_single_feature(m):
    assert m["cv_auc"] >= 0.65 and min(m["fold_auc"]) > 0.6
    assert m["cv_auc"] >= max(m["single_feature_auc"].values()) - 0.01          # the model must add something over one raw layer
    assert m["auc_within"]["inside_BMC"] >= 0.65                                  # and must work inside the city, not only outside


def test_stream_threshold_does_not_drive_the_result(m):
    aucs = [v["auc"] for v in m["stream_threshold_sensitivity"].values()]
    assert max(aucs) - min(aucs) < 0.02


def test_susceptibility_raster_is_a_probability_on_the_dem_grid(m):
    with rasterio.open(OUT / "susceptibility_30m.tif") as r:
        p = r.read(1)
        assert r.transform == rasterio.open(OUTPUTS / "terrain" / "terrain_features.tif").transform
    bmc = tr.real_inputs()["bmc"]
    assert p.shape == bmc.shape and np.isfinite(p[bmc]).all()
    assert np.nanmin(p) >= 0 and np.nanmax(p) <= 1


def test_out_of_fold_map_is_defined_exactly_on_the_eligible_cells(m):
    with rasterio.open(OUT / "oof_30m.tif") as r:
        o = r.read(1)
    assert np.isfinite(o).sum() == m["n_cells"]
