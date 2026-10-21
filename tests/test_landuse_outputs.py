"""T4 outputs: metrics files and maps from the two trained seeds. Thresholds were fixed before the numbers were read."""
import json

import numpy as np
import pytest
import rasterio

from src import grid
from src.config import OUTPUTS

OUT = OUTPUTS / "landuse"
needed = [OUT / f for f in ["metrics_seed0.json", "metrics_seed1.json", "metrics_lightgbm_baseline.json",
                            "landuse_2023_seed0.tif", "landuse_2023_seed1.tif",
                            "landuse_2025_26_seed0.tif", "landuse_2025_26_seed1.tif"]]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in needed), reason="T4 outputs not produced yet")


@pytest.fixture(scope="module")
def metrics():
    return {k: json.loads((OUT / f"metrics_{k}.json").read_text()) for k in ["seed0", "seed1", "lightgbm_baseline"]}


def _map(name):
    with rasterio.open(OUT / name) as r:
        assert r.transform == grid.load_grid().transform
        return r.read(1)


def test_unet_beats_majority_class_and_the_per_pixel_baseline(metrics):
    for s in ["seed0", "seed1"]:
        m = metrics[s]
        oa = m["validation_blocks"]["overall_accuracy"]
        assert oa > m["majority_class_validation_accuracy"] + 0.25
        assert oa > metrics["lightgbm_baseline"]["overall_accuracy"] + 0.05      # spatial context must add value
        assert m["validation_blocks"]["macro_f1"] > metrics["lightgbm_baseline"]["macro_f1"]


def test_not_badly_overfitted_and_stable_across_seeds(metrics):
    for s in ["seed0", "seed1"]:
        v, t = metrics[s]["validation_blocks"], metrics[s]["training_blocks"]
        assert t["overall_accuracy"] - v["overall_accuracy"] < 0.12
    a, b = metrics["seed0"]["validation_blocks"], metrics["seed1"]["validation_blocks"]
    assert abs(a["overall_accuracy"] - b["overall_accuracy"]) < 0.01
    assert abs(a["macro_f1"] - b["macro_f1"]) < 0.03


def test_maps_are_complete_class_maps_on_the_grid():
    for name in ["landuse_2023_seed0.tif", "landuse_2023_seed1.tif", "landuse_2025_26_seed0.tif", "landuse_2025_26_seed1.tif"]:
        a = _map(name)
        assert a.shape == grid.load_grid().shape and a.dtype == np.uint8
        assert set(np.unique(a)) == {0, 1, 2, 3, 4, 5}, name


def test_seeds_agree_on_the_map_and_the_new_season_does_not_collapse():
    m = grid.study_mask()
    s0, s1 = _map("landuse_2025_26_seed0.tif"), _map("landuse_2025_26_seed1.tif")
    assert (s0[m] == s1[m]).mean() > 0.90                                        # pre-declared stability threshold
    for seed in ["seed0", "seed1"]:
        a23, a25 = _map(f"landuse_2023_{seed}.tif")[m], _map(f"landuse_2025_26_{seed}.tif")[m]
        assert abs((a25 == 4).mean() - (a23 == 4).mean()) < 0.03                 # built-up share barely moves between seasons
