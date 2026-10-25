"""T9 scenario engine: runoff, food and open-space proxies, block selection and map edits (synthetic maps)."""
import numpy as np
import pytest

from src import scenarios as sc

W, T, C, G, B, X, GARDEN, WETLAND, CORRIDOR = 0, 1, 2, 3, 4, 5, 6, 7, 8


def test_scs_runoff_matches_the_textbook_formula():
    S = 25400 / 74 - 254
    Ia = 0.2 * S
    assert sc.scs_runoff_mm(100.0, 74) == pytest.approx((100 - Ia) ** 2 / (100 - Ia + S))
    assert sc.scs_runoff_mm(100.0, 100) == pytest.approx(100.0)
    assert sc.scs_runoff_mm(5.0, 60) == 0.0                                              # rain below the initial abstraction


def test_runoff_volume_adds_class_runoff_and_subtracts_wetland_storage():
    cn = {G: 79, WETLAND: 98}
    m = np.full((10, 10), G, dtype=np.uint8)
    base = sc.runoff_volume_m3(m, np.ones_like(m, bool), 100.0, cn, wetland_storage_m=0.3)
    assert base == pytest.approx(100 * 100 * sc.scs_runoff_mm(100.0, 79) / 1000)
    m2 = m.copy(); m2[:5] = WETLAND
    after = sc.runoff_volume_m3(m2, np.ones_like(m, bool), 100.0, cn, wetland_storage_m=0.3)
    assert after < base                                                                  # storage outweighs the higher wetland CN
    assert sc.runoff_volume_m3(m2, np.ones_like(m, bool), 100.0, cn, wetland_storage_m=100.0) == 0.0   # never negative


def test_food_index_counts_cropland_and_gardens_only():
    m = np.array([[C, C, GARDEN, T, B]], dtype=np.uint8)
    assert sc.food_index_ha(m, np.ones_like(m, bool)) == pytest.approx(3 / 100)


def test_open_space_share_counts_osm_space_plus_new_gardens_and_corridors_over_baseline_built_area():
    m = np.array([[B, B, B, B, GARDEN, CORRIDOR, WETLAND, T]], dtype=np.uint8)
    osm = np.zeros_like(m, bool); osm[0, 7] = True
    share = sc.open_space_share(m, np.ones_like(m, bool), osm, built_px=4)
    assert share == pytest.approx(3 / 4)                                                 # garden + corridor + OSM park; not the wetland


def test_block_selection_follows_the_score_and_stops_once_the_budget_is_met():
    blocks = sc.block_ids((20, 20), size=10)                                             # four 10 x 10 blocks: ids 0..3
    cand = np.zeros((20, 20), bool); cand[:10, :] = True; cand[10:, :3] = True           # blocks 0, 1 full; 2 30%; 3 empty
    score = {0: 1.0, 1: 5.0, 2: 9.0, 3: 0.0}
    chosen = sc.select_blocks(cand, blocks, score, budget_px=150, min_frac=0.5)
    assert chosen == [1, 0]                                                              # block 2 is ineligible (30% < 50%)
    assert sc.select_blocks(cand, blocks, score, budget_px=50, min_frac=0.5) == [1]
    assert sc.select_blocks(cand, blocks, score, budget_px=50, min_frac=0.5, exclude={1}) == [0]


def test_apply_changes_only_candidate_cells_inside_chosen_blocks():
    m = np.full((20, 20), G, dtype=np.uint8); m[0, 0] = B
    blocks = sc.block_ids((20, 20), size=10)
    cand = (m == G)
    out = sc.apply_edit(m, cand, blocks, [0], WETLAND)
    assert out[0, 0] == B                                                                # built-up never edited
    assert (out[:10, :10][cand[:10, :10]] == WETLAND).all() and (out[10:, :] == G).all() and (out[:10, 10:] == G).all()
    assert (m[1, 1] == G)                                                                # input not modified in place


def test_candidates_follow_the_registered_rules():
    m = np.array([[G, X, C, B, T, W]], dtype=np.uint8)
    flood = np.array([[90, 90, 90, 90, 90, 90]], dtype=float)
    near_water = np.ones_like(m, bool); built_share = np.full(m.shape, 0.8); riparian = np.ones_like(m, bool)
    c = sc.candidates(m, np.ones_like(m, bool), flood, near_water, built_share, riparian)
    assert c["wetland"].tolist() == [[True, True, True, False, False, False]]
    assert c["garden"].tolist() == [[True, True, False, False, False, False]]
    assert c["corridor"].tolist() == [[True, True, True, False, False, False]]
    c2 = sc.candidates(m, np.ones_like(m, bool), flood * 0 + 50, near_water, built_share * 0, riparian * False)
    assert not c2["wetland"].any() and not c2["garden"].any() and not c2["corridor"].any()
