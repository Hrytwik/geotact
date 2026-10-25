"""T4b: WorldCover mapping and the consensus-label rule (pure functions on small arrays)."""
import numpy as np
import pytest

from src import landuse as lu

W, T, C, G, B, X = 0, 1, 2, 3, 4, 5                                                      # chapter class indices


def test_worldcover_codes_map_to_the_chapter_classes_and_unusable_codes_are_ignored():
    wc = np.array([10, 95, 20, 30, 40, 50, 60, 80, 90, 70, 100, 0, 255], dtype=np.uint8)
    out = lu.wc_labels(wc)
    assert out.tolist() == [T, T, G, G, C, B, X, W, W, lu.IGNORE, lu.IGNORE, lu.IGNORE, lu.IGNORE]


def test_consensus_keeps_only_pixels_where_both_products_agree():
    esri = np.array([T, B, C, G, lu.IGNORE, W], dtype=np.uint8)
    wc = np.array([T, T, C, C, T, lu.IGNORE], dtype=np.uint8)
    out = lu.consensus(esri, wc)
    assert out.tolist() == [T, lu.IGNORE, C, lu.IGNORE, lu.IGNORE, lu.IGNORE]


def test_labels_from_dispatches_on_the_mode_and_rejects_unknown_modes():
    io = np.array([2, 7, 5], dtype=np.uint8)                                              # Esri: trees, built-up, crops
    wc = np.array([10, 10, 40], dtype=np.uint8)                                           # WorldCover: trees, trees, crops
    assert lu.labels_from(io, wc, "esri").tolist() == [T, B, C]
    assert lu.labels_from(io, wc, "consensus").tolist() == [T, lu.IGNORE, C]
    with pytest.raises(ValueError):
        lu.labels_from(io, wc, "other")


def test_consensus_never_adds_labels_that_esri_does_not_have():
    rng = np.random.default_rng(0)
    io = rng.choice([1, 2, 4, 5, 7, 8, 11], size=2000).astype(np.uint8)
    wc = rng.choice([10, 20, 30, 40, 50, 60, 80, 90], size=2000).astype(np.uint8)
    e, c = lu.labels_from(io, wc, "esri"), lu.labels_from(io, wc, "consensus")
    kept = c != lu.IGNORE
    assert (c[kept] == e[kept]).all() and kept.sum() <= (e != lu.IGNORE).sum()
