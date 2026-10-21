"""T6h: one-shot hotspot test for every benchmark model (no torch): Holm correction, score maps, the one-shot guard."""
import json

import numpy as np
import pytest

from src import flood_cnn_r3 as r3


def test_holm_matches_a_hand_worked_example():
    # sorted p: 0.01, 0.02, 0.03, 0.04 with m = 4 -> 0.04, 0.06, 0.06, 0.06 (running max keeps it monotone)
    out = r3.holm([0.04, 0.01, 0.03, 0.02])
    assert out == pytest.approx([0.06, 0.04, 0.06, 0.06])


def test_holm_caps_at_one_and_never_lowers_a_p_value():
    p = [0.9, 0.5, 0.001]
    out = r3.holm(p)
    assert all(o >= x for o, x in zip(out, p)) and max(out) <= 1.0


def test_score_map_scores_only_the_reference_cells_in_chunks():
    ref = np.zeros((6, 7), bool); ref[1:5, 2:6] = True
    seen = []

    def fn(rows, cols):
        seen.append(len(rows))
        return rows * 10.0 + cols

    m = r3.score_map(ref, fn, chunk=5)
    assert np.isnan(m[~ref]).all()
    rr, cc = np.nonzero(ref)
    assert np.allclose(m[ref], rr * 10.0 + cc)
    assert max(seen) <= 5 and sum(seen) == ref.sum()


def test_score_map_rejects_non_finite_scores_in_the_reference():
    ref = np.ones((3, 3), bool)
    with pytest.raises(ValueError):
        r3.score_map(ref, lambda r, c: np.where(r == 1, np.nan, 1.0))


def _toy_world(seed=0):
    rng = np.random.default_rng(seed)
    ref = np.ones((60, 60), bool)
    good = rng.random((60, 60))
    pts = [[(int(r), int(c))] for r, c in zip(rng.integers(5, 55, 16), rng.integers(5, 55, 16))]
    return ref, good, pts


def test_hotspot_test_passes_a_score_that_peaks_at_the_hotspots_and_not_a_random_one():
    ref, rand, pts = _toy_world()
    informed = rand.copy()
    for p in pts:
        informed[p[0][0] - 2:p[0][0] + 3, p[0][1] - 2:p[0][1] + 3] += 5.0
    res = r3.test_scores({"informed": informed, "random": rand}, ref, pts, radius_cells=3.0, n_perm=500)
    assert res["informed"]["pass"] and res["informed"]["mean_percentile"] > 85      # 25 of the 29 disk cells are boosted, so about 89
    assert not res["random"]["pass"]
    assert set(res["informed"]) >= {"mean_percentile", "p", "p_holm", "ci95", "share_top20pct", "null_mean"}


def test_holm_is_applied_across_all_models_tested():
    ref, rand, pts = _toy_world(1)
    res = r3.test_scores({"a": rand, "b": rand[::-1].copy(), "c": rand.T.copy()}, ref, pts, radius_cells=3.0, n_perm=200)
    assert all(res[k]["p_holm"] >= res[k]["p"] for k in res)


def test_one_shot_guard_refuses_a_second_run(tmp_path):
    f = tmp_path / "r3_hotspot_test.json"
    r3.guard_one_shot(f)                                                    # first time: fine
    f.write_text(json.dumps({"done": True}))
    with pytest.raises(FileExistsError):
        r3.guard_one_shot(f)
