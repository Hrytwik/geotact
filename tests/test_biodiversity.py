"""T12b bird-habitat layer: visits, guild flags, effort without the own guild, name resolution, radius covariates, decision rule
(synthetic data)."""
import numpy as np
import pandas as pd
import pytest

from src import biodiversity as bd


def records():
    return pd.DataFrame(dict(
        species=["Ardea alba", "Dicrurus macrocercus", "Picus xanthopygaeus", "Ardea alba", "Dicrurus macrocercus", "Dicrurus macrocercus"],
        lat=[20.30001, 20.30001, 20.30001, 20.25, 20.25, 20.25], lon=[85.80001, 85.80001, 85.80001, 85.82, 85.82, 85.82],
        date=["2024-01-05", "2024-01-05", "2024-01-05", "2024-01-05", "2024-01-05", "2024-02-01"]))


GUILD_OF = {"Ardea alba": "wetland", "Picus xanthopygaeus": "tree", "Dicrurus macrocercus": None}


def test_visits_group_records_by_place_and_day_and_flag_guilds():
    v = bd.make_visits(records(), GUILD_OF).sort_values(["lat", "date"]).reset_index(drop=True)
    assert len(v) == 3                                                                      # two places, one of them on two days
    assert v.n_species.tolist() == [2, 1, 3]
    assert v.has_wetland.tolist() == [1, 0, 1] and v.has_tree.tolist() == [0, 0, 1]


def test_effort_counts_only_species_outside_the_guild():
    v = bd.make_visits(records(), GUILD_OF).sort_values(["lat", "date"]).reset_index(drop=True)
    assert v.effort_wetland.tolist() == pytest.approx(np.log1p([1, 1, 2]))                  # the egret itself is not effort for wetland
    assert v.effort_tree.tolist() == pytest.approx(np.log1p([2, 1, 2]))


def test_names_resolve_exactly_then_through_synonyms_and_otherwise_stay_unmatched():
    known = {"Bubulcus ibis", "Ardea alba"}
    syn = {"Ardea ibis": ["Bubulcus ibis"], "Foo bar": ["Baz qux"]}
    got = bd.resolve_names(["Ardea alba", "Ardea ibis", "Foo bar"], known, lambda n: syn.get(n, []))
    assert got == {"Ardea alba": "Ardea alba", "Ardea ibis": "Bubulcus ibis", "Foo bar": None}


def test_radius_shares_match_a_brute_force_disk_and_handle_the_edge():
    rng = np.random.default_rng(0)
    m = rng.integers(0, 6, (40, 50)).astype(np.uint8)
    S = bd.class_shares(m, radius_px=4)
    yy, xx = np.mgrid[-4:5, -4:5]; disk = (yy ** 2 + xx ** 2) <= 16
    for r, c in [(20, 25), (0, 0), (39, 10)]:
        win = [(r + dy, c + dx) for dy, dx in zip(yy[disk], xx[disk]) if 0 <= r + dy < 40 and 0 <= c + dx < 50]
        vals = np.array([m[a, b] for a, b in win])
        for k in range(6):
            assert S[k, r, c] == pytest.approx((vals == k).mean(), abs=1e-5)
    assert np.allclose(S.sum(axis=0), 1.0, atol=1e-5)


def test_masked_mean_ignores_missing_values():
    a = np.ones((20, 20), "float32"); a[5:10, 5:10] = np.nan; a[15, 15] = 3.0
    out = bd.masked_disk_mean(a, radius_px=2)
    assert np.isnan(out[7, 7])                                                               # every cell within 2 px is missing
    assert out[2, 2] == pytest.approx(1.0) and out[15, 15] == pytest.approx((12 + 3.0) / 13)  # 13 cells in a radius-2 disk


def test_decision_rule_needs_both_skill_and_a_gain_over_effort_only():
    assert bd.decide(0.75, [0.70, 0.80], [0.02, 0.09])
    assert not bd.decide(0.69, [0.64, 0.74], [0.02, 0.09])
    assert not bd.decide(0.75, [0.70, 0.80], [-0.01, 0.09])


def test_location_bootstrap_resamples_whole_locations():
    rng = np.random.default_rng(1)
    loc = np.repeat(np.arange(40), 5)
    y = (rng.random(200) < 0.4).astype(int)
    a = y + rng.normal(0, 0.8, 200); b = rng.normal(0, 1, 200)
    r = bd.location_bootstrap(a, b, y, loc, n_boot=300)
    assert r["auc_ci"][0] < r["auc"] < r["auc_ci"][1] and r["diff_ci"][0] > 0
