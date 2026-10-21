"""T6e outputs: consistency of the recorded one-shot result with the registered criterion (does not pin pass or fail)."""
import json

import pytest

from src.config import OUTPUTS

F = OUTPUTS / "flood" / "hotspot_test.json"
pytestmark = pytest.mark.skipif(not F.exists(), reason="T6e not run yet")


def test_recorded_verdict_matches_the_registered_criterion_and_all_methods_are_present():
    r = json.loads(F.read_text())
    assert {"model", "low_elevation", "low_HAND", "high_TWI", "near_drain"} <= set(r)
    m = r["model"]
    assert r["PASS"] == (m["mean_percentile"] >= 70.0 and m["p"] < 0.05)
    assert m["n_hotspots"] == 16 and m["n_perm"] == 5000
    lo, hi = m["ci95"]
    assert lo < m["mean_percentile"] < hi
    assert 45 < m["null_mean"] < 55                                   # the null of the (amended) disk-mean statistic sits near 50
