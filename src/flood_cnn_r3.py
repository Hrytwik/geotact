"""T6h: ONE-SHOT hotspot test (R3) for every benchmark model (Amendment 1 in docs/build_plan.md). Same statistic, null and
reference as T6e (src/flood_hotspots.py): percentile of the score among BMC cells that are not mapped water, mean over a 150 m disk,
maximum over each hotspot's points, mean over the 16 hotspots, 5000 random hotspot sets. The result is REPORTED for every model
and never used to rank or select; p-values are given uncorrected and Holm-corrected across the models tested. No torch here.

Stage 1 (fit_all_and_score in flood_cnn_baselines.py and flood_cnn.py) refits each model on all field points and saves a score map
WITHOUT touching the hotspots. Stage 2 (this file) reads the maps and runs the test once; it refuses to run again."""
import json
from pathlib import Path

import numpy as np

from src import flood_hotspots as fh
from src.config import OUTPUTS

OUT = OUTPUTS / "flood_cnn"
RESULT = OUT / "r3_hotspot_test.json"
MODELS = ("cnn", "mlp", "logistic", "forest", "lightgbm", "frozen_t6d", "layer_hand")


def holm(p):
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty_like(adj)
    out[order] = adj
    return out.tolist()


def score_map(ref, fn, chunk=4096):
    """Grid of fn(rows, cols) on the reference cells (computed in chunks), NaN elsewhere. Non-finite scores are an error."""
    rows, cols = np.nonzero(ref)
    out = np.full(ref.shape, np.nan, dtype="float32")
    for s in range(0, len(rows), chunk):
        v = np.asarray(fn(rows[s:s + chunk], cols[s:s + chunk]), dtype="float64")
        if not np.isfinite(v).all():
            raise ValueError("scores contain NaN or inf on reference cells")
        out[rows[s:s + chunk], cols[s:s + chunk]] = v
    return out


def test_scores(score_maps, ref, points_by_hotspot, radius_cells, n_perm=5000, seed=0):
    counts = [len(p) for p in points_by_hotspot]
    res = {}
    for name, score in score_maps.items():
        refm = np.asarray(ref, bool) & np.isfinite(score)
        dm = fh.disk_mean(fh.percentile_raster(score, refm), radius_cells)
        v = fh.hotspot_values(dm, points_by_hotspot)
        r = fh.permutation_test(v, dm, refm, counts, n_perm, seed)
        res[name] = dict(**r, ci95=fh.bootstrap_ci(v, seed=seed), median_percentile=float(np.median(v)),
                         share_top20pct=float((v >= 80).mean()), n_hotspots=len(v), per_hotspot=v.round(1).tolist(),
                         **{"pass": bool(r["mean_percentile"] >= fh.PASS_MEAN_PERCENTILE and r["p"] < fh.PASS_P)})
    for name, ph in zip(res, holm([res[k]["p"] for k in res])):
        res[name]["p_holm"] = ph
    return res


def guard_one_shot(path=RESULT):
    if Path(path).exists():
        raise FileExistsError(f"{path} exists: the hotspot test is one-shot and has already been run")


def run(n_perm=5000, seed=0):
    import rasterio

    from src import terrain as tr
    guard_one_shot()
    d = tr.real_inputs()
    ref = d["bmc"] & ~d["water"]
    h = fh.load_hotspot_cells()
    scores = {k: np.load(OUT / f"r3_scores_{k}.npy") for k in ("cnn", "mlp", "logistic", "forest", "lightgbm")}
    with rasterio.open(OUTPUTS / "terrain" / "terrain_features.tif") as r:
        hand = r.read(r.descriptions.index("hand") + 1)
    scores["frozen_t6d"] = rasterio.open(OUTPUTS / "flood" / "susceptibility_30m.tif").read(1)
    scores["layer_hand"] = -hand
    scores = {k: scores[k] for k in MODELS}
    res = test_scores(scores, ref, h["points_by_hotspot"], fh.RADIUS_M / abs(h["transform"].a), n_perm, seed)
    for k, r in res.items():
        print(f"{k:12s} mean percentile {r['mean_percentile']:5.1f} [{r['ci95'][0]:5.1f}, {r['ci95'][1]:5.1f}]  null {r['null_mean']:5.1f}  "
              f"p {r['p']:.4f}  p_holm {r['p_holm']:.4f}  top20% {int(round(r['share_top20pct'] * 16))}/16  "
              f"{'PASS' if r['pass'] else 'fail'}", flush=True)
    RESULT.write_text(json.dumps(dict(n_perm=n_perm, seed=seed, criterion=f"mean percentile >= {fh.PASS_MEAN_PERCENTILE} and p < {fh.PASS_P}",
                                      hotspots=h["names"], models=res), indent=1))
    return res


if __name__ == "__main__":
    run()
