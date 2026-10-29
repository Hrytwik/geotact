"""T6j Part B: urban street-flooding model (protocol: Amendments 2 and 2a in docs/results_log.md). Sample assembly, the training
regimes and the two evaluation tasks. No torch here (LightGBM runs in this process)."""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

BUILTUP_MIN = 0.5
REGIMES = ("S_full", "S", "U", "S+U")


def urban_cells(csv_path, project, drop_doubtful=False):
    """Unique (row, col) 30 m cells of the usable urban rows. `project(lon, lat)` maps coordinates to (rows, cols) of the grid."""
    d = pd.read_csv(csv_path)
    d = d[d.use_in_training.astype(bool)]
    if drop_doubtful:
        d = d[~d.doubtful.astype(bool)]
    r, c = project(d.lon.values, d.lat.values)
    return np.unique(np.column_stack([np.floor(np.asarray(r)).astype(int), np.floor(np.asarray(c)).astype(int)]), axis=0)


def assemble(base, urban, fold_map, builtup_share, min_dist_cells=2):
    """Survey samples (rows, cols, kind, fold) plus the urban positives (kind 'urban'). Background cells closer than min_dist_cells
    to an urban cell are dropped. 'builtup' = built-up share >= 0.5 at the centre cell (NaN counts as not built-up)."""
    rows, cols, kind = np.asarray(base["rows"]), np.asarray(base["cols"]), np.asarray(base["kind"])
    urban = np.asarray(urban).reshape(-1, 2)
    keep = np.ones(len(rows), bool)
    if len(urban):
        d = np.hypot(rows[:, None] - urban[None, :, 0], cols[:, None] - urban[None, :, 1]).min(axis=1)
        keep = ~((kind == "background") & (d < min_dist_cells))
    rows, cols, kind = rows[keep], cols[keep], kind[keep]
    rows = np.r_[rows, urban[:, 0]]; cols = np.r_[cols, urban[:, 1]]; kind = np.r_[kind, np.array(["urban"] * len(urban))]
    share = np.asarray(builtup_share, dtype=float)[rows, cols]
    return dict(rows=rows, cols=cols, kind=kind, fold=np.asarray(fold_map)[rows, cols], builtup=np.nan_to_num(share, nan=0.0) >= BUILTUP_MIN)


def regime(name, kind, builtup):
    """(training mask, label) for a training regime. S_full: survey flood cells against the full background (Part A / T6g training);
    S, U, S+U: positives against built-up negatives only."""
    kind, builtup = np.asarray(kind), np.asarray(builtup, bool)
    other = (kind == "background") | (kind == "nonflood")
    if name == "S_full":
        pos, neg = kind == "flood", other
    elif name == "S":
        pos, neg = kind == "flood", other & builtup
    elif name == "U":
        pos, neg = kind == "urban", other & builtup
    elif name == "S+U":
        pos, neg = (kind == "flood") | (kind == "urban"), other & builtup
    else:
        raise ValueError(f"unknown regime {name!r}")
    return pos | neg, pos.astype(int)


def task(name, kind, builtup):
    """Evaluation task: mask of the samples used, binary label, and the kind labels that flood_cnn_eval.evaluate expects."""
    kind, builtup = np.asarray(kind), np.asarray(builtup, bool)
    other = (kind == "background") | (kind == "nonflood")
    if name == "urban":
        pos, neg = kind == "urban", other & builtup
    elif name == "survey":
        pos, neg = kind == "flood", other
    else:
        raise ValueError(f"unknown task {name!r}")
    lab = np.where(pos, "flood", kind)
    return dict(mask=pos | neg, y=pos.astype(int), kind=lab)


# ---------------------------------------------------------------- real run (Part B of Amendment 2)
def _project_factory(transform):
    import pyproj
    from rasterio.transform import rowcol

    from src.config import CRS
    tr = pyproj.Transformer.from_crs(4326, CRS, always_xy=True)

    def project(lon, lat):
        x, y = tr.transform(lon, lat)
        return rowcol(transform, x, y)
    return project


def build(drop_doubtful=False, seed=0):
    from src import flood_cnn_baselines as bl
    from src import flood_cnn_data as fd
    from src import flood_model as fm
    d = fd.build(seed)
    s = d["samples"]
    urban = urban_cells("docs/urban_waterlogging_points.csv", _project_factory(d["transform"]), drop_doubtful)
    fold_map = fm.block_folds(d["shape"], fm.BLOCK, 5, seed)
    smp = assemble(s, urban, fold_map, d["stack"][fd.CHANNELS.index("share_built")])
    X, names = bl.features(d["stack"], smp["rows"], smp["cols"], fd.CHANNELS)
    return smp, X, names, urban


def fit_score(smp, X, regimes=REGIMES, seed=0):
    """Nested-tuned logistic / forest / LightGBM plus their soft vote for each regime, scored on EVERY sample by the model of its
    outer fold. Returns {(regime, model): score vector over all samples}."""
    from src import flood_cnn_baselines as bl
    makers = {"logistic": lambda p: bl.logistic_factory(**p)(), "forest": lambda p: bl.forest_factory(seed, **p)(),
              "lightgbm": lambda p: bl.lgbm_factory(seed, **p)()}
    from src import flood_tuning as tu
    out = {}
    for rg in regimes:
        m, y = regime(rg, smp["kind"], smp["builtup"])
        res = {}
        for name, make in makers.items():
            res[name] = tu.nested_family(X[m], y[m], smp["fold"][m], make, tu.GRIDS[name], X_eval=X, folds_eval=smp["fold"])
            out[(rg, name)] = res[name]["outer_eval"]
            print(f"  {rg:6s} {name:9s} done; chosen per outer fold: {[tuple(c.values()) for c in res[name]['chosen']]}", flush=True)
        out[(rg, "soft_vote")] = tu.soft_vote_eval(res, smp["fold"])
    return out


def baselines(smp, X, names):
    out = {}
    for label, col, sign in (("builtup_share_centre", "share_built", 1), ("builtup_share_9x9", "share_built_m9", 1),
                             ("low_hand", "hand", -1), ("near_drain", "log_dist_drain", -1)):
        v = sign * X[:, names.index(col)].astype(float)
        out[("baseline", label)] = np.where(np.isfinite(v), v, np.nanmedian(v))
    return out


def evaluate_all(scores, smp, n_boot=1000):
    from src import flood_cnn_eval as ev
    res = {}
    for key, sc in scores.items():
        res[key] = {}
        for tname in ("urban", "survey"):
            t = task(tname, smp["kind"], smp["builtup"])
            m = t["mask"]
            res[key][tname] = ev.evaluate(sc[m], t["y"][m], t["kind"][m], n_boot=n_boot)
    return res


def select(scores, res, smp, n_boot=1000):
    """Registered rule: highest urban out-of-fold AUC among the fitted candidates; candidates whose paired CI to the best includes zero
    are tied, and a tie goes to the best S+U candidate."""
    from src import flood_cnn_eval as ev
    cands = [k for k in scores if k[0] in REGIMES]
    best = max(cands, key=lambda k: res[k]["urban"]["auc"])
    t = task("urban", smp["kind"], smp["builtup"]); m = t["mask"]
    paired = {k: ev.paired_auc_diff(scores[best][m], scores[k][m], t["y"][m], t["kind"][m], n_boot=n_boot) for k in cands if k != best}
    tied = [k for k, p in paired.items() if p["ci"][0] <= 0 <= p["ci"][1]] + [best]
    su = [k for k in tied if k[0] == "S+U"]
    chosen = max(su, key=lambda k: res[k]["urban"]["auc"]) if su else best
    return best, chosen, paired


def run(n_boot=1000):
    import json
    import pickle

    from src.config import OUTPUTS
    out_dir = OUTPUTS / "flood_cnn"
    smp, X, names, urban = build(False)
    n_pos = {k: int((smp["kind"] == k).sum()) for k in ("flood", "urban", "nonflood", "background")}
    print("samples:", n_pos, "| built-up share of background:", round(float(smp["builtup"][smp["kind"] == "background"].mean()), 2), flush=True)
    print("urban positives per outer fold:", np.bincount(smp["fold"][smp["kind"] == "urban"], minlength=5).tolist(), flush=True)
    scores = {**fit_score(smp, X), **baselines(smp, X, names)}
    res = evaluate_all(scores, smp, n_boot)
    best, chosen, paired = select(scores, res, smp, n_boot)
    for tname in ("urban", "survey"):
        print(f"\nTASK: {tname}", flush=True)
        for k in sorted(scores, key=lambda k: -res[k][tname]["auc"]):
            r = res[k][tname]
            print(f"  {k[0]:9s} {k[1]:20s} AUC {r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]  AP {r['ap']:.3f}", flush=True)
    print(f"\nBEST by urban AUC: {best}   CHOSEN final urban model (tie rule to S+U): {chosen}", flush=True)
    with open(out_dir / "urban_scores.pkl", "wb") as f:
        pickle.dump(dict(scores=scores, kind=smp["kind"], fold=smp["fold"], rows=smp["rows"], cols=smp["cols"], builtup=smp["builtup"]), f)
    (out_dir / "urban_results.json").write_text(json.dumps(dict(
        counts=n_pos, results={f"{k[0]}|{k[1]}": v for k, v in res.items()}, best=list(best), chosen=list(chosen),
        paired_vs_best={f"{k[0]}|{k[1]}": v for k, v in paired.items()}), indent=1))
    return smp, X, names, scores, res, chosen


def sensitivity(chosen_main=None, n_boot=500):
    """Re-run the regimes that use urban cells without the 4 doubtful rows; report the urban AUC next to the main run."""
    import json

    from src.config import OUTPUTS
    smp, X, names, urban = build(True)
    scores = fit_score(smp, X, regimes=("U", "S+U"))
    res = evaluate_all(scores, smp, n_boot)
    main = json.loads((OUTPUTS / "flood_cnn" / "urban_results.json").read_text())["results"]
    print(f"\nSENSITIVITY without the doubtful rows ({len(urban)} urban cells):", flush=True)
    for k in sorted(scores):
        a = res[k]["urban"]["auc"]; b = main[f"{k[0]}|{k[1]}"]["urban"]["auc"]
        print(f"  {k[0]:5s} {k[1]:10s} urban AUC {a:.3f} (main run with all {len(urban) + 4}: {b:.3f}; difference {a - b:+.3f})", flush=True)
    (OUTPUTS / "flood_cnn" / "urban_sensitivity.json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in res.items()}, indent=1))


def choose_c(X, y, folds, grid=(0.01, 0.1, 1.0, 10.0)):
    """Logistic C with the best pooled out-of-fold AUC under spatial CV on the given samples; returns (C, {C: AUC})."""
    from sklearn.metrics import roc_auc_score

    from src import flood_cnn_baselines as bl
    table = {c: float(roc_auc_score(y, bl.cv_scores(X, y, folds, bl.logistic_factory(c)))) for c in grid}
    return max(table, key=table.get), table


def fit_final_and_score():
    """T6j stage 1 for the hotspot second look: the U-logistic model refitted on ALL urban cells against all built-up negatives,
    scored on every non-water city cell. Saves the score map and the built-up mask. Never reads the hotspots."""
    from src import flood_cnn_baselines as bl
    from src import flood_cnn_data as fd
    from src import flood_cnn_r3 as r3
    from src import terrain as tr
    from src.config import OUTPUTS
    smp, X, names, urban = build(False)
    m, y = regime("U", smp["kind"], smp["builtup"])
    c, table = choose_c(X[m], y[m], smp["fold"][m])
    print(f"C by 5-fold spatial CV on all {int(y[m].sum())} urban cells: {c}   (pooled AUC by C: { {k: round(v, 3) for k, v in table.items()} })", flush=True)
    model = bl.logistic_factory(c)()
    model.fit(X[m], y[m])
    d = fd.build(0)
    stack, bi = d["stack"], fd.CHANNELS.index("share_built")
    inp = tr.real_inputs()
    ref = inp["bmc"] & ~inp["water"]
    means = bl.local_means(stack)
    centre = np.asarray(stack, dtype="float32")
    score = r3.score_map(ref, lambda r, cc: model.predict_proba(np.hstack([centre[:, r, cc].T, means[:, r, cc].T]))[:, 1])
    builtup = np.nan_to_num(stack[bi], nan=0.0) >= BUILTUP_MIN
    out = OUTPUTS / "flood_cnn"
    np.save(out / "r3_scores_urban_logistic.npy", score)
    np.save(out / "r3_builtup_mask.npy", builtup)
    print(f"scored {int(np.isfinite(score).sum())} city cells, range {np.nanmin(score):.4f}-{np.nanmax(score):.4f}; built-up cells in the reference: "
          f"{int((ref & builtup).sum())} of {int(ref.sum())}", flush=True)


if __name__ == "__main__":
    import sys

    from src.wardmap import memory_guard
    memory_guard(3.0)
    fit_final_and_score() if "--final" in sys.argv else sensitivity() if "--sensitivity" in sys.argv else run()
