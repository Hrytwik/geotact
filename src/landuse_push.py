"""T13: land-use accuracy push with any method (protocol in docs/results_log.md). Candidate feature sets x learners, scored on the 300
hand labels with the T7 stratified estimator; the headline is NESTED selection (the candidate for each held-out fold is chosen on the
training folds only). No torch here: the spring-2026 ensemble probabilities come from the live pipeline (T12)."""
import json

import numpy as np

GRIDS = {"logistic": [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)],
         "forest": [dict(min_samples_leaf=m, max_features=f) for m in (1, 3, 10) for f in ("sqrt", 0.3)]}
SETS = ("F0", "F1", "F2", "F3", "F4")
CANDIDATES = [(s, l) for s in SETS for l in ("logistic", "forest")]


def dequantise(v):
    """AlphaEarth int8 -> embedding value: sign(v) * (v / 127.5)^2; -128 (no data) -> NaN."""
    v = np.asarray(v)
    d = (np.sign(v) * (v.astype("float32") / 127.5) ** 2).astype("float32")
    return np.where(v == -128, np.nan, d)


def indices(b):
    """NDVI, MNDWI, NDBI from the 10 Sentinel-2 bands (order B02 B03 B04 B05 B06 B07 B08 B8A B11 B12), reflectance scale."""
    b = np.asarray(b, "float32")
    nd = lambda a, c: (a - c) / np.where(np.abs(a + c) > 1e-6, a + c, np.nan)
    return np.stack([nd(b[6], b[2]), nd(b[1], b[8]), nd(b[8], b[6])])


def stratified(strata, correct, W):
    from src import landuse_accuracy as la
    return la.stratified_accuracy(strata, correct, W)[0]


def make_learner(name, params):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=params["C"], class_weight="balanced", max_iter=5000))
    if name == "forest":
        from sklearn.ensemble import RandomForestClassifier
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0, n_jobs=-1, **params))
    raise ValueError(name)


def _oof(X, y, folds, factory):
    out = np.empty(len(y), dtype=y.dtype)
    for k in np.unique(folds):
        te = folds == k
        out[te] = factory().fit(X[~te], y[~te]).predict(X[te])
    return out


def _best_params(X, y, strata, W, folds, learner, grid, make):
    """Hyper-parameters with the best stratified accuracy of leave-one-fold-out predictions over the given folds."""
    scores = [stratified(strata, _oof(X, y, folds, lambda p=p: make(learner, p)) == y, W) for p in grid]
    i = int(np.argmax(scores))
    return grid[i], scores[i]


def nested_select(sets, cands, y, strata, W, folds, grids=GRIDS, make=make_learner):
    """Outer leave-one-fold-out. Inside each outer training set, every candidate gets its best setting by inner CV, and the candidate
    with the best inner accuracy predicts the held-out fold. Also returns each candidate's own outer predictions."""
    y, strata, folds = np.asarray(y), np.asarray(strata), np.asarray(folds)
    oof = np.empty(len(y), dtype=y.dtype)
    per = {f"{s}|{l}": dict(oof=np.empty(len(y), dtype=y.dtype), params=[]) for s, l in cands}
    chosen = []
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        best = None
        for s, l in cands:
            X = np.asarray(sets[s])
            p, acc = _best_params(X[tr], y[tr], strata[tr], W, folds[tr], l, grids[l], make)
            m = make(l, p).fit(X[tr], y[tr])
            per[f"{s}|{l}"]["oof"][te] = m.predict(X[te])
            per[f"{s}|{l}"]["params"].append(p)
            if best is None or acc > best[3]:
                best = (s, l, p, acc)
        s, l, p, _ = best
        oof[te] = per[f"{s}|{l}"]["oof"][te]
        chosen.append((s, l, p))
    for key, d in per.items():
        d["accuracy"] = stratified(strata, d["oof"] == y, W)
    return dict(oof=oof, chosen=chosen, per_candidate=per, accuracy=stratified(strata, oof == y, W))


def margin_from_proba(P):
    """Gap between the largest and second-largest class probability per row (small = unsure)."""
    top2 = np.sort(np.asarray(P, float), axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def pick_uncertain(rows, cols, margin, n, min_px, avoid=None):
    """Up to n pixels in order of smallest margin, each at least min_px from every pick and from every point in `avoid`."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    got = []
    av = np.zeros((0, 2)) if avoid is None else np.asarray(avoid, float).reshape(-1, 2)
    for i in np.argsort(margin, kind="stable"):
        p = np.array([rows[i], cols[i]], float)
        if len(av) and np.hypot(*(av - p).T).min() < min_px:
            continue
        if got and np.hypot(rows[got] - p[0], cols[got] - p[1]).min() < min_px:
            continue
        got.append(int(i))
        if len(got) == n:
            break
    return np.array(got, dtype=int)


# ---------------------------------------------------------------- real data
LIVE_SPRING = "2026_spring"


def layers26():
    from src import landuse_improve as li
    from src.config import OUTPUTS
    d = OUTPUTS / "twin_live" / LIVE_SPRING
    return li._layers(d / "probs_ens.tif", d / "ens.tif")


def raster_stack(name):
    import rasterio

    from src.config import RAW
    with rasterio.open(RAW / name) as r:
        return r.read()


def feature_sets(L23, L26, S26, AEF, rr, cc):
    """F0..F4 at the given pixels (rows rr, cols cc). S26: 10 x H x W reflectance; AEF: 64 x H x W raw int8."""
    from src import landuse_improve as li
    f = lambda L: li.features_from(L["probs"][:, rr, cc], L["esri"][rr, cc], L["wc"][rr, cc], L["nd"][:, rr, cc], L["built"][rr, cc])
    b = S26[:, rr, cc].astype("float32")
    spec = np.vstack([b, indices(b)]).T
    emb = dequantise(AEF[:, rr, cc]).T
    F0, F1 = f(L23), f(L26)
    return {"F0": F0, "F1": F1, "F2": np.hstack([F1, spec]), "F3": np.hstack([F1, spec, emb]), "F4": emb}


def run():
    import pickle

    import rasterio

    from src import landuse_accuracy as la
    from src import landuse_improve as li
    from src.config import OUTPUTS
    out = OUTPUTS / "landuse"
    d, W = la.load_inputs()
    L23, L26 = li._layers(), layers26()
    rr, cc = rasterio.transform.rowcol(L23["tf"], d.x_utm.values, d.y_utm.values)
    rr, cc = np.asarray(rr), np.asarray(cc)
    S26 = (raster_stack("s2_2026_spring.tif") / 10000.0).astype("float32")
    S26[S26 <= 0] = np.nan
    AEF = raster_stack("aef_2025.tif")
    sets = feature_sets(L23, L26, S26, AEF, rr, cc)
    y = np.array([la.CLASSES.index(c) for c in d.ref])
    strata, folds = d.stratum.values, li.point_folds(d.x_utm.values, d.y_utm.values, 4000.0, 5, 0)
    t4c = li.cv_predict(sets["F0"], y, folds, li.correction_model)
    acc_t4c = stratified(strata, t4c == y, W)
    print(f"T4c out-of-fold accuracy recomputed: {100 * acc_t4c:.1f}% (registered 67.8%)", flush=True)
    res = nested_select(sets, CANDIDATES, y, strata, W, folds)
    yref = d.ref.values
    names = lambda a: np.array([la.CLASSES[k] for k in a])
    report = dict(t4c_oof_accuracy=acc_t4c, nested=dict(accuracy=res["accuracy"], chosen=[list(c) for c in res["chosen"]],
                  vs_t4c=la.paired_difference(strata, res["oof"] == y, t4c == y, W, 2000)), candidates={})
    for k, v in res["per_candidate"].items():
        acc, se = la.stratified_accuracy(strata, v["oof"] == y, W)
        report["candidates"][k] = dict(accuracy=acc, ci=[acc - 1.96 * se, acc + 1.96 * se], params=v["params"],
                                       vs_t4c=la.paired_difference(strata, v["oof"] == y, t4c == y, W, 2000))
        print(f"{k:12s} {100 * acc:.1f}% [{100 * (acc - 1.96 * se):.1f}, {100 * (acc + 1.96 * se):.1f}]  vs T4c "
              f"{100 * report['candidates'][k]['vs_t4c']['diff']:+.1f}", flush=True)
    n = report["nested"]
    acc, se = la.stratified_accuracy(strata, res["oof"] == y, W)
    n["ci"] = [acc - 1.96 * se, acc + 1.96 * se]
    n["users"] = la.users_accuracy(strata, names(res["oof"]), yref, W, la.CLASSES)
    n["producers"] = la.producers_accuracy(strata, names(res["oof"]), yref, W, la.CLASSES)
    n["replace"] = bool(n["accuracy"] - acc_t4c >= 0.03 and n["vs_t4c"]["ci"][0] > 0)
    print(f"NESTED (primary): {100 * acc:.1f}% [{100 * n['ci'][0]:.1f}, {100 * n['ci'][1]:.1f}]  vs T4c {100 * n['vs_t4c']['diff']:+.1f} "
          f"[{100 * n['vs_t4c']['ci'][0]:+.1f}, {100 * n['vs_t4c']['ci'][1]:+.1f}]  chosen per fold {n['chosen']}  REPLACE: {n['replace']}", flush=True)
    if n["replace"]:
        from collections import Counter
        cnt = Counter((s, l) for s, l, _ in res["chosen"])
        top = max(cnt.values())
        tied = [c for c in cnt if cnt[c] == top]
        s, l = max(tied, key=lambda c: report["candidates"][f"{c[0]}|{c[1]}"]["accuracy"])
        p, _ = _best_params(sets[s], y, strata, W, folds, l, GRIDS[l], make_learner)
        model = make_learner(l, p).fit(sets[s], y)
        n["final"] = dict(set=s, learner=l, params=p)
        with open(out / "t13_model.pkl", "wb") as fh:
            pickle.dump(dict(model=model, set=s, learner=l, params=p), fh)
        apply_full(model, s, L23, L26, S26, AEF, out / "landuse_2026_t13.tif")
    (out / "t13_scores.json").write_text(json.dumps(report, indent=1, default=float))
    return report


def apply_full(model, set_name, L23, L26, S26, AEF, path, rows_per_chunk=100):
    from src import landuse_improve as li
    H, Wd = L23["esri"].shape
    cls = np.zeros((H, Wd), "uint8")
    margin = np.zeros((H, Wd), "float32")
    for r0 in range(0, H, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, H)
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(Wd), indexing="ij")
        F = feature_sets(L23, L26, S26, AEF, rr.ravel(), cc.ravel())[set_name]
        P = model.predict_proba(F)
        cls[r0:r1] = model.classes_[P.argmax(1)].reshape(r1 - r0, Wd) if hasattr(model, "classes_") else P.argmax(1).reshape(r1 - r0, Wd)
        margin[r0:r1] = margin_from_proba(P).reshape(r1 - r0, Wd)
    from src import landuse as lu
    li._write(path, cls, lu.OUT / "landuse_2023_seed0.tif", "uint8")
    li._write(path.with_name(path.stem + "_margin.tif"), np.round(margin * 100), lu.OUT / "landuse_2023_seed0.tif", "uint8")


def t4c_margin(path, chunk=100):
    """Confidence margin (percent) of the CURRENT product's correction model on every pixel of the 2023 features."""
    import pickle

    from src import landuse as lu
    from src import landuse_improve as li
    from src.config import HOME_OUTPUTS
    with open(HOME_OUTPUTS / "landuse" / "correction_model.pkl", "rb") as fh:
        model = pickle.load(fh)
    L = li._layers()
    H, Wd = L["esri"].shape
    m = np.zeros((H, Wd), "float32")
    for r0 in range(0, H, chunk):
        r1 = min(r0 + chunk, H)
        sl = (slice(r0, r1), slice(None))
        F = li.features_from(L["probs"][:, r0:r1].reshape(6, -1), L["esri"][sl].ravel(), L["wc"][sl].ravel(), L["nd"][:, r0:r1].reshape(8, -1), L["built"][sl].ravel())
        m[r0:r1] = margin_from_proba(model.predict_proba(F)).reshape(r1 - r0, Wd)
    li._write(path, np.round(m * 100), lu.OUT / "landuse_2023_seed0.tif", "uint8")


def active_sheet(margin_path, n=100, min_m=200.0, sample=300_000, seed=2028):
    """TRAINING-ONLY extra points for the authors: the n least-sure pixels (smallest top-two probability gap) of a random pixel sample,
    at least min_m apart and min_m from the 300 accuracy points; written as KML + sheet in validation/active_learning/."""
    import rasterio
    import simplekml
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from pyproj import Transformer

    from src import landuse_accuracy as la
    from src.config import CRS, ROOT
    with rasterio.open(margin_path) as r:
        m, tf = r.read(1).astype("float32"), r.transform
    rng = np.random.default_rng(seed)
    flat = rng.choice(m.size, size=min(sample, m.size), replace=False)
    rows, cols = np.unravel_index(flat, m.shape)
    d, _ = la.load_inputs()
    ar, ac = rasterio.transform.rowcol(tf, d.x_utm.values, d.y_utm.values)
    pick = pick_uncertain(rows, cols, m[rows, cols], n, min_m / abs(tf.a), np.column_stack([ar, ac]))
    to_wgs = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    out = ROOT / "validation" / "active_learning"
    out.mkdir(parents=True, exist_ok=True)
    kml, wb = simplekml.Kml(name="Extra land-use training points (not accuracy)"), Workbook()
    ws = wb.active; ws.title = "labels"
    ws.append(["id", "lat", "lon", "label", "confidence", "imagery_date", "notes"])
    for i, k in enumerate(pick, 1):
        x, y = tf * (cols[k] + 0.5, rows[k] + 0.5)
        lon, lat = to_wgs.transform(x, y)
        pid = f"A{i:03d}"
        kml.newpoint(name=pid, coords=[(lon, lat)]).style.iconstyle.scale = 0.6
        sq = [to_wgs.transform(x + dx, y + dy) for dx, dy in [(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)]]
        pol = kml.newpolygon(name=f"{pid} pixel", outerboundaryis=sq)
        pol.style.polystyle.fill = 0; pol.style.linestyle.color = simplekml.Color.yellow; pol.style.linestyle.width = 2
        ws.append([pid, round(lat, 6), round(lon, 6), None, None, None, None])
    for col, opts in (("D", ",".join(la.CLASSES)), ("E", "Sure,Unsure")):
        dv = DataValidation(type="list", formula1='"' + opts + '"', allow_blank=True); ws.add_data_validation(dv); dv.add(f"{col}2:{col}{len(pick) + 1}")
    g = wb.create_sheet("how_to_label")
    for line in ["Same rules as the 300 accuracy points (validation/validation_labels.xlsx, how_to_label tab).",
                 "These points are where the map is least sure. They are used ONLY to train the map, never to measure its accuracy.",
                 "Use the most recent Google Earth imagery and write its date."]:
        g.append([line])
    kml.save(str(out / "active_points.kml")); wb.save(out / "active_labels.xlsx")
    return len(pick)


if __name__ == "__main__":
    import sys

    from src.config import OUTPUTS
    from src.wardmap import memory_guard
    memory_guard(5.0)
    if "--active" in sys.argv:
        mp = OUTPUTS / "landuse" / "landuse_2023_corr_margin.tif"
        if not mp.exists():
            t4c_margin(mp)
        print("active-learning points written:", active_sheet(mp), flush=True)
    else:
        run()
