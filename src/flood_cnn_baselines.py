"""T6g baselines for the deep flood model (protocol in docs/build_plan.md): LightGBM and logistic regression on centre-cell
features plus 9 x 9 means, single-layer scores, and the frozen T6d susceptibility. Own process: no torch."""
import numpy as np
from scipy import ndimage as ndi

WINDOW = 9
# direction in which each layer is read as "more flood-prone" (protocol item d)
SINGLE_LAYERS = {"hand": -1, "log_dist_drain": -1, "local_relief": -1, "twi": +1}


def local_means(stack, size=WINDOW):
    """Mean over a size x size window of the finite values only, per channel; NaN where the whole window is NaN."""
    stack = np.asarray(stack, dtype="float64")
    fin = np.isfinite(stack)
    out = np.empty(stack.shape, dtype="float32")
    for c in range(stack.shape[0]):
        num = ndi.uniform_filter(np.where(fin[c], stack[c], 0.0), size=size, mode="reflect")
        den = ndi.uniform_filter(fin[c].astype("float64"), size=size, mode="reflect")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[c] = np.where(den > 1e-9, num / den, np.nan)
    return out


def features(stack, rows, cols, channels, size=WINDOW):
    """Centre-cell values then size x size window means for each (row, col); names follow the column order."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    centre = np.asarray(stack, dtype="float32")[:, rows, cols].T
    means = local_means(stack, size)[:, rows, cols].T
    return np.hstack([centre, means]), list(channels) + [f"{c}_m{size}" for c in channels]


def single_layer_scores(stack, rows, cols, channels, signs=SINGLE_LAYERS):
    """One score per layer (signed so that higher = more flood-prone); NaN filled with the layer's median over the samples."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    out = {}
    for name, sign in signs.items():
        v = sign * np.asarray(stack, dtype="float64")[list(channels).index(name), rows, cols]
        out[name] = np.where(np.isfinite(v), v, np.nanmedian(v))
    return out


def logistic_factory(C=1.0):
    def make():
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
    return make


def forest_factory(seed=0):
    def make():
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                    n_jobs=4, random_state=seed))
    return make


def lgbm_factory(seed=0):
    def make():
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=20.0, n_jobs=4,
                                  verbose=-1, random_state=seed)
    return make


def cv_scores(X, y, folds, make_model):
    """Out-of-fold probability for every sample: each fold is scored by a model that never saw it."""
    X, y, folds = np.asarray(X), np.asarray(y), np.asarray(folds)
    oof = np.full(len(y), np.nan)
    for k in np.unique(folds):
        te = folds == k
        model = make_model()
        model.fit(X[~te], y[~te])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return oof


# ---------------------------------------------------------------- real run (own process: LightGBM, no torch)
def run(seed=0, n_boot=1000):
    import json

    import rasterio

    from src import flood_cnn_data as fd
    from src import flood_cnn_eval as ev
    from src.config import OUTPUTS

    out_dir = OUTPUTS / "flood_cnn"
    out_dir.mkdir(exist_ok=True)
    d = fd.build(seed)
    s, stack = d["samples"], d["stack"]
    rows, cols, y, kind, folds = s["rows"], s["cols"], s["y"], s["kind"], s["fold"]
    X, names = features(stack, rows, cols, fd.CHANNELS)
    print(f"samples {len(y)}  flood {int(y.sum())}  features {X.shape[1]}  flood per fold "
          f"{[int(((folds == k) & (y == 1)).sum()) for k in np.unique(folds)]}", flush=True)

    scores = {"lightgbm": cv_scores(X, y, folds, lgbm_factory(seed)), "logistic": cv_scores(X, y, folds, logistic_factory()),
              "forest": cv_scores(X, y, folds, forest_factory(seed))}
    for name, v in single_layer_scores(stack, rows, cols, fd.CHANNELS).items():
        scores[f"layer_{name}"] = v
    with rasterio.open(OUTPUTS / "flood" / "susceptibility_30m.tif") as r:
        t6d = r.read(1)[rows, cols]
    print(f"frozen T6d susceptibility: {int(np.isnan(t6d).sum())} of {len(t6d)} samples are NaN (filled with the minimum)", flush=True)
    scores["frozen_t6d"] = np.where(np.isfinite(t6d), t6d, np.nanmin(t6d))

    res = {k: ev.evaluate(v, y, kind, n_boot=n_boot, seed=seed) for k, v in scores.items()}
    for k, r in res.items():
        print(f"{k:22s} AUC {r['auc']:.3f} {r['auc_ci'][0]:.3f}-{r['auc_ci'][1]:.3f}  AP {r['ap']:.3f}  AUC vs field non-flood {r['auc_nonflood']:.3f}", flush=True)
    np.savez_compressed(out_dir / "baseline_oof.npz", y=y, kind=kind, fold=folds, rows=rows, cols=cols, **scores)
    (out_dir / "baselines.json").write_text(json.dumps(dict(seed=seed, n_boot=n_boot, models=res), indent=1))
    return res




# ---------------------------------------------------------------- T6h stage 1: refit on ALL field points, save score maps
def fit_all_and_score(seed=0):
    """Fit logistic, forest and LightGBM on all samples and score every non-water city cell. Never reads the hotspots."""
    from src import flood_cnn_data as fd
    from src import flood_cnn_r3 as r3
    from src.config import OUTPUTS

    d = fd.build(seed)
    s, stack = d["samples"], d["stack"]
    ref = d["bmc"] & ~d["water"]
    X, _ = features(stack, s["rows"], s["cols"], fd.CHANNELS)
    means = local_means(stack)
    centre = np.asarray(stack, dtype="float32")

    def cell_features(rows, cols):
        return np.hstack([centre[:, rows, cols].T, means[:, rows, cols].T])

    for name, make in (("logistic", logistic_factory()), ("forest", forest_factory(seed)), ("lightgbm", lgbm_factory(seed))):
        model = make()
        model.fit(X, s["y"])
        m = r3.score_map(ref, lambda r, c: model.predict_proba(cell_features(r, c))[:, 1])
        np.save(OUTPUTS / "flood_cnn" / f"r3_scores_{name}.npy", m)
        print(f"{name}: scored {int(np.isfinite(m).sum())} cells, score range {np.nanmin(m):.4f}-{np.nanmax(m):.4f}", flush=True)


if __name__ == "__main__":
    import sys

    from src.wardmap import memory_guard
    memory_guard(3.0)
    fit_all_and_score() if "--fit-all" in sys.argv else run()
