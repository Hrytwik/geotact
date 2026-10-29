"""T6i: nested hyper-parameter search, soft vote and stacking for the flood models (protocol: Amendment 2 in docs/results_log.md).
No torch here. For every outer spatial fold the parameters are chosen by an inner spatial CV on the remaining folds only; the outer
fold is scored once, by a model refitted on all remaining folds."""
import itertools

import numpy as np

from src.flood_cnn_baselines import cv_scores


def grid(options):
    keys = list(options)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(options[k] for k in keys))]


def percentile_map(train_scores):
    """Function giving the mid-rank percentile (0-1) of a score among `train_scores`; saturates at 0 and 1 outside their range."""
    s = np.sort(np.asarray(train_scores, dtype=float))

    def f(x):
        x = np.asarray(x, dtype=float)
        return (np.searchsorted(s, x, side="left") + np.searchsorted(s, x, side="right")) / (2.0 * len(s))
    return f


def nested_family(X, y, folds, factory, configs, refit_factory=None, X_eval=None, folds_eval=None):
    """Nested search for one model family. `factory(params)` returns a fresh estimator (fit / predict_proba). Returns the outer
    out-of-fold score of every sample, the inner out-of-fold scores of each outer training set (for ensembling), the chosen
    parameters and every config's inner AUC."""
    from sklearn.metrics import roc_auc_score
    X, y, folds = np.asarray(X), np.asarray(y), np.asarray(folds)
    refit_factory = refit_factory or factory
    outer = np.full(len(y), np.nan)
    outer_eval = None if X_eval is None else np.full(len(X_eval), np.nan)
    inner_oof, chosen, inner_auc = [], [], []
    for k in np.unique(folds):
        tr = folds != k
        scored = []
        for params in configs:
            oof = cv_scores(X[tr], y[tr], folds[tr], lambda p=params: factory(p))
            scored.append((float(roc_auc_score(y[tr], oof)), oof))
        best = int(np.argmax([a for a, _ in scored]))
        inner_oof.append(scored[best][1]); chosen.append(configs[best]); inner_auc.append([a for a, _ in scored])
        model = refit_factory(configs[best])
        model.fit(X[tr], y[tr])
        outer[~tr] = model.predict_proba(X[~tr])[:, 1]
        if outer_eval is not None:                        # also score a different sample set that lies in the same outer fold
            te_e = np.asarray(folds_eval) == k
            outer_eval[te_e] = model.predict_proba(np.asarray(X_eval)[te_e])[:, 1]
    return dict(outer=outer, inner_oof=inner_oof, chosen=chosen, inner_auc=inner_auc, outer_eval=outer_eval)


def soft_vote(results, folds):
    """Mean over models of each model's outer score mapped to its percentile among that model's inner out-of-fold training scores."""
    folds = np.asarray(folds)
    out = np.full(len(folds), np.nan)
    for i, k in enumerate(np.unique(folds)):
        te = folds == k
        out[te] = np.mean([percentile_map(r["inner_oof"][i])(r["outer"][te]) for r in results.values()], axis=0)
    return out


def soft_vote_eval(results, folds_eval):
    """soft_vote for the extra evaluation set scored by nested_family(X_eval=...): the same percentile mapping per outer fold."""
    folds_eval = np.asarray(folds_eval)
    out = np.full(len(folds_eval), np.nan)
    for i, k in enumerate(np.unique(folds_eval)):
        te = folds_eval == k
        out[te] = np.mean([percentile_map(r["inner_oof"][i])(r["outer_eval"][te]) for r in results.values()], axis=0)
    return out


def stack(results, y, folds):
    """Logistic regression (balanced) on the models' percentile scores, fitted on the inner out-of-fold scores of each outer
    training set and applied to the outer fold."""
    from sklearn.linear_model import LogisticRegression
    y, folds = np.asarray(y), np.asarray(folds)
    out = np.full(len(folds), np.nan)
    for i, k in enumerate(np.unique(folds)):
        tr, te = folds != k, folds == k
        maps = {n: percentile_map(r["inner_oof"][i]) for n, r in results.items()}
        Xtr = np.column_stack([maps[n](r["inner_oof"][i]) for n, r in results.items()])
        Xte = np.column_stack([maps[n](r["outer"][te]) for n, r in results.items()])
        lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xtr, y[tr])
        out[te] = lr.predict_proba(Xte)[:, 1]
    return out


# ---------------------------------------------------------------- real run (Part A of Amendment 2)
GRIDS = {
    "logistic": grid({"C": [0.01, 0.1, 1.0, 10.0]}),
    "forest": grid({"min_samples_leaf": [5, 20], "max_features": ["sqrt", 0.5]}),
    "lightgbm": grid({"num_leaves": [7, 15], "learning_rate": [0.03, 0.05], "n_estimators": [200, 400]}),
    "mlp": grid({"width": [32, 64], "dropout": [0.3, 0.5], "steps": [600, 1500]}),
}
OUT_NAME = "tuning_{}.pkl"


def load_part_a():
    """The T6g samples, folds and 32 features, checked against the saved T6g baseline scores."""
    from src import flood_cnn_baselines as bl
    from src import flood_cnn_data as fd
    from src.config import OUTPUTS
    d = fd.build(0)
    s = d["samples"]
    base = np.load(OUTPUTS / "flood_cnn" / "baseline_oof.npz", allow_pickle=True)
    assert np.array_equal(base["rows"], s["rows"]) and np.array_equal(base["y"], s["y"]), "samples differ from the T6g baseline run"
    X, _ = bl.features(d["stack"], s["rows"], s["cols"], fd.CHANNELS)
    return X, s["y"], s["kind"], s["fold"], base


def _save(name, res):
    import pickle

    from src.config import OUTPUTS
    with open(OUTPUTS / "flood_cnn" / OUT_NAME.format(name), "wb") as f:
        pickle.dump(res, f)


def run_tabular(seed=0):
    from sklearn.metrics import roc_auc_score

    from src import flood_cnn_baselines as bl
    X, y, kind, folds, _ = load_part_a()
    makers = {"logistic": lambda p: bl.logistic_factory(**p)(),
              "forest": lambda p: bl.forest_factory(seed, **p)(),
              "lightgbm": lambda p: bl.lgbm_factory(seed, **p)()}
    for name, make in makers.items():
        res = nested_family(X, y, folds, make, GRIDS[name])
        _save(name, res)
        print(f"{name:9s} outer pooled AUC {roc_auc_score(y, res['outer']):.3f}  chosen per outer fold: {res['chosen']}", flush=True)


def combine(n_boot=1000):
    import json
    import pickle

    from src import flood_cnn_eval as ev
    from src.config import OUTPUTS
    from src.flood_cnn_r3 import holm
    X, y, kind, folds, base = load_part_a()
    results = {}
    for name in GRIDS:
        with open(OUTPUTS / "flood_cnn" / OUT_NAME.format(name), "rb") as f:
            results[name] = pickle.load(f)
    cands = {name: r["outer"] for name, r in results.items()}
    cands["soft_vote (PRIMARY)"] = soft_vote(results, folds)
    cands["stacking"] = stack(results, y, folds)
    ref = base["logistic"]
    out = {}
    for k, v in cands.items():
        out[k] = dict(**ev.evaluate(v, y, kind, n_boot=n_boot), vs_t6g_logistic=ev.paired_auc_diff(v, ref, y, kind, n_boot=n_boot))
    for k, ph in zip(out, holm([out[k]["vs_t6g_logistic"]["p"] for k in out])):
        out[k]["vs_t6g_logistic"]["p_holm"] = ph
    out["t6g_untuned_logistic"] = ev.evaluate(ref, y, kind, n_boot=n_boot)
    out["_chosen"] = {n: r["chosen"] for n, r in results.items()}
    (OUTPUTS / "flood_cnn" / "tuning_result.json").write_text(json.dumps(out, indent=1, default=str))
    np.savez_compressed(OUTPUTS / "flood_cnn" / "tuning_scores.npz", y=y, kind=kind, fold=folds, **{k.split(" ")[0]: v for k, v in cands.items()})
    r0 = out["t6g_untuned_logistic"]
    print(f"{'t6g untuned logistic':22s} AUC {r0['auc']:.3f} [{r0['auc_ci'][0]:.3f}, {r0['auc_ci'][1]:.3f}]  AP {r0['ap']:.3f}")
    for k in cands:
        r, d = out[k], out[k]["vs_t6g_logistic"]
        print(f"{k:22s} AUC {r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]  AP {r['ap']:.3f}  vs field non-flood {r['auc_nonflood']:.3f}  "
              f"minus T6g logistic {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] p {d['p']:.3f} Holm {d['p_holm']:.3f}")
    return out


if __name__ == "__main__":
    import sys

    from src.wardmap import memory_guard
    memory_guard(3.0)
    run_tabular() if "--tabular" in sys.argv else combine()
