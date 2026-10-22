"""T6i: nested hyper-parameter search, soft vote and stacking for the flood models (protocol: Amendment 2 in docs/build_plan.md).
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


def nested_family(X, y, folds, factory, configs, refit_factory=None):
    """Nested search for one model family. `factory(params)` returns a fresh estimator (fit / predict_proba). Returns the outer
    out-of-fold score of every sample, the inner out-of-fold scores of each outer training set (for ensembling), the chosen
    parameters and every config's inner AUC."""
    from sklearn.metrics import roc_auc_score
    X, y, folds = np.asarray(X), np.asarray(y), np.asarray(folds)
    refit_factory = refit_factory or factory
    outer = np.full(len(y), np.nan)
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
    return dict(outer=outer, inner_oof=inner_oof, chosen=chosen, inner_auc=inner_auc)


def soft_vote(results, folds):
    """Mean over models of each model's outer score mapped to its percentile among that model's inner out-of-fold training scores."""
    folds = np.asarray(folds)
    out = np.full(len(folds), np.nan)
    for i, k in enumerate(np.unique(folds)):
        te = folds == k
        out[te] = np.mean([percentile_map(r["inner_oof"][i])(r["outer"][te]) for r in results.values()], axis=0)
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
