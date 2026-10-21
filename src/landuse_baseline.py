"""Per-pixel LightGBM baseline for T4: does the U-Net's spatial context beat a classifier that sees one pixel?

Runs in its OWN process and must never import torch: LightGBM and PyTorch each ship a libomp, and loading both in
one macOS process crashed the seed-0 run with SIGSEGV inside libomp (tests/test_landuse.py regression tests).
"""
import json

import numpy as np

from src import landuse as lu


def run(n_train=400_000, seed=0):
    import lightgbm as lgb
    d = lu.load_inputs("s2_2023", 2023)
    X, Y = d["X"], d["Y"]
    train_w, val_w = lu.split_patches(Y.shape, 128, 64, 512, 0.25, seed=0)
    mean, std = lu.band_stats(X, train_w)
    Xs = lu.standardise(X, mean, std)
    cw = lu.class_weights(Y, train_w)
    trn, val = lu._val_canvas(Y.shape, train_w), lu._val_canvas(Y.shape, val_w)
    feats = Xs.reshape(Xs.shape[0], -1)
    flat = Y.ravel()
    ti = np.flatnonzero((trn & (Y != lu.IGNORE)).ravel())
    ti = np.random.default_rng(seed).choice(ti, size=min(n_train, ti.size), replace=False)
    clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.1, n_jobs=4, verbose=-1, random_state=seed)
    clf.fit(feats[:, ti].T, flat[ti], sample_weight=cw[flat[ti]])
    vi = np.flatnonzero((val & (Y != lu.IGNORE)).ravel())
    c = lu.confusion(clf.predict(feats[:, vi].T), flat[vi])
    res = dict(confusion=c.tolist(), **lu._jsonable(lu.metrics(c)), n_train_pixels=int(ti.size), classes=lu.CLASSES)
    (lu.OUT / "metrics_lightgbm_baseline.json").write_text(json.dumps(res, indent=1))
    print(f"[baseline] per-pixel LightGBM | validation OA {res['overall_accuracy']:.4f} macro-F1 {res['macro_f1']:.4f}", flush=True)
    return res


if __name__ == "__main__":
    run()
