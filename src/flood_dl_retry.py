"""T6k: flood deep-learning retry (protocol in docs/results_log.md): CNN with extra urban positives, transfer learning from the
land-use U-Net encoder, a tuned CNN, and a learning-curve diagnostic. Torch process (no LightGBM)."""
import numpy as np

from src import flood_cnn as fc


def cells_30m_to_10m(rows30, cols30, tf30, tf10):
    x = tf30.c + (np.asarray(cols30) + 0.5) * tf30.a
    y = tf30.f + (np.asarray(rows30) + 0.5) * tf30.e
    return np.floor((y - tf10.f) / tf10.e).astype(int), np.floor((x - tf10.c) / tf10.a).astype(int)


def encoder_features(encoder, X, rows, cols, size=64, batch=64, device="cpu"):
    """Mean of the encoder's last feature map over a size x size patch centred on each (row, col); image reflect-padded."""
    import torch
    h = size // 2
    Xp = np.pad(np.asarray(X, dtype="float32"), ((0, 0), (h, h), (h, h)), mode="reflect")
    encoder.to(device).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            P = np.stack([Xp[:, r:r + size, c:c + size] for r, c in zip(rows[s:s + batch], cols[s:s + batch])])
            f = encoder(torch.as_tensor(P, device=device))[-1]
            out.append(f.mean(dim=(2, 3)).cpu().numpy())
    return np.concatenate(out)


def cv_with_extra(x, y, folds, x_extra, folds_extra, train, predict, seeds=fc.SEEDS):
    """Out-of-fold scores; extra positives join a fold's training set only when they do not lie in the held-out fold."""
    x, y, folds = np.asarray(x), np.asarray(y), np.asarray(folds)
    oof = np.zeros((len(seeds), len(y)))
    for si, s in enumerate(seeds):
        for k in np.unique(folds):
            te = folds == k
            ke = np.asarray(folds_extra) != k
            Xtr = np.concatenate([x[~te], np.asarray(x_extra)[ke]])
            Ytr = np.concatenate([y[~te], np.ones(int(ke.sum()), dtype=y.dtype)])
            oof[si, te] = predict(train(Xtr, Ytr, seed=s), x[te])
    return oof.mean(axis=0)


def subsample_positives(y, train_idx, frac, rng):
    train_idx = np.asarray(train_idx)
    pos, neg = train_idx[np.asarray(y)[train_idx] == 1], train_idx[np.asarray(y)[train_idx] == 0]
    keep = rng.choice(pos, size=int(round(frac * len(pos))), replace=False)
    return np.sort(np.concatenate([keep, neg]))


def crop_centre(x, size):
    if size == x.shape[-1]:
        return x
    h = (x.shape[-1] - size) // 2
    return x[..., h:h + size, h:h + size]


class CNNEstimator:
    """fit / predict_proba wrapper so the CNN can be tuned with flood_tuning.nested_family."""

    def __init__(self, patch=33, dropout=0.3, steps=fc.STEPS, seeds=fc.SEEDS):
        self.patch, self.dropout, self.steps, self.seeds = patch, dropout, steps, tuple(seeds)

    def fit(self, X, y):
        Xc = crop_centre(np.asarray(X, dtype="float32"), self.patch)
        self.models = [fc.train_cnn(Xc, y, seed=s, steps=self.steps, dropout=self.dropout) for s in self.seeds]
        return self

    def predict_proba(self, X):
        Xc = crop_centre(np.asarray(X, dtype="float32"), self.patch)
        p = np.mean([fc.predict(m, Xc) for m in self.models], axis=0)
        return np.column_stack([1 - p, p])


# ---------------------------------------------------------------- real runs
def _load():
    from src import flood_cnn_data as fd
    from src.config import OUTPUTS
    d = fd.build(0)
    s = d["samples"]
    base = np.load(OUTPUTS / "flood_cnn" / "baseline_oof.npz", allow_pickle=True)
    tune = np.load(OUTPUTS / "flood_cnn" / "tuning_scores.npz", allow_pickle=True)
    assert np.array_equal(base["rows"], s["rows"]) and np.array_equal(base["y"], s["y"])
    return d, s, dict(logistic=base["logistic"], soft_vote=tune["soft_vote"])


def _record(name, score, s, refs, extra=None):
    import json

    from src import flood_cnn_eval as ev
    from src.config import OUTPUTS
    r = ev.evaluate(score, s["y"], s["kind"], n_boot=1000)
    r["vs"] = {k: ev.paired_auc_diff(score, v, s["y"], s["kind"], n_boot=1000) for k, v in refs.items()}
    r["meets_rule"] = bool(r["auc"] >= 0.75 and r["auc_ci"][0] >= 0.70)
    if extra:
        r.update(extra)
    f = OUTPUTS / "flood_cnn" / "dl_retry.json"
    allr = json.loads(f.read_text()) if f.exists() else {}
    allr[name] = r
    f.write_text(json.dumps(allr, indent=1, default=float))
    print(f"{name}: AUC {r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]  AP {r['ap']:.3f}  "
          f"minus logistic {r['vs']['logistic']['diff']:+.3f} [{r['vs']['logistic']['ci'][0]:+.3f}, {r['vs']['logistic']['ci'][1]:+.3f}]  "
          f"minus soft vote {r['vs']['soft_vote']['diff']:+.3f}  meets rule: {r['meets_rule']}", flush=True)
    return r


def run_transfer():
    import rasterio
    import torch

    from src import flood_cnn_baselines as bl
    from src import flood_cnn_data as fd
    from src import grid
    from src import landuse as lu
    from src.config import RAW
    d, s, refs = _load()
    st = np.load(lu.OUT / "band_stats.npz")
    Xs = lu.standardise((grid.read_aligned(RAW / "s2_2023.tif") / 10000.0).astype("float32"), st["mean"], st["std"])
    model = lu.build_model(in_ch=10)
    model.load_state_dict(torch.load(lu.OUT / "unet_seed0.pt", map_location="cpu"))
    with rasterio.open(lu.OUT / "landuse_2023_seed0.tif") as r:
        tf10 = r.transform
    r10, c10 = cells_30m_to_10m(s["rows"], s["cols"], d["transform"], tf10)
    deep = encoder_features(model.encoder, Xs, r10, c10, size=64, batch=64, device=lu._device())
    del Xs
    tab, _ = bl.features(d["stack"], s["rows"], s["cols"], fd.CHANNELS)
    F = np.hstack([deep, tab])

    def head():
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))

    oof = bl.cv_scores(F, s["y"], s["fold"], head)
    return _record("B_transfer_unet_encoder", oof, s, refs, dict(n_deep_features=int(deep.shape[1])))


def _patches(d, s):
    from src import flood_cnn_data as fd
    return fd.PatchSource(d["stack"]).get(s["rows"], s["cols"])


def run_more_positives():
    from src import flood_model as fm
    from src import flood_urban as ur
    d, s, refs = _load()
    P = _patches(d, s)
    urban = ur.urban_cells("docs/urban_waterlogging_points.csv", ur._project_factory(d["transform"]))
    from src import flood_cnn_data as fd
    Pu = fd.PatchSource(d["stack"]).get(urban[:, 0], urban[:, 1])
    fu = fm.block_folds(d["shape"], fm.BLOCK, 5, 0)[urban[:, 0], urban[:, 1]]
    oof = cv_with_extra(P, s["y"], s["fold"], Pu, fu, train=lambda X, Y, seed: fc.train_cnn(X, Y, seed=seed), predict=fc.predict)
    return _record("A_cnn_survey_plus_urban_positives", oof, s, refs, dict(n_urban_positives=int(len(urban))))


def run_learning_curve():
    from sklearn.metrics import roc_auc_score
    d, s, refs = _load()
    P = _patches(d, s)
    y, folds = s["y"], s["fold"]
    out = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        oof = np.zeros(len(y))
        for k in np.unique(folds):
            tr = np.flatnonzero(folds != k)
            idx = subsample_positives(y, tr, frac, np.random.default_rng(100 + int(k)))
            m = fc.train_cnn(P[idx], y[idx], seed=0)
            oof[folds == k] = fc.predict(m, P[folds == k])
        out[str(frac)] = float(roc_auc_score(y, oof))
        print(f"learning curve: {frac:.2f} of training flood cells -> pooled AUC {out[str(frac)]:.3f}", flush=True)
    import json

    from src.config import OUTPUTS
    f = OUTPUTS / "flood_cnn" / "dl_retry.json"
    allr = json.loads(f.read_text()) if f.exists() else {}
    allr["C_learning_curve_seed0"] = out
    f.write_text(json.dumps(allr, indent=1, default=float))
    return out


def run_tuned_cnn():
    from src import flood_tuning as tu
    d, s, refs = _load()
    P = _patches(d, s)
    configs = tu.grid({"patch": [17, 33], "dropout": [0.3, 0.5]})
    res = tu.nested_family(P, s["y"], s["fold"], lambda p: CNNEstimator(p["patch"], p["dropout"], fc.STEPS, seeds=(0,)), configs,
                           refit_factory=lambda p: CNNEstimator(p["patch"], p["dropout"], fc.STEPS, seeds=fc.SEEDS))
    return _record("D_tuned_cnn", res["outer"], s, refs, dict(chosen=res["chosen"]))


if __name__ == "__main__":
    import sys

    from src.wardmap import memory_guard
    memory_guard(4.5)
    {"--transfer": run_transfer, "--more-positives": run_more_positives, "--learning-curve": run_learning_curve,
     "--tuned": run_tuned_cnn}[sys.argv[1]]()
