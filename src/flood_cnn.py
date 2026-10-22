"""T6g: the deep flood models (a patch CNN and a tabular MLP) with the training settings registered in docs/build_plan.md
(Amendment 1b). torch, CPU only. Own process: LightGBM and the other baselines live in flood_cnn_baselines.py."""
import numpy as np
import torch
from torch import nn

STEPS = 1500
BATCH = 64
LR, WEIGHT_DECAY = 1e-3, 1e-2
SEEDS = (0, 1, 2)
torch.set_num_threads(4)


# ---------------------------------------------------------------- normalisation
def fit_zscore(x, chunk=1000):
    """Per-channel mean and std over the finite pixels of x (N, C, H, W); a constant channel gets std 1."""
    c = x.shape[1]
    n, s, s2 = np.zeros(c), np.zeros(c), np.zeros(c)
    for i in range(0, len(x), chunk):
        b = x[i:i + chunk].astype("float64")
        fin = np.isfinite(b)
        b = np.where(fin, b, 0.0)
        n += fin.sum(axis=(0, 2, 3)); s += b.sum(axis=(0, 2, 3)); s2 += (b ** 2).sum(axis=(0, 2, 3))
    mean = s / np.maximum(n, 1)
    std = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0.0))
    return mean.astype("float32"), np.where(std < 1e-6, 1.0, std).astype("float32")


def apply_zscore(x, mean, std):
    z = (np.asarray(x, dtype="float32") - mean[None, :, None, None]) / std[None, :, None, None]
    return np.nan_to_num(z, nan=0.0)


# ---------------------------------------------------------------- models
def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))


class FloodCNN(nn.Module):
    """Three conv blocks (32-64-64), global average pooling, dropout 0.3, one logit. Normalises its own input."""

    def __init__(self, channels, mean=None, std=None):
        super().__init__()
        m = np.zeros(channels, "float32") if mean is None else mean
        s = np.ones(channels, "float32") if std is None else std
        self.register_buffer("mean", torch.as_tensor(m).view(1, -1, 1, 1))
        self.register_buffer("std", torch.as_tensor(s).view(1, -1, 1, 1))
        self.features = nn.Sequential(_block(channels, 32), _block(32, 64), _block(64, 64))
        self.drop, self.out = nn.Dropout(0.3), nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nan_to_num((x - self.mean) / self.std, nan=0.0)
        return self.out(self.drop(self.features(x).mean(dim=(2, 3)))).squeeze(-1)


class TabMLP(nn.Module):
    """Two hidden layers of 64, ReLU, dropout 0.3. Median-imputes and z-scores its own input with training statistics."""

    def __init__(self, d, median=None, mean=None, std=None, width=64, dropout=0.3):
        super().__init__()
        for name, v, default in (("median", median, 0.0), ("mean", mean, 0.0), ("std", std, 1.0)):
            self.register_buffer(name, torch.as_tensor(np.full(d, default, "float32") if v is None else v))
        self.net = nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, width), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, x):
        x = torch.where(torch.isnan(x), self.median, x)
        return self.net((x - self.mean) / self.std).squeeze(-1)


# ---------------------------------------------------------------- training
def augment(x, rng):
    """Random horizontal flip and a random multiple of 90-degree rotation per patch (pixel values are only moved)."""
    out = np.empty_like(x)
    k, flip = rng.integers(0, 4, size=len(x)), rng.integers(0, 2, size=len(x))
    for i in range(len(x)):
        p = np.rot90(x[i], k=int(k[i]), axes=(1, 2))
        out[i] = p[:, :, ::-1] if flip[i] else p
    return out


def balanced_batch(y, batch, rng):
    """Indices for one batch: half flood (drawn with replacement, there are few), half other."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    half = batch // 2
    return np.concatenate([rng.choice(pos, size=half, replace=True), rng.choice(neg, size=batch - half, replace=len(neg) < batch - half)])


def _fit(model, get_batch, y, seed, steps, batch):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(steps):
        idx = balanced_batch(y, batch, rng)
        xb = torch.as_tensor(get_batch(idx, rng))
        loss = loss_fn(model(xb), torch.as_tensor(y[idx], dtype=torch.float32))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    return model


def train_cnn(x, y, seed=0, steps=STEPS, batch=BATCH):
    x, y = np.asarray(x, dtype="float32"), np.asarray(y)
    mean, std = fit_zscore(x)
    torch.manual_seed(seed)                                                # before construction: seeds the initial weights
    return _fit(FloodCNN(x.shape[1], mean, std), lambda idx, rng: augment(x[idx], rng), y, seed, steps, batch)


def train_mlp(X, y, seed=0, steps=STEPS, batch=BATCH, width=64, dropout=0.3):
    X, y = np.asarray(X, dtype="float32"), np.asarray(y)
    median = np.nanmedian(X, axis=0)
    filled = np.where(np.isnan(X), median, X)
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    torch.manual_seed(seed)
    return _fit(TabMLP(X.shape[1], median, mean, np.where(std < 1e-6, 1.0, std), width, dropout), lambda idx, rng: X[idx], y, seed,
                steps, batch)


class MLPEstimator:
    """fit / predict_proba wrapper (mean over seeds) so the tabular MLP can be tuned like the other families."""

    def __init__(self, width=64, dropout=0.3, steps=STEPS, seeds=SEEDS):
        self.width, self.dropout, self.steps, self.seeds = width, dropout, steps, tuple(seeds)

    def fit(self, X, y):
        self.models = [train_mlp(X, y, seed=sd, steps=self.steps, width=self.width, dropout=self.dropout) for sd in self.seeds]
        return self

    def predict_proba(self, X):
        p = np.mean([_predict(m, X) for m in self.models], axis=0)
        return np.column_stack([1 - p, p])


def _predict(model, x, batch=512):
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(torch.as_tensor(np.asarray(x[i:i + batch], dtype="float32")))).numpy()
                               for i in range(0, len(x), batch)])


predict = predict_tab = _predict


# ---------------------------------------------------------------- spatial cross-validation
def _cv(x, y, folds, seeds, train, steps, batch, return_per_seed):
    x, y, folds = np.asarray(x), np.asarray(y), np.asarray(folds)
    per_seed = np.full((len(seeds), len(y)), np.nan)
    for si, seed in enumerate(seeds):
        for k in np.unique(folds):
            te = folds == k
            model = train(x[~te], y[~te], seed=seed, steps=steps, batch=batch)
            per_seed[si, te] = _predict(model, x[te])
    return per_seed if return_per_seed else per_seed.mean(axis=0)


def cv_cnn(x, y, folds, seeds=SEEDS, steps=STEPS, batch=BATCH, return_per_seed=False):
    """Out-of-fold CNN probability for every sample, averaged over seeds; each fold is scored by a model trained without it."""
    return _cv(x, y, folds, seeds, train_cnn, steps, batch, return_per_seed)


def cv_mlp(X, y, folds, seeds=SEEDS, steps=STEPS, batch=BATCH, return_per_seed=False):
    return _cv(X, y, folds, seeds, train_mlp, steps, batch, return_per_seed)


# ---------------------------------------------------------------- real run
def run(steps=STEPS, seeds=SEEDS, n_boot=1000, tag="benchmark"):
    import json

    from src import flood_cnn_baselines as bl
    from src import flood_cnn_data as fd
    from src import flood_cnn_eval as ev
    from src.config import OUTPUTS

    out_dir = OUTPUTS / "flood_cnn"
    d = fd.build(0)
    s = d["samples"]
    rows, cols, y, kind, folds = s["rows"], s["cols"], s["y"], s["kind"], s["fold"]
    base = np.load(out_dir / "baseline_oof.npz", allow_pickle=True)
    assert np.array_equal(base["rows"], rows) and np.array_equal(base["cols"], cols) and np.array_equal(base["y"], y), \
        "baseline scores are not aligned with these samples; rerun python -m src.flood_cnn_baselines"
    X, _ = bl.features(d["stack"], rows, cols, fd.CHANNELS)
    patches = fd.PatchSource(d["stack"]).get(rows, cols)
    del d
    print(f"patches {patches.shape}  {patches.nbytes / 1e6:.0f} MB  steps {steps}  seeds {tuple(seeds)}", flush=True)

    scores = {k: base[k] for k in base.files if k not in ("y", "kind", "fold", "rows", "cols")}
    per_seed_auc = {}
    for name, fn, data in (("mlp", cv_mlp, X), ("cnn", cv_cnn, patches)):
        ps = fn(data, y, folds, seeds=seeds, steps=steps, return_per_seed=True)
        scores[name] = ps.mean(axis=0)
        per_seed_auc[name] = [float(ev.evaluate(p, y, kind, n_boot=0)["auc"]) for p in ps]
        print(f"{name}: per-seed pooled AUC {np.round(per_seed_auc[name], 3).tolist()}", flush=True)
        np.save(out_dir / f"{tag}_{name}_per_seed_oof.npy", ps)

    res = {k: ev.evaluate(v, y, kind, n_boot=n_boot) for k, v in scores.items()}
    order = sorted(res, key=lambda k: -res[k]["auc"])
    paired = {k: ev.paired_auc_diff(scores["cnn"], scores[k], y, kind, n_boot=n_boot) for k in order if k != "cnn"}
    print("rank by pooled spatial-CV AUC (diff = CNN minus model, paired 95% CI):", flush=True)
    for k in order:
        r = res[k]
        extra = "" if k == "cnn" else f"  CNN-minus: {paired[k]['diff']:+.3f} [{paired[k]['ci'][0]:+.3f}, {paired[k]['ci'][1]:+.3f}]"
        print(f"  {k:22s} AUC {r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]  AP {r['ap']:.3f}  vs field non-flood {r['auc_nonflood']:.3f}{extra}", flush=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(dict(steps=steps, seeds=list(seeds), n_boot=n_boot, ranking=order, models=res,
                                                         cnn_minus=paired, per_seed_auc=per_seed_auc), indent=1))
    return res


def fit_all_and_score(seeds=SEEDS, steps=STEPS):
    """T6h stage 1: train the CNN and the MLP (mean over seeds) on ALL field samples and score every non-water city cell.
    Never reads the hotspots."""
    from src import flood_cnn_baselines as bl
    from src import flood_cnn_data as fd
    from src import flood_cnn_r3 as r3
    from src.config import OUTPUTS

    d = fd.build(0)
    s, stack = d["samples"], d["stack"]
    ref = d["bmc"] & ~d["water"]
    src = fd.PatchSource(stack)
    X, _ = bl.features(stack, s["rows"], s["cols"], fd.CHANNELS)
    means = bl.local_means(stack)
    centre = np.asarray(stack, dtype="float32")
    patches = src.get(s["rows"], s["cols"])
    cnns = [train_cnn(patches, s["y"], seed=sd, steps=steps) for sd in seeds]
    mlps = [train_mlp(X, s["y"], seed=sd, steps=steps) for sd in seeds]
    del patches

    def cnn_fn(rows, cols):
        x = src.get(rows, cols)
        return np.mean([_predict(m, x) for m in cnns], axis=0)

    def mlp_fn(rows, cols):
        x = np.hstack([centre[:, rows, cols].T, means[:, rows, cols].T])
        return np.mean([_predict(m, x) for m in mlps], axis=0)

    for name, fn in (("cnn", cnn_fn), ("mlp", mlp_fn)):
        m = r3.score_map(ref, fn, chunk=2048)
        np.save(OUTPUTS / "flood_cnn" / f"r3_scores_{name}.npy", m)
        print(f"{name}: scored {int(np.isfinite(m).sum())} cells, score range {np.nanmin(m):.4f}-{np.nanmax(m):.4f}", flush=True)


def run_mlp_tuning():
    """Part A: nested search for the tabular MLP (3 seeds everywhere), saved for the combine step."""
    from sklearn.metrics import roc_auc_score

    from src import flood_tuning as tu
    X, y, kind, folds, _ = tu.load_part_a()
    res = tu.nested_family(X, y, folds, lambda p: MLPEstimator(p["width"], p["dropout"], p["steps"], SEEDS), tu.GRIDS["mlp"])
    tu._save("mlp", res)
    print(f"mlp       outer pooled AUC {roc_auc_score(y, res['outer']):.3f}  chosen per outer fold: {res['chosen']}", flush=True)


if __name__ == "__main__":
    import argparse

    from src.wardmap import memory_guard
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tag", default="benchmark")
    ap.add_argument("--limit-gb", type=float, default=4.0)
    ap.add_argument("--fit-all", action="store_true")
    ap.add_argument("--tune-mlp", action="store_true")
    a = ap.parse_args()
    memory_guard(a.limit_gb)
    if a.tune_mlp:
        run_mlp_tuning()
    else:
        fit_all_and_score(a.seeds, a.steps) if a.fit_all else run(a.steps, a.seeds, a.n_boot, a.tag)
