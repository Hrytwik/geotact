"""T5d: spatio-temporal deep model for tree-cover loss from the greenness time series (protocol in docs/results_log.md).

A U-Net reads 4 consecutive offset-corrected NDVI seasons and predicts other / stable tree / tree loss; weak labels come from
the Esri annual maps (2018-2021, 2019-2022, 2020-2023 windows); the 2022-23 to 2025-26 window is scored against the authors'
40 hand-checked points. Torch process."""
import numpy as np
import pandas as pd

from src.landuse import IGNORE

FIRST_SEASON = 2018
N_SEASONS = 8
TREE = 1
LOSS_TO = [2, 3, 4, 5]                    # cropland, grass/scrub, built-up, bare ground (water is not counted as tree loss)


def window_indices(start_year):
    i = start_year - FIRST_SEASON
    if i < 0 or i + 3 > N_SEASONS - 1:
        raise ValueError(f"no 4-season window starts in {start_year}")
    return [i, i + 1, i + 2, i + 3]


def window_labels(esri_start, esri_end):
    s, e = np.asarray(esri_start), np.asarray(esri_end)
    lab = np.zeros(s.shape, dtype=np.uint8)
    lab[(s == TREE) & (e == TREE)] = 1
    lab[(s == TREE) & np.isin(e, LOSS_TO)] = 2
    lab[(s == IGNORE) | (e == IGNORE)] = IGNORE
    return lab


def truth_from_sheet(sheet):
    """Tree loss = trees in early 2023 and not trees now; rows with an 'Unsure' or empty answer are dropped."""
    from src.landuse_accuracy import CLASSES
    lk = {c.lower(): c for c in CLASSES}
    lk["unsure"] = "Unsure"
    d = sheet.copy()
    d["t23"] = d.trees_in_2023.astype(str).str.strip().str.capitalize()
    d["now"] = d.now_covers.astype(str).str.strip().str.lower().map(lk)
    keep = d.t23.isin(["Yes", "No"]) & d.now.notna() & (d.now != "Unsure")
    d = d[keep]
    return pd.DataFrame(dict(id=d.id.values, truth=((d.t23 == "Yes") & (d.now != "Trees")).astype(int).values))


def auc_ci(score, y, n_boot=2000, seed=0):
    from sklearn.metrics import roc_auc_score
    score, y = np.asarray(score, float), np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    b = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                       np.r_[score[rng.choice(pos, len(pos))], score[rng.choice(neg, len(neg))]]) for _ in range(n_boot)]
    return float(roc_auc_score(y, score)), (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def paired_auc_diff(a, b, y, n_boot=2000, seed=0):
    from sklearn.metrics import roc_auc_score
    a, b, y = np.asarray(a, float), np.asarray(b, float), np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = []
    for _ in range(n_boot):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return dict(diff=float(roc_auc_score(y, a) - roc_auc_score(y, b)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def loss_rich_windows(Ys, windows, patch=128, min_share=0.01):
    """(window index, patch origin) pairs whose patch has at least min_share tree-loss pixels."""
    return [(k, (r, c)) for k, Y in enumerate(Ys) for r, c in windows if (Y[r:r + patch, c:c + patch] == 2).mean() >= min_share]


def draw(Ys, windows, rich, rng, rich_share=0.0):
    """One training patch: with probability rich_share from the loss-rich list, otherwise a uniform window and origin."""
    if rich and rng.random() < rich_share:
        return rich[int(rng.integers(len(rich)))]
    return int(rng.integers(len(Ys))), windows[int(rng.integers(len(windows)))]


def inverse_frequency_weights(Ys, windows, patch=128, n_classes=3):
    from src import landuse as lu
    counts = np.zeros(n_classes)
    for Y in Ys:
        y = lu._gather(Y, windows, patch)
        counts += np.bincount(y[y != IGNORE], minlength=n_classes)[:n_classes]
    present = counts > 0
    w = np.zeros(n_classes)
    w[present] = counts[present].sum() / counts[present]
    w[present] /= w[present].mean()
    w[~present] = w[present].max() if present.any() else 1.0
    return w.astype("float32")


def fit_windows(Xs, Ys, windows, patch=128, steps=3000, batch_size=16, lr=1e-3, seed=0, device="cpu", class_w=None, rich=None, rich_share=0.0):
    """Train one U-Net on several time windows of the same grid: every batch item draws a random window and patch origin."""
    import torch
    from torch import nn

    from src import landuse as lu
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = lu.build_model(in_ch=Xs[0].shape[0], n_classes=3).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    w = None if class_w is None else torch.as_tensor(class_w, dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    losses = []
    for _ in range(steps):
        xs, ys = [], []
        for _ in range(batch_size):
            k, (r, c) = draw(Ys, windows, rich, rng, rich_share)
            x, y = lu._augment(Xs[k][:, r:r + patch, c:c + patch], Ys[k][r:r + patch, c:c + patch], rng)
            xs.append(x); ys.append(y)
        yb = torch.as_tensor(np.stack(ys).astype("int64"), device=device)
        if not (yb != IGNORE).any():
            continue
        xb = torch.as_tensor(np.stack(xs), dtype=torch.float32, device=device)
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward(); opt.step(); sched.step()
        losses.append(float(loss.detach()))
    model.eval()
    return model, losses


def bitemporal_input(X1, X2, mean, std):
    """Two dates' bands, each standardised with the same band statistics, stacked (NaN becomes the band mean)."""
    from src import landuse as lu
    return np.concatenate([lu.standardise(np.asarray(X1, "float32"), mean, std), lu.standardise(np.asarray(X2, "float32"), mean, std)])


def run_bitemporal(seeds=(0, 1), steps=3000):
    import json

    import rasterio

    from src import grid
    from src import landuse as lu
    from src.config import OUTPUTS, RAW
    from src.vegetation_crosscheck import conversion_masks, ndvi_change
    st = np.load(lu.OUT / "band_stats.npz")
    rd_s2 = lambda n: (grid.read_aligned(RAW / f"{n}.tif") / 10000.0).astype("float32")
    E21 = lu.map_labels(grid.read_aligned(RAW / "io_lulc_2021.tif", resampling="nearest", dtype="uint8"))
    E23 = lu.map_labels(grid.read_aligned(RAW / "io_lulc_2023.tif", resampling="nearest", dtype="uint8"))
    s21, s23 = rd_s2("s2_2021"), rd_s2("s2_2023")
    Xtr = bitemporal_input(s21, s23, st["mean"], st["std"]); del s21
    Ytr = window_labels(E21, E23)
    train_w, val_w = lu.split_patches(Ytr.shape, 128, 64, 512, 0.25, seed=0)
    cw = lu.class_weights(Ytr, train_w, n_classes=3)
    rich = loss_rich_windows([Ytr], train_w, 128, 0.01)
    print(f"bitemporal: label counts {[int((Ytr == k).sum()) for k in range(3)]} | loss-rich patches {len(rich)} of {len(train_w)} | weights {np.round(cw, 2).tolist()}", flush=True)
    device = lu._device()
    Xap = bitemporal_input(s23, rd_s2("s2_2025_26_dry"), st["mean"], st["std"]); del s23
    probs, val = [], {}
    for s in seeds:
        model, losses = fit_windows([Xtr], [Ytr], train_w, steps=steps, seed=s, device=device, class_w=cw, rich=rich, rich_share=0.5)
        print(f"seed {s}: final loss {np.mean(losses[-100:]):.4f}", flush=True)
        probs.append(lu.predict_probs(model, Xap, device=device).astype("float32"))
        pv = lu.predict_probs(model, Xtr, device=device).argmax(0)
        ok = lu._val_canvas(Ytr.shape, val_w) & (Ytr != IGNORE)
        val[f"seed{s}"] = np.round(lu.metrics(lu.confusion(pv[ok], Ytr[ok], n_classes=3))["f1"], 3).tolist()
    loss_prob = np.mean(probs, axis=0)[2]
    with rasterio.open(lu.OUT / "landuse_2023_seed0.tif") as r:
        meta = r.meta.copy()
    meta.update(dtype="uint8", count=1, compress="deflate", nodata=None)
    with rasterio.open(OUTPUTS / "vegetation" / "dl_tree_loss_prob_bitemporal_2023_2026.tif", "w", **meta) as dst:
        dst.write(np.round(loss_prob * 100).astype("uint8"), 1)
    sheet = pd.read_excel("validation/tree_loss_labels.xlsx", sheet_name="labels")
    key = pd.read_csv("validation/tree_loss_key_private_do_not_open.csv")
    t = truth_from_sheet(sheet).merge(key[["id", "row", "col"]], on="id")
    rr, cc, y = t.row.values, t.col.values, t.truth.values
    chg = ndvi_change(lu.ndvi_seasons())
    s_drop = -chg[rr, cc]; s_drop = np.where(np.isfinite(s_drop), s_drop, np.nanmedian(s_drop))
    rdm = lambda f: rasterio.open(lu.OUT / f).read(1)
    flag = conversion_masks(rdm("landuse_2023_seed0.tif"), rdm("landuse_2023_seed1.tif"), rdm("landuse_2025_26_seed0.tif"),
                            rdm("landuse_2025_26_seed1.tif"), grid.study_mask())["converted"][rr, cc].astype(float)
    s_dl = loss_prob[rr, cc]
    auc, ci = auc_ci(s_dl, y)
    res = dict(n_points=int(len(t)), n_loss=int(y.sum()), heldout_f1_2021_2023=val, deep_bitemporal=dict(auc=auc, ci=list(ci)),
               deep_minus_greenness_drop=paired_auc_diff(s_dl, s_drop, y), deep_minus_unet_flag=paired_auc_diff(s_dl, flag, y),
               city_share_loss_prob_over_50=float((loss_prob[grid.study_mask()] > 0.5).mean()))
    res["meets_rule"] = bool(auc >= 0.75 and ci[0] >= 0.60 and res["deep_minus_greenness_drop"]["ci"][1] >= 0)
    (OUTPUTS / "vegetation" / "dl_tree_loss_bitemporal.json").write_text(json.dumps(res, indent=1, default=float))
    return res


# ---------------------------------------------------------------- real run
def run(seeds=(0, 1), steps=3000, attempt=1):
    import json

    import rasterio

    from src import grid
    from src import landuse as lu
    from src.config import OUTPUTS, RAW
    from src.landuse_improve import stack_inputs
    from src.vegetation_crosscheck import conversion_masks, ndvi_change
    nd_raw = lu.ndvi_seasons()
    nd = stack_inputs(np.zeros((0,) + nd_raw.shape[1:], "float32"), nd_raw)
    mu, sd = float(nd.mean()), float(nd.std())
    Z = ((nd - mu) / sd).astype("float32")
    E = {y: lu.map_labels(grid.read_aligned(RAW / f"io_lulc_{y}.tif", resampling="nearest", dtype="uint8")) for y in range(2018, 2024)}
    Xs = [Z[window_indices(y)] for y in (2018, 2019, 2020)]
    Ys = [window_labels(E[y], E[y + 3]) for y in (2018, 2019, 2020)]
    train_w, val_w = lu.split_patches(Ys[0].shape, 128, 64, 512, 0.25, seed=0)
    if attempt == 1:
        cw, rich, rich_share = np.mean([lu.class_weights(Y, train_w, n_classes=3) for Y in Ys], axis=0), None, 0.0
    else:                                                                  # attempt 2: targets the class collapse of attempt 1
        cw, rich, rich_share = inverse_frequency_weights(Ys, train_w, 128, 3), loss_rich_windows(Ys, train_w, 128, 0.01), 0.5
        print(f"attempt 2: loss-rich training patches {len(rich)} of {len(train_w) * len(Ys)}", flush=True)
    sfx = "" if attempt == 1 else f"_attempt{attempt}"
    counts = {f"{y}-{y + 3}": {n: int((Y == k).sum()) for k, n in enumerate(("other", "stable_tree", "tree_loss"))} for y, Y in zip((2018, 2019, 2020), Ys)}
    print("training label counts:", counts, "| class weights", np.round(cw, 2).tolist(), flush=True)
    device = lu._device()
    target = Z[window_indices(2022)]
    probs, val = [], {}
    for s in seeds:
        model, losses = fit_windows(Xs, Ys, train_w, steps=steps, seed=s, device=device, class_w=cw, rich=rich, rich_share=rich_share)
        print(f"seed {s}: final loss {np.mean(losses[-100:]):.4f}", flush=True)
        probs.append(lu.predict_probs(model, target, device=device).astype("float32"))
        pv = lu.predict_probs(model, Xs[2], device=device).argmax(0)                       # internal check on the 2020-2023 window
        canvas = lu._val_canvas(Ys[2].shape, val_w)
        ok = canvas & (Ys[2] != IGNORE)
        cm = lu.confusion(pv[ok], Ys[2][ok], n_classes=3)
        val[f"seed{s}"] = dict(confusion_rows_esri_cols_pred=cm.tolist(), **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in lu.metrics(cm).items()})
    loss_prob = np.mean(probs, axis=0)[2]
    with rasterio.open(lu.OUT / "landuse_2023_seed0.tif") as r:
        meta = r.meta.copy()
    meta.update(dtype="uint8", count=1, compress="deflate", nodata=None)
    with rasterio.open(OUTPUTS / "vegetation" / f"dl_tree_loss_prob_2022_2026{sfx}.tif", "w", **meta) as dst:
        dst.write(np.round(loss_prob * 100).astype("uint8"), 1)
    # test on the 40 hand-checked points
    sheet = pd.read_excel("validation/tree_loss_labels.xlsx", sheet_name="labels")
    key = pd.read_csv("validation/tree_loss_key_private_do_not_open.csv")
    t = truth_from_sheet(sheet).merge(key[["id", "group", "row", "col"]], on="id")
    rr, cc = t.row.values, t.col.values
    rd = lambda f: rasterio.open(lu.OUT / f).read(1)
    flag = conversion_masks(rd("landuse_2023_seed0.tif"), rd("landuse_2023_seed1.tif"), rd("landuse_2025_26_seed0.tif"),
                            rd("landuse_2025_26_seed1.tif"), grid.study_mask())["converted"]
    chg = ndvi_change(nd_raw)
    s_dl, s_drop, s_flag = loss_prob[rr, cc], -chg[rr, cc], flag[rr, cc].astype(float)
    s_drop = np.where(np.isfinite(s_drop), s_drop, np.nanmedian(s_drop))
    y = t.truth.values
    res = dict(n_points=int(len(t)), n_loss=int(y.sum()), training_label_counts=counts, internal_validation_2020_2023=val)
    for name, sc in (("deep_temporal_model", s_dl), ("greenness_drop", s_drop), ("unet_both_seed_flag", s_flag)):
        auc, ci = auc_ci(sc, y)
        res[name] = dict(auc=auc, ci=list(ci))
    res["deep_minus_greenness_drop"] = paired_auc_diff(s_dl, s_drop, y)
    res["deep_minus_unet_flag"] = paired_auc_diff(s_dl, s_flag, y)
    strict = (flag[rr, cc]) & (chg[rr, cc] <= -0.05)
    res["stricter_rule"] = dict(flagged=int(strict.sum()), real_loss_among_flagged=int(y[strict].sum()),
                                plain_flag_flagged=int(flag[rr, cc].sum()), plain_flag_real_loss=int(y[flag[rr, cc]].sum()))
    d = res["deep_temporal_model"]
    res["meets_rule"] = bool(d["auc"] >= 0.75 and d["ci"][0] >= 0.60 and res["deep_minus_greenness_drop"]["ci"][1] >= 0)
    (OUTPUTS / "vegetation" / f"dl_tree_loss{sfx}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    import json

    from src.wardmap import memory_guard
    memory_guard(4.5)
    import sys
    r = run_bitemporal() if "--bitemporal" in sys.argv else run(attempt=2 if "--attempt2" in sys.argv else 1)
    print(json.dumps({k: v for k, v in r.items() if k not in ("internal_validation_2020_2023",)}, indent=1, default=float))
