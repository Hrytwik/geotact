"""T4: land-use classification with a U-Net on Sentinel-2.

Inputs: Sentinel-2 composite (10 bands, reflectance). Weak labels: Esri/Impact Observatory annual LULC
collapsed to the six chapter classes (shared with the validation sample in src.validation_points).
The model learns another model's labels, so agreement with them is NOT accuracy; real accuracy comes
only from the independent hand-labelled sample (T7).
"""
import math

import numpy as np

from src import grid
from src.config import OUTPUTS, RAW
from src.validation_points import CLASSES, IO_TO_CLASS

IGNORE = 255
OUT = OUTPUTS / "landuse"


# ---------------------------------------------------------------- labels / data
def map_labels(io):
    """Esri class codes -> chapter class index 0-5; unusable codes (nodata, snow, cloud, unmapped) -> IGNORE."""
    lut = np.full(256, IGNORE, dtype=np.uint8)
    for code, cls in IO_TO_CLASS.items():
        lut[code] = cls
    return lut[np.asarray(io, dtype=np.uint8)]


# ESA WorldCover codes -> chapter classes (water and herbaceous wetland to Water, as Esri's flooded vegetation); snow and moss unusable
WC_TO_CLASS = {10: 1, 95: 1, 20: 3, 30: 3, 40: 2, 50: 4, 60: 5, 80: 0, 90: 0}


def wc_labels(wc):
    lut = np.full(256, IGNORE, dtype=np.uint8)
    for code, cls in WC_TO_CLASS.items():
        lut[code] = cls
    return lut[np.asarray(wc, dtype=np.uint8)]


def consensus(esri_cls, wc_cls):
    """Keep a label only where both products give the same usable class; everywhere else IGNORE."""
    esri_cls, wc_cls = np.asarray(esri_cls, dtype=np.uint8), np.asarray(wc_cls, dtype=np.uint8)
    keep = (esri_cls == wc_cls) & (esri_cls != IGNORE)
    return np.where(keep, esri_cls, IGNORE).astype(np.uint8)


def labels_from(io, wc, mode="esri"):
    if mode == "esri":
        return map_labels(io)
    if mode == "consensus":
        return consensus(map_labels(io), wc_labels(wc))
    raise ValueError(f"unknown label mode {mode!r}")


def load_inputs(composite="s2_2023", io_year=2023, labels="esri"):
    """X: (10, H, W) reflectance (NaN = no data); Y: (H, W) uint8 chapter classes (IGNORE = unusable)."""
    X = grid.read_aligned(RAW / f"{composite}.tif") / 10000.0
    io = grid.read_aligned(RAW / f"io_lulc_{io_year}.tif", resampling="nearest", dtype="uint8")
    wc = grid.read_aligned(RAW / "worldcover_2021.tif", resampling="nearest", dtype="uint8") if labels == "consensus" else None
    return dict(X=X.astype("float32"), Y=labels_from(io, wc, labels))


# ---------------------------------------------------------------- spatial split
def split_patches(shape, patch=128, stride=64, block=512, val_frac=0.25, seed=0):
    """Train / validation patch origins. Whole blocks go to one split and every patch lies inside a single
    block, so no pixel (and no 128 px context) is shared between the splits."""
    H, W = shape
    by_block = {}
    for r in range(0, H, block):
        for c in range(0, W, block):
            r1, c1 = min(r + block, H), min(c + block, W)
            wins = [(rr, cc) for rr in range(r, r1 - patch + 1, stride) for cc in range(c, c1 - patch + 1, stride)]
            if wins:
                by_block[(r // block, c // block)] = wins
    keys = sorted(by_block)
    order = np.random.default_rng(seed).permutation(len(keys))
    n_val = max(1, round(val_frac * len(keys)))
    val_keys = {keys[i] for i in order[:n_val]}
    train = [w for k in keys if k not in val_keys for w in by_block[k]]
    val = [w for k in keys if k in val_keys for w in by_block[k]]
    return train, val


def _gather(arr, windows, patch):
    return np.concatenate([arr[..., r:r + patch, c:c + patch].reshape(*arr.shape[:-2], -1) for r, c in windows], axis=-1)


def band_stats(X, windows, patch=128):
    """Per-band mean/std from the given windows only (use training windows; never validation)."""
    v = _gather(X, windows, patch)
    return np.nanmean(v, axis=1).astype("float32"), np.nanstd(v, axis=1).astype("float32")


def standardise(X, mean, std):
    Z = (X - mean[:, None, None]) / np.maximum(std, 1e-6)[:, None, None]
    return np.nan_to_num(Z, nan=0.0).astype("float32")


def class_weights(Y, windows, patch=128, n_classes=6):
    """Inverse-sqrt-frequency weights, mean 1 over present classes; absent classes get the largest weight."""
    y = _gather(Y, windows, patch)
    counts = np.bincount(y[y != IGNORE], minlength=n_classes)[:n_classes].astype(float)
    present = counts > 0
    w = np.zeros(n_classes)
    w[present] = np.sqrt(counts[present].sum() / counts[present])
    w[present] /= w[present].mean()
    w[~present] = w[present].max()
    return w.astype("float32")


# ---------------------------------------------------------------- model / training
def build_model(in_ch=10, n_classes=6, encoder="resnet18"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=in_ch, classes=n_classes)


def _augment(x, y, rng):
    k = int(rng.integers(4))
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(0, 1))
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    gain = rng.normal(1.0, 0.05, size=(x.shape[0], 1, 1)).astype("float32")     # mimic year-to-year radiometric shifts
    off = rng.normal(0.0, 0.05, size=(x.shape[0], 1, 1)).astype("float32")
    return np.ascontiguousarray(x) * gain + off, np.ascontiguousarray(y)


def fit(model, X, Y, windows, patch=128, steps=1000, batch_size=16, lr=1e-3, class_w=None, device="cpu",
        augment=True, seed=0, weight_decay=1e-4, log_every=0):
    """Train with class-weighted cross-entropy; batches without a labelled pixel are skipped. Returns the losses."""
    import torch
    from torch import nn
    rng = np.random.default_rng(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    w = None if class_w is None else torch.as_tensor(class_w, dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    losses = []
    for step in range(steps):
        idx = rng.integers(len(windows), size=batch_size)
        xs, ys = [], []
        for i in idx:
            r, c = windows[i]
            x, y = X[:, r:r + patch, c:c + patch], Y[r:r + patch, c:c + patch]
            if augment:
                x, y = _augment(x, y, rng)
            xs.append(x)
            ys.append(y)
        xb = torch.as_tensor(np.stack(xs), dtype=torch.float32, device=device)
        yb = torch.as_tensor(np.stack(ys).astype("int64"), device=device)
        if not (yb != IGNORE).any():
            continue
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1}/{steps} loss {np.mean(losses[-log_every:]):.4f}", flush=True)
    return losses


# ---------------------------------------------------------------- inference
def predict_map(model, X, tile=256, margin=32, device="cpu", return_conf=False):
    """Seam-free tiled prediction: each tile is predicted with `margin` pixels of context on every side and only
    its centre is kept; the image is reflect-padded so border tiles get context too."""
    import torch
    with torch.no_grad():
        return _predict_tiles(torch, model, X, tile, margin, device, return_conf)


def _predict_tiles(torch, model, X, tile, margin, device, return_conf):
    model.to(device).eval()
    C, H, W = X.shape
    s = tile - 2 * margin
    Xp = np.pad(X, ((0, 0), (margin, margin), (margin, margin)), mode="reflect")
    pred = np.zeros((H, W), dtype=np.uint8)
    conf = np.zeros((H, W), dtype=np.uint8)
    for r in range(0, H, s):
        for c in range(0, W, s):
            hh, ww = min(s, H - r), min(s, W - c)
            crop = Xp[:, r:r + hh + 2 * margin, c:c + ww + 2 * margin]
            ph, pw = (-crop.shape[1]) % 32, (-crop.shape[2]) % 32
            crop = np.pad(crop, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            logits = model(torch.as_tensor(crop[None], dtype=torch.float32, device=device))[0]
            logits = logits[:, margin:margin + hh, margin:margin + ww]
            pred[r:r + hh, c:c + ww] = logits.argmax(0).cpu().numpy()
            if return_conf:
                conf[r:r + hh, c:c + ww] = (logits.softmax(0).max(0).values * 100).cpu().numpy().astype(np.uint8)
    return (pred, conf) if return_conf else pred


# ---------------------------------------------------------------- metrics
def confusion(pred, ref, n_classes=6):
    """Rows = reference, columns = prediction; pixels with an IGNORE label on either side are dropped."""
    pred, ref = np.asarray(pred).ravel().astype(int), np.asarray(ref).ravel().astype(int)
    ok = (ref != IGNORE) & (pred != IGNORE)
    return np.bincount(ref[ok] * n_classes + pred[ok], minlength=n_classes ** 2).reshape(n_classes, n_classes)


def metrics(c):
    c = np.asarray(c, dtype=float)
    d, row, col = np.diag(c), c.sum(1), c.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall, precision = d / row, d / col
        f1 = 2 * precision * recall / (precision + recall)
        iou = d / (row + col - d)
    return dict(overall_accuracy=float(d.sum() / c.sum()), recall=recall, precision=precision, f1=f1, iou=iou,
                macro_f1=float(np.nanmean(f1)))


# ---------------------------------------------------------------- full run
def _val_canvas(shape, windows, patch=128):
    cv = np.zeros(shape, dtype=bool)
    for r, c in windows:
        cv[r:r + patch, c:c + patch] = True
    return cv


def _jsonable(m):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}


def _write_map(path, arr, nodata=IGNORE):
    import rasterio
    g = grid.load_grid()
    with rasterio.open(path, "w", driver="GTiff", height=g.shape[0], width=g.shape[1], count=1,
                       dtype="uint8", crs=g.crs, transform=g.transform, nodata=nodata, compress="deflate", tiled=True) as dst:
        dst.write(arr, 1)


def _device(device=None):
    import torch
    return device or ("mps" if torch.backends.mps.is_available() else "cpu")


def train(seed=0, steps=3000, batch_size=16, lr=1e-3, device=None, log_every=250, tag=None, labels="esri"):
    """Train on the 2023 composite with Esri 2023 labels; saves weights, band statistics and the loss curve."""
    import time

    import pandas as pd
    import torch
    device = _device(device)
    tag = tag or f"seed{seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_inputs("s2_2023", 2023, labels)
    X, Y = d["X"], d["Y"]
    train_w, val_w = split_patches(Y.shape, 128, 64, 512, 0.25, seed=0)           # split seed is fixed, independent of `seed`
    mean, std = band_stats(X, train_w)
    Xs = standardise(X, mean, std)
    cw = class_weights(Y, train_w)
    print(f"[{tag}] device {device} | train patches {len(train_w)} | val patches {len(val_w)} | class weights {cw.round(2)}", flush=True)
    torch.manual_seed(seed)
    model = build_model()
    t0 = time.time()
    losses = fit(model, Xs, Y, train_w, steps=steps, batch_size=batch_size, lr=lr, class_w=cw, device=device, seed=seed,
                 log_every=log_every)
    print(f"[{tag}] trained in {time.time() - t0:.0f} s | final loss {np.mean(losses[-100:]):.4f}", flush=True)
    torch.save(model.state_dict(), OUT / f"unet_{tag}.pt")
    np.savez(OUT / "band_stats.npz", mean=mean, std=std)
    pd.Series(losses, name="loss").to_csv(OUT / f"loss_{tag}.csv", index=False)
    (OUT / f"config_{tag}.json").write_text(__import__("json").dumps(
        dict(seed=seed, steps=steps, batch_size=batch_size, lr=lr, device=device, composite="s2_2023", labels="io_lulc_2023" if labels == "esri" else "io_lulc_2023+worldcover_2021_consensus")))
    return tag


def evaluate(tag="seed0", device=None, labels="esri"):
    """Reload saved weights; agreement with the weak labels on held-out blocks; predict 2023 and 2025-26 maps."""
    import json

    import torch
    device = _device(device)
    d = load_inputs("s2_2023", 2023, labels)
    X, Y = d["X"], d["Y"]
    train_w, val_w = split_patches(Y.shape, 128, 64, 512, 0.25, seed=0)
    st = np.load(OUT / "band_stats.npz")
    mean, std = st["mean"], st["std"]
    Xs = standardise(X, mean, std)
    model = build_model()
    model.load_state_dict(torch.load(OUT / f"unet_{tag}.pt", map_location="cpu"))
    val, trn = _val_canvas(Y.shape, val_w), _val_canvas(Y.shape, train_w)
    pred, conf = predict_map(model, Xs, tile=256, margin=32, device=device, return_conf=True)
    res = {}
    for name, sel in [("validation_blocks", val), ("training_blocks", trn)]:
        c = confusion(pred[sel], Y[sel])
        res[name] = dict(confusion=c.tolist(), **_jsonable(metrics(c)))
    ok = val & (Y != IGNORE)
    res["majority_class_validation_accuracy"] = float(np.bincount(Y[ok], minlength=6).max() / ok.sum())
    res["classes"] = CLASSES
    res["config"] = json.loads((OUT / f"config_{tag}.json").read_text())
    print(f"[{tag}] agreement with Esri labels | validation OA {res['validation_blocks']['overall_accuracy']:.4f} "
          f"macro-F1 {res['validation_blocks']['macro_f1']:.4f} | training OA {res['training_blocks']['overall_accuracy']:.4f}", flush=True)
    (OUT / f"metrics_{tag}.json").write_text(json.dumps(res, indent=1))
    _write_map(OUT / f"landuse_2023_{tag}.tif", pred)
    _write_map(OUT / f"confidence_2023_{tag}.tif", conf)
    X2 = grid.read_aligned(RAW / "s2_2025_26_dry.tif") / 10000.0
    pred2 = predict_map(model, standardise(X2.astype("float32"), mean, std), tile=256, margin=32, device=device)
    _write_map(OUT / f"landuse_2025_26_{tag}.tif", pred2)
    m = grid.study_mask()
    for nm, arr in [("Esri 2023 (labels)", Y), ("U-Net on 2023 composite", pred), ("U-Net on 2025-26 composite", pred2)]:
        v = arr[m]
        v = v[v != IGNORE]
        print(f"[{tag}] inside BMC {nm:28s}", {CLASSES[k]: round(100 * float((v == k).mean()), 1) for k in range(6)}, flush=True)
    return res


def run(seed=0, steps=3000, batch_size=16, lr=1e-3, device=None, tag=None, labels="esri"):
    tag = train(seed, steps, batch_size, lr, device, tag=tag, labels=labels)
    return evaluate(tag, device, labels)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--labels", choices=["esri", "consensus"], default="esri")
    ap.add_argument("--eval-only", action="store_true", help="skip training; evaluate saved weights for --tag")
    a = ap.parse_args()
    from src.wardmap import memory_guard
    memory_guard(5.0)
    if a.eval_only:
        evaluate(a.tag or f"seed{a.seed}", labels=a.labels)
    else:
        run(seed=a.seed, steps=a.steps, batch_size=a.batch, tag=a.tag, labels=a.labels)
