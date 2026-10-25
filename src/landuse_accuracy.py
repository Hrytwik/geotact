"""T7: accuracy of the land-use maps from the authors' 300 hand labels (protocol in docs/build_plan.md). Stratified random sample
with strata = Esri/IO 2023 classes; estimators after Olofsson et al. (2014) and Stehman (2014); ratio-estimator variances."""
import numpy as np
import pandas as pd

CLASSES = ["Water", "Trees", "Cropland", "Grass/Scrub", "Built-up", "Bare ground"]


def normalise_labels(s):
    lookup = {c.lower(): c for c in CLASSES}
    out = s.astype(str).str.strip().str.lower().map(lookup)
    if out.isna().any():
        raise ValueError(f"unknown class labels: {sorted(set(s[out.isna()].astype(str)))}")
    return out


def normalise_confidence(s):
    out = s.astype(str).str.strip().str.lower().map({"sure": "Sure", "unsure": "Unsure"})
    if out.isna().any():
        raise ValueError(f"unknown confidence values: {sorted(set(s[out.isna()].astype(str)))}")
    return out


def _moments(strata, a, b=None):
    """Per stratum: n, mean of a, variance of a (ddof 1), and the same for b with the covariance."""
    out = {}
    for h in np.unique(strata):
        m = strata == h
        n = int(m.sum())
        d = dict(n=n, ma=float(a[m].mean()), va=float(a[m].var(ddof=1)) if n > 1 else 0.0)
        if b is not None:
            d.update(mb=float(b[m].mean()), vb=float(b[m].var(ddof=1)) if n > 1 else 0.0,
                     cab=float(np.cov(a[m], b[m], ddof=1)[0, 1]) if n > 1 else 0.0)
        out[h] = d
    return out


def stratified_accuracy(strata, correct, W):
    strata, correct = np.asarray(strata), np.asarray(correct, dtype=float)
    mo = _moments(strata, correct)
    acc = sum(W[h] * d["ma"] for h, d in mo.items())
    var = sum(W[h] ** 2 * d["va"] / d["n"] for h, d in mo.items())
    return float(acc), float(np.sqrt(var))


def ratio_estimate(strata, num, den, W):
    """Estimate and standard error of sum(W*mean(num)) / sum(W*mean(den)) (Cochran ratio estimator, no finite-population term)."""
    strata, num, den = np.asarray(strata), np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    mo = _moments(strata, num, den)
    Y = sum(W[h] * d["ma"] for h, d in mo.items()); X = sum(W[h] * d["mb"] for h, d in mo.items())
    if X == 0:
        return float("nan"), float("nan")
    R = Y / X
    var = sum(W[h] ** 2 / d["n"] * (d["va"] + R * R * d["vb"] - 2 * R * d["cab"]) for h, d in mo.items()) / X ** 2
    return float(R), float(np.sqrt(max(var, 0.0)))


def users_accuracy(strata, mapped, ref, W, classes):
    mapped, ref = np.asarray(mapped), np.asarray(ref)
    return {c: ratio_estimate(strata, (mapped == c) & (ref == c), mapped == c, W) for c in classes}


def producers_accuracy(strata, mapped, ref, W, classes):
    mapped, ref = np.asarray(mapped), np.asarray(ref)
    return {c: ratio_estimate(strata, (mapped == c) & (ref == c), ref == c, W) for c in classes}


def area_confusion(strata, mapped, ref, W, classes):
    """Estimated proportion of the area in each (mapped, reference) cell: sum over strata of weight x share of the stratum's points."""
    strata, mapped, ref = np.asarray(strata), np.asarray(mapped), np.asarray(ref)
    p = pd.DataFrame(0.0, index=classes, columns=classes)
    for h in np.unique(strata):
        m = strata == h
        for i in classes:
            for j in classes:
                p.loc[i, j] += W[h] * ((mapped[m] == i) & (ref[m] == j)).sum() / m.sum()
    return p


def paired_difference(strata, correct_a, correct_b, W, n_boot=2000, seed=0):
    """Overall accuracy of map A minus map B on the same points; stratified bootstrap (resampling points within strata)."""
    strata, a, b = np.asarray(strata), np.asarray(correct_a, float), np.asarray(correct_b, float)
    diff = stratified_accuracy(strata, a, W)[0] - stratified_accuracy(strata, b, W)[0]
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == h) for h in np.unique(strata)]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        d.append(stratified_accuracy(strata[idx], a[idx], W)[0] - stratified_accuracy(strata[idx], b[idx], W)[0])
    d = np.asarray(d)
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return dict(diff=float(diff), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p=max(p, 1.0 / (n_boot + 1)) if p < 1 else 1.0)


# ---------------------------------------------------------------- real run
def load_inputs():
    from src.config import ROOT
    v = ROOT / "validation"
    lab = pd.read_excel(v / "validation_labels.xlsx", sheet_name="labels")
    key = pd.read_csv(v / "strata_private_do_not_open.csv")
    w = pd.read_csv(v / "strata_weights.csv").set_index("cls").weight.to_dict()
    d = key.merge(lab[["id", "lat", "lon", "label", "confidence"]], on="id", suffixes=("", "_lab"))
    assert len(d) == 300 and np.allclose(d.lat, d.lat_lab) and np.allclose(d.lon, d.lon_lab), "labels and sample points do not line up"
    d["ref"] = normalise_labels(d.label)
    d["conf"] = normalise_confidence(d.confidence)
    return d, w


def sample_map(path, x, y):
    import rasterio
    with rasterio.open(path) as r:
        rows, cols = rasterio.transform.rowcol(r.transform, x, y)
        a = r.read(1)
    return np.array([CLASSES[a[i, j]] for i, j in zip(rows, cols)])


def run(n_boot=2000):
    import json

    from src.config import OUTPUTS
    d, W = load_inputs()
    lu = OUTPUTS / "landuse"
    maps = {"esri_2023": d.stratum.values}
    for name, f in (("unet_2023_seed0", "landuse_2023_seed0.tif"), ("unet_2023_seed1", "landuse_2023_seed1.tif"),
                    ("unet_2025_26_seed0", "landuse_2025_26_seed0.tif"), ("unet_2025_26_seed1", "landuse_2025_26_seed1.tif"),
                    ("cons_2023_seed0", "landuse_2023_cons_seed0.tif"), ("cons_2023_seed1", "landuse_2023_cons_seed1.tif"),
                    ("cons_2025_26_seed0", "landuse_2025_26_cons_seed0.tif"), ("cons_2025_26_seed1", "landuse_2025_26_cons_seed1.tif")):
        maps[name] = sample_map(lu / f, d.x_utm.values, d.y_utm.values)
    res = {}
    for subset, mask in (("all_300", np.ones(len(d), bool)), ("sure_only", (d.conf == "Sure").values)):
        s = d.stratum.values[mask]; ref = d.ref.values[mask]
        res[subset] = {}
        for name, m in maps.items():
            m = m[mask]
            acc, se = stratified_accuracy(s, m == ref, W)
            res[subset][name] = dict(n=int(mask.sum()), overall=acc, se=se, ci=[acc - 1.96 * se, acc + 1.96 * se],
                                     users={c: v for c, v in users_accuracy(s, m, ref, W, CLASSES).items()},
                                     producers={c: v for c, v in producers_accuracy(s, m, ref, W, CLASSES).items()})
        res[subset]["diff_unet2023_minus_esri2023"] = paired_difference(
            s, maps["unet_2023_seed0"][mask] == ref, maps["esri_2023"][mask] == ref, W, n_boot)
        res[subset]["diff_unet2025_minus_esri2023"] = paired_difference(
            s, maps["unet_2025_26_seed0"][mask] == ref, maps["esri_2023"][mask] == ref, W, n_boot)
        res[subset]["diff_cons2023_minus_unet2023"] = paired_difference(
            s, maps["cons_2023_seed0"][mask] == ref, maps["unet_2023_seed0"][mask] == ref, W, n_boot)
        res[subset]["diff_cons2023_minus_esri2023"] = paired_difference(
            s, maps["cons_2023_seed0"][mask] == ref, maps["esri_2023"][mask] == ref, W, n_boot)
        res[subset]["diff_cons2025_minus_unet2025"] = paired_difference(
            s, maps["cons_2025_26_seed0"][mask] == ref, maps["unet_2025_26_seed0"][mask] == ref, W, n_boot)
    conf = area_confusion(d.stratum.values, maps["unet_2023_seed0"], d.ref.values, W, CLASSES)
    res["area_confusion_unet_2023_seed0"] = conf.round(4).to_dict()
    res["ref_class_counts"] = d.ref.value_counts().to_dict()
    (OUTPUTS / "landuse" / "accuracy.json").write_text(json.dumps(res, indent=1, default=float))
    rows = []
    for name in maps:
        r = res["all_300"][name]
        for c in CLASSES:
            rows.append(dict(map=name, cls=c, users=round(r["users"][c][0], 3), users_se=round(r["users"][c][1], 3),
                             producers=round(r["producers"][c][0], 3), producers_se=round(r["producers"][c][1], 3)))
    pd.DataFrame(rows).to_csv("docs/landuse_accuracy_by_class.csv", index=False)
    for name in maps:
        r = res["all_300"][name]; s = res["sure_only"][name]
        print(f"{name:20s} overall {100 * r['overall']:.1f}% [{100 * r['ci'][0]:.1f}, {100 * r['ci'][1]:.1f}]   sure-only ({s['n']} pts) {100 * s['overall']:.1f}%")
    for k in ("diff_unet2023_minus_esri2023", "diff_unet2025_minus_esri2023", "diff_cons2023_minus_unet2023", "diff_cons2023_minus_esri2023", "diff_cons2025_minus_unet2025"):
        r = res["all_300"][k]
        print(f"{k}: {100 * r['diff']:+.1f} points [{100 * r['ci'][0]:+.1f}, {100 * r['ci'][1]:+.1f}] p {r['p']:.3f}")
    print("reference class counts:", res["ref_class_counts"])
    return res


if __name__ == "__main__":
    run()
