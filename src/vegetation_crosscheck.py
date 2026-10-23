"""T5b: cross-check of the land-use U-Net's tree loss (2023 -> 2025-26) against the offset-corrected NDVI change (protocol in
docs/build_plan.md). Both products come from Sentinel-2, so this is corroboration, not independent validation, and not accuracy."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

TREE, BUILT = 1, 4
BLOCK_PX = 200                                     # 2 km blocks on the 10 m grid


def conversion_masks(a23s0, a23s1, a26s0, a26s1, study):
    study = np.asarray(study, bool)
    t23 = (a23s0 == TREE) & (a23s1 == TREE)
    b23 = (a23s0 == BUILT) & (a23s1 == BUILT)
    t26 = (a26s0 == TREE) & (a26s1 == TREE)
    b26 = (a26s0 == BUILT) & (a26s1 == BUILT)
    other26 = (a26s0 != TREE) & (a26s0 != BUILT) & (a26s1 != TREE) & (a26s1 != BUILT)
    return {
        "converted": t23 & b26 & study, "stable_tree": t23 & t26 & study, "stable_built": b23 & b26 & study,
        "tree_to_other": t23 & other26 & study, "built_to_tree": b23 & t26 & study,
        "converted_seed0": (a23s0 == TREE) & (a26s0 == BUILT) & study, "converted_seed1": (a23s1 == TREE) & (a26s1 == BUILT) & study,
        "disagree_2023": (((a23s0 == TREE) & (a23s1 == BUILT)) | ((a23s0 == BUILT) & (a23s1 == TREE))) & study,
        "disagree_2026": (((a26s0 == TREE) & (a26s1 == BUILT)) | ((a26s0 == BUILT) & (a26s1 == TREE))) & study,
    }


def ndvi_change(stack):
    """Mean of seasons 2024_25 and 2025_26 minus mean of 2021_22 and 2022_23 (indices 6, 7 and 3, 4 of the 8-season stack)."""
    stack = np.asarray(stack, dtype="float64")
    return np.nanmean(stack[[6, 7]], axis=0) - np.nanmean(stack[[3, 4]], axis=0)


def prob_less(x, y):
    """P(x < y) over all pairs, ties counted half (Mann-Whitney)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = rankdata(np.concatenate([x, y]))
    u_x = r[:len(x)].sum() - len(x) * (len(x) + 1) / 2
    return float(1 - u_x / (len(x) * len(y)))


def _groups(block_ids):
    order = np.argsort(block_ids, kind="stable")
    ids, starts = np.unique(np.asarray(block_ids)[order], return_index=True)
    return {int(i): chunk for i, chunk in zip(ids, np.split(order, starts[1:]))}


def block_bootstrap_prob_less(x, x_blocks, y, y_blocks, n_boot=1000, seed=0):
    """prob_less with a 95 percent interval from resampling whole spatial blocks (keeps neighbouring pixels together)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gx, gy = _groups(x_blocks), _groups(y_blocks)
    blocks = np.array(sorted(set(gx) | set(gy)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        xi = [gx[b] for b in pick if b in gx]; yi = [gy[b] for b in pick if b in gy]
        if not xi or not yi:
            continue
        vals.append(prob_less(x[np.concatenate(xi)], y[np.concatenate(yi)]))
    return prob_less(x, y), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def ward_table(wards, tree23, converted, change, px_ha=0.01, min_ha=10.0):
    """Per ward: 2023 tree area, converted area, loss fraction and mean NDVI change over the 2023 tree pixels; wards under min_ha dropped."""
    rows = []
    for w in np.unique(wards[wards > 0]):
        sel = (wards == w) & tree23
        tree_ha = sel.sum() * px_ha
        if tree_ha < min_ha:
            continue
        conv_ha = (converted & sel).sum() * px_ha
        rows.append(dict(ward=int(w), tree_ha=float(tree_ha), converted_ha=float(conv_ha), loss_frac=float(conv_ha / tree_ha),
                         ndvi_change_mean=float(np.nanmean(change[sel]))))
    return pd.DataFrame(rows, columns=["ward", "tree_ha", "converted_ha", "loss_frac", "ndvi_change_mean"])


def spearman_bootstrap(a, b, n_boot=1000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        r = spearmanr(a[i], b[i])[0]
        if np.isfinite(r):
            boots.append(r)
    return float(spearmanr(a, b)[0]), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def decide(pixel, ward):
    """Registered decision rule. pixel: prob, ci, median_diff (converted minus stable built); ward: rho, ci."""
    pix = pixel["prob"] > 0.60 and pixel["ci"][0] > 0.5 and pixel["median_diff"] <= -0.02
    wrd = ward["rho"] < -0.3 and ward["ci"][1] < 0
    if pix and wrd:
        return "corroborated: both the pixel test and the ward test are supported"
    if pix:
        return "partly corroborated: the pixel test is supported, the ward test is not"
    if wrd:
        return "unconfirmed: only the ward test is supported (case not covered by the registered rule, treated as unconfirmed)"
    return "unconfirmed: neither test is supported"


def run(n_boot=1000, seed=0):
    import json

    import rasterio

    from src import grid
    from src.config import OUTPUTS
    lu, veg = OUTPUTS / "landuse", OUTPUTS / "vegetation"
    rd = lambda name: rasterio.open(lu / name).read(1)
    a23s0, a23s1, a26s0, a26s1 = rd("landuse_2023_seed0.tif"), rd("landuse_2023_seed1.tif"), rd("landuse_2025_26_seed0.tif"), rd("landuse_2025_26_seed1.tif")
    study = grid.study_mask()
    m = conversion_masks(a23s0, a23s1, a26s0, a26s1, study)
    change = ndvi_change(rasterio.open(veg / "ndvi_norm_stack.tif").read())
    ok = np.isfinite(change)
    rr, cc = np.indices(change.shape)
    block = (rr // BLOCK_PX) * 100 + (cc // BLOCK_PX)
    ha = lambda k: float(m[k].sum() / 100.0)
    print("GROUPS (pixels / hectares):", flush=True)
    for k in m:
        print(f"  {k:16s} {int(m[k].sum()):9d} px  {ha(k):9.1f} ha", flush=True)
    tree23 = (a23s0 == TREE) & (a23s1 == TREE) & study
    print(f"2023 tree area, both seeds: {tree23.sum() / 100:.0f} ha; converted share {100 * m['converted'].sum() / tree23.sum():.1f}% ; "
          f"seed disagreement tree/built: 2023 {int(m['disagree_2023'].sum())} px, 2025-26 {int(m['disagree_2026'].sum())} px", flush=True)

    rng = np.random.default_rng(seed)
    conv = m["converted"] & ok
    sb = np.flatnonzero((m["stable_built"] & ok).ravel())
    sb = rng.choice(sb, size=min(100_000, len(sb)), replace=False)
    st = np.flatnonzero((m["stable_tree"] & ok).ravel())
    st = rng.choice(st, size=min(100_000, len(st)), replace=False)
    cv = np.flatnonzero(conv.ravel())
    flat_change, flat_block = change.ravel(), block.ravel()
    est, ci = block_bootstrap_prob_less(flat_change[cv], flat_block[cv], flat_change[sb], flat_block[sb], n_boot, seed)
    med = dict(converted=float(np.median(flat_change[cv])), stable_built=float(np.median(flat_change[sb])), stable_tree=float(np.median(flat_change[st])))
    pixel = dict(prob=est, ci=ci, median_diff=med["converted"] - med["stable_built"], n_converted=int(len(cv)), n_stable_built_sample=int(len(sb)), medians=med)
    est_t, ci_t = block_bootstrap_prob_less(flat_change[cv], flat_block[cv], flat_change[st], flat_block[st], n_boot, seed)
    print(f"\nPIXEL TEST: NDVI change median: converted {med['converted']:+.3f}, stable built {med['stable_built']:+.3f}, stable tree {med['stable_tree']:+.3f}", flush=True)
    print(f"  P(converted change < stable built change) = {est:.3f}  95% block-bootstrap CI [{ci[0]:.3f}, {ci[1]:.3f}]  median difference {pixel['median_diff']:+.3f}", flush=True)
    print(f"  reference: P(converted < stable tree) = {est_t:.3f} [{ci_t[0]:.3f}, {ci_t[1]:.3f}]", flush=True)
    for k in ("converted_seed0", "converted_seed1"):
        sel = np.flatnonzero((m[k] & ok).ravel())
        print(f"  single-seed group {k}: {len(sel)} px, median NDVI change {np.median(flat_change[sel]):+.3f}", flush=True)

    wt = ward_table(grid.ward_raster(), tree23, m["converted"], change)
    rho, rci = spearman_bootstrap(wt.loss_frac.values, wt.ndvi_change_mean.values, n_boot, seed)
    t5 = pd.read_csv(veg / "ward_summary.csv").set_index("ward")
    t5m = t5.reindex(wt.ward.values).median_slope_per_yr.values
    good = np.isfinite(t5m)
    rho5, rci5 = spearman_bootstrap(wt.loss_frac.values[good], t5m[good], n_boot, seed)
    ward = dict(rho=rho, ci=rci, n_wards=int(len(wt)), rho_vs_t5_slope=rho5, ci_vs_t5_slope=rci5)
    print(f"\nWARD TEST ({len(wt)} wards with >= 10 ha of 2023 trees): Spearman rho (loss fraction vs mean NDVI change) = {rho:+.2f}  95% CI [{rci[0]:+.2f}, {rci[1]:+.2f}]", flush=True)
    print(f"  loss fraction vs T5 ward median slope: rho {rho5:+.2f} [{rci5[0]:+.2f}, {rci5[1]:+.2f}]", flush=True)
    # post-hoc (NOT part of the registered verdict): the ward mean above includes the converted pixels themselves, so part of that
    # correlation is mechanical; repeat it with the mean NDVI change over STABLE-tree pixels only
    ws = ward_table(grid.ward_raster(), m["stable_tree"], m["converted"], change, min_ha=0).set_index("ward").reindex(wt.ward.values)
    okw = np.isfinite(ws.ndvi_change_mean.values)
    rho_s, ci_s = spearman_bootstrap(wt.loss_frac.values[okw], ws.ndvi_change_mean.values[okw], n_boot, seed)
    ward["posthoc_rho_stable_tree_only"] = rho_s; ward["posthoc_ci"] = ci_s
    print(f"  POST-HOC (not registered): loss fraction vs NDVI change over stable-tree pixels only: rho {rho_s:+.2f} [{ci_s[0]:+.2f}, {ci_s[1]:+.2f}]; "
          f"{int(wt.tree_ha.sum() / (tree23.sum() / 100) * 100)}% of the 2023 tree area lies in these {len(wt)} wards (largest ward {wt.tree_ha.max():.0f} ha)", flush=True)
    verdict = decide(pixel, ward)
    print(f"\nDECISION (registered rule): {verdict}", flush=True)
    wt.to_csv("docs/vegetation_ward_crosscheck.csv", index=False)
    (OUTPUTS / "vegetation" / "crosscheck.json").write_text(json.dumps(dict(
        groups={k: dict(px=int(v.sum()), ha=ha(k)) for k, v in m.items()}, tree23_ha=float(tree23.sum() / 100), pixel=pixel, ward=ward, verdict=verdict), indent=1))
    return verdict


if __name__ == "__main__":
    from src.wardmap import memory_guard
    memory_guard(3.5)
    run()
