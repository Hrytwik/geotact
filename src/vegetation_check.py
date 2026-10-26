"""T15: vegetation-loss map and its design-based accuracy check (protocol in docs/build_plan.md). The loss threshold is frozen from the
38 T5c points before any new label; a blind stratified random sample (loss / no loss) is labelled by the authors; Olofsson-style estimators
give precision, recall, overall accuracy and the loss area."""
import json

import numpy as np
import pandas as pd

VEGETATED = (1, 2, 3)                       # Trees, Cropland, Grass/Scrub
LOSS_COVER = {"Built-up", "Bare ground", "Water"}
PLAN = {"loss": 50, "no_loss": 50}
SEED = 2029


def youden_threshold(change, truth):
    """Cut on the NDVI change (loss = change <= tau) maximising TPR - FPR; tau sits midway to the next observed value; ties take the
    middle of the best plateau."""
    change, truth = np.asarray(change, float), np.asarray(truth, int)
    u = np.unique(change)
    pos, neg = truth == 1, truth == 0
    j = np.array([(change[pos] <= t).mean() - (change[neg] <= t).mean() for t in u])
    best = np.flatnonzero(np.isclose(j, j.max()))
    i = best[len(best) // 2]
    return float((u[i] + u[i + 1]) / 2) if i + 1 < len(u) else float(u[i])


def loss_map(change, vegetated, tau):
    pop = np.asarray(vegetated, bool) & np.isfinite(change)
    with np.errstate(invalid="ignore"):
        return pop & (np.nan_to_num(change, nan=np.inf) <= tau), pop


def draw_sample(loss, pop, plan, min_px, avoid=None, seed=SEED):
    """Random pixels per stratum (loss / no_loss inside the population), all at least min_px apart and from every `avoid` point."""
    from scipy import ndimage as ndi
    far = np.ones(loss.shape, bool)
    if avoid is not None and len(avoid):
        a = np.zeros(loss.shape, bool)
        av = np.asarray(avoid, int)
        ok = (av[:, 0] >= 0) & (av[:, 0] < loss.shape[0]) & (av[:, 1] >= 0) & (av[:, 1] < loss.shape[1])
        a[av[ok, 0], av[ok, 1]] = True
        far = ndi.distance_transform_edt(~a) >= min_px
    rng = np.random.default_rng(seed)
    got = []
    for name, n in plan.items():
        cand = np.flatnonzero((pop & far & (loss if name == "loss" else ~loss)).ravel())
        taken = 0
        for idx in rng.permutation(cand):
            r, c = divmod(int(idx), loss.shape[1])
            if all((r - a) ** 2 + (c - b) ** 2 >= min_px ** 2 for _, a, b in got):
                got.append((name, r, c)); taken += 1
                if taken == n:
                    break
    return pd.DataFrame(got, columns=["stratum", "row", "col"])


def truth(sheet):
    d = sheet.copy()
    d["before"] = d.vegetation_before.astype(str).str.strip().str.capitalize()
    lk = {c.lower(): c for c in ["Water", "Trees", "Cropland", "Grass/Scrub", "Built-up", "Bare ground", "Unsure"]}
    d["now"] = d.now_covers.astype(str).str.strip().str.lower().map(lk)
    d = d[d.before.isin(["Yes", "No"]) & d.now.notna() & (d.now != "Unsure")]
    return pd.DataFrame(dict(id=d.id.values, loss=((d.before == "Yes") & d.now.isin(LOSS_COVER)).astype(int).values))


def score(truth_df, key, W):
    from src import landuse_accuracy as la
    d = key.merge(truth_df, on="id")
    s = d.stratum.values
    mapped, ref = (s == "loss"), d.loss.values.astype(bool)
    return dict(n=int(len(d)), users_accuracy_loss=la.ratio_estimate(s, mapped & ref, mapped, W),
                producers_accuracy_loss=la.ratio_estimate(s, mapped & ref, ref, W),
                overall=la.stratified_accuracy(s, mapped == ref, W), loss_area_share=la.stratified_accuracy(s, ref, W))


# ---------------------------------------------------------------- real build
def build():
    import rasterio
    import simplekml
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from pyproj import Transformer

    from src import grid
    from src import landuse as lu
    from src import landuse_accuracy as la
    from src.config import CRS, OUTPUTS, ROOT
    from src.vegetation_crosscheck import ndvi_change
    from src.vegetation_dl import truth_from_sheet
    change = ndvi_change(lu.ndvi_seasons())
    sheet = pd.read_excel(ROOT / "validation" / "tree_loss_labels.xlsx", sheet_name="labels")
    key = pd.read_csv(ROOT / "validation" / "tree_loss_key_private_do_not_open.csv")
    t = truth_from_sheet(sheet).merge(key[["id", "row", "col"]], on="id")
    tau = youden_threshold(change[t.row.values, t.col.values], t.truth.values)
    with rasterio.open(OUTPUTS / "landuse" / "landuse_2023_corr.tif") as r:
        cm, tf, meta = r.read(1), r.transform, r.meta.copy()
    loss, pop = loss_map(change, np.isin(cm, VEGETATED) & grid.study_mask(), tau)
    W = {"loss": float(loss.sum() / pop.sum()), "no_loss": float((pop & ~loss).sum() / pop.sum())}
    d, _ = la.load_inputs()
    ar, ac = rasterio.transform.rowcol(tf, d.x_utm.values, d.y_utm.values)
    avoid = np.vstack([np.column_stack([ar, ac]), key[["row", "col"]].to_numpy()])
    pts = draw_sample(loss, pop, PLAN, min_px=10, avoid=avoid)
    pts = pts.sample(frac=1, random_state=SEED).reset_index(drop=True)
    pts.insert(0, "id", [f"V{i + 1:03d}" for i in range(len(pts))])
    to_wgs = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    xy = [tf * (c + 0.5, r + 0.5) for r, c in zip(pts.row, pts.col)]
    pts["lon"], pts["lat"] = zip(*[to_wgs.transform(x, y) for x, y in xy])
    out = ROOT / "validation" / "vegetation_check"
    out.mkdir(parents=True, exist_ok=True)
    pts.to_csv(out / "vegetation_key_private_do_not_open.csv", index=False)
    kml = simplekml.Kml(name="Vegetation change check points")
    wb = Workbook(); ws = wb.active; ws.title = "labels"
    ws.append(["id", "lat", "lon", "vegetation_before", "now_covers", "imagery_date_before", "imagery_date_now", "confidence", "notes"])
    for p, (x, y) in zip(pts.itertuples(), xy):
        kml.newpoint(name=p.id, coords=[(p.lon, p.lat)]).style.iconstyle.scale = 0.6
        pol = kml.newpolygon(name=f"{p.id} pixel", outerboundaryis=[to_wgs.transform(x + dx, y + dy) for dx, dy in [(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)]])
        pol.style.polystyle.fill = 0; pol.style.linestyle.color = simplekml.Color.yellow; pol.style.linestyle.width = 2
        ws.append([p.id, round(p.lat, 6), round(p.lon, 6), None, None, None, None, None, None])
    n = len(pts) + 1
    for col, opts in (("D", "Yes,No,Unsure"), ("E", "Water,Trees,Cropland,Grass/Scrub,Built-up,Bare ground,Unsure"), ("H", "Sure,Unsure")):
        dv = DataValidation(type="list", formula1='"' + opts + '"', allow_blank=True); ws.add_data_validation(dv); dv.add(f"{col}2:{col}{n}")
    g = wb.create_sheet("how_to_check")
    for line in ["1. Open vegetation_points.kml in Google Earth Pro; double-click a point (V001...) to fly to it; judge only the yellow 10 m square.",
                 "2. Historical imagery (clock icon): go to 2022 or early 2023. vegetation_before: were there trees, crops or grass? Yes / No / Unsure.",
                 "3. Go to the most recent image. now_covers: what covers most of the square now? Write both image dates.",
                 "Points were picked at random; some changed, many did not. You are not told which."]:
        g.append([line])
    kml.save(str(out / "vegetation_points.kml")); wb.save(out / "vegetation_labels.xlsx")
    meta.update(dtype="uint8", nodata=255, compress="deflate")
    m = np.full(cm.shape, 255, "uint8"); m[pop] = 0; m[loss] = 1
    with rasterio.open(OUTPUTS / "vegetation" / "vegetation_loss_2022_2026.tif", "w", **meta) as dst:
        dst.write(m, 1)
    info = dict(tau=tau, n_train_points=int(len(t)), weights=W, loss_ha=float(loss.sum() / 100), population_ha=float(pop.sum() / 100),
                sample=pts.stratum.value_counts().to_dict())
    (OUTPUTS / "vegetation" / "vegetation_check.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    return info


def run_score():
    from src.config import OUTPUTS, ROOT
    out = ROOT / "validation" / "vegetation_check"
    W = json.loads((OUTPUTS / "vegetation" / "vegetation_check.json").read_text())["weights"]
    r = score(truth(pd.read_excel(out / "vegetation_labels.xlsx", sheet_name="labels")), pd.read_csv(out / "vegetation_key_private_do_not_open.csv"), W)
    (OUTPUTS / "vegetation" / "vegetation_check_scores.json").write_text(json.dumps(r, indent=1, default=float))
    return r


if __name__ == "__main__":
    import sys
    print(run_score() if "--score" in sys.argv else build())
