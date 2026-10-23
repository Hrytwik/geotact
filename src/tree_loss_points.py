"""Hand check of the U-Net tree-loss map (vegetation claim). 40 blind points in Google Earth: 25 places both U-Net runs call
tree (2023) -> built-up (2025-26), 8 'stayed tree' and 7 'stayed built-up' controls, shuffled so the checker cannot tell which is
which. The key is kept in a separate private file. Same format as the 300 land-use points (src/validation_points.py)."""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

SEED = 2027
PLAN = {"converted": 25, "stable_tree": 8, "stable_built": 7}
CLASSES = ["Water", "Trees", "Cropland", "Grass/Scrub", "Built-up", "Bare ground"]
THRESHOLDS = (6, 4, 2, 0)                    # core rule: neighbours (of 8) that must belong to the same group, relaxed if too few


def core_mask(mask, min_same):
    """Pixels of the group with at least min_same of their 8 neighbours also in the group."""
    mask = np.asarray(mask, bool)
    k = np.ones((3, 3)); k[1, 1] = 0
    return mask & (ndi.convolve(mask.astype(float), k, mode="constant") >= min_same)


def pick_points(cand, n, rng, min_dist_px=10):
    """Up to n random candidate pixels, none closer than min_dist_px (pixels) to another."""
    idx = np.flatnonzero(np.asarray(cand).ravel())
    rr, cc = np.unravel_index(rng.permutation(idx), cand.shape)
    got = []
    for r, c in zip(rr, cc):
        if all(np.hypot(r - a, c - b) >= min_dist_px for a, b in got):
            got.append((int(r), int(c)))
            if len(got) == n:
                break
    return np.array(got, dtype=int).reshape(-1, 2)


def choose(groups, plan, rng, min_dist_px=10):
    out = []
    for name, n in plan.items():
        picks, used = np.zeros((0, 2), int), 0
        for t in THRESHOLDS:
            cand = core_mask(groups[name], t) if t > 0 else np.asarray(groups[name], bool)
            picks, used = pick_points(cand, n, rng, min_dist_px), t
            if len(picks) >= n:
                break
        out += [dict(group=name, row=int(r), col=int(c), relax=used) for r, c in picks]
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


def score(sheet, key):
    """Join the filled sheet (id, trees_in_2023, now_covers) to the private key and score each group."""
    d = key.merge(sheet[["id", "trees_in_2023", "now_covers"]], on="id", how="left")
    ans = lambda s: s.fillna("").astype(str)
    d["trees_in_2023"], d["now_covers"] = ans(d.trees_in_2023), ans(d.now_covers)
    d["unsure"] = (d.trees_in_2023 == "Unsure") | (d.now_covers == "Unsure")
    d["answered"] = (d.trees_in_2023 != "") & (d.now_covers != "")
    res = {}
    c = d[d.group == "converted"]
    c_ans = c[c.answered]
    ok = ((c_ans.trees_in_2023 == "Yes") & (c_ans.now_covers == "Built-up")).sum()
    loss = ((c_ans.trees_in_2023 == "Yes") & (c_ans.now_covers != "Trees") & (c_ans.now_covers != "Unsure")).sum()
    n_sure = int((~c_ans.unsure).sum())
    ok_sure = ((c_ans.trees_in_2023 == "Yes") & (c_ans.now_covers == "Built-up") & ~c_ans.unsure).sum()
    res["converted"] = dict(n=len(c), answered=len(c_ans), unsure=int(c_ans.unsure.sum()),
                            tree_to_built_lower=float(ok / len(c)) if len(c) else float("nan"),
                            tree_to_built_upper=float(ok_sure / n_sure) if n_sure else float("nan"),
                            tree_loss_any_lower=float(loss / len(c)) if len(c) else float("nan"),
                            tree_to_built_ci_lower_case=wilson(int(ok), len(c)), tree_to_built_ci_upper_case=wilson(int(ok_sure), n_sure))
    for g, (before, now) in (("stable_tree", ("Yes", "Trees")), ("stable_built", ("No", "Built-up"))):
        s = d[(d.group == g) & d.answered]
        res[g] = dict(n=int((d.group == g).sum()), answered=len(s), agree=int(((s.trees_in_2023 == before) & (s.now_covers == now)).sum()))
    return res


def build():
    import rasterio
    import simplekml
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from pyproj import Transformer

    from src import grid
    from src.config import CRS, OUTPUTS, ROOT
    from src.vegetation_crosscheck import conversion_masks
    rd = lambda n: rasterio.open(OUTPUTS / "landuse" / n).read(1)
    with rasterio.open(OUTPUTS / "landuse" / "landuse_2023_seed0.tif") as r:
        tf = r.transform
    m = conversion_masks(rd("landuse_2023_seed0.tif"), rd("landuse_2023_seed1.tif"), rd("landuse_2025_26_seed0.tif"), rd("landuse_2025_26_seed1.tif"), grid.study_mask())
    rng = np.random.default_rng(SEED)
    picks = choose({k: m[k] for k in PLAN}, PLAN, rng)
    order = rng.permutation(len(picks))
    to_wgs = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    rows = []
    for i, j in enumerate(order, 1):
        p = picks[j]
        x, y = tf * (p["col"] + 0.5, p["row"] + 0.5)
        lon, lat = to_wgs.transform(x, y)
        rows.append(dict(id=f"T{i:02d}", lat=round(lat, 6), lon=round(lon, 6), x_utm=x, y_utm=y, **p))
    df = pd.DataFrame(rows)
    out = ROOT / "validation"
    out.mkdir(exist_ok=True)
    df[["id", "group", "row", "col", "relax", "lat", "lon"]].to_csv(out / "tree_loss_key_private_do_not_open.csv", index=False)

    kml = simplekml.Kml(name="Bhubaneswar tree-loss check points")
    for p in df.itertuples():
        pt = kml.newpoint(name=p.id, coords=[(p.lon, p.lat)])
        pt.style.iconstyle.scale = 0.6
        sq = [to_wgs.transform(p.x_utm + dx, p.y_utm + dy) for dx, dy in [(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)]]
        pol = kml.newpolygon(name=f"{p.id} pixel", outerboundaryis=sq)
        pol.style.polystyle.fill = 0
        pol.style.linestyle.color = simplekml.Color.yellow
        pol.style.linestyle.width = 2
    kml.save(str(out / "tree_loss_points.kml"))

    wb = Workbook(); ws = wb.active; ws.title = "labels"
    ws.append(["id", "lat", "lon", "trees_in_2023", "now_covers", "imagery_date_2023", "imagery_date_now", "confidence", "notes"])
    for p in df.itertuples():
        ws.append([p.id, p.lat, p.lon, None, None, None, None, None, None])
    n = len(df) + 1
    for col, opts in (("D", "Yes,No,Unsure"), ("E", ",".join(CLASSES) + ",Unsure"), ("H", "Sure,Unsure")):
        dv = DataValidation(type="list", formula1='"' + opts + '"', allow_blank=True)
        ws.add_data_validation(dv); dv.add(f"{col}2:{col}{n}")
    for col, w in zip("ABCDEFGHI", [7, 11, 11, 14, 14, 16, 16, 12, 30]):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    guide = wb.create_sheet("how_to_check")
    for line in [
        "1. Open tree_loss_points.kml in Google Earth Pro (File > Open). Double-click a point (T01...) to fly to it.",
        "2. Look ONLY inside the small yellow square (10 m x 10 m); if a building edge touches it, judge the centre.",
        "3. Click the clock icon (Historical Imagery) and slide to the image closest to EARLY 2023 (Jan-Apr 2023 if it exists).",
        "   Question 1 (trees_in_2023): were there trees (canopy) in the square then? Yes / No / Unsure. Write the image date.",
        "4. Slide to the MOST RECENT image (late 2025 or 2026 if it exists).",
        "   Question 2 (now_covers): what covers most of the square now? Pick from the list. Write the image date.",
        "5. Mark confidence Sure or Unsure. Don't skip points. If there is no image near 2023, use the closest date and write it down.",
        "",
        "Classes: Water, Trees (canopy, orchards, forest), Cropland, Grass/Scrub, Built-up (buildings, roads, paved, construction), Bare ground.",
        "You will not be told which points the model thinks changed: that is deliberate. Some points did not change at all.",
    ]:
        guide.append([line])
    guide.column_dimensions["A"].width = 120
    wb.save(out / "tree_loss_labels.xlsx")
    return df


if __name__ == "__main__":
    from src.wardmap import memory_guard
    memory_guard(3.0)
    d = build()
    print(d.group.value_counts().to_dict(), "| relax thresholds used:", d.groupby("group").relax.agg(lambda s: sorted(set(s))).to_dict())
