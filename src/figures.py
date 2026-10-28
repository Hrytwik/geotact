"""Chapter figures for docs/figures/ (Springer: maps and plots as 600 dpi TIFF, the diagram also as EPS; a PNG preview of each).
Palettes were checked with the dataviz validator: twin and intervention types use three categorical hues that pass all-pairs
colour-vision checks; the six land-use classes cannot pass all-pairs on one map (documented limit past three hues), so the
land-use figure adds one small panel per class as a second encoding."""
import json

import numpy as np

INK, INK2, MUTED, SURFACE, GREY = "#0b0b0b", "#52514e", "#a3a29c", "#fcfcfb", "#e2e1dc"
LANDUSE = ["#2a78d6", "#00691f", "#eda100", "#6fd2b0", "#e34948", "#8a8a8a"]      # Water, Trees, Cropland, Grass/Scrub, Built-up, Bare
LU_NAMES = ["Water", "Trees", "Cropland", "Grass/Scrub", "Built-up", "Bare ground"]
THREE = ["#2a78d6", "#008300", "#eda100"]                                           # water/wetland, green/corridor, farm/garden
DIV_LOSS, DIV_MID, DIV_GAIN = "#c0392b", "#f0efec", "#2a78d6"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
WIDTH_IN = 6.5


def crop_window(mask, pad=20):
    """(r0, r1, c0, c1) of the mask's bounding box plus padding, clipped to the grid."""
    rr, cc = np.nonzero(mask)
    return (max(int(rr.min()) - pad, 0), min(int(rr.max()) + 1 + pad, mask.shape[0]),
            max(int(cc.min()) - pad, 0), min(int(cc.max()) + 1 + pad, mask.shape[1]))


def extent(tf, win):
    """imshow extent (left, right, bottom, top) of a (r0, r1, c0, c1) window on an affine grid."""
    r0, r1, c0, c1 = win
    return (tf.c + c0 * tf.a, tf.c + c1 * tf.a, tf.f + r1 * tf.e, tf.f + r0 * tf.e)


# ---------------------------------------------------------------- drawing helpers
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
                         "legend.frameon": False, "savefig.facecolor": "white", "figure.facecolor": "white"})


def _map_axes(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)


def _scalebar(ax, km=2, loc=None):
    """Scale bar (left) and north arrow (right) in a strip added below the map, so neither covers data."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    H = y1 - y0
    ax.set_ylim(y0 - 0.10 * H, y1)
    yb = y0 - 0.07 * H
    x = x0 + 0.03 * (x1 - x0)
    ax.plot([x, x + km * 1000], [yb, yb], color=INK, lw=1.5, solid_capstyle="butt", clip_on=False)
    ax.text(x + km * 500, yb + 0.012 * H, f"{km} km", ha="center", va="bottom", fontsize=6)
    xa = x0 + 0.95 * (x1 - x0)
    ax.annotate("N", xy=(xa, y0 - 0.005 * H), xytext=(xa, y0 - 0.085 * H), ha="center", va="bottom", fontsize=7,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))


def _boundary(ax, gdf_mun, lw=0.8):
    gdf_mun.boundary.plot(ax=ax, color=INK, lw=lw)


def _save(fig, name, out, eps=False):
    kw = dict(bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **kw)
    fig.savefig(out / f"{name}.png", dpi=200, **kw)
    if eps:
        fig.savefig(out / f"{name}.eps", **kw)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"saved {name}", flush=True)


def _layers():
    import geopandas as gpd

    from src.config import INTERIM
    wm = INTERIM / "wardmap.gpkg"
    return {k: gpd.read_file(wm, layer=k) for k in ("municipal_poly", "wards", "drain_line", "river", "canal", "wb")}


def _grid():
    import rasterio

    from src import grid
    from src.config import OUTPUTS
    mask = grid.study_mask()
    with rasterio.open(OUTPUTS / "landuse" / "landuse_2023_corr.tif") as r:
        return mask, r.transform, r.read(1)


# ---------------------------------------------------------------- figures
def fig_study_area(out, L):
    import geopandas as gpd
    import matplotlib.pyplot as plt

    from src.config import INTERIM
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.2))
    L["wards"].plot(ax=ax, facecolor="#f0efec", edgecolor=MUTED, lw=0.3)
    for k in ("river", "canal"):
        L[k].plot(ax=ax, facecolor="#9ec5f4", edgecolor="none")
    L["wb"][L["wb"].geom_type.isin(["Polygon", "MultiPolygon"])].plot(ax=ax, facecolor="#9ec5f4", edgecolor="none")
    L["drain_line"].plot(ax=ax, color="#2a78d6", lw=0.6)
    _boundary(ax, L["municipal_poly"], 1.0)
    h = gpd.read_file(INTERIM / "hotspots.gpkg", layer="hotspots")
    ax.scatter(h.geometry.x, h.geometry.y, marker="^", s=22, facecolor=INK, edgecolor="white", lw=0.5, zorder=5)
    for hs, g in h.groupby("hotspot"):
        ax.annotate(str(hs).replace("WL", ""), (g.geometry.x.mean(), g.geometry.y.mean()), xytext=(4, -1), textcoords="offset points",
                    fontsize=5.5, color=INK, zorder=6, va="center", bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.75))
    _map_axes(ax); _scalebar(ax)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.0, label="BMC boundary"), Patch(facecolor="#f0efec", edgecolor=MUTED, lw=0.3, label="Wards (67)"),
                       Patch(facecolor="#9ec5f4", label="River, canals, water bodies"), Line2D([], [], color="#2a78d6", lw=0.6, label="Drains"),
                       Line2D([], [], marker="^", ls="", color=INK, markersize=5, label="Waterlogging hotspots (WL1-WL16)")],
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6)
    _save(fig, "fig01_study_area", out)


def fig_workflow(out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

    def box(x, y, w, h, title, body, fill="#f0efec"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=MUTED, lw=0.6))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=6.8, weight="bold")
        ax.text(x + w / 2, y + h - 6.3, body, ha="center", va="top", fontsize=5.6, color=INK2, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))

    box(1, 30, 20, 27, "Open data", "Sentinel-2 / Sentinel-1\nCopernicus GLO-30 DEM\nEsri LULC, WorldCover\nLandsat LST, ERA5\nGBIF, OpenStreetMap\nBMC ward map")
    box(26, 41, 22, 16, "Land use", "U-Net ensemble +\ncorrection model\n(300 hand labels)", "#e6f0fb")
    box(26, 22, 22, 16, "Vegetation change", "8-season NDVI;\ngreenness-drop index", "#e6f4e6")
    box(26, 3, 22, 16, "Flood susceptibility", "radar water frequency;\nfield-survey models", "#e6f0fb")
    box(53, 22, 20, 35, "Graph twin", "water, green, farm\nand habitat patches\nas nodes; connectivity,\nadjacency and drainage\nedges; yearly history;\nrebuilt per clear\nsatellite window")
    box(78, 41, 21, 16, "Scenarios", "wetlands, gardens,\ncorridors -> runoff,\nfood, habitat, heat")
    box(78, 22, 21, 16, "Optimisation", "NSGA-II, 4 objectives,\n150 ha budget,\narchive of plans")
    box(53, 3, 46, 14, "Evaluation (rules fixed before results)", "design-based accuracy on hand labels; spatial cross-validation;\nhotspot test; convergence and reproducibility checks", "#fdf3dc")
    for y in (49, 30, 11):
        arrow(21, 43.5, 26, y)
    for y in (49, 30, 11):
        arrow(48, y, 53, 39.5 if y != 11 else 30)
    arrow(73, 45, 78, 49); arrow(73, 33, 78, 30)
    arrow(88.5, 41, 88.5, 38)
    _save(fig, "fig02_workflow", out, eps=True)


def fig_landuse(out, mask, tf, lu):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    win = crop_window(mask); r0, r1, c0, c1 = win
    sub = np.where(mask, lu, 255)[r0:r1, c0:c1].astype(float)
    sub[sub == 255] = np.nan
    ext = extent(tf, win)
    fig = plt.figure(figsize=(WIDTH_IN, 4.6))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.5, 1.5, 1, 1], wspace=0.04, hspace=0.12)
    ax = fig.add_subplot(gs[:, :2])
    ax.imshow(sub, cmap=ListedColormap(LANDUSE), vmin=-0.5, vmax=5.5, extent=ext, interpolation="nearest")
    _map_axes(ax); _scalebar(ax)
    ax.set_title("(a) Land use, product map (2023)", fontsize=7, loc="left")
    ax.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(LANDUSE, LU_NAMES)], loc="upper left", fontsize=5.8, ncol=1)
    for k in range(6):
        a = fig.add_subplot(gs[k // 2, 2 + k % 2])
        layer = np.where(np.isnan(sub), np.nan, (sub == k).astype(float))
        a.imshow(layer, cmap=ListedColormap([GREY, LANDUSE[k]]), vmin=0, vmax=1, extent=ext, interpolation="nearest")
        _map_axes(a)
        share = float(np.nanmean(layer)) * 100
        a.set_title(f"{LU_NAMES[k]} ({share:.0f}%)", fontsize=6, loc="left", pad=2)
    fig.text(0.62, 0.07, "(b) Each class on its own (second encoding for colour-vision deficiency)", fontsize=6, color=INK2)
    _save(fig, "fig03_landuse", out)


def fig_vegetation(out, mask, tf):
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch

    from src import landuse as lu_mod
    from src.config import OUTPUTS
    from src.vegetation_crosscheck import ndvi_change
    with rasterio.open(OUTPUTS / "vegetation" / "vegetation_loss_2022_2026.tif") as r:
        loss = r.read(1)
    change = ndvi_change(lu_mod.ndvi_seasons()).astype("float32")
    win = crop_window(mask); r0, r1, c0, c1 = win; ext = extent(tf, win)
    pop = (loss != 255)
    ch = np.where(pop & mask, change, np.nan)[r0:r1, c0:c1]
    base = np.where(mask, 0.0, np.nan)[r0:r1, c0:c1]
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.6))
    cmap = LinearSegmentedColormap.from_list("div", [DIV_LOSS, DIV_MID, DIV_GAIN])
    axs[1].imshow(base, cmap=ListedColormap([GREY]), extent=ext, interpolation="nearest")
    im = axs[0].imshow(ch, cmap=cmap, vmin=-0.3, vmax=0.3, extent=ext, interpolation="nearest")
    cb = fig.colorbar(im, ax=axs[0], orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.8)
    cb.set_label("NDVI change, 2021-23 to 2024-26 (vegetated land)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    axs[0].set_title("(a) Greenness change (white: not vegetated in 2023)", fontsize=7, loc="left")
    L_mun = _layers()["municipal_poly"]
    for a in axs:
        L_mun.boundary.plot(ax=a, color=INK, lw=0.6)
    fl = np.where(mask & pop, (loss == 1).astype(float), np.nan)[r0:r1, c0:c1]
    axs[1].imshow(fl, cmap=ListedColormap(["#9cc99c", DIV_LOSS]), vmin=0, vmax=1, extent=ext, interpolation="nearest")
    axs[1].set_title("(b) Flagged vegetation loss", fontsize=7, loc="left")
    axs[1].legend(handles=[Patch(facecolor=DIV_LOSS, label="Flagged loss (482.6 ha)"), Patch(facecolor="#9cc99c", label="Vegetated, no flag"),
                           Patch(facecolor=GREY, label="Not vegetated in 2023")], loc="upper left", bbox_to_anchor=(0.0, 0.0), fontsize=5.5, ncol=3)
    for a in axs:
        _map_axes(a)
    _scalebar(axs[0]); _scalebar(axs[1])
    _save(fig, "fig04_vegetation", out)


def fig_flood(out, L):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from rasterio import features

    from src.config import INTERIM, OUTPUTS
    with rasterio.open(OUTPUTS / "flood" / "susceptibility_30m.tif") as r:
        tf, shape = r.transform, r.shape
    score = np.load(OUTPUTS / "flood_cnn" / "r3_scores_logistic.npy")
    m30 = features.rasterize([(g, 1) for g in L["municipal_poly"].geometry], out_shape=shape, transform=tf).astype(bool)
    win = crop_window(m30, pad=8); r0, r1, c0, c1 = win; ext = extent(tf, win)
    s = np.where(m30 & np.isfinite(score), score, np.nan)[r0:r1, c0:c1]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.0))
    im = ax.imshow(s, cmap=LinearSegmentedColormap.from_list("seq", SEQ), extent=ext, interpolation="nearest",
                   vmin=np.nanpercentile(s, 2), vmax=np.nanpercentile(s, 98))
    _boundary(ax, L["municipal_poly"], 0.8)
    p = gpd.read_file(INTERIM / "panigrahi2026_fig3_points.gpkg")
    fl = p[p.cls != "non_flood"]; nf = p[p.cls == "non_flood"]
    ax.scatter(fl.geometry.x, fl.geometry.y, s=7, marker="o", facecolor="white", edgecolor=INK, lw=0.4, zorder=5)
    ax.scatter(nf.geometry.x, nf.geometry.y, s=8, marker="s", facecolor=MUTED, edgecolor=INK, lw=0.3, zorder=5)
    h = gpd.read_file(INTERIM / "hotspots.gpkg", layer="hotspots")
    ax.scatter(h.geometry.x, h.geometry.y, marker="^", s=24, facecolor="#eda100", edgecolor=INK, lw=0.5, zorder=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("Flood susceptibility (field-survey logistic model score)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=INK, markersize=4, label="Field-surveyed flood sites (129)"),
                       Line2D([], [], marker="s", ls="", mfc=MUTED, mec=INK, markersize=4, label="Field-surveyed non-flood sites (25)"),
                       Line2D([], [], marker="^", ls="", mfc="#eda100", mec=INK, markersize=5, label="Sanitation-plan hotspots (16 hotspots, 19 points)")],
              loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=6)
    _map_axes(ax); _scalebar(ax)
    _save(fig, "fig05_flood", out)


def fig_twin(out, L):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from src.config import OUTPUTS
    n = gpd.read_file(OUTPUTS / "twin_v2" / "nodes.gpkg").rename(columns={"type": "kind"})   # .type is the geometry type in geopandas
    e = pd.read_csv(OUTPUTS / "twin_v2" / "edges.csv")
    xy = n.set_index("id")[["x", "y"]]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.0))
    L["municipal_poly"].plot(ax=ax, facecolor="#f7f6f3", edgecolor="none")
    order = {"water": THREE[0], "green": THREE[1], "farm": THREE[2]}
    for t, c in order.items():
        n[n["kind"] == t].plot(ax=ax, facecolor=c, edgecolor="none", alpha=0.85)
    styles = {"connectivity": ("#7d7c78", 0.25, 0.35), "adjacent": (INK, 0.5, 0.8), "drainage": ("#104281", 0.7, 0.9)}
    for kind, (col, lw, al) in styles.items():
        k = e[e.kind == kind]
        seg = [((xy.at[u, "x"], xy.at[u, "y"]), (xy.at[v, "x"], xy.at[v, "y"])) for u, v in zip(k.u, k.v) if u in xy.index and v in xy.index]
        ax.add_collection(LineCollection(seg, colors=col, linewidths=lw, alpha=al, zorder=4))
    ax.scatter(xy.x, xy.y, s=1.5, color=INK, zorder=5)
    _boundary(ax, L["municipal_poly"], 0.8)
    cnt = n["kind"].value_counts().to_dict(); ec = e.kind.value_counts().to_dict()
    ax.legend(handles=[Patch(facecolor=THREE[0], label=f"Water patches ({cnt.get('water', 0)})"),
                       Patch(facecolor=THREE[1], label=f"Green patches ({cnt.get('green', 0)})"),
                       Patch(facecolor=THREE[2], label=f"Farm patches ({cnt.get('farm', 0)})"),
                       Line2D([], [], color="#7d7c78", lw=0.8, label=f"Connectivity edges ({ec.get('connectivity', 0):,})"),
                       Line2D([], [], color=INK, lw=0.8, label=f"Adjacency edges ({ec.get('adjacent', 0)})"),
                       Line2D([], [], color="#104281", lw=0.8, label=f"Drainage edges, downhill ({ec.get('drainage', 0)})")],
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6)
    _map_axes(ax); _scalebar(ax)
    _save(fig, "fig06_twin_graph", out)


def _knee_plan():
    """Exact-front knee plan of T19b as a class map (same knee definition as optimise_convergence.headlines)."""
    import pandas as pd

    from src import optimise as op
    from src.config import OUTPUTS
    R = pd.read_csv(OUTPUTS / "optimise_v3_t19b" / "front.csv")
    M = R[R.on_exact_front].reset_index(drop=True)
    Fm = np.column_stack([M.runoff_m3, -M.food_gj, -M.eca_gain_ha, M.heat_c])
    norm = (Fm - Fm.min(axis=0)) / np.where(np.ptp(Fm, axis=0) > 0, np.ptp(Fm, axis=0), 1)
    k = M.iloc[int(np.argmin(np.linalg.norm(norm, axis=1)))]
    energy, heat = op.v3_inputs()
    data, ids, d, cand, blocks, base, built_px = op.precompute(150.0, map_name="landuse_2023_corr.tif", suffix="_v3", energy=energy, heat=heat)
    x = np.array(list(map(int, k.x.split(","))))
    return op.solution_map(x, ids, d["cm"], cand, blocks), k


def fig_scenarios(out, mask, tf, L):
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from src import scenarios as sc
    from src.config import OUTPUTS
    with rasterio.open(OUTPUTS / "scenarios_v3" / "combined.tif") as r:
        comb = r.read(1)
    knee, k = _knee_plan()
    win = crop_window(mask); r0, r1, c0, c1 = win; ext = extent(tf, win)
    code = {sc.WETLAND: 1, sc.CORRIDOR: 2, sc.GARDEN: 3}

    def show(a, m, title):
        v = np.where(mask, 0, np.nan)
        for src_val, dst in code.items():
            v = np.where(mask & (m == src_val), dst, v)
        a.imshow(v[r0:r1, c0:c1], cmap=ListedColormap([GREY] + THREE), vmin=-0.5, vmax=3.5, extent=ext, interpolation="nearest")
        L["municipal_poly"].boundary.plot(ax=a, color=INK, lw=0.6)
        _map_axes(a)
        a.set_title(title, fontsize=7, loc="left")

    fig, axs = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.7))
    show(axs[0], comb, "(a) Hand-made combined scenario (151 ha)")
    show(axs[1], knee, f"(b) Optimised balanced plan ({k.area_ha:.0f} ha)")
    _scalebar(axs[0]); _scalebar(axs[1])
    axs[0].legend(handles=[Patch(facecolor=THREE[0], label="Wetland restoration"), Patch(facecolor=THREE[1], label="Tree corridor"),
                           Patch(facecolor=THREE[2], label="Community garden")], loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=6)
    _save(fig, "fig07_scenarios", out)


def fig_tradeoffs(out):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap

    from src.config import OUTPUTS
    R = pd.read_csv(OUTPUTS / "optimise_v3_t19b" / "front.csv")
    M = R[R.on_exact_front].reset_index(drop=True)
    b = json.loads((OUTPUTS / "scenarios_v3" / "metrics.json").read_text())["scenarios"]
    base, comb = b["baseline"], b["combined"]
    runoff = 100 * (M.runoff_m3 / base["runoff_m3"] - 1); food = M.food_gj - base["food_gj"]; hab = M.eca_gain_ha; heat = M.heat_c
    Fm = np.column_stack([M.runoff_m3, -M.food_gj, -M.eca_gain_ha, M.heat_c])
    norm = (Fm - Fm.min(axis=0)) / np.where(np.ptp(Fm, axis=0) > 0, np.ptp(Fm, axis=0), 1)
    kn = int(np.argmin(np.linalg.norm(norm, axis=1)))
    c_run = 100 * (comb["runoff_m3"] / base["runoff_m3"] - 1); c_food = comb["food_gj"] - base["food_gj"]; c_hab = comb["eca_ha"] - base["eca_ha"]
    cmap = LinearSegmentedColormap.from_list("seq", SEQ[::-1])          # darker = more cooling
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.0), constrained_layout=True)
    for a, y, cy, lab in ((axs[0], food, c_food, "Food energy change (GJ per year)"), (axs[1], hab, c_hab, "Connected habitat gain (ha)")):
        sc_ = a.scatter(runoff, y, c=heat, cmap=cmap, s=3, lw=0, rasterized=True)
        a.scatter([runoff[kn]], [y[kn]], marker="*", s=70, color="#eda100", edgecolor=INK, lw=0.6, zorder=5, label="Balanced (knee) plan")
        a.scatter([c_run], [cy], marker="D", s=22, color="white", edgecolor=INK, lw=0.8, zorder=5, label="Hand-made combined plan")
        a.scatter([0], [0], marker="P", s=40, color=INK, zorder=5, label="Today")
        a.axhline(0, color=MUTED, lw=0.5, zorder=0); a.axvline(0, color=MUTED, lw=0.5, zorder=0)
        a.set_xlabel("Storm runoff change (%)"); a.set_ylabel(lab)
        a.grid(color="#efeeea", lw=0.4); a.set_axisbelow(True)
    axs[0].legend(loc="upper left", fontsize=5.8)
    cb = fig.colorbar(sc_, ax=axs, fraction=0.04, pad=0.01)
    cb.set_label("Residents' heat exposure change (°C)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    axs[0].set_title("(a) Runoff vs food", fontsize=7, loc="left"); axs[1].set_title("(b) Runoff vs habitat", fontsize=7, loc="left")
    _save(fig, "fig08_tradeoffs", out)


def run(only=None):
    from src.config import ROOT
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    _style()
    L = _layers()
    mask, tf, lu = _grid()
    jobs = {"study_area": lambda: fig_study_area(out, L), "workflow": lambda: fig_workflow(out),
            "landuse": lambda: fig_landuse(out, mask, tf, lu), "vegetation": lambda: fig_vegetation(out, mask, tf),
            "flood": lambda: fig_flood(out, L), "twin": lambda: fig_twin(out, L),
            "scenarios": lambda: fig_scenarios(out, mask, tf, L), "tradeoffs": lambda: fig_tradeoffs(out)}
    for name, job in jobs.items():
        if only is None or name in only:
            job()


if __name__ == "__main__":
    import sys

    from src.wardmap import memory_guard
    memory_guard(4.0)
    run(set(sys.argv[1:]) or None)
