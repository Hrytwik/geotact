"""T10: multi-objective choice of intervention sites with NSGA-II (design registered in docs/build_plan.md).

Each eligible 1 ha block takes one option: 0 none, 1 wetland, 2 garden, 3 corridor. Runoff and food change are exact sums over
blocks; connectivity uses a first-order surrogate (sum of single-block gains) during the search and is re-evaluated exactly on
the final fronts. Total converted area is limited by a budget."""
from dataclasses import dataclass

import numpy as np

OPTIONS = ("none", "wetland", "garden", "corridor")


@dataclass
class SiteData:
    allowed: np.ndarray          # (n_blocks, 4) bool, column 0 (none) always True
    area: np.ndarray             # (n_blocks, 4) ha converted by each option
    d_runoff: np.ndarray         # (n_blocks, 4) m3 change in storm runoff
    d_food: np.ndarray           # (n_blocks, 4) ha change in food-potential area
    d_conn: np.ndarray           # (n_blocks, 4) ha gain in equivalent connected habitat (block converted alone)
    budget: float                # ha
    base_runoff: float
    base_food: float
    d_heat: np.ndarray = None    # optional (n_blocks, 4): change in residents' mean summer surface temperature (deg C), minimised


def _take(arr, X):
    X = np.asarray(X, dtype=int)
    return arr[np.arange(arr.shape[0])[None, :], X]


def objectives(d, X):
    """Minimisation form: [runoff, -food area, -connectivity gain] for each row of X."""
    cols = [d.base_runoff + _take(d.d_runoff, X).sum(axis=1), -(d.base_food + _take(d.d_food, X).sum(axis=1)),
            -_take(d.d_conn, X).sum(axis=1)]
    if d.d_heat is not None:
        cols.append(_take(d.d_heat, X).sum(axis=1))
    return np.column_stack(cols)


def area_used(d, X):
    return _take(d.area, X).sum(axis=1)


def pareto_mask(F):
    F = np.asarray(F, dtype=float)
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        keep[i] = not dom.any()
    return keep


def repair(d, X, rng):
    X = np.array(X, dtype=int, copy=True)
    n = np.arange(X.shape[1])
    for row in X:
        row[~d.allowed[n, row]] = 0
        used = d.area[n, row].sum()
        while used > d.budget + 1e-9:
            on = np.flatnonzero(row)
            k = on[rng.integers(len(on))]
            used -= d.area[k, row[k]]
            row[k] = 0
    return X


def hypervolume(F, lo, hi, ref=1.1):
    from pymoo.indicators.hv import HV
    span = np.where(np.asarray(hi) - np.asarray(lo) > 0, np.asarray(hi) - np.asarray(lo), 1.0)
    return float(HV(ref_point=np.full(np.asarray(F).shape[1], ref)).do((np.asarray(F) - lo) / span))


def archive_update(AX, AF, X, F):
    """External non-dominated archive: merge new solutions, drop duplicates (same decision vector) and dominated rows."""
    X, F = np.vstack([AX, np.asarray(X, int)]), np.vstack([AF, np.asarray(F, float)])
    X, i = np.unique(X, axis=0, return_index=True)
    F = F[i]
    keep = pareto_mask(F)
    return X[keep], F[keep]


def run_nsga(d, pop=200, gens=300, seed=0, archive_every=None, return_final=False):
    """NSGA-II on the site problem. With archive_every = k, the feasible population is merged into an external non-dominated
    archive every k generations (and at the end) and the archive is returned instead of the final population (T19b)."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.core.mutation import Mutation
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from pymoo.core.sampling import Sampling
    from pymoo.operators.crossover.ux import UniformCrossover
    from pymoo.optimize import minimize

    n = d.allowed.shape[0]
    fallback = np.random.default_rng(seed + 7919)                                 # used only if pymoo passes no generator
    rng_of = lambda kw: kw.get("random_state") or fallback                        # pymoo 0.6.2 hands operators a seeded Generator
    choices = [np.flatnonzero(d.allowed[i]) for i in range(n)]
    mean_area = float(d.area[:, 1:][d.allowed[:, 1:]].mean()) if d.allowed[:, 1:].any() else 1.0

    class SiteProblem(Problem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=4 if d.d_heat is not None else 3, n_ieq_constr=1, xl=0, xu=3, vtype=int)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = objectives(d, X)
            out["G"] = (area_used(d, X) - d.budget)[:, None]

    class SiteSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            rng = rng_of(kwargs)
            X = np.zeros((n_samples, n), dtype=int)
            for s in range(n_samples):
                p = min(1.0, d.budget / mean_area / n * rng.uniform(0.2, 1.5))
                for i in np.flatnonzero(rng.random(n) < p):
                    opts = choices[i][choices[i] > 0]
                    if len(opts):
                        X[s, i] = rng.choice(opts)
            return repair(d, X, rng)

    class SiteMutation(Mutation):
        def _do(self, problem, X, **kwargs):
            rng = rng_of(kwargs)
            X = np.array(X, dtype=int, copy=True)
            hit = rng.random(X.shape) < 1.0 / n
            for s, i in zip(*np.nonzero(hit)):
                X[s, i] = rng.choice(choices[i])
            return X

    class SiteRepair(Repair):
        def _do(self, problem, X, **kwargs):
            return repair(d, X, rng_of(kwargs))

    class Archive(Callback):
        def __init__(self):
            super().__init__()
            self.X, self.F = np.empty((0, n), int), np.empty((0, 4 if d.d_heat is not None else 3))

        def notify(self, algorithm):
            if algorithm.n_gen % archive_every == 0 or algorithm.n_gen >= gens:
                P = algorithm.pop
                ok = P.get("CV").ravel() <= 0
                self.X, self.F = archive_update(self.X, self.F, P.get("X")[ok].astype(int), P.get("F")[ok])

    algo = NSGA2(pop_size=pop, sampling=SiteSampling(), crossover=UniformCrossover(), mutation=SiteMutation(),
                 repair=SiteRepair(), eliminate_duplicates=True)
    cb = Archive() if archive_every else None
    res = minimize(SiteProblem(), algo, ("n_gen", gens), seed=seed, verbose=False, **({"callback": cb} if cb is not None else {}))
    if cb is not None:
        Xp = np.atleast_2d(res.X).astype(int)
        X, _ = archive_update(cb.X, cb.F, Xp, objectives(d, Xp))
        return (X, objectives(d, X), Xp, objectives(d, Xp)) if return_final else (X, objectives(d, X))
    X = np.atleast_2d(res.X).astype(int)
    return X, objectives(d, X)


def exact_row(dd, x, ids, d, cand, blocks, base, built_px, energy=None, heat=None):
    """Exact map metrics of one plan x next to the search's additive predictions (one row of the front table)."""
    from src import scenarios as sc
    fc, fpc = ("food_gj", "food_pred_gj") if energy is not None else ("food_ha", "food_pred_ha")
    m = solution_map(x, ids, d["cm"], cand, blocks)
    ex = sc.metrics(m, d["mask"], d["osm"], built_px)
    f = objectives(dd, x[None])[0]
    a = dd.area[np.arange(len(x)), x]
    return dict(budget_ha=dd.budget, area_ha=float(area_used(dd, x[None])[0]),
                wetland_ha=float(a[x == 1].sum()), garden_ha=float(a[x == 2].sum()), corridor_ha=float(a[x == 3].sum()),
                runoff_m3=ex["runoff_m3"], runoff_pred_m3=f[0],
                **{fc: ex["food_ha"] if energy is None else food_gj(m, d["mask"], energy), fpc: -f[1]},
                eca_gain_ha=ex["eca_ha"] - base["eca_ha"], eca_gain_surrogate_ha=-f[2], open_space_share=ex["open_space_share"],
                **({} if heat is None else dict(heat_c=ht_exact(m, d["cm"], heat), heat_pred_c=f[3])),
                x=",".join(map(str, x)))


# ---------------------------------------------------------------- real run (T10)
def precompute(budget=150.0, cache=True, map_name="landuse_2023_seed0.tif", suffix="", energy=None, heat=None):
    """SiteData for the eligible blocks of T9: exact runoff and food changes, single-block connectivity gains (cached).
    energy = dict(rice_gj_ha, veg_gj_ha): food in gigajoules per year instead of hectares. heat = dict(beta, G, wsum): adds the heat column."""
    import json

    import pandas as pd
    import rasterio

    from src import scenarios as sc
    from src.config import OUTPUTS
    out = OUTPUTS / f"optimise{suffix}"
    out.mkdir(exist_ok=True)
    f = out / "site_data.npz"
    d = sc.inputs(map_name)
    cm, mask = d["cm"], d["mask"]
    cand = sc.candidates(cm, mask, d["flood"], d["near_water"], d["built_share"], d["riparian"])
    blocks = sc.block_ids(cm.shape, 10)
    nb = int(blocks.max()) + 1
    size = np.bincount(blocks.ravel(), minlength=nb)
    keys = ("wetland", "garden", "corridor"); newc = {"wetland": sc.WETLAND, "garden": sc.GARDEN, "corridor": sc.CORRIDOR}
    cnt = {k: np.bincount(blocks[cand[k]], minlength=nb) for k in keys}
    elig = {k: (cnt[k] >= 0.5 * size) & (cnt[k] > 0) for k in keys}
    ids = np.flatnonzero(elig["wetland"] | elig["garden"] | elig["corridor"])
    built_px = int(((cm == sc.BUILT) & mask).sum())
    base = sc.metrics(cm, mask, d["osm"], built_px)
    base_food = base["food_ha"] if energy is None else food_gj(cm, mask, energy)
    if cache and f.exists():
        z = np.load(f, allow_pickle=True)
        if np.array_equal(z["ids"], ids) and bool(z["energy"]) == (energy is not None) and bool(z["heat"]) == (heat is not None):
            data = SiteData(z["allowed"], z["area"], z["d_runoff"], z["d_food"], z["d_conn"], budget, float(base["runoff_m3"]), float(base_food),
                            d_heat=z["d_heat"] if heat is not None else None)
            return data, ids, d, cand, blocks, base, built_px
    n = len(ids)
    allowed = np.zeros((n, 4), bool); allowed[:, 0] = True
    area, d_runoff, d_food, d_conn, d_heat = (np.zeros((n, 4)) for _ in range(5))
    qC = {k: sc.scs_runoff_mm(100.0, v) for k, v in sc.CN["C"].items()}
    rows_of = {int(b): np.flatnonzero(blocks.ravel() == b) for b in ids}
    flat_cm = cm.ravel()
    for j, b in enumerate(ids):
        idx = rows_of[int(b)]
        for o, k in enumerate(keys, start=1):
            if not elig[k][b]:
                continue
            cells = idx[cand[k].ravel()[idx] & ~np.isin(flat_cm[idx], sc.NEVER_EDIT)]
            orig = flat_cm[cells]
            allowed[j, o] = True
            area[j, o] = len(cells) / 100.0
            dq = sum(qC[newc[k]] - qC[int(c)] for c in orig)                 # mm summed over cells
            d_runoff[j, o] = dq / 1000.0 * 100.0 - (0.3 * 100.0 * len(cells) if k == "wetland" else 0.0)
            n_garden, n_crop = (len(cells) if k == "garden" else 0), int((orig == sc.CROP).sum())
            d_food[j, o] = (n_garden / 100.0 - n_crop / 100.0) if energy is None else (n_garden * energy["veg_gj_ha"] - n_crop * energy["rice_gj_ha"]) / 100.0
            m2 = cm.copy(); m2.ravel()[cells] = newc[k]
            d_conn[j, o] = sc.metrics(m2, mask, d["osm"], built_px)["eca_ha"] - base["eca_ha"]
            if heat is not None:
                from src import heat as ht
                d_heat[j, o] = ht.exposure_change_fast(m2, cm, heat["beta"], heat["G"], heat["wsum"])
        if j % 100 == 0:
            print(f"  precompute {j}/{n}", flush=True)
    np.savez(f, ids=ids, allowed=allowed, area=area, d_runoff=d_runoff, d_food=d_food, d_conn=d_conn, d_heat=d_heat,
             energy=energy is not None, heat=heat is not None)
    return (SiteData(allowed, area, d_runoff, d_food, d_conn, budget, float(base["runoff_m3"]), float(base_food),
                     d_heat=d_heat if heat is not None else None), ids, d, cand, blocks, base, built_px)


def food_gj(classmap, mask, energy):
    """Food energy per year (GJ): rice land at the rice energy per hectare plus gardens at the vegetable energy per hectare."""
    from src import food
    from src import scenarios as sc
    m = np.asarray(classmap)[np.asarray(mask, bool)]
    return food.food_energy_gj(int((m == sc.CROP).sum()) / 100.0, int((m == sc.GARDEN).sum()) / 100.0, energy["rice_gj_ha"], energy["veg_gj_ha"])


def ht_exact(m, cm, heat):
    from src import heat as ht
    return ht.exposure_change_fast(m, cm, heat["beta"], heat["G"], heat["wsum"])


def v3_inputs(rice_yield=None):
    """Food energy factors and the adopted heat model for the four-objective run (T14); stops if the heat model was not adopted."""
    import json

    from src import food
    from src import grid
    from src import heat as ht
    from src.config import OUTPUTS
    hm = json.loads((OUTPUTS / "heat" / "heat_model.json").read_text())
    if not hm["adopted"]:
        raise SystemExit(f"heat model not adopted (spatial-CV R2 {hm['cv_r2']:.2f} < 0.30): the registered rule keeps heat out")
    fa = food.factors(rice_yield) if rice_yield else food.factors()
    _, _, pop = ht.inputs()
    w = np.where(grid.study_mask(), pop, 0.0).astype("float32")
    heat = dict(beta=np.array([hm["beta"][k] for k in ht.FEATURES]), G=ht.exposure_kernels(w), wsum=float(w.sum()))
    return dict(rice_gj_ha=fa["rice_gj_ha"], veg_gj_ha=fa["veg_gj_ha"]), heat


def solution_map(x, ids, cm, cand, blocks):
    from src import scenarios as sc
    newc = {1: (sc.WETLAND, "wetland"), 2: (sc.GARDEN, "garden"), 3: (sc.CORRIDOR, "corridor")}
    m = cm.copy()
    for o, (cls, k) in newc.items():
        ch = [int(b) for b in ids[x == o]]
        if ch:
            m = sc.apply_edit(m, cand[k], blocks, ch, cls)
    return m


def run(budgets=(150.0, 75.0, 300.0), seeds=(0, 1, 2, 3, 4), pop=200, gens=300, map_name="landuse_2023_seed0.tif", suffix="",
        energy=None, heat=None):
    import dataclasses
    import json

    import pandas as pd
    import rasterio

    from src import scenarios as sc
    from src.config import OUTPUTS
    out = OUTPUTS / f"optimise{suffix}"
    data, ids, d, cand, blocks, base, built_px = precompute(budgets[0], map_name=map_name, suffix=suffix, energy=energy, heat=heat)
    fc, fpc = ("food_gj", "food_pred_gj") if energy is not None else ("food_ha", "food_pred_ha")
    print(f"blocks {len(ids)} | allowed pairs {int(data.allowed[:, 1:].sum())} | base runoff {base['runoff_m3']:,.0f} food {data.base_food:.1f} {fc[5:]} ECA {base['eca_ha']:.1f}", flush=True)
    summary, all_rows = {}, []
    for budget in budgets:
        dd = dataclasses.replace(data, budget=budget)
        use_seeds = seeds if budget == budgets[0] else seeds[:1]
        fronts = {}
        for s in use_seeds:
            X, F = run_nsga(dd, pop, gens, s)
            fronts[s] = X
            print(f"  budget {budget:.0f} ha seed {s}: {len(X)} front solutions", flush=True)
        U = np.unique(np.vstack(list(fronts.values())), axis=0)
        R = pd.DataFrame([exact_row(dd, x, ids, d, cand, blocks, base, built_px, energy, heat) for x in U])
        Fe = np.column_stack([R.runoff_m3, -R[fc], -R.eca_gain_ha] + ([R.heat_c] if heat is not None else []))
        R["on_exact_front"] = pareto_mask(Fe)
        lo, hi = Fe[R.on_exact_front].min(axis=0), Fe[R.on_exact_front].max(axis=0)
        hv = {}
        for s, X in fronts.items():
            sel = R.x.isin([",".join(map(str, x)) for x in X]) & R.on_exact_front
            Fs = Fe[R.x.isin([",".join(map(str, x)) for x in X]).values]
            Fs = Fs[pareto_mask(Fs)]
            hv[int(s)] = hypervolume(Fs, lo, hi)
        corr = float(np.corrcoef(R.eca_gain_ha, R.eca_gain_surrogate_ha)[0, 1]) if len(R) > 2 else float("nan")
        summary[f"{int(budget)}"] = dict(n_union=len(R), n_exact_front=int(R.on_exact_front.sum()), hv_by_seed=hv,
                                         surrogate_vs_exact_eca_corr=corr, surrogate_minus_exact_eca_mean=float((R.eca_gain_surrogate_ha - R.eca_gain_ha).mean()),
                                         runoff_pred_vs_exact_max_abs=float((R.runoff_pred_m3 - R.runoff_m3).abs().max()),
                                         food_pred_vs_exact_max_abs=float((R[fpc] - R[fc]).abs().max()),
                                         **({} if heat is None else dict(heat_pred_vs_exact_max_abs=float((R.heat_pred_c - R.heat_c).abs().max()))))
        all_rows.append(R)
    A = pd.concat(all_rows)
    A.to_csv(out / "front.csv", index=False)
    # comparison with the hand-made T9 combined scenario (152 ha) and representative 150 ha solutions
    t9 = json.loads((OUTPUTS / f"scenarios{suffix}" / "metrics.json").read_text())["scenarios"]["combined"]
    c = np.array([t9["runoff_m3"], -t9[fc], -(t9["eca_ha"] - base["eca_ha"])] + ([t9["heat_c"]] if heat is not None else []))
    M = A[(A.budget_ha == budgets[0]) & A.on_exact_front]
    Fm = np.column_stack([M.runoff_m3, -M[fc], -M.eca_gain_ha] + ([M.heat_c] if heat is not None else []))
    dominating = np.all(Fm <= c, axis=1) & np.any(Fm < c, axis=1)
    norm = (Fm - Fm.min(axis=0)) / np.where(np.ptp(Fm, axis=0) > 0, np.ptp(Fm, axis=0), 1)
    reps = {"min_runoff": int(np.argmin(Fm[:, 0])), "max_food": int(np.argmin(Fm[:, 1])), "max_connectivity": int(np.argmin(Fm[:, 2])),
            "knee": int(np.argmin(np.linalg.norm(norm, axis=1)))}
    if heat is not None:
        reps["min_heat"] = int(np.argmin(Fm[:, 3]))
    summary["t9_combined_dominated_by"] = int(dominating.sum())
    summary["representatives_150ha"] = {k: M.iloc[i].drop("x").to_dict() for k, i in reps.items()}
    meta = dict(driver="GTiff", height=d["cm"].shape[0], width=d["cm"].shape[1], count=1, dtype="uint8", crs=d["crs"], transform=d["tf"], compress="deflate")
    for k, i in reps.items():
        x = np.array(list(map(int, M.iloc[i].x.split(","))))
        with rasterio.open(out / f"solution_{k}.tif", "w", **meta) as dst:
            dst.write(solution_map(x, ids, d["cm"], cand, blocks).astype("uint8"), 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    M.drop(columns="x").sort_values("runoff_m3").to_csv(f"docs/optimisation_front_150ha{suffix}.csv", index=False)
    return summary


if __name__ == "__main__":
    import json

    from src.wardmap import memory_guard
    memory_guard(4.0)
    import sys
    if "--v3" in sys.argv:                                   # T14: four objectives on the current product map
        mp = sys.argv[sys.argv.index("--v3") + 1]
        energy, heat = v3_inputs()
        s = run(map_name=mp, suffix="_v3", energy=energy, heat=heat)
    else:
        s = run(map_name="landuse_2023_corr.tif", suffix="_v2") if "--v2" in sys.argv else run()
    print(json.dumps({k: v for k, v in s.items() if k != "representatives_150ha"}, indent=1, default=float))
    for k, v in s["representatives_150ha"].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
