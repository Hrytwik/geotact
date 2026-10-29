"""T19: convergence of the T14 NSGA-II search (protocol in docs/results_log.md). Seeds x generation budgets on the cached site data,
hypervolume of each front in one fixed normalisation (the non-dominated union of every run)."""
import json

import numpy as np

from src import optimise as op

GENS = (300, 1000, 3000, 10000)
SEEDS = (0, 1, 2, 3, 4)


def site_data(path, budget=150.0):
    z = np.load(path, allow_pickle=True)
    return op.SiteData(z["allowed"], z["area"], z["d_runoff"], z["d_food"], z["d_conn"], budget, 0.0, 0.0, d_heat=z["d_heat"])


def front_hv(fronts, lo, hi):
    """Hypervolume of the non-dominated part of the stacked fronts (list of objective arrays) in the fixed normalisation."""
    F = np.vstack(fronts)
    return op.hypervolume(F[op.pareto_mask(F)], lo, hi)


def judge(hv, gens=GENS):
    """hv = {setting: {"union": float, "seeds": [floats]}}; returns per-setting gain to the next setting, spread, converged flag,
    and whether the first setting is adequate (union within 1% of the last setting's union)."""
    out = {}
    for i, g in enumerate(gens):
        s = np.asarray(hv[g]["seeds"], float)
        spread = float((s.max() - s.min()) / s.mean())
        gain = float(hv[gens[i + 1]]["union"] / hv[g]["union"] - 1) if i + 1 < len(gens) else float("nan")
        out[g] = dict(union=hv[g]["union"], seed_spread=spread, gain_to_next=gain,
                      converged=bool(i + 1 < len(gens) and gain < 0.005 and spread < 0.02))
    first, last = gens[0], gens[-1]
    out["first_adequate"] = bool(hv[first]["union"] >= 0.99 * hv[last]["union"])
    conv = [g for g in gens if out[g]["converged"]]
    out["shortest_converged"] = conv[0] if conv else None
    return out


def run():
    from src.config import OUTPUTS
    d = site_data(OUTPUTS / "optimise_v3" / "site_data.npz")
    F = {}
    settings = [(g, 200) for g in GENS] + [(3000, 500)]
    for g, pop in settings:
        for s in SEEDS:
            F[(g, pop, s)] = op.run_nsga(d, pop, g, s)[1]
            print(f"gens {g} pop {pop} seed {s}: {len(F[(g, pop, s)])} solutions", flush=True)
    U = np.vstack(list(F.values()))
    U = U[op.pareto_mask(U)]
    lo, hi = U.min(axis=0), U.max(axis=0)
    hv = {}
    for g, pop in settings:
        key = g if pop == 200 else f"{g}_pop{pop}"
        hv[key] = dict(union=front_hv([F[(g, pop, s)] for s in SEEDS], lo, hi),
                       seeds=[op.hypervolume(F[(g, pop, s)][op.pareto_mask(F[(g, pop, s)])], lo, hi) for s in SEEDS])
    res = dict(hv=hv, judgement=judge(hv), n_union_nondominated=int(len(U)))
    (OUTPUTS / "optimise_v3" / "convergence.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def run_t19b(gens=10000, archive_every=25):
    """T19b: (A) pop 200 with an external archive, seeds 0-4 (its final populations are the T19 setting); (B) pop 500, seeds 0-2."""
    from src.config import OUTPUTS
    out = OUTPUTS / "optimise_v3"
    d = site_data(out / "site_data.npz")
    A, P, B = {}, {}, {}
    for s in SEEDS:
        Xa, Fa, _, Fp = op.run_nsga(d, 200, gens, s, archive_every=archive_every, return_final=True)
        A[s], P[s] = (Xa, Fa), Fp
        np.savez(out / f"t19b_archive_seed{s}.npz", X=Xa, F=Fa)
        print(f"A seed {s}: archive {len(Xa)}, final {len(Fp)}", flush=True)
    for s in SEEDS[:3]:
        Xb, Fb = op.run_nsga(d, 500, gens, s)
        B[s] = (Xb, Fb)
        np.savez(out / f"t19b_pop500_seed{s}.npz", X=Xb, F=Fb)
        print(f"B seed {s}: {len(Xb)}", flush=True)
    U = np.vstack([v[1] for v in A.values()] + list(P.values()) + [v[1] for v in B.values()])
    U = U[op.pareto_mask(U)]
    lo, hi = U.min(axis=0), U.max(axis=0)
    hv = dict(archive=front_hv([v[1] for v in A.values()], lo, hi), final_pop200=front_hv(list(P.values()), lo, hi),
              pop500=front_hv([v[1] for v in B.values()], lo, hi))
    XA = np.vstack([v[0] for v in A.values()]); FA = np.vstack([v[1] for v in A.values()])
    XA, FA = op.archive_update(np.empty((0, XA.shape[1]), int), np.empty((0, FA.shape[1])), XA, FA)
    np.savez(out / "t19b_archive_union.npz", X=XA, F=FA)
    res = dict(hv=hv, n_archive_union=int(len(XA)), archive_sizes={int(s): int(len(v[0])) for s, v in A.items()},
               archive_adopted=bool(hv["archive"] >= 0.995 * hv["pop500"]))
    (out / "t19b.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def headlines(R, b):
    """T14/T19 headline numbers of an exact front table R against the baseline metrics b (same definitions as in T19)."""
    all4 = (R.runoff_m3 < b["runoff_m3"]) & (R.food_gj > b["food_gj"]) & (R.eca_gain_ha > 0) & (R.heat_c < 0)
    Fm = np.column_stack([R.runoff_m3, -R.food_gj, -R.eca_gain_ha, R.heat_c])
    norm = (Fm - Fm.min(axis=0)) / np.where(np.ptp(Fm, axis=0) > 0, np.ptp(Fm, axis=0), 1)
    k = R.iloc[int(np.argmin(np.linalg.norm(norm, axis=1)))]
    return dict(n_front=int(len(R)), better_all_four=int(all4.sum()),
                min_runoff_pct=float(100 * (R.runoff_m3.min() / b["runoff_m3"] - 1)), max_food_gj=float(R.food_gj.max() - b["food_gj"]),
                max_eca_ha=float(R.eca_gain_ha.max()), min_heat_c=float(R.heat_c.min()),
                best_all4_runoff_pct=float(100 * (R[all4].runoff_m3.min() / b["runoff_m3"] - 1)) if all4.any() else None,
                knee=dict(runoff_pct=float(100 * (k.runoff_m3 / b["runoff_m3"] - 1)), food_gj=float(k.food_gj - b["food_gj"]),
                          eca_ha=float(k.eca_gain_ha), heat_c=float(k.heat_c), wetland_ha=float(k.wetland_ha), garden_ha=float(k.garden_ha),
                          corridor_ha=float(k.corridor_ha)))


def exact_archive():
    """T19b, archive adopted: exact four-objective values of every plan in the non-dominated archive union at 150 ha."""
    import pandas as pd

    from src.config import OUTPUTS
    out = OUTPUTS / "optimise_v3_t19b"
    out.mkdir(exist_ok=True)
    energy, heat = op.v3_inputs()
    data, ids, d, cand, blocks, base, built_px = op.precompute(150.0, map_name="landuse_2023_corr.tif", suffix="_v3", energy=energy, heat=heat)
    z = np.load(OUTPUTS / "optimise_v3" / "t19b_archive_union.npz")
    rows = []
    for i, x in enumerate(z["X"]):
        rows.append(op.exact_row(data, x, ids, d, cand, blocks, base, built_px, energy, heat))
        if (i + 1) % 100 == 0 or i + 1 == len(z["X"]):
            import time
            print(f"  exact {i + 1}/{len(z['X'])} at {time.strftime('%H:%M:%S')}", flush=True)
    R = pd.DataFrame(rows)
    Fe = np.column_stack([R.runoff_m3, -R.food_gj, -R.eca_gain_ha, R.heat_c])
    R["on_exact_front"] = op.pareto_mask(Fe)
    R.to_csv(out / "front.csv", index=False)
    b = json.loads((OUTPUTS / "scenarios_v3" / "metrics.json").read_text())["scenarios"]["baseline"]
    M = R[R.on_exact_front]
    t9 = json.loads((OUTPUTS / "scenarios_v3" / "metrics.json").read_text())["scenarios"]["combined"]
    c = np.array([t9["runoff_m3"], -t9["food_gj"], -(t9["eca_ha"] - b["eca_ha"]), t9["heat_c"]])
    Fm = np.column_stack([M.runoff_m3, -M.food_gj, -M.eca_gain_ha, M.heat_c])
    res = dict(n_archive=int(len(R)), headlines=headlines(M, b), t9_combined_dominated_by=int((np.all(Fm <= c, axis=1) & np.any(Fm < c, axis=1)).sum()))
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    import sys
    if "--exact-archive" in sys.argv:
        from src.wardmap import memory_guard
        memory_guard(4.0)
        print(json.dumps(exact_archive(), indent=1, default=float))
        sys.exit()
    print(json.dumps(run_t19b() if "--t19b" in sys.argv else run(), indent=1, default=float))
