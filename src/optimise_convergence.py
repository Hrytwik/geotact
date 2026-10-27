"""T19: convergence of the T14 NSGA-II search (protocol in docs/build_plan.md). Seeds x generation budgets on the cached site data,
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


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=float))
