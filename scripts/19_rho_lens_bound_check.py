"""
19_rho_lens_bound_check.py — why did the lens get saltier under the physical filter?

    conda run -n pg python 19_rho_lens_bound_check.py [--out ROOT]

Field ρ_lens posterior of the joint BEL (17_salinity_v2.py) for: the pre-freeze site subset (logit bound 0.4 Ω·m);
the site + physical subset with the bound 0.4 (v1.0: filter only); the site + physical subset with the bound raised
to 0.69415 Ω·m (rejected alternative: filter + bound); five random subsets of the site set with the same
size as the site + physical set (refit noise). Joint-BEL posterior seeds default_rng([129000, 0]) (as 17); random
subsets default_rng([133000, i]); Archie draws default_rng([131000, 0]) (as 17).
Writes ROOT/results/paper/rho_lens_bound_check.md and ROOT/data/processed/paper/rho_lens_bound_check.json.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import read_trace  # noqa: E402
read_trace.install("19_rho_lens_bound_check.py")
import westhoek_common as wc  # noqa: E402

_sp = importlib.util.spec_from_file_location("s17", os.path.join(SCRIPT_DIR, "17_salinity_v2.py"))
s17 = importlib.util.module_from_spec(_sp)
with contextlib.redirect_stdout(io.StringIO()):
    _sp.loader.exec_module(s17)
s6 = s17.s6


def q(v, d=2):
    lo, md, hi = np.percentile(v, [5, 50, 95]); return f"{md:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def field(D, T, th, bound, obs):
    s6.BOUNDS["rho_lens"] = (bound, 3.0, True)
    J = s6.JointBEL(D, T, th)
    b, p, _ = J.posterior(obs["field"], s17.N_POST, np.random.default_rng([s17.JOINT_SEED, 0]))
    bc, pc, A, info = s17.constrained(b, p, np.random.default_rng([s17.ARCHIE2_SEED, 0]))
    rl = np.asarray(pc["rho_lens"])
    return dict(rl_u=np.asarray(p["rho_lens"]), rl=rl, fr=A["F"] * A["rsw"], sw=100 * A["F"] * A["rsw"] / rl,
                rej=info["joint_rate"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=wc.ROOT); a = ap.parse_args()
    res = os.path.join(a.out, "results", "paper"); dpp = os.path.join(a.out, "data", "processed", "paper")
    os.makedirs(res, exist_ok=True); os.makedirs(dpp, exist_ok=True)
    s6.init(); s6.TX = s6.S16.TX
    obs, _ = s6.b13.observed_sets()
    C = s6.load_combined()
    D, T, th = C["tr"]["D"], C["tr"]["T"], C["tr"]["th"]
    phys = np.where([t["rho_lens"] >= s6.PHYS_RHO_LENS_MIN for t in th])[0]
    sub = lambda idx: (D[idx], T[idx], [th[k] for k in idx])
    cases = {"site, bound 0.4 (pre-freeze headline)": field(D, T, th, 0.4, obs),
             "site + physical, bound 0.4 (v1.0: filter only)": field(*sub(phys), 0.4, obs),
             f"site + physical, bound {s6.PHYS_RHO_LENS_MIN:.3f} (rejected: filter + bound)":
                 field(*sub(phys), s6.PHYS_RHO_LENS_MIN, obs)}
    rand = []
    for i in range(5):
        idx = np.sort(np.random.default_rng([133000, i]).choice(len(D), len(phys), replace=False))
        rand.append(field(*sub(idx), 0.4, obs))
    rl_tr = np.array([t["rho_lens"] for t in th])
    L = ["# Field ρ_lens under the physical filter: filter vs logit bound", "",
         f"Training ρ_lens: site {q(rl_tr)} (n = {len(rl_tr)}); site + physical {q(rl_tr[phys])} (n = {len(phys)}). "
         "Field posterior of the joint BEL (Ω·m), before and after the per-draw physical constraint; lens % seawater "
         "= 100·F·ρ_sw/ρ_lens of the accepted draws.", "",
         "| joint-BEL refit | ρ_lens, unconstrained | ρ_lens, constrained | accepted F·ρ_sw (Ω·m) | lens % seawater | "
         "joint rejection |", "|---|---|---|---|---|---|"]
    for k, r in cases.items():
        L.append(f"| {k} | {q(r['rl_u'])} | {q(r['rl'])} | {q(r['fr'], 3)} | {q(r['sw'], 1)} | {100*r['rej']:.1f} % |")
    for i, r in enumerate(rand):
        L.append(f"| random {len(phys)}-sample subset of site {i}, bound 0.4 (refit noise) | {q(r['rl_u'])} | {q(r['rl'])} | "
                 f"{q(r['fr'], 3)} | {q(r['sw'], 1)} | {100*r['rej']:.1f} % |")
    c0, c1, c2 = cases.values()
    med = lambda r, k: float(np.median(r[k])); p95 = lambda r, k: float(np.percentile(r[k], 95))
    L += ["", "**Reading.** The filter alone leaves the field ρ_lens posterior unchanged (median "
          f"{med(c0, 'rl_u'):.2f} → {med(c1, 'rl_u'):.2f} Ω·m, P95 {p95(c0, 'rl_u'):.2f} → {p95(c1, 'rl_u'):.2f}): the field "
          f"posterior hardly reaches the removed region ({100*np.mean(c0['rl_u'] < s6.PHYS_RHO_LENS_MIN):.1f} % of it below "
          f"{s6.PHYS_RHO_LENS_MIN:.3f} Ω·m). Raising the joint-BEL logit bound from 0.4 to {s6.PHYS_RHO_LENS_MIN:.3f} Ω·m "
          f"lowers it (median {med(c1, 'rl_u'):.2f} → {med(c2, 'rl_u'):.2f}, P95 {p95(c1, 'rl_u'):.2f} → {p95(c2, 'rl_u'):.2f}); "
          f"the random-subset refits span medians {min(med(r, 'rl_u') for r in rand):.2f}–{max(med(r, 'rl_u') for r in rand):.2f} "
          f"and P95 {min(p95(r, 'rl_u') for r in rand):.2f}–{max(p95(r, 'rl_u') for r in rand):.2f}. The Archie factor F·ρ_sw of "
          f"the accepted draws does not change ({med(c0, 'fr'):.3f} vs {med(c2, 'fr'):.3f} Ω·m), so the saltier lens "
          f"({med(c0, 'sw'):.1f} → {med(c2, 'sw'):.1f} % seawater) comes from the lower ρ_lens, i.e. from the bound: with "
          "the posterior (≈ 0.9–1.5 Ω·m) close to the new lower bound, the logit coordinate is strongly non-linear "
          "there, and KDE sampling in canonical space followed by the back-transform compresses the upper tail and "
          "moves the median towards the bound. It is a transformation effect, not information from the data — "
          "hence v1.0 keeps the transform bound at 0.4 Ω·m and applies the physics through the training-set filter and "
          "the per-draw rejection."]
    with open(os.path.join(res, "rho_lens_bound_check.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    summ = {k: dict(rl_u=q(r["rl_u"]), rl=q(r["rl"]), sw=q(r["sw"], 1)) for k, r in list(cases.items())}
    summ["random"] = [dict(rl_u=q(r["rl_u"]), sw=q(r["sw"], 1)) for r in rand]
    json.dump(summ, open(os.path.join(dpp, "rho_lens_bound_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
