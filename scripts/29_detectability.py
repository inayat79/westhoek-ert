"""
29_detectability.py
===================
Westhoek ERT v1.5 — Task 2: can the ERT data tell fresh from brackish water below the lens? Forward modelling only.
EXPLORATORY; designed after the DP results were known (docs/decision_log.md v1.5-0).

  run    [--workers 6] [--out ROOT]   noise-free responses of T2 with ρ_below ∈ {3, 5, 8, 12, 20, 40, 60} Ω·m and the clay top
                                      as T2 / −30 / −26 mTAW, on both layouts → ROOT/data/processed/v1_5/detect/
  resume [--workers 6] [--out ROOT]   as run (finished responses are skipped)
  report [--out ROOT]                 χ²/N tables, separability ranges, trade-off, field check (X5)
                                      → ROOT/results/v1_5/3_detectability.md, figs/F_15_detectability.png

χ²/N = mean of ((ln R_a − ln R_b)/ε_b)², ε_b = √(0.03² + (1e-4 Ω / R_b)²) (the v1.0 error model on the reference).
No random draws (noise-free forward modelling).
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
read_trace.install("29_detectability.py")
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


g28 = _load("g28", "28_dp_synthetic.py")
RHOS = [3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 60.0]
CLAYS = ["T2", -30.0, -26.0]
REF_RHO = 12.0
EXPLORATORY = g28.EXPLORATORY


def SRC():
    return os.path.join(wc.ROOT, "data", "processed", "v1_5", "detect")


def ctag(c):
    return "T2" if c == "T2" else f"{c:g}"


class Variant(g28.TruthModel):
    """T2 with ρ_below everywhere = rho and the clay top = T2's (CPT step / −27.5) or a fixed elevation."""
    def __init__(self, rho, clay):
        super().__init__("T2")
        self.lr["rho_below"] = np.full(len(self.D), np.log10(rho))
        if clay != "T2":
            self.par["clay"] = np.full(len(self.D), float(clay))


def fwd_job(job):
    rho, clay, lay = job
    p = os.path.join(SRC(), f"{lay}_rho{rho:g}_clay{ctag(clay)}.npz")
    if os.path.exists(p):
        return p
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    elec, q, _ = g28.layout(lay); scheme = g28.scheme_of(elec, q)
    with contextlib.redirect_stdout(io.StringIO()):
        fmesh = mt.createParaMesh(scheme.sensors(), **g28.sb.FWD)
    cx = np.array([c.center().x() for c in fmesh.cells()]); cz = np.array([c.center().y() for c in fmesh.cells()])
    v = Variant(rho, clay).log10rho(cx, cz)
    for _ in range(3):
        with contextlib.redirect_stdout(io.StringIO()):
            out = ert.simulate(fmesh, scheme=g28.scheme_of(elec, q), res=10 ** v, noiseLevel=0, verbose=False)
        R = np.array(out["rhoa"])
        if len(R) == len(q["C1"]):
            break
    else:
        raise RuntimeError(f"forward {job}: wrong number of data")
    np.savez(p + ".tmp.npz", R=R); os.replace(p + ".tmp.npz", p)
    return p


def cmd_run(args):
    os.makedirs(SRC(), exist_ok=True)
    jobs = [(r, c, l) for l in g28.LAYOUTS for c in CLAYS for r in RHOS]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        done = list(ex.map(fwd_job, jobs))
    print(f"[detect] {len(done)} responses")


def chi2n(Ra, Rb):
    ok = (Ra > 0) & (Rb > 0)
    eps = np.sqrt(0.03 ** 2 + (1e-4 / Rb[ok]) ** 2)
    return float(np.mean(((np.log(Ra[ok]) - np.log(Rb[ok])) / eps) ** 2))


def measured(lay):
    """(R measured, primary mask, label): L18 2331 published data with the G1 mask; L24 1806 positive ERT.dat data with the
    v1.4 Stage-1 mask."""
    if lay == "L18":
        L = open(os.path.join(wc.ROOT, "data", "raw", "land_data.dat")).read().split("\n")
        n = int(L[0].split()[0]); R = np.array([float(l.split()[4]) for l in L[1:1 + n]])
        drop = {int(l.split("#")[0]) for l in open(os.path.join(wc.ROOT, "results", "deep_all_droplist.txt")) if l.split("#")[0].strip()}
        return R, np.array([j not in drop for j in range(n)]), "2018 data (6.5-year gap)"
    E = g28.g27.read_ert()
    keep = np.load(os.path.join(wc.ROOT, "data", "processed", "v1_4", "samedate", "cpt.npz"))["keep"]
    return E["R"], keep.astype(bool), "ERT.dat (same epoch, date unconfirmed)"


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = os.path.join(args.out, "results", "v1_5"); figs = os.path.join(res, "figs"); os.makedirs(figs, exist_ok=True)
    dpp = os.path.join(args.out, "data", "processed", "v1_5"); os.makedirs(dpp, exist_ok=True)
    s18 = _load("p18", "18_paper.py"); s18.mpl()
    R = {(l, r, ctag(c)): np.load(os.path.join(SRC(), f"{l}_rho{r:g}_clay{ctag(c)}.npz"))["R"]
         for l in g28.LAYOUTS for c in CLAYS for r in RHOS}
    T = {(t, l): np.load(os.path.join(g28.SRC(), f"fwd_{t}_{l}.npz"))["R"] for t in ("T1", "T3") for l in g28.LAYOUTS}
    out = dict(rhos=RHOS, clays=[ctag(c) for c in CLAYS], vs_ref={}, vs_T1={}, field={})
    L = ["# 3 — Can the ERT data tell fresh from brackish water below the lens? (Task 2)", "", f"**{EXPLORATORY}**", "",
         "Noise-free forward responses of T2 with ρ_below replaced everywhere, and with the clay top as in T2 (CPT step / −27.5) "
         "or moved to −30 or −26 mTAW; both layouts (L18: 2018 layout, 2331 quadrupoles; L24: ERT.dat layout, 1806 quadrupoles). "
         "χ²/N against a reference response with the v1.0 error model of the reference (3 % + 1e-4 Ω). χ²/N < 1: the "
         "two models cannot be told apart at the data error.", ""]
    for l in g28.LAYOUTS:
        ref = R[(l, REF_RHO, "T2")]
        L += [f"## {l}: χ²/N against ρ_below = 12 Ω·m (T2 clay top) and against T1", "",
              "| clay top | " + " | ".join(f"ρ_below = {r:g}" for r in RHOS) + " |", "|---|" + "---|" * len(RHOS)]
        for c in CLAYS:
            v = [chi2n(R[(l, r, ctag(c))], ref) for r in RHOS]
            out["vs_ref"][f"{l}|{ctag(c)}"] = v
            L.append(f"| {ctag(c) if c != 'T2' else 'T2 (CPT step / −27.5)'} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
        L.append("")
        L += ["| vs T1, clay top | " + " | ".join(f"ρ_below = {r:g}" for r in RHOS) + " |", "|---|" + "---|" * len(RHOS)]
        for c in CLAYS:
            v = [chi2n(R[(l, r, ctag(c))], T[("T1", l)]) for r in RHOS]
            out["vs_T1"][f"{l}|{ctag(c)}"] = v
            L.append(f"| {ctag(c) if c != 'T2' else 'T2'} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
        L.append("")
        for thr in (1.0, 2.0):
            ok = [r for r, x in zip(RHOS, out["vs_ref"][f"{l}|T2"]) if x < thr]
            out.setdefault("insep", {})[f"{l}|{thr:g}"] = ok
        trade = [(ctag(c), r, x) for c in CLAYS if c != "T2" for r, x in zip(RHOS, out["vs_ref"][f"{l}|{ctag(c)}"]) if x < 1.0 and r != REF_RHO]
        out.setdefault("trade", {})[l] = trade
        L += [f"- Not separable from ρ_below = 12 Ω·m (T2 clay top): χ²/N < 1 for ρ_below ∈ {{{', '.join(f'{r:g}' for r in out['insep'][f'{l}|1'])}}}; "
              f"< 2 for {{{', '.join(f'{r:g}' for r in out['insep'][f'{l}|2'])}}} Ω·m.",
              "- Trade-off with the clay top (a moved clay top with another ρ_below at χ²/N < 1 against the reference): "
              + ("; ".join(f"clay {c} mTAW with ρ_below {r:g} Ω·m (χ²/N {x:.2f})" for c, r, x in trade) or "none") + ".", ""]
        meas, mask, lab = measured(l)
        f = {}
        for t in ("T1", "T3"):
            f[t] = dict(primary=chi2n(T[(t, l)][mask], meas[mask]), all=chi2n(T[(t, l)], meas), n=int(mask.sum()), n_all=int(len(meas)))
        out["field"][l] = dict(label=lab, **f)
    L += ["## Field check: χ²/N of the noise-free truth responses against the measured data", "",
          "| layout | measured data | T1 (2024 logs) | T3 (brackish below) | T1 all data | T3 all data |", "|---|---|---|---|---|---|"]
    for l in g28.LAYOUTS:
        F = out["field"][l]
        L.append(f"| {l} | {F['label']}; {F['T1']['n']} G1/Stage-1 data ({F['T1']['n_all']} all) | {F['T1']['primary']:.2f} | "
                 f"{F['T3']['primary']:.2f} | {F['T1']['all']:.2f} | {F['T3']['all']:.2f} |")
    x5 = out["field"]["L24"]["T1"]["primary"] <= out["field"]["L24"]["T3"]["primary"]
    out["X5"] = bool(x5)
    L += ["", f"**X5** (ERT.dat fits T1 at least as well as T3): **{'yes' if x5 else 'no'}** — χ²/N "
          f"{out['field']['L24']['T1']['primary']:.2f} (T1) vs {out['field']['L24']['T3']['primary']:.2f} (T3). Both truths are "
          "built from 1D logs interpolated laterally, so neither is expected to fit to χ²/N ≈ 1."]
    with open(os.path.join(res, "3_detectability.md"), "w", encoding="utf-8") as f_:
        f_.write("\n".join(L) + "\n")
    with open(os.path.join(dpp, "detect.json"), "w", encoding="utf-8") as f_:
        json.dump(out, f_, indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    cols = {"T2": "k", "-30": "#1f4e9c", "-26": "#d35400"}
    for ax, l in zip(axes, g28.LAYOUTS):
        for c in CLAYS:
            ax.plot(RHOS, out["vs_ref"][f"{l}|{ctag(c)}"], "-o", ms=3, color=cols[ctag(c)],
                    label=f"clay top {'T2 (CPT step / −27.5)' if c == 'T2' else f'{c:g} mTAW'}")
        ax.axhline(1, color="0.4", ls="--", lw=0.8); ax.axhline(2, color="0.4", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(RHOS); ax.set_xticklabels([f"{r:g}" for r in RHOS], fontsize=6)
        ax.set_xlabel("ρ_below (Ω·m)"); ax.set_title(f"{l}: χ²/N against ρ_below = 12 Ω·m", fontsize=8)
    axes[0].set_ylabel("χ²/N (v1.0 error model)"); axes[1].legend(fontsize=6)
    fig.suptitle(EXPLORATORY, fontsize=7); fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(figs, "F_15_detectability.png"), dpi=300); plt.close(fig)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "resume"):
        s = sub.add_parser(c); s.add_argument("--out", default=wc.ROOT); s.add_argument("--workers", type=int, default=6)
    sub.add_parser("report").add_argument("--out", default=wc.ROOT)
    a = ap.parse_args()
    (cmd_report if a.cmd == "report" else cmd_run)(a)


if __name__ == "__main__":
    main()
