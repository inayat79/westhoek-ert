"""
07_lambda_lcurve.py
===================
Westhoek ERT — λ-sweep summary and L-curve (model roughness vs data misfit).

For each run: λ, χ², iterations, stop reason (χ² ≤ 1 / ΔΦ < 2 % / iteration cap)
and the model roughness ‖Cw · ln m‖ (the regularisation term the inversion
minimises, with the same weighted C as pyGIMLi — verified in 02 --verify).

OUTPUTS
  figures/runs/lcurve_<name>/21_lcurve.png
  results/lcurve_<name>.txt

RUN
  python 07_lambda_lcurve.py --name G1 --tags lam5_x lam10_x lam20_x lam50_x lam100_x
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "processed"))
FIG_DIR       = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "figures"))
RESULTS_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))


def main():
    ap = argparse.ArgumentParser(description="λ-sweep summary and L-curve")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    rows = []
    for t in args.tags:
        d = np.load(os.path.join(PROCESSED_DIR, "runs", t, "westhoek_inversion_results.npz"), allow_pickle=True)
        rows.append(dict(tag=t, lam=float(d["lam"]), chi2=float(d["chi2"]), n_iter=int(d["n_iter"]),
                         max_iter=int(d["max_iter"]), stop=str(d["stop_reason"]),
                         rough=float(d["roughness"]), n_data=int(d["n_data"]),
                         runtime=float(d["runtime_total_s"]), negresp=bool(d["negresp"])))
    rows.sort(key=lambda r: r["lam"])

    ok = [r for r in rows if r["chi2"] <= 1.0]
    best = max(ok, key=lambda r: r["lam"]) if ok else None

    L = [f"λ SWEEP / L-CURVE '{args.name}'", "",
         f"{'tag':<22} {'λ':>6} {'n_data':>6} {'χ²':>8} {'iters':>5} {'stop reason':>14} "
         f"{'‖C ln m‖':>9} {'runtime s':>9} {'negResp':>7}"]
    for r in rows:
        L.append(f"{r['tag']:<22} {r['lam']:6g} {r['n_data']:6d} {r['chi2']:8.3f} {r['n_iter']:5d} "
                 f"{r['stop']:>14} {r['rough']:9.3f} {r['runtime']:9.0f} {'yes' if r['negresp'] else 'no':>7}")
    L += ["", (f"Largest λ reaching χ² ≤ 1: λ = {best['lam']:g} (run '{best['tag']}', χ² = {best['chi2']:.3f})"
               if best else "No run reached χ² ≤ 1.")]
    capped = [r["tag"] for r in rows if r["stop"] == "iteration cap"]
    if capped:
        L.append(f"WARNING: stopped at the iteration cap (not converged): {capped}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = [r["chi2"] for r in rows]; y = [r["rough"] for r in rows]
    ax.plot(x, y, "-", color="0.5", lw=1)
    for r in rows:
        mk = {"chi2<=1": "o", "dPhi<2%": "s", "iteration cap": "x"}.get(r["stop"], "d")
        ax.plot(r["chi2"], r["rough"], mk, ms=8, color="#1F77B4" if r["chi2"] <= 1 else "#D62728")
        ax.annotate(f"λ={r['lam']:g}", (r["chi2"], r["rough"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.axvline(1.0, color="k", ls="--", lw=0.8, label="χ² = 1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Data misfit χ²")
    ax.set_ylabel("Model roughness ‖Cw · ln m‖")
    ax.set_title(f"L-curve — {args.name}\n(o: stopped χ²≤1, s: ΔΦ<2 %, x: iteration cap)", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fdir = os.path.join(FIG_DIR, "runs", f"lcurve_{args.name}")
    os.makedirs(fdir, exist_ok=True)
    fig.savefig(os.path.join(fdir, "21_lcurve.png"), dpi=150)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rep = os.path.join(RESULTS_DIR, f"lcurve_{args.name}.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[saved] {rep}\n[saved] {os.path.join(fdir, '21_lcurve.png')}")


if __name__ == "__main__":
    main()
