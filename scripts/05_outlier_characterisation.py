"""
05_outlier_characterisation.py
==============================
Westhoek ERT — characterise suspect electrodes from a run's residuals and write
the inputs for the three outlier treatments (G1–G3).

For each flagged electrode (default list + the run's worst electrodes):
  - outliers where it acts as current (C1/C2) vs potential (P1/P2) electrode
  - mean normalised residual and sign consistency (raw and role-aligned*)
  - |r| vs potential-dipole length |x(P1) − x(P2)| for data using it
    (Spearman ρ; negative = misfit grows as the dipole gets shorter)

* role-aligned: r multiplied by −1 when the electrode is C2 or P2. A position
  error of one electrode changes the geometric factor with opposite sign in the
  two roles, so position errors should be sign-consistent after alignment;
  contact problems (noise) should not be.

Outlier rule (as 04): |r| > 3 × 1.4826 × MAD(r).

OUTPUTS
  results/<tag>_outlier_characterisation.txt
  figures/runs/<tag>/17_outlier_dipole_length.png
  results/<tag>_G2_droplist_flagged_electrodes.txt   (all data using flagged electrodes)
  results/<tag>_G3_err_scale.txt                     ('<index> <factor>' for 02 --err-scale-file)

RUN
  python 05_outlier_characterisation.py --tag deep_all [--electrodes 39,40,...] [--n-worst 10]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "processed"))
FIG_DIR       = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "figures"))
RESULTS_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))

DEFAULT_FLAGGED = [39, 40, 41, 42, 43, 44, 45, 55, 59, 65, 66]
MAD_FACTOR = 3.0 * 1.4826


def main():
    ap = argparse.ArgumentParser(description="Characterise suspect electrodes")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--electrodes", default=",".join(map(str, DEFAULT_FLAGGED)),
                    help="comma-separated electrode numbers to examine")
    ap.add_argument("--n-worst", type=int, default=10,
                    help="also flag this many worst electrodes (mean |r|) from the run")
    args = ap.parse_args()

    npz = os.path.join(PROCESSED_DIR, "runs", args.tag, "westhoek_inversion_results.npz")
    d = np.load(npz, allow_pickle=True)
    r = (np.log(d["d_obs"]) - np.log(d["d_pred"])) / d["d_err"]
    C1, C2, P1, P2 = (d[k].astype(int) for k in ("C1", "C2", "P1", "P2"))
    keep = d["keep_idx"].astype(int)
    x_of = dict(zip(d["elec_num"].astype(int), d["elec_x"]))
    med = np.median(r); mad = np.median(np.abs(r - med)); thr = MAD_FACTOR * mad
    out = np.abs(r) > thr
    dip = np.abs(np.array([x_of[e] for e in P1]) - np.array([x_of[e] for e in P2]))

    # electrodes: requested list + the run's n worst by mean |r|
    given = [int(e) for e in args.electrodes.split(",") if e.strip()]
    mean_abs = {e: np.mean(np.abs(r[(C1 == e) | (C2 == e) | (P1 == e) | (P2 == e)]))
                for e in x_of if ((C1 == e) | (C2 == e) | (P1 == e) | (P2 == e)).any()}
    worst = sorted(mean_abs, key=lambda e: -mean_abs[e])[:args.n_worst]
    new = [e for e in worst if e not in given]
    flagged = sorted(set(given) | set(worst))

    L = [f"OUTLIER CHARACTERISATION — run '{args.tag}'  ({npz})",
         f"n={len(r)}  median r={med:.3f}  MAD={mad:.3f}  outlier threshold |r| > {thr:.3f}  "
         f"→ {out.sum()} outliers ({100*out.mean():.1f}%)",
         f"electrodes examined: given {given}; new from this run's {args.n_worst} worst: {new or 'none'}",
         "",
         f"{'elec':>4} {'x':>6} {'n':>4} {'mean|r|':>7} {'nOut':>4} {'out/n as C = rate':>17} "
         f"{'out/n as P = rate':>17} "
         f"{'mean r':>7} {'sgn raw':>7} {'sgn al':>6} {'sgnOut al':>9} {'ρS(|r|,dip)@P':>13} {'p':>7}  verdict"]
    rows = []
    for e in flagged:
        use = (C1 == e) | (C2 == e) | (P1 == e) | (P2 == e)
        asC = (C1 == e) | (C2 == e)
        asP = (P1 == e) | (P2 == e)
        sign = np.where((C2 == e) | (P2 == e), -1.0, 1.0)      # role alignment
        ra = r * sign
        def cons(v):
            v = v[v != 0]
            return max(np.mean(v > 0), np.mean(v < 0)) if v.size else np.nan
        s_raw, s_al = cons(r[use]), cons(ra[use])
        s_out_al = cons(ra[use & out])
        sel = asP & ~asC
        if sel.sum() >= 8:
            rho_s, p_s = stats.spearmanr(np.abs(r[sel]), dip[sel])
        else:
            rho_s, p_s = np.nan, np.nan
        nC, nP = int((out & asC).sum()), int((out & asP & ~asC).sum())
        # outlier RATES per role (an electrode is a potential electrode far more
        # often than a current electrode in a multiple-gradient array)
        rateC = nC / max(asC.sum(), 1)
        rateP = nP / max((asP & ~asC).sum(), 1)
        # heuristic verdict (hypothesis only)
        if nC + nP == 0:
            verdict = "few outliers"
        elif rateC > 1.5 * rateP and (np.isnan(s_out_al) or s_out_al < 0.7):
            verdict = "contact-like (current-driven, mixed sign)"
        elif s_out_al >= 0.8 and (np.isnan(rho_s) or rho_s < -0.2):
            verdict = "position-like (sign-consistent, short dipoles worse)"
        elif s_out_al >= 0.8:
            verdict = "sign-consistent, no dipole-length trend"
        elif rateP > 1.5 * rateC and s_out_al < 0.7 and rho_s < -0.2 and p_s < 0.05:
            verdict = "potential-side noise (random sign, short dipoles worse)"
        elif s_out_al < 0.7:
            verdict = "random sign, no clear role/dipole pattern"
        else:
            verdict = "mixed / unclear"
        rows.append((e, use, asC, asP, ra))
        L.append(f"{e:4d} {x_of[e]:6.1f} {use.sum():4d} {mean_abs.get(e, np.nan):7.2f} {int((out & use).sum()):4d} "
                 f"{nC:3d}/{asC.sum():<3d}={100*rateC:3.0f}% {nP:3d}/{(asP & ~asC).sum():<3d}={100*rateP:3.0f}% "
                 f"{np.mean(r[use]):+7.2f} {s_raw:7.2f} {s_al:6.2f} {s_out_al:9.2f} "
                 f"{rho_s:13.2f} {p_s:7.1e}  {verdict}")
    L += ["",
          "Columns: out@C = outliers where the electrode is C1/C2; out@P = outliers where it is only P1/P2;",
          "sgn raw / sgn al = fraction of residuals with the majority sign (raw / role-aligned, all data",
          "using the electrode); sgnOut al = same for its outliers only; ρS = Spearman correlation of |r|",
          "with |x(P1)−x(P2)| for data where the electrode is a potential electrode (negative → worse",
          "for short dipoles). 0.5 = random sign, 1.0 = fully consistent.",
          "Verdicts are heuristic HYPOTHESES, not diagnoses."]

    # ── G2 / G3 inputs ────────────────────────────────────────────────────────
    anyflag = np.zeros(len(r), bool)
    for e in flagged:
        anyflag |= (C1 == e) | (C2 == e) | (P1 == e) | (P2 == e)
    g_med = np.median(r[anyflag]); g_mad = np.median(np.abs(r[anyflag] - g_med))
    g_scale = max(1.0, 1.4826 * g_mad)
    o_med = np.median(r[~anyflag]); o_mad = np.median(np.abs(r[~anyflag] - o_med))
    L += ["",
          f"Flagged electrode set used for G2/G3: {flagged}",
          f"Data using ≥1 flagged electrode: {anyflag.sum()} of {len(r)} ({100*anyflag.mean():.1f}%)",
          f"Robust residual scale 1.4826·MAD: flagged group = {1.4826*g_mad:.3f}, "
          f"other data = {1.4826*o_mad:.3f}  → G3 error factor = {g_scale:.3f}"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    g2 = os.path.join(RESULTS_DIR, f"{args.tag}_G2_droplist_flagged_electrodes.txt")
    with open(g2, "w", encoding="utf-8") as f:
        f.write(f"# All data using flagged electrodes {flagged} (run '{args.tag}'); "
                "0-based land_data.dat indices\n")
        f.writelines(f"{i}\n" for i in sorted(keep[anyflag]))
    g3 = os.path.join(RESULTS_DIR, f"{args.tag}_G3_err_scale.txt")
    with open(g3, "w", encoding="utf-8") as f:
        f.write(f"# Error factor {g_scale:.4f} = 1.4826·MAD of normalised residuals of data using "
                f"flagged electrodes {flagged} (run '{args.tag}')\n# <0-based index> <factor>\n")
        f.writelines(f"{i} {g_scale:.4f}\n" for i in sorted(keep[anyflag]))
    L += [f"G2 drop-list: {g2}", f"G3 error-scale file: {g3}"]

    rep = os.path.join(RESULTS_DIR, f"{args.tag}_outlier_characterisation.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    # ── Figure: role-aligned residual vs potential-dipole length ─────────────
    n = len(rows); nc = 4; nr_ = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr_, nc, figsize=(16, 3.2 * nr_), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (e, use, asC, asP, ra) in zip(axes, rows):
        pc = use & asC; pp = use & asP & ~asC
        ax.scatter(dip[pp], ra[pp], s=10, c="#2C6FAC", alpha=0.7, label="as P1/P2")
        ax.scatter(dip[pc], ra[pc], s=10, c="#D9534F", alpha=0.7, marker="^", label="as C1/C2")
        ax.axhline(0, color="k", lw=0.7)
        for s in (-thr, thr):
            ax.axhline(s, color="gray", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"electrode {e} (x={x_of[e]:.0f} m)", fontsize=9)
        ax.set_yscale("symlog", linthresh=3)
        ax.grid(True, which="both", alpha=0.2)
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].legend(fontsize=7)
    fig.supxlabel("Potential-dipole length |x(P1) − x(P2)| (m)")
    fig.supylabel("Role-aligned normalised residual (symlog)")
    fig.suptitle(f"Suspect electrodes — residual vs potential-dipole length, run '{args.tag}'",
                 fontweight="bold")
    fig.tight_layout()
    fdir = os.path.join(FIG_DIR, "runs", args.tag)
    os.makedirs(fdir, exist_ok=True)
    fp = os.path.join(fdir, "17_outlier_dipole_length.png")
    fig.savefig(fp, dpi=150)
    print(f"[saved] {rep}\n[saved] {fp}")


if __name__ == "__main__":
    main()
