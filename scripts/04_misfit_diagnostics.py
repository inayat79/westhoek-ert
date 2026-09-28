"""
04_misfit_diagnostics.py
========================
Westhoek ERT — data misfit diagnostics for a 02_land_inversion.py run.

INPUT  (from 02_land_inversion.py):
    data/processed/westhoek_inversion_results.npz                (no --tag)
    data/processed/runs/<tag>/westhoek_inversion_results.npz     (--tag <tag>)

OUTPUTS (figures/ or figures/runs/<tag>/):
    12_residual_histogram.png      — normalised log-space residuals + N(0,1) overlay
    13_residual_pseudosection.png  — residuals at pseudosection positions
    14_residual_vs_K_R.png         — residual vs |K| and vs |R|
    15_residual_per_electrode.png  — mean |residual| per electrode vs x
results/<tag>_misfit_report.txt    — χ², RMS, skew/kurtosis, worst data & electrodes
results/<tag>_droplist.txt         — |r| > 3 × 1.4826 × MAD(r) (0-based land_data.dat indices)

Normalised residual (log space):  r_i = (ln ρa_obs,i − ln ρa_pred,i) / err_i
with err_i the relative error assigned in 02 (original errors, also for robust runs).

RUN:
    python 04_misfit_diagnostics.py [--tag <tag>] [--input <npz>]
"""

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "processed"))
FIG_DIR       = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "figures"))
RESULTS_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))
ELEC_FILE     = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "raw", "elec.dat"))

MAD_FACTOR  = 3.0 * 1.4826   # outlier threshold in units of MAD (≈ 3σ for Gaussian data)
WORST_FRAC  = 0.05           # report the worst 5 % of measurements
N_WORST_EL  = 10             # report the 10 worst electrodes


# ─────────────────────────────────────────────────────────────────────────────
def water_lines(elec_file, land_x, land_z):
    """
    High-water crossing: x where the land surface drops below the mean marine
    Hsea level in elec.dat. Low water is not in elec.dat; the land line was
    acquired at low tide, so its seaward end approximates the low-water line.
    """
    hsea = []
    with open(elec_file) as f:
        for line in f:
            p = line.strip().split('\t')
            if p[0].strip().isdigit() and len(p) > 3 and p[3].strip() not in ('\\', ''):
                hsea.append(float(p[3]))
    if not hsea:
        return None, None, None
    h = float(np.mean(hsea))
    order = np.argsort(land_x)
    x, z = land_x[order], land_z[order]
    below = np.where(z < h)[0]
    x_hw = None
    if below.size:
        i = below[0]
        # linear interpolation of the crossing between electrodes i-1 and i
        x_hw = x[i] if i == 0 else x[i-1] + (h - z[i-1]) * (x[i] - x[i-1]) / (z[i] - z[i-1])
    return h, x_hw, float(x[-1])


def main():
    ap = argparse.ArgumentParser(description="Westhoek ERT — misfit diagnostics")
    ap.add_argument("--tag",   default=None, help="run tag (reads data/processed/runs/<tag>/)")
    ap.add_argument("--input", default=None, help="explicit path to a results .npz")
    args = ap.parse_args()

    tag = args.tag or "reference"
    if args.input:
        npz = args.input
    elif args.tag:
        npz = os.path.join(PROCESSED_DIR, "runs", args.tag, "westhoek_inversion_results.npz")
    else:
        npz = os.path.join(PROCESSED_DIR, "westhoek_inversion_results.npz")
    fig_dir = os.path.join(FIG_DIR, "runs", args.tag) if args.tag else FIG_DIR
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    d = np.load(npz, allow_pickle=True)
    needed = {"d_obs", "d_pred", "d_err", "C1", "C2", "P1", "P2", "R", "K",
              "keep_idx", "elec_num", "elec_x", "elec_z"}
    missing = needed - set(d.files)
    if missing:
        raise KeyError(f"{npz} lacks {sorted(missing)} — rerun it with the current 02_land_inversion.py")

    d_obs, d_pred, err = d["d_obs"], d["d_pred"], d["d_err"]
    C1, C2, P1, P2 = (d[k].astype(int) for k in ("C1", "C2", "P1", "P2"))
    R, K, keep_idx = d["R"], d["K"], d["keep_idx"].astype(int)
    elec_num, elec_x, elec_z = d["elec_num"].astype(int), d["elec_x"], d["elec_z"]
    x_of = dict(zip(elec_num, elec_x))

    log_ratio = np.log(d_obs) - np.log(d_pred)
    r = log_ratio / err
    n = len(r)
    chi2 = float(np.mean(r**2))
    rms_norm = float(np.sqrt(chi2))
    rms_pct = float(100 * np.sqrt(np.mean(log_ratio**2)))   # ≈ relative RMS misfit in %
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r))                           # excess kurtosis (0 for Gaussian)
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    thr = MAD_FACTOR * mad
    print(f"[{tag}] n={n}  χ²={chi2:.3f}  RMS(norm)={rms_norm:.3f}  RMS={rms_pct:.2f}%  "
          f"skew={skew:.2f}  excess kurtosis={kurt:.2f}  median={med:.2f}  MAD={mad:.3f}")

    # Pseudosection coordinates (same convention as 01_data_exploration.py)
    xa, xb, xm, xn = (np.array([x_of[e] for e in col]) for col in (C1, C2, P1, P2))
    x_mid = (xa + xb + xm + xn) / 4.0
    pseudo_z = -0.2 * np.maximum(np.abs(xb - xa), np.abs(xn - xm))

    h_sea, x_hw, x_lw = water_lines(ELEC_FILE, elec_x, elec_z)

    def mark_water(ax):
        if x_hw is not None:
            ax.axvline(x_hw, color="steelblue", ls="--", lw=1.0,
                       label=f"High-water line ({h_sea:.2f} mTAW, elec.dat)")
        if x_lw is not None:
            ax.axvline(x_lw, color="gray", ls=":", lw=1.0,
                       label="≈ Low-water line (seaward end of land line)")
        if x_hw is not None and x_lw is not None:
            ax.axvspan(x_hw, x_lw, color="orange", alpha=0.08, label="Intertidal zone")

    title_sfx = f"— run '{tag}'  (χ²={chi2:.2f}, n={n})"
    vlim = max(3.0, float(np.percentile(np.abs(r), 99)))

    # ── 12 Histogram ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-vlim * 1.2, vlim * 1.2, 81)
    ax.hist(np.clip(r, bins[0], bins[-1]), bins=bins, density=True,
            color="#2C6FAC", alpha=0.75, label="Normalised residuals")
    xx = np.linspace(bins[0], bins[-1], 400)
    ax.plot(xx, stats.norm.pdf(xx), "k-", lw=1.5, label="N(0, 1) — expected if errors are correct")
    ax.plot(xx, stats.norm.pdf(xx, med, 1.4826 * mad), "r--", lw=1.2,
            label=f"N(median, 1.4826·MAD) = N({med:.2f}, {1.4826*mad:.2f})")
    for s in (-thr, thr):
        ax.axvline(s, color="red", ls=":", lw=0.9)
    ax.set_xlabel("(ln ρa,obs − ln ρa,pred) / err")
    ax.set_ylabel("Density")
    ax.set_title(f"Normalised residual distribution {title_sfx}", fontsize=10)
    ax.text(0.02, 0.97, f"skew = {skew:.2f}\nexcess kurtosis = {kurt:.2f}\n"
            f"drop threshold |r| > {thr:.2f}", transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "12_residual_histogram.png"), dpi=150)
    plt.close(fig)

    # ── 13 Residual pseudosection ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    order = np.argsort(np.abs(r))           # draw largest residuals on top
    sc = ax.scatter(x_mid[order], pseudo_z[order], c=r[order], cmap="RdBu_r",
                    norm=mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim),
                    s=14, marker="s", linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Normalised residual  (+ = observed > predicted)")
    mark_water(ax)
    ax.set_xlabel("Midpoint x (m)")
    ax.set_ylabel("Pseudo-depth (m)")
    ax.set_title(f"Residual pseudosection {title_sfx}", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "13_residual_pseudosection.png"), dpi=150)
    plt.close(fig)

    # ── 14 Residual vs |K| and |R| ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, v, lab in ((axes[0], np.abs(K), "|geometric factor K| (m)"),
                       (axes[1], np.abs(R), "|R| (Ω)")):
        ax.scatter(v, r, s=6, alpha=0.5, c="#2C6FAC", linewidths=0)
        # running median in log-spaced bins to expose systematic trends
        edges = np.logspace(np.log10(v.min()), np.log10(v.max()), 16)
        idx = np.digitize(v, edges)
        cen, runmed = [], []
        for b in range(1, len(edges)):
            sel = idx == b
            if sel.sum() >= 10:
                cen.append(np.sqrt(edges[b-1] * edges[b]))
                runmed.append(np.median(r[sel]))
        ax.plot(cen, runmed, "r-o", ms=4, lw=1.5, label="Binned median")
        ax.axhline(0, color="k", lw=0.8)
        for s in (-thr, thr):
            ax.axhline(s, color="red", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Normalised residual")
    fig.suptitle(f"Residual vs geometric factor and resistance {title_sfx}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "14_residual_vs_K_R.png"), dpi=150)
    plt.close(fig)

    # ── 15 Per-electrode mean |r| ───────────────────────────────────────────
    el_mean, el_count = {}, {}
    for e in elec_num:
        use = (C1 == e) | (C2 == e) | (P1 == e) | (P2 == e)
        el_count[e] = int(use.sum())
        el_mean[e] = float(np.mean(np.abs(r[use]))) if use.any() else np.nan
    em = np.array([el_mean[e] for e in elec_num])
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(elec_x, em, "k-", lw=0.6)
    ax.scatter(elec_x, em, c=em, cmap="viridis", s=22, zorder=3)
    worst_el = sorted((e for e in elec_num if el_count[e] > 0), key=lambda e: -el_mean[e])[:N_WORST_EL]
    for e in worst_el:
        ax.annotate(str(e), (x_of[e], el_mean[e]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color="red")
    ax.axhline(np.sqrt(2 / np.pi), color="gray", ls="-.", lw=0.8,
               label="E|r| for N(0,1) = 0.80")
    mark_water(ax)
    ax.set_xlabel("Electrode x (m)")
    ax.set_ylabel("Mean |normalised residual|")
    ax.set_title(f"Per-electrode mean |residual| (all measurements using the electrode) {title_sfx}",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "15_residual_per_electrode.png"), dpi=150)
    plt.close(fig)

    # ── Report ──────────────────────────────────────────────────────────────
    n_worst = max(1, int(np.ceil(WORST_FRAC * n)))
    worst = np.argsort(-np.abs(r))[:n_worst]
    drop = np.where(np.abs(r) > thr)[0]

    # zone statistics (by pseudosection midpoint)
    zones = []
    if x_hw is not None:
        zones = [("landward of high-water line", x_mid < x_hw),
                 ("intertidal (high- to low-water)", x_mid >= x_hw)]

    rep = os.path.join(RESULTS_DIR, f"{tag}_misfit_report.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write(f"WESTHOEK ERT — MISFIT REPORT — run '{tag}'\n")
        f.write(f"Source: {npz}\n\n")
        f.write(f"n_data               : {n}\n")
        f.write(f"chi2 (log space)     : {chi2:.4f}\n")
        f.write(f"RMS normalised       : {rms_norm:.4f}\n")
        f.write(f"RMS log misfit       : {rms_pct:.2f} %\n")
        f.write(f"mean / median r      : {np.mean(r):.3f} / {med:.3f}\n")
        f.write(f"skewness             : {skew:.3f}\n")
        f.write(f"excess kurtosis      : {kurt:.3f}\n")
        f.write(f"MAD (about median)   : {mad:.4f}   → robust σ = 1.4826·MAD = {1.4826*mad:.4f}\n")
        f.write(f"drop threshold       : |r| > 3 × 1.4826 × MAD = {thr:.3f}  → {len(drop)} data "
                f"({100*len(drop)/n:.1f} %)\n")
        if x_hw is not None:
            f.write(f"high-water crossing  : x = {x_hw:.1f} m ({h_sea:.2f} mTAW, from elec.dat Hsea)\n")
            f.write(f"low water (approx.)  : x = {x_lw:.1f} m (seaward end of land line)\n")
        for name, sel in zones:
            if sel.any():
                f.write(f"  {name:<34}: n={sel.sum():5d}  mean|r|={np.mean(np.abs(r[sel])):.2f}  "
                        f"chi2={np.mean(r[sel]**2):.2f}  median r={np.median(r[sel]):+.2f}  "
                        f"outliers={np.sum(np.abs(r[sel]) > thr)}\n")

        f.write(f"\nWORST {100*WORST_FRAC:.0f}% OF MEASUREMENTS ({n_worst}) — index = 0-based line in land_data.dat\n")
        f.write(f"{'index':>6} {'C1':>4} {'C2':>4} {'P1':>4} {'P2':>4} {'R (Ohm)':>11} {'K (m)':>10} "
                f"{'x_mid':>7} {'residual':>9}\n")
        for i in worst:
            f.write(f"{keep_idx[i]:6d} {C1[i]:4d} {C2[i]:4d} {P1[i]:4d} {P2[i]:4d} {R[i]:11.5f} "
                    f"{K[i]:10.2f} {x_mid[i]:7.1f} {r[i]:+9.2f}\n")

        f.write(f"\n{N_WORST_EL} WORST ELECTRODES (mean |r| over all measurements using them)\n")
        f.write(f"{'elec':>5} {'x (m)':>8} {'z (mTAW)':>9} {'n_meas':>7} {'mean|r|':>8}\n")
        z_of = dict(zip(elec_num, elec_z))
        for e in worst_el:
            f.write(f"{e:5d} {x_of[e]:8.2f} {z_of[e]:9.3f} {el_count[e]:7d} {el_mean[e]:8.2f}\n")
    print(f"[saved] {rep}")

    dl = os.path.join(RESULTS_DIR, f"{tag}_droplist.txt")
    with open(dl, "w", encoding="utf-8") as f:
        f.write(f"# Drop-list from run '{tag}': |r| > 3 x 1.4826 x MAD = {thr:.3f}  ({len(drop)} data)\n")
        f.write("# 0-based measurement index in land_data.dat order (use with 02 --drop-list)\n")
        for i in sorted(keep_idx[drop]):
            f.write(f"{i}\n")
    print(f"[saved] {dl}  ({len(drop)} measurements)")
    print(f"[saved] figures 12–15 in {fig_dir}")


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
