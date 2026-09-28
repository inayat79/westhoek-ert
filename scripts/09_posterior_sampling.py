"""
09_posterior_sampling.py
========================
Westhoek ERT — samples from the linearised (Gaussian) posterior of a 02 run,
and a λ-mixture of several runs on the same mesh.

Posterior in ln ρ space (Laplace approximation around the inverted model m̂):
    m ~ N(m̂, A⁻¹),   A = JᵀWd²J + λ·CᵀC
with exactly the ingredients of the verified R_m computation in 02 (log-log
Jacobian, Wd = 1/err, weighted constraint matrix incl. zWeight, C++ λ).
With A = L Lᵀ (dense Cholesky):  m_k = m̂ + L⁻ᵀ z_k,  z_k ~ N(0, I).

Checks: the per-cell sample std must match √diag(A⁻¹) stored by 02 (sigma_log_m).

Outputs (figures/runs/<tag>_<suffix>/, results/runs/<tag>_<suffix>/; cells with
resolution radius r > r_max hatched/masked; sections down to −45 mTAW):
    23_prob_rho_lt3_gt20.png           P(ρ < 3 Ω·m), P(ρ > 20 Ω·m) per cell
    24_interface_bands.png             lens top / base: median + 5–95 % over samples vs analytic band
    25_posterior_samples.png           m̂ and 6 random samples
    26_lambda_mixture.png              P(ρ < 3) and interface bands, single λ vs pooled λ mixture
    posterior_sampling_report.txt      numbers for the report
    posterior_interface_bands.npz      per-column band statistics

RUN
    python 09_posterior_sampling.py --tag g1_lam50 --suffix v2 --r-max 10 \
        --mix-tags g1_lam5 g1_lam10 deep_drop_data g1_lam50 g1_lam100
"""

import argparse
import os
import time

import numpy as np
import scipy.linalg as sla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import westhoek_common as wc

LN3, LN20 = np.log(3.0), np.log(20.0)
Z90 = 1.6449
MEM_LIMIT_GB = 4.0
ZONES = {"dune (x 0–90 m)": (0, 90), "lens (x 90–370 m)": (90, 370), "discharge (x 380–470 m)": (380, 470)}
CMAP_RHO = "RdBu_r"
NORM_RHO = mcolors.LogNorm(0.5, 500)


# ─────────────────────────────────────────────────────────────────────────────
def draw_samples(run, n, rng, label=""):
    """Return (ln m̂, samples (n_cells, n), L, timings) for one run."""
    t = time.time()
    L, A, JtWJ, lam = wc.cholesky_A(run)
    t_chol = time.time() - t
    Z = rng.standard_normal((A.shape[0], n))
    X = sla.solve_triangular(L, Z, lower=True, trans="T")        # Lᵀ x = z  →  x = L⁻ᵀ z
    lnm = np.log(run["model"])
    std = X.std(axis=1, ddof=1)
    ratio = std / run["sigma_log_m"]
    print(f"[{label}] λ={lam:g}: Cholesky {t_chol:.2f}s; sample std / √diag(A⁻¹): "
          f"median {np.median(ratio):.4f}, min {ratio.min():.4f}, max {ratio.max():.4f}")
    return lnm, lnm[:, None] + X, dict(lam=lam, ratio=ratio, t_chol=t_chol, n=A.shape[0], L=L)


def interface_samples(cs, lnS, r_grid_log10, r_max, chunk=100):
    """Main-lens top/base elevation per column and sample; also a mask r > r_max."""
    nx, ns = len(cs.x), lnS.shape[1]
    top = np.full((nx, ns), np.nan); base = np.full((nx, ns), np.nan)
    for i in range(0, ns, chunk):
        V = cs(lnS[:, i:i + chunk] / np.log(10))
        t, b = wc.main_lens(V, cs.z)
        top[:, i:i + chunk], base[:, i:i + chunk] = t, b
    dz = cs.z[0] - cs.z[1]
    def masked(Zc):
        k = np.clip(np.floor((cs.z[0] - np.nan_to_num(Zc, nan=cs.z[0])) / dz).astype(int), 0, len(cs.z) - 1)
        rr = r_grid_log10[k, np.arange(nx)[:, None]]
        return np.isfinite(Zc) & (rr > np.log10(r_max))
    return top, base, masked(top), masked(base)


def band_stats(Z, M=None):
    """Per column: present fraction, median, P5, P95 (optionally ignoring masked samples)."""
    Zu = np.where(M, np.nan, Z) if M is not None else Z
    pres = np.isfinite(Zu).mean(axis=1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(Zu, axis=1)
            p5 = np.nanpercentile(Zu, 5, axis=1)
            p95 = np.nanpercentile(Zu, 95, axis=1)
    return pres, med, p5, p95


def analytic_band(cs, run):
    """Main-lens top/base of m̂ with σ_z = σ_log10 / |∂log10ρ/∂z| at the crossing."""
    V = cs(np.log10(run["model"])); S = cs(run["sigma_log_m"] / np.log(10))
    t, b = wc.main_lens(V[:, :, None], cs.z)
    dz = cs.z[0] - cs.z[1]
    out = []
    for Zc in (t[:, 0], b[:, 0]):
        sig = np.full(len(cs.x), np.nan)
        for j, zc in enumerate(Zc):
            if not np.isfinite(zc):
                continue
            k = int(np.clip(np.floor((cs.z[0] - zc) / dz), 0, len(cs.z) - 2))
            g = (V[k + 1, j] - V[k, j]) / (cs.z[k + 1] - cs.z[k])
            sig[j] = abs(S[k, j] / g) if g != 0 else np.nan
        out.append((Zc, sig))
    return out


def draw_band(ax, x, pres, med, p5, p95, color, label, min_pres=0.5):
    ok = pres >= min_pres
    m = np.where(ok, med, np.nan)
    ax.fill_between(x, np.where(ok, p5, np.nan), np.where(ok, p95, np.nan), color=color, alpha=0.35, lw=0,
                    zorder=6)
    ax.plot(x, m, "-", color=color, lw=1.5, zorder=7, label=label)


def zone_prob_summary(P, run, poor, depth):
    lines = []
    cx, cz = run["cell_x"], run["cell_z"]
    for zn, (x0, x1) in ZONES.items():
        s = (cx >= x0) & (cx < x1) & ~poor
        if not s.any():
            continue
        hi = s & (P > 0.9); mid = s & (P >= 0.1) & (P <= 0.9); lo = s & (P < 0.1)
        def rng_(sel):
            return (f"elev {cz[sel].min():.1f} … {cz[sel].max():.1f} mTAW, depth {depth[sel].min():.1f}–"
                    f"{depth[sel].max():.1f} m" if sel.any() else "—")
        lines.append(f"  {zn:<24} unmasked cells {s.sum():5d}: P>0.9 {100*hi.sum()/s.sum():5.1f}%  "
                     f"0.1–0.9 {100*mid.sum()/s.sum():5.1f}%  P<0.1 {100*lo.sum()/s.sum():5.1f}%")
        lines.append(f"  {'':<24}   P>0.9: {rng_(hi)}")
        lines.append(f"  {'':<24}   0.1–0.9: {rng_(mid)}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Linearised posterior sampling")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--suffix", default="v2")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--r-max", type=float, default=10.0)
    ap.add_argument("--r-max-extra", nargs="*", type=float, default=[15.0],
                    help="additional r_max values for the lens-base band table")
    ap.add_argument("--mix-tags", nargs="*", default=[])
    ap.add_argument("--report-x", nargs="*", type=float, default=[150, 200, 250, 300, 350])
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    name = f"{args.tag}_{args.suffix}"
    fig_dir = os.path.join(wc.FIG_DIR, "runs", name)
    res_dir = os.path.join(wc.RESULTS_DIR, "runs", name)
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)
    for f in ("23_prob_rho_lt3_gt20.png", "24_interface_bands.png", "25_posterior_samples.png",
              "26_lambda_mixture.png"):
        if os.path.exists(os.path.join(fig_dir, f)):
            raise SystemExit(f"{f} already exists in {fig_dir} — choose a new --suffix (no overwriting).")

    run = wc.load_run(args.tag)
    n = len(run["model"])
    mem = (2 * n * n + n * args.n * (1 + len(args.mix_tags))) * 8 / 1e9
    R = [f"POSTERIOR SAMPLING — run '{args.tag}'  (λ = {float(run['lam']):g}, χ² = {float(run['chi2']):.3f})",
         f"n_cells = {n}; dense A and L: {2*n*n*8/1e6:.0f} MB; samples: {n*args.n*8/1e6:.0f} MB per run; "
         f"peak estimate {mem:.2f} GB (limit {MEM_LIMIT_GB} GB)"]
    print("\n".join(R))
    if mem > MEM_LIMIT_GB:
        raise SystemExit("Dense Cholesky would exceed the memory limit — stopping to ask before proceeding.")

    mesh, err = wc.rebuild_paradomain(run)
    polys = wc.cell_polygons(mesh)
    poor = run["res_radius"] > args.r_max
    ex, ez = wc.land_surface()
    depth = np.interp(run["cell_x"], ex, ez) - run["cell_z"]
    cs = wc.ColumnSampler(mesh, np.arange(np.ceil(ex.min()), np.floor(ex.max()) + 1, 2.0))
    r_grid = cs(np.log10(run["res_radius"]))

    # ── single-λ posterior ───────────────────────────────────────────────────
    lnm, S1, info = draw_samples(run, args.n, rng, args.tag)
    R += [f"Samples: {args.n}, seed {args.seed}",
          f"CHECK sample std / √diag(A⁻¹) (02's sigma_log_m): median {np.median(info['ratio']):.4f}, "
          f"min {info['ratio'].min():.4f}, max {info['ratio'].max():.4f}; "
          f"cells within ±5 %: {100*np.mean(np.abs(info['ratio'] - 1) <= 0.05):.1f}% "
          f"(expected sampling s.d. of the ratio ≈ {1/np.sqrt(2*(args.n-1)):.3f})"]
    P3 = (S1 < LN3).mean(axis=1); P20 = (S1 > LN20).mean(axis=1)

    # ── figure 23: probabilities ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    for ax, P, ttl in ((axes[0], P3, "P(ρ < 3 Ω·m)"), (axes[1], P20, "P(ρ > 20 Ω·m)")):
        pc = wc.draw_cells(ax, polys, P, "cividis", mcolors.Normalize(0, 1), mask=poor)
        wc.hatch_cells(ax, polys, poor)
        Pg = cs(P); XX, ZZ = np.meshgrid(cs.x, cs.z)
        Pg[cs(poor.astype(float)) > 0.5] = np.nan
        c = ax.contour(XX, ZZ, Pg, levels=[0.1, 0.5, 0.9], colors=["white", "0.6", "k"], linewidths=1.0)
        ax.clabel(c, fmt="%.1f", fontsize=7)
        wc.setup_section(ax, f"{ttl} — {args.n} linearised-posterior samples, run '{args.tag}' "
                             f"(λ={info['lam']:g})")
        fig.colorbar(pc, ax=ax, pad=0.01, fraction=0.025, label="probability")
    wc.white_legend(axes[1], handles=[Patch(facecolor="white", edgecolor="0.35", hatch="////",
                                            label=f"resolution radius r > {args.r_max:g} m (masked)")],
                    loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "23_prob_rho_lt3_gt20.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    R += ["", "P(ρ < 3 Ω·m) by zone (unmasked cells, r ≤ r_max):"] + zone_prob_summary(P3, run, poor, depth)

    # ── interfaces ───────────────────────────────────────────────────────────
    top1, base1, mt1, mb1 = interface_samples(cs, S1, r_grid, args.r_max)
    (ztop, stop_), (zbase, sbase) = analytic_band(cs, run)
    # analytic σ_z from the variance of the interpolated value wᵀA⁻¹w (step 0a)
    swtop, _ = wc.sigma_z_interp(cs, info["L"], np.log10(run["model"]), ztop)
    swbase, _ = wc.sigma_z_interp(cs, info["L"], np.log10(run["model"]), zbase)
    # lens-base masks for additional r_max values (step 0c)
    extra_base = {}
    for rx in args.r_max_extra:
        _, _, _, mbx = interface_samples(cs, S1, r_grid, rx)
        extra_base[rx] = band_stats(base1, mbx)
    single = dict(top=band_stats(top1, mt1), base=band_stats(base1, mb1),
                  top_raw=band_stats(top1), base_raw=band_stats(base1))

    def analytic_masked(zc):
        dzg = cs.z[0] - cs.z[1]
        k = np.clip(np.floor((cs.z[0] - np.nan_to_num(zc, nan=cs.z[0])) / dzg).astype(int), 0, len(cs.z) - 1)
        return np.isfinite(zc) & (r_grid[k, np.arange(len(cs.x))] > np.log10(args.r_max))

    def interface_fig(ax, stats, title, bg=True):
        if bg:
            wc.draw_cells(ax, polys, run["model"], CMAP_RHO, NORM_RHO, alpha=0.45)
            wc.hatch_cells(ax, polys, poor)
        draw_band(ax, cs.x, *stats["top"], "navy", "lens top: median, 5–95 % band")
        draw_band(ax, cs.x, *stats["base"], "darkorange", "lens base: median, 5–95 % band")
        for zc, sg, sw, col in ((ztop, stop_, swtop, "navy"), (zbase, sbase, swbase, "darkorange")):
            zc = np.where(analytic_masked(zc), np.nan, zc)      # same r_max mask as the samples
            ax.plot(cs.x, zc - Z90 * sg, ":", color=col, lw=0.9, zorder=8)
            ax.plot(cs.x, zc + Z90 * sg, ":", color=col, lw=0.9, zorder=8)
            ax.plot(cs.x, zc - Z90 * sw, "--", color=col, lw=0.8, zorder=8)
            ax.plot(cs.x, zc + Z90 * sw, "--", color=col, lw=0.8, zorder=8)
        wc.setup_section(ax, title)

    fig, ax = plt.subplots(figsize=(15, 5.8))
    interface_fig(ax, single, f"Main-lens ρ = 3 Ω·m interfaces from {args.n} posterior samples "
                              f"(λ={info['lam']:g}); dotted: analytic ±1.645σ_z of m̂")
    h, _ = ax.get_legend_handles_labels()
    h += [Line2D([0], [0], color="0.3", ls=":", lw=0.9, label="analytic 90 % band, cell σ"),
          Line2D([0], [0], color="0.3", ls="--", lw=0.8, label="analytic 90 % band, wᵀA⁻¹w"),
          Patch(facecolor="white", edgecolor="0.35", hatch="////", label=f"r > {args.r_max:g} m")]
    wc.white_legend(ax, handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    fig.savefig(os.path.join(fig_dir, "24_interface_bands.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ── figure 25: m̂ and 6 random samples ────────────────────────────────────
    pick = rng.choice(args.n, 6, replace=False)
    fig, axes = plt.subplots(4, 2, figsize=(16, 13))
    axes = axes.ravel()
    for ax, (vals, ttl) in zip(axes, [(np.exp(lnm), "m̂ (inverted model)")] +
                               [(np.exp(S1[:, k]), f"posterior sample #{k}") for k in pick]):
        pc = wc.draw_cells(ax, polys, vals, CMAP_RHO, NORM_RHO)
        wc.hatch_cells(ax, polys, poor)
        wc.setup_section(ax, ttl)
    axes[-1].axis("off")
    cb = fig.colorbar(pc, ax=axes[-1], location="left", fraction=0.6, label="ρ (Ω·m)")
    axes[-1].text(0.55, 0.5, f"run '{args.tag}', λ = {info['lam']:g}\nsamples m̂ + L⁻ᵀz, z ~ N(0, I)\n"
                              f"hatched: r > {args.r_max:g} m", transform=axes[-1].transAxes, va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "25_posterior_samples.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── λ mixture ────────────────────────────────────────────────────────────
    mix = None
    if args.mix_tags:
        pooled = []
        R += ["", f"λ MIXTURE: {args.n} samples from each of {args.mix_tags}"]
        for t in args.mix_tags:
            rr = wc.load_run(t)
            if rr["cell_x"].shape != run["cell_x"].shape or not np.allclose(rr["cell_x"], run["cell_x"]):
                raise SystemExit(f"Mixture run '{t}' is on a different mesh.")
            _, Sk, ik = draw_samples(rr, args.n, rng, t)
            R.append(f"  {t:<16} λ={ik['lam']:>5g}  χ²={float(rr['chi2']):.3f}  std check median "
                     f"{np.median(ik['ratio']):.4f}, max {ik['ratio'].max():.4f}")
            pooled.append(Sk)
        SP = np.hstack(pooled); del pooled
        P3m = (SP < LN3).mean(axis=1)
        topm, basem, mtm, mbm = interface_samples(cs, SP, r_grid, args.r_max)
        mix = dict(top=band_stats(topm, mtm), base=band_stats(basem, mbm),
                   top_raw=band_stats(topm), base_raw=band_stats(basem))
        R += ["", "P(ρ < 3 Ω·m) by zone — POOLED λ mixture:"] + zone_prob_summary(P3m, run, poor, depth)
        dP = np.abs(P3m - P3)[~poor]
        R.append(f"|P_pooled − P_single| (unmasked cells): median {np.median(dP):.3f}, P95 {np.percentile(dP, 95):.3f}, "
                 f"max {dP.max():.3f}")

        fig, axes = plt.subplots(2, 2, figsize=(22, 10))
        for ax, P, ttl in ((axes[0, 0], P3, f"P(ρ<3) — single λ = {info['lam']:g}"),
                           (axes[0, 1], P3m, f"P(ρ<3) — pooled λ ∈ {{5, 10, 20, 50, 100}}")):
            pc = wc.draw_cells(ax, polys, P, "cividis", mcolors.Normalize(0, 1), mask=poor)
            wc.hatch_cells(ax, polys, poor)
            wc.setup_section(ax, ttl)
            fig.colorbar(pc, ax=ax, pad=0.01, fraction=0.025, label="probability")
        interface_fig(axes[1, 0], single, f"interfaces — single λ = {info['lam']:g}")
        interface_fig(axes[1, 1], mix, "interfaces — pooled λ mixture")
        h, _ = axes[1, 1].get_legend_handles_labels()
        h += [Line2D([0], [0], color="0.3", ls=":", lw=0.9, label=f"analytic 90 % band (m̂, λ={info['lam']:g})")]
        wc.white_legend(axes[1, 1], handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "26_lambda_mixture.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── band table at the requested x ───────────────────────────────────────
    def row(st, j):
        pres, med, p5, p95 = (a[j] for a in st)
        if pres == 0 or not np.isfinite(med):
            return f"{'absent':>28}"
        return f"{med:7.2f} [{p5:6.2f}, {p95:6.2f}] w{p95-p5:5.2f} ({100*pres:3.0f}%)"

    R += ["", "LENS TOP / BASE ELEVATION (mTAW): median [P5, P95] w=90% width (fraction of samples with the "
              "crossing).", "'masked' = crossings with r > r_max removed; 'raw' = all crossings.",
          "Analytic: main-lens crossing of m̂ ± 1.645·σ_z  (σ_z = σ_log10 / |∂log10ρ/∂z|).", ""]
    for kind, zc, sg in (("top", ztop, stop_), ("base", zbase, sbase)):
        R.append(f"— lens {kind} —")
        R.append(f"{'x':>5} | {'single λ masked':>32} | {'single λ raw':>32} | {'pooled masked':>32} | "
                 f"{'pooled raw':>32} | {'analytic m̂ ± 90%':>24} | r at m̂ crossing")
        for x in args.report_x:
            j = int(np.argmin(np.abs(cs.x - x)))
            an = (f"{zc[j]:7.2f} ± {Z90*sg[j]:5.2f} (w{2*Z90*sg[j]:5.2f})" if np.isfinite(zc[j]) else f"{'absent':>24}")
            if np.isfinite(zc[j]):
                k = int(np.clip(np.floor((cs.z[0] - zc[j]) / (cs.z[0] - cs.z[1])), 0, len(cs.z) - 1))
                rr_ = f"{10**r_grid[k, j]:5.1f} m"
            else:
                rr_ = "  —"
            R.append(f"{x:5.0f} | {row(single[kind], j):>32} | {row(single[kind + '_raw'], j):>32} | "
                     + (f"{row(mix[kind], j):>32} | {row(mix[kind + '_raw'], j):>32} | " if mix else "")
                     + f"{an:>24} | {rr_}")
        R.append("")

    # step 0a/0c: 90 % widths — sampled vs analytic (cell σ) vs analytic (wᵀA⁻¹w), single λ
    R += ["90 % WIDTH OF THE INTERFACE BAND (m), single λ: sampled (P95−P5, all crossings) vs analytic with "
          "cell σ vs analytic with the variance of the interpolated value wᵀA⁻¹w"]
    for kind, zc, sg, sw in (("top", ztop, stop_, swtop), ("base", zbase, sbase, swbase)):
        R.append(f"— lens {kind} —   x | m̂ crossing | sampled | analytic cell σ | analytic wᵀA⁻¹w | "
                 f"sampled/wᵀA⁻¹w")
        pres, med, p5, p95 = single[kind + "_raw"]
        for x in args.report_x:
            j = int(np.argmin(np.abs(cs.x - x)))
            if not np.isfinite(zc[j]):
                R.append(f"  {x:5.0f} | absent"); continue
            ws = p95[j] - p5[j]
            R.append(f"  {x:5.0f} | {zc[j]:7.2f} | {ws:6.2f} | {2*Z90*sg[j]:6.2f} | {2*Z90*sw[j]:6.2f} | "
                     f"{ws/(2*Z90*sw[j]):5.2f}")
    okb = np.isfinite(swbase) & (single["base_raw"][0] >= 0.5)
    qb = (single["base_raw"][3] - single["base_raw"][2])[okb] / (2 * Z90 * swbase[okb])
    R += [f"  whole profile, lens base: sampled / wᵀA⁻¹w width median {np.median(qb):.2f}, "
          f"P10 {np.percentile(qb, 10):.2f}, P90 {np.percentile(qb, 90):.2f} over {okb.sum()} columns", ""]
    R.append("LENS-BASE BANDS BY r_max (single λ): median [P5, P95] (share of samples with an unmasked crossing)")
    hdr = "    x | " + " | ".join(f"{'r_max = ' + format(v, 'g') + ' m':>34}" for v in [args.r_max] + args.r_max_extra)
    R.append(hdr)
    for x in args.report_x:
        j = int(np.argmin(np.abs(cs.x - x)))
        cells = [row(single["base"], j)] + [row(extra_base[v], j) for v in args.r_max_extra]
        R.append(f"{x:5.0f} | " + " | ".join(f"{c:>34}" for c in cells))
    R.append("")

    # sampled vs analytic 90 % widths along the whole profile (unmasked single-λ)
    for kind, zc, sg in (("top", ztop, stop_), ("base", zbase, sbase)):
        pres, med, p5, p95 = single[kind]
        ok = (pres >= 0.5) & np.isfinite(sg) & np.isfinite(med)
        if ok.any():
            q = (p95 - p5)[ok] / (2 * Z90 * sg[ok])
            dm = np.abs(med[ok] - zc[ok])
            R.append(f"lens {kind}: sampled/analytic 90% width over {ok.sum()} unmasked columns: median {np.median(q):.2f}, "
                     f"P10 {np.percentile(q, 10):.2f}, P90 {np.percentile(q, 90):.2f};  "
                     f"|sampled median − m̂ crossing| median {np.median(dm):.2f} m, P90 {np.percentile(dm, 90):.2f} m")
        if mix:
            pres, med, p5, p95 = mix[kind]
            okm = (pres >= 0.5) & np.isfinite(sg) & np.isfinite(med)
            if okm.any():
                q = (p95 - p5)[okm] / (2 * Z90 * sg[okm])
                R.append(f"lens {kind} (pooled): sampled/analytic(λ={info['lam']:g}) 90% width: median {np.median(q):.2f}, "
                         f"P90 {np.percentile(q, 90):.2f} over {okm.sum()} columns")

    np.savez(os.path.join(res_dir, "posterior_interface_bands.npz"), x=cs.x,
             analytic_top=ztop, analytic_top_sigma=stop_, analytic_base=zbase, analytic_base_sigma=sbase,
             analytic_top_sigma_w=swtop, analytic_base_sigma_w=swbase,
             **{f"single_{k}_{q}": v for k in ("top", "base", "top_raw", "base_raw")
                for q, v in zip(("present", "median", "p5", "p95"), single[k])},
             **({f"pooled_{k}_{q}": v for k in ("top", "base", "top_raw", "base_raw")
                 for q, v in zip(("present", "median", "p5", "p95"), mix[k])} if mix else {}),
             P_lt3=P3, P_gt20=P20, **({"P_lt3_pooled": P3m} if mix else {}))
    rep = os.path.join(res_dir, "posterior_sampling_report.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(R) + "\n")
    print("\n".join(R))
    print(f"[saved] {rep}\n[saved] figures 23–26 in {fig_dir}")


if __name__ == "__main__":
    main()
