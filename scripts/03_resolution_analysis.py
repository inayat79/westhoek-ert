"""
03_resolution_analysis.py
=========================
Westhoek ERT – resolution & linearised uncertainty analysis of a 02 run.

All model-space quantities are drawn on the inversion mesh itself (cells
filled with their values; the paraDomain is rebuilt from the run's settings and
verified against the stored cell centres — see westhoek_common.py). Section
figures go down to −45 mTAW at a stated vertical exaggeration, poorly resolved
cells are hatched, colour convention everywhere: blue = conductive, red =
resistive.

INPUT   data/processed[/runs/<tag>]/westhoek_inversion_results.npz  (from 02)

OUTPUTS (figures/[runs/<tag>[_<suffix>]/])
    05_resistivity_model.png       resistivity section + hydrogeological labels
    06_resolution_diagonal.png     Rm_ii (mesh-dependent; see 16 for the radius)
    07_cumulative_sensitivity.png  log10 normalised cumulative sensitivity
    08_uncertainty_map.png         σ_log10(ρ) and multiplicative uncertainty factor
    16_resolution_radius.png       resolution radius (Friedel 2003), log colours
    09_interface_uncertainty.png   ρ = 3 Ω·m lens top / lens base + analytic 90 % band
    10_psf_averaging_kernels.png   PSF (Rm column) and averaging kernel (Rm row / area)
    11_lambda_comparison.png       regularisation spread across λ runs vs linearised σ
results/[runs/<tag>[_<suffix>]/]  westhoek_psf.npz, westhoek_uq_summary.txt

RUN
    python 03_resolution_analysis.py --tag g1_lam50 --suffix v2 --mask-mode radius --r-max 10 \
        --ensemble-tags g1_lam5 g1_lam10 deep_drop_data g1_lam50 g1_lam100
"""

import argparse
import io
import contextlib
import os

import numpy as np
import scipy.linalg as sla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import westhoek_common as wc

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
RHO_SALT_THR  = 3.0      # Ω·m — saltwater (< threshold)
RHO_FRESH_THR = 20.0     # Ω·m — freshwater (> threshold)
RM_WELL_RESOLVED  = 0.5
RM_POOR_THRESHOLD = 0.2
Z90 = 1.6449             # 90 % two-sided normal quantile

# PSF / averaging-kernel targets (x, z in m) — inside land coverage
PSF_SITES = {
    "Dune aquifer"  : (60,  -15),
    "Lens interior" : (250, -10),
    "Lens base"     : (250, -25),
    "Discharge zone": (430, -15),
}

# Hydrogeological labels (x, z) — placed clear of legends (legends sit below the axes)
LABELS = [
    (30,  1.0, "Dune\nfreshwater"),
    (230, -4.0, "Saltwater lens"),
    (425, -9.0, "Discharge\nzone"),
    (200, -32.0, "Deeper brackish / fresher\n(poorly resolved)"),
]

CMAP_RHO = "RdBu_r"      # blue = conductive, red = resistive
NORM_RHO = mcolors.LogNorm(vmin=0.5, vmax=500)
RHO_TICKS = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]

MASK = {"mode": "rm", "r_max": None}


# ─────────────────────────────────────────────────────────────────────────────
def poor_mask_cells(res):
    """Poorly resolved cells: Rm < 0.2 (rm mode) or resolution radius > r_max (radius mode)."""
    if MASK["mode"] == "rm":
        return res["Rm_diag"] < RM_POOR_THRESHOLD
    return res["res_radius"] > MASK["r_max"]


def mask_desc():
    return (f"Rm < {RM_POOR_THRESHOLD}" if MASK["mode"] == "rm"
            else f"resolution radius r > {MASK['r_max']:g} m")


def uq_label(res):
    if res.get("sigma_scale", 1.0) != 1.0:
        return f"  [σ scaled by √χ² = {res['sigma_scale']:.2f}]"
    if np.isfinite(res.get("chi2", np.nan)) and res["chi2"] > 2:
        return f"  [UNSCALED, χ² = {res['chi2']:.1f}]"
    return ""


def load_results(tag, path, scale_by_chi2=False):
    res = wc.load_run(tag, path)
    need = {"model", "cell_x", "cell_z", "Rm_diag", "cum_sens", "sigma_log_m"}
    miss = need - set(res)
    if miss:
        raise KeyError(f"Missing keys in results file: {miss}")
    res["chi2"] = float(res["chi2"]) if "chi2" in res else np.nan
    res["sigma_scale"] = 1.0
    if np.isfinite(res["chi2"]) and res["chi2"] > 2:
        print("\n" + "!" * 78)
        print(f"  WARNING: χ² = {res['chi2']:.2f} > 2 — linearised σ likely underestimated by "
              f"≈ √χ² = {np.sqrt(res['chi2']):.2f}×. "
              + ("Scaling σ (--scale-by-chi2)." if scale_by_chi2 else "Pass --scale-by-chi2 to inflate σ."))
        print("!" * 78 + "\n")
    if scale_by_chi2 and np.isfinite(res["chi2"]):
        res["sigma_scale"] = float(np.sqrt(res["chi2"]))
        res["sigma_log_m"] = res["sigma_log_m"] * res["sigma_scale"]
    if "res_radius" not in res:
        if MASK["mode"] == "radius":
            raise KeyError("Run has no res_radius (rerun with the current 02) — use --mask-mode rm")
    res["log10_rho"] = np.log10(res["model"])
    res["sigma_log10"] = res["sigma_log_m"] / np.log(10)
    res["rho_unc_factor"] = np.exp(res["sigma_log_m"])
    print(f"[load] n_cells={len(res['model'])}  ρ [{res['model'].min():.2f}, {res['model'].max():.2f}] Ω·m  "
          f"χ²={res['chi2']:.3f}  λ={float(res['lam']):g}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
def rho_colorbar(fig, pc, ax):
    cb = fig.colorbar(pc, ax=ax, pad=0.01, fraction=0.025)
    cb.set_label("Resistivity ρ (Ω·m)")
    cb.set_ticks(RHO_TICKS); cb.set_ticklabels([f"{t:g}" for t in RHO_TICKS])
    return cb


def section(fig, ax, M, values, cmap, norm, title, hatch=True, **kw):
    pc = wc.draw_cells(ax, M["polys"], values, cmap, norm, **kw)
    if hatch:
        wc.hatch_cells(ax, M["polys"], M["poor"])
    wc.setup_section(ax, title)
    return pc


def hatch_handle():
    return Patch(facecolor="white", edgecolor="0.35", hatch="////", label=f"poorly resolved: {mask_desc()}")


def legend_below(ax, handles, ncol=4):
    wc.white_legend(ax, handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol)


# ─────────────────────────────────────────────────────────────────────────────
def plot_resistivity_model(res, M, out_dir):
    fig, ax = plt.subplots(figsize=(15, 5.6))
    pc = section(fig, ax, M, res["model"], CMAP_RHO, NORM_RHO,
                 "Westhoek land ERT — inverted resistivity (cells of the inversion mesh)")
    V = M["cs"](res["log10_rho"])
    XX, ZZ = np.meshgrid(M["cs"].x, M["cs"].z)
    ax.contour(XX, ZZ, V, levels=[np.log10(RHO_SALT_THR)], colors="navy", linewidths=1.4, linestyles="--")
    ax.contour(XX, ZZ, V, levels=[np.log10(RHO_FRESH_THR)], colors="darkred", linewidths=1.1, linestyles=":")
    for x, z, t in LABELS:
        wc.label(ax, x, z, t)
    rho_colorbar(fig, pc, ax)
    legend_below(ax, [Line2D([0], [0], color="navy", ls="--", lw=1.4, label=f"ρ = {RHO_SALT_THR:g} Ω·m"),
                      Line2D([0], [0], color="darkred", ls=":", lw=1.1, label=f"ρ = {RHO_FRESH_THR:g} Ω·m"),
                      Line2D([0], [0], color="k", lw=0.8, label="land surface"), hatch_handle()])
    fig.savefig(os.path.join(out_dir, "05_resistivity_model.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_resolution_diagonal(res, M, out_dir):
    fig, ax = plt.subplots(figsize=(15, 5.6))
    pc = section(fig, ax, M, np.clip(res["Rm_diag"], 0, 1), "viridis", mcolors.Normalize(0, 1),
                 "Model resolution diagonal Rm_ii  (depends on cell size — compare figure 16)")
    fig.colorbar(pc, ax=ax, pad=0.01, fraction=0.025, label="Rm_ii")
    frac_well = np.mean(res["Rm_diag"] >= RM_WELL_RESOLVED)
    wc.label(ax, 60, -40, f"Rm ≥ {RM_WELL_RESOLVED}: {100*frac_well:.1f}% of cells", fontsize=8)
    legend_below(ax, [hatch_handle()], ncol=1)
    fig.savefig(os.path.join(out_dir, "06_resolution_diagonal.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_sensitivity(res, M, out_dir):
    cs = res["cum_sens"] / res["cum_sens"].max()
    fig, ax = plt.subplots(figsize=(15, 5.6))
    pc = section(fig, ax, M, np.log10(np.clip(cs, 1e-6, None)), "YlOrRd", mcolors.Normalize(-5, 0),
                 "Data coverage — log10 normalised cumulative sensitivity Σ(Wd·J)²")
    fig.colorbar(pc, ax=ax, pad=0.01, fraction=0.025, label="log10(normalised cumulative sensitivity)")
    legend_below(ax, [hatch_handle()], ncol=1)
    fig.savefig(os.path.join(out_dir, "07_cumulative_sensitivity.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_map(res, M, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    s = res["sigma_log10"]; f = res["rho_unc_factor"]
    ok = ~M["poor"]
    vmax_s = np.percentile(s[ok], 98) if ok.any() else 0.5
    pc0 = section(fig, axes[0], M, s, "hot_r", mcolors.Normalize(0, vmax_s),
                  "(a) linearised posterior σ_log10(ρ)" + uq_label(res), mask=M["poor"])
    fig.colorbar(pc0, ax=axes[0], pad=0.01, fraction=0.025, label="σ_log10(ρ) (decades)")
    vmax_f = np.percentile(f[ok], 98) if ok.any() else 3
    pc1 = section(fig, axes[1], M, f, "plasma_r", mcolors.Normalize(1, vmax_f),
                  "(b) multiplicative uncertainty factor exp(σ_ln)  (ρ/f … ρ·f = 68 %)" + uq_label(res),
                  mask=M["poor"])
    fig.colorbar(pc1, ax=axes[1], pad=0.01, fraction=0.025, label="factor")
    legend_below(axes[1], [hatch_handle()], ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_uncertainty_map.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_resolution_radius(res, M, out_dir):
    r = res["res_radius"]
    fig, ax = plt.subplots(figsize=(15, 5.6))
    pc = section(fig, ax, M, r, "magma_r", mcolors.LogNorm(1, 100),
                 "Resolution radius r = √(A / (π·Rm_ii))  (Friedel 2003)", hatch=False)
    fig.colorbar(pc, ax=ax, pad=0.01, fraction=0.025, label="resolution radius r (m)")
    V = M["cs"](res["log10_rho"]); XX, ZZ = np.meshgrid(M["cs"].x, M["cs"].z)
    ax.contour(XX, ZZ, V, levels=[np.log10(RHO_SALT_THR)], colors="cyan", linewidths=1.4, linestyles="--")
    h = [Line2D([0], [0], color="cyan", ls="--", lw=1.4, label=f"ρ = {RHO_SALT_THR:g} Ω·m")]
    if MASK["mode"] == "radius":
        R = M["cs"](np.log10(r))
        ax.contour(XX, ZZ, R, levels=[np.log10(MASK["r_max"])], colors="white", linewidths=1.3)
        h.append(Line2D([0], [0], color="white", lw=1.3, label=f"r = r_max = {MASK['r_max']:g} m"))
        ax.set_facecolor("0.85")
    legend_below(ax, h, ncol=2)
    fig.savefig(os.path.join(out_dir, "16_resolution_radius.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def interface_table(res, M):
    """All ρ = 3 Ω·m crossings per column + analytic σ_z and r_max mask for each."""
    cs = M["cs"]
    V = cs(res["log10_rho"]); S = cs(res["sigma_log10"])
    Rm = cs(np.log10(res["res_radius"])) if MASK["mode"] == "radius" else cs(res["Rm_diag"])
    cols, masks, sig = [], [], []
    dz = cs.z[1] - cs.z[0]
    for j in range(len(cs.x)):
        cc = wc.crossings(V[:, j], cs.z)
        cols.append(cc)
        mk, sg = [], []
        for kind, zc in cc:
            k = int(np.clip(np.floor((zc - cs.z[0]) / dz), 0, len(cs.z) - 2))
            grad = (V[k + 1, j] - V[k, j]) / dz                       # d log10ρ / dz
            s = S[k, j] + (zc - cs.z[k]) / dz * (S[k + 1, j] - S[k, j])
            sg.append(abs(s / grad) if grad != 0 else np.nan)
            m_here = Rm[k, j] + (zc - cs.z[k]) / dz * (Rm[k + 1, j] - Rm[k, j])
            mk.append(m_here > np.log10(MASK["r_max"]) if MASK["mode"] == "radius" else m_here < RM_POOR_THRESHOLD)
        masks.append(mk); sig.append(sg)
    return cols, masks, sig


def plot_interface_uncertainty(res, M, out_dir):
    cols, masks, sig = interface_table(res, M)
    x = M["cs"].x
    fig, ax = plt.subplots(figsize=(15, 5.6))
    pc = section(fig, ax, M, res["model"], CMAP_RHO, NORM_RHO,
                 "ρ = 3 Ω·m interfaces (lens top / lens base) with analytic 90 % band ±1.645·σ_z" + uq_label(res),
                 alpha=0.55)
    # bands: same topology rule as the lines
    kinds = [tuple(k for k, _ in c) for c in cols]
    j = 0
    while j < len(x):
        j2 = j
        while j2 + 1 < len(x) and kinds[j2 + 1] == kinds[j]:
            j2 += 1
        for k, kind in enumerate(kinds[j]):
            zz = np.array([cols[i][k][1] for i in range(j, j2 + 1)])
            ss = np.array([sig[i][k] for i in range(j, j2 + 1)])
            mm = np.array([masks[i][k] for i in range(j, j2 + 1)])
            zz = np.where(mm, np.nan, zz)
            ax.fill_between(x[j:j2 + 1], zz - Z90 * ss, zz + Z90 * ss, color="gold" if kind == "base" else "lightskyblue",
                            alpha=0.6, lw=0, zorder=6)
        j = j2 + 1
    wc.draw_interfaces(ax, x, cols, masks)
    rho_colorbar(fig, pc, ax)
    h, l = ax.get_legend_handles_labels()
    h += [Patch(color="lightskyblue", alpha=0.6, label="90 % band, top"),
          Patch(color="gold", alpha=0.6, label="90 % band, base"), hatch_handle()]
    legend_below(ax, h, ncol=3)
    fig.savefig(os.path.join(out_dir, "09_interface_uncertainty.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    allsig = np.array([s for i, sg in enumerate(sig) for s, m in zip(sg, masks[i]) if not m and np.isfinite(s)])
    if allsig.size:
        print(f"[interface] analytic σ_z (unmasked crossings): median {np.median(allsig):.2f} m, "
              f"P90 {np.percentile(allsig, 90):.2f} m")
    return cols, masks, sig


# ─────────────────────────────────────────────────────────────────────────────
def plot_psf(res, M, out_dir):
    """PSF = column of R_m; averaging kernel = row of R_m divided by cell area."""
    L, A, JtWJ, lam = wc.cholesky_A(res)
    cx, cz, area = res["cell_x"], res["cell_z"], res["cell_area"]
    out = {}
    fig, axes = plt.subplots(len(PSF_SITES), 2, figsize=(15, 3.3 * len(PSF_SITES)))
    lines = []
    for i, (name, (tx, tz)) in enumerate(PSF_SITES.items()):
        j = int(np.argmin((cx - tx) ** 2 + (cz - tz) ** 2))
        e = np.zeros(len(cx)); e[j] = 1.0
        psf = sla.cho_solve((L, True), JtWJ @ e)           # column j of R_m
        row = JtWJ @ sla.cho_solve((L, True), e)            # row j of R_m (A symmetric)
        ak = row / area
        # normalise by the value AT THE TARGET (the per-area averaging kernel has
        # sign-oscillating side lobes in the tiny near-surface cells that exceed it)
        psf_n = psf / psf[j]
        ak_n = ak / ak[j]
        fp = wc.lobe_fwhm(M["mesh"], psf_n, j, cx, cz)
        fa = wc.lobe_fwhm(M["mesh"], ak_n, j, cx, cz)
        side_psf = float(np.max(np.abs(psf_n)))
        side_ak = float(np.max(np.abs(ak_n)))
        # share of the averaging kernel's total weight (Rm row, NOT divided by area)
        r_j = res["res_radius"][j] if "res_radius" in res else np.nan
        ex_, ez_ = wc.land_surface()
        dep = np.interp(cx, ex_, ez_) - cz
        near = np.hypot(cx - cx[j], cz - cz[j]) <= r_j
        top2 = (dep <= 2.0) & ~near
        rest = ~near & ~top2
        shares = {k: (row[s].sum() / row.sum(), np.abs(row[s]).sum() / np.abs(row).sum())
                  for k, s in (("within r", near), ("top 2 m", top2), ("elsewhere", rest))}
        out[name] = dict(idx=j, x=cx[j], z=cz[j], psf=psf, avg_kernel=ak, Rm_jj=psf[j],
                         r=r_j, fwhm_psf=fp, fwhm_ak=fa, max_abs_psf_rel=side_psf, max_abs_ak_rel=side_ak,
                         row_sum=float(row.sum()), shares=shares)
        lines.append(f"      averaging-kernel weight (Rm row, Σ={row.sum():.3f}): " + "; ".join(
            f"{k}: {100*v[0]:5.1f}% of Σrow, {100*v[1]:5.1f}% of Σ|row|" for k, v in shares.items()))
        lines.append(f"  {name:<15} cell {j:5d} ({cx[j]:6.1f},{cz[j]:6.1f})  Rm_jj={psf[j]:.3f}  "
                     f"r={out[name]['r']:.1f} m | PSF main-lobe FWHM Δx={fp[0]:.1f} Δz={fp[1]:.1f} m "
                     f"({fp[2]} cells), max|PSF|/PSF_target={side_psf:.2f} | avg-kernel main-lobe FWHM "
                     f"Δx={fa[0]:.1f} Δz={fa[1]:.1f} m ({fa[2]} cells), max|AK|/AK_target={side_ak:.1f}")
        for k, (vals, lab, fw) in enumerate(((psf_n, "PSF (Rm column) / value at target", fp),
                                             (ak_n, "averaging kernel (Rm row / cell area) / value at target",
                                              fa))):
            ax = axes[i, k]
            pc = wc.draw_cells(ax, M["polys"], vals, "RdBu_r", mcolors.Normalize(-1, 1))
            wc.hatch_cells(ax, M["polys"], M["poor"])
            wc.setup_section(ax, f"{name} ({tx}, {tz}) — {lab}", ve=1.5,
                             xlim=(max(-5, tx - 90), min(510, tx + 90)))
            ax.title.set_fontsize(9)
            ax.plot(cx[j], cz[j], "k*", ms=11, zorder=8)
            wc.label(ax, 0.5, 0.08, f"main-lobe FWHM Δx≈{fw[0]:.0f} m, Δz≈{fw[1]:.0f} m; "
                     f"colours clipped at ±1 (max |value| {np.max(np.abs(vals)):.1f})", fontsize=7.5,
                     transform=ax.transAxes)
            fig.colorbar(pc, ax=ax, pad=0.01, fraction=0.03)
    fig.suptitle(f"Point-spread functions and averaging kernels (λ={lam:g}); hatched: {mask_desc()}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "10_psf_averaging_kernels.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[PSF]\n" + "\n".join(lines))
    return out, lines


# ─────────────────────────────────────────────────────────────────────────────
def plot_lambda_comparison(res, M, out_dir, tags):
    """Regularisation spread (std of ln ρ across λ runs on the same mesh) vs linearised σ_ln."""
    models = {}
    for t in tags:
        r = wc.load_run(t)
        if r["cell_x"].shape != res["cell_x"].shape or not np.allclose(r["cell_x"], res["cell_x"]):
            raise SystemExit(f"Ensemble run '{t}' is on a different mesh.")
        models[float(r["lam"])] = np.log(r["model"])
    lams = sorted(models)
    spread = np.vstack([models[l] for l in lams]).std(axis=0)
    sig = res["sigma_log_m"]
    ratio = np.log2(spread / sig)
    fig, axes = plt.subplots(3, 1, figsize=(15, 14))
    vmax = np.percentile(np.r_[spread[~M["poor"]], sig[~M["poor"]]], 98)
    pc0 = section(fig, axes[0], M, spread, "hot_r", mcolors.Normalize(0, vmax),
                  f"Regularisation spread: std of ln ρ across λ = {', '.join(f'{l:g}' for l in lams)}",
                  mask=M["poor"])
    fig.colorbar(pc0, ax=axes[0], pad=0.01, fraction=0.025, label="regularisation spread (ln units)")
    pc1 = section(fig, axes[1], M, sig, "hot_r", mcolors.Normalize(0, vmax),
                  f"Linearised posterior σ_ln(ρ) of this run (λ = {float(res['lam']):g})" + uq_label(res),
                  mask=M["poor"])
    fig.colorbar(pc1, ax=axes[1], pad=0.01, fraction=0.025, label="σ_ln (ln units)")
    pc2 = section(fig, axes[2], M, ratio, "RdBu_r", mcolors.Normalize(-2, 2),
                  "log2(regularisation spread / linearised σ):  0 = equal, +1 = spread 2× larger",
                  mask=M["poor"])
    cb = fig.colorbar(pc2, ax=axes[2], pad=0.01, fraction=0.025, label="spread / σ  (log2 scale)")
    cb.set_ticks([-2, -1, 0, 1, 2]); cb.set_ticklabels(["¼", "½", "1", "2", "4"])
    legend_below(axes[2], [hatch_handle()], ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "11_lambda_comparison.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    ok = ~M["poor"]
    print(f"[λ] regularisation spread / σ (unmasked cells): median {np.median(2**ratio[ok]):.2f}, "
          f"P10 {np.percentile(2**ratio[ok], 10):.2f}, P90 {np.percentile(2**ratio[ok], 90):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
def summary_table(res, psf):
    print("=" * 96)
    print(f"{'WESTHOEK ERT — RESOLUTION & UNCERTAINTY SUMMARY':^96}")
    print("=" * 96)
    print(f"  n_cells        : {len(res['model'])}")
    print(f"  χ² (data fit)  : {res['chi2']:.3f}   λ = {float(res['lam']):g}")
    print(f"  σ scaling      : ×{res['sigma_scale']:.3f}")
    print(f"  mask           : {mask_desc()} → {100*np.mean(poor_mask_cells(res)):.1f}% of cells")
    if "res_radius" in res:
        print(f"  resolution radius: median {np.median(res['res_radius']):.1f} m")
    print("-" * 96)
    print(f"{'Site':<16} {'x':>6} {'z':>6} {'ρ (Ω·m)':>8} {'Rm':>6} {'r (m)':>6} {'σ_log10':>8} {'±factor':>8} "
          f"{'FWHM PSF Δx/Δz':>16} {'FWHM AK Δx/Δz':>15}")
    for name, p in psf.items():
        j = p["idx"]
        print(f"{name:<16} {p['x']:6.1f} {p['z']:6.1f} {res['model'][j]:8.2f} {res['Rm_diag'][j]:6.3f} "
              f"{p['r']:6.1f} {res['sigma_log10'][j]:8.4f} {res['rho_unc_factor'][j]:8.3f} "
              f"{p['fwhm_psf'][0]:7.1f}/{p['fwhm_psf'][1]:<7.1f} {p['fwhm_ak'][0]:7.1f}/{p['fwhm_ak'][1]:<7.1f}")
    if psf:
        print("-" * 96)
        print("Averaging-kernel weight shares (Rm row, not divided by area): signed share of Σrow / share of Σ|row|")
        for name, p in psf.items():
            print(f"{name:<16} Σrow={p['row_sum']:.3f}  " + "  ".join(
                f"{k}: {100*v[0]:5.1f}% / {100*v[1]:5.1f}%" for k, v in p["shares"].items()))
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser(description="Westhoek ERT — resolution & uncertainty analysis")
    ap.add_argument("--tag", default=None, help="run tag (data/processed/runs/<tag>/)")
    ap.add_argument("--input", default=None, help="explicit results npz")
    ap.add_argument("--suffix", default=None, help="write to figures/runs/<tag>_<suffix>/ (new folder)")
    ap.add_argument("--no-psf", action="store_true")
    ap.add_argument("--ensemble-tags", nargs="*", default=None,
                    help="existing runs on the same mesh with different λ for figure 11")
    ap.add_argument("--scale-by-chi2", action="store_true")
    ap.add_argument("--mask-mode", choices=("rm", "radius"), default="rm")
    ap.add_argument("--r-max", type=float, default=None)
    args = ap.parse_args()
    if args.mask_mode == "radius" and args.r_max is None:
        ap.error("--mask-mode radius requires --r-max")
    MASK["mode"], MASK["r_max"] = args.mask_mode, args.r_max

    name = (args.tag or "reference") + (f"_{args.suffix}" if args.suffix else "")
    fig_dir = os.path.join(wc.FIG_DIR, "runs", name) if (args.tag or args.suffix) else wc.FIG_DIR
    res_dir = os.path.join(wc.RESULTS_DIR, "runs", name) if (args.tag or args.suffix) else wc.RESULTS_DIR
    if args.suffix and (os.path.exists(fig_dir) and os.listdir(fig_dir)):
        raise SystemExit(f"{fig_dir} exists and is not empty — choose a new --suffix (no overwriting).")
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)

    res = load_results(args.tag, args.input, args.scale_by_chi2)
    print("[mesh] rebuilding paraDomain …")
    mesh, err = wc.rebuild_paradomain(res)
    print(f"[mesh] {mesh.cellCount()} cells, max centre offset vs stored {err:.1e} m")
    ex, _ = wc.land_surface()
    M = dict(mesh=mesh, polys=wc.cell_polygons(mesh), poor=poor_mask_cells(res),
             cs=wc.ColumnSampler(mesh, np.arange(np.ceil(ex.min()), np.floor(ex.max()) + 1, 1.0)))

    plot_resistivity_model(res, M, fig_dir)
    plot_resolution_diagonal(res, M, fig_dir)
    plot_cumulative_sensitivity(res, M, fig_dir)
    plot_uncertainty_map(res, M, fig_dir)
    if "res_radius" in res:
        plot_resolution_radius(res, M, fig_dir)
    plot_interface_uncertainty(res, M, fig_dir)
    psf = {}
    if not args.no_psf:
        psf, _ = plot_psf(res, M, fig_dir)
        np.savez(os.path.join(res_dir, "westhoek_psf.npz"),
                 **{f"{k.lower().replace(' ', '_')}_{q}": np.asarray(v[q])
                    for k, v in psf.items() for q in ("psf", "avg_kernel", "idx", "x", "z", "fwhm_psf", "fwhm_ak")},
                 cell_x=res["cell_x"], cell_z=res["cell_z"])
    if args.ensemble_tags:
        plot_lambda_comparison(res, M, fig_dir, args.ensemble_tags)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary_table(res, psf)
    print(buf.getvalue())
    with open(os.path.join(res_dir, "westhoek_uq_summary.txt"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"✓ figures in {fig_dir}\n✓ results in {res_dir}")


if __name__ == "__main__":
    main()
