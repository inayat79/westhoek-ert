"""
06_compare_runs.py
==================
Westhoek ERT — compare inverted models of runs that share the same mesh.

  - Δlog10(ρ) of each run relative to a reference run, and pairwise between runs
  - ρ = 3 Ω·m contours of all runs overlaid on one section
  - max / 95th-percentile |Δlog10ρ| in the saltwater lens (x 90–370 m) and the
    discharge zone (x 380–470 m), top 30 m below surface (optionally only cells
    with resolution radius ≤ --r-max)
  - depth of the upper and lower ρ = 3 Ω·m interface along the profile and its
    change relative to the reference run

OUTPUTS
  figures/runs/compare_<name>/18_delta_log10rho.png
  figures/runs/compare_<name>/19_rho3_contours.png
  figures/runs/compare_<name>/20_interface_depth.png
  results/compare_<name>_model_comparison.txt

RUN
  python 06_compare_runs.py --name outlier_treatments --ref deep_all \
      --tags deep_drop_data deep_drop_elec deep_downweight [--r-max 10]
"""

import argparse
import itertools
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "processed"))
FIG_DIR       = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "figures"))
RESULTS_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))
ELEC_FILE     = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "raw", "elec.dat"))

ZONES = {"lens (x 90–370 m)": (90, 370), "discharge (x 380–470 m)": (380, 470)}
TOP_DEPTH = 30.0            # m below surface considered for zone statistics
LOG3 = np.log10(3.0)
COLORS = ["k", "#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E"]


def land_surface():
    ex, ez = [], []
    for line in open(ELEC_FILE):
        p = line.strip().split("\t")
        if p[0].strip().isdigit() and (len(p) < 4 or p[3].strip() in ("\\", "")):
            ex.append(float(p[1])); ez.append(float(p[2]))
    o = np.argsort(ex)
    return np.array(ex)[o], np.array(ez)[o]


def interfaces(xg, zg, G, surf_z):
    """Upper (first drop below 3 Ω·m going down) and lower (next rise above) interface elevation."""
    top = np.full(len(xg), np.nan); bot = np.full(len(xg), np.nan)
    for i in range(len(xg)):
        col = G[:, i]                     # zg ascending → iterate from the surface down
        state_below = None
        for k in range(len(zg) - 1, 0, -1):
            v1, v0 = col[k], col[k - 1]
            if np.isnan(v1) or np.isnan(v0) or zg[k] > surf_z[i]:
                continue
            if np.isnan(top[i]) and v1 >= LOG3 > v0:
                top[i] = zg[k] + (LOG3 - v1) / (v0 - v1) * (zg[k - 1] - zg[k])
            elif not np.isnan(top[i]) and v1 < LOG3 <= v0:
                bot[i] = zg[k] + (LOG3 - v1) / (v0 - v1) * (zg[k - 1] - zg[k])
                break
        # lens reaching the surface: upper interface = surface
        if np.isnan(top[i]):
            first = next((col[k] for k in range(len(zg) - 1, -1, -1)
                          if not np.isnan(col[k]) and zg[k] <= surf_z[i]), np.nan)
            if first < LOG3:
                top[i] = surf_z[i]
                for k in range(len(zg) - 1, 0, -1):
                    v1, v0 = col[k], col[k - 1]
                    if np.isnan(v1) or np.isnan(v0) or zg[k] > surf_z[i]:
                        continue
                    if v1 < LOG3 <= v0:
                        bot[i] = zg[k] + (LOG3 - v1) / (v0 - v1) * (zg[k - 1] - zg[k])
                        break
    return top, bot


def main():
    ap = argparse.ArgumentParser(description="Compare runs on the same mesh")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--r-max", type=float, default=None,
                    help="only use cells with resolution radius ≤ r_max (reference run) in zone stats")
    args = ap.parse_args()

    tags = [args.ref] + args.tags
    runs = {t: np.load(os.path.join(PROCESSED_DIR, "runs", t, "westhoek_inversion_results.npz"),
                       allow_pickle=True) for t in tags}
    ref = runs[args.ref]
    cx, cz = ref["cell_x"], ref["cell_z"]
    for t in args.tags:
        if runs[t]["cell_x"].shape != cx.shape or not (np.allclose(runs[t]["cell_x"], cx)
                                                       and np.allclose(runs[t]["cell_z"], cz)):
            raise SystemExit(f"Run '{t}' is on a different mesh than '{args.ref}' — cannot compare cell-wise.")
    L10 = {t: np.log10(runs[t]["model"]) for t in tags}

    ex, ez = land_surface()
    depth = np.interp(cx, ex, ez) - cz
    resolved = np.ones_like(cx, bool)
    if args.r_max is not None and "res_radius" in ref.files:
        resolved = ref["res_radius"] <= args.r_max

    fdir = os.path.join(FIG_DIR, "runs", f"compare_{args.name}")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # regular grid for plotting / interfaces
    xg = np.arange(np.ceil(cx.min()), np.floor(cx.max()) + 1, 2.0)
    zg = np.arange(np.ceil(cz.min() * 4) / 4, cz.max(), 0.25)
    XG, ZG = np.meshgrid(xg, zg)
    surf = np.interp(xg, ex, ez)
    pts = np.column_stack([cx, cz])
    G = {t: griddata(pts, L10[t], (XG, ZG), method="linear") for t in tags}
    for t in tags:
        G[t][ZG > np.interp(XG, ex, ez)] = np.nan

    # ── 18: Δlog10ρ maps ─────────────────────────────────────────────────────
    pairs = [(t, args.ref) for t in args.tags] + list(itertools.combinations(args.tags, 2))
    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 2.9 * len(pairs)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (a, b) in zip(axes, pairs):
        D = G[a] - G[b]
        im = ax.pcolormesh(xg, zg, D, cmap="RdBu_r", vmin=-0.5, vmax=0.5, shading="auto", rasterized=True)
        ax.contour(XG, ZG, G[b], levels=[LOG3], colors="k", linewidths=0.8, linestyles="--")
        ax.contour(XG, ZG, G[a], levels=[LOG3], colors="m", linewidths=0.8)
        for (x0, x1) in ZONES.values():
            ax.axvline(x0, color="gray", lw=0.6, ls=":"); ax.axvline(x1, color="gray", lw=0.6, ls=":")
        ax.set_title(f"Δlog10ρ = {a} − {b}   (magenta: ρ=3 of {a}, black dashed: ρ=3 of {b})", fontsize=9)
        ax.set_ylabel("Elev. (m)")
        fig.colorbar(im, ax=ax, pad=0.01, label="Δlog10ρ")
    axes[-1].set_xlabel("Distance along profile (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "18_delta_log10rho.png"), dpi=150)
    plt.close(fig)

    # ── 19: ρ = 3 Ω·m contours of all runs on one section ────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.pcolormesh(xg, zg, G[args.ref], cmap="RdBu_r", vmin=np.log10(0.5), vmax=np.log10(500),
                       shading="auto", rasterized=True, alpha=0.6)
    fig.colorbar(im, ax=ax, pad=0.01, label=f"log10ρ of {args.ref}")
    from matplotlib.lines import Line2D
    handles = []
    for c, t in zip(COLORS, tags):
        ax.contour(XG, ZG, G[t], levels=[LOG3], colors=c, linewidths=1.4 if t == args.ref else 1.1)
        handles.append(Line2D([0], [0], color=c, lw=1.3, label=t))
    ax.plot(ex, ez, "k-", lw=0.8)
    ax.legend(handles=handles, fontsize=8, loc="lower left", title="ρ = 3 Ω·m", title_fontsize=8)
    ax.set_xlabel("Distance along profile (m)"); ax.set_ylabel("Elevation (m)")
    ax.set_title(f"ρ = 3 Ω·m contours — {', '.join(tags)}", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "19_rho3_contours.png"), dpi=150)
    plt.close(fig)

    # ── interfaces ───────────────────────────────────────────────────────────
    IF = {t: interfaces(xg, zg, G[t], surf) for t in tags}
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(ex, ez, color="0.4", lw=1, label="surface")
    for c, t in zip(COLORS, tags):
        axes[0].plot(xg, IF[t][0], color=c, lw=1.2, label=f"{t} upper")
        axes[0].plot(xg, IF[t][1], color=c, lw=1.2, ls="--", label=f"{t} lower")
        if t != args.ref:
            axes[1].plot(xg, IF[t][0] - IF[args.ref][0], color=c, lw=1.0, label=f"{t} upper")
            axes[1].plot(xg, IF[t][1] - IF[args.ref][1], color=c, lw=1.0, ls="--", label=f"{t} lower")
    for ax in axes:
        for (x0, x1) in ZONES.values():
            ax.axvspan(x0, x1, color="orange", alpha=0.06)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Interface elevation (m)")
    axes[0].legend(fontsize=7, ncol=4)
    axes[0].set_title("ρ = 3 Ω·m interfaces along the profile (upper: top of low-ρ zone, lower: its base)",
                      fontsize=10, fontweight="bold")
    axes[1].axhline(0, color="k", lw=0.7)
    axes[1].set_ylabel(f"Δ vs {args.ref} (m)")
    axes[1].set_xlabel("Distance along profile (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "20_interface_depth.png"), dpi=150)
    plt.close(fig)

    # ── report ───────────────────────────────────────────────────────────────
    Lr = [f"MODEL COMPARISON '{args.name}' — reference run '{args.ref}', runs {args.tags}",
          f"Zone statistics over cells with depth ≤ {TOP_DEPTH:g} m below surface"
          + (f" and resolution radius ≤ {args.r_max:g} m (in '{args.ref}')" if args.r_max else ""),
          ""]
    Lr.append(f"{'pair':<40} " + "  ".join(f"{z:>32}" for z in ZONES))
    Lr.append(f"{'':<40} " + "  ".join(f"{'max|Δlog10ρ|  P95   (n cells)':>32}" for _ in ZONES))
    for a, b in pairs:
        cells = []
        for (x0, x1) in ZONES.values():
            sel = (cx >= x0) & (cx <= x1) & (depth <= TOP_DEPTH) & resolved
            dd = np.abs(L10[a][sel] - L10[b][sel])
            cells.append(f"{dd.max():12.3f} {np.percentile(dd, 95):6.3f} ({sel.sum():5d})" if sel.any() else "n/a")
        Lr.append(f"{a + ' − ' + b:<40} " + "  ".join(f"{c:>32}" for c in cells))
    Lr += ["", "INTERFACE ELEVATION CHANGE vs reference (m): median |Δ| / max |Δ| / columns with interface in both"]
    for t in args.tags:
        for which, k in (("upper", 0), ("lower", 1)):
            parts = []
            for zn, (x0, x1) in ZONES.items():
                sel = (xg >= x0) & (xg <= x1)
                dz = IF[t][k][sel] - IF[args.ref][k][sel]
                ok = ~np.isnan(dz)
                parts.append(f"{zn}: {np.median(np.abs(dz[ok])):.2f} / {np.max(np.abs(dz[ok])):.2f} / {ok.sum()}"
                             if ok.any() else f"{zn}: n/a")
            Lr.append(f"  {t:<18} {which:<5}  " + "   ".join(parts))
    rep = os.path.join(RESULTS_DIR, f"compare_{args.name}_model_comparison.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(Lr) + "\n")
    print("\n".join(Lr))
    print(f"[saved] {rep}\n[saved] figures 18–20 in {fdir}")


if __name__ == "__main__":
    main()
