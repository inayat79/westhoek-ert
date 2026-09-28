"""
08_doi_index.py
===============
Westhoek ERT — depth-of-investigation (DOI) index (Oldenburg & Li 1999).

Two inversions of the same data on the same mesh, with smallness+smoothness
regularisation (02 --ctype 10) and homogeneous reference/starting models
m_r1 = f1·median(ρa) and m_r2 = f2·median(ρa) (02 --ref-factor):

    DOI_j = (ln m1_j − ln m2_j) / (ln m_r1 − ln m_r2)

DOI ≈ 0: cell determined by the data; DOI → 1: cell follows the reference model.
Common cut-offs 0.1–0.2.

OUTPUTS
  figures/runs/doi_<name>/22_doi_index.png
  results/doi_<name>.txt

RUN
  python 08_doi_index.py --name G1 --low doi_low_tag --high doi_high_tag [--model tag]
"""

import argparse
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


def load(tag):
    return np.load(os.path.join(PROCESSED_DIR, "runs", tag, "westhoek_inversion_results.npz"), allow_pickle=True)


def main():
    ap = argparse.ArgumentParser(description="DOI index (Oldenburg & Li 1999)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--low", required=True, help="run with the low reference model")
    ap.add_argument("--high", required=True, help="run with the high reference model")
    ap.add_argument("--model", default=None, help="run whose ρ = 3 Ω·m contour is overlaid")
    args = ap.parse_args()

    a, b = load(args.low), load(args.high)
    if a["cell_x"].shape != b["cell_x"].shape or not np.allclose(a["cell_x"], b["cell_x"]):
        raise SystemExit("DOI runs are on different meshes.")
    for d, t in ((a, args.low), (b, args.high)):
        if np.isnan(float(d["ref_value"])):
            raise SystemExit(f"Run '{t}' has no reference model (use 02 --ref-factor).")
    mr1, mr2 = float(a["ref_value"]), float(b["ref_value"])
    bad_fit = [t for d, t in ((a, args.low), (b, args.high)) if float(d["chi2"]) > 2]
    if bad_fit:
        print("!" * 78)
        print(f"  WARNING: {bad_fit} do not fit the data (χ² > 2). The DOI index assumes both")
        print("  reference inversions explain the data equally well; with a poor fit the")
        print("  index reflects the regularisation, not the data. Treat it as INVALID.")
        print("!" * 78)
    doi = (np.log(a["model"]) - np.log(b["model"])) / (np.log(mr1) - np.log(mr2))
    cx, cz = a["cell_x"], a["cell_z"]

    ex, ez = [], []
    for line in open(ELEC_FILE):
        p = line.strip().split("\t")
        if p[0].strip().isdigit() and (len(p) < 4 or p[3].strip() in ("\\", "")):
            ex.append(float(p[1])); ez.append(float(p[2]))
    o = np.argsort(ex); ex, ez = np.array(ex)[o], np.array(ez)[o]
    depth = np.interp(cx, ex, ez) - cz

    xg = np.arange(np.ceil(cx.min()), np.floor(cx.max()) + 1, 2.0)
    zg = np.arange(np.ceil(cz.min()), cz.max(), 0.5)
    XG, ZG = np.meshgrid(xg, zg)
    G = griddata(np.column_stack([cx, cz]), doi, (XG, ZG), method="linear")
    G[ZG > np.interp(XG, ex, ez)] = np.nan

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.pcolormesh(xg, zg, np.abs(G), cmap="viridis_r", vmin=0, vmax=1, shading="auto", rasterized=True)
    fig.colorbar(im, ax=ax, pad=0.01, label="|DOI index|")
    cs = ax.contour(XG, ZG, np.abs(G), levels=[0.1, 0.2], colors=["white", "red"], linewidths=[1.2, 1.2])
    ax.clabel(cs, fmt="%.1f", fontsize=8)
    if args.model:
        m = load(args.model)
        Gm = griddata(np.column_stack([m["cell_x"], m["cell_z"]]), np.log10(m["model"]), (XG, ZG), method="linear")
        Gm[ZG > np.interp(XG, ex, ez)] = np.nan
        ax.contour(XG, ZG, Gm, levels=[np.log10(3)], colors="cyan", linewidths=1.3, linestyles="--")
    ax.plot(ex, ez, "k-", lw=0.8)
    ax.set_xlabel("Distance along profile (m)"); ax.set_ylabel("Elevation (m)")
    ax.set_title(f"DOI index (Oldenburg & Li 1999) — ref {mr1:.2f} vs {mr2:.2f} Ω·m; "
                 f"white 0.1, red 0.2" + (f"; cyan dashed ρ=3 Ω·m of {args.model}" if args.model else ""),
                 fontsize=10)
    fig.tight_layout()
    fdir = os.path.join(FIG_DIR, "runs", f"doi_{args.name}")
    os.makedirs(fdir, exist_ok=True)
    fig.savefig(os.path.join(fdir, "22_doi_index.png"), dpi=150)

    # depth (below surface) at which |DOI| first exceeds the cut-off, per 10 m column
    L = [f"DOI INDEX '{args.name}': low ref '{args.low}' ({mr1:.3f} Ω·m, χ²={float(a['chi2']):.2f}), "
         f"high ref '{args.high}' ({mr2:.3f} Ω·m, χ²={float(b['chi2']):.2f})", ""]
    for cut in (0.1, 0.2):
        dd = []
        for x0 in np.arange(ex.min(), ex.max(), 10.0):
            sel = (cx >= x0) & (cx < x0 + 10)
            if not sel.any():
                continue
            o = np.argsort(depth[sel]); dep, dv = depth[sel][o], np.abs(doi[sel][o])
            # shallowest depth below which the running median of |DOI| stays above the cut-off
            above = np.where(dv > cut)[0]
            dd.append(dep[above[0]] if above.size else dep[-1])
        dd = np.array(dd)
        L.append(f"|DOI| > {cut}: depth below surface (first cell, 10 m columns): "
                 f"median {np.median(dd):.1f} m, P10 {np.percentile(dd, 10):.1f} m, P90 {np.percentile(dd, 90):.1f} m")
    for zmax in (10, 20, 30, 40):
        sel = depth <= zmax
        L.append(f"cells ≤ {zmax:2d} m depth: fraction with |DOI| ≤ 0.1: {np.mean(np.abs(doi[sel]) <= 0.1):.2f}, "
                 f"≤ 0.2: {np.mean(np.abs(doi[sel]) <= 0.2):.2f}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rep = os.path.join(RESULTS_DIR, f"doi_{args.name}.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[saved] {rep}\n[saved] {os.path.join(fdir, '22_doi_index.png')}")


if __name__ == "__main__":
    main()
