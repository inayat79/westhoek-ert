#!/usr/bin/env python3
"""v2.0 A11 (decision log v2.0-0): the paper figures under the WRR v5 numbering → results/figs_wrr_v5/ (TIFF + PDF at
190 mm, the v1.9.1 submission style of `make_paper_figures.py --wrr`), built with the unchanged `paper_figures.py`
functions, with four additions drawn on top of the returned figures:

  Figure 5   the two depth windows of A5 at each sounding (sounding window solid, image window dashed), as thin brackets
  Figure 6   panel (a): the five sounding values ρ_below,DP at DP3–DP7 as thin labelled vertical lines
  Figure S12 (scores) the pre-specified row (BEL v3, 2018 data) marked *, the prior-v4 rows marked † (exploratory)
  Figure S7  (bias rule) open markers for the cases with |q − 0.5| ≤ 0.1

Nothing in results/figs_wrr/ is overwritten. PLOTTING ONLY.
"""
import json
import os
import sys

os.environ.pop("WESTHOEK_READ_TRACE", None)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import paper_figures as pf  # noqa: E402
import ert_plot_style as eps  # noqa: E402

ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
OUT = os.path.join(ROOT, "results", "figs_wrr_v5")
STYLE = os.path.join(ROOT, "results", "figs_style")
A5 = os.path.join(ROOT, "data", "processed", "v2_0", "a5.json")
# WRR v5 label → v1.8 short name (brief, A11 table)
V5 = [("Figure1", "Fig01"), ("Figure2", "Fig02"), ("Figure3", "Fig04"), ("Figure4", "Fig08"), ("Figure5", "Fig09"),
      ("Figure6", "Fig11"), ("FigureS1", "FigS04"), ("FigureS2", "FigS02"), ("FigureS3", "Fig06"), ("FigureS4", "Fig03"),
      ("FigureS5", "FigS01"), ("FigureS6", "Fig05"), ("FigureS7", "FigS03"), ("FigureS8", "FigS10"), ("FigureS9", "Fig07"),
      ("FigureS10", "FigS08"), ("FigureS11", "FigS05"), ("FigureS12", "Fig10"), ("FigureS13", "FigS06"), ("FigureS14", "FigS07"),
      ("FigureS15", "FigS09")]


def rebuild_legend(fig, extra_h, ncol=None):
    """Replace the figure-level legend (legend_below) by one with extra entries appended."""
    lg = fig.legends[-1]
    H, L = list(lg.legend_handles), [t.get_text() for t in lg.get_texts()]
    nc = ncol or lg._ncols
    lg.remove()
    H += extra_h; L += [h.get_label() for h in extra_h]
    pf.legend_below(fig, H, L, ncol=nc)


def add_fig5(fig):
    A = json.load(open(A5, encoding="utf-8"))["dps"]
    _, names = pf.dp_records()
    axes = [a for a in fig.axes if a.get_xscale() == "log"]
    for ax, nm in zip(axes, names):
        v = A[nm]
        if "window_top" not in v:
            continue
        for x, z0, z1, ls in ((200.0, v["window_top"], v["window_bottom"], "-"), (120.0, v["image_base"] - 4.0, v["image_base"] - 10.0, "--")):
            ax.plot([x, x], [z0, z1], color="k", lw=0.7, ls=ls, zorder=6, solid_capstyle="butt")
            for z in (z0, z1):
                ax.plot([x / 1.12, x], [z, z], color="k", lw=0.7, zorder=6, solid_capstyle="butt")
    rebuild_legend(fig, [Line2D([], [], color="k", lw=0.7, ls="-", label="Sounding window for ρ$_\\mathbf{below,DP}$ (A5)"),
                         Line2D([], [], color="k", lw=0.7, ls="--", label="Image window, 4–10 m below its base (E20)")])


def add_fig6(fig):
    R = json.load(open(os.path.join(ROOT, "data", "processed", "v1_4", "dp_results.json"), encoding="utf-8"))["dps"]
    ax = fig.axes[0]
    for ln in list(ax.lines):                                   # the three dotted v1.5 reference values are replaced
        x = ln.get_xdata()
        if len(x) == 2 and x[0] == x[1] and round(float(x[0]), 2) in (10.89, 13.57, 14.80):
            ln.remove()
    ytop = ax.get_ylim()[1]
    for k_, nm in enumerate(("DP3", "DP4", "DP5", "DP6", "DP7")):
        rb = R[nm]["per"]["primary|A"]["rho_below"]
        ax.axvline(rb, color="#1f4e9c", ls=":", lw=0.7, zorder=2)
        ax.text(rb * 1.03, ytop * (0.97 if k_ % 2 == 0 else 0.80), nm, rotation=90, ha="left", va="top", fontsize=eps.size("tick"), color="#1f4e9c",
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.8), zorder=7)
    rebuild_legend(fig, [Line2D([], [], color="#1f4e9c", ls=":", lw=0.7, label="ρ$_\\mathbf{below,DP}$ at DP3–DP7 (2024 soundings)")])


def add_figS12(fig):
    ax = fig.axes[0]
    labs = [t.get_text() for t in ax.get_yticklabels()]
    new = []
    for t in labs:
        if "BEL" in t and "v3" in t and "2018" in t:
            t = t + " *"
        elif "v4" in t and "BEL" in t:
            t = t + " †"
        new.append(t)
    ax.set_yticklabels(new)
    H = [Line2D([], [], ls="", marker="", label="* pre-specified comparison (BEL, prior v3, 2018 data)"),
         Line2D([], [], ls="", marker="", label="† exploratory (BEL, prior v4)"),
         Line2D([], [], ls="", marker="", label="‡ CRPS from three quantiles")]
    pf.legend_below(fig, H, [h.get_label() for h in H], ncol=3, handlelength=0, handletextpad=0)
    return labs, new


def add_figS7(fig, fit, fit_lam):
    near = lambda p: abs(p["q"] - 0.5) <= 0.1
    ax = fig.axes[0]
    pcs = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
    n_open = 0
    for c, p in zip(pcs[:len(fit["pts"])], fit["pts"]):
        if near(p):
            fc = c.get_facecolors()[0]; c.set_facecolors("none"); c.set_edgecolors([fc]); c.set_linewidths(0.9); n_open += 1
    axb = fig.axes[1]
    pcb = [c for c in axb.collections if isinstance(c, matplotlib.collections.PathCollection)]
    for c, (l, fl) in zip(pcb, [(l, fl) for l, fl in fit_lam.items() if fl]):
        fc = c.get_facecolors()[0]
        F = np.array([[0, 0, 0, 0] if near(p) else fc for p in fl["pts"]]); E = np.array([fc if near(p) else matplotlib.colors.to_rgba(eps.EDGE_DARK) for p in fl["pts"]])
        c.set_facecolors(F); c.set_edgecolors(E); c.set_linewidths([0.9 if near(p) else 0.4 for p in fl["pts"]])
    rebuild_legend(fig, [Line2D([], [], marker="o", ls="", mfc="none", mec="0.3", mew=0.9, label="open: |q − 0.5| ≤ 0.1")])
    return n_open


def main():
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    os.makedirs(OUT, exist_ok=True)
    pf.DATA_DIR = os.path.join(STYLE, "data")
    byname = {n.split("_")[0]: (n, fn) for n, fn, _ in pf.FIGURES}
    log = {}
    s03 = None
    for label, short in V5:
        if only and label not in only:
            continue
        name, fn = byname[short]
        if short == "FigS03":
            s03 = s03 or pf.figS03_data(STYLE)
            fig, _ = pf.figS03(*s03[:3])
        else:
            fig, _ = fn()
        note = ""
        if label == "Figure5":
            add_fig5(fig); note = "A5 depth windows added"
        elif label == "Figure6":
            add_fig6(fig); note = "DP3–DP7 ρ_below lines added (replacing the three dotted v1.5 values)"
        elif label == "FigureS12":
            _, new = add_figS12(fig); note = "pre-specified row * and prior-v4 rows † marked: " + "; ".join(t for t in new if t.endswith(("*", "†")))
        elif label == "FigureS7":
            n = add_figS7(fig, s03[0], s03[1]); note = f"{n} open markers in (a) for |q − 0.5| ≤ 0.1"
        ov = eps.overlap_checks(fig)
        files, smallest, clipped = eps.savefig(fig, label, OUT, 190, formats=("tif", "pdf"))
        plt.close(fig)
        log[label] = dict(source=short, function=name, note=note, smallest_font_pt=smallest, clipped=clipped,
                          overlaps=(ov if isinstance(ov, (list, dict)) else str(ov)), files=[os.path.relpath(f, ROOT) for f in files])
        print(label, short, note, "smallest", smallest, "clipped", clipped, flush=True)
    p = os.path.join(OUT, "figs_v5_log.json")
    old = json.load(open(p, encoding="utf-8")) if os.path.exists(p) and only else {}
    old.update(log)
    json.dump(old, open(p, "w", encoding="utf-8"), indent=1, default=str)


if __name__ == "__main__":
    main()
