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
import matplotlib.ticker  # noqa: E402
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
        for x, z0, z1, ls in ((200.0, v["window_top"], v["window_bottom"], "-"), (120.0, v["image_base"] - 4.0, v["image_base"] - 10.0, ":")):
            ax.plot([x, x], [z0, z1], color="k", lw=0.7, ls=ls, zorder=6, solid_capstyle="butt")
            for z in (z0, z1):
                ax.plot([x / 1.12, x], [z, z], color="k", lw=0.7, zorder=6, solid_capstyle="butt")
    rebuild_legend(fig, [Line2D([], [], color="k", lw=0.7, ls="-", label="Sounding window for ρ$_\\mathbf{below,DP}$ (A5)"),
                         Line2D([], [], color="k", lw=0.7, ls=":", label="Image window, 4–10 m below its base (E20)")])


def add_fig6(fig):
    R = json.load(open(os.path.join(ROOT, "data", "processed", "v1_4", "dp_results.json"), encoding="utf-8"))["dps"]
    ax = fig.axes[0]
    for ln in list(ax.lines):        # the three dotted v1.5 reference values are replaced; the grey 8 Ω·m line (v1.5 rule threshold) is removed (v2.1)
        x = ln.get_xdata()
        if len(x) == 2 and x[0] == x[1] and round(float(x[0]), 2) in (10.89, 13.57, 14.80, 8.0):
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
        # v2.1: one uniform collection for the filled and one for the open markers (per-point colours made the SVG write
        # every marker separately with its own clip path, i.e. one PowerClip per marker in CorelDRAW); same points, order kept
        fc = c.get_facecolors()[0]; off = np.asarray(c.get_offsets()); mk_ = c.get_paths()[0]; sz = c.get_sizes()[0]; z = c.get_zorder()
        m = np.array([near(p) for p in fl["pts"]]); lab = c.get_label(); c.remove()
        axb.scatter(off[~m, 0], off[~m, 1], c=[fc], s=sz, marker=mk_, edgecolors=eps.EDGE_DARK, linewidths=0.4, zorder=z, label=lab)
        if m.any():
            axb.scatter(off[m, 0], off[m, 1], facecolors="none", edgecolors=[fc], s=sz, marker=mk_, linewidths=0.9, zorder=z, label=f"_{lab} open")
    lg = fig.legends[-1]; H, L = list(lg.legend_handles), [t.get_text() for t in lg.get_texts()]; nc = lg._ncols; lg.remove()
    i = L.index("")                                              # first blank slot (end of the definitions column)
    H[i] = Line2D([], [], marker="o", ls="", mfc="none", mec="0.3", mew=0.9); L[i] = "open: |q − 0.5| ≤ 0.1"
    pf.legend_below(fig, H, L, ncol=nc)
    return n_open


V20 = os.path.join(ROOT, "data", "processed", "v2_0")
COREL = os.path.join(ROOT, "Coral Draw", "v2.1")      # v2.1: new folder only; nothing else in "Coral Draw" is written
LEGEND_NCOL = {"Figure5": 3, "Figure6": 3, "FigureS7": 3, "FigureS12": 3}   # v2.1: full-width rows; more columns would overlap at ≥ 7 pt


def legend_v21(fig, label):
    """v2.1 legend style confirmed in the author's CorelDRAW edits: the legend below the panels spans the figure width in
    compact rows, in a thin black dotted frame with square corners."""
    lg = fig.legends[-1]
    H, L = list(lg.legend_handles), [t.get_text() for t in lg.get_texts()]
    hl, htp = lg.handlelength, lg.handletextpad
    lg.remove()
    leg = pf.legend_below(fig, H, L, ncol=LEGEND_NCOL[label], mode="expand", handlelength=hl, handletextpad=htp)
    fr = leg.get_frame(); fr.set_linestyle((0, (1, 1.5))); fr.set_linewidth(0.5); fr.set_edgecolor("k"); fr.set_boxstyle("square", pad=0)
    return leg


def boot_ci(v, idx):
    v = np.asarray(v, float); b = np.nanmean(v[idx], 1)
    return dict(mean=float(np.nanmean(v)), lo=float(np.percentile(b, 5)), hi=float(np.percentile(b, 95)))


def fig_s12():
    """Figure S12 (scores at DP2–DP7), v2.1: CRPS of the linearised bands from the samples (A3, energy form; no ‡) and a new
    row 2 for BEL with prior v3 on the 2025 line (A1, post hoc, §). Otherwise as paper_figures.fig10."""
    O = json.load(open(os.path.join(ROOT, "data", "processed", "v1_6", "scores.json"), encoding="utf-8"))
    A3 = json.load(open(os.path.join(V20, "a3.json"), encoding="utf-8")); A1 = json.load(open(os.path.join(V20, "a1.json"), encoding="utf-8"))
    DPS = ["DP2", "DP3", "DP4", "DP5", "DP6", "DP7"]
    idx = np.random.default_rng([160100, 0]).integers(0, len(DPS), (10_000, len(DPS)))            # as 31_scores.cmd_report
    FS = {m: dict(O["field"][m]) for m in pf.FIELD_METHODS}
    for m in pf.FIELD_METHODS:
        if m.startswith("LIN"):
            FS[m]["CRPS"] = boot_ci([A3["per_sounding"][m][n]["crps"] for n in DPS], idx); FS[m]["approx"] = 0
    r1 = {r["dp"]: r for r in A1["rows"]}
    FS["BEL3-24"] = dict(IS=boot_ci([r1[n]["bel"]["is_"] for n in DPS], idx), CRPS=boot_ci([r1[n]["bel"]["crps"] for n in DPS], idx),
                         width=A1["summary"]["median_width"], cov=A1["summary"]["k"], n=6, approx=0)
    methods = ["BEL3-18", "BEL3-24"] + pf.FIELD_METHODS[1:]
    lab = dict(pf.MLAB, **{"BEL3-24": "BEL (prior v3), 2025 ERT line"})
    mk = lambda m: "h" if m == "BEL3-24" else pf.mmark(m)
    suffix = lambda m: " *" if m == "BEL3-18" else (" §" if m == "BEL3-24" else (" †" if m.startswith("BEL4") else ""))
    fig = pf.new_fig(98)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
    axes = [fig.add_subplot(gs[0, 0])]; axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0])); axes.append(fig.add_subplot(gs[0, 2]))
    ys = np.arange(len(methods))[::-1]
    for i_, (ax, k) in enumerate(((axes[0], "IS"), (axes[1], "CRPS"))):
        for yv, m in zip(ys, methods):
            o = FS[m][k]
            ax.errorbar(o["mean"], yv, xerr=[[o["mean"] - o["lo"]], [o["hi"] - o["mean"]]], fmt=mk(m), color=pf.mcol(m),
                        ms=4.2 if m == "BEL3-24" else 3.5, lw=1, capsize=2, mec="k", mew=0.3)
        ax.set_xlabel("Mean 90 % interval score (m)" if k == "IS" else "Mean CRPS (m)")
        pf.finish(ax); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        eps.panel_label(ax, "ab"[i_])
    axes[0].set_yticks(ys); axes[0].set_yticklabels([f"{i + 1}  {lab[m]}{suffix(m)}" for i, m in enumerate(methods)])
    plt.setp(axes[1].get_yticklabels(), visible=False)
    ax = axes[2]
    for m in methods:
        o = FS[m]; ax.plot(o["width"], o["cov"] / o["n"], mk(m), color=pf.mcol(m), ms=5.8 if m == "BEL3-24" else 5, mec="k", mew=0.3, zorder=4)
    pts = sorted([(FS[m]["cov"] / FS[m]["n"], np.log10(FS[m]["width"]), i + 1) for i, m in enumerate(methods)])
    groups = []
    for c_, lw_, k_ in pts:
        if groups and groups[-1][0] == c_ and lw_ - groups[-1][1][-1] < 0.04:
            groups[-1][1].append(lw_); groups[-1][2].append(k_)
        else:
            groups.append((c_, [lw_], [k_]))
    for c_, lws, ks in groups:
        ks = sorted(ks); lb = (f"{ks[0]}–{ks[-1]}" if ks == list(range(ks[0], ks[-1] + 1)) and len(ks) > 2 else ", ".join(map(str, ks)))
        ax.annotate(lb, (10 ** np.mean(lws), c_), xytext=(0, -6 if c_ > 0.97 else 5), textcoords="offset points", ha="center",
                    va="top" if c_ > 0.97 else "bottom", fontsize=eps.size("tick"), zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
    for g_ in O["synthetic"]:
        ax.plot(g_["width"], g_["cov"], pf.mmark(g_["method"]), mfc="none", color=pf.mcol(g_["method"]), ms=4, mew=0.9)
        ax.plot(g_["prior_width"], g_["prior_cov"], pf.mmark("PRIOR"), mfc="none", color=pf.C["prior"], ms=3, mew=0.9)
    ax.axhline(0.9, color="0.5", lw=0.6, ls=":"); ax.set_xscale("log")
    ax.set_xlabel("Sharpness: median 90 % width (m)"); ax.set_ylabel("Share inside the 90 % interval")
    pf.finish(ax); eps.panel_label(ax, "c")
    h = [Line2D([], [], marker="o", ls="", color="0.4", mec="k", label=f"Field (1–{len(methods)})"),
         Line2D([], [], marker="o", ls="", mfc="none", color="0.4", label="Synthetic"),
         Line2D([], [], marker=pf.mmark("PRIOR"), ls="", mfc="none", color=pf.C["prior"], label="Their priors")]
    eps.styled_legend(ax, handles=h, loc="lower right")
    H = [Line2D([], [], ls="", marker="", label="* pre-specified comparison (BEL, prior v3, 2018 data)"),
         Line2D([], [], marker="h", ls="", color=pf.mcol("BEL"), mec="k", mew=0.3, label="§ post hoc, no information from the soundings"),
         Line2D([], [], ls="", marker="", label="† exploratory (BEL, prior v4)")]
    pf.legend_below(fig, H, [x.get_label() for x in H], ncol=3)
    return fig, dict(methods=methods)


def unclip_markers(fig, tol_px=0.5):
    """v2.1, CorelDRAW export only: scatter collections whose every marker (centre ± radius + edge) lies inside the axes lose
    their clip path. make_paper_figures.unclip keeps them when the collection has no face colour (open markers), because
    matplotlib then reports no usable window extent; each kept clip path becomes a PowerClip in CorelDRAW."""
    from matplotlib.collections import PathCollection
    fig.canvas.draw(); n = 0
    for ax in fig.axes:
        b = ax.bbox
        for c in ax.collections:
            if not isinstance(c, PathCollection) or not c.get_clip_on() or not len(c.get_offsets()):
                continue
            xy = c.get_offset_transform().transform(np.asarray(c.get_offsets(), float))
            rad = np.sqrt(np.max(c.get_sizes())) / 2 * fig.dpi / 72 + np.max(np.atleast_1d(c.get_linewidths())) * fig.dpi / 72
            if np.all(np.isfinite(xy)) and (xy[:, 0] - rad >= b.x0 - tol_px).all() and (xy[:, 0] + rad <= b.x1 + tol_px).all() and                     (xy[:, 1] - rad >= b.y0 - tol_px).all() and (xy[:, 1] + rad <= b.y1 + tol_px).all():
                c.set_clip_on(False); n += 1
    return n


def build(label, short, byname, s03, corel=False):
    eps.corel_mode(corel)
    name, fn = byname[short]; note = ""
    if short == "FigS03":
        fig, _ = pf.figS03(*s03[:3])
    elif short == "Fig09":
        fig, _ = pf.fig09(img_col="#4D4D4D")            # v2.1: dark grey dashed as in v1.9.1 (v2.0 used the red default by mistake)
    elif short == "Fig10":
        fig, _ = fig_s12(); note = "v2.1: CRPS of the linearised bands from samples (A3), no ‡; row 2 = BEL prior v3 on the 2025 line (A1, §)"
    else:
        fig, _ = fn()
    if label == "Figure5":
        add_fig5(fig); note = "A5 depth windows (sounding solid, image dotted); 2025 image dark grey dashed"
    elif label == "Figure6":
        add_fig6(fig); note = "DP3–DP7 ρ_below lines; grey 8 Ω·m line removed"
    elif label == "FigureS7":
        n = add_figS7(fig, s03[0], s03[1]); note = f"{n} open markers in (a) for |q − 0.5| ≤ 0.1"
    if label in LEGEND_NCOL:
        legend_v21(fig, label); note += "; legend: thin dotted square frame, full width, 3 columns"
    return fig, name, note


def main():
    import argparse
    import make_paper_figures as mpf
    ap = argparse.ArgumentParser(); ap.add_argument("only", nargs="?", default=None); ap.add_argument("--corel", action="store_true")
    ap.add_argument("--csv", action="store_true", help="also write the plotted values to results/figs_wrr_v5/data/<label>.csv")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    pf.DATA_DIR = os.path.join(STYLE, "data")
    byname = {n.split("_")[0]: (n, fn) for n, fn, _ in pf.FIGURES}
    s03 = pf.figS03_data(STYLE) if (only is None or "FigureS7" in only) else None
    out = COREL if a.corel else OUT
    for d in ([os.path.join(out, x) for x in ("SVG", "PDF", "preview")] if a.corel else [out]):
        os.makedirs(d, exist_ok=True)
    log = {}
    for label, short in V5:
        if only and label not in only:
            continue
        fig, name, note = build(label, short, byname, s03, corel=a.corel)
        ov = eps.overlap_checks(fig)
        if a.corel:
            r = dict(source=short, function=name, note=note, text_overlaps=mpf.text_overlaps(fig))
            r["smallest_pt"], r["clipped"] = eps.layout_checks(fig)
            r["rasterized_switched_off"] = mpf.vectorize(fig); r["translucent_hatches_split"] = mpf.split_translucent_hatches(fig)
            r["clip_kept"], r["clip_removed"] = mpf.unclip(fig)
            r["marker_clips_removed"] = unclip_markers(fig); r["clip_kept"] -= r["marker_clips_removed"]; mpf.assign_gids(fig)
            r["rows"], _ = mpf.figure_rows(fig)
            r["size_mm"] = [round(v * 25.4, 2) for v in fig.get_size_inches()]
            for ext, kw in (("svg", dict(format="svg", dpi=mpf.IMG_DPI, metadata={"Date": None})),
                            ("pdf", dict(format="pdf", dpi=mpf.IMG_DPI, metadata={"CreationDate": None, "ModDate": None})),
                            ("png", dict(format="png", dpi=300))):
                pth = os.path.join(out, "preview" if ext == "png" else ext.upper(), f"{label}.{ext}")
                fig.savefig(pth, bbox_inches=None, **kw)
                if ext == "svg":
                    r["svg_font_rewrites"] = mpf.svg_explicit_fonts(pth); r["svg_mathtext_runs"] = mpf.svg_mathtext_runs(pth)
            print(label, short, note, "smallest", r["smallest_pt"], "clipped", r["clipped"], "clips kept", r["clip_kept"], flush=True)
        else:
            if a.csv:
                os.makedirs(os.path.join(OUT, "data"), exist_ok=True); mpf.write_figure_csv(fig, label, short, os.path.join(OUT, "data"))
            files, smallest, clipped = eps.savefig(fig, label, OUT, 190, formats=("tif", "pdf"))
            r = dict(source=short, function=name, note=note, smallest_font_pt=smallest, clipped=clipped,
                     overlaps=(ov if isinstance(ov, (list, dict)) else str(ov)), files=[os.path.relpath(f, ROOT) for f in files])
            print(label, short, note, "smallest", smallest, "clipped", clipped, flush=True)
        log[label] = r
        plt.close(fig)
    eps.corel_mode(False)
    p = os.path.join(out, "figs_v21_corel_log.json" if a.corel else "figs_v5_log.json")
    old = json.load(open(p, encoding="utf-8")) if os.path.exists(p) and only else {}
    old.update(log)
    json.dump(old, open(p, "w", encoding="utf-8"), indent=1, default=str, ensure_ascii=False)


if __name__ == "__main__":
    main()
