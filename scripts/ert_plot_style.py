"""
ert_plot_style.py
=================
Westhoek ERT v1.8 — paper figure style, after `Claude outputs/ERT_Figure_Style_Spec.md`, with the paper adaptations
(a)–(k) of the v1.8 prompt. Plotting only: no data are read or computed here.

The spec gives sizes for figures about 20 in (508 mm) wide. Paper figures are drawn at their print size (190 mm full
width, 90 mm single column), so every spec size in points (fonts, line widths, tick lengths) is multiplied by
SCALE = 190 mm / 508 mm (the full-width ratio) — the same point sizes at print for 190 mm and 90 mm figures. Dense-grid mode subtracts 4 spec points before scaling. A floor of MIN_FONT_PT applies to
all text (journal minimum); the inventory reports where the floor is active.

Use:
    import ert_plot_style as eps
    eps.apply(print_width_mm=190, dense=False)
    fig = plt.figure(figsize=eps.figsize(190, 120))
    ...
    eps.savefig(fig, "Fig02_ReferenceModel", outdir, print_width_mm=190)
"""

import os
import tempfile

import matplotlib
from matplotlib import font_manager
import cmcrameri.cm  # noqa: F401  registers the Crameri maps as "cmc.<name>" (cmcrameri 1.10)

SHOW_TITLES = False                      # journal version: panel labels only, no titles / suptitles (rule g)
# Resistivity colour map: Crameri "batlow" (perceptually uniform, colour-vision-deficiency friendly; v1.9 for AGU / WRR,
# replacing the spec's "rainbow"). Source: Crameri, F. (2018), Scientific colour maps, Zenodo, doi:10.5281/zenodo.1243862;
# Crameri, Shephard & Heron (2020), Nat. Commun. 11, 5444. Low values (conductive) are dark.
RES_CMAP = "cmc.batlow"
BANNED_CMAPS = ("jet", "rainbow", "turbo", "hsv", "nipy_spectral", "gist_")    # rainbow-like (AGU graphics guidance)
MISFIT_CMAP = "RdBu_r"                   # misfit / differences, symmetric about zero (spec §2)
SPEC_WIDTH_MM = 508.0                    # the spec's reference figure width (20 in)
MIN_FONT_PT = 7.0                        # floor for any text at print size
DPI = 1000                               # rule k (updated): PNG and RGB TIFF at 1000 dpi + PDF; 300 dpi draft PNG
MM = 1 / 25.4

# one resistivity scale for every section in the paper (rule b): the v1.7.1 figures use LogNorm(0.5, 500) Ω·m
RES_LIM = (0.5, 500.0)
RES_TICKS = (1, 3, 10, 30, 100)

# method / series colours (rule f)
# v1.9.1: prior grey #767676 (4.5:1 against white; was 0.55 drawn with alpha, 2.2–2.9:1) and predicted / model-column red
# #DC3D38 (4.4:1; tomato was 2.95:1), so that every series reaches the WCAG 3:1 contrast for graphical objects.
COLORS = dict(bel="#6A3D9A", logmid="#1A9E77", thr3="#D9601A", prior="#767676", maxgrad="0.35",
              observed="steelblue", predicted="#DC3D38")
EDGE_DARK = "#333333"          # thin dark edge on light markers (v1.9.1)
HALO = 2.5                     # white halo (pt) of lines and labels on dark cells (v1.9.1)

# spec sizes in spec points (§1, §4, §5, §6)
_SPEC = dict(title=26, panel=22, axis=24, tick=22, cbar_label=24, cbar_tick=22, legend=22, stats=18,
             spine=3.0, tick_len=9.0, tick_w=2.4, minor_len=5.0, minor_w=1.6, grid_lw=0.9, line=2.8, marker=8.0,
             marker_edge=0.8, cbar_outline=2.4, legend_frame=2.4, stats_frame=2.2, section_outline=2.5,
             elec_marker=60.0, elec_edge=0.9)
FULL_WIDTH_MM = 190.0
_STATE = dict(scale=FULL_WIDTH_MM / SPEC_WIDTH_MM, dense=False, width=FULL_WIDTH_MM)


def font_file(weight="bold"):
    """Path of the font file matplotlib uses for Times New Roman (raises if it would fall back)."""
    return font_manager.findfont(font_manager.FontProperties(family="Times New Roman", weight=weight), fallback_to_default=False)


def size(key):
    """A spec size converted to print points (dense-grid mode: −4 spec points for text); text is floored at MIN_FONT_PT."""
    v = _SPEC[key]
    text = key in ("title", "panel", "axis", "tick", "cbar_label", "cbar_tick", "legend", "stats")
    if text and _STATE["dense"]:
        v -= 4
    v *= _STATE["scale"]
    return max(v, MIN_FONT_PT) if text else v


def raw_text_size(key, dense=None, print_width_mm=None):
    """Text size before the floor (for the inventory)."""
    d = _STATE["dense"] if dense is None else dense
    return (_SPEC[key] - (4 if d else 0)) * FULL_WIDTH_MM / SPEC_WIDTH_MM


def apply(print_width_mm=190.0, dense=False):
    """Set rcParams for a figure printed at print_width_mm (190 or 90 mm); point sizes do not depend on the width."""
    _STATE.update(scale=FULL_WIDTH_MM / SPEC_WIDTH_MM, dense=dense, width=print_width_mm)
    s = size
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.weight": "bold",
        "mathtext.fontset": "stix", "font.size": s("tick"),
        "axes.titlesize": s("panel"), "axes.titleweight": "bold", "axes.titlepad": 14 * _STATE["scale"],
        "axes.labelsize": s("axis"), "axes.labelweight": "bold", "axes.linewidth": s("spine"),
        "axes.grid": False, "grid.linestyle": "--", "grid.alpha": 0.18, "grid.linewidth": s("grid_lw"),
        "xtick.labelsize": s("tick"), "ytick.labelsize": s("tick"), "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.major.size": s("tick_len"), "ytick.major.size": s("tick_len"),
        "xtick.major.width": s("tick_w"), "ytick.major.width": s("tick_w"), "xtick.minor.size": s("minor_len"),
        "ytick.minor.size": s("minor_len"), "xtick.minor.width": s("minor_w"), "ytick.minor.width": s("minor_w"),
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "lines.linewidth": s("line"), "lines.markersize": s("marker"), "lines.markeredgewidth": s("marker_edge"),
        "legend.fontsize": s("legend"), "legend.frameon": True, "legend.fancybox": False, "legend.framealpha": 0.95,
        "legend.edgecolor": "black",
        "figure.titlesize": s("title"), "figure.titleweight": "bold", "figure.dpi": 150,
        "savefig.dpi": DPI, "pdf.fonttype": 42, "ps.fonttype": 42, "image.cmap": RES_CMAP,
        "axes.unicode_minus": True, "hatch.linewidth": 0.5,
    })


def figsize(width_mm, height_mm):
    """Figure size in inches from print size in mm (rule: printed height ≤ 225 mm)."""
    if height_mm > 225:
        raise ValueError(f"printed height {height_mm:.0f} mm exceeds 225 mm")
    return (width_mm * MM, height_mm * MM)


def title(ax, text):
    """In-figure title, shown only when SHOW_TITLES is True (rule g)."""
    if SHOW_TITLES:
        ax.set_title(text, fontsize=size("panel"), fontweight="bold")


# Panel letters (v1.9, WRR): one size in every figure (the spec panel size at 190 mm, not reduced in dense mode), at the
# upper left of the panel just outside the frame: left edge on the left spine, baseline PANEL_GAP_PT above the top spine.
PANEL_PT = round(_SPEC["panel"] * FULL_WIDTH_MM / SPEC_WIDTH_MM, 2)     # 8.23 pt
PANEL_GAP_PT = 3.0


def panel_label(ax, letter, text=None, wrap=False, **_old):
    """Bold panel letter '(a)' above the upper-left corner of the axes, outside the frame. A short identifier follows on
    the same line, or on a second line (wrap=True) where the panel is too narrow; use the same choice for all panels of a
    figure. (The v1.8 arguments loc/inside are accepted and ignored: every letter is now placed the same way.)"""
    from matplotlib.transforms import offset_copy
    s = f"({letter})" if letter else ""
    if text:
        s = (s + ("\n" if wrap else " ") + text) if s else text
    tr = offset_copy(ax.transAxes, fig=ax.figure, x=0.0, y=PANEL_GAP_PT, units="points")
    t = ax.text(0.0, 1.0, s, transform=tr, ha="left", va="baseline", fontsize=PANEL_PT, fontweight="bold", zorder=50,
                clip_on=False, linespacing=1.15)
    t.set_gid("panel_label")
    return t


def panel_ident(ax, text):
    """Identifier of a sub-panel without its own letter (e.g. 'Sample 2'), at the panel-letter position."""
    t = panel_label(ax, None, text=text)
    t.set_gid("panel_ident")
    return t


def styled_legend(ax_or_fig, *args, **kw):
    """Legend with square corners, black frame (spec §6)."""
    kw.setdefault("fontsize", size("legend")); kw.setdefault("frameon", True); kw.setdefault("fancybox", False)
    kw.setdefault("framealpha", 0.95); kw.setdefault("edgecolor", "black")
    leg = ax_or_fig.legend(*args, **kw)
    leg.get_frame().set_linewidth(size("legend_frame"))
    for t in leg.get_texts():
        t.set_fontweight("bold")
    leg.set_zorder(40)
    return leg


def add_colorbar(fig, ax, mappable, label, ticks=None, log_ticks_fmt=True, cax=None, orientation="vertical", **kw):
    """Colourbar to the right (spec §5: pad 0.02, fraction 0.030), bold label and ticks, outline at spec width."""
    if cax is None:
        kw.setdefault("pad", 0.02); kw.setdefault("fraction", 0.030)
    cb = fig.colorbar(mappable, ax=None if cax is not None else ax, cax=cax, orientation=orientation, **kw)
    cb.set_label(label, fontsize=size("cbar_label"), fontweight="bold")
    if ticks is not None:
        cb.set_ticks(list(ticks))
        if log_ticks_fmt:
            cb.set_ticklabels([f"{t:g}" for t in ticks])
        cb.minorticks_off()
    cb.ax.tick_params(labelsize=size("cbar_tick"), width=size("tick_w"), length=size("tick_len") * 0.6)
    for t in cb.ax.get_yticklabels() + cb.ax.get_xticklabels():
        t.set_fontweight("bold")
    cb.outline.set_linewidth(size("cbar_outline"))
    return cb


def res_norm():
    """The single resistivity scale of the paper (rule b)."""
    from matplotlib.colors import LogNorm
    return LogNorm(*RES_LIM)


def res_ticks():
    return [t for t in RES_TICKS if RES_LIM[0] <= t <= RES_LIM[1]]


def stats_box(ax, text, loc="upper right"):
    """Rounded white statistics box (spec §6)."""
    xy, ha, va = {"upper right": ((0.98, 0.98), "right", "top"), "upper left": ((0.02, 0.98), "left", "top"),
                  "lower right": ((0.98, 0.02), "right", "bottom"), "lower left": ((0.02, 0.02), "left", "bottom")}[loc]
    return ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size("stats"), fontweight="bold", zorder=45,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=size("stats_frame"), alpha=0.92))


def electrode_markers(ax, x, z_surface, offset=0.25, label="Electrodes", **kw):
    """Black ▼ markers 0.25 m above the topographic surface at each electrode (rule e)."""
    import numpy as np
    kw.setdefault("s", _SPEC["elec_marker"] * _STATE["scale"] ** 2)          # scatter size is an area (points²)
    return ax.scatter(np.asarray(x), np.asarray(z_surface) + offset, marker="v", c="black", edgecolors="white",
                      linewidths=size("elec_edge"), zorder=30, label=label, clip_on=False, **kw)


def finish_axes(ax, grid=True):
    """Spines on all sides, inward ticks on all sides, minor ticks, dashed light grid (spec §4)."""
    from matplotlib.ticker import AutoMinorLocator
    for s_ in ax.spines.values():
        s_.set_linewidth(size("spine")); s_.set_visible(True)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if ax.get_xscale() == "linear":
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    if grid:
        ax.grid(True, which="major", ls="--", alpha=0.18, lw=size("grid_lw"))


def _all_axes(f):
    out = list(f.axes)
    for sf in getattr(f, "subfigs", []):
        out += _all_axes(sf)
    return out


def _legends(f):
    out = [ax.get_legend() for ax in _all_axes(f) if ax.get_legend() is not None] + list(f.legends)
    for sf in getattr(f, "subfigs", []):
        out += list(sf.legends)
    return out


def _is_cbar(ax):
    return getattr(ax, "_colorbar", None) is not None or ax.get_label() == "<colorbar>"


def overlap_checks(fig, shrink_px=0.5):
    """After drawing. (1) Legends over data: a legend box overlapping a plotted line (vertices or segments), a marker, a
    data patch (bars, filled areas, cells) or a panel letter, inside the visible part of the axes. (2) Panel letters
    overlapping a legend, a colour bar, another axes, a tick label or any other text."""
    from matplotlib.collections import PathCollection, Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch as _P
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    axes = _all_axes(fig); legs = _legends(fig)
    letters = [t for t in fig.findobj(Text) if t.get_gid() in ("panel_label", "panel_ident") and t.get_visible()]
    lbb = {id(t): t.get_window_extent(r) for t in letters}
    leg_hits, let_hits = [], []

    def shrink(b, d=shrink_px):
        return Bbox.from_extents(b.x0 + d, b.y0 + d, b.x1 - d, b.y1 - d)
    for lg in legs:
        bb = shrink(lg.get_window_extent(r))
        name = "/".join(t.get_text()[:25] for t in lg.get_texts()[:2])
        for ax in axes:
            if _is_cbar(ax) or not ax.get_visible():
                continue
            vis = Bbox.intersection(bb, ax.bbox)
            if vis is None or vis.width <= 0 or vis.height <= 0:
                continue
            for a in list(ax.lines) + list(ax.collections) + list(ax.patches):
                if not a.get_visible() or a in (ax.patch,):
                    continue
                box = vis if a.get_clip_on() else bb
                try:
                    if isinstance(a, Line2D):
                        p = a.get_transform().transform_path(a.get_path())
                        hit = (len(p.vertices) > 1 and p.intersects_bbox(box, filled=False)) or \
                              (a.get_marker() not in (None, "None", "", " ") and box.count_contains(p.vertices) > 0)
                    elif isinstance(a, PathCollection):
                        off = a.get_offset_transform().transform(a.get_offsets())
                        hit = len(off) > 0 and box.count_contains(off) > 0
                    elif isinstance(a, Collection):
                        tr = a.get_transform()
                        hit = any(tr.transform_path(p).intersects_bbox(box, filled=True) for p in a.get_paths())
                    elif isinstance(a, _P):
                        hit = a.get_transform().transform_path(a.get_path()).intersects_bbox(box, filled=True)
                    else:
                        hit = False
                except Exception:
                    hit = False
                if hit:
                    leg_hits.append(f"legend '{name}' over {type(a).__name__} '{(a.get_label() or '')[:30]}'")
        for t in letters:
            if bb.overlaps(lbb[id(t)]):
                leg_hits.append(f"legend '{name}' over panel letter '{t.get_text()[:12]}'")
    # panel letters
    others = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t.get_gid() not in ("panel_label", "panel_ident")
              and not (t.axes is not None and not t.axes.get_visible())]
    tick_all, tick_ok = set(), set()
    for ax in axes:                            # only tick labels that are drawn (inside the axis view interval)
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_all.update((id(tk.label1), id(tk.label2)))
            for tk in axis._update_ticks():
                for lab in (tk.label1, tk.label2):
                    if lab.get_visible() and lab.get_text():
                        tick_ok.add(id(lab))
    for t in letters:
        b = shrink(lbb[id(t)]); nm = t.get_text().split("\n")[0][:20]
        for lg in legs:
            if b.overlaps(lg.get_window_extent(r)):
                let_hits.append(f"'{nm}' over a legend")
        for ax in axes:
            if ax is t.axes:
                continue
            tb = ax.get_window_extent(r) if not _is_cbar(ax) else ax.get_tightbbox(r)
            if tb is not None and b.overlaps(tb):
                let_hits.append(f"'{nm}' over {'colour bar' if _is_cbar(ax) else 'axes'} {ax.get_label() or ''}")
        for o in others:
            if id(o) in tick_all and id(o) not in tick_ok:
                continue                       # an undrawn tick label
            try:
                ob = o.get_window_extent(r)
            except Exception:
                continue
            if ob.width > 0 and b.overlaps(ob) and o is not t:
                if lg_child(o, legs):
                    continue                   # legend texts are covered by the legend test above
                let_hits.append(f"'{nm}' over text '{o.get_text()[:25]}'")
    return dict(legend_overlaps=leg_hits, letter_overlaps=let_hits, n_legends=len(legs), n_letters=len(letters))


def lg_child(o, legs):
    return any(o in lg.get_texts() or o is lg.get_title() for lg in legs)


def layout_checks(fig, tol_px=1.0):
    """After drawing: (smallest visible text size in pt, list of clipped artists). An artist is clipped when its window
    extent leaves the figure by more than tol_px (constrained layout is required to keep everything inside)."""
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    sizes, clipped = [], []
    ticklabels = set()                       # tick labels are checked through the axes' tight boxes (only drawn ticks count)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                ticklabels.update((id(tk.label1), id(tk.label2)))
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.axes is not None and t.get_clip_on() and not t.axes.get_visible():
            continue
        sizes.append(t.get_fontsize())
        if id(t) in ticklabels:
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width > 0 and (bb.x0 < -tol_px or bb.y0 < -tol_px or bb.x1 > W + tol_px or bb.y1 > H + tol_px):
            clipped.append(f"text {t.get_text()[:40]!r}")
    for ax in fig.axes:
        bb = ax.get_tightbbox(r)
        if bb is not None and (bb.x0 < -tol_px or bb.y0 < -tol_px or bb.x1 > W + tol_px or bb.y1 > H + tol_px):
            clipped.append(f"axes {ax.get_label() or ax}: {bb.bounds}")
    return (min(sizes) if sizes else None), clipped


def colormaps_in(fig):
    """Names of all colour maps used by the figure's mappables (collections, images, quad meshes)."""
    names = set()
    for obj in fig.findobj(lambda a: hasattr(a, "get_cmap") and hasattr(a, "get_array") and a.get_array() is not None):
        if getattr(obj.get_array(), "ndim", 0) == 3:      # RGB(A) image: no colour map applied
            continue
        if getattr(obj, "colors", None) is not None and type(obj).__name__.endswith("ContourSet"):
            continue                                        # contour lines in explicit colours (internal ListedColormap)
        try:
            names.add(obj.get_cmap().name)
        except Exception:
            pass
    return sorted(names)


def savefig(fig, name, outdir, print_width_mm, dpi=1000, draft_dpi=300, formats=("png", "tif", "pdf")):
    """Save at the exact print size (no bbox cropping): PNG and RGB TIFF (LZW) at `dpi` with the resolution in the file
    metadata, PDF (vector text and lines, TrueType fonts), and a `draft_dpi` PNG copy in <outdir>/draft/. Every file is
    written to a temporary name in its target folder and then renamed (atomic). Returns (paths, smallest_font_pt, clipped)."""
    from PIL import Image
    os.makedirs(outdir, exist_ok=True); dd = os.path.join(outdir, "draft"); os.makedirs(dd, exist_ok=True)
    w_mm = fig.get_size_inches()[0] * 25.4
    if abs(w_mm - print_width_mm) > 0.5:
        raise ValueError(f"{name}: figure width {w_mm:.1f} mm ≠ print width {print_width_mm} mm")
    smallest, clipped = layout_checks(fig)
    out = []

    def atomic(final, write):
        fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(final)[1], dir=os.path.dirname(final)); os.close(fd)
        try:
            write(tmp); os.replace(tmp, final)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        out.append(final)

    if "png" in formats:
        png = os.path.join(outdir, f"{name}.png")
        atomic(png, lambda t: fig.savefig(t, format="png", dpi=dpi, metadata={"Software": None}))
        im = Image.open(png); im.load()
    else:                                    # TIFF only: render to a temporary PNG that is removed again
        fd, tmp_png = tempfile.mkstemp(suffix=".png", dir=outdir); os.close(fd)
        try:
            fig.savefig(tmp_png, format="png", dpi=dpi, metadata={"Software": None})
            im = Image.open(tmp_png); im.load()
        finally:
            os.remove(tmp_png)
    if "tif" in formats:
        atomic(os.path.join(outdir, f"{name}.tif"),
               lambda t: im.convert("RGB").save(t, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi)))
    atomic(os.path.join(outdir, f"{name}.pdf"),
           lambda t: fig.savefig(t, format="pdf", dpi=dpi, metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}))
    atomic(os.path.join(dd, f"{name}.png"), lambda t: fig.savefig(t, format="png", dpi=draft_dpi, metadata={"Software": None}))
    return out, smallest, clipped
