"""
paper_figures.py
================
Westhoek ERT v1.8 — the 21 paper figures redrawn in the paper style (`ert_plot_style.py`). PLOTTING ONLY: every value
comes from the saved files of tag v1.7.1, read with the existing loaders of the analysis scripts, which are imported
unchanged and without side effects (no number registration, no writes; `WESTHOEK_READ_TRACE` is removed before import).
Figure content (data, panels, panel order, axis ranges) is that of v1.7.1; only the look changes.

Each `figXX()` returns (fig, meta); meta holds the plotted values used by the spot checks of make_paper_figures.py.
"""

import contextlib
import importlib.util
import io
import json
import os
import sys

os.environ.pop("WESTHOEK_READ_TRACE", None)
import numpy as np  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import westhoek_common as wc  # noqa: E402
import ert_plot_style as eps  # noqa: E402

ROOT = wc.ROOT
C = eps.COLORS
DARK = "#4D4D4D"
DP_X = {"DP1": 454.1, "DP2": 404.7, "DP3": 352.8, "DP4": 304.8, "DP5": 253.6, "DP6": 203.5, "DP7": 152.7}   # Table 3 of the draft
DATA_DIR = None            # set by make_paper_figures.py: <out>/data (derived arrays stored for reproducible redraws)
RHO = "ρ"
RHO_BELOW = "ρ$_\\mathbf{below}$"
_M = {}


def mod(key):
    """Existing analysis scripts, loaded once and unchanged (stdout of their import suppressed)."""
    files = dict(s18="18_paper.py", g25="25_generalisation.py", g27="27_dp_groundtruth.py", k1="21_k1_comparison.py",
                 g28="28_dp_synthetic.py", g30="30_bel_v4.py", m32="32_location_map.py")
    if key not in _M:
        spec = importlib.util.spec_from_file_location(f"pf_{key}", os.path.join(SCRIPT_DIR, files[key]))
        m = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(m)
        if key == "s18":                     # set in 18_paper.main() only; point them at the project caches (read-only use)
            m.DPAP = os.path.join(ROOT, "data", "processed", "paper"); m.DS6 = os.path.join(ROOT, "data", "processed", "stage6_v2")
            m.RECOMPUTE = False
        _M[key] = m
    return _M[key]


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


# ─────────────────────────────────────────────────────────────────────────────
# common drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def new_fig(h_mm, dense=False, w_mm=190.0, constrained=True):
    eps.apply(w_mm, dense)
    return plt.figure(figsize=eps.figsize(w_mm, h_mm), layout="constrained" if constrained else None)


def band(ax, x, lo, hi, col, alpha, ls="-", lw=0.6, **kw):
    """v1.9.1: a 90 % band whose edges are drawn as thin lines in the line style of its median, with a light fill, so that
    overlapping bands can be told apart without colour (same polygon as before; only its edge is now stroked)."""
    return ax.fill_between(x, lo, hi, facecolor=matplotlib.colors.to_rgba(col, alpha), edgecolor=col, linewidth=lw, linestyle=ls, **kw)


def halo(lw=None):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw or eps.HALO, foreground="white")]


def trim_height(fig, pad_mm=1.0, tol_mm=0.3):
    """v1.9: remove the empty band that constrained layout leaves above and below an aspect-locked section: shrink the
    figure height to the drawn content (+ pad) and let the layout re-run (the axes width, hence the scale, is unchanged)."""
    for _ in range(4):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        slack = h - bb.height - 2 * pad_mm / 25.4
        if slack < tol_mm / 25.4:
            break
        fig.set_size_inches(w, h - slack)
    return fig.get_size_inches()[1] * 25.4


def cells(ax, polys, values, cmap, norm, mask=None):
    """Exact mesh cells; only this layer is rasterised; edge colour = face colour (no seams in PDF viewers)."""
    from matplotlib.collections import PolyCollection
    v = np.ma.masked_invalid(np.asarray(values, float))
    if mask is not None:
        v = np.ma.masked_where(mask, v)
    cm = (matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap).with_extremes(bad=(1, 1, 1, 0))
    pc = PolyCollection(polys, array=v, cmap=cm, norm=norm, edgecolors="face", linewidths=eps.COREL_CELL_LW if eps.is_corel() else 0.3, antialiased=True, rasterized=eps.raster())
    ax.add_collection(pc)
    return pc


def hatch_cells(ax, polys, mask, label=None):
    from matplotlib.collections import PolyCollection
    sel = [p for p, m in zip(polys, mask) if m]
    if sel:
        ax.add_collection(PolyCollection(sel, facecolors="none", edgecolors="0.35", hatch="////", linewidths=0.0, label=label))


def ve_text(ax, ve=3):
    ax.text(0.995, 0.02, f"V.E. {ve:g}×", transform=ax.transAxes, ha="right", va="bottom", fontsize=eps.size("tick"), zorder=12,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.4", alpha=0.9))


def dune_hatch(ax, label=False, text=True):
    import matplotlib as mpl_
    mpl_.rcParams["hatch.linewidth"] = 0.5
    y0, y1 = ax.get_ylim(); xa = min(ax.get_xlim()[0], 0) - 10
    ax.add_patch(Rectangle((xa, y0), 90.0 - xa, y1 - y0, facecolor=(0.55, 0.55, 0.55, 0.28), edgecolor=(0.3, 0.3, 0.3, 0.6),
                           hatch="x", lw=0, zorder=8, label="Dunes x < 90 m: weakly constrained" if label else None))
    if text:
        ax.text(45.0, y0 + 0.04 * (y1 - y0), "dunes:\nweakly\nconstrained", ha="center", va="bottom", fontsize=eps.size("tick"),
                zorder=13, color="0.1", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.85))


def section(ax, z_floor=-45.0, ve=3, xlim=(-5.0, 510.0), surface=True):
    ex, ez = wc.land_surface()
    if surface:
        ax.plot(ex, ez, "k-", lw=0.8, zorder=6)
    ax.set_xlim(*xlim); ax.set_ylim(z_floor, max(ez) + 2); ax.set_aspect(ve)
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("Elevation (mTAW)")
    ve_text(ax, ve)


def electrodes(ax):
    ex, ez = wc.land_surface()
    return eps.electrode_markers(ax, ex, ez, offset=0.25)


def finish(ax, grid=True, minor=True):
    from matplotlib.ticker import AutoMinorLocator, NullLocator
    for s_ in ax.spines.values():
        s_.set_linewidth(eps.size("spine")); s_.set_visible(True)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for axis, sc in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if sc == "linear":
            axis.set_minor_locator(AutoMinorLocator() if minor else NullLocator())
    if grid:
        ax.grid(True, which="major", ls="--", alpha=0.18, lw=eps.size("grid_lw"))


def legend_below(ax_or_fig, handles=None, labels=None, ncol=2, **kw):
    if hasattr(ax_or_fig, "transAxes"):
        kw.setdefault("loc", "upper center"); kw.setdefault("bbox_to_anchor", (0.5, -0.16))
    else:
        kw.setdefault("loc", "outside lower center")
    args = (handles, labels) if handles is not None else ()
    return eps.styled_legend(ax_or_fig, *args, ncol=ncol, **kw)


def res_colorbar(fig, ax, pc, **kw):
    return eps.add_colorbar(fig, ax, pc, f"{RHO} (Ω·m)", ticks=[1, 3, 10, 30, 100, 300], **kw)


def xwin(x, v, lo=130.0, hi=450.0):
    return np.where((np.asarray(x) >= lo) & (np.asarray(x) <= hi), v, np.nan)


def dp_records():
    wc.require_input("dp_results", "the comparison with the soundings (Tables 1–2, SI Tables S1, S7–S11, Figures 4–6 and S11–S14, expectations E15–E21)", path=os.path.join(ROOT, "data", "processed", "v1_4", "dp_results.json"))
    R = json.load(open(os.path.join(ROOT, "data", "processed", "v1_4", "dp_results.json"), encoding="utf-8"))
    names = sorted(R["dps"], key=lambda n: R["dps"][n]["index"])
    return R, names


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 1 — location map (rule h: fonts, spines, ticks, legend frame, saving only; dense sizes)
# ─────────────────────────────────────────────────────────────────────────────
def fig01():
    m = mod("m32")
    import matplotlib.patheffects as pe
    from matplotlib.colors import TwoSlopeNorm, LightSource
    from matplotlib.patches import ConnectionPatch
    from pyproj import Transformer
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader
    from PIL import Image
    S = json.load(open(os.path.join(m.BASE, "sources.json"), encoding="utf-8"))
    for k, o in S.items():
        if k.startswith(("copdem", "emodnet", "ortho")) and m.sha256(os.path.join(ROOT, o["file"])) != o["sha256"]:
            raise SystemExit(f"cache file changed: {o['file']}")
    eps.apply(190, dense=True)
    fs, fs2 = eps.size("tick"), eps.size("axis")                     # 7.0 / 7.5 pt
    G = quiet(m.geometry); z, sea, _ = quiet(m.relief_grid)
    T72 = (Transformer.from_crs(4326, 31370, always_xy=True), Transformer.from_crs(31370, 4326, always_xy=True))
    T35 = (Transformer.from_crs(4326, 3035, always_xy=True), Transformer.from_crs(3035, 4326, always_xy=True))
    fig = plt.figure(figsize=eps.figsize(190, 160))
    R = lambda x, y, w, h: [x / 190, y / 160, w / 190, h / 160]
    halo = [pe.withStroke(linewidth=1.8, foreground="w")]
    COL = m.COL
    bx0, bx1, by0, by1 = m.EXT_B; wb = 122.0; hb = wb * (by1 - by0) / (bx1 - bx0)
    axb = fig.add_axes(R(63, 93, wb, hb))
    import cmcrameri.cm as cmc
    cm = cmc.oleron; nrm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)          # v1.9: Crameri oleron (sea | land at 0 m)
    ls = LightSource(azdeg=315, altdeg=40)
    zc = np.where(sea, np.clip(z, -40, -0.01), np.clip(z, 0.01, 40))
    rgb = ls.shade(zc, cmap=cm, norm=nrm, blend_mode="soft", vert_exag=8, dx=m.GRID_B, dy=m.GRID_B)
    axb.imshow(rgb, extent=m.EXT_B, origin="upper", interpolation="bilinear", zorder=1)
    rd = shapereader.Reader(os.path.join(m.NE_DIR, "ne_10m_admin_0_boundary_lines_land.shp"))
    for g in rd.geometries():
        for ln in getattr(g, "geoms", [g]):
            lo, la = np.array(ln.coords).T
            if lo.min() < 3.2 and lo.max() > 2.0 and la.min() < 51.4 and la.max() > 50.7:
                xx, yy = T72[0].transform(lo, la); axb.plot(xx, yy, color="k", lw=0.8, ls=(0, (4, 1.5, 1, 1.5)), zorder=4)
    for name, (lo, la) in m.TOWNS.items():
        xx, yy = T72[0].transform(lo, la); axb.plot(xx, yy, "s", ms=2.6, mfc="k", mec="w", mew=0.4, zorder=6)
        dxy = dict(Dunkerque=(4, -7), Nieuwpoort=(-4, -8)).get(name, (3, -7))
        axb.annotate(name, (xx, yy), xytext=dxy, textcoords="offset points", fontsize=fs, ha="left" if dxy[0] > 0 else "right",
                     path_effects=halo, zorder=7)
    for txt, (lo, la), kw in (("North Sea", (2.48, 51.135), dict(style="italic", color="#0b2a55")), ("France", (2.45, 51.046), {}),
                              ("Belgium", (2.66, 51.068), {})):
        xx, yy = T72[0].transform(lo, la)
        axb.text(xx, yy, txt, fontsize=fs2, ha="center", va="center", path_effects=halo, zorder=7, clip_on=True, **kw)
    cx0, cx1, cy0, cy1 = m.EXT_C
    axb.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec=COL["box_c"], lw=0.9, zorder=8))
    axb.set_xlim(bx0, bx1); axb.set_ylim(by0, by1); axb.set_xticks([]); axb.set_yticks([]); axb.set_aspect("equal")
    gb = m.Grat(31370)
    n_before = len(axb.lines)
    gb.draw(axb, m.EXT_B, 5 / 60, 2.5 / 60, lambda v: m.dm(v, "E"), lambda v: m.dm(v, "N", 1), fs=fs)
    for ln in axb.lines[n_before:]:
        ln.set_gid("a11y:map graticule")
    m.north_arrow(axb, (0.045, 0.60), T72, fs=fs); m.scale_bar(axb, 5000, "5 km", xy=(0.80, 0.07), fs=fs)
    eps.panel_label(axb, "b")
    cax = fig.add_axes(R(80, 93 + hb + 3.6, 58, 2.0))                 # v1.9: starts right of panel letter (b)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cm), cax=cax, orientation="horizontal", ticks=[-40, -20, -10, 0, 10, 20, 40])
    cb.ax.tick_params(labelsize=fs, length=2, width=0.5, pad=1, direction="in"); cb.outline.set_linewidth(eps.size("cbar_outline"))
    cb.ax.xaxis.set_ticks_position("top"); cb.ax.xaxis.set_label_position("top")
    cb.set_label("Elevation / depth (m); land clipped at 40 m", fontsize=fs, labelpad=1.5, fontweight="bold")
    pa = ccrs.epsg(3035)
    axa = fig.add_axes(R(9, 93, 44, hb), projection=pa)
    lon_a, lat_a = (-1.5, 8.0), (48.6, 54.2)
    xa, ya = T35[0].transform(np.array([lon_a[0], lon_a[1], lon_a[0], lon_a[1], 3.0, 3.0]), np.array([lat_a[0], lat_a[0], lat_a[1], lat_a[1], 48.6, 54.2]))
    ycen = 0.5 * (min(ya) + max(ya)); yh = max(ya) - min(ya); xw = yh * 44 / hb; xcen = 0.5 * (min(xa) + max(xa))
    EXT_A = (xcen - xw / 2, xcen + xw / 2, ycen - yh / 2, ycen + yh / 2)
    axa.set_extent(EXT_A, crs=pa)
    axa.add_feature(cfeature.ShapelyFeature(shapereader.Reader(os.path.join(m.NE_DIR, "ne_10m_land.shp")).geometries(), ccrs.PlateCarree(), gid="a11y:map base (land)",
                                            facecolor="#e9e4d6", edgecolor="none"), zorder=1)
    axa.set_facecolor("#cfe3f0")
    axa.add_feature(cfeature.ShapelyFeature(shapereader.Reader(os.path.join(m.NE_DIR, "ne_10m_coastline.shp")).geometries(), ccrs.PlateCarree(),
                                            facecolor="none", edgecolor="0.25", lw=0.4), zorder=2)
    axa.add_feature(cfeature.ShapelyFeature(shapereader.Reader(os.path.join(m.NE_DIR, "ne_10m_admin_0_boundary_lines_land.shp")).geometries(),
                                            ccrs.PlateCarree(), facecolor="none", edgecolor="0.3", lw=0.5, linestyle=(0, (3, 1.2))), zorder=2)
    t = np.linspace(0, 1, 50)
    bxs = np.r_[bx0 + t * (bx1 - bx0), np.full(50, bx1), bx1 - t * (bx1 - bx0), np.full(50, bx0)]
    bys = np.r_[np.full(50, by0), by0 + t * (by1 - by0), np.full(50, by1), by1 - t * (by1 - by0)]
    blo, bla = T72[1].transform(bxs, bys); X3, Y3 = T35[0].transform(blo, bla)
    axa.plot(X3, Y3, color=COL["box_b"], lw=1.2, transform=pa, zorder=5)
    for txt, (lo, la), kw in (("North\nSea", (3.4, 53.0), dict(style="italic", color="#0b2a55")), ("UK", (0.2, 52.4), {}), ("France", (3.2, 49.6), {}),
                              ("Belgium", (5.0, 50.45), {}), ("NL", (5.6, 52.3), {})):
        xx, yy = T35[0].transform(lo, la); axa.text(xx, yy, txt, fontsize=fs, ha="center", va="center", transform=pa, zorder=6, **kw)
    ga = m.Grat(3035)
    n_before = len(axa.lines)
    ga.draw(axa, EXT_A, 2.0, 2.0, lambda v: m.dm(v, "E"), lambda v: m.dm(v, "N"), fs=fs)
    for ln in axa.lines[n_before:]:
        ln.set_gid("a11y:map graticule")
    m.north_arrow(axa, (0.88, 0.80), T35, fs=fs); m.scale_bar(axa, 200000, "200 km", xy=(0.06, 0.06), fs=fs)
    eps.panel_label(axa, "a")
    axc = fig.add_axes(R(12, 12, 70.0, 70.0))
    img = np.array(Image.open(os.path.join(m.BASE, f"ortho_{m.ORTHO_LAYER}_{m.ORTHO_RES:g}m.png")))
    axc.imshow(img, extent=m.EXT_C, origin="upper", interpolation="lanczos", zorder=1)
    E, (M1, M2), E24, cp = G["E"], G["M"], G["E24"], G["cpt"]
    stroke = lambda w, c="k": [pe.Stroke(linewidth=w, foreground=c), pe.Normal()]
    axc.plot(*E.T, "-", color=COL["land"], lw=2.0, zorder=4, path_effects=stroke(3.0), label="2018 land ERT (K1)")
    axc.plot([M1[0], M2[0]], [M1[1], M2[1]], color=COL["marine"], lw=1.2, ls=(0, (5, 2.5)), zorder=4, path_effects=stroke(2.2),
             label="2018 marine ERT (high tide)")
    axc.plot(*E24[[0, -1]].T, ls=(0, (1.0, 1.0)), color=COL["e24"], lw=1.3, zorder=5, path_effects=stroke(2.3), label="2025 ERT",
             gid="a11y:outlined (black casing)", dash_capstyle="round")              # v1.9.1: dotted, cased (was solid, colour only)
    pin = m.pin_marker()
    axc.plot(*cp.T, ls="none", marker=pin, ms=9, mfc=COL["cpt"], mec="k", mew=0.6, zorder=7, label="CPT / CPT-EC (2024)†")
    for i, s in enumerate(G["cpt_names"]):
        axc.annotate(f"DP{int(s[1:]) - 1}", cp[i], xytext=(5, 1), textcoords="offset points", fontsize=fs, ha="left", va="bottom",
                     path_effects=halo, zorder=8)
    for lab, p, dxy in (("E1", E[0], (6, -2)), ("E102", E[-1], (-6, 4)), ("E103", M1, (-12, -12)), ("E538", M2, (7, -6))):
        axc.annotate(lab, p, xytext=dxy, textcoords="offset points", fontsize=fs, ha="left" if dxy[0] > 0 else "right", path_effects=halo, zorder=8)
    for txt, xy, kw in (("Westhoek dunes", (23290, 199590), dict(color="w", path_effects=stroke(1.6))),
                        ("Beach", (22960, 199700), dict(rotation=34)), ("North Sea", (22830, 200040), dict(style="italic", color="#0b2a55"))):
        axc.text(*xy, txt, fontsize=fs2, ha="center", va="center", zorder=6, **({"path_effects": halo} | kw))
    axc.set_xlim(cx0, cx1); axc.set_ylim(cy0, cy1); axc.set_xticks([]); axc.set_yticks([]); axc.set_aspect("equal")
    gc = m.Grat(31370)
    gc.draw(axc, m.EXT_C, 0.2 / 60, 0.1 / 60, lambda v: m.dm(v, "E", 1), lambda v: m.dm(v, "N", 1), lines=False, fs=fs)
    m.north_arrow(axc, (0.06, 0.16), T72, fs=fs); m.scale_bar(axc, 200, "200 m", xy=(0.06, 0.05), fs=fs)
    eps.styled_legend(axc, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=fs, handlelength=2.6, borderpad=0.4, labelspacing=0.3)
    eps.panel_label(axc, "c")
    for (xa_, ya_), (xc_, yc_) in (((cx0, cy0), (0.09, 1)), ((cx1, cy0), (1, 1))):     # v1.9: left line ends right of letter (c)
        fig.add_artist(ConnectionPatch(xyA=(xa_, ya_), coordsA=axb.transData, xyB=(xc_, yc_), coordsB=axc.transAxes, color="k", lw=0.6, zorder=0))
    for ax in (axa, axb, axc):
        for s_ in ax.spines.values():
            s_.set_linewidth(eps.size("spine"))
    axd = fig.add_axes(R(106, 22, 80, 50))
    axd.axvspan(0, 90, color="#9bbf62", alpha=0.25, lw=0, zorder=0)
    axd.plot(G["ex"], G["ez"], "-", color=COL["land"], lw=1.2, zorder=3, label="2018 electrodes (K1)")
    axd.plot(G["ex"], G["ez"], "o", ms=1.4, color=COL["land"], zorder=3)
    axd.plot(G["cpt_x"], G["cpt_z"], ls="none", marker=pin, ms=8, mfc=COL["cpt"], mec="k", mew=0.6, zorder=5, label="CPT ground level (2024)†")
    for i, s in enumerate(G["cpt_names"]):
        axd.annotate(f"DP{int(s[1:]) - 1}", (G["cpt_x"][i], G["cpt_z"][i]), xytext=(0, 11), textcoords="offset points", fontsize=fs, ha="center")
    yb = -1.6
    axd.plot([148.75, 503.75], [yb, yb], "-", color="k", lw=2.4, solid_capstyle="butt", zorder=4)
    axd.plot([148.75, 503.75], [yb, yb], ls=(0, (1.0, 1.0)), color=COL["e24"], lw=1.4, dash_capstyle="butt", zorder=5,
             gid="a11y:outlined (black casing)")                                  # v1.9.1: dotted as in (c)
    axd.text(326, yb + 0.45, "2025 ERT (x = 148.75–503.75 m)", fontsize=fs, ha="center", va="bottom")
    for xv, lab in ((90, "dune\nfoot"), (G["x_hw"], f"HW 2018\n(x = {G['x_hw']:.0f} m)"), (G["x_lw"], "LW")):
        axd.axvline(xv, color="0.25", lw=0.6, ls=(0, (3, 2)), zorder=1)
        axd.text(xv + (-4 if xv > 480 else 4), 8.4, lab, fontsize=fs, ha="right" if xv > 480 else "left", va="top")
    axd.text(45, 3.6, "dunes", fontsize=fs, ha="center", style="italic"); axd.text(215, 5.9, "beach", fontsize=fs, ha="center", style="italic")
    axd.set_xlim(0, 505); axd.set_ylim(-2.4, 8.6)
    axd.set_xlabel("x along K1 (m; E1 = 0, dune → sea)", fontsize=fs2); axd.set_ylabel("Elevation (mTAW)", fontsize=fs2)
    finish(axd, grid=False)
    axd.tick_params(labelsize=fs)
    eps.styled_legend(axd, loc="upper right", bbox_to_anchor=(0.99, 0.86), fontsize=fs, handlelength=2.0)
    eps.panel_label(axd, "d")
    return fig, dict(dense=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 2 — reference model
# ─────────────────────────────────────────────────────────────────────────────
def fig02():
    import matplotlib.tri as mtri
    s18 = mod("s18"); ctx = quiet(s18.ref_ctx); ref = ctx["ref"]
    fig = new_fig(72)
    ax = fig.add_subplot()
    pc = cells(ax, ctx["polys"], ref["model"], eps.RES_CMAP, eps.res_norm())
    cx, cz, rr = ref["cell_x"], ref["cell_z"], ref["res_radius"]
    tri = mtri.Triangulation(cx, cz); ex, ez = wc.land_surface()
    xm, zm = cx[tri.triangles].mean(1), cz[tri.triangles].mean(1)
    tri.set_mask((zm > np.interp(xm, ex, ez)) | (np.ptp(cx[tri.triangles], 1) > 25))
    cs = ax.tricontour(tri, rr, levels=[10, 15], colors=["k", "0.25"], linewidths=[1.1, 0.9], linestyles=["-", "--"], zorder=7)
    cs.set_path_effects(halo())                                            # v1.9.1: white halo on the dark (conductive) cells
    for t in ax.clabel(cs, fmt={10: "r = 10 m", 15: "r = 15 m"}, fontsize=eps.size("tick"), inline=True):
        t.set_path_effects(halo())
    section(ax, z_floor=-45)
    dune_hatch(ax); electrodes(ax)
    cax = ax.inset_axes([1.012, 0.0, 0.016, 1.0])                 # v1.9: colour bar exactly as tall as the section
    eps.add_colorbar(fig, None, pc, f"{RHO} (Ω·m)", ticks=[1, 3, 10, 30, 100, 300], cax=cax)
    finish(ax, grid=False)
    h = [Line2D([], [], color="k", lw=1.1), Line2D([], [], color="0.25", lw=0.9, ls="--"),
         Line2D([], [], marker="v", ls="", color="k", mec="w", ms=4)]
    legend_below(ax, h, ["r = 10 m", "r = 15 m", "Electrodes"], ncol=3, bbox_to_anchor=(0.5, -0.22))
    trim_height(fig)                                             # v1.9: no empty band above the section
    return fig, dict(chi2=float(ref["chi2"]), lam=float(ref["lam"]))


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 3 — linearised lens base
# ─────────────────────────────────────────────────────────────────────────────
def fig03():
    import matplotlib.colors as mcolors
    s18 = mod("s18"); ctx = quiet(s18.ref_ctx); ref = ctx["ref"]
    v3 = np.load(s18.LIN_V3); v4 = np.load(s18.LIN_V4)
    fig = new_fig(150)
    axes = fig.subplots(2, 1)
    ax = axes[0]
    pc = cells(ax, ctx["polys"], v3["P_lt3_pooled"], "Blues", mcolors.Normalize(0, 1))
    hatch_cells(ax, ctx["polys"], ref["res_radius"] > 10, label="r > 10 m")
    section(ax, z_floor=-45); dune_hatch(ax); electrodes(ax)
    eps.add_colorbar(fig, ax, pc, "P(ρ < 3 Ω·m)", ticks=[0, 0.25, 0.5, 0.75, 1.0], log_ticks_fmt=False)
    finish(ax, grid=False); eps.panel_label(ax, "a")
    legend_below(ax, ncol=2, bbox_to_anchor=(0.5, -0.28))
    ax = axes[1]
    x = v4["x"]; meta = {}
    for key, col, lab, ls_ in (("pooled_thr3", C["thr3"], "3 Ω·m contour", "-"), ("pooled_logmid", C["logmid"], "Log-midpoint", "--")):
        b = v4[key]; ok = (b[0] >= 0.5) & (x >= 130) & (x <= 450)
        band(ax, x, np.where(ok, b[2], np.nan), np.where(ok, b[3], np.nan), col, 0.22, ls_)
        ax.plot(x, np.where(ok, b[1], np.nan), color=col, lw=1.4, ls=ls_, label=f"{lab}: median, 90 % (λ-pooled)")
        meta[key] = {xx: [float(v) for v in b[:, int(np.argmin(np.abs(x - xx)))]] for xx in (150, 200, 250, 300, 350)}
    ex, ez = wc.land_surface()
    ax.plot(ex, ez, "k-", lw=0.8, label="Land surface")
    ax.set_xlim(-5, 510); ax.set_ylim(-32, 8); ax.set_aspect(3)
    dune_hatch(ax)
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("Elevation (mTAW)"); ve_text(ax, 3)
    cb = fig.colorbar(pc, ax=ax, fraction=0.030, pad=0.02); cb.ax.set_visible(False)          # aligns (b) with (a)
    finish(ax); eps.panel_label(ax, "b")
    legend_below(ax, ncol=3, bbox_to_anchor=(0.5, -0.36))
    return fig, meta


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 4 — synthetic test of the linearised intervals
# ─────────────────────────────────────────────────────────────────────────────
def fig04():
    s18 = mod("s18"); c11 = s18.c11
    rows = quiet(s18.synthetic_rows); summ = s18.a2_summary(rows); relb = quiet(s18.reliability)
    fig = new_fig(215, dense=True)
    axes = fig.subplots(3, 2)
    cols = dict(thr3=C["thr3"], maxgrad=C["maxgrad"], logmid=C["logmid"])
    names = {"thr3": "3 Ω·m", "maxgrad": "maximum gradient", "logmid": "log-midpoint"}
    ax = axes[0, 0]; w = 0.25
    for k, dname in enumerate(c11.DEFS):
        for gi, grp in enumerate(("≤10", "10–15")):
            s_ = summ[(grp, dname)]
            ax.bar(gi + (k - 1) * w, 100 * s_["cov"], w * 0.9, color=cols[dname], edgecolor="k", lw=0.4, hatch=HATCH[dname],
                   label=names[dname] if gi == 0 else None)
            ax.plot(gi + (k - 1) * w, 100 * s_["cov_pooled"], "k_", ms=9, mew=1.6, label="λ-pooled" if (gi == 0 and k == 0) else None)
    ax.axhline(90, color="k", ls="--", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["r ≤ 10 m", "10 < r ≤ 15 m"]); ax.set_ylim(0, 105)
    ax.set_ylabel("Coverage of 90 % bands (%)")
    finish(ax, minor=False); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    eps.panel_label(ax, "a")
    legend_below(ax, ncol=2, bbox_to_anchor=(0.5, -0.14))
    ax = axes[0, 1]
    for k, dname in enumerate(c11.DEFS):
        for gi, grp in enumerate(("≤10", "10–15")):
            bs = [np.nanmedian([x["bias"] for x in rows if x["truth"] == t and x["d"] == dname and x["lam"] == "50" and x["grp"] == grp])
                  for t in c11.A2_TRUTHS]
            ax.bar(gi + (k - 1) * w, np.mean(bs), w * 0.9, color=cols[dname], edgecolor="k", lw=0.4, hatch=HATCH[dname])
            ax.plot(np.full(len(bs), gi + (k - 1) * w), bs, "k.", ms=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["r ≤ 10 m", "10 < r ≤ 15 m"])
    ax.set_ylabel("Median bias m̂ − true (m)\n(− = too deep)")
    finish(ax); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); eps.panel_label(ax, "b")
    lams = [5, 10, 20, 50, 100]
    for col_i, (key, ylab) in enumerate((("cov", "Coverage (%)"), ("bias", "Median bias (m)"))):
        ax = axes[1, col_i]
        for dname in ("thr3", "logmid"):
            for grp, ls in (("≤10", "-"), ("10–15", "--")):
                sel = lambda l: [x for x in rows if x["d"] == dname and x["lam"] == str(l) and x["grp"] == grp and x["r"] < 5]
                v = [100 * np.mean([x["cov"] for x in sel(l)]) if key == "cov" else np.nanmedian([x["bias"] for x in sel(l)]) for l in lams]
                ax.plot(lams, v, ls, marker=MARK[dname], ms=3.5, color=cols[dname], mec="k", mew=0.3,
                        label=f"{names[dname]}, " + {"≤10": "r ≤ 10 m", "10–15": "10 < r ≤ 15 m"}[grp])
        ax.axhline(90 if key == "cov" else 0, color="k", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_xticks(lams); ax.set_xticklabels([str(l) for l in lams]); ax.minorticks_off()
        ax.set_xlabel("λ"); ax.set_ylabel(ylab)
        finish(ax); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if key == "cov":
            ax.set_ylim(0, 105); eps.styled_legend(ax, loc="lower left")
        eps.panel_label(ax, "cd"[col_i])
    for col_i, grp in enumerate(("≤10", "10–15")):
        ax = axes[2, col_i]
        ax.plot([0, 1], [0, 1], "--", color="red", lw=eps.size("line") * 1.2, label="Perfect reliability")
        for mode, col, ls in (("single", "k", "-"), ("pooled", "0.5", "--")):
            mp, ob, n = relb[(mode, grp)]
            ax.plot(mp, ob, ls, marker="o", ms=3, color=col, label="λ = 50" if mode == "single" else "λ-pooled")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("Predicted P(ρ < 3 Ω·m)"); ax.set_ylabel("Observed frequency")
        finish(ax)
        eps.panel_label(ax, "ef"[col_i], text=("r ≤ 10 m", "10 < r ≤ 15 m")[col_i])
    legend_below(fig, *axes[2, 0].get_legend_handles_labels(), ncol=3)       # v1.9: shared, below (e, f), off the curves
    return fig, {}


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 5 / Fig. S3 — factorial test (25_generalisation)
# ─────────────────────────────────────────────────────────────────────────────
_G25 = {}


def g25_rows():
    if not _G25:
        g = mod("g25"); d = g.SRC
        done = [t for t in g.TRUTHS if all(os.path.exists(os.path.join(d, t, f"r{r}.npz")) for r in range(g.N_REAL))]
        rows = [r for t in done for r in quiet(g.truth_metrics, d, t)]
        _G25.update(g=g, done=done, rows=rows)
    return _G25


def fig05():
    G_ = g25_rows(); g, rows = G_["g"], G_["rows"]
    wq = (np.log10(3) - np.log10(g.WESTHOEK["rho_lens"])) / (np.log10(g.WESTHOEK["rho_below"]) - np.log10(g.WESTHOEK["rho_lens"]))
    cmap = {2: "#9ecae1", 4: "#4292c6", 8: "#08519c", 16: "#08306b"}
    fig = new_fig(118, dense=True)
    axes = fig.subplots(2, 4, sharey=True)
    for i, (ax, x) in enumerate(zip(axes.ravel(), g.DEFS)):
        for r in g.pick(rows, definition=x, treatment="lam50", group="≤10"):
            if not r["n"]:
                continue
            qq = r["q"] if x.startswith("thr") else g.q_of(3.0, g.params(r["truth"]))
            mk = "o" if r["tau"] == 0 else "^"
            ax.scatter(qq, 100 * r["coverage"], c=cmap[r["k"]], marker=mk, s=KSIZE[r["k"]], zorder=3,       # v1.9.1: thin dark edge
                       edgecolor="k" if r["rho_lens"] == 0.8 else eps.EDGE_DARK, lw=1.0 if r["rho_lens"] == 0.8 else 0.4)
        ax.axhline(90, color="0.3", lw=0.8, ls="--")
        ax.axvspan(1.0, 3.0, color="0.9", zorder=0); ax.axvline(0.5, color="0.45", lw=0.6, ls=":")
        ax.axvline(wq, color=C["thr3"], lw=1.0); ax.set_xlim(0, 2.8); ax.set_ylim(-3, 103)
        ax.set_xlabel("q of this contour" if x.startswith("thr") else "q of the 3 Ω·m contour")
        finish(ax)
        lab = g.DEF_LAB[x].replace("max. gradient", "max. gradient").replace("oracle √(ρ_lens·ρ_below)", "oracle")
        eps.panel_label(ax, "abcdefgh"[i], text=lab, wrap=True)
    axes[0, 0].set_ylabel("Coverage of the\n90 % band (%)"); axes[1, 0].set_ylabel("Coverage of the\n90 % band (%)")
    h = [Line2D([], [], marker="o", ls="", color=cmap[k], mec=eps.EDGE_DARK, mew=0.4, ms=np.sqrt(KSIZE[k]), label=f"k = {k}") for k in g.KS] + \
        [Line2D([], [], marker="^", ls="", color="0.5", mec=eps.EDGE_DARK, mew=0.4, label="τ = 5 m"),
         Line2D([], [], marker="o", ls="", mfc="0.5", mec="k", mew=1.0, label="ρ$_\\mathbf{lens}$ = 0.8 Ω·m"),
         Line2D([], [], color="0.3", ls="--", label="Nominal 90 %"), Line2D([], [], color=C["thr3"], label=f"Westhoek, 3 Ω·m (q = {wq:.2f})"),
         Patch(facecolor="0.9", label="q > 1: threshold outside the lens–below contrast")]
    legend_below(fig, h, [x.get_label() for x in h], ncol=5)
    return fig, dict(wq=wq)


def figS03_data(outdir):
    """Re-run the seeded fit of 25_generalisation (fit_rule, unchanged) and verify it against v13_summary.json. Writes the
    scatter points to <outdir>/data/FigS03_points.csv (new file). Raises if any stored constant is not reproduced."""
    G_ = g25_rows(); g, rows, done = G_["g"], G_["rows"], G_["done"]
    fit = quiet(g.fit_rule, rows, "lam50", seed=[136200])
    fit_lam = {int(l): quiet(g.fit_rule, rows, f"lam{int(l)}_r01", seed=[136201]) for l in [5.0, 10.0, 20.0, 50.0, 100.0]}
    grad = [t for t in g.TRUTHS if g.params(t)["tau"] > 0]
    fit_grad = quiet(g.fit_rule, rows, "lam50", truths=grad, seed=[136200])
    V = quiet(g.verdicts, rows)
    S = json.load(open(os.path.join(ROOT, "data", "processed", "v1_3", "v13_summary.json"), encoding="utf-8"))
    chk = {}
    def same(a, b):
        return all(a.get(k) == b.get(k) for k in ("a", "r2", "n", "a90"))
    chk["fit"] = same(fit, S["fit"])
    chk["fit_lam"] = all(same(fit_lam[int(k)], v) for k, v in S["fit_lam"].items())
    chk["fit_grad"] = same(fit_grad, S["fit_grad"])
    e9n, e9s = V["E9"], S["verdicts"]["E9"]
    chk["E9"] = (e9n["n_ok"], e9n["n"]) == (e9s["n_ok"], e9s["n"])
    info = dict(a=fit["a"], r2=fit["r2"], n=fit["n"], a90=fit["a90"], E9=f"{e9n['n_ok']}/{e9n['n']}", stored_E9=f"{e9s['n_ok']}/{e9s['n']}",
                checks=chk)
    if not all(chk.values()):
        raise SystemExit(f"Fig. S3: the re-run fit does not reproduce v13_summary.json: {info}")
    os.makedirs(os.path.join(outdir, "data"), exist_ok=True)
    p = os.path.join(outdir, "data", "FigS03_points.csv")
    keys = ["set", "lam", "truth", "d", "k", "z_b", "rho_lens", "q", "rbar", "x", "y", "pred", "resid"]
    txt = ",".join(keys) + "\n"
    for name, fr, lam in [("lam50", fit, 50)] + [(f"lam{l}_r01", fl, l) for l, fl in fit_lam.items()] + [("gradual_lam50", fit_grad, 50)]:
        for q_ in fr["pts"]:
            txt += ",".join(str(v) for v in [name, lam, q_["truth"], q_["d"], q_["k"], q_["z_b"], q_["rho_lens"], q_["q"], q_["rbar"],
                                             q_["x"], q_["y"], q_["pred"], q_["resid"]]) + "\n"
    if os.path.exists(p):                     # never overwritten: a re-run must reproduce the stored points exactly
        if open(p, encoding="utf-8").read() != txt:
            raise SystemExit(f"{p}: re-computed points differ from the stored file")
        info["points_file"] = "existing file reproduced exactly"
    else:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
        info["points_file"] = "written"
    return fit, fit_lam, fit_grad, info


def figS03(fit, fit_lam, fit_grad):
    g = g25_rows()["g"]
    cmap = {2: "#9ecae1", 4: "#4292c6", 8: "#08519c", 16: "#08306b"}
    fig = new_fig(108)
    axes = fig.subplots(1, 2)
    ax = axes[0]
    for p in fit["pts"]:
        ax.scatter(p["x"] * fit["a"], p["y"], c=cmap[p["k"]], s=KSIZE[p["k"]] * 1.2, marker={"thr2": "s", "thr3": "o", "thr5": "D"}[p["d"]], zorder=3,
                   edgecolors=eps.EDGE_DARK, linewidths=0.4)                     # v1.9.1: thin dark edge
    for p in fit_grad["pts"]:
        ax.scatter(p["x"] * fit["a"], p["y"], facecolor="none", edgecolor="#d35400", s=18, marker="^", zorder=3)
    lim = max(1, np.nanmax(np.abs([p["y"] for p in fit["pts"]] + [p["x"] * fit["a"] for p in fit["pts"]]))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "--", color="red", lw=eps.size("line") * 1.2); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Predicted bias −a·r̄·Φ⁻¹(q) (m)"); ax.set_ylabel("Observed median bias (m)")
    finish(ax)
    hh = [Line2D([], [], marker=m_, ls="", color="0.4", label=l) for m_, l in (("s", "2 Ω·m"), ("o", "3 Ω·m"), ("D", "5 Ω·m"))]
    hh += [Line2D([], [], marker="^", ls="", mfc="none", mec="#d35400", label="τ = 5 m (not fitted)")]
    hh += [Line2D([], [], marker="o", ls="", color=cmap[k], mec=eps.EDGE_DARK, mew=0.4, ms=np.sqrt(KSIZE[k] * 1.2), label=f"k = {k}") for k in g.KS]
    eps.panel_label(ax, "a")
    eps.stats_box(ax, f"λ = 50\na = {fit['a']:.3f} [{fit['a90'][0]:.3f}, {fit['a90'][1]:.3f}]\nR² = {fit['r2']:.2f}", loc="upper right")
    ax = axes[1]
    lc = {5: "#fdae6b", 10: "#fd8d3c", 20: "#e6550d", 50: "#a63603", 100: "#7f2704"}
    for l, fl in fit_lam.items():
        if fl:
            ax.scatter([p["x"] * fl["a"] for p in fl["pts"]], [p["y"] for p in fl["pts"]], c=lc[l], s=12, marker=LAM_MARK[l],
                       edgecolors=eps.EDGE_DARK, linewidths=0.4,
                       label=f"λ = {l}: a = {fl['a']:.3f}, R² = {fl['r2']:.2f}")
    lim2 = ax.get_xlim(); m_ = max(abs(lim2[0]), abs(lim2[1]), 1)
    ax.plot([-m_, m_], [-m_, m_], "--", color="red", lw=eps.size("line") * 1.2)
    ax.set_xlabel("Predicted bias −a$_\\mathbf{λ}$·r̄·Φ⁻¹(q) (m)"); ax.set_ylabel("Observed median bias (m)")
    finish(ax); eps.panel_label(ax, "b")
    hb, lb = ax.get_legend_handles_labels()                                  # v1.9: one legend below both panels, off the data
    blank = Line2D([], [], ls="", marker="")
    H = hh[:4] + [blank] + hh[4:] + [blank] + hb                              # columns: definitions | k | λ of (b)
    legend_below(fig, H, [x.get_label() for x in hh[:4]] + [""] + [x.get_label() for x in hh[4:]] + [""] + [f"(b) {x}" for x in lb], ncol=3)
    return fig, {}


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 6 — BEL prior and calibration
# ─────────────────────────────────────────────────────────────────────────────
def fig06():
    s18 = mod("s18"); b13 = s18.b13; TX = s18.TX
    B = quiet(s18.bel_products)
    exs = json.loads(str(B["examples"]))
    fig = new_fig(205, dense=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 0.07, 1.0, 1.0])
    ex, ez = wc.land_surface()
    xg = np.arange(0, 505.5, 1.0); zg = np.arange(8.0, -45.01, -0.25)
    X, Z = np.meshgrid(xg, zg)
    for i, th in enumerate(exs):
        ax = fig.add_subplot(gs[0, i])
        pm = b13.PriorModel2(th)
        V = pm.log10rho(X.ravel(), Z.ravel()).reshape(X.shape); V[Z > np.interp(X, ex, ez)] = np.nan
        pc = ax.pcolormesh(xg, zg, 10 ** V, cmap=eps.RES_CMAP, norm=eps.res_norm(), shading="auto", rasterized=eps.raster(),
                           edgecolors="face", antialiased=True, **({"linewidth": eps.COREL_CELL_LW} if eps.is_corel() else {}))
        ax.plot(TX, pm.target(), "k-", lw=0.9, path_effects=halo())        # v1.9.1: halo on dark cells
        ax.plot(ex, ez, "k-", lw=0.6)
        ax.set_xlim(0, 505); ax.set_ylim(-45, 9); ax.set_aspect("auto")
        dune_hatch(ax, text=False); electrodes(ax)
        ax.set_xlabel("Distance (m)")
        finish(ax, grid=False)
        txt = f"Sample {i + 1}" + (", discharge zone" if th["dis_present"] else "")
        if i == 0:
            ax.set_ylabel("Elevation (mTAW)"); eps.panel_label(ax, "a", text=txt)
        else:
            eps.panel_ident(ax, txt)
    cax = fig.add_subplot(gs[1, :])
    eps.add_colorbar(fig, None, pc, f"{RHO} (Ω·m)", ticks=[1, 3, 10, 30, 100, 300], cax=cax, orientation="horizontal")
    for k, (key, fkey, lab, ttl) in enumerate((("t2_train", "t2_field", "Hotelling T² (data PCA)", "T²"),
                                               ("t2r_train", "t2r_field", "Robust T² (FastMCD)", "Robust T²"),
                                               ("q_train", "q_field", "Q (residual outside the PCA)", "Q"))):
        ax = fig.add_subplot(gs[2, k])
        v = B[key]; fv = B[fkey]
        bins = np.logspace(np.log10(v.min()), np.log10(max(v.max(), fv)), 45)
        ax.hist(v, bins=bins, color="steelblue", edgecolor="black", lw=eps.size("marker_edge") * 0.6); ax.set_xscale("log")
        ax.axvline(fv, color="navy", lw=eps.size("line") * 1.15, label=f"Field ({100*np.mean(v < fv):.0f}th pct.)")
        ax.axvline(np.percentile(v, 99), color="red", lw=0.9, ls="--", label="Training P99")
        ax.set_xlabel(lab)
        finish(ax)
        ttl = f"{ttl}, field at {100*np.mean(v < fv):.0f}th pct."      # v1.9: was in the legend, which covered the data
        if k == 0:
            ax.set_ylabel("Training samples"); eps.panel_label(ax, "b", text=ttl)
        else:
            eps.panel_ident(ax, ttl)
        h_b = ax.get_legend_handles_labels()
    ax = fig.add_subplot(gs[3, 0:2])
    rk = B["ho_rank"].ravel()
    ax.hist(rk, bins=20, range=(0, 1), color="steelblue", edgecolor="black", lw=eps.size("marker_edge") * 0.6)
    ax.axhline(rk.size / 20, color="k", ls="-.", lw=0.9, label="Uniform")
    ax.set_xlabel("Rank of the true lens base among 2000 posterior samples"); ax.set_ylabel("Count")
    finish(ax); eps.panel_label(ax, "c")
    h_c = ax.get_legend_handles_labels()
    legend_below(fig, h_b[0] + h_c[0], ["Field value (b)", "Training P99 (b)", "Uniform (c)"], ncol=3)   # v1.9: one legend, off the data
    ax = fig.add_subplot(gs[3, 2])
    ax.plot(TX, 100 * B["ho_cov"].mean(0), "-o", ms=2.5, color="steelblue", mec="k", mew=0.3)
    ax.axhspan(85, 95, color="green", alpha=0.12); ax.axhline(90, color="k", lw=0.7, ls=":")
    ax.set_ylim(60, 100); ax.set_xlabel("Distance (m)"); ax.set_ylabel("Coverage (%)")
    finish(ax); eps.panel_label(ax, "d")
    return fig, dict(ho_cov=float(100 * B["ho_cov"].mean()))


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 7 — field lens base
# ─────────────────────────────────────────────────────────────────────────────
def fig07():
    s18 = mod("s18"); TX = s18.TX
    B = quiet(s18.bel_products); v4 = np.load(s18.LIN_V4); pres = quiet(s18.lens_presence)
    fig = new_fig(158)
    axes = fig.subplots(2, 1)
    ex, ez = wc.land_surface()
    surf = lambda x: np.interp(x, ex, ez); clip = lambda x, v: np.minimum(v, surf(x))
    h = TX >= 130
    ax = axes[0]
    x4 = v4["x"]; b = v4["pooled_logmid"]; ok = b[0] >= 0.5
    band(ax, x4, np.where(ok, clip(x4, b[2]), np.nan), np.where(ok, clip(x4, b[3]), np.nan), C["logmid"], 0.2, "--")
    ax.plot(x4, np.where(ok, clip(x4, b[1]), np.nan), color=C["logmid"], lw=1.2, ls="--", label="Linearised log-midpoint, λ-pooled: median, 90 %")
    lo, md, hi = np.percentile(B["Pf"], [5, 50, 95], 0)
    band(ax, TX[h], clip(TX[h], lo[h]), clip(TX[h], hi[h]), C["bel"], 0.25, "-")
    ax.plot(TX[h], clip(TX[h], md[h]), color=C["bel"], lw=1.7, label="BEL (prior v3): median, 90 %")
    xs = np.array([100.0, 110.0, 120.0]); ps = [pres[str(int(x))]["p"] for x in xs]
    ax.plot(xs, surf(xs) + 1.5, "v", color=C["bel"], ms=4.5, mec="k", mew=0.4, label=f"P(lens present), x = 100–120 m: {min(ps):.2f}")
    ax.plot(ex, ez, "k-", lw=0.8, label="Land surface")
    ax.set_xlim(-5, 510); ax.set_ylim(-32, 9); ax.set_aspect(3); dune_hatch(ax)
    ax.set_ylabel("Elevation (mTAW)"); ax.set_xlabel("Distance (m)"); ve_text(ax, 3)
    finish(ax); eps.panel_label(ax, "a")
    legend_below(ax, ncol=2, bbox_to_anchor=(0.5, -0.30))
    ax = axes[1]
    styles = [("#b2182b", "-"), ("#2166ac", "--"), ("#8c6d31", "-.")]
    for k, ((lo_, hi_), (col, ls)) in enumerate(zip(B["bins"], styles)):
        l, mm, hh = np.percentile(B["Pbins"][k], [5, 50, 95], 0)
        band(ax, TX[h], clip(TX[h], l[h]), clip(TX[h], hh[h]), col, 0.14, ls)
        ax.plot(TX[h], clip(TX[h], mm[h]), color=col, ls=ls, lw=1.5,
                label=f"{RHO_BELOW} {lo_}–{hi_} Ω·m (held-out coverage {100*B['bins_cov'][k]:.0f} %)")
    ax.plot(TX[h], clip(TX[h], md[h]), color=C["bel"], ls=":", lw=1.6, label=f"All {RHO_BELOW} (panel a)")
    ax.plot(ex, ez, "k-", lw=0.8)
    ax.set_xlim(-5, 510); ax.set_ylim(-32, 9); ax.set_aspect(3); dune_hatch(ax, label=False)
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("Elevation (mTAW)"); ve_text(ax, 3)
    finish(ax); eps.panel_label(ax, "b")
    legend_below(ax, ncol=2, bbox_to_anchor=(0.5, -0.30))
    meta = {xx: [float(np.median(B["Pf"][:, int(np.argmin(np.abs(TX - xx)))])), float(np.percentile(B["Pf"][:, int(np.argmin(np.abs(TX - xx)))], 5)),
                 float(np.percentile(B["Pf"][:, int(np.argmin(np.abs(TX - xx)))], 95))] for xx in (150, 200, 250, 300, 350)}
    return fig, dict(bel=meta, Pf=B["Pf"], TX=TX)


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 8 — profile against the soundings and 1980
# ─────────────────────────────────────────────────────────────────────────────
def fig08():
    s18 = mod("s18"); g27 = mod("g27"); k1 = mod("k1")
    R, names = dp_records()
    v4 = np.load(s18.LIN_V4); x4, b4, r4 = v4["x"], v4["pooled_logmid"], v4["r_logmid"]
    zb_ = np.load(os.path.join(ROOT, "data", "processed", "v1_1", "k1_field_bel.npz")); TX, Pf = zb_["TX"], zb_["Pf"]
    lb = k1.read_lebbe(); ex, ez = wc.land_surface()
    fig = new_fig(104)
    ax = fig.add_subplot()
    ok = (b4[0] >= 0.5) & (r4 <= 10.0)
    ax.axvspan(*k1.DIS_ZONE, color="#e6c229", alpha=0.16, lw=0, zorder=0)
    band(ax, x4, np.where(ok, b4[2], np.nan), np.where(ok, b4[3], np.nan), C["logmid"], 0.18, "--")
    h_lm, = ax.plot(x4, np.where(ok, b4[1], np.nan), color=C["logmid"], lw=1.1, ls="--")
    lo, md, hi = np.percentile(Pf, [5, 50, 95], 0); hh = TX >= 130
    band(ax, TX[hh], lo[hh], hi[hh], C["bel"], 0.22, "-"); h_bel, = ax.plot(TX[hh], md[hh], color=C["bel"], lw=1.7)
    v = lb["saline_base_mTAW"]; m = np.isfinite(v)
    h80 = ax.errorbar(lb["x_ours_m"][m], v[m], xerr=k1.X_ERR, yerr=k1.Z_ERR, fmt="o", color="#b2182b", ms=4, lw=0.8, capsize=1.5)
    for nm in names:
        for fl in R["dps"][nm]["fine_layers"]:
            x = R["dps"][nm]["our_x"]
            ax.add_patch(Rectangle((x - 4, fl["bottom"]), 8, fl["top"] - fl["bottom"], facecolor="none", edgecolor=DARK, hatch="xxxx", lw=0.5))
    P_ = g27.P_
    xd = np.array([P_(R, nm)["x"] for nm in names]); yd = np.array([P_(R, nm)["logmid"] for nm in names])
    y3 = np.array([P_(R, nm)["thr3"] for nm in names])
    h_dp = ax.errorbar(xd, yd, xerr=1.5, fmt="D", color="k", ms=4.5, lw=1.0, capsize=2)
    h_dp3, = ax.plot(xd, y3, "D", mfc="none", mec="k", ms=4.5)
    for nm, xx, yy in zip(names, xd, yd):
        ax.text(xx, (yy if np.isfinite(yy) else -30) - 2.4, nm, ha="center", va="top", fontsize=eps.size("tick"))
    h_s, = ax.plot(ex, ez, "k-", lw=0.9); ax.axvline(161, color="steelblue", lw=0.8, ls=":")
    ax.set_xlim(0, 505); ax.set_ylim(-34, 8); dune_hatch(ax)
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("Elevation (mTAW)"); ve_text(ax, 3); ax.set_aspect(3)
    finish(ax)
    blank = Patch(facecolor="none", edgecolor="none")
    hs = [blank, h_lm, h_bel, Patch(facecolor="#e6c229", alpha=0.35, edgecolor="none"), h_s, blank, h80, blank, h_dp, h_dp3,
          Patch(facecolor="none", edgecolor=DARK, hatch="xxxx")]
    ls_ = ["This study, ERT of 7 March 2018", "λ-pooled log-midpoint (r ≤ 10 m): median, 90 %", "BEL (prior v3): median, 90 %",
           "Discharge zone (this study)", "Land surface (2018 electrodes)", "Lebbe (1981), K1, logs of 1980", "Saline base (< 2.5 Ω·m)",
           "Soundings DP1–DP7 (CPT / CPT-EC, 23 September 2024)", "Sounding base, log-midpoint (offset ≤ 1.5 m)", "Sounding 3 Ω·m crossing",
           "CPT fine-grained layers (I$_\\mathbf{SBT}$ > 2.6)"]
    leg = legend_below(ax, hs, ls_, ncol=2, bbox_to_anchor=(0.5, -0.24))
    for t in leg.get_texts():
        if t.get_text() in (ls_[0], ls_[5], ls_[7]):
            t.set_fontstyle("italic")
    trim_height(fig)                                             # v1.9: no empty band above the section
    return fig, {}


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 9 / Fig. S5 — per-sounding panels
# ─────────────────────────────────────────────────────────────────────────────
IMG_COL = C["predicted"]          # v1.9.1: #DC3D38 (tomato #FF6347 was 2.95:1 against white)
# v1.9 second cues (no distinction by colour alone; AGU graphics guidance)
HATCH = dict(thr3="////", maxgrad="....", logmid="")
MARK = dict(thr3="o", logmid="s", maxgrad="D")
KSIZE = {2: 7, 4: 13, 8: 22, 16: 34}                      # marker area (pt²) grows with the contrast k
LAM_MARK = {5: "v", 10: "s", 20: "D", 50: "o", 100: "^"}
T_MARK = {"T1": "o", "T2": "s", "T3": "D"}
# v1.9.1: CPT class strips of Fig. S10 at ≥ 3:1 against white, light → dark kept (were #f6e7b4, #d9c48f: 1.2–1.7:1)
SAND = ("#B58C45", "#907237", "#7a6a55")


def _dp_label(ax, letter, nm, extra=None):
    eps.panel_label(ax, letter, text=f"{nm}\nx = {DP_X[nm]:.1f} m" + (f"\n{extra}" if extra else ""))


def fig09(img_col=None):
    img_col = img_col or IMG_COL
    g27 = mod("g27"); R, names = dp_records(); logs = g27.read_dp_values(); n = len(names)
    fig = new_fig(100, dense=True)
    axes = fig.subplots(1, n, sharey=True)
    meta = {}
    for j, (ax, nm) in enumerate(zip(axes, names)):
        o = R["dps"][nm]["per"]["primary|A"]; s_ = o["sd"]; lg = logs[min(logs, key=lambda k: abs(k - R["dps"][nm]["local_x"]))]
        bl = s_["band_logmid_pooled"]; bt = s_["band_thr3"]
        if bl[0] >= 0.5:
            ax.axhspan(bl[2], bl[3], facecolor="none", edgecolor=C["logmid"], hatch="////", lw=0)
        if bt[0] >= 0.5:
            ax.axhspan(bt[2], bt[3], color=C["thr3"], alpha=0.25, lw=0)
        ax.plot(lg["r"], o["surface"] - lg["depth"], color="0.78", lw=0.4)
        ax.plot(10 ** np.array(o["v_grid"]), o["z_grid"], color=C["observed"], lw=1.1)
        ic = np.array([np.nan if v is None else v for v in o["img_col"]["v"]], float)
        ax.plot(10 ** ic, o["img_col"]["z"], color=img_col, lw=1.3, ls="--")
        for v, ls in ((3.0, "--"), (20.0, "-.")):
            ax.axvline(v, color="0.45", lw=0.6, ls=ls)
        ax.set_xscale("log"); ax.set_xlim(0.3, 300); ax.set_ylim(-36, 6)
        finish(ax)
        _dp_label(ax, "abcdefg"[j], nm)
        ax.text(0.97, 0.02, f"r = {mod('g27').f_r(s_['r'])} m", transform=ax.transAxes, ha="right", va="bottom", fontsize=eps.size("tick"),
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=50)
        meta[nm] = dict(logmid=[float(v) for v in bl], thr3=[float(v) for v in bt])
    axes[0].set_ylabel("Elevation (mTAW)"); axes[n // 2].set_xlabel(f"{RHO} (Ω·m)")
    h = [Line2D([], [], color=C["observed"], label="CPT-EC log, 0.5 m median"), Line2D([], [], color="0.78", label="CPT-EC log, 1 cm"),
         Line2D([], [], color=img_col, ls="--", label="2025 ERT line, image at the sounding (λ = 50)"),
         Patch(facecolor="none", edgecolor=C["logmid"], hatch="////", label="Log-midpoint, λ-pooled, 90 %"),
         Patch(facecolor=C["thr3"], alpha=0.35, label="3 Ω·m contour, λ = 50, 90 %"),
         Line2D([], [], color="0.45", ls="--", lw=0.6, label="3 Ω·m"), Line2D([], [], color="0.45", ls="-.", lw=0.6, label="20 Ω·m")]
    legend_below(fig, h, [x.get_label() for x in h], ncol=4)
    return fig, meta


def figS05():
    g27 = mod("g27"); R, names = dp_records(); logs = g27.read_dp_values(); n = len(names)
    fig = new_fig(118, dense=True)
    gs = fig.add_gridspec(1, 2 * n, width_ratios=[5, 1] * n, wspace=0.02)
    ax0 = None
    for j, nm in enumerate(names):
        o = R["dps"][nm]["per"]["primary|A"]; lg = logs[min(logs, key=lambda k: abs(k - R["dps"][nm]["local_x"]))]
        ax = fig.add_subplot(gs[0, 2 * j], sharey=ax0); ax0 = ax0 or ax
        z1 = o["surface"] - lg["depth"]
        ax.axhspan(o["bel"][1], o["bel"][2], color=C["bel"], alpha=0.18, lw=0); ax.axhline(o["bel"][0], color=C["bel"], lw=1.1)
        lp = o["logmid_pooled"]
        if lp[0] >= 0.5:
            ax.axhspan(lp[2], lp[3], facecolor="none", edgecolor=C["logmid"], hatch="////", lw=0)
            ax.axhline(lp[1], color=C["logmid"], lw=0.9, ls="-.")
        ax.axhline(o["sal1980"], color="#b2182b", lw=1.0, ls=":")
        ax.plot(lg["r"], z1, color="0.78", lw=0.4); ax.plot(10 ** np.array(o["v_grid"]), o["z_grid"], color=C["observed"], lw=1.0)
        rc = np.array([np.nan if v is None else v for v in o["ref_col"]["v"]], float)
        ax.plot(10 ** rc, o["ref_col"]["z"], color=IMG_COL, lw=1.2, ls="--")
        for v, ls in ((2.5, ":"), (3.0, "--"), (20.0, "-.")):
            ax.axvline(v, color="0.45", lw=0.6, ls=ls)
        if np.isfinite(o["logmid"]):
            ax.plot([0.35], [o["logmid"]], marker=">", color="k", ms=5, clip_on=False)
        ax.set_xscale("log"); ax.set_xlim(0.3, 300); ax.set_ylim(-36, 6)
        finish(ax)
        _dp_label(ax, "abcdefg"[j], nm)
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
        sx = fig.add_subplot(gs[0, 2 * j + 1], sharey=ax0)
        I = np.array([np.nan if v is None else v for v in R["dps"][nm]["isbt"]], float); ze = np.array(R["dps"][nm]["cpt_elev"])
        colr = np.where(I < g27.ISBT_SAND, SAND[0], np.where(I <= g27.ISBT_FINE, SAND[1], SAND[2]))
        for zi, ci in zip(ze, colr):
            sx.axhspan(zi - 0.1, zi + 0.1, facecolor=ci, lw=0, hatch="xxx" if ci == SAND[2] else None, edgecolor="0.9",
                       gid="a11y-series:CPT class strip")
        sx.set_xticks([]); plt.setp(sx.get_yticklabels(), visible=False)
        for s_ in sx.spines.values():
            s_.set_linewidth(eps.size("spine"))
        sx.tick_params(which="both", direction="in", left=False, right=False)
    ax0.set_ylabel("Elevation (mTAW)")
    mains = [a_ for a_ in fig.axes if a_.get_xscale() == "log"]; mains[len(mains) // 2].set_xlabel(f"{RHO} (Ω·m)")
    h = [Line2D([], [], color="0.78", lw=0.8, label="CPT-EC log (1 cm)"), Line2D([], [], color=C["observed"], label="CPT-EC log, 0.5 m median"),
         Line2D([], [], color=IMG_COL, ls="--", label="2018 reference model column (λ = 50)"),
         Patch(facecolor=C["bel"], alpha=0.3, label="2018 BEL lens base: median, 90 %"),
         Patch(facecolor="none", edgecolor=C["logmid"], hatch="////", label="2018 log-midpoint, λ-pooled"),
         Line2D([], [], color="#b2182b", ls=":", label="1980 saline base"), Line2D([], [], marker=">", ls="", color="k", label="Sounding base (log-midpoint)"),
         Patch(facecolor=SAND[0], label="CPT clean sand"), Patch(facecolor=SAND[1], label="Sand mixtures"), Patch(facecolor=SAND[2], hatch="xxx", edgecolor="0.9", label="Fine-grained")]
    legend_below(fig, h, [x.get_label() for x in h], ncol=4)
    return fig, {}


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 10 / Fig. S10 — scores
# ─────────────────────────────────────────────────────────────────────────────
FIELD_METHODS = ["BEL3-18", "BEL4-18", "BEL4-18s", "BEL4-24", "LIN-LM-18", "LIN-3-18", "LIN-MG-18", "LIN-LM-24", "LIN-3-24", "LIN-MG-24",
                 "PRIOR3", "PRIOR4", "PRIOR4-s", "PRIOR4-L24"]
MLAB = {"BEL3-18": "BEL (prior v3), 2018", "BEL4-18": "BEL (prior v4), 2018", "BEL4-18s": "BEL (prior v4), 2018 beach stretch",
        "BEL4-24": "BEL (prior v4), 2025 ERT line", "LIN-LM-18": "Log-midpoint, 2018", "LIN-3-18": "3 Ω·m contour, 2018",
        "LIN-MG-18": "Maximum gradient, 2018", "LIN-LM-24": "Log-midpoint, 2025 ERT line", "LIN-3-24": "3 Ω·m contour, 2025 ERT line",
        "LIN-MG-24": "Maximum gradient, 2025 ERT line", "PRIOR3": "Prior v3", "PRIOR4": "Prior v4", "PRIOR4-s": "Prior v4, beach stretch",
        "PRIOR4-L24": "Prior v4, 2025 layout"}


def mmark(m):
    """Marker by method family (second cue to the colour)."""
    if m.startswith("BEL"):
        return "o"
    if m.startswith("PRIOR"):
        return "v"
    if "-LM-" in m or "logmid" in m:
        return "s"
    if "-3-" in m or "thr3" in m:
        return "^"
    return "D"


def mcol(m):
    if m.startswith("BEL"):
        return C["bel"]
    if m.startswith("PRIOR"):
        return C["prior"]
    if "-LM-" in m or "logmid" in m:
        return C["logmid"]
    if "-3-" in m or "thr3" in m:
        return C["thr3"]
    return C["maxgrad"]


def fig10():
    O = json.load(open(os.path.join(ROOT, "data", "processed", "v1_6", "scores.json"), encoding="utf-8"))
    FS, SG = O["field"], O["synthetic"]
    fig = new_fig(92)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
    axes = [fig.add_subplot(gs[0, 0])]; axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0])); axes.append(fig.add_subplot(gs[0, 2]))
    ys = np.arange(len(FIELD_METHODS))[::-1]
    for i_, (ax, k) in enumerate(((axes[0], "IS"), (axes[1], "CRPS"))):
        for yv, m in zip(ys, FIELD_METHODS):
            o = FS[m][k]
            ax.errorbar(o["mean"], yv, xerr=[[o["mean"] - o["lo"]], [o["hi"] - o["mean"]]], fmt=mmark(m), color=mcol(m), ms=3.5, lw=1,
                        capsize=2, mec="k", mew=0.3)
        ax.set_xlabel("Mean 90 % interval score (m)" if k == "IS" else "Mean CRPS (m)")
        finish(ax); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        eps.panel_label(ax, "ab"[i_])
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([f"{i + 1}  {MLAB[m]}" + (" ‡" if FS[m]["approx"] else "") for i, m in enumerate(FIELD_METHODS)])
    plt.setp(axes[1].get_yticklabels(), visible=False)
    ax = axes[2]
    for i, m in enumerate(FIELD_METHODS):
        o = FS[m]; ax.plot(o["width"], o["cov"] / o["n"], mmark(m), color=mcol(m), ms=5, mec="k", mew=0.3, zorder=4)
    pts = sorted([(FS[m]["cov"] / FS[m]["n"], np.log10(FS[m]["width"]), i + 1) for i, m in enumerate(FIELD_METHODS)])
    groups = []
    for c_, lw_, k_ in pts:                      # numbers of overlapping points (same coverage, width within 10 %) share one label
        if groups and groups[-1][0] == c_ and lw_ - groups[-1][1][-1] < 0.04:
            groups[-1][1].append(lw_); groups[-1][2].append(k_)
        else:
            groups.append((c_, [lw_], [k_]))
    for c_, lws, ks in groups:
        ks = sorted(ks); lab = (f"{ks[0]}–{ks[-1]}" if ks == list(range(ks[0], ks[-1] + 1)) and len(ks) > 2 else ", ".join(map(str, ks)))
        ax.annotate(lab, (10 ** np.mean(lws), c_), xytext=(0, -6 if c_ > 0.97 else 5), textcoords="offset points", ha="center",
                    va="top" if c_ > 0.97 else "bottom",
                    fontsize=eps.size("tick"), zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
    for g in SG:
        ax.plot(g["width"], g["cov"], mmark(g["method"]), mfc="none", color=mcol(g["method"]), ms=4, mew=0.9)
        ax.plot(g["prior_width"], g["prior_cov"], mmark("PRIOR"), mfc="none", color=C["prior"], ms=3, mew=0.9)
    ax.axhline(0.9, color="0.5", lw=0.6, ls=":"); ax.set_xscale("log")
    ax.set_xlabel("Sharpness: median 90 % width (m)"); ax.set_ylabel("Share inside the 90 % interval")
    finish(ax); eps.panel_label(ax, "c")
    h = [Line2D([], [], marker="o", ls="", color="0.4", mec="k", label="Field (1–14)"),
         Line2D([], [], marker="o", ls="", mfc="none", color="0.4", label="Synthetic"),
         Line2D([], [], marker=mmark("PRIOR"), ls="", mfc="none", color=C["prior"], label="Their priors")]
    eps.styled_legend(ax, handles=h, loc="lower right")
    meta = {m: FS[m] for m in FIELD_METHODS}
    return fig, meta


def s10_label(g):
    grp, lab = g["group"], g["label"]
    lab2 = (lab.replace("LIN logmid", "log-midpoint").replace("LIN maxgrad", "maximum gradient").replace("LIN thr3", "3 Ω·m contour"))
    if grp == "held-out":
        return {"BEL v3 (L18)": "Held-out: BEL (prior v3)", "BEL v4 (L18)": "Held-out: BEL (prior v4), 2018 layout",
                "BEL v4 (L24)": "Held-out: BEL (prior v4), 2025 layout"}[lab]
    if grp == "v1.0 truths":
        return "Site-like truths: " + ("BEL (prior v3)" if lab.startswith("BEL") else lab2)
    if grp == "v1.3 factorial":
        return "Factorial test: " + lab2
    if grp.startswith("v1.5 T1–T3"):
        return f"Sounding-based truths, {'2018' if 'L18' in grp else '2025'} layout: " + lab2
    if grp == "v1.5 BEL v4":
        return {"BEL v4 on T1/L18": "BEL (prior v4), soundings (T1), 2018 layout", "BEL v4 on T3/L18": "BEL (prior v4), brackish below (T3), 2018 layout",
                "BEL v4 on T1/L24": "BEL (prior v4), soundings (T1), 2025 layout",
                "BEL v4 on T1/L18s (L24-stretch subset)": "BEL (prior v4), soundings (T1), 2018 beach stretch"}[lab]
    return f"{grp}: {lab}"


def figS10():
    O = json.load(open(os.path.join(ROOT, "data", "processed", "v1_6", "scores.json"), encoding="utf-8"))
    SG = O["synthetic"]
    fig = new_fig(165)
    axes = fig.subplots(1, 2, sharey=True, gridspec_kw=dict(width_ratios=[1, 1]))
    ys = np.arange(len(SG))[::-1]
    for yv, g in zip(ys, SG):
        c = mcol(g["method"])
        axes[0].plot(g["IS"], yv, "o", color=c, ms=4, mec="k", mew=0.3); axes[0].plot(g["prior_IS"], yv, "|", color=C["prior"], ms=7, mew=1.2)
        axes[1].barh(yv, g["skill_IS"], color=c, height=0.6, edgecolor="k", lw=0.3)
    axes[0].set_xscale("log"); axes[0].set_yticks(ys)
    axes[0].set_yticklabels([s10_label(g) + (" ‡" if g["approx"] else "") for g in SG])
    axes[0].set_xlabel("Mean interval score (m); grey tick: prior"); axes[1].set_xlabel("Interval-score skill against the prior")
    axes[1].axvline(0, color="k", lw=0.6); axes[1].axvline(0.3, color="0.5", lw=0.5, ls=":")
    for i, ax in enumerate(axes):
        finish(ax); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator()); eps.panel_label(ax, "ab"[i])
    plt.setp(axes[1].get_yticklabels(), visible=False)
    return fig, {}


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 11 / Fig. S7 — water below the lens (BEL prior v4)
# ─────────────────────────────────────────────────────────────────────────────
def _v15_inputs():
    g30, g28 = mod("g30"), mod("g28")
    D4, D4L, DLT = g30.D4(), g30.D4L(), g30.DLT()
    Dg = {d["name"]: d for d in g28.dp_inputs()}
    return g30, g28, D4, D4L, DLT, Dg


def fig11(bins_a=None):
    """bins_a: histogram bin edges of panel (a) (v2.2: aligned to the prior bounds); default as v1.8."""
    g30, g28, D4, D4L, DLT, Dg = _v15_inputs()
    Z = np.load(os.path.join(DLT, "layout_vs_time.npz"), allow_pickle=True); S = np.load(os.path.join(D4, "synth.npz"))
    F = np.load(os.path.join(D4, "bel_v4.npz"), allow_pickle=True); L = np.load(os.path.join(D4L, "bel_v4_l24.npz"), allow_pickle=True)
    Z3 = np.load(os.path.join(DLT, "t1_subset.npz"))
    names = [d["name"] for d in sorted(g28.dp_inputs(), key=lambda d: -d["x"])]
    pooled = lambda Zs, pre: np.concatenate([10 ** Zs[k][:, 3] for k in Zs.files if k.startswith(pre) and k.endswith("_P2")])
    pT18, pT24, pT1s = pooled(S, "T1_r"), pooled(Z, "T1L24_r"), pooled(Z3, "T1sub_r")
    r_full, r_sub, r_ert = 10 ** F["P2"][:, 3], 10 ** Z["sub_P2"][:, 3], 10 ** L["P2"][:, 3]
    x0 = float(Z["x0"])
    fig = new_fig(90)
    axes = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axes[0]; bins = np.logspace(np.log10(1.5), np.log10(120), 40) if bins_a is None else np.asarray(bins_a, float)
    ax.hist(pT18, bins, density=True, alpha=0.40, color="#1f4e9c", label="Synthetic from soundings, 2018 layout")
    ax.hist(pT24, bins, density=True, facecolor=matplotlib.colors.to_rgba("#7fa6e0", 0.45), edgecolor="#1f4e9c", hatch="////", lw=0,
            label="Synthetic from soundings, 2025 layout")
    ax.hist(pT1s, bins, density=True, histtype="step", lw=1.2, ls="-.", color="#1f4e9c", label="Synthetic from soundings, 2018 beach stretch")
    ax.hist(r_full, bins, density=True, histtype="step", lw=1.2, color="#d35400", label="2018 data, full line")
    ax.hist(r_sub, bins, density=True, histtype="step", lw=1.2, ls="--", color="#d35400", label="2018 data, beach stretch")
    ax.hist(r_ert, bins, density=True, histtype="step", lw=1.6, ls=":", color="k", label="2025 ERT line")
    for v in (10.89, 13.57, 14.80):
        ax.axvline(v, color="#1f4e9c", ls=":", lw=0.7)
    ax.axvline(8.0, color="0.5", ls="-", lw=0.5)
    ax.set_xscale("log"); ax.set_xlabel(f"{RHO_BELOW} (Ω·m)"); ax.set_ylabel("Density")
    finish(ax); eps.panel_label(ax, "a")
    ax = axes[1]
    for P, TX, c, lab, a, ls_ in ((F["Pf"], F["TX"], "#d35400", "2018 data, full line", 0.15, "-"),
                                  (Z["sub_Pf"], Z["TX"], "steelblue", "2018 data, beach stretch", 0.25, "-.")):
        lo, md, hi = np.percentile(P, [5, 50, 95], 0)
        band(ax, TX, lo, hi, c, a, ls_); ax.plot(TX, md, color=c, lw=1.3, ls=ls_, label=lab)
    ax.plot(L["TX"], np.median(L["Pf"], 0), color="k", lw=1.0, ls="--", label="2025 ERT line, median")
    ax.axvline(x0, color="0.5", lw=0.6, ls=":"); ax.text(x0 + 2, 3.5, "2025 ERT line start", fontsize=eps.size("tick"), color="0.3")
    ax.plot([Dg[n]["x"] for n in names], [Dg[n]["base"] for n in names], "kD", ms=4, label="Sounding base (2024)")
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("Elevation (mTAW)"); ax.set_ylim(-25, 5)
    finish(ax); eps.panel_label(ax, "b")
    h1, l1 = axes[0].get_legend_handles_labels(); h2, l2 = axes[1].get_legend_handles_labels()
    legend_below(fig, h1 + h2, l1 + l2, ncol=3)
    return fig, {}


def figS07():
    g30, g28, D4, D4L, DLT, Dg = _v15_inputs()
    R, names = dp_records()
    S = np.load(os.path.join(D4, "synth.npz")); L = np.load(os.path.join(D4L, "bel_v4_l24.npz"), allow_pickle=True)
    TXL = L["TX"]
    fig = new_fig(90)
    axes = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axes[0]; bins = np.logspace(np.log10(1.5), np.log10(120), 40)
    labs = {"T1": "BEL (prior v4), synthetic from soundings (T1)", "T3": "BEL (prior v4), brackish below (T3)"}
    for t, col in (("T1", "#1f4e9c"), ("T3", "#d35400")):
        pooled = np.concatenate([10 ** S[f"{k[:-3]}_P2"][:, 3] for k in S.files if k.startswith(t) and k.endswith("_P2")])
        ax.hist(pooled, bins, density=True, facecolor=matplotlib.colors.to_rgba(col, 0.45), label=labs[t] + ", 2018 layout",
                hatch="\\\\\\\\" if t == "T3" else None, edgecolor="#8a3a00" if t == "T3" else "none", lw=0)
    ax.hist(10 ** L["P2"][:, 3], bins, density=True, histtype="step", lw=1.3, color="k", label="BEL (prior v4), 2025 ERT line")
    for v in (10.89, 13.57, 14.80):
        ax.axvline(v, color="#1f4e9c", ls=":", lw=0.7)
    ax.axvline(4.93, color="#d35400", ls=":", lw=0.7); ax.set_xscale("log"); ax.set_xlabel(f"{RHO_BELOW} (Ω·m)"); ax.set_ylabel("Density")
    finish(ax); eps.panel_label(ax, "a")
    ax = axes[1]
    lo, md, hi = np.percentile(L["Pf"], [5, 50, 95], 0)
    band(ax, TXL, lo, hi, C["bel"], 0.25, "-"); ax.plot(TXL, md, color=C["bel"], lw=1.4, label="BEL (prior v4), 2025 ERT line")
    base_rows = []
    for n in names:
        o = R["dps"][n]["per"]["primary|A"]; s_ = o["sd"]
        base_rows.append(dict(x=o["x"], dp_base=o["logmid"], lin=s_["band_logmid_pooled"][1:4]))
    xr = np.array([r["x"] for r in base_rows])
    ax.errorbar(xr + 3, [r["lin"][0] for r in base_rows], yerr=[[r["lin"][0] - r["lin"][1] for r in base_rows], [r["lin"][2] - r["lin"][0] for r in base_rows]],
                fmt="^", color=C["logmid"], ms=3.5, lw=0.8, mec="k", mew=0.3, label="Linearised log-midpoint, λ-pooled")
    ax.plot(xr, [r["dp_base"] for r in base_rows], "kD", ms=4, label="Sounding base (2024)")
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("Elevation (mTAW)"); ax.set_ylim(-25, 5)
    finish(ax); eps.panel_label(ax, "b")
    h1, l1 = axes[0].get_legend_handles_labels(); h2, l2 = axes[1].get_legend_handles_labels()
    legend_below(fig, h1 + h2, l1 + l2, ncol=2)
    return fig, {}


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary
# ─────────────────────────────────────────────────────────────────────────────
def figS01():
    s18 = mod("s18"); p04 = s18.p04
    F = s18.run_npz("deep_all")
    drop = s18.droplist(os.path.join(wc.RESULTS_DIR, "deep_all_droplist.txt"))
    keep = F["keep_idx"].astype(int); is_out = np.array([k in drop for k in keep])
    en, exx = F["elec_num"].astype(int), F["elec_x"]
    q4 = [F[k].astype(int) for k in ("C1", "C2", "P1", "P2")]
    rate = []
    for e in en:
        use = (q4[0] == e) | (q4[1] == e) | (q4[2] == e) | (q4[3] == e)
        rate.append(is_out[use].mean() if use.any() else np.nan)
    rate = np.array(rate)
    _, x_hw, x_lw = p04.water_lines(wc.ELEC_FILE, exx, F["elec_z"])
    flagged = [39, 40, 41, 42, 43, 44, 45, 55, 59, 65, 66]
    ctx = quiet(s18.ref_ctx)
    G1 = s18.run_npz("deep_drop_data")
    # lens bases of G1 / G2 from the stored models (18_paper.lens_base, unchanged). pyGIMLi's column sampler on the rebuilt mesh
    # is not bit-reproducible between processes (≤ 0.2 m at a few columns), so the arrays are stored once and reused.
    p = os.path.join(DATA_DIR, "FigS01_lens_bases.csv") if DATA_DIR else None
    xl = ctx["cs_line"].x
    if p and os.path.exists(p):
        A = np.genfromtxt(p, delimiter=",", names=True); bl1, bl2 = A["G1_logmid"], A["G2_logmid"]
        assert np.allclose(A["x"], xl)
    else:
        G2 = s18.run_npz("deep_drop_elec")
        bl1, bl2 = s18.lens_base(G1, ctx["cs_line"]), s18.lens_base(G2, ctx["cs_line"])
        if p:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write("x,G1_logmid,G2_logmid\n" + "".join(f"{a!r},{b!r},{c!r}\n" for a, b, c in zip(xl, bl1, bl2)))
    fig = new_fig(150)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.75, 1.25])
    ax = fig.add_subplot(gs[0])
    fl = np.isin(en, flagged)
    ax.bar(exx[~fl], 100 * rate[~fl], width=4, color="0.55", label="Electrodes")
    ax.bar(exx[fl], 100 * rate[fl], width=4, color="#c0392b", hatch="////", edgecolor="k", lw=0.3, label="Flagged electrodes (dropped in G2)")
    ax.axvline(x_hw, color="steelblue", ls="--", lw=1.0, label=f"High-water line (x = {x_hw:.0f} m)")
    ax.axvspan(x_hw, x_lw, color="steelblue", alpha=0.18, lw=0, label="Intertidal zone")
    ax.axhline(100 * is_out.mean(), color="k", lw=0.7, ls=":", label=f"All data ({100*is_out.mean():.1f} %)")
    ax.set_xlim(-5, 510); ax.set_ylim(0, max(60, 100 * np.nanmax(rate) + 5))
    dune_hatch(ax, label=False)
    ax.set_ylabel("Outlier rate of data\nusing the electrode (%)"); ax.set_xlabel("Distance (m)")
    finish(ax); eps.panel_label(ax, "a")
    legend_below(ax, ncol=3, bbox_to_anchor=(0.5, -0.32))
    ax = fig.add_subplot(gs[1])
    pc = cells(ax, ctx["polys"], G1["model"], eps.RES_CMAP, eps.res_norm())
    hatch_cells(ax, ctx["polys"], G1["res_radius"] > 10)
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]
    ax.plot(xl, xwin(xl, bl1), color=C["logmid"], lw=1.5, label="Log-midpoint lens base, G1 (outlier data dropped)", zorder=9, path_effects=halo)
    ax.plot(xl, xwin(xl, bl2), color="k", lw=1.5, ls="--", dashes=(4, 2), label="Log-midpoint lens base, G2 (flagged electrodes dropped)", zorder=9,
            path_effects=halo)
    section(ax, z_floor=-40); dune_hatch(ax, label=False); electrodes(ax)
    res_colorbar(fig, ax, pc)
    finish(ax, grid=False); eps.panel_label(ax, "b")
    legend_below(ax, ncol=2, bbox_to_anchor=(0.5, -0.26))
    return fig, {}


def figS02():
    J = json.load(open(os.path.join(ROOT, "data", "processed", "v1_4", "raw2018.json"), encoding="utf-8")); fa = J["fit_all"]
    P = np.load(os.path.join(ROOT, "data", "processed", "v1_4", "raw2018_pairs.npz"))
    Rb = (np.abs(P["Rn"]) + np.abs(P["Rr"])) / 2; dd = np.abs(P["Rn"] - P["Rr"]) / np.sqrt(2)
    fig = new_fig(88)
    axes = fig.subplots(1, 2)
    ax = axes[0]; pe_ = P["published_electrodes"]
    ax.loglog(Rb[~pe_], dd[~pe_], "x", ms=2.0, mew=0.4, color=C["prior"], label="Pairs (dune electrodes involved)", rasterized=eps.raster())
    ax.loglog(Rb[pe_], dd[pe_], ".", ms=2.5, color="steelblue", label="Pairs on published electrodes", rasterized=eps.raster())
    rr = np.logspace(np.log10(Rb.min()), np.log10(Rb.max()), 100)
    for k, col, ls in (("A", "#d35400", "-"), ("B", "#6a3d9a", "--")):
        p = fa[k]["params"]
        s = p[0] + p[1] * rr if k == "A" else np.sqrt((p[0] * rr) ** 2 + p[1] ** 2)
        ax.loglog(rr, s, ls, color=col, lw=1.2, label=f"Model {k}" + (" (best AIC)" if fa["best"] == k else ""))
    ax.loglog(rr, np.sqrt((0.03 * rr) ** 2 + 1e-4 ** 2), "k:", lw=1.3, label="Error model used: 3 % + 10$^{-4}$ Ω")
    ax.set_xlabel("|R| (Ω)"); ax.set_ylabel("|R$_\\mathbf{n}$ − R$_\\mathbf{r}$| / √2 (Ω)")
    finish(ax); eps.panel_label(ax, "a")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=3)       # v1.9: legend of (a) below the figure, off the data
    ax = axes[1]
    re = np.abs(P["Rn"] - P["Rr"]) / ((np.abs(P["Rn"] + P["Rr"])) / 2)
    ax.loglog(100 * P["stack"], 100 * re, ".", ms=2.0, color="steelblue", rasterized=eps.raster())
    lim = [1e-3, 1e3]; ax.plot(lim, lim, "--", color="red", lw=eps.size("line") * 1.2)
    ax.axhline(3, color="k", ls=":", lw=1); ax.axvline(1, color="0.5", ls="--", lw=0.8)
    ax.set_xlim(1e-3, 1e2); ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel("Stacking error (%)"); ax.set_ylabel("Reciprocal error (%)")
    ax.tick_params(axis="x", which="major", pad=5.5)                       # v1.10: corner labels 10⁻³ / 10⁻³ no longer touch
    finish(ax); eps.panel_label(ax, "b")
    return fig, {}


def figS04():
    g27 = mod("g27")
    wc.require_input("cpt_json", "the comparison with the soundings (Tables 1–2, SI Tables S1, S7–S11, Figures 4–6 and S11–S14, expectations E15–E21)")
    J = json.load(open(os.path.join(ROOT, "data", "processed", "v1_4", "cpt.json"), encoding="utf-8")); Cc = J["cpts"]; ids = sorted(Cc)
    fig = new_fig(180, dense=True)
    gs = fig.add_gridspec(4, len(ids), height_ratios=[0.09, 1, 1, 1])
    axes = np.empty((3, len(ids)), dtype=object)
    for i in range(3):
        for j in range(len(ids)):
            axes[i, j] = fig.add_subplot(gs[i + 1, j], sharey=axes[0, 0] if (i or j) else None)
    for j, s in enumerate(ids):
        hx = fig.add_subplot(gs[0, j]); hx.axis("off")
        hx.text(0.5, 0.0, f"{Cc[s]['dp']}\nx = {DP_X[Cc[s]['dp']]:.1f} m", ha="center", va="bottom", fontsize=eps.size("panel"), transform=hx.transAxes)
    for j, s in enumerate(ids):
        c = Cc[s]; z = np.array(c["elev"]); I = np.array([np.nan if v is None else v for v in c["isbt"]], float)
        axes[0, j].semilogx(c["qc"], z, color="0.2", lw=0.8)
        axes[1, j].plot(c["Rf"], z, color="#8c6d31", lw=0.8); axes[1, j].set_xlim(0, 6)
        ax = axes[2, j]
        ax.axvspan(0, g27.ISBT_SAND, color="#f6e7b4", lw=0); ax.axvspan(g27.ISBT_SAND, g27.ISBT_FINE, color="#e0cfa0", lw=0)
        ax.axvspan(g27.ISBT_FINE, 4.5, facecolor="#b7a58a", hatch="xx", edgecolor="0.85", lw=0)
        ax.plot(I, z, color="k", lw=0.8); ax.set_xlim(1, 4.0)
        for r in c["fine_layers"]:
            ax.axhspan(r["bottom"], r["top"], xmin=0.92, xmax=1.0, color=DARK, lw=0)
        for a in axes[:, j]:
            a.axhline(-30, color="red", lw=0.6, ls=":"); finish(a, grid=False)
        axes[0, j].set_xlabel("q$_\\mathbf{c}$ (MPa)"); axes[1, j].set_xlabel("R$_\\mathbf{f}$ (%)"); ax.set_xlabel("I$_\\mathbf{SBT}$")
        if j == 0:
            for i in range(3):
                eps.panel_label(axes[i, j], "abc"[i])
        else:
            for i in range(3):
                plt.setp(axes[i, j].get_yticklabels(), visible=False)
    for i in range(3):
        axes[i, 0].set_ylabel("Elevation (mTAW)")
    axes[0, 0].set_ylim(-36, 6)
    return fig, {}


def figS06():
    g28 = mod("g28")
    J = json.load(open(os.path.join(ROOT, "data", "processed", "v1_5", "dp_synth_summary.json"), encoding="utf-8"))
    M, Fv = J["metrics"], J["field"]; D = g28.dp_inputs(); names = [d["name"] for d in D]
    fig = new_fig(105)
    ax = fig.add_subplot()
    xs = np.array([d["x"] for d in D]); ax.plot(xs, [d["base"] for d in D], "k_", ms=16, mew=2, label="True base (sounding log-midpoint)")
    cols = {"T1": "#1f4e9c", "T2": "#8c6d31", "T3": "#b2182b"}
    tl = {"T1": "Logs (T1)", "T2": "Three units (T2)", "T3": "Brackish below (T3)"}
    for k, t in enumerate(g28.TRUTHS):
        dx = (k - 1) * 5
        ax.plot(xs + dx, [M[f"{t}|L24|{n}"]["thr3"] for n in names], T_MARK[t], color=cols[t], ms=4.5, mec="k", mew=0.3,
                label=f"{tl[t]}: 3 Ω·m contour, λ = 50")
        ax.plot(xs + dx, [M[f"{t}|L24|{n}"]["logmid_pooled"] for n in names], T_MARK[t], color=cols[t], ms=4.5, mfc="none", mew=0.9,
                label=f"{tl[t]}: log-midpoint, λ-pooled")
    fb = np.array([d["base"] for d in D])
    ax.plot(xs + 9, fb + np.array([Fv[n]["L24"]["thr3_off"] for n in names]), "h", color="0.3", ms=5, label="2025 ERT line: 3 Ω·m contour")
    ax.plot(xs + 9, fb + np.array([Fv[n]["L24"]["logmid_off"] for n in names]), "h", color="0.3", ms=5, mfc="none", mew=0.9,
            label="2025 ERT line: log-midpoint")
    for n, x in zip(names, xs):
        ax.text(x, -24, n, ha="center", fontsize=eps.size("tick"))
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("Elevation (mTAW)"); ax.set_ylim(-25, 0)
    finish(ax)
    legend_below(ax, ncol=2, bbox_to_anchor=(0.5, -0.16))
    return fig, {}


def figS08():
    s18 = mod("s18")
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon
    z = np.load(os.path.join(s18.DS6, "field_v2.npz"))
    sm = json.load(open(os.path.join(s18.DS6, "summary_v2.json"), encoding="utf-8"))
    XG, ZG = z["XG"], z["ZG"]; ex, ez = wc.land_surface()
    fig = new_fig(175)
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[2.1, 0.9]).ravel()
    gs = sf_top.add_gridspec(2, 1)
    for i, (key, cmap, ttl) in enumerate((("Psal", "Blues", "P(seawater > 50 %)"), ("Pfresh", "Oranges", "P(EC$_\\mathbf{25}$ < 1500 µS cm$^{-1}$), fresh"))):
        ax = sf_top.add_subplot(gs[i])
        Mm = np.where(z["Nv"] > 0, z[key], np.nan)
        for j in range(Mm.shape[1]):
            col = Mm[:, j]; ok = np.where(np.isfinite(col))[0]
            if ok.size:
                col[:ok[0]] = col[ok[0]]; col[ok[-1] + 1:] = col[ok[-1]]
        pc = ax.pcolormesh(XG, ZG, np.ma.masked_invalid(Mm), cmap=cmap, norm=mcolors.Normalize(0, 1), shading="auto", rasterized=eps.raster(),
                           edgecolors="face", antialiased=True, **({"linewidth": eps.COREL_CELL_LW} if eps.is_corel() else {}))
        top = np.minimum(np.interp(XG, ex, ez), np.median(z["par_z_wt"]))
        clip_poly = Polygon(np.column_stack([np.r_[XG, XG[::-1]], np.r_[top, np.full_like(XG, np.median(z["par_z_deep"]))]]),
                            closed=True, facecolor="none", edgecolor="none", transform=ax.transData)
        ax.add_patch(clip_poly); pc.set_clip_path(clip_poly)
        ax.plot(ex, ez, "k-", lw=0.8)
        ax.set_xlim(-5, 510); ax.set_ylim(-45, 9); ax.set_aspect(3); dune_hatch(ax); electrodes(ax)
        ax.set_ylabel("Elevation (mTAW)"); ax.set_xlabel("Distance (m)"); ve_text(ax, 3)
        eps.add_colorbar(sf_top, ax, pc, "Probability", ticks=[0, 0.25, 0.5, 0.75, 1.0], log_ticks_fmt=False)
        finish(ax, grid=False); eps.panel_label(ax, "ab"[i], text=ttl)
    ax = sf_bot.add_subplot()
    units = ("lens", "below", "dis", "dune")
    nm = {"lens": "Saline lens", "below": "Water below the lens", "dis": "Discharge-zone water", "dune": "Dune aquifer (weakly constrained)"}
    right = []
    for k, u in enumerate(units):
        v = z[f"usw_{u}"]; v = v[np.isfinite(v)]
        p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
        y = len(units) - 1 - k
        ax.plot([p5, p95], [y, y], "-", color="#1f4e9c", lw=1.2)
        ax.plot([p25, p75], [y, y], "-", color="#1f4e9c", lw=4.5, alpha=0.6)
        ax.plot(p50, y, "o", color="k", ms=4)
        ps = np.mean(v > 50); ec = z[f"uec_{u}"]; pf = np.mean(ec[np.isfinite(ec)] < 1500)
        right.append((y, f"P(> 50 %) = {ps:.2f}; P(fresh) = {pf:.2f}"))
        if u == "dune":
            ax.axhspan(y - 0.4, y + 0.4, facecolor=(0.55, 0.55, 0.55, 0.2), edgecolor=(0.3, 0.3, 0.3, 0.5), hatch="x", lw=0)
    ax.axvline(50, color="k", ls="--", lw=0.8)
    ax.set_yticks(range(len(units))); ax.set_yticklabels([nm[u] for u in units[::-1]])
    ax.set_ylim(-0.5, len(units) - 0.5)
    ax.set_xscale("log"); ax.set_xlim(0.3, 120); ax.set_xlabel("% seawater (median; 50 % and 90 % intervals)")
    finish(ax); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.tick_params(axis="y", which="both", right=False)
    ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim()); ax2.set_yticks([y for y, _ in right]); ax2.set_yticklabels([t for _, t in right])
    ax2.tick_params(axis="y", length=0); ax2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for s_ in ax2.spines.values():
        s_.set_linewidth(eps.size("spine"))
    eps.panel_label(ax, "c")
    return fig, dict(rsw=sm["rsw"])


def figS09():
    s18 = mod("s18"); k1 = mod("k1")
    z = np.load(os.path.join(ROOT, "data", "processed", "v1_1", "k1_field_bel.npz")); TX, Pf = z["TX"], z["Pf"]
    v4 = np.load(s18.LIN_V4); x4, b4, r4 = v4["x"], v4["pooled_logmid"], v4["r_logmid"]
    lb = k1.read_lebbe(); ex, ez = wc.land_surface()
    fig = new_fig(96)
    ax = fig.add_subplot()
    ok = (b4[0] >= 0.5) & (r4 <= 10.0)
    ax.axvspan(*k1.DIS_ZONE, color="#e6c229", alpha=0.16, lw=0, zorder=0)
    band(ax, x4, np.where(ok, b4[2], np.nan), np.where(ok, b4[3], np.nan), C["logmid"], 0.18, "--")
    h_lm, = ax.plot(x4, np.where(ok, b4[1], np.nan), color=C["logmid"], lw=1.1, ls="--")
    lo, md, hi = np.percentile(Pf, [5, 50, 95], 0); h = TX >= 130
    band(ax, TX[h], lo[h], hi[h], C["bel"], 0.22, "-"); h_bel, = ax.plot(TX[h], md[h], color=C["bel"], lw=1.7)
    hl = {}
    for key, mk, col in (("saline_base_mTAW", "o", "#b2182b"), ("fresh_top_mTAW", "s", "#2166ac"), ("lower_brackish_top_mTAW", "^", "#8c6d31")):
        v = lb[key]; m = np.isfinite(v)
        hl[key] = ax.errorbar(lb["x_ours_m"][m], v[m], xerr=k1.X_ERR, yerr=k1.Z_ERR, fmt=mk, color=col, ms=4, lw=0.8, capsize=1.5)
    h_clay, = ax.plot(lb["x_ours_m"], lb["clay_top_mTAW"], "-.", color="0.35", lw=1.0)
    h_surf, = ax.plot(ex, ez, "k-", lw=0.9)
    ax.axvline(161, color="steelblue", lw=0.8, ls=":"); ax.text(163, 4.8, "high-water line", fontsize=eps.size("tick"), color="steelblue")
    ax.set_xlim(0, 505); ax.set_ylim(-32, 8); dune_hatch(ax)
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("Elevation (mTAW)"); ve_text(ax, 3); ax.set_aspect(3)
    finish(ax)
    blank = Patch(facecolor="none", edgecolor="none")
    handles = [blank, h_lm, h_bel, Patch(facecolor="#e6c229", alpha=0.35, edgecolor="none"), h_surf,
               blank, hl["saline_base_mTAW"], hl["fresh_top_mTAW"], hl["lower_brackish_top_mTAW"], h_clay]
    labels = ["This study, ERT data of 2018", "λ-pooled log-midpoint (r ≤ 10 m): median, 90 %", "BEL (prior v3): median, 90 %",
              "Discharge zone (this study)", "Land surface (2018 electrodes)", "Lebbe (1981), K1, logs of June–August 1980",
              "Saline base (< 2.5 Ω·m)", "Fresh top (> 20 Ω·m)", "Lower brackish top", "Clay top"]
    legend_below(ax, handles, labels, ncol=2, bbox_to_anchor=(0.5, -0.26))
    trim_height(fig)                                             # v1.9: no empty band above the section
    return fig, {}


FIGURES = [
    ("Fig01_LocationMap", fig01, "F_01_location_map"), ("Fig02_ReferenceModel", fig02, "F3_reference"),
    ("Fig03_LinearisedLensBase", fig03, "F4_linearised"), ("Fig04_SyntheticLinearised", fig04, "F5_synthetic"),
    ("Fig05_FactorialCoverage", fig05, "F_10_coverage_map"), ("Fig06_BELCalibration", fig06, "F6_bel_calibration"),
    ("Fig07_FieldLensBase", fig07, "F7_field_lens_base"), ("Fig08_ProfileVsSoundings", fig08, "F_13_dp_section"),
    ("Fig09_ERT2025AtSoundings", fig09, "F_12b_dp_vs_2025_ert_line"), ("Fig10_Scores", fig10, "F_20_scores"),
    ("Fig11_WaterBelowLens", fig11, "F_19_layout_vs_time"), ("FigS01_Outliers", figS01, "F2_outliers"),
    ("FigS02_Errors", figS02, "F_S10_errors"), ("FigS03_BiasRule", None, "F_11_bias_rule"),
    ("FigS04_CPTLogs", figS04, "F_S11_cpt_logs"), ("FigS05_SoundingsVs2018", figS05, "F_12_dp_vs_2018"),
    ("FigS06_SyntheticVsField", figS06, "F_14_synthetic_vs_field"), ("FigS07_BELWaterBelowLens", figS07, "F_17_bel_synthetic_and_2025_ert_line"),
    ("FigS08_Salinity", figS08, "F8_salinity"), ("FigS09_K1_1980vs2018", figS09, "F_S3_k1_1980_vs_2018"),
    ("FigS10_SyntheticScores", figS10, "F_S13_synthetic_scores"),
]
