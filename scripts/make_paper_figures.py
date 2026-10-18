"""
make_paper_figures.py
=====================
Westhoek ERT v1.8 — redraw the 21 paper figures in the paper style from the saved files of tag v1.7.1 (plotting only).

  python make_paper_figures.py                     all figures → results/figs_style/ (PNG + RGB TIFF at 1000 dpi, PDF,
                                                   300 dpi draft PNG in draft/), then the checks → results/figs_style/checks.json,
                                                   inputs_sha256_after.txt, protected_sha256_{before,after}.txt, contact_sheet.png
  python make_paper_figures.py --only Fig09_ERT2025AtSoundings --draft-only --out <dir>
                                                   quick look: draft PNG only, no checks
  python make_paper_figures.py --corel             v1.10: CorelDRAW versions → Coral Draw/ (SVG, EPS, PDF, preview PNG; one
                                                   font, live text, all-vector data, named groups); results/ is not written
  python make_paper_figures.py --wrr               v1.9: WRR (AGU) export → results/figs_wrr/ under the WRR numbering: TIFF
                                                   (1000 dpi, RGB, LZW) and PDF, 300 dpi draft PNG, figure data data/<label>.csv,
                                                   colour maps, CVD sheet, value comparison with tag v1.8, checks.json

Checks (full run): inputs unchanged (SHA-256 vs inputs_sha256_before.txt of Stage 1); numbers.md, captions.md, tables.md
and all table CSVs unchanged; no file under results/ (except results/figs_style/), data/processed/ or figures/ created,
changed or removed; no number registered by importing the analysis scripts; values of Figs. 3, 7, 9 and 10 against
results/paper/numbers.md and Table 3 of the draft; smallest font ≥ 7 pt and nothing clipped at the figure edge.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import zipfile

os.environ.pop("WESTHOEK_READ_TRACE", None)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
OUT = os.path.join(ROOT, "results", "figs_style")
DRAFT_DOCX = os.path.join(ROOT, "Claude outputs", "Westhoek_paper1_full_draft_v3.docx")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def protected_files():
    fs = [os.path.join(ROOT, "results", "paper", n) for n in ("numbers.md", "captions.md", "tables.md")]
    fs += [p for p in sorted(glob.glob(os.path.join(ROOT, "results", "**", "*.csv"), recursive=True))
           if not rel(p).startswith("results/figs_style/")]
    return fs


def snapshot():
    snap = {}
    for top in ("results", os.path.join("data", "processed"), "figures"):
        for d, dirs, files in os.walk(os.path.join(ROOT, top)):
            r = rel(d)
            if r.startswith("results/figs_style"):
                continue
            snap[r + "/"] = ("dir",)
            for f in files:
                p = os.path.join(d, f); st = os.stat(p)
                snap[rel(p)] = (st.st_size, st.st_mtime_ns)
    return snap


def numbers_md():
    out = {}
    for line in open(os.path.join(ROOT, "results", "paper", "numbers.md"), encoding="utf-8"):
        m = re.match(r"\| \d+ \| `([^`]+)` \| (.*?) \| ", line)
        if m:
            out[m.group(1)] = m.group(2).replace("\\|", "|")
    return out


def table3():
    """Table 3 of the draft (python-docx-free): DP → list of the eight cells after the DP name."""
    x = zipfile.ZipFile(DRAFT_DOCX).read("word/document.xml").decode("utf-8")
    paras = ["".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", q)) for q in re.findall(r"<w:p[ >].*?</w:p>", x, flags=re.S)]
    i = next(k for k, p in enumerate(paras) if p.startswith("Table 3."))
    rows, j = {}, i
    while j < len(paras) and len(rows) < 7:
        if re.fullmatch(r"DP[1-7]", paras[j]):
            rows[paras[j]] = paras[j + 1:j + 9]; j += 9
        else:
            j += 1
    return rows


def fmt3(b, share=False):
    s = f"{b[1]:.2f} [{b[2]:.2f}, {b[3]:.2f}]" + (f" ({100*b[0]:.0f}%)" if share else "")
    return s


def spot_checks(meta, pf):
    N = numbers_md(); out = {}
    s18 = pf.mod("s18")
    if "Fig03_LinearisedLensBase" in meta:
        m = meta["Fig03_LinearisedLensBase"]["pooled_logmid"]
        out["Fig03"] = [dict(key=f"lin_logmid_pooled_x{x}", plotted=fmt3(m[x], True), numbers=N.get(f"lin_logmid_pooled_x{x}"),
                             ok=fmt3(m[x], True) == N.get(f"lin_logmid_pooled_x{x}")) for x in (150, 200, 250, 300, 350)]
    if "Fig07_FieldLensBase" in meta:
        Pf, TX = meta["Fig07_FieldLensBase"]["Pf"], meta["Fig07_FieldLensBase"]["TX"]
        import numpy as np
        out["Fig07"] = []
        for x in (150, 200, 250, 300, 350):
            v = s18.q(Pf[:, int(np.argmin(np.abs(TX - x)))])
            out["Fig07"].append(dict(key=f"bel_base_x{x}", plotted=v, numbers=N.get(f"bel_base_x{x}"), ok=v == N.get(f"bel_base_x{x}")))
    if "Fig09_ERT2025AtSoundings" in meta:
        T3 = table3(); out["Fig09"] = []; out["Fig09_median_notes"] = []
        for nm, v in meta["Fig09_ERT2025AtSoundings"].items():
            for k, col in (("logmid", 5), ("thr3", 6)):
                # Fig. 9 draws the P5–P95 bands only (no median line): the drawn values are P5 and P95
                b = v[k]; t = T3[nm][col]
                lims = f"[{b[2]:.2f}, {b[3]:.2f}]".replace("-", "−"); t_lims = t[t.index("["):]
                out["Fig09"].append(dict(key=f"{nm} {k} P5–P95 (Table 3 col '{('linearised log-midpoint 2025', 'linearised 3 Ω·m 2025 (λ = 50)')[col - 5]}')",
                                         plotted=lims, table3=t_lims, ok=lims == t_lims))
                med = f"{b[1]:.2f}".replace("-", "−"); t_med = t[:t.index(" [")]
                if med != t_med:
                    out["Fig09_median_notes"].append(f"{nm} {k}: band median {med} vs Table 3 central value {t_med} (median not drawn in Fig. 9)")
    if "Fig10_Scores" in meta:
        FS = meta["Fig10_Scores"]; out["Fig10"] = []
        ci = lambda o, k: f"{o[k]['mean']:.2f} [{o[k]['lo']:.2f}, {o[k]['hi']:.2f}]"
        sk = lambda o, k: f"{o[k]['skill']:+.2f} [{o[k]['skill_lo']:+.2f}, {o[k]['skill_hi']:+.2f}]"
        for m in ("BEL3-18", "BEL4-18", "BEL4-18s", "BEL4-24"):
            o = FS[m]
            v = (f"IS {ci(o, 'IS')}, skill {sk(o, 'IS')}; CRPS {ci(o, 'CRPS')}, skill {sk(o, 'CRPS')}; "
                 f"{o['cov']} of {o['n']} inside; sharpness {o['width']:.2f} m")
            key = f"v16_field_{m.replace('-', '_')}"
            out["Fig10"].append(dict(key=key, plotted=v, numbers=N.get(key), ok=v == N.get(key)))
        bel = [FS[m] for m in ("BEL3-18", "BEL4-18", "BEL4-18s", "BEL4-24")]
        v = f"{FS['PRIOR4']['width']:.2f} m (PRIOR4); BEL {min(o['width'] for o in bel):.2f}–{max(o['width'] for o in bel):.2f} m"
        out["Fig10"].append(dict(key="v16_field_prior_sharpness", plotted=v, numbers=N.get("v16_field_prior_sharpness"),
                                 ok=v == N.get("v16_field_prior_sharpness")))
        out["Fig10_summary"] = dict(bel_IS_skill=f"{min(o['IS']['skill'] for o in bel):+.2f} to {max(o['IS']['skill'] for o in bel):+.2f}",
                                    bel_width=f"{min(o['width'] for o in bel):.2f}–{max(o['width'] for o in bel):.2f} m",
                                    prior_width=f"{min(FS[m]['width'] for m in FS if m.startswith('PRIOR')):.2f}–{max(FS[m]['width'] for m in FS if m.startswith('PRIOR')):.2f} m",
                                    bel4_24_inside=f"{FS['BEL4-24']['cov']} of {FS['BEL4-24']['n']}",
                                    v15_bel_l24_base_in=N.get("v15_bel_l24_base_in"))
    return out


def contact_sheet(names, out):
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    W, cols, pad = 900, 3, 30
    tiles = []
    for n in names:
        im = Image.open(os.path.join(OUT, "draft", f"{n}.png")).convert("RGB")
        im = im.resize((W, int(im.size[1] * W / im.size[0])))
        t = Image.new("RGB", (W, im.size[1] + 40), "white"); t.paste(im, (0, 40)); ImageDraw.Draw(t).text((5, 8), n, fill="black")
        tiles.append(t)
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    H = sum(max(t.size[1] for t in r) + pad for r in rows)
    sheet = Image.new("RGB", (cols * (W + pad), H), "white"); y = 0
    for r in rows:
        for i, t in enumerate(r):
            sheet.paste(t, (i * (W + pad), y))
        y += max(t.size[1] for t in r) + pad
    sheet.save(out, dpi=(150, 150))


# ----------------------------------------------------------------------------------------------------------------------
# v1.9: WRR (AGU) export — results/figs_wrr/
# ----------------------------------------------------------------------------------------------------------------------
OUT_WRR = os.path.join(ROOT, "results", "figs_wrr")
WRR_NAMES = [("Figure1", "Fig01"), ("Figure2", "Fig02"), ("Figure3", "Fig04"), ("Figure4", "Fig08"), ("Figure5", "Fig09"),
             ("Figure6", "Fig10"), ("Figure7", "Fig11"), ("FigureS1", "FigS04"), ("FigureS2", "FigS02"), ("FigureS3", "Fig06"),
             ("FigureS4", "Fig03"), ("FigureS5", "FigS01"), ("FigureS6", "Fig05"), ("FigureS7", "FigS03"), ("FigureS8", "Fig07"),
             ("FigureS9", "FigS08"), ("FigureS10", "FigS05"), ("FigureS11", "FigS10"), ("FigureS12", "FigS06"),
             ("FigureS13", "FigS07"), ("FigureS14", "FigS09")]
# Series that v1.8 told apart by colour only, and the second cue added in v1.9 (WRR label, v1.8 figure, panel, pair, cue).
# The colours are those of the figure; "colours" gives the two (or more) hex/named colours compared under CVD below.
CVD_PAIRS = [
    ("Figure3", "Fig04", "a–d", "3 Ω·m vs maximum gradient vs log-midpoint bars", "hatch: //// (3 Ω·m), .... (max. gradient), none (log-midpoint)", ("#D9601A", "0.35", "#1A9E77")),
    ("Figure3", "Fig04", "e–f", "3 Ω·m vs log-midpoint lines (same marker 'o')", "markers: ○ 3 Ω·m, □ log-midpoint", ("#D9601A", "#1A9E77")),
    ("Figure4", "Fig08", "a–b", "BEL vs log-midpoint median lines", "log-midpoint dashed", ("#6A3D9A", "#1A9E77")),
    ("Figure6", "Fig10", "a–c", "method colours (BEL, 3 Ω·m, log-midpoint, prior)", "marker by method family: ○ BEL, △ 3 Ω·m, □ log-midpoint, ▽ prior, ◇ other; field filled, synthetic open", ("#6A3D9A", "#D9601A", "#1A9E77", "0.55")),
    ("Figure7", "Fig11", "a", "2018 data full line (orange step) vs 2025 ERT line (black step); 2018 vs 2025 layout synthetic (dark vs light blue fill)", "2025 ERT line dotted; 2025-layout histogram hatched", ("#d35400", "k", "#1f4e9c", "#7fa6e0")),
    ("Figure7", "Fig11", "b", "2018 full line (orange) vs 2018 beach stretch (steelblue) medians", "beach stretch dash-dot", ("#d35400", "steelblue")),
    ("FigureS1", "FigS04", "all", "fine-grained I_SBT class band vs sand", "fine-grained band hatched (xx)", ("#b7a58a", "white")),
    ("FigureS2", "FigS02", "a", "pairs with dune electrodes (grey) vs other pairs", "dune pairs drawn as ×", ("0.6", "k")),
    ("FigureS4", "Fig03", "b", "3 Ω·m vs log-midpoint median lines", "log-midpoint dashed", ("#D9601A", "#1A9E77")),
    ("FigureS5", "FigS01", "a", "flagged electrodes (red bars) vs others", "flagged bars hatched (////)", ("#c0392b", "0.6")),
    ("FigureS6", "Fig05", "all", "contrast k = 2, 4, 8, 16 (blue shades)", "marker area grows with k (7, 13, 22, 34 pt²)", ("#9ecae1", "#4292c6", "#08519c", "#08306b")),
    ("FigureS7", "FigS03", "a", "contrast k (blue shades)", "marker area grows with k", ("#9ecae1", "#4292c6", "#08519c", "#08306b")),
    ("FigureS7", "FigS03", "b", "λ = 5, 10, 20, 50, 100 (orange shades)", "marker per λ: ▽ □ ◇ ○ △", ("#fdae6b", "#fd8d3c", "#e6550d", "#a63603", "#7f2704")),
    ("FigureS8", "Fig07", "a", "BEL vs log-midpoint median lines", "log-midpoint dashed", ("#6A3D9A", "#1A9E77")),
    ("FigureS10", "FigS05", "all", "log-midpoint median (dashed green) vs 2018 model column (dashed tomato); fine-grained I_SBT strip", "log-midpoint median dash-dot; fine-grained strip hatched", ("#1A9E77", "tomato", "#7a6a55", "#d9c48f")),
    ("FigureS12", "FigS06", "all", "truths T1, T2, T3 (blue, brown, red)", "marker per truth: ○ T1, □ T2, ◇ T3 (filled 3 Ω·m, open log-midpoint); field ⬡", ("#1f4e9c", "#8c6d31", "#b2182b")),
    ("FigureS13", "FigS07", "a", "T1 vs T3 histograms (blue vs orange fill)", "T3 histogram hatched", ("#1f4e9c", "#d35400")),
    ("FigureS14", "FigS09", "all", "BEL vs log-midpoint lines; clay top was dashed", "log-midpoint dashed; clay top dash-dot", ("#6A3D9A", "#1A9E77", "0.35")),
]
CVD_ALREADY = [
    ("Figure5", "Fig09", "log-midpoint band (hatched outline) vs 3 Ω·m band (filled); CPT-EC (solid steelblue) vs 2025 image (dashed dark grey)"),
    ("FigureS8", "Fig07", "panel b: ρ_below classes have distinct line styles (solid/dashed/dash-dot; all ρ_below dotted)"),
    ("FigureS3", "Fig06", "single-colour histograms; P99 and uniform reference lines dashed"),
    ("FigureS9", "FigS08", "P(seawater) and P(fresh) are separate panels (Blues, Oranges)"),
    ("Figure2, S3, S5", "Fig02, Fig06, FigS01", "resistivity: batlow (monotonic lightness, conductive dark); contours r = 10 m solid, r = 15 m dashed"),
]


def all_axes(f):
    out = list(f.axes)
    for a in list(out):                         # inset axes (e.g. the colour bar of Figure 2)
        out += [c for c in getattr(a, "child_axes", []) if c not in out]
    for sf in getattr(f, "subfigs", []):
        out += all_axes(sf)
    return out


def _is_cbar(ax):
    return getattr(ax, "_colorbar", None) is not None or ax.get_label() == "<colorbar>"


def _ax_label(ax, which):
    get = ax.get_xlabel if which == "x" else ax.get_ylabel
    s = get()
    if not s:
        sh = ax.get_shared_x_axes() if which == "x" else ax.get_shared_y_axes()
        for o in sh.get_siblings(ax):
            s = (o.get_xlabel if which == "x" else o.get_ylabel)()
            if s:
                break
    return s.replace("$", "").replace("\\mathbf", "").replace("{", "").replace("}", "").replace("\n", " ")


def figure_rows(fig):
    """Plotted values of every axes (colour bars skipped): lines, filled areas, cells, scatter points, bars, error bars."""
    import numpy as np
    import matplotlib
    from matplotlib.collections import PolyCollection, PathCollection, QuadMesh, LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, Polygon
    from matplotlib.image import AxesImage
    rows, notes = [], []
    g = lambda v: "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{float(v):.6g}"
    axes = [a for a in all_axes(fig) if not _is_cbar(a)]
    for ia, ax in enumerate(axes):
        pl = next((t.get_text().split(")")[0] + ")" for t in ax.texts if re.match(r"^\([a-z]\)", t.get_text())), None)
        panel = pl.strip("()") if pl else f"ax{ia}"
        xl, yl = _ax_label(ax, "x"), _ax_label(ax, "y")
        kind_of = lambda tr: "data" if tr == ax.transData else ("hline" if tr == ax.get_yaxis_transform() else
                                                                ("vline" if tr == ax.get_xaxis_transform() else "other"))

        def add(series, kind, item, x, y, value="", size="", vname=""):
            rows.append([panel, series, kind, str(item), g(x) if x != "" else "", g(y) if y != "" else "", value, size, xl, yl, vname])

        for i, ln in enumerate(ax.lines):
            lab = ln.get_label() or ""; lab = lab if not lab.startswith("_") else f"line{i}"
            xy = np.asarray(ln.get_xydata(), float); k = kind_of(ln.get_transform())
            if k == "hline":
                add(lab, "hline", 0, "", xy[0, 1]); continue
            if k == "vline":
                add(lab, "vline", 0, xy[0, 0], ""); continue
            for j, (x, y) in enumerate(xy):
                if np.isfinite(x) and np.isfinite(y):
                    add(lab, "line", j, x, y)
        for i, c in enumerate(ax.collections):
            lab = c.get_label() or ""; lab = lab if not lab.startswith("_") else f"{type(c).__name__}{i}"
            arr = c.get_array()
            arr = None if arr is None else np.ma.asarray(arr).ravel()
            k = kind_of(c.get_transform()) if not isinstance(c, PathCollection) else kind_of(c.get_offset_transform())
            vname = ""
            if arr is not None and c.norm is not None:
                vname = f"colour value ({type(c.norm).__name__})"
            if isinstance(c, QuadMesh):
                co = c.get_coordinates(); xc = 0.25 * (co[:-1, :-1, 0] + co[1:, :-1, 0] + co[:-1, 1:, 0] + co[1:, 1:, 0])
                yc = 0.25 * (co[:-1, :-1, 1] + co[1:, :-1, 1] + co[:-1, 1:, 1] + co[1:, 1:, 1])
                for j, (x, y, v) in enumerate(zip(xc.ravel(), yc.ravel(), arr)):
                    if v is not np.ma.masked:
                        add(lab, "grid cell (centre)", j, x, y, g(v), "", vname)
                continue
            if isinstance(c, PathCollection):
                off = np.asarray(c.get_offsets(), float); sz = c.get_sizes()
                for j, (x, y) in enumerate(off):
                    v = g(arr[j]) if arr is not None and j < arr.size and arr[j] is not np.ma.masked else ""
                    s_ = g(sz[j % len(sz)]) if len(sz) else ""
                    add(lab, "point", j, x, y, v, s_, vname)
                continue
            paths = c.get_paths()
            if arr is not None and arr.size == len(paths):
                for j, (p, v) in enumerate(zip(paths, arr)):
                    if v is np.ma.masked or len(p.vertices) == 0:
                        continue
                    vv = p.vertices; x, y = vv[:, 0].mean(), vv[:, 1].mean()
                    add(lab, "mesh cell (centroid)" if isinstance(c, PolyCollection) else "path (centroid)", j, x, y, g(v), "", vname)
                continue
            kind = "segment" if isinstance(c, LineCollection) else ("contour" if "Contour" in type(c).__name__ else "area")
            if k == "other":
                notes.append(f"{panel}: {lab} drawn in axes coordinates, not exported"); continue
            if len(paths) > 300:                    # hatch masks / overlay cells: centroids only
                for j, p in enumerate(paths):
                    vv = p.vertices
                    if len(vv) == 0:
                        continue
                    add(lab, kind + " (centroid)", j, vv[:, 0].mean(), vv[:, 1].mean())
                continue
            for j, p in enumerate(paths):
                for q, (x, y) in enumerate(p.vertices):
                    if np.isfinite(x) and np.isfinite(y):
                        add(lab, kind, f"{j}.{q}", x if k != "hline" else "", y if k != "vline" else "")
        for i, p in enumerate(ax.patches):
            lab = p.get_label() or ""; lab = lab if not lab.startswith("_") else f"patch{i}"
            k = kind_of(p.get_transform())
            if isinstance(p, Rectangle):
                x, y, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
                if k == "data":
                    add(lab, "bar (x left, y bottom; value = height; size = width)", i, x, y, g(h), g(w))
                elif k == "vline":
                    add(lab, "x-span (x left; size = width)", i, x, "", "", g(w))
                elif k == "hline":
                    add(lab, "y-span (y bottom; value = height)", i, "", y, g(h))
            elif isinstance(p, Polygon):
                for q, (x, y) in enumerate(p.get_xy()):
                    add(lab, "polygon" if k == "data" else k + " polygon", f"{i}.{q}", x if k != "hline" else "", y if k != "vline" else "")
        for im in ax.images:
            if isinstance(im, AxesImage):
                a_ = im.get_array(); ext = im.get_extent()
                notes.append(f"{panel}: raster image {a_.shape[1]}×{a_.shape[0]} px, extent x {ext[0]:.6g}–{ext[1]:.6g}, "
                             f"y {ext[2]:.6g}–{ext[3]:.6g} (RGB rendering of the sources in the caption; not exported)")
    return rows, notes


def write_figure_csv(fig, label, old, out):
    rows, notes = figure_rows(fig)
    import csv
    hdr = ["panel", "series", "kind", "item", "x", "y", "value", "size", "x_axis (name and unit)", "y_axis (name and unit)",
           "value_name (unit)"]
    p = os.path.join(out, f"{label}.csv")
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(hdr); w.writerows(rows)
    return dict(file=rel(p), rows=len(rows), notes=notes), rows


def load_v18_module(pf):
    """paper_figures.py of tag v1.8, loaded from git as a separate module (for the value comparison only)."""
    import subprocess, types
    src = subprocess.run(["git", "show", "refs/tags/v1.8:scripts/paper_figures.py"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
    m = types.ModuleType("paper_figures_v18"); m.__file__ = os.path.join(SCRIPT_DIR, "paper_figures.py")
    exec(compile(src, m.__file__ + " (v1.8)", "exec"), m.__dict__)
    m._M = pf._M; m.DATA_DIR = pf.DATA_DIR
    return m


def cvd_sheet(labels, draft_dir, out):
    """Normal, deuteranopia, protanopia, tritanopia (colorspacious, severity 100) for every draft, one row per figure."""
    import numpy as np
    from PIL import Image, ImageDraw
    from colorspacious import cspace_convert
    Image.MAX_IMAGE_PIXELS = None
    W, pad, hdr = 700, 16, 34
    kinds = [("normal", None), ("deuteranomaly 100", "deuteranomaly"), ("protanomaly 100", "protanomaly"), ("tritanomaly 100", "tritanomaly")]
    rows = []
    for lab in labels:
        im = Image.open(os.path.join(draft_dir, f"{lab}.png")).convert("RGB")
        im = im.resize((W, int(im.size[1] * W / im.size[0])), Image.LANCZOS)
        a = np.asarray(im, float) / 255.0; tiles = []
        for name, cvd in kinds:
            if cvd is None:
                b = a
            else:
                b = np.clip(cspace_convert(a, {"name": "sRGB1+CVD", "cvd_type": cvd, "severity": 100}, "sRGB1"), 0, 1)
            t = Image.new("RGB", (W, im.size[1] + hdr), "white"); t.paste(Image.fromarray((b * 255).round().astype("uint8")), (0, hdr))
            ImageDraw.Draw(t).text((5, 10), f"{lab} - {name}", fill="black"); tiles.append(t)
        rows.append(tiles)
    H = sum(r[0].size[1] + pad for r in rows)
    sheet = Image.new("RGB", (4 * (W + pad), H), "white"); y = 0
    for r in rows:
        for i, t in enumerate(r):
            sheet.paste(t, (i * (W + pad), y))
        y += r[0].size[1] + pad
    sheet.save(out, dpi=(150, 150))


def cvd_delta_e():
    """ΔE (CAM02-UCS) between the colours of each pair, normal and simulated; < ~10 is hard to tell apart in thin lines."""
    import itertools
    import numpy as np
    from matplotlib.colors import to_rgb
    from colorspacious import cspace_convert
    res = []
    for lab, old, panel, what, cue, cols in CVD_PAIRS:
        rgb = np.array([to_rgb(c) for c in cols])
        d = {}
        for name, cvd in (("normal", None), ("deuteranomaly", "deuteranomaly"), ("protanomaly", "protanomaly"), ("tritanomaly", "tritanomaly")):
            sim = rgb if cvd is None else np.clip(cspace_convert(rgb, {"name": "sRGB1+CVD", "cvd_type": cvd, "severity": 100}, "sRGB1"), 0, 1)
            J = cspace_convert(sim, "sRGB1", "CAM02-UCS")
            d[name] = round(float(min(np.linalg.norm(J[i] - J[j]) for i, j in itertools.combinations(range(len(cols)), 2))), 1)
        res.append(dict(label=lab, old=old, panel=panel, pair=what, cue=cue, colours=list(cols), min_deltaE=d))
    return res


def paper_colours_delta_e():
    import itertools
    import numpy as np
    from matplotlib.colors import to_rgb
    from colorspacious import cspace_convert
    P = {"BEL #6A3D9A": "#6A3D9A", "log-midpoint #1A9E77": "#1A9E77", "3 Ω·m #D9601A": "#D9601A"}
    out = {}
    for (a, ca), (b, cb) in itertools.combinations(P.items(), 2):
        rgb = np.array([to_rgb(ca), to_rgb(cb)]); out[f"{a} vs {b}"] = {}
        for cvd in (None, "deuteranomaly", "protanomaly", "tritanomaly"):
            sim = rgb if cvd is None else np.clip(cspace_convert(rgb, {"name": "sRGB1+CVD", "cvd_type": cvd, "severity": 100}, "sRGB1"), 0, 1)
            J = cspace_convert(sim, "sRGB1", "CAM02-UCS")
            out[f"{a} vs {b}"][cvd or "normal"] = round(float(np.linalg.norm(J[0] - J[1])), 1)
    return out


def wrr_snapshot():
    snap = {}
    for top in ("results", os.path.join("data", "processed"), "figures"):
        for d, dirs, files in os.walk(os.path.join(ROOT, top)):
            r = rel(d)
            if r.startswith("results/figs_wrr"):
                continue
            snap[r + "/"] = ("dir",)
            for f in files:
                p = os.path.join(d, f); st = os.stat(p)
                snap[rel(p)] = (st.st_size, st.st_mtime_ns)
    return snap


def wrr_protected():
    return [p for p in protected_files() if not rel(p).startswith("results/figs_wrr/")] +         [p for p in sorted(glob.glob(os.path.join(OUT, "data", "*.csv")))]


# ----------------------------------------------------------------------------------------------------------------------
# v1.9.1: accessibility audit — WCAG contrast of every series colour against white, CIELAB ΔE of colour pairs per panel
# under normal vision and simulated deuteranopia / protanopia / tritanopia, and the non-colour cue of close pairs
# ----------------------------------------------------------------------------------------------------------------------
# Background traces, exempt from the 3:1 rule (they carry no information on their own; listed in SUMMARY_v1.9.1.md)
A11Y_BACKGROUND = {"CPT-EC log, 1 cm": "background trace; the 0.5 m median carries the information"}
CVD_TYPES = (("deut", "deuteranomaly"), ("prot", "protanomaly"), ("trit", "tritanomaly"))


def _rel_lum(rgb):
    import numpy as np
    c = np.asarray(rgb, float)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def wcag_contrast(rgb, bg=(1.0, 1.0, 1.0)):
    a, b = _rel_lum(rgb), _rel_lum(bg)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def _over_white(rgba):
    r, g, b, a = rgba
    return (a * r + 1 - a, a * g + 1 - a, a * b + 1 - a)


def _hex(rgb):
    return "#%02X%02X%02X" % tuple(int(round(255 * min(max(v, 0), 1))) for v in rgb)


def lab_delta_e(c1, c2):
    """CIELAB ΔE (CIE76, D65) between two sRGB colours: normal, and under the three simulations (severity 100)."""
    import numpy as np
    from colorspacious import cspace_convert
    rgb = np.array([c1, c2], float); out = {}
    for key, cvd in (("normal", None),) + CVD_TYPES:
        sim = rgb if cvd is None else np.clip(cspace_convert(rgb, {"name": "sRGB1+CVD", "cvd_type": cvd, "severity": 100}, "sRGB1"), 0, 1)
        L = cspace_convert(sim, "sRGB1", "CIELab")
        out[key] = round(float(np.linalg.norm(L[0] - L[1])), 1)
    return out


def _gid_exempt(a):
    g = a.get_gid() or ""
    return g[5:] if g.startswith("a11y:") else ""


def _casing(ln):
    """Dark casing drawn by a path effect (Stroke / withStroke with a dark foreground): the line's boundary is then the
    casing, whose contrast counts."""
    for p in ln.get_path_effects() or []:
        fg = getattr(p, "_gc", {}).get("foreground")
        if fg is not None:
            from matplotlib.colors import to_rgba
            if wcag_contrast(_over_white(to_rgba(fg))) >= 3:
                return True
    return False


def a11y_entries(fig):
    """Every series colour drawn in the figure: one entry per (panel, artist, role) with the colour composited over white,
    its WCAG contrast, a style signature (line style, marker, hatch, fill) and whether it is exempt (colour-mapped cells,
    background spans, light band fills with drawn edges, listed background traces)."""
    import numpy as np
    from matplotlib.collections import PathCollection, PolyCollection, Collection, LineCollection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, Polygon, Patch as _P
    out = []
    axes = [a for a in all_axes(fig) if not _is_cbar(a) and a.get_visible()]
    for ia, ax in enumerate(axes):
        pl = next((t.get_text().split(")")[0] + ")" for t in ax.texts if re.match(r"^\([a-z]\)", t.get_text())), None)
        panel = pl if pl else f"ax{ia}"
        blended = lambda tr: tr != ax.transData and tr in (ax.get_yaxis_transform(), ax.get_xaxis_transform())

        def add(label, role, rgba, style, exempt="", against=None):
            if rgba is None:
                return
            rgba = to_rgba(rgba)
            if rgba[3] == 0:
                return
            rgb = _over_white(rgba)
            if min(rgb) > 0.985:                   # white (halo, frame fill): not a series colour
                return
            bg = (1.0, 1.0, 1.0) if against is None else _over_white(to_rgba(against))
            out.append(dict(panel=panel, label=label, role=role, hex=_hex(rgb), opaque_hex=_hex(rgba[:3]), alpha=round(rgba[3], 2),
                            contrast=round(wcag_contrast(rgb, bg), 2), against="white" if against is None else _hex(bg),
                            style=style, exempt=exempt))
        for i, ln in enumerate(ax.lines):
            if not ln.get_visible():
                continue
            lab = ln.get_label() or ""; lab = lab if not lab.startswith("_") else f"line{i}"
            ex = A11Y_BACKGROUND.get(lab, "") or _gid_exempt(ln)
            if ex == "" and to_rgba(ln.get_color())[:3] == to_rgba("0.78")[:3] and len(ln.get_xdata()) > 200:
                ex = "background trace (1 cm log)"
            if ex == "" and _casing(ln):
                ex = "outlined (dark casing)"
            a = ln.get_alpha(); a = 1.0 if a is None else a
            ls = ln.get_linestyle(); mk = ln.get_marker()
            sty = f"line {ls}" if ls not in ("None", "", " ", "none") and ln.get_linewidth() > 0 else ""
            if mk not in (None, "None", "", " ", "none"):
                filled = ln.get_markerfacecolor() not in ("none", "None") and to_rgba(ln.get_markerfacecolor())[3] > 0
                sty += f" marker {mk}{'' if filled else ' open'}"
            sty = sty.strip()
            if "line" in sty:
                add(lab, "line", to_rgba(ln.get_color(), a), sty, ex)
            if "marker" in sty:
                mfc, mec = ln.get_markerfacecolor(), ln.get_markeredgecolor()
                if mfc not in ("none", "None"):
                    edge_ok = ln.get_markeredgewidth() > 0 and wcag_contrast(_over_white(to_rgba(mec, a))) >= 3
                    add(lab, "marker face", to_rgba(mfc, a), sty, ex or ("outlined (edge ≥ 3:1)" if edge_ok else ""))
                if ln.get_markeredgewidth() > 0:
                    add(lab, "marker edge", to_rgba(mec, a), sty, ex)
        for i, c in enumerate(ax.collections):
            if not c.get_visible():
                continue
            lab = c.get_label() or ""; lab = lab if not lab.startswith("_") else f"{type(c).__name__}{i}"
            if c.get_array() is not None and c.norm is not None and not (isinstance(c, PathCollection) and False):
                if not isinstance(c, PathCollection):
                    continue                       # colour-mapped cells / meshes: colour map, not a series colour
            try:
                tr_bl = blended(c.get_transform())
            except Exception:
                tr_bl = False
            h = c.get_hatch()
            if isinstance(c, PathCollection):
                p0 = c.get_paths()[0].vertices if len(c.get_paths()) else np.zeros((0, 2))
                mk = f"path{len(p0)}"
                fc = c.get_facecolors(); ec = c.get_edgecolors(); lw = c.get_linewidths()
                filled = len(fc) > 0 and fc[:, 3].max() > 0
                sty = f"marker {mk}{'' if filled else ' open'} size {','.join(sorted({f'{s:g}' for s in np.round(c.get_sizes(), 0)}))}"
                edge_ok = len(lw) and max(lw) > 0 and len(ec) and min(wcag_contrast(_over_white(e)) for e in ec) >= 3
                for f in {tuple(np.round(x, 4)) for x in fc}:
                    add(lab, "marker face", f, sty, _gid_exempt(c) or ("outlined (edge ≥ 3:1)" if edge_ok else ""))
                if len(lw) and max(lw) > 0:
                    for e in {tuple(np.round(x, 4)) for x in ec}:
                        add(lab, "marker edge", e, sty)
                continue
            if isinstance(c, LineCollection):
                ls = c.get_linestyle()[0] if c.get_linestyle() else "-"
                for e in {tuple(np.round(x, 4)) for x in c.get_colors()}:
                    add(lab, "line", e, f"segments {ls}")
                continue
            fc = c.get_facecolors(); ec = c.get_edgecolors(); lw = c.get_linewidths()
            has_edge = len(ec) and len(lw) and max(lw) > 0 and ec[:, 3].max() > 0
            els = c.get_linestyle()[0] if has_edge and c.get_linestyle() else None
            sty = f"area{' hatch ' + h if h else ''}{' edge ' + str(els) if has_edge else ''}"
            for f in {tuple(np.round(x, 4)) for x in fc}:
                ex = _gid_exempt(c) or ("background span" if tr_bl else ("light band fill (edges drawn)" if f[3] < 1 and has_edge else
                                                                        ("light band fill" if f[3] < 1 else "")))
                add(lab, "area fill", f, sty, ex)
            if has_edge:
                for e in {tuple(np.round(x, 4)) for x in ec}:
                    add(lab, "area edge", e, sty, "background span" if tr_bl else "")
        for i, p in enumerate(ax.patches):
            if not p.get_visible():
                continue
            lab = p.get_label() or ""; lab = lab if not lab.startswith("_") else f"patch{i}"
            bl = blended(p.get_data_transform())
            h = p.get_hatch(); fc = p.get_facecolor(); ec = p.get_edgecolor(); lw = p.get_linewidth()
            kind = "bar" if isinstance(p, Rectangle) and not bl else ("span" if bl else "patch")
            sty = f"{kind}{' hatch ' + h if h else ''}"
            edge_dark = lw > 0 and ec[3] > 0 and wcag_contrast(_over_white(ec)) >= 3
            forced = (p.get_gid() or "").startswith("a11y-series:")       # data drawn as a span (CPT class strips)
            bl = bl and not forced
            ex = "background span" if bl else ("light fill" if fc[3] < 1 and not edge_dark else "")
            if fc[3] > 0:
                add(lab, f"{kind} fill", fc, sty, ex or ("outlined (edge ≥ 3:1)" if edge_dark and wcag_contrast(_over_white(fc)) < 3 else ""))
            if (lw > 0 and ec[3] > 0) or h:
                opaque = fc[3] >= 0.99
                add(lab, f"{kind} {'hatch' if h else 'edge'}", ec, sty, "background span" if bl else "",
                    against=fc if (h and opaque) else None)
    return out


def a11y_pairs(entries):
    """Per panel: every pair of distinct series colours (exempt entries left out), CIELAB ΔE normal / deut / prot / trit,
    and the non-colour cue between the series that use them (style signatures differ)."""
    import itertools
    from matplotlib.colors import to_rgb
    by_panel = {}
    for e in entries:
        if e["exempt"]:
            continue
        by_panel.setdefault(e["panel"], {}).setdefault(e["hex"], set()).add((e["label"], e["style"]))
    res = []
    cache = {}
    for panel, cols in by_panel.items():
        for c1, c2 in itertools.combinations(sorted(cols), 2):
            key = (c1, c2)
            if key not in cache:
                cache[key] = lab_delta_e(to_rgb(c1), to_rgb(c2))
            d = cache[key]
            if {l for l, _ in cols[c1]} & {l for l, _ in cols[c2]}:
                continue                           # two colours of one series (e.g. a bar and its edge)
            s1 = {s for _, s in cols[c1]}; s2 = {s for _, s in cols[c2]}
            cue = "" if (s1 - s2) or (s2 - s1) else "NONE"
            if not cue:
                cue = "; ".join(sorted(s1 - s2))[:60] + "  vs  " + "; ".join(sorted(s2 - s1))[:60]
            res.append(dict(panel=panel, c1=c1, c2=c2, dE=d, min_dE=min(d.values()),
                            series1=sorted({l for l, _ in cols[c1]})[:3], series2=sorted({l for l, _ in cols[c2]})[:3], cue=cue))
    return res


def wrr_main(a):
    import numpy as np
    out = OUT_WRR
    for d in (out, os.path.join(out, "draft"), os.path.join(out, "data")):
        os.makedirs(d, exist_ok=True)
    inputs = [l.split("  ", 1)[1] for l in open(os.path.join(OUT, "inputs_sha256_before.txt"), encoding="utf-8").read().splitlines() if l]
    if not a.only:
        with open(os.path.join(out, "inputs_sha256_before.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines(f"{sha(os.path.join(ROOT, p))}  {p}\n" for p in inputs)
        with open(os.path.join(out, "protected_sha256_before.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines(f"{sha(p)}  {rel(p)}\n" for p in wrr_protected())
        style_before = {rel(p): sha(p) for p in sorted(glob.glob(os.path.join(OUT, "**", "*"), recursive=True)) if os.path.isfile(p)}
    snap0 = wrr_snapshot()
    t_all = time.time()
    import matplotlib.pyplot as plt
    import paper_figures as pf
    import ert_plot_style as eps
    pf.DATA_DIR = os.path.join(OUT, "data")           # v1.8 stored arrays (FigS01 lens bases), read only
    old_mod = None if a.no_compare else load_v18_module(pf)
    byname = {n.split("_")[0]: (n, fn, old) for n, fn, old in pf.FIGURES}
    only = set(a.only.split(",")) if a.only else None
    res, meta, data_info, cmaps, compare, a11y = {}, {}, {}, {}, {}, {}
    s03 = None
    for label, short in WRR_NAMES:
        if only and label not in only:
            continue
        name, fn, old = byname[short]; t0 = time.time()

        def build(module):
            if name == "FigS03_BiasRule":
                return module.figS03(*s03[:3])
            if name == "Fig09_ERT2025AtSoundings":
                return module.fig09(img_col=a.fig9_color)
            return getattr(module, fn.__name__)()
        if name == "FigS03_BiasRule" and s03 is None:
            s03 = pf.figS03_data(OUT)              # re-run + exact check against v13_summary.json; compares FigS03_points.csv, no write
        fig, m = build(pf)
        if name == "FigS03_BiasRule":
            m = dict(fit_check=s03[3])
        cmaps[label] = eps.colormaps_in(fig)
        ov = eps.overlap_checks(fig)
        ae = a11y_entries(fig); ap = a11y_pairs(ae)
        a11y[label] = dict(entries=ae, pairs=ap)
        low = sorted({(e["hex"], e["contrast"], e["label"][:30]) for e in ae if not e["exempt"] and e["contrast"] < 3})
        close = [(p["panel"], p["c1"], p["c2"], p["min_dE"], p["cue"]) for p in ap if p["min_dE"] < 10]
        info, rows_new = write_figure_csv(fig, label, short, os.path.join(out, "data"))
        data_info[label] = info
        res_cells = None
        if short in ("Fig02", "Fig06", "FigS01"):
            res_cells = [np.ma.copy(c.get_array()) for ax in all_axes(fig) if not _is_cbar(ax) for c in ax.collections
                         if c.get_array() is not None and c.norm is not None and type(c.norm).__name__ == "LogNorm"]
        if a.draft_only:
            smallest, clipped = eps.layout_checks(fig); files = []
            fig.savefig(os.path.join(out, "draft", f"{label}.png"), dpi=300)
        else:
            files, smallest, clipped = eps.savefig(fig, label, out, 190, formats=("tif", "pdf"))
        w, h = fig.get_size_inches() * 25.4
        plt.close(fig)
        if old_mod is not None:
            # Inline contour labels cut gaps into the contour paths that depend on the figure size (rendering, not data):
            # for the comparison both versions are rebuilt with clabel switched off.
            import matplotlib.axes
            n_clabel = sum(len(getattr(c, "labelTexts", []) or []) for ax in all_axes(fig) for c in ax.collections)
            if n_clabel:
                keep = matplotlib.axes.Axes.clabel; matplotlib.axes.Axes.clabel = lambda self, *q, **k: []
                try:
                    fn_, _ = build(pf); rows_new, _ = figure_rows(fn_); plt.close(fn_)
                    fo, _ = build(old_mod)
                finally:
                    matplotlib.axes.Axes.clabel = keep
            else:
                fo, _ = build(old_mod)
            rows_old, _ = figure_rows(fo)
            strip = lambda R_: [r_[:7] + r_[8:] for r_ in R_]       # marker size (column "size") is a v1.9 cue, compared separately
            cmp = dict(rows_equal=strip(rows_old) == strip(rows_new), sizes_equal=rows_old == rows_new, n_rows=len(rows_new), cmaps_v18=eps.colormaps_in(fo),
                       contour_labels_off_for_comparison=bool(n_clabel))
            if not cmp["rows_equal"]:
                cmp["n_diff"] = sum(1 for x, y in zip(strip(rows_old), strip(rows_new)) if x != y) + abs(len(rows_old) - len(rows_new))
                cmp["first_diff"] = next(([x, y] for x, y in zip(strip(rows_old), strip(rows_new)) if x != y), None)
            if res_cells is not None:
                oc = [np.ma.copy(c.get_array()) for ax in all_axes(fo) if not _is_cbar(ax) for c in ax.collections
                      if c.get_array() is not None and c.norm is not None and type(c.norm).__name__ == "LogNorm"]
                cmp["res_cells"] = dict(n_layers=len(res_cells), n_cells=int(sum(x.size for x in res_cells)),
                                        identical=len(oc) == len(res_cells) and all(                       # NaN cells (outside the mesh) count as equal
                                            np.array_equal(np.ma.getdata(x), np.ma.getdata(y), equal_nan=True)
                                            and np.array_equal(np.ma.getmaskarray(x), np.ma.getmaskarray(y)) for x, y in zip(oc, res_cells)))
            plt.close(fo)
            compare[label] = cmp
        res[label] = dict(v18=name, old=old, size_mm=[round(w, 1), round(h, 1)], smallest_pt=smallest, clipped=clipped, overlaps=ov,
                          seconds=round(time.time() - t0, 1), files=[rel(p) for p in files],
                          bytes={os.path.splitext(p)[1] + ("(draft)" if "draft" in p else ""): os.path.getsize(p) for p in files})
        meta[name] = m
        print(f"[fig] {label} ({short}): {w:.0f}×{h:.0f} mm, smallest {smallest:.2f} pt, clipped {len(clipped)}, cmaps {cmaps[label]}, "
              f"data {info['rows']} rows, same data as v1.8: {compare.get(label, {}).get('rows_equal')}, "
              f"legends {ov['n_legends']} / letters {ov['n_letters']}, overlaps {len(ov['legend_overlaps'])} + {len(ov['letter_overlaps'])}, "
              f"{time.time() - t0:.0f} s", flush=True)
        for x in ov["legend_overlaps"] + ov["letter_overlaps"]:
            print("   overlap:", x, flush=True)
        for x in low:
            print("   contrast < 3:", x, flush=True)
        for x in close:
            print("   ΔE < 10:", x, flush=True)
        if label in compare and not compare[label]["rows_equal"]:
            print("   first diff:", compare[label].get("n_diff"), compare[label].get("first_diff"), flush=True)
    if only:
        return
    checks = dict(figures=res, seconds=round(time.time() - t_all), colormaps=cmaps, data=data_info, compare_v18=compare)
    json.dump(a11y, open(os.path.join(out, "a11y.json"), "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    checks["a11y_low_contrast"] = sorted({(l, e["hex"], e["contrast"], e["label"]) for l, v in a11y.items() for e in v["entries"]
                                          if not e["exempt"] and e["contrast"] < 3})
    checks["a11y_pairs_without_cue"] = [(l, p["panel"], p["c1"], p["c2"], p["min_dE"]) for l, v in a11y.items() for p in v["pairs"]
                                        if p["min_dE"] < 10 and p["cue"] == "NONE"]
    checks["banned_colormaps_found"] = sorted({c for v in cmaps.values() for c in v if any(c.lower().startswith(b) for b in eps.BANNED_CMAPS)
                                               or c.lower().replace("cmc.", "") in ("jet", "rainbow", "turbo", "hsv", "nipy_spectral")})
    after = [(sha(os.path.join(ROOT, p)), p) for p in inputs]
    with open(os.path.join(out, "inputs_sha256_after.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{h}  {p}\n" for h, p in after)
    checks["inputs_identical"] = (open(os.path.join(out, "inputs_sha256_before.txt"), encoding="utf-8").read()
                                  == open(os.path.join(out, "inputs_sha256_after.txt"), encoding="utf-8").read())
    checks["inputs_identical_to_v18_before"] = (open(os.path.join(OUT, "inputs_sha256_before.txt"), encoding="utf-8").read()
                                                == open(os.path.join(out, "inputs_sha256_after.txt"), encoding="utf-8").read())
    checks["n_inputs"] = len(inputs)
    with open(os.path.join(out, "protected_sha256_after.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{sha(p)}  {rel(p)}\n" for p in wrr_protected())
    checks["protected_identical"] = (open(os.path.join(out, "protected_sha256_before.txt"), encoding="utf-8").read()
                                     == open(os.path.join(out, "protected_sha256_after.txt"), encoding="utf-8").read())
    checks["n_protected"] = len(wrr_protected())
    style_after = {rel(p): sha(p) for p in sorted(glob.glob(os.path.join(OUT, "**", "*"), recursive=True)) if os.path.isfile(p)}
    checks["figs_style_unchanged"] = style_before == style_after
    checks["n_figs_style_files"] = len(style_after)
    snap1 = wrr_snapshot()
    checks["side_effects"] = dict(created=sorted(set(snap1) - set(snap0)), removed=sorted(set(snap0) - set(snap1)),
                                  changed=sorted(k for k in snap0 if k in snap1 and snap0[k] != snap1[k]))
    checks["numbers_registered_by_imports"] = len(pf.mod("s18").NUM.rows)
    checks["spot"] = spot_checks(meta, pf)
    checks["figS03_fit"] = meta.get("FigS03_BiasRule", {}).get("fit_check")
    checks["cvd_pairs"] = cvd_delta_e(); checks["cvd_already_distinct"] = CVD_ALREADY
    checks["paper_colours_deltaE"] = paper_colours_delta_e()
    labels = [l for l, _ in WRR_NAMES]
    cvd_sheet(labels, os.path.join(out, "draft"), os.path.join(out, "cvd_check.png"))
    global_out = OUT
    try:
        globals()["OUT"] = out
        contact_sheet(labels, os.path.join(out, "contact_sheet.png"))
    finally:
        globals()["OUT"] = global_out
    json.dump(checks, open(os.path.join(out, "checks.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    ok = (checks["inputs_identical"] and checks["protected_identical"] and checks["figs_style_unchanged"]
          and not any(checks["side_effects"].values()) and checks["numbers_registered_by_imports"] == 0
          and not checks["banned_colormaps_found"]
          and all(c["rows_equal"] and c.get("res_cells", {"identical": True})["identical"] for c in compare.values())
          and all(r["ok"] for k, v in checks["spot"].items() if isinstance(v, list) for r in v if isinstance(r, dict))
          and all(r["smallest_pt"] >= 7.0 - 1e-9 and not r["clipped"] for r in res.values())
          and all(not r["overlaps"]["legend_overlaps"] and not r["overlaps"]["letter_overlaps"] for r in res.values())
          and not checks["a11y_low_contrast"] and not checks["a11y_pairs_without_cue"])
    checks["n_legend_overlaps"] = sum(len(r["overlaps"]["legend_overlaps"]) for r in res.values())
    checks["n_letter_overlaps"] = sum(len(r["overlaps"]["letter_overlaps"]) for r in res.values())
    json.dump(checks, open(os.path.join(out, "checks.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"[checks] all passed: {ok}  ({checks['seconds']} s)")



# ----------------------------------------------------------------------------------------------------------------------
# v1.10: font inventory (which font draws each glyph of each text) and text roles
# ----------------------------------------------------------------------------------------------------------------------
_CMAPS = {}


def _cmap(path):
    if path not in _CMAPS:
        from matplotlib.ft2font import FT2Font
        f = FT2Font(path)
        _CMAPS[path] = (f.family_name + (" " + f.style_name if f.style_name not in ("Regular",) else ""), set(f.get_charmap()))
    return _CMAPS[path]


def drawn_texts(fig):
    """ids of the Text artists that are actually drawn (Text.draw is recorded during one render)."""
    from matplotlib.text import Text
    seen = set(); orig = Text.draw

    def rec(self, renderer):
        if self.get_visible() and self.get_text().strip():
            seen.add(id(self))
        return orig(self, renderer)
    Text.draw = rec
    try:
        fig.canvas.draw()
    finally:
        Text.draw = orig
    return seen


def text_role(t, fig):
    """Role of a text artist: panel letter, axis label, tick label, colour-bar label / tick, legend, annotation."""
    import ert_plot_style as eps
    gid = t.get_gid() or ""
    if gid in ("panel_label", "panel_ident") or gid.startswith("panel_letter"):
        return "panel letter / identifier"
    ax = t.axes
    for a in all_axes(fig):
        cb = _is_cbar(a)
        if t is a.xaxis.label or t is a.yaxis.label:
            return "colour-bar label" if cb else "axis label"
        for axis in (a.xaxis, a.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if t is tk.label1 or t is tk.label2:
                    return "colour-bar tick" if cb else "tick label"
    for lg in eps._legends(fig):
        if t in lg.get_texts() or t is lg.get_title():
            return "legend"
    return "annotation"


def text_inventory(fig):
    """Every drawn, non-empty text: role, string, size (pt), and the font of each glyph as matplotlib resolves it
    (plain text: first font of the family fallback list whose character map has the glyph; mathtext: the parser's glyph
    fonts)."""
    from matplotlib import font_manager
    from matplotlib.mathtext import MathTextParser
    from matplotlib.text import Text
    fig.canvas.draw()
    drawn = set()
    for a in all_axes(fig):
        for axis in (a.xaxis, a.yaxis):
            for tk in axis._update_ticks():
                drawn.update((id(tk.label1), id(tk.label2)))
    tick_all = set()
    for a in all_axes(fig):
        for axis in (a.xaxis, a.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_all.update((id(tk.label1), id(tk.label2)))
    mp_ = MathTextParser("path")
    out = []
    shown = drawn_texts(fig)
    for t in fig.findobj(Text):
        s = t.get_text()
        if id(t) not in shown:
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        prop = t.get_fontproperties()
        s2, ismath = t._preprocess_math(s)
        fonts = {}
        if ismath:
            try:
                vp = mp_.parse(s2, 72, prop)
                for font, fsize, num, ox, oy in vp.glyphs:
                    nm = font.family_name + ("" if font.style_name == "Regular" else " " + font.style_name)
                    fonts.setdefault(nm, []).append(chr(num))
            except Exception as e:
                fonts.setdefault(f"(parse error {e})", []).append(s2)
        else:
            paths = font_manager.fontManager._find_fonts_by_props(prop)
            for ch in s2:
                if ch in "\n\t ":
                    continue
                nm = None
                for p in paths:
                    fam, cm = _cmap(str(p))
                    if ord(ch) in cm:
                        nm = fam; break
                fonts.setdefault(nm or "(missing glyph)", []).append(ch)
        out.append(dict(role=text_role(t, fig), text=s, size=round(t.get_fontsize(), 2), weight=str(t.get_fontweight()),
                        style=t.get_fontstyle(), math=bool(ismath), fonts={k: "".join(v) for k, v in fonts.items()}))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# v1.10: CorelDRAW versions → "Coral Draw/" (SVG, EPS, PDF, 300 dpi preview): one font, live text, all-vector data,
# named groups, clip paths only where data leave the axes
# ----------------------------------------------------------------------------------------------------------------------
OUT_COREL = os.path.join(ROOT, "Coral Draw")
IMG_DPI = 600                    # embedded map imagery of Figure 1 (orthophoto, relief) at print size


def text_overlaps(fig, shrink_px=0.5):
    """Pairs of drawn texts whose boxes overlap (texts of one legend excluded; the legend lays them out)."""
    import ert_plot_style as eps
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    drawn, tick_all = set(), set()
    for a in all_axes(fig):
        for axis in (a.xaxis, a.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_all.update((id(tk.label1), id(tk.label2)))
            for tk in axis._update_ticks():
                drawn.update((id(tk.label1), id(tk.label2)))
    legs = eps._legends(fig)
    in_leg = {id(t): i for i, lg in enumerate(legs) for t in list(lg.get_texts()) + [lg.get_title()]}
    T = []
    shown = drawn_texts(fig)
    for t in fig.findobj(Text):
        if id(t) not in shown:
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        try:
            b = t.get_window_extent(r)
        except Exception:
            continue
        if b.width <= 0:
            continue
        T.append((t, Bbox.from_extents(b.x0 + shrink_px, b.y0 + shrink_px, b.x1 - shrink_px, b.y1 - shrink_px)))
    hits = []
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            (t1, b1), (t2, b2) = T[i], T[j]
            if id(t1) in in_leg and in_leg.get(id(t1)) == in_leg.get(id(t2)):
                continue
            if b1.overlaps(b2):
                hits.append(f"{t1.get_text()[:25]!r} / {t2.get_text()[:25]!r}")
    return hits


def split_translucent_hatches(fig):
    """A translucent hatched area is written to SVG as a pattern whose background carries the fill; CorelDRAW makes that
    background opaque. Split each such patch / collection into the translucent fill (no hatch) and a hatch-only copy on
    top, so both survive the import. Opaque hatched fills (bars, strips) are left as they are. Returns the count."""
    from matplotlib.patches import PathPatch
    from matplotlib.collections import PolyCollection
    n = 0
    for ax in all_axes(fig):
        for p in list(ax.patches):
            h = p.get_hatch(); fc = p.get_facecolor()
            if h and 0 < fc[3] < 1:
                hc = getattr(p, "_hatch_color", (0, 0, 0, 1))
                q = PathPatch(p.get_path(), transform=p.get_transform(), facecolor="none", edgecolor=hc, linewidth=0,
                              hatch=h, zorder=p.get_zorder(), clip_on=p.get_clip_on(), label="_hatch")
                q._hatch_color = hc
                if p.get_clip_path() is not None:
                    q.set_clip_path(p.get_clip_path())
                p.set_hatch(None); ax.add_artist(q); n += 1
        for c in list(ax.collections):
            if type(c).__name__ not in ("PolyCollection", "FillBetweenPolyCollection") or not c.get_hatch():
                continue
            fc = c.get_facecolors()
            if len(fc) and (0 < fc[:, 3]).any() and (fc[:, 3] < 1).any():
                hc = getattr(c, "_hatch_color", (0, 0, 0, 1))
                q = PolyCollection([pp.vertices for pp in c.get_paths()], facecolors="none", edgecolors=[hc], linewidths=0,
                                   hatch=c.get_hatch(), zorder=c.get_zorder(), transform=c.get_transform(), label="_hatch")
                q._hatch_color = hc
                c.set_hatch(None); ax.add_collection(q, autolim=False); n += 1
    return n


def vectorize(fig):
    """No rasterised layer: switch rasterisation off for every artist (colour-bar solids included). Returns the count."""
    n = 0
    for a in fig.findobj():
        if hasattr(a, "get_rasterized") and a.get_rasterized() and type(a).__name__ != "AxesImage":
            a.set_rasterized(False); n += 1
    return n


def cut_polygons_to_axes(ax, c):
    """Cut every polygon of a data-space PolyCollection to the axes view rectangle (linear axes only), so that the layer
    needs no clip path. Values stay attached to their cells; cells entirely outside the view are dropped (they were not
    visible). Only the drawing geometry changes; the plotted values were checked before."""
    import numpy as np
    from matplotlib.transforms import Bbox
    if ax.get_xscale() != "linear" or ax.get_yscale() != "linear" or c.get_transform() != ax.transData:
        return False
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    bb = Bbox.from_extents(x0, y0, x1, y1)
    arr = c.get_array(); paths = c.get_paths()
    keep_p, keep_i = [], []
    for i, p in enumerate(paths):
        v = p.vertices
        if len(v) == 0:
            continue
        if v[:, 0].min() >= x0 and v[:, 0].max() <= x1 and v[:, 1].min() >= y0 and v[:, 1].max() <= y1:
            keep_p.append(p); keep_i.append(i); continue
        q = p.clip_to_bbox(bb, inside=True)
        if len(q.vertices) >= 3:
            keep_p.append(q); keep_i.append(i)
    c.set_verts([q.vertices for q in keep_p])          # PolyCollection takes vertex arrays (not Path objects)
    if arr is not None:
        a = np.ma.asarray(arr).ravel()
        if a.size == len(paths):
            c.set_array(a[keep_i])
    fc = c.get_facecolors()
    if c.get_array() is None and len(fc) == len(paths) and len(paths) > 1:
        c.set_facecolor(fc[keep_i])
    return True


def _view_box(ax):
    from shapely.geometry import box
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    return box(x0, y0, x1, y1)


def _linear_data(ax, a):
    return ax.get_xscale() == "linear" and ax.get_yscale() == "linear" and a.get_transform() == ax.transData


def _custom_clip(ax, a):
    cp = a.get_clip_path()
    return cp is not None and getattr(cp, "_patch", None) is not ax.patch


def quadmesh_to_cells(ax, qm):
    """Replace a QuadMesh by a PolyCollection of its unmasked cells, cut to the axes view and to a custom clip polygon
    (if any), with the same colour map, norm, values, z-order and edges, so that no clip path is needed."""
    import numpy as np
    from matplotlib.collections import PolyCollection
    from shapely.geometry import Polygon as SP
    from shapely.prepared import prep
    if ax.get_xscale() != "linear" or ax.get_yscale() != "linear":
        return False
    region = _view_box(ax)
    cp = qm.get_clip_path()
    if _custom_clip(ax, qm):
        patch = getattr(cp, "_patch", None)
        if patch is None or not hasattr(patch, "get_xy"):
            return False
        region = region.intersection(SP(patch.get_xy()))
    pr = prep(region)
    co = qm.get_coordinates(); arr = np.ma.asarray(qm.get_array())
    ny, nx = co.shape[0] - 1, co.shape[1] - 1
    vals = arr.reshape(ny, nx)
    polys, keep = [], []
    for j in range(ny):
        for i in range(nx):
            v = vals[j, i]
            if v is np.ma.masked:
                continue
            q = [co[j, i], co[j, i + 1], co[j + 1, i + 1], co[j + 1, i]]
            g = SP(q)
            if pr.contains(g):
                polys.append(np.asarray(q)); keep.append(v)
            elif pr.intersects(g):
                h = g.intersection(region)
                for part in getattr(h, "geoms", [h]):
                    if part.geom_type == "Polygon" and part.area > 0:
                        polys.append(np.asarray(part.exterior.coords)); keep.append(v)
    pc = PolyCollection(polys, array=np.asarray(keep, float), cmap=qm.get_cmap(), norm=qm.norm, edgecolors="face",
                        linewidths=qm.get_linewidth(), antialiased=True, zorder=qm.get_zorder(), clip_on=False)
    pc.set_label(qm.get_label()); pc.set_gid(qm.get_gid())
    qm.remove(); ax.add_collection(pc, autolim=False)
    return True


def cut_lines(ax, a):
    """Cut a LineCollection or a marker-less Line2D at the axes view (shapely), so that it needs no clip path."""
    import numpy as np
    from shapely.geometry import LineString
    region = _view_box(ax)
    if type(a).__name__ == "LineCollection":
        segs = []
        for sgm in a.get_segments():
            if len(sgm) < 2:
                continue
            g = LineString(sgm).intersection(region)
            for part in getattr(g, "geoms", [g]):
                if part.geom_type == "LineString" and len(part.coords) >= 2:
                    segs.append(np.asarray(part.coords))
        cols = a.get_colors()
        a.set_segments(segs)
        if len(cols) > 1:
            return False if len(cols) != len(segs) else True
        return True
    if type(a).__name__ == "Line2D" and a.get_marker() in ("None", None, "", " ", "none"):
        xy = np.asarray(a.get_xydata(), float)
        parts, cur = [], []
        for p in xy:
            if np.all(np.isfinite(p)):
                cur.append(p)
            elif cur:
                parts.append(cur); cur = []
        if cur:
            parts.append(cur)
        out = []
        for pp in parts:
            if len(pp) < 2:
                continue
            g = LineString(pp).intersection(region)
            for part in getattr(g, "geoms", [g]):
                if part.geom_type == "LineString" and len(part.coords) >= 2:
                    out.append(np.asarray(part.coords)); out.append(np.array([[np.nan, np.nan]]))
        if not out:
            return False
        xy2 = np.vstack(out[:-1]); a.set_data(xy2[:, 0], xy2[:, 1])
        return True
    return False


def unclip(fig, tol_px=0.5):
    """Remove the clip path of every artist whose drawn extent lies inside its axes (CorelDRAW would turn each clip path
    into a PowerClip). Artists that really extend beyond the axes keep it. Returns (kept, removed) counts."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    kept = removed = 0
    for ax in all_axes(fig):
        box = Bbox.from_extents(ax.bbox.x0 - tol_px, ax.bbox.y0 - tol_px, ax.bbox.x1 + tol_px, ax.bbox.y1 + tol_px)
        arts = list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.images) + list(ax.texts)
        arts += ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines()
        if _is_cbar(ax):                        # colour-bar solids fill exactly the colour-bar axes
            for a in arts:
                if a.get_clip_on():
                    a.set_clip_on(False); removed += 1
            continue
        for a in arts:
            if not a.get_visible() or not a.get_clip_on():
                continue
            if type(a).__name__ == "QuadMesh" and a.get_array() is not None:
                try:
                    if quadmesh_to_cells(ax, a):
                        removed += 1; continue
                except Exception:
                    pass
            if _custom_clip(ax, a):
                kept += 1; continue                 # a clip path drawn on purpose (e.g. water-table outline): keep
            if type(a).__name__ == "Rectangle" and a.get_data_transform() == ax.transData and \
                    ax.get_xscale() == "linear" and ax.get_yscale() == "linear":
                (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
                rx0, rx1 = sorted((a.get_x(), a.get_x() + a.get_width())); ry0, ry1 = sorted((a.get_y(), a.get_y() + a.get_height()))
                nx0, nx1, ny0, ny1 = max(rx0, x0), min(rx1, x1), max(ry0, y0), min(ry1, y1)
                if nx1 > nx0 and ny1 > ny0 and a.get_angle() == 0:
                    a.set_xy((nx0, ny0)); a.set_width(nx1 - nx0); a.set_height(ny1 - ny0)
                    a.set_clip_on(False); removed += 1; continue
            if type(a).__name__ in ("LineCollection", "Line2D") and _linear_data(ax, a):
                try:
                    if cut_lines(ax, a):                 # (a LineCollection reports no window extent: always cut)
                        a.set_clip_on(False); removed += 1; continue
                except Exception:
                    pass
            if type(a).__name__ == "PolyCollection" and len(a.get_paths()) > 1:
                if cut_polygons_to_axes(ax, a):
                    a.set_clip_on(False); removed += 1; continue
            try:
                b = a.get_window_extent(r)
            except Exception:
                b = None
            if b is not None and b.width >= 0 and box.x0 <= b.x0 and box.y0 <= b.y0 and b.x1 <= box.x1 and b.y1 <= box.y1:
                a.set_clip_on(False); removed += 1
            else:
                kept += 1
    return kept, removed


def _safe(s):
    s = re.sub(r"\$[^$]*\$", lambda m: re.sub(r"\\mathbf|\\mathdefault|[{}\\^_$]", "", m.group(0)), s or "")
    s = re.sub(r"[^0-9A-Za-zΑ-Ωα-ω.+-]+", "_", s).strip("_")
    return s[:48] or "x"


def assign_gids(fig):
    """Named groups for CorelDRAW's Object Manager: one group per panel (panel_a …) with its axes, ticks, labels,
    series, legend and colour bar inside; panel letters as panel_letter_a. Ids are unique within the figure."""
    import ert_plot_style as eps
    used = {}

    def uid(g):
        n = used.get(g, 0); used[g] = n + 1
        return g if n == 0 else f"{g}_{n + 1}"
    axes = all_axes(fig); names = {}; prev = "main"; k = {}
    for ax in axes:
        if _is_cbar(ax):
            continue
        pl = next((t for t in ax.texts if (t.get_gid() or "") == "panel_label"), None)
        if pl is not None:
            prev = re.match(r"\(([a-z])\)", pl.get_text()).group(1); k[prev] = 0; names[ax] = f"panel_{prev}"
        else:
            k[prev] = k.get(prev, 0) + 1; names[ax] = f"panel_{prev}" if prev == "main" and k[prev] == 1 else f"panel_{prev}_{k[prev]}"
    for ax in axes:
        if _is_cbar(ax):
            cb = ax._colorbar
            owner = getattr(cb.mappable, "axes", None)
            base = names.get(owner, "figure")
            ax.set_gid(uid(f"{base}/colorbar"))
            for axis, nm in ((ax.xaxis, "ticks_x"), (ax.yaxis, "ticks_y")):
                axis.set_gid(uid(f"{base}/colorbar/{nm}"))
            continue
        n = names[ax]
        ax.set_gid(uid(n)); ax.patch.set_gid(uid(f"{n}/axes/background"))
        for sn, sp in ax.spines.items():
            sp.set_gid(uid(f"{n}/axes/frame_{sn}"))
        ax.xaxis.set_gid(uid(f"{n}/ticks_x")); ax.yaxis.set_gid(uid(f"{n}/ticks_y"))
        ax.xaxis.label.set_gid(uid(f"{n}/labels/x_label")); ax.yaxis.label.set_gid(uid(f"{n}/labels/y_label"))
        for i, ln in enumerate(ax.lines):
            lab = ln.get_label() or ""
            ln.set_gid(uid(f"{n}/series/{_safe(lab) if not lab.startswith('_') else f'line_{i}'}"))
        for i, c in enumerate(ax.collections):
            lab = c.get_label() or ""
            tn = type(c).__name__
            kind = ("cells" if tn in ("PolyCollection", "QuadMesh") and c.get_array() is not None else
                    "contours" if "Contour" in tn else "band" if tn in ("PolyCollection", "FillBetweenPolyCollection") else
                    "points" if tn == "PathCollection" else tn.lower())
            c.set_gid(uid(f"{n}/series/{_safe(lab) if not lab.startswith('_') else kind}"))
        for i, p in enumerate(ax.patches):
            lab = p.get_label() or ""
            p.set_gid(uid(f"{n}/patches/{_safe(lab) if not lab.startswith('_') else type(p).__name__.lower()}"))
        for i, im in enumerate(ax.images):
            im.set_gid(uid(f"{n}/image_{i}"))
        for t in ax.texts:
            g = t.get_gid() or ""
            if g == "panel_label":
                t.set_gid(uid(f"panel_letter_{n.split('_')[1]}"))
            elif g == "panel_ident":
                t.set_gid(uid(f"{n}/labels/identifier"))
            else:
                t.set_gid(uid(f"{n}/labels/{_safe(t.get_text())}"))
        lg = ax.get_legend()
        if lg is not None:
            lg.set_gid(uid(f"{n}/legend"))
    for i, lg in enumerate(fig.legends):
        lg.set_gid(uid("figure/legend" if i == 0 else f"figure/legend_{i + 1}"))
    for i, t in enumerate(fig.texts):
        t.set_gid(uid(f"figure/labels/{_safe(t.get_text())}"))
    fig.patch.set_gid("figure/background")


def opaque_for_eps(fig):
    """PostScript has no transparency: composite every translucent face / edge colour over white (same look where
    nothing overlaps; overlapping translucent areas become opaque — SVG and PDF keep the transparency)."""
    import numpy as np
    from matplotlib.colors import to_rgba
    n = 0

    def comp(c):
        c = np.asarray(c, float)
        if c.ndim == 1:
            return comp(c[None])[0]
        a = c[:, 3:4]
        return np.hstack([c[:, :3] * a + (1 - a), np.where(a > 0, 1.0, 0.0)])
    for a in fig.findobj():
        try:
            al = a.get_alpha()
        except Exception:
            continue
        if hasattr(a, "get_facecolor") and hasattr(a, "set_facecolor"):
            try:
                fc = a.get_facecolor(); ec = a.get_edgecolor()
                fcn = np.asarray(fc, float); ecn = np.asarray(ec, float)
                if (fcn.size and (fcn[..., 3] < 1).any() and (fcn[..., 3] > 0).any()) or al is not None:
                    if al is not None:
                        fcn = np.atleast_2d(fcn).copy(); fcn[:, 3] *= al
                        ecn = np.atleast_2d(ecn).copy() if ecn.size else ecn
                        if ecn.size:
                            ecn[:, 3] *= al
                        a.set_alpha(None)
                    a.set_facecolor(comp(fcn) if fcn.size else fc)
                    if ecn.size:
                        a.set_edgecolor(comp(ecn))
                    n += 1
            except Exception:
                pass
        elif al is not None and hasattr(a, "get_color") and hasattr(a, "set_color"):
            try:
                c = to_rgba(a.get_color(), al); a.set_alpha(None); a.set_color(tuple(comp(c))); n += 1
            except Exception:
                pass
    return n


_ADV = {}


def _advance(ch, size, bold=True, italic=False):
    """Advance width (px at 72 dpi = pt) of one Times New Roman character, as matplotlib's mathtext places it."""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font, LOAD_NO_HINTING
    key = (bold, italic)
    if key not in _ADV:
        fp = font_manager.FontProperties(family="Times New Roman", weight="bold" if bold else "normal",
                                         style="italic" if italic else "normal")
        _ADV[key] = FT2Font(font_manager.findfont(fp, fallback_to_default=False))
    f = _ADV[key]
    f.set_size(size, 72)
    return f.load_char(ord(ch), flags=LOAD_NO_HINTING).linearHoriAdvance / 65536.0


def svg_mathtext_runs(path, tol=0.6):
    """matplotlib writes mathtext as <tspan x="x1 x2 x3 …"> with one position per character; CorelDRAW misplaces such
    runs (overlapping words, lost text after a subscript). Rewrite every multi-position tspan as tspans with a single
    start position, split where a character does not follow the previous one at its natural advance, and keep spaces
    (xml:space="preserve"). Returns the number of tspans rewritten."""
    import html
    s = open(path, encoding="utf-8").read()
    pat = re.compile(r'<tspan style="([^"]*)" x="([^"]+)" y="([^"]+)">([^<]*)</tspan>')
    n = 0

    def rep(m):
        nonlocal n
        style, xs, y, txt = m.group(1), m.group(2).split(), m.group(3), html.unescape(m.group(4))
        if len(xs) <= 1 or len(xs) != len(txt):
            return m.group(0)
        size = float(re.search(r"font-size: ([0-9.]+)px", style).group(1))
        bold = "font-weight: bold" in style; italic = "font-style: italic" in style
        xs = [float(v) for v in xs]
        runs, start = [], 0
        for i in range(1, len(txt)):
            if abs(xs[i] - (xs[i - 1] + _advance(txt[i - 1], size, bold, italic))) > tol:
                runs.append((start, i)); start = i
        runs.append((start, len(txt)))
        # CorelDRAW drops leading spaces of a run: start each run at its first non-space character, at its own position
        runs = [(next((k for k in range(a, b) if not txt[k].isspace()), b), b) for a, b in runs]   # incl. U+00A0
        runs = [(a, b) for a, b in runs if a < b]
        n += 1
        # mathtext writes spaces as U+00A0, which CorelDRAW renders too narrow: plain spaces (kept by xml:space="preserve")
        return "".join(f'<tspan style="{style}" x="{xs[a]:.6g}" y="{y}">{html.escape(txt[a:b].replace(chr(160), " "), quote=False)}</tspan>'
                       for a, b in runs)
    s2 = pat.sub(rep, s)
    s2 = s2.replace("<text>", '<text xml:space="preserve">')
    # with xml:space="preserve" the indentation between tspans would become text: remove it inside <text> elements
    s2 = re.sub(r'(<text xml:space="preserve">)\s+(?=<tspan)', r"\1", s2)
    s2 = re.sub(r"</tspan>\s+(?=<tspan|</text>)", "</tspan>", s2)
    # CorelDRAW still misplaces several positioned tspans in one <text> (rotated labels, spaces between runs): write each
    # run of a multi-run text as its own <text> element (one text object per run in CorelDRAW)
    tsp = re.compile(r'<tspan (style="[^"]*") (x="[^"]*") (y="[^"]*")>([^<]*)</tspan>')

    def split_text(m):
        parts = tsp.findall(m.group(1))
        if len(parts) < 2:
            return m.group(0)
        return "".join(f'<text xml:space="preserve" {st} {x} {y}>{t}</text>' for st, x, y, t in parts)
    s2 = re.sub(r'<text xml:space="preserve">((?:<tspan [^>]*>[^<]*</tspan>)+)</text>', split_text, s2)
    open(path, "w", encoding="utf-8", newline="\n").write(s2)
    return n


def svg_explicit_fonts(path):
    """Rewrite matplotlib's CSS shorthand 'font: [italic] 700 7.5px 'Times New Roman', serif' as explicit properties
    (font-family, font-size, font-weight, font-style): CorelDRAW reads these and keeps the bold weight."""
    s = open(path, encoding="utf-8").read()
    pat = re.compile(r"font:\s*(italic\s+|oblique\s+)?(?:(\d{3}|bold|normal)\s+)?([0-9.]+)px\s+([^;\"]+)")

    def rep(m):
        style = "italic" if m.group(1) else "normal"
        w = m.group(2) or "400"
        w = "bold" if w in ("700", "bold") or (w.isdigit() and int(w) >= 600) else "normal"
        fam = m.group(4).strip()
        return f"font-family: {fam}; font-size: {m.group(3)}px; font-weight: {w}; font-style: {style}"
    s2, n = pat.subn(rep, s)
    open(path, "w", encoding="utf-8", newline="\n").write(s2)
    return n


def _read_csv_rows(p):
    import csv
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return rows[1:]


def corel_main(a):
    import numpy as np
    import ert_plot_style as eps
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1792281600")      # fixed EPS creation date: reproducible files
    eps.corel_mode(True)
    out = OUT_COREL
    for d in ("SVG", "EPS", "PDF", "preview"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    import matplotlib.pyplot as plt
    import paper_figures as pf
    pf.DATA_DIR = os.path.join(OUT, "data")                  # v1.8 stored arrays (FigS01 lens bases), read only
    byname = {n.split("_")[0]: (n, fn, old) for n, fn, old in pf.FIGURES}
    only = set(a.only.split(",")) if a.only else None
    t_all = time.time(); res = {}; s03 = None
    snap0 = wrr_snapshot()
    inputs = [l.split("  ", 1)[1] for l in open(os.path.join(OUT, "inputs_sha256_before.txt"), encoding="utf-8").read().splitlines() if l]
    inp0 = {p: sha(os.path.join(ROOT, p)) for p in inputs}
    prot0 = {rel(p): sha(p) for p in wrr_protected()}
    wrr0 = {rel(p): sha(p) for p in sorted(glob.glob(os.path.join(OUT_WRR, "**", "*"), recursive=True)) if os.path.isfile(p)}
    for label, short in WRR_NAMES:
        if only and label not in only:
            continue
        name, fn, old = byname[short]; t0 = time.time()

        def build(corel):
            eps.corel_mode(corel)
            if name == "FigS03_BiasRule":
                return pf.figS03(*s03[:3])[0]
            if name == "Fig09_ERT2025AtSoundings":
                return pf.fig09(img_col=a.fig9_color)[0]
            return fn()[0]
        if name == "FigS03_BiasRule" and s03 is None:
            eps.corel_mode(True); s03 = pf.figS03_data(OUT)
        fig = build(True)
        r = dict(v18=name, v191_csv=f"results/figs_wrr/data/{label}.csv")
        rows, _ = figure_rows(fig)
        ref = _read_csv_rows(os.path.join(OUT_WRR, "data", f"{label}.csv"))
        cells_c = [np.ma.copy(c.get_array()) for ax in all_axes(fig) if not _is_cbar(ax) for c in ax.collections
                   if c.get_array() is not None and c.norm is not None and type(c).__name__ in ("PolyCollection", "QuadMesh")]
        r["fonts"] = text_inventory(fig)
        r["overlaps"] = eps.overlap_checks(fig)
        r["text_overlaps"] = text_overlaps(fig)
        r["smallest_pt"], r["clipped"] = eps.layout_checks(fig)
        r["rasterized_switched_off"] = vectorize(fig)
        r["translucent_hatches_split"] = split_translucent_hatches(fig)
        kept, removed = unclip(fig)
        r["clip_kept"], r["clip_removed"] = kept, removed
        assign_gids(fig)
        w, h = fig.get_size_inches() * 25.4
        r["size_mm"] = [round(w, 2), round(h, 2)]
        paths = {}
        for ext, kw in (("svg", dict(format="svg", dpi=IMG_DPI, metadata={"Date": None})),
                        ("pdf", dict(format="pdf", dpi=IMG_DPI, metadata={"CreationDate": None, "ModDate": None})),
                        ("png", dict(format="png", dpi=300))):
            p = os.path.join(out, "preview" if ext == "png" else ext.upper(), f"{label}.{ext}")
            fig.savefig(p, bbox_inches=None, **kw); paths[ext] = p
            if ext == "svg":
                r["svg_font_rewrites"] = svg_explicit_fonts(p)
                r["svg_mathtext_runs"] = svg_mathtext_runs(p)
        cells = [c for ax in all_axes(fig) if not _is_cbar(ax) for c in ax.collections
                 if c.get_array() is not None and type(c).__name__ in ("PolyCollection", "QuadMesh")]
        r["n_vector_cells"] = int(sum(len(c.get_paths()) for c in cells))
        # fallback for figures that are too heavy for CorelDRAW: > 30 MB SVG, or > 20,000 vector cells (CorelDRAW 2023 did
        # not finish importing Figure S9's 52,000 cells within 40 min)
        if os.path.getsize(paths["svg"]) > 30e6 or r["n_vector_cells"] > 20000:
            for c in cells:
                c.set_rasterized(True)
            for ext in ("svg", "pdf"):
                p = os.path.join(out, ext.upper(), f"{label}_cells600dpi.{ext}")
                fig.savefig(p, format=ext, dpi=IMG_DPI, bbox_inches=None,
                            metadata={"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None})
                paths[f"{ext}_fallback"] = p
                if ext == "svg":
                    svg_explicit_fonts(p); svg_mathtext_runs(p)
            for c in cells:
                c.set_rasterized(False)
            r["fallback_cells600dpi"] = True
        r["eps_opaque_artists"] = opaque_for_eps(fig)
        p = os.path.join(out, "EPS", f"{label}.eps")
        fig.savefig(p, format="eps", dpi=IMG_DPI, bbox_inches=None, metadata={"Creator": None, "CreationDate": None}); paths["eps"] = p
        plt.close(fig)
        r["files"] = {k: dict(file=rel(v), mb=round(os.path.getsize(v) / 1e6, 2)) for k, v in paths.items()}
        # ---- comparisons that rebuild figures (after saving: a rebuild resets the rcParams)
        r["data_identical_to_v191"] = rows == ref
        n_clabel = sum(len(getattr(c, "labelTexts", []) or []) for ax in all_axes(fig) for c in ax.collections)
        if not r["data_identical_to_v191"] and n_clabel:
            # inline contour labels cut gaps whose width follows the label font: compare with clabel off in both builds
            import matplotlib.axes
            keep = matplotlib.axes.Axes.clabel; matplotlib.axes.Axes.clabel = lambda self, *q, **k: []
            try:
                fa = build(True); ra, _ = figure_rows(fa); plt.close(fa)
                fb = build(False); rb, _ = figure_rows(fb); plt.close(fb)
            finally:
                matplotlib.axes.Axes.clabel = keep; eps.corel_mode(True)
            r["data_identical_to_v191"] = ra == rb
            r["contour_labels_off_for_comparison"] = True
            r["data_identical_outside_contours"] = [x for x in rows if "Contour" not in x[1]] == [x for x in ref if "Contour" not in x[1]]
        if not r["data_identical_to_v191"]:
            r["data_first_diff"] = next(([x, y] for x, y in zip(ref, rows) if x != y), None); r["data_rows"] = (len(ref), len(rows))
        # cell arrays: compare with the submission build (rasterised, v1.9.1 fonts)
        if short in ("Fig02", "Fig06", "FigS01", "Fig03", "FigS08"):
            f2 = build(False)
            cells_s = [np.ma.copy(c.get_array()) for ax in all_axes(f2) if not _is_cbar(ax) for c in ax.collections
                       if c.get_array() is not None and c.norm is not None and type(c).__name__ in ("PolyCollection", "QuadMesh")]
            plt.close(f2); eps.corel_mode(True)
            r["cells"] = dict(n_layers=len(cells_c), n_cells=int(sum(x.size for x in cells_c)),
                              identical=len(cells_c) == len(cells_s) and all(
                                  np.array_equal(np.ma.getdata(x), np.ma.getdata(y), equal_nan=True)
                                  and np.array_equal(np.ma.getmaskarray(x), np.ma.getmaskarray(y)) for x, y in zip(cells_c, cells_s)))
        eps.corel_mode(True)
        res[label] = r
        fams = sorted({f for e in r["fonts"] for f in e["fonts"]})
        print(f"[corel] {label}: data same {r['data_identical_to_v191']}, cells {r.get('cells', {}).get('identical')}, fonts {fams}, "
              f"overlaps {len(r['overlaps']['legend_overlaps'])}+{len(r['overlaps']['letter_overlaps'])}, text overlaps {len(r['text_overlaps'])}, "
              f"smallest {r['smallest_pt']:.2f}, clipped {len(r['clipped'])}, clips kept {kept} removed {removed}, "
              f"svg {r['files']['svg']['mb']} MB, {time.time() - t0:.0f} s", flush=True)
        for x in r["text_overlaps"][:6]:
            print("   text overlap:", x, flush=True)
    if only:
        json.dump(res, open(os.path.join(out, "_partial_checks.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
        return
    chk = dict(figures=res, seconds=round(time.time() - t_all))
    chk["inputs_identical"] = all(sha(os.path.join(ROOT, p)) == h for p, h in inp0.items())
    chk["protected_identical"] = {rel(p): sha(p) for p in wrr_protected()} == prot0
    chk["figs_wrr_unchanged"] = {rel(p): sha(p) for p in sorted(glob.glob(os.path.join(OUT_WRR, "**", "*"), recursive=True)) if os.path.isfile(p)} == wrr0
    snap1 = wrr_snapshot()
    chk["side_effects"] = dict(created=sorted(set(snap1) - set(snap0)), removed=sorted(set(snap0) - set(snap1)),
                               changed=sorted(k for k in snap0 if k in snap1 and snap0[k] != snap1[k]))
    json.dump(chk, open(os.path.join(out, "checks_corel.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print("[corel] done", chk["inputs_identical"], chk["protected_identical"], chk["figs_wrr_unchanged"], {k: len(v) for k, v in chk["side_effects"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated figure names")
    ap.add_argument("--draft-only", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fig9-color", default="#4D4D4D", help="image column of Fig. 9 (tomato was too close to the 3 Ω·m band)")
    ap.add_argument("--wrr", action="store_true", help="v1.9 WRR export to results/figs_wrr/ (--only takes WRR labels)")
    ap.add_argument("--no-compare", action="store_true", help="--wrr: skip the redraw with the v1.8 code")
    ap.add_argument("--corel", action="store_true", help="v1.10 CorelDRAW versions to 'Coral Draw/' (--only takes WRR labels)")
    a = ap.parse_args()
    if a.corel:
        return corel_main(a)
    if a.wrr:
        return wrr_main(a)
    full = not a.only and not a.draft_only and os.path.abspath(a.out) == os.path.abspath(OUT)
    os.makedirs(a.out, exist_ok=True)
    if full:
        with open(os.path.join(OUT, "protected_sha256_before.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines(f"{sha(p)}  {rel(p)}\n" for p in protected_files())
        snap0 = snapshot()
    t_all = time.time()
    import matplotlib.pyplot as plt
    import paper_figures as pf
    import ert_plot_style as eps
    pf.DATA_DIR = os.path.join(a.out, "data") if full else None
    only = set(a.only.split(",")) if a.only else None
    res, meta = {}, {}
    for name, fn, old in pf.FIGURES:
        if only and name not in only:
            continue
        t0 = time.time()
        if name == "FigS03_BiasRule":
            if a.draft_only:
                G_ = pf.g25_rows(); g = G_["g"]
                fit = pf.quiet(g.fit_rule, G_["rows"], "lam50", seed=[136200])
                fit_lam = {int(l): pf.quiet(g.fit_rule, G_["rows"], f"lam{int(l)}_r01", seed=[136201]) for l in [5.0, 10.0, 20.0, 50.0, 100.0]}
                fit_grad = pf.quiet(g.fit_rule, G_["rows"], "lam50", truths=[t for t in g.TRUTHS if g.params(t)["tau"] > 0], seed=[136200])
                info = {}
            else:
                fit, fit_lam, fit_grad, info = pf.figS03_data(a.out)
            fig, m = pf.figS03(fit, fit_lam, fit_grad); m = dict(fit_check=info)
        elif name == "Fig09_ERT2025AtSoundings":
            fig, m = pf.fig09(img_col=a.fig9_color)
        else:
            fig, m = fn()
        if a.draft_only:
            smallest, clipped = eps.layout_checks(fig)
            os.makedirs(os.path.join(a.out, "draft"), exist_ok=True)
            fig.savefig(os.path.join(a.out, "draft", f"{name}.png"), dpi=300); files = []
        else:
            files, smallest, clipped = eps.savefig(fig, name, a.out, 190)
        w, h = fig.get_size_inches() * 25.4
        plt.close(fig)
        res[name] = dict(old=old, size_mm=[round(w, 1), round(h, 1)], smallest_pt=smallest, clipped=clipped,
                         seconds=round(time.time() - t0, 1), files=[rel(p) for p in files],
                         bytes={os.path.splitext(p)[1]: os.path.getsize(p) for p in files})
        meta[name] = m
        print(f"[fig] {name}: {w:.0f}×{h:.0f} mm, smallest {smallest:.2f} pt, clipped {len(clipped)}, {time.time() - t0:.0f} s", flush=True)
    if not full:
        return
    checks = dict(figures=res, seconds=round(time.time() - t_all))
    before = [l.split("  ", 1) for l in open(os.path.join(OUT, "inputs_sha256_before.txt"), encoding="utf-8").read().splitlines() if l]
    after = [(sha(os.path.join(ROOT, p)), p) for _, p in before]
    with open(os.path.join(OUT, "inputs_sha256_after.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{h}  {p}\n" for h, p in after)
    checks["inputs_identical"] = [(h, p) for h, p in before] == list(after)
    checks["n_inputs"] = len(before)
    with open(os.path.join(OUT, "protected_sha256_after.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{sha(p)}  {rel(p)}\n" for p in protected_files())
    checks["protected_identical"] = (open(os.path.join(OUT, "protected_sha256_before.txt"), encoding="utf-8").read()
                                     == open(os.path.join(OUT, "protected_sha256_after.txt"), encoding="utf-8").read())
    checks["n_protected"] = len(protected_files())
    snap1 = snapshot()
    checks["side_effects"] = dict(created=sorted(set(snap1) - set(snap0)), removed=sorted(set(snap0) - set(snap1)),
                                  changed=sorted(k for k in snap0 if k in snap1 and snap0[k] != snap1[k]))
    checks["numbers_registered_by_imports"] = len(pf.mod("s18").NUM.rows)
    checks["spot"] = spot_checks(meta, pf)
    checks["figS03_fit"] = meta.get("FigS03_BiasRule", {}).get("fit_check")
    checks["fig9_image_colour"] = a.fig9_color
    contact_sheet([n for n, _, _ in pf.FIGURES], os.path.join(OUT, "contact_sheet.png"))
    json.dump(checks, open(os.path.join(OUT, "checks.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    ok = (checks["inputs_identical"] and checks["protected_identical"] and not any(checks["side_effects"].values())
          and checks["numbers_registered_by_imports"] == 0
          and all(r["ok"] for k, v in checks["spot"].items() if isinstance(v, list) for r in v if isinstance(r, dict))
          and all(r["smallest_pt"] >= 7.0 - 1e-9 and not r["clipped"] for r in res.values()))
    print(f"[checks] all passed: {ok}  ({checks['seconds']} s)")


if __name__ == "__main__":
    main()
