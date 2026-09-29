"""
18_paper.py
===========
Westhoek ERT — paper figures, tables and quoted numbers.

  all [--out ROOT] [--only F1,F2,...] [--recompute]

Figures F1–F8 (+ appendix A1) → ROOT/figures/paper/*.pdf (vector; dense meshes rasterised inside) + *.png (300 dpi);
tables T1–T3 (+ appendix TA1) → ROOT/results/paper/tables.md and T*.csv; every quoted number → ROOT/results/paper/
numbers.md (+ numbers.json), each with the function that computes it, the cached input it comes from and — where the
same number was reported before — the stored report used to cross-check it (run_all.py --check).

Inputs (read-only, from the project): data/raw, data/processed/runs (inversions), data/processed/synthetic*
(benchmark runs), data/processed/bel*/ (prior samples, twin forward, approximation error), results/runs/g1_lam50_v3|v4
(linearised posterior products), ROOT/data/processed/stage6_v2 (17_salinity_v2.py). Heavy BEL products and the
fine-mesh forward check are cached in ROOT/data/processed/paper/ (rebuilt with --recompute or when missing).

Headline subset (v1.0 freeze): "site" + "physical" (ρ_lens ≥ stage6_analysis.PHYS_RHO_LENS_MIN = 0.69415 Ω·m)
of prior v2 + v3, for the base-only BEL, the second BEL (D1, D2, logit-bounded log10 ρ; transform bounds ρ_lens
0.4–3.0, ρ_below 2–20, ρ_discharge 2.5–150 Ω·m — the ρ_lens transform bound is NOT the physical bound, see
19_rho_lens_bound_check.py) and the joint BEL (17_salinity_v2.py). The pre-freeze site-only field lens base is recomputed
with its original seeds for the comparison. Second-BEL seeds default_rng([132000, 3]) field, [132000, 60000 + j]
held-out; its targets are cached in ROOT/data/processed/paper/second_bel_targets_phys.npz.

Conventions: blue = conductive (RdBu_r on ρ; Blues for P(saline)); vertical exaggeration stated on every section;
dune zone x < 90 m hatched as weakly constrained. Seeds: BEL site-v3 lens base and held-out = the Stage-6 refit seeds
(default_rng([129000, 1]) field, [129000, 90000 + j] held-out); new BEL posteriors default_rng([132000, k]):
Stage-4 truths k = 20000 + 100·truth + r, twin k = 30000 + r, ρ_below bins k = 40 + bin (field) and
50000 + 1000·bin + j (held-out), full prior v3 k = 2, prior examples k = 7; MCD default_rng(77000).
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import read_trace  # noqa: E402
read_trace.install("18_paper.py")
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


s15 = _load("s15", "15_stage5_closeout.py")
g14, b13, c11, sb = s15.g14, s15.b13, s15.c11, s15.sb
p04 = _load("p04", "04_misfit_diagnostics.py")
import stage6_analysis as s6  # noqa: E402

TX = b13.TX
XREP = (150, 200, 250, 300, 350)
DUNE_X = 90.0
SEED = 132000
PROJ = wc.ROOT
RUNS = os.path.join(wc.PROCESSED_DIR, "runs")
LIN_V3 = os.path.join(wc.RESULTS_DIR, "runs", "g1_lam50_v3", "posterior_interface_bands.npz")
LIN_V4 = os.path.join(wc.RESULTS_DIR, "runs", "g1_lam50_v4", "interface_definitions_bands.npz")
LAM_TAGS = [("g1_lam5", 5), ("g1_lam10", 10), ("deep_drop_data", 20), ("g1_lam50", 50), ("g1_lam100", 100)]
S = "18_paper.py"


def rel(p):
    return os.path.relpath(p, PROJ).replace("\\", "/")


# ─────────────────────────────────────────────────────────────────────────────
# number registry
# ─────────────────────────────────────────────────────────────────────────────
class Numbers:
    def __init__(self):
        self.rows = []

    def add(self, key, value, desc, fn, inputs, ref=None):
        """ref = (stored report relative to the project, context substring, match substring)."""
        self.rows.append(dict(key=key, value=str(value), desc=desc, script=f"{S}:{fn}", inputs=inputs, ref=ref))

    def write(self, res):
        L = ["# Numbers quoted in the paper", "",
             "Every value is recomputed by `scripts/18_paper.py` (or `17_salinity_v2.py`) from the cached inputs listed. "
             "'Cross-check' = the stored report where the same number was first reported; `run_all.py --check` verifies "
             "that the line matching the context contains the value.", "",
             "| # | key | value | quantity | computed by | from | cross-check |", "|---|---|---|---|---|---|---|"]
        for i, r in enumerate(self.rows, 1):
            ref = f"`{r['ref'][0]}`" if r["ref"] else "— (first reported here)"
            L.append(f"| {i} | `{r['key']}` | {md(r['value'])} | {md(r['desc'])} | `{r['script']}` | {md(r['inputs'])} | {ref} |")
        with open(os.path.join(res, "numbers.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        json.dump(self.rows, open(os.path.join(res, "numbers.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


NUM = Numbers()


def q(v, d=2):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    lo, md, hi = np.percentile(v, [5, 50, 95])
    return f"{md:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ─────────────────────────────────────────────────────────────────────────────
# figure helpers
# ─────────────────────────────────────────────────────────────────────────────
def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "savefig.dpi": 300,
                         "axes.titleweight": "bold", "font.family": "DejaVu Sans"})
    return plt


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGP, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[fig] {name}", flush=True)


def dune_hatch(ax, label=False, text=True):
    """Dune zone x < 90 m: translucent grey, sparse hatch, in-panel label (weakly constrained)."""
    from matplotlib.patches import Rectangle
    import matplotlib as mpl_
    mpl_.rcParams["hatch.linewidth"] = 0.45
    y0, y1 = ax.get_ylim(); xa = min(ax.get_xlim()[0], 0) - 10
    ax.add_patch(Rectangle((xa, y0), DUNE_X - xa, y1 - y0, facecolor=(0.55, 0.55, 0.55, 0.28), edgecolor=(0.3, 0.3, 0.3, 0.6),
                           hatch="x", lw=0, zorder=8, label="dunes x < 90 m: weakly constrained" if label else None))
    if text:
        ax.text(DUNE_X / 2, y0 + 0.04 * (y1 - y0), "dunes:\nweakly\nconstrained", ha="center", va="bottom", fontsize=5.8,
                zorder=13, color="0.15", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.85))


def legend_below(ax, ncol=2, dy=-0.22, **kw):
    kw.setdefault("fontsize", 6.3)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, dy), ncol=ncol, facecolor="white", edgecolor="0.6",
                    framealpha=1.0, **kw)
    leg.set_zorder(30)
    return leg


def cbar_space(fig, ax, mappable):
    """Reserve the same colorbar space as a sibling panel so stacked sections align."""
    cb = fig.colorbar(mappable, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.set_visible(False)


def ve_text(ax, ve):
    ax.text(0.995, 0.02, f"V.E. {ve:g}×", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, zorder=12,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9))


def panel(ax, s):
    """Panel letter as a prefix of the (existing) title."""
    ax.set_title(f"({s}) " + ax.get_title(), fontsize=ax.title.get_fontsize())


COL = {"thr3": "#d35400", "logmid": "#1f4e9c", "maxgrad": "#7f8c8d", "bel": "#6a3d9a"}   # one colour per method


def xwin(x, v, lo=130.0, hi=450.0):
    """Mask a lens-base line / band outside the headline window x = 130–450 m."""
    return np.where((np.asarray(x) >= lo) & (np.asarray(x) <= hi), v, np.nan)


def lambert72_to_lonlat(x, y):
    """Belgian Lambert 72 (EPSG:31370) → lon/lat (International 1924 ellipsoid, no datum shift: ~100 m; inset only)."""
    a, f = 6378388.0, 1 / 297.0
    e = np.sqrt(2 * f - f * f)
    p1, p2 = np.radians(49.8333339), np.radians(51.16666723333333)
    lam0 = np.radians(4.367486666666666); E0, N0 = 150000.013, 5400088.438
    m = lambda ph: np.cos(ph) / np.sqrt(1 - (e * np.sin(ph)) ** 2)
    t = lambda ph: np.tan(np.pi / 4 - ph / 2) / ((1 - e * np.sin(ph)) / (1 + e * np.sin(ph))) ** (e / 2)
    n = (np.log(m(p1)) - np.log(m(p2))) / (np.log(t(p1)) - np.log(t(p2)))
    Fc = m(p1) / (n * t(p1) ** n)
    dx, dy = x - E0, -(y - N0)
    tt = (np.hypot(dx, dy) / (a * Fc)) ** (1 / n)
    phi = np.pi / 2 - 2 * np.arctan(tt)
    for _ in range(10):
        phi = np.pi / 2 - 2 * np.arctan(tt * ((1 - e * np.sin(phi)) / (1 + e * np.sin(phi))) ** (e / 2))
    return float(np.degrees(np.arctan2(dx, dy) / n + lam0)), float(np.degrees(phi))


# schematic coastline and borders (approximate, hand-digitised; location inset only)
COAST = [(1.60, 50.87), (1.72, 50.93), (1.86, 50.96), (2.13, 51.01), (2.37, 51.05), (2.546, 51.09), (2.73, 51.15),
         (2.92, 51.23), (3.13, 51.32), (3.29, 51.35), (3.37, 51.37), (3.55, 51.41), (3.60, 51.44), (3.90, 51.42)]
BORDERS = [[(2.546, 51.09), (2.60, 51.00), (2.64, 50.85)], [(3.37, 51.37), (3.53, 51.28), (3.90, 51.22)]]


def location_inset(ax, lon, lat):
    ins = ax.inset_axes([0.02, 0.02, 0.40, 0.36])
    cx, cy = zip(*COAST)
    ins.fill(list(cx) + [3.90, 1.60], list(cy) + [50.80, 50.80], color="#efe6cf", lw=0)
    ins.plot(cx, cy, color="0.35", lw=0.7)
    for b in BORDERS:
        bx, by = zip(*b); ins.plot(bx, by, color="0.35", lw=0.6, ls="--")
    ins.plot(lon, lat, "*", color="#c0392b", ms=8, mec="k", mew=0.4)
    for tx, ty, lab in ((2.05, 50.88, "FR"), (3.15, 51.02, "BE"), (3.62, 51.29, "NL")):
        ins.text(tx, ty, lab, fontsize=6, color="0.25")
    ins.text(2.35, 51.42, "North Sea", fontsize=6, style="italic", color="#1f4e9c")
    ins.set_xlim(1.60, 3.90); ins.set_ylim(50.80, 51.62); ins.set_aspect(1 / np.cos(np.radians(51.2)))
    ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor("#dbe9f6")
    ins.set_title("location (schematic)", fontsize=5.8, pad=1.5)
    return ins


def rho_norm():
    import matplotlib.colors as mcolors
    return mcolors.LogNorm(0.5, 500)


_CTX = {}


def ref_ctx():
    """Reference run, its paraDomain mesh and column samplers (built once)."""
    if not _CTX:
        ref = wc.load_run("g1_lam50")
        mesh, _ = wc.rebuild_paradomain(ref)
        _CTX.update(ref=ref, mesh=mesh, polys=wc.cell_polygons(mesh),
                    cs_line=wc.ColumnSampler(mesh, np.arange(0.0, 506.0, 2.0)), cs_tx=wc.ColumnSampler(mesh, TX))
    return _CTX


def lens_base(run, cs, defn="logmid"):
    return c11.interfaces_all(cs(np.log10(run["model"]))[:, :, None], cs.z)[defn][:, 0]


def run_npz(tag):
    return wc.load_run(tag)


# ─────────────────────────────────────────────────────────────────────────────
# F1 — site
# ─────────────────────────────────────────────────────────────────────────────
def read_gps():
    """GPS_coordinate.xlsx (Belgian Lambert-72), parsed with the standard library (no openpyxl in the env)."""
    z = zipfile.ZipFile(os.path.join(PROJ, "data", "raw", "GPS_coordinate.xlsx"))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
    rows = []
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{{{ns['m']}}}row"):
        vals = []
        for c in row.findall("m:c", ns):
            v = c.find("m:v", ns)
            if v is None:
                vals.append(None); continue
            vals.append(shared[int(v.text)] if c.get("t") == "s" else float(v.text))
        rows.append(vals)
    out = {}
    for r in rows:
        if r and isinstance(r[0], float):
            out[int(r[0])] = (r[1], r[2], r[3])
    return out


def fig_site():
    plt = mpl()
    land = np.loadtxt(os.path.join(PROJ, "data", "raw", "elec.dat"), skiprows=1, usecols=(0, 1, 2), max_rows=102)
    ex, ez = land[:, 1], land[:, 2]
    h_sea, x_hw, x_lw = p04.water_lines(wc.ELEC_FILE, ex, ez)
    gps = read_gps()
    nq = int(open(os.path.join(PROJ, "data", "raw", "land_data.dat")).readline().split()[0])
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    x0, y0 = gps[1][0], gps[1][1]
    P = lambda e: ((gps[e][0] - x0), (gps[e][1] - y0))
    a1, a2 = P(1), P(102); m1, m2 = P(103), P(538)
    ax.plot([a1[0], a2[0]], [a1[1], a2[1]], "-", color="#1f4e9c", lw=2.2, label="land profile (102 electrodes, 5 m spacing)")
    ax.plot([m1[0], m2[0]], [m1[1], m2[1]], "-", color="#e0a040", lw=1.6, label="marine profile (floating, high tide)")
    L_land = np.hypot(a2[0] - a1[0], a2[1] - a1[1]); u = ((a2[0] - a1[0]) / L_land, (a2[1] - a1[1]) / L_land)
    for xx in range(0, 501, 100):
        px, py = a1[0] + u[0] * xx * L_land / ex[-1], a1[1] + u[1] * xx * L_land / ex[-1]
        ax.plot(px, py, "k|", ms=6, mew=1.0)
        ax.annotate(f"{xx}", (px, py), xytext=(6, -4), textcoords="offset points", fontsize=6.5)
    hx = a1[0] + u[0] * x_hw * L_land / ex[-1]; hy = a1[1] + u[1] * x_hw * L_land / ex[-1]
    ax.plot(hx, hy, "o", color="steelblue", ms=5, label=f"high-water line (x = {x_hw:.0f} m)")
    dx_, dy_ = a1[0] + u[0] * DUNE_X * L_land / ex[-1], a1[1] + u[1] * DUNE_X * L_land / ex[-1]
    ax.plot([a1[0], dx_], [a1[1], dy_], "-", color="0.45", lw=6, alpha=0.35, solid_capstyle="butt",
            label="dunes x < 90 m (weakly constrained)")
    ax.text(-18, -48, "dunes (SE)", fontsize=6.5, color="0.25", ha="right")
    ax.text(-480, 380, "North Sea\n(NW)", fontsize=7, style="italic", color="#1f4e9c", ha="center")
    ax.annotate("", xy=(0.93, 0.95), xytext=(0.93, 0.78), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    ax.text(0.93, 0.72, "N", transform=ax.transAxes, ha="center", fontsize=8, fontweight="bold")
    ax.set_xlim(-600, 120); ax.set_ylim(-60, 660)
    location_inset(ax, *lambert72_to_lonlat(gps[1][0], gps[1][1]))
    ax.set_aspect("equal"); ax.set_xlabel(f"Easting − {x0:.0f} m (Belgian Lambert 72)")
    ax.set_ylabel(f"Northing − {y0:.0f} m"); ax.grid(True, alpha=0.25)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0), fontsize=6.5, facecolor="white", edgecolor="0.6")
    ax.text(1.04, 0.5, "Wells K0, K1 (Paepen et al. 2020):\npositions to be added once located", transform=ax.transAxes,
            fontsize=6.5, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="#fff6d5", ec="0.5"))
    ax.set_title("Land and marine ERT profiles, Westhoek (De Panne)"); panel(ax, "a")
    ax = fig.add_subplot(gs[1])
    ax.fill_between(ex, ez, -1.5, color="#e8d9b0", lw=0)
    ax.plot(ex, ez, "k-", lw=0.9, label="land surface")
    ax.plot(ex, ez, "v", color="k", ms=2.2, label=f"electrodes ({len(ex)})")
    ax.axhline(h_sea, color="steelblue", ls="--", lw=1.0, label=f"high tide during marine survey ({h_sea:.2f} mTAW)")
    ax.axhline(-0.15, color="navy", ls=":", lw=1.0, label="low tide during land survey (−0.15 mTAW)")
    ax.axvline(x_hw, color="steelblue", lw=0.8)
    ax.text(x_hw + 3, 6.2, f"high-water line\nx = {x_hw:.0f} m", fontsize=6.5, color="steelblue")
    ax.text(x_lw - 3, 6.2, "low-water line\n(end of line)", fontsize=6.5, ha="right", color="navy")
    ax.set_xlim(-5, 510); ax.set_ylim(-1.5, 8.5); ax.set_aspect(10)
    dune_hatch(ax)
    ax.text(0.99, 0.62, "spring tidal range ≈ 5 m;\nlines show survey-day levels", transform=ax.transAxes, ha="right",
            va="center", fontsize=6.3)
    ax.set_xlabel("Distance along profile x (m)"); ax.set_ylabel("Elevation (mTAW)")
    ve_text(ax, 10); legend_below(ax, ncol=2, dy=-0.32)
    ax.set_title("Topography, electrodes and tide levels"); panel(ax, "b")
    save(fig, "F1_site")
    fn, src = "fig_site", "`data/raw/elec.dat`, `data/raw/land_data.dat`"
    NUM.add("n_electrodes", len(ex), "land electrodes", fn, src)
    NUM.add("electrode_spacing_m", f"{np.median(np.diff(ex)):.1f}", "median electrode spacing (m)", fn, src)
    NUM.add("profile_length_m", f"{ex[-1]:.1f}", "land profile length (m)", fn, src)
    NUM.add("n_quadrupoles", nq, "measured quadrupoles (land)", fn, src,
            ref=("results/deep_all_misfit_report.txt", "n_data", f": {nq}"))
    NUM.add("h_high_tide_mTAW", f"{h_sea:.2f}", "sea level during the marine survey (mean Hsea, elec.dat)", fn, src)
    NUM.add("x_high_water_m", f"{x_hw:.1f}", "high-water line (land surface = marine-survey sea level)", fn, src,
            ref=("results/deep_all_misfit_report.txt", "high-water crossing", f"x = {x_hw:.1f} m"))
    NUM.add("surface_range_mTAW", f"{ez.min():.2f} – {ez.max():.2f}", "land-surface elevation range", fn, src)


# ─────────────────────────────────────────────────────────────────────────────
# F2 — outliers, G1 vs G2
# ─────────────────────────────────────────────────────────────────────────────
def droplist(p):
    return {int(l) for l in open(p) if l.strip() and not l.startswith("#")}


def fig_outliers():
    plt = mpl()
    F = run_npz("deep_all")
    r = (np.log(F["d_obs"]) - np.log(F["d_pred"])) / F["d_err"]
    mad = float(np.median(np.abs(r - np.median(r)))); thr = 3 * 1.4826 * mad
    drop = droplist(os.path.join(wc.RESULTS_DIR, "deep_all_droplist.txt"))
    keep = F["keep_idx"].astype(int)
    is_out = np.array([k in drop for k in keep])
    en, exx = F["elec_num"].astype(int), F["elec_x"]
    q4 = [F[k].astype(int) for k in ("C1", "C2", "P1", "P2")]
    rate, n_use = [], []
    for e in en:
        use = (q4[0] == e) | (q4[1] == e) | (q4[2] == e) | (q4[3] == e)
        n_use.append(use.sum()); rate.append(is_out[use].mean() if use.any() else np.nan)
    rate = np.array(rate)
    _, x_hw, x_lw = p04.water_lines(wc.ELEC_FILE, exx, F["elec_z"])
    flagged = [39, 40, 41, 42, 43, 44, 45, 55, 59, 65, 66]
    ctx = ref_ctx()
    G1, G2 = run_npz("deep_drop_data"), run_npz("deep_drop_elec")
    bl1, bl2 = lens_base(G1, ctx["cs_line"]), lens_base(G2, ctx["cs_line"])
    t1 = lens_base(G1, ctx["cs_tx"]); t2 = lens_base(G2, ctx["cs_tx"])
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), gridspec_kw=dict(height_ratios=[0.8, 1.2], hspace=0.62))
    ax = axes[0]
    fl = np.isin(en, flagged)
    ax.bar(exx[~fl], 100 * rate[~fl], width=4, color="0.55", label="electrodes")
    ax.bar(exx[fl], 100 * rate[fl], width=4, color="#c0392b", label="flagged electrodes (G2 drops their data)")
    ax.axvline(x_hw, color="steelblue", ls="--", lw=1.0, label=f"high-water line (x = {x_hw:.0f} m)")
    ax.axvspan(x_hw, x_lw, color="steelblue", alpha=0.18, lw=0, label="intertidal zone")
    ax.axhline(100 * is_out.mean(), color="k", lw=0.7, ls=":", label=f"all data ({100*is_out.mean():.1f} %)")
    ax.set_xlim(-5, 510); ax.set_ylim(0, max(60, 100 * np.nanmax(rate) + 5))
    dune_hatch(ax, label=False)
    ax.set_ylabel("outlier rate of data\nusing the electrode (%)"); ax.set_xlabel("Electrode position x (m)")
    ax.set_title(f"Per-datum outlier rate by electrode, run F ({len(drop)} of {len(r)} data)")
    legend_below(ax, ncol=3, dy=-0.3); panel(ax, "a")
    ax = axes[1]
    pc = wc.draw_cells(ax, ctx["polys"], G1["model"], "RdBu_r", rho_norm())
    wc.hatch_cells(ax, ctx["polys"], G1["res_radius"] > 10)
    xl = ctx["cs_line"].x
    ax.plot(xl, xwin(xl, bl1), color=COL["logmid"], lw=1.5, label="log-midpoint lens base, G1 (drop outlier data)", zorder=9)
    ax.plot(xl, xwin(xl, bl2), color=COL["logmid"], lw=1.5, ls="--", dashes=(4, 2),
            label="log-midpoint lens base, G2 (drop flagged electrodes)", zorder=9)
    wc.setup_section(ax, None, ve=3, z_floor=-40)
    dune_hatch(ax, label=False)
    ax.set_title("G1 (λ = 20) with the log-midpoint lens base of G1 and G2 (hatched: r > 10 m)")
    cb = fig.colorbar(pc, ax=ax, fraction=0.025, pad=0.01); cb.set_label("ρ (Ω·m)")
    cbar_space(fig, axes[0], pc)
    legend_below(ax, ncol=2, dy=-0.25); panel(ax, "b")
    save(fig, "F2_outliers")
    fn, src = "fig_outliers", "`data/processed/runs/deep_all|deep_drop_data|deep_drop_elec`, `results/deep_all_droplist.txt`"
    NUM.add("outlier_threshold", f"{thr:.3f}", "outlier threshold on normalised log residuals (run F)", fn, src,
            ref=("results/deep_all_outlier_characterisation.txt", "outlier threshold", f"|r| > {thr:.3f}"))
    NUM.add("n_outliers", f"{len(drop)} ({100*len(drop)/len(r):.1f}%)", "per-datum outliers removed (G1)", fn, src,
            ref=("results/deep_all_outlier_characterisation.txt", "outliers", f"{len(drop)} outliers ({100*len(drop)/len(r):.1f}%)"))
    NUM.add("n_G1", int(G1["n_data"]), "data in G1", fn, src,
            ref=("results/deep_drop_data_misfit_report.txt", "n_data", f": {int(G1['n_data'])}"))
    NUM.add("n_G2", int(G2["n_data"]), "data in G2 (flagged electrodes dropped)", fn, src,
            ref=("results/deep_drop_elec_misfit_report.txt", "n_data", f": {int(G2['n_data'])}"))
    sea = exx >= x_hw
    NUM.add("outlier_rate_seaward_vs_landward", f"{100*np.nanmean(rate[sea]):.1f} % vs {100*np.nanmean(rate[~sea]):.1f} %",
            "mean per-electrode outlier rate seaward / landward of the high-water line", fn, src)
    NUM.add("max_electrode_outlier_rate", f"{100*np.nanmax(rate):.0f} % (electrode {en[np.nanargmax(rate)]}, x = "
            f"{exx[np.nanargmax(rate)]:.0f} m)", "highest per-electrode outlier rate", fn, src)
    m = (TX >= 130) & np.isfinite(t1) & np.isfinite(t2)
    d = np.abs(t2 - t1)[m]
    NUM.add("G2_vs_G1_base", f"{np.median(d):.2f} / {d.max():.2f} m", "log-midpoint lens base G2 − G1 (λ = 20), "
            "median / max |Δ| over x = 130–450 m", fn, src)


# ─────────────────────────────────────────────────────────────────────────────
# F3 — reference model
# ─────────────────────────────────────────────────────────────────────────────
def fig_reference():
    plt = mpl()
    import matplotlib.tri as mtri
    ctx = ref_ctx(); ref = ctx["ref"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    pc = wc.draw_cells(ax, ctx["polys"], ref["model"], "RdBu_r", rho_norm())
    cx, cz, rr = ref["cell_x"], ref["cell_z"], ref["res_radius"]
    tri = mtri.Triangulation(cx, cz)
    ex, ez = wc.land_surface()
    xm, zm = cx[tri.triangles].mean(1), cz[tri.triangles].mean(1)
    tri.set_mask((zm > np.interp(xm, ex, ez)) | (np.ptp(cx[tri.triangles], 1) > 25))
    cs = ax.tricontour(tri, rr, levels=[10, 15], colors=["k", "0.35"], linewidths=[1.1, 0.9], linestyles=["-", "--"], zorder=7)
    ax.clabel(cs, fmt={10: "r = 10 m", 15: "r = 15 m"}, fontsize=6.5, inline=True)
    wc.setup_section(ax, None, ve=3, z_floor=-45)
    dune_hatch(ax)
    ax.set_title(f"Reference model g1_lam50 (λ = {float(ref['lam']):g}, χ² = {float(ref['chi2']):.2f}) with "
                 "resolution-radius contours")
    cb = fig.colorbar(pc, ax=ax, fraction=0.025, pad=0.01); cb.set_label("ρ (Ω·m)")
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color="k", lw=1.1), Line2D([], [], color="0.35", lw=0.9, ls="--")]; l += ["r = 10 m", "r = 15 m"]
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6.3, facecolor="white")
    save(fig, "F3_reference")
    fn, src = "fig_reference", "`data/processed/runs/g1_lam50/westhoek_inversion_results.npz`"
    NUM.add("ref_chi2", f"{float(ref['chi2']):.3f}", "reference χ²", fn, src, ref=("data/processed/runs/g1_lam50/westhoek_inversion_results.npz", "npz:chi2:.3f", f"{float(ref['chi2']):.3f}"))
    NUM.add("ref_lambda", f"{float(ref['lam']):g}", "reference λ", fn, src)
    NUM.add("ref_n_iter", int(ref["n_iter"]), "Gauss–Newton iterations", fn, src,
            ref=("data/processed/runs/g1_lam50/westhoek_inversion_results.npz", "npz:n_iter:d", f"{int(ref['n_iter'])}"))
    NUM.add("ref_n_cells", len(ref["model"]), "paraDomain cells", fn, src, ref=("data/processed/runs/g1_lam50/westhoek_inversion_results.npz", "npz:len(model):d", f"{len(ref['model'])}"))
    s = np.interp(cx, ex, ez); top30 = (s - cz) <= 30
    for rmax in (10, 15):
        NUM.add(f"ref_share_r{rmax}_top30", f"{100*np.mean(rr[top30] <= rmax):.0f} %",
                f"share of cells within 30 m of the surface with r ≤ {rmax} m", fn, src)
    lens = (cx >= 130) & (cx <= 370) & (cz > s - 10)
    NUM.add("ref_rho_lens_top10m", f"{np.median(ref['model'][lens]):.2f}", "median ρ in the upper 10 m, x = 130–370 m (Ω·m)", fn, src)


# ─────────────────────────────────────────────────────────────────────────────
# F4 — linearised UQ
# ─────────────────────────────────────────────────────────────────────────────
def fig_linearised():
    plt = mpl()
    ctx = ref_ctx(); ref = ctx["ref"]
    v3 = np.load(LIN_V3); v4 = np.load(LIN_V4)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), gridspec_kw=dict(hspace=0.42))
    ax = axes[0]
    import matplotlib.colors as mcolors
    pc = wc.draw_cells(ax, ctx["polys"], v3["P_lt3_pooled"], "Blues", mcolors.Normalize(0, 1))
    wc.hatch_cells(ax, ctx["polys"], ref["res_radius"] > 10, label="r > 10 m")
    wc.setup_section(ax, None, ve=3, z_floor=-45)
    dune_hatch(ax)
    ax.set_title("P(ρ < 3 Ω·m), linearised posterior pooled over λ = 5–100")
    cb = fig.colorbar(pc, ax=ax, fraction=0.025, pad=0.01); cb.set_label("probability")
    legend_below(ax, ncol=1, dy=-0.2); panel(ax, "a")
    ax = axes[1]
    x = v4["x"]
    for key, col, lab in (("pooled_thr3", COL["thr3"], "3 Ω·m contour"), ("pooled_logmid", COL["logmid"], "log-midpoint")):
        b = v4[key]; ok = (b[0] >= 0.5) & (x >= 130) & (x <= 450)
        ax.fill_between(x, np.where(ok, b[2], np.nan), np.where(ok, b[3], np.nan), color=col, alpha=0.22, lw=0)
        ax.plot(x, np.where(ok, b[1], np.nan), color=col, lw=1.4, label=f"{lab}: median, 90 % (λ-pooled)")
    ex, ez = wc.land_surface()
    ax.plot(ex, ez, "k-", lw=0.8, label="land surface")
    ax.set_xlim(-5, 510); ax.set_ylim(-32, 8); ax.set_aspect(3)
    dune_hatch(ax)
    ax.set_xlabel("Distance along profile (m)"); ax.set_ylabel("Elevation (mTAW)"); ve_text(ax, 3)
    ax.set_title("Lens base: 3 Ω·m contour vs log-midpoint (linearised, λ-pooled)")
    cbar_space(fig, ax, pc)
    legend_below(ax, ncol=3, dy=-0.36); panel(ax, "b")
    save(fig, "F4_linearised")
    fn, src = "fig_linearised", "`results/runs/g1_lam50_v4/interface_definitions_bands.npz`"
    rep = "results/runs/g1_lam50_v4/interface_definitions_report.md"
    fmt = lambda b: f"{b[1]:.2f} [{b[2]:.2f}, {b[3]:.2f}] ({100*b[0]:.0f}%)"
    for xx in XREP:
        j = int(np.argmin(np.abs(x - xx)))
        NUM.add(f"lin_logmid_pooled_x{xx}", fmt(v4["pooled_logmid"][:, j]), f"linearised log-midpoint base, λ-pooled, x = {xx} m "
                "(median [P5, P95] (share present))", fn, src, ref=(rep, f"| {xx} |", fmt(v4["pooled_logmid"][:, j])))
        NUM.add(f"lin_thr3_single_x{xx}", fmt(v4["single_thr3"][:, j]), f"3 Ω·m base, λ = 50, x = {xx} m", fn, src,
                ref=(rep, f"| {xx} |", fmt(v4["single_thr3"][:, j])))


# ─────────────────────────────────────────────────────────────────────────────
# synthetic benchmark (F5, T2)
# ─────────────────────────────────────────────────────────────────────────────
def synthetic_rows():
    """A2 runs: per (truth, definition, λ or pooled, realisation, column) coverage / bias / width, grouped by r at the
    λ = 50 crossing of the same definition."""
    spec = sb.load_spec(); cols = sb.COLS
    truths = {t: c11.truth_defs(spec, t, cols) for t in c11.A2_TRUTHS}
    rows = []
    for t in c11.A2_TRUTHS:
        for r in range(sb.N_REAL):
            p = os.path.join(c11.OUT_DATA, "a2_runs", f"{t}_r{r:02d}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p, allow_pickle=True)
            for dname in c11.DEFS:
                tv = truths[t][dname]; r50 = d[f"lam50_r_{dname}"]
                for lam in ("50", "5", "10", "20", "100", "pooled"):
                    key = f"lam{lam}_band_{dname}" if lam != "pooled" else f"pooled_band_{dname}"
                    if key not in d.files:
                        continue
                    bb = d[key]; mh = d[f"lam{lam}_mhat_{dname}"] if lam != "pooled" else None
                    for j in range(len(cols)):
                        if not np.isfinite(tv[j]) or not np.isfinite(r50[j]):
                            continue
                        g = "≤10" if r50[j] <= 10 else ("10–15" if r50[j] <= 15 else None)
                        if g is None:
                            continue
                        rows.append(dict(truth=t, r=r, d=dname, lam=lam, grp=g,
                                         cov=bool(bb[0, j] >= 0.5 and bb[2, j] <= tv[j] <= bb[3, j]),
                                         bias=(mh[j] - tv[j]) if mh is not None else np.nan,
                                         w=bb[3, j] - bb[2, j] if bb[0, j] >= 0.5 else np.nan))
    return rows


def a2_summary(rows):
    """A2 report convention: rows grouped by r at the definition's OWN λ = 50 crossing; mean over the four truths."""
    out = {}
    for grp in ("≤10", "10–15"):
        for dname in c11.DEFS:
            per = []
            for t in c11.A2_TRUTHS:
                rs = [x for x in rows if x["truth"] == t and x["d"] == dname and x["lam"] == "50" and x["grp"] == grp]
                if rs:
                    per.append((np.mean([x["cov"] for x in rs]), np.nanmedian([x["bias"] for x in rs]),
                                np.nanmedian([x["w"] for x in rs])))
            pooled = [x for x in rows if x["d"] == dname and x["lam"] == "pooled" and x["grp"] == grp]
            out[(grp, dname)] = dict(cov=np.mean([p[0] for p in per]), absbias=np.mean([abs(p[1]) for p in per]),
                                     width=np.median([p[2] for p in per]),
                                     cov_pooled=np.mean([x["cov"] for x in pooled]) if pooled else np.nan,
                                     width_pooled=np.nanmedian([x["w"] for x in pooled]) if pooled else np.nan)
    return out


def reliability():
    fwd = np.load(os.path.join(sb.SYN_DATA, "truth_forward.npz"))
    rel = {"single": [], "pooled": []}
    for t in sb.TRUTHS:
        lt3 = fwd[f"cell_log10_{t}"] < np.log10(3)
        for r in range(sb.N_REAL):
            p = os.path.join(sb.SYN_DATA, "runs", f"{t}_r{r:02d}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p, allow_pickle=True)
            rel["single"].append((d["lam50_P3"], lt3, d["lam50_radius"]))
            if "pooled_P3" in d.files:
                rel["pooled"].append((d["pooled_P3"], lt3, d["lam50_radius"]))
    bins = np.linspace(0, 1, 11); out = {}
    for mode in ("single", "pooled"):
        P = np.concatenate([a for a, _, _ in rel[mode]]); T = np.concatenate([b for _, b, _ in rel[mode]])
        R = np.concatenate([c for _, _, c in rel[mode]])
        for gname, lo, hi in (("≤10", 0, 10), ("10–15", 10, 15)):
            s = (R > lo) & (R <= hi); idx = np.clip(np.digitize(P[s], bins) - 1, 0, 9)
            out[(mode, gname)] = ([P[s][idx == i].mean() if (idx == i).any() else np.nan for i in range(10)],
                                  [T[s][idx == i].mean() if (idx == i).any() else np.nan for i in range(10)],
                                  [int((idx == i).sum()) for i in range(10)])
    return out


def a1_bias():
    """A1 convention (Stage-4 runs, all nine truths, 3 Ω·m base, realisations 0–4, grouped by r at λ = 50)."""
    fwd = np.load(os.path.join(sb.SYN_DATA, "truth_forward.npz"))
    rows = []
    for t in sb.TRUTHS:
        tb = fwd[f"base_{t}"]
        for r in range(sb.N_REAL_LAMBDA):
            d = np.load(os.path.join(c11.S4_RUNS, f"{t}_r{r:02d}.npz"), allow_pickle=True)
            r50 = d["lam50_r_base"]
            for lam in (5, 10, 20, 50, 100):
                mh = d[f"lam{lam}_mhat_base"]
                for j in range(len(sb.COLS)):
                    if not np.isfinite(tb[j]) or not np.isfinite(r50[j]):
                        continue
                    g = "≤10" if r50[j] <= 10 else ("10–15" if r50[j] <= 15 else None)
                    if g:
                        rows.append((lam, g, mh[j] - tb[j]))
    return {(lam, g): float(np.nanmedian([b for l_, g_, b in rows if l_ == lam and g_ == g]))
            for lam in (5, 10, 20, 50, 100) for g in ("≤10", "10–15")}


def fig_synthetic():
    plt = mpl()
    rows = synthetic_rows(); summ = a2_summary(rows); relb = reliability(); a1 = a1_bias()
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.6), gridspec_kw=dict(hspace=0.62, wspace=0.3))
    cols = COL
    names = {"thr3": "3 Ω·m", "maxgrad": "max-gradient", "logmid": "log-midpoint"}
    ax = axes[0, 0]; w = 0.25
    for k, dname in enumerate(c11.DEFS):
        for gi, grp in enumerate(("≤10", "10–15")):
            s_ = summ[(grp, dname)]
            ax.bar(gi + (k - 1) * w, 100 * s_["cov"], w * 0.9, color=cols[dname], label=names[dname] if gi == 0 else None)
            ax.plot(gi + (k - 1) * w, 100 * s_["cov_pooled"], "k_", ms=9, mew=1.6,
                    label="λ-pooled" if (gi == 0 and k == 0) else None)
    ax.axhline(90, color="k", ls="--", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["r ≤ 10 m", "10 < r ≤ 15 m"]); ax.set_ylim(0, 105)
    ax.set_ylabel("coverage of 90 % bands (%)"); ax.set_title("Coverage by definition (λ = 50)")
    legend_below(ax, ncol=4, dy=-0.16, fontsize=6); panel(ax, "a")
    ax = axes[0, 1]
    for k, dname in enumerate(c11.DEFS):
        for gi, grp in enumerate(("≤10", "10–15")):
            bs = [np.nanmedian([x["bias"] for x in rows if x["truth"] == t and x["d"] == dname and x["lam"] == "50"
                                and x["grp"] == grp]) for t in c11.A2_TRUTHS]
            ax.bar(gi + (k - 1) * w, np.mean(bs), w * 0.9, color=cols[dname])
            ax.plot(np.full(len(bs), gi + (k - 1) * w), bs, "k.", ms=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["r ≤ 10 m", "10 < r ≤ 15 m"])
    ax.set_ylabel("median bias m̂ − true (m)\n(− = too deep)"); ax.set_title("Bias by definition (dots: truths)")
    panel(ax, "b")
    lams = [5, 10, 20, 50, 100]
    for col_i, (key, ylab, ttl) in enumerate((("cov", "coverage (%)", "Coverage vs λ"), ("bias", "median bias (m)", "Bias vs λ"))):
        ax = axes[1, col_i]
        for dname in ("thr3", "logmid"):
            for grp, ls in (("≤10", "-"), ("10–15", "--")):
                sel = lambda l: [x for x in rows if x["d"] == dname and x["lam"] == str(l) and x["grp"] == grp and x["r"] < 5]
                v = [100 * np.mean([x["cov"] for x in sel(l)]) if key == "cov" else np.nanmedian([x["bias"] for x in sel(l)])
                     for l in lams]
                ax.plot(lams, v, ls, marker="o", ms=3, color=cols[dname], label=f"{names[dname]}, r {grp} m")
        ax.axhline(90 if key == "cov" else 0, color="k", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_xticks(lams); ax.set_xticklabels([str(l) for l in lams]); ax.minorticks_off()
        ax.set_xlabel("λ"); ax.set_ylabel(ylab); ax.set_title(f"{ttl} (realisations 0–4)"); ax.grid(True, alpha=0.3)
        if key == "cov":
            ax.set_ylim(0, 105); ax.legend(loc="lower left", fontsize=5.8, facecolor="white", framealpha=0.95)
        panel(ax, "cd"[col_i])
    for col_i, grp in enumerate(("≤10", "10–15")):
        ax = axes[2, col_i]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect reliability")
        for mode, col, ls in (("single", "k", "-"), ("pooled", "0.5", "--")):
            mp, ob, n = relb[(mode, grp)]
            ax.plot(mp, ob, ls, marker="o", ms=3, color=col, label="λ = 50" if mode == "single" else "λ-pooled")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("predicted P(ρ < 3 Ω·m)"); ax.set_ylabel("observed frequency")
        ax.set_title(f"Reliability, cells with r {grp.replace('≤', '≤ ')} m"); ax.grid(True, alpha=0.3)
        wc.white_legend(ax, loc="upper left", fontsize=6)
        panel(ax, "ef"[col_i])
    save(fig, "F5_synthetic")
    fn = "fig_synthetic"; src = "`data/processed/synthetic_s5/a2_runs/*.npz`, `data/processed/synthetic/truth_spec.json`"
    rep2 = "results/synthetic_s5/a2_interface_definitions.md"
    for grp in ("≤10", "10–15"):
        line = f"**All four truths, r {grp} m:**"
        for dname in c11.DEFS:
            s = summ[(grp, dname)]
            NUM.add(f"syn_cov_{dname}_r{'10' if grp == '≤10' else '10-15'}", f"{100*s['cov']:.0f}%",
                    f"synthetic coverage, {names[dname]}, λ = 50, r {grp} m (mean over T1, T1h, T2a, T3)", fn, src,
                    ref=(rep2, line, f"{dname}: coverage {100*s['cov']:.0f}%"))
            NUM.add(f"syn_absbias_{dname}_r{'10' if grp == '≤10' else '10-15'}", f"{s['absbias']:.2f} m",
                    f"synthetic |bias|, {names[dname]}, λ = 50, r {grp} m (mean over truths)", fn, src,
                    ref=(rep2, line, f"{dname}: coverage {100*s['cov']:.0f}% (mean over truths), |bias| {s['absbias']:.2f} m"))
    srcA1 = "`data/processed/synthetic/runs/*.npz` (Stage-4 λ pool, all nine truths)"
    for grp in ("≤10", "10–15"):
        row = "| **all** | " + " | ".join(f"{a1[(l, grp)]:+.2f}" for l in (5, 10, 20, 50, 100)) + " |"
        NUM.add(f"a1_bias_vs_lambda_r{'10' if grp == '≤10' else '10-15'}", " / ".join(f"{a1[(l, grp)]:+.2f}" for l in (5, 10, 20, 50, 100)),
                f"median bias of the 3 Ω·m base at λ = 5/10/20/50/100, r {grp} m (m)", fn, srcA1,
                ref=("results/synthetic_s5/a1_bias_vs_lambda.md", "| **all** |", row))
    rep = "results/synthetic/benchmark_report.md"
    for mode in ("single", "pooled"):
        mp, ob, n = relb[(mode, "≤10")]
        NUM.add(f"rel_{mode}_r10_top", f"{ob[9]:.3f} (n = {n[9]})", f"observed frequency in the P(ρ<3) 0.9–1.0 bin, "
                f"{'λ = 50' if mode == 'single' else 'λ-pooled'}, r ≤ 10 m", fn, "`data/processed/synthetic/runs/*.npz`",
                ref=(rep, "| 0.9–1.0 |", f"{n[9]} | {ob[9]:.3f}"))
        NUM.add(f"rel_{mode}_r10_bottom", f"{ob[0]:.3f} (n = {n[0]})", f"observed frequency in the 0.0–0.1 bin, "
                f"{'λ = 50' if mode == 'single' else 'λ-pooled'}, r ≤ 10 m", fn, "`data/processed/synthetic/runs/*.npz`",
                ref=(rep, "| 0.0–0.1 |", f"{n[0]} | {ob[0]:.3f}"))
    return rows, summ


# ─────────────────────────────────────────────────────────────────────────────
# BEL products (F6, F7, T2, T3)
# ─────────────────────────────────────────────────────────────────────────────
def bel_products(recompute=False):
    p = os.path.join(DPAP, "bel_products.npz")
    if os.path.exists(p) and not recompute:
        z = np.load(p, allow_pickle=True)
        return {k: (z[k].item() if z[k].dtype == object and z[k].shape == () else z[k]) for k in z.files}
    t0 = time.time()
    s6.init(); s6.TX = TX
    obs, truths = b13.observed_sets()
    C0 = s6.load_combined()                                   # pre-freeze headline: site subset only (field only)
    out = dict(n_train_site=len(C0["tr"]["D"]), n_held_site=len(C0["ho"]["D"]))
    out["Pf_site"] = b13.BEL(C0["tr"]["D"], C0["tr"]["T"]).posterior(obs["field"], 2000,
                                                                      np.random.default_rng([129000, 1]), "kde")
    del C0
    C = s6.load_combined(physical=True)                       # v1.0 headline: site + physical
    D, T, th = C["tr"]["D"], C["tr"]["T"], C["tr"]["th"]
    bel = b13.BEL(D, T)
    out.update(n_train=len(D), n_held=len(C["ho"]["D"]), kd=bel.pd.k, kt=bel.pt.k)
    out["Pf"] = bel.posterior(obs["field"], 2000, np.random.default_rng([129000, 1]), "kde")
    cov, rank, wid, bias = [], [], [], []
    for j, (y, t) in enumerate(zip(C["ho"]["D"], C["ho"]["T"])):
        P = bel.posterior(y, 2000, np.random.default_rng([129000, 90_000 + j]), "kde")
        lo, md, hi = np.percentile(P, [5, 50, 95], 0)
        cov.append((t >= lo) & (t <= hi)); rank.append(np.mean(P < t, 0)); wid.append(hi - lo); bias.append(md - t)
    out.update(ho_cov=np.array(cov), ho_rank=np.array(rank), ho_wid=np.array(wid), ho_bias=np.array(bias))
    print(f"[bel] held-out done ({time.time()-t0:.0f} s)", flush=True)
    # falsification in the BEL data-PCA space
    sc = bel.Sd; var = sc.var(0, ddof=1)
    sf = bel.pd.fwd(obs["field"])[0]
    mu, Cm = b13.pilot.fast_mcd(sc); Ci = np.linalg.inv(Cm)
    out.update(t2_train=np.sum(sc ** 2 / var, 1), t2_field=float(np.sum(sf ** 2 / var)),
               t2r_train=np.einsum("ij,jk,ik->i", sc - mu, Ci, sc - mu), t2r_field=float((sf - mu) @ Ci @ (sf - mu)),
               q_train=bel.pd.resid(D), q_field=float(bel.pd.resid(obs["field"])[0]))
    # Stage-4 truths and the twin
    spec4 = sb.load_spec(); cols = TX >= 130
    tstats = {}
    for ti, t in enumerate(("T1", "T1h", "T2a", "T3")):
        tv = c11.truth_defs(spec4, t, TX)["logmid"]
        cv, bi, wd = [], [], []
        for r in range(20):
            y = truths[(t, r)]
            if y is None:
                continue
            P = bel.posterior(y, 2000, np.random.default_rng([SEED, 20_000 + 100 * ti + r]), "kde")
            lo, md, hi = np.percentile(P, [5, 50, 95], 0)
            cv.append(((tv >= lo) & (tv <= hi))[cols]); bi.append((md - tv)[cols]); wd.append((hi - lo)[cols])
        tstats[t] = dict(n=len(cv), cov=float(np.mean(cv)), bias=float(np.median(bi)), width=float(np.median(wd)))
    th_tw = json.load(open(os.path.join(s15.D4, "twin_theta.json")))
    pm = b13.PriorModel2(th_tw)
    R_tw = np.load(os.path.join(s15.D4, "twin_forward.npz"))["R"]
    tw_truth = {"twin (own target)": pm.target(), "twin vs T1 b(x)": sb.Truth(spec4, "T1").base(TX)}
    acc = {k: ([], [], []) for k in tw_truth}
    for r in range(20):
        Rn = R_tw + np.sqrt((0.03 * R_tw) ** 2 + 1e-8) * np.random.default_rng(1000 + r).standard_normal(len(R_tw))
        if not np.all(Rn > 0):
            continue
        P = bel.posterior(np.log10(Rn), 2000, np.random.default_rng([SEED, 30_000 + r]), "kde")
        lo, md, hi = np.percentile(P, [5, 50, 95], 0)
        for k, tv in tw_truth.items():
            acc[k][0].append(((tv >= lo) & (tv <= hi))[cols]); acc[k][1].append((md - tv)[cols]); acc[k][2].append((hi - lo)[cols])
    for k, (cv, bi, wd) in acc.items():
        tstats[k] = dict(n=len(cv), cov=float(np.mean(cv)), bias=float(np.median(bi)), width=float(np.median(wd)))
    out["truth_stats"] = tstats
    print(f"[bel] truths + twin done ({time.time()-t0:.0f} s)", flush=True)
    # ρ_below-conditional BEL
    rb = np.array([t["rho_below"] for t in th]); rbh = np.array([t["rho_below"] for t in C["ho"]["th"]])
    bins = ((2, 4), (4, 8), (8, 20)); Pb, bcov, bn = [], [], []
    for k, (lo_, hi_) in enumerate(bins):
        m = (rb >= lo_) & (rb < hi_); mh = np.where((rbh >= lo_) & (rbh < hi_))[0]
        bk = b13.BEL(D[m], T[m])
        Pb.append(bk.posterior(obs["field"], 2000, np.random.default_rng([SEED, 40 + k]), "kde"))
        cc = []
        for j in mh:
            P = bk.posterior(C["ho"]["D"][j], 2000, np.random.default_rng([SEED, 50_000 + 1000 * k + int(j)]), "kde")
            lo, hi = np.percentile(P, [5, 95], 0); cc.append((C["ho"]["T"][j] >= lo) & (C["ho"]["T"][j] <= hi))
        bcov.append(float(np.mean(cc))); bn.append((int(m.sum()), len(mh)))
    out.update(Pbins=np.array(Pb), bins_cov=np.array(bcov), bins_n=np.array(bn), bins=np.array(bins))
    print(f"[bel] ρ_below bins done ({time.time()-t0:.0f} s)", flush=True)
    # second BEL (D1, D2, logit-bounded log10 ρ) on site + physical
    pY = os.path.join(DPAP, "second_bel_targets_phys.npz")
    if os.path.exists(pY) and not recompute:
        zY = np.load(pY); Ytr, Yho = zY["Ytr"], zY["Yho"]
    else:
        Ytr = np.array([g14.derived_targets(t) for t in th]); Yho = np.array([g14.derived_targets(t) for t in C["ho"]["th"]])
        np.savez(pY, Ytr=Ytr, Yho=Yho)
    s15.RHO_BOUNDS.update(rho_lens=(0.4, 3.0), rho_below=(2.0, 20.0), rho_discharge=(2.5, 150.0))
    b2 = s15.BEL2(D, Ytr, bounded=True)
    out["P2"] = b2.posterior(obs["field"], 2000, np.random.default_rng([SEED, 3]))
    c2 = []
    for j, y in enumerate(C["ho"]["D"]):
        P = b2.posterior(y, 2000, np.random.default_rng([SEED, 60_000 + j])); lo, hi = np.percentile(P, [5, 95], 0)
        c2.append((Yho[j] >= lo) & (Yho[j] <= hi))
    out["cov2"] = np.array(c2); out["Y2_train"] = Ytr
    print(f"[bel] second BEL done ({time.time()-t0:.0f} s)", flush=True)
    # full prior v3 (no site filter), field only
    Cf = s6.load_combined(site_only=False)
    out["Pf_full"] = b13.BEL(Cf["tr"]["D"], Cf["tr"]["T"]).posterior(obs["field"], 2000, np.random.default_rng([SEED, 2]), "kde")
    out["n_train_full"] = len(Cf["tr"]["D"])
    # prior examples (site training samples)
    ex_i = np.random.default_rng([SEED, 7]).choice(len(th), 3, replace=False)
    out["examples"] = json.dumps([th[i] for i in ex_i])
    out["runtime_s"] = time.time() - t0
    np.savez(p, **{k: (np.array(v, dtype=object) if isinstance(v, dict) else v) for k, v in out.items()})
    print(f"[bel] products cached ({out['runtime_s']:.0f} s)", flush=True)
    return out


def fig_bel_calibration(B):
    plt = mpl()
    exs = json.loads(str(B["examples"]))
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 0.09, 1.0, 1.0], hspace=0.7, wspace=0.35)
    ex, ez = wc.land_surface()
    xg = np.arange(0, 505.5, 1.0); zg = np.arange(8.0, -45.01, -0.25)
    X, Z = np.meshgrid(xg, zg)
    for i, th in enumerate(exs):
        ax = fig.add_subplot(gs[0, i])
        pm = b13.PriorModel2(th)
        V = pm.log10rho(X.ravel(), Z.ravel()).reshape(X.shape); V[Z > np.interp(X, ex, ez)] = np.nan
        pc = ax.pcolormesh(xg, zg, 10 ** V, cmap="RdBu_r", norm=rho_norm(), shading="auto", rasterized=True)
        ax.plot(TX, pm.target(), "k-", lw=0.9)
        ax.plot(ex, ez, "k-", lw=0.6)
        ax.set_xlim(0, 505); ax.set_ylim(-45, 9); ax.set_aspect("auto")
        dune_hatch(ax, text=False)
        ax.set_title(f"sample {i+1}" + (", discharge zone" if th["dis_present"] else ""), fontsize=7)
        ax.tick_params(labelsize=6); ax.set_xlabel("x (m)", fontsize=6.5)
        if i == 0:
            ax.set_ylabel("mTAW"); panel(ax, "a")
    cax = fig.add_subplot(gs[1, :])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal"); cb.set_label("ρ (Ω·m); black line: lens-base target; "
                                                                           "grey hatch: dunes x < 90 m (weakly constrained)", fontsize=6.5)
    cax.tick_params(labelsize=6)
    for k, (key, fkey, lab, ttl) in enumerate((("t2_train", "t2_field", "Hotelling T² (data PCA)", "T²"),
                                               ("t2r_train", "t2r_field", "robust T² (FastMCD)", "robust T²"),
                                               ("q_train", "q_field", "Q (residual outside the PCA)", "Q"))):
        ax = fig.add_subplot(gs[2, k])
        v = B[key]; fv = B[fkey]
        bins = np.logspace(np.log10(v.min()), np.log10(max(v.max(), fv)), 45)
        ax.hist(v, bins=bins, color="0.6"); ax.set_xscale("log")
        ax.axvline(fv, color="navy", lw=1.5, label=f"field ({100*np.mean(v < fv):.0f}th pct.)")
        ax.axvline(np.percentile(v, 99), color="#c0392b", lw=0.9, ls="--", label="training P99")
        ax.set_xlabel(lab); ax.set_title(ttl)
        if k == 0:
            ax.set_ylabel("training samples"); panel(ax, "b")
        wc.white_legend(ax, fontsize=5.6, loc="upper right")
    ax = fig.add_subplot(gs[3, 0:2])
    rk = B["ho_rank"].ravel()
    ax.hist(rk, bins=20, range=(0, 1), color="#1f77b4", alpha=0.85)
    ax.axhline(rk.size / 20, color="k", ls="--", lw=0.9, label="uniform")
    ax.set_xlabel("rank of the true lens base among 2000 posterior samples"); ax.set_ylabel("count")
    ax.set_title(f"Held-out rank histogram ({B['ho_rank'].shape[0]} samples × {len(TX)} columns)")
    wc.white_legend(ax, fontsize=6, loc="lower center"); panel(ax, "c")
    ax = fig.add_subplot(gs[3, 2])
    ax.plot(TX, 100 * B["ho_cov"].mean(0), "-o", ms=2.5, color="#1f77b4")
    ax.axhspan(85, 95, color="green", alpha=0.12); ax.axhline(90, color="k", lw=0.7, ls=":")
    ax.set_ylim(60, 100); ax.set_xlabel("x (m)"); ax.set_ylabel("coverage (%)"); ax.set_title("Held-out coverage")
    ax.grid(True, alpha=0.3); panel(ax, "d")
    save(fig, "F6_bel_calibration")
    from scipy.stats import chisquare
    fn, src = "bel_products / fig_bel_calibration", "`data/processed/bel/v2/train`, `data/processed/bel_prior_v3/train` (site + physical subset)"
    rep = "results/stage6/1_prior_v3_refit.md"
    NUM.add("bel_n_train_site_v3", f"{B['n_train_site']} / {B['n_held_site']}", "training / held-out samples, site subset "
            "(pre-freeze)", fn, src, ref=(rep, "site subset", f"{B['n_train_site']} training / {B['n_held_site']} held-out"))
    NUM.add("bel_n_train_site_phys", f"{B['n_train']} / {B['n_held']}", "training / held-out samples, site + physical "
            "subset (v1.0 headline)", fn, src)
    NUM.add("bel_kd_kt", f"{B['kd']} / {B['kt']}", "data / target PCA components (99 % variance)", fn, src)
    NUM.add("bel_heldout_coverage", f"{100*B['ho_cov'].mean():.1f} %", "BEL held-out 90 % coverage (lens base; site + physical)",
            fn, src)
    pc_ = 100 * B["ho_cov"].mean(0)
    NUM.add("bel_heldout_cov_per_column", f"{pc_.min():.0f}–{pc_.max():.0f} %", "per-column held-out coverage range", fn, src)
    h = np.histogram(rk, bins=10, range=(0, 1))[0]
    NUM.add("bel_rank_chi2", f"{chisquare(h).statistic:.1f} (p = {chisquare(h).pvalue:.2g})",
            "rank-histogram χ² (10 bins; pooled over columns, not independent)", fn, src)
    NUM.add("bel_heldout_bias_width", f"{np.median(B['ho_bias']):+.2f} / {np.median(B['ho_wid']):.2f} m",
            "held-out median bias / median 90 % width", fn, src)
    for nm, a, b_ in (("t2", B["t2_train"], B["t2_field"]), ("t2_robust", B["t2r_train"], B["t2r_field"]),
                      ("q", B["q_train"], B["q_field"])):
        NUM.add(f"falsify_{nm}_pct", f"{100*np.mean(a < b_):.1f}", f"field percentile of {nm.replace('_', ' ')} among the "
                "training samples (site + physical, prior v3)", fn, src)


def lens_presence():
    """P(lens present) at x = 100–120 m: the Gate-A classifier (prior v2, 14_gateA_sensitivity.cmd_nuisance) reproduced —
    x_L is not affected by the site / physical filters. Cached in DPAP/lens_presence_v2.json."""
    p = os.path.join(DPAP, "lens_presence_v2.json")
    if os.path.exists(p) and not RECOMPUTE:
        return json.load(open(p, encoding="utf-8"))
    A = g14.load_all(); bel = b13.BEL(A["Dtr"], A["Ttr"])
    obs, _ = b13.observed_sets()
    sd = np.sqrt(bel.Sd.var(0, ddof=1))
    Xtr = bel.Sd / sd; Xho = bel.pd.fwd(A["Dho"]) / sd; xf = bel.pd.fwd(obs["field"]) / sd
    thtr = [A["th"][i] for i in A["ktr"]]; thho = [A["th"][i] for i in A["kho"]]
    out = {}
    for x in (100, 110, 120):
        ytr = np.array([float(t["x_L"] <= x) for t in thtr]); yho = np.array([float(t["x_L"] <= x) for t in thho])
        predict, C, _ = g14.logistic_cv(Xtr, ytr)
        ph = predict(Xho); brier = float(np.mean((ph - yho) ** 2)); p0 = float(yho.mean())
        out[str(x)] = dict(p=float(predict(xf)[0]), brier=brier, noskill=p0 * (1 - p0), C=float(C))
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    return out


def fig_field_base(B):
    plt = mpl()
    v4 = np.load(LIN_V4)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), gridspec_kw=dict(hspace=0.5))
    ex, ez = wc.land_surface()
    surf = lambda x: np.interp(x, ex, ez)
    clip = lambda x, v: np.minimum(v, surf(x))                      # bands and medians clipped at the land surface
    pres = lens_presence()
    h = TX >= 130
    ax = axes[0]
    x4 = v4["x"]; b = v4["pooled_logmid"]; ok = b[0] >= 0.5
    ax.fill_between(x4, np.where(ok, clip(x4, b[2]), np.nan), np.where(ok, clip(x4, b[3]), np.nan), color=COL["logmid"],
                    alpha=0.2, lw=0)
    ax.plot(x4, np.where(ok, clip(x4, b[1]), np.nan), color=COL["logmid"], lw=1.2,
            label="linearised log-midpoint, λ-pooled: median, 90 %")
    lo, md, hi = np.percentile(B["Pf"], [5, 50, 95], 0)
    ax.fill_between(TX[h], clip(TX[h], lo[h]), clip(TX[h], hi[h]), color=COL["bel"], alpha=0.25, lw=0)
    ax.plot(TX[h], clip(TX[h], md[h]), color=COL["bel"], lw=1.7, label="BEL-KDE (site + physical, prior v3): median, 90 %")
    xs = np.array([100.0, 110.0, 120.0]); ps = [pres[str(int(x))]["p"] for x in xs]
    ax.plot(xs, surf(xs) + 1.5, "v", color=COL["bel"], ms=4.5, mec="k", mew=0.4,
            label=f"P(lens present), x = 100–120 m: {min(ps):.2f}")
    ax.plot(ex, ez, "k-", lw=0.8, label="land surface")
    ax.set_xlim(-5, 510); ax.set_ylim(-32, 9); ax.set_aspect(3); dune_hatch(ax)
    ax.set_ylabel("Elevation (mTAW)"); ve_text(ax, 3)
    ax.set_title("Field lens base: BEL vs linearised")
    legend_below(ax, ncol=2, dy=-0.14); panel(ax, "a")
    ax = axes[1]
    styles = [("#b2182b", "-"), ("#1b9e77", "--"), ("#8c6d31", "-.")]
    for k, ((lo_, hi_), (col, ls)) in enumerate(zip(B["bins"], styles)):
        l, m, hh = np.percentile(B["Pbins"][k], [5, 50, 95], 0)
        ax.fill_between(TX[h], clip(TX[h], l[h]), clip(TX[h], hh[h]), color=col, alpha=0.14, lw=0)
        ax.plot(TX[h], clip(TX[h], m[h]), color=col, ls=ls, lw=1.5,
                label=f"ρ_below {lo_}–{hi_} Ω·m (held-out cov. {100*B['bins_cov'][k]:.0f} %)")
    ax.plot(TX[h], clip(TX[h], md[h]), color=COL["bel"], ls=":", lw=1.6, label="all ρ_below (panel a)")
    ax.plot(ex, ez, "k-", lw=0.8)
    ax.set_xlim(-5, 510); ax.set_ylim(-32, 9); ax.set_aspect(3); dune_hatch(ax, label=False)
    ax.set_xlabel("Distance along profile (m)"); ax.set_ylabel("Elevation (mTAW)"); ve_text(ax, 3)
    ax.set_title("BEL lens base conditional on the resistivity of the water below the lens")
    legend_below(ax, ncol=2, dy=-0.3); panel(ax, "b")
    save(fig, "F7_field_lens_base")
    fn, src = "bel_products / fig_field_base", "`data/processed/bel/v2/train`, `data/processed/bel_prior_v3/train` (site + physical subset)"
    rep = "results/stage6/1_prior_v3_refit.md"
    for xx in XREP:
        j = int(np.argmin(np.abs(TX - xx)))
        NUM.add(f"bel_base_x{xx}", q(B["Pf"][:, j]), f"BEL lens base (site + physical, prior v3), x = {xx} m, median "
                "[P5, P95] (mTAW)", fn, src)
        NUM.add(f"bel_base_site_only_x{xx}", q(B["Pf_site"][:, j]), f"pre-freeze BEL lens base (site only), x = {xx} m",
                fn, src, ref=(rep, f"| {xx} |", q(B["Pf_site"][:, j])))
        shift = float(np.median(B["Pf"][:, j]) - np.median(B["Pf_site"][:, j]))
        NUM.add(f"physical_filter_shift_x{xx}", f"{shift:+.2f} m" + (" (beyond ±1.3 m)" if abs(shift) > 1.3 else ""),
                f"field lens-base median shift, site + physical − site only, x = {xx} m", fn, src)
        for k, (lo_, hi_) in enumerate(B["bins"]):
            NUM.add(f"bel_base_rhobelow{lo_}-{hi_}_x{xx}", q(B["Pbins"][k][:, j]),
                    f"BEL lens base | ρ_below {lo_}–{hi_} Ω·m, x = {xx} m", fn, src)
    names2 = ["D1", "D2", "rho_lens", "rho_below", "rho_discharge"]
    for i, n2 in enumerate(names2):
        v = B["P2"][:, i]; v = 10 ** v if n2.startswith("rho") else v
        NUM.add(f"second_bel_{n2}", f"{q(v, 2)}; held-out coverage {100*B['cov2'][:, i].mean():.1f} %",
                f"second BEL (site + physical): field {n2}{' (Ω·m)' if n2.startswith('rho') else ' (log10)'} median "
                "[P5, P95]; held-out 90 % coverage", "bel_products", src)
    NUM.add("second_bel_P_D1_gt_0", f"{np.mean(B['P2'][:, 0] > 0):.2f}", "second BEL: P(D1 > 0) — water beneath the "
            "saline body at x 380–470 m more resistive than beneath the lens (quoted instead of the ρ_discharge median)",
            "bel_products", src)
    lo, hi = np.log10(2.5), np.log10(150.0); top = lo + 0.9 * (hi - lo)
    NUM.add("second_bel_rho_dis_top10", f"{100*np.mean(B['P2'][:, 4] > top):.1f} % (prior {100*np.mean(B['Y2_train'][:, 4] > top):.1f} %)",
            "second BEL: field ρ_discharge posterior mass above 99.6 Ω·m (top 10 % of the log range); the upper limit is "
            "physical (150 Ω·m bulk ≈ 300 μS/cm EC25 pore water at F = 3.2), the median is transform-sensitive and not "
            "quoted", "bel_products", src)
    for k, (lo_, hi_) in enumerate(B["bins"]):
        NUM.add(f"bel_rhobelow{lo_}-{hi_}_n_cov", f"{B['bins_n'][k][0]} / {B['bins_n'][k][1]}; {100*B['bins_cov'][k]:.1f} %",
                f"ρ_below {lo_}–{hi_} Ω·m: training / held-out samples; held-out coverage", fn, src)
    thick = np.interp(TX, ex, ez)[None, :] - B["Pf"]
    m = (TX >= 130) & (TX <= 370)
    NUM.add("bel_lens_thickness_130_370", q(np.median(thick[:, m], axis=1), 1), "lens thickness below the surface, "
            "median over x = 130–370 m (m), posterior median [P5, P95]", fn, src)


# ─────────────────────────────────────────────────────────────────────────────
# F8 — salinity
# ─────────────────────────────────────────────────────────────────────────────
def fig_salinity():
    plt = mpl()
    z = np.load(os.path.join(DS6, "field_v2.npz"))
    sm = json.load(open(os.path.join(DS6, "summary_v2.json"), encoding="utf-8"))
    XG, ZG = z["XG"], z["ZG"]
    ex, ez = wc.land_surface()
    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.9], hspace=0.45)
    import matplotlib.colors as mcolors
    for i, (key, cmap, ttl) in enumerate((("Psal", "Blues", "P(% seawater > 50)"),
                                          ("Pfresh", "Oranges", "P(EC25 < 1500 μS/cm) — fresh"))):
        ax = fig.add_subplot(gs[i])
        M = np.where(z["Nv"] > 0, z[key], np.nan)
        for j in range(M.shape[1]):                     # extend the top / bottom valid value of each column outward
            col = M[:, j]; ok = np.where(np.isfinite(col))[0]
            if ok.size:
                col[:ok[0]] = col[ok[0]]; col[ok[-1] + 1:] = col[ok[-1]]
        pc = ax.pcolormesh(XG, ZG, np.ma.masked_invalid(M), cmap=cmap, norm=mcolors.Normalize(0, 1), shading="auto",
                           rasterized=True)
        # smooth clip: median water table / land surface on top, median deep-unit top below (no cell stair-steps)
        from matplotlib.patches import Polygon
        top = np.minimum(np.interp(XG, ex, ez), np.median(z["par_z_wt"]))
        clip_poly = Polygon(np.column_stack([np.r_[XG, XG[::-1]], np.r_[top, np.full_like(XG, np.median(z["par_z_deep"]))]]),
                            closed=True, facecolor="none", edgecolor="none", transform=ax.transData)
        ax.add_patch(clip_poly); pc.set_clip_path(clip_poly)
        ax.plot(ex, ez, "k-", lw=0.8)
        ax.set_xlim(-5, 510); ax.set_ylim(-45, 9); ax.set_aspect(3); dune_hatch(ax)
        ax.set_ylabel("Elevation (mTAW)"); ve_text(ax, 3)
        ax.set_title(f"{ttl} (field)")
        cb = fig.colorbar(pc, ax=ax, fraction=0.02, pad=0.01); cb.set_label("probability")
        panel(ax, "ab"[i])
    ax.set_xlabel("Distance along profile (m)")
    ax = fig.add_subplot(gs[2])
    units = ("lens", "below", "dis", "dune")
    names = {"lens": "saline lens", "below": "water below the lens", "dis": "discharge-zone water",
             "dune": "dune aquifer (weakly constrained)"}
    for k, u in enumerate(units):
        v = z[f"usw_{u}"]; v = v[np.isfinite(v)]
        p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
        y = len(units) - 1 - k
        ax.plot([p5, p95], [y, y], "-", color="#1f4e9c", lw=1.2)
        ax.plot([p25, p75], [y, y], "-", color="#1f4e9c", lw=4.5, alpha=0.6)
        ax.plot(p50, y, "o", color="k", ms=4)
        ps = np.mean(v > 50); ec = z[f"uec_{u}"]; pf = np.mean(ec[np.isfinite(ec)] < 1500)
        ax.text(1.02, (y + 0.5) / len(units), f"P(> 50 %) = {ps:.2f}\nP(fresh) = {pf:.2f}", va="center", fontsize=6.3,
                transform=ax.transAxes)
        if u == "dune":
            ax.axhspan(y - 0.4, y + 0.4, facecolor=(0.55, 0.55, 0.55, 0.2), edgecolor=(0.3, 0.3, 0.3, 0.5), hatch="x", lw=0)
    ax.axvline(50, color="k", ls="--", lw=0.8)
    ax.set_yticks(range(len(units))); ax.set_yticklabels([names[u] for u in units[::-1]])
    ax.set_ylim(-0.5, len(units) - 0.5)
    ax.set_xscale("log"); ax.set_xlim(0.3, 120); ax.set_xlabel("% seawater (median; 50 % and 90 % intervals)")
    ax.set_title(f"Per-unit salinity (ρ_sw {sm['rsw']['min']:.3f}–{sm['rsw']['max']:.3f} Ω·m tied to T; "
                 "draws > 100 % seawater rejected)")
    panel(ax, "c")
    save(fig, "F8_salinity")
    # appendix A1: isohaline vs parametric base
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    iso = z["iso"]; TXz = z["TX"]
    msk = (XG >= TXz[0]) & (XG <= TXz[-1])
    lo, md, hi = (np.where(msk, np.nanpercentile(iso, p, axis=0), np.nan) for p in (5, 50, 95))
    ax.fill_between(XG, lo, hi, color="navy", alpha=0.2, lw=0); ax.plot(XG, md, color="navy", lw=1.3, label="50 % isohaline: median, 90 %")
    l2, m2, h2 = np.percentile(z["base"], [5, 50, 95], 0)
    ax.fill_between(TXz, l2, h2, color="purple", alpha=0.12, lw=0); ax.plot(TXz, m2, "--", color="purple", lw=1.2, label="parametric base (joint BEL)")
    ax.plot(ex, ez, "k-", lw=0.8); ax.set_xlim(-5, 510); ax.set_ylim(-32, 8); ax.set_aspect(3); dune_hatch(ax)
    ax.set_xlabel("Distance along profile (m)"); ax.set_ylabel("Elevation (mTAW)"); ve_text(ax, 3)
    ax.set_title("Appendix: salinity-defined lens base vs parametric base (equal by construction where the lens is > 50 %)")
    wc.white_legend(ax, loc="lower left", fontsize=6.3)
    save(fig, "A1_isohaline")
    fn, src = "fig_salinity (values from 17_salinity_v2.py:cmd_run)", "`data/processed/stage6_v2/field_v2.npz`"
    rep = "results/stage6_v2/2_salinity_v2.md"
    for u in units:
        a = z[f"uarea_{u}"]; v = z[f"usw_{u}"]; ec = z[f"uec_{u}"]
        ok = np.isfinite(v)
        row = f"{s6.q(a, 0)} | {s6.q(v, 1)} | {s6.q(ec, 0)} | {np.mean(v[ok] > 50):.2f} | {np.mean(ec[ok] < 1500):.2f} |"
        NUM.add(f"unit_{u}", row.rstrip(" |").replace(" | ", "; "), f"{names[u]}: area (m²/m); % seawater; EC25 (μS/cm); "
                "P(> 50 %); P(fresh) — median [P5, P95]", fn, src, ref=(rep, f"| {names[u].replace(' (weakly constrained)', '')}", row))
    NUM.add("rsw_range", f"{sm['rsw']['min']:.3f}–{sm['rsw']['max']:.3f} Ω·m", "seawater resistivity range (S 31–34 PSU, "
            "T 8–14 °C)", "17_salinity_v2.py:cmd_run", "PSS-78", ref=("results/stage6_v2/1_seawater_resistivity.md", "ρ_sw range",
                                                                    f"{sm['rsw']['min']:.3f}–{sm['rsw']['max']:.3f} Ω·m"))
    NUM.add("ec25_sw_range", f"{sm['ec25_sw'][0]:.2f}–{sm['ec25_sw'][1]:.2f} mS/cm", "EC25 of seawater, S = 31–34 PSU (PSS-78)",
            "17_salinity_v2.py:cmd_run", "PSS-78", ref=("results/stage6_v2/1_seawater_resistivity.md", "EC25 of seawater",
                                                       f"{sm['ec25_sw'][0]:.2f}–{sm['ec25_sw'][1]:.2f} mS/cm"))
    NUM.add("pdis_field", f"{float(z['pdis']):.2f}", "P(discharge zone present | field)", fn, src)
    for u, c in sm["ucov"].items():
        NUM.add(f"unit_{u}_heldout_cov", f"area {100*c['area']:.1f}% (physical truths {100*c['area_phys']:.1f}%), "
                f"% seawater {100*c['sw']:.1f}%, EC25 {100*c['ec25']:.1f}%",
                f"{names[u]}: held-out coverage of the 90 % intervals", "17_salinity_v2.py:cmd_run",
                "`data/processed/stage6_v2/summary_v2.json`",
                ref=(rep, f"| {names[u].replace(' (weakly constrained)', '')}",
                     f"{100*c['area']:.1f}% / {100*c['area_phys']:.1f}% | {100*c['sw']:.1f}% | {100*c['ec25']:.1f}%"))
    jc = sm["joint_cov"]
    NUM.add("joint_heldout_base_cov", f"{100*jc['base']:.1f} % ({100*jc['base_col_min']:.0f}–{100*jc['base_col_max']:.0f} % per column)",
            "joint BEL held-out coverage, lens base", "17_salinity_v2.py:cmd_run", "`data/processed/stage6_v2/summary_v2.json`",
            ref=(rep, "Lens base (all", f"**{100*jc['base']:.1f} %**"))
    for k, v in jc["params"].items():
        NUM.add(f"joint_heldout_cov_{k}", f"{100*v:.1f} %", f"joint BEL held-out coverage, {k}", "17_salinity_v2.py:cmd_run",
                "`data/processed/stage6_v2/summary_v2.json`", ref=(rep, f"| {k} |", f"{100*v:.1f}% |"))
    fl = sm["filter"]
    NUM.add("physical_filter_counts", f"training {fl['n_site'][0]} → {fl['n_phys'][0]}; held-out {fl['n_site'][1]} → {fl['n_phys'][1]}",
            f"site → site + physical subset (ρ_lens ≥ {fl['rho_lens_min']:.5f} Ω·m)", "17_salinity_v2.py:cmd_run",
            "`data/processed/stage6_v2/summary_v2.json`")
    cn = sm["constraint"]
    NUM.add("constraint_joint_rejection_field", f"{100*cn['joint_rate']:.1f} %", "field: share of (posterior sample, F, S, T) "
            "draws rejected for > 100 % seawater in any unit", "17_salinity_v2.py:cmd_run", "`data/processed/stage6_v2/summary_v2.json`",
            ref=(rep, "rejected", f"**{100*cn['joint_rate']:.1f} % rejected**"))
    for u in units:
        NUM.add(f"constraint_rejection_{u}", f"{100*cn['unit_rate'][u]:.1f} %", f"field: rejection rate, {names[u]}",
                "17_salinity_v2.py:cmd_run", "`data/processed/stage6_v2/summary_v2.json`",
                ref=(rep, f"| {names[u].replace(' (weakly constrained)', '')} |", f"{100*cn['unit_rate'][u]:.1f} % |"))
    NUM.add("constraint_heldout", f"median rejection {100*cn['heldout_median_joint_rate']:.1f} %; {cn['heldout_truth_impossible']} "
            f"unphysical truths; {cn['heldout_post_empty']} posteriors with < 50 physical draws (of {sm['n_heldout']})",
            "held-out: effect of the physical constraint (prior ρ_lens partly below F·ρ_sw)", "17_salinity_v2.py:cmd_run",
            "`data/processed/stage6_v2/summary_v2.json`")


# ─────────────────────────────────────────────────────────────────────────────
# forward accuracy (T3)
# ─────────────────────────────────────────────────────────────────────────────
def fwd_accuracy(recompute=False):
    p = os.path.join(DPAP, "fwd_fine_g1_lam50.npz")
    ref = wc.load_run("g1_lam50")
    if os.path.exists(p) and not recompute:
        R_fine = np.load(p)["R_fine"]
    else:
        import pygimli.meshtools as mt
        from scipy.spatial import cKDTree
        pd_, _ = wc.rebuild_paradomain(ref)
        scheme = b13.pilot.build_scheme()
        fmesh = mt.createParaMesh(scheme.sensors(), **sb.FWD)
        tree = cKDTree(np.column_stack([ref["cell_x"], ref["cell_z"]]))
        vals = np.empty(fmesh.cellCount())
        for c in fmesh.cells():
            pc = c.center(); cell = pd_.findCell(pc)
            vals[c.id()] = ref["model"][cell.id()] if cell is not None else ref["model"][tree.query([pc.x(), pc.y()])[1]]
        with contextlib.redirect_stdout(io.StringIO()):
            R_fine = b13.pilot.simulate_R(fmesh, vals, scheme.size())
        np.savez(p, R_fine=R_fine)
    z = (np.log(R_fine) - np.log(ref["d_pred"] / ref["K"])) / ref["d_err"]
    xo = dict(zip(ref["elec_num"].astype(int), ref["elec_x"]))
    mid = sum(np.array([xo[e] for e in ref[k].astype(int)]) for k in ("C1", "C2", "P1", "P2")) / 4
    return float(np.mean(np.abs(z) > 1)), float(np.mean(np.abs(z[mid < 90]) > 1))


# ─────────────────────────────────────────────────────────────────────────────
# tables
# ─────────────────────────────────────────────────────────────────────────────
def md(x):
    """Escape '|' inside a markdown table cell."""
    return str(x).replace("|", "\\|")


def tables(B, summ):
    L = ["# Paper tables", ""]
    # T1
    spec = json.load(open(b13.SPEC2, encoding="utf-8"))
    L += ["## T1 — prior (site + physical subset of prior v3) and conversion parameters", "",
          "| parameter | distribution | range | unit | source / justification |", "|---|---|---|---|---|"]
    csv = ["parameter,distribution,range,unit,source"]
    for k, v in spec["params"].items():
        rng_ = v.get("range")
        rs = f"p = {v['p']}" if rng_ is None else (f"{rng_[0]:g} – {rng_[1]:g}" if rng_[1] is not None else f"{rng_[0]:g} – x_L + 20")
        why = v.get("why", "")
        if k == "rho_lens":
            rs = f"{s6.PHYS_RHO_LENS_MIN:.3f} – 3.0 (physical filter of 0.4 – 3.0)"
            why = ("saline by construction; lower bound = min F × min ρ_sw (2.8 × 0.2479 Ω·m, PSS-78): below it the lens "
                   "would exceed 100 % seawater for every Archie draw")
        if k == "rho_discharge":
            rs = "2.5 – 150"; why = "prior v2 2.5–40 (Hermans & Paepen 2020 range, confirmed) extended to 150 Ω·m in prior v3 (posterior mass piled at the 40 Ω·m ceiling)"
        if k == "z_deep":
            rs = "−38 – −22 (site subset of −45 – −20)"; why = "Kortrijk-clay aquifer base −25…−30 mTAW at K1, ≈ −35 mTAW at K0 (Paepen et al. 2020, citing Lebbe 1981)"
        L.append(f"| {k} | {v.get('dist', '')} | {md(rs)} | {v.get('unit', '')} | {md(why)} |")
        csv.append(",".join([k, v.get("dist", ""), rs, v.get("unit", ""), '"' + why.replace('"', "'") + '"']))
    extra = [("site constraint", "—", "lens base above the deep-unit top (all lens columns)", "", "physical consistency with the deep unit (site subset)"),
             ("noise", "Gaussian", "3 % + 1e-4 Ω", "", "field error model (reference inversion χ² ≈ 0.75)"),
             ("mesh error", "Gaussian", "mean + covariance of log10 R_fine − log10 R_medium (100 dual samples)", "", "approximation-error correction"),
             ("F (formation factor)", "uniform", "2.8 – 3.6", "", "Paepen et al. (2020), centred on 3.2"),
             ("S (seawater salinity)", "uniform", "31 – 34", "PSU", "decision (southern North Sea coastal water); EC25 via PSS-78"),
             ("T (groundwater temperature)", "uniform", "8 – 14", "°C", "decision (central 11 °C); same T for ρ_sw and EC25"),
             ("α (temperature coefficient)", "fixed", "0.02", "/°C", "linear compensation to 25 °C")]
    for r in extra:
        L.append("| " + " | ".join(md(x) for x in r) + " |"); csv.append(",".join(f'"{x}"' for x in r))
    with open(os.path.join(RESP, "T1_prior.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv) + "\n")
    # T2
    L += ["", "## T2 — synthetic validation of the lens base: coverage of 90 % intervals, bias, width", "",
          "Linearised: truths T1, T1h, T2a, T3 × 20 realisations, λ = 50 (pooled = λ 5–100, realisations 0–4), columns "
          "x = 100–450 m grouped by the resolution radius at the method's own λ = 50 crossing; coverage and |bias| = mean "
          "over the four truths, width = median. BEL-KDE (site + physical, prior v3): held-out prior samples (all 36 columns); "
          "Stage-4 truths and the BEL-builder twin of T1 (20 realisations, fine-mesh data, columns x = 130–450 m); "
          "bias = median(estimate − truth), negative = too deep.", "",
          "| method | test set | coverage | bias (m) | width (m) |", "|---|---|---|---|---|"]
    csv = ["method,test_set,coverage_pct,bias_m,width_m"]
    names = {"thr3": "linearised 3 Ω·m", "logmid": "linearised log-midpoint", "maxgrad": "linearised max-gradient"}
    for dname in ("thr3", "logmid"):
        for grp in ("≤10", "10–15"):
            s = summ[(grp, dname)]
            for lab, cv, wd, bs in (("λ = 50", s["cov"], s["width"], f"{s['absbias']:.2f} (abs.)"),
                                    ("λ-pooled", s["cov_pooled"], s["width_pooled"], "—")):
                L.append(f"| {names[dname]} ({lab}) | 4 truths, r {grp} m | {100*cv:.0f} % | {bs} | {wd:.2f} |")
                csv.append(f"{names[dname]} ({lab}),4 truths r {grp} m,{100*cv:.1f},{bs},{wd:.3f}")
    L.append(f"| BEL-KDE (site + physical, prior v3) | held-out prior ({B['n_held']}) | {100*B['ho_cov'].mean():.1f} % | "
             f"{np.median(B['ho_bias']):+.2f} | {np.median(B['ho_wid']):.2f} |")
    csv.append(f"BEL-KDE,held-out prior,{100*B['ho_cov'].mean():.2f},{np.median(B['ho_bias']):.3f},{np.median(B['ho_wid']):.3f}")
    ts = B["truth_stats"]
    for k, v in ts.items():
        L.append(f"| BEL-KDE (site + physical, prior v3) | {k} (n = {v['n']}) | {100*v['cov']:.0f} % | {v['bias']:+.2f} | {v['width']:.2f} |")
        csv.append(f"BEL-KDE,{k},{100*v['cov']:.1f},{v['bias']:.3f},{v['width']:.3f}")
    with open(os.path.join(RESP, "T2_synthetic.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv) + "\n")
    for k, v in ts.items():
        NUM.add(f"bel_truth_{re.sub('[^A-Za-z0-9]+', '_', k)}", f"{100*v['cov']:.0f} % / {v['bias']:+.2f} / {v['width']:.2f} m",
                f"BEL (site + physical, prior v3) on {k}: coverage / bias / width", "bel_products / tables",
                "`data/processed/synthetic/truth_forward.npz`, `data/processed/bel_v4/twin_forward.npz`")
    # T3 sensitivity
    ctx = ref_ctx(); cs = ctx["cs_tx"]
    xs = np.isin(TX, XREP)
    base = {t: lens_base(run_npz(t), cs) for t in ("deep_drop_data", "deep_drop_elec", "deep_downweight", "g1_lam50",
                                                    "g1b_thr299", "g1_lam5", "g1_lam10", "g1_lam100")}
    def dd(a, b):
        d = np.abs(a - b)[xs]; return f"{np.nanmedian(d):.2f} / {np.nanmax(d):.2f}"
    lamB = np.array([base[t] for t, _ in LAM_TAGS])
    v4 = np.load(LIN_V4)
    jj = [int(np.argmin(np.abs(v4["x"] - x))) for x in XREP]
    mdB = np.median(B["Pf"], 0)
    frac, frac_dune = fwd_accuracy()
    ap = np.load(os.path.join(b13.D2, "approx_error.npz")); nl = 0.03 / np.log(10)
    sm = json.load(open(os.path.join(DS6, "summary_v2.json"), encoding="utf-8"))
    sens = sm["sens"]; ref_ = sens["reference (F ~ U(2.8, 3.6))"]
    rows3 = [
        ("outliers", "G2 (drop 11 flagged electrodes) vs G1, λ = 20", "log-midpoint m̂ base", dd(base["deep_drop_elec"], base["deep_drop_data"])),
        ("outliers", "G3 (down-weight flagged electrodes) vs G1, λ = 20", "log-midpoint m̂ base", dd(base["deep_downweight"], base["deep_drop_data"])),
        ("outliers", "G1b (|r| > 2.99) vs G1, λ = 50", "log-midpoint m̂ base", dd(base["g1b_thr299"], base["g1_lam50"])),
        ("regularisation", "λ = 5 … 100 (G1)", "log-midpoint m̂ base, max − min over λ",
         f"{np.nanmedian(np.ptp(lamB, 0)[xs]):.2f} / {np.nanmax(np.ptp(lamB, 0)[xs]):.2f}"),
        ("interface definition", "3 Ω·m vs log-midpoint, λ = 50", "m̂ base",
         f"{np.nanmedian(np.abs(v4['mhat_thr3'][jj] - v4['mhat_logmid'][jj])):.2f} / {np.nanmax(np.abs(v4['mhat_thr3'][jj] - v4['mhat_logmid'][jj])):.2f}"),
        ("mesh (inversion)", "reference model on the fine forward mesh", "share of G1 data with |Δ| > 1σ (x_mid < 90 m)",
         f"{100*frac:.1f} % ({100*frac_dune:.1f} %)"),
        ("mesh (BEL forward)", "medium vs fine mesh, 100 dual prior samples", "median |mean δ| / median s.d. δ, in units of 3 % noise",
         f"{np.median(np.abs(ap['mean']))/nl:.2f} / {np.median(ap['sd'])/nl:.2f}"),
        ("prior subset", "site + physical vs site only (BEL; physical filter)", "BEL median base",
         f"{np.median(np.abs(np.median(B['Pf_site'], 0) - mdB)[xs]):.2f} / {np.max(np.abs(np.median(B['Pf_site'], 0) - mdB)[xs]):.2f}"),
        ("prior subset", "site + physical vs full prior v3 (BEL)", "BEL median base",
         f"{np.median(np.abs(np.median(B['Pf_full'], 0) - mdB)[xs]):.2f} / {np.max(np.abs(np.median(B['Pf_full'], 0) - mdB)[xs]):.2f}"),
    ]
    for k, (lo_, hi_) in enumerate(B["bins"]):
        d = np.abs(np.median(B["Pbins"][k], 0) - mdB)[xs]
        rows3.append(("ρ_below", f"{lo_}–{hi_} Ω·m vs all (BEL)", "BEL median base", f"{np.median(d):.2f} / {d.max():.2f}"))
    for Fv in ("2.8", "3.6"):
        s = sens[f"F = {Fv}"]
        rows3.append(("F (Archie)", f"F = {Fv} vs F ~ U(2.8, 3.6)", "% seawater lens / below / discharge (medians)",
                      " / ".join(f"{a:.1f}" for a in s["sw"]) + "  (ref " + " / ".join(f"{a:.1f}" for a in ref_["sw"]) + ")"))
        rows3.append(("F (Archie)", f"F = {Fv}", "saline beach area (m²/m, median)",
                      f"{s['areas']['saline_beach']:.0f} (ref {ref_['areas']['saline_beach']:.0f})"))
    for k, v in sens.items():
        if k.startswith("ρ_below"):
            rows3.append(("ρ_below (salinity)", k, "50 % isohaline at x = 150 / 250 / 350 (mTAW)", " / ".join(f"{a:.2f}" for a in v["iso"])
                          + "  (ref " + " / ".join(f"{a:.2f}" for a in ref_["iso"]) + ")"))
    L += ["", "## T3 — sensitivity summary", "",
          "Lens-base rows: |Δ| of the field lens base at x = 150, 200, 250, 300, 350 m — median / max (m).", "",
          "| factor | comparison | quantity | effect |", "|---|---|---|---|"]
    csv = ["factor,comparison,quantity,effect"]
    for r in rows3:
        L.append("| " + " | ".join(md(x) for x in r) + " |"); csv.append(",".join(f'"{x}"' for x in r))
        NUM.add("sens_" + re.sub("[^A-Za-z0-9]+", "_", re.sub(r"\(n = \d+\)", "", f"{r[0]}_{r[1]}")).strip("_")[:60],
                r[3], f"{r[0]}: {r[1]} — {r[2]}",
                "tables", "see T3 inputs")
    with open(os.path.join(RESP, "T3_sensitivity.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv) + "\n")
    # appendix TA1: class areas
    z = np.load(os.path.join(DS6, "field_v2.npz"))
    L += ["", "## TA1 (appendix) — class areas of the field section (m² per m of coast)", "",
          "| class | dunes x 0–90 m (weakly constrained) | beach x 90–505 m |", "|---|---|---|"]
    for c in ("saline", "brackish", "fresh"):
        L.append(f"| {c} | {s6.q(z[f'area_{c}_dunes'], 0)} | {s6.q(z[f'area_{c}_beach'], 0)} |")
        for zz in ("dunes", "beach"):
            NUM.add(f"class_area_{c}_{zz}", s6.q(z[f"area_{c}_{zz}"], 0), f"appendix: {c} area, {zz} (m²/m)",
                    "tables", "`data/processed/stage6_v2/field_v2.npz`")
    with open(os.path.join(RESP, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("[tables] written", flush=True)
    fnm, src = "tables", "`data/processed/runs/*`, `results/runs/g1_lam50_v4`, `data/processed/bel/v2/approx_error.npz`"
    NUM.add("fwd_accuracy_share", f"{100*frac:.1f} %", "share of G1 data whose inversion-mesh response differs from the fine "
            "mesh by > 1σ", fnm, src, ref=("results/bel_v3/3_forward_accuracy.md", "share with |z| > 1", f"**{100*frac:.1f} %**"))


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# numbers added in v1.0.1 (appended after all v1.0 numbers so that existing rows keep their position)
# ─────────────────────────────────────────────────────────────────────────────
def extra_numbers():
    import scipy.linalg as sla
    fn = "extra_numbers"
    F = run_npz("deep_all")
    NUM.add("run_F_chi2", f"{float(F['chi2']):.4f}", "run F (all 2331 data, λ = 20, paraDepth 60 m): χ² (log space)", fn,
            "`data/processed/runs/deep_all`", ref=("results/deep_all_misfit_report.txt", "chi2 (log space)", f"{float(F['chi2']):.4f}"))
    Au = run_npz("deep_auto")
    NUM.add("auto_depth_chi2", f"{float(Au['chi2']):.2f} (paraDepth {float(Au['para_depth_used']):.0f} m)",
            "run with pyGIMLi's automatic mesh depth (all data, λ = 20, paraDX 0.33): χ² (log space)", fn,
            "`data/processed/runs/deep_auto`",
            ref=("data/processed/runs/deep_auto/westhoek_inversion_results.npz", "npz:chi2:.2f", f"{float(Au['chi2']):.2f}"))
    ref = run_npz("g1_lam50")
    L_, A_, _, _ = wc.cholesky_A(ref)
    Z = np.random.default_rng(20260927).standard_normal((A_.shape[0], 1000))      # = 09_posterior_sampling.py, first draw
    ratio = sla.solve_triangular(L_, Z, lower=True, trans="T").std(axis=1, ddof=1) / ref["sigma_log_m"]
    rep9 = "results/runs/g1_lam50_v3/posterior_sampling_report.txt"
    NUM.add("posterior_samples", "1000 per run (seed 20260927); 5000 pooled over λ = 5, 10, 20, 50, 100",
            "linearised posterior samples (Cholesky of A = JᵀWd²J + λCᵀC)", fn, "`data/processed/runs/g1_lam*`",
            ref=(rep9, "Samples:", "Samples: 1000, seed 20260927"))
    NUM.add("posterior_std_check", f"{np.median(ratio):.4f} (min {ratio.min():.4f}, max {ratio.max():.4f})",
            "sample std / √diag(A⁻¹) per cell, g1_lam50 (median, range)", fn, "`data/processed/runs/g1_lam50`",
            ref=(rep9, "CHECK sample std", f"median {np.median(ratio):.4f}, min {ratio.min():.4f}, max {ratio.max():.4f}"))
    spec = sb.load_spec()
    for t in ("T1", "T1h"):
        v = spec["values"][t]
        row = " | ".join(f"{v[k]:.2f}" for k in ("cap", "fresh", "lens", "brackish", "discharge", "deep"))
        NUM.add(f"truth_rho_{t}", row.replace(" | ", " / "), f"synthetic truth {t}: ρ of cap / fresh aquifer / lens / "
                "brackish / discharge / deep (Ω·m)" + ("; T2a–T6 use the T1 values" if t == "T1" else ""), fn,
                "`data/processed/synthetic/truth_spec.json`", ref=("results/synthetic/truth_models.md", f"| {t} |", f"| {t} | {row} |"))
    import pygimli.meshtools as mt
    scheme = b13.pilot.build_scheme()
    with contextlib.redirect_stdout(io.StringIO()):
        nf = mt.createParaMesh(scheme.sensors(), **sb.FWD).cellCount()
        nm = mt.createParaMesh(scheme.sensors(), **b13.MESH["medium"]).cellCount()
    NUM.add("fwd_mesh_cells_fine", f"{nf:,}".replace(",", " "), "fine forward mesh (synthetic truths, dual prior samples, "
            "forward-accuracy check): cells", fn, "pyGIMLi createParaMesh with 10_synthetic_benchmark.FWD",
            ref=("results/bel_v3/3_forward_accuracy.md", "fine forward mesh", f"{nf:,}".replace(",", " ") + " cells"))
    NUM.add("fwd_mesh_cells_medium", f"{nm:,}".replace(",", " "), "medium forward mesh (BEL prior samples): cells", fn,
            "pyGIMLi createParaMesh with 13_bel_full.MESH['medium']")
    pres = lens_presence()
    for x in (100, 110, 120):
        v = pres[str(x)]
        NUM.add(f"lens_present_x{x}", f"{v['p']:.2f} (held-out Brier {v['brier']:.3f} vs no-skill {v['noskill']:.3f})",
                f"P(lens present | field) at x = {x} m (Gate-A classifier, prior v2)", "lens_presence",
                "`data/processed/bel/v2/train`",
                ref=("results/bel_v3/2_nuisance.md", f"lens present at x = {x} m",
                     f"| {v['brier']:.3f} | {v['noskill']:.3f} | {1 - v['brier']/v['noskill']:+.2f} | {v['p']:.2f} |"))
    gps = read_gps()
    brg = lambda a, b: np.degrees(np.arctan2(gps[b][0] - gps[a][0], gps[b][1] - gps[a][1])) % 360
    bl, bm = brg(1, 102), brg(103, 538)
    u = np.array([np.sin(np.radians(bl)), np.cos(np.radians(bl))]); nvec = np.array([u[1], -u[0]])
    off = [float(np.dot(np.array(gps[e][:2]) - np.array(gps[1][:2]), nvec)) for e in (103, 538)]
    NUM.add("marine_line_geometry", f"bearings {bl:.1f}° (land) vs {bm:.1f}° (marine), difference {abs(bm - bl):.1f}°; "
            f"marine end points {abs(off[0]):.0f} m and {abs(off[1]):.0f} m off the land line",
            "land (electrodes 1–102) vs marine profile (103–538) geometry: not collinear", fn, "`data/raw/GPS_coordinate.xlsx`")
    z = np.load(os.path.join(DS6, "field_v2.npz"))
    dis = (z["XG"] >= 330)[None, :] & (z["Nv"] > 0.5 * z["base"].shape[0])
    NUM.add("pfresh_cell_max_discharge", f"{np.nanmax(np.where(dis, z['Pfresh'], np.nan)):.2f}",
            "largest cell-wise P(EC25 < 1500 μS/cm) at x ≥ 330 m (the unit value P(fresh) of the discharge water is "
            "`unit_dis`)", fn, "`data/processed/stage6_v2/field_v2.npz`")


def captions():
    """results/paper/captions.md — one caption per figure; numbers taken from the registry (numbers.md)."""
    N = {r["key"]: r["value"] for r in NUM.rows}
    g = lambda k: N[k]
    ts = {k[len("bel_truth_"):]: v for k, v in N.items() if k.startswith("bel_truth_")}
    C = ["# Figure captions (v1.0.1)", "",
         "Numbers are those of `results/paper/numbers.md` (key in brackets where it helps). Blue = conductive throughout; "
         "the dune zone x < 90 m is hatched as weakly constrained; V.E. is stated in every section.", ""]
    C += ["**Figure 1.** Westhoek site (De Panne, Belgium). (a) Land ERT profile — "
          f"{g('n_electrodes')} electrodes at {g('electrode_spacing_m')} m spacing over {g('profile_length_m')} m, multiple-gradient "
          f"array at low tide (−0.15 mTAW), {g('n_quadrupoles')} quadrupoles — and the marine profile (floating electrodes, "
          f"high tide {g('h_high_tide_mTAW')} mTAW). The marine line is not collinear with the land line "
          f"({g('marine_line_geometry')}); only the land data are inverted here. Inset: location on the southern North Sea "
          "coast (schematic coastline and borders). Wells K0 and K1 (Paepen et al. 2020) will be added once located. "
          f"(b) Topography, electrodes and survey-day water levels; high-water line at x = {g('x_high_water_m')} m; spring "
          "tidal range ≈ 5 m (field description). V.E. 10×.", ""]
    C += ["**Figure 2.** Outlier diagnostics. (a) Share of per-datum outliers among the data using each electrode, run F "
          f"(all data, λ = 20, χ² = {g('run_F_chi2')}); outliers are |r| > 3·1.4826·MAD = {g('outlier_threshold')} of the "
          f"normalised log residuals r ({g('n_outliers').replace(' (', ' data, ').rstrip(')')} of all data); red = the 11 flagged electrodes whose data G2 drops; shaded = "
          "intertidal zone. (b) G1 model (outlier data dropped) with the log-midpoint lens base of G1 (solid) and G2 "
          f"(flagged electrodes dropped; dashed), x = 130–450 m; median / max difference {g('G2_vs_G1_base')}. Both are shown "
          "at **λ = 20**, the λ of the outlier-treatment comparison (the reference model uses λ = 50). Hatched: resolution "
          "radius r > 10 m. V.E. 3×.", ""]
    C += [f"**Figure 3.** Reference model g1_lam50 (G1 data, {g('ref_n_cells')} cells, λ = {g('ref_lambda')}, χ² = "
          f"{g('ref_chi2')} after {g('ref_n_iter')} iterations) with resolution-radius contours r = 10 m and 15 m "
          f"(Friedel 2003). Cells within 30 m of the surface with r ≤ 10 m: {g('ref_share_r10_top30')}. V.E. 3×.", ""]
    C += ["**Figure 4.** Linearised uncertainty. (a) P(ρ < 3 Ω·m) from linearised posterior samples "
          f"({g('posterior_samples')}; sample std / √diag(A⁻¹) = {g('posterior_std_check')}), pooled over λ = 5–100; "
          "hatched r > 10 m. (b) Lens base, λ-pooled median and 90 % band for the 3 Ω·m contour (orange) and the "
          "log-midpoint definition (blue), x = 130–450 m. V.E. 3×.", ""]
    C += ["**Figure 5.** Synthetic benchmark (truths T1, T1h, T2a, T3; 20 noise realisations; truth resistivities "
          f"T1 {g('truth_rho_T1')} Ω·m, T1h {g('truth_rho_T1h')} Ω·m for cap / fresh / lens / brackish / discharge / deep; "
          f"fine forward mesh of {g('fwd_mesh_cells_fine')} cells). (a) Coverage of the 90 % bands by interface "
          "definition at λ = 50 (bars) and λ-pooled (ticks), grouped by resolution radius at the estimated base. (b) "
          "Median bias (dots: individual truths). (c, d) Coverage and bias vs λ. (e, f) Reliability of P(ρ < 3 Ω·m).", ""]
    C += ["**Figure 6.** BEL calibration (site + physical subset of prior v3, "
          f"{g('bel_n_train_site_phys')} training / held-out samples). (a) Three prior samples (black: lens-base target). (b) "
          f"Prior falsification of the field data among the training samples: percentile {g('falsify_t2_pct')} for T², "
          f"{g('falsify_t2_robust_pct')} for robust T² and {g('falsify_q_pct')} for Q (none beyond P99). (c) Held-out rank histogram of the true lens base; "
          f"(d) held-out coverage per column ({g('bel_heldout_coverage')} overall). The excess at rank ≈ 1 and the deficit at "
          "0.7–0.95 mean the true base lies above (shallower than) the posterior more often than nominal: the conditional "
          "bias of the non-unique data–target relation, also seen on the Stage-4 truths and the BEL-builder twin (coverage / "
          f"bias / width: T1 {ts.get('T1', '—')}, twin {ts.get('twin_own_target_', '—')}; Table T2); accepted and reported "
          "alongside the intervals (decision log).", ""]
    C += ["**Figure 7.** Field lens base. (a) BEL-KDE (site + physical, prior v3; purple, from x = 130 m) and the linearised "
          "log-midpoint band (blue, λ-pooled); bands and medians are clipped at the land surface. Triangles: P(lens present) "
          f"at x = 100, 110, 120 m = {g('lens_present_x100').split(' ')[0]}, {g('lens_present_x110').split(' ')[0]}, "
          f"{g('lens_present_x120').split(' ')[0]} (Gate-A classifier). (b) BEL lens base conditional on the resistivity of "
          "the water below the lens (ρ_below bins; line style per bin). V.E. 3×.", ""]
    C += ["**Figure 8.** Salinity (joint BEL + Archie; ρ_sw tied to temperature; draws with > 100 % seawater rejected). "
          "(a) Cell-wise P(% seawater > 50) and (b) cell-wise P(EC25 < 1500 μS/cm), shown from the median water table / land "
          "surface down to the median deep-unit top. (c) Per-unit % seawater (median, 50 % and 90 % intervals) with the "
          "unit probabilities. **Cell vs unit:** the unit value P(fresh) of the discharge-zone water "
          f"(unit_dis: {g('unit_dis').split('; ')[-1]}) is the probability that the discharge water is fresh; the map shows, per "
          "cell, the probability that the cell lies in fresh water, which is lower (at most "
          f"{g('pfresh_cell_max_discharge')} at x ≥ 330 m) because the extent and position of the discharge body are "
          "uncertain. V.E. 3×.", ""]
    C += ["**Figure A1.** 50 % seawater isohaline vs the parametric (joint BEL) lens base; equal by construction where the "
          "lens exceeds 50 % seawater (homogeneous lens resistivity, sharp base). V.E. 3×.", ""]
    with open(os.path.join(RESP, "captions.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(C) + "\n")
    print("[captions] written", flush=True)


PARTS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "T")
RECOMPUTE = False


def cmd_all(args):
    global FIGP, RESP, DPAP, DS6, RECOMPUTE
    RECOMPUTE = bool(args.recompute)
    t0 = time.time()
    root = os.path.abspath(args.out)
    FIGP = os.path.join(root, "figures", "paper"); RESP = os.path.join(root, "results", "paper")
    DPAP = os.path.join(root, "data", "processed", "paper"); DS6 = os.path.join(root, "data", "processed", "stage6_v2")
    for d in (FIGP, RESP, DPAP):
        os.makedirs(d, exist_ok=True)
    only = set(args.only.split(",")) if args.only else set(PARTS)
    if "F1" in only: fig_site()
    if "F2" in only: fig_outliers()
    if "F3" in only: fig_reference()
    if "F4" in only: fig_linearised()
    summ = None
    if "F5" in only or "T" in only:
        _, summ = fig_synthetic()
    B = bel_products(args.recompute) if only & {"F6", "F7", "T"} else None
    if "F6" in only: fig_bel_calibration(B)
    if "F7" in only: fig_field_base(B)
    if "F8" in only: fig_salinity()
    if "T" in only: tables(B, summ)
    if only == set(PARTS):
        extra_numbers()
        NUM.write(RESP)
        captions()
        print(f"[numbers] {len(NUM.rows)} numbers → {os.path.join(RESP, 'numbers.md')}")
    print(f"[18_paper] done in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("all"); pa.add_argument("--out", default=wc.ROOT); pa.add_argument("--only", default="")
    pa.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    dict(all=cmd_all)[a.cmd](a)


if __name__ == "__main__":
    main()
