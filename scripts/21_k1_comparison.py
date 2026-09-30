"""
21_k1_comparison.py
===================
Westhoek ERT v1.1 — profile location (K1) and the comparison of the 1980 K1 borehole-log structure (Lebbe 1981) with
the 2018 lens base of this study.

  location [--out ROOT]   Task 0: electrodes 1, 102, 538 → WGS84 (pyproj, EPSG:31370 → EPSG:4326); distance from the
                          coastal end of the French–Belgian border (51.08944 N, 2.54528 E) along the coast (azimuth 57°)
                          and in a straight line → ROOT/results/v1_1/0_profile_location.md
  run [--out ROOT]        Task 3: comparison numbers → ROOT/data/processed/v1_1/k1_comparison.json
  report [--out ROOT]     Task 3: ROOT/results/v1_1/3_k1_comparison.md and figs/F_S3_k1_1980_vs_2018{,_notitle}.png

Inputs: data/raw/GPS_coordinate.xlsx; data/external/lebbe1981_K1_digitised.csv (hand-digitised, see its header);
BEL field lens base = the v1.0 headline (base-only BEL, site + physical subset of prior v3), recomputed with the v1.0
functions and seed (default_rng([129000, 1])); linearised λ-pooled log-midpoint bands from
results/runs/g1_lam50_v4/interface_definitions_bands.npz. No random draws of its own.
"""

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import read_trace  # noqa: E402
read_trace.install("21_k1_comparison.py")
import westhoek_common as wc  # noqa: E402

BORDER = (51.08944, 2.54528)            # coastal end of the French–Belgian border (lat, lon)
COAST_AZ = 57.0                         # coast direction (deg from north)
XREP = (150, 200, 250, 300, 350)
X_EXTRA = (400, 450)                    # v1.1.1: seaward positions, kept in a separate block ("extra")
DIS_ZONE = (380.0, 480.0)               # discharge zone of this study, shaded in F_S3
LEBBE_CSV = os.path.join(wc.ROOT, "data", "external", "lebbe1981_K1_digitised.csv")
X_ERR, Z_ERR = 30.0, 1.0


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def read_lebbe():
    rows = [r for r in csv.DictReader(l for l in open(LEBBE_CSV, encoding="utf-8") if not l.startswith("#"))]
    f = lambda v: float(v) if v.strip() else np.nan
    keys = ("x_ours_m", "x_lebbe_m", "ground_1980_mTAW", "saline_base_mTAW", "fresh_top_mTAW",
            "lower_brackish_top_mTAW", "clay_top_mTAW")
    return {k: np.array([f(r[k]) for r in rows]) for k in keys} | {"note": [r["note"] for r in rows]}


def interp_boundary(xq, x, v):
    """Linear interpolation between the digitised points that have a value; constant beyond the end points."""
    ok = np.isfinite(v)
    return np.interp(xq, x[ok], v[ok])


def paths(root):
    res = os.path.join(root, "results", "v1_1"); dpp = os.path.join(root, "data", "processed", "v1_1")
    for d in (res, os.path.join(res, "figs"), dpp):
        os.makedirs(d, exist_ok=True)
    return res, dpp


# ─────────────────────────────────────────────────────────────────────────────
def cmd_location(args):
    from pyproj import Geod, Transformer
    res, _ = paths(args.out)
    s18 = _load("p18", "18_paper.py")
    gps = s18.read_gps()
    tr = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
    geod = Geod(ellps="WGS84")
    L = ["# 0 — Profile location: the land line is K1", "",
         "Electrode coordinates from `data/raw/GPS_coordinate.xlsx` (Belgian Lambert 72, EPSG:31370) converted to WGS84 "
         "with pyproj (EPSG:31370 → EPSG:4326). Reference point: coastal end of the French–Belgian border, "
         f"{BORDER[0]:.5f} N, {BORDER[1]:.5f} E. Along-coast distance = projection of the geodesic separation onto the "
         f"coast azimuth {COAST_AZ:.0f}°; cross-shore = the perpendicular component (+ = seaward, to the north-west).", "",
         "| electrode | role | Lambert 72 x, y (m) | WGS84 lat, lon | straight-line distance from the border (m) | "
         "along-coast (m) | cross-shore (m) |", "|---|---|---|---|---|---|---|"]
    out = {}
    for e, role in ((1, "first land electrode (dunes)"), (102, "last land electrode (low-water line)"),
                    (538, "last marine electrode (offshore)")):
        x, y = gps[e][0], gps[e][1]
        lon, lat = tr.transform(x, y)
        az, _, d = geod.inv(BORDER[1], BORDER[0], lon, lat)
        a = np.radians(az - COAST_AZ)
        along, cross = d * np.cos(a), -d * np.sin(a)
        out[e] = dict(lat=lat, lon=lon, dist=d, along=along, cross=cross)
        L.append(f"| {e} | {role} | {x:.1f}, {y:.1f} | {lat:.5f} N, {lon:.5f} E | {d:.0f} | {along:.0f} | {cross:+.0f} |")
    al = sorted([out[1]["along"], out[102]["along"]]); sl = sorted([out[1]["dist"], out[102]["dist"]])
    L += ["", f"**Result.** The land line (electrodes 1–102) lies {al[0]/1000:.2f}–{al[1]/1000:.2f} km "
          "north-east of the coastal end of the French–Belgian border, measured along the coast "
          f"(straight-line {sl[0]/1000:.2f}–{sl[1]/1000:.2f} km; the along-coast value depends on the assumed coast "
          f"azimuth of {COAST_AZ:.0f}°). This is profile **K1** of Lebbe (1981) "
          "and Paepen et al. (2020), not K0 (at the border, ≈ 1 km to the south-west).", "",
          "**Sources.**",
          "- Paepen et al. (2020), HESS 24:3539, Section 3.1.1: land ERT data were acquired on 7 March 2018 on \"one profile "
          "1 km east of the Belgian–French border, K1\" and on 11 October 2018 on \"one profile at the border, K0\"; "
          "Fig. 5 caption: \"K1 (7 March 2018) and K0 (11 October 2018). Profile numbering based on Lebbe (1981)\" "
          "(text extracted from `docs/literature/Paepen2020_HESS24_3539.pdf`); Fig. 1 (profile locations) cited from the "
          "project brief — the figure could not be rendered in this environment.",
          "- Hermans and Paepen (2020), GRL, Fig. 3: the same line position (cited from the project brief; the figure "
          "itself could not be rendered in this environment).",
          "- Survey date of our land data: 7 March 2018 (Paepen et al., 2020; to be confirmed by the authors); survey "
          "duration: not reported (\"multiple days\")."]
    with open(os.path.join(res, "0_profile_location.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
def field_products():
    """v1.0 headline field lens base (base-only BEL, site + physical, prior v3) and λ-pooled log-midpoint bands."""
    import stage6_analysis as s6
    s18 = _load("p18", "18_paper.py")
    s6.init(); s6.TX = s18.TX
    C = s6.load_combined(physical=True)
    obs, _ = s18.b13.observed_sets()
    Pf = s18.b13.BEL(C["tr"]["D"], C["tr"]["T"]).posterior(obs["field"], 2000, np.random.default_rng([129000, 1]), "kde")
    v4 = np.load(s18.LIN_V4)
    return s18.TX, Pf, v4["x"], v4["pooled_logmid"], v4["r_logmid"]


def cmd_run(args):
    _, dpp = paths(args.out)
    lb = read_lebbe()
    TX, Pf, x4, b4, r4 = field_products()
    out = dict(x=list(XREP), saline_base=[], fresh_top=[], bel=[], logmid=[], bel_minus_sal=[], logmid_minus_sal=[],
               sal_in_bel90=[])
    for x in XREP:
        sb_ = float(interp_boundary(x, lb["x_ours_m"], lb["saline_base_mTAW"]))
        ft = float(interp_boundary(x, lb["x_ours_m"], lb["fresh_top_mTAW"]))
        j = int(np.argmin(np.abs(TX - x))); k = int(np.argmin(np.abs(x4 - x)))
        lo, md, hi = np.percentile(Pf[:, j], [5, 50, 95])
        lm = [float(b4[1, k]), float(b4[2, k]), float(b4[3, k])]
        out["saline_base"].append(sb_); out["fresh_top"].append(ft)
        out["bel"].append([float(md), float(lo), float(hi)]); out["logmid"].append(lm)
        out["bel_minus_sal"].append(float(md - sb_)); out["logmid_minus_sal"].append(float(lm[0] - sb_))
        out["sal_in_bel90"].append(bool(lo <= sb_ <= hi))
    ex_ = dict(x=list(X_EXTRA), saline_base=[], fresh_top=[], bel=[], logmid=[], r_logmid=[], bel_minus_sal=[],
               logmid_minus_sal=[], sal_in_bel90=[])
    for x in X_EXTRA:
        sb_ = float(interp_boundary(x, lb["x_ours_m"], lb["saline_base_mTAW"]))
        ft = float(interp_boundary(x, lb["x_ours_m"], lb["fresh_top_mTAW"]))
        j = int(np.argmin(np.abs(TX - x))); k = int(np.argmin(np.abs(x4 - x)))
        lo, md, hi = np.percentile(Pf[:, j], [5, 50, 95])
        lm = [float(b4[1, k]), float(b4[2, k]), float(b4[3, k])]
        ex_["saline_base"].append(sb_); ex_["fresh_top"].append(ft); ex_["r_logmid"].append(float(r4[k]))
        ex_["bel"].append([float(md), float(lo), float(hi)]); ex_["logmid"].append(lm)
        ex_["bel_minus_sal"].append(float(md - sb_)); ex_["logmid_minus_sal"].append(float(lm[0] - sb_))
        ex_["sal_in_bel90"].append(bool(lo <= sb_ <= hi))
    out["extra"] = ex_
    np.savez(os.path.join(dpp, "k1_field_bel.npz"), TX=TX, Pf=Pf)
    json.dump(out, open(os.path.join(dpp, "k1_comparison.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res, dpp = paths(args.out)
    s18 = _load("p18", "18_paper.py")
    s18.mpl()
    c = json.load(open(os.path.join(dpp, "k1_comparison.json"), encoding="utf-8"))
    z = np.load(os.path.join(dpp, "k1_field_bel.npz")); TX, Pf = z["TX"], z["Pf"]
    v4 = np.load(s18.LIN_V4); x4, b4 = v4["x"], v4["pooled_logmid"]
    lb = read_lebbe()
    ex, ez = wc.land_surface()
    L = ["# 3 — K1: 1980 borehole-log structure (Lebbe 1981) vs the 2018 lens base (this study)", "",
         "1980 boundaries interpolated linearly from `data/external/lebbe1981_K1_digitised.csv` (vertical ±1 m, x ±30 m). "
         "BEL = v1.0 headline (base-only BEL, site + physical, prior v3); log-midpoint = linearised λ-pooled (v1.0). "
         "Elevations in mTAW; differences = 2018 median − 1980 saline base (negative = 2018 base deeper).", "",
         "| x (m) | 1980 saline base | 1980 fresh top | BEL median [P5, P95] | BEL − 1980 saline base | 1980 saline base in "
         "BEL 90 % | log-midpoint median [P5, P95] | log-midpoint − 1980 saline base |", "|---|---|---|---|---|---|---|---|"]
    for i, x in enumerate(c["x"]):
        b, l = c["bel"][i], c["logmid"][i]
        L.append(f"| {x} | {c['saline_base'][i]:.2f} | {c['fresh_top'][i]:.2f} | {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}] | "
                 f"{c['bel_minus_sal'][i]:+.2f} | {'yes' if c['sal_in_bel90'][i] else 'no'} | "
                 f"{l[0]:.2f} [{l[1]:.2f}, {l[2]:.2f}] | {c['logmid_minus_sal'][i]:+.2f} |")
    e = c["extra"]
    L += ["", "Seaward positions (v1.1.1; the lens base here lies near the discharge zone, x = 380–480 m; r = resolution "
          "radius at the λ = 50 log-midpoint crossing):", "",
          "| x (m) | 1980 saline base | 1980 fresh top | BEL median [P5, P95] | BEL − 1980 saline base | 1980 saline base in "
          "BEL 90 % | log-midpoint median [P5, P95] | log-midpoint − 1980 saline base | r (m) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i, x in enumerate(e["x"]):
        b, l = e["bel"][i], e["logmid"][i]
        L.append(f"| {x} | {e['saline_base'][i]:.2f} | {e['fresh_top'][i]:.2f} | {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}] | "
                 f"{e['bel_minus_sal'][i]:+.2f} | {'yes' if e['sal_in_bel90'][i] else 'no'} | "
                 f"{l[0]:.2f} [{l[1]:.2f}, {l[2]:.2f}] | {e['logmid_minus_sal'][i]:+.2f} | {e['r_logmid'][i]:.1f} |")
    L += ["", f"The 1980 saline base lies inside the BEL 90 % interval at {sum(c['sal_in_bel90'])} of {len(c['x'])} "
          "positions. These are two different times (1980 logs, 2018 ERT) and two different definitions (resistivity "
          "< 2.5 Ω·m in logs vs the centre of the base transition in the models); the comparison shows consistency or "
          "change, it does not validate either."]
    with open(os.path.join(res, "3_k1_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    # figure
    from matplotlib.patches import Patch
    r4 = v4["r_logmid"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ok = (b4[0] >= 0.5) & (r4 <= 10.0)                               # log-midpoint shown only where r ≤ 10 m
    ax.axvspan(*DIS_ZONE, color="#e6c229", alpha=0.16, lw=0, zorder=0)
    ax.fill_between(x4, np.where(ok, b4[2], np.nan), np.where(ok, b4[3], np.nan), color=s18.COL["logmid"], alpha=0.18, lw=0)
    h_lm, = ax.plot(x4, np.where(ok, b4[1], np.nan), color=s18.COL["logmid"], lw=1.1)
    lo, md, hi = np.percentile(Pf, [5, 50, 95], 0); h = TX >= 130
    ax.fill_between(TX[h], lo[h], hi[h], color=s18.COL["bel"], alpha=0.22, lw=0)
    h_bel, = ax.plot(TX[h], md[h], color=s18.COL["bel"], lw=1.7)
    hl = {}
    for key, mk, col in (("saline_base_mTAW", "o", "#b2182b"), ("fresh_top_mTAW", "s", "#1b9e77"),
                         ("lower_brackish_top_mTAW", "^", "#8c6d31")):
        v = lb[key]; m = np.isfinite(v)
        hl[key] = ax.errorbar(lb["x_ours_m"][m], v[m], xerr=X_ERR, yerr=Z_ERR, fmt=mk, color=col, ms=4, lw=0.8, capsize=1.5)
    h_clay, = ax.plot(lb["x_ours_m"], lb["clay_top_mTAW"], "--", color="0.35", lw=1.0)
    h_surf, = ax.plot(ex, ez, "k-", lw=0.9)
    ax.axvline(161, color="steelblue", lw=0.8, ls=":"); ax.text(163, 4.8, "high-water line", fontsize=6.3, color="steelblue")
    ax.set_xlim(0, 505); ax.set_ylim(-32, 8)
    s18.dune_hatch(ax)
    ax.set_xlabel("Distance along profile x (m)"); ax.set_ylabel("Elevation (mTAW)"); s18.ve_text(ax, 3)
    ax.set_aspect(3)
    blank = Patch(facecolor="none", edgecolor="none")
    handles = [blank, h_lm, h_bel, Patch(facecolor="#e6c229", alpha=0.35, edgecolor="none"), h_surf,
               blank, hl["saline_base_mTAW"], hl["fresh_top_mTAW"], hl["lower_brackish_top_mTAW"], h_clay]
    labels = ["This study, ERT data of 2018", "λ-pooled log-midpoint (r ≤ 10 m): median, P5–P95",
              "BEL (site + physical): median, P5–P95", "discharge zone (this study)", "land surface (2018 electrodes)",
              "Lebbe (1981), K1, logs of June–August 1980", "saline base (< 2.5 Ω·m)", "fresh top (> 20 Ω·m)",
              "lower brackish top", "clay top"]
    leg = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=6, facecolor="white",
                    edgecolor="0.6", framealpha=1.0, handletextpad=0.6, columnspacing=1.5)
    for t in leg.get_texts():
        if t.get_text() in (labels[0], labels[5]):
            t.set_fontweight("bold")
    ax.set_title("Profile K1: 1980 borehole logs vs 2018 lens base")
    fig.savefig(os.path.join(res, "figs", "F_S3_k1_1980_vs_2018.png"), dpi=300, bbox_inches="tight")
    ax.set_title("")
    fig.savefig(os.path.join(res, "figs", "F_S3_k1_1980_vs_2018_notitle.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("location", "run", "report"):
        sub.add_parser(c).add_argument("--out", default=wc.ROOT)
    a = ap.parse_args()
    dict(location=cmd_location, run=cmd_run, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
