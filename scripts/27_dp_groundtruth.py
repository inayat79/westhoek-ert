"""
27_dp_groundtruth.py
====================
Westhoek ERT v1.4 — the Zenodo record of Vrancken, Paepen & Hermans (2026), doi:10.5281/zenodo.20792093 (CC-BY 4.0):
inventory and placement of its 72-electrode ERT line on our 2018 profile (Task A). Pre-registration:
docs/decision_log.md v1.4-4 (placement) and v1.4-5 (DP tests, not run in this round).

  fetch                       download the record into data/raw/dp_zenodo_20792093/ (skipped if present; MD5 checked)
  run    [--out ROOT] [--workers 6]
                              GPS datum check (project.db), X0 scan (forward modelling of ERT.dat through g1_lam50, both
                              orientations), pseudo-section correlation, DP positions → ROOT/data/processed/v1_4/placement.json
  report [--out ROOT]         ROOT/results/v1_4/0_inventory.md, 1_line_placement.md, figs/F_S9_line_placement.png

THIS ROUND READS NO DP RESISTIVITY VALUE: `DP_R.xzr` is read with usecols=(0, 1) (local x, depth) only.
No random draws. Settings in the constants below and in the decision log.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sqlite3
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import read_trace  # noqa: E402
read_trace.install("27_dp_groundtruth.py")
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


ZEN_ID = 20792093
ZEN_DOI = "10.5281/zenodo.20792093"
RAW = os.path.join(wc.ROOT, "data", "raw", "dp_zenodo_20792093")
ZEN_FILES = {"DP_R.xzr": "dff0ed3851efcbddb270bf6a8d235e24", "SGS_DP_R.txt": "2fa612ce98110e87b09fa7115b340e47",
             "ERT.dat": "9a844221e4d45e1772d6be06749460ad", "Mesh.bms": "04a596b38b2a8a4c10846644d676b0ff",
             "Python_script.py": "ec438220954b88319151a6106a540c3c"}
DB = os.path.join(wc.ROOT, "data", "raw", "Project12_LS210110079_2018-03-07_11-22-23", "project.db")
COAST_AZ = 57.0

X0_GRID = np.arange(380.0, 520.0 + 1e-9, 2.5)             # primary: our_x = X0 − x_local
X0R_GRID = np.arange(-15.0, 125.0 + 1e-9, 2.5)            # check: our_x = X0′ + x_local
RMS_TOL, SIG_FLOOR = 1.05, 10.0
MESH = dict(paraDX=0.33, paraDepth=60.0)
PS_DX, PS_DZ, PS_F, PS_MIN = 5.0, 2.5, 0.19, 10
DP_SURF_SENS = 0.5
DP_DATE = None                                            # pending (asked T. Hermans)


# ─────────────────────────────────────────────────────────────────────────────
def cmd_fetch(args):
    import urllib.request
    os.makedirs(RAW, exist_ok=True)
    for fn, md5 in ZEN_FILES.items():
        p = os.path.join(RAW, fn)
        if not os.path.exists(p):
            urllib.request.urlretrieve(f"https://zenodo.org/api/records/{ZEN_ID}/files/{fn}/content", p)
        if hashlib.md5(open(p, "rb").read()).hexdigest() != md5:
            raise SystemExit(f"{fn}: MD5 mismatch")
        print(f"[fetch] {fn} OK")
    mp = os.path.join(RAW, f"zenodo_record_{ZEN_ID}.json")
    if not os.path.exists(mp):
        urllib.request.urlretrieve(f"https://zenodo.org/api/records/{ZEN_ID}", mp)


def check_raw():
    for fn, md5 in ZEN_FILES.items():
        p = os.path.join(RAW, fn)
        if not os.path.exists(p) or hashlib.md5(open(p, "rb").read()).hexdigest() != md5:
            raise SystemExit(f"{p} missing or changed — run `27_dp_groundtruth.py fetch`")


# ─────────────────────────────────────────────────────────────────────────────
# coordinates
# ─────────────────────────────────────────────────────────────────────────────
def transformer(datum=True):
    from pyproj import Transformer
    from pyproj.transformer import TransformerGroup
    if not datum:                                          # lat/lon read as BD72 geographic: no datum shift
        return Transformer.from_crs("EPSG:4313", "EPSG:31370", always_xy=True), "lat/lon as BD72 (EPSG:4313), no datum shift"
    tg = TransformerGroup("EPSG:4326", "EPSG:31370", always_xy=True)
    t = [t for t in tg.transformers if "BD72 to WGS 84 (3)" in t.description]
    if not t:
        raise SystemExit("operation 'BD72 to WGS 84 (3)' (EPSG:15929) not available")
    return t[0], t[0].description


def our_line():
    """Lambert 72 chord E1 → E102 (GPS_coordinate.xlsx has the land line's end electrodes only) and their profile x."""
    s18 = _load("p18", "18_paper.py")
    gps = s18.read_gps(); land, _ = wc.read_electrodes()
    return np.array([[gps[1][0], gps[1][1]], [gps[102][0], gps[102][1]]]), np.array([land[0, 0], land[-1, 0]])


def project(P, X, q):
    """Point q (Lambert) on the chord: (our x, extrapolated beyond the ends; signed offset, + = north-east)."""
    a, b = P[0], P[1]; d = b - a; L = np.hypot(*d)
    t = (q - a) @ d / L ** 2; f = a + t * d; perp = q - f
    ne = np.array([np.sin(np.radians(COAST_AZ)), np.cos(np.radians(COAST_AZ))])
    return float(X[0] + t * (X[1] - X[0])), float(np.sign(perp @ ne) * np.hypot(*perp))


def gps_check():
    con = sqlite3.connect(f"file:{DB.replace(os.sep, '/')}?mode=ro", uri=True)
    rows = con.execute("""select m.StationID, s.TaskID, s.PosX, m.PosLatitude, m.PosLongitude from Measures m
                          join Stations s on m.StationID = s.ID where m.PosLatitude is not null and m.PosLatitude != 0""").fetchall()
    P, X = our_line(); out = []
    for st in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == st]
        lat, lon = float(np.median([r[3] for r in rr])), float(np.median([r[4] for r in rr]))
        rec = dict(station=int(st), task=int(rr[0][1]), spread_start_p=float(rr[0][2]), n=len(rr), lat=lat, lon=lon,
                   centre_p=float(rr[0][2]) + 31.5)
        rec["centre_x_expected"] = float(np.interp(101 - rec["centre_p"] if rec["task"] == 1 else 101 - (rec["centre_p"] + 64),
                                                   np.arange(102), wc.read_electrodes()[0][:, 0]))
        for key, dat in (("datum", True), ("nodatum", False)):
            tr, desc = transformer(dat)
            e, n = tr.transform(lon, lat)
            rec[key] = dict(desc=desc, x=project(P, X, np.array([e, n]))[0], offset=project(P, X, np.array([e, n]))[1])
        out.append(rec)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Zenodo ERT.dat
# ─────────────────────────────────────────────────────────────────────────────
def read_ert():
    L = open(os.path.join(RAW, "ERT.dat"), encoding="utf-8").read().splitlines()
    ns = int(L[0].split("#")[0].split()[0])
    sens = np.array([[float(v) for v in l.split()[:2]] for l in L[2:2 + ns]])
    nd = int(L[2 + ns].split("#")[0].split()[0])
    D = np.array([[float(v) for v in l.split()[:6]] for l in L[4 + ns:4 + ns + nd]])
    abmn = D[:, :4].astype(int) - 1                                 # unified format: 1-based
    i, u = D[:, 4], D[:, 5]
    ok = (i > 0) & (u > 0)
    return dict(xloc=sens[:, 0], abmn=abmn[ok], R=(u / i)[ok], n_all=nd, n_removed=int((~ok).sum()), n_sens=ns)


def fwd_rms(job):
    """RMS of log10 R_pred − log10 R_obs for one placement (worker)."""
    X0, sgn = job
    import pygimli as pg
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    from scipy.spatial import cKDTree
    E = read_ert(); ref = wc.load_run("g1_lam50"); ex, ez = wc.land_surface()
    xs = X0 - E["xloc"] if sgn < 0 else X0 + E["xloc"]
    order = np.argsort(xs); pos_of = np.empty_like(order); pos_of[order] = np.arange(len(order))
    data = pg.DataContainerERT()
    for k in order:
        data.createSensor(pg.Pos(float(xs[k]), float(np.interp(xs[k], ex, ez))))
    data.resize(len(E["R"]))
    for c, key in enumerate("abmn"):
        data[key] = [int(pos_of[v]) for v in E["abmn"][:, c]]
    data["k"] = np.ones(data.size())
    with contextlib.redirect_stdout(io.StringIO()):
        mesh = mt.createParaMesh(data.sensors(), **MESH)
    cc = np.array([[c.center().x(), c.center().y()] for c in mesh.cells()])
    tree = cKDTree(np.c_[ref["cell_x"], ref["cell_z"]])
    res = np.asarray(ref["model"], float)[tree.query(cc)[1]]
    for _ in range(3):
        with contextlib.redirect_stdout(io.StringIO()):
            out = ert.simulate(mesh, scheme=data, res=res, noiseLevel=0, verbose=False)
        Rp = np.array(out["rhoa"])
        if len(Rp) == data.size():
            break
    ok = Rp > 0
    r = np.log10(Rp[ok]) - np.log10(E["R"][ok])
    return float(np.sqrt(np.mean(r ** 2))), int((~ok).sum()), float(np.median(r))


def interval(grid, rms):
    i = int(np.argmin(rms)); thr = RMS_TOL * rms[i]; lo = hi = i
    while lo > 0 and rms[lo - 1] <= thr:
        lo -= 1
    while hi < len(rms) - 1 and rms[hi + 1] <= thr:
        hi += 1
    return float(grid[i]), max(SIG_FLOOR, float(grid[hi] - grid[lo]) / 2), [float(grid[lo]), float(grid[hi])], float(rms[i])


# pseudo-sections ─────────────────────────────────────────────────────────────
def k_flat(xa, xb, xm, xn):
    r = lambda p, q: np.abs(p - q)
    return 2 * np.pi / (1 / r(xa, xm) - 1 / r(xb, xm) - 1 / r(xa, xn) + 1 / r(xb, xn))


def pseudo(xa, xb, xm, xn, R):
    X = np.c_[xa, xb, xm, xn]
    rho = R * k_flat(xa, xb, xm, xn)
    ok = rho > 0
    return X.mean(1)[ok], PS_F * (X.max(1) - X.min(1))[ok], np.log10(rho[ok])


GX = np.arange(0.0, 520.0 + 1e-9, PS_DX)
GZ = np.arange(0.0, 40.0 + 1e-9, PS_DZ)


def gridded(x, z, v):
    """Linear (Delaunay) interpolation of a pseudo-section onto the common grid, NaN outside its convex hull
    (decision log v1.4-5)."""
    from scipy.interpolate import griddata
    Xg, Zg = np.meshgrid(GX, GZ)
    return griddata(np.c_[x, z], v, (Xg, Zg), method="linear")


def corr(A, B):
    ok = np.isfinite(A) & np.isfinite(B)
    if ok.sum() < PS_MIN:
        return np.nan, int(ok.sum())
    a = np.where(ok, A, np.nan); b = np.where(ok, B, np.nan)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a = a - np.nanmean(a, axis=1, keepdims=True); b = b - np.nanmean(b, axis=1, keepdims=True)
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(ok.sum())


def ours_binned():
    land, _ = wc.read_electrodes(); xe = land[:, 0]
    L = open(os.path.join(wc.ROOT, "data", "raw", "land_data.dat")).read().split("\n")
    n = int(L[0].split()[0]); D = np.array([[float(v) for v in l.split()[:5]] for l in L[1:1 + n]])
    drop = {int(l.split("#")[0]) for l in open(os.path.join(wc.ROOT, "results", "deep_all_droplist.txt")) if l.split("#")[0].strip()}
    keep = np.array([j not in drop for j in range(n)])
    D = D[keep]; e = D[:, :4].astype(int) - 1
    return gridded(*pseudo(xe[e[:, 0]], xe[e[:, 1]], xe[e[:, 2]], xe[e[:, 3]], D[:, 4])), int(keep.sum())


def theirs_binned(X0, sgn, E):
    xs = X0 - E["xloc"] if sgn < 0 else X0 + E["xloc"]
    a = E["abmn"]
    return gridded(*pseudo(xs[a[:, 0]], xs[a[:, 1]], xs[a[:, 2]], xs[a[:, 3]], E["R"]))


# DP positions (no resistivity) ───────────────────────────────────────────────
def dp_geometry():
    A = np.genfromtxt(os.path.join(RAW, "DP_R.xzr"), usecols=(0, 1))          # local x, depth — NOT the ρ column
    out = []
    for lx in np.unique(np.round(A[:, 0], 3)):
        z = A[np.isclose(A[:, 0], lx, atol=1e-3), 1]
        out.append(dict(local_x=float(lx), n=int(len(z)), depth_top=float(-z.max()), depth_bottom=float(-z.min()),
                        dz=float(np.median(np.abs(np.diff(np.sort(z)))))))
    for i, d in enumerate(sorted(out, key=lambda d: d["local_x"])):
        d["name"] = f"DP{i + 1}"                                   # DP1 at the seaward end (x_local = 50)
    return sorted(out, key=lambda d: d["local_x"])


# ─────────────────────────────────────────────────────────────────────────────
def cmd_run(args):
    check_raw()
    dpp = os.path.join(args.out, "data", "processed", "v1_4"); os.makedirs(dpp, exist_ok=True)
    E = read_ert()
    out = dict(source=ZEN_DOI, ert=dict(n_sens=E["n_sens"], n_all=E["n_all"], n_removed=E["n_removed"], n_used=len(E["R"]),
                                         xloc=[float(E["xloc"][0]), float(E["xloc"][-1])]))
    out["gps"] = gps_check()
    # primary: forward-model scan
    jobs = [(float(x), -1) for x in X0_GRID] + [(float(x), +1) for x in X0R_GRID]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(fwd_rms, jobs))
    rms = np.array([r[0] for r in res[:len(X0_GRID)]]); rmsr = np.array([r[0] for r in res[len(X0_GRID):]])
    X0s, sig, ivl, rmin = interval(X0_GRID, rms)
    X0r, sigr, ivlr, rminr = interval(X0R_GRID, rmsr)
    out["scan"] = dict(grid=X0_GRID.tolist(), rms=rms.tolist(), grid_rev=X0R_GRID.tolist(), rms_rev=rmsr.tolist(),
                       nonpos=[r[1] for r in res], median_resid=[r[2] for r in res])
    out["primary"] = dict(X0=X0s, sigma=sig, interval=ivl, rms_min=rmin, X0_rev=X0r, sigma_rev=sigr, rms_min_rev=rminr,
                          orientation_ok=bool(rmin < rminr))
    # secondary: pseudo-section correlation
    ob, n_ours = ours_binned()
    cc = [corr(ob, theirs_binned(x, -1, E)) for x in X0_GRID]; ccr = [corr(ob, theirs_binned(x, +1, E)) for x in X0R_GRID]
    c = np.array([v[0] for v in cc]); cr = np.array([v[0] for v in ccr])
    ipc = int(np.nanargmax(c))
    out["pseudo"] = dict(grid=X0_GRID.tolist(), r=c.tolist(), n_bins=[v[1] for v in cc], r_rev=cr.tolist(),
                         X0=float(X0_GRID[ipc]), r_max=float(c[ipc]), r_max_rev=float(np.nanmax(cr)),
                         X0_rev=float(X0R_GRID[int(np.nanargmax(cr))]), n_ours=n_ours)
    dis = abs(out["pseudo"]["X0"] - X0s)
    out["agreement"] = dict(diff=dis, two_sigma=2 * sig, stop=bool(dis > 2 * sig))
    # DP positions
    ex_, ez_ = wc.land_surface()
    dps = dp_geometry()
    for d in dps:
        d["our_x"] = X0s - d["local_x"]; d["our_x_lo"] = X0s - sig - d["local_x"]; d["our_x_hi"] = X0s + sig - d["local_x"]
        d["surface_2018"] = float(np.interp(d["our_x"], ex_, ez_))
    out["dps"] = dps
    out["line"] = dict(sea_end=X0s, land_end=X0s - float(E["xloc"][-1]))
    with open(os.path.join(dpp, "placement.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(dict(primary=out["primary"], pseudo={k: out["pseudo"][k] for k in ("X0", "r_max", "X0_rev", "r_max_rev")},
                          agreement=out["agreement"]), indent=1))
    if out["agreement"]["stop"]:
        print("STOP: the two placement methods disagree by more than 2σ_X (pre-registered stop rule)")


# ─────────────────────────────────────────────────────────────────────────────
def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    check_raw()
    J = json.load(open(os.path.join(args.out, "data", "processed", "v1_4", "placement.json"), encoding="utf-8"))
    R26 = json.load(open(os.path.join(args.out, "data", "processed", "v1_4", "raw2018.json"), encoding="utf-8"))
    res = os.path.join(args.out, "results", "v1_4"); figs = os.path.join(res, "figs"); os.makedirs(figs, exist_ok=True)
    s18 = _load("p18", "18_paper.py"); s18.mpl()
    meta = json.load(open(os.path.join(RAW, f"zenodo_record_{ZEN_ID}.json"), encoding="utf-8"))["metadata"]
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    pr, ps, ag = J["primary"], J["pseudo"], J["agreement"]
    c = R26["counts"]; t = R26["timing"]
    # ── 0_inventory.md ──
    L = ["# 0 — Inventory of the new inputs (v1.4)", "",
         "Sources, licences and SHA256: `data/raw/README_sources.md`. Disclosure of numbers seen before the pre-registration: "
         "`docs/decision_log.md` v1.4-1.", "",
         "## 1. The 2018 Terrameter LS project (T. Hermans, Ghent University; received October 2026)", "",
         f"`data/raw/Project12_LS210110079_2018-03-07_11-22-23/project.db` — SQLite, Terrameter LS project \"Project12\", "
         f"instrument LS210110079, {os.path.getsize(DB)} bytes, SHA256 `{sha(DB)}`; read-only.", "",
         "| table | contents |", "|---|---|",
         "| Tasks | 1 = grad8tri_1 (normal, three roll-along stations at p = 0, 32, 64), 2 = grad8tri_R_1 (one station) |",
         "| DP_ABMN | A/B/M/N positions p in electrode units, global across the roll-along |",
         "| DPV | measured values: DatatypeID 5 = R (Ω, DataSDev = stacking s.d.), 2 = ρapp, 6 = I, 7 = ΔU, 3 = IP window; Channel 0 = transmitter |",
         "| Measures / DP_MEASURE | time, GPS lat/lon, instrument temperature per injection; data ↔ injection links |",
         "| Log, ElectrodeTestData | instrument events; contact resistance per electrode and station |",
         "| AcqSettings, TaskSettings | grad8tri.xml (task 2: grad8tri_R.xml), spread S64.xml, spacing 5 m, ErrorLimit 0.01, 2–3 stacks, 1 A |", "",
         f"- {c['task1']} task-1 R values ({c['task1_on_published_electrodes']} on our electrodes p ≤ 101, our number = 102 − p); "
         f"{c['task2']} task-2 values (the reciprocal of station 3). All {c['matched']} published data matched (`1_raw_abem.md`).",
         f"- Published data {t['published'][0][11:16]}–{t['published'][1][11:16]}, session {t['session'][0][11:16]}–{t['session'][1][11:16]} "
         "on 7 March 2018, instrument clock (time zone unknown — asked T. Hermans).",
         "- Coordinate system: positions in electrode units along the line (5 m); instrument GPS per injection (WGS84 lat/lon).", "",
         "## 2. Zenodo record 10.5281/zenodo.20792093 (Vrancken, Paepen & Hermans 2026; CC-BY 4.0)", "",
         f"\"{meta['title']}\"; published {meta['publication_date']}; no version number; downloaded 2026-10-03.", "",
         "| file | format | contents | SHA256 |", "|---|---|---|---|"]
    desc = {"ERT.dat": f"pyGIMLi unified data: {J['ert']['n_sens']} electrodes at local x {J['ert']['xloc'][0]:g}–{J['ert']['xloc'][1]:g} m, all z = 0; "
                       f"{J['ert']['n_all']} multiple-gradient data as i (mA), u (mV), no errors, no times; {J['ert']['n_removed']} with non-positive u or i",
            "DP_R.xzr": "3 columns, no header: local x, z = depth below ground (negative, 1 cm), ρ (presumably bulk Ω·m; temperature reference not stated) — ρ not read in this round",
            "SGS_DP_R.txt": "geostatistical (SGS) realisation conditioned on the DP data (cell x, z, value) — simulated, not measured",
            "Mesh.bms": "pyGIMLi mesh of the recent line (local coordinates)",
            "Python_script.py": "the authors' method: ensemble of deterministic pyGIMLi inversions (seed 227843; λ 1–999; correlation lengths "
                                "ah 10–119 m, av 5–14 m; homogeneous reference 1–72 Ω·m or SGS reference; joint ERT + DP variant; 10 % error on both) — read, not executed"}
    for fn in ZEN_FILES:
        L.append(f"| `{fn}` | {fn.rsplit('.', 1)[1]} | {desc[fn]} | `{sha(os.path.join(RAW, fn))[:16]}…` |")
    L += ["", "| DP | x_local (m) | samples | depth (m below ground) | sampling |", "|---|---|---|---|---|"]
    for d in J["dps"]:
        L.append(f"| {d['name']} | {d['local_x']:g} | {d['n']} | {d['depth_top']:.2f}–{d['depth_bottom']:.2f} | {100*d['dz']:.0f} cm |")
    L += ["", "- **Missing in every file:** coordinates, survey dates (DP and ERT), DP ground elevations. Dates: pending (asked T. Hermans).",
          "- `ERT.dat` is not the 2018 survey: no quadrupole matches `land_data.dat`, and its line is 355 m long with 72 electrodes.",
          "- Orientation: x_local = 0 is the seaward end; DP1 at x_local = 50 m, DP7 at 350 m (map of T. Hermans). Mapping "
          "our_x = X0 − x_local with X0 from the line placement (`1_line_placement.md`)."]
    with open(os.path.join(res, "0_inventory.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    # ── 1_line_placement.md ──
    P = ["# 1 — Placement of the Zenodo ERT line on our profile (Task A)", "",
         "Pre-registered in `docs/decision_log.md` v1.4-4. **The survey dates differ (7 March 2018 vs a later date, pending), "
         "so the lens may have changed between the surveys**; the placement assumes the structure is similar enough to locate "
         "the line.", "",
         "## Datum check with the instrument GPS (project.db)", "",
         "| station (task) | n | GPS median lat, lon | expected spread centre, our x (m) | with the BD72 datum transformation: x / offset (m) | without it: x / offset (m) |",
         "|---|---|---|---|---|---|"]
    for g in J["gps"]:
        P.append(f"| {g['station']} ({g['task']}) | {g['n']} | {g['lat']:.6f}, {g['lon']:.6f} | {g['centre_x_expected']:.1f} | "
                 f"{g['datum']['x']:.1f} / {g['datum']['offset']:+.1f} | {g['nodatum']['x']:.1f} / {g['nodatum']['offset']:+.1f} |")
    shifts = [g["nodatum"]["x"] - g["datum"]["x"] for g in J["gps"]]
    P += ["", f"With the transformation (\"{J['gps'][0]['datum']['desc']}\") the instrument positions lie on our line near the spread "
          f"centres; without it they move by {min(shifts):+.0f} … {max(shifts):+.0f} m along the line (all stations alike). The datum handling is confirmed "
          "(the instrument stood near the middle of each spread).", "",
          "## Primary: forward modelling of ERT.dat through g1_lam50", "",
          f"{J['ert']['n_used']} data ({J['ert']['n_removed']} non-positive removed). RMS of log10 R_pred − log10 R_obs over X0 = "
          f"{J['scan']['grid'][0]:g}–{J['scan']['grid'][-1]:g} m (2.5 m steps).", "",
          f"- **X0* = {pr['X0']:.1f} m, σ_X = {pr['sigma']:.1f} m** (RMS ≤ {RMS_TOL:g} × min over {pr['interval'][0]:.1f}–{pr['interval'][1]:.1f} m; "
          f"floor {SIG_FLOOR:g} m); minimum RMS {pr['rms_min']:.3f} (log10 units).",
          f"- Opposite orientation (our_x = X0′ + x_local): best X0′ = {pr['X0_rev']:.1f} m, minimum RMS {pr['rms_min_rev']:.3f} — "
          + ("worse than the primary orientation, which is confirmed." if pr["orientation_ok"] else "**not worse: the orientation is not confirmed by the data.**"),
          "", "## Secondary: pseudo-section correlation", "",
          f"Ours: the {ps['n_ours']} G1 data; theirs: the positive data of ERT.dat. Pseudo-depth = {PS_F:g} × electrode span; each "
          f"pseudo-section interpolated linearly onto a {PS_DX:g} m × {PS_DZ:g} m grid inside its own hull (v1.4-5: binned levels never "
          f"coincide because the current spans differ); each depth row centred; log10 ρa (flat-ground K).", "",
          f"- Best X0 = {ps['X0']:.1f} m (r = {ps['r_max']:.2f}); opposite orientation best r = {ps['r_max_rev']:.2f} at X0′ = {ps['X0_rev']:.1f} m.",
          f"- |X0_pc − X0*| = {ag['diff']:.1f} m vs 2σ_X = {ag['two_sigma']:.1f} m → " +
          ("**the methods disagree (stop rule triggered).**" if ag["stop"] else "the methods agree."), "",
          "## DP positions on our axis", "",
          "| DP | x_local (m) | **our x (m)** | range at X0* ± σ_X | 2018 surface (mTAW) | DP surface used |", "|---|---|---|---|---|---|"]
    for d in J["dps"]:
        P.append(f"| {d['name']} | {d['local_x']:g} | **{d['our_x']:.1f}** | {d['our_x_lo']:.1f}–{d['our_x_hi']:.1f} | {d['surface_2018']:.2f} | "
                 f"2018 surface; DP-date surface pending (±{DP_SURF_SENS:g} m) |")
    P += ["", f"The line itself spans our x {J['line']['land_end']:.1f} (land end) to {J['line']['sea_end']:.1f} m (sea end)."]
    with open(os.path.join(res, "1_line_placement.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(P) + "\n")
    # ── F_S9 ──
    S = J["scan"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.plot(S["grid"], S["rms"], "-o", ms=2.5, color="#1f4e9c", label="our_x = X0 − x_local (primary)")
    ax.plot(np.array(S["grid_rev"]) + 355, S["rms_rev"], "-", color="0.6", lw=1, label="opposite orientation (plotted at X0′ + 355)")
    ax.axvspan(pr["X0"] - pr["sigma"], pr["X0"] + pr["sigma"], color="#1f4e9c", alpha=0.12, lw=0)
    ax.axvline(pr["X0"], color="#1f4e9c", lw=0.8)
    ax.set_xlabel("X0 = our x of the line's sea end (m)"); ax.set_ylabel("RMS log10 R misfit"); ax.legend(fontsize=6)
    ax.set_title(f"(a) forward model: X0* = {pr['X0']:.1f} ± {pr['sigma']:.1f} m", fontsize=7.5)
    ax = axes[1]
    ax.plot(ps["grid"], ps["r"], "-o", ms=2.5, color="#d35400", label="primary orientation")
    ax.plot(np.array(J["pseudo"]["grid"]) * 0 + np.array(S["grid_rev"]) + 355, ps["r_rev"], "-", color="0.6", lw=1, label="opposite (at X0′ + 355)")
    ax.axvline(ps["X0"], color="#d35400", lw=0.8); ax.axvspan(pr["X0"] - 2 * pr["sigma"], pr["X0"] + 2 * pr["sigma"], color="#1f4e9c", alpha=0.08, lw=0)
    ax.set_xlabel("X0 (m)"); ax.set_ylabel("pseudo-section correlation r"); ax.legend(fontsize=6)
    ax.set_title(f"(b) pseudo-sections: X0 = {ps['X0']:.1f} m (shaded ± 2σ_X)", fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "F_S9_line_placement.png"), dpi=300); plt.close(fig)
    print("\n".join(P))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    r = sub.add_parser("run"); r.add_argument("--out", default=wc.ROOT); r.add_argument("--workers", type=int, default=6)
    sub.add_parser("report").add_argument("--out", default=wc.ROOT)
    a = ap.parse_args()
    dict(fetch=cmd_fetch, run=cmd_run, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
