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
    if os.path.exists(os.path.join(args.out, "data", "processed", "v1_4", "cpt.json")):
        Lc, Mc = report_cpt(args, plt, s18)
        print("\n".join(Lc + [""] + Mc))
    report_method(args)
    if os.path.exists(os.path.join(args.out, "data", "processed", "v1_4", "dp_results.json")):
        Ld, Md = report_dp(args, plt, s18)
        print("\n".join(Ld + [""] + Md))


# ─────────────────────────────────────────────────────────────────────────────
# Tasks E and F — CPT soundings of E. Vrancken (23 September 2024): geometry and lithology. No DP resistivity.
# ─────────────────────────────────────────────────────────────────────────────
CPT_DIR = os.path.join(wc.ROOT, "data", "raw", "cpt_vrancken_2024")
CPT_ZIP = os.path.join(CPT_DIR, "CPT.zip")
CPT_DATE = "2024-09-23"
PA = 0.1                                                   # MPa
ISBT_SAND, ISBT_FINE = 2.05, 2.6
FINE_MIN = 2                                               # consecutive 0.2 m samples (≥ 0.4 m)
DEEP_Z, DEEP_THICK = -20.0, 2.0
RF_STEP, RF_RUN = 2.0, 5                                   # descriptive clay marker: Rf > 2 % over ≥ 1 m (5 samples) below −20 mTAW


def rf_step(z, Rf):
    """Top (mTAW) of the first ≥ 1 m run with Rf > 2 % below −20 mTAW (descriptive marker of the clay top)."""
    m = (np.asarray(Rf) > RF_STEP) & (np.asarray(z) < DEEP_Z)
    for i in range(len(m) - RF_RUN + 1):
        if m[i:i + RF_RUN].all():
            return float(z[i] + 0.1)
    return None


def cpt_check():
    """The extracted files must be byte-identical to the members of the original archive."""
    import zipfile
    if not os.path.exists(CPT_ZIP):
        raise SystemExit(f"{CPT_ZIP} missing (unpublished CPT data from E. Vrancken; not redistributed)")
    z = zipfile.ZipFile(CPT_ZIP); out = {}
    for n in z.namelist():
        p = os.path.join(CPT_DIR, "CPT", n)
        b = z.read(n)
        if not os.path.exists(p) or open(p, "rb").read() != b:
            raise SystemExit(f"{p} differs from the archive member {n}")
        out[n] = hashlib.sha256(b).hexdigest()
    out["CPT.zip"] = hashlib.sha256(open(CPT_ZIP, "rb").read()).hexdigest()
    return out


def fnum(s):
    return float(s.replace(",", "."))


def read_cpt():
    L = open(os.path.join(CPT_DIR, "CPT", "Location.txt"), encoding="utf-8").read().splitlines()
    loc = {}
    for l in L[1:]:
        if l.strip():
            p = l.split(";"); loc[p[0]] = dict(x=fnum(p[1]), y=fnum(p[2]), z=fnum(p[3]))
    cpts = {}
    for sid in sorted(loc):
        k = int(sid[1:]); fn = os.path.join(CPT_DIR, "CPT", f"Proef{k}.txt")
        H = open(fn, encoding="utf-8").read().splitlines()
        hdr = H[2].split()
        rows = np.array([[fnum(v) for v in l.split()] for l in H[4:] if l.strip()])
        cpts[sid] = dict(file=os.path.basename(fn), title=H[0].strip(), header_surface=float(hdr[0]), type=hdr[1],
                         cone=hdr[3], diameter_mm=float(hdr[5]), columns=H[3].split(), depth=rows[:, 0], level=rows[:, 1],
                         qc=rows[:, 2], fs=rows[:, 3], Rf=rows[:, 4], **loc[sid])
    return cpts


def isbt(qc, Rf):
    with np.errstate(divide="ignore", invalid="ignore"):
        I = np.sqrt((3.47 - np.log10(qc / PA)) ** 2 + (np.log10(Rf) + 1.22) ** 2)
    return np.where((qc > 0) & (Rf > 0), I, np.nan)


def runs_of(mask, z, dz=0.2, nmin=FINE_MIN):
    out = []; i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            n = j - i + 1
            if n >= nmin:
                out.append(dict(top=float(z[i] + dz / 2), bottom=float(z[j] - dz / 2), thick=float(n * dz), n=int(n)))
            i = j + 1
        else:
            i += 1
    return out


def cmd_cpt(args):
    """Tasks E and F: no DP resistivity value is read."""
    check_raw()
    dpp = os.path.join(args.out, "data", "processed", "v1_4"); os.makedirs(dpp, exist_ok=True)
    sha = cpt_check()
    cpts = read_cpt()
    P, X = our_line(); ex, ez = wc.land_surface()
    pl = json.load(open(os.path.join(args.out, "data", "processed", "v1_4", "placement.json"), encoding="utf-8"))
    X0s = pl["primary"]["X0"]; dpg = {d["name"]: d for d in dp_geometry()}
    out = dict(sha256=sha, date=CPT_DATE, cpts={})
    for sid, c in cpts.items():
        k = int(sid[1:]); dp = f"DP{k - 1}"
        x, off = project(P, X, np.array([c["x"], c["y"]]))
        I = isbt(c["qc"], c["Rf"]); zz = c["z"] - c["depth"]
        fine = runs_of(I > ISBT_FINE, zz)
        deep = [r for r in fine if r["bottom"] < DEEP_Z and r["thick"] > DEEP_THICK]
        cls = np.where(I < ISBT_SAND, "sand", np.where(I <= ISBT_FINE, "mixture", "fine"))
        out["cpts"][sid] = dict(dp=dp, file=c["file"], title=c["title"], lambert_x=c["x"], lambert_y=c["y"], our_x=x, offset=off,
                                z=c["z"], header_surface=c["header_surface"], surf2018=float(np.interp(x, ex, ez)),
                                type=c["type"], cone=c["cone"], diameter_mm=c["diameter_mm"], columns=c["columns"],
                                end_depth=float(c["depth"][-1]), dp_end_depth=dpg[dp]["depth_bottom"], dp_local_x=dpg[dp]["local_x"],
                                x_taskA=X0s - dpg[dp]["local_x"], n=int(len(I)), dz=float(np.median(np.diff(c["depth"]))),
                                level_minus_z_from_header=float(np.median(c["level"] - (c["header_surface"] - c["depth"]))),
                                frac={k_: float(np.mean(cls == k_)) for k_ in ("sand", "mixture", "fine")},
                                fine_layers=fine, deep_unit=(deep[0] if deep else None), rf_step=rf_step(zz, c["Rf"]),
                                elev=zz.tolist(), qc=c["qc"].tolist(), fs=c["fs"].tolist(), Rf=c["Rf"].tolist(),
                                isbt=[None if not np.isfinite(v) else float(v) for v in I])
    C = out["cpts"]
    xs = np.array([C[s]["our_x"] for s in sorted(C)]); xl = np.array([C[s]["dp_local_x"] for s in sorted(C)])
    X0f = float(np.mean(xs + xl)); res = xs - (X0f - xl)
    out["fit"] = dict(X0_fit=X0f, rms=float(np.sqrt(np.mean(res ** 2))), X0_taskA=X0s, sigma_taskA=pl["primary"]["sigma"],
                      resid_taskA=[float(v) for v in xs - (X0s - xl)], spacing=[float(v) for v in -np.diff(xs)])
    with open(os.path.join(dpp, "cpt.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["fit"], indent=1))


def report_cpt(args, plt, s18):
    J = json.load(open(os.path.join(args.out, "data", "processed", "v1_4", "cpt.json"), encoding="utf-8"))
    res = os.path.join(args.out, "results", "v1_4"); figs = os.path.join(res, "figs")
    C = J["cpts"]; F = J["fit"]; ids = sorted(C)
    # 5_cpt_inventory.md
    L = ["# 5 — CPT inventory and surveyed DP geometry (Task E)", "",
         "Mechanical CPT soundings of E. Vrancken (Ghent University; unpublished, used with permission, not to be "
         "redistributed; received 8 October 2026), carried out on **23 September 2024**. Per E. Vrancken and T. Hermans the "
         "Zenodo DP resistivity logs are CPT-EC measurements from the same pushes: same date and positions. Raw files: "
         "`data/raw/cpt_vrancken_2024/` (git-ignored; available from the data owner). Extracted files verified byte-identical "
         "to the archive members.", "",
         "| file | SHA256 |", "|---|---|"] + [f"| `{k}` | `{v}` |" for k, v in sorted(J["sha256"].items())]
    c0 = C[ids[0]]
    L += ["", "## Format (verified)", "",
          "- `Location.txt`: columns Proef;x;y;z(mTAW), decimal comma; S02–S08. Coordinates are Belgian Lambert 72 "
          f"(EPSG:31370): projected on our line they lie within {max(abs(C[s]['offset']) for s in ids):.1f} m of it, which "
          "confirms the system.",
          f"- `ProefN.txt`: a title line, a header line and its values (surface level, type \"{c0['type']}\", cone {c0['cone']}, "
          f"{c0['diameter_mm']:g} mm), then columns {', '.join(c0['columns'])}; {c0['dz']:.1f} m steps, decimal comma. qc and fs in "
          "N/mm² = MPa, Rf in %. **No pore-pressure column despite the \"CPTU\" label.**",
          "- Flags: " + "; ".join(
              ([f"{C[s]['file']} header surface {C[s]['header_surface']:.2f} mTAW vs Location.txt {C[s]['z']:.2f} (Location.txt used)"
                for s in ids if abs(C[s]['header_surface'] - C[s]['z']) > 0.005] +
               [f"{C[s]['file']} is titled \"{C[s]['title']}\"" for s in ids if C[s]["title"].replace(" ", "") != f"Proef{int(s[1:])}"])) + ".",
          "", "## Geometry: one row per DP (CPT S0k = DP(k − 1))", "",
          "| DP | CPT | Lambert 72 x, y (m) | our x (m) | offset (m) | CPT ground (mTAW) | 2018 surface (mTAW) | CPT − 2018 (m) | "
          "CPT end depth (m) | DP end depth (m) | Task A: X0* − x_local (m) | surveyed − Task A (m) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in zip(ids, F["resid_taskA"]):
        c = C[s]
        L.append(f"| {c['dp']} | {s} | {c['lambert_x']:.2f}, {c['lambert_y']:.2f} | {c['our_x']:.1f} | {c['offset']:+.1f} | {c['z']:.2f} | "
                 f"{c['surf2018']:.2f} | {c['z'] - c['surf2018']:+.2f} | {c['end_depth']:.2f} | {c['dp_end_depth']:.2f} | "
                 f"{c['x_taskA']:.1f} | {r:+.1f} |")
    L += ["", f"Spacing between consecutive CPTs: {min(F['spacing']):.1f}–{max(F['spacing']):.1f} m. End depths: CPT − DP = "
          f"{min(C[s]['end_depth'] - C[s]['dp_end_depth'] for s in ids):+.2f} … {max(C[s]['end_depth'] - C[s]['dp_end_depth'] for s in ids):+.2f} m.", "",
          "## Independent check of the Task A placement", "",
          f"The surveyed positions differ from the Task A positions (X0* = {F['X0_taskA']:.1f} ± {F['sigma_taskA']:.1f} m) by "
          f"{min(F['resid_taskA']):+.1f} … {max(F['resid_taskA']):+.1f} m. Least-squares X0 fitted to the surveyed positions "
          f"(our_x = X0 − x_local): **X0 = {F['X0_fit']:.1f} m** (RMS residual {F['rms']:.1f} m), {F['X0_fit'] - F['X0_taskA']:+.1f} m from "
          "X0*. The ERT-based placement method is confirmed within its stated uncertainty.", "",
          "Note on our x: positions are projected on the chord E1 → E102 (503.3 m in Lambert 72) and scaled to our profile x "
          "(elec.dat, 0–504.8 m along the surface), as in Task A. Distances measured along the chord alone are ≈ 0.3 % (up to "
          "1.4 m) smaller; the facts list of this round used those (e.g. S02 at 452.7 m)."]
    with open(os.path.join(res, "5_cpt_inventory.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    G = ["dp,cpt,lambert_x,lambert_y,our_x,offset_m,cpt_ground_mTAW,surface_2018_mTAW,ground_minus_2018_m,cpt_end_depth_m,"
         "dp_end_depth_m,x_local,x_taskA,surveyed_minus_taskA_m"]
    for s, r in zip(ids, F["resid_taskA"]):
        c = C[s]
        G.append(f"{c['dp']},{s},{c['lambert_x']:.2f},{c['lambert_y']:.2f},{c['our_x']:.2f},{c['offset']:.2f},{c['z']:.2f},"
                 f"{c['surf2018']:.2f},{c['z'] - c['surf2018']:.2f},{c['end_depth']:.2f},{c['dp_end_depth']:.2f},{c['dp_local_x']:g},"
                 f"{c['x_taskA']:.1f},{r:.2f}")
    with open(os.path.join(args.out, "data", "processed", "v1_4", "T_S6_geometry.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(G) + "\n")
    # 6_lithology.md + T_S9
    M = ["# 6 — Lithology from the CPTs (Task F)", "",
         f"Non-normalised soil-behaviour-type index (Robertson 2010): I_SBT = √[(3.47 − log10(qc/pa))² + (log10 Rf + 1.22)²], "
         f"pa = {PA:g} MPa; no unit weights or pore pressure needed (none were measured). Classes: clean sand I_SBT < {ISBT_SAND:g}; "
         f"sand mixtures {ISBT_SAND:g}–{ISBT_FINE:g}; fine-grained > {ISBT_FINE:g}. Elevation = Location.txt ground level − depth. "
         "Without pore pressure the classification is approximate, especially in fine-grained layers.", "",
         "| CPT (DP) | our x (m) | clean sand / mixtures / fine (%) | fine-grained layers ≥ 0.4 m (top → bottom, mTAW) | "
         f"first fine-grained unit > {DEEP_THICK:g} m reaching below {DEEP_Z:g} mTAW: top (thickness) | first fine-grained layer "
         f"(≥ 0.4 m) below {DEEP_Z:g} mTAW: top (descriptive) | Rf > 2 % over ≥ 1 m below {DEEP_Z:g} mTAW: top (descriptive) |",
         "|---|---|---|---|---|---|---|"]
    T = ["cpt,dp,our_x,layer_top_mTAW,layer_bottom_mTAW,thickness_m,deep_unit"]
    for s in ids:
        c = C[s]; fr = c["frac"]
        lay = "; ".join(f"{r['top']:.1f} → {r['bottom']:.1f}" for r in c["fine_layers"]) or "none"
        du = c["deep_unit"]
        M.append(f"| {s} ({c['dp']}) | {c['our_x']:.1f} | {100*fr['sand']:.0f} / {100*fr['mixture']:.0f} / {100*fr['fine']:.0f} | {lay} | "
                 + (f"{du['top']:.1f} ({du['thick']:.1f} m)" if du else f"none (log ends at {c['z'] - c['end_depth']:.1f})") + " | "
                 + next((f"{r['top']:.1f}" for r in c["fine_layers"] if r["top"] < DEEP_Z), "none") + " | "
                 + (f"{c['rf_step']:.1f}" if c["rf_step"] is not None else "none") + " |")
        for r in c["fine_layers"]:
            T.append(f"{s},{c['dp']},{c['our_x']:.1f},{r['top']:.1f},{r['bottom']:.1f},{r['thick']:.1f},{bool(du and r['top'] == du['top'])}")
    dtops = [C[s]["deep_unit"]["top"] for s in ids if C[s]["deep_unit"]]
    rft = [C[s]["rf_step"] for s in ids if C[s]["rf_step"] is not None]
    ftops = [next(r["top"] for r in C[s]["fine_layers"] if r["top"] < DEEP_Z) for s in ids
             if any(r["top"] < DEEP_Z for r in C[s]["fine_layers"])]
    M += ["", "Comparison with the deep unit of the synthetic truths and the reference model (top at −30 mTAW; descriptive): "
          + (f"the CPTs that reach it place the first thick fine-grained unit at {min(dtops):.1f} … {max(dtops):.1f} mTAW." if dtops
             else "no CPT shows a fine-grained unit thicker than 2 m below −20 mTAW.")
          + (f" Descriptively, {len(ftops)} CPTs ({', '.join(s for s in ids if any(r['top'] < DEEP_Z for r in C[s]['fine_layers']))}) "
             f"show thin fine-grained layers interbedded with sandier samples from {max(ftops):.1f} … {min(ftops):.1f} mTAW downwards; "
             "split by samples just below the 2.6 boundary, they never form a continuous unit > 2 m." if ftops else "")
          + (" A sharper marker (descriptive, not pre-registered): the friction ratio jumps to Rf > 2 % (qc drops to ≈ 4 MPa) "
             f"at {max(rft):.1f} … {min(rft):.1f} mTAW in {len(rft)} CPTs ({', '.join(s for s in ids if C[s]['rf_step'] is not None)}) "
             "— most likely the top of the Kortrijk clay. In this material I_SBT lies on the 2.6 class boundary, which is why "
             f"the class rule splits it into thin pieces. It lies {min(r + 30 for r in rft):.1f}–{max(r + 30 for r in rft):.1f} m "
             "above the −30 mTAW top of the deep unit of the synthetic truths and the reference model, and within or just below "
             "the 1980 clay top (−25.5 to −28 mTAW, Lebbe 1981)." if rft else "")
          + " CPTs that end above it (refusal in dense sand, qc up to ≈ 60 MPa): " + ", ".join(
              f"{s} at {C[s]['z'] - C[s]['end_depth']:.1f} mTAW" for s in ids if C[s]["rf_step"] is None) + "."]
    with open(os.path.join(res, "6_lithology.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(M) + "\n")
    with open(os.path.join(res, "T_S9_cpt_lithology.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(T) + "\n")
    with open(os.path.join(res, "T_S9_cpt_lithology.md"), "w", encoding="utf-8") as f:
        f.write("# T_S9 — fine-grained layers (I_SBT > 2.6, ≥ 0.4 m) per CPT (Markdown copy of `T_S9_cpt_lithology.csv`)\n\n"
                "| CPT | DP | our x (m) | top (mTAW) | bottom (mTAW) | thickness (m) | first thick unit below −20 |\n|---|---|---|---|---|---|---|\n"
                + "\n".join("| " + " | ".join(l.split(",")) + " |" for l in T[1:]) + "\n")
    # F_S11
    fig, axes = plt.subplots(3, len(ids), figsize=(1.6 * len(ids), 8.6), sharey=True)
    for j, s in enumerate(ids):
        c = C[s]; z = np.array(c["elev"]); I = np.array([np.nan if v is None else v for v in c["isbt"]], float)
        axes[0, j].semilogx(c["qc"], z, color="0.2", lw=0.8); axes[0, j].set_title(f"{s} ({c['dp']})\nx = {c['our_x']:.0f} m", fontsize=7)
        axes[1, j].plot(c["Rf"], z, color="#8c6d31", lw=0.8); axes[1, j].set_xlim(0, 6)
        ax = axes[2, j]
        ax.axvspan(0, ISBT_SAND, color="#f6e7b4", lw=0); ax.axvspan(ISBT_SAND, ISBT_FINE, color="#e0cfa0", lw=0)
        ax.axvspan(ISBT_FINE, 4.5, color="#b7a58a", lw=0)
        ax.plot(I, z, color="k", lw=0.8); ax.set_xlim(1, 4.0)
        for r in c["fine_layers"]:
            ax.axhspan(r["bottom"], r["top"], xmin=0.92, xmax=1.0, color="#6a3d9a", lw=0)
        for a in axes[:, j]:
            a.axhline(-30, color="#c0392b", lw=0.6, ls=":"); a.tick_params(labelsize=6)
        axes[0, j].set_xlabel("qc (MPa)", fontsize=6.5); axes[1, j].set_xlabel("Rf (%)", fontsize=6.5)
        ax.set_xlabel("I_SBT", fontsize=6.5)
    for i in range(3):
        axes[i, 0].set_ylabel("Elevation (mTAW)", fontsize=7)
    axes[0, 0].set_ylim(-36, 6)
    fig.suptitle("CPT logs of 23 September 2024 (E. Vrancken). I_SBT shading: clean sand < 2.05, mixtures 2.05–2.6, fine-grained > 2.6; "
                 "purple bars: fine-grained layers ≥ 0.4 m; red dotted: −30 mTAW (deep unit of the synthetic truths)", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(figs, "F_S11_cpt_logs.png"), dpi=300); plt.close(fig)
    return L, M


# ─────────────────────────────────────────────────────────────────────────────
# Task D — same-epoch inversion of the Zenodo ERT.dat (decision log v1.4-6, v1.4-8). No DP resistivity value.
# ─────────────────────────────────────────────────────────────────────────────
X0_CPT = 503.75                                            # least-squares fit to the surveyed CPT positions (v1.4-8)
SD_LAMS = [50.0, 5.0, 10.0, 20.0, 100.0]
SD_DIR_NAME = "samedate"


def cpt_topography():
    """our_x → ground elevation: CPT ground levels linear in x; beyond the outermost CPTs the 2018 surface shifted by
    the outermost CPT's difference (v1.4-8)."""
    J = json.load(open(os.path.join(wc.ROOT, "data", "processed", "v1_4", "cpt.json"), encoding="utf-8"))
    C = sorted(J["cpts"].values(), key=lambda c: c["our_x"])
    xc = np.array([c["our_x"] for c in C]); zc = np.array([c["z"] for c in C])
    ex, ez = wc.land_surface()
    def f(x):
        x = np.asarray(x, float); z = np.interp(x, xc, zc)
        lo, hi = x < xc[0], x > xc[-1]
        z = np.where(lo, np.interp(x, ex, ez) + (zc[0] - np.interp(xc[0], ex, ez)), z)
        z = np.where(hi, np.interp(x, ex, ez) + (zc[-1] - np.interp(xc[-1], ex, ez)), z)
        return z
    return f


@contextlib.contextmanager
def surface_override(xs, zs):
    """ColumnSampler and others read wc.land_surface(); point it at the topography of the recent line."""
    orig = wc.land_surface
    wc.land_surface = lambda: (np.asarray(xs, float), np.asarray(zs, float))
    try:
        yield
    finally:
        wc.land_surface = orig


def sd_columns():
    """Columns: each DP at its primary (surveyed) x and at the Task-A x for X0* − σ, X0*, X0* + σ."""
    J = json.load(open(os.path.join(wc.ROOT, "data", "processed", "v1_4", "cpt.json"), encoding="utf-8"))
    pl = json.load(open(os.path.join(wc.ROOT, "data", "processed", "v1_4", "placement.json"), encoding="utf-8"))
    X0s, sg = pl["primary"]["X0"], pl["primary"]["sigma"]
    xs = set()
    for c in J["cpts"].values():
        xs.add(round(c["our_x"], 3))
        for X0 in (X0s - sg, X0s, X0s + sg):
            xs.add(round(X0 - c["dp_local_x"], 3))
    return np.array(sorted(xs))


def steep_logmid(v, z, zstop=-25.0):
    """Log-midpoint anchored on the steepest increase above zstop (secondary definition; as for the DP logs)."""
    D = (v[1:] - v[:-1]) / (z[:-1] - z[1:]); zm = 0.5 * (z[:-1] + z[1:])
    W = np.isfinite(D) & (zm >= zstop)
    if not W.any() or not np.max(D[W]) > 0:
        return np.nan
    g = np.max(D[W]); sel = W & (D >= 0.9 * g); zg = float(np.sum(D[sel] * zm[sel]) / np.sum(D[sel]))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        up = np.nanmedian(np.where((z > zg) & (z <= zg + 3.0), v, np.nan)); lo = np.nanmedian(np.where((z < zg) & (z >= zg - 3.0), v, np.nan))
    thr = 0.5 * (up + lo); a, b = v[:-1] - thr, v[1:] - thr
    cr = np.isfinite(a) & np.isfinite(b) & ((a <= 0) != (b <= 0)) & (np.abs(zm - zg) <= 10.0)
    if not cr.any():
        return np.nan
    t = a[cr] / (a[cr] - b[cr]); zc = z[:-1][cr] + t * (z[1:] - z[:-1])[cr]
    return float(zc[np.argmin(np.abs(zc - zg))])


def cmd_samedate(args):
    check_raw()
    c11 = _load("c11", "11_stage4_closeout.py"); sb = c11.sb
    m02 = c11._load("inv02", "02_land_inversion.py"); m02.VERBOSE = False
    od = os.path.join(wc.ROOT, "data", "processed", "v1_4", SD_DIR_NAME); os.makedirs(od, exist_ok=True)
    E = read_ert(); topo = cpt_topography(); xcol = sd_columns()
    for case in ("cpt", "flat"):
        out = os.path.join(od, f"{case}.npz")
        if os.path.exists(out):
            print(f"[samedate] {case}: exists"); continue
        n_el = len(E["xloc"]); num = n_el - np.arange(n_el)              # local index i0 → number increasing with our_x
        xs = X0_CPT - E["xloc"]; zs = topo(xs) if case == "cpt" else np.zeros_like(xs)
        elec = {int(num[i]): (float(xs[i]), float(zs[i])) for i in range(n_el)}
        a = E["abmn"]
        base = {k: np.array([int(num[v]) for v in a[:, c]]) for c, k in enumerate(("C1", "C2", "P1", "P2"))}
        order = np.argsort(xs)
        with surface_override(xs[order], zs[order]), contextlib.redirect_stdout(io.StringIO()):
            land = dict(base, R=E["R"], n=len(E["R"]), err=np.zeros(len(E["R"])))
            data = m02.build_datacont(land, elec, err_rel=sb.ERR_REL)[0]
            mgr = m02.run_inversion(data, lam=20.0, para_depth=sb.PARA_DEPTH, para_dx=sb.PARA_DX, max_iter=sb.MAX_ITER)
            ncell = mgr.paraDomain.cellCount(); pd_mesh = pg_mesh_copy(mgr.paraDomain)
            cs = wc.ColumnSampler(pd_mesh, xcol)
            pre, _ = sb.one_inversion(m02, land, elec, 20.0, np.zeros(ncell), cs, [139000, 0])
            r = pre["resid"]; med = np.median(r); mad = np.median(np.abs(r - med))
            keep = np.abs(r) <= 3 * 1.4826 * mad
            land2 = dict({k: v[keep] for k, v in base.items()}, R=E["R"][keep], n=int(keep.sum()), err=np.zeros(int(keep.sum())))
            save = dict(case=case, X0=X0_CPT, xcol=xcol, z=cs.z, surf=cs.surf, n_data=len(E["R"]), n_kept=int(keep.sum()),
                        stage1_chi2=pre["chi2"], keep=keep, elec_x=xs, elec_z=zs)
            pooled = []
            for lam in SD_LAMS:
                info, S = sb.one_inversion(m02, land2, elec, lam, np.zeros(ncell), cs, [139000, int(lam)])
                V = cs(np.log10(info["model"].astype(float)))
                hat = c11.interfaces_all(V[:, :, None], cs.z)
                pre_ = f"lam{int(lam)}_"
                save.update({pre_ + "chi2": info["chi2"], pre_ + "stop": info["stop"], pre_ + "model": info["model"],
                             pre_ + "col": V.astype(np.float32), pre_ + "radius": info["radius"]})
                for d in ("thr3", "maxgrad", "logmid"):
                    save[pre_ + f"mhat_{d}"] = hat[d][:, 0]
                    save[pre_ + f"r_{d}"] = c11.r_at(cs, info["radius"], hat[d][:, 0])
                save[pre_ + "mhat_steep"] = np.array([steep_logmid(V[:, j], cs.z) for j in range(len(xcol))])
                bands = c11.sample_defs(cs, S)
                for d in ("thr3", "maxgrad", "logmid"):
                    save[pre_ + f"band_{d}"] = bands[d]
                pooled.append(S.astype(np.float32))
            P = np.hstack(pooled).astype(np.float64)
            pb = c11.sample_defs(cs, P)
            for d in ("thr3", "maxgrad", "logmid"):
                save[f"pooled_band_{d}"] = pb[d]
            Vs = cs(P / np.log(10))
            st = np.array([[steep_logmid(Vs[:, j, k], cs.z) for k in range(Vs.shape[2])] for j in range(len(xcol))])
            save["pooled_band_steep"] = c11.band_stats(st)
        np.savez(out + ".tmp.npz", **save); os.replace(out + ".tmp.npz", out)
        print(f"[samedate] {case}: kept {int(keep.sum())}/{len(E['R'])}; χ² λ=50 {save['lam50_chi2']:.2f}", flush=True)


def pg_mesh_copy(m):
    import pygimli as pg
    return pg.Mesh(m)


# ─────────────────────────────────────────────────────────────────────────────
# Task C — DP logs vs the 2018 results and vs the same-epoch image (E15–E21). Pre-registration: v1.4-0, v1.4-6, v1.4-8.
# ─────────────────────────────────────────────────────────────────────────────
DP_SURVEY = "2024-09-23"
GAP_DAYS = 2392
GAP_TXT = "time gap 7 March 2018 → 23 September 2024: 2392 days (≈ 6.5 years)"
SD_DATE_TXT = "ERT.dat date: probably the same campaign, unconfirmed (asked T. Hermans)"
TEMP_B = 1.26
SMOOTH_N, DZG, Z_STOP, RB_OFF, FRESH_R = 51, 0.2, -25.0, 2.0, 20.0
N_ARCH = 20000
XR15, XR16, RMAX15, RMAX_SD = (100.0, 450.0), (150.0, 350.0), 15.0, 10.0
CLEAN_MIN = 1.0


def read_dp_values():
    A = np.genfromtxt(os.path.join(RAW, "DP_R.xzr"))
    logs = {}
    for lx in np.unique(np.round(A[:, 0], 3)):
        s_ = A[np.isclose(A[:, 0], lx, atol=1e-3)]
        s_ = s_[np.argsort(-s_[:, 1])]
        logs[float(lx)] = dict(depth=-s_[:, 1], r=s_[:, 2])          # depth positive down
    return logs


def dp_itf(zg, v):
    from scipy.ndimage import median_filter  # noqa: F401  (smoothing done by the caller)
    V = v[:, None, None]
    out = dict(thr3=float(wc.main_lens(V, zg, wc.LOG3)[1][0, 0]), thr2p5=float(wc.main_lens(V, zg, np.log10(2.5))[1][0, 0]))
    out["logmid"] = steep_logmid(v, zg, Z_STOP)
    return out


def dp_process(lg, surface, factor, cls_at_depth):
    from scipy.ndimage import median_filter
    d = lg["depth"]; r = lg["r"] * factor
    z1 = surface - d
    sm = median_filter(np.log10(r), size=SMOOTH_N, mode="nearest")
    zg = np.arange(np.floor(z1[0] / DZG) * DZG, z1[-1] - 1e-9, -DZG); zg = zg[(zg <= z1[0]) & (zg >= z1[-1])]
    v = np.interp(-zg, -z1, sm)
    o = dp_itf(zg, v)
    base = o["logmid"]
    fr = []
    m = sm >= np.log10(FRESH_R); i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            fr.append(dict(top=float(z1[i]), bottom=float(z1[j]), thick=float(z1[i] - z1[j]))); i = j + 1
        else:
            i += 1
    o.update(top=float(z1[0]), bottom=float(z1[-1]), fresh_runs=fr,
             fresh_below_max=max([f["thick"] for f in fr if np.isfinite(base) and f["top"] < base], default=0.0),
             base_depth=float(surface - base) if np.isfinite(base) else np.nan, rho_below=np.nan, rho_below_unscreened=np.nan,
             n_window=0, n_clean=0, clean_m=0.0, rho_lens=np.nan, q=np.nan, z_grid=zg.tolist(), v_grid=v.tolist())
    if np.isfinite(base):
        lim = max(z1[-1], Z_STOP); w = (z1 <= base - RB_OFF) & (z1 >= lim)
        clean = np.array([cls_at_depth(x) == "sand" for x in d])
        o["n_window"] = int(w.sum()); o["n_clean"] = int((w & clean).sum()); o["clean_m"] = float((w & clean).sum() * 0.01)
        if w.any():
            o["rho_below_unscreened"] = float(np.median(r[w]))
        if o["clean_m"] >= CLEAN_MIN:
            o["rho_below"] = float(np.median(r[w & clean]))
        lw = z1 >= base + RB_OFF
        if lw.any():
            o["rho_lens"] = float(np.median(r[lw]))
        if np.isfinite(o["rho_lens"]) and np.isfinite(o["rho_below"]) and o["rho_below"] != o["rho_lens"]:
            o["q"] = float((wc.LOG3 - np.log10(o["rho_lens"])) / (np.log10(o["rho_below"]) - np.log10(o["rho_lens"])))
    return o


def cmd_dp(args):
    check_raw()
    dpp = os.path.join(args.out, "data", "processed", "v1_4"); os.makedirs(dpp, exist_ok=True)
    s18 = _load("p18", "18_paper.py"); k1 = _load("k1c", "21_k1_comparison.py"); s17 = _load("s17", "17_salinity_v2.py")
    J = json.load(open(os.path.join(dpp, "cpt.json"), encoding="utf-8"))
    pl = json.load(open(os.path.join(dpp, "placement.json"), encoding="utf-8"))
    X0s, sg = pl["primary"]["X0"], pl["primary"]["sigma"]
    ex, ez = wc.land_surface()
    v4 = np.load(s18.LIN_V4)
    pb = os.path.join(args.out, "data", "processed", "v1_1", "k1_field_bel.npz")
    pb = pb if os.path.exists(pb) else os.path.join(wc.ROOT, "data", "processed", "v1_1", "k1_field_bel.npz")
    zb = np.load(pb); TX, Pf = zb["TX"], zb["Pf"]
    lb = k1.read_lebbe()
    num0 = {r_["key"]: r_["value"] for r_ in json.load(open(os.path.join(wc.ROOT, "results", "paper", "numbers.json"), encoding="utf-8"))}
    SD = np.load(os.path.join(wc.ROOT, "data", "processed", "v1_4", SD_DIR_NAME, "cpt.npz"), allow_pickle=True)
    logs = read_dp_values()
    ref = wc.load_run("g1_lam50"); mesh, _ = wc.rebuild_paradomain(ref)
    geoms = [("primary", None, None)] + [(f"X0{dX:+g}_s{ds:+g}", dX, ds) for dX in (-sg, 0.0, sg) for ds in (-0.5, 0.0, 0.5)]
    xs_all = set()
    for c in J["cpts"].values():
        xs_all.add(round(c["our_x"], 3))
        for dX in (-sg, 0.0, sg):
            xs_all.add(round(X0s + dX - c["dp_local_x"], 3))
    xs_all = np.array(sorted(xs_all)); cs = wc.ColumnSampler(mesh, xs_all); col = cs(np.log10(ref["model"]))
    res = dict(gap=GAP_TXT, date=DP_SURVEY, sd_date=SD_DATE_TXT, rho_below_field=num0["second_bel_rho_below"], dps={})
    for sid in sorted(J["cpts"]):
        c = J["cpts"][sid]; nm = c["dp"]; i = int(nm[2:])
        lg = logs[min(logs, key=lambda k: abs(k - c["dp_local_x"]))]
        cd = c["z"] - np.array(c["elev"]); cI = np.array([np.nan if v is None else v for v in c["isbt"]], float)
        ccls = np.where(cI < ISBT_SAND, "sand", np.where(cI <= ISBT_FINE, "mixture", "fine"))
        cls_at = lambda dd, cd=cd, ccls=ccls: ccls[int(np.argmin(np.abs(cd - dd)))]
        fine_depth = [(c["z"] - f["top"], c["z"] - f["bottom"]) for f in c["fine_layers"]]
        per = {}
        for gname, dX, ds in geoms:
            x = c["our_x"] if dX is None else X0s + dX - c["dp_local_x"]
            surf = c["z"] if dX is None else float(np.interp(x, ex, ez)) + ds
            for scen, fac in (("A", 1.0), ("B", TEMP_B)):
                o = dp_process(lg, surf, fac, cls_at)
                j = int(np.argmin(np.abs(TX - x))); k = int(np.argmin(np.abs(v4["x"] - x)))
                q = np.percentile(Pf[:, j], [50, 5, 95]); kc = int(np.argmin(np.abs(xs_all - x)))
                ks = int(np.argmin(np.abs(SD["xcol"] - x)))
                o.update(x=float(x), surface=float(surf), r_ref=float(v4["r_logmid"][k]),
                         included=bool(np.isfinite(v4["r_logmid"][k]) and v4["r_logmid"][k] <= RMAX15),
                         bel=[float(v) for v in q], logmid_pooled=[float(v) for v in v4["pooled_logmid"][:, k]],
                         thr3_lam50=[float(v) for v in v4["single_thr3"][:, k]],
                         sal1980=float(k1.interp_boundary(x, lb["x_ours_m"], lb["saline_base_mTAW"])))
                # same-epoch image at this x
                bimg = float(SD["lam50_mhat_logmid"][ks]); colimg = SD["lam50_col"][:, ks].astype(float); zimg = SD["z"]
                wimg = (zimg <= bimg - 4.0) & (zimg >= bimg - 10.0) & np.isfinite(colimg)
                o["sd"] = dict(x=float(SD["xcol"][ks]), r=float(SD["lam50_r_logmid"][ks]), thr3=float(SD["lam50_mhat_thr3"][ks]),
                               logmid=bimg, steep=float(SD["lam50_mhat_steep"][ks]),
                               band_thr3=[float(v) for v in SD["lam50_band_thr3"][:, ks]],
                               band_logmid_pooled=[float(v) for v in SD["pooled_band_logmid"][:, ks]],
                               band_steep_pooled=[float(v) for v in SD["pooled_band_steep"][:, ks]],
                               rho_below_img=float(10 ** np.median(colimg[wimg])) if wimg.any() else np.nan)
                o["lith_near_base"] = bool(np.isfinite(o["base_depth"]) and any(t <= o["base_depth"] + 1.0 and b_ >= o["base_depth"] - 1.0
                                                                                   for t, b_ in [(min(a, b), max(a, b)) for a, b in fine_depth]))
                if gname == "primary":
                    o["ref_col"] = dict(z=cs.z.tolist(), v=[None if not np.isfinite(t) else float(t) for t in col[:, kc]])
                    o["img_col"] = dict(z=zimg.tolist(), v=[None if not np.isfinite(t) else float(t) for t in colimg])
                    if np.isfinite(o["rho_below"]):
                        a = s17.archie_v2(N_ARCH, np.random.default_rng([137200, i]))
                        sw = 100 * a["F"] * a["rsw"] / o["rho_below"]
                        o["sw"] = [float(np.median(sw)), float(np.percentile(sw, 5)), float(np.percentile(sw, 95)), float(np.mean(sw > 100))]
                    else:
                        o["sw"] = None
                per[f"{gname}|{scen}"] = o
        res["dps"][nm] = dict(index=i, cpt=sid, local_x=c["dp_local_x"], our_x=c["our_x"], ground=c["z"], per=per,
                              fine_layers=c["fine_layers"], isbt=c["isbt"], cpt_elev=c["elev"])
    res["verdicts"] = {k: dp_verdicts(res["dps"], k) for k in [f"{g}|{s}" for g, _, _ in geoms for s in ("A", "B")]}
    conf = [n for n, d in res["dps"].items() if d["per"]["primary|A"]["lith_near_base"]]
    res["confounded"] = conf
    res["verdicts_noconf"] = dp_verdicts({n: d for n, d in res["dps"].items() if n not in conf}, "primary|A") if conf else None
    res["final"] = final_verdicts(res)
    with open(os.path.join(dpp, "dp_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res["final"], indent=1))


def _h(b):
    return None if b is None else bool(b)


def dp_verdicts(dps, key):
    V = {}
    D = {n: d["per"][key] for n, d in dps.items()}
    e15 = [n for n, o in D.items() if o["included"] and XR15[0] <= o["x"] <= XR15[1]]
    ins = [n for n in e15 if np.isfinite(D[n]["logmid"]) and D[n]["bel"][1] <= D[n]["logmid"] <= D[n]["bel"][2]]
    V["E15"] = dict(n=len(e15), n_ok=len(ins), dps=e15, ok=ins, holds=_h(len(ins) / len(e15) >= 2 / 3) if e15 else None)
    e16 = [n for n, o in D.items() if o["included"] and XR16[0] <= o["x"] <= XR16[1]]
    low = [n for n in e16 if np.isfinite(D[n]["rho_below"]) and D[n]["rho_below"] < 12.45]
    thick = [n for n in e16 if D[n]["fresh_below_max"] > 3.0]
    V["E16"] = dict(n=len(e16), n_rho=sum(np.isfinite(D[n]["rho_below"]) for n in e16), n_ok=len(low), dps=e16, ok=low, thick=thick,
                    holds=_h(len(low) > len(e16) / 2 and not thick) if e16 else None)
    e17 = [n for n, o in D.items() if o["included"] and np.isfinite(o["logmid"]) and abs(o["logmid_pooled"][1] - o["thr3_lam50"][1]) > 2.0]
    cl = [n for n in e17 if abs(D[n]["logmid"] - D[n]["logmid_pooled"][1]) < abs(D[n]["logmid"] - D[n]["thr3_lam50"][1])]
    V["E17"] = dict(n=len(e17), n_ok=len(cl), dps=e17, ok=cl, holds=_h(len(cl) > len(e17) / 2) if e17 else None)
    r10 = [n for n, o in D.items() if np.isfinite(o["sd"]["r"]) and o["sd"]["r"] <= RMAX_SD]
    testable = len(r10) >= 3
    e18 = [n for n in r10 if np.isfinite(D[n]["q"]) and 0 < D[n]["q"] < 1 and np.isfinite(D[n]["thr3"]) and np.isfinite(D[n]["sd"]["thr3"])]
    ok18 = [n for n in e18 if (D[n]["sd"]["thr3"] < D[n]["thr3"]) == (D[n]["q"] > 0.5)]
    V["E18"] = dict(n=len(e18), n_ok=len(ok18), dps=e18, ok=ok18, r10=r10,
                    holds=(_h(len(ok18) / len(e18) >= 2 / 3) if e18 else None) if testable else "not testable")
    inb = lambda b, v: np.isfinite(v) and b[0] >= 0.5 and b[2] <= v <= b[3]
    c_lm = [n for n in r10 if inb(D[n]["sd"]["band_logmid_pooled"], D[n]["logmid"])]
    c_t3 = [n for n in r10 if inb(D[n]["sd"]["band_thr3"], D[n]["thr3"])]
    V["E19"] = dict(n=len(r10), n_logmid=len(c_lm), n_thr3=len(c_t3), ok_logmid=c_lm, ok_thr3=c_t3,
                    holds=_h(len(c_lm) > len(c_t3)) if testable else "not testable")
    e20 = [n for n in r10 if np.isfinite(D[n]["rho_below"]) and np.isfinite(D[n]["sd"]["rho_below_img"])]
    ok20 = [n for n in e20 if D[n]["sd"]["rho_below_img"] < D[n]["rho_below"]]
    V["E20"] = dict(n=len(e20), n_ok=len(ok20), dps=e20, ok=ok20,
                    holds=(_h(len(ok20) / len(e20) >= 2 / 3) if e20 else None) if testable else "not testable")
    no_l = [n for n, o in D.items() if np.isfinite(o["logmid"]) and not o["lith_near_base"]]
    V["E21"] = dict(n=len(D), n_ok=len(no_l), ok=no_l, holds=_h(len(no_l) >= 5))
    return V


def final_verdicts(res):
    V = res["verdicts"]; prim = V["primary|A"]; out = {}
    lab = lambda h: "not testable" if h == "not testable" else ("untestable (no DP in range)" if h is None else ("held" if h else "not held"))
    for e in ("E15", "E16", "E17", "E18", "E19", "E20", "E21"):
        h = prim[e]["holds"]; flags = []
        sec = [k for k in V if k.endswith("|A") and k != "primary|A" and V[k][e]["holds"] != h]
        if sec:
            flags.append("position-sensitive")
        if V["primary|B"][e]["holds"] != h:
            flags.append("temperature-sensitive")
        if res["verdicts_noconf"] is not None and e != "E21" and res["verdicts_noconf"][e]["holds"] != h:
            flags.append("lithology-confounded")
        out[e] = dict(verdict=lab(h), flags=flags, differs_in=sec)
    return out


def report_dp(args, plt, s18):
    """2_dp_comparison.md, 3_same_epoch.md, T_S6, T_S7, F_12, F_12b, F_13, F_1b."""
    dpp = os.path.join(args.out, "data", "processed", "v1_4"); res = os.path.join(args.out, "results", "v1_4"); figs = os.path.join(res, "figs")
    R = json.load(open(os.path.join(dpp, "dp_results.json"), encoding="utf-8"))
    names = sorted(R["dps"], key=lambda n: R["dps"][n]["index"])
    f_ = lambda v, fmt="{:.2f}": "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)
    P = lambda n, k="primary|A": R["dps"][n]["per"][k]
    G = R["gap"]
    # ── 2_dp_comparison.md ──
    L = ["# 2 — DP logs against the 2018 results (Task C)", "",
         f"**{G}.** DP logs = CPT-EC of the CPT pushes (E. Vrancken, 23 September 2024); positions and ground levels surveyed "
         "(`5_cpt_inventory.md`); elevation = CPT ground − depth. Scenario A (primary): ρ as measured, in situ; scenario B: × 1.26 "
         "(values referenced to 25 °C). ρ_below,DP from clean-sand samples only (CPT I_SBT < 2.05). Pre-registration: "
         "`docs/decision_log.md` v1.4-0, v1.4-6, v1.4-8.", "",
         "## Per DP (primary geometry, scenario A)", "",
         "| DP | CPT | our x (m) | ground (mTAW) | DP log top / bottom (mTAW) | **DP log-midpoint base** | DP 3 / 2.5 Ω·m | ≥ 20 Ω·m runs (mTAW) | "
         "ρ_lens,DP | ρ_below,DP (clean sand; unscreened) | clean sand in window | q_DP | fine layer within ±1 m of base |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        o = P(n); fr = "; ".join(f"{x['top']:.1f}–{x['bottom']:.1f}" for x in o["fresh_runs"]) or "none"
        L.append(f"| {n} | {R['dps'][n]['cpt']} | {o['x']:.1f} | {o['surface']:.2f} | {o['top']:.2f} / {o['bottom']:.2f} | **{f_(o['logmid'])}** | "
                 f"{f_(o['thr3'])} / {f_(o['thr2p5'])} | {fr} | {f_(o['rho_lens'])} | {f_(o['rho_below'])} ({f_(o['rho_below_unscreened'])}) | "
                 f"{o['clean_m']:.2f} m of {o['n_window'] * 0.01:.2f} m | {f_(o['q'])} | {'yes' if o['lith_near_base'] else 'no'} |")
    L += ["", "## Against 2018 (primary, A; difference = DP − 2018, positive = DP shallower)", "",
          "| DP | x | r (2018, m) | incl. | BEL median [P5, P95] | DP − BEL | in BEL 90 % | log-mid λ-pooled | DP − log-mid | 3 Ω·m λ = 50 | DP 3 Ω·m − 2018 3 Ω·m | 1980 saline base | DP − 1980 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        o = P(n); lm = o["logmid"]
        L.append(f"| {n} | {o['x']:.1f} | {f_(o['r_ref'], '{:.1f}')} | {'yes' if o['included'] else 'no'} | {o['bel'][0]:.2f} [{o['bel'][1]:.2f}, {o['bel'][2]:.2f}] | "
                 f"{f_(lm - o['bel'][0], '{:+.2f}')} | {'yes' if np.isfinite(lm) and o['bel'][1] <= lm <= o['bel'][2] else 'no'} | "
                 f"{o['logmid_pooled'][1]:.2f} [{o['logmid_pooled'][2]:.2f}, {o['logmid_pooled'][3]:.2f}] | {f_(lm - o['logmid_pooled'][1], '{:+.2f}')} | "
                 f"{o['thr3_lam50'][1]:.2f} | {f_(o['thr3'] - o['thr3_lam50'][1], '{:+.2f}')} | {o['sal1980']:.2f} | {f_(lm - o['sal1980'], '{:+.2f}')} |")
    L += ["", "## Salinity of the water below the lens (clean-sand ρ_below,DP; v1.0 Archie draws, T 8–14 °C)", "",
          "| DP | ρ_below,DP A (Ω·m) | % seawater A, median [P5, P95] | ρ_below,DP B = 1.26 × A | % seawater B |", "|---|---|---|---|---|"]
    for n in names:
        o = P(n); sw = o["sw"]; swb = (None if sw is None else [v / TEMP_B for v in sw[:3]])
        L.append(f"| {n} | {f_(o['rho_below'])} | " + (f"{sw[0]:.1f} [{sw[1]:.1f}, {sw[2]:.1f}]" if sw else "not estimable") +
                 f" | {f_(o['rho_below'] * TEMP_B if np.isfinite(o['rho_below']) else np.nan)} | " +
                 (f"{swb[0]:.1f} [{swb[1]:.1f}, {swb[2]:.1f}]" if swb else "not estimable") + " |")
    L += ["", f"Field BEL ρ_below posterior (profile-wide, 2018): {R['rho_below_field'].split(';')[0]} Ω·m.", ""]
    L += verdict_lines(R, ("E15", "E16", "E17", "E21"))
    with open(os.path.join(res, "2_dp_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    # ── 3_same_epoch.md ──
    SDn = np.load(os.path.join(wc.ROOT, "data", "processed", "v1_4", SD_DIR_NAME, "cpt.npz"), allow_pickle=True)
    SDf = np.load(os.path.join(wc.ROOT, "data", "processed", "v1_4", SD_DIR_NAME, "flat.npz"), allow_pickle=True)
    M = ["# 3 — Same-epoch test: ERT.dat inverted with our pipeline vs the DP logs (Task D)", "",
         f"**{G}; {R['sd_date']}.** ERT.dat electrodes at our_x = {X0_CPT:g} − x_local (X0 fitted to the surveyed CPT positions), "
         "on the CPT ground levels (flat topography as a check); Stage-1 G1 outlier rule, v1.0 error model, λ = 5–100 with "
         "pooling, linearised posterior (seeds `default_rng([139000, λ])`). Pre-registration: v1.4-6, v1.4-8.", "",
         f"- Data: {int(SDn['n_data'])} positive data; Stage-1 rule kept {int(SDn['n_kept'])} ({int(SDn['n_data']) - int(SDn['n_kept'])} removed); "
         f"stage-1 χ² {float(SDn['stage1_chi2']):.2f}. χ² per λ (topography): " +
         ", ".join(f"λ = {int(l)}: {float(SDn[f'lam{int(l)}_chi2']):.2f}" for l in SD_LAMS) + "; flat: " +
         ", ".join(f"{int(l)}: {float(SDf[f'lam{int(l)}_chi2']):.2f}" for l in SD_LAMS) + ".", "",
         "| DP | x (m) | image r (m) | DP 3 Ω·m | image 3 Ω·m (λ = 50) [P5, P95] | q_DP | side as predicted | DP log-mid | image log-mid λ-pooled [P5, P95] | "
         "image log-mid, steepest-increase anchor (pooled) | DP ρ_below (clean) | image ρ 4–10 m below its base | flat: image 3 Ω·m / log-mid |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    T7 = ["dp,x,image_r,dp_thr3,image_thr3,image_thr3_p5,image_thr3_p95,q_dp,dp_logmid,image_logmid_pooled,image_logmid_p5,image_logmid_p95,"
          "image_steep_pooled,dp_rho_below,image_rho_below,flat_thr3,flat_logmid"]
    for n in names:
        o = P(n); s_ = o["sd"]; ks = int(np.argmin(np.abs(SDf["xcol"] - o["x"])))
        side = "—" if not (np.isfinite(o["q"]) and 0 < o["q"] < 1 and np.isfinite(o["thr3"]) and np.isfinite(s_["thr3"])) else \
            ("yes" if (s_["thr3"] < o["thr3"]) == (o["q"] > 0.5) else "no")
        bt, bl, bs = s_["band_thr3"], s_["band_logmid_pooled"], s_["band_steep_pooled"]
        M.append(f"| {n} | {o['x']:.1f} | {f_(s_['r'], '{:.1f}')} | {f_(o['thr3'])} | {f_(s_['thr3'])} [{f_(bt[2])}, {f_(bt[3])}] | {f_(o['q'])} | {side} | "
                 f"{f_(o['logmid'])} | {f_(bl[1])} [{f_(bl[2])}, {f_(bl[3])}] | {f_(bs[1])} [{f_(bs[2])}, {f_(bs[3])}] | {f_(o['rho_below'])} | "
                 f"{f_(s_['rho_below_img'])} | {f_(float(SDf['lam50_mhat_thr3'][ks]))} / {f_(float(SDf['lam50_mhat_logmid'][ks]))} |")
        T7.append(",".join(f_(v, "{:.3f}") if not isinstance(v, str) else v for v in
                           [n, o["x"], s_["r"], o["thr3"], s_["thr3"], bt[2], bt[3], o["q"], o["logmid"], bl[1], bl[2], bl[3], bs[1],
                            o["rho_below"], s_["rho_below_img"], float(SDf["lam50_mhat_thr3"][ks]), float(SDf["lam50_mhat_logmid"][ks])]))
    M += [""] + verdict_lines(R, ("E18", "E19", "E20"))
    with open(os.path.join(res, "3_same_epoch.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(M) + "\n")
    with open(os.path.join(res, "T_S7_same_epoch.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(T7).replace("—", "") + "\n")
    with open(os.path.join(res, "T_S7_same_epoch.md"), "w", encoding="utf-8") as f:
        f.write("# T_S7 — same-epoch image vs DP (Markdown copy of `T_S7_same_epoch.csv`)\n\n" + G + "; " + R["sd_date"] + ".\n\n"
                + "| " + " | ".join(T7[0].split(",")) + " |\n|" + "---|" * len(T7[0].split(",")) + "\n"
                + "\n".join("| " + " | ".join(l.split(",")) + " |" for l in T7[1:]) + "\n")
    # ── T_S6 (geometry + resistivity) ──
    geo = open(os.path.join(dpp, "T_S6_geometry.csv"), encoding="utf-8").read().splitlines()
    gh = geo[0].split(","); gm = {l.split(",")[0]: l for l in geo[1:]}
    cols = ["included", "dp_logmid", "dp_thr3", "dp_thr2p5", "rho_lens", "rho_below_clean", "rho_below_unscreened", "clean_m", "q_dp",
            "fresh_below_max_m", "bel_med", "bel_p5", "bel_p95", "logmid_pooled_med", "thr3_lam50_med", "sal1980", "dp_minus_bel",
            "dp_minus_logmid", "dp_minus_thr3", "inside_bel90", "lith_near_base", "sw_A_med", "sw_A_p5", "sw_A_p95"]
    rows = [",".join(gh + cols)]
    for n in names:
        o = P(n); sw = o["sw"] or [np.nan] * 3; lm = o["logmid"]
        vals = [o["included"], lm, o["thr3"], o["thr2p5"], o["rho_lens"], o["rho_below"], o["rho_below_unscreened"], o["clean_m"], o["q"],
                o["fresh_below_max"], o["bel"][0], o["bel"][1], o["bel"][2], o["logmid_pooled"][1], o["thr3_lam50"][1], o["sal1980"],
                lm - o["bel"][0], lm - o["logmid_pooled"][1], lm - o["thr3_lam50"][1],
                bool(np.isfinite(lm) and o["bel"][1] <= lm <= o["bel"][2]), o["lith_near_base"], sw[0], sw[1], sw[2]]
        rows.append(gm[n] + "," + ",".join(str(v) if isinstance(v, bool) else ("" if not np.isfinite(v) else f"{v:.4g}") for v in vals))
    with open(os.path.join(res, "T_S6_dp_comparison.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    hdr = rows[0].split(",")
    with open(os.path.join(res, "T_S6_dp_comparison.md"), "w", encoding="utf-8") as f:
        f.write("# T_S6 — DP logs: surveyed geometry and comparison with 2018 (Markdown copy of `T_S6_dp_comparison.csv`)\n\n"
                + G + "; primary geometry, scenario A.\n\n| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n"
                + "\n".join("| " + " | ".join(l.split(",")) + " |" for l in rows[1:]) + "\n")
    figures_dp(R, names, plt, s18, figs)
    return L, M


def verdict_lines(R, es):
    F = R["final"]; V = R["verdicts"]["primary|A"]; out = [f"## Pre-registered expectations — {R['gap']}", ""]
    desc = dict(E15="DP log-midpoint base inside the 2018 BEL 90 % interval for ≥ 2/3 of the included DPs at x = 100–450 m",
                E16="most included DPs at x = 150–350 m with ρ_below,DP < 12.45 Ω·m and none with > 3 m of ≥ 20 Ω·m below the base",
                E17="DP base closer to the 2018 log-midpoint where log-midpoint and 3 Ω·m differ by > 2 m",
                E18="image 3 Ω·m crossing on the side of the DP 3 Ω·m crossing predicted by q_DP, at ≥ 2/3 of DPs with r ≤ 10 m",
                E19="λ-pooled log-midpoint band contains the DP base at more DPs than the λ = 50 3 Ω·m band contains the DP 3 Ω·m crossing",
                E20="image ρ 4–10 m below its base lower than ρ_below,DP at ≥ 2/3 of DPs with r ≤ 10 m",
                E21="no fine-grained layer within ±1 m of the DP log-midpoint base at ≥ 5 of 7 DPs")
    for e in es:
        v = V[e]; fv = F[e]
        if e == "E19":
            det = f"log-midpoint {v['n_logmid']} vs 3 Ω·m {v['n_thr3']} of {v['n']} DPs with r ≤ 10 m"
        elif e == "E16":
            det = f"{v['n_ok']} of {v['n']} below 12.45 Ω·m ({v['n_rho']} estimable; DPs {', '.join(v['dps']) or 'none'}); thick ≥ 20 Ω·m zones below the base: {', '.join(v['thick']) or 'none'}"
        elif e == "E21":
            det = f"{v['n_ok']} of {v['n']} DPs without a fine-grained layer near the base; lithology-confounded: {', '.join(R['confounded']) or 'none'}"
        else:
            det = f"{v['n_ok']} of {v['n']} ({', '.join(v.get('ok', [])) or 'none'} of {', '.join(v.get('dps', [])) or 'none'})"
        out.append(f"- **{e}** ({desc[e]}): **{fv['verdict']}**" + (f" — *{', '.join(fv['flags'])}*" if fv["flags"] else "") + f" — {det}."
                   + (f" Differs in: {', '.join(fv['differs_in'])}." if fv["differs_in"] else ""))
    return out


def figures_dp(R, names, plt, s18, figs):
    COL = s18.COL; logs = read_dp_values(); n = len(names)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    # F_12: DP vs 2018, with I_SBT strip
    fig = plt.figure(figsize=(2.05 * n, 5.6))
    gs = fig.add_gridspec(1, 2 * n, width_ratios=[5, 1] * n, wspace=0.08)
    ax0 = None
    for j, nm in enumerate(names):
        o = R["dps"][nm]["per"]["primary|A"]; lg = logs[min(logs, key=lambda k: abs(k - R["dps"][nm]["local_x"]))]
        ax = fig.add_subplot(gs[0, 2 * j], sharey=ax0); ax0 = ax0 or ax
        z1 = o["surface"] - lg["depth"]
        ax.axhspan(o["bel"][1], o["bel"][2], color=COL["bel"], alpha=0.18, lw=0); ax.axhline(o["bel"][0], color=COL["bel"], lw=1.1)
        lp = o["logmid_pooled"]
        if lp[0] >= 0.5:
            ax.axhspan(lp[2], lp[3], facecolor="none", edgecolor=COL["logmid"], hatch="////", lw=0, alpha=0.6)
            ax.axhline(lp[1], color=COL["logmid"], lw=0.9, ls="--")
        ax.axhline(o["sal1980"], color="#b2182b", lw=1.0, ls=":")
        ax.plot(lg["r"], z1, color="0.75", lw=0.4); ax.plot(10 ** np.array(o["v_grid"]), o["z_grid"], color="k", lw=1.0)
        rc = np.array([np.nan if v is None else v for v in o["ref_col"]["v"]], float)
        ax.plot(10 ** rc, o["ref_col"]["z"], color="#d35400", lw=1.2)
        for v, ls in ((2.5, ":"), (3.0, "--"), (20.0, "-.")):
            ax.axvline(v, color="0.45", lw=0.6, ls=ls)
        if np.isfinite(o["logmid"]):
            ax.plot([0.35], [o["logmid"]], marker=">", color="k", ms=6, clip_on=False)
        ax.set_xscale("log"); ax.set_xlim(0.3, 300); ax.set_ylim(-36, 6)
        ax.set_title(f"{nm} ({R['dps'][nm]['cpt']})\nx = {o['x']:.0f} m", fontsize=7); ax.set_xlabel("ρ (Ω·m)", fontsize=6.5)
        ax.tick_params(labelsize=6)
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
        sx = fig.add_subplot(gs[0, 2 * j + 1], sharey=ax0)
        I = np.array([np.nan if v is None else v for v in R["dps"][nm]["isbt"]], float); ze = np.array(R["dps"][nm]["cpt_elev"])
        colr = np.where(I < ISBT_SAND, "#f6e7b4", np.where(I <= ISBT_FINE, "#d9c48f", "#7a6a55"))
        for zi, ci in zip(ze, colr):
            sx.axhspan(zi - 0.1, zi + 0.1, color=ci, lw=0)
        sx.set_xticks([]); plt.setp(sx.get_yticklabels(), visible=False); sx.set_xlabel("I_SBT", fontsize=5.5)
    ax0.set_ylabel("Elevation (mTAW)")
    h = [Line2D([], [], color="0.75", lw=0.8, label="DP log (1 cm)"), Line2D([], [], color="k", label="DP log, 0.5 m median"),
         Line2D([], [], color="#d35400", label="2018 reference model (λ = 50)"), Patch(facecolor=COL["bel"], alpha=0.3, label="2018 BEL lens base: median, 90 %"),
         Patch(facecolor="none", edgecolor=COL["logmid"], hatch="////", label="2018 log-midpoint, λ-pooled"),
         Line2D([], [], color="#b2182b", ls=":", label="1980 saline base"), Line2D([], [], marker=">", ls="", color="k", label="DP log-midpoint base"),
         Patch(facecolor="#f6e7b4", label="CPT clean sand"), Patch(facecolor="#d9c48f", label="sand mixtures"), Patch(facecolor="#7a6a55", label="fine-grained")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=6.3, frameon=False)
    fig.suptitle(f"DP logs (CPT-EC, 23 Sep 2024) vs the 2018 results — {R['gap']}; lines at 2.5 / 3 / 20 Ω·m", fontsize=8)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.2)
    fig.savefig(os.path.join(figs, "F_12_dp_vs_2018.png"), dpi=300); plt.close(fig)
    # F_12b: DP vs same-epoch image
    fig, axes = plt.subplots(1, n, figsize=(1.75 * n, 5.2), sharey=True)
    for ax, nm in zip(axes, names):
        o = R["dps"][nm]["per"]["primary|A"]; s_ = o["sd"]; lg = logs[min(logs, key=lambda k: abs(k - R["dps"][nm]["local_x"]))]
        bl = s_["band_logmid_pooled"]; bt = s_["band_thr3"]
        if bl[0] >= 0.5:
            ax.axhspan(bl[2], bl[3], facecolor="none", edgecolor=COL["logmid"], hatch="////", lw=0, alpha=0.6)
        if bt[0] >= 0.5:
            ax.axhspan(bt[2], bt[3], color=COL["thr3"], alpha=0.18, lw=0)
        ax.plot(lg["r"], o["surface"] - lg["depth"], color="0.75", lw=0.4); ax.plot(10 ** np.array(o["v_grid"]), o["z_grid"], color="k", lw=1.0)
        ic = np.array([np.nan if v is None else v for v in o["img_col"]["v"]], float)
        ax.plot(10 ** ic, o["img_col"]["z"], color="#1b9e77", lw=1.3)
        for v, ls in ((3.0, "--"), (20.0, "-.")):
            ax.axvline(v, color="0.45", lw=0.6, ls=ls)
        ax.set_xscale("log"); ax.set_xlim(0.3, 300); ax.set_ylim(-36, 6)
        ax.set_title(f"{nm}: x = {o['x']:.0f} m\nimage r = {f_r(s_['r'])} m", fontsize=7); ax.tick_params(labelsize=6)
    axes[0].set_ylabel("Elevation (mTAW)")
    h = [Line2D([], [], color="k", label="DP log, 0.5 m median"), Line2D([], [], color="#1b9e77", label="ERT.dat image (our pipeline, λ = 50)"),
         Patch(facecolor="none", edgecolor=COL["logmid"], hatch="////", label="image log-midpoint, λ-pooled 90 %"),
         Patch(facecolor=COL["thr3"], alpha=0.3, label="image 3 Ω·m, λ = 50 90 %")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.5, frameon=False)
    fig.suptitle(f"Same epoch: DP logs vs ERT.dat inverted with our pipeline ({R['sd_date']})", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95)); fig.savefig(os.path.join(figs, "F_12b_dp_vs_samedate.png"), dpi=300); plt.close(fig)
    # F_13: section
    k1 = _load("k1c", "21_k1_comparison.py")
    v4 = np.load(s18.LIN_V4); x4, b4, r4 = v4["x"], v4["pooled_logmid"], v4["r_logmid"]
    pb = os.path.join(wc.ROOT, "data", "processed", "v1_1", "k1_field_bel.npz"); zb_ = np.load(pb); TX, Pf = zb_["TX"], zb_["Pf"]
    lb = k1.read_lebbe(); ex, ez = wc.land_surface()
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ok = (b4[0] >= 0.5) & (r4 <= 10.0)
    ax.axvspan(*k1.DIS_ZONE, color="#e6c229", alpha=0.16, lw=0, zorder=0)
    ax.fill_between(x4, np.where(ok, b4[2], np.nan), np.where(ok, b4[3], np.nan), color=COL["logmid"], alpha=0.18, lw=0)
    h_lm, = ax.plot(x4, np.where(ok, b4[1], np.nan), color=COL["logmid"], lw=1.1)
    lo, md, hi = np.percentile(Pf, [5, 50, 95], 0); hh = TX >= 130
    ax.fill_between(TX[hh], lo[hh], hi[hh], color=COL["bel"], alpha=0.22, lw=0); h_bel, = ax.plot(TX[hh], md[hh], color=COL["bel"], lw=1.7)
    v = lb["saline_base_mTAW"]; m = np.isfinite(v)
    h80 = ax.errorbar(lb["x_ours_m"][m], v[m], xerr=k1.X_ERR, yerr=k1.Z_ERR, fmt="o", color="#b2182b", ms=4, lw=0.8, capsize=1.5)
    for nm in names:
        for fl in R["dps"][nm]["fine_layers"]:
            x = R["dps"][nm]["our_x"]
            ax.add_patch(plt.Rectangle((x - 4, fl["bottom"]), 8, fl["top"] - fl["bottom"], facecolor="none", edgecolor="#6a3d9a", hatch="xxxx", lw=0.5))
    xd = np.array([P_(R, nm)["x"] for nm in names]); yd = np.array([P_(R, nm)["logmid"] for nm in names]); y3 = np.array([P_(R, nm)["thr3"] for nm in names])
    h_dp = ax.errorbar(xd, yd, xerr=1.5, fmt="D", color="k", ms=4.5, lw=1.0, capsize=2)
    h_dp3, = ax.plot(xd, y3, "D", mfc="none", mec="k", ms=4.5)
    for nm, xx, yy in zip(names, xd, yd):
        ax.text(xx, (yy if np.isfinite(yy) else -30) - 2.4, nm, ha="center", fontsize=5.8)
    h_s, = ax.plot(ex, ez, "k-", lw=0.9); ax.axvline(161, color="steelblue", lw=0.8, ls=":")
    ax.set_xlim(0, 505); ax.set_ylim(-34, 8); s18.dune_hatch(ax)
    ax.set_xlabel("Distance along profile x (m)"); ax.set_ylabel("Elevation (mTAW)"); s18.ve_text(ax, 3); ax.set_aspect(3)
    blank = Patch(facecolor="none", edgecolor="none")
    hs = [blank, h_lm, h_bel, Patch(facecolor="#e6c229", alpha=0.35, edgecolor="none"), h_s, blank, h80, blank, h_dp, h_dp3,
          Patch(facecolor="none", edgecolor="#6a3d9a", hatch="xxxx")]
    ls_ = ["This study, ERT of 7 March 2018", "λ-pooled log-midpoint (r ≤ 10 m): median, P5–P95", "BEL (site + physical): median, P5–P95",
           "discharge zone (this study)", "land surface (2018 electrodes)", "Lebbe (1981), K1, logs of 1980", "saline base (< 2.5 Ω·m)",
           "DP / CPT, 23 September 2024 (Vrancken et al.)", "DP log-midpoint base (surveyed x, offset ≤ 1.5 m)", "DP 3 Ω·m crossing",
           "CPT fine-grained layers (I_SBT > 2.6)"]
    leg = ax.legend(hs, ls_, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=6, facecolor="white", edgecolor="0.6", framealpha=1.0)
    for t in leg.get_texts():
        if t.get_text() in (ls_[0], ls_[5], ls_[7]):
            t.set_fontweight("bold")
    fig.savefig(os.path.join(figs, "F_13_dp_section.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    fig_location_v14(R, names, plt, s18, os.path.join(figs, "F_1b_location_inset.png"))


def f_r(v):
    return "—" if v is None or not np.isfinite(v) else f"{v:.0f}"


def P_(R, nm):
    return R["dps"][nm]["per"]["primary|A"]


def fig_location_v14(R, names, plt, s18, path):
    from cartopy.io import shapereader
    from shapely.geometry import box
    tr, _ = transformer(True)
    P, X = our_line()
    J = json.load(open(os.path.join(wc.ROOT, "data", "processed", "v1_4", "cpt.json"), encoding="utf-8"))
    pl = json.load(open(os.path.join(wc.ROOT, "data", "processed", "v1_4", "placement.json"), encoding="utf-8"))
    bb = box(2.50, 51.06, 2.62, 51.13)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for name, kw in (("coastline", dict(color="0.3", lw=0.8)), ("admin_0_boundary_lines_land", dict(color="0.3", lw=0.8, ls="--"))):
        for g in shapereader.Reader(os.path.join(s18.NE_DIR, f"ne_10m_{name}.shp")).geometries():
            g = g.intersection(bb)
            if g.is_empty:
                continue
            for part in getattr(g, "geoms", [g]):
                lon, lat = np.asarray(part.xy); e, n_ = tr.transform(lon, lat); ax.plot(e, n_, **kw)
    ax.plot(P[:, 0], P[:, 1], "-", color="#1f4e9c", lw=2.0, label="2018 land line = K1 (Lebbe 1981)")
    b_e, b_n = tr.transform(2.54528, 51.08944); az = np.radians(COAST_AZ - 90.0); L_ = 500.0
    ax.plot([b_e - 0.4 * L_ * np.sin(az), b_e + 0.6 * L_ * np.sin(az)], [b_n - 0.4 * L_ * np.cos(az), b_n + 0.6 * L_ * np.cos(az)],
            color="0.45", lw=1.4, ls=(0, (3, 2)), label="K0 (Lebbe 1981; approximate)")
    # Zenodo line from the fitted X0 (ends on our chord)
    d = P[1] - P[0]; Lc = np.hypot(*d)
    pt = lambda x: P[0] + d * (x - X[0]) / (X[1] - X[0])
    a_, b_ = pt(X0_CPT - 355.0), pt(X0_CPT)
    ax.plot([a_[0], b_[0]], [a_[1], b_[1]], "-", color="#d35400", lw=1.0, label="recent ERT line (Zenodo; date unconfirmed)")
    cx = [J["cpts"][s]["lambert_x"] for s in sorted(J["cpts"])]; cy = [J["cpts"][s]["lambert_y"] for s in sorted(J["cpts"])]
    ax.plot(cx, cy, "o", ms=3.2, color="k", label="CPT / DP S02–S08 (surveyed; 23 Sep 2024)")
    ax.text(cx[0] + 25, cy[0] - 10, "DP1", fontsize=6); ax.text(cx[-1] + 25, cy[-1] - 10, "DP7", fontsize=6)
    ax.text(b_e + 40, b_n - 120, "FR | BE", fontsize=6, color="0.3")
    cxm, cym = P.mean(0)
    ax.set_xlim(cxm - 1300, cxm + 700); ax.set_ylim(cym - 1000, cym + 900); ax.set_aspect("equal")
    ax.set_xlabel("Lambert 72 x (m)", fontsize=7); ax.set_ylabel("Lambert 72 y (m)", fontsize=7); ax.tick_params(labelsize=5.5)
    ax.ticklabel_format(useOffset=False, style="plain"); ax.legend(fontsize=5.3, loc="lower right", framealpha=1)
    ax.set_title("Location (coastline and border: Natural Earth 1:10 m)", fontsize=6.5)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)




def report_method(args):
    """4_inreview_method.md: the authors' approach, read from Python_script.py (not executed; their ensemble not run)."""
    res = os.path.join(args.out, "results", "v1_4")
    L = ["# 4 — The in-review method of Vrancken, Paepen & Hermans (record of their approach)", "",
         "Source: `Python_script.py` of Zenodo record 10.5281/zenodo.20792093 (CC-BY 4.0), read but not executed; their "
         "ensemble was not run here. The accompanying paper is in review; this summary describes only what the script shows.", "",
         "- **Data.** `ERT.dat` (pyGIMLi format, i and u per quadrupole) on `Mesh.bms`; data with i ≤ 0 or u ≤ 0 removed; "
         "numerical geometric factors; ρa = k·R; error = 10 % relative (`ert.estimateError(relativeError=0.1)`).",
         "- **Deterministic ensemble.** Repeated pyGIMLi inversions in log ρ (TransLog for data and model), starting from a "
         "homogeneous model at the median ρa; per member a random λ from 1–999 (Python `random`, seed 227843, 100 distinct "
         "values), and — in commented-out variants — random correlation lengths ah 10–119 m and av 5–14 m (geostatistical "
         "regularisation), a homogeneous reference model of 1–72 Ω·m, or an SGS reference model (`SGS_DP_R.txt`, a "
         "geostatistical simulation conditioned on the DP data). The active code runs λ = 17.",
         "- **Joint ERT + DP variant.** The DP values (`DP_R.xzr`: local x, z, ρ) enter as point data through a "
         "`PriorModelling` operator combined with the ERT operator in a `JointModelling` inversion; 10 % error on the DP data "
         "(comment: \"Grunenbaum: 10 %; example online 20 %\").",
         "- **Outputs per member.** model, response, χ², relative and absolute RMS, data and errors (`np.savez`).",
         "- **Differences from our pipeline.** Their uncertainty comes from the spread of an ensemble of regularised "
         "deterministic models (λ, correlation lengths, reference model), not from a linearised posterior or a learned "
         "prior; 10 % instead of our 3 % + 1e-4 Ω error model; no outlier removal beyond non-positive values; flat "
         "topography (all electrodes at z = 0). Our same-epoch test (`3_same_epoch.md`) inverts their ERT.dat with our "
         "pipeline instead."]
    with open(os.path.join(res, "4_inreview_method.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    r = sub.add_parser("run"); r.add_argument("--out", default=wc.ROOT); r.add_argument("--workers", type=int, default=6)
    sub.add_parser("report").add_argument("--out", default=wc.ROOT)
    sub.add_parser("cpt").add_argument("--out", default=wc.ROOT)
    sub.add_parser("samedate").add_argument("--out", default=wc.ROOT)
    sub.add_parser("dp").add_argument("--out", default=wc.ROOT)
    a = ap.parse_args()
    dict(fetch=cmd_fetch, run=cmd_run, report=cmd_report, cpt=cmd_cpt, samedate=cmd_samedate, dp=cmd_dp)[a.cmd](a)


if __name__ == "__main__":
    main()
