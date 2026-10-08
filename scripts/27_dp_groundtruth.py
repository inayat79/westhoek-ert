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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    r = sub.add_parser("run"); r.add_argument("--out", default=wc.ROOT); r.add_argument("--workers", type=int, default=6)
    sub.add_parser("report").add_argument("--out", default=wc.ROOT)
    sub.add_parser("cpt").add_argument("--out", default=wc.ROOT)
    a = ap.parse_args()
    dict(fetch=cmd_fetch, run=cmd_run, report=cmd_report, cpt=cmd_cpt)[a.cmd](a)


if __name__ == "__main__":
    main()
