"""
10_synthetic_benchmark.py
=========================
Westhoek ERT — synthetic benchmark of the linearised UQ pipeline (Stage 4).

Subcommands (run in this order):
  spec     zone geometry + zone medians from g1_lam50 → results/synthetic/truth_models.md,
           data/processed/synthetic/truth_spec.json  (written BEFORE any model is built)
  build    fine forward mesh (≠ inversion mesh), forward-mesh accuracy check, noise-free
           resistances of every truth model on the exact G1 quadrupoles; true cell averages
           on the inversion mesh and true interface depths per column
  run      parallel, resumable driver: each job = (truth model, realisation):
             noise (3 % + 1e-4 Ω) → [T6: corruption + Stage-1 outlier rule] → inversion(s)
             → resolution → 1000 linearised-posterior samples → per-run summaries
           Inversion, error model and resolution use the functions of 02_land_inversion.py
           (identical pipeline to the field data).
  metrics  coverage / bias / band width / reliability / T4, T5, T6 metrics → results/synthetic/

Seeds (all fixed and recorded in truth_spec.json):
  noise            1000 + realisation                (paired across truth models)
  T6 corruption    5000 + realisation
  posterior        7_000_000 + 100_000·truth_index + 100·realisation + λ

Only writes to data/processed/synthetic/, figures/synthetic/, results/synthetic/.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import scipy.linalg as sla

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import westhoek_common as wc  # noqa: E402

SYN_DATA = os.path.join(wc.PROCESSED_DIR, "synthetic")
SYN_FIG = os.path.join(wc.FIG_DIR, "synthetic")
SYN_RES = os.path.join(wc.RESULTS_DIR, "synthetic")
SPEC_JSON = os.path.join(SYN_DATA, "truth_spec.json")
REF_TAG = "g1_lam50"             # zone medians and lens-base line come from this run
G1_TAG = "deep_drop_data"        # its keep_idx = the exact G1 quadrupoles

TRUTHS = ["T1", "T1h", "T2a", "T2b", "T2c", "T3", "T4", "T5", "T6"]
N_REAL = 20
N_REAL_LAMBDA = 5
LAM_MAIN = 50.0
LAM_EXTRA = [5.0, 10.0, 20.0, 100.0]
N_SAMPLES = 1000
ERR_REL, ERR_ABS = 0.03, 1e-4
PARA_DEPTH, PARA_DX, MAX_ITER = 60.0, 0.33, 30
COLS = np.arange(100.0, 451.0, 10.0)          # metric columns
Z_WT = 3.0                                    # dune water table (mTAW), top of the fresh aquifer
X_LENS0, X_DIS0, X_DIS1 = 90.0, 370.0, 470.0  # lens start, discharge zone start/end
Z_DIS_BOT = -15.0                             # base of the discharge zone (mTAW)
Z_DEEP = -30.0                                # top of the deep zone (mTAW)
CLAY = dict(x0=20.0, x1=95.0, z0=-8.0, z1=-5.0, rho=5.0)
TRANSITION = 5.0                              # T3: log-linear transition thickness (m), centred on the base
FWD = dict(paraDX=0.2, paraDepth=PARA_DEPTH, paraMaxCellSize=5.0, quality=34.0)


def seeds(truth, real, lam=None):
    ti = TRUTHS.index(truth)
    return dict(noise=1000 + real, corrupt=5000 + real,
                post=None if lam is None else 7_000_000 + 100_000 * ti + 100 * real + int(lam))


# ─────────────────────────────────────────────────────────────────────────────
# Truth models
# ─────────────────────────────────────────────────────────────────────────────
class Truth:
    """Zone-based resistivity model; geometry and values from truth_spec.json."""

    def __init__(self, spec, name):
        self.name = name
        self.v = dict(spec["values"][name])
        self.bx = np.array(spec["base_line"]["x"]); self.bz = np.array(spec["base_line"]["z"])
        self.sx = np.array(spec["surface"]["x"]); self.sz = np.array(spec["surface"]["z"])
        self.flat = spec["variants"].get(name, {}).get("flat_base")
        self.transition = spec["variants"].get(name, {}).get("transition", 0.0)
        self.clay = spec["variants"].get(name, {}).get("clay")
        self.no_discharge = spec["variants"].get(name, {}).get("no_discharge", False)

    def base(self, x):
        b = np.interp(x, self.bx, self.bz)
        if self.flat is not None:
            b = np.where((x >= X_LENS0) & (x < X_DIS0), self.flat, b)
        if self.no_discharge:
            b370, b470 = np.interp(X_DIS0 - 1e-6, self.bx, self.bz), np.interp(X_DIS1, self.bx, self.bz)
            lin = b370 + (x - X_DIS0) / (X_DIS1 - X_DIS0) * (b470 - b370)
            b = np.where((x >= X_DIS0) & (x < X_DIS1), lin, b)
        return b

    def labels(self, x, z):
        """0 cap, 1 fresh, 2 lens, 3 brackish, 4 discharge, 5 deep (clay handled in log10rho)."""
        x = np.asarray(x, float); z = np.asarray(z, float)
        b = self.base(x)
        lab = np.full(x.shape, 3, int)
        dune = x < X_LENS0
        lab[dune & (z > Z_WT)] = 0
        lab[dune & (z <= Z_WT)] = 1
        lens = ~dune & (z > b)
        lab[lens] = 2
        if not self.no_discharge:
            dis = (x >= X_DIS0) & (x < X_DIS1) & (z <= b) & (z > Z_DIS_BOT)
            lab[dis] = 4
        lab[(z <= Z_DEEP) & ~lens] = 5          # the lens overrides the deep zone (T2c: base at −35)
        return lab

    def log10rho(self, x, z):
        x = np.asarray(x, float); z = np.asarray(z, float)
        names = ["cap", "fresh", "lens", "brackish", "discharge", "deep"]
        vals = np.log10(np.array([self.v[n] for n in names]))
        lab = self.labels(x, z)
        out = vals[lab]
        if self.transition:
            # log-linear transition centred on the lens base (lens part x 90–370 m only)
            b = self.base(x)
            d = z - b
            sel = (x >= X_LENS0) & (x < X_DIS0) & (np.abs(d) < self.transition / 2) & (z > Z_DEEP)
            below = vals[3]
            f = (d[sel] + self.transition / 2) / self.transition        # 0 at bottom … 1 at top
            out[sel] = below + f * (vals[2] - below)
        if self.clay:
            c = self.clay
            sel = (x >= c["x0"]) & (x <= c["x1"]) & (z >= c["z0"]) & (z <= c["z1"])
            out[sel] = np.log10(c["rho"])
        return out


def true_interfaces(truth, xcols, dz=0.02):
    """Main-lens top/base of the truth on vertical columns (fine sampling)."""
    ex, ez = wc.land_surface()
    s = np.interp(xcols, ex, ez)
    z = np.arange(np.floor(ez.max() / dz) * dz, wc.Z_FLOOR - 1e-9, -dz)
    X, Z = np.meshgrid(xcols, z)
    V = truth.log10rho(X, Z)
    V[Z > s[None, :] - 1e-6] = np.nan
    top, base = wc.main_lens(V[:, :, None], z)
    return top[:, 0], base[:, 0]


def cell_truth_average(truth, mesh, n_side=6):
    """Geometric-mean true resistivity over each inversion cell (triangle quadrature points)."""
    pts = []
    for i in range(n_side + 1):
        for j in range(n_side + 1 - i):
            pts.append((i / n_side, j / n_side))
    bary = np.array([(1 - a - b, a, b) for a, b in pts])        # (npts, 3)
    out = np.empty(mesh.cellCount())
    for c in mesh.cells():
        P = np.array([[n.pos().x(), n.pos().y()] for n in c.nodes()])[:3]
        q = bary @ P
        out[c.id()] = np.mean(truth.log10rho(q[:, 0], q[:, 1]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
def cmd_spec(args):
    os.makedirs(SYN_DATA, exist_ok=True); os.makedirs(SYN_RES, exist_ok=True)
    ref = wc.load_run(REF_TAG)
    mesh, _ = wc.rebuild_paradomain(ref)
    ex, ez = wc.land_surface()
    xb = np.arange(np.ceil(ex.min()), np.floor(ex.max()) + 1, 1.0)
    cs = wc.ColumnSampler(mesh, xb)
    _, base = wc.main_lens(cs(np.log10(ref["model"]))[:, :, None], cs.z)
    base = base[:, 0]
    ok = np.isfinite(base) & (xb >= X_LENS0)
    bz = np.interp(xb, xb[ok], base[ok])
    # 5-point running median to remove single-column jitter of the extraction
    bz_s = np.array([np.median(bz[max(0, i - 2):i + 3]) for i in range(len(bz))])
    spec = dict(
        reference_run=REF_TAG, g1_quadrupoles_from=G1_TAG,
        surface=dict(x=list(map(float, ex)), z=list(map(float, ez))),
        base_line=dict(x=list(map(float, xb)), z=list(map(float, bz_s)),
                       source=f"main-lens ρ=3 Ω·m base of {REF_TAG} (raw, incl. r>r_max), NaN-gaps "
                              "linearly filled, 5-point running median"),
        geometry=dict(Z_WT=Z_WT, X_LENS0=X_LENS0, X_DIS0=X_DIS0, X_DIS1=X_DIS1, Z_DIS_BOT=Z_DIS_BOT,
                      Z_DEEP=Z_DEEP, CLAY=CLAY, TRANSITION=TRANSITION),
    )
    # zone medians from the reference model (cell centres)
    tmp = dict(spec, values={"T1": {k: 1.0 for k in ("cap", "fresh", "lens", "brackish", "discharge", "deep")}},
               variants={})
    lab = Truth(tmp, "T1").labels(ref["cell_x"], ref["cell_z"])
    names = ["cap", "fresh", "lens", "brackish", "discharge", "deep"]
    med = {n: float(np.median(ref["model"][lab == i])) for i, n in enumerate(names)}
    cnt = {n: int((lab == i).sum()) for i, n in enumerate(names)}
    T1 = dict(med)
    T1h = dict(med, lens=1.0, fresh=2 * med["fresh"])
    values = {t: dict(T1) for t in TRUTHS}
    values["T1h"] = T1h
    variants = {"T2a": {"flat_base": -15.0}, "T2b": {"flat_base": -25.0}, "T2c": {"flat_base": -35.0},
                "T3": {"transition": TRANSITION}, "T4": {"clay": CLAY}, "T5": {"no_discharge": True},
                "T6": {"outliers": dict(electrodes=list(range(39, 46)), role="P1/P2", fraction=0.4,
                                        residuals=f"bootstrap of F (deep_all) outlier normalised residuals "
                                                  "where electrodes 39–45 act as P1/P2, random sign",
                                        rule="Stage 1: invert all G1 data at λ=20 (maxIter 30), drop "
                                             "|r| > 3·1.4826·MAD(r), then invert kept data")}}
    spec.update(values=values, variants=variants, zone_cell_counts=cnt, zone_medians=med,
                noise=dict(rel=ERR_REL, abs_ohm=ERR_ABS, realisations=N_REAL),
                inversion=dict(lam_main=LAM_MAIN, lam_extra=LAM_EXTRA, realisations_lambda=N_REAL_LAMBDA,
                               paraDepth=PARA_DEPTH, paraDX=PARA_DX, maxIter=MAX_ITER, zWeight=0.5,
                               err_rel=ERR_REL, samples=N_SAMPLES),
                forward_mesh=FWD, metric_columns=list(map(float, COLS)),
                seeds=dict(noise="1000 + realisation", corrupt="5000 + realisation",
                           posterior="7_000_000 + 100_000·truth_index + 100·realisation + λ",
                           truth_index=TRUTHS))
    with open(SPEC_JSON, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)

    L = ["# Synthetic truth models — Stage 4", "",
         f"Written before any model was built. Reference run for zone values and the lens-base line: `{REF_TAG}` "
         f"(G1 data, λ = 50, paraDepth 60 m, paraDX 0.33). Machine-readable copy: `data/processed/synthetic/truth_spec.json`.",
         "", "## Zone geometry (elevations in mTAW, x along profile in m)", "",
         "| zone | definition |", "|---|---|",
         f"| dry sand cap | x < {X_LENS0:g}, above the dune water table z = {Z_WT:g} |",
         f"| dune fresh aquifer | x < {X_LENS0:g}, from z = {Z_WT:g} down to {Z_DEEP:g} |",
         f"| saline lens | x ≥ {X_LENS0:g}, from the surface down to the lens-base line b(x) "
         f"(for {X_DIS0:g} ≤ x < {X_DIS1:g} this is the thin saline layer above the discharge zone; "
         f"seaward of {X_DIS1:g} the seaward saltwater) |",
         f"| discharge zone | {X_DIS0:g} ≤ x < {X_DIS1:g}, from b(x) down to {Z_DIS_BOT:g} |",
         f"| brackish (below lens) | x ≥ {X_LENS0:g}, below the lens / discharge zone, down to {Z_DEEP:g} |",
         f"| deep | z ≤ {Z_DEEP:g} everywhere |", "",
         f"Lens-base line b(x): {spec['base_line']['source']}. Values at selected x: " +
         ", ".join(f"x={x:g}: {np.interp(x, xb, bz_s):.1f}" for x in (90, 150, 200, 250, 300, 350, 400, 450, 480, 505)) + ".",
         "", "## Zone medians of g1_lam50 (cell centres in each zone)", "",
         "| zone | cells | median ρ (Ω·m) |", "|---|---|---|"]
    L += [f"| {n} | {cnt[n]} | {med[n]:.2f} |" for n in names]
    L += ["", "## Truth models — every value used (Ω·m)", "",
          "| model | cap | fresh | lens | brackish | discharge | deep | modification |", "|---|---|---|---|---|---|---|---|"]
    mods = {"T1": "Westhoek-like; lens x 90–370 m, base = b(x)",
            "T1h": "T1, high contrast: lens 1.0 Ω·m, fresh aquifer 2× median",
            "T2a": "T1, flat lens base at −15 for 90 ≤ x < 370", "T2b": "T1, flat lens base at −25",
            "T2c": "T1, flat lens base at −35 (below Z_DEEP: lens replaces deep zone down to −35)",
            "T3": f"T1, {TRANSITION:g} m log-linear transition centred on b(x) (x 90–370 m)",
            "T4": f"T1 + clay {CLAY['rho']:g} Ω·m at x {CLAY['x0']:g}–{CLAY['x1']:g}, z {CLAY['z0']:g} … {CLAY['z1']:g} "
                  "(overrides other zones; x 90–95 overlaps the lens edge)",
            "T5": "T1 without discharge zone: lens continues to x 470 with base interpolated linearly "
                  "between b(370) and b(470); brackish below",
            "T6": "T1 + charge-up outliers on electrodes 39–45 as P1/P2 (40 % of those data), Stage-1 rule applied"}
    for t in TRUTHS:
        v = values[t]
        L.append(f"| {t} | {v['cap']:.2f} | {v['fresh']:.2f} | {v['lens']:.2f} | {v['brackish']:.2f} | "
                 f"{v['discharge']:.2f} | {v['deep']:.2f} | {mods[t]} |")
    L += ["", "Note on T2c: the flat base at −35 lies below the deep-zone top (−30); for x 90–370 the lens "
          "resistivity is used down to −35 (the lens overrides the deep zone there).",
          "", "## Data, noise, forward modelling", "",
          f"- Electrodes: the 102 land electrodes with topography; quadrupoles: exactly the {G1_TAG} (G1) kept set.",
          f"- Forward mesh: pyGIMLi createParaMesh with paraDX {FWD['paraDX']}, paraDepth {FWD['paraDepth']:g}, "
          f"paraMaxCellSize {FWD['paraMaxCellSize']} m², quality {FWD['quality']} (inversion mesh: paraDX 0.33, "
          "default max cell size and quality) — a different, finer triangulation; resistivity = truth at the cell centre.",
          f"- Noise on R: Gaussian with σ = √((0.03 R)² + (1e-4 Ω)²), {N_REAL} realisations per model, seed 1000 + realisation "
          "(the same noise draws are used for every truth model → paired comparisons).",
          "- True cell value for the reliability diagram: geometric mean of the truth over each inversion cell "
          "(28 barycentric quadrature points per triangle).",
          "- True interface depths: main-lens ρ = 3 Ω·m crossings of the truth on vertical columns (dz = 0.02 m).",
          "", "## Inversion and UQ (identical to the field pipeline)", "",
          f"- 02_land_inversion.py functions: build_datacont (err 3 % + 1e-4 Ω floor, numerical K), run_inversion "
          f"(paraDepth {PARA_DEPTH:g}, paraDX {PARA_DX}, zWeight 0.5, maxIter {MAX_ITER}), compute_resolution_matrix.",
          f"- λ = {LAM_MAIN:g} for all realisations; realisations 0–{N_REAL_LAMBDA-1} also λ = 5, 10, 20, 100 and the pooled mixture.",
          f"- {N_SAMPLES} linearised-posterior samples per inversion; seeds 7 000 000 + 100 000·truth_index + 100·realisation + λ."]
    path = os.path.join(SYN_RES, "truth_models.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"[saved] {path}\n[saved] {SPEC_JSON}")


# ─────────────────────────────────────────────────────────────────────────────
def load_spec():
    with open(SPEC_JSON, encoding="utf-8") as f:
        return json.load(f)


def g1_scheme():
    """G1 quadrupoles (electrode numbers) and the land-electrode table."""
    g1 = wc.load_run(G1_TAG)
    return dict(C1=g1["C1"].astype(int), C2=g1["C2"].astype(int), P1=g1["P1"].astype(int),
                P2=g1["P2"].astype(int), keep_idx=g1["keep_idx"].astype(int),
                elec_num=g1["elec_num"].astype(int), elec_x=g1["elec_x"], elec_z=g1["elec_z"])


def make_scheme(sch):
    import pygimli as pg
    data = pg.DataContainerERT()
    idx = {}
    for i, (e, x, z) in enumerate(zip(sch["elec_num"], sch["elec_x"], sch["elec_z"])):
        data.createSensor(pg.Pos(float(x), float(z))); idx[int(e)] = i
    data.resize(len(sch["C1"]))
    for key, col in (("a", "C1"), ("b", "C2"), ("m", "P1"), ("n", "P2")):
        data[key] = [idx[int(e)] for e in sch[col]]
    return data


def cmd_build(args):
    import pygimli as pg
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    spec = load_spec()
    os.makedirs(SYN_DATA, exist_ok=True)
    sch = g1_scheme()
    scheme = make_scheme(sch)
    print(f"[build] scheme: {scheme.size()} quadrupoles, {scheme.sensorCount()} electrodes")
    t = time.time()
    fmesh = mt.createParaMesh(scheme.sensors(), **FWD)
    print(f"[build] forward mesh: {fmesh.cellCount()} cells, {fmesh.nodeCount()} nodes ({time.time()-t:.1f}s)")
    cx = np.array([c.center().x() for c in fmesh.cells()]); cz = np.array([c.center().y() for c in fmesh.cells()])

    # forward accuracy check: homogeneous 10 Ω·m, R from the fine mesh × numerical K of the
    # inversion pipeline (build_datacont computes K on its own mesh) should give ρa ≈ 10
    scheme["k"] = np.ones(scheme.size())
    out = ert.simulate(fmesh, scheme=scheme, res=np.full(fmesh.cellCount(), 10.0), verbose=False)
    R_h = np.array(out["rhoa"])        # with k = 1, ρa = R
    K_num = np.array(ert.createGeometricFactors(make_scheme(sch), numerical=True))
    dev = R_h * K_num / 10.0 - 1.0
    acc = dict(median_abs_pct=float(100 * np.median(np.abs(dev))), p95_abs_pct=float(100 * np.percentile(np.abs(dev), 95)),
               max_abs_pct=float(100 * np.max(np.abs(dev))), bias_pct=float(100 * np.median(dev)))
    print(f"[build] forward accuracy (homogeneous 10 Ω·m, fine-mesh R × inversion-pipeline K): "
          f"median |dev| {acc['median_abs_pct']:.2f}%, P95 {acc['p95_abs_pct']:.2f}%, max {acc['max_abs_pct']:.2f}%, "
          f"median dev {acc['bias_pct']:+.2f}%")

    ref = wc.load_run(REF_TAG)
    imesh, _ = wc.rebuild_paradomain(ref)
    truthR, cellavg, ttop, tbase = {}, {}, {}, {}
    for tname in TRUTHS:
        tr = Truth(spec, tname)
        res = 10 ** tr.log10rho(cx, cz)
        t = time.time()
        out = ert.simulate(fmesh, scheme=scheme, res=res, verbose=False)
        truthR[tname] = np.array(out["rhoa"])
        cellavg[tname] = cell_truth_average(tr, imesh)
        ttop[tname], tbase[tname] = true_interfaces(tr, COLS)
        print(f"[build] {tname}: forward {time.time()-t:.1f}s, R [{truthR[tname].min():.4g}, {truthR[tname].max():.4g}] Ω, "
              f"negative R: {(truthR[tname] <= 0).sum()}")
    np.savez(os.path.join(SYN_DATA, "truth_forward.npz"),
             **{f"R_{t}": truthR[t] for t in TRUTHS}, **{f"cell_log10_{t}": cellavg[t] for t in TRUTHS},
             **{f"top_{t}": ttop[t] for t in TRUTHS}, **{f"base_{t}": tbase[t] for t in TRUTHS},
             cols=COLS, keep_idx=sch["keep_idx"], fwd_cells=fmesh.cellCount(),
             fwd_accuracy=json.dumps(acc))
    spec["forward_mesh_cells"] = int(fmesh.cellCount()); spec["forward_accuracy"] = acc
    with open(SPEC_JSON, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    with open(os.path.join(SYN_RES, "truth_models.md"), "a", encoding="utf-8") as f:
        f.write(f"\n## Forward mesh check (added by `build`)\n\nForward mesh: {fmesh.cellCount()} cells "
                f"(inversion mesh: {imesh.cellCount()}). Homogeneous 10 Ω·m: fine-mesh R × inversion-pipeline "
                f"numerical K deviates from 10 Ω·m by median {acc['median_abs_pct']:.2f} %, P95 {acc['p95_abs_pct']:.2f} %, "
                f"max {acc['max_abs_pct']:.2f} % (median signed {acc['bias_pct']:+.2f} %) — the modelling error between the "
                "two discretisations, i.e. no inverse crime.\n")
    for tname in TRUTHS:
        print(f"   true base at x=150,250,350: " + ", ".join(
            f"{tbase[tname][np.argmin(np.abs(COLS - x))]:.2f}" for x in (150, 250, 350)) + f"   ({tname})")
    print(f"[saved] {os.path.join(SYN_DATA, 'truth_forward.npz')}")


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────
def load_02():
    path = os.path.join(SCRIPT_DIR, "02_land_inversion.py")
    spec = importlib.util.spec_from_file_location("inv02", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def f_outlier_residuals():
    """F's outlier normalised residuals where electrodes 39–45 act as P1/P2."""
    d = wc.load_run("deep_all")
    r = (np.log(d["d_obs"]) - np.log(d["d_pred"])) / d["d_err"]
    med = np.median(r); mad = np.median(np.abs(r - med)); thr = 3 * 1.4826 * mad
    P1, P2, C1, C2 = (d[k].astype(int) for k in ("P1", "P2", "C1", "C2"))
    E = np.arange(39, 46)
    asP = (np.isin(P1, E) | np.isin(P2, E)) & ~(np.isin(C1, E) | np.isin(C2, E))
    return np.abs(r[asP & (np.abs(r) > thr)])


def one_inversion(m02, land, elec, lam, mesh_check, cs, rng_seed, max_iter=MAX_ITER):
    """Run the field pipeline for one λ; return summaries + posterior samples (ln ρ)."""
    t0 = time.time()
    data = m02.build_datacont(land, elec, err_rel=ERR_REL)[0]
    mgr = m02.run_inversion(data, lam=lam, para_depth=PARA_DEPTH, para_dx=PARA_DX, max_iter=max_iter)
    model = np.array(mgr.model)
    pd = mgr.paraDomain
    if pd.cellCount() != len(mesh_check):
        raise RuntimeError("inversion mesh differs from the reference mesh")
    d_obs, d_pred, d_err = np.array(data["rhoa"]), np.array(mgr.inv.response), np.array(data["err"])
    r = (np.log(d_obs) - np.log(d_pred)) / d_err
    chi2, chi2_pg, n_iter = float(np.mean(r ** 2)), float(mgr.inv.chi2()), m02.n_iterations(mgr)
    stop = "chi2<=1" if chi2_pg <= 1 else ("iteration cap" if n_iter >= max_iter else "dPhi<2%")
    t_inv = time.time() - t0
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        res = m02.compute_resolution_matrix(mgr, data, lam_expected=lam)
    area = np.array(pd.cellSizes())
    radius = np.sqrt(area / (np.pi * np.maximum(res["Rm_diag"], 1e-6)))
    run = dict(J=res["J"], data_err=res["err"], C=res["C"], lam=lam, model=model)
    L, A, _, _ = wc.cholesky_A(run)
    rng = np.random.default_rng(rng_seed)
    S = np.log(model)[:, None] + sla.solve_triangular(L, rng.standard_normal((len(model), N_SAMPLES)),
                                                      lower=True, trans="T")
    # interfaces of m̂ + analytic bands (cell σ and wᵀA⁻¹w) + r at the crossing
    V = cs(np.log10(model))
    top, base = wc.main_lens(V[:, :, None], cs.z)
    top, base = top[:, 0], base[:, 0]
    sw_base, _ = wc.sigma_z_interp(cs, L, np.log10(model), base)
    sw_top, _ = wc.sigma_z_interp(cs, L, np.log10(model), top)
    Rg = cs(np.log10(radius))
    dz = cs.z[0] - cs.z[1]
    def r_at(zc):
        out = np.full(len(zc), np.nan)
        for j, z in enumerate(zc):
            if np.isfinite(z):
                k = int(np.clip(np.floor((cs.z[0] - z) / dz), 0, len(cs.z) - 1))
                out[j] = 10 ** Rg[k, j]
        return out
    return dict(lam=lam, chi2=chi2, chi2_pg=chi2_pg, n_iter=n_iter, stop=stop, t_inv=t_inv,
                t_total=time.time() - t0, n_data=int(len(d_obs)), model=model.astype(np.float32),
                radius=radius.astype(np.float32), P3=(S < np.log(3)).mean(axis=1).astype(np.float32),
                mhat_top=top, mhat_base=base, sw_top=sw_top, sw_base=sw_base,
                r_top=r_at(top), r_base=r_at(base), resid=r), S


def sample_bands(cs, S, chunk=100):
    """Main-lens base/top per column over samples → (present, median, p5, p95) each."""
    nx, ns = len(cs.x), S.shape[1]
    top = np.full((nx, ns), np.nan); base = np.full((nx, ns), np.nan)
    for i in range(0, ns, chunk):
        t, b = wc.main_lens(cs(S[:, i:i + chunk] / np.log(10)), cs.z)
        top[:, i:i + chunk], base[:, i:i + chunk] = t, b
    import warnings
    out = {}
    for name, Z in (("top", top), ("base", base)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[name] = np.vstack([np.isfinite(Z).mean(axis=1), np.nanmedian(Z, axis=1),
                                   np.nanpercentile(Z, 5, axis=1), np.nanpercentile(Z, 95, axis=1)])
    return out


def run_job(m02, spec, fwd, sch, mesh, cs, truth, real, log):
    out_path = os.path.join(SYN_DATA, "runs", f"{truth}_r{real:02d}.npz")
    if os.path.exists(out_path):
        log(f"skip {truth} r{real} (exists)"); return
    t_job = time.time()
    sd = seeds(truth, real)
    R_true = fwd[f"R_{truth}"]
    rng = np.random.default_rng(sd["noise"])
    sig = np.sqrt((ERR_REL * R_true) ** 2 + ERR_ABS ** 2)
    R = R_true + sig * rng.standard_normal(len(R_true))
    elec = {int(e): (float(x), float(z)) for e, x, z in zip(sch["elec_num"], sch["elec_x"], sch["elec_z"])}
    base_land = {k: sch[k] for k in ("C1", "C2", "P1", "P2")}
    extra = {}
    # noisy R ≤ 0 (possible where R ≈ the 1e-4 Ω floor) cannot enter the log-space inversion;
    # as in field practice such readings are discarded (count recorded)
    keep = R > 0
    extra["n_nonpositive_dropped"] = int((~keep).sum())
    if truth == "T6":
        rc = np.random.default_rng(sd["corrupt"])
        E = np.arange(39, 46)
        asP = (np.isin(sch["P1"], E) | np.isin(sch["P2"], E)) & ~(np.isin(sch["C1"], E) | np.isin(sch["C2"], E))
        cand = np.where(asP)[0]
        bad = rc.choice(cand, int(round(0.4 * len(cand))), replace=False)
        pool = f_outlier_residuals()
        rr = rc.choice(pool, len(bad)) * rc.choice([-1.0, 1.0], len(bad))
        R = R.copy()
        R[bad] = R[bad] * np.exp(rr * (sig[bad] / R_true[bad]))       # multiplicative: sign preserved
        pos = keep.copy()
        land = dict({k: v[pos] for k, v in base_land.items()}, R=R[pos], n=int(pos.sum()), err=np.zeros(int(pos.sum())))
        pre, _ = one_inversion(m02, land, elec, 20.0, mesh, cs, 1)       # Stage-1 λ=20 all-data pass
        r = pre["resid"]; med = np.median(r); mad = np.median(np.abs(r - med))
        keep = pos.copy()
        keep[np.where(pos)[0]] = np.abs(r) <= 3 * 1.4826 * mad
        isbad = np.zeros(len(R), bool); isbad[bad] = True
        extra = dict(t6_n_corrupted=len(bad), t6_n_dropped=int((~keep).sum()),
                     t6_corrupted_dropped=int((~keep & isbad).sum()), t6_clean_dropped=int((~keep & ~isbad).sum()),
                     t6_pre_chi2=pre["chi2"], t6_pre_stop=pre["stop"], t6_pre_iter=pre["n_iter"])
        log(f"{truth} r{real}: corrupted {len(bad)}, Stage-1 dropped {(~keep).sum()} "
            f"({extra['t6_corrupted_dropped']} corrupted, {extra['t6_clean_dropped']} clean)")
    land = dict({k: v[keep] for k, v in base_land.items()}, R=R[keep], n=int(keep.sum()), err=np.zeros(int(keep.sum())))
    lams = [LAM_MAIN] + (LAM_EXTRA if real < N_REAL_LAMBDA else [])
    runs, pooled = {}, []
    for lam in lams:
        info, S = one_inversion(m02, land, elec, lam, mesh, cs, seeds(truth, real, lam)["post"])
        info["bands"] = sample_bands(cs, S)
        runs[lam] = info
        if real < N_REAL_LAMBDA:
            pooled.append(S.astype(np.float32))
        log(f"{truth} r{real} λ={lam:g}: χ²={info['chi2']:.3f} ({info['stop']}, {info['n_iter']} it) "
            f"{info['t_total']:.0f}s")
    save = dict(truth=truth, real=real, n_data=int(keep.sum()), keep=keep, cols=cs.x, **extra)
    for lam, info in runs.items():
        p = f"lam{int(lam)}_"
        for k in ("chi2", "chi2_pg", "n_iter", "stop", "t_inv", "t_total", "n_data", "model", "radius", "P3",
                  "mhat_top", "mhat_base", "sw_top", "sw_base", "r_top", "r_base"):
            save[p + k] = info[k]
        for kind in ("top", "base"):
            save[p + f"band_{kind}"] = info["bands"][kind]
    if pooled:
        SP = np.hstack(pooled).astype(np.float64); del pooled
        save["pooled_P3"] = (SP < np.log(3)).mean(axis=1).astype(np.float32)
        pb = sample_bands(cs, SP)
        save["pooled_band_top"], save["pooled_band_base"] = pb["top"], pb["base"]
    save["t_job"] = time.time() - t_job
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path + ".tmp.npz", **save)
    os.replace(out_path + ".tmp.npz", out_path)
    log(f"done {truth} r{real} in {save['t_job']:.0f}s")


def cmd_worker(args):
    import contextlib, io
    spec = load_spec()
    fwd = np.load(os.path.join(SYN_DATA, "truth_forward.npz"))
    sch = g1_scheme()
    ref = wc.load_run(REF_TAG)
    mesh, _ = wc.rebuild_paradomain(ref)
    cs = wc.ColumnSampler(mesh, COLS)
    with contextlib.redirect_stdout(io.StringIO()):
        m02 = load_02()
    m02.VERBOSE = False
    logf = open(os.path.join(SYN_DATA, "logs", f"worker_{args.worker_id}.log"), "a", encoding="utf-8")
    def log(msg):
        logf.write(time.strftime("%H:%M:%S ") + msg + "\n"); logf.flush()
    for job in args.jobs:
        truth, real = job.split(":")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run_job(m02, spec, fwd, sch, np.asarray(ref["cell_x"]), cs, truth, int(real), log)
        except Exception as e:  # keep going; the job can be rerun (resumable)
            import traceback
            log(f"FAILED {job}: {e}\n{traceback.format_exc()}")
    log("worker finished")


def cmd_run(args):
    os.makedirs(os.path.join(SYN_DATA, "logs"), exist_ok=True)
    jobs = [(t, r) for r in range(N_REAL) for t in TRUTHS]
    todo = [j for j in jobs if not os.path.exists(os.path.join(SYN_DATA, "runs", f"{j[0]}_r{j[1]:02d}.npz"))]
    if args.only:
        todo = [j for j in todo if j[0] in args.only]
    # heavy jobs (5 λ) first, spread round-robin
    todo.sort(key=lambda j: (j[1] >= N_REAL_LAMBDA, j[1], TRUTHS.index(j[0])))
    parts = [todo[i::args.workers] for i in range(args.workers)]
    print(f"[run] {len(todo)} jobs → {args.workers} workers")
    procs = []
    for w, part in enumerate(parts):
        if not part:
            continue
        wd = os.path.join(SYN_DATA, "work", f"w{w}")      # negResp dumps land here, not in the project
        os.makedirs(wd, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "worker", "--worker-id", str(w), "--jobs"] + \
              [f"{t}:{r}" for t, r in part]
        env = dict(os.environ, OMP_NUM_THREADS="2")
        procs.append(subprocess.Popen(cmd, cwd=wd, env=env, stdout=subprocess.DEVNULL,
                                      stderr=open(os.path.join(SYN_DATA, "logs", f"worker_{w}.err"), "a")))
    for p in procs:
        p.wait()
    done = sum(os.path.exists(os.path.join(SYN_DATA, "runs", f"{t}_r{r:02d}.npz")) for t, r in jobs)
    print(f"[run] finished: {done}/{len(jobs)} jobs have results")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def mixture_interval(mus, sigs, q=(0.05, 0.95)):
    """Quantiles of an equal-weight mixture of normals (finite components only)."""
    from scipy.stats import norm
    ok = np.isfinite(mus) & np.isfinite(sigs) & (sigs > 0)
    if not ok.any():
        return np.nan, np.nan
    m, s = mus[ok], sigs[ok]
    grid = np.linspace((m - 5 * s).min(), (m + 5 * s).max(), 4001)
    cdf = norm.cdf((grid[:, None] - m[None]) / s[None]).mean(axis=1)
    return tuple(float(np.interp(qq, cdf, grid)) for qq in q)


def cmd_metrics(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(SYN_FIG, exist_ok=True); os.makedirs(SYN_RES, exist_ok=True)
    fwd = np.load(os.path.join(SYN_DATA, "truth_forward.npz"))
    spec = load_spec()
    ref = wc.load_run(REF_TAG)
    cx, cz = ref["cell_x"], ref["cell_z"]
    Z90 = 1.6449
    lamk = f"lam{int(LAM_MAIN)}_"
    rows_runs, cov_rows, rel = [], [], {"single": [], "pooled": []}
    per_truth = {}
    for t in TRUTHS:
        tb = fwd[f"base_{t}"]
        recs = []
        for r in range(N_REAL):
            p = os.path.join(SYN_DATA, "runs", f"{t}_r{r:02d}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p, allow_pickle=True)
            for lam in [LAM_MAIN] + LAM_EXTRA:
                k = f"lam{int(lam)}_"
                if k + "chi2" in d.files:
                    rows_runs.append((t, r, lam, int(d[k + "n_data"]), float(d[k + "chi2"]), int(d[k + "n_iter"]),
                                      str(d[k + "stop"]), float(d[k + "t_total"])))
            if "t6_pre_chi2" in d.files:
                rows_runs.append((t, r, 20.0, -1, float(d["t6_pre_chi2"]), int(d["t6_pre_iter"]),
                                  "Stage-1 pre-pass: " + str(d["t6_pre_stop"]), np.nan))
            recs.append(d)
            # reliability (single λ = 50 all realisations; pooled for r < 5)
            truth_lt3 = fwd[f"cell_log10_{t}"] < np.log10(3)
            rad = d[k.replace(k, lamk) + "radius"]
            rel["single"].append((d[lamk + "P3"], truth_lt3, rad))
            if "pooled_P3" in d.files:
                rel["pooled"].append((d["pooled_P3"], truth_lt3, rad))
        per_truth[t] = recs
        # coverage per column
        for d in recs:
            r_base = d[lamk + "r_base"]
            b1 = d[lamk + "band_base"]            # (4, ncol): present, median, p5, p95
            mh = d[lamk + "mhat_base"]; sw = d[lamk + "sw_base"]
            pooled = "pooled_band_base" in d.files
            for j, x in enumerate(COLS):
                tr = tb[j]
                grp = ("≤10" if r_base[j] <= 10 else "10–15" if r_base[j] <= 15 else ">15") if np.isfinite(r_base[j]) else "no est."
                row = dict(truth=t, real=int(d["real"]), x=x, grp=grp, true=tr, est=mh[j], sw=sw[j],
                           cov_s=(b1[0, j] >= 0.5) and np.isfinite(tr) and (b1[2, j] <= tr <= b1[3, j]),
                           cov_a=np.isfinite(mh[j]) and np.isfinite(tr) and abs(tr - mh[j]) <= Z90 * sw[j],
                           width_s=b1[3, j] - b1[2, j] if b1[0, j] >= 0.5 else np.nan,
                           width_a=2 * Z90 * sw[j], true_exists=np.isfinite(tr))
                if pooled:
                    pb = d["pooled_band_base"]
                    mus = np.array([d[f"lam{int(l)}_mhat_base"][j] for l in [LAM_MAIN] + LAM_EXTRA])
                    sgs = np.array([d[f"lam{int(l)}_sw_base"][j] for l in [LAM_MAIN] + LAM_EXTRA])
                    lo, hi = mixture_interval(mus, sgs)
                    row.update(cov_ps=(pb[0, j] >= 0.5) and np.isfinite(tr) and (pb[2, j] <= tr <= pb[3, j]),
                               cov_pa=np.isfinite(lo) and np.isfinite(tr) and (lo <= tr <= hi),
                               width_ps=pb[3, j] - pb[2, j] if pb[0, j] >= 0.5 else np.nan, width_pa=hi - lo)
                cov_rows.append(row)

    # ── all runs table ───────────────────────────────────────────────────────
    with open(os.path.join(SYN_RES, "all_runs.csv"), "w", encoding="utf-8") as f:
        f.write("truth,realisation,lambda,n_data,chi2,iterations,stop_reason,runtime_s\n")
        for rr in rows_runs:
            f.write(",".join(str(v) for v in rr) + "\n")

    import collections
    L = ["# Synthetic benchmark — results", ""]
    L += ["## Runs: χ² and stop reason (all runs in all_runs.csv)", "",
          "| truth | λ | runs | χ² median [min, max] | stop reasons |", "|---|---|---|---|---|"]
    g = collections.defaultdict(list)
    for t, r, lam, n, c2, it, st, rt in rows_runs:
        g[(t, lam if not st.startswith("Stage") else "pre20")].append((c2, st))
    for (t, lam), v in sorted(g.items(), key=lambda kv: (TRUTHS.index(kv[0][0]),
                                                         -1.0 if kv[0][1] == "pre20" else float(kv[0][1]))):
        c = np.array([a for a, _ in v]); sts = collections.Counter(s for _, s in v)
        L.append(f"| {t} | {lam} | {len(v)} | {np.median(c):.3f} [{c.min():.3f}, {c.max():.3f}] | "
                 + ", ".join(f"{k}: {n}" for k, n in sts.items()) + " |")

    # ── coverage table ───────────────────────────────────────────────────────
    L += ["", "## Lens-base coverage (90 % bands), bias and width", "",
          "Columns x = 100–450 m every 10 m; grouped by the resolution radius at the ESTIMATED (m̂, λ=50) base "
          "crossing. Coverage = share of (column, realisation) pairs where the 90 % band contains the TRUE base. "
          "Sampled = P5–P95 of posterior samples (requires a crossing in ≥ 50 % of samples, otherwise a miss); "
          "analytic = m̂ crossing ± 1.645·σ_z with σ_z from wᵀA⁻¹w. Pooled = λ ∈ {5,10,20,50,100} mixture "
          "(realisations 0–4). Bias = median(estimated − true) of the m̂ crossing (m, + = too shallow). "
          "Width = median 90 % sampled width (m). σ-inflation = P90(|m̂ − true| / σ_z) / 1.645: the factor by which "
          "the analytic σ_z would have to be multiplied to reach 90 % coverage.", "",
          "| truth | r at est. base | n | cov. sampled | cov. analytic | cov. pooled sampled | cov. pooled analytic | "
          "bias (m) | width (m) | pooled width (m) | σ-inflation |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for t in TRUTHS:
        for grp in ("≤10", "10–15", ">15", "no est."):
            rs = [c for c in cov_rows if c["truth"] == t and c["grp"] == grp and c["true_exists"]]
            if not rs:
                continue
            f = lambda k: np.mean([c[k] for c in rs if k in c]) if any(k in c for c in rs) else np.nan
            npool = sum(1 for c in rs if "cov_ps" in c)
            bias = np.nanmedian([c["est"] - c["true"] for c in rs]) if grp != "no est." else np.nan
            ws = np.nanmedian([c["width_s"] for c in rs]) if rs else np.nan
            wps = np.nanmedian([c.get("width_ps", np.nan) for c in rs]) if npool else np.nan
            zs = np.array([abs(c["est"] - c["true"]) / c["sw"] for c in rs
                           if np.isfinite(c["est"]) and np.isfinite(c["sw"]) and c["sw"] > 0])
            infl = np.percentile(zs, 90) / 1.6449 if zs.size else np.nan
            summary[(t, grp)] = dict(n=len(rs), cs=f("cov_s"), ca=f("cov_a"), cps=f("cov_ps"), cpa=f("cov_pa"),
                                     bias=bias, w=ws, wp=wps, infl=infl)
            L.append(f"| {t} | {grp} | {len(rs)} | {100*f('cov_s'):.0f}% | {100*f('cov_a'):.0f}% | "
                     + (f"{100*f('cov_ps'):.0f}% (n={npool})" if npool else "—") + " | "
                     + (f"{100*f('cov_pa'):.0f}%" if npool else "—") + f" | {bias:+.2f} | {ws:.2f} | "
                     + (f"{wps:.2f}" if npool else "—") + f" | {infl:.1f}× |")
    # all truths pooled
    for grp in ("≤10", "10–15"):
        rs = [c for c in cov_rows if c["grp"] == grp and c["true_exists"]]
        if rs:
            L.append(f"| **all** | {grp} | {len(rs)} | **{100*np.mean([c['cov_s'] for c in rs]):.0f}%** | "
                     f"**{100*np.mean([c['cov_a'] for c in rs]):.0f}%** | "
                     f"{100*np.mean([c['cov_ps'] for c in rs if 'cov_ps' in c]):.0f}% | "
                     f"{100*np.mean([c['cov_pa'] for c in rs if 'cov_pa' in c]):.0f}% | "
                     f"{np.nanmedian([c['est'] - c['true'] for c in rs]):+.2f} | "
                     f"{np.nanmedian([c['width_s'] for c in rs]):.2f} | "
                     f"{np.nanmedian([c.get('width_ps', np.nan) for c in rs]):.2f} | "
                     f"{np.nanpercentile([abs(c['est'] - c['true']) / c['sw'] for c in rs if np.isfinite(c['est']) and c['sw'] > 0], 90) / 1.6449:.1f}× |")

    # ── diagnosis: is the NOISE part calibrated? (scatter across realisations vs σ_z) ──
    L += ["", "## Diagnosis — random scatter vs predicted σ_z, and bias-corrected coverage", "",
          "Per truth model and column: empirical s.d. of the m̂ base across the 20 noise realisations (what the "
          "posterior should describe if it only had to capture noise) vs the median predicted σ_z (wᵀA⁻¹w). "
          "Bias-corrected coverage = coverage of the analytic band after subtracting the column's median bias "
          "(an oracle correction, only to show how much of the failure is bias).", "",
          "| truth | r at est. base | columns | median s.d.(m̂) / σ_z | analytic coverage | bias-corrected coverage |",
          "|---|---|---|---|---|---|"]
    for t in TRUTHS:
        for grp in ("≤10", "10–15"):
            ratios, hit, hitc = [], [], []
            for x in COLS:
                rs = [c for c in cov_rows if c["truth"] == t and c["x"] == x and np.isfinite(c["est"])
                      and np.isfinite(c["true"]) and np.isfinite(c["sw"]) and c["sw"] > 0]
                rg = [c for c in rs if c["grp"] == grp]
                if len(rs) < 10 or not rg:
                    continue
                est = np.array([c["est"] for c in rs]); mb = np.median(est - np.array([c["true"] for c in rs]))
                ratios.append(np.std(est, ddof=1) / np.median([c["sw"] for c in rs]))
                hit += [abs(c["est"] - c["true"]) <= Z90 * c["sw"] for c in rg]
                hitc += [abs(c["est"] - mb - c["true"]) <= Z90 * c["sw"] for c in rg]
            if ratios:
                L.append(f"| {t} | {grp} | {len(ratios)} | {np.median(ratios):.2f} | {100*np.mean(hit):.0f}% | "
                         f"{100*np.mean(hitc):.0f}% |")

    # ── T2 bias vs true depth ────────────────────────────────────────────────
    L += ["", "## Bias of the m̂ lens base against true depth (T1 and flat-base T2a/b/c; x 100–360 m)", "",
          "| truth | true base (m) | columns×realisations | est. present | median bias (m) | P10 / P90 bias | "
          "share with r ≤ 10 m at est. |", "|---|---|---|---|---|---|---|"]
    for t in ("T1", "T2a", "T2b", "T2c"):
        rs = [c for c in cov_rows if c["truth"] == t and c["x"] < 370]
        if not rs:
            continue
        b = np.array([c["est"] - c["true"] for c in rs]); pres = np.isfinite(b)
        L.append(f"| {t} | {np.nanmedian([c['true'] for c in rs]):.1f} | {len(rs)} | {100*pres.mean():.0f}% | "
                 f"{np.nanmedian(b):+.2f} | {np.nanpercentile(b, 10):+.2f} / {np.nanpercentile(b, 90):+.2f} | "
                 f"{100*np.mean([c['grp'] == '≤10' for c in rs]):.0f}% |")

    # ── reliability ──────────────────────────────────────────────────────────
    bins = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    L += ["", "## Reliability of P(ρ < 3 Ω·m)", "",
          "Cells binned by predicted P; observed = share whose TRUE cell-averaged (geometric-mean) resistivity < 3 Ω·m. "
          "All truth models pooled.", ""]
    for ax, (grpname, lo, hi) in zip(axes, (("r ≤ 10 m", 0, 10), ("10 < r ≤ 15 m", 10, 15))):
        L += [f"**{grpname}**", "", "| P bin | single λ: n | observed | pooled λ: n | observed |", "|---|---|---|---|---|"]
        curves = {}
        for mode in ("single", "pooled"):
            P = np.concatenate([a for a, _, rr in rel[mode]]); T = np.concatenate([b for _, b, rr in rel[mode]])
            R_ = np.concatenate([rr for _, _, rr in rel[mode]])
            sel = (R_ > lo) & (R_ <= hi)
            idx = np.clip(np.digitize(P[sel], bins) - 1, 0, 9)
            obs = [T[sel][idx == i].mean() if (idx == i).any() else np.nan for i in range(10)]
            n = [(idx == i).sum() for i in range(10)]
            mp = [P[sel][idx == i].mean() if (idx == i).any() else np.nan for i in range(10)]
            curves[mode] = (mp, obs, n)
        for i in range(10):
            L.append(f"| {bins[i]:.1f}–{bins[i+1]:.1f} | {curves['single'][2][i]} | {curves['single'][1][i]:.3f} | "
                     f"{curves['pooled'][2][i]} | {curves['pooled'][1][i]:.3f} |")
        L.append("")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect reliability")
        for mode, col, mk in (("single", "#1F77B4", "o"), ("pooled", "#D62728", "s")):
            mp, obs, n = curves[mode]
            ax.plot(mp, obs, "-" + mk, color=col, label=f"{'single λ=50' if mode == 'single' else 'pooled λ mixture'}")
            for x_, y_, nn in zip(mp, obs, n):
                if np.isfinite(x_) and nn:
                    ax.annotate(f"{nn}", (x_, y_), textcoords="offset points", xytext=(4, -10), fontsize=6.5, color=col)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("predicted P(ρ < 3 Ω·m) (bin mean)"); ax.set_ylabel("observed frequency of true ρ < 3 Ω·m")
        ax.set_title(f"Reliability — cells with {grpname}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        wc.white_legend(ax, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(SYN_FIG, "reliability_diagram.png"), dpi=170)
    plt.close(fig)

    # ── T4, T5, T6 ───────────────────────────────────────────────────────────
    L += ["## T4 — clay (5 Ω·m) flagged as saline?", ""]
    c = CLAY
    inclay = (cx >= c["x0"]) & (cx <= c["x1"]) & (cz >= c["z0"]) & (cz <= c["z1"])
    for mode, key in (("single λ=50", lamk + "P3"), ("pooled", "pooled_P3")):
        P = np.array([d[key][inclay] for d in per_truth["T4"] if key in d.files])
        Rr = np.array([d[lamk + "radius"][inclay] for d in per_truth["T4"] if key in d.files])
        if P.size:
            L.append(f"- {mode}: {inclay.sum()} clay cells × {len(P)} realisations — mean P(ρ<3) {P.mean():.3f}; "
                     f"share of cells with P > 0.5: {100*np.mean(P > 0.5):.1f}%, P > 0.9: {100*np.mean(P > 0.9):.1f}% "
                     f"(r ≤ 10 m cells: {100*np.mean(Rr <= 10):.0f}% of clay cells; mean P there "
                     f"{P[Rr <= 10].mean() if (Rr <= 10).any() else np.nan:.3f})")
    Pt1 = np.array([d[lamk + "P3"][inclay] for d in per_truth["T1"]])
    L.append(f"- same cells in T1 (no clay; true ρ = fresh aquifer / lens edge): mean P(ρ<3) {Pt1.mean():.3f}")
    L += ["", "## T5 — false-positive discharge zone", "",
          "Test cells = cells (x 370–470 m) that are truly NON-saline in T1 (discharge zone) but truly SALINE in T5 "
          "(true cell-averaged ρ vs 3 Ω·m). A cell is 'confidently non-saline' if P(ρ<3) < 0.1; a realisation "
          "'detects a discharge zone' if ≥ 25 % of the test cells with r ≤ 15 m are confidently non-saline. "
          "In T1 this is a true positive, in T5 a false positive.", ""]
    box = ((cx >= X_DIS0) & (cx < X_DIS1) & (fwd["cell_log10_T1"] >= np.log10(3))
           & (fwd["cell_log10_T5"] < np.log10(3)))
    L.append(f"- test cells: {box.sum()} (x {cx[box].min():.0f}–{cx[box].max():.0f} m, "
             f"z {cz[box].min():.1f} … {cz[box].max():.1f} mTAW)")
    for t in ("T1", "T5"):
        for mode, key in (("single λ=50", lamk + "P3"), ("pooled", "pooled_P3")):
            vals, flags = [], []
            for d in per_truth[t]:
                if key not in d.files:
                    continue
                ok = box & (d[lamk + "radius"] <= 15)
                share = np.mean(d[key][ok] < 0.1) if ok.any() else np.nan
                vals.append(share); flags.append(share >= 0.25)
            if vals:
                L.append(f"- {t}, {mode}: share of box cells with P<0.1 median {np.nanmedian(vals):.2f}; "
                         f"'discharge detected' in {sum(flags)}/{len(flags)} realisations"
                         + ("  ← FALSE POSITIVE rate" if t == "T5" else "  ← true-positive rate"))
    L += ["", "## T6 — charge-up outliers + Stage-1 rule", ""]
    t6 = per_truth["T6"]; t1 = {int(d["real"]): d for d in per_truth["T1"]}
    if t6:
        nc = np.array([d["t6_n_corrupted"] for d in t6]); nd = np.array([d["t6_n_dropped"] for d in t6])
        ncd = np.array([d["t6_corrupted_dropped"] for d in t6]); ncl = np.array([d["t6_clean_dropped"] for d in t6])
        L.append(f"- corrupted per realisation {int(np.median(nc))}; Stage-1 rule dropped median {int(np.median(nd))} "
                 f"(of which corrupted {int(np.median(ncd))} = {100*np.median(ncd/nc):.0f}% of the corrupted; "
                 f"clean {int(np.median(ncl))})")
        sh = []
        for d in t6:
            r = int(d["real"])
            if r in t1:
                sh.append(d[lamk + "mhat_base"] - t1[r][lamk + "mhat_base"])
        sh = np.array(sh)
        L.append(f"- lens-base shift T6 − T1 (m̂, λ=50, paired noise): median {np.nanmedian(sh):+.2f} m, "
                 f"P10/P90 {np.nanpercentile(sh, 10):+.2f}/{np.nanpercentile(sh, 90):+.2f} m, "
                 f"max |shift| {np.nanmax(np.abs(sh)):.2f} m; per x: " + ", ".join(
                     f"{x:g}: {np.nanmedian(sh[:, j]):+.2f}" for j, x in enumerate(COLS) if x in (150, 200, 250, 300, 350)))
    rep = os.path.join(SYN_RES, "benchmark_report.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    import pickle
    with open(os.path.join(SYN_DATA, "coverage_rows.pkl"), "wb") as f:
        pickle.dump(cov_rows, f)
    print("\n".join(L))
    print(f"[saved] {rep}\n[saved] {os.path.join(SYN_FIG, 'reliability_diagram.png')}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Westhoek synthetic benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spec"); sub.add_parser("build"); sub.add_parser("metrics")
    pr = sub.add_parser("run"); pr.add_argument("--workers", type=int, default=3)
    pr.add_argument("--only", nargs="*", default=None)
    pw = sub.add_parser("worker"); pw.add_argument("--worker-id", type=int, required=True)
    pw.add_argument("--jobs", nargs="+", required=True)
    args = ap.parse_args()
    dict(spec=cmd_spec, build=cmd_build, run=cmd_run, worker=cmd_worker, metrics=cmd_metrics)[args.cmd](args)


if __name__ == "__main__":
    main()
