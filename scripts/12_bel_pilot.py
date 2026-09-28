"""
12_bel_pilot.py
===============
Westhoek ERT — Stage 5b pilot: Bayesian Evidential Learning (BEL) prior and prior falsification.

Subcommands
  spec       write results/bel/prior_spec.md + data/processed/bel/prior_spec.json (BEFORE any sampling);
             the dune water-table range is read from the resistive cap of g1_lam50
  timing     choose the forward mesh (coarsest candidate whose homogeneous-half-space R stays within 1 % of
             the synthetic forward mesh), time 10 prior forward runs, project the cost of 3000 samples
  sample     draw N prior samples (default 300), forward-model on the chosen mesh, add 3 % + 1e-4 Ω noise
             (parallel, resumable)
  falsify    PCA of log10 R of the prior-predicted data; Mahalanobis (classical + robust MCD) and Q (residual)
             statistics of the field data (G1 set) and T1 / T1h synthetic data against the prior cloud

Seeds: parameters of sample i: numpy default_rng([PRIOR_SEED, i]); noise of sample i: default_rng([NOISE_SEED, i]);
timing samples use i = 100000 + k. MCD: default_rng(MCD_SEED). All recorded in prior_spec.json.
Outputs only in data/processed/bel/, figures/bel/, results/bel/.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import westhoek_common as wc  # noqa: E402

BEL_DATA = os.path.join(wc.PROCESSED_DIR, "bel")
BEL_FIG = os.path.join(wc.FIG_DIR, "bel")
BEL_RES = os.path.join(wc.RESULTS_DIR, "bel")
SPEC_JSON = os.path.join(BEL_DATA, "prior_spec.json")
PRIOR_SEED, NOISE_SEED, MCD_SEED = 55_000, 66_000, 77_000
CTRL_X = [100.0, 170.0, 240.0, 310.0, 380.0, 450.0]
ERR_REL, ERR_ABS = 0.03, 1e-4


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Prior
# ─────────────────────────────────────────────────────────────────────────────
def draw(spec, i):
    """One prior sample θ (dict), seeded by (PRIOR_SEED, i)."""
    rng = np.random.default_rng([PRIOR_SEED, i])
    P = spec["params"]
    U = lambda k: float(rng.uniform(*P[k]["range"]))
    LU = lambda k: float(10 ** rng.uniform(*np.log10(P[k]["range"])))
    th = dict(x_L=U("x_L"), x_S=U("x_S"), tau=U("tau"), z_wt=U("z_wt"), z_deep=U("z_deep"),
              base_depth=[float(rng.uniform(*P["base_depth"]["range"])) for _ in CTRL_X])
    for k in ("rho_cap", "rho_aquifer", "rho_lens", "rho_below", "rho_discharge", "rho_deep"):
        th[k] = LU(k)
    th["dis_present"] = bool(rng.uniform() < P["dis_present"]["p"])
    th["dis_x0"] = U("dis_x0"); th["dis_width"] = U("dis_width")
    th["dis_skin"] = U("dis_skin"); th["dis_bottom"] = U("dis_bottom")
    th["clay_present"] = bool(rng.uniform() < P["clay_present"]["p"])
    th["clay_top"] = U("clay_top"); th["clay_thick"] = U("clay_thick"); th["rho_clay"] = LU("rho_clay")
    return th


class PriorModel:
    """log10 ρ(x, z) of one prior sample θ."""

    def __init__(self, th):
        self.th = th
        self.ex, self.ez = wc.land_surface()
        ctrl_z = np.interp(CTRL_X, self.ex, self.ez) - np.array(th["base_depth"])
        self.base_f = PchipInterpolator(CTRL_X, ctrl_z, extrapolate=False)

    def surface(self, x):
        return np.interp(x, self.ex, self.ez)

    def base(self, x):
        b = self.base_f(np.clip(x, CTRL_X[0], CTRL_X[-1]))
        return np.asarray(b, float)

    def log10rho(self, x, z):
        th = self.th
        x = np.asarray(x, float); z = np.asarray(z, float)
        s = self.surface(x)
        L = lambda k: np.log10(th[k])
        out = np.full(x.shape, L("rho_below"))
        dune = x < th["x_L"]
        out[dune] = np.where(z[dune] > th["z_wt"], L("rho_cap"), L("rho_aquifer"))
        if th["clay_present"]:
            c = dune & (z <= th["clay_top"]) & (z > th["clay_top"] - th["clay_thick"])
            out[c] = L("rho_clay")
        lens_x = (x >= th["x_L"]) & (x <= th["x_S"])
        b = self.base(x)
        d = z - b                                                    # + above the base
        if th["tau"] > 0:
            f = np.clip((d + th["tau"] / 2) / th["tau"], 0, 1)       # 0 below … 1 above
        else:
            f = (d > 0).astype(float)
        out[lens_x] = (L("rho_below") + f * (L("rho_lens") - L("rho_below")))[lens_x]
        if th["dis_present"]:
            x0, x1 = th["dis_x0"], min(th["dis_x0"] + th["dis_width"], 505.0)
            dz = (x >= x0) & (x <= x1)
            skin = dz & (z > s - th["dis_skin"])
            body = dz & (z <= s - th["dis_skin"]) & (z > th["dis_bottom"])
            out[skin] = L("rho_lens")
            out[body] = L("rho_discharge")
        out[z <= th["z_deep"]] = L("rho_deep")
        return out


# ─────────────────────────────────────────────────────────────────────────────
def cmd_spec(args):
    os.makedirs(BEL_DATA, exist_ok=True); os.makedirs(BEL_RES, exist_ok=True)
    ref = wc.load_run("g1_lam50")
    s4 = json.load(open(os.path.join(wc.PROCESSED_DIR, "synthetic", "truth_spec.json"), encoding="utf-8"))
    med = s4["zone_medians"]
    # water table under the dunes: log-midpoint between cap and aquifer medians in g1_lam50, x 10–50 m
    # (x < 10 m: start of the electrode line, poorly constrained; x > 50 m: dune toe where the cap thins
    #  against the lens edge — both give readouts that are not the water table)
    mesh, _ = wc.rebuild_paradomain(ref)
    xs = np.arange(10.0, 51.0, 2.0)
    cs = wc.ColumnSampler(mesh, xs, dz=0.1)
    V = cs(np.log10(ref["model"]))
    thr = 0.5 * (np.log10(med["cap"]) + np.log10(med["fresh"]))
    zwt = []
    for j in range(len(xs)):
        c = wc.crossings(V[:, j], cs.z, level=thr)
        first_drop = [z for k, z in c if k == "top"]            # ≥thr above, <thr below
        if first_drop:
            zwt.append(first_drop[0])
    zwt = np.array(zwt)
    wt_lo, wt_hi = float(np.floor(zwt.min() - 1.0)), float(np.ceil(zwt.max() + 1.0))
    P = {
        "x_L": dict(range=[50.0, 130.0], dist="uniform", unit="m",
                    why="landward lens edge; g1_lam50 3 Ω·m contour reaches the surface at x ≈ 65–95 m; ±30 m around it"),
        "x_S": dict(range=[300.0, 505.0], dist="uniform", unit="m",
                    why="seaward end of the saline lens; ≥ 505 m (end of the land line) = lens continues to the low-water line "
                        "(as in g1_lam50 and T1). Values < 505 put 'water below the lens' at the surface seaward of x_S. CONFIRM"),
        "base_depth": dict(range=[3.0, 30.0], dist="uniform (each of 6 control points)", unit="m below surface",
                           why=f"control points at x = {CTRL_X}; PCHIP curve; g1_lam50 base 8–20 m below surface, "
                               "T2 tests went to 37 m; 3–30 m spans both"),
        "tau": dict(range=[0.0, 5.0], dist="uniform", unit="m", why="log-linear transition centred on the base (as T3)"),
        "rho_cap": dict(range=[50.0, 1000.0], dist="log-uniform", unit="Ω·m",
                        why=f"g1_lam50 cap median {med['cap']:.0f}; ×/÷ ≥ 4; dry dune sand"),
        "rho_aquifer": dict(range=[5.0, 60.0], dist="log-uniform", unit="Ω·m",
                            why=f"median {med['fresh']:.1f}; T1h {2*med['fresh']:.1f}; ≥ ×/÷ 2 around the median, "
                                "upper end allows fresher water than the smoothed inversion shows"),
        "rho_lens": dict(range=[0.4, 3.0], dist="log-uniform", unit="Ω·m",
                         why=f"median {med['lens']:.2f}; T1h 1.0; seawater-saturated sand ≈ 0.5–2 Ω·m. Upper end capped at "
                             "3 Ω·m so the lens is saline by construction — CONFIRM (factor 2 above the median is 2.84)"),
        "rho_below": dict(range=[2.0, 20.0], dist="log-uniform", unit="Ω·m",
                          why=f"median {med['brackish']:.2f}; ≥ ×/÷ 2.5; brackish to fresh water below the lens"),
        "rho_discharge": dict(range=[2.5, 40.0], dist="log-uniform", unit="Ω·m",
                              why=f"median {med['discharge']:.2f} (smoothed; equals the brackish median); upper end allows "
                                  "fresh discharge water — CONFIRM"),
        "rho_deep": dict(range=[3.0, 50.0], dist="log-uniform", unit="Ω·m",
                         why=f"median {med['deep']:.2f}, P75 of the deep g1_lam50 cells ≈ 23; widened to 50 (poorly resolved)"),
        "z_deep": dict(range=[-45.0, -20.0], dist="uniform", unit="mTAW",
                       why="top of the deep unit; T1 used −30; below the resolved depth, so wide"),
        "z_wt": dict(range=[wt_lo, wt_hi], dist="uniform", unit="mTAW",
                     why=f"dune water table read from g1_lam50: crossing of the cap/aquifer log-midpoint "
                         f"({10**thr:.0f} Ω·m) at x 10–50 m (line start and dune toe excluded) gives {zwt.min():.2f} … {zwt.max():.2f} mTAW "
                         f"(median {np.median(zwt):.2f}); ±1 m"),
        "dis_present": dict(p=0.5, dist="Bernoulli", why="discharge zone present or absent (T5 showed ERT cannot tell)"),
        "dis_x0": dict(range=[330.0, 420.0], dist="uniform", unit="m", why="g1_lam50 fresher zone starts x ≈ 370–380"),
        "dis_width": dict(range=[30.0, 120.0], dist="uniform", unit="m", why="g1_lam50 fresher zone ≈ 90 m wide; end ≤ 505 m"),
        "dis_skin": dict(range=[0.5, 5.0], dist="uniform", unit="m", why="saline skin above the discharge zone; g1_lam50 1–5 m"),
        "dis_bottom": dict(range=[-25.0, -8.0], dist="uniform", unit="mTAW", why="T1 used −15"),
        "clay_present": dict(p=0.5, dist="Bernoulli",
                             why="dune conductive layer (clay OR saltwater — indistinguishable for ERT)"),
        "clay_top": dict(range=[-15.0, -2.0], dist="uniform", unit="mTAW", why="T4 used −5; within resolved depth"),
        "clay_thick": dict(range=[1.0, 6.0], dist="uniform", unit="m", why="T4 used 3 m"),
        "rho_clay": dict(range=[1.0, 6.0], dist="log-uniform", unit="Ω·m", why="as requested"),
    }
    spec = dict(params=P, ctrl_x=CTRL_X, seeds=dict(prior=f"default_rng([{PRIOR_SEED}, i])",
                noise=f"default_rng([{NOISE_SEED}, i])", timing="i = 100000 + k", mcd=MCD_SEED),
                noise=dict(rel=ERR_REL, abs_ohm=ERR_ABS), water_table_readout=dict(x=list(map(float, xs)),
                z=list(map(float, zwt))), g1_zone_medians=med)
    with open(SPEC_JSON, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    Lm = ["# BEL prior specification (Stage 5b pilot)", "",
          "Written before any prior sample was drawn. Machine-readable: `data/processed/bel/prior_spec.json`.", "",
          "## Geometry", "",
          "- Land surface = the 102 land electrodes; section down to the forward-mesh depth (60 m).",
          f"- Dunes (x < x_L): dry cap above the water table z_wt, dune aquifer below, down to z_deep; optional conductive "
          f"layer (clay or saltwater) for x < x_L between clay_top − clay_thick and clay_top.",
          f"- Lens (x_L ≤ x ≤ x_S): from the surface down to the base curve (PCHIP through depths below surface at x = "
          f"{', '.join(f'{x:g}' for x in CTRL_X)} m, held constant beyond the end points), with a log-linear transition "
          "of thickness τ centred on the base into the 'water below the lens'.",
          "- Seaward of x_S and below the lens: 'water below the lens' down to z_deep.",
          "- Discharge zone (if present): x_0 … min(x_0 + width, 505) m, a saline skin (lens resistivity) of thickness "
          "`dis_skin` at the surface, discharge-zone water from there down to `dis_bottom`.",
          "- Below z_deep: deep unit everywhere.", "",
          "## Parameters", "", "| parameter | distribution | range | unit | justification |", "|---|---|---|---|---|"]
    for k, v in P.items():
        rng_s = f"{v['range'][0]:g} – {v['range'][1]:g}" if "range" in v else f"p = {v['p']}"
        Lm.append(f"| {k} | {v['dist']} | {rng_s} | {v.get('unit', '')} | {v['why']} |")
    Lm += ["", "## Please confirm", "",
           "1. **rho_lens upper bound 3.0 Ω·m** — keeps the lens saline by definition; a looser bound (e.g. 5) would let "
           "the prior contain 'lens' bodies that are not saline by the 3 Ω·m convention.",
           "2. **x_S semantics** — x_S ≥ 505 m means the lens runs to the end of the land line (field-like); smaller values "
           "end it earlier with brackish water at the surface seaward of it.",
           "3. **rho_discharge up to 40 Ω·m** — much fresher than anything the smoothed inversion shows (it smears the zone).",
           "4. **Depths of the base control points are below the local surface** (not elevations).",
           "5. **Noise** — 3 % + 1e-4 Ω Gaussian on R, as in Stage 4.",
           f"6. **Water table** {wt_lo:g} … {wt_hi:g} mTAW from the g1_lam50 cap (single inversion, smoothed; ±1 m added).",
           "", "## Seeds", "", f"- parameters of sample i: default_rng([{PRIOR_SEED}, i]); noise: default_rng([{NOISE_SEED}, i]); "
           f"timing samples i = 100000 + k; robust covariance (MCD): default_rng({MCD_SEED})."]
    with open(os.path.join(BEL_RES, "prior_spec.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(Lm) + "\n")
    print("\n".join(Lm))


# ─────────────────────────────────────────────────────────────────────────────
MESH_CANDIDATES = {
    "fine (synthetic fwd)": dict(paraDX=0.2, paraDepth=60.0, paraMaxCellSize=5.0, quality=34.0),
    "medium": dict(paraDX=0.33, paraDepth=60.0, paraMaxCellSize=15.0, quality=33.0),
    "coarse": dict(paraDX=0.5, paraDepth=60.0, paraMaxCellSize=40.0, quality=32.0),
    "coarsest": dict(paraDX=0.5, paraDepth=60.0, paraMaxCellSize=0.0, quality=30.0),
}


RETRIES = {"n": 0}


def simulate_R(mesh, res, n_expected, tries=3):
    """
    ert.simulate on a FRESH scheme (k = 1 → ρa = R). pyGIMLi 1.5.5 occasionally returned an
    empty result for an unchanged call (seen on the 17 684-cell mesh); retry up to `tries`
    times, count retries, and fail loudly rather than return a gap.
    """
    from pygimli.physics import ert
    for k in range(tries):
        out = ert.simulate(mesh, scheme=build_scheme(), res=res, verbose=False)
        R = np.array(out["rhoa"])
        if len(R) == n_expected:
            return R
        RETRIES["n"] += 1
    raise RuntimeError(f"simulate returned {len(R)} values for {n_expected} quadrupoles after {tries} tries")


def forward(mesh, scheme, th):
    cx = np.array([c.center().x() for c in mesh.cells()]); cz = np.array([c.center().y() for c in mesh.cells()])
    res = 10 ** PriorModel(th).log10rho(cx, cz)
    return simulate_R(mesh, res, scheme.size())


def build_scheme():
    sb = _load("sb10", "10_synthetic_benchmark.py")
    sch = sb.g1_scheme()
    scheme = sb.make_scheme(sch)
    scheme["k"] = np.ones(scheme.size())
    return scheme


def cmd_timing(args):
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    spec = json.load(open(SPEC_JSON, encoding="utf-8"))
    scheme = build_scheme()
    res = {}
    for name, kw in MESH_CANDIDATES.items():
        print(f"[timing] building {name} {kw}", flush=True)
        m = mt.createParaMesh(scheme.sensors(), **kw)
        print(f"[timing]   {m.cellCount()} cells — homogeneous forward …", flush=True)
        t = time.time()
        R = simulate_R(m, np.full(m.cellCount(), 10.0), scheme.size())
        print(f"[timing]   {len(R)} values, {time.time()-t:.1f}s (retries so far {RETRIES['n']})", flush=True)
        res[name] = dict(mesh=m, R=R, cells=m.cellCount(), t_hom=time.time() - t)
    Rf = res["fine (synthetic fwd)"]["R"]
    L = ["# B2 — forward mesh choice and timing", "",
         "Criterion: homogeneous 10 Ω·m half-space, R on the candidate vs the synthetic forward mesh (= geometric factors) "
         "within 1 % for every G1 quadrupole.", "",
         "| mesh | settings | cells | max |ΔR|/R vs fine | P95 | one homogeneous forward (s) |", "|---|---|---|---|---|---|"]
    ok = []
    for name, r in res.items():
        dev = np.abs(r["R"] / Rf - 1)
        L.append(f"| {name} | {MESH_CANDIDATES[name]} | {r['cells']} | {100*dev.max():.2f}% | {100*np.percentile(dev, 95):.2f}% "
                 f"| {r['t_hom']:.1f} |")
        if dev.max() <= 0.01:
            ok.append(name)
    chosen = min(ok, key=lambda n: res[n]["cells"])
    mesh = res[chosen]["mesh"]
    times = []
    for k in range(10):
        th = draw(spec, 100000 + k)
        t = time.time(); forward(mesh, scheme, th); times.append(time.time() - t)
    t_med = float(np.median(times))
    n_cpu = os.cpu_count() or 1
    workers = 4
    proj = 3000 * t_med
    L += ["", f"Chosen: **{chosen}** ({res[chosen]['cells']} cells) — the coarsest candidate within 1 %.", "",
          f"10 prior forward runs (single process, machine otherwise idle): median {t_med:.1f} s, "
          f"range {min(times):.1f}–{max(times):.1f} s.", "",
          f"Projection for 3000 prior samples: {proj/3600:.1f} h in one process; with {workers} parallel workers "
          f"(each pyGIMLi forward uses 2 threads; {n_cpu} logical CPUs) ≈ {proj/3600/workers*1.6:.1f} h, assuming the "
          f"≈ 1.6× per-run slowdown under 4-way load seen in Stage 4 (≈ {proj/3600/workers:.1f} h if it scaled perfectly). "
          f"Storage: 3000 × 1974 × 2 arrays × 8 B ≈ {3000*1974*2*8/1e6:.0f} MB.", "",
          f"Empty-result retries during this timing run: {RETRIES['n']} (pyGIMLi 1.5.5 ert.simulate occasionally "
          "returned an empty container for an unchanged call on the 17 684-cell mesh; every call is now retried and "
          "retries are counted)."]
    json.dump(dict(chosen=chosen, settings=MESH_CANDIDATES[chosen], t_median=t_med, times=times),
              open(os.path.join(BEL_DATA, "forward_mesh_choice.json"), "w"), indent=1)
    with open(os.path.join(BEL_RES, "b2_timing.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
def cmd_worker(args):
    import pygimli.meshtools as mt
    spec = json.load(open(SPEC_JSON, encoding="utf-8"))
    choice = json.load(open(os.path.join(BEL_DATA, "forward_mesh_choice.json")))
    scheme = build_scheme()
    mesh = mt.createParaMesh(scheme.sensors(), **choice["settings"])
    logf = open(os.path.join(BEL_DATA, "logs", f"sample_worker_{args.worker_id}.log"), "a", encoding="utf-8")
    for i in args.ids:
        out = os.path.join(BEL_DATA, "prior", f"s{i:05d}.npz")
        if os.path.exists(out):
            continue
        try:
            th = draw(spec, i)
            t = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                R = forward(mesh, scheme, th)
            rng = np.random.default_rng([NOISE_SEED, i])
            Rn = R + np.sqrt((ERR_REL * R) ** 2 + ERR_ABS ** 2) * rng.standard_normal(len(R))
            np.savez(out + ".tmp.npz", R=R, R_noisy=Rn, theta=json.dumps(th))
            os.replace(out + ".tmp.npz", out)
            logf.write(f"{time.strftime('%H:%M:%S')} s{i} {time.time()-t:.1f}s\n"); logf.flush()
        except Exception as e:
            import traceback
            logf.write(f"FAILED s{i}: {e}\n{traceback.format_exc()}\n"); logf.flush()
    logf.write("worker finished\n")


def cmd_sample(args):
    os.makedirs(os.path.join(BEL_DATA, "prior"), exist_ok=True); os.makedirs(os.path.join(BEL_DATA, "logs"), exist_ok=True)
    ids = [i for i in range(args.n_samples) if not os.path.exists(os.path.join(BEL_DATA, "prior", f"s{i:05d}.npz"))]
    parts = [ids[k::args.workers] for k in range(args.workers)]
    procs = []
    for w, part in enumerate(parts):
        if not part:
            continue
        wd = os.path.join(BEL_DATA, "work", f"w{w}"); os.makedirs(wd, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "worker", "--worker-id", str(w), "--ids"] + [str(i) for i in part]
        procs.append(subprocess.Popen(cmd, cwd=wd, env=dict(os.environ, OMP_NUM_THREADS="2"), stdout=subprocess.DEVNULL,
                                      stderr=open(os.path.join(BEL_DATA, "logs", f"sample_worker_{w}.err"), "a")))
    for p in procs:
        p.wait()
    n = len([f for f in os.listdir(os.path.join(BEL_DATA, "prior")) if f.endswith(".npz") and "tmp" not in f])
    print(f"[sample] {n} prior samples available")


# ─────────────────────────────────────────────────────────────────────────────
def fast_mcd(X, h_frac=0.75, n_starts=200, n_csteps=30, seed=MCD_SEED):
    """Minimum Covariance Determinant (FastMCD-style: random p+1 subsets + C-steps), with consistency factor."""
    from scipy.stats import chi2
    rng = np.random.default_rng(seed)
    n, p = X.shape
    h = int(np.ceil(h_frac * n))
    best = (np.inf, None, None)
    for _ in range(n_starts):
        idx = rng.choice(n, p + 1, replace=False)
        mu = X[idx].mean(0); C = np.cov(X[idx].T) + 1e-9 * np.eye(p)
        for _ in range(n_csteps):
            try:
                d = np.einsum("ij,jk,ik->i", X - mu, np.linalg.inv(C), X - mu)
            except np.linalg.LinAlgError:
                break
            sub = np.argsort(d)[:h]
            mu_new = X[sub].mean(0); C_new = np.cov(X[sub].T)
            if np.allclose(mu_new, mu) and np.allclose(C_new, C):
                break
            mu, C = mu_new, C_new
        det = np.linalg.slogdet(C)[1]
        if det < best[0]:
            best = (det, mu, C)
    _, mu, C = best
    d = np.einsum("ij,jk,ik->i", X - mu, np.linalg.inv(C), X - mu)
    C = C * np.median(d) / chi2.ppf(0.5, p)          # consistency correction
    return mu, C


def cmd_falsify(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(BEL_FIG, exist_ok=True)
    files = sorted(f for f in os.listdir(os.path.join(BEL_DATA, "prior")) if f.endswith(".npz") and "tmp" not in f)
    D, TH = [], []
    for f in files:
        z = np.load(os.path.join(BEL_DATA, "prior", f), allow_pickle=True)
        if np.all(z["R_noisy"] > 0):
            D.append(np.log10(z["R_noisy"])); TH.append(json.loads(str(z["theta"])))
    D = np.array(D); n = len(D)
    # observed data sets (G1 quadrupoles, same order)
    g1 = wc.load_run("deep_drop_data")
    fwd = np.load(os.path.join(wc.PROCESSED_DIR, "synthetic", "truth_forward.npz"))
    obs = {"field (G1)": np.log10(g1["R"])}
    for t in ("T1", "T1h"):
        R = fwd[f"R_{t}"]
        rng = np.random.default_rng(1000)                         # Stage-4 noise realisation 0
        obs[f"{t} (synthetic, noise r0)"] = np.log10(R + np.sqrt((ERR_REL * R) ** 2 + ERR_ABS ** 2) * rng.standard_normal(len(R)))
    mu = D.mean(0); Xc = D - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = s ** 2 / (n - 1); cum = np.cumsum(ev) / ev.sum()
    k = int(np.searchsorted(cum, 0.99) + 1); k = min(k, n // 10)
    scores = Xc @ Vt[:k].T
    # classical Mahalanobis in the PC space (leave-one-out for the prior reference distribution)
    var = scores.var(0, ddof=1)
    d_prior = np.array([np.sum(scores[i] ** 2 / np.delete(scores, i, 0).var(0, ddof=1)) for i in range(n)])
    Q_prior = np.sum((Xc - scores @ Vt[:k]) ** 2, axis=1)
    mcd_mu, mcd_C = fast_mcd(scores)
    Ci = np.linalg.inv(mcd_C)
    d_rob_prior = np.einsum("ij,jk,ik->i", scores - mcd_mu, Ci, scores - mcd_mu)
    from scipy.stats import chi2
    L = ["# B4 — prior falsification (pilot, {} prior samples)".format(n), "",
         f"PCA of log10 R (noisy prior-predicted data, {D.shape[1]} G1 quadrupoles), centred on the prior mean. "
         f"k = {k} components ({100*cum[k-1]:.2f} % of the variance; k capped at n/10). "
         "T² = Mahalanobis distance² in the k-PC space (prior reference = leave-one-out distances of the prior samples); "
         "robust T² with a FastMCD covariance (h = 75 %, seed {}); Q = squared residual outside the k-PC subspace "
         "(tests data patterns the prior cannot produce at all).".format(MCD_SEED), "",
         "| data | T² | percentile vs prior | χ²_k p-value | robust T² | robust percentile | Q | Q percentile vs prior |",
         "|---|---|---|---|---|---|---|---|"]
    obs_scores = {}
    for name, y in obs.items():
        yc = y - mu; sc = yc @ Vt[:k].T; obs_scores[name] = sc
        t2 = float(np.sum(sc ** 2 / var)); t2r = float((sc - mcd_mu) @ Ci @ (sc - mcd_mu))
        q = float(np.sum((yc - sc @ Vt[:k]) ** 2))
        L.append(f"| {name} | {t2:.1f} | {100*np.mean(d_prior < t2):.1f} | {chi2.sf(t2, k):.2e} | {t2r:.1f} | "
                 f"{100*np.mean(d_rob_prior < t2r):.1f} | {q:.3f} | {100*np.mean(Q_prior < q):.1f} |")
    # which components / parameters are responsible (field)
    f_sc = obs_scores["field (G1)"]
    z = f_sc / np.sqrt(var)
    pnames = ["x_L", "x_S", "tau", "z_wt", "z_deep", "rho_cap", "rho_aquifer", "rho_lens", "rho_below", "rho_discharge",
              "rho_deep", "dis_present", "clay_present"] + [f"base_depth_{int(x)}" for x in CTRL_X]
    def pval(th, p):
        if p.startswith("base_depth_"):
            return th["base_depth"][CTRL_X.index(float(p.split("_")[-1]))]
        v = th[p]
        return float(np.log10(v)) if p.startswith("rho_") else float(v)
    PM = np.array([[pval(th, p) for p in pnames] for th in TH])
    L += ["", "## Field data per principal component", "",
          "z = field score / prior s.d.; 'range' = where the field score falls within the prior scores (percentile). "
          "Parameters: the three with the largest |correlation| between the prior samples' PC score and the parameter "
          "(ρ in log10). A field |z| far outside the prior cloud, with a strongly correlated parameter, points to that "
          "parameter's range (direction = sign(z)·sign(corr)).", "",
          "| PC | var % | field z | field percentile | most correlated parameters (corr) |", "|---|---|---|---|---|"]
    for i in range(k):
        cr = [np.corrcoef(scores[:, i], PM[:, j])[0, 1] for j in range(len(pnames))]
        top = np.argsort(-np.abs(cr))[:3]
        L.append(f"| {i+1} | {100*ev[i]/ev.sum():.1f} | {z[i]:+.2f} | {100*np.mean(scores[:, i] < f_sc[i]):.1f} | "
                 + ", ".join(f"{pnames[j]} ({cr[j]:+.2f})" for j in top) + " |")
    L += ["", "Suggested direction per flagged PC (|z| > 3 or field percentile < 0.5 / > 99.5): the parameter with the "
          "largest |corr| should extend towards values implied by sign(z)·sign(corr) (+ = larger).", ""]
    flagged = [i for i in range(k) if abs(z[i]) > 3 or not (0.5 <= 100 * np.mean(scores[:, i] < f_sc[i]) <= 99.5)]
    for i in flagged:
        cr = np.array([np.corrcoef(scores[:, i], PM[:, j])[0, 1] for j in range(len(pnames))])
        j = int(np.argmax(np.abs(cr)))
        L.append(f"- PC{i+1}: field z = {z[i]:+.1f}; {pnames[j]} (corr {cr[j]:+.2f}) → needs "
                 f"{'larger' if np.sign(z[i]) * np.sign(cr[j]) > 0 else 'smaller'} values than the prior range "
                 f"[{PM[:, j].min():.2f}, {PM[:, j].max():.2f}] provides")
    if not flagged:
        L.append("- no individual component is flagged")
    # figure: PC1–PC2, PC1–PC3 and Q
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (a, b) in zip(axes[:2], ((0, 1), (0, 2))):
        ax.scatter(scores[:, a], scores[:, b], s=8, c="0.6", label=f"prior ({n})")
        for (name, sc), mk, col in zip(obs_scores.items(), ("*", "o", "s"), ("red", "navy", "darkorange")):
            ax.plot(sc[a], sc[b], mk, color=col, ms=12, label=name)
        ax.set_xlabel(f"PC{a+1}"); ax.set_ylabel(f"PC{b+1}"); ax.grid(True, alpha=0.3)
        wc.white_legend(ax, loc="best")
    ax = axes[2]
    ax.hist(Q_prior, bins=30, color="0.6", label="prior Q")
    for (name, y), col in zip(obs.items(), ("red", "navy", "darkorange")):
        yc = y - mu; q = np.sum((yc - (yc @ Vt[:k].T) @ Vt[:k]) ** 2)
        ax.axvline(q, color=col, lw=2, label=name)
    ax.set_xscale("log"); ax.set_xlabel("Q (residual outside k PCs)"); wc.white_legend(ax, loc="best")
    fig.suptitle(f"Prior falsification — PCA of log10 R, {n} prior samples, k = {k}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(BEL_FIG, "b4_prior_falsification.png"), dpi=150)
    with open(os.path.join(BEL_RES, "b4_prior_falsification.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spec"); sub.add_parser("timing"); sub.add_parser("falsify")
    ps = sub.add_parser("sample"); ps.add_argument("--n-samples", type=int, default=300); ps.add_argument("--workers", type=int, default=4)
    pw = sub.add_parser("worker"); pw.add_argument("--worker-id", type=int, required=True)
    pw.add_argument("--ids", type=int, nargs="+", required=True)
    args = ap.parse_args()
    dict(spec=cmd_spec, timing=cmd_timing, sample=cmd_sample, worker=cmd_worker, falsify=cmd_falsify)[args.cmd](args)


if __name__ == "__main__":
    main()
