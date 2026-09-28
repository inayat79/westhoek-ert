"""
13_bel_full.py
==============
Westhoek ERT — Stage 5b: Bayesian Evidential Learning (BEL) full run.

Subcommands (in order)
  spec      prior_spec_v2 (results/bel/prior_spec_v2.md + data/processed/bel/v2/prior_spec_v2.json)
  sample    parallel, resumable forward modelling (noise-free R stored):
              --set dual   : 100 prior samples on BOTH the fine and the medium mesh (approximation error)
              --set train  : 3300 prior samples on the medium mesh (0–2999 training, 3000–3299 held out)
  approx    mean and covariance of δ = log10 R_fine − log10 R_medium from the dual set; size vs the noise
  falsify   B4-style falsification of the v2 prior against the field data (stops the pipeline if outside P99)
  bel       data PCA, target PCA, CCA, posterior (Gaussian regression and BEL1D-style KDE), Gate C
            validation (held-out + Stage-4 truths), field posterior, discharge / dune-layer probabilities

Data model for every prior sample i (applied at analysis time, seeded):
  log10 R_i = log10 R_medium,i + mean(δ) + e_i,   e_i ~ N(0, Cov(δ))   (approximation-error correction)
  R_i ← R_i + sqrt((0.03 R_i)² + (1e-4 Ω)²) · n_i                       (measurement noise)
Field and Stage-4 truth data are used as they are (fine-mesh truths + Stage-4 noise seeds 1000 + r).

Seeds (recorded in prior_spec_v2.json): parameters default_rng([88000, i]); noise default_rng([99000, i]);
approximation-error draw default_rng([111000, i]); dual set i = 500000 + j; posterior default_rng([123000, k]);
KDE bandwidths are deterministic (Scott). Outputs only in data/processed/bel/v2/, figures/bel/, results/bel/.
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


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


pilot = _load("bel12", "12_bel_pilot.py")

D2 = os.path.join(wc.PROCESSED_DIR, "bel", "v2")
FIG = os.path.join(wc.FIG_DIR, "bel")
RES = os.path.join(wc.RESULTS_DIR, "bel")
SPEC2 = os.path.join(D2, "prior_spec_v2.json")
PRIOR_SEED, NOISE_SEED, APPROX_SEED, POST_SEED = 88000, 99000, 111000, 123000
DUAL_OFFSET = 500000
N_TRAIN, N_HELD = 3000, 300
N_DUAL = 100
CTRL_X = [100.0, 170.0, 240.0, 310.0, 380.0, 450.0]
TX = np.arange(100.0, 451.0, 10.0)            # target columns
ERR_REL, ERR_ABS = 0.03, 1e-4
MESH = {"fine": dict(paraDX=0.2, paraDepth=60.0, paraMaxCellSize=5.0, quality=34.0),
        "medium": dict(paraDX=0.33, paraDepth=60.0, paraMaxCellSize=15.0, quality=33.0)}
N_POST = 2000
X_END = 505.0


# ─────────────────────────────────────────────────────────────────────────────
# Prior v2
# ─────────────────────────────────────────────────────────────────────────────
def draw(spec, i):
    """One prior sample (dict) with the contrast rules enforced by redrawing the resistivity triple."""
    rng = np.random.default_rng([PRIOR_SEED, i])
    P = spec["params"]
    U = lambda k: float(rng.uniform(*P[k]["range"]))
    LU = lambda k: float(10 ** rng.uniform(*np.log10(P[k]["range"])))
    th = dict(x_L=U("x_L"), tau=U("tau"), z_wt=U("z_wt"), z_deep=U("z_deep"),
              base_depth=[float(rng.uniform(*P["base_depth"]["range"])) for _ in CTRL_X],
              rho_cap=LU("rho_cap"), rho_aquifer=LU("rho_aquifer"), rho_deep=LU("rho_deep"))
    redraws = 0
    while True:
        rl, rb, rd = LU("rho_lens"), LU("rho_below"), LU("rho_discharge")
        if rb >= 1.5 * rl and rd >= 1.5 * rl:
            break
        redraws += 1
    th.update(rho_lens=rl, rho_below=rb, rho_discharge=rd, n_redraws=redraws)
    th["dis_present"] = bool(rng.uniform() < P["dis_present"]["p"])
    th["x_S"] = U("x_S"); th["dis_width"] = U("dis_width")
    th["dis_skin"] = U("dis_skin"); th["dis_bottom"] = U("dis_bottom")
    th["clay_present"] = bool(rng.uniform() < P["clay_present"]["p"])
    th["clay_x0"] = U("clay_x0")
    th["clay_x1"] = float(rng.uniform(P["clay_x1"]["range"][0], th["x_L"] + 20.0))
    th["clay_top"] = U("clay_top"); th["clay_thick"] = U("clay_thick"); th["rho_clay"] = LU("rho_clay")
    return th


class PriorModel2:
    def __init__(self, th):
        self.th = th
        self.ex, self.ez = wc.land_surface()
        ctrl_z = np.interp(CTRL_X, self.ex, self.ez) - np.array(th["base_depth"])
        self.bf = PchipInterpolator(CTRL_X, ctrl_z, extrapolate=False)

    def surface(self, x):
        return np.interp(x, self.ex, self.ez)

    def base(self, x):
        return np.asarray(self.bf(np.clip(x, CTRL_X[0], CTRL_X[-1])), float)

    def dis_span(self):
        th = self.th
        if not th["dis_present"]:
            return None
        return th["x_S"], min(th["x_S"] + th["dis_width"], X_END)

    def log10rho(self, x, z):
        th = self.th
        x = np.asarray(x, float); z = np.asarray(z, float)
        s = self.surface(x)
        L = lambda k: np.log10(th[k])
        out = np.full(x.shape, L("rho_below"))
        dune = x < th["x_L"]
        out[dune] = np.where(z[dune] > th["z_wt"], L("rho_cap"), L("rho_aquifer"))
        span = self.dis_span()
        lens_x = ~dune
        if span:
            in_dis = (x >= span[0]) & (x < span[1])
            lens_x &= ~in_dis
        b = self.base(x); d = z - b
        f = np.clip((d + th["tau"] / 2) / th["tau"], 0, 1) if th["tau"] > 0 else (d > 0).astype(float)
        out[lens_x] = (L("rho_below") + f * (L("rho_lens") - L("rho_below")))[lens_x]
        if span:
            skin = in_dis & (z > s - th["dis_skin"])
            body = in_dis & (z <= s - th["dis_skin"]) & (z > th["dis_bottom"])
            out[skin] = L("rho_lens"); out[body] = L("rho_discharge")
        if th["clay_present"]:
            c = ((x >= th["clay_x0"]) & (x <= th["clay_x1"]) & (z <= th["clay_top"])
                 & (z > th["clay_top"] - th["clay_thick"]))
            out[c] = L("rho_clay")
        # the deep unit lies below everything EXCEPT the lens (incl. its transition), so a lens base below
        # z_deep is not truncated and the target (base centre) stays consistent with the model (as T2c)
        in_lens = lens_x & (f > 0)
        out[(z <= th["z_deep"]) & ~in_lens] = L("rho_deep")
        return out

    def target(self, xs=TX):
        """Lens-base elevation (mTAW) = centre of the base transition; skin base over the discharge zone;
        surface (zero lens thickness) landward of x_L."""
        th = self.th
        t = self.base(xs)
        s = self.surface(xs)
        t = np.where(xs < th["x_L"], s, t)
        span = self.dis_span()
        if span:
            t = np.where((xs >= span[0]) & (xs < span[1]), s - th["dis_skin"], t)
        return t


# ─────────────────────────────────────────────────────────────────────────────
def cmd_spec(args):
    os.makedirs(D2, exist_ok=True); os.makedirs(RES, exist_ok=True)
    v1 = json.load(open(pilot.SPEC_JSON, encoding="utf-8"))
    P = json.loads(json.dumps(v1["params"]))
    P["rho_lens"] = dict(range=[0.4, 3.0], dist="log-uniform", unit="Ω·m", why="confirmed (saline by construction)")
    P["rho_below"]["why"] += "; contrast rule ρ_below ≥ 1.5 ρ_lens"
    P["rho_discharge"] = dict(range=[2.5, 40.0], dist="log-uniform", unit="Ω·m",
                              why="confirmed unchanged; contrast rule ρ_discharge ≥ 1.5 ρ_lens")
    P["x_S"] = dict(range=[330.0, 420.0], dist="uniform", unit="m",
                    why="seaward limit of the upper-beach lens = landward edge of the discharge zone (used only when "
                        "the discharge zone is present; otherwise the lens runs to the end of the line, 505 m)")
    P.pop("dis_x0", None)
    P["dis_width"]["why"] = "discharge zone x_S … min(x_S + width, 505 m); seaward of it the lens resumes"
    P["z_wt"] = dict(range=[1.0, 5.0], dist="uniform", unit="mTAW", why="confirmed")
    P["clay_x0"] = dict(range=[0.0, 60.0], dist="uniform", unit="m", why="left edge of the dune conductive layer")
    P["clay_x1"] = dict(range=[60.0, None], dist="uniform on [60, x_L + 20]", unit="m",
                        why="right edge; may reach 20 m under the lens edge (overrides lens/water there)")
    P["clay_top"] = dict(range=[-12.0, -3.0], dist="uniform", unit="mTAW", why="as specified")
    P["clay_thick"] = dict(range=[1.0, 6.0], dist="uniform", unit="m", why="as specified")
    P["rho_clay"] = dict(range=[1.0, 6.0], dist="log-uniform", unit="Ω·m", why="clay OR saltwater")
    spec = dict(version=2, params=P, ctrl_x=CTRL_X, target_x=list(map(float, TX)),
                seeds=dict(prior=f"default_rng([{PRIOR_SEED}, i])", noise=f"default_rng([{NOISE_SEED}, i])",
                           approx_error=f"default_rng([{APPROX_SEED}, i])", dual=f"i = {DUAL_OFFSET} + j",
                           posterior=f"default_rng([{POST_SEED}, k]) (k = dataset index)"),
                sets=dict(train=[0, N_TRAIN - 1], held_out=[N_TRAIN, N_TRAIN + N_HELD - 1],
                          dual=[DUAL_OFFSET, DUAL_OFFSET + N_DUAL - 1]),
                meshes=MESH, noise=dict(rel=ERR_REL, abs_ohm=ERR_ABS))
    with open(SPEC2, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    # rejection rate of the contrast rules (exact, from the draws of the training + held-out set)
    red = np.array([draw(spec, i)["n_redraws"] for i in range(N_TRAIN + N_HELD)])
    acc = 1.0 / (1.0 + red.mean())
    L = ["# BEL prior specification v2 (Stage 5b full run)", "",
         "Supersedes prior_spec.md (pilot). Machine-readable: `data/processed/bel/v2/prior_spec_v2.json`. "
         "All depths reported in mTAW.", "", "## Changes from the pilot (as confirmed)", "",
         "- Lens resistivity 0.4–3.0 Ω·m. Contrast rules ρ_below ≥ 1.5·ρ_lens and ρ_discharge ≥ 1.5·ρ_lens: the "
         "resistivity triple (lens, below, discharge) is redrawn until both hold "
         f"(per-draw acceptance {100*acc:.1f} %, i.e. rejection rate {100*(1-acc):.1f} %; mean {red.mean():.2f} "
         f"redraws per sample, max {red.max()}; over the {N_TRAIN + N_HELD} training + held-out samples).",
         "- x_S = seaward limit of the upper-beach lens = landward edge of the discharge zone, U(330, 420) m, "
         "used when the discharge zone is present (p = 0.5); then the zone spans x_S … min(x_S + width, 505) m "
         "(width U(30, 120) m) with a saline skin of thickness U(0.5, 5) m over discharge water down to "
         "U(−25, −8) mTAW; seaward of the zone the lens resumes. If absent, the lens runs to the end of the line.",
         "- Discharge-zone resistivity 2.5–40 Ω·m (unchanged). Base control depths below the local surface (unchanged).",
         "- Water table under the dunes +1 … +5 mTAW.",
         "- Dune conductive layer (p = 0.5): left edge U(0, 60) m, right edge U(60, x_L + 20) m, top U(−12, −3) mTAW, "
         "thickness U(1, 6) m, 1–6 Ω·m (log-uniform). Where it reaches under the lens edge it overrides the lens.",
         "", "## BEL target and auxiliary variables", "",
         f"- Lens-base elevation at x = {TX[0]:g}–{TX[-1]:g} m every 10 m ({len(TX)} values) = the CENTRE of the base "
         "transition (the PCHIP base curve), consistent with the log-midpoint definition on true models. Over the "
         "discharge zone: the base of the saline skin. Landward of x_L (no lens): the land surface (zero lens thickness) "
         "— occurs only at x = 100–120 m.",
         "- Also recorded: discharge present / absent, ρ_discharge, dune layer present / absent.", "",
         "## Parameters", "", "| parameter | distribution | range | unit | justification |", "|---|---|---|---|---|"]
    for k, v in P.items():
        rng_s = (f"{v['range'][0]:g} – {v['range'][1] if v['range'][1] is not None else 'x_L+20'}" if "range" in v
                 else f"p = {v['p']}")
        L.append(f"| {k} | {v['dist']} | {rng_s} | {v.get('unit', '')} | {v['why']} |")
    L += ["", "## Seeds", ""] + [f"- {k}: {v}" for k, v in spec["seeds"].items()]
    with open(os.path.join(RES, "prior_spec_v2.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
def cmd_worker(args):
    import pygimli.meshtools as mt
    spec = json.load(open(SPEC2, encoding="utf-8"))
    scheme = pilot.build_scheme()
    meshes = {k: mt.createParaMesh(scheme.sensors(), **v) for k, v in MESH.items()
              if k == "medium" or args.set == "dual"}
    cells = {k: (np.array([c.center().x() for c in m.cells()]), np.array([c.center().y() for c in m.cells()]))
             for k, m in meshes.items()}
    logf = open(os.path.join(D2, "logs", f"{args.set}_worker_{args.worker_id}.log"), "a", encoding="utf-8")
    for i in args.ids:
        out = os.path.join(D2, args.set, f"s{i:06d}.npz")
        if os.path.exists(out):
            continue
        try:
            th = draw(spec, i)
            pm = PriorModel2(th)
            t0 = time.time(); R = {}
            for k, m in meshes.items():
                with contextlib.redirect_stdout(io.StringIO()):
                    R[k] = pilot.simulate_R(m, 10 ** pm.log10rho(*cells[k]), scheme.size())
            np.savez(out + ".tmp.npz", **{f"R_{k}": v for k, v in R.items()}, theta=json.dumps(th),
                     target=pm.target(), dis_present=th["dis_present"], rho_discharge=th["rho_discharge"],
                     clay_present=th["clay_present"], retries=pilot.RETRIES["n"])
            os.replace(out + ".tmp.npz", out)
            logf.write(f"{time.strftime('%H:%M:%S')} s{i} {time.time()-t0:.1f}s retries={pilot.RETRIES['n']}\n"); logf.flush()
        except Exception as e:
            import traceback
            logf.write(f"FAILED s{i}: {e}\n{traceback.format_exc()}\n"); logf.flush()
    logf.write("worker finished\n")


def cmd_sample(args):
    os.makedirs(os.path.join(D2, args.set), exist_ok=True); os.makedirs(os.path.join(D2, "logs"), exist_ok=True)
    ids = (list(range(DUAL_OFFSET, DUAL_OFFSET + N_DUAL)) if args.set == "dual"
           else list(range(N_TRAIN + N_HELD)))
    ids = [i for i in ids if not os.path.exists(os.path.join(D2, args.set, f"s{i:06d}.npz"))]
    parts = [ids[k::args.workers] for k in range(args.workers)]
    procs = []
    for w, part in enumerate(parts):
        if not part:
            continue
        wd = os.path.join(D2, "work", f"{args.set}_w{w}"); os.makedirs(wd, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "worker", "--set", args.set, "--worker-id", str(w),
               "--ids"] + [str(i) for i in part]
        procs.append(subprocess.Popen(cmd, cwd=wd, env=dict(os.environ, OMP_NUM_THREADS="2"), stdout=subprocess.DEVNULL,
                                      stderr=open(os.path.join(D2, "logs", f"{args.set}_worker_{w}.err"), "a")))
    for p in procs:
        p.wait()
    n = len([f for f in os.listdir(os.path.join(D2, args.set)) if f.endswith(".npz") and "tmp" not in f])
    print(f"[sample] {args.set}: {n} samples")


# ─────────────────────────────────────────────────────────────────────────────
def load_set(name, ids=None):
    d = os.path.join(D2, name)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and "tmp" not in f)
    out = []
    for f in files:
        i = int(f[1:-4])
        if ids is not None and i not in ids:
            continue
        z = np.load(os.path.join(d, f), allow_pickle=True)
        out.append((i, {k: z[k] for k in z.files}))
    return out


def approx_stats():
    p = os.path.join(D2, "approx_error.npz")
    if not os.path.exists(p):
        raise SystemExit("run `approx` first")
    z = np.load(p)
    return z["mean"], z["dev"]          # dev: (n_dual, n_data) centred δ (covariance = devᵀdev/(n−1))


def cmd_approx(args):
    dual = load_set("dual")
    dl = np.array([np.log10(z["R_fine"]) - np.log10(z["R_medium"]) for _, z in dual])
    mu = dl.mean(0); dev = dl - mu
    sd = dev.std(0, ddof=1)
    noise_log10 = ERR_REL / np.log(10)               # ≈ σ of log10 R for 3 % noise (floor ignored)
    np.savez(os.path.join(D2, "approx_error.npz"), mean=mu, dev=dev, sd=sd)
    ev = np.linalg.svd(dev, compute_uv=False) ** 2 / (len(dl) - 1)
    L = ["# Mesh approximation error (medium vs fine forward mesh)", "",
         f"δ = log10 R_fine − log10 R_medium over {len(dl)} prior samples (dual set, seeds i = {DUAL_OFFSET} + j), "
         f"G1 quadrupoles ({dl.shape[1]}). Reference: 3 % noise ≈ {noise_log10:.4f} in log10 R.", "",
         "| quantity | median | P95 | max | vs 3 % noise (median / P95) |", "|---|---|---|---|---|",
         f"| |mean δ| per datum | {np.median(np.abs(mu)):.5f} | {np.percentile(np.abs(mu), 95):.5f} | {np.abs(mu).max():.5f} | "
         f"{np.median(np.abs(mu))/noise_log10:.2f} / {np.percentile(np.abs(mu), 95)/noise_log10:.2f} |",
         f"| s.d. of δ per datum | {np.median(sd):.5f} | {np.percentile(sd, 95):.5f} | {sd.max():.5f} | "
         f"{np.median(sd)/noise_log10:.2f} / {np.percentile(sd, 95)/noise_log10:.2f} |", "",
         f"Mean δ in % of R: median {100*np.median(np.abs(10**mu-1)):.2f} %, max {100*np.abs(10**mu-1).max():.2f} %. "
         f"Covariance rank ≤ {len(dl)-1}; its first 5 eigenvalues carry "
         f"{100*ev[:5].sum()/ev.sum():.0f} % of the total variance (the error is spatially coherent).", "",
         "Applied to every medium-mesh prior sample: log10 R + mean δ + e, e ~ N(0, Cov δ) drawn as devᵀz/√(n−1), "
         f"z ~ N(0, I), seed default_rng([{APPROX_SEED}, i]); then 3 % + 1e-4 Ω noise."]
    with open(os.path.join(RES, "approx_error.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def prior_data(samples, mu, dev):
    """Corrected, noisy log10 R for prior samples (list of (i, dict))."""
    out, keep = [], []
    n1 = dev.shape[0] - 1
    for i, z in samples:
        l = np.log10(z["R_medium"]) + mu
        e = dev.T @ np.random.default_rng([APPROX_SEED, i]).standard_normal(dev.shape[0]) / np.sqrt(n1)
        R = 10 ** (l + e)
        Rn = R + np.sqrt((ERR_REL * R) ** 2 + ERR_ABS ** 2) * np.random.default_rng([NOISE_SEED, i]).standard_normal(len(R))
        if np.all(Rn > 0):
            out.append(np.log10(Rn)); keep.append(i)
    return np.array(out), keep


def observed_sets():
    """Field (G1) and Stage-4 truth data sets (fine mesh, Stage-4 noise seeds)."""
    g1 = wc.load_run("deep_drop_data")
    obs = {"field": np.log10(g1["R"])}
    fwd = np.load(os.path.join(wc.PROCESSED_DIR, "synthetic", "truth_forward.npz"))
    truths = {}
    for t in ("T1", "T2a", "T3", "T1h"):
        R = fwd[f"R_{t}"]
        for r in range(20):
            rng = np.random.default_rng(1000 + r)
            Rn = R + np.sqrt((ERR_REL * R) ** 2 + ERR_ABS ** 2) * rng.standard_normal(len(R))
            truths[(t, r)] = np.log10(Rn) if np.all(Rn > 0) else None
    return obs, truths


def cmd_falsify(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import chi2
    mu_d, dev = approx_stats()
    train = load_set("train", set(range(N_TRAIN)))
    D, _ = prior_data(train, mu_d, dev)
    obs, truths = observed_sets()
    n = len(D); m = D.mean(0); Xc = D - m
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = s ** 2 / (n - 1); cum = np.cumsum(ev) / ev.sum()
    k = int(np.searchsorted(cum, 0.99) + 1)
    sc = Xc @ Vt[:k].T; var = sc.var(0, ddof=1)
    t2p = np.sum(sc ** 2 / var, axis=1)                       # n = 3000 ≫ k: in-sample is fine
    Qp = np.sum((Xc - sc @ Vt[:k]) ** 2, axis=1)
    mcd_mu, mcd_C = pilot.fast_mcd(sc)
    Ci = np.linalg.inv(mcd_C)
    t2rp = np.einsum("ij,jk,ik->i", sc - mcd_mu, Ci, sc - mcd_mu)
    rows = {"field (G1)": obs["field"], "T1 r0": truths[("T1", 0)], "T1h r0": truths[("T1h", 0)]}
    L = [f"# Prior v2 falsification ({n} training samples, medium mesh + approximation-error correction)", "",
         f"PCA of log10 R: k = {k} components (≥ 99 % variance: {100*cum[k-1]:.2f} %). Robust covariance: FastMCD "
         f"(seed {pilot.MCD_SEED}).", "", "| data | T² | T² percentile | robust T² percentile | Q | Q percentile | verdict |",
         "|---|---|---|---|---|---|---|"]
    verdict = {}
    for name, y in rows.items():
        yc = y - m; s_ = yc @ Vt[:k].T
        t2 = float(np.sum(s_ ** 2 / var)); t2r = float((s_ - mcd_mu) @ Ci @ (s_ - mcd_mu))
        q = float(np.sum((yc - s_ @ Vt[:k]) ** 2))
        pc = [100 * np.mean(t2p < t2), 100 * np.mean(t2rp < t2r), 100 * np.mean(Qp < q)]
        out = max(pc) > 99
        verdict[name] = out
        L.append(f"| {name} | {t2:.1f} | {pc[0]:.1f} | {pc[1]:.1f} | {q:.3f} | {pc[2]:.1f} | "
                 f"{'OUTSIDE P99' if out else 'inside'} |")
    json.dump(dict(field_outside=bool(verdict["field (G1)"])), open(os.path.join(D2, "falsify_verdict.json"), "w"))
    with open(os.path.join(RES, "falsification_v2.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
# BEL
# ─────────────────────────────────────────────────────────────────────────────
class PCA:
    def __init__(self, X, frac=0.99, k=None):
        self.m = X.mean(0)
        U, s, Vt = np.linalg.svd(X - self.m, full_matrices=False)
        ev = s ** 2 / (len(X) - 1); self.cum = np.cumsum(ev) / ev.sum()
        self.k = k or int(np.searchsorted(self.cum, frac) + 1)
        self.V = Vt[:self.k]; self.ev = ev

    def fwd(self, X):
        return (np.atleast_2d(X) - self.m) @ self.V.T

    def inv(self, S):
        return S @ self.V + self.m

    def resid(self, X):
        Xc = np.atleast_2d(X) - self.m
        return np.sum((Xc - (Xc @ self.V.T) @ self.V) ** 2, axis=1)


class CCA:
    """Classical CCA on (standardised) PC scores. Canonical variates have unit variance on the training set."""

    def __init__(self, X, Y, reg=1e-8):
        self.mx, self.my = X.mean(0), Y.mean(0)
        Xc, Yc = X - self.mx, Y - self.my
        n = len(X)
        Cxx = Xc.T @ Xc / (n - 1) + reg * np.eye(X.shape[1]); Cyy = Yc.T @ Yc / (n - 1) + reg * np.eye(Y.shape[1])
        Cxy = Xc.T @ Yc / (n - 1)
        def isqrt(C):
            w, V = np.linalg.eigh(C); return V @ np.diag(w ** -0.5) @ V.T
        Wx, Wy = isqrt(Cxx), isqrt(Cyy)
        U, r, Vt = np.linalg.svd(Wx @ Cxy @ Wy, full_matrices=False)
        q = min(X.shape[1], Y.shape[1])
        self.A = Wx @ U[:, :q]; self.B = Wy @ Vt.T[:, :q]; self.r = r[:q]
        self.Binv = np.linalg.inv(self.B) if self.B.shape[0] == self.B.shape[1] else np.linalg.pinv(self.B)

    def cx(self, X):
        return (np.atleast_2d(X) - self.mx) @ self.A

    def cy(self, Y):
        return (np.atleast_2d(Y) - self.my) @ self.B

    def y_from_c(self, C):
        return C @ self.Binv + self.my


def kde_conditional_sample(cd_train, ct_train, cd_obs, n, rng):
    """BEL1D-style: 2-D Gaussian KDE of (c_d, c_t) per canonical pair; sample c_t | c_d = cd_obs by inverse CDF."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(np.vstack([cd_train, ct_train]))       # Scott bandwidth (deterministic)
    grid = np.linspace(ct_train.min() - 1.0, ct_train.max() + 1.0, 800)
    dens = kde(np.vstack([np.full_like(grid, cd_obs), grid]))
    if not np.isfinite(dens).all() or dens.sum() <= 0:
        return rng.normal(0, 1, n)
    cdf = np.cumsum(dens); cdf /= cdf[-1]
    return np.interp(rng.uniform(size=n), cdf, grid)


class BEL:
    def __init__(self, D, T, frac_d=0.99, frac_t=0.99):
        self.pd = PCA(D, frac_d); self.pt = PCA(T, frac_t)
        self.Sd = self.pd.fwd(D); self.St = self.pt.fwd(T)
        self.cca = CCA(self.Sd, self.St)
        self.Cd = self.cca.cx(self.Sd); self.Ct = self.cca.cy(self.St)
        q = self.Ct.shape[1]
        # Gaussian regression per canonical pair: c_t = a c_d + b + e
        self.lin = []
        for i in range(q):
            a, b = np.polyfit(self.Cd[:, i], self.Ct[:, i], 1)
            res = self.Ct[:, i] - (a * self.Cd[:, i] + b)
            self.lin.append((a, b, res.std(ddof=2)))

    def posterior(self, y, n, rng, method="kde"):
        cd = self.cca.cx(self.pd.fwd(y))[0]
        q = self.Ct.shape[1]
        C = np.empty((n, q))
        for i in range(q):
            if method == "kde":
                C[:, i] = kde_conditional_sample(self.Cd[:, i], self.Ct[:, i], cd[i], n, rng)
            else:
                a, b, s = self.lin[i]
                C[:, i] = a * cd[i] + b + s * rng.standard_normal(n)
        return self.pt.inv(self.cca.y_from_c(C))

    def nonlinearity(self, i, nb=20):
        """Share of the linear-fit residual variance explained by a binned-mean (nonparametric) fit."""
        x, y = self.Cd[:, i], self.Ct[:, i]
        a, b, s = self.lin[i]
        r_lin = y - (a * x + b)
        edges = np.quantile(x, np.linspace(0, 1, nb + 1)); idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
        bm = np.array([y[idx == j].mean() for j in range(nb)])
        r_np = y - bm[idx]
        return float(1 - r_np.var() / r_lin.var())


def knn_aux(Sd_train, var, sd_obs, aux, k=100):
    """k-NN (whitened data-PC space) estimate of auxiliary variables."""
    d = np.sum((Sd_train - sd_obs) ** 2 / var, axis=1)
    idx = np.argsort(d)[:k]
    return idx


def cmd_bel(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    global FIG, RES
    test = os.environ.get("BEL_TEST") == "1"
    if test:   # code test on partial data: 80/20 split of what exists, separate output folders
        FIG = os.path.join(FIG, "_test"); RES = os.path.join(RES, "_test"); os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    if not test:
        verdict = json.load(open(os.path.join(D2, "falsify_verdict.json")))
        if verdict["field_outside"]:
            raise SystemExit("field data outside the prior P99 — stopping (see falsification_v2.md)")
    mu_d, dev = approx_stats()
    if test:
        allp = load_set("train"); cut = int(0.8 * len(allp)); tr, ho = allp[:cut], allp[cut:]
    else:
        tr = load_set("train", set(range(N_TRAIN)))
        ho = load_set("train", set(range(N_TRAIN, N_TRAIN + N_HELD)))
    Dtr, ktr = prior_data(tr, mu_d, dev); Dho, kho = prior_data(ho, mu_d, dev)
    tmap = {i: z for i, z in tr + ho}
    Ttr = np.array([tmap[i]["target"] for i in ktr]); Tho = np.array([tmap[i]["target"] for i in kho])
    aux_tr = dict(dis=np.array([bool(tmap[i]["dis_present"]) for i in ktr]),
                  rdis=np.array([float(tmap[i]["rho_discharge"]) for i in ktr]),
                  clay=np.array([bool(tmap[i]["clay_present"]) for i in ktr]))
    bel = BEL(Dtr, Ttr)
    obs, truths = observed_sets()
    L = [f"# BEL — Stage 5b full run", "",
         f"Training: {len(ktr)} prior samples (of {N_TRAIN}; {N_TRAIN-len(ktr)} dropped for a non-positive noisy R); "
         f"held out: {len(kho)}. Medium mesh + approximation-error correction + 3 % / 1e-4 Ω noise.", ""]
    # data PCA and the field residual
    Qtr = bel.pd.resid(Dtr); qf = float(bel.pd.resid(obs["field"])[0])
    sf = bel.pd.fwd(obs["field"])[0]
    pct = [100 * np.mean(bel.Sd[:, i] < sf[i]) for i in range(bel.pd.k)]
    L += ["## Dimension reduction", "",
          f"- Data PCA: k_d = {bel.pd.k} components ({100*bel.pd.cum[bel.pd.k-1]:.2f} % of the variance). Field residual "
          f"outside them Q = {qf:.3f} (percentile {100*np.mean(Qtr < qf):.1f} of the training Q; median training Q "
          f"{np.median(Qtr):.3f}). Field scores within the training range in every kept component: "
          f"{'yes' if all(0 < p < 100 for p in pct) else 'NO'} (percentiles {min(pct):.1f}–{max(pct):.1f}).",
          f"- Target PCA: k_t = {bel.pt.k} components ({100*bel.pt.cum[bel.pt.k-1]:.2f} %)."]
    # pilot PC29 coverage
    pdat = []
    for f in sorted(os.listdir(os.path.join(wc.PROCESSED_DIR, "bel", "prior"))):
        if f.endswith(".npz") and "tmp" not in f:
            z = np.load(os.path.join(wc.PROCESSED_DIR, "bel", "prior", f))
            if np.all(z["R_noisy"] > 0):
                pdat.append(np.log10(z["R_noisy"]))
    pdat = np.array(pdat); pm_ = pdat.mean(0)
    Vp = np.linalg.svd(pdat - pm_, full_matrices=False)[2]
    v29 = Vp[28]
    p_new = (Dtr - pm_) @ v29; p_f = float((obs["field"] - pm_) @ v29); p_old = (pdat - pm_) @ v29
    L.append(f"- Pilot PC29 direction: field projection {p_f:+.3f}; pilot prior range [{p_old.min():+.3f}, {p_old.max():+.3f}] "
             f"(field percentile {100*np.mean(p_old < p_f):.1f}); v2 training range [{p_new.min():+.3f}, {p_new.max():+.3f}] "
             f"(field percentile {100*np.mean(p_new < p_f):.1f}) → "
             f"{'covered' if 0.5 < 100*np.mean(p_new < p_f) < 99.5 else 'NOT covered'}.")
    # CCA
    nl = [bel.nonlinearity(i) for i in range(len(bel.cca.r))]
    method = "kde" if max(nl[:4]) > 0.05 else "gauss"
    L += ["", "## CCA", "", "| pair | canonical correlation | nonlinearity (share of linear-residual variance explained by a "
          "binned mean) |", "|---|---|---|"]
    L += [f"| {i+1} | {bel.cca.r[i]:.3f} | {nl[i]:.3f} |" for i in range(len(bel.cca.r))]
    L += ["", f"Posterior method: **{'KDE in canonical space (BEL1D)' if method == 'kde' else 'Gaussian regression'}** "
          f"(rule: KDE if any of the first four pairs has nonlinearity > 0.05; max = {max(nl[:4]):.3f}). "
          "Both methods are validated below."]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for i, ax in enumerate(axes):
        ax.scatter(bel.Cd[:, i], bel.Ct[:, i], s=3, alpha=0.4, c="0.4")
        a, b, s = bel.lin[i]; xx = np.linspace(bel.Cd[:, i].min(), bel.Cd[:, i].max(), 50)
        ax.plot(xx, a * xx + b, "r-", lw=1.2)
        cf = bel.cca.cx(bel.pd.fwd(obs["field"]))[0, i]
        ax.axvline(cf, color="navy", lw=1.2, ls="--", label="field")
        ax.set_title(f"pair {i+1}: r = {bel.cca.r[i]:.2f}, NL = {nl[i]:.2f}", fontsize=10)
        ax.set_xlabel("data canonical variate"); ax.set_ylabel("target canonical variate"); ax.grid(True, alpha=0.3)
    wc.white_legend(axes[0], loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "bel_cca_pairs.png"), dpi=150); plt.close(fig)

    # ── Gate C: held-out prior samples ───────────────────────────────────────
    def post_stats(P, truth):
        lo, med, hi = np.percentile(P, [5, 50, 95], axis=0)
        return lo, med, hi, (truth >= lo) & (truth <= hi), np.mean(P < truth, axis=0)
    res_ho = {}
    for mth in ("kde", "gauss"):
        cov, ranks, widths = [], [], []
        for j, y in enumerate(Dho):
            P = bel.posterior(y, N_POST, np.random.default_rng([POST_SEED, 10_000 + j]), mth)
            lo, med, hi, c, rk = post_stats(P, Tho[j])
            cov.append(c); ranks.append(rk); widths.append(hi - lo)
        res_ho[mth] = (np.array(cov), np.array(ranks), np.array(widths))
    L += ["", "## Gate C — held-out prior samples", "",
          f"{len(Dho)} held-out samples × {len(TX)} columns; 90 % interval = P5–P95 of {N_POST} posterior samples.", "",
          "| method | coverage (all) | per-column coverage min / median / max | median width (m) | rank-histogram χ² (10 bins) |",
          "|---|---|---|---|---|"]
    from scipy.stats import chisquare
    for mth, (cv, rk, wd) in res_ho.items():
        pc = cv.mean(0); h = np.histogram(rk.ravel(), bins=10, range=(0, 1))[0]
        L.append(f"| {mth} | {100*cv.mean():.1f}% | {100*pc.min():.0f}% / {100*np.median(pc):.0f}% / {100*pc.max():.0f}% | "
                 f"{np.median(wd):.2f} | {chisquare(h).statistic:.1f} (p = {chisquare(h).pvalue:.2g}) |")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, mth in zip(axes[:2], ("kde", "gauss")):
        ax.hist(res_ho[mth][1].ravel(), bins=20, range=(0, 1), color="#1F77B4", alpha=0.8)
        ax.axhline(res_ho[mth][1].size / 20, color="k", ls="--", lw=1, label="uniform")
        ax.set_title(f"rank histogram — held-out, {mth}", fontsize=10); ax.set_xlabel("rank of the true base among posterior samples")
        wc.white_legend(ax, loc="lower center")
    for mth, col in (("kde", "#1F77B4"), ("gauss", "#D62728")):
        axes[2].plot(TX, 100 * res_ho[mth][0].mean(0), "-o", ms=3, color=col, label=mth)
    axes[2].axhspan(85, 95, color="green", alpha=0.12, label="target 85–95 %")
    axes[2].set_ylim(0, 100); axes[2].set_xlabel("x (m)"); axes[2].set_ylabel("coverage of 90 % interval (%)")
    axes[2].set_title("held-out coverage per column", fontsize=10); axes[2].grid(True, alpha=0.3)
    wc.white_legend(axes[2], loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "bel_gateC_heldout.png"), dpi=150); plt.close(fig)

    # ── Gate C: Stage-4 truths vs linearised results ─────────────────────────
    c11 = _load("c11", "11_stage4_closeout.py")
    s4spec = c11.sb.load_spec()
    L += ["", "## Gate C — Stage-4 truths (fine-mesh data, Stage-4 noise seeds 1000 + r, 20 realisations)", "",
          "Truth = centre of the true base transition (= log-midpoint truth). Columns x = 100–450 m. Linearised results "
          "from the Stage-5 A2 runs (λ = 50; single-λ posterior samples): 3 Ω·m and log-midpoint definitions. "
          "'r ≤ 10' = the subset of (column, realisation) pairs whose linearised log-midpoint crossing has resolution "
          "radius ≤ 10 m.", "",
          "| truth | subset | n | BEL-kde cov / bias / width | BEL-gauss cov / bias / width | lin. 3 Ω·m cov / bias / width "
          "| lin. log-mid cov / bias / width |", "|---|---|---|---|---|---|---|"]
    gate = {}
    skipped = {}
    for t in ("T1", "T2a", "T3", "T1h"):
        tv = c11.truth_defs(s4spec, t, TX)["logmid"]
        tv3 = c11.truth_defs(s4spec, t, TX)["thr3"]
        rows = []
        for r in range(20):
            y = truths[(t, r)]
            if y is None:          # a noisy R ≤ 0: BEL needs the full data vector → skipped for ALL methods
                skipped.setdefault(t, []).append(r)
                continue
            a2 = np.load(os.path.join(wc.PROCESSED_DIR, "synthetic_s5", "a2_runs", f"{t}_r{r:02d}.npz"), allow_pickle=True)
            ent = {}
            for mth in ("kde", "gauss"):
                P = bel.posterior(y, N_POST, np.random.default_rng([POST_SEED, 20_000 + 100 * ["T1", "T2a", "T3", "T1h"].index(t) + r]), mth)
                lo, med, hi = np.percentile(P, [5, 50, 95], axis=0)
                ent[mth] = (lo, med, hi)
            for j, x in enumerate(TX):
                jj = int(np.argmin(np.abs(a2["cols"] - x)))
                row = dict(r_ok=float(a2["lam50_r_logmid"][jj]) <= 10 if np.isfinite(a2["lam50_r_logmid"][jj]) else False)
                for mth in ("kde", "gauss"):
                    lo, med, hi = (v[j] for v in ent[mth])
                    row[mth] = (lo <= tv[j] <= hi, med - tv[j], hi - lo)
                for dname, tt in (("thr3", tv3), ("logmid", tv)):
                    b = a2[f"lam50_band_{dname}"][:, jj]; mh = a2[f"lam50_mhat_{dname}"][jj]
                    row[dname] = ((b[0] >= 0.5) and (b[2] <= tt[j] <= b[3]), mh - tt[j], b[3] - b[2] if b[0] >= 0.5 else np.nan)
                rows.append(row)
        for sub in ("all", "r ≤ 10"):
            rs = rows if sub == "all" else [x for x in rows if x["r_ok"]]
            if not rs:
                continue
            cell = lambda k: (f"{100*np.mean([x[k][0] for x in rs]):.0f}% / {np.nanmedian([x[k][1] for x in rs]):+.2f} / "
                              f"{np.nanmedian([x[k][2] for x in rs]):.2f}")
            gate[(t, sub)] = {k: np.mean([x[k][0] for x in rs]) for k in ("kde", "gauss", "thr3", "logmid")}
            L.append(f"| {t} | {sub} | {len(rs)} | {cell('kde')} | {cell('gauss')} | {cell('thr3')} | {cell('logmid')} |")
    L.append("")
    L.append("Realisations skipped because a noisy R ≤ 0 (BEL needs every datum; skipped for all methods): "
             + (", ".join(f"{t}: {v}" for t, v in skipped.items()) if skipped else "none"))
    # ── field ────────────────────────────────────────────────────────────────
    Pf = {m: bel.posterior(obs["field"], N_POST, np.random.default_rng([POST_SEED, 0]), m) for m in ("kde", "gauss")}
    a4 = np.load(os.path.join(wc.RESULTS_DIR, "runs", "g1_lam50_v4", "interface_definitions_bands.npz"))
    L += ["", "## Field — lens-base elevation (mTAW)", "",
          f"BEL: median [P5, P95] of {N_POST} posterior samples (seed default_rng([{POST_SEED}, 0])). Linearised "
          "(g1_lam50, single λ = 50): 3 Ω·m and log-midpoint sampled bands (Stage-5 A4).", "",
          "| x | BEL-kde | BEL-gauss | lin. 3 Ω·m | lin. log-midpoint |", "|---|---|---|---|---|"]
    fmt = lambda a, b, c: f"{b:.2f} [{a:.2f}, {c:.2f}]"
    for x in (150, 200, 250, 300, 350):
        j = int(np.argmin(np.abs(TX - x))); jj = int(np.argmin(np.abs(a4["x"] - x)))
        bl = {m: np.percentile(Pf[m][:, j], [5, 50, 95]) for m in Pf}
        s3, sl = a4["single_thr3"][:, jj], a4["single_logmid"][:, jj]
        L.append(f"| {x} | {fmt(*bl['kde'])} | {fmt(*bl['gauss'])} | {fmt(s3[2], s3[1], s3[3])} | {fmt(sl[2], sl[1], sl[3])} |")
    # auxiliary: k-NN in whitened data-PC space
    var = bel.Sd.var(0, ddof=1)
    kk = 100
    idx = knn_aux(bel.Sd, var, bel.pd.fwd(obs["field"])[0], None, kk)
    p_dis = aux_tr["dis"][idx].mean(); p_clay = aux_tr["clay"][idx].mean()
    rd = aux_tr["rdis"][idx][aux_tr["dis"][idx]]
    # calibration on the held-out set
    ho_dis = np.array([bool(tmap[i]["dis_present"]) for i in kho]); ho_clay = np.array([bool(tmap[i]["clay_present"]) for i in kho])
    pd_ho, pc_ho = [], []
    for j, y in enumerate(Dho):
        ii = knn_aux(bel.Sd, var, bel.pd.fwd(y)[0], None, kk)
        pd_ho.append(aux_tr["dis"][ii].mean()); pc_ho.append(aux_tr["clay"][ii].mean())
    pd_ho, pc_ho = np.array(pd_ho), np.array(pc_ho)
    brier = lambda p, o: float(np.mean((p - o) ** 2))
    L += ["", "## Field — discharge zone and dune conductive layer", "",
          f"k-nearest-neighbour estimate (k = {kk}) in the whitened data-PC space (k_d = {bel.pd.k}); calibrated on the "
          "held-out set (Brier score; 0.25 = no skill for p = 0.5).", "",
          f"- P(discharge zone present | field) = {p_dis:.2f}   (held-out Brier {brier(pd_ho, ho_dis):.3f}; "
          f"AUC-like separation: mean P when present {pd_ho[ho_dis].mean():.2f} vs absent {pd_ho[~ho_dis].mean():.2f})",
          f"- ρ_discharge | present (neighbours with a discharge zone, n = {len(rd)}): median {np.median(rd):.1f} Ω·m, "
          f"P5–P95 {np.percentile(rd, 5):.1f}–{np.percentile(rd, 95):.1f} Ω·m (prior 2.5–40)" if len(rd) else
          "- ρ_discharge: no neighbours with a discharge zone",
          f"- P(dune conductive layer present | field) = {p_clay:.2f}   (held-out Brier {brier(pc_ho, ho_clay):.3f}; "
          f"mean P when present {pc_ho[ho_clay].mean():.2f} vs absent {pc_ho[~ho_clay].mean():.2f})"]
    # field figure
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ex, ez = wc.land_surface()
    ax.plot(ex, ez, "k-", lw=0.9, label="land surface")
    lo, med, hi = np.percentile(Pf[method], [5, 50, 95], axis=0)
    ax.fill_between(TX, lo, hi, color="purple", alpha=0.25, lw=0); ax.plot(TX, med, color="purple", lw=1.8,
                                                                           label=f"BEL ({method}): median, 90 %")
    for key, col, lab in (("single_thr3", "darkorange", "linearised 3 Ω·m"), ("single_logmid", "teal", "linearised log-midpoint")):
        b = a4[key]; ok = b[0] >= 0.5
        ax.fill_between(a4["x"], np.where(ok, b[2], np.nan), np.where(ok, b[3], np.nan), color=col, alpha=0.18, lw=0)
        ax.plot(a4["x"], np.where(ok, b[1], np.nan), color=col, lw=1.3, label=f"{lab}: median, 90 %")
    ax.set_xlim(90, 460); ax.set_ylim(-35, 8); ax.set_aspect(3)
    ax.set_xlabel("Distance along profile (m)"); ax.set_ylabel("Elevation (mTAW)")
    ax.text(0.995, 0.02, "V.E. 3×", transform=ax.transAxes, ha="right", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    ax.set_title("Field lens base — BEL posterior vs linearised bands", fontweight="bold")
    wc.white_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    fig.savefig(os.path.join(FIG, "bel_field_lens_base.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
    np.savez(os.path.join(D2, "bel_field_posterior.npz"), x=TX, kde=Pf["kde"], gauss=Pf["gauss"],
             p_dis=p_dis, p_clay=p_clay, rho_dis_neighbours=rd)
    with open(os.path.join(RES, "bel_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("spec", "approx", "falsify", "bel"):
        sub.add_parser(c)
    ps = sub.add_parser("sample"); ps.add_argument("--set", choices=("dual", "train"), required=True)
    ps.add_argument("--workers", type=int, default=4)
    pw = sub.add_parser("worker"); pw.add_argument("--set", choices=("dual", "train"), required=True)
    pw.add_argument("--worker-id", type=int, required=True); pw.add_argument("--ids", type=int, nargs="+", required=True)
    args = ap.parse_args()
    dict(spec=cmd_spec, sample=cmd_sample, worker=cmd_worker, approx=cmd_approx, falsify=cmd_falsify,
         bel=cmd_bel)[args.cmd](args)


if __name__ == "__main__":
    main()
