"""
14_gateA_sensitivity.py
=======================
Westhoek ERT — decisions + Gate A close-out (after Stage 5b).

Subcommands
  tradeoffs   1a  nearest-100 training samples to the field data / T1 r0 in whitened data-PCA space:
                  parameter distributions and rank correlations of base depth with the nuisance parameters
  subsets     1b  BEL-KDE on training subsets that change one assumption at a time (held-out filtered the same way)
  build1c     1c  fine-mesh forward of the controlled truths T1m (ρ_below, ρ_deep at prior medians) and T1d40
                  (deep-unit top at −40 mTAW)
  drive1c/worker  1c  linearised λ = 50 (+ λ pool for r < 5) posterior sampling for the controlled truths
  metrics1c   1c  BEL-KDE vs linearised log-midpoint coverage / bias / width on the controlled truths
  nuisance    2   cross-validated L2 logistic regression for presence variables; second BEL on D1, D2, log ρ
  fwdcheck    3   g1_lam50 model forwarded on the fine mesh vs the inversion's own response

Seeds: BEL posterior default_rng([124000, k]); logistic CV folds default_rng(125000); controlled truths:
noise 1000 + r (Stage-4 seeds), posterior 7 000 000 + 100 000·ti + 100·r + λ with ti = 9 (T1m), 10 (T1d40).
Outputs only in results/bel_v3/, figures/bel_v3/, data/processed/bel_v3/.
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
import warnings

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


b13 = _load("b13", "13_bel_full.py")
c11 = _load("c11", "11_stage4_closeout.py")
sb = c11.sb

D3 = os.path.join(wc.PROCESSED_DIR, "bel_v3")
FIG = os.path.join(wc.FIG_DIR, "bel_v3")
RES = os.path.join(wc.RESULTS_DIR, "bel_v3")
POST_SEED, CV_SEED = 124000, 125000
XREP = (150, 200, 250, 300, 350)
N_POST = 2000
NEW_TRUTHS = {"T1m": 9, "T1d40": 10}


def ensure_dirs():
    for d in (D3, FIG, RES):
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def load_all():
    """Training / held-out data (corrected + noisy), targets, parameters; full-prior BEL."""
    mu_d, dev = b13.approx_stats()
    tr = b13.load_set("train", set(range(b13.N_TRAIN)))
    ho = b13.load_set("train", set(range(b13.N_TRAIN, b13.N_TRAIN + b13.N_HELD)))
    Dtr, ktr = b13.prior_data(tr, mu_d, dev); Dho, kho = b13.prior_data(ho, mu_d, dev)
    zmap = {i: z for i, z in tr + ho}
    th = {i: json.loads(str(zmap[i]["theta"])) for i in ktr + kho}
    Ttr = np.array([zmap[i]["target"] for i in ktr]); Tho = np.array([zmap[i]["target"] for i in kho])
    return dict(Dtr=Dtr, ktr=ktr, Dho=Dho, kho=kho, Ttr=Ttr, Tho=Tho, th=th, zmap=zmap)


PARAMS = ["rho_lens", "rho_below", "rho_deep", "z_deep", "tau", "x_S", "rho_discharge", "rho_aquifer", "x_L",
          "z_wt", "dis_present", "clay_present"]


def pval(t, p):
    v = t[p]
    return float(np.log10(v)) if p.startswith("rho_") else float(v)


def cmd_tradeoffs(args):
    ensure_dirs()
    from scipy.stats import spearmanr
    A = load_all(); bel = b13.BEL(A["Dtr"], A["Ttr"])
    obs, truths = b13.observed_sets()
    var = bel.Sd.var(0, ddof=1)
    PM = np.array([[pval(A["th"][i], p) for p in PARAMS] for i in A["ktr"]])
    L = ["# 1a — trade-offs among the training samples nearest the data", "",
         "Nearest 100 training samples in the whitened data-PCA space (k_d = {}; distance Σ score²/var). ρ in log10. "
         "Rank correlations (Spearman) of the lens-base elevation (mTAW; larger = shallower) with each parameter, "
         "within the 100 neighbours. x_S and ρ_discharge only act when a discharge zone is present.".format(bel.pd.k), ""]
    for name, y in (("field (G1)", obs["field"]), ("T1 realisation 0", truths[("T1", 0)])):
        d = np.sum((bel.Sd - bel.pd.fwd(y)[0]) ** 2 / var, axis=1)
        nn = np.argsort(d)[:100]
        L += [f"## {name}", "", "| parameter | prior median [P10, P90] | neighbours median [P10, P90] |", "|---|---|---|"]
        for j, p in enumerate(PARAMS):
            a, b = PM[:, j], PM[nn, j]
            L.append(f"| {p} | {np.median(a):.2f} [{np.percentile(a, 10):.2f}, {np.percentile(a, 90):.2f}] | "
                     f"{np.median(b):.2f} [{np.percentile(b, 10):.2f}, {np.percentile(b, 90):.2f}] |")
        L += ["", "| base at x | " + " | ".join(PARAMS[:6]) + " |", "|---|" + "---|" * 6]
        for x in (200, 250, 300):
            j = int(np.argmin(np.abs(b13.TX - x)))
            base = A["Ttr"][nn, j]
            cells = []
            for k in range(6):
                rho, pv = spearmanr(base, PM[nn, k])
                cells.append(f"{rho:+.2f}{'*' if pv < 0.05 else ''}")
            L.append(f"| {x} | " + " | ".join(cells) + " |")
        b = A["Ttr"][nn][:, [int(np.argmin(np.abs(b13.TX - x))) for x in (200, 250, 300)]]
        L += ["", f"Neighbours' base at x = 200/250/300: median " + ", ".join(f"{v:.1f}" for v in np.median(b, 0))
              + " mTAW; * p < 0.05.", ""]
    with open(os.path.join(RES, "1a_tradeoffs.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
def field_summary(bel, y, seed):
    P = bel.posterior(y, N_POST, np.random.default_rng([POST_SEED, seed]), "kde")
    return {x: np.percentile(P[:, int(np.argmin(np.abs(b13.TX - x)))], [5, 50, 95]) for x in XREP}


def cmd_subsets(args):
    ensure_dirs()
    A = load_all()
    obs, _ = b13.observed_sets()
    thtr = [A["th"][i] for i in A["ktr"]]; thho = [A["th"][i] for i in A["kho"]]
    zd_med = np.median([t["z_deep"] for t in thtr]); rb_med = np.median([t["rho_below"] for t in thtr])
    rl_med = np.median([t["rho_lens"] for t in thtr])
    subsets = {
        f"deep top above median (> {zd_med:.1f})": lambda t: t["z_deep"] > zd_med,
        f"deep top below median (≤ {zd_med:.1f})": lambda t: t["z_deep"] <= zd_med,
        f"ρ_below lower half (≤ {rb_med:.2f})": lambda t: t["rho_below"] <= rb_med,
        f"ρ_below upper half (> {rb_med:.2f})": lambda t: t["rho_below"] > rb_med,
        f"ρ_lens lower half (≤ {rl_med:.2f})": lambda t: t["rho_lens"] <= rl_med,
        f"ρ_lens upper half (> {rl_med:.2f})": lambda t: t["rho_lens"] > rl_med,
        "transition < 2 m": lambda t: t["tau"] < 2.0,
        "transition > 2 m": lambda t: t["tau"] > 2.0,
        "T1-like (deep top < −30, ρ_below 3–8)": lambda t: t["z_deep"] < -30 and 3 <= t["rho_below"] <= 8,
    }
    full = b13.BEL(A["Dtr"], A["Ttr"])
    base = field_summary(full, obs["field"], 0)
    L = ["# 1b — prior subsets: BEL-KDE retrained on one-assumption subsets", "",
         "Same settings as the full run (data PCA ≥ 99 %, target PCA ≥ 99 %, CCA, KDE in canonical space); PCA/CCA "
         "refitted on each subset. Held-out samples filtered the same way. Field: median [P5, P95] (mTAW) and the shift "
         f"of the median vs the full prior next to the 90 % half-width. Posterior seeds default_rng([{POST_SEED}, k]).", "",
         "Full prior field: " + "; ".join(f"x={x}: {base[x][1]:.2f} [{base[x][0]:.2f}, {base[x][2]:.2f}]" for x in XREP), "",
         "| subset | n train / held-out | held-out coverage | " + " | ".join(f"x={x}: median (shift / half-width)" for x in XREP)
         + " |", "|---|---|---|" + "---|" * len(XREP)]
    for k, (name, fn) in enumerate(subsets.items(), start=1):
        mtr = np.array([fn(t) for t in thtr]); mho = np.array([fn(t) for t in thho])
        bel = b13.BEL(A["Dtr"][mtr], A["Ttr"][mtr])
        cov = []
        for j in np.where(mho)[0]:
            P = bel.posterior(A["Dho"][j], N_POST, np.random.default_rng([POST_SEED, 1000 * k + j]), "kde")
            lo, hi = np.percentile(P, [5, 95], axis=0)
            cov.append((A["Tho"][j] >= lo) & (A["Tho"][j] <= hi))
        fs = field_summary(bel, obs["field"], 100 * k)
        cells = []
        for x in XREP:
            lo, md, hi = fs[x]
            cells.append(f"{md:.2f} ({md - base[x][1]:+.2f} / {0.5*(hi-lo):.2f})")
        L.append(f"| {name} | {mtr.sum()} / {mho.sum()} | {100*np.mean(cov):.1f}% | " + " | ".join(cells) + " |")
        print(L[-1], flush=True)
    with open(os.path.join(RES, "1b_prior_subsets.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
def truth_spec_1c():
    """Stage-4 spec extended with the two controlled truths (in memory; written to bel_v3)."""
    spec = sb.load_spec()
    P = json.load(open(b13.SPEC2, encoding="utf-8"))["params"]
    gm = lambda k: float(np.sqrt(P[k]["range"][0] * P[k]["range"][1]))      # log-uniform median
    spec["values"]["T1m"] = dict(spec["values"]["T1"], brackish=gm("rho_below"), discharge=gm("rho_below"),
                                 deep=gm("rho_deep"))
    spec["values"]["T1d40"] = dict(spec["values"]["T1"])
    spec["variants"]["T1m"] = {}
    spec["variants"]["T1d40"] = {"z_deep": -40.0}
    return spec


class TruthZ(sb.Truth):
    """Stage-4 Truth with an optional deep-unit top override (T1d40)."""

    def __init__(self, spec, name):
        super().__init__(spec, name)
        self.z_deep = spec["variants"].get(name, {}).get("z_deep", sb.Z_DEEP)

    def labels(self, x, z):
        old = sb.Z_DEEP
        sb.Z_DEEP = self.z_deep
        try:
            return super().labels(x, z)
        finally:
            sb.Z_DEEP = old


def cmd_build1c(args):
    ensure_dirs()
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    spec = truth_spec_1c()
    json.dump({k: spec["values"][k] for k in NEW_TRUTHS} | {"variants": {k: spec["variants"][k] for k in NEW_TRUTHS}},
              open(os.path.join(D3, "truths_1c.json"), "w"), indent=1)
    sch = sb.g1_scheme(); scheme = sb.make_scheme(sch); scheme["k"] = np.ones(scheme.size())
    fmesh = mt.createParaMesh(scheme.sensors(), **sb.FWD)
    cx = np.array([c.center().x() for c in fmesh.cells()]); cz = np.array([c.center().y() for c in fmesh.cells()])
    out = {}
    for t in NEW_TRUTHS:
        tr = TruthZ(spec, t)
        with contextlib.redirect_stdout(io.StringIO()):
            out[f"R_{t}"] = b13.pilot.simulate_R(fmesh, 10 ** tr.log10rho(cx, cz), scheme.size())
        tv = interfaces_truth(tr)
        out[f"logmid_{t}"], out[f"thr3_{t}"] = tv["logmid"], tv["thr3"]
        print(f"[build1c] {t}: values {spec['values'][t]}, variant {spec['variants'][t]}; R [{out[f'R_{t}'].min():.3g}, "
              f"{out[f'R_{t}'].max():.3g}]; true base x=150/250/350: "
              + ", ".join(f"{tv['logmid'][int(np.argmin(np.abs(sb.COLS - x)))]:.2f}" for x in (150, 250, 350)))
    np.savez(os.path.join(D3, "truths_1c_forward.npz"), cols=sb.COLS, **out)


def interfaces_truth(tr, dz=0.02):
    ex, ez = wc.land_surface()
    s = np.interp(sb.COLS, ex, ez)
    z = np.arange(np.floor(ez.max() / dz) * dz, wc.Z_FLOOR - 1e-9, -dz)
    X, Zg = np.meshgrid(sb.COLS, z)
    V = tr.log10rho(X, Zg); V[Zg > s[None, :] - 1e-6] = np.nan
    r = c11.interfaces_all(V[:, :, None], z)
    return {d: r[d][:, 0] for d in ("thr3", "logmid")}


def job_1c(m02, fwd, sch, cell_x, cs, truth, real, log):
    out = os.path.join(D3, "runs_1c", f"{truth}_r{real:02d}.npz")
    if os.path.exists(out):
        return
    t0 = time.time()
    R_true = fwd[f"R_{truth}"]
    rng = np.random.default_rng(1000 + real)
    sig = np.sqrt((sb.ERR_REL * R_true) ** 2 + sb.ERR_ABS ** 2)
    R = R_true + sig * rng.standard_normal(len(R_true)); keep = R > 0
    elec = {int(e): (float(x), float(z)) for e, x, z in zip(sch["elec_num"], sch["elec_x"], sch["elec_z"])}
    land = dict({k: sch[k][keep] for k in ("C1", "C2", "P1", "P2")}, R=R[keep], n=int(keep.sum()), err=np.zeros(int(keep.sum())))
    lams = [50.0] + ([5.0, 10.0, 20.0, 100.0] if real < 5 else [])
    save = dict(truth=truth, real=real, cols=cs.x, nonpositive=int((~keep).sum())); pooled = []
    ti = NEW_TRUTHS[truth]
    for lam in lams:
        info, S = sb.one_inversion(m02, land, elec, lam, cell_x, cs, 7_000_000 + 100_000 * ti + 100 * real + int(lam))
        hat = c11.interfaces_all(cs(np.log10(info["model"].astype(float)))[:, :, None], cs.z)
        bands = c11.sample_defs(cs, S)
        p = f"lam{int(lam)}_"
        save.update({p + "chi2": info["chi2"], p + "stop": info["stop"]})
        for d in ("thr3", "logmid"):
            save[p + f"mhat_{d}"] = hat[d][:, 0]; save[p + f"band_{d}"] = bands[d]
            save[p + f"r_{d}"] = c11.r_at(cs, info["radius"], hat[d][:, 0])
        if real < 5:
            pooled.append(S.astype(np.float32))
        log(f"1c {truth} r{real} λ={lam:g}: χ²={info['chi2']:.3f} ({info['stop']})")
    if pooled:
        pb = c11.sample_defs(cs, np.hstack(pooled).astype(np.float64))
        for d in ("thr3", "logmid"):
            save[f"pooled_band_{d}"] = pb[d]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out + ".tmp.npz", **save); os.replace(out + ".tmp.npz", out)
    log(f"done 1c {truth} r{real} in {time.time()-t0:.0f}s")


def cmd_worker(args):
    fwd = np.load(os.path.join(D3, "truths_1c_forward.npz"))
    sch = sb.g1_scheme(); ref = wc.load_run(sb.REF_TAG)
    mesh, _ = wc.rebuild_paradomain(ref); cs = wc.ColumnSampler(mesh, sb.COLS)
    m02 = _load("inv02", "02_land_inversion.py"); m02.VERBOSE = False
    logf = open(os.path.join(D3, "logs", f"1c_worker_{args.worker_id}.log"), "a", encoding="utf-8")
    log = lambda m: (logf.write(time.strftime("%H:%M:%S ") + m + "\n"), logf.flush())
    for job in args.jobs:
        t, r = job.split(":")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                job_1c(m02, fwd, sch, np.asarray(ref["cell_x"]), cs, t, int(r), log)
        except Exception as e:
            import traceback
            log(f"FAILED {job}: {e}\n{traceback.format_exc()}")
    log("worker finished")


def cmd_drive1c(args):
    os.makedirs(os.path.join(D3, "logs"), exist_ok=True)
    jobs = [f"{t}:{r}" for r in range(20) for t in NEW_TRUTHS
            if not os.path.exists(os.path.join(D3, "runs_1c", f"{t}_r{r:02d}.npz"))]
    jobs.sort(key=lambda j: (int(j.split(":")[1]) >= 5, int(j.split(":")[1])))
    parts = [jobs[i::args.workers] for i in range(args.workers)]
    procs = []
    for w, part in enumerate(parts):
        if not part:
            continue
        wd = os.path.join(D3, "work", f"w{w}"); os.makedirs(wd, exist_ok=True)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", "--worker-id", str(w), "--jobs"]
                                      + part, cwd=wd, env=dict(os.environ, OMP_NUM_THREADS="2"), stdout=subprocess.DEVNULL,
                                      stderr=open(os.path.join(D3, "logs", f"1c_worker_{w}.err"), "a")))
    for p in procs:
        p.wait()
    print("[drive1c] finished")


def cmd_metrics1c(args):
    ensure_dirs()
    A = load_all(); bel = b13.BEL(A["Dtr"], A["Ttr"])
    fwd = np.load(os.path.join(D3, "truths_1c_forward.npz"))
    jj = [int(np.argmin(np.abs(sb.COLS - x))) for x in b13.TX]
    L = ["# 1c — controlled truths: BEL-KDE vs linearised log-midpoint", "",
         "T1m = T1 with ρ_below (and the discharge-zone water) and ρ_deep at the prior-v2 medians (geometric means of the "
         "log-uniform ranges); T1d40 = T1 with the deep-unit top at −40 mTAW. Fine forward mesh, Stage-4 noise seeds "
         "1000 + r, 20 realisations. Truth = base-transition centre (log-midpoint truth). Columns x = 130–450 m "
         "(headline range). BEL-KDE with the full v2 prior (posterior seeds default_rng([124000, 5000 + 100·t + r])); "
         "linearised λ = 50 and λ-pooled (r < 5) log-midpoint bands. Cells: coverage / bias (m) / width (m).", "",
         "| truth | subset | n | BEL-KDE | lin. log-mid λ=50 | lin. log-mid pooled λ (n) |", "|---|---|---|---|---|---|"]
    ref = {}
    for ti, t in enumerate(NEW_TRUTHS):
        tv = fwd[f"logmid_{t}"][jj]
        rows = []
        for r in range(20):
            p = os.path.join(D3, "runs_1c", f"{t}_r{r:02d}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p, allow_pickle=True)
            R = fwd[f"R_{t}"]; rng = np.random.default_rng(1000 + r)
            Rn = R + np.sqrt((sb.ERR_REL * R) ** 2 + sb.ERR_ABS ** 2) * rng.standard_normal(len(R))
            if not np.all(Rn > 0):
                continue
            P = bel.posterior(np.log10(Rn), N_POST, np.random.default_rng([POST_SEED, 5000 + 100 * ti + r]), "kde")
            lo, med, hi = np.percentile(P, [5, 50, 95], axis=0)
            for k, x in enumerate(b13.TX):
                if x < 130:
                    continue
                j = jj[k]
                b = d["lam50_band_logmid"][:, j]; mh = d["lam50_mhat_logmid"][j]
                row = dict(r_ok=bool(np.isfinite(d["lam50_r_logmid"][j]) and d["lam50_r_logmid"][j] <= 10),
                           bel=(lo[k] <= tv[k] <= hi[k], med[k] - tv[k], hi[k] - lo[k]),
                           lin=((b[0] >= 0.5) and (b[2] <= tv[k] <= b[3]), mh - tv[k], b[3] - b[2] if b[0] >= 0.5 else np.nan))
                if "pooled_band_logmid" in d.files:
                    pb = d["pooled_band_logmid"][:, j]
                    row["pool"] = ((pb[0] >= 0.5) and (pb[2] <= tv[k] <= pb[3]), mh - tv[k], pb[3] - pb[2] if pb[0] >= 0.5 else np.nan)
                rows.append(row)
        for sub in ("all", "r ≤ 10"):
            rs = rows if sub == "all" else [x for x in rows if x["r_ok"]]
            if not rs:
                continue
            c = lambda k: (f"{100*np.mean([x[k][0] for x in rs if k in x]):.0f}% / "
                           f"{np.nanmedian([x[k][1] for x in rs if k in x]):+.2f} / {np.nanmedian([x[k][2] for x in rs if k in x]):.2f}")
            npool = sum(1 for x in rs if "pool" in x)
            L.append(f"| {t} | {sub} | {len(rs)} | {c('bel')} | {c('lin')} | {c('pool')} ({npool}) |")
    L += ["", "Reference (Stage 5b, same metric, all columns 100–450 m): T1 BEL-KDE 88 % / −3.13 / 9.20; "
          "T1 lin. log-mid 99 % / +0.75 / 9.26."]
    with open(os.path.join(RES, "1c_controlled_truths.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
def logistic_cv(X, y, Cs=(0.001, 0.01, 0.1, 1.0, 10.0), k=5, seed=CV_SEED):
    """L2 logistic regression; C chosen by k-fold CV log-loss. Returns predict function and chosen C."""
    from scipy.optimize import minimize
    def fit(X, y, C):
        Xb = np.hstack([X, np.ones((len(X), 1))])
        def f(w):
            z = Xb @ w; p = 1 / (1 + np.exp(-z))
            ll = -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            reg = 0.5 / C * np.sum(w[:-1] ** 2)
            g = Xb.T @ (p - y); g[:-1] += w[:-1] / C
            return ll + reg, g
        return minimize(f, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B").x
    pred = lambda w, X: 1 / (1 + np.exp(-(np.hstack([X, np.ones((len(X), 1))]) @ w)))
    rng = np.random.default_rng(seed); fold = rng.integers(0, k, len(y))
    losses = {}
    for C in Cs:
        l = 0.0
        for f_ in range(k):
            tr, te = fold != f_, fold == f_
            w = fit(X[tr], y[tr], C); p = np.clip(pred(w, X[te]), 1e-9, 1 - 1e-9)
            l += -np.sum(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
        losses[C] = l / len(y)
    Cb = min(losses, key=losses.get)
    w = fit(X, y, Cb)
    return (lambda Xn: pred(w, np.atleast_2d(Xn))), Cb, losses


def derived_targets(t):
    """D1, D2 and log10 resistivities for one prior sample (evaluated on its own model)."""
    pm = b13.PriorModel2(t)
    ex, ez = wc.land_surface()
    def below_window(xs):
        vals = []
        for x in xs:
            tb = float(pm.target(np.array([x]))[0])
            zz = np.linspace(tb - 7.0, tb - 3.0, 9)
            vals.append(pm.log10rho(np.full_like(zz, x), zz).mean())
        return float(np.mean(vals))
    D1 = below_window(np.arange(380, 471, 10.0)) - below_window(np.arange(200, 301, 10.0))
    X, Z = np.meshgrid(np.arange(25, 91, 2.5), np.arange(-12, -2.99, 0.25))
    V = pm.log10rho(X, Z); V[Z > np.interp(X, ex, ez)] = np.nan
    D2 = float(np.nanmin(V))
    return np.array([D1, D2, np.log10(t["rho_lens"]), np.log10(t["rho_below"]), np.log10(t["rho_discharge"])])


def cmd_nuisance(args):
    ensure_dirs()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = load_all(); bel = b13.BEL(A["Dtr"], A["Ttr"])
    obs, _ = b13.observed_sets()
    sd = np.sqrt(bel.Sd.var(0, ddof=1))
    Xtr = bel.Sd / sd; Xho = bel.pd.fwd(A["Dho"]) / sd; xf = bel.pd.fwd(obs["field"]) / sd
    thtr = [A["th"][i] for i in A["ktr"]]; thho = [A["th"][i] for i in A["kho"]]
    targets = {"discharge zone present": lambda t: t["dis_present"], "dune conductive layer present": lambda t: t["clay_present"]}
    for x in (100, 110, 120):
        targets[f"lens present at x = {x} m"] = (lambda xx: (lambda t: t["x_L"] <= xx))(x)
    L = ["# 2 — nuisance features", "", "## 2a — presence probabilities (cross-validated L2 logistic regression)", "",
         f"Features: the k_d = {bel.pd.k} whitened data-PC scores. C chosen by 5-fold CV log-loss (folds seed {CV_SEED}). "
         "Brier score on the held-out set vs the no-skill value p(1−p) for the held-out base rate p.", "",
         "| variable | C | held-out Brier | no-skill p(1−p) | skill (1 − Brier/no-skill) | field P |", "|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4.2))
    for ax, (name, fn) in zip(axes, targets.items()):
        ytr = np.array([float(fn(t)) for t in thtr]); yho = np.array([float(fn(t)) for t in thho])
        predict, C, _ = logistic_cv(Xtr, ytr)
        ph = predict(Xho); pf = float(predict(xf)[0])
        brier = float(np.mean((ph - yho) ** 2)); p0 = yho.mean(); ns = p0 * (1 - p0)
        L.append(f"| {name} | {C:g} | {brier:.3f} | {ns:.3f} | {1 - brier/ns:+.2f} | {pf:.2f} |")
        bins = np.linspace(0, 1, 6); idx = np.clip(np.digitize(ph, bins) - 1, 0, 4)
        mp = [ph[idx == i].mean() if (idx == i).any() else np.nan for i in range(5)]
        ob = [yho[idx == i].mean() if (idx == i).any() else np.nan for i in range(5)]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.plot(mp, ob, "o-", color="#1F77B4")
        for a, b, n in zip(mp, ob, [(idx == i).sum() for i in range(5)]):
            if np.isfinite(a):
                ax.annotate(str(n), (a, b), textcoords="offset points", xytext=(4, -10), fontsize=7)
        ax.axvline(pf, color="red", lw=1.2, ls=":", label=f"field P = {pf:.2f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.set_title(f"{name}\nBrier {brier:.3f} (no-skill {ns:.3f})", fontsize=9)
        ax.set_xlabel("predicted P"); ax.set_ylabel("observed frequency"); wc.white_legend(ax, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "2a_presence_reliability.png"), dpi=150); plt.close(fig)

    # 2b — second BEL on continuous nuisance targets
    Ytr = np.array([derived_targets(t) for t in thtr]); Yho = np.array([derived_targets(t) for t in thho])
    names = ["D1 (discharge water − below-lens water, log10)", "D2 (min log10ρ, x 25–90, −3…−12 mTAW)",
             "log10 ρ_lens", "log10 ρ_below", "log10 ρ_discharge"]
    ym, ys = Ytr.mean(0), Ytr.std(0)
    cca = b13.CCA(bel.Sd, (Ytr - ym) / ys)
    Cd = cca.cx(bel.Sd); Ct = cca.cy((Ytr - ym) / ys)
    def post(y, rng):
        cd = cca.cx(bel.pd.fwd(y))[0]
        C = np.column_stack([b13.kde_conditional_sample(Cd[:, i], Ct[:, i], cd[i], N_POST, rng) for i in range(Ct.shape[1])])
        return cca.y_from_c(C) * ys + ym
    cov = []
    for j, y in enumerate(A["Dho"]):
        P = post(y, np.random.default_rng([POST_SEED, 30_000 + j])); lo, hi = np.percentile(P, [5, 95], axis=0)
        cov.append((Yho[j] >= lo) & (Yho[j] <= hi))
    cov = np.array(cov)
    Pf = post(obs["field"], np.random.default_rng([POST_SEED, 30_999]))
    L += ["", "## 2b — second BEL on continuous nuisance targets", "",
          "Same data PCA (k_d = {}); CCA between the data scores and the 5 standardised targets; KDE in canonical space. "
          "D1 = mean log10ρ 3–7 m below the lens/skin base at x 380–470 m minus the same at x 200–300 m (the unit beneath "
          "the saline body: discharge water where the zone is present, otherwise the water below the lens). "
          "D2 = minimum log10ρ in x 25–90 m between −3 and −12 mTAW. Canonical correlations: {}.".format(
              bel.pd.k, ", ".join(f"{r:.2f}" for r in cca.r)), "",
          "| target | held-out 90 % coverage | prior P5–P95 | field median [P5, P95] |", "|---|---|---|---|"]
    for i, n in enumerate(names):
        L.append(f"| {n} | {100*cov[:, i].mean():.1f}% | {np.percentile(Ytr[:, i], 5):.2f} – {np.percentile(Ytr[:, i], 95):.2f} | "
                 f"{np.median(Pf[:, i]):.2f} [{np.percentile(Pf[:, i], 5):.2f}, {np.percentile(Pf[:, i], 95):.2f}] |")
    L += ["", f"- P(D1 > 0 | field) = {np.mean(Pf[:, 0] > 0):.2f}  (prior {np.mean(Ytr[:, 0] > 0):.2f}) — the water beneath "
          "the saline body at x 380–470 m is more resistive than beneath the lens",
          f"- P(10^D2 < 3 Ω·m | field) = {np.mean(10 ** Pf[:, 1] < 3):.2f}  (prior {np.mean(10 ** Ytr[:, 1] < 3):.2f}) — a "
          "saline-grade conductive body under the dunes",
          f"- In Ω·m: ρ_lens {10**np.median(Pf[:, 2]):.2f} [{10**np.percentile(Pf[:, 2], 5):.2f}, {10**np.percentile(Pf[:, 2], 95):.2f}], "
          f"ρ_below {10**np.median(Pf[:, 3]):.2f} [{10**np.percentile(Pf[:, 3], 5):.2f}, {10**np.percentile(Pf[:, 3], 95):.2f}], "
          f"ρ_discharge {10**np.median(Pf[:, 4]):.2f} [{10**np.percentile(Pf[:, 4], 5):.2f}, {10**np.percentile(Pf[:, 4], 95):.2f}] "
          "(ρ_discharge acts only where a discharge zone exists — marginalised over presence)."]
    np.savez(os.path.join(D3, "nuisance_field_posterior.npz"), targets=np.array(names), field=Pf)
    with open(os.path.join(RES, "2_nuisance.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
def cmd_fwdcheck(args):
    ensure_dirs()
    import pygimli as pg
    import pygimli.meshtools as mt
    ref = wc.load_run("g1_lam50")
    pd_, _ = wc.rebuild_paradomain(ref)
    scheme = b13.pilot.build_scheme()
    fmesh = mt.createParaMesh(scheme.sensors(), **sb.FWD)
    # model onto the fine mesh: containing paraDomain cell, else nearest cell (as pyGIMLi's prolongation)
    from scipy.spatial import cKDTree
    pcx, pcz = ref["cell_x"], ref["cell_z"]
    tree = cKDTree(np.column_stack([pcx, pcz]))
    vals = np.empty(fmesh.cellCount())
    for c in fmesh.cells():
        pc = c.center(); cell = pd_.findCell(pc)
        vals[c.id()] = ref["model"][cell.id()] if cell is not None else ref["model"][tree.query([pc.x(), pc.y()])[1]]
    with contextlib.redirect_stdout(io.StringIO()):
        R_fine = b13.pilot.simulate_R(fmesh, vals, scheme.size())
    R_inv = ref["d_pred"] / ref["K"]            # inversion response in resistance units
    z = (np.log(R_fine) - np.log(R_inv)) / ref["d_err"]
    xo = dict(zip(ref["elec_num"].astype(int), ref["elec_x"]))
    xs = {k: np.array([xo[e] for e in ref[k].astype(int)]) for k in ("C1", "C2", "P1", "P2")}
    mid = sum(xs.values()) / 4; dip = np.abs(xs["P1"] - xs["P2"])
    az = np.abs(z); frac = float(np.mean(az > 1))
    from scipy.stats import spearmanr
    L = ["# 3 — forward accuracy of the inversion mesh (g1_lam50)", "",
         "The g1_lam50 model mapped onto the fine forward mesh (paraDX 0.2, max cell 5 m², 17 684 cells; containing cell, "
         "else nearest) and forwarded; compared with the inversion's own response (its refined forward mesh) for the "
         f"{len(z)} G1 data. z = (ln R_fine − ln R_inv) / err (err = 3 % + 1e-4 Ω, relative).", "",
         f"- |z|: median {np.median(az):.2f}, P90 {np.percentile(az, 90):.2f}, P95 {np.percentile(az, 95):.2f}, max {az.max():.2f}; "
         f"mean z {z.mean():+.2f}",
         f"- share with |z| > 1: **{100*frac:.1f} %** → " + ("**FLAG: > 5 %**" if frac > 0.05 else "below the 5 % flag"),
         f"- |z| vs potential-dipole length: Spearman {spearmanr(az, dip)[0]:+.2f}; vs midpoint x: {spearmanr(az, mid)[0]:+.2f}"]
    big = az > 1
    if big.any():
        L.append(f"- data with |z| > 1: midpoint x median {np.median(mid[big]):.0f} m (range {mid[big].min():.0f}–{mid[big].max():.0f}), "
                 f"dipole length median {np.median(dip[big]):.0f} m (all data: {np.median(dip):.0f} m)")
    for lo, hi in ((0, 90), (90, 200), (200, 370), (370, 510)):
        s = (mid >= lo) & (mid < hi)
        L.append(f"  - midpoint x {lo}–{hi} m: n = {s.sum()}, share |z| > 1 = {100*np.mean(az[s] > 1):.1f} %, median |z| {np.median(az[s]):.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(np.clip(z, -5, 5), bins=60, color="#1F77B4"); axes[0].axvline(-1, color="r", ls=":"); axes[0].axvline(1, color="r", ls=":")
    axes[0].set_xlabel("(ln R_fine − ln R_inv) / noise"); axes[0].set_title("fine vs inversion-mesh response", fontsize=10)
    sc = axes[1].scatter(mid, -0.2 * np.maximum(np.abs(xs["C2"] - xs["C1"]), dip), c=np.clip(z, -3, 3), cmap="RdBu_r", s=10,
                         vmin=-3, vmax=3)
    fig.colorbar(sc, ax=axes[1], label="z (clipped ±3)"); axes[1].set_xlabel("midpoint x (m)"); axes[1].set_ylabel("pseudo-depth (m)")
    axes[1].set_title("where the differences sit", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_forward_accuracy.png"), dpi=150); plt.close(fig)
    with open(os.path.join(RES, "3_forward_accuracy.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("tradeoffs", "subsets", "build1c", "metrics1c", "nuisance", "fwdcheck"):
        sub.add_parser(c)
    pd = sub.add_parser("drive1c"); pd.add_argument("--workers", type=int, default=4)
    pw = sub.add_parser("worker"); pw.add_argument("--worker-id", type=int, required=True); pw.add_argument("--jobs", nargs="+")
    args = ap.parse_args()
    dict(tradeoffs=cmd_tradeoffs, subsets=cmd_subsets, build1c=cmd_build1c, drive1c=cmd_drive1c, worker=cmd_worker,
         metrics1c=cmd_metrics1c, nuisance=cmd_nuisance, fwdcheck=cmd_fwdcheck)[args.cmd](args)


if __name__ == "__main__":
    main()
