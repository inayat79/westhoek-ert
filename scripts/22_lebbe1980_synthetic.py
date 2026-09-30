"""
22_lebbe1980_synthetic.py (renamed from 19_lebbe1980_synthetic.py in v1.1.1)
=========================
Westhoek ERT v1.1 — Task 1: synthetic test with the 1980 K1 borehole-log structure (Lebbe 1981) as truth.

  run [--workers 4] [--out ROOT]   truths → fine-mesh forward (G1 quadrupoles) → noisy data → v1.0 pipeline
                                   (inversions λ = 50 and the λ pool, linearised bands, falsification, base-only BEL,
                                   second BEL, joint BEL + salinity) → ROOT/data/processed/v1_1/lebbe1980/
  report [--out ROOT]              ROOT/results/v1_1/1_lebbe1980_synthetic.md, figs/F_S1_lebbe1980_synthetic.png,
                                   ROOT/data/processed/v1_1/lebbe1980_summary.json (read by 18_paper.py)
  worker                           (internal)

Truths (pre-registered in docs/decision_log.md): T80a base case; T80b no transition; T80c fresh 30 Ω·m; T80d fresh
100 Ω·m; T80e saline base and fresh top shifted by +30 m in x; T80a_ht = T80a data + the field's residual pattern.
Post hoc robustness checks (v1.1.1, docs/decision_log.md v1.1.1-0): T80f = T80a with the clay at 4 Ω·m; T80g = T80a with
the fresh top 5 m deeper and the brackish transition extended down to it.
Boundaries from data/external/lebbe1981_K1_digitised.csv, linear in x between control points, constant beyond.
Seeds: noise default_rng([134000, truth_index, seed]); heavy-tail residual resampling default_rng([134500, seed]);
linearised posterior samples default_rng([134050, truth_index, seed, λ]); BEL posteriors default_rng([134100 + k,
truth_index, seed]) with k = 0 base-only, 1 second BEL, 2 joint BEL, 3 Archie draws; true-salinity Archie draws
default_rng([134200, truth_index]). Truth indices: T80a 0, T80b 1, T80c 2, T80d 3, T80e 4, T80a_ht 5, T80f 6, T80g 7.
Pipeline functions reused unchanged: 11_stage4_closeout.a2_job (keyword arguments added in v1.1 with v1.0 defaults),
10_synthetic_benchmark (G1 scheme, fine mesh FWD, error model), 13_bel_full.BEL, 15_stage5_closeout.BEL2,
14_gateA_sensitivity.derived_targets, stage6_analysis (site + physical set, JointBEL), 17_salinity_v2 (Archie v2,
per-draw rejection, unit statistics).
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import read_trace  # noqa: E402
read_trace.install("22_lebbe1980_synthetic.py")
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


c11 = _load("c11", "11_stage4_closeout.py")
sb = c11.sb
k1 = _load("k1c", "21_k1_comparison.py")

TRUTHS = ["T80a", "T80b", "T80c", "T80d", "T80e", "T80a_ht", "T80f", "T80g"]     # append only: indices seed the noise
CLAY_T80F = 4.0
TI = {t: i for i, t in enumerate(TRUTHS)}
N_SEED = 5
LAMS = [50.0, 5.0, 10.0, 20.0, 100.0]
XREP = (150, 200, 250, 300, 350)
RHO = dict(unsat=300.0, dune=50.0, dune_edge=6.0, lens=1.24, trans=6.0, fresh=50.0, lower=6.0)
FRESH = {"T80c": 30.0, "T80d": 100.0}
Z_WT, X_DUNE, X_EDGE = 4.5, 85.0, 70.0
N_POST = 2000


def dirs(root):
    d = os.path.join(root, "data", "processed", "v1_1", "lebbe1980")
    res = os.path.join(root, "results", "v1_1")
    for p in (d, os.path.join(d, "runs"), os.path.join(d, "logs"), os.path.join(d, "work"), res, os.path.join(res, "figs")):
        os.makedirs(p, exist_ok=True)
    return d, res


# ─────────────────────────────────────────────────────────────────────────────
# truth models
# ─────────────────────────────────────────────────────────────────────────────
def clay_rho():
    """Deep-unit resistivity of the v1.0 reference model (g1_lam50 deep-zone median, Stage-4 truth spec)."""
    return float(sb.load_spec()["values"]["T1"]["deep"])


def boundaries(truth, x):
    lb = k1.read_lebbe(); xs = lb["x_ours_m"]
    sal = lb["saline_base_mTAW"]; clay = lb["clay_top_mTAW"]
    ft = np.where(np.isfinite(lb["fresh_top_mTAW"]), lb["fresh_top_mTAW"], sal)          # fresh zone absent → at sal
    lbt = np.where(np.isfinite(lb["lower_brackish_top_mTAW"]), lb["lower_brackish_top_mTAW"], clay)   # absent → clay
    shift = 30.0 if truth.startswith("T80e") else 0.0
    b = dict(sal=np.interp(x - shift, xs, sal), ft=np.interp(x - shift, xs, ft), lbt=np.interp(x, xs, lbt),
             clay=np.interp(x, xs, clay))
    if truth.startswith("T80b"):
        b["ft"] = b["sal"].copy()
    if truth.startswith("T80g"):                    # fresh top 5 m deeper, transition extended; never below the clay
        b["ft"] = np.maximum(b["ft"] - 5.0, b["clay"])
    b["ft"] = np.minimum(b["ft"], b["sal"]); b["lbt"] = np.minimum(b["lbt"], b["ft"])
    return b


def log10rho(truth, x, z):
    x = np.asarray(x, float); z = np.asarray(z, float)
    b = boundaries(truth, x)
    fresh = FRESH.get(truth.split("_")[0], RHO["fresh"])
    r = np.full(x.shape, fresh)
    r = np.where(z > b["sal"], RHO["lens"], r)
    r = np.where((z <= b["sal"]) & (z > b["ft"]), RHO["trans"], r)
    r = np.where((z <= b["lbt"]) & (z > b["clay"]), RHO["lower"], r)
    dune = x < X_DUNE
    r = np.where(dune & (x < X_EDGE), RHO["dune"], np.where(dune, RHO["dune_edge"], r))
    r = np.where(dune & (z > Z_WT), RHO["unsat"], r)
    r = np.where(z <= b["clay"], CLAY_T80F if truth.startswith("T80f") else clay_rho(), r)
    return np.log10(r)


def true_below(truth, x):
    """Thickness-weighted mean log10 ρ between the saline base and the clay top."""
    b = boundaries(truth, np.atleast_1d(np.asarray(x, float)))
    fresh = np.log10(FRESH.get(truth.split("_")[0], RHO["fresh"]))
    th = [b["sal"] - b["ft"], b["ft"] - b["lbt"], b["lbt"] - b["clay"]]
    lv = [np.log10(RHO["trans"]), fresh, np.log10(RHO["lower"])]
    return sum(t * v for t, v in zip(th, lv)) / sum(th)


def forward_truths(d):
    p = os.path.join(d, "truth_forward.npz")
    need = [t for t in TRUTHS if not t.endswith("_ht")]
    old = dict(np.load(p)) if os.path.exists(p) else {}
    if all(f"R_{t}" in old for t in need):
        return np.load(p)
    import pygimli.meshtools as mt
    sch = sb.g1_scheme(); scheme = sb.make_scheme(sch); scheme["k"] = np.ones(scheme.size())
    b13 = _load("b13", "13_bel_full.py")
    with contextlib.redirect_stdout(io.StringIO()):
        fmesh = mt.createParaMesh(scheme.sensors(), **sb.FWD)
    cx = np.array([c.center().x() for c in fmesh.cells()]); cz = np.array([c.center().y() for c in fmesh.cells()])
    out = dict(cx=cx, cz=cz, clay_rho=clay_rho())
    out.update(old)                                                   # existing truths kept bit-identical
    for t in need:
        if f"R_{t}" in out:
            continue
        v = log10rho(t, cx, cz)
        with contextlib.redirect_stdout(io.StringIO()):
            out[f"R_{t}"] = b13.pilot.simulate_R(fmesh, 10 ** v, scheme.size())
        out[f"cell_log10_{t}"] = v.astype(np.float32)
        print(f"[forward] {t}", flush=True)
    np.savez(p, **out)
    return np.load(p)


# ─────────────────────────────────────────────────────────────────────────────
# noise and inversion jobs
# ─────────────────────────────────────────────────────────────────────────────
def gaussian(R, truth, seed):
    rng = np.random.default_rng([134000, TI[truth], seed])
    sig = np.sqrt((sb.ERR_REL * R) ** 2 + sb.ERR_ABS ** 2)
    return R + sig * rng.standard_normal(len(R))


def heavy_tail(R, seed):
    """Gaussian noise (as T80a) plus the reference inversion's standardised residuals, resampled with replacement
    among the data that share the P1 electrode, applied in log space: ln R += err_i · r*_i."""
    ref = wc.load_run("g1_lam50"); g1 = wc.load_run(sb.G1_TAG)
    assert np.array_equal(ref["keep_idx"], g1["keep_idx"]), "g1_lam50 and the G1 scheme must share the data order"
    r = (np.log(ref["d_obs"]) - np.log(ref["d_pred"])) / ref["d_err"]
    p1 = ref["P1"].astype(int); rng = np.random.default_rng([134500, seed])
    rstar = np.empty_like(r)
    for e in np.unique(p1):
        i = np.where(p1 == e)[0]
        rstar[i] = r[i][rng.integers(0, len(i), len(i))]
    return gaussian(R, "T80a", seed) * np.exp(ref["d_err"] * rstar)


def cmd_worker(args):
    d, _ = dirs(args.out)
    fwd = forward_truths(d)
    spec = sb.load_spec(); sch = sb.g1_scheme()
    ref = wc.load_run(sb.REF_TAG); mesh, _ = wc.rebuild_paradomain(ref); cs = wc.ColumnSampler(mesh, sb.COLS)
    m02 = c11._load("inv02", "02_land_inversion.py"); m02.VERBOSE = False
    logf = open(os.path.join(d, "logs", f"worker_{args.worker_id}.log"), "a", encoding="utf-8")
    log = lambda m: (logf.write(time.strftime("%H:%M:%S ") + m + "\n"), logf.flush())
    for job in args.jobs:
        t, s = job.split(":"); s = int(s)
        R_true = fwd["R_T80a"] if t == "T80a_ht" else fwd[f"R_{t}"]
        noise = (lambda R, s=s: heavy_tail(R, s)) if t == "T80a_ht" else (lambda R, t=t, s=s: gaussian(R, t, s))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c11.a2_job(m02, spec, None, sch, np.asarray(ref["cell_x"]), cs, t, s, log, R_true=R_true, noise=noise,
                           out=os.path.join(d, "runs", f"{t}_s{s}.npz"), lams=LAMS,
                           post_seed=lambda lam, t=t, s=s: [134050, TI[t], s, int(lam)], save_extra=True)
        except Exception as e:
            import traceback
            log(f"FAILED {job}: {e}\n{traceback.format_exc()}")
    log("worker finished")


def run_inversions(d, workers):
    jobs = [f"{t}:{s}" for s in range(N_SEED) for t in TRUTHS if not os.path.exists(os.path.join(d, "runs", f"{t}_s{s}.npz"))]
    parts = [jobs[i::workers] for i in range(workers)]
    procs = []
    for w, part in enumerate(parts):
        if not part:
            continue
        wd = os.path.join(d, "work", f"w{w}"); os.makedirs(wd, exist_ok=True)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", "--worker-id", str(w), "--out",
                                       ARGS_OUT, "--jobs"] + part, cwd=wd, env=dict(os.environ, OMP_NUM_THREADS="2"),
                                      stdout=subprocess.DEVNULL, stderr=open(os.path.join(d, "logs", f"worker_{w}.err"), "a")))
    print(f"[run] {len(jobs)} inversion jobs on {len(procs)} workers", flush=True)
    for p in procs:
        p.wait()


# ─────────────────────────────────────────────────────────────────────────────
# BEL analyses of every synthetic data set
# ─────────────────────────────────────────────────────────────────────────────
def bel_analyses(d):
    import stage6_analysis as s6
    s15 = _load("s15", "15_stage5_closeout.py"); g14, b13 = s15.g14, s15.b13
    s17 = _load("s17", "17_salinity_v2.py")
    s6.init(); s6.TX = b13.TX; TX = b13.TX
    C = s6.load_combined(physical=True)
    D, T, th = C["tr"]["D"], C["tr"]["T"], C["tr"]["th"]
    bel = b13.BEL(D, T)
    sc = bel.Sd; var = sc.var(0, ddof=1); mu, Cm = b13.pilot.fast_mcd(sc); Ci = np.linalg.inv(Cm)
    t2tr = np.sum(sc ** 2 / var, 1); t2rtr = np.einsum("ij,jk,ik->i", sc - mu, Ci, sc - mu); qtr = bel.pd.resid(D)
    Ytr = np.array([g14.derived_targets(t) for t in th])
    s15.RHO_BOUNDS.update(rho_lens=(0.4, 3.0), rho_below=(2.0, 20.0), rho_discharge=(2.5, 150.0))
    b2 = s15.BEL2(D, Ytr, bounded=True)
    J = s6.JointBEL(D, T, th)
    ex, ez = wc.land_surface(); surf = np.interp(TX, ex, ez); m = (TX >= 130) & (TX <= 370)
    out = {}
    pold = os.path.join(d, "bel_results.npz")
    old = dict(np.load(pold, allow_pickle=True)) if os.path.exists(pold) else {}
    done = {k.split("__")[0] for k in old if "__" in k}
    for t in TRUTHS:
        for s in range(N_SEED):
            p = os.path.join(d, "runs", f"{t}_s{s}.npz")
            if not os.path.exists(p) or f"{t}_s{s}" in done:
                continue
            z = np.load(p, allow_pickle=True)
            if not np.all(z["keep"]):
                print(f"[bel] {t} s{s}: non-positive noisy R — BEL needs the full data vector, skipped", flush=True)
                continue
            y = np.log10(z["R_noisy"]); ti = TI[t]
            P = bel.posterior(y, N_POST, np.random.default_rng([134100, ti, s]), "kde")
            sf = bel.pd.fwd(y)[0]
            fal = [float(np.mean(t2tr < np.sum(sf ** 2 / var))), float(np.mean(t2rtr < (sf - mu) @ Ci @ (sf - mu))),
                   float(np.mean(qtr < bel.pd.resid(y)[0]))]
            P2 = b2.posterior(y, N_POST, np.random.default_rng([134101, ti, s]))
            bj, pj, _ = J.posterior(y, N_POST, np.random.default_rng([134102, ti, s]))
            bc, pc, A, info = s17.constrained(bj, pj, np.random.default_rng([134103, ti, s]))
            st = s17.stats(bc, pc, A)
            out[f"{t}_s{s}"] = dict(base=P.astype(np.float32), thick=np.median(surf[None, m] - P[:, m], axis=1),
                                    rho=(10 ** P2[:, 2:5]).astype(np.float32), fal=np.array(fal),
                                    sw={u: np.asarray(st["usw"][u]) for u in s17.UNITS}, rej=info["joint_rate"])
            print(f"[bel] {t} s{s}: ρ_below median {np.median(10 ** P2[:, 3]):.2f} Ω·m", flush=True)
    flat = {k: v for k, v in old.items() if k != "TX"}                # existing data sets kept bit-identical
    for k, v in out.items():
        flat[f"{k}__base"] = v["base"]; flat[f"{k}__thick"] = v["thick"]; flat[f"{k}__rho"] = v["rho"]
        flat[f"{k}__fal"] = v["fal"]; flat[f"{k}__rej"] = np.array(v["rej"])
        for u, a in v["sw"].items():
            flat[f"{k}__sw_{u}"] = a
    np.savez(os.path.join(d, "bel_results.npz"), TX=TX, **flat)


def cmd_run(args):
    global ARGS_OUT
    ARGS_OUT = os.path.abspath(args.out)
    d, _ = dirs(args.out)
    t0 = time.time()
    forward_truths(d)
    print(f"[run] forward done ({time.time()-t0:.0f} s); clay ρ = {clay_rho():.2f} Ω·m", flush=True)
    run_inversions(d, args.workers)
    print(f"[run] inversions done ({time.time()-t0:.0f} s)", flush=True)
    bel_analyses(d)
    print(f"[run] BEL analyses done ({time.time()-t0:.0f} s)", flush=True)


ARGS_OUT = wc.ROOT


# ─────────────────────────────────────────────────────────────────────────────
def q(v, d=2):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    lo, md, hi = np.percentile(v, [5, 50, 95]); return f"{md:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def cmd_report(args):
    d_src = os.path.join(wc.ROOT, "data", "processed", "v1_1", "lebbe1980")        # cached run products (read-only)
    d, res = dirs(args.out)
    s17 = _load("s17", "17_salinity_v2.py")
    Z = np.load(os.path.join(d_src, "bel_results.npz"), allow_pickle=True)
    TX = Z["TX"]; jx = [int(np.argmin(np.abs(TX - x))) for x in XREP]
    runs = {k: np.load(os.path.join(d_src, "runs", f"{k}.npz"), allow_pickle=True) for k in
            sorted({k.split("__")[0] for k in Z.files if "__" in k})}
    fwd = np.load(os.path.join(d_src, "truth_forward.npz"))
    arch = s17.archie_v2(20000, np.random.default_rng([134200, 0]))
    num0 = {r["key"]: r["value"] for r in json.load(open(os.path.join(wc.RESULTS_DIR, "paper", "numbers.json"), encoding="utf-8"))}
    f_lo, f_hi = [float(v) for v in num0["second_bel_rho_below"].split(";")[0].split("[")[1].rstrip("] ").split(",")]
    L = ["# 1 — Synthetic test with the 1980 K1 structure (Lebbe 1981) as truth", "",
         "Pre-registered expectations and analysis choices: `docs/decision_log.md` (v1.1). Truths built from "
         "`data/external/lebbe1981_K1_digitised.csv` on our geometry; forward on the fine Stage-4 mesh for the 1974 G1 "
         f"quadrupoles; 3 % + 1e-4 Ω Gaussian noise, {N_SEED} seeds per truth; T80a_ht adds the field residual pattern. "
         "Processing = the unchanged v1.0 pipeline. Clay (deep-unit) resistivity = "
         f"{float(fwd['clay_rho']):.2f} Ω·m (v1.0 reference-model deep-zone median, "
         "`data/processed/synthetic/truth_spec.json`).", "",
         "BEL numbers: posterior samples of all seeds pooled (median [P5, P95]); 'median of medians' = the median over "
         "seeds of the per-seed posterior median (the E1/E2 statistic).", ""]
    summ = {}
    for t in TRUTHS:
        keys = [k for k in runs if k.startswith(t + "_s")]
        if not keys:
            continue
        tt = t.split("_")[0]
        b = boundaries(tt, np.array(XREP, float))
        base = np.vstack([Z[f"{k}__base"] for k in keys]); rho = np.vstack([Z[f"{k}__rho"] for k in keys])
        rb_med = [float(np.median(Z[f"{k}__rho"][:, 1])) for k in keys]
        tb = float(np.mean(true_below(tt, np.array(XREP, float))))
        sw_true = 100 * arch["F"] * arch["rsw"] / 10 ** tb
        sw_below = np.concatenate([Z[f"{k}__sw_below"] for k in keys])
        cov_sal = np.mean([(np.percentile(Z[f"{k}__base"][:, j], 5) <= b["sal"][i] <= np.percentile(Z[f"{k}__base"][:, j], 95))
                           for k in keys for i, j in enumerate(jx)])
        cov_ft = np.mean([(np.percentile(Z[f"{k}__base"][:, j], 5) <= b["ft"][i] <= np.percentile(Z[f"{k}__base"][:, j], 95))
                          for k in keys for i, j in enumerate(jx)])
        med = [float(np.median(base[:, j])) for j in jx]
        inside = [bool(min(b["sal"][i], b["ft"][i]) - 1e-9 <= med[i] <= max(b["sal"][i], b["ft"][i]) + 1e-9) for i in range(5)]
        lm = [[float(np.nanmedian([runs[k]["pooled_band_logmid"][1, int(np.argmin(np.abs(runs[k]["cols"] - x)))] for k in keys]))]
              for x in XREP]
        fal = np.array([Z[f"{k}__fal"] for k in keys])
        chi2 = [float(runs[k]["lam50_chi2"]) for k in keys]
        L += [f"## {t}", "",
              f"{len(keys)} seeds; χ² of the λ = 50 inversion {min(chi2):.2f}–{max(chi2):.2f}; falsification percentiles "
              f"(median over seeds) T² {100*np.median(fal[:, 0]):.1f}, robust T² {100*np.median(fal[:, 1]):.1f}, "
              f"Q {100*np.median(fal[:, 2]):.1f}.", "",
              "| x (m) | true saline base | true fresh top | BEL lens base, median [P5, P95] | BEL median inside [fresh top, "
              "saline base] | λ-pooled log-midpoint (median over seeds) |", "|---|---|---|---|---|---|"]
        for i, (x, j) in enumerate(zip(XREP, jx)):
            L.append(f"| {x} | {b['sal'][i]:.2f} | {b['ft'][i]:.2f} | {q(base[:, j])} | {'yes' if inside[i] else 'no'} | "
                     f"{lm[i][0]:.2f} |")
        thick = np.concatenate([Z[f"{k}__thick"] for k in keys])
        rel = ("above" if np.median(rb_med) > f_hi else "below" if np.median(rb_med) < f_lo else "inside")
        L.append(f"- ρ_below vs the field 90 % interval [{f_lo:.2f}, {f_hi:.2f}] Ω·m: median of medians "
                 f"{np.median(rb_med):.2f} Ω·m is **{rel}** it (per-seed medians {min(rb_med):.2f}–{max(rb_med):.2f} Ω·m).")
        L += ["", f"- ρ_lens {q(rho[:, 0])} Ω·m (true {RHO['lens']:.2f}); ρ_below {q(rho[:, 1])} Ω·m, median of medians "
              f"**{np.median(rb_med):.2f} Ω·m** (true thickness-weighted log-mean between saline base and clay "
              f"{10**tb:.2f} Ω·m); ρ_discharge {q(rho[:, 2])} Ω·m.",
              f"- % seawater, below unit: recovered {q(sw_below, 1)}; 'true' (same salinity step on {10**tb:.2f} Ω·m) "
              f"{q(sw_true, 1)}. Other units: lens {q(np.concatenate([Z[f'{k}__sw_lens'] for k in keys]), 1)}, discharge "
              f"{q(np.concatenate([Z[f'{k}__sw_dis'] for k in keys]), 1)}, dune {q(np.concatenate([Z[f'{k}__sw_dune'] for k in keys]), 1)}.",
              f"- Lens thickness over x = 130–370 m: {q(thick, 1)} m.",
              f"- Coverage of the BEL 90 % lens-base intervals (x = 150–350 m, all seeds): true saline base "
              f"{100*cov_sal:.0f} %, true fresh top {100*cov_ft:.0f} %.", ""]
        summ[t] = dict(rho_below_mom=float(np.median(rb_med)), rho_below=q(rho[:, 1]), rho_lens=q(rho[:, 0]),
                       rho_dis=q(rho[:, 2]), true_below=float(10 ** tb), sw_below=q(sw_below, 1), sw_true=q(sw_true, 1),
                       cov_sal=float(cov_sal), cov_ft=float(cov_ft), inside=inside, base_med=med,
                       sal=[float(v) for v in b["sal"]], ft=[float(v) for v in b["ft"]], n=len(keys),
                       fal=[float(v) for v in np.median(fal, 0)], field_rel=rel, field_lo=f_lo, field_hi=f_hi)
    a = summ.get("T80a"); bb = summ.get("T80b")
    e1 = a and 2.6 <= a["rho_below_mom"] <= 12.5
    e2 = a and a["rho_below_mom"] >= 15.0
    e3 = a and all(a["inside"])
    L += ["## Pre-registered expectations", "",
          f"- **E1** (ρ_below of T80a in [2.6, 12.5] Ω·m): {'**holds**' if e1 else '**does not hold**'} — median of "
          f"medians {a['rho_below_mom']:.2f} Ω·m." if a else "- E1: T80a missing.",
          f"- **E2** (≥ 15 Ω·m): {'**holds** — Section 4.3 must be rewritten' if e2 else '**does not hold**'}"
          + ("" if e1 or e2 else " (value between 12.5 and 15 Ω·m: inconclusive)") + ".",
          f"- **E3** (T80a BEL lens base inside the transition at every x = 150–350 m): {'**holds**' if e3 else '**does not hold**'} "
          f"({sum(a['inside'])} of 5 positions)." if a else "",
          f"- **E4** (T80b vs T80a): ρ_below median of medians {bb['rho_below_mom']:.2f} vs {a['rho_below_mom']:.2f} Ω·m "
          f"(change {bb['rho_below_mom'] - a['rho_below_mom']:+.2f} Ω·m, {100*(bb['rho_below_mom']/a['rho_below_mom']-1):+.0f} %)."
          if a and bb else "- E4: T80a or T80b missing.", "",
          "**What this shows and does not show.** These are synthetic data from a structure built after the 1980 logs; "
          "they show what the unchanged pipeline would report if the subsurface looked like that, not that the 1980 "
          "structure exists today."]
    # observations after the results (not pre-registered), computed from the cached products
    allmed = {k: float(np.median(Z[f"{k}__rho"][:, 1])) for k in runs}
    num = {r["key"]: r["value"] for r in json.load(open(os.path.join(wc.RESULTS_DIR, "paper", "numbers.json"), encoding="utf-8"))}
    field_rb = num["second_bel_rho_below"].split(";")[0]
    field_p95 = float(field_rb.split(",")[1].strip(" ]"))
    tseeds = [allmed[k] for k in runs if k.startswith("T80a_s")]
    lm_bias = []
    for k in runs:
        tt = k.split("_s")[0].split("_")[0]; bnd = boundaries(tt, np.array(XREP, float))
        for i, x in enumerate(XREP):
            jj = int(np.argmin(np.abs(runs[k]["cols"] - x)))
            lm_bias.append(runs[k]["pooled_band_logmid"][1, jj] - bnd["sal"][i])
    a_, b_ = summ.get("T80a", {}), summ.get("T80b", {})
    p95s = [float(np.percentile(Z[f"{k}__rho"][:, 1], 95)) for k in runs]
    L += ["", "## Observations after the results (not pre-registered)", "",
          f"- **Prior ceiling.** The prior of ρ_below is 2–20 Ω·m. Every synthetic truth has a true log-mean between the "
          f"saline base and the clay of {min(v['true_below'] for v in summ.values()):.1f}–"
          f"{max(v['true_below'] for v in summ.values()):.1f} Ω·m, and the recovered ρ_below (per-seed posterior medians) is "
          f"{min(allmed.values()):.2f}–{max(allmed.values()):.2f} Ω·m for all {len(allmed)} synthetic data sets, with "
          f"posterior P95 {min(p95s):.1f}–{max(p95s):.1f} Ω·m: the estimate saturates below the prior ceiling and cannot "
          f"resolve how fresh the fresh zone is (fresh zone 30, 50 and 100 Ω·m give {summ['T80c']['rho_below_mom']:.1f}, "
          f"{summ['T80a']['rho_below_mom']:.1f} and {summ['T80d']['rho_below_mom']:.1f} Ω·m).",
          f"- **Separation from the field value.** {sum(v > field_p95 for v in allmed.values())} of {len(allmed)} synthetic per-seed medians "
          f"({min(allmed.values()):.2f}–{max(allmed.values()):.2f} Ω·m) lie above the field posterior P95 of "
          f"{field_p95:.2f} Ω·m (field {field_rb} Ω·m, `second_bel_rho_below`).",
          f"- **E2 margin.** The T80a statistic ({np.median(tseeds):.2f} Ω·m) exceeds the 15 Ω·m threshold by "
          f"{np.median(tseeds) - 15:.2f} Ω·m; the five seeds give {min(tseeds):.2f}–{max(tseeds):.2f} Ω·m "
          f"({sum(v < 15 for v in tseeds)} of 5 below 15).",
          f"- **Salinity of the water below the lens.** Recovered {a_.get('sw_below', '—')} % seawater vs 'true' "
          f"{a_.get('sw_true', '—')} % for T80a: biased towards saline because ρ_below cannot exceed 20 Ω·m.",
          f"- **Linearised log-midpoint.** λ-pooled median minus the true saline base over all data sets and x = 150–350 m: "
          f"median {np.nanmedian(lm_bias):+.2f} m (range {np.nanmin(lm_bias):+.2f} … {np.nanmax(lm_bias):+.2f} m; positive = "
          "shallower): for this structure the linearised base is shallow, while the BEL intervals cover both true "
          "boundaries.",
          f"- **Heavy-tailed residuals.** T80a_ht ({summ.get('T80a_ht', {}).get('rho_below_mom', float('nan')):.2f} Ω·m) vs "
          f"T80a ({a_.get('rho_below_mom', float('nan')):.2f} Ω·m): the field residual pattern does not change the result."]
    with open(os.path.join(res, "1_lebbe1980_synthetic.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    csv = ["truth,n_seeds,rho_below_median_of_medians,rho_below_pooled,true_below_logmean,rho_lens_pooled,"
           "sw_below_recovered,sw_below_true,cov_saline_base,cov_fresh_top,bel_inside_transition"]
    for t, v in summ.items():
        csv.append(f"{t},{v['n']},{v['rho_below_mom']:.2f},\"{v['rho_below']}\",{v['true_below']:.2f},\"{v['rho_lens']}\","
                   f"\"{v['sw_below']}\",\"{v['sw_true']}\",{100*v['cov_sal']:.0f}%,{100*v['cov_ft']:.0f}%,{sum(v['inside'])}/5")
    with open(os.path.join(res, "T_S1_lebbe1980.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv) + "\n")
    json.dump(dict(summ=summ, E1=bool(e1), E2=bool(e2), E3=bool(e3)),
              open(os.path.join(d, "..", "lebbe1980_summary.json"), "w", encoding="utf-8"), indent=1)
    figure(res, d_src, Z, fwd)
    print("\n".join(L))


def figure(res, d_src, Z, fwd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pygimli.meshtools as mt
    s18 = _load("p18", "18_paper.py"); s18.mpl()
    sch = sb.g1_scheme(); scheme = sb.make_scheme(sch)
    with contextlib.redirect_stdout(io.StringIO()):
        fmesh = mt.createParaMesh(scheme.sensors(), **sb.FWD)
    ctx = s18.ref_ctx()
    run0 = np.load(os.path.join(d_src, "runs", "T80a_s0.npz"), allow_pickle=True)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), gridspec_kw=dict(hspace=0.45))
    for ax, polys, vals, ttl in ((axes[0], wc.cell_polygons(fmesh), 10 ** fwd["cell_log10_T80a"], "(a) T80a truth (1980 K1 structure)"),
                                 (axes[1], ctx["polys"], run0["lam50_model"], "(b) deterministic inversion of T80a, seed 0 (λ = 50)")):
        pc = wc.draw_cells(ax, polys, vals, "RdBu_r", s18.rho_norm())
        wc.setup_section(ax, None, ve=3, z_floor=-40); s18.dune_hatch(ax); ax.set_title(ttl)
        cb = fig.colorbar(pc, ax=ax, fraction=0.025, pad=0.01); cb.set_label("ρ (Ω·m)")
    ax = axes[2]
    TX = Z["TX"]; base = np.vstack([Z[k] for k in Z.files if k.startswith("T80a_s") and k.endswith("__base")])
    lo, md, hi = np.percentile(base, [5, 50, 95], 0); h = TX >= 130
    ax.fill_between(TX[h], lo[h], hi[h], color=s18.COL["bel"], alpha=0.25, lw=0)
    ax.plot(TX[h], md[h], color=s18.COL["bel"], lw=1.6, label="BEL lens base, T80a (5 seeds pooled): median, P5–P95")
    xx = np.linspace(0, 505, 506); b = boundaries("T80a", xx)
    ax.plot(xx, b["sal"], color="#b2182b", lw=1.2, label="true saline base")
    ax.plot(xx, b["ft"], color="#1b9e77", lw=1.2, ls="--", label="true fresh top")
    ex, ez = wc.land_surface(); ax.plot(ex, ez, "k-", lw=0.8)
    ax.set_xlim(-5, 510); ax.set_ylim(-32, 9); ax.set_aspect(3); s18.dune_hatch(ax)
    ax.set_xlabel("Distance along profile (m)"); ax.set_ylabel("Elevation (mTAW)"); s18.ve_text(ax, 3)
    ax.set_title("(c) BEL lens base for T80a vs the true boundaries")
    s18.cbar_space(fig, ax, pc); s18.legend_below(ax, ncol=2, dy=-0.25, fontsize=6)
    fig.savefig(os.path.join(res, "figs", "F_S1_lebbe1980_synthetic.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run"); pr.add_argument("--workers", type=int, default=4); pr.add_argument("--out", default=wc.ROOT)
    sub.add_parser("report").add_argument("--out", default=wc.ROOT)
    pw = sub.add_parser("worker"); pw.add_argument("--worker-id", type=int, required=True); pw.add_argument("--out", default=wc.ROOT)
    pw.add_argument("--jobs", nargs="+", required=True)
    a = ap.parse_args()
    dict(run=cmd_run, report=cmd_report, worker=cmd_worker)[a.cmd](a)


if __name__ == "__main__":
    main()
