"""
25_generalisation.py
====================
Westhoek ERT v1.3 — does the fixed-contour failure generalise? (pre-registration: docs/decision_log.md, v1.3)

  field  [--out ROOT]                      Task 0: like-for-like field numbers (λ-pooled 3 Ω·m base, λ = 50 log-midpoint
                                           base) at x = 150–350 m → ROOT/results/v1_3/0_field_like_for_like.md, field.json
  run    [--workers 6] [--out ROOT]        Task 1: 32 factorial truths → fine-mesh forward → noisy data → v1.0 inversion and
         [--time-only]                     linearised posterior (5 realisations at λ = 50; realisations 0–1 also λ = 5, 10,
                                           20, 100) → per-sample interface depths of every definition
                                           → ROOT/data/processed/v1_3/general/<truth>/; --time-only times 5 inversions
                                           and writes ROOT/results/v1_3/1_plan.md
  resume [--workers 6] [--out ROOT]        as run, skipping truths with a checkpoint (summary.json) and finished realisations
  report [--out ROOT]                      Tasks 1–2: metrics, E9–E12, rule fit, figures, tables, archive + SHA256SUMS
  worker                                   (internal)

Truths G_<ρ_lens>_<k>_<z_b>_<τ>: x ≥ 90 m lens ρ_lens from the surface to the flat base z_b, ρ_below = k·ρ_lens down to
−30 mTAW, deep 8.44 Ω·m below; τ = 5: log-linear transition of 5 m centred on z_b; x < 90 m: v1.0 T1 reference values.
Seeds: noise default_rng([136000, ti, real]); posterior samples default_rng([136100, ti, real, λ]); rule-fit bootstrap
default_rng([136200]) (λ = 50) and default_rng([136201]) (per λ). ti = index in TRUTHS.
Reused unchanged: 10_synthetic_benchmark (G1 scheme, fine mesh FWD, error model, one_inversion), 11_stage4_closeout
(interfaces_all, band_stats, r_at), westhoek_common.main_lens (its `level` argument gives the 2 / 5 Ω·m and oracle
contours), 13_bel_full.pilot.simulate_R (fine-mesh forward, as 22_lebbe1980_synthetic.py).
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import read_trace  # noqa: E402
read_trace.install("25_generalisation.py")
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


c11 = _load("c11", "11_stage4_closeout.py")
sb = c11.sb

RHO_LENS = (0.8, 1.42)
KS = (2, 4, 8, 16)
ZBS = (-8, -12, -18)
TAU_ZB = -12
TRUTHS = ([f"G_{rl:g}_{k}_{zb}_0" for rl in RHO_LENS for k in KS for zb in ZBS] +
          [f"G_{rl:g}_{k}_{TAU_ZB}_5" for rl in RHO_LENS for k in KS])
TI = {t: i for i, t in enumerate(TRUTHS)}
N_REAL, N_REAL_LAM = 5, 2
LAMS_ALL = [50.0, 5.0, 10.0, 20.0, 100.0]
THRS = (2.0, 3.0, 5.0)
DEFS = ("thr2", "thr3", "thr5", "oracle", "maxgrad", "logmid", "maxgrad_o", "logmid_o")
DEFS_MAIN = DEFS[:6]
Z90 = 1.6449
XREP = (150, 200, 250, 300, 350)
WESTHOEK = dict(rho_lens=1.42, rho_below=5.32)
GROUPS = ("≤10", "10–15")
SRC = os.path.join(wc.ROOT, "data", "processed", "v1_3", "general")


def dirs(root):
    d = os.path.join(root, "data", "processed", "v1_3", "general")
    res = os.path.join(root, "results", "v1_3")
    for p in (d, os.path.join(d, "_logs"), os.path.join(d, "_work"), res, os.path.join(res, "figs")):
        os.makedirs(p, exist_ok=True)
    return d, res


# ─────────────────────────────────────────────────────────────────────────────
# truths
# ─────────────────────────────────────────────────────────────────────────────
def params(name):
    _, rl, k, zb, tau = name.split("_")
    return dict(rho_lens=float(rl), k=int(k), z_b=float(zb), tau=float(tau), rho_below=float(rl) * int(k))


def ref_values():
    return sb.load_spec()["values"]["T1"]


def log10rho(name, x, z):
    p = params(name); v = ref_values()
    x = np.asarray(x, float); z = np.asarray(z, float)
    lens, below, deep = np.log10(p["rho_lens"]), np.log10(p["rho_below"]), np.log10(v["deep"])
    out = np.where(z > p["z_b"], lens, below)
    if p["tau"] > 0:
        d = z - p["z_b"]; sel = np.abs(d) < p["tau"] / 2
        f = (d + p["tau"] / 2) / p["tau"]                               # 0 at the bottom … 1 at the top
        out = np.where(sel, below + f * (lens - below), out)
    out = np.where(z <= sb.Z_DEEP, deep, out)
    dune = x < sb.X_LENS0
    dv = np.where(z > sb.Z_WT, np.log10(v["cap"]), np.where(z > sb.Z_DEEP, np.log10(v["fresh"]), deep))
    return np.where(dune, dv, out)


def q_of(thr, p):
    return float((np.log10(thr) - np.log10(p["rho_lens"])) / np.log10(p["k"]))


def oracle_level(p):
    return 0.5 * (np.log10(p["rho_lens"]) + np.log10(p["rho_below"]))


def truth_spec(name):
    p = params(name); v = ref_values()
    return dict(name=name, index=TI[name], **p, oracle=float(10 ** oracle_level(p)), deep=v["deep"], z_deep=sb.Z_DEEP,
                x_lens0=sb.X_LENS0, dune=dict(cap=v["cap"], fresh=v["fresh"], z_wt=sb.Z_WT),
                q={f"thr{t:g}": q_of(t, p) for t in THRS} | {"oracle": 0.5},
                note="lens from the surface to z_b for x ≥ 90 m (no discharge zone); τ = log-linear transition centred on z_b")


# ─────────────────────────────────────────────────────────────────────────────
# interface definitions (m̂ and samples)
# ─────────────────────────────────────────────────────────────────────────────
def depths_all(V, z, p):
    """V (nz, nx, ns) log10ρ → {definition: (nx, ns)}."""
    out = {f"thr{t:g}": wc.main_lens(V, z, level=np.log10(t))[1] for t in THRS}
    lo = oracle_level(p)
    out["oracle"] = wc.main_lens(V, z, level=lo)[1]
    r = c11.interfaces_all(V, z)
    out["maxgrad"], out["logmid"] = r["maxgrad"], r["logmid"]
    ro = c11.interfaces_all(V + (wc.LOG3 - lo), z)                    # anchored on the oracle crossing (diagnostic)
    out["maxgrad_o"], out["logmid_o"] = ro["maxgrad"], ro["logmid"]
    return out


def sample_depths(cs, S, p, chunk=100):
    out = {d: np.full((len(cs.x), S.shape[1]), np.nan) for d in DEFS}
    for i in range(0, S.shape[1], chunk):
        r = depths_all(cs(S[:, i:i + chunk] / np.log(10)), cs.z, p)
        for d in DEFS:
            out[d][:, i:i + chunk] = r[d]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# forward, jobs, workers
# ─────────────────────────────────────────────────────────────────────────────
def forward_truth(d, name, fmesh=None):
    tdir = os.path.join(d, name); os.makedirs(tdir, exist_ok=True)
    p = os.path.join(tdir, "truth_forward.npz")
    if os.path.exists(p):
        return np.load(p)
    import pygimli.meshtools as mt
    b13 = _load("b13", "13_bel_full.py")
    if fmesh is None:
        sch = sb.g1_scheme(); scheme = sb.make_scheme(sch)
        with contextlib.redirect_stdout(io.StringIO()):
            fmesh = mt.createParaMesh(scheme.sensors(), **sb.FWD)
    if fmesh.cellCount() != 17684:
        raise RuntimeError(f"forward mesh has {fmesh.cellCount()} cells, expected 17 684")
    cx = np.array([c.center().x() for c in fmesh.cells()]); cz = np.array([c.center().y() for c in fmesh.cells()])
    v = log10rho(name, cx, cz)
    n = len(sb.g1_scheme()["C1"])
    with contextlib.redirect_stdout(io.StringIO()):
        R = b13.pilot.simulate_R(fmesh, 10 ** v, n)
    with open(os.path.join(tdir, "truth_spec.json"), "w", encoding="utf-8") as f:
        json.dump(truth_spec(name), f, indent=1)
    np.savez(p + ".tmp.npz", R=R, cell_log10=v.astype(np.float32), fwd_cells=fmesh.cellCount())
    os.replace(p + ".tmp.npz", p)
    return np.load(p)


def lams_for(real):
    return LAMS_ALL if real < N_REAL_LAM else [50.0]


def run_job(d, name, real, m02, sch, cell_x, cs, log):
    tdir = os.path.join(d, name)
    out = os.path.join(tdir, f"r{real}.npz")
    if os.path.exists(out):
        log(f"skip {name} r{real}"); return
    t0 = time.time()
    p = params(name); ti = TI[name]
    R_true = forward_truth(d, name)["R"]
    rng = np.random.default_rng([136000, ti, real])
    sig = np.sqrt((sb.ERR_REL * R_true) ** 2 + sb.ERR_ABS ** 2)
    R = R_true + sig * rng.standard_normal(len(R_true))
    keep = R > 0
    np.savez(os.path.join(tdir, f"noisy_r{real}.npz"), R_noisy=R, keep=keep, R_true=R_true,
             seed=np.array([136000, ti, real]))
    elec = {int(e): (float(x), float(z)) for e, x, z in zip(sch["elec_num"], sch["elec_x"], sch["elec_z"])}
    land = dict({k: sch[k][keep] for k in ("C1", "C2", "P1", "P2")}, R=R[keep], n=int(keep.sum()),
                err=np.zeros(int(keep.sum())))
    zb = np.full(len(cs.x), p["z_b"])
    save = dict(truth=name, real=real, cols=cs.x, n_nonpositive_dropped=int((~keep).sum()))
    pooled = {dd: [] for dd in DEFS}
    for lam in lams_for(real):
        tl = time.time()
        info, S = sb.one_inversion(m02, land, elec, lam, cell_x, cs, [136100, ti, real, int(lam)])
        hat = depths_all(cs(np.log10(info["model"].astype(float)))[:, :, None], cs.z, p)
        dep = sample_depths(cs, S, p)
        pre = f"lam{int(lam)}_"
        save.update({pre + "chi2": info["chi2"], pre + "stop": info["stop"], pre + "n_iter": info["n_iter"],
                     pre + "model": info["model"], pre + "radius": info["radius"],
                     pre + "r_true": c11.r_at(cs, info["radius"], zb), pre + "t": time.time() - tl})
        for dd in DEFS:
            save[pre + f"mhat_{dd}"] = hat[dd][:, 0]
            save[pre + f"band_{dd}"] = c11.band_stats(dep[dd])
            save[pre + f"r_{dd}"] = c11.r_at(cs, info["radius"], hat[dd][:, 0])
            save[pre + f"depths_{dd}"] = dep[dd].astype(np.float32)
            if real < N_REAL_LAM:
                pooled[dd].append(dep[dd])
        log(f"{name} r{real} λ={lam:g}: χ²={info['chi2']:.3f} ({info['stop']}) {time.time()-tl:.0f}s")
    if real < N_REAL_LAM:
        for dd in DEFS:
            save[f"pooled_band_{dd}"] = c11.band_stats(np.hstack(pooled[dd]))
    save["t_job"] = time.time() - t0
    np.savez(out + ".tmp.npz", **save); os.replace(out + ".tmp.npz", out)
    log(f"done {name} r{real} in {save['t_job']:.0f}s")


def cmd_worker(args):
    d, _ = dirs(args.out)
    sch = sb.g1_scheme()
    ref = wc.load_run(sb.REF_TAG); mesh, _ = wc.rebuild_paradomain(ref); cs = wc.ColumnSampler(mesh, sb.COLS)
    m02 = c11._load("inv02", "02_land_inversion.py"); m02.VERBOSE = False
    logf = open(os.path.join(d, "_logs", f"worker_{args.worker_id}.log"), "a", encoding="utf-8")
    log = lambda m: (logf.write(time.strftime("%H:%M:%S ") + m + "\n"), logf.flush())
    for job in args.jobs:
        kind, name, real = job.split(":")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if kind == "F":
                    t0 = time.time(); forward_truth(d, name); log(f"forward {name} {time.time()-t0:.0f}s")
                else:
                    run_job(d, name, int(real), m02, sch, np.asarray(ref["cell_x"]), cs, log)
        except Exception as e:
            import traceback
            log(f"FAILED {job}: {e}\n{traceback.format_exc()}")
    log("worker finished")


def launch(d, jobs, workers, out_root):
    parts = [jobs[i::workers] for i in range(workers)]
    procs = []
    for w, part in enumerate(parts):
        if not part:
            continue
        wd = os.path.join(d, "_work", f"w{w}"); os.makedirs(wd, exist_ok=True)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", "--worker-id", str(w), "--out",
                                       out_root, "--jobs"] + part, cwd=wd, env=dict(os.environ, OMP_NUM_THREADS="2"),
                                      stdout=subprocess.DEVNULL, stderr=open(os.path.join(d, "_logs", f"worker_{w}.err"), "a")))
    for pr in procs:
        pr.wait()


def config(name):
    p = params(name); ti = TI[name]
    return dict(truth=name, truth_index=ti, params=p, realisations=N_REAL, lams={str(r): lams_for(r) for r in range(N_REAL)},
                seeds=dict(noise=[[136000, ti, r] for r in range(N_REAL)],
                           posterior=[[136100, ti, r, int(l)] for r in range(N_REAL) for l in lams_for(r)]),
                noise=dict(rel=sb.ERR_REL, abs_ohm=sb.ERR_ABS), samples=sb.N_SAMPLES,
                inversion=dict(paraDepth=sb.PARA_DEPTH, paraDX=sb.PARA_DX, maxIter=sb.MAX_ITER, zWeight=0.5),
                forward_mesh=sb.FWD, forward_cells=17684, columns=list(map(float, sb.COLS)), definitions=list(DEFS),
                saved="per inversion: model, radius, χ², m̂ / band / r per definition, r at z_b, per-sample interface "
                      "depths per definition (float32); full posterior samples not saved")


def checkpoint(d, name):
    """Write config.json and summary.json (the per-truth checkpoint) once all realisations exist."""
    tdir = os.path.join(d, name)
    if not all(os.path.exists(os.path.join(tdir, f"r{r}.npz")) for r in range(N_REAL)):
        return False
    with open(os.path.join(tdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config(name), f, indent=1)
    rows = truth_metrics(d, name)
    with open(os.path.join(tdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    return True


def finished(d, name):
    return os.path.exists(os.path.join(d, name, "summary.json"))


def cmd_run(args, resume=False):
    d, res = dirs(args.out)
    out_root = os.path.abspath(args.out)
    t0 = time.time()
    if args.time_only:
        name = TRUTHS[0]
        forward_truth(d, name)
        tf = time.time() - t0
        jobs = [f"J:{name}:0"]                                         # realisation 0: λ = 50, 5, 10, 20, 100
        launch(d, jobs, 1, out_root)
        z = np.load(os.path.join(d, name, "r0.npz"), allow_pickle=True)
        ts = [float(z[f"lam{int(l)}_t"]) for l in LAMS_ALL]
        plan(res, tf, ts, args.workers)
        return
    todo = [t for t in TRUTHS if not finished(d, t)]
    if not resume and len(todo) < len(TRUTHS):
        print(f"[run] {len(TRUTHS) - len(todo)} truths already finished; continuing (same as resume)", flush=True)
    launch(d, [f"F:{t}:0" for t in todo if not os.path.exists(os.path.join(d, t, "truth_forward.npz"))], args.workers, out_root)
    print(f"[run] forward done ({time.time()-t0:.0f} s)", flush=True)
    jobs = [f"J:{t}:{r}" for r in range(N_REAL) for t in todo if not os.path.exists(os.path.join(d, t, f"r{r}.npz"))]
    jobs.sort(key=lambda j: int(j.split(":")[2]) >= N_REAL_LAM)        # heavy (5 λ) jobs first
    print(f"[run] {len(jobs)} inversion jobs on {args.workers} workers", flush=True)
    launch(d, jobs, args.workers, out_root)
    n = sum(checkpoint(d, t) for t in todo)
    print(f"[run] {n}/{len(todo)} truths checkpointed ({time.time()-t0:.0f} s)", flush=True)


def plan(res, tf, ts, workers):
    n_inv = len(TRUTHS) * (N_REAL + N_REAL_LAM * 4)
    t_inv = float(np.mean(ts))
    serial = n_inv * t_inv + len(TRUTHS) * tf
    # parallel slowdown measured in v1.1 (22_lebbe1980_synthetic.py, 4 workers): assumed 1.5×
    wall = serial / workers * 1.5
    L = ["# 1 — Run plan (Task 1)", "",
         f"Timed on {time.strftime('%Y-%m-%d')}: forward of one truth on the 17 684-cell mesh {tf:.0f} s; five inversions "
         f"(truth {TRUTHS[0]}, realisation 0, λ = 50, 5, 10, 20, 100; each = inversion + resolution + 1000 samples + all "
         f"interface definitions) " + ", ".join(f"{t:.0f}" for t in ts) + f" s (mean {t_inv:.0f} s), one worker, "
         "OMP_NUM_THREADS = 2.", "",
         f"- Inversions: {len(TRUTHS)} truths × ({N_REAL} at λ = 50 + {N_REAL_LAM} × 4 extra λ) = **{n_inv}**.",
         f"- Serial time: {serial/3600:.1f} h. With {workers} parallel workers and an assumed 1.5× slowdown under load: "
         f"**≈ {wall/3600:.1f} h** wall-clock.",
         f"- Threshold in the brief: ≈ 36 h → {'within budget; full design run' if wall <= 36 * 3600 else 'EXCEEDS the budget: stop and propose a reduced design'}."]
    with open(os.path.join(res, "1_plan.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
# metrics
# ─────────────────────────────────────────────────────────────────────────────
def grp_of(r):
    if not np.isfinite(r):
        return None
    return "≤10" if r <= 10 else ("10–15" if r <= 15 else None)


def truth_metrics(d, name):
    """Rows per definition × treatment × group (pre-registered metric definitions)."""
    p = params(name); zb = p["z_b"]
    Z = [np.load(os.path.join(d, name, f"r{r}.npz"), allow_pickle=True) for r in range(N_REAL)]
    rtrue = [z["lam50_r_true"] for z in Z]
    ncol = len(Z[0]["cols"])
    treats = {"lam50": [(r, "lam50_") for r in range(N_REAL)], "pooled": [(r, "pooled_") for r in range(N_REAL_LAM)]}
    treats.update({f"lam{int(l)}_r01": [(r, f"lam{int(l)}_") for r in range(N_REAL_LAM)] for l in LAMS_ALL})
    rows = []
    for dd in DEFS:
        q = q_of(float(dd[3:]), p) if dd.startswith("thr") else (0.5 if dd == "oracle" else None)
        for tr, pairs in treats.items():
            for g in GROUPS:
                cov, bias, rr, miss = [], [], [], 0
                for r, pre in pairs:
                    band = Z[r][pre + f"band_{dd}"]
                    for j in range(ncol):
                        if grp_of(rtrue[r][j]) != g:
                            continue
                        b = band[:, j]
                        cov.append(bool(b[0] >= 0.5 and b[2] <= zb <= b[3]))
                        est = b[1] if pre == "pooled_" else Z[r][pre + f"mhat_{dd}"][j]
                        if np.isfinite(est):
                            bias.append(float(est - zb))
                        else:
                            miss += 1
                        rr.append(float(rtrue[r][j]))
                row = dict(truth=name, **p, definition=dd, q=q, treatment=tr, group=g, n=len(cov),
                           coverage=float(np.mean(cov)) if cov else None, bias=float(np.median(bias)) if bias else None,
                           n_missing=miss, rbar=float(np.mean(rr)) if rr else None, scatter=None)
                if tr == "lam50" and cov:
                    ratios = []
                    for j in range(ncol):
                        if grp_of(float(np.median([rtrue[r][j] for r in range(N_REAL)]))) != g:
                            continue
                        mh = np.array([Z[r][f"lam50_mhat_{dd}"][j] for r in range(N_REAL)])
                        bs = [Z[r][f"lam50_band_{dd}"][:, j] for r in range(N_REAL)]
                        sp = [(b[3] - b[2]) / (2 * Z90) for b in bs if b[0] >= 0.5]
                        if np.isfinite(mh).sum() >= 3 and sp and np.median(sp) > 0:
                            ratios.append(float(np.nanstd(mh, ddof=1) / np.median(sp)))
                    row["scatter"] = float(np.median(ratios)) if ratios else None
                rows.append(row)
    return rows


def all_rows(d):
    rows = []
    for t in TRUTHS:
        p = os.path.join(d, t, "summary.json")
        if os.path.exists(p):
            rows += json.load(open(p, encoding="utf-8"))
    return rows


def pick(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def ppf(q):
    from scipy.stats import norm
    return float(norm.ppf(q))


def verdicts(rows):
    out = {}
    # E9
    cases = []
    for r in rows:
        if r["definition"] in ("thr2", "thr3", "thr5") and r["treatment"] == "lam50" and r["group"] == "≤10" and \
                r["n"] >= 5 and r["bias"] is not None and 0 < r["q"] < 1 and abs(r["q"] - 0.5) > 0.1:
            pred = -np.sign(ppf(r["q"]))
            cases.append(dict(truth=r["truth"], d=r["definition"], q=r["q"], bias=r["bias"], ok=bool(np.sign(r["bias"]) == pred)))
    out["E9"] = dict(n=len(cases), n_ok=sum(c["ok"] for c in cases), share=np.mean([c["ok"] for c in cases]) if cases else None,
                     fails=[c for c in cases if not c["ok"]])
    out["E9"]["holds"] = bool(out["E9"]["share"] is not None and out["E9"]["share"] >= 0.9)
    # E10
    sharp = [t for t in TRUTHS if params(t)["tau"] == 0]
    ok10, n10, nod = 0, 0, []
    for t in sharp:
        r = pick(rows, truth=t, definition="oracle", treatment="lam50", group="≤10")
        if not r or not r[0]["n"] or r[0]["bias"] is None:
            nod.append(t); continue
        n10 += 1; ok10 += abs(r[0]["bias"]) <= 0.5 and r[0]["coverage"] >= 0.8
    out["E10"] = dict(n=n10, n_ok=int(ok10), share=ok10 / n10 if n10 else None, no_data=nod)
    out["E10"]["holds"] = bool(n10 and ok10 / n10 >= 0.8)
    # E11 (as written), plus the pre-registered breakdowns
    def e11(defs, truths):
        res = {}
        for dd in defs:
            ok, n = 0, 0
            for t in truths:
                r = pick(rows, truth=t, definition=dd, treatment="lam50", group="≤10")
                if not r or not r[0]["n"]:
                    continue
                n += 1; ok += r[0]["coverage"] >= 0.85
            res[dd] = dict(n=n, n_ok=int(ok), share=ok / n if n else None)
        return res
    inside3 = [t for t in TRUTHS if 0 < q_of(3.0, params(t)) < 1]
    out["E11"] = dict(all=e11(("logmid", "maxgrad"), TRUTHS), inside3=e11(("logmid", "maxgrad"), inside3),
                      oracle_anchor=e11(("logmid_o", "maxgrad_o"), TRUTHS), n_inside3=len(inside3))
    out["E11"]["holds"] = bool(all(v["share"] is not None and v["share"] >= 0.8 for v in out["E11"]["all"].values()))
    return out


def fit_rule(rows, treatment="lam50", truths=None, seed=None, nboot=2000):
    """bias = −a·r̄·Φ⁻¹(q) over truths × {2, 3, 5} with 0 < q < 1 (r ≤ 10 m); r̄ from λ = 50 (same truth, group)."""
    truths = truths or [t for t in TRUTHS if params(t)["tau"] == 0]
    pts = []
    for t in truths:
        rb = pick(rows, truth=t, definition="thr3", treatment="lam50", group="≤10")
        rbar = rb[0]["rbar"] if rb else None
        for dd in ("thr2", "thr3", "thr5"):
            r = pick(rows, truth=t, definition=dd, treatment=treatment, group="≤10")
            if not r or r[0]["bias"] is None or rbar is None or not 0 < r[0]["q"] < 1:
                continue
            pts.append(dict(truth=t, d=dd, q=r[0]["q"], rbar=rbar, x=-rbar * ppf(r[0]["q"]), y=r[0]["bias"],
                            k=params(t)["k"], z_b=params(t)["z_b"], rho_lens=params(t)["rho_lens"]))
    if len(pts) < 3:
        return None
    x = np.array([p["x"] for p in pts]); y = np.array([p["y"] for p in pts])
    a = float(x @ y / (x @ x))
    r2 = float(1 - np.sum((y - a * x) ** 2) / np.sum((y - y.mean()) ** 2))
    out = dict(a=a, r2=r2, n=len(pts), pts=[dict(p, pred=a * p["x"], resid=p["y"] - a * p["x"]) for p in pts])
    if seed is not None:
        rng = np.random.default_rng(seed); ut = sorted({p["truth"] for p in pts}); bs = []
        idx = {t: [i for i, p in enumerate(pts) if p["truth"] == t] for t in ut}
        for _ in range(nboot):
            sel = np.concatenate([idx[t] for t in rng.choice(ut, len(ut))])
            xs, ys = x[sel], y[sel]
            bs.append(xs @ ys / (xs @ xs))
        out["a90"] = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Task 0
# ─────────────────────────────────────────────────────────────────────────────
def cmd_field(args):
    _, res = dirs(args.out)
    v4 = np.load(os.path.join(wc.RESULTS_DIR, "runs", "g1_lam50_v4", "interface_definitions_bands.npz"))
    fmt = lambda b: f"{b[1]:.2f} [{b[2]:.2f}, {b[3]:.2f}] ({100*b[0]:.0f}%)"
    out = dict(x=list(XREP))
    L = ["# 0 — Field: like-for-like interface definitions", "",
         "Source: `results/runs/g1_lam50_v4/interface_definitions_bands.npz` (v1.0 A4: bands of the cached field "
         "linearised posterior samples of g1_lam50 and the λ pool 5–100, posterior seeds 20260928 + run index). "
         "median [P5, P95] (share of samples in which the interface exists), mTAW. The two left columns are the paper's "
         "Table 2 pair (λ = 50 for the 3 Ω·m contour, λ-pooled for the log-midpoint); the two right columns are the "
         "like-for-like additions.", "",
         "| x (m) | 3 Ω·m, λ = 50 (Table 2) | log-midpoint, λ-pooled (Table 2) | **3 Ω·m, λ-pooled** | **log-midpoint, λ = 50** |",
         "|---|---|---|---|---|"]
    for k in ("single_thr3", "pooled_logmid", "pooled_thr3", "single_logmid"):
        out[k] = []
    for xx in XREP:
        j = int(np.argmin(np.abs(v4["x"] - xx)))
        for k in ("single_thr3", "pooled_logmid", "pooled_thr3", "single_logmid"):
            out[k].append([float(v) for v in v4[k][:, j]])
        L.append(f"| {xx} | {fmt(v4['single_thr3'][:, j])} | {fmt(v4['pooled_logmid'][:, j])} | "
                 f"{fmt(v4['pooled_thr3'][:, j])} | {fmt(v4['single_logmid'][:, j])} |")
    d1 = [b[1] - a[1] for a, b in zip(out["pooled_thr3"], out["pooled_logmid"])]
    d2 = [b[1] - a[1] for a, b in zip(out["single_thr3"], out["single_logmid"])]
    out["diff_pooled_logmid_minus_thr3"] = d1; out["diff_lam50_logmid_minus_thr3"] = d2
    L += ["", "Median log-midpoint minus median 3 Ω·m base (positive = log-midpoint shallower): λ-pooled " +
          ", ".join(f"{v:+.2f}" for v in d1) + " m; λ = 50 " + ", ".join(f"{v:+.2f}" for v in d2) + " m (x = 150 … 350 m)."]
    with open(os.path.join(res, "0_field_like_for_like.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    dpp = os.path.join(args.out, "data", "processed", "v1_3"); os.makedirs(dpp, exist_ok=True)
    with open(os.path.join(dpp, "field.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
# report
# ─────────────────────────────────────────────────────────────────────────────
DEF_LAB = dict(thr2="2 Ω·m", thr3="3 Ω·m", thr5="5 Ω·m", oracle="oracle √(ρ_lens·ρ_below)", maxgrad="max. gradient",
               logmid="log-midpoint", maxgrad_o="max. gradient (oracle anchor)", logmid_o="log-midpoint (oracle anchor)")


def f_(v, fmt="{:.2f}", none="—"):
    return none if v is None else fmt.format(v)


def sha256sums(folder):
    lines = []
    for root, _, files in os.walk(folder):
        for fn in sorted(files):
            if fn == "SHA256SUMS" or root.split(os.sep)[-1].startswith("_"):
                continue
            p = os.path.join(root, fn)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            lines.append(f"{h}  {os.path.relpath(p, folder).replace(os.sep, '/')}")
    lines.sort(key=lambda s: s.split("  ", 1)[1])
    with open(os.path.join(folder, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # the inversion cache is always read from the project; reports, archive and the summary go to --out
    d = SRC
    res = os.path.join(args.out, "results", "v1_3"); dpp = os.path.join(args.out, "data", "processed", "v1_3")
    for q_ in (res, os.path.join(res, "figs"), dpp):
        os.makedirs(q_, exist_ok=True)
    in_project = os.path.abspath(args.out) == os.path.abspath(wc.ROOT)
    s18 = _load("p18", "18_paper.py"); s18.mpl()
    done = [t for t in TRUTHS if all(os.path.exists(os.path.join(d, t, f"r{r}.npz")) for r in range(N_REAL))]
    per = {t: truth_metrics(d, t) for t in done}                      # recomputed from the saved runs
    rows = [r for t in done for r in per[t]]
    V = verdicts(rows)
    fit = fit_rule(rows, "lam50", seed=[136200])
    fit_lam = {int(l): fit_rule(rows, f"lam{int(l)}_r01", seed=[136201]) for l in [5.0, 10.0, 20.0, 50.0, 100.0]}
    grad = [t for t in TRUTHS if params(t)["tau"] > 0]
    fit_grad = fit_rule(rows, "lam50", truths=grad, seed=[136200])
    wq = (np.log10(3) - np.log10(WESTHOEK["rho_lens"])) / (np.log10(WESTHOEK["rho_below"]) - np.log10(WESTHOEK["rho_lens"]))

    # ── table T_S4 (CSV + Markdown) ──
    cols = ["truth", "rho_lens", "k", "z_b", "tau", "definition", "q", "treatment", "group", "n", "coverage", "bias",
            "n_missing", "scatter", "rbar"]
    main = [r for r in rows if r["treatment"] in ("lam50", "pooled")]
    with open(os.path.join(res, "T_S4_factorial.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in main:
            f.write(",".join("" if r[c] is None else (f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")
    M = ["# T_S4 — factorial results (Markdown copy of `T_S4_factorial.csv`)", "",
         "Coverage of the 90 % band / median bias (m, negative = too deep) / scatter ratio (λ = 50 only) / n pairs; "
         "groups by r at the true base (λ = 50).", ""]
    for g in GROUPS:
        M += [f"## r {'≤ 10 m' if g == '≤10' else '10–15 m'}", "",
              "| truth | r̄ (m) | " + " | ".join(f"{DEF_LAB[x]} λ50 | {DEF_LAB[x]} pooled" for x in DEFS) + " |",
              "|---|---|" + "---|---|" * len(DEFS)]
        for t in done:
            cells = []
            rb = pick(rows, truth=t, definition="thr3", treatment="lam50", group=g)
            for x in DEFS:
                for tr in ("lam50", "pooled"):
                    r = pick(rows, truth=t, definition=x, treatment=tr, group=g)[0]
                    cells.append("—" if not r["n"] else f"{100*r['coverage']:.0f}% / {f_(r['bias'], '{:+.2f}')}"
                                 + (f" / {f_(r['scatter'], '{:.2f}')}" if tr == "lam50" else "") + f" / {r['n']}")
            M.append(f"| {t} | {f_(rb[0]['rbar'], '{:.1f}') if rb else '—'} | " + " | ".join(cells) + " |")
        M.append("")
    with open(os.path.join(res, "T_S4_factorial.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(M) + "\n")

    # ── 2_factorial_results.md ──
    L = ["# 2 — Factorial synthetic study (Task 1)", "",
         f"{len(done)} of {len(TRUTHS)} truths finished. Design, seeds and metric definitions: `docs/decision_log.md` "
         "(v1.3, pre-registered). Full table: `T_S4_factorial.csv` / `T_S4_factorial.md`.", "",
         "## Truths", "", "| truth | ρ_lens | ρ_below | k | z_b | τ | q (2 / 3 / 5 Ω·m) | oracle (Ω·m) |", "|---|---|---|---|---|---|---|---|"]
    for t in TRUTHS:
        p = params(t)
        L.append(f"| {t} | {p['rho_lens']:g} | {p['rho_below']:.2f} | {p['k']} | {p['z_b']:g} | {p['tau']:g} | "
                 + " / ".join(f"{q_of(x, p):.2f}" for x in THRS) + f" | {10**oracle_level(p):.2f} |")
    L += ["", "q outside (0, 1): the threshold lies outside the lens–below contrast (its crossing falls on the deep unit or "
          "is absent).", "", "## Coverage / bias at λ = 50, r ≤ 10 m (per truth)", "",
          "| truth | r̄ (m) | n | " + " | ".join(DEF_LAB[x] for x in DEFS) + " |", "|---|---|---|" + "---|" * len(DEFS)]
    for t in done:
        rb = pick(rows, truth=t, definition="thr3", treatment="lam50", group="≤10")[0]
        cells = []
        for x in DEFS:
            r = pick(rows, truth=t, definition=x, treatment="lam50", group="≤10")[0]
            cells.append("—" if not r["n"] else f"{100*r['coverage']:.0f}% / {f_(r['bias'], '{:+.2f}')}")
        L.append(f"| {t} | {f_(rb['rbar'], '{:.1f}')} | {rb['n']} | " + " | ".join(cells) + " |")
    L += ["", "## Summary over truths (λ = 50 and λ-pooled; r ≤ 10 m)", "",
          "| definition | treatment | truths with data | median coverage | truths with coverage ≥ 85 % | median \\|bias\\| (m) | median scatter ratio |",
          "|---|---|---|---|---|---|---|"]
    agg = {}
    for x in DEFS:
        for tr in ("lam50", "pooled"):
            rr = [r for r in pick(rows, definition=x, treatment=tr, group="≤10") if r["n"]]
            if not rr:
                continue
            cv = [r["coverage"] for r in rr]; bb = [abs(r["bias"]) for r in rr if r["bias"] is not None]
            sc = [r["scatter"] for r in rr if r["scatter"] is not None]
            agg[f"{x}_{tr}"] = dict(n=len(rr), cov_med=float(np.median(cv)), n85=int(sum(c >= 0.85 for c in cv)),
                                    absbias_med=float(np.median(bb)) if bb else None, scatter_med=float(np.median(sc)) if sc else None)
            a_ = agg[f"{x}_{tr}"]
            L.append(f"| {DEF_LAB[x]} | {tr} | {a_['n']} | {100*a_['cov_med']:.0f}% | {a_['n85']} | "
                     f"{f_(a_['absbias_med'])} | {f_(a_['scatter_med'])} |")
    L += ["", "## Pre-registered expectations", ""]
    e9, e10, e11 = V["E9"], V["E10"], V["E11"]
    L.append(f"- **E9** (sign of the fixed-contour bias predicted by q, ≥ 90 % of cases): **{'holds' if e9['holds'] else 'does not hold'}** — "
             f"{e9['n_ok']} of {e9['n']} cases ({100*(e9['share'] or 0):.0f} %)."
             + (" Mismatches: " + "; ".join(f"{c['truth']} {c['d']} (q = {c['q']:.2f}, bias {c['bias']:+.2f} m)" for c in e9["fails"]) if e9["fails"] else ""))
    L.append(f"- **E10** (oracle: \\|bias\\| ≤ 0.5 m and coverage ≥ 80 % in ≥ 80 % of sharp truths): **{'holds' if e10['holds'] else 'does not hold'}** — "
             f"{e10['n_ok']} of {e10['n']} sharp truths with r ≤ 10 m data ({100*(e10['share'] or 0):.0f} %)"
             + (f"; no r ≤ 10 m data: {', '.join(e10['no_data'])}" if e10["no_data"] else "") + ".")
    a, b = e11["all"]["logmid"], e11["all"]["maxgrad"]
    L.append(f"- **E11** (log-midpoint and max. gradient coverage ≥ 85 % in ≥ 80 % of all truths): **{'holds' if e11['holds'] else 'does not hold'}** — "
             f"log-midpoint {a['n_ok']}/{a['n']} ({100*(a['share'] or 0):.0f} %), max. gradient {b['n_ok']}/{b['n']} ({100*(b['share'] or 0):.0f} %). "
             f"Pre-registered breakdowns: truths with 3 Ω·m inside the contrast ({e11['n_inside3']}): log-midpoint "
             f"{e11['inside3']['logmid']['n_ok']}/{e11['inside3']['logmid']['n']}, max. gradient {e11['inside3']['maxgrad']['n_ok']}/{e11['inside3']['maxgrad']['n']}; "
             f"oracle-anchored diagnostic, all truths: log-midpoint {e11['oracle_anchor']['logmid_o']['n_ok']}/{e11['oracle_anchor']['logmid_o']['n']}, "
             f"max. gradient {e11['oracle_anchor']['maxgrad_o']['n_ok']}/{e11['oracle_anchor']['maxgrad_o']['n']}.")
    if fit:
        L.append(f"- **E12** (R² ≥ 0.5 of the one-constant fit): **{'holds' if fit['r2'] >= 0.5 else 'does not hold'}** — "
                 f"R² = {fit['r2']:.2f}, a = {fit['a']:.3f} [{fit['a90'][0]:.3f}, {fit['a90'][1]:.3f}], n = {fit['n']} (`3_rule_fit.md`).")
    with open(os.path.join(res, "2_factorial_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # ── 3_rule_fit.md ──
    R = ["# 3 — Test of the offset rule (Task 2)", "",
         "Rule: an inverted log-resistivity profile that is an error function of width s around the true interface "
         "crosses the threshold ρ_t at Δz = s·Φ⁻¹(q), q = (log ρ_t − log ρ_lens)/(log ρ_below − log ρ_lens); as a bias in "
         "elevation (negative = too deep): bias = −s·Φ⁻¹(q), with s = a·r̄ (r̄ = mean resolution radius at the true base, "
         "λ = 50, r ≤ 10 m). Least squares through the origin over the (truth × threshold) cases with 0 < q < 1; bootstrap "
         "over truths (2000 resamples).", ""]
    if fit:
        R += [f"## Sharp truths, λ = 50 (realisations 0–4)", "",
              f"a = **{fit['a']:.3f}** (90 % bootstrap [{fit['a90'][0]:.3f}, {fit['a90'][1]:.3f}]), R² = **{fit['r2']:.2f}**, "
              f"n = {fit['n']} cases; s = a·r̄ ≈ {fit['a']*np.median([p['rbar'] for p in fit['pts']]):.2f} m at the median r̄ "
              f"({np.median([p['rbar'] for p in fit['pts']]):.1f} m).", "",
              "| truth | contour | q | r̄ (m) | observed bias | predicted | residual |", "|---|---|---|---|---|---|---|"]
        R += [f"| {p['truth']} | {p['d'][3:]} Ω·m | {p['q']:.2f} | {p['rbar']:.1f} | {p['y']:+.2f} | {p['pred']:+.2f} | {p['resid']:+.2f} |"
              for p in fit["pts"]]
        for key in ("k", "z_b", "rho_lens"):
            groups = sorted({p[key] for p in fit["pts"]})
            R.append("")
            R.append(f"Median residual by {key}: " + ", ".join(
                f"{g:g}: {np.median([p['resid'] for p in fit['pts'] if p[key] == g]):+.2f} m" for g in groups))
    R += ["", "## One constant per λ (realisations 0–1, the same r̄ from λ = 50)", "",
          "| λ | a | 90 % bootstrap | R² | n |", "|---|---|---|---|---|"]
    for l, fl in fit_lam.items():
        if fl:
            R.append(f"| {l} | {fl['a']:.3f} | [{fl['a90'][0]:.3f}, {fl['a90'][1]:.3f}] | {fl['r2']:.2f} | {fl['n']} |")
    if fit_grad:
        R += ["", "## Gradual truths (τ = 5 m), λ = 50", "",
              f"a = {fit_grad['a']:.3f} (90 % [{fit_grad['a90'][0]:.3f}, {fit_grad['a90'][1]:.3f}]), R² = {fit_grad['r2']:.2f}, "
              f"n = {fit_grad['n']}; sharp-truth a = {fit['a']:.3f}" + (" — the gradual truths behave as a wider s." if fit and fit_grad['a'] > fit['a90'][1]
                                                                         else " — not wider than the sharp-truth interval." if fit else "")]
    with open(os.path.join(res, "3_rule_fit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R) + "\n")

    # ── figures ──
    figs = os.path.join(res, "figs")
    cmap = {2: "#9ecae1", 4: "#4292c6", 8: "#08519c", 16: "#08306b"}
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharey=True)
    for ax, x in zip(axes.ravel(), DEFS):
        for r in pick(rows, definition=x, treatment="lam50", group="≤10"):
            if not r["n"]:
                continue
            qq = r["q"] if x.startswith("thr") else q_of(3.0, params(r["truth"]))
            mk = "o" if r["tau"] == 0 else "^"
            ax.scatter(qq, 100 * r["coverage"], c=cmap[r["k"]], marker=mk, s=22, edgecolor="k" if r["rho_lens"] == 0.8 else "none", lw=0.5, zorder=3)
        ax.axhline(90, color="0.3", lw=0.8, ls="--")
        ax.axvspan(1.0, 3.0, color="0.92", zorder=0); ax.axvline(0.5, color="0.6", lw=0.6, ls=":")
        ax.axvline(wq, color="#d35400", lw=1.0); ax.set_xlim(0, 2.8); ax.set_ylim(-3, 103)
        ax.set_title(DEF_LAB[x], fontsize=8)
        ax.set_xlabel("q of this contour" if x.startswith("thr") else "q of the 3 Ω·m contour (truth)", fontsize=7)
    axes[0, 0].set_ylabel("coverage of the 90 % band (%)"); axes[1, 0].set_ylabel("coverage of the 90 % band (%)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=cmap[k], label=f"k = {k}") for k in KS] + \
        [Line2D([], [], marker="^", ls="", color="0.5", label="τ = 5 m"), Line2D([], [], marker="o", ls="", mfc="0.5", mec="k", label="ρ_lens = 0.8"),
         Line2D([], [], color="0.3", ls="--", label="nominal 90 %"), Line2D([], [], color="#d35400", label=f"Westhoek, 3 Ω·m (q = {wq:.2f})")]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=7, frameon=False)
    fig.suptitle("Coverage vs q (λ = 50, r ≤ 10 m); grey: threshold outside the lens–below contrast", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(os.path.join(figs, "F_10_coverage_map.png"), dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    if fit:
        ax = axes[0]
        for p in fit["pts"]:
            ax.scatter(p["x"] * fit["a"], p["y"], c=cmap[p["k"]], s=18, marker={"thr2": "s", "thr3": "o", "thr5": "D"}[p["d"]])
        if fit_grad:
            for p in fit_grad["pts"]:
                ax.scatter(p["x"] * fit["a"], p["y"], facecolor="none", edgecolor="#d35400", s=24, marker="^")
        lim = max(1, np.nanmax(np.abs([p["y"] for p in fit["pts"]] + [p["x"] * fit["a"] for p in fit["pts"]]))) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("predicted bias −a·r̄·Φ⁻¹(q) (m)"); ax.set_ylabel("observed median bias (m)")
        ax.set_title(f"(a) λ = 50: a = {fit['a']:.3f} [{fit['a90'][0]:.3f}, {fit['a90'][1]:.3f}], R² = {fit['r2']:.2f}", fontsize=8)
        ax.grid(alpha=0.3)
    ax = axes[1]
    lc = {5: "#fdae6b", 10: "#fd8d3c", 20: "#e6550d", 50: "#a63603", 100: "#7f2704"}
    for l, fl in fit_lam.items():
        if fl:
            ax.scatter([p["x"] * fl["a"] for p in fl["pts"]], [p["y"] for p in fl["pts"]], c=lc[l], s=14,
                       label=f"λ = {l}: a = {fl['a']:.3f}, R² = {fl['r2']:.2f}")
    lim2 = ax.get_xlim(); m_ = max(abs(lim2[0]), abs(lim2[1]), 1)
    ax.plot([-m_, m_], [-m_, m_], "k-", lw=0.8); ax.set_xlabel("predicted bias −a_λ·r̄·Φ⁻¹(q) (m)"); ax.set_ylabel("observed median bias (m)")
    ax.set_title("(b) one a per λ (realisations 0–1)", fontsize=8); ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "F_11_bias_rule.png"), dpi=300); plt.close(fig)

    if fit:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
        for ax, key in zip(axes, ("k", "z_b", "rho_lens")):
            ax.scatter([p[key] for p in fit["pts"]], [p["resid"] for p in fit["pts"]], s=14, c="#1f4e9c")
            ax.axhline(0, color="k", lw=0.8); ax.set_xlabel(key)
            if key == "k":
                ax.set_xscale("log", base=2)
        axes[0].set_ylabel("residual (observed − predicted), m")
        fig.suptitle("Rule-fit residuals (sharp truths, λ = 50, r ≤ 10 m)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(figs, "F_S8_rule_residuals.png"), dpi=300); plt.close(fig)

    # F_S7: heat map of coverage (λ = 50, r ≤ 10 m), truth × definition
    C = np.full((len(done), len(DEFS)), np.nan); Bm = np.full_like(C, np.nan)
    for i, t in enumerate(done):
        for j, x in enumerate(DEFS):
            r = pick(rows, truth=t, definition=x, treatment="lam50", group="≤10")[0]
            if r["n"]:
                C[i, j] = 100 * r["coverage"]; Bm[i, j] = r["bias"] if r["bias"] is not None else np.nan
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(done) + 1.6))
    im = ax.imshow(C, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if np.isfinite(C[i, j]):
                ax.text(j, i, f"{C[i, j]:.0f}|{Bm[i, j]:+.1f}" if np.isfinite(Bm[i, j]) else f"{C[i, j]:.0f}|—", ha="center", va="center", fontsize=5)
    ax.set_xticks(range(len(DEFS))); ax.set_xticklabels([DEF_LAB[x] for x in DEFS], rotation=30, ha="right", fontsize=6)
    ax.set_yticks(range(len(done))); ax.set_yticklabels(done, fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label("coverage (%)", fontsize=7)
    ax.set_title("Coverage (%) | median bias (m), λ = 50, r ≤ 10 m (full table: T_S4_factorial.csv)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "F_S7_factorial_table.png"), dpi=300); plt.close(fig)

    # ── machine-readable summary (read by 18_paper.py) ──
    summ = dict(n_truths=len(done), westhoek_q3=float(wq), verdicts=V, agg=agg,
                fit={k: v for k, v in (fit or {}).items() if k != "pts"},
                fit_lam={str(l): ({k: v for k, v in fl.items() if k != "pts"} if fl else None) for l, fl in fit_lam.items()},
                fit_grad={k: v for k, v in (fit_grad or {}).items() if k != "pts"})
    with open(os.path.join(dpp, "v13_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, default=float)

    # ── archive + checksums ──
    arc = os.path.join(res, "archive"); os.makedirs(arc, exist_ok=True)
    for t in done:
        os.makedirs(os.path.join(arc, t), exist_ok=True)
        for fn, obj in (("config.json", config(t)), ("summary.json", per[t])):
            with open(os.path.join(arc, t, fn), "w", encoding="utf-8", newline="\n") as f:
                json.dump(obj, f, indent=1)
        shutil.copy2(os.path.join(d, t, "truth_spec.json"), os.path.join(arc, t, "truth_spec.json"))
    for fn in ("T_S4_factorial.csv", "T_S4_factorial.md"):
        shutil.copy2(os.path.join(res, fn), os.path.join(arc, fn))
    n2 = sha256sums(arc)
    n1 = sha256sums(d) if in_project else "not rewritten (rebuild outside the project)"
    print("\n".join(L))
    print(f"[report] {len(done)} truths; SHA256SUMS: {n1} files (data), {n2} files (archive)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("field").add_argument("--out", default=wc.ROOT)
    for c in ("run", "resume"):
        s = sub.add_parser(c)
        s.add_argument("--out", default=wc.ROOT); s.add_argument("--workers", type=int, default=6)
        s.add_argument("--time-only", action="store_true")
    sub.add_parser("report").add_argument("--out", default=wc.ROOT)
    w = sub.add_parser("worker")
    w.add_argument("--worker-id", type=int, required=True); w.add_argument("--out", default=wc.ROOT)
    w.add_argument("--jobs", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "resume":
        cmd_run(a, resume=True)
    else:
        dict(field=cmd_field, report=cmd_report, worker=cmd_worker)[a.cmd](a)


if __name__ == "__main__":
    main()
