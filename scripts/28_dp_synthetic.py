"""
28_dp_synthetic.py
==================
Westhoek ERT v1.5 — Task 1: synthetic truths built from the 2024 DP/CPT logs. EXPLORATORY; designed after the DP results
were known (docs/decision_log.md v1.5-0).

  run    [--workers 6] [--time-only] [--out ROOT]   truths T1–T3 × layouts L18, L24: fine-mesh forward, 5 noise realisations,
                                                    Stage-1 G1 rule, λ = 50 (+ λ = 5, 10, 20, 100 for realisations 0–1),
                                                    linearised posterior, interfaces at the DP x positions
                                                    → ROOT/data/processed/v1_5/dp_synth/; --time-only times 3 inversions
                                                    and writes ROOT/results/v1_5/1_plan.md
  resume [--workers 6] [--out ROOT]                 as run, skipping finished jobs
  report [--out ROOT]                               ROOT/results/v1_5/2_dp_synthetic.md, T_S10_dp_synthetic.{csv,md},
                                                    figs/F_14_synthetic_vs_field.png, F_S12_dp_truths.png
  worker                                            (internal)

Seeds: noise default_rng([150000, truth, layout, real]); posterior default_rng([150100, truth, layout, real, λ]); Stage-1
posterior (unused samples) default_rng([150200, truth, layout, real]). truth 1–3 (T1, T2, T3), layout 0 = L18, 1 = L24.
Reused: 10_synthetic_benchmark (FWD, error model, one_inversion), 11_stage4_closeout (interfaces_all, sample_defs,
band_stats, r_at), 02_land_inversion (build_datacont, run_inversion), 27_dp_groundtruth (DP logs, CPT topography,
steepest-increase log-midpoint, ERT.dat), 13_bel_full.pilot (forward simulation with retries).
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
read_trace.install("28_dp_synthetic.py")
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


g27 = _load("g27", "27_dp_groundtruth.py")
c11 = g27._load("c11", "11_stage4_closeout.py")
sb = c11.sb

TRUTHS = ["T1", "T2", "T3"]
ABL = ["T2a", "T2b", "T2c"]                                  # v1.5-1 ablation of the log-midpoint's shallow bias
N_REAL_ABL = 3
LAYOUTS = ["L18", "L24"]
N_REAL, N_REAL_LAM = 5, 2
LAMS = [50.0, 5.0, 10.0, 20.0, 100.0]
DEEP = 8.44017866592723                                    # v1.0 T1 deep median (truth_spec.json)
RHO_BRACKISH = 4.93
CLAY_DEFAULT = -27.5
BLOCK = 0.5
Z_DEEPEST = -35.0
EXPLORATORY = "Exploratory; designed after the DP results were known (decision log v1.5-0)."


def dirs(root):
    d = os.path.join(root, "data", "processed", "v1_5", "dp_synth")
    res = os.path.join(root, "results", "v1_5")
    for p in (d, os.path.join(d, "_logs"), os.path.join(d, "_work"), res, os.path.join(res, "figs")):
        os.makedirs(p, exist_ok=True)
    return d, res


def SRC():
    return os.path.join(wc.ROOT, "data", "processed", "v1_5", "dp_synth")


# ─────────────────────────────────────────────────────────────────────────────
# truths
# ─────────────────────────────────────────────────────────────────────────────
def dp_inputs():
    """Per DP (DP1 … DP7, sorted by x): x, ground, depth/ρ log, v1.4 base, ρ_lens, ρ_below, clay top, clay ρ."""
    v14 = os.path.join(wc.ROOT, "data", "processed", "v1_4")
    R = json.load(open(os.path.join(v14, "dp_results.json"), encoding="utf-8"))
    C = json.load(open(os.path.join(v14, "cpt.json"), encoding="utf-8"))
    logs = g27.read_dp_values()
    out = []
    for n in sorted(R["dps"], key=lambda k: R["dps"][k]["our_x"]):
        d = R["dps"][n]; o = d["per"]["primary|A"]; c = C["cpts"][d["cpt"]]
        lg = logs[min(logs, key=lambda k: abs(k - d["local_x"]))]
        z1 = c["z"] - lg["depth"]
        clay = c["rf_step"] if c["rf_step"] is not None else CLAY_DEFAULT
        below = z1 <= clay
        rclay = float(np.median(lg["r"][below])) if below.sum() * 0.01 >= 0.5 else DEEP
        out.append(dict(name=n, cpt=d["cpt"], x=d["our_x"], ground=c["z"], z1=z1, r=lg["r"], base=o["logmid"],
                        rho_lens=o["rho_lens"], rho_below=o["rho_below"], clay=clay, rho_clay=rclay,
                        clay_source="CPT Rf step" if c["rf_step"] is not None else "default −27.5 mTAW",
                        clay_rho_source="DP log below the clay top" if below.sum() * 0.01 >= 0.5 else "reference deep 8.44 Ω·m"))
    return out


def t1_profile(dp):
    """0.5 m block averages (in log10) of the 0.5 m running median; deepest 2 m median below; 8.44 below −35 mTAW."""
    from scipy.ndimage import median_filter
    sm = median_filter(np.log10(dp["r"]), size=51, mode="nearest"); z1 = dp["z1"]
    blk = np.floor(z1 / BLOCK).astype(int)
    vals = {b: float(np.mean(sm[blk == b])) for b in np.unique(blk)}
    bmin, bmax = min(vals), max(vals)
    deep = float(np.median(sm[z1 <= z1[-1] + 2.0]))
    def f(z):
        z = np.asarray(z, float); b = np.floor(z / BLOCK).astype(int)
        v = np.array([vals.get(int(k), np.nan) for k in b.ravel()]).reshape(z.shape)
        v = np.where(b > bmax, vals[bmax], v)
        v = np.where(b < bmin, deep, v)
        v = np.where(z < Z_DEEPEST, np.log10(DEEP), v)
        return np.where(np.isnan(v), deep, v)
    return f


def dune_ref(x, z):
    v = sb.load_spec()["values"]["T1"]
    return np.where(z > sb.Z_WT, np.log10(v["cap"]), np.where(z > sb.Z_DEEP, np.log10(v["fresh"]), np.log10(v["deep"])))


class TruthModel:
    def __init__(self, name):
        self.name = name; self.D = dp_inputs(); self.xs = np.array([d["x"] for d in self.D])
        if name == "T1":
            self.prof = [t1_profile(d) for d in self.D]
        else:
            self.par = {k: np.array([d[k] for d in self.D]) for k in ("base", "clay")}
            self.lr = {k: np.log10(np.array([d[k] for d in self.D])) for k in ("rho_lens", "rho_below", "rho_clay")}
            if name == "T3":
                self.lr["rho_below"] = np.full(len(self.D), np.log10(RHO_BRACKISH))
            if name == "T2a":                                    # flat base at the DP3–DP7 mean
                m = [i for i, d in enumerate(self.D) if d["name"] in ("DP3", "DP4", "DP5", "DP6", "DP7")]
                self.par["base"] = np.full(len(self.D), float(np.mean(self.par["base"][m])))
            if name == "T2b":                                    # no clay: ρ_below down to −35 mTAW, 8.44 Ω·m below
                self.par["clay"] = np.full(len(self.D), Z_DEEPEST)
                self.lr["rho_clay"] = np.full(len(self.D), np.log10(DEEP))
            if name == "T2c":                                    # ρ_lens = 1.5 Ω·m everywhere (DP1's top unit → lens)
                self.lr["rho_lens"] = np.full(len(self.D), np.log10(1.5))

    def log10rho(self, x, z):
        x = np.asarray(x, float); z = np.asarray(z, float); shp = np.broadcast(x, z).shape
        x = np.broadcast_to(x, shp).ravel(); z = np.broadcast_to(z, shp).ravel()
        xc = np.clip(x, self.xs[0], self.xs[-1])
        if self.name == "T1":
            P = np.array([p(z) for p in self.prof])                  # (n_dp, n)
            j = np.clip(np.searchsorted(self.xs, xc) - 1, 0, len(self.xs) - 2)
            w = (xc - self.xs[j]) / (self.xs[j + 1] - self.xs[j]); n = np.arange(len(x))
            v = (1 - w) * P[j, n] + w * P[j + 1, n]
        else:
            I = lambda arr: np.interp(xc, self.xs, arr)
            base, clay = I(self.par["base"]), I(self.par["clay"])
            v = np.where(z > base, I(self.lr["rho_lens"]), np.where(z > clay, I(self.lr["rho_below"]), I(self.lr["rho_clay"])))
        return np.where(x < sb.X_LENS0, dune_ref(x, z), v).reshape(shp)


# ─────────────────────────────────────────────────────────────────────────────
# layouts, forward
# ─────────────────────────────────────────────────────────────────────────────
def layout(name):
    """Electrode dict {number: (x, z)}, quadrupoles (C1, C2, P1, P2 numbers), surface (xs, zs) of the layout."""
    if name == "L18":
        land, _ = wc.read_electrodes()
        elec = {i + 1: (float(x), float(z)) for i, (x, z) in enumerate(land)}
        L = open(os.path.join(wc.ROOT, "data", "raw", "land_data.dat")).read().split("\n")
        n = int(L[0].split()[0]); D = np.array([[float(v) for v in l.split()[:5]] for l in L[1:1 + n]])
        q = {k: D[:, i].astype(int) for i, k in enumerate(("C1", "C2", "P1", "P2"))}
        return elec, q, (land[:, 0], land[:, 1])
    E = g27.read_ert(); topo = g27.cpt_topography()
    n_el = len(E["xloc"]); num = n_el - np.arange(n_el); xs = g27.X0_CPT - E["xloc"]; zs = topo(xs)
    elec = {int(num[i]): (float(xs[i]), float(zs[i])) for i in range(n_el)}
    q = {k: np.array([int(num[v]) for v in E["abmn"][:, c]]) for c, k in enumerate(("C1", "C2", "P1", "P2"))}
    o = np.argsort(xs)
    return elec, q, (xs[o], zs[o])


def scheme_of(elec, q):
    import pygimli as pg
    data = pg.DataContainerERT(); idx = {}
    for k, e in enumerate(sorted(elec)):
        data.createSensor(pg.Pos(*elec[e])); idx[e] = k
    data.resize(len(q["C1"]))
    for key, col in (("a", "C1"), ("b", "C2"), ("m", "P1"), ("n", "P2")):
        data[key] = [idx[int(v)] for v in q[col]]
    data["k"] = np.ones(data.size())
    return data


def forward(d, truth, lay):
    p = os.path.join(d, f"fwd_{truth}_{lay}.npz")
    if os.path.exists(p):
        return np.load(p)
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    elec, q, _ = layout(lay); scheme = scheme_of(elec, q)
    with contextlib.redirect_stdout(io.StringIO()):
        fmesh = mt.createParaMesh(scheme.sensors(), **sb.FWD)
    cx = np.array([c.center().x() for c in fmesh.cells()]); cz = np.array([c.center().y() for c in fmesh.cells()])
    v = TruthModel(truth).log10rho(cx, cz)
    for _ in range(3):
        with contextlib.redirect_stdout(io.StringIO()):
            out = ert.simulate(fmesh, scheme=scheme_of(elec, q), res=10 ** v, noiseLevel=0, verbose=False)
        R = np.array(out["rhoa"])
        if len(R) == len(q["C1"]):
            break
    else:
        raise RuntimeError(f"forward {truth} {lay}: wrong number of data")
    np.savez(p + ".tmp.npz", R=R, cells=fmesh.cellCount(), cell_log10=v.astype(np.float32))
    os.replace(p + ".tmp.npz", p)
    return np.load(p)


# ─────────────────────────────────────────────────────────────────────────────
# inversion job
# ─────────────────────────────────────────────────────────────────────────────
def dp_x():
    return np.array([d["x"] for d in dp_inputs()])


def run_job(d, truth, lay, real, m02, log):
    out = os.path.join(d, f"{truth}_{lay}_r{real}.npz")
    if os.path.exists(out):
        log(f"skip {truth} {lay} r{real}"); return
    t0 = time.time(); li = LAYOUTS.index(lay)
    abl = truth in ABL
    ti = ABL.index(truth) + 1 if abl else TRUTHS.index(truth) + 1
    s_noise, s_post, s_pre = (151000, 151100, 151200) if abl else (150000, 150100, 150200)
    R_true = forward(d, truth, lay)["R"]
    elec, q, (sx, sz) = layout(lay)
    rng = np.random.default_rng([s_noise, ti, li, real])
    R = R_true + np.sqrt((sb.ERR_REL * R_true) ** 2 + sb.ERR_ABS ** 2) * rng.standard_normal(len(R_true))
    pos = R > 0
    xcol = dp_x()
    with g27.surface_override(sx, sz):
        land = dict({k: v[pos] for k, v in q.items()}, R=R[pos], n=int(pos.sum()), err=np.zeros(int(pos.sum())))
        data = m02.build_datacont(land, elec, err_rel=sb.ERR_REL)[0]
        mgr = m02.run_inversion(data, lam=20.0, para_depth=sb.PARA_DEPTH, para_dx=sb.PARA_DX, max_iter=sb.MAX_ITER)
        ncell = mgr.paraDomain.cellCount(); cs = wc.ColumnSampler(g27.pg_mesh_copy(mgr.paraDomain), xcol)
        pre, _ = sb.one_inversion(m02, land, elec, 20.0, np.zeros(ncell), cs, [s_pre, ti, li, real])
        r = pre["resid"]; med = np.median(r); mad = np.median(np.abs(r - med)); keep = np.abs(r) <= 3 * 1.4826 * mad
        land2 = dict({k: v[keep] for k, v in land.items() if k in ("C1", "C2", "P1", "P2")}, R=land["R"][keep], n=int(keep.sum()),
                     err=np.zeros(int(keep.sum())))
        save = dict(truth=truth, layout=lay, real=real, xcol=xcol, z=cs.z, n_data=int(pos.sum()), n_kept=int(keep.sum()),
                    stage1_chi2=pre["chi2"])
        lams = LAMS if (abl or real < N_REAL_LAM) else [50.0]; pooled = []
        for lam in lams:
            info, S = sb.one_inversion(m02, land2, elec, lam, np.zeros(ncell), cs, [s_post, ti, li, real, int(lam)])
            V = cs(np.log10(info["model"].astype(float))); hat = c11.interfaces_all(V[:, :, None], cs.z); p = f"lam{int(lam)}_"
            save.update({p + "chi2": info["chi2"], p + "col": V.astype(np.float32)})
            for dd in ("thr3", "maxgrad", "logmid"):
                save[p + f"mhat_{dd}"] = hat[dd][:, 0]
            save[p + "r_logmid"] = c11.r_at(cs, info["radius"], hat["logmid"][:, 0])
            save[p + "mhat_steep"] = np.array([g27.steep_logmid(V[:, j], cs.z) for j in range(len(xcol))])
            bands = c11.sample_defs(cs, S)
            for dd in ("thr3", "logmid"):
                save[p + f"band_{dd}"] = bands[dd]
            if abl or real < N_REAL_LAM:
                pooled.append(S.astype(np.float32))
            log(f"{truth} {lay} r{real} λ={lam:g}: χ²={info['chi2']:.2f}")
        if pooled:
            P = np.hstack(pooled).astype(np.float64); pb = c11.sample_defs(cs, P)
            for dd in ("thr3", "logmid"):
                save[f"pooled_band_{dd}"] = pb[dd]
            Vs = cs(P / np.log(10))
            st = np.array([[g27.steep_logmid(Vs[:, j, k], cs.z) for k in range(Vs.shape[2])] for j in range(len(xcol))])
            save["pooled_band_steep"] = c11.band_stats(st)
    save["t_job"] = time.time() - t0
    np.savez(out + ".tmp.npz", **save); os.replace(out + ".tmp.npz", out)
    log(f"done {truth} {lay} r{real} in {save['t_job']:.0f}s")


def cmd_worker(args):
    d, _ = dirs(args.out)
    m02 = c11._load("inv02", "02_land_inversion.py"); m02.VERBOSE = False
    logf = open(os.path.join(d, "_logs", f"worker_{args.worker_id}.log"), "a", encoding="utf-8")
    log = lambda m: (logf.write(time.strftime("%H:%M:%S ") + m + "\n"), logf.flush())
    for job in args.jobs:
        kind, truth, lay, real = job.split(":")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if kind == "F":
                    forward(d, truth, lay); log(f"forward {truth} {lay}")
                else:
                    run_job(d, truth, lay, int(real), m02, log)
        except Exception as e:
            import traceback
            log(f"FAILED {job}: {e}\n{traceback.format_exc()}")
    log("worker finished")


def launch(d, jobs, workers, out_root):
    parts = [jobs[i::workers] for i in range(workers)]; procs = []
    for w, part in enumerate(parts):
        if not part:
            continue
        wd = os.path.join(d, "_work", f"w{w}"); os.makedirs(wd, exist_ok=True)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", "--worker-id", str(w), "--out", out_root,
                                       "--jobs"] + part, cwd=wd, env=dict(os.environ, OMP_NUM_THREADS="2", MPLBACKEND="Agg"),
                                      stdout=subprocess.DEVNULL, stderr=open(os.path.join(d, "_logs", f"worker_{w}.err"), "a")))
    for p in procs:
        p.wait()


def cmd_run(args):
    d, res = dirs(args.out); root = os.path.abspath(args.out); t0 = time.time()
    if args.time_only:
        m02 = c11._load("inv02", "02_land_inversion.py"); m02.VERBOSE = False
        tf = {}
        for lay in LAYOUTS:
            t = time.time(); forward(d, "T2", lay); tf[lay] = time.time() - t
        # time: one realisation-2 job of T2 on each layout = Stage-1 + λ = 50 (3 inversions in all)
        tj = {}
        logs = []
        for lay in LAYOUTS:
            t = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                run_job(d, "T2", lay, 2, m02, lambda m: logs.append(m))
            tj[lay] = (time.time() - t) / 2
        plan(res, tf, tj, args.workers); return
    TT, NR = (ABL, N_REAL_ABL) if args.ablation else (TRUTHS, N_REAL)
    launch(d, [f"F:{t}:{l}:0" for t in TT for l in LAYOUTS if not os.path.exists(os.path.join(d, f"fwd_{t}_{l}.npz"))],
           args.workers, root)
    jobs = [f"J:{t}:{l}:{r}" for r in range(NR) for t in TT for l in LAYOUTS
            if not os.path.exists(os.path.join(d, f"{t}_{l}_r{r}.npz"))]
    jobs.sort(key=lambda j: int(j.split(":")[3]) >= N_REAL_LAM)
    print(f"[run] {len(jobs)} jobs on {args.workers} workers", flush=True)
    launch(d, jobs, args.workers, root)
    n = sum(os.path.exists(os.path.join(d, f"{t}_{l}_r{r}.npz")) for t in TT for l in LAYOUTS for r in range(NR))
    print(f"[run] {n}/{len(TT) * len(LAYOUTS) * NR} jobs done ({time.time() - t0:.0f} s)", flush=True)


def plan(res, tf, tj, workers):
    n_inv = len(TRUTHS) * len(LAYOUTS) * (N_REAL * 2 + N_REAL_LAM * 4)
    per = {l: tj[l] for l in LAYOUTS}
    serial = sum(len(TRUTHS) * (N_REAL * 2 + N_REAL_LAM * 4) * per[l] for l in LAYOUTS) + 4 * sum(tf.values())
    wall = serial / workers * 2.0
    L = ["# 1 — Run plan (v1.5)", "", EXPLORATORY, "",
         "Timed: forward of T2 on the fine mesh " + ", ".join(f"{l} {tf[l]:.0f} s" for l in LAYOUTS) + "; four inversions "
         "(T2, realisation 2: Stage-1 + λ = 50 on each layout, plus a mesh build; each inversion incl. resolution, 1000 samples and the interfaces) "
         "mean " + ", ".join(f"{l} {per[l]:.0f} s" for l in LAYOUTS) + " per inversion, one process.", "",
         f"- Task 1: {n_inv} inversions (3 truths × 2 layouts × (5 Stage-1 + 5 λ = 50 + 2 × 4 extra λ)); serial ≈ {serial / 3600:.1f} h; "
         f"with {workers} workers and an assumed 2× slowdown under load (v1.3 measured 2.3×) ≈ **{wall / 3600:.1f} h**.",
         f"- Limit in the brief: 24 h → {'within; full design run' if wall <= 24 * 3600 else 'EXCEEDED: stop and ask'}.",
         "- Task 2 (forward only): 42 + 4 fine-mesh forward runs ≈ " + f"{(21 * sum(tf.values()) + 2 * sum(tf.values())) / 60:.0f} min serial.",
         "- Task 3 (BEL v4): timed separately (`4_bel_v4.md`)."]
    with open(os.path.join(res, "1_plan.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "resume"):
        s = sub.add_parser(c); s.add_argument("--out", default=wc.ROOT); s.add_argument("--workers", type=int, default=6)
        s.add_argument("--time-only", action="store_true"); s.add_argument("--ablation", action="store_true")
    sub.add_parser("report").add_argument("--out", default=wc.ROOT)
    w = sub.add_parser("worker"); w.add_argument("--worker-id", type=int, required=True); w.add_argument("--out", default=wc.ROOT)
    w.add_argument("--jobs", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd in ("run", "resume"):
        cmd_run(a)
    elif a.cmd == "report":
        cmd_report(a)
    else:
        cmd_worker(a)


def metrics():
    """Per truth, layout, DP: offsets from the true base, coverage, damping and contrast (decision log v1.5-0)."""
    D = dp_inputs(); out = {}
    for t in TRUTHS:
        for lay in LAYOUTS:
            Z = [np.load(os.path.join(SRC(), f"{t}_{lay}_r{r}.npz"), allow_pickle=True) for r in range(N_REAL)]
            z = Z[0]["z"]
            for j, dp in enumerate(D):
                base = dp["base"]; rb_true = RHO_BRACKISH if t == "T3" else dp["rho_below"]; rl_true = dp["rho_lens"]
                thr3 = np.array([float(zz["lam50_mhat_thr3"][j]) for zz in Z]); lm50 = np.array([float(zz["lam50_mhat_logmid"][j]) for zz in Z])
                cov3 = [bool(b_[0] >= 0.5 and b_[2] <= base <= b_[3]) for b_ in (zz["lam50_band_thr3"][:, j] for zz in Z)]
                pl = [zz["pooled_band_logmid"][:, j] for zz in Z[:N_REAL_LAM]]; ps = [zz["pooled_band_steep"][:, j] for zz in Z[:N_REAL_LAM]]
                rbi, rli = [], []
                for zz, b_ in zip(Z, lm50):
                    col = zz["lam50_col"][:, j].astype(float); ok = np.isfinite(col)
                    w = ok & (z <= b_ - 4) & (z >= b_ - 10); u = ok & (z >= b_ + 4)
                    rbi.append(np.median(col[w]) if w.any() else np.nan); rli.append(np.median(col[u]) if u.any() else np.nan)
                rbi_m, rli_m = float(np.nanmedian(rbi)), float(np.nanmedian(rli))
                ratio = (rbi_m - rli_m) / (np.log10(rb_true) - np.log10(rl_true)) if rb_true != rl_true else np.nan
                med = lambda v: float(np.nanmedian(v))
                out[f"{t}|{lay}|{dp['name']}"] = dict(
                    truth=t, layout=lay, dp=dp["name"], x=dp["x"], base=base, rho_below_true=rb_true, rho_lens_true=rl_true,
                    thr3=med(thr3), thr3_off=med(thr3) - base, logmid50_off=med(lm50) - base,
                    logmid_pooled=med([b_[1] for b_ in pl]), logmid_off=med([b_[1] for b_ in pl]) - base,
                    steep_pooled=med([b_[1] for b_ in ps]), steep_off=med([b_[1] for b_ in ps]) - base,
                    cov_thr3=float(np.mean(cov3)), cov_logmid=float(np.mean([b_[0] >= 0.5 and b_[2] <= base <= b_[3] for b_ in pl])),
                    cov_steep=float(np.mean([b_[0] >= 0.5 and b_[2] <= base <= b_[3] for b_ in ps])),
                    rho_below_img=float(10 ** rbi_m), rho_lens_img=float(10 ** rli_m), contrast_ratio=float(ratio),
                    chi2_50=med([float(zz["lam50_chi2"]) for zz in Z]), n_kept=int(np.median([int(zz["n_kept"]) for zz in Z])),
                    n_data=int(Z[0]["n_data"]))
    return out


def field_values():
    R = json.load(open(os.path.join(wc.ROOT, "data", "processed", "v1_4", "dp_results.json"), encoding="utf-8"))
    out = {}
    for n, d in R["dps"].items():
        o = d["per"]["primary|A"]; b = o["logmid"]; s_ = o["sd"]
        out[n] = dict(L24=dict(thr3_off=s_["thr3"] - b, logmid_off=s_["band_logmid_pooled"][1] - b, steep_off=s_["band_steep_pooled"][1] - b,
                               rho_below_img=s_["rho_below_img"]),
                      L18=dict(thr3_off=o["thr3_lam50"][1] - b, logmid_off=o["logmid_pooled"][1] - b, steep_off=np.nan, rho_below_img=np.nan))
    return out


X_DPS = ("DP3", "DP4", "DP5", "DP6", "DP7")


def questions(M):
    g = lambda t, l, n: M[f"{t}|{l}|{n}"]
    x1 = [n for n in X_DPS if g("T1", "L24", n)["logmid_off"] >= 2.0 and abs(g("T1", "L24", n)["thr3_off"]) <= 2.5]
    x2 = {t: [n for n in X_DPS if g(t, "L24", n)["rho_below_img"] < 0.5 * g(t, "L24", n)["rho_below_true"]] for t in ("T1", "T2")}
    x3 = [n for n in X_DPS if abs(g("T3", "L24", n)["logmid_off"]) <= 2.0]
    return dict(X1=dict(ok=x1, holds=len(x1) >= 3), X2=dict(ok=x2, holds=all(len(v) >= 4 for v in x2.values())),
                X3=dict(ok=x3, holds=len(x3) >= 3))


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, res = dirs(args.out); figs = os.path.join(res, "figs")
    dpp = os.path.join(args.out, "data", "processed", "v1_5"); os.makedirs(dpp, exist_ok=True)
    s18 = _load("p18", "18_paper.py"); s18.mpl()
    M = metrics(); Fv = field_values(); Q = questions(M); D = dp_inputs(); names = [d["name"] for d in D]
    f_ = lambda v, fmt="{:+.2f}": "—" if v is None or not np.isfinite(v) else fmt.format(v)
    L = ["# 2 — Synthetic truths built from the 2024 DP/CPT logs (Task 1)", "", f"**{EXPLORATORY}**", "",
         "T1 = the 2024 logs (0.5 m blocks); T2 = three sharp units per DP (lens / clean-sand ρ_below / clay); T3 = T2 with "
         f"ρ_below = {RHO_BRACKISH:g} Ω·m (the 2018 BEL median, the \"brackish\" hypothesis). Layouts: L18 (2018 geometry; the truth is "
         "the 2024 state) and L24 (geometry of the 2025 ERT line, ERT.dat). 5 noise realisations; Stage-1 G1 rule, v1.0 error model, λ = 50 and the "
         "λ pool (5–100), linearised posterior; values at each DP x, offsets = image − true base (positive = image shallower). "
         "Medians over realisations; coverage = share of realisations whose 90 % band contains the true base.", "",
         "## Truth parameters per DP", "", "| DP | x (m) | true base (mTAW) | ρ_lens | ρ_below (T1/T2) | clay top (mTAW, source) | clay ρ (source) |",
         "|---|---|---|---|---|---|---|"]
    for d in D:
        L.append(f"| {d['name']} | {d['x']:.1f} | {d['base']:.2f} | {d['rho_lens']:.2f} | {d['rho_below']:.2f} | {d['clay']:.1f} ({d['clay_source']}) | "
                 f"{d['rho_clay']:.2f} ({d['clay_rho_source']}) |")
    T10 = ["truth,layout,dp,x,true_base,thr3_off,logmid_pooled_off,steep_pooled_off,logmid_lam50_off,cov_thr3,cov_logmid,cov_steep,"
           "rho_below_true,rho_below_img,img_over_true,contrast_ratio,chi2_lam50,n_kept,n_data"]
    for t in TRUTHS:
        for lay in LAYOUTS:
            L += ["", f"## {t} on {lay}", "",
                  "| DP | image 3 Ω·m − base | log-mid (pooled) − base | steepest-increase log-mid − base | cover 3 Ω·m / log-mid / steep | "
                  "image ρ below / true ρ below (Ω·m) | recovered / true log-contrast | field (v1.4): 3 Ω·m / log-mid − DP base |",
                  "|---|---|---|---|---|---|---|---|"]
            for n in names:
                m = M[f"{t}|{lay}|{n}"]; fv = Fv[n][lay]
                L.append(f"| {n} | {f_(m['thr3_off'])} | {f_(m['logmid_off'])} | {f_(m['steep_off'])} | {100*m['cov_thr3']:.0f} / {100*m['cov_logmid']:.0f} / "
                         f"{100*m['cov_steep']:.0f} % | {m['rho_below_img']:.2f} / {m['rho_below_true']:.2f} | {f_(m['contrast_ratio'], '{:.2f}')} | "
                         f"{f_(fv['thr3_off'])} / {f_(fv['logmid_off'])} |")
                T10.append(",".join(str(v) for v in [t, lay, n, f"{m['x']:.1f}", f"{m['base']:.2f}", f"{m['thr3_off']:.3f}", f"{m['logmid_off']:.3f}",
                                                      f"{m['steep_off']:.3f}", f"{m['logmid50_off']:.3f}", f"{m['cov_thr3']:.2f}", f"{m['cov_logmid']:.2f}",
                                                      f"{m['cov_steep']:.2f}", f"{m['rho_below_true']:.2f}", f"{m['rho_below_img']:.2f}",
                                                      f"{m['rho_below_img'] / m['rho_below_true']:.3f}", f"{m['contrast_ratio']:.3f}",
                                                      f"{m['chi2_50']:.2f}", m["n_kept"], m["n_data"]]))
            ch = [M[f"{t}|{lay}|{n}"]["chi2_50"] for n in names][0]
            L.append(f"\nλ = 50 χ² (median over realisations): {ch:.2f}; data kept by the Stage-1 rule: {M[f'{t}|{lay}|{names[0]}']['n_kept']} of "
                     f"{M[f'{t}|{lay}|{names[0]}']['n_data']}.")
    x1, x2, x3 = Q["X1"], Q["X2"], Q["X3"]
    L += ["", "## Questions (exploratory)", "",
          f"- **X1** (T1 on L24 reproduces the field pattern: at ≥ 3 of DP3–DP7 the log-midpoint ≥ 2 m shallower and the 3 Ω·m within ±2.5 m): "
          f"**{'yes' if x1['holds'] else 'no'}** — {len(x1['ok'])} of 5 ({', '.join(x1['ok']) or 'none'}).",
          f"- **X2** (image ρ below < 0.5 × true at ≥ 4 of DP3–DP7, T1 and T2 on L24): **{'yes' if x2['holds'] else 'no'}** — T1 "
          f"{len(x2['ok']['T1'])} of 5, T2 {len(x2['ok']['T2'])} of 5.",
          f"- **X3** (T3 on L24: log-midpoint within ±2 m at ≥ 3 of DP3–DP7): **{'yes' if x3['holds'] else 'no'}** — {len(x3['ok'])} of 5 "
          f"({', '.join(x3['ok']) or 'none'}).",
          "- **X4** (T1 vs T2, L24): see the table below."]
    L += ["", "| DP | log-mid − base: T1 / T2 / T3 | 3 Ω·m − base: T1 / T2 / T3 | image ρ below: T1 / T2 / T3 |", "|---|---|---|---|"]
    for n in names:
        g = [M[f"{t}|L24|{n}"] for t in TRUTHS]
        L.append(f"| {n} | " + " / ".join(f_(m["logmid_off"]) for m in g) + " | " + " / ".join(f_(m["thr3_off"]) for m in g) + " | "
                 + " / ".join(f"{m['rho_below_img']:.2f}" for m in g) + " |")
    with open(os.path.join(res, "2_dp_synthetic.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(res, "T_S10_dp_synthetic.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(T10) + "\n")
    hdr = T10[0].split(",")
    with open(os.path.join(res, "T_S10_dp_synthetic.md"), "w", encoding="utf-8") as f:
        f.write(f"# T_S10 — synthetic truths from the DP logs (Markdown copy of `T_S10_dp_synthetic.csv`)\n\n{EXPLORATORY}\n\n| "
                + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(l.split(",")) + " |" for l in T10[1:]) + "\n")
    json.dump(dict(metrics=M, questions=Q, field=Fv), open(os.path.join(dpp, "dp_synth_summary.json"), "w", encoding="utf-8"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    # F_14
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    xs = np.array([d["x"] for d in D]); ax.plot(xs, [d["base"] for d in D], "k_", ms=16, mew=2, label="true base (DP log-midpoint)")
    cols = {"T1": "#1f4e9c", "T2": "#1b9e77", "T3": "#d35400"}
    for k, t in enumerate(TRUTHS):
        dx = (k - 1) * 5
        ax.plot(xs + dx, [M[f"{t}|L24|{n}"]["thr3"] for n in names], "o", color=cols[t], ms=4, label=f"{t}: 3 Ω·m (λ = 50)")
        ax.plot(xs + dx, [M[f"{t}|L24|{n}"]["logmid_pooled"] for n in names], "^", color=cols[t], ms=4, mfc="none", label=f"{t}: log-midpoint (pooled)")
    fb = np.array([d["base"] for d in D])
    ax.plot(xs + 9, fb + np.array([Fv[n]["L24"]["thr3_off"] for n in names]), "s", color="0.3", ms=4, label="field, 2025 ERT line: 3 Ω·m")
    ax.plot(xs + 9, fb + np.array([Fv[n]["L24"]["logmid_off"] for n in names]), "s", color="0.3", ms=4, mfc="none", label="field, 2025 ERT line: log-midpoint")
    for n, x in zip(names, xs):
        ax.text(x, -24, n, ha="center", fontsize=6)
    ax.set_xlabel("x (m)"); ax.set_ylabel("Elevation (mTAW)"); ax.set_ylim(-25, 0); ax.legend(fontsize=5.8, ncol=2, loc="lower right")
    ax.set_title("Synthetic (L24) vs field: lens base by definition — " + EXPLORATORY, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "F_14_synthetic_vs_field.png"), dpi=300)
    wc.save_publication(fig, args.out, "F_14_synthetic_vs_field", clear_axes_titles=(ax,)); plt.close(fig)
    # F_S12
    xg = np.arange(0, 506, 1.0); zg = np.arange(6.0, -40.01, -0.25); X, Zg = np.meshgrid(xg, zg)
    topo = g27.cpt_topography(); surf = topo(xg)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
    for ax, t in zip(axes, TRUTHS):
        V = TruthModel(t).log10rho(X, Zg); V[Zg > surf[None, :]] = np.nan
        pc = ax.pcolormesh(xg, zg, 10 ** V, cmap="RdBu_r", norm=s18.rho_norm(), shading="auto", rasterized=True)
        ax.plot(xs, [d["base"] for d in D], "kD", ms=3); ax.plot(xg, surf, "k-", lw=0.7)
        ax.set_ylabel("mTAW"); ax.set_title({"T1": "T1 — 2024 logs (0.5 m blocks)", "T2": "T2 — three units per DP",
                                             "T3": f"T3 — T2 with ρ_below = {RHO_BRACKISH:g} Ω·m"}[t], fontsize=8)
        s18.dune_hatch(ax)
    axes[-1].set_xlabel("x (m)"); fig.colorbar(pc, ax=axes, fraction=0.025).set_label("ρ (Ω·m)")
    fig.suptitle(EXPLORATORY, fontsize=7)
    fig.savefig(os.path.join(figs, "F_S12_dp_truths.png"), dpi=300); plt.close(fig)
    print("\n".join(L))
    if all(os.path.exists(os.path.join(SRC(), f"{t}_{l}_r{r}.npz")) for t in ABL for l in LAYOUTS for r in range(N_REAL_ABL)):
        report_ablation(args, plt)


def report_ablation(args, plt):
    """6_ablation.md and F_18 (decision log v1.5-1 (3))."""
    _, res = dirs(args.out); figs = os.path.join(res, "figs"); dpp = os.path.join(args.out, "data", "processed", "v1_5")
    D = dp_inputs(); names = [d["name"] for d in D]; X = ("DP3", "DP4", "DP5", "DP6", "DP7")
    def offs(t, lay, nreal):
        Z = [np.load(os.path.join(SRC(), f"{t}_{lay}_r{r}.npz"), allow_pickle=True) for r in range(nreal)]
        pooled_n = nreal if t in ABL else N_REAL_LAM
        tb = TruthModel(t).par["base"]                     # the variant's own true base (T2a: flat)
        o = {}
        for j, d in enumerate(D):
            o[d["name"]] = dict(logmid=float(np.median([zz["pooled_band_logmid"][1, j] for zz in Z[:pooled_n]])) - float(tb[j]),
                                thr3=float(np.median([zz["lam50_mhat_thr3"][j] for zz in Z])) - float(tb[j]), base=float(tb[j]))
        return o
    E = g27.read_ert(); line_start = g27.X0_CPT - float(E["xloc"].max())
    dp7 = [d for d in D if d["name"] == "DP7"][0]
    R = {}
    for lay in LAYOUTS:
        R[f"T2|{lay}"] = offs("T2", lay, N_REAL)
        for t in ABL:
            R[f"{t}|{lay}"] = offs(t, lay, N_REAL_ABL)
    ok = {k: [n for n in X if abs(v[n]["logmid"]) <= 2.0] for k, v in R.items()}
    f_ = lambda v: "—" if not np.isfinite(v) else f"{v:+.2f}"
    desc = dict(T2="T2 (reference, v1.5)", T2a="T2a flat base at the DP3–DP7 mean", T2b="T2b no clay (ρ_below to −35 mTAW)",
                T2c="T2c ρ_lens = 1.5 Ω·m everywhere (DP1 top unit → lens)")
    L = ["# 6 — Ablation of the log-midpoint's shallow bias (v1.5-1 (3))", "", f"**{EXPLORATORY}** Logged before running "
         "(`docs/decision_log.md` v1.5-1).", "",
         "λ-pooled log-midpoint median − true base (positive = too shallow; median over realisations) and λ = 50 3 Ω·m − true base. "
         "T2a–T2c: 3 realisations, each pooled over λ = 5–100; T2: the v1.5 runs (5 realisations, pooling over realisations 0–1). "
         "Criterion: log-midpoint within ±2 m at ≥ 3 of DP3–DP7.", "",
         f"DP7 (x = {dp7['x']:.1f} m) lies **{dp7['x'] - line_start:.2f} m** from the landward end of the L24 line (x = {line_start:.2f} m); "
         "on L18 it is mid-line.", ""]
    for lay in LAYOUTS:
        L += [f"## {lay}", "", "| variant | " + " | ".join(f"{n}: log-mid / 3 Ω·m" for n in X) + " | within ±2 m | criterion met |",
              "|---|" + "---|" * len(X) + "---|---|"]
        for t in ["T2"] + ABL:
            k = f"{t}|{lay}"
            L.append(f"| {desc[t]} | " + " | ".join(f"{f_(R[k][n]['logmid'])} / {f_(R[k][n]['thr3'])}" for n in X)
                     + f" | {len(ok[k])} of 5 ({', '.join(ok[k]) or 'none'}) | {'**yes**' if len(ok[k]) >= 3 else 'no'} |")
        L.append("")
    fixes = sorted({k.split("|")[0] for k in ok if len(ok[k]) >= 3 and not k.startswith("T2|")})
    L += ["## Which change brings the log-midpoint within ±2 m?", "",
          ("Variants meeting the criterion on at least one layout: " + ", ".join(desc[t] for t in fixes) + "." if fixes
           else "None of the three single changes brings the log-midpoint within ±2 m at ≥ 3 of DP3–DP7 on either layout.")]
    with open(os.path.join(res, "6_ablation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    json.dump(dict(offsets=R, ok=ok, fixes=fixes, dp7_from_line_end=dp7["x"] - line_start),
              open(os.path.join(dpp, "ablation.json"), "w", encoding="utf-8"), indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    cols = dict(T2="k", T2a="#1f4e9c", T2b="#1b9e77", T2c="#d35400")
    xs = np.arange(len(X))
    for ax, lay in zip(axes, LAYOUTS):
        for k_, t in enumerate(["T2"] + ABL):
            ax.plot(xs + (k_ - 1.5) * 0.12, [R[f"{t}|{lay}"][n]["logmid"] for n in X], "o", color=cols[t], ms=4, label=desc[t])
        ax.axhspan(-2, 2, color="0.9", lw=0); ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xticks(xs); ax.set_xticklabels(X, fontsize=7); ax.set_title(f"{lay}: log-midpoint − true base", fontsize=8)
    axes[0].set_ylabel("m (positive = too shallow)"); axes[1].legend(fontsize=5.8)
    fig.suptitle(EXPLORATORY + " Grey band: ±2 m.", fontsize=7); fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(figs, "F_18_ablation.png"), dpi=300); plt.close(fig)
    print("\n".join(L))



if __name__ == "__main__":
    main()
