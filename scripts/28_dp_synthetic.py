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
    t0 = time.time(); ti, li = TRUTHS.index(truth) + 1, LAYOUTS.index(lay)
    R_true = forward(d, truth, lay)["R"]
    elec, q, (sx, sz) = layout(lay)
    rng = np.random.default_rng([150000, ti, li, real])
    R = R_true + np.sqrt((sb.ERR_REL * R_true) ** 2 + sb.ERR_ABS ** 2) * rng.standard_normal(len(R_true))
    pos = R > 0
    xcol = dp_x()
    with g27.surface_override(sx, sz):
        land = dict({k: v[pos] for k, v in q.items()}, R=R[pos], n=int(pos.sum()), err=np.zeros(int(pos.sum())))
        data = m02.build_datacont(land, elec, err_rel=sb.ERR_REL)[0]
        mgr = m02.run_inversion(data, lam=20.0, para_depth=sb.PARA_DEPTH, para_dx=sb.PARA_DX, max_iter=sb.MAX_ITER)
        ncell = mgr.paraDomain.cellCount(); cs = wc.ColumnSampler(g27.pg_mesh_copy(mgr.paraDomain), xcol)
        pre, _ = sb.one_inversion(m02, land, elec, 20.0, np.zeros(ncell), cs, [150200, ti, li, real])
        r = pre["resid"]; med = np.median(r); mad = np.median(np.abs(r - med)); keep = np.abs(r) <= 3 * 1.4826 * mad
        land2 = dict({k: v[keep] for k, v in land.items() if k in ("C1", "C2", "P1", "P2")}, R=land["R"][keep], n=int(keep.sum()),
                     err=np.zeros(int(keep.sum())))
        save = dict(truth=truth, layout=lay, real=real, xcol=xcol, z=cs.z, n_data=int(pos.sum()), n_kept=int(keep.sum()),
                    stage1_chi2=pre["chi2"])
        lams = LAMS if real < N_REAL_LAM else [50.0]; pooled = []
        for lam in lams:
            info, S = sb.one_inversion(m02, land2, elec, lam, np.zeros(ncell), cs, [150100, ti, li, real, int(lam)])
            V = cs(np.log10(info["model"].astype(float))); hat = c11.interfaces_all(V[:, :, None], cs.z); p = f"lam{int(lam)}_"
            save.update({p + "chi2": info["chi2"], p + "col": V.astype(np.float32)})
            for dd in ("thr3", "maxgrad", "logmid"):
                save[p + f"mhat_{dd}"] = hat[dd][:, 0]
            save[p + "r_logmid"] = c11.r_at(cs, info["radius"], hat["logmid"][:, 0])
            save[p + "mhat_steep"] = np.array([g27.steep_logmid(V[:, j], cs.z) for j in range(len(xcol))])
            bands = c11.sample_defs(cs, S)
            for dd in ("thr3", "logmid"):
                save[p + f"band_{dd}"] = bands[dd]
            if real < N_REAL_LAM:
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
    launch(d, [f"F:{t}:{l}:0" for t in TRUTHS for l in LAYOUTS if not os.path.exists(os.path.join(d, f"fwd_{t}_{l}.npz"))],
           args.workers, root)
    jobs = [f"J:{t}:{l}:{r}" for r in range(N_REAL) for t in TRUTHS for l in LAYOUTS
            if not os.path.exists(os.path.join(d, f"{t}_{l}_r{r}.npz"))]
    jobs.sort(key=lambda j: int(j.split(":")[3]) >= N_REAL_LAM)
    print(f"[run] {len(jobs)} jobs on {args.workers} workers", flush=True)
    launch(d, jobs, args.workers, root)
    n = sum(os.path.exists(os.path.join(d, f"{t}_{l}_r{r}.npz")) for t in TRUTHS for l in LAYOUTS for r in range(N_REAL))
    print(f"[run] {n}/{len(TRUTHS) * len(LAYOUTS) * N_REAL} jobs done ({time.time() - t0:.0f} s)", flush=True)


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
        s.add_argument("--time-only", action="store_true")
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


def cmd_report(args):
    raise SystemExit("report: written after the runs")


if __name__ == "__main__":
    main()
