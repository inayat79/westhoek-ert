"""
33_review_v20.py
================
Westhoek ERT v2.0 — analyses requested by the internal review of WRR v4 (POST HOC; rules in docs/decision_log.md v2.0-0).

  a1 [--time-only] [--workers 6]   A1: BEL with prior v3 on the 2025 line (L24) — forward ensemble, then the BELs
                                   → data/processed/v2_0/bel_v3_l24/
  a1-analyse                       A1: fit the BELs from the finished ensemble
  worker                           (internal)

Seeds and ids: see the decision log v2.0-0.
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
import westhoek_common as wc  # noqa: E402

POST_HOC = "Post hoc; requested by the internal review of WRR v4 (decision log v2.0-0)."


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def D20(*p):
    return os.path.join(wc.ROOT, "data", "processed", "v2_0", *p)


def R20(*p):
    return os.path.join(wc.ROOT, "results", "v2_0", *p)


# ─────────────────────────────────────────────────────────────────────────────
# A1 — prior v3 on the L24 geometry
# ─────────────────────────────────────────────────────────────────────────────
A1_DUAL0, A1_NDUAL = 31_500_000, 100
A1_SEED_BASE, A1_SEED_2 = 30_129_000, 30_132_000
V2_IDS = list(range(0, 3300))                      # prior v2: 3000 training + 300 held-out
V3_IDS = list(range(20000, 21573))                 # prior v3 extension: 1430 training + 143 held-out


def a1_dir():
    return D20("bel_v3_l24")


_MODS = {}


def mods():
    if not _MODS:
        g30 = _load("g30", "30_bel_v4.py")
        s16 = _load("s16", "16_stage6_salinity.py")
        _MODS.update(g30=g30, s16=s16, b13=g30.b13)
    return _MODS


def draw_v3(i):
    """Exactly the prior-v3 parameter draw of id i (v2 ids: prior v2 spec; v3 ids and the duals: draw3)."""
    M = mods(); b13, s16 = M["b13"], M["s16"]
    if i < 20000:
        return b13.draw(json.load(open(b13.SPEC2, encoding="utf-8")), i)
    return s16.draw3(i)


def is_train(i):
    return (i < 3000) or (20000 <= i < 21430)


def cmd_worker(args):
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    M = mods(); g30, b13 = M["g30"], M["b13"]
    g28, elec, qk, _, (xs, zs) = g30.l24_geometry()
    scheme = g28.scheme_of(elec, qk)
    meshes = {k: mt.createParaMesh(scheme.sensors(), **b13.MESH[k]) for k in (("medium", "fine") if args.dual else ("medium",))}
    cc = {k: (np.array([c.center().x() for c in m.cells()]), np.array([c.center().y() for c in m.cells()])) for k, m in meshes.items()}
    logf = open(os.path.join(a1_dir(), "logs", f"worker_{args.worker_id}.log"), "a", encoding="utf-8")
    for i in args.ids:
        out = os.path.join(a1_dir(), "dual" if args.dual else "train", f"s{i:08d}.npz")
        if os.path.exists(out):
            continue
        try:
            th = draw_v3(i); t0 = time.time(); save = {}
            with g28.g27.surface_override(xs, zs):
                pm = b13.PriorModel2(th)
            for k, (cx, cz) in cc.items():
                for _ in range(3):
                    with contextlib.redirect_stdout(io.StringIO()):
                        o = ert.simulate(meshes[k], scheme=g28.scheme_of(elec, qk), res=10 ** pm.log10rho(cx, cz), noiseLevel=0, verbose=False)
                    R = np.array(o["rhoa"])
                    if len(R) == len(qk["C1"]):
                        break
                else:
                    raise RuntimeError("wrong number of data")
                save[f"R_{k}"] = R
            np.savez(out + ".tmp.npz", theta=json.dumps(th), target=pm.target(g30.TX_L24), t=time.time() - t0, **save)
            os.replace(out + ".tmp.npz", out)
            logf.write(f"{time.strftime('%H:%M:%S')} s{i} {time.time() - t0:.1f}s\n"); logf.flush()
        except Exception as e:
            import traceback
            logf.write(f"FAILED s{i}: {e}\n{traceback.format_exc()}\n"); logf.flush()
    logf.write("worker finished\n")


def launch(ids, workers, dual=False):
    parts = [ids[k::workers] for k in range(workers)]; procs = []
    for w, part in enumerate(parts):
        if not part:
            continue
        wd = os.path.join(a1_dir(), "work", f"w{w}"); os.makedirs(wd, exist_ok=True)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", "--worker-id", str(w)]
                                      + (["--dual"] if dual else []) + ["--ids"] + [str(i) for i in part], cwd=wd,
                                      env=dict(os.environ, OMP_NUM_THREADS="2", MPLBACKEND="Agg"), stdout=subprocess.DEVNULL,
                                      stderr=open(os.path.join(a1_dir(), "logs", f"worker_{w}.err"), "a")))
    for p in procs:
        p.wait()


def cmd_a1(args):
    for d in ("train", "dual", "logs", "work"):
        os.makedirs(os.path.join(a1_dir(), d), exist_ok=True)
    ids = V2_IDS + V3_IDS
    if args.time_only:
        tid = ids[:10] + V3_IDS[:10]; launch(tid, 1)
        ts = [float(np.load(os.path.join(a1_dir(), "train", f"s{i:08d}.npz"))["t"]) for i in tid]
        dual = [A1_DUAL0, A1_DUAL0 + 1]; launch(dual, 1, dual=True)
        td = float(np.mean([np.load(os.path.join(a1_dir(), "dual", f"s{i:08d}.npz"))["t"] for i in dual]))
        per = float(np.mean(ts)); n = len(ids)
        wall = (n * per + A1_NDUAL * td) / args.workers * 1.3
        plan = dict(n_timed=len(ts), mean_s=per, dual_mean_s=td, n_total=n, n_dual=A1_NDUAL, workers=args.workers, factor=1.3,
                    wall_h=wall / 3600, limit_h=6.0, go=bool(wall / 3600 <= 6.0))
        json.dump(plan, open(os.path.join(a1_dir(), "plan.json"), "w", encoding="utf-8"), indent=1)
        print(json.dumps(plan, indent=1)); return
    t0 = time.time()
    todo = [i for i in ids if not os.path.exists(os.path.join(a1_dir(), "train", f"s{i:08d}.npz"))]
    print(f"[a1] {len(todo)} samples", flush=True); launch(todo, args.workers)
    dual = [i for i in range(A1_DUAL0, A1_DUAL0 + A1_NDUAL) if not os.path.exists(os.path.join(a1_dir(), "dual", f"s{i:08d}.npz"))]
    launch(dual, args.workers, dual=True)
    print(f"[a1] forward done ({time.time() - t0:.0f} s)", flush=True)


def cmd_a1_analyse(args):
    """BEL with prior v3 unchanged on the L24 geometry: own approximation-error set, held-out set, falsification,
    field posterior and the DP2–DP7 scores against its own prior (as 30_bel_v4.l24_analyse; v3 bounds; A1 seeds)."""
    M = mods(); g30, b13 = M["g30"], M["b13"]
    import stage6_analysis as s6
    g31 = _load("g31", "31_scores.py"); s15, g14 = g30.s15, g30.g14
    TXL = g30.TX_L24; s6.init(); s6.TX = TXL
    g28, elec, qk, keep, _ = g30.l24_geometry()
    dd = os.path.join(a1_dir(), "dual"); dual = [np.load(os.path.join(dd, f)) for f in sorted(os.listdir(dd)) if f.endswith(".npz") and "tmp" not in f]
    dl = np.array([np.log10(z["R_fine"]) - np.log10(z["R_medium"]) for z in dual]); mu_d = dl.mean(0); dev = dl - mu_d
    d = os.path.join(a1_dir(), "train"); sets = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz") and "tmp" not in f:
            z = np.load(os.path.join(d, f), allow_pickle=True)
            sets.append((int(f[1:-4]), {"R_medium": z["R_medium"], "theta": z["theta"], "target": z["target"]}))
    exp_ids = set(V2_IDS + V3_IDS); have = {i for i, _ in sets}
    C = {}
    for name, ss in (("tr", [x for x in sets if is_train(x[0])]), ("ho", [x for x in sets if not is_train(x[0])])):
        D, kp = b13.prior_data(ss, mu_d, dev); zm = {i: z for i, z in ss}
        th = [json.loads(str(zm[i]["theta"])) for i in kp]; T = np.array([zm[i]["target"] for i in kp])
        m = np.array([s6.site(t, tt) and t["rho_lens"] >= s6.PHYS_RHO_LENS_MIN for t, tt in zip(th, T)], dtype=bool)
        C[name] = dict(D=D[m], T=T[m], th=[t for t, k in zip(th, m) if k], n_all=len(ss), n_pos=len(kp), n_kept=int(m.sum()))
    E = g28.g27.read_ert(); y = np.log10(E["R"][keep])
    D, T, th = C["tr"]["D"], C["tr"]["T"], C["tr"]["th"]
    bel = b13.BEL(D, T); Pf = bel.posterior(y, 2000, np.random.default_rng([A1_SEED_BASE, 1]), "kde")
    cov, wid = [], []
    for j, (yy, t) in enumerate(zip(C["ho"]["D"], C["ho"]["T"])):
        P = bel.posterior(yy, 2000, np.random.default_rng([A1_SEED_BASE, 90_000 + j]), "kde"); lo, hi = np.percentile(P, [5, 95], 0)
        cov.append((t >= lo) & (t <= hi)); wid.append(hi - lo)
    sc = bel.Sd; var = sc.var(0, ddof=1); sf = bel.pd.fwd(y[None])[0]
    mu, Cm = b13.pilot.fast_mcd(sc); Ci = np.linalg.inv(Cm)
    fal = dict(t2=float(np.mean(np.sum(sc ** 2 / var, 1) < np.sum(sf ** 2 / var))),
               t2r=float(np.mean(np.einsum("ij,jk,ik->i", sc - mu, Ci, sc - mu) < (sf - mu) @ Ci @ (sf - mu))),
               q=float(np.mean(bel.pd.resid(D) < bel.pd.resid(y[None])[0])))
    Ytr = np.array([g14.derived_targets(t) for t in th]); Yho = np.array([g14.derived_targets(t) for t in C["ho"]["th"]])
    b2 = s15.BEL2(D, Ytr, bounded=True); P2 = b2.posterior(y[None], 2000, np.random.default_rng([A1_SEED_2, 3]))
    c2 = []
    for j, yy in enumerate(C["ho"]["D"]):
        P = b2.posterior(yy, 2000, np.random.default_rng([A1_SEED_2, 60_000 + j])); lo, hi = np.percentile(P, [5, 95], 0)
        c2.append((Yho[j] >= lo) & (Yho[j] <= hi))
    # field scores at DP2–DP7 against its own prior (training targets); prior CRPS subsample seeds [160000, prior set, 500 + j]
    DPs = [x for x in g31.dp_list() if x["dp"] in g31.DPS]; rows = []
    for i, dp in enumerate(DPs):
        j = int(np.argmin(np.abs(TXL - dp["x"]))); yb = dp["logmid"]
        sb_ = g31.from_samples(Pf[:, j], yb, [160000, g31.SETS["field"], 100 * i + j])
        l, mq, u = np.percentile(T[:, j], [5, 50, 95])
        sp = dict(l=l, m=mq, u=u, is_=float(g31.interval_score(l, u, yb)), crps=float(g31.crps_sorted(g31.cap(T[:, j], [160000, g31.SETS["prior"], 500 + j]), yb)[0]))
        rows.append(dict(dp=dp["dp"], col=float(TXL[j]), y=yb, bel=sb_, prior=sp, inside=bool(sb_["l"] <= yb <= sb_["u"]), prior_inside=bool(l <= yb <= u)))
    k = sum(r["inside"] for r in rows)
    IS = float(np.mean([r["bel"]["is_"] for r in rows])); ISp = float(np.mean([r["prior"]["is_"] for r in rows]))
    CR = float(np.mean([r["bel"]["crps"] for r in rows])); CRp = float(np.mean([r["prior"]["crps"] for r in rows]))
    summ = dict(k=int(k), n=len(rows), cp90=clopper_pearson(int(k), len(rows)), median_width=float(np.median([r["bel"]["u"] - r["bel"]["l"] for r in rows])),
                IS=IS, CRPS=CR, prior_IS=ISp, prior_CRPS=CRp, skill_IS=1 - IS / ISp, skill_CRPS=1 - CR / CRp,
                prior_k=int(sum(r["prior_inside"] for r in rows)))
    out = dict(note=POST_HOC, n_expected=len(exp_ids), n_have=len(have & exp_ids), missing=sorted(exp_ids - have)[:50], n_dual=len(dual),
               n={k_: {kk: v for kk, v in C[k_].items() if kk.startswith("n_")} for k_ in ("tr", "ho")}, kd=int(bel.pd.k), kt=int(bel.pt.k),
               heldout=dict(cov=float(np.mean(cov)) if cov else np.nan, cov_k=int(np.sum(cov)), cov_n=int(np.size(cov)),
                            width_median=float(np.median(wid)) if wid else np.nan,
                            cov2_rho_below=float(np.mean(np.array(c2)[:, 3])) if c2 else np.nan),
               falsification=fal, approx=dict(n=len(dl), mean_abs_median=float(np.median(np.abs(mu_d))), sd_median=float(np.median(dev.std(0, ddof=1)))),
               rho_below=[float(np.median(10 ** P2[:, 3])), float(np.percentile(10 ** P2[:, 3], 5)), float(np.percentile(10 ** P2[:, 3], 95))],
               rows=rows, summary=summ, expectation=dict(holds=bool(k >= 5 and summ["skill_IS"] > 0)))
    sfx = "_smoke" if os.environ.get("A1_SMOKE") else ""
    np.savez(D20(f"a1_bel_v3_l24{sfx}.npz"), Pf=Pf, P2=P2, TX=TXL, ho_cov=np.array(cov), Ttr=T)
    json.dump(out, open(D20(f"a1{sfx}.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("[a1-analyse]", json.dumps(summ), out["heldout"], fal, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# A3 (L18) — regenerate the field linearised samples exactly as 11_stage4_closeout.cmd_a4
# ─────────────────────────────────────────────────────────────────────────────
A4_TAGS = ["g1_lam50", "g1_lam5", "g1_lam10", "deep_drop_data", "g1_lam100"]      # order = seed offset (20260928 + i)


def depth_samples(c11, cs, S, chunk=100):
    """Per-sample interface depths (columns × samples) for c11.DEFS — the arrays inside c11.sample_defs."""
    out = {d: np.full((len(cs.x), S.shape[1]), np.nan) for d in c11.DEFS}
    for i in range(0, S.shape[1], chunk):
        r = c11.interfaces_all(cs(S[:, i:i + chunk] / np.log(10)), cs.z)
        for d in c11.DEFS:
            out[d][:, i:i + chunk] = r[d]
    return out


def lin_samples(tags, seeds, cs, c11):
    import scipy.linalg as sla
    per, pooled = {}, []
    for t, seed in zip(tags, seeds):
        run = wc.load_run(t)
        L_, _, _, lam = wc.cholesky_A(run)
        rng = np.random.default_rng(seed)
        S = np.log(run["model"])[:, None] + sla.solve_triangular(L_, rng.standard_normal((len(run["model"]), 1000)), lower=True, trans="T")
        per[t] = depth_samples(c11, cs, S); pooled.append(S.astype(np.float32))
        print(f"[lin] {t} λ={lam:g} seed {seed}", flush=True)
    P = depth_samples(c11, cs, np.hstack(pooled).astype(np.float64))
    return per, P


def cmd_a3_lin18(args):
    c11 = _load("c11", "11_stage4_closeout.py")
    ref = wc.load_run("g1_lam50"); mesh, _ = wc.rebuild_paradomain(ref)
    ex, ez = wc.land_surface()
    cs = wc.ColumnSampler(mesh, np.arange(np.ceil(ex.min()), np.floor(ex.max()) + 1, 2.0))
    per, P = lin_samples(A4_TAGS, [20260928 + i for i in range(len(A4_TAGS))], cs, c11)
    st = np.load(os.path.join(wc.RESULTS_DIR, "runs", "g1_lam50_v4", "interface_definitions_bands.npz"))
    chk = {}
    for d in c11.DEFS:
        for kind, Z in (("single", per["g1_lam50"][d]), ("pooled", P[d])):
            b = c11.band_stats(Z); s = st[f"{kind}_{d}"]
            diff = float(np.nanmax(np.abs(b - s))) if np.isfinite(b).any() else 0.0
            chk[f"{kind}_{d}"] = dict(max_abs_diff=diff, nan_pattern_equal=bool(np.array_equal(np.isnan(b), np.isnan(s))))
    # decision log v2.0-1: acceptance on the 3 Ω·m and maximum-gradient bands; the log-midpoint differences are recorded
    ok = all(v["max_abs_diff"] <= 1e-6 and v["nan_pattern_equal"] for k, v in chk.items() if "logmid" not in k)
    os.makedirs(D20(), exist_ok=True)
    json.dump(dict(check=chk, accepted=ok, tolerance_m=1e-6, note="acceptance on thr3/maxgrad only; logmid differences recorded (v2.0-1)"),
              open(D20("a3_lin18_check.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(chk, indent=1), "accepted", ok)
    if not ok:
        raise SystemExit("regenerated L18 bands differ from the stored bands — stopping (decision log v2.0-0)")
    np.savez_compressed(D20("a3_lin18_samples.npz"), x=cs.x, **{f"single_{d}": per["g1_lam50"][d].astype(np.float32) for d in c11.DEFS},
                        **{f"pooled_{d}": P[d].astype(np.float32) for d in c11.DEFS})


# ─────────────────────────────────────────────────────────────────────────────
# A9 — mesh, λ and stopping (stored runs; nothing rerun)
# ─────────────────────────────────────────────────────────────────────────────
def _mx(a):
    a = np.abs(np.asarray(a, float)); return float(np.nanmax(a)) if np.isfinite(a).any() else 0.0


def cmd_a3_lin24(args):
    """Rerun of the 2025-line (L24, CPT topography) λ pool exactly as 27_dp_groundtruth.cmd_samedate (case "cpt"),
    keeping what the stored product lacks: per-sample interface depths at the columns (A3), final-iteration
    objective terms (A9), and J / data errors at λ = 50 (A5 cumulative sensitivity). Seeds as stored: [139000, λ]."""
    out = D20("a3_lin24.npz")
    if os.path.exists(out):
        print("[a3-lin24] exists"); return
    g27 = _load("g27", "27_dp_groundtruth.py"); c11 = g27._load("c11", "11_stage4_closeout.py"); sb = c11.sb
    m02 = c11._load("inv02", "02_land_inversion.py"); m02.VERBOSE = False
    SD = np.load(os.path.join(wc.ROOT, "data", "processed", "v1_4", g27.SD_DIR_NAME, "cpt.npz"), allow_pickle=True)
    E = g27.read_ert(); topo = g27.cpt_topography(); xcol = g27.sd_columns()
    assert np.allclose(xcol, SD["xcol"])
    n_el = len(E["xloc"]); num = n_el - np.arange(n_el)
    xs = g27.X0_CPT - E["xloc"]; zs = topo(xs)
    elec = {int(num[i]): (float(xs[i]), float(zs[i])) for i in range(n_el)}
    a = E["abmn"]; keep = SD["keep"].astype(bool)
    base = {k: np.array([int(num[v]) for v in a[:, c]]) for c, k in enumerate(("C1", "C2", "P1", "P2"))}
    land2 = dict({k: v[keep] for k, v in base.items()}, R=E["R"][keep], n=int(keep.sum()), err=np.zeros(int(keep.sum())))
    order = np.argsort(xs)
    cap = {}
    orig_res = m02.compute_resolution_matrix
    def res_capture(mgr, data, lam_expected=None):
        r = orig_res(mgr, data, lam_expected=lam_expected); cap["res"] = r; cap["pd"] = g27.pg_mesh_copy(mgr.paraDomain); return r
    m02.compute_resolution_matrix = res_capture
    save = dict(note=POST_HOC, xcol=xcol, n_kept=int(keep.sum()))
    pooled = []; acc = {}
    with g27.surface_override(xs[order], zs[order]):
        ncell = len(SD["lam50_model"])
        # mesh: as built by the manager from these data (independent of λ and iterations); one-iteration run
        with contextlib.redirect_stdout(io.StringIO()):
            data0 = m02.build_datacont(land2, elec, err_rel=sb.ERR_REL)[0]
            mgr0 = m02.run_inversion(data0, lam=50.0, para_depth=sb.PARA_DEPTH, para_dx=sb.PARA_DX, max_iter=1)
        pd0 = g27.pg_mesh_copy(mgr0.paraDomain); del mgr0
        assert pd0.cellCount() == ncell, "mesh differs from the stored L24 mesh"
        cs = wc.ColumnSampler(pd0, xcol)
        for lam in g27.SD_LAMS:
            with contextlib.redirect_stdout(io.StringIO()):
                info, S = sb.one_inversion(m02, land2, elec, lam, np.zeros(ncell), cs, [139000, int(lam)])
            res = cap["res"]; m = info["model"].astype(float)
            Cw = res["C"]; phim = float(np.linalg.norm(Cw @ np.log(m)) ** 2)
            phid = info["n_data"] * info["chi2"]
            p = f"lam{int(lam)}_"
            V = cs(np.log10(m))
            acc[p + "col_maxdiff_log10"] = _mx(V - SD[p + "col"])
            D = depth_samples(c11, cs, S)
            for d in c11.DEFS:
                save[p + "dep_" + d] = D[d].astype(np.float32)
                b = c11.band_stats(D[d]); acc[p + "band_" + d + "_maxdiff_m"] = _mx(b[1:] - SD[p + "band_" + d][1:])
                acc[p + "band_" + d + "_share_maxdiff"] = _mx(b[0] - SD[p + "band_" + d][0])
            save.update({p + "chi2": info["chi2"], p + "chi2_pg": info["chi2_pg"], p + "n_iter": info["n_iter"], p + "stop": info["stop"],
                         p + "n_data": info["n_data"], p + "phi_d": phid, p + "phi_m": phim, p + "phi": phid + lam * phim, p + "model": m})
            if int(lam) == 50:
                J = np.asarray(res["J"]); err = np.asarray(res["err"])
                save["lam50_J"] = J.astype(np.float32); save["lam50_err"] = err
                save["lam50_cumsens"] = np.asarray(res["cum_sens"], float)                     # Σ(J/ε)², normalised (02_land_inversion)
                save["lam50_cumsens_col"] = cs(np.log10(np.clip(save["lam50_cumsens"], 1e-30, None))).astype(np.float32)
            save[p + "col"] = V.astype(np.float32)
            pooled.append(S.astype(np.float32))
            print(f"[a3-lin24] λ={lam:g}: n_iter {info['n_iter']} stop {info['stop']} χ² {info['chi2']:.3f} Φd {phid:.0f} Φm {phim:.1f}", flush=True)
        P = depth_samples(c11, cs, np.hstack(pooled).astype(np.float64))
        for d in c11.DEFS:
            save["pooled_dep_" + d] = P[d].astype(np.float32)
            b = c11.band_stats(P[d]); acc["pooled_band_" + d + "_maxdiff_m"] = _mx(b[1:] - SD["pooled_band_" + d][1:])
        # mesh: vertical size of cells with centres at −20 … −10 mTAW, x 100–450 m
        dz = []
        for c in cap["pd"].cells():
            ys = [n.pos().y() for n in c.nodes()]; xx = [n.pos().x() for n in c.nodes()]
            cz, cx = np.mean(ys), np.mean(xx)
            if -20 <= cz <= -10 and 100 <= cx <= 450:
                dz.append(max(ys) - min(ys))
        save["mesh_dz_window"] = np.array(dz); save["mesh_ncell"] = cap["pd"].cellCount()
        save["z"] = cs.z; save["surf"] = cs.surf
    m02.compute_resolution_matrix = orig_res
    ok = all(v <= 0.10 for k, v in acc.items() if k.endswith("_m")) and all(v <= 0.01 for k, v in acc.items() if "col_maxdiff" in k)
    json.dump(dict(check=acc, accepted=ok, rule="bands within 0.10 m, model columns within 0.01 log10 (v2.0-0)"),
              open(D20("a3_lin24_check.json"), "w", encoding="utf-8"), indent=1)
    print("[a3-lin24] accepted", ok, flush=True)
    np.savez(out + ".tmp.npz", **save); os.replace(out + ".tmp.npz", out)


# ─────────────────────────────────────────────────────────────────────────────
# A2 — BEL posterior support (truncation above the ground surface)
# ─────────────────────────────────────────────────────────────────────────────
def truncate_cols(P, surf):
    """Column-wise: samples whose base lies above the ground surface of that column become NaN."""
    P = np.array(P, float); P[P > np.asarray(surf)[None, :]] = np.nan
    return P


def canon_info(bel, y):
    """Leading-5 canonical correlations, KDE (Scott) bandwidths and the field's position in canonical space."""
    from scipy.stats import gaussian_kde
    cd = bel.cca.cx(bel.pd.fwd(np.atleast_2d(y)))[0]
    q = min(5, bel.Ct.shape[1]); out = dict(r=[float(v) for v in bel.cca.r[:q]], pairs=[])
    for i in range(q):
        k = gaussian_kde(np.vstack([bel.Cd[:, i], bel.Ct[:, i]]))
        pct = float(np.mean(bel.Cd[:, i] < cd[i]))
        out["pairs"].append(dict(pair=i + 1, r=float(bel.cca.r[i]), kde_factor=float(k.factor), bw_cd=float(np.sqrt(k.covariance[0, 0])),
                                 bw_ct=float(np.sqrt(k.covariance[1, 1])), field_cd=float(cd[i]), field_pct=pct,
                                 field_two_sided=float(2 * min(pct, 1 - pct))))
    Z = bel.Cd[:, :q]; mu = Z.mean(0); Ci = np.linalg.inv(np.cov(Z.T))
    m2 = np.einsum("ij,jk,ik->i", Z - mu, Ci, Z - mu); mf = float((cd[:q] - mu) @ Ci @ (cd[:q] - mu))
    out.update(n_train=int(len(bel.Cd)), kd=int(bel.pd.k), kt=int(bel.pt.k), mahal2_field=mf, mahal2_pct=float(np.mean(m2 < mf)))
    return out


def cmd_a2(args):
    import csv
    M = mods(); g30, s16, b13 = M["g30"], M["s16"], M["b13"]
    import stage6_analysis as s6
    g31 = _load("g31", "31_scores.py"); g27 = _load("g27", "27_dp_groundtruth.py")
    TX, TXL = b13.TX, g30.TX_L24
    ex, ez = wc.land_surface(); surf18 = np.interp(TX, ex, ez); surf24 = g27.cpt_topography()(TXL)
    P = D20()
    B3 = np.load(os.path.join(wc.ROOT, "data", "processed", "paper", "bel_products.npz"), allow_pickle=True)
    Z4 = np.load(os.path.join(g30.D4(), "bel_v4.npz"), allow_pickle=True); ZL = np.load(os.path.join(g30.D4L(), "bel_v4_l24.npz"), allow_pickle=True)
    Zs = np.load(os.path.join(g30.DLT(), "layout_vs_time.npz"), allow_pickle=True)
    K1 = np.load(os.path.join(wc.ROOT, "data", "processed", "v1_1", "k1_field_bel.npz"))
    info = dict(note=POST_HOC, k1_equals_bel_products=bool(np.array_equal(K1["Pf"], B3["Pf"])), canon={}, regen={})
    obs, _ = b13.observed_sets()
    T = {}
    # BEL v3 (L18)
    s6.init(); s6.TX = TX
    C = s6.load_combined(physical=True); bel = b13.BEL(C["tr"]["D"], C["tr"]["T"])
    info["regen"]["BEL3-18"] = bool(np.array_equal(bel.posterior(obs["field"], 2000, np.random.default_rng([129000, 1]), "kde"), B3["Pf"]))
    info["canon"]["BEL3-18"] = canon_info(bel, obs["field"]); T["PRIOR3"] = np.asarray(C["tr"]["T"], float); del C, bel
    print("[a2] v3 done", flush=True)
    C = g30.load_v4(); bel = b13.BEL(C["tr"]["D"], C["tr"]["T"])
    info["regen"]["BEL4-18"] = bool(np.array_equal(bel.posterior(obs["field"], g30.N_POST, np.random.default_rng([g30.SEED_BASE, 1]), "kde"), Z4["Pf"]))
    info["canon"]["BEL4-18"] = canon_info(bel, obs["field"]); T["PRIOR4"] = np.asarray(C["tr"]["T"], float); del C, bel
    print("[a2] v4 done", flush=True)
    g28, keep, C = g30.l24_sets(); s6.init(); s6.TX = TXL; bel = b13.BEL(C["tr"]["D"], C["tr"]["T"])
    y = np.log10(g28.g27.read_ert()["R"][keep])
    info["regen"]["BEL4-24"] = bool(np.array_equal(bel.posterior(y, g30.N_POST, np.random.default_rng([g30.L24_SEED_BASE, 1]), "kde"), ZL["Pf"]))
    info["canon"]["BEL4-24"] = canon_info(bel, y); T["PRIOR4-L24"] = np.asarray(C["tr"]["T"], float); del C, bel
    print("[a2] v4 L24 done", flush=True)
    _, _, S, _ = g30.stretch_sets(); T["PRIOR4-s"] = np.asarray(S["tr"]["T"], float); del S
    # truncation
    post = {"BEL3-18": (B3["Pf"], TX, surf18, "PRIOR3"), "BEL4-18": (Z4["Pf"], TX, surf18, "PRIOR4"),
            "BEL4-18s": (Zs["sub_Pf"], TX, surf18, "PRIOR4-s"), "BEL4-24": (ZL["Pf"], TXL, surf24, "PRIOR4-L24")}
    psurf = {"PRIOR3": (TX, surf18), "PRIOR4": (TX, surf18), "PRIOR4-s": (TX, surf18), "PRIOR4-L24": (TXL, surf24)}
    DPs = g31.dp_list(); rows = []; tab = {}
    share = {}
    for meth, (Pf, TXm, sf, pr) in post.items():
        Pt = truncate_cols(Pf, sf); share[meth] = [float(v) for v in np.mean(np.isnan(Pt), 0)]
        tab[meth] = {}
        for i, d in enumerate(DPs):
            j = int(np.argmin(np.abs(TXm - d["x"])))
            q0 = np.percentile(Pf[:, j], [50, 5, 95]); xs = Pt[:, j][np.isfinite(Pt[:, j])]; q1 = np.percentile(xs, [50, 5, 95])
            tab[meth][d["dp"]] = dict(col=float(TXm[j]), surface=float(sf[j]), n_removed=int(len(Pf) - len(xs)), orig=[float(v) for v in q0],
                                      trunc=[float(v) for v in q1], y=d["logmid"])
            if d["logmid"] is not None and np.isfinite(d["logmid"]):
                for tag, X in (("orig", Pf[:, j]), ("trunc", xs)):
                    sc = g31.from_samples(X, d["logmid"], [160000, g31.SETS["field"], 100 * i + j])
                    rows.append(dict(version=tag, method=meth, prior=pr, dp=d["dp"], col=float(TXm[j]), y=d["logmid"], l=sc["l"], m=sc["m"], u=sc["u"],
                                     **{"is": sc["is_"], "crps": sc["crps"]}))
    pshare = {}
    for pr, (TXm, sf) in psurf.items():
        Tt = truncate_cols(T[pr], sf); pshare[pr] = int(np.isnan(Tt).sum())
        for tag, X in (("orig", T[pr]), ("trunc", Tt)):
            Pobj = g31.Prior(pr, X, TXm)
            for i, d in enumerate(DPs):
                if d["logmid"] is None or not np.isfinite(d["logmid"]):
                    continue
                j = Pobj.col(d["x"]); sc = Pobj.score(j, d["logmid"])
                rows.append(dict(version=tag, method=pr, prior=pr, dp=d["dp"], col=float(TXm[j]), y=d["logmid"], l=sc["l"], m=sc["m"], u=sc["u"],
                                 **{"is": sc["is_"], "crps": sc["crps"]}))
    info["removed_share_by_column"] = share; info["prior_samples_above_ground"] = pshare; info["dp_table"] = tab
    # check: the 'orig' rows reproduce cases.csv
    Cc = g31.load_cases(); ref = {(r["method"], r["dp"]): r for r in Cc if r["set"] == "field" and r["truth_kind"] == "logmid"}
    dif = [abs(r["is"] - ref[(r["method"], r["dp"])]["is"]) + abs(r["crps"] - ref[(r["method"], r["dp"])]["crps"])
           for r in rows if r["version"] == "orig" and (r["method"], r["dp"]) in ref]
    info["orig_reproduces_cases_csv_maxdiff"] = float(max(dif)); info["orig_n_compared"] = len(dif)
    # field summaries (as 31_scores.cmd_report) for both versions
    idx = np.random.default_rng([160100, 0]).integers(0, len(g31.DPS), (g31.N_BOOT, len(g31.DPS)))
    summ = {}
    for tag in ("orig", "trunc"):
        Cv = [dict(r, set="field", truth_kind="logmid", approx=0) for r in rows if r["version"] == tag]
        F = {m: g31.field_vec(Cv, m) for m in list(post) + list(psurf)}
        S_ = {}
        for m, v in F.items():
            p = F[v["prior"]] if not m.startswith("PRIOR") else None
            S_[m] = dict(prior=v["prior"], n=int(np.isfinite(v["is_"]).sum()), cov=int(np.sum((v["l"] <= v["y"]) & (v["y"] <= v["u"]))),
                         width=float(np.nanmedian(v["u"] - v["l"])), mae=float(np.nanmean(np.abs(v["m"] - v["y"]))),
                         IS=g31.fsum(v["is_"], None if p is None else p["is_"], idx), CRPS=g31.fsum(v["crps"], None if p is None else p["crps"], idx),
                         inside={n: bool(l <= yy <= u) for n, l, yy, u in zip(g31.DPS, v["l"], v["y"], v["u"])})
        summ[tag] = S_
    info["field_summary"] = summ
    json.dump(info, open(P + "/a2.json", "w", encoding="utf-8"), indent=1, default=float)
    with open(P + "/a2_field_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("[a2] regen", info["regen"], "orig reproduces cases.csv:", info["orig_reproduces_cases_csv_maxdiff"], flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# A6 — bias rule out of sample
# ─────────────────────────────────────────────────────────────────────────────
def rule_metrics(y, yh, q):
    y, yh, q = map(np.asarray, (y, yh, q))
    m = np.abs(q - 0.5) > 0.1
    sgn = lambda a, b: float(np.mean(np.sign(a) == np.sign(b))) if len(a) else None
    return dict(n=int(len(y)), rmse=float(np.sqrt(np.mean((y - yh) ** 2))),
                r2_centred=float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)) if len(y) > 1 else None,
                sign_all=sgn(y, yh), n_margin=int(m.sum()), sign_margin=sgn(y[m], yh[m]))


def cmd_a6(args):
    g25 = _load("g25", "25_generalisation.py"); sb = g25.sb if hasattr(g25, "sb") else _load("sb10", "10_synthetic_benchmark.py")
    g28 = _load("g28", "28_dp_synthetic.py")
    from scipy.stats import norm
    d = g25.SRC
    done = [t for t in g25.TRUTHS if all(os.path.exists(os.path.join(d, t, f"r{r}.npz")) for r in range(g25.N_REAL))]
    rows = [r for t in done for r in g25.truth_metrics(d, t)]
    fit = g25.fit_rule(rows, "lam50"); a = fit["a"]; pts = fit["pts"]
    out = dict(note=POST_HOC, a_full=a, r2_full=fit["r2"], n_fit=fit["n"])
    # LOTO
    loto = []
    for t in sorted({p["truth"] for p in pts}):
        tr = [p for p in pts if p["truth"] != t]; x = np.array([p["x"] for p in tr]); yy = np.array([p["y"] for p in tr])
        at = float(x @ yy / (x @ x))
        for p in pts:
            if p["truth"] == t:
                loto.append(dict(truth=t, d=p["d"], q=p["q"], rbar=p["rbar"], y=p["y"], a_loto=at, pred=at * p["x"]))
    out["loto"] = dict(cases=loto, **rule_metrics([c["y"] for c in loto], [c["pred"] for c in loto], [c["q"] for c in loto]))
    out["in_sample"] = rule_metrics([p["y"] for p in pts], [a * p["x"] for p in pts], [p["q"] for p in pts])
    # E9 cases and the near-midpoint sharp cases
    V = g25.verdicts(rows); out["E9"] = dict(n=V["E9"]["n"], n_ok=V["E9"]["n_ok"])
    e9 = []; near = []
    for r in rows:
        if r["definition"] in ("thr2", "thr3", "thr5") and r["treatment"] == "lam50" and r["group"] == "≤10" and \
                r["n"] >= 5 and r["bias"] is not None and 0 < r["q"] < 1:
            rec = dict(truth=r["truth"], contour=r["definition"], lam=50, r_group="≤10", n=r["n"], q=r["q"], bias=r["bias"],
                       sharp=bool(g25.params(r["truth"])["tau"] == 0), rbar=r["rbar"])
            if abs(r["q"] - 0.5) > 0.1:
                rec["pred_sign"] = float(-np.sign(norm.ppf(r["q"]))); rec["ok"] = bool(np.sign(r["bias"]) == rec["pred_sign"]); e9.append(rec)
            elif rec["sharp"]:
                rec["too_deep"] = bool(r["bias"] < 0); near.append(rec)
    out["E9"]["cases"] = e9; out["E9"]["n_listed"] = len(e9); out["near_midpoint_sharp"] = near
    # out of sample (i): v1.0 site-like truths, 3 Ω·m main-lens base, λ = 50, r ≤ 10 m
    spec = json.load(open(os.path.join(wc.PROCESSED_DIR, "synthetic", "truth_spec.json"), encoding="utf-8"))
    ref = wc.load_run(sb.REF_TAG); mesh, _ = wc.rebuild_paradomain(ref); cs = wc.ColumnSampler(mesh, sb.COLS)
    L3 = np.log10(3.0); oos = {}
    v10 = []
    for t in ("T1", "T1h", "T2a", "T3"):
        tr = sb.Truth(spec, t); v = tr.v
        q_true = float((L3 - np.log10(v["lens"])) / (np.log10(v["brackish"]) - np.log10(v["lens"])))
        bias, rr, vl, vb = [], [], [], []
        for real in range(20):
            Z = np.load(os.path.join(wc.PROCESSED_DIR, "synthetic", "runs", f"{t}_r{real:02d}.npz"), allow_pickle=True)
            cols = Z["cols"]; zb = tr.base(cols); mh = Z["lam50_mhat_base"]; rb = Z["lam50_r_base"]
            Vc = cs(np.log10(Z["lam50_model"].astype(float)))
            for j in range(len(cols)):
                if not (np.isfinite(rb[j]) and rb[j] <= 10 and np.isfinite(mh[j])):
                    continue
                bias.append(mh[j] - zb[j]); rr.append(rb[j])
                col = Vc[:, j]; ok = np.isfinite(col)
                aa = ok & (cs.z >= zb[j] + 4); bb = ok & (cs.z <= zb[j] - 4) & (cs.z >= zb[j] - 10)
                if aa.any() and bb.any():
                    vl.append(np.median(col[aa])); vb.append(np.median(col[bb]))
        lens_r, below_r = float(np.median(vl)), float(np.median(vb)); q_rec = float((L3 - lens_r) / (below_r - lens_r))
        rbar = float(np.mean(rr)); yb = float(np.median(bias))
        v10.append(dict(truth=t, n=len(bias), rbar=rbar, y=yb, q_true=q_true, q_rec=q_rec, rho_lens_rec=10 ** lens_r, rho_below_rec=10 ** below_r,
                        pred_true=float(-a * rbar * norm.ppf(q_true)), pred_rec=float(-a * rbar * norm.ppf(q_rec)) if 0 < q_rec < 1 else None))
    oos["v10_site_like"] = v10
    # (ii) v1.5 sounding-based truths T1–T3 on L18 and L24, DP2–DP7 columns (r of the log-midpoint, as stored)
    D = g28.dp_inputs(); v15 = []
    for t in g28.TRUTHS:
        for lay in g28.LAYOUTS:
            Z = [np.load(os.path.join(g28.SRC(), f"{t}_{lay}_r{r}.npz"), allow_pickle=True) for r in range(g28.N_REAL)]
            z = Z[0]["z"]
            for j, dp in enumerate(D):
                if dp["name"] == "DP1":
                    continue
                base = dp["base"]; rb_true = g28.RHO_BRACKISH if t == "T3" else dp["rho_below"]; rl_true = dp["rho_lens"]
                r_ = np.array([float(zz["lam50_r_logmid"][j]) for zz in Z]); rbar = float(np.nanmedian(r_))
                if not rbar <= 10:
                    continue
                y = float(np.nanmedian([float(zz["lam50_mhat_thr3"][j]) for zz in Z]) - base)
                q_true = float((L3 - np.log10(rl_true)) / (np.log10(rb_true) - np.log10(rl_true)))
                vl, vb = [], []
                for zz in Z:
                    col = zz["lam50_col"][:, j].astype(float); ok = np.isfinite(col)
                    aa = ok & (z >= base + 4); bb = ok & (z <= base - 4) & (z >= base - 10)
                    if aa.any() and bb.any():
                        vl.append(np.median(col[aa])); vb.append(np.median(col[bb]))
                lr, br = float(np.median(vl)), float(np.median(vb)); q_rec = float((L3 - lr) / (br - lr)) if br != lr else np.nan
                v15.append(dict(truth=t, layout=lay, dp=dp["name"], rbar=rbar, y=y, q_true=q_true, q_rec=q_rec, rho_lens_rec=10 ** lr, rho_below_rec=10 ** br,
                                pred_true=float(-a * rbar * norm.ppf(q_true)) if 0 < q_true < 1 else None,
                                pred_rec=float(-a * rbar * norm.ppf(q_rec)) if 0 < q_rec < 1 else None))
    oos["v15_sounding"] = v15
    def ev(cases, k):
        c = [x for x in cases if x[f"pred_{k}"] is not None]
        return rule_metrics([x["y"] for x in c], [x[f"pred_{k}"] for x in c], [x[f"q_{k}"] for x in c]) if c else None
    out["oos"] = oos
    out["oos_metrics"] = {f"{s}|{k}": ev(oos[s], k) for s in oos for k in ("true", "rec")}
    json.dump(out, open(D20("a6.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("[a6] a", a, "LOTO", {k: v for k, v in out["loto"].items() if k != "cases"}, "E9 listed", len(e9), "near", len(near), flush=True)
    for k, v in out["oos_metrics"].items():
        print("  ", k, v)


# ─────────────────────────────────────────────────────────────────────────────
# A7 — sounding-base definition
# ─────────────────────────────────────────────────────────────────────────────
def steep_logmid_window(v, z, lo, hi):
    """27_dp_groundtruth.steep_logmid with the steepest-increase search restricted to lo ≤ z_mid ≤ hi (A7, v2.0-0)."""
    D = (v[1:] - v[:-1]) / (z[:-1] - z[1:]); zm = 0.5 * (z[:-1] + z[1:])
    W = np.isfinite(D) & (zm >= lo) & (zm <= hi)
    if not W.any() or not np.max(D[W]) > 0:
        return np.nan
    g = np.max(D[W]); sel = W & (D >= 0.9 * g); zg = float(np.sum(D[sel] * zm[sel]) / np.sum(D[sel]))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        up = np.nanmedian(np.where((z > zg) & (z <= zg + 3.0), v, np.nan)); lo_ = np.nanmedian(np.where((z < zg) & (z >= zg - 3.0), v, np.nan))
    thr = 0.5 * (up + lo_); a, b = v[:-1] - thr, v[1:] - thr
    cr = np.isfinite(a) & np.isfinite(b) & ((a <= 0) != (b <= 0)) & (np.abs(zm - zg) <= 10.0)
    if not cr.any():
        return np.nan
    t = a[cr] / (a[cr] - b[cr]); zc = z[:-1][cr] + t * (z[1:] - z[:-1])[cr]
    return float(zc[np.argmin(np.abs(zc - zg))])


def cmd_a7(args):
    g27 = _load("g27", "27_dp_groundtruth.py"); g31 = _load("g31", "31_scores.py")
    wc.require_input("dp_results", "the comparison with the soundings (Tables 1–2, SI Tables S1, S7–S11, Figures 4–6 and S11–S14, expectations E15–E21)", path=os.path.join(wc.ROOT, "data", "processed", "v1_4", "dp_results.json"))
    R = json.load(open(os.path.join(wc.ROOT, "data", "processed", "v1_4", "dp_results.json"), encoding="utf-8"))
    names = sorted(R["dps"], key=lambda n: R["dps"][n]["index"])
    O = {n: R["dps"][n]["per"]["primary|A"] for n in names}
    base = {n: dict(logmid=O[n]["logmid"], thr2p5=O[n]["thr2p5"], thr3=O[n]["thr3"]) for n in names}
    # step-specific alternatives
    zg = lambda n: np.array(O[n]["z_grid"]); vg = lambda n: np.array(O[n]["v_grid"])
    dp7_main = steep_logmid_window(vg("DP7"), zg("DP7"), -25.0, -15.0)
    dp1_shallow = steep_logmid_window(vg("DP1"), zg("DP1"), -8.0, np.inf)
    alts = {"primary (log-midpoint)": {n: base[n]["logmid"] for n in names},
            "2.5 Ω·m crossing": {n: base[n]["thr2p5"] for n in names},
            "3 Ω·m crossing": {n: base[n]["thr3"] for n in names},
            "DP7 main step": dict({n: base[n]["logmid"] for n in names}, DP7=dp7_main),
            "DP1 shallow saline layer": dict({n: base[n]["logmid"] for n in names}, DP1=dp1_shallow),
            "DP7 main step + DP1 shallow": dict({n: base[n]["logmid"] for n in names}, DP7=dp7_main, DP1=dp1_shallow)}
    C = g31.load_cases()
    F = {(r["method"], r["dp"]): r for r in C if r["set"] == "field" and r["truth_kind"] == "logmid"}
    meths = g31.FIELD_METHODS
    out = dict(note=POST_HOC, dp7_first_step=base["DP7"]["logmid"], dp7_main_step=dp7_main, dp1_deep_step=base["DP1"]["logmid"],
               dp1_shallow=dp1_shallow, bases=alts, counts={}, e15={})
    for an, yb in alts.items():
        out["counts"][an] = {}
        for dset, dps in (("DP2–DP7", g31.DPS), ("DP1–DP7", ["DP1"] + list(g31.DPS))):
            cnt = {}
            for m in meths:
                k = n = 0; ins = []
                for dp in dps:
                    r = F.get((m, dp)); y = yb[dp]
                    if r is None or y is None or not np.isfinite(y):
                        continue
                    n += 1
                    if np.isfinite(r["l"]) and r["l"] <= y <= r["u"]:
                        k += 1; ins.append(dp)
                cnt[m] = dict(k=k, n=n, inside=ins)
            out["counts"][an][dset] = cnt
        e15 = [n for n in names if O[n]["included"] and g27.XR15[0] <= O[n]["x"] <= g27.XR15[1]]
        ok = [n for n in e15 if yb[n] is not None and np.isfinite(yb[n]) and O[n]["bel"][1] <= yb[n] <= O[n]["bel"][2]]
        out["e15"][an] = dict(n=len(e15), n_ok=len(ok), dps=e15, ok=ok, holds=bool(len(ok) / len(e15) >= 2 / 3))
    out["thr2p5_vs_1980"] = {n: dict(thr2p5_2024=base[n]["thr2p5"], sal1980=O[n]["sal1980"], diff=base[n]["thr2p5"] - O[n]["sal1980"])
                             for n in ("DP3", "DP4", "DP5", "DP6", "DP7")}
    json.dump(out, open(D20("a7.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("[a7] DP7 main", dp7_main, "DP1 shallow", dp1_shallow)
    for an in alts:
        print(" ", an, "E15", out["e15"][an]["n_ok"], "/", out["e15"][an]["n"], {m: f'{v["k"]}/{v["n"]}' for m, v in out["counts"][an]["DP2–DP7"].items()})


# ─────────────────────────────────────────────────────────────────────────────
# A5 — damping with matched windows
# ─────────────────────────────────────────────────────────────────────────────
def cmd_a5(args):
    g27 = _load("g27", "27_dp_groundtruth.py"); s17 = _load("s17", "17_salinity_v2.py")
    v14 = os.path.join(wc.ROOT, "data", "processed", "v1_4")
    wc.require_input("dp_results", "the comparison with the soundings (Tables 1–2, SI Tables S1, S7–S11, Figures 4–6 and S11–S14, expectations E15–E21)", path=os.path.join(v14, "dp_results.json"))
    R = json.load(open(os.path.join(v14, "dp_results.json"), encoding="utf-8"))
    J = json.load(open(os.path.join(v14, "cpt.json"), encoding="utf-8"))
    SD = np.load(os.path.join(v14, g27.SD_DIR_NAME, "cpt.npz"), allow_pickle=True)
    A3p = D20("a3_lin24.npz"); A3 = np.load(A3p, allow_pickle=True) if os.path.exists(A3p) else None
    logs = g27.read_dp_values(); zimg = SD["z"]
    out = dict(note=POST_HOC, image_source="stored lam50_col (v1.4 samedate, CPT topography)", dps={})
    if A3 is not None:
        out["image_rerun_maxdiff_log10_at_dps"] = None
    sw_of = lambda rho, i: (lambda a: 100 * a["F"] * a["rsw"] / rho)(s17.archie_v2(g27.N_ARCH, np.random.default_rng([137200, i])))
    q4 = lambda v: [float(np.median(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95)), float(np.mean(v > 100))]
    diffs = []
    for nm in sorted(R["dps"], key=lambda n: R["dps"][n]["index"]):
        dd = R["dps"][nm]; o = dd["per"]["primary|A"]; i = dd["index"]; c = J["cpts"][dd["cpt"]]
        lg = logs[min(logs, key=lambda k: abs(k - dd["local_x"]))]
        d = lg["depth"]; r = lg["r"]; z1 = c["z"] - d
        cd = c["z"] - np.array(c["elev"]); cI = np.array([np.nan if v is None else v for v in c["isbt"]], float)
        cls = lambda dep: ("sand" if cI[int(np.argmin(np.abs(cd - dep)))] < g27.ISBT_SAND else "other")
        base = o["logmid"]; rec = dict(x=o["x"], base=base, rho_below_dp=o["rho_below"])
        if not np.isfinite(base):
            out["dps"][nm] = rec; continue
        lim = max(z1[-1], g27.Z_STOP); w = (z1 <= base - g27.RB_OFF) & (z1 >= lim)
        clean = np.array([cls(x) == "sand" for x in d]); sel = w & clean
        rec.update(window_top=float(base - g27.RB_OFF), window_bottom=float(lim), n_samples=int(sel.sum()), clean_m=float(sel.sum() * 0.01),
                   rho_below_dp_recomputed=float(np.median(r[sel])) if sel.any() else np.nan)
        ks = int(np.argmin(np.abs(SD["xcol"] - dd["our_x"]))); col = SD["lam50_col"][:, ks].astype(float)
        ok = np.isfinite(col)
        if A3 is not None:
            diffs.append(float(np.nanmax(np.abs(A3["lam50_col"][:, ks] - SD["lam50_col"][:, ks]))))
        if sel.any():
            vi = np.interp(-z1[sel], -zimg[ok], col[ok])
            rec["rho_img_matched"] = float(10 ** np.median(vi))
            rec["ratio_matched"] = rec["rho_img_matched"] / o["rho_below"] if np.isfinite(o["rho_below"]) else np.nan
            rec["sw_img_matched"] = q4(sw_of(rec["rho_img_matched"], i))
        bimg = float(SD["lam50_mhat_logmid"][ks]); wo = (zimg <= bimg - 4.0) & (zimg >= bimg - 10.0) & ok
        rec.update(image_base=bimg, rho_img_old=float(10 ** np.median(col[wo])) if wo.any() else np.nan, rho_img_old_stored=o["sd"]["rho_below_img"])
        rec["ratio_old"] = rec["rho_img_old"] / o["rho_below"] if np.isfinite(o["rho_below"]) else np.nan
        rec["sw_img_old"] = q4(sw_of(rec["rho_img_old"], i)) if np.isfinite(rec["rho_img_old"]) else None
        rec["sw_dp"] = q4(sw_of(o["rho_below"], i)) if np.isfinite(o["rho_below"]) else None
        rec["sw_dp_stored"] = o["sw"]
        if A3 is not None:
            cc = A3["lam50_cumsens_col"][:, ks].astype(float); okc = np.isfinite(cc)
            if sel.any():
                vals = np.interp(-z1[sel], -zimg[okc], cc[okc]); rec["log10_cumsens_window"] = float(np.median(np.unique(np.round(vals, 9))))
            lens = okc & (zimg >= base) & (zimg <= c["z"])
            rec["log10_cumsens_lens"] = float(np.median(np.unique(np.round(cc[lens], 9)))) if lens.any() else np.nan
            rec["log10_cumsens_at_base"] = float(np.interp(-base, -zimg[okc], cc[okc]))
        out["dps"][nm] = rec
    if A3 is not None:
        out["image_rerun_maxdiff_log10_at_dps"] = max(diffs); out["image_rerun_accepted"] = bool(max(diffs) <= 0.01)
    x5 = [n for n in ("DP3", "DP4", "DP5", "DP6", "DP7") if np.isfinite(out["dps"][n].get("ratio_matched", np.nan))]
    out["expectation"] = dict(n=len(x5), n_below_half=sum(out["dps"][n]["ratio_matched"] < 0.5 for n in x5), dps=x5,
                              below_half=[n for n in x5 if out["dps"][n]["ratio_matched"] < 0.5])
    out["expectation"]["holds"] = bool(out["expectation"]["n_below_half"] >= 4)
    json.dump(out, open(D20("a5.json"), "w", encoding="utf-8"), indent=1, default=float)
    for n, v in out["dps"].items():
        print(n, {k: (round(v[k], 3) if isinstance(v.get(k), float) else v.get(k)) for k in ("rho_below_dp", "rho_below_dp_recomputed", "rho_img_matched", "ratio_matched", "rho_img_old", "rho_img_old_stored", "ratio_old", "log10_cumsens_window")})
    print(out["expectation"], out.get("image_rerun_maxdiff_log10_at_dps"))


# ─────────────────────────────────────────────────────────────────────────────
# A3 — scores from samples, per-sounding table, paired tests
# ─────────────────────────────────────────────────────────────────────────────
def prior_T():
    """Training targets (lens base per column) of the four priors; cached (deterministic loaders)."""
    p = D20("prior_T.npz")
    if os.path.exists(p):
        z = np.load(p); return {k: z[k] for k in z.files}
    M = mods(); g30, b13 = M["g30"], M["b13"]
    import stage6_analysis as s6
    out = {}
    s6.init(); s6.TX = b13.TX; out["PRIOR3"] = np.asarray(s6.load_combined(physical=True)["tr"]["T"], float)
    out["PRIOR4"] = np.asarray(g30.load_v4()["tr"]["T"], float)
    _, _, C = g30.l24_sets(); out["PRIOR4-L24"] = np.asarray(C["tr"]["T"], float); del C
    _, _, S, _ = g30.stretch_sets(); out["PRIOR4-s"] = np.asarray(S["tr"]["T"], float); del S
    np.savez(p, **out)
    return out


def energy_crps(x, y, seed, ncap=2000):
    xs = np.asarray(x, float); xs = xs[np.isfinite(xs)]
    if len(xs) > ncap:
        xs = np.random.default_rng(seed).choice(xs, ncap, replace=False)
    xs = np.sort(xs); n = len(xs); i = np.arange(1, n + 1)
    return float(np.mean(np.abs(xs - y)) - np.sum((2 * i - n - 1) * xs) / n ** 2)


def clopper_pearson(k, n, a=0.10):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1)); hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def sign_flip(d):
    """Exact sign-flip test over all 2^n patterns. One-sided H1: mean(d) < 0 (BEL lower)."""
    import itertools
    d = np.asarray(d, float); d = d[np.isfinite(d)]; n = len(d)
    E = np.array(list(itertools.product([1, -1], repeat=n))); T = (E * np.abs(d)).mean(1); t0 = d.mean()
    return dict(n=n, mean_diff=float(t0), n_lower=int(np.sum(d < 0)), p_one=float(np.mean(T <= t0 + 1e-12)),
                p_two=float(np.mean(np.abs(T) >= abs(t0) - 1e-12)), n_patterns=len(E))


def cmd_a3(args):
    M = mods(); g30, b13 = M["g30"], M["b13"]
    g31 = _load("g31", "31_scores.py")
    TX, TXL = b13.TX, g30.TX_L24
    L18 = np.load(D20("a3_lin18_samples.npz")); L24 = np.load(D20("a3_lin24.npz"), allow_pickle=True)
    chk24 = json.load(open(D20("a3_lin24_check.json"), encoding="utf-8"))
    B3 = np.load(os.path.join(wc.ROOT, "data", "processed", "paper", "bel_products.npz"), allow_pickle=True)
    Z4 = np.load(os.path.join(g30.D4(), "bel_v4.npz"), allow_pickle=True); ZL = np.load(os.path.join(g30.D4L(), "bel_v4_l24.npz"), allow_pickle=True)
    Zs = np.load(os.path.join(g30.DLT(), "layout_vs_time.npz"), allow_pickle=True)
    PT = prior_T()
    DPs = [d for d in g31.dp_list() if d["dp"] in g31.DPS]
    meths = list(g31.FIELD_METHODS)
    def samples(m, x):
        """Predictive samples of method m at DP position x, and the column used."""
        if m in ("BEL3-18", "BEL4-18", "BEL4-18s", "BEL4-24"):
            P, T_ = {"BEL3-18": (B3["Pf"], TX), "BEL4-18": (Z4["Pf"], TX), "BEL4-18s": (Zs["sub_Pf"], TX), "BEL4-24": (ZL["Pf"], TXL)}[m]
            j = int(np.argmin(np.abs(T_ - x))); return P[:, j], T_[j]
        if m.startswith("PRIOR"):
            T_ = TXL if m == "PRIOR4-L24" else TX; j = int(np.argmin(np.abs(T_ - x))); return PT[m][:, j], T_[j]
        key = {"LIN-LM": "pooled_logmid", "LIN-3": "single_thr3", "LIN-MG": "pooled_maxgrad"}[m[:-3]]
        if m.endswith("-18"):
            k = int(np.argmin(np.abs(L18["x"] - x))); return L18[key][k].astype(float), L18["x"][k]
        k = int(np.argmin(np.abs(L24["xcol"] - x)))
        key24 = {"pooled_logmid": "pooled_dep_logmid", "single_thr3": "lam50_dep_thr3", "pooled_maxgrad": "pooled_dep_maxgrad"}[key]
        return L24[key24][k].astype(float), L24["xcol"][k]
    def score(x, y, seed):
        x = np.asarray(x, float); share = float(np.mean(np.isfinite(x)))
        if share < 0.5:
            return dict(share=share, l=np.nan, m=np.nan, u=np.nan, is_=np.nan, crps=np.nan, ae=np.nan, inside=False)
        xf = x[np.isfinite(x)]; l, md, u = np.percentile(xf, [5, 50, 95])
        return dict(share=share, l=float(l), m=float(md), u=float(u), is_=float(g31.interval_score(l, u, y)), crps=energy_crps(xf, y, seed),
                    ae=float(abs(md - y)), inside=bool(l <= y <= u))
    C = g31.load_cases(); ref = {(r["method"], r["dp"]): r for r in C if r["set"] == "field" and r["truth_kind"] == "logmid"}
    tab = {m: {} for m in meths}; rep = {0.5: {m: {} for m in meths}, 1.0: {m: {} for m in meths}}
    for mi, m in enumerate(meths):
        for di, d in enumerate(DPs):
            x, col = samples(m, d["x"]); y = d["logmid"]
            s = score(x, y, [200300, mi, di]); s["col"] = float(col)
            r0 = ref.get((m, d["dp"]))
            s["stored_is"] = r0["is"] if r0 else np.nan; s["stored_crps"] = r0["crps"] if r0 else np.nan; s["stored_approx"] = r0["approx"] if r0 else None
            tab[m][d["dp"]] = s
            for si, sg in ((1, 0.5), (2, 1.0)):
                xf = np.asarray(x, float); xf = xf[np.isfinite(xf)] if np.mean(np.isfinite(xf)) >= 0.5 else np.array([np.nan])
                xn = xf + np.random.default_rng([200310, si, mi, di]).normal(0, sg, len(xf))
                rep[sg][m][d["dp"]] = score(xn, y, [200300, mi, di])
    vec = lambda T_, m, k: np.array([T_[m][n][k] for n in g31.DPS], float)
    def summary(T_):
        out = {}
        for m in meths:
            k = int(sum(T_[m][n]["inside"] for n in g31.DPS)); n_ = int(np.isfinite(vec(T_, m, "is_")).sum())
            pr = None if m.startswith("PRIOR") else {"BEL3-18": "PRIOR3", "BEL4-18": "PRIOR4", "BEL4-18s": "PRIOR4-s", "BEL4-24": "PRIOR4-L24"}.get(m, "PRIOR4" if m.endswith("-18") else "PRIOR4-L24")
            o = dict(prior=pr, k=k, n=n_, n_dp=len(g31.DPS), cp90=clopper_pearson(k, len(g31.DPS)), width=float(np.nanmedian(vec(T_, m, "u") - vec(T_, m, "l"))),
                     IS=float(np.nanmean(vec(T_, m, "is_"))), CRPS=float(np.nanmean(vec(T_, m, "crps"))), MAE=float(np.nanmean(vec(T_, m, "ae"))))
            if pr:
                o["skill_IS"] = 1 - o["IS"] / float(np.nanmean(vec(T_, pr, "is_"))); o["skill_CRPS"] = 1 - o["CRPS"] / float(np.nanmean(vec(T_, pr, "crps")))
            out[m] = o
        return out
    res = dict(note=POST_HOC, lin24_check=chk24, per_sounding=tab, summary=summary(tab),
               sigma_rep={str(s): dict(per_sounding=rep[s], summary=summary(rep[s])) for s in rep})
    # exact sign-flip tests
    lins = ["LIN-LM-18", "LIN-3-18", "LIN-MG-18", "LIN-LM-24", "LIN-3-24", "LIN-MG-24"]
    res["sign_flip"] = {f"{b} vs {l}": {k: sign_flip(vec(tab, b, kk) - vec(tab, l, kk)) for k, kk in (("IS", "is_"), ("CRPS", "crps"))}
                        for b in ("BEL3-18", "BEL4-24") for l in lins}
    res["sign_flip_sigma_rep"] = {str(s): {f"{b} vs {l}": {k: sign_flip(vec(rep[s], b, kk) - vec(rep[s], l, kk)) for k, kk in (("IS", "is_"), ("CRPS", "crps"))}
                                           for b in ("BEL3-18", "BEL4-24") for l in lins} for s in rep}
    # leave-one-sounding-out ranking
    loo = {}
    for k, kk in (("IS", "is_"), ("CRPS", "crps")):
        ranks = {"BEL3-18": [], "BEL4-24": []}; beats = {f"{b} vs {l}": [] for b in ranks for l in lins}
        for leave in g31.DPS:
            keep = [n for n in g31.DPS if n != leave]
            mean = {m: float(np.nanmean([tab[m][n][kk] for n in keep])) for m in meths}
            order = sorted(meths, key=lambda m: (np.isnan(mean[m]), mean[m]))
            for b in ranks:
                ranks[b].append(order.index(b) + 1)
                for l in lins:
                    beats[f"{b} vs {l}"].append(bool(mean[b] < mean[l]))
        full = {m: float(np.nanmean(vec(tab, m, kk))) for m in meths}
        loo[k] = dict(full_rank={b: sorted(meths, key=lambda m: full[m]).index(b) + 1 for b in ranks}, ranks=ranks,
                      beats_in_all_6={kk_: all(v) for kk_, v in beats.items()}, beats_count={kk_: int(sum(v)) for kk_, v in beats.items()})
    res["loo"] = loo
    # Clopper–Pearson for the field coverages under the A7 bases and the held-out sets
    a7p = D20("a7.json")
    if os.path.exists(a7p):
        A7 = json.load(open(a7p, encoding="utf-8"))
        res["cp90_a7"] = {an: {ds: {m: dict(k=v["k"], n=v["n"], cp90=clopper_pearson(v["k"], v["n"])) for m, v in dv.items()} for ds, dv in av.items()}
                          for an, av in A7["counts"].items()}
    ho = {}
    for st, mth in (("ho_v3", "BEL3"), ("ho_v4", "BEL4"), ("ho_v4L24", "BEL4-L24")):
        rr = [r for r in C if r["set"] == st and r["method"] == mth]
        k = int(sum(r["l"] <= r["y"] <= r["u"] for r in rr)); ho[st] = dict(k=k, n=len(rr), cp90=clopper_pearson(k, len(rr)))
    res["cp90_heldout"] = ho
    json.dump(res, open(D20("a3.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("[a3] done; IS:", {m: round(v["IS"], 2) for m, v in res["summary"].items()})
    for k, v in res["sign_flip"].items():
        print("  ", k, "IS p1", round(v["IS"]["p_one"], 4), "p2", round(v["IS"]["p_two"], 4), "CRPS p1", round(v["CRPS"]["p_one"], 4))


# ─────────────────────────────────────────────────────────────────────────────
# A4 — outlier treatments and error model at the soundings
# ─────────────────────────────────────────────────────────────────────────────
A4_TREAT = {  # name: (index, {λ: run tag}, drop list, error-scale file)
    "G1b": (1, {5: "v20_g1b_lam5", 10: "v20_g1b_lam10", 20: "v20_g1b_lam20", 50: "g1b_thr299", 100: "v20_g1b_lam100"},
            "results/bel_v3/g1b_droplist_thr299.txt", None),
    "G2": (2, {5: "v20_g2_lam5", 10: "v20_g2_lam10", 20: "deep_drop_elec", 50: "v20_g2_lam50", 100: "v20_g2_lam100"},
           "results/deep_all_G2_droplist_flagged_electrodes.txt", None),
    "G3": (3, {5: "v20_g3_lam5", 10: "v20_g3_lam10", 20: "deep_downweight", 50: "v20_g3_lam50", 100: "v20_g3_lam100"},
           None, "results/deep_all_G3_err_scale.txt"),
    "RECIP": (4, {5: "v20_recip_lam5", 10: "v20_recip_lam10", 20: "v20_recip_lam20", 50: "g1_lam50_recip", 100: "v20_recip_lam100"},
              "results/deep_all_droplist.txt", "data/processed/v1_4/recip_err_scale.txt"),
}


def read_index_file(rel, with_value=False):
    out = {}
    for ln in open(os.path.join(wc.ROOT, rel), encoding="utf-8"):
        s = ln.split()
        if s and not s[0].startswith("#"):
            out[int(s[0])] = float(s[1]) if with_value else True
    return out


def bel_retrain(name, g1_keep, b13, s6, return_ho=False, cap_k=None):
    """BEL v3 refit within the 1974 G1 quadrupoles for one treatment (decision log v2.0-0); returns (BEL, field vector, info)."""
    obs, _ = b13.observed_sets(); y = obs["field"]
    _, _, drop, scale_file = A4_TREAT[name]
    cols = np.arange(len(g1_keep)); scale = np.ones(len(g1_keep))
    if name == "G2":
        dl = read_index_file(drop); cols = np.array([j for j, k in enumerate(g1_keep) if int(k) not in dl])
    if scale_file:
        sc = read_index_file(scale_file, with_value=True); scale = np.array([sc.get(int(k), 1.0) for k in g1_keep])
    def pd_treat(samples, mu, dev):
        out, keep = [], []
        n1 = dev.shape[0] - 1
        for i, z in samples:
            l = np.log10(z["R_medium"]) + mu
            e = dev.T @ np.random.default_rng([b13.APPROX_SEED, i]).standard_normal(dev.shape[0]) / np.sqrt(n1)
            R = 10 ** (l + e)
            sig = np.sqrt((b13.ERR_REL * R) ** 2 + b13.ERR_ABS ** 2); zz = np.random.default_rng([b13.NOISE_SEED, i]).standard_normal(len(R))
            if name == "RECIP":                                            # decision log v2.0-2: scaled error in log space
                Rn = R * np.exp(scale * sig / R * zz)
            else:
                Rn = R + scale * sig * zz
            if np.all(Rn[cols] > 0):
                out.append(np.log10(Rn[cols])); keep.append(i)
        return np.array(out), keep
    s6.init(); s6.TX = b13.TX                                        # init() builds stage6's own copy of 13_bel_full: patch that copy
    orig = s6.b13.prior_data; s6.b13.prior_data = pd_treat
    try:
        C = s6.load_combined(physical=True)
    finally:
        s6.b13.prior_data = orig
    assert C["tr"]["D"].shape[1] == len(cols), "treatment prior_data not applied"
    info = dict(n_cols=int(len(cols)), n_scaled=int(np.sum(scale != 1)), scale_min=float(scale.min()), scale_max=float(scale.max()),
                n_train=int(len(C["tr"]["D"])), n_dropped_cols=int(len(g1_keep) - len(cols)))
    bel = bel_fit(b13, C["tr"]["D"], C["tr"]["T"], cap_k)
    if return_ho:
        return bel, y[cols], info, (C["ho"]["D"], C["ho"]["T"])
    return bel, y[cols], info


def bel_fit(b13, D, T, cap_k=None):
    """13_bel_full.BEL, optionally with the data PCA capped at cap_k components (v2.0-4)."""
    if cap_k is None:
        return b13.BEL(D, T)
    bel = b13.BEL.__new__(b13.BEL)
    bel.pd = b13.PCA(D, k=cap_k); bel.pt = b13.PCA(T, 0.99)
    bel.Sd = bel.pd.fwd(D); bel.St = bel.pt.fwd(T)
    bel.cca = b13.CCA(bel.Sd, bel.St); bel.Cd = bel.cca.cx(bel.Sd); bel.Ct = bel.cca.cy(bel.St)
    bel.lin = []
    for i in range(bel.Ct.shape[1]):
        a, b_ = np.polyfit(bel.Cd[:, i], bel.Ct[:, i], 1); r = bel.Ct[:, i] - (a * bel.Cd[:, i] + b_); bel.lin.append((a, b_, r.std(ddof=2)))
    return bel


def cmd_a4diag(args):
    """Decision log v2.0-4: held-out check of every A4 BEL refit; RECIP with the data PCA capped at 114."""
    M = mods(); b13 = M["b13"]
    import stage6_analysis as s6
    g31 = _load("g31", "31_scores.py"); TX = b13.TX
    DPs = [d for d in g31.dp_list() if d["dp"] in g31.DPS]; jb = [int(np.argmin(np.abs(TX - d["x"]))) for d in DPs]
    g1_keep = np.asarray(wc.load_run("deep_drop_data")["keep_idx"]).astype(int)
    out = dict(note=POST_HOC + " Diagnostic (v2.0-4).", refits={})
    def check(name, bel, yv, ho, ti):
        Dh, Th = ho; cov, wid = [], []
        for j, (yy, t) in enumerate(zip(Dh, Th)):
            P = bel.posterior(yy, 2000, np.random.default_rng([129000, 90_000 + j]), "kde"); lo, hi = np.percentile(P, [5, 95], 0)
            cov.append((t >= lo) & (t <= hi)); wid.append(hi - lo)
        Pf = bel.posterior(yv, 2000, np.random.default_rng([200500, ti]), "kde")
        lo, hi = np.percentile(Pf[:, jb], [5, 95], 0); md = np.median(Pf[:, jb], 0)
        ins = [bool(l <= d["logmid"] <= u) for l, u, d in zip(lo, hi, DPs)]
        c = float(np.mean(cov))
        o = dict(n_heldout=int(len(Dh)), heldout_cov=c, heldout_width_median=float(np.median(wid)), field_width_dp_median=float(np.median(hi - lo)),
                 field_inside=int(sum(ins)), field_p50=[float(v) for v in md], kd=int(bel.pd.k), kt=int(bel.pt.k), cca_r5=[float(v) for v in bel.cca.r[:5]],
                 unreliable=bool(not (0.85 <= c <= 0.95)))
        print(f"[a4diag] {name}: ho cov {c:.3f} width {o['heldout_width_median']:.2f}; field width {o['field_width_dp_median']:.2f}; inside {o['field_inside']}/6; kd {o['kd']}", flush=True)
        return o
    s6.init(); s6.TX = TX
    C = s6.load_combined(physical=True); bel = b13.BEL(C["tr"]["D"], C["tr"]["T"]); obs, _ = b13.observed_sets()
    # G1: the paper fit; its field posterior is the paper's ([129000, 1]), so the DP check uses that seed
    o = check("G1", bel, obs["field"], (C["ho"]["D"], C["ho"]["T"]), 0)
    Pf = bel.posterior(obs["field"], 2000, np.random.default_rng([129000, 1]), "kde"); lo, hi = np.percentile(Pf[:, jb], [5, 95], 0)
    o.update(field_width_dp_median=float(np.median(hi - lo)), field_inside=int(sum(l <= d["logmid"] <= u for l, u, d in zip(lo, hi, DPs))), field_seed="[129000, 1] (paper)")
    out["refits"]["G1"] = o; out["refits"]["G1b"] = dict(identical_to="G1"); del C, bel
    for name in ("G2", "G3", "RECIP"):
        ti = A4_TREAT[name][0]
        bel, yv, info, ho = bel_retrain(name, g1_keep, b13, s6, return_ho=True)
        out["refits"][name] = dict(check(name, bel, yv, ho, ti), n_train=info["n_train"]); del bel
    bel, yv, info, ho = bel_retrain("RECIP", g1_keep, b13, s6, return_ho=True, cap_k=114)
    out["refits"]["RECIP_k114"] = dict(check("RECIP_k114", bel, yv, ho, 4), n_train=info["n_train"])
    out["recip_k114_6of6"] = bool(out["refits"]["RECIP_k114"]["field_inside"] == 6)
    json.dump(out, open(D20("a4diag.json"), "w", encoding="utf-8"), indent=1, default=float)


def cmd_a4(args):
    M = mods(); b13 = M["b13"]
    import stage6_analysis as s6
    c11 = _load("c11", "11_stage4_closeout.py"); g31 = _load("g31", "31_scores.py")
    DPs = g31.dp_list(); TX = b13.TX
    ref = wc.load_run("g1_lam50"); mesh, _ = wc.rebuild_paradomain(ref); ex, ez = wc.land_surface()
    grid = np.arange(np.ceil(ex.min()), np.floor(ex.max()) + 1, 2.0)
    xdp = np.array([grid[int(np.argmin(np.abs(grid - d["x"])))] for d in DPs])
    cs = wc.ColumnSampler(mesh, xdp)
    g1 = wc.load_run("deep_drop_data"); g1_keep = np.asarray(g1["keep_idx"]).astype(int)
    assert len(g1_keep) == 1974
    L18 = np.load(D20("a3_lin18_samples.npz")); kk = [int(np.argmin(np.abs(L18["x"] - x))) for x in xdp]
    B3 = np.load(os.path.join(wc.ROOT, "data", "processed", "paper", "bel_products.npz"), allow_pickle=True)
    jb = [int(np.argmin(np.abs(TX - d["x"]))) for d in DPs]
    q = lambda v: (lambda f: [float(np.median(f)), float(np.percentile(f, 5)), float(np.percentile(f, 95)), float(np.mean(np.isfinite(v)))]
                   if len(f) else [np.nan] * 3 + [0.0])(np.asarray(v, float)[np.isfinite(v)])
    out = dict(note=POST_HOC, columns=xdp.tolist(), treatments={})
    def readout(thr3_50, lm_pool, bel):
        rows = {}
        for i, d in enumerate(DPs):
            y = d["logmid"]; r = {}
            for k, S in (("thr3_lam50", thr3_50[i]), ("logmid_pooled", lm_pool[i]), ("bel", bel[i])):
                b = q(S); ok = b[3] >= 0.5 and np.isfinite(b[1])
                r[k] = dict(p50=b[0], p5=b[1], p95=b[2], share=b[3], inside=bool(ok and b[1] <= y <= b[2]))
            r["y"] = y; rows[d["dp"]] = r
        cnt = {k: int(sum(rows[n][k]["inside"] for n in g31.DPS)) for k in ("thr3_lam50", "logmid_pooled", "bel")}
        return rows, cnt
    # G1 = the paper's products (A3 regeneration, paper BEL)
    rows, cnt = readout([L18["single_thr3"][k].astype(float) for k in kk], [L18["pooled_logmid"][k].astype(float) for k in kk], [B3["Pf"][:, j] for j in jb])
    out["treatments"]["G1"] = dict(rows=rows, counts=cnt, runs={l: t for l, t in POOL_2018}, bel="paper BEL v3 (1718 training samples, 1974 G1 data)")
    import scipy.linalg as sla
    for name, (ti, runs, _, _) in A4_TREAT.items():
        meta, single, pooled = {}, None, []
        for lam in (5, 10, 20, 50, 100):
            run = wc.load_run(runs[lam])
            assert len(run["model"]) == ref["model"].size, "mesh differs"
            L_, _, _, _ = wc.cholesky_A(run)
            S = np.log(run["model"])[:, None] + sla.solve_triangular(L_, np.random.default_rng([200400, ti, lam]).standard_normal((len(run["model"]), 1000)),
                                                                     lower=True, trans="T")
            Dd = depth_samples(c11, cs, S)
            if lam == 50:
                single = Dd["thr3"]
            pooled.append(Dd["logmid"])
            meta[lam] = dict(tag=runs[lam], n_data=int(len(run["R"])), chi2=float(run["chi2"]), n_iter=int(run["n_iter"]), stop=str(run["stop_reason"]))
            print(f"[a4] {name} λ={lam}: {runs[lam]} n={meta[lam]['n_data']} χ²={meta[lam]['chi2']:.3f} {meta[lam]['stop']}", flush=True)
        P = np.hstack(pooled)
        if name == "G1b":
            belS = [B3["Pf"][:, j] for j in jb]; binfo = dict(identical_to="G1", reason="the 62 data G1b adds to G1 have no training responses")
        else:
            bel, yv, binfo = bel_retrain(name, g1_keep, b13, s6)
            Pf = bel.posterior(yv, 2000, np.random.default_rng([200500, ti]), "kde"); belS = [Pf[:, j] for j in jb]
            binfo.update(kd=int(bel.pd.k), kt=int(bel.pt.k)); del bel
            print(f"[a4] {name} BEL: {binfo}", flush=True)
        rows, cnt = readout([single[i] for i in range(len(DPs))], [P[i] for i in range(len(DPs))], belS)
        out["treatments"][name] = dict(rows=rows, counts=cnt, runs=meta, bel=binfo)
    c = {k: [v["counts"][k] for v in out["treatments"].values()] for k in ("thr3_lam50", "logmid_pooled", "bel")}
    out["expectation"] = dict(bel_min=min(c["bel"]), lin_range={k: max(c[k]) - min(c[k]) for k in ("thr3_lam50", "logmid_pooled")})
    out["expectation"]["holds"] = bool(out["expectation"]["bel_min"] >= 5 and all(v <= 1 for v in out["expectation"]["lin_range"].values()))
    json.dump(out, open(D20("a4.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("[a4]", {n: v["counts"] for n, v in out["treatments"].items()}, out["expectation"], flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# A10 (optional) — (a) blocky (L1) inversion of the G1 data
# ─────────────────────────────────────────────────────────────────────────────
def cmd_a10a(args):
    import scipy.linalg as sla
    t0 = time.time()
    c11 = _load("c11", "11_stage4_closeout.py"); sb = c11.sb; g31 = _load("g31", "31_scores.py")
    m02 = c11._load("inv02", "02_land_inversion.py"); m02.VERBOSE = False
    with contextlib.redirect_stdout(io.StringIO()):
        land, keep_idx = m02.apply_drop_list(m02.parse_crtomo(m02.LAND_FILE), os.path.join(wc.ROOT, "results", "deep_all_droplist.txt"))
        elec, _ = m02.parse_electrodes(m02.ELEC_FILE)
        data = m02.build_datacont(land, elec, err_rel=sb.ERR_REL)[0]
        mgr = m02.run_inversion(data, lam=50.0, para_depth=sb.PARA_DEPTH, para_dx=sb.PARA_DX, max_iter=sb.MAX_ITER, blocky=True)
        res = m02.compute_resolution_matrix(mgr, data, lam_expected=50.0)
    model = np.array(mgr.model); t_inv = time.time() - t0
    ref = wc.load_run("g1_lam50"); assert len(model) == ref["model"].size, "mesh differs"
    r = (np.log(np.array(data["rhoa"])) - np.log(np.array(mgr.inv.response))) / np.array(data["err"])
    chi2, chi2_pg, n_iter = float(np.mean(r ** 2)), float(mgr.inv.chi2()), m02.n_iterations(mgr)
    stop = "chi2<=1" if chi2_pg <= 1 else ("iteration cap" if n_iter >= sb.MAX_ITER else "dPhi<2%")
    L_, _, _, _ = wc.cholesky_A(dict(J=res["J"], data_err=res["err"], C=res["C"], lam=50.0, model=model))
    S = np.log(model)[:, None] + sla.solve_triangular(L_, np.random.default_rng([200600, 0]).standard_normal((len(model), 1000)), lower=True, trans="T")
    mesh, _ = wc.rebuild_paradomain(ref); ex, ez = wc.land_surface()
    grid = np.arange(np.ceil(ex.min()), np.floor(ex.max()) + 1, 2.0)
    DPs = g31.dp_list(); xdp = np.array([grid[int(np.argmin(np.abs(grid - d["x"])))] for d in DPs]); cs = wc.ColumnSampler(mesh, xdp)
    D = depth_samples(c11, cs, S); hat = c11.interfaces_all(cs(np.log10(model))[:, :, None], cs.z)
    rows = {}
    for i, d in enumerate(DPs):
        rr = {}
        for k in ("thr3", "logmid"):
            x = D[k][i]; sh = float(np.mean(np.isfinite(x))); xf = x[np.isfinite(x)]
            if sh >= 0.5:
                l, m, u = np.percentile(xf, [5, 50, 95])
                rr[k] = dict(share=sh, p50=float(m), p5=float(l), p95=float(u), mhat=float(hat[k][i, 0]),
                             inside_logmid=bool(l <= d["logmid"] <= u), inside_thr3=bool(np.isfinite(d["thr3"]) and l <= d["thr3"] <= u),
                             IS=float(g31.interval_score(l, u, d["logmid"])))
            else:
                rr[k] = dict(share=sh, mhat=float(hat[k][i, 0]))
        rows[d["dp"]] = dict(x=d["x"], col=float(xdp[i]), y_logmid=d["logmid"], y_thr3=d["thr3"], **rr)
    cnt = {k: dict(vs_logmid=int(sum(rows[n][k].get("inside_logmid", False) for n in g31.DPS)),
                   vs_thr3=int(sum(rows[n][k].get("inside_thr3", False) for n in g31.DPS))) for k in ("thr3", "logmid")}
    out = dict(note=POST_HOC, n_data=int(len(r)), chi2=chi2, chi2_pg=chi2_pg, n_iter=n_iter, stop=stop, t_inversion_s=t_inv,
               t_total_s=time.time() - t0, rows=rows, counts_dp2_dp7=cnt,
               caveat="A = JᵀWd²J + λCᵀC with C the (unweighted) constraint matrix of compute_resolution_matrix: the IRLS weights of the "
                      "L1 model norm at the final iterate are not in A, so the bands are the smooth-norm linearisation around the blocky model.")
    json.dump(out, open(D20("a10a.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("[a10a]", chi2, stop, n_iter, cnt, f"{time.time() - t0:.0f} s", flush=True)


def cmd_a10b(args):
    """BEL v3 (paper fit) on the 8 sharp factorial truths with z_b = −12 m, realisations 0–4 (stored v1.3 noisy data)."""
    t0 = time.time()
    M = mods(); b13 = M["b13"]
    import stage6_analysis as s6
    g25 = _load("g25", "25_generalisation.py"); g31 = _load("g31", "31_scores.py")
    s6.init(); s6.TX = b13.TX; TX = b13.TX
    C = s6.load_combined(physical=True); bel = b13.BEL(C["tr"]["D"], C["tr"]["T"]); T = np.asarray(C["tr"]["T"], float); del C
    cols = np.where((TX >= 130) & (TX <= 450))[0]
    pq = np.percentile(T, [5, 95], 0)
    truths = [f"G_{rl:g}_{k}_-12_0" for rl in (0.8, 1.42) for k in (2, 4, 8, 16)]
    out = dict(note=POST_HOC, columns=TX[cols].tolist(), truths={})
    for t in truths:
        ti = g25.TI[t]; zb = g25.params(t)["z_b"]; recs = []
        for real in range(5):
            z = np.load(os.path.join(g25.SRC, t, f"noisy_r{real}.npz"))
            R = z["R_noisy"]
            if not np.all(R > 0):
                recs.append(dict(real=real, skipped="non-positive R")); continue
            P = bel.posterior(np.log10(R), 2000, np.random.default_rng([200700, ti, real]), "kde")
            lo, hi = np.percentile(P[:, cols], [5, 95], 0)
            IS = g31.interval_score(lo, hi, zb); ISp = g31.interval_score(pq[0, cols], pq[1, cols], zb)
            recs.append(dict(real=real, cov=float(np.mean((lo <= zb) & (zb <= hi))), IS=float(np.mean(IS)), IS_prior=float(np.mean(ISp)),
                             median_bias=float(np.median(np.median(P[:, cols], 0) - zb)), width=float(np.median(hi - lo))))
        ok = [r for r in recs if "cov" in r]
        out["truths"][t] = dict(rho_lens=g25.params(t)["rho_lens"], k=g25.params(t)["k"], z_b=zb, reals=recs,
                                cov=float(np.mean([r["cov"] for r in ok])), IS=float(np.mean([r["IS"] for r in ok])),
                                IS_prior=float(np.mean([r["IS_prior"] for r in ok])), median_bias=float(np.median([r["median_bias"] for r in ok])))
        out["truths"][t]["skill_IS"] = 1 - out["truths"][t]["IS"] / out["truths"][t]["IS_prior"]
        print("[a10b]", t, {k: round(v, 3) for k, v in out["truths"][t].items() if isinstance(v, float)}, flush=True)
    out["t_total_s"] = time.time() - t0
    json.dump(out, open(D20("a10b.json"), "w", encoding="utf-8"), indent=1, default=float)


POOL_2018 = [(5, "g1_lam5"), (10, "g1_lam10"), (20, "deep_drop_data"), (50, "g1_lam50"), (100, "g1_lam100")]


def run_objective(run):
    """Final-iteration quantities of a stored run: χ² (mean of squared normalised log residuals, as stored), Φd = N·χ²,
    Φm = ‖C ln m‖² (C = the weighted constraint matrix incl. zWeight, as in the normal matrix), Φ = Φd + λΦm."""
    lam = float(run["lam"]); n = int(len(run["R"]))
    chi2 = float(run["chi2"]); phid = n * chi2
    phim = float(run["roughness"]) ** 2
    return dict(lam=lam, n_data=n, n_iter=int(run["n_iter"]), stop=str(run["stop_reason"]), chi2=chi2,
                chi2_pygimli=float(run["chi2_pygimli"]), phi_d=phid, phi_m=phim, phi=phid + lam * phim, lam_phi_m=lam * phim)


def cmd_a9(args):
    os.makedirs(D20(), exist_ok=True)
    out = dict(note=POST_HOC, pool_2018=[])
    for lam, tag in POOL_2018:
        r = wc.load_run(tag); o = run_objective(r); o["tag"] = tag
        out["pool_2018"].append(o)
    # mesh: vertical cell size at −10 … −20 mTAW (cell centres), lens part x = 100–450 m
    ref = wc.load_run("g1_lam50"); mesh, _ = wc.rebuild_paradomain(ref)
    dz, dx, cz = [], [], []
    for c in mesh.cells():
        ys = [n.pos().y() for n in c.nodes()]; xs = [n.pos().x() for n in c.nodes()]
        zc, xc = c.center().y(), c.center().x()
        if -20 <= zc <= -10 and 100 <= xc <= 450:
            dz.append(max(ys) - min(ys)); dx.append(max(xs) - min(xs)); cz.append(zc)
    dz = np.array(dz); dx = np.array(dx)
    out["mesh"] = dict(n_cells=int(mesh.cellCount()), n_cells_window=len(dz), cell_type="triangles (pyGIMLi createParaMesh, paraDX 0.33, paraDepth 60)",
                       dz_median=float(np.median(dz)), dz_p5=float(np.percentile(dz, 5)), dz_p95=float(np.percentile(dz, 95)),
                       dz_min=float(dz.min()), dz_max=float(dz.max()), dx_median=float(np.median(dx)))
    json.dump(out, open(D20("a9.json"), "w", encoding="utf-8"), indent=1)
    for o in out["pool_2018"]:
        print(f"λ {o['lam']:>5g} {o['tag']:<15} iter {o['n_iter']} stop {o['stop']:<8} χ² {o['chi2']:.3f} (pg {o['chi2_pygimli']:.3f}) "
              f"Φd {o['phi_d']:.0f} Φm {o['phi_m']:.1f} λΦm {o['lam_phi_m']:.0f} Φ {o['phi']:.0f}")
    print(out["mesh"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("a1"); a.add_argument("--time-only", action="store_true"); a.add_argument("--workers", type=int, default=6)
    sub.add_parser("a9"); sub.add_parser("a3-lin18"); sub.add_parser("a3-lin24"); sub.add_parser("a2"); sub.add_parser("a6"); sub.add_parser("a7"); sub.add_parser("a5"); sub.add_parser("a3"); sub.add_parser("a4"); sub.add_parser("a10a"); sub.add_parser("a10b"); sub.add_parser("a4diag"); sub.add_parser("a1-analyse"); sub.add_parser("prior-T")
    w = sub.add_parser("worker"); w.add_argument("--worker-id", type=int, required=True); w.add_argument("--dual", action="store_true")
    w.add_argument("--ids", type=int, nargs="+", required=True)
    args = ap.parse_args()
    dict(a1=cmd_a1, a2=cmd_a2, a6=cmd_a6, a7=cmd_a7, a5=cmd_a5, a3=cmd_a3, a4=cmd_a4, a10a=cmd_a10a, a10b=cmd_a10b, a4diag=cmd_a4diag, **{"a1-analyse": cmd_a1_analyse}, **{"prior-T": lambda a: prior_T()}, worker=cmd_worker, a9=cmd_a9, **{"a3-lin18": cmd_a3_lin18, "a3-lin24": cmd_a3_lin24})[args.cmd](args)


if __name__ == "__main__":
    main()
