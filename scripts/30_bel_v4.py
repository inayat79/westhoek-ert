"""
30_bel_v4.py
============
Westhoek ERT v1.5 — Task 3: BEL with a physically wider prior (prior v4). EXPLORATORY; designed after the DP results were
known (docs/decision_log.md v1.5-0). Prior v4 = prior v3 with exactly two changes: ρ_below log-uniform 2–100 Ω·m (v3:
2–20) and the deep-unit top U(−29, −26) mTAW (v3: U(−45, −20), "site" subset −38 … −22); ρ_discharge log-uniform
2.5–150 Ω·m as the combined v3 set; everything else as prior v2/v3. No DP resistivity value is used to build it.

  run    [--workers 6] [--time-only] [--out ROOT]  forward-model the v4 ensemble (4430 training + 443 held-out, as v3) and
                                                   20 dual samples (fine + medium), then fit the BELs → ROOT/data/processed/v1_5/bel_v4/
  resume [--workers 6] [--out ROOT]                as run, skipping finished samples
  report [--out ROOT]                              ROOT/results/v1_5/4_bel_v4.md, T_S11_bel_v4.{csv,md}, figs/F_16_bel_v3_v4.png
  worker                                           (internal)

Seeds (the v2/v3 scheme with ids offset by 10 000 000): parameters default_rng([88000, id]), noise default_rng([99000, id]),
approximation error default_rng([111000, id]) for ids 10 000 000 … 10 004 872 (training < 10 004 430); dual samples ids
10 500 000 … 10 500 019; base-BEL posteriors default_rng([10 129 000, k]); second BEL default_rng([10 132 000, k]).
Reused: 13_bel_full (draw, PriorModel2, prior_data, approximation error, PCA–CCA–KDE BEL, observed field data),
15_stage5_closeout.BEL2, 14_gateA_sensitivity.derived_targets, stage6_analysis.site, 18_paper.bel_products (v3 values).
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
read_trace.install("30_bel_v4.py")
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


s15 = _load("s15", "15_stage5_closeout.py")
g14, b13 = s15.g14, s15.b13
ID0, N_TR, N_HO = 10_000_000, 4430, 443
DUAL0, N_DUAL = 10_500_000, 20
SEED_BASE, SEED_2 = 10_129_000, 10_132_000
RHO_BELOW_V4, Z_DEEP_V4, RHO_DIS_V4 = (2.0, 100.0), (-29.0, -26.0), (2.5, 150.0)
N_POST = 2000
EXPLORATORY = "Exploratory; designed after the DP results were known (decision log v1.5-0)."
POST_HOC = "post hoc comparison; prior v4 was designed after the DP results"


def D4():
    return os.path.join(wc.ROOT, "data", "processed", "v1_5", "bel_v4")


def spec_v4():
    spec = json.load(open(b13.SPEC2, encoding="utf-8"))
    spec["params"]["rho_below"] = dict(range=list(RHO_BELOW_V4), dist="log-uniform", unit="Ω·m", why="prior v4")
    spec["params"]["z_deep"] = dict(range=list(Z_DEEP_V4), dist="uniform", unit="mTAW", why="prior v4")
    spec["params"]["rho_discharge"] = dict(range=list(RHO_DIS_V4), dist="log-uniform", unit="Ω·m", why="combined prior v3")
    return spec


def draw4(i):
    return b13.draw(spec_v4(), i)


def ensure():
    for d in ("train", "dual", "logs", "work"):
        os.makedirs(os.path.join(D4(), d), exist_ok=True)


def cmd_worker(args):
    import pygimli.meshtools as mt
    scheme = b13.pilot.build_scheme()
    meshes = {k: mt.createParaMesh(scheme.sensors(), **b13.MESH[k]) for k in (("medium", "fine") if args.dual else ("medium",))}
    cc = {k: (np.array([c.center().x() for c in m.cells()]), np.array([c.center().y() for c in m.cells()])) for k, m in meshes.items()}
    logf = open(os.path.join(D4(), "logs", f"worker_{args.worker_id}.log"), "a", encoding="utf-8")
    for i in args.ids:
        out = os.path.join(D4(), "dual" if args.dual else "train", f"s{i:08d}.npz")
        if os.path.exists(out):
            continue
        try:
            th = draw4(i); pm = b13.PriorModel2(th); t0 = time.time(); save = {}
            for k, (cx, cz) in cc.items():
                with contextlib.redirect_stdout(io.StringIO()):
                    save[f"R_{k}"] = b13.pilot.simulate_R(meshes[k], 10 ** pm.log10rho(cx, cz), scheme.size())
            np.savez(out + ".tmp.npz", theta=json.dumps(th), target=pm.target(), t=time.time() - t0, **save)
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
        wd = os.path.join(D4(), "work", f"w{w}"); os.makedirs(wd, exist_ok=True)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", "--worker-id", str(w)]
                                      + (["--dual"] if dual else []) + ["--ids"] + [str(i) for i in part], cwd=wd,
                                      env=dict(os.environ, OMP_NUM_THREADS="2", MPLBACKEND="Agg"), stdout=subprocess.DEVNULL,
                                      stderr=open(os.path.join(D4(), "logs", f"worker_{w}.err"), "a")))
    for p in procs:
        p.wait()


def cmd_run(args):
    ensure(); t0 = time.time()
    if args.time_only:
        ids = list(range(ID0, ID0 + 20))
        launch(ids, 1)
        ts = [float(np.load(os.path.join(D4(), "train", f"s{i:08d}.npz"))["t"]) for i in ids]
        n = N_TR + N_HO; per = float(np.mean(ts))
        wall = n * per / args.workers * 2.0
        plan = dict(n_timed=len(ts), mean_s=per, n_total=n, workers=args.workers, wall_h=wall / 3600)
        json.dump(plan, open(os.path.join(D4(), "plan.json"), "w", encoding="utf-8"), indent=1)
        print(json.dumps(plan, indent=1)); return
    todo = [i for i in range(ID0, ID0 + N_TR + N_HO) if not os.path.exists(os.path.join(D4(), "train", f"s{i:08d}.npz"))]
    print(f"[v4] {len(todo)} samples to forward-model", flush=True)
    launch(todo, args.workers)
    dual = [i for i in range(DUAL0, DUAL0 + N_DUAL) if not os.path.exists(os.path.join(D4(), "dual", f"s{i:08d}.npz"))]
    launch(dual, min(args.workers, len(dual)) or 1, dual=True)
    print(f"[v4] forward done ({time.time() - t0:.0f} s)", flush=True)
    analyse()
    print(f"[v4] BEL done ({time.time() - t0:.0f} s)", flush=True)


def load_v4():
    mu_d, dev = b13.approx_stats()
    d = os.path.join(D4(), "train"); sets = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz") and "tmp" not in f:
            z = np.load(os.path.join(d, f), allow_pickle=True)
            sets.append((int(f[1:-4]), {"R_medium": z["R_medium"], "theta": z["theta"], "target": z["target"]}))
    s6 = _load("s6x", "stage6_analysis.py") if False else __import__("stage6_analysis")
    out = {}
    for name, ss in (("tr", [x for x in sets if x[0] < ID0 + N_TR]), ("ho", [x for x in sets if x[0] >= ID0 + N_TR])):
        D, keep = b13.prior_data(ss, mu_d, dev)
        zm = {i: z for i, z in ss}
        th = [json.loads(str(zm[i]["theta"])) for i in keep]; T = np.array([zm[i]["target"] for i in keep])
        m = np.array([s6.site(t, tt) and t["rho_lens"] >= s6.PHYS_RHO_LENS_MIN for t, tt in zip(th, T)], dtype=bool)
        out[name] = dict(D=D[m], T=T[m], th=[t for t, k in zip(th, m) if k], n_all=len(ss), n_pos=len(keep), n_kept=int(m.sum()))
    return out


def analyse():
    import stage6_analysis as s6
    s6.init(); s6.TX = b13.TX
    C = load_v4(); obs, _ = b13.observed_sets()
    D, T, th = C["tr"]["D"], C["tr"]["T"], C["tr"]["th"]
    bel = b13.BEL(D, T)
    Pf = bel.posterior(obs["field"], N_POST, np.random.default_rng([SEED_BASE, 1]), "kde")
    cov, wid = [], []
    for j, (y, t) in enumerate(zip(C["ho"]["D"], C["ho"]["T"])):
        P = bel.posterior(y, N_POST, np.random.default_rng([SEED_BASE, 90_000 + j]), "kde"); lo, hi = np.percentile(P, [5, 95], 0)
        cov.append((t >= lo) & (t <= hi)); wid.append(hi - lo)
    sc = bel.Sd; var = sc.var(0, ddof=1); sf = bel.pd.fwd(obs["field"])[0]
    mu, Cm = b13.pilot.fast_mcd(sc); Ci = np.linalg.inv(Cm)
    fal = dict(t2=float(np.mean(np.sum(sc ** 2 / var, 1) < np.sum(sf ** 2 / var))),
               t2r=float(np.mean(np.einsum("ij,jk,ik->i", sc - mu, Ci, sc - mu) < (sf - mu) @ Ci @ (sf - mu))),
               q=float(np.mean(bel.pd.resid(D) < bel.pd.resid(obs["field"])[0])))
    Ytr = np.array([g14.derived_targets(t) for t in th]); Yho = np.array([g14.derived_targets(t) for t in C["ho"]["th"]])
    s15.RHO_BOUNDS.update(rho_lens=(0.4, 3.0), rho_below=RHO_BELOW_V4, rho_discharge=RHO_DIS_V4)
    b2 = s15.BEL2(D, Ytr, bounded=True)
    P2 = b2.posterior(obs["field"], N_POST, np.random.default_rng([SEED_2, 3]))
    c2 = []
    for j, y in enumerate(C["ho"]["D"]):
        P = b2.posterior(y, N_POST, np.random.default_rng([SEED_2, 60_000 + j])); lo, hi = np.percentile(P, [5, 95], 0)
        c2.append((Yho[j] >= lo) & (Yho[j] <= hi))
    # approximation error of the v4 dual samples vs the v2 correction
    mu_d, dev = b13.approx_stats()
    dd = []
    for f in sorted(os.listdir(os.path.join(D4(), "dual"))):
        if f.endswith(".npz") and "tmp" not in f:
            z = np.load(os.path.join(D4(), "dual", f)); dd.append(np.log10(z["R_fine"]) - np.log10(z["R_medium"]))
    dd = np.array(dd)
    approx = dict(n=len(dd), v4_mean_abs_median=float(np.median(np.abs(dd.mean(0)))), v4_sd_median=float(np.median(dd.std(0, ddof=1))),
                  v2_mean_abs_median=float(np.median(np.abs(mu_d))), v2_sd_median=float(np.median(dev.std(0, ddof=1))),
                  resid_after_v2_correction_median=float(np.median(np.abs(dd - mu_d))))
    np.savez(os.path.join(D4(), "bel_v4.npz"), Pf=Pf, P2=P2, ho_cov=np.array(cov), ho_wid=np.array(wid), cov2=np.array(c2),
             Y2_train=Ytr, fal=json.dumps(fal), approx=json.dumps(approx), n=json.dumps({k: {kk: v for kk, v in C[k].items() if kk.startswith("n_")} for k in ("tr", "ho")}),
             kd=bel.pd.k, kt=bel.pt.k, TX=b13.TX)


# ─────────────────────────────────────────────────────────────────────────────
def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = os.path.join(args.out, "results", "v1_5"); figs = os.path.join(res, "figs"); os.makedirs(figs, exist_ok=True)
    dpp = os.path.join(args.out, "data", "processed", "v1_5"); os.makedirs(dpp, exist_ok=True)
    s18 = _load("p18", "18_paper.py"); s18.mpl()
    Z = np.load(os.path.join(D4(), "bel_v4.npz"), allow_pickle=True)
    s18.DPAP = os.path.join(wc.ROOT, "data", "processed", "paper")         # cached v1.0 BEL products (v3)
    B3 = s18.bel_products()
    TX = Z["TX"]; Pf4, P24 = Z["Pf"], Z["P2"]; Pf3, P23 = B3["Pf"], B3["P2"]
    fal = json.loads(str(Z["fal"])); ap = json.loads(str(Z["approx"])); nn = json.loads(str(Z["n"]))
    plan = json.load(open(os.path.join(D4(), "plan.json"), encoding="utf-8"))
    R = json.load(open(os.path.join(wc.ROOT, "data", "processed", "v1_4", "dp_results.json"), encoding="utf-8"))
    names = sorted(R["dps"], key=lambda n: R["dps"][n]["index"])
    q = lambda v: [float(np.median(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))]
    fm = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    rb3, rb4 = q(10 ** P23[:, 3]), q(10 ** P24[:, 3])
    prior4 = q(10 ** Z["Y2_train"][:, 3])
    rows = []
    for n in names:
        o = R["dps"][n]["per"]["primary|A"]; j = int(np.argmin(np.abs(TX - o["x"])))
        b3, b4 = q(Pf3[:, j]), q(Pf4[:, j])
        rows.append(dict(dp=n, x=o["x"], tx=float(TX[j]), v3=b3, v4=b4, dp_base=o["logmid"], dp_rho=o["rho_below"],
                         in3=bool(b3[1] <= o["logmid"] <= b3[2]), in4=bool(b4[1] <= o["logmid"] <= b4[2])))
    e15 = [r for r in rows if r["dp"] != "DP1"]
    x6a = rb4[0] > 4.93; x6b = rb4[1] <= 10.89 and rb4[2] >= 14.80; x6c = sum(r["in4"] for r in e15) >= 5
    w3, w4 = np.log10(rb3[2] / rb3[1]), np.log10(rb4[2] / rb4[1]); wp = np.log10(prior4[2] / prior4[1])
    L = ["# 4 — BEL with a physically wider prior (prior v4; Task 3)", "", f"**{EXPLORATORY}**", "",
         "## Prior v4", "",
         "Prior v3 with two changes (decision log v1.5-0):", "",
         f"- **ρ_below log-uniform {RHO_BELOW_V4[0]:g}–{RHO_BELOW_V4[1]:g} Ω·m** (v3: 2–20). Lebbe (1981) and Paepen et al. (2020) "
         "classify sediment above 20 Ω·m as fresh, so a ceiling at 20 Ω·m excludes fresh water by construction. With Archie's law "
         "and the formation factor range of this study (2.8–3.6), fresh pore water of 500–800 µS/cm (12.5–20 Ω·m; the range "
         "used in the draft, §2.7) gives a bulk resistivity of about 35–72 Ω·m, and fresher water more; 100 Ω·m covers this. "
         "[CHECK: add a published range of bulk resistivities of fresh dune and beach sands.] The DP values were not used.",
         f"- **Deep-unit top U({Z_DEEP_V4[0]:g}, {Z_DEEP_V4[1]:g}) mTAW.** Prior v3 drew it from U(−45, −20) and kept −38 … −22 in the "
         "site subset (the −30 mTAW of earlier summaries is the synthetic truths' and reference model's value, not v3's). Sources: "
         "Lebbe (1981), Kortrijk-clay top −25.5 to −28 mTAW on K1; the CPT friction-ratio step at −26.8 … −28.6 mTAW "
         "(`v1_4/6_lithology.md`), a lithological marker. The deep resistivity prior stays log-uniform 3–50 Ω·m.",
         f"- Unchanged: everything else, incl. the contrast rule ρ_below ≥ 1.5 ρ_lens; ρ_discharge log-uniform {RHO_DIS_V4[0]:g}–{RHO_DIS_V4[1]:g} "
         "Ω·m (the combined v3 set). The second BEL's transform bound for ρ_below follows the prior (2–100 Ω·m).", "",
         f"Ensemble: {nn['tr']['n_all']} training + {nn['ho']['n_all']} held-out samples (as v3); after positive-data, site and physical "
         f"filters {nn['tr']['n_kept']} training / {nn['ho']['n_kept']} held-out. Forward: medium mesh, v2 approximation-error correction, "
         f"3 % + 1e-4 Ω noise. Timing: 20 forward runs, mean {plan['mean_s']:.1f} s → projected {plan['wall_h']:.1f} h on "
         f"{plan['workers']} workers (limit 36 h).", "",
         "## Checks (as v3)", "",
         f"- Prior falsification against the 2018 G1 data (share of training samples below the field value): T² {100*fal['t2']:.1f} %, "
         f"robust T² {100*fal['t2r']:.1f} %, Q {100*fal['q']:.1f} %.",
         f"- Held-out 90 % coverage of the lens base: **{100*Z['ho_cov'].mean():.1f} %** (median width {np.median(Z['ho_wid']):.1f} m); "
         f"second BEL ρ_below: {100*Z['cov2'][:, 3].mean():.1f} %.",
         f"- Approximation error, {ap['n']} v4 dual samples (fine vs medium): median |mean δ| {ap['v4_mean_abs_median']:.4f} (v2 dual "
         f"{ap['v2_mean_abs_median']:.4f}), median s.d. {ap['v4_sd_median']:.4f} (v2 {ap['v2_sd_median']:.4f}) in log10 R; after the v2 "
         f"correction the median |residual| is {ap['resid_after_v2_correction_median']:.4f}.", "",
         "## Field posterior (2018 data)", "",
         f"ρ_below: **v4 {fm(rb4)} Ω·m** vs v3 {fm(rb3)} Ω·m; v4 prior (training set) {fm(prior4)} Ω·m. Log10 width of the 90 % "
         f"interval: posterior {w4:.2f} vs prior {wp:.2f} (v3 {w3:.2f}).", "",
         f"| DP | x (m) | BEL column (m) | v3 base median [P5, P95] | v4 base median [P5, P95] | DP base, 2024 ({POST_HOC}) | in v3 90 % | in v4 90 % |",
         "|---|---|---|---|---|---|---|---|"]
    T11 = ["dp,x,bel_column,v3_med,v3_p5,v3_p95,v4_med,v4_p5,v4_p95,dp_base_posthoc,dp_rho_below_posthoc,in_v3,in_v4"]
    for r in rows:
        L.append(f"| {r['dp']} | {r['x']:.1f} | {r['tx']:g} | {fm(r['v3'])} | {fm(r['v4'])} | {r['dp_base']:.2f} | {'yes' if r['in3'] else 'no'} | "
                 f"{'yes' if r['in4'] else 'no'} |")
        T11.append(",".join(str(v) for v in [r["dp"], f"{r['x']:.1f}", f"{r['tx']:g}", *[f"{v:.2f}" for v in r["v3"]], *[f"{v:.2f}" for v in r["v4"]],
                                               f"{r['dp_base']:.2f}", f"{r['dp_rho']:.2f}", r["in3"], r["in4"]]))
    T11.append(",".join(["rho_below", "", "", *[f"{v:.2f}" for v in rb3], *[f"{v:.2f}" for v in rb4], "", "", "", ""]))
    learn = w4 > 0.8 * wp
    L += ["", f"## X6 — {POST_HOC}", "",
          f"- v4 ρ_below median > 4.93 Ω·m: **{'yes' if x6a else 'no'}** ({rb4[0]:.2f} Ω·m).",
          f"- v4 90 % interval contains 10.89–14.80 Ω·m (DP3–DP5): **{'yes' if x6b else 'no'}** ([{rb4[1]:.2f}, {rb4[2]:.2f}]).",
          f"- Base intervals contain the DP base at ≥ 5 of DP2–DP7: **{'yes' if x6c else 'no'}** ({sum(r['in4'] for r in e15)} of {len(e15)}; v3 "
          f"{sum(r['in3'] for r in e15)} of {len(e15)}).",
          ("- **Little learning on ρ_below**: the posterior 90 % width is " + f"{100*w4/wp:.0f} % of the prior's (log10), so the data "
           "barely constrain it — consistent with the equivalence of Task 2." if learn else
           f"- The posterior narrows ρ_below to {100*w4/wp:.0f} % of the prior 90 % width (log10).")]
    with open(os.path.join(res, "4_bel_v4.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(res, "T_S11_bel_v4.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(T11) + "\n")
    hdr = T11[0].split(",")
    with open(os.path.join(res, "T_S11_bel_v4.md"), "w", encoding="utf-8") as f:
        f.write(f"# T_S11 — BEL v3 vs v4 at the DP positions (Markdown copy of `T_S11_bel_v4.csv`)\n\n{EXPLORATORY} DP columns: {POST_HOC}.\n\n| "
                + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(l.split(",")) + " |" for l in T11[1:]) + "\n")
    json.dump(dict(rb3=rb3, rb4=rb4, prior4=prior4, width_post=w4, width_prior=wp, X6=dict(median=x6a, interval=x6b, base=x6c),
                   n_in4=sum(r["in4"] for r in e15), n_in3=sum(r["in3"] for r in e15), cov=float(Z["ho_cov"].mean()), fal=fal, approx=ap,
                   plan=plan, n=nn, learn_little=bool(learn)), open(os.path.join(dpp, "bel_v4_summary.json"), "w", encoding="utf-8"), indent=1)
    # F_16
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axes[0]
    for k, (P, col, lab, dx) in enumerate(((Pf3, "#6a3d9a", "BEL v3", -3), (Pf4, "#1b9e77", "BEL v4", 3))):
        lo, md, hi = np.percentile(P, [5, 50, 95], 0); hh = TX >= 130
        ax.fill_between(TX[hh], lo[hh], hi[hh], color=col, alpha=0.2, lw=0); ax.plot(TX[hh], md[hh], color=col, lw=1.5, label=f"{lab}: median, 90 %")
    ax.plot([r["x"] for r in rows], [r["dp_base"] for r in rows], "D", color="k", ms=4.5, label=f"DP base 2024 ({POST_HOC})")
    ex, ez = wc.land_surface(); ax.plot(ex, ez, "k-", lw=0.8)
    ax.set_xlim(100, 470); ax.set_ylim(-30, 8); ax.set_xlabel("x (m)"); ax.set_ylabel("Elevation (mTAW)")
    ax.set_title("(a) lens base, 2018 data", fontsize=8); ax.legend(fontsize=6, loc="lower left")
    ax = axes[1]
    bins = np.logspace(np.log10(1.5), np.log10(120), 40)
    ax.hist(10 ** P23[:, 3], bins, color="#6a3d9a", alpha=0.45, density=True, label="v3 posterior")
    ax.hist(10 ** P24[:, 3], bins, color="#1b9e77", alpha=0.45, density=True, label="v4 posterior")
    ax.hist(10 ** Z["Y2_train"][:, 3], bins, histtype="step", color="#1b9e77", lw=0.9, density=True, label="v4 prior (training)")
    for v in sorted({R["dps"][n]["per"]["primary|A"]["rho_below"] for n in names}):
        ax.axvline(v, color="k", lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("ρ_below (Ω·m)"); ax.set_title("(b) ρ_below; dotted: DP values (post hoc)", fontsize=7)
    ax.legend(fontsize=6)
    fig.suptitle(EXPLORATORY, fontsize=7); fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(figs, "F_16_bel_v3_v4.png"), dpi=300); plt.close(fig)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "resume"):
        s = sub.add_parser(c); s.add_argument("--out", default=wc.ROOT); s.add_argument("--workers", type=int, default=6)
        s.add_argument("--time-only", action="store_true")
    sub.add_parser("report").add_argument("--out", default=wc.ROOT)
    w = sub.add_parser("worker"); w.add_argument("--worker-id", type=int, required=True); w.add_argument("--dual", action="store_true")
    w.add_argument("--ids", type=int, nargs="+", required=True)
    a = ap.parse_args()
    dict(run=cmd_run, resume=cmd_run, report=cmd_report, worker=cmd_worker)[a.cmd](a)


if __name__ == "__main__":
    main()
