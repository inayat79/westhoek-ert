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
  synth / l24 / layout [--out ROOT]                v1.5-1 / v1.5-2 checks (decision log): BEL v4 on synthetic data, same-epoch
                                                   BEL on ERT.dat, and layout vs time (T1/L24; 2018 data on the L24 stretch)
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
    wc.require_input("dp_results", "the comparison with the soundings (Tables 1–2, SI Tables S1, S7–S11, Figures 4–6 and S11–S14, expectations E15–E21)", path=os.path.join(wc.ROOT, "data", "processed", "v1_4", "dp_results.json"))
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
    if os.path.exists(os.path.join(D4(), "synth.npz")) and os.path.exists(os.path.join(D4L(), "bel_v4_l24.npz")):
        report_checks(args)
    if os.path.exists(os.path.join(DLT(), "layout_vs_time.npz")):
        report_layout(args)


# ─────────────────────────────────────────────────────────────────────────────
# v1.5-1 (1): BEL v4 on synthetic data (T1/L18, T3/L18) — no new prior ensemble
# ─────────────────────────────────────────────────────────────────────────────
FRESH_MED, FRESH_LO, FRESH_HI = 8.0, 10.89, 14.80


def fit_v4():
    """The v4 BELs exactly as in analyse() (same training set)."""
    import stage6_analysis as s6
    s6.init(); s6.TX = b13.TX
    C = load_v4(); D, T, th = C["tr"]["D"], C["tr"]["T"], C["tr"]["th"]
    bel = b13.BEL(D, T)
    Ytr = np.array([g14.derived_targets(t) for t in th])
    s15.RHO_BOUNDS.update(rho_lens=(0.4, 3.0), rho_below=RHO_BELOW_V4, rho_discharge=RHO_DIS_V4)
    return bel, s15.BEL2(D, Ytr, bounded=True)


def cmd_synth(args):
    g28 = _load("g28", "28_dp_synthetic.py")
    bel, b2 = fit_v4()
    keep = np.asarray(wc.load_run("deep_drop_data")["keep_idx"], int)
    out = {}
    for truth in ("T1", "T3"):
        ti = g28.TRUTHS.index(truth) + 1
        R_true = np.load(os.path.join(g28.SRC(), f"fwd_{truth}_L18.npz"))["R"]
        for real in range(g28.N_REAL):
            rng = np.random.default_rng([150000, ti, 0, real])
            R = R_true + np.sqrt((0.03 * R_true) ** 2 + 1e-4 ** 2) * rng.standard_normal(len(R_true))
            y = R[keep]
            if np.any(y <= 0):
                out[f"{truth}_r{real}_skipped"] = np.array(1); continue
            y = np.log10(y)
            out[f"{truth}_r{real}_base"] = bel.posterior(y, N_POST, np.random.default_rng([SEED_BASE, 300_000 + 10 * ti + real]), "kde").astype(np.float32)
            out[f"{truth}_r{real}_P2"] = b2.posterior(y, N_POST, np.random.default_rng([SEED_2, 300_000 + 10 * ti + real])).astype(np.float32)
            print(f"[synth] {truth} r{real}: ρ_below median {np.median(10 ** out[f'{truth}_r{real}_P2'][:, 3]):.2f} Ω·m", flush=True)
    np.savez(os.path.join(D4(), "synth.npz"), TX=b13.TX, **out)


def fresh(meds, pooled):
    m = float(np.median(meds)); lo, hi = np.percentile(pooled, [5, 95])
    return dict(median=m, p5=float(lo), p95=float(hi), fresh=bool(m >= FRESH_MED and lo <= FRESH_LO and hi >= FRESH_HI))


# ─────────────────────────────────────────────────────────────────────────────
# v1.5-1 (2): same-epoch BEL v4 on ERT.dat (L24 geometry, own prior ensemble)
# ─────────────────────────────────────────────────────────────────────────────
L24_ID0, L24_DUAL0, L24_NDUAL = 20_000_000, 21_500_000, 100
L24_SEED_BASE, L24_SEED_2 = 20_129_000, 20_132_000
TX_L24 = np.arange(150.0, 451.0, 10.0)


def D4L():
    return os.path.join(wc.ROOT, "data", "processed", "v1_5", "bel_v4_l24")


def l24_geometry():
    """Electrodes (L24 layout), the 1643 Stage-1-kept quadrupoles, a k = 1 scheme, and the CPT topography (0–505 m)."""
    g28 = _load("g28", "28_dp_synthetic.py")
    elec, q, _ = g28.layout("L24")
    keep = np.load(os.path.join(wc.ROOT, "data", "processed", "v1_4", "samedate", "cpt.npz"))["keep"].astype(bool)
    qk = {k: v[keep] for k, v in q.items()}
    xs = np.arange(0.0, 506.0, 1.0); zs = g28.g27.cpt_topography()(xs)
    return g28, elec, qk, keep, (xs, zs)


def l24_worker(args):
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    g28, elec, qk, _, (xs, zs) = l24_geometry()
    scheme = g28.scheme_of(elec, qk)
    meshes = {k: mt.createParaMesh(scheme.sensors(), **b13.MESH[k]) for k in (("medium", "fine") if args.dual else ("medium",))}
    cc = {k: (np.array([c.center().x() for c in m.cells()]), np.array([c.center().y() for c in m.cells()])) for k, m in meshes.items()}
    logf = open(os.path.join(D4L(), "logs", f"worker_{args.worker_id}.log"), "a", encoding="utf-8")
    for i in args.ids:
        out = os.path.join(D4L(), "dual" if args.dual else "train", f"s{i:08d}.npz")
        if os.path.exists(out):
            continue
        try:
            th = draw4(i); t0 = time.time(); save = {}
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
            np.savez(out + ".tmp.npz", theta=json.dumps(th), target=pm.target(TX_L24), t=time.time() - t0, **save)
            os.replace(out + ".tmp.npz", out)
            logf.write(f"{time.strftime('%H:%M:%S')} s{i} {time.time() - t0:.1f}s\n"); logf.flush()
        except Exception as e:
            import traceback
            logf.write(f"FAILED s{i}: {e}\n{traceback.format_exc()}\n"); logf.flush()
    logf.write("worker finished\n")


def l24_launch(ids, workers, dual=False):
    parts = [ids[k::workers] for k in range(workers)]; procs = []
    for w, part in enumerate(parts):
        if not part:
            continue
        wd = os.path.join(D4L(), "work", f"w{w}"); os.makedirs(wd, exist_ok=True)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", "--l24", "--worker-id", str(w)]
                                      + (["--dual"] if dual else []) + ["--ids"] + [str(i) for i in part], cwd=wd,
                                      env=dict(os.environ, OMP_NUM_THREADS="2", MPLBACKEND="Agg"), stdout=subprocess.DEVNULL,
                                      stderr=open(os.path.join(D4L(), "logs", f"worker_{w}.err"), "a")))
    for p in procs:
        p.wait()


def cmd_l24(args):
    for d in ("train", "dual", "logs", "work"):
        os.makedirs(os.path.join(D4L(), d), exist_ok=True)
    t0 = time.time()
    if args.time_only:
        ids = list(range(L24_ID0, L24_ID0 + 20)); l24_launch(ids, 1)
        ts = [float(np.load(os.path.join(D4L(), "train", f"s{i:08d}.npz"))["t"]) for i in ids]
        per = float(np.mean(ts)); n = N_TR + N_HO
        dual = [i for i in range(L24_DUAL0, L24_DUAL0 + 2)]; l24_launch(dual, 1, dual=True)
        td = float(np.mean([np.load(os.path.join(D4L(), "dual", f"s{i:08d}.npz"))["t"] for i in dual]))
        wall = (n * per + L24_NDUAL * td) / args.workers * 2.0
        plan = dict(n_timed=len(ts), mean_s=per, dual_mean_s=td, n_total=n, n_dual=L24_NDUAL, workers=args.workers, wall_h=wall / 3600)
        json.dump(plan, open(os.path.join(D4L(), "plan.json"), "w", encoding="utf-8"), indent=1)
        print(json.dumps(plan, indent=1)); return
    todo = [i for i in range(L24_ID0, L24_ID0 + N_TR + N_HO) if not os.path.exists(os.path.join(D4L(), "train", f"s{i:08d}.npz"))]
    print(f"[l24] {len(todo)} samples", flush=True); l24_launch(todo, args.workers)
    dual = [i for i in range(L24_DUAL0, L24_DUAL0 + L24_NDUAL) if not os.path.exists(os.path.join(D4L(), "dual", f"s{i:08d}.npz"))]
    l24_launch(dual, args.workers, dual=True)
    print(f"[l24] forward done ({time.time() - t0:.0f} s)", flush=True)
    l24_analyse()
    print(f"[l24] BEL done ({time.time() - t0:.0f} s)", flush=True)


def l24_analyse():
    import stage6_analysis as s6
    s6.init(); s6.TX = TX_L24
    g28, elec, qk, keep, _ = l24_geometry()
    dual = [np.load(os.path.join(D4L(), "dual", f)) for f in sorted(os.listdir(os.path.join(D4L(), "dual"))) if f.endswith(".npz") and "tmp" not in f]
    dl = np.array([np.log10(z["R_fine"]) - np.log10(z["R_medium"]) for z in dual]); mu_d = dl.mean(0); dev = dl - mu_d
    d = os.path.join(D4L(), "train"); sets = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz") and "tmp" not in f:
            z = np.load(os.path.join(d, f), allow_pickle=True)
            sets.append((int(f[1:-4]), {"R_medium": z["R_medium"], "theta": z["theta"], "target": z["target"]}))
    C = {}
    for name, ss in (("tr", [x for x in sets if x[0] < L24_ID0 + N_TR]), ("ho", [x for x in sets if x[0] >= L24_ID0 + N_TR])):
        D, kp = b13.prior_data(ss, mu_d, dev); zm = {i: z for i, z in ss}
        th = [json.loads(str(zm[i]["theta"])) for i in kp]; T = np.array([zm[i]["target"] for i in kp])
        m = np.array([s6.site(t, tt) and t["rho_lens"] >= s6.PHYS_RHO_LENS_MIN for t, tt in zip(th, T)], dtype=bool)
        C[name] = dict(D=D[m], T=T[m], th=[t for t, k in zip(th, m) if k], n_all=len(ss), n_kept=int(m.sum()))
    E = g28.g27.read_ert(); y = np.log10(E["R"][keep])
    D, T, th = C["tr"]["D"], C["tr"]["T"], C["tr"]["th"]
    bel = b13.BEL(D, T); Pf = bel.posterior(y, N_POST, np.random.default_rng([L24_SEED_BASE, 1]), "kde")
    cov = []
    for j, (yy, t) in enumerate(zip(C["ho"]["D"], C["ho"]["T"])):
        P = bel.posterior(yy, N_POST, np.random.default_rng([L24_SEED_BASE, 90_000 + j]), "kde"); lo, hi = np.percentile(P, [5, 95], 0)
        cov.append((t >= lo) & (t <= hi))
    sc = bel.Sd; var = sc.var(0, ddof=1); sf = bel.pd.fwd(y[None])[0]
    mu, Cm = b13.pilot.fast_mcd(sc); Ci = np.linalg.inv(Cm)
    fal = dict(t2=float(np.mean(np.sum(sc ** 2 / var, 1) < np.sum(sf ** 2 / var))),
               t2r=float(np.mean(np.einsum("ij,jk,ik->i", sc - mu, Ci, sc - mu) < (sf - mu) @ Ci @ (sf - mu))),
               q=float(np.mean(bel.pd.resid(D) < bel.pd.resid(y[None])[0])))
    Ytr = np.array([g14.derived_targets(t) for t in th]); Yho = np.array([g14.derived_targets(t) for t in C["ho"]["th"]])
    s15.RHO_BOUNDS.update(rho_lens=(0.4, 3.0), rho_below=RHO_BELOW_V4, rho_discharge=RHO_DIS_V4)
    b2 = s15.BEL2(D, Ytr, bounded=True); P2 = b2.posterior(y[None], N_POST, np.random.default_rng([L24_SEED_2, 3]))
    c2 = []
    for j, yy in enumerate(C["ho"]["D"]):
        P = b2.posterior(yy, N_POST, np.random.default_rng([L24_SEED_2, 60_000 + j])); lo, hi = np.percentile(P, [5, 95], 0)
        c2.append((Yho[j] >= lo) & (Yho[j] <= hi))
    np.savez(os.path.join(D4L(), "bel_v4_l24.npz"), Pf=Pf, P2=P2, ho_cov=np.array(cov), cov2=np.array(c2), Y2_train=Ytr, TX=TX_L24,
             fal=json.dumps(fal), n=json.dumps({k: {kk: v for kk, v in C[k].items() if kk.startswith("n_")} for k in ("tr", "ho")}),
             approx=json.dumps(dict(n=len(dl), mean_abs_median=float(np.median(np.abs(mu_d))), sd_median=float(np.median(dev.std(0, ddof=1))))))


# ─────────────────────────────────────────────────────────────────────────────
def report_checks(args):
    """5_bel_checks.md and F_17 (v1.5-1 items 1 and 2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = os.path.join(args.out, "results", "v1_5"); figs = os.path.join(res, "figs")
    dpp = os.path.join(args.out, "data", "processed", "v1_5")
    wc.require_input("dp_results", "the comparison with the soundings (Tables 1–2, SI Tables S1, S7–S11, Figures 4–6 and S11–S14, expectations E15–E21)", path=os.path.join(wc.ROOT, "data", "processed", "v1_4", "dp_results.json"))
    R = json.load(open(os.path.join(wc.ROOT, "data", "processed", "v1_4", "dp_results.json"), encoding="utf-8"))
    names = sorted(R["dps"], key=lambda n: R["dps"][n]["index"])
    S = np.load(os.path.join(D4(), "synth.npz")); L = np.load(os.path.join(D4L(), "bel_v4_l24.npz"), allow_pickle=True)
    g28 = _load("g28", "28_dp_synthetic.py"); Dg = {d["name"]: d for d in g28.dp_inputs()}
    fm = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    q = lambda v: [float(np.median(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))]
    out = {}
    for truth in ("T1", "T3"):
        keys = [k[:-3] for k in S.files if k.startswith(truth) and k.endswith("_P2")]
        meds = [np.median(10 ** S[f"{k}_P2"][:, 3]) for k in keys]; pooled = np.concatenate([10 ** S[f"{k}_P2"][:, 3] for k in keys])
        cov = {n: [] for n in names}
        for k in keys:
            P = S[f"{k}_base"]
            for n in names:
                j = int(np.argmin(np.abs(S["TX"] - Dg[n]["x"]))); lo, hi = np.percentile(P[:, j], [5, 95]); cov[n].append(bool(lo <= Dg[n]["base"] <= hi))
        out[truth] = dict(fresh(meds, pooled), n=len(keys), skipped=sum(1 for k in S.files if k.startswith(truth) and k.endswith("skipped")),
                          per_real=[float(m) for m in meds], cov={n: float(np.mean(v)) for n, v in cov.items()},
                          base_med={n: float(np.median([np.median(S[f'{k}_base'][:, int(np.argmin(np.abs(S['TX'] - Dg[n]['x'])))]) for k in keys])) for n in names})
    P2 = L["P2"]; rbL = 10 ** P2[:, 3]
    out["ERT"] = dict(fresh([np.median(rbL)], rbL), cov=float(np.mean(L["ho_cov"])), cov2=float(np.mean(L["cov2"][:, 3])),
                      fal=json.loads(str(L["fal"])), n=json.loads(str(L["n"])), approx=json.loads(str(L["approx"])))
    t1, er = out["T1"]["fresh"], out["ERT"]["fresh"]
    readout = ("fresh water on T1 and on ERT.dat → the 2018 value points to change between 2018 and 2024" if t1 and er else
               "fresh water on T1 but not on ERT.dat → model-class error on real data" if t1 else
               "no fresh water on T1 → a method / model-class limit, independent of time")
    TXL = L["TX"]; base_rows = []
    for n in names:
        o = R["dps"][n]["per"]["primary|A"]; j = int(np.argmin(np.abs(TXL - o["x"])))
        b = q(L["Pf"][:, j]); s_ = o["sd"]
        base_rows.append(dict(dp=n, x=o["x"], col=float(TXL[j]), bel=b, dp_base=o["logmid"], inside=bool(b[1] <= o["logmid"] <= b[2]),
                              lin_logmid=s_["band_logmid_pooled"][1:4], lin_thr3=s_["band_thr3"][1:4]))
    out["ERT"]["base"] = base_rows
    plan = json.load(open(os.path.join(D4L(), "plan.json"), encoding="utf-8"))
    M = ["# 5 — BEL checks: synthetic data and the 2025 ERT line (ERT.dat; v1.5-1)", "", f"**{EXPLORATORY}** Decisions and read-out logged "
         "before running (`docs/decision_log.md` v1.5-1).", "",
         "## (1) BEL v4 on synthetic data (T1 and T3 on the 2018 layout, 1974 G1 quadrupoles)", "",
         "| truth | realisations | ρ_below: median of posterior medians | pooled 90 % interval | per realisation | true ρ_below (DP3–DP5) | sees fresh water? |",
         "|---|---|---|---|---|---|---|"]
    for t in ("T1", "T3"):
        o = out[t]
        M.append(f"| {t} | {o['n']} (skipped {o['skipped']}) | {o['median']:.2f} | [{o['p5']:.2f}, {o['p95']:.2f}] | "
                 + ", ".join(f"{v:.2f}" for v in o["per_real"]) + f" | {'10.89, 13.57, 14.80' if t == 'T1' else '4.93'} | "
                 + ("**yes**" if o["fresh"] else "**no**") + " |")
    M += ["", "Base intervals at the DP x positions (share of realisations whose 90 % interval contains the true base; median of "
          "the posterior medians):", "", "| DP | x | true base | T1: coverage / median | T3: coverage / median |", "|---|---|---|---|---|"]
    for n in names:
        M.append(f"| {n} | {Dg[n]['x']:.1f} | {Dg[n]['base']:.2f} | {100*out['T1']['cov'][n]:.0f} % / {out['T1']['base_med'][n]:.2f} | "
                 f"{100*out['T3']['cov'][n]:.0f} % / {out['T3']['base_med'][n]:.2f} |")
    e = out["ERT"]
    M += ["", "## (2) BEL v4 on the 2025 ERT line (ERT.dat, April 2025, about seven months after the CPTs (E. Vrancken, pers. comm.); L24 geometry, own prior ensemble)", "",
          f"Ensemble {e['n']['tr']['n_all']} training + {e['n']['ho']['n_all']} held-out (after filters {e['n']['tr']['n_kept']} / "
          f"{e['n']['ho']['n_kept']}); timing: 20 runs, mean {plan['mean_s']:.1f} s (dual {plan['dual_mean_s']:.1f} s) → projected "
          f"{plan['wall_h']:.1f} h on {plan['workers']} workers (limit 24 h). Approximation error ({e['approx']['n']} dual samples): median "
          f"|mean δ| {e['approx']['mean_abs_median']:.4f}, median s.d. {e['approx']['sd_median']:.4f} (log10 R).",
          f"- Prior falsification against ERT.dat (share of training samples below the field value): T² {100*e['fal']['t2']:.1f} %, "
          f"robust T² {100*e['fal']['t2r']:.1f} %, Q {100*e['fal']['q']:.1f} %.",
          f"- Held-out 90 % coverage: lens base {100*e['cov']:.1f} %, ρ_below {100*e['cov2']:.1f} %.",
          f"- **ρ_below (2025 ERT line): {e['median']:.2f} [{e['p5']:.2f}, {e['p95']:.2f}] Ω·m**; DP3–DP5 (clean sand): "
          "10.89–14.80 Ω·m → finds fresh water: " + ("**yes**" if e["fresh"] else "**no**") + ".", "",
          "| DP | x | BEL column | BEL base median [P5, P95] | DP base | inside | linearised λ-pooled log-midpoint (v1.4 T_S7) | linearised λ = 50 3 Ω·m |",
          "|---|---|---|---|---|---|---|---|"]
    for r in base_rows:
        M.append(f"| {r['dp']} | {r['x']:.1f} | {r['col']:g} | {fm(r['bel'])} | {r['dp_base']:.2f} | {'yes' if r['inside'] else 'no'} | "
                 f"{fm(r['lin_logmid'])} | {fm(r['lin_thr3'])} |")
    M += ["", f"## Read-out (logged before running)", "", f"**{readout}.**", "", "**Note (v1.7-1):** this test was pre-registered and evaluated assuming a near-simultaneous survey; the 2025 ERT line (ERT.dat) was measured in April 2025, about seven months after the CPTs of 23 September 2024 (E. Vrancken, pers. comm.). The gap is reported as a limitation; no verdict changes."]
    with open(os.path.join(res, "5_bel_checks.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(M) + "\n")
    out["readout"] = readout
    json.dump(out, open(os.path.join(dpp, "bel_checks.json"), "w", encoding="utf-8"), indent=1)
    # F_17
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axes[0]; bins = np.logspace(np.log10(1.5), np.log10(120), 40)
    for t, col in (("T1", "#1f4e9c"), ("T3", "#d35400")):
        pooled = np.concatenate([10 ** S[f"{k[:-3]}_P2"][:, 3] for k in S.files if k.startswith(t) and k.endswith("_P2")])
        ax.hist(pooled, bins, density=True, alpha=0.45, color=col, label=f"BEL v4 on {t}/L18")
    ax.hist(rbL, bins, density=True, histtype="step", lw=1.3, color="k", label="BEL v4 on the 2025 ERT line")
    for v in (10.89, 13.57, 14.80):
        ax.axvline(v, color="#1f4e9c", ls=":", lw=0.7)
    ax.axvline(4.93, color="#d35400", ls=":", lw=0.7); ax.set_xscale("log"); ax.set_xlabel("ρ_below (Ω·m)")
    ax.set_title("(a) ρ_below posteriors; dotted: true / DP values", fontsize=7); ax.legend(fontsize=5.8)
    ax = axes[1]
    lo, md, hi = np.percentile(L["Pf"], [5, 50, 95], 0)
    ax.fill_between(TXL, lo, hi, color="#1b9e77", alpha=0.25, lw=0); ax.plot(TXL, md, color="#1b9e77", lw=1.4, label="BEL v4 on the 2025 ERT line")
    xr = [r["x"] for r in base_rows]
    ax.errorbar(np.array(xr) + 3, [r["lin_logmid"][0] for r in base_rows],
                yerr=[[r["lin_logmid"][0] - r["lin_logmid"][1] for r in base_rows], [r["lin_logmid"][2] - r["lin_logmid"][0] for r in base_rows]],
                fmt="^", color="#1f4e9c", ms=3.5, lw=0.8, label="linearised log-midpoint, λ-pooled (v1.4)")
    ax.plot(xr, [r["dp_base"] for r in base_rows], "kD", ms=4, label="DP base (2024)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("Elevation (mTAW)"); ax.set_ylim(-25, 5); ax.legend(fontsize=5.8, loc="lower right")
    ax.set_title("(b) 2025 ERT line: lens base", fontsize=7)
    fig.suptitle(EXPLORATORY, fontsize=7); fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(figs, "F_17_bel_synthetic_and_same_epoch.png"), dpi=300)
    wc.save_publication(fig, args.out, "F_17_bel_synthetic_and_2025_ert_line"); plt.close(fig)
    print("\n".join(M))


# ─────────────────────────────────────────────────────────────────────────────
# v1.5-2 (B): layout vs time for ρ_below — no new prior ensemble
# ─────────────────────────────────────────────────────────────────────────────
LAY_SEED_FIELD, LAY_SEED_HO_BASE, LAY_SEED_HO_2 = 500_000, 590_000, 560_000


def DLT():
    return os.path.join(wc.ROOT, "data", "processed", "v1_5", "bel_v4_layout")


def l24_sets():
    """The L24 training / held-out sets exactly as in l24_analyse()."""
    import stage6_analysis as s6
    s6.init(); s6.TX = TX_L24
    g28, _, _, keep, _ = l24_geometry()
    dual = [np.load(os.path.join(D4L(), "dual", f)) for f in sorted(os.listdir(os.path.join(D4L(), "dual"))) if f.endswith(".npz") and "tmp" not in f]
    dl = np.array([np.log10(z["R_fine"]) - np.log10(z["R_medium"]) for z in dual]); mu_d = dl.mean(0); dev = dl - mu_d
    d = os.path.join(D4L(), "train"); sets = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz") and "tmp" not in f:
            z = np.load(os.path.join(d, f), allow_pickle=True)
            sets.append((int(f[1:-4]), {"R_medium": z["R_medium"], "theta": z["theta"], "target": z["target"]}))
    C = {}
    for name, ss in (("tr", [x for x in sets if x[0] < L24_ID0 + N_TR]), ("ho", [x for x in sets if x[0] >= L24_ID0 + N_TR])):
        D, kp = b13.prior_data(ss, mu_d, dev); zm = {i: z for i, z in ss}
        th = [json.loads(str(zm[i]["theta"])) for i in kp]; T = np.array([zm[i]["target"] for i in kp])
        m = np.array([s6.site(t, tt) and t["rho_lens"] >= s6.PHYS_RHO_LENS_MIN for t, tt in zip(th, T)], dtype=bool)
        C[name] = dict(D=D[m], T=T[m], th=[t for t, k in zip(th, m) if k], n_all=len(ss), n_kept=int(m.sum()))
    return g28, keep, C


def l24_stretch_cols():
    """2018 G1 quadrupoles (columns of the BEL data vector) whose four electrodes all lie at x ≥ the L24 line start."""
    g27 = _load("g27x", "27_dp_groundtruth.py")
    E = g27.read_ert(); x0 = float(g27.X0_CPT - E["xloc"].max())
    sch = b13.pilot.build_scheme()
    x = np.array([p[0] for p in sch.sensors()])
    A = np.array([np.asarray(sch(k), int) for k in ("a", "b", "m", "n")]).T
    assert A.min() >= 0 and len(A) == len(wc.load_run("deep_drop_data")["keep_idx"])
    return np.where((x[A] >= x0).all(1))[0], x0


def prior_data_cols(samples, mu, dev, cols):
    """13_bel_full.prior_data on the full G1 vector (same seeds), restricted to `cols`; positivity checked on `cols`."""
    out, keep = [], []
    n1 = dev.shape[0] - 1
    for i, z in samples:
        l = np.log10(z["R_medium"]) + mu
        e = dev.T @ np.random.default_rng([b13.APPROX_SEED, i]).standard_normal(dev.shape[0]) / np.sqrt(n1)
        R = 10 ** (l + e)
        Rn = R + np.sqrt((b13.ERR_REL * R) ** 2 + b13.ERR_ABS ** 2) * np.random.default_rng([b13.NOISE_SEED, i]).standard_normal(len(R))
        if np.all(Rn[cols] > 0):
            out.append(np.log10(Rn[cols])); keep.append(i)
    return np.array(out), keep


def falsify(bel, D, y):
    sc = bel.Sd; var = sc.var(0, ddof=1); sf = bel.pd.fwd(y[None])[0]
    mu, Cm = b13.pilot.fast_mcd(sc); Ci = np.linalg.inv(Cm)
    return dict(t2=float(np.mean(np.sum(sc ** 2 / var, 1) < np.sum(sf ** 2 / var))),
                t2r=float(np.mean(np.einsum("ij,jk,ik->i", sc - mu, Ci, sc - mu) < (sf - mu) @ Ci @ (sf - mu))),
                q=float(np.mean(bel.pd.resid(D) < bel.pd.resid(y[None])[0])))


def stretch_sets():
    """The v1.5-2 B2 training / held-out sets (L18 v4 ensemble restricted to the L24-stretch columns) and the base BEL."""
    import stage6_analysis as s6
    s6.init(); s6.TX = b13.TX
    cols, x0 = l24_stretch_cols()
    mu_d, dev = b13.approx_stats()
    d = os.path.join(D4(), "train"); sets = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz") and "tmp" not in f:
            z = np.load(os.path.join(d, f), allow_pickle=True)
            sets.append((int(f[1:-4]), {"R_medium": z["R_medium"], "theta": z["theta"], "target": z["target"]}))
    S = {}
    for name, ss in (("tr", [x for x in sets if x[0] < ID0 + N_TR]), ("ho", [x for x in sets if x[0] >= ID0 + N_TR])):
        Dd, kp = prior_data_cols(ss, mu_d, dev, cols); zm = {i: z for i, z in ss}
        th_ = [json.loads(str(zm[i]["theta"])) for i in kp]; T_ = np.array([zm[i]["target"] for i in kp])
        m = np.array([s6.site(t, tt) and t["rho_lens"] >= s6.PHYS_RHO_LENS_MIN for t, tt in zip(th_, T_)], dtype=bool)
        S[name] = dict(D=Dd[m], T=T_[m], th=[t for t, k in zip(th_, m) if k], n_all=len(ss), n_pos=len(kp), n_kept=int(m.sum()))
    return cols, x0, S, b13.BEL(S["tr"]["D"], S["tr"]["T"])


def cmd_layout_t1(args):
    """v1.5-3: BEL v4 (B2 BELs) on T1/L18 synthetic data restricted to the L24-stretch columns."""
    g28 = _load("g28", "28_dp_synthetic.py")
    os.makedirs(DLT(), exist_ok=True); t0 = time.time()
    cols, x0, S, bel = stretch_sets()
    Ytr = np.array([g14.derived_targets(t) for t in S["tr"]["th"]])
    s15.RHO_BOUNDS.update(rho_lens=(0.4, 3.0), rho_below=RHO_BELOW_V4, rho_discharge=RHO_DIS_V4)
    b2 = s15.BEL2(S["tr"]["D"], Ytr, bounded=True)
    keep = np.asarray(wc.load_run("deep_drop_data")["keep_idx"], int)
    ti = g28.TRUTHS.index("T1") + 1
    R_true = np.load(os.path.join(g28.SRC(), "fwd_T1_L18.npz"))["R"]
    out = {}
    for real in range(g28.N_REAL):
        rng = np.random.default_rng([150000, ti, 0, real])
        R = R_true + np.sqrt((0.03 * R_true) ** 2 + 1e-4 ** 2) * rng.standard_normal(len(R_true))
        y = R[keep][cols]
        if np.any(y <= 0):
            out[f"T1sub_r{real}_skipped"] = np.array(1); continue
        y = np.log10(y)
        out[f"T1sub_r{real}_base"] = bel.posterior(y, N_POST, np.random.default_rng([SEED_BASE, 510_000 + real]), "kde").astype(np.float32)
        out[f"T1sub_r{real}_P2"] = b2.posterior(y, N_POST, np.random.default_rng([SEED_2, 510_000 + real])).astype(np.float32)
        print(f"[layout-t1] T1/subset r{real}: ρ_below median {np.median(10 ** out[f'T1sub_r{real}_P2'][:, 3]):.2f} Ω·m", flush=True)
    np.savez(os.path.join(DLT(), "t1_subset.npz"), TX=b13.TX, cols=cols, **out)
    print(f"[layout-t1] done ({time.time() - t0:.0f} s)", flush=True)


def cmd_layout(args):
    import stage6_analysis as s6
    os.makedirs(DLT(), exist_ok=True); t0 = time.time(); out = {}
    # (B1) BEL v4 (L24 ensemble) on T1/L24 synthetic data
    g28, keep, C = l24_sets()
    assert g28.sb.ERR_REL == b13.ERR_REL and g28.sb.ERR_ABS == b13.ERR_ABS
    D, T, th = C["tr"]["D"], C["tr"]["T"], C["tr"]["th"]
    bel = b13.BEL(D, T); Ytr = np.array([g14.derived_targets(t) for t in th])
    s15.RHO_BOUNDS.update(rho_lens=(0.4, 3.0), rho_below=RHO_BELOW_V4, rho_discharge=RHO_DIS_V4)
    b2 = s15.BEL2(D, Ytr, bounded=True)
    ti, li = g28.TRUTHS.index("T1") + 1, g28.LAYOUTS.index("L24")
    R_true = np.load(os.path.join(g28.SRC(), "fwd_T1_L24.npz"))["R"]; assert len(R_true) == len(keep)
    for real in range(g28.N_REAL):
        rng = np.random.default_rng([150000, ti, li, real])
        R = R_true + np.sqrt((b13.ERR_REL * R_true) ** 2 + b13.ERR_ABS ** 2) * rng.standard_normal(len(R_true))
        y = R[keep]
        if np.any(y <= 0):
            out[f"T1L24_r{real}_skipped"] = np.array(1); continue
        y = np.log10(y)
        out[f"T1L24_r{real}_base"] = bel.posterior(y, N_POST, np.random.default_rng([L24_SEED_BASE, 300_000 + 10 * ti + real]), "kde").astype(np.float32)
        out[f"T1L24_r{real}_P2"] = b2.posterior(y, N_POST, np.random.default_rng([L24_SEED_2, 300_000 + 10 * ti + real])).astype(np.float32)
        print(f"[layout] T1/L24 r{real}: ρ_below median {np.median(10 ** out[f'T1L24_r{real}_P2'][:, 3]):.2f} Ω·m", flush=True)
    # (B2) BEL v4 (L18 ensemble) on the 2018 field data restricted to the L24 stretch
    cols, x0, S, bel = stretch_sets()
    obs, _ = b13.observed_sets(); y = obs["field"][cols]
    D, T, th = S["tr"]["D"], S["tr"]["T"], S["tr"]["th"]
    Pf = bel.posterior(y, N_POST, np.random.default_rng([SEED_BASE, LAY_SEED_FIELD + 1]), "kde")
    cov = []
    for j, (yy, t) in enumerate(zip(S["ho"]["D"], S["ho"]["T"])):
        P = bel.posterior(yy, N_POST, np.random.default_rng([SEED_BASE, LAY_SEED_HO_BASE + j]), "kde"); lo, hi = np.percentile(P, [5, 95], 0)
        cov.append((t >= lo) & (t <= hi))
    fal = falsify(bel, D, y)
    Ytr = np.array([g14.derived_targets(t) for t in th]); Yho = np.array([g14.derived_targets(t) for t in S["ho"]["th"]])
    s15.RHO_BOUNDS.update(rho_lens=(0.4, 3.0), rho_below=RHO_BELOW_V4, rho_discharge=RHO_DIS_V4)
    b2 = s15.BEL2(D, Ytr, bounded=True)
    P2 = b2.posterior(y, N_POST, np.random.default_rng([SEED_2, LAY_SEED_FIELD + 3]))
    c2 = []
    for j, yy in enumerate(S["ho"]["D"]):
        P = b2.posterior(yy, N_POST, np.random.default_rng([SEED_2, LAY_SEED_HO_2 + j])); lo, hi = np.percentile(P, [5, 95], 0)
        c2.append((Yho[j] >= lo) & (Yho[j] <= hi))
    print(f"[layout] 2018 L24 stretch ({len(cols)} data): ρ_below median {np.median(10 ** P2[:, 3]):.2f} Ω·m", flush=True)
    np.savez(os.path.join(DLT(), "layout_vs_time.npz"), TX=b13.TX, TX_L24=TX_L24, cols=cols, x0=x0, sub_Pf=Pf, sub_P2=P2,
             sub_ho_cov=np.array(cov), sub_cov2=np.array(c2), sub_fal=json.dumps(fal),
             sub_n=json.dumps({k: {kk: v for kk, v in S[k].items() if kk.startswith("n_")} for k in ("tr", "ho")}),
             sub_kd=bel.pd.k, sub_kt=bel.pt.k, **out)
    print(f"[layout] done ({time.time() - t0:.0f} s)", flush=True)


def report_layout(args):
    """7_layout_vs_time.md, F_19 and layout_vs_time.json (v1.5-2 B)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = os.path.join(args.out, "results", "v1_5"); figs = os.path.join(res, "figs")
    dpp = os.path.join(args.out, "data", "processed", "v1_5")
    Z = np.load(os.path.join(DLT(), "layout_vs_time.npz"), allow_pickle=True)
    S = np.load(os.path.join(D4(), "synth.npz")); F = np.load(os.path.join(D4(), "bel_v4.npz"), allow_pickle=True)
    L = np.load(os.path.join(D4L(), "bel_v4_l24.npz"), allow_pickle=True)
    g28 = _load("g28", "28_dp_synthetic.py"); Dg = {d["name"]: d for d in g28.dp_inputs()}
    names = [d["name"] for d in sorted(g28.dp_inputs(), key=lambda d: -d["x"])]
    q = lambda v: [float(np.median(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))]
    fm = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    col = lambda TX, x: int(np.argmin(np.abs(TX - x)))

    def synth(Zs, pre, TX):
        keys = [k[:-3] for k in Zs.files if k.startswith(pre) and k.endswith("_P2")]
        meds = [float(np.median(10 ** Zs[f"{k}_P2"][:, 3])) for k in keys]
        pooled = np.concatenate([10 ** Zs[f"{k}_P2"][:, 3] for k in keys])
        cov = {n: float(np.mean([np.percentile(Zs[f"{k}_base"][:, col(TX, Dg[n]['x'])], 5) <= Dg[n]["base"]
                                 <= np.percentile(Zs[f"{k}_base"][:, col(TX, Dg[n]['x'])], 95) for k in keys])) for n in names}
        bm = {n: float(np.median([np.median(Zs[f"{k}_base"][:, col(TX, Dg[n]['x'])]) for k in keys])) for n in names}
        return dict(fresh(meds, pooled), n=len(keys), skipped=sum(1 for k in Zs.files if k.startswith(pre) and k.endswith("skipped")),
                    per_real=meds, cov=cov, base_med=bm), pooled

    def field(P2, Pf, TX):
        rb = 10 ** P2[:, 3]
        base = {n: dict(bel=q(Pf[:, col(TX, Dg[n]["x"])]), col=float(TX[col(TX, Dg[n]["x"])])) for n in names}
        for n in names:
            base[n]["inside"] = bool(base[n]["bel"][1] <= Dg[n]["base"] <= base[n]["bel"][2])
        return dict(fresh([np.median(rb)], rb), base=base), rb

    o = {}
    o["T1_L18"], pT18 = synth(S, "T1_r", S["TX"])
    o["T1_L24"], pT24 = synth(Z, "T1L24_r", Z["TX_L24"])
    o["F18_full"], r_full = field(F["P2"], F["Pf"], F["TX"])
    o["F18_full"].update(cov=float(np.mean(F["ho_cov"])), cov2=float(np.mean(F["cov2"][:, 3])), fal=json.loads(str(F["fal"])))
    o["F18_sub"], r_sub = field(Z["sub_P2"], Z["sub_Pf"], Z["TX"])
    o["F18_sub"].update(cov=float(np.mean(Z["sub_ho_cov"])), cov2=float(np.mean(Z["sub_cov2"][:, 3])), fal=json.loads(str(Z["sub_fal"])),
                        n=json.loads(str(Z["sub_n"])), n_data=int(len(Z["cols"])), kd=int(Z["sub_kd"]), kt=int(Z["sub_kt"]))
    o["ERT"], r_ert = field(L["P2"], L["Pf"], L["TX"])
    r = o["T1_L24"]["median"] / o["T1_L18"]["median"]; m = o["F18_sub"]["median"]
    rules = dict(i=bool(1 / 1.5 <= r <= 1.5 and m < 8.0), ii=bool(m >= 8.0), iii=bool(r > 1.5))
    txt = dict(i="the 2018–2024 difference is in the data (change since 2018, or something specific to the 2018 data), not the layout",
               ii="the dune part / layout of the 2018 survey drove the low 2018 value (layout or model-class effect, not time)",
               iii="the L24 layout tends to read ρ_below high — a caveat on the same-epoch value")
    hit = [k for k in ("i", "ii", "iii") if rules[k]]
    readout = "; ".join(f"({k}) {txt[k]}" for k in hit) if hit else "none of the logged outcomes applies (r < 1/1.5 and m < 8 Ω·m)"
    o.update(ratio=r, rules=rules, readout=readout, x0=float(Z["x0"]))
    M = ["# 7 — Layout vs time for ρ_below (v1.5-2 B)", "", f"**{EXPLORATORY}** Decisions and read-out logged before running "
         "(`docs/decision_log.md` v1.5-2). No new prior ensemble: (B1) the L24 BEL v4 (v1.5-1) on T1/L24 synthetic data; (B2) the "
         f"L18 BEL v4 retrained on the 2018 G1 quadrupoles with all four electrodes at x ≥ {o['x0']:.2f} m (the L24 line start).", "",
         "2025 ERT line (ERT.dat): April 2025, about seven months after the CPTs (E. Vrancken, pers. comm.); the CPTs / DP logs date from 23 September 2024.", "", "**Note (v1.7-1):** this test was pre-registered and evaluated assuming a near-simultaneous survey; the 2025 ERT line (ERT.dat) was measured in April 2025, about seven months after the CPTs of 23 September 2024 (E. Vrancken, pers. comm.). The gap is reported as a limitation; no verdict changes.", "",
         "## ρ_below", "",
         "| data | BEL (ensemble) | data used | ρ_below: median (of posterior medians) | 90 % interval (pooled) | per realisation | fresh (≥ 8 and interval ⊇ 10.89–14.80)? |",
         "|---|---|---|---|---|---|---|"]
    t18, t24, fu, su, er = o["T1_L18"], o["T1_L24"], o["F18_full"], o["F18_sub"], o["ERT"]
    yn = lambda b: "**yes**" if b else "**no**"
    M += [f"| T1/L18 (v1.5-1) | L18 v4 | 1974 G1 | {t18['median']:.2f} | [{t18['p5']:.2f}, {t18['p95']:.2f}] | "
          + ", ".join(f"{v:.2f}" for v in t18["per_real"]) + f" | {yn(t18['fresh'])} |",
          f"| T1/L24 (B1) | L24 v4 | 1643 Stage-1 kept | {t24['median']:.2f} | [{t24['p5']:.2f}, {t24['p95']:.2f}] | "
          + ", ".join(f"{v:.2f}" for v in t24["per_real"]) + f" | {yn(t24['fresh'])} |",
          f"| 2018 field, full line (v1.5 Task 3) | L18 v4 | 1974 G1 | {fu['median']:.2f} | [{fu['p5']:.2f}, {fu['p95']:.2f}] | — | {yn(fu['fresh'])} |",
          f"| 2018 field, L24 stretch (B2) | L18 v4, retrained on the subset | {su['n_data']} of 1974 G1 | {su['median']:.2f} | "
          f"[{su['p5']:.2f}, {su['p95']:.2f}] | — | {yn(su['fresh'])} |",
          f"| 2025 ERT line (ERT.dat; v1.5-1) | L24 v4 | 1643 Stage-1 kept | {er['median']:.2f} | [{er['p5']:.2f}, {er['p95']:.2f}] | — | {yn(er['fresh'])} |", "",
          f"Ratio r = T1/L24 ÷ T1/L18 = **{r:.2f}**; 2018 L24-stretch median m = **{m:.2f} Ω·m**.", "",
          "## Checks on the 2018 L24-stretch BEL (B2)", "",
          f"- Training {su['n']['tr']['n_kept']} (of {su['n']['tr']['n_all']}; {su['n']['tr']['n_pos']} positive on the subset), held-out "
          f"{su['n']['ho']['n_kept']}; PCA components: data {su['kd']}, target {su['kt']}.",
          f"- Prior falsification (share of training samples below the field value): T² {100*su['fal']['t2']:.1f} %, robust T² "
          f"{100*su['fal']['t2r']:.1f} %, Q {100*su['fal']['q']:.1f} % (full line: {100*fu['fal']['t2']:.1f} / {100*fu['fal']['t2r']:.1f} / "
          f"{100*fu['fal']['q']:.1f} %).",
          f"- Held-out 90 % coverage: lens base {100*su['cov']:.1f} %, ρ_below {100*su['cov2']:.1f} % (full line: {100*fu['cov']:.1f} %, "
          f"{100*fu['cov2']:.1f} %).", "",
          "## Lens base at the DP x positions", "",
          "| DP | x | true / DP base | T1/L18: coverage / median | T1/L24: coverage / median | 2018 full: median [P5, P95] | 2018 L24 stretch: median [P5, P95] | inside (full / stretch) |",
          "|---|---|---|---|---|---|---|---|"]
    for n in names:
        M.append(f"| {n} | {Dg[n]['x']:.1f} | {Dg[n]['base']:.2f} | {100*t18['cov'][n]:.0f} % / {t18['base_med'][n]:.2f} | "
                 f"{100*t24['cov'][n]:.0f} % / {t24['base_med'][n]:.2f} | {fm(fu['base'][n]['bel'])} | {fm(su['base'][n]['bel'])} | "
                 f"{'yes' if fu['base'][n]['inside'] else 'no'} / {'yes' if su['base'][n]['inside'] else 'no'} |")
    nin = lambda b: sum(b["base"][n]["inside"] for n in names if n != "DP1")
    o.update(base_in_full=nin(fu), base_in_sub=nin(su))
    M += ["", f"DP2–DP7 bases inside the 90 % intervals: 2018 full {nin(fu)} of 6, 2018 L24 stretch {nin(su)} of 6.", "",
          "## Read-out (logged before running)", "",
          f"(i) 1/1.5 ≤ r ≤ 1.5 and m < 8 Ω·m: {'yes' if rules['i'] else 'no'}; (ii) m ≥ 8 Ω·m: {'yes' if rules['ii'] else 'no'}; "
          f"(iii) r > 1.5: {'yes' if rules['iii'] else 'no'}.", "", f"**{readout}.**", "",
          "## 2018 survey timing (v1.5-2 A2)", "",
          "Measurement started at low tide (T. Hermans, pers. comm.). Estimated morning low water at Nieuwpoort on 7 March 2018 ≈ "
          "10:55 CET (09:55 UTC) ± about 30 min, from the official 2017 Nieuwpoort tide table at the same lunar phase (17–18 March "
          "2017, 5 days after full moon), not from the 2018 table. The first instrument record is at 11:22–11:24, so the clock was "
          "most likely on local time (CET): the survey ran from about low water (11:24 CET) to 13:41 CET on a rising tide. "
          "**Estimated; to be confirmed with the 2018 tide table or the Nieuwpoort gauge record.**"]
    pT1s = None
    if os.path.exists(os.path.join(DLT(), "t1_subset.npz")):            # v1.5-3
        Z3 = np.load(os.path.join(DLT(), "t1_subset.npz"))
        o["T1_sub"], pT1s = synth(Z3, "T1sub_r", Z3["TX"]); ts_ = o["T1_sub"]; t, mf = ts_["median"], su["median"]
        rr = dict(a=bool(t / 1.5 <= mf <= 1.5 * t), b=bool(t > 1.5 * mf), c=bool(t < mf / 1.5))
        tx3 = dict(a="the 2018 data on this stretch are consistent with the 2024 state: no evidence of change in the water below the lens",
                   b="on this stretch the 2018 data read more conductive than the 2024 state would: change since 2018 or model error remain possible",
                   c="the 2018 data read fresher than the 2024 state")
        ro3 = "; ".join(f"({k}) {tx3[k]}" for k in ("a", "b", "c") if rr[k])
        o.update(ratio_sub=mf / t, rules_sub=rr, readout_sub=ro3)
        M += ["", "## v1.5-3: T1 on the same 1141-quadrupole subset (logged before running)", "",
              "BEL's ρ_below is a single profile-wide value, so every ρ_below here is a layout-weighted effective value, not the "
              "resistivity at a given place.", "",
              f"The B2 BELs (retrained on the subset, same approximation-error subset) on T1/L18 realisations 0–4 restricted to the same "
              f"{su['n_data']} quadrupoles (skipped: {ts_['skipped']}).", "",
              "| data | ρ_below: median (of posterior medians) | 90 % interval (pooled) | per realisation |", "|---|---|---|---|",
              f"| T1, full 2018 layout (v1.5-1) | {t18['median']:.2f} | [{t18['p5']:.2f}, {t18['p95']:.2f}] | " + ", ".join(f"{v:.2f}" for v in t18["per_real"]) + " |",
              f"| **T1, L24-stretch subset (v1.5-3)** | **{t:.2f}** | [{ts_['p5']:.2f}, {ts_['p95']:.2f}] | " + ", ".join(f"{v:.2f}" for v in ts_["per_real"]) + " |",
              f"| 2018 field, L24-stretch subset (B2) | {mf:.2f} | [{su['p5']:.2f}, {su['p95']:.2f}] | — |", "",
              f"Field ÷ T1 on the subset = {mf:.2f} / {t:.2f} = **{mf / t:.2f}**; subsetting changes T1 by a factor {t / t18['median']:.2f}.", "",
              "Base coverage of the T1/subset BEL (share of realisations whose 90 % interval contains the true base; median of the posterior medians):", "",
              "| DP | x | true base | coverage / median |", "|---|---|---|---|"]
        for n in names:
            if n != "DP1":
                M.append(f"| {n} | {Dg[n]['x']:.1f} | {Dg[n]['base']:.2f} | {100*ts_['cov'][n]:.0f} % / {ts_['base_med'][n]:.2f} |")
        o["T1_sub_base_in"] = int(sum(ts_["cov"][n] == 1.0 for n in names if n != "DP1"))
        M += ["", f"(a) t/1.5 ≤ m ≤ 1.5 t: {'yes' if rr['a'] else 'no'}; (b) t > 1.5 m: {'yes' if rr['b'] else 'no'}; "
              f"(c) t < m/1.5: {'yes' if rr['c'] else 'no'} (t = {t:.2f}, m = {mf:.2f} Ω·m).", "", f"**{ro3}.**"]
    with open(os.path.join(res, "7_layout_vs_time.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(M) + "\n")
    json.dump(o, open(os.path.join(dpp, "layout_vs_time.json"), "w", encoding="utf-8"), indent=1)
    # F_19
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axes[0]; bins = np.logspace(np.log10(1.5), np.log10(120), 40)
    ax.hist(pT18, bins, density=True, alpha=0.40, color="#1f4e9c", label="T1/L18 (synthetic)")
    ax.hist(pT24, bins, density=True, alpha=0.40, color="#7fa6e0", label="T1/L24 (synthetic)")
    if pT1s is not None:
        ax.hist(pT1s, bins, density=True, histtype="step", lw=1.2, ls="-.", color="#1f4e9c", label="T1, L24-stretch subset (synthetic)")
    ax.hist(r_full, bins, density=True, histtype="step", lw=1.2, color="#d35400", label="2018, full line")
    ax.hist(r_sub, bins, density=True, histtype="step", lw=1.2, ls="--", color="#d35400", label="2018, L24 stretch")
    ax.hist(r_ert, bins, density=True, histtype="step", lw=1.3, color="k", label="2025 ERT line")
    for v in (10.89, 13.57, 14.80):
        ax.axvline(v, color="#1f4e9c", ls=":", lw=0.7)
    ax.axvline(8.0, color="0.5", ls="-", lw=0.5)
    ax.set_xscale("log"); ax.set_xlabel("ρ_below (Ω·m)"); ax.set_title("(a) ρ_below posteriors (BEL v4)", fontsize=7); ax.legend(fontsize=5.6)
    ax = axes[1]
    for P, TX, c, lab, a in ((F["Pf"], F["TX"], "#d35400", "2018, full line", 0.15), (Z["sub_Pf"], Z["TX"], "#1b9e77", "2018, L24 stretch", 0.25)):
        lo, md, hi = np.percentile(P, [5, 50, 95], 0)
        ax.fill_between(TX, lo, hi, color=c, alpha=a, lw=0); ax.plot(TX, md, color=c, lw=1.3, label=lab)
    ax.plot(L["TX"], np.median(L["Pf"], 0), color="k", lw=1.0, ls="--", label="2025 ERT line, median")
    ax.axvline(o["x0"], color="0.5", lw=0.6, ls=":"); ax.text(o["x0"] + 2, 3.5, "L24 start", fontsize=6, color="0.4")
    ax.plot([Dg[n]["x"] for n in names], [Dg[n]["base"] for n in names], "kD", ms=4, label="DP base (2024)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("Elevation (mTAW)"); ax.set_ylim(-25, 5); ax.legend(fontsize=5.6, loc="lower right")
    ax.set_title("(b) lens base, 90 % intervals", fontsize=7)
    fig.suptitle(EXPLORATORY, fontsize=7); fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(figs, "F_19_layout_vs_time.png"), dpi=300)
    wc.save_publication(fig, args.out, "F_19_layout_vs_time"); plt.close(fig)
    print("\n".join(M))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "resume"):
        s = sub.add_parser(c); s.add_argument("--out", default=wc.ROOT); s.add_argument("--workers", type=int, default=6)
        s.add_argument("--time-only", action="store_true")
    sub.add_parser("report").add_argument("--out", default=wc.ROOT)
    sub.add_parser("synth").add_argument("--out", default=wc.ROOT)
    sub.add_parser("layout").add_argument("--out", default=wc.ROOT)
    sub.add_parser("layout-t1").add_argument("--out", default=wc.ROOT)
    l = sub.add_parser("l24"); l.add_argument("--out", default=wc.ROOT); l.add_argument("--workers", type=int, default=6)
    l.add_argument("--time-only", action="store_true")
    w = sub.add_parser("worker"); w.add_argument("--worker-id", type=int, required=True); w.add_argument("--dual", action="store_true")
    w.add_argument("--l24", action="store_true"); w.add_argument("--ids", type=int, nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "worker" and a.l24:
        l24_worker(a); return
    dict(run=cmd_run, resume=cmd_run, report=cmd_report, worker=cmd_worker, synth=cmd_synth, l24=cmd_l24, layout=cmd_layout, **{"layout-t1": cmd_layout_t1})[a.cmd](a)


if __name__ == "__main__":
    main()
