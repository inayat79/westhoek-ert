"""
31_scores.py
============
Westhoek ERT v1.6 — proper scoring rules for the lens base (interval score, CRPS) for every method, at field scale (DP2–DP7)
and on the synthetic sets, with the prior predictive as baseline. EXPLORATORY: scores chosen after the DP results were known;
metric definitions logged before computing (`docs/decision_log.md` v1.6-0). No new inversions, forward runs, data or seeds.

  run    [--out ROOT]   per-case scores → ROOT/data/processed/v1_6/cases.csv, regen_check.json (regenerates the unstored BEL
                        posteriors of the held-out sets and the v1.0 truths from the stored fits and seeds; must reproduce the
                        stored summaries exactly)
  report [--out ROOT]   ROOT/results/v1_6/1_scores.md, SUMMARY.md, T_S12_scores.{csv,md}, figs/F_20_scores.png,
                        figs/F_S13_synthetic_scores.png; ROOT/data/processed/v1_6/scores.json

Seeds: CRPS subsamples default_rng([160000, set, case]) (prior predictive: case = column); field bootstrap
default_rng([160100, 0]) (10 000 resamples over DP2–DP7, shared by all methods).
"""

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import read_trace  # noqa: E402
read_trace.install("31_scores.py")
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


ALPHA = 0.1
N_CAP = 2000
N_BOOT = 10_000
TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)
HEADER = ("**Exploratory: scores chosen after the DP results were known; metric definitions logged before computing** "
          "(`docs/decision_log.md` v1.6-0).")
SETS = {"field": 0, "ho_v3": 1, "ho_v4": 2, "ho_v4L24": 3, "v10_bel": 4, "v10_lin": 5, "v13_lin": 6, "v15_lin": 7, "v15_bel": 8,
        "prior": 9}
FIELDS = ["set", "method", "prior", "case", "truth", "real", "dp", "col", "x", "y", "l", "m", "u", "is", "crps", "approx", "r",
          "truth_kind"]


def D6(root=None):
    return os.path.join(root or wc.ROOT, "data", "processed", "v1_6")


# ─────────────────────────────────────────────────────────────────────────────
# scores
# ─────────────────────────────────────────────────────────────────────────────
def interval_score(l, u, y):
    return (u - l) + (2 / ALPHA) * (l - y) * (y < l) + (2 / ALPHA) * (y - u) * (y > u)


def cap(x, seed):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) > N_CAP:
        x = np.random.default_rng(seed).choice(x, N_CAP, replace=False)
    return np.sort(x)


def crps_sorted(xs, y):
    """Plug-in CRPS of the empirical distribution of the sorted sample xs at y (vectorised over y)."""
    n = len(xs); y = np.atleast_1d(np.asarray(y, float))
    i = np.arange(1, n + 1)
    half_spread = np.sum((2 * i - n - 1) * xs) / n ** 2           # ½ E|X − X′|
    cs = np.concatenate([[0.0], np.cumsum(xs)])
    k = np.searchsorted(xs, y, side="right")
    e_abs = (y * k - cs[k] + (cs[-1] - cs[k]) - y * (n - k)) / n
    return e_abs - half_spread


def crps_quantiles(p5, p50, p95, y):
    q = np.interp(TAUS, [0.05, 0.5, 0.95], [p5, p50, p95])
    return float(2 * np.mean(((y < q).astype(float) - TAUS) * (q - y)))


def from_samples(x, y, seed):
    xs = np.asarray(x, float); xs = xs[np.isfinite(xs)]
    l, m, u = np.percentile(xs, [5, 50, 95])
    return dict(l=l, m=m, u=u, is_=float(interval_score(l, u, y)), crps=float(crps_sorted(cap(xs, seed), y)[0]), approx=0)


def from_band(b, y):
    """Stored band (share defined, P50, P5, P95); None when the interface is defined in < 50 % of samples."""
    if not (np.isfinite(b[0]) and b[0] >= 0.5 and np.all(np.isfinite(b[1:]))):
        return None
    return dict(l=float(b[2]), m=float(b[1]), u=float(b[3]), is_=float(interval_score(b[2], b[3], y)),
                crps=crps_quantiles(b[2], b[1], b[3], y), approx=1)


class Prior:
    """Prior predictive of the base per column: P5/P50/P95 from all samples, CRPS from ≤ 2000 (subsample per column)."""

    def __init__(self, name, T, TX):
        self.name, self.T, self.TX = name, np.asarray(T, float), np.asarray(TX, float)
        self.q = np.nanpercentile(self.T, [5, 50, 95], 0)
        self.xs = [cap(self.T[:, j], [160000, SETS["prior"], 100 * (hash_name(name)) + j]) for j in range(self.T.shape[1])]

    def col(self, x):
        return int(np.argmin(np.abs(self.TX - x)))

    def score(self, j, y):
        l, m, u = self.q[:, j]
        return dict(l=float(l), m=float(m), u=float(u), is_=float(interval_score(l, u, y)), crps=float(crps_sorted(self.xs[j], y)[0]),
                    approx=0)


def hash_name(name):
    return {"PRIOR3": 1, "PRIOR4": 2, "PRIOR4-L24": 3, "PRIOR4-s": 4}[name]


class Rows:
    def __init__(self):
        self.rows = []

    def add(self, set_, method, prior, case, sc, y, **kw):
        if sc is None:
            return
        self.rows.append(dict(set=set_, method=method, prior=prior, case=case, y=float(y), l=sc["l"], m=sc["m"], u=sc["u"],
                              **{"is": sc["is_"], "crps": sc["crps"], "approx": sc["approx"]}, **kw))


# ─────────────────────────────────────────────────────────────────────────────
# run
# ─────────────────────────────────────────────────────────────────────────────
def dp_list():
    R = json.load(open(os.path.join(wc.ROOT, "data", "processed", "v1_4", "dp_results.json"), encoding="utf-8"))
    names = sorted(R["dps"], key=lambda n: R["dps"][n]["index"])
    return [dict(dp=n, **{k: R["dps"][n]["per"]["primary|A"][k] for k in ("x", "logmid", "thr3", "rho_below")}) for n in names]


def cmd_run(args):
    t0 = time.time()
    os.makedirs(D6(args.out), exist_ok=True)
    g30 = _load("g30", "30_bel_v4.py"); b13, g14, s15 = g30.b13, g30.g14, g30.s15
    import stage6_analysis as s6
    c11, sb = s15.c11, s15.sb
    s18 = _load("p18", "18_paper.py")
    TX, TXL = b13.TX, g30.TX_L24
    rows = Rows(); check = {}
    eq = lambda a, b: bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))

    # ── BEL v3 (v1.0 headline fit): held-out and v1.0 truths, regenerated ──
    s6.init(); s6.TX = TX
    C3 = s6.load_combined(physical=True)
    bel3 = b13.BEL(C3["tr"]["D"], C3["tr"]["T"])
    P3 = Prior("PRIOR3", C3["tr"]["T"], TX)
    B3 = np.load(os.path.join(wc.ROOT, "data", "processed", "paper", "bel_products.npz"), allow_pickle=True)
    cov, rank, wid, bias = [], [], [], []
    for j, (y, t) in enumerate(zip(C3["ho"]["D"], C3["ho"]["T"])):
        P = bel3.posterior(y, 2000, np.random.default_rng([129000, 90_000 + j]), "kde")
        lo, md, hi = np.percentile(P, [5, 50, 95], 0)
        cov.append((t >= lo) & (t <= hi)); rank.append(np.mean(P < t, 0)); wid.append(hi - lo); bias.append(md - t)
        for k in range(len(TX)):
            if np.isfinite(t[k]):
                rows.add("ho_v3", "BEL3", "PRIOR3", j, from_samples(P[:, k], t[k], [160000, SETS["ho_v3"], j * 100 + k]), t[k],
                         col=TX[k], x=TX[k], truth_kind="target")
                rows.add("ho_v3", "PRIOR3", "PRIOR3", j, P3.score(k, t[k]), t[k], col=TX[k], x=TX[k], truth_kind="target")
    check["v3_heldout"] = dict(cov=eq(cov, B3["ho_cov"]), rank=eq(rank, B3["ho_rank"]), wid=eq(wid, B3["ho_wid"]), bias=eq(bias, B3["ho_bias"]),
                               n=len(cov))
    print(f"[31] v3 held-out regenerated: {check['v3_heldout']} ({time.time()-t0:.0f} s)", flush=True)
    obs, truths = b13.observed_sets(); spec4 = sb.load_spec(); cols = TX >= 130; ts = B3["truth_stats"].item(); tchk = {}
    for ti, t in enumerate(("T1", "T1h", "T2a", "T3")):
        tv = c11.truth_defs(spec4, t, TX)["logmid"]; cv, bi, wd = [], [], []
        for r in range(20):
            y = truths[(t, r)]
            if y is None:
                continue
            P = bel3.posterior(y, 2000, np.random.default_rng([s18.SEED, 20_000 + 100 * ti + r]), "kde")
            lo, md, hi = np.percentile(P, [5, 50, 95], 0)
            cv.append(((tv >= lo) & (tv <= hi))[cols]); bi.append((md - tv)[cols]); wd.append((hi - lo)[cols])
            for k in np.where(cols & np.isfinite(tv))[0]:
                rows.add("v10_bel", "BEL3", "PRIOR3", f"{t}_r{r}", from_samples(P[:, k], tv[k], [160000, SETS["v10_bel"], 10000 * ti + 100 * r + k]),
                         tv[k], truth=t, real=r, col=TX[k], x=TX[k], truth_kind="logmid")
                rows.add("v10_bel", "PRIOR3", "PRIOR3", f"{t}_r{r}", P3.score(k, tv[k]), tv[k], truth=t, real=r, col=TX[k], x=TX[k],
                         truth_kind="logmid")
        new = dict(n=len(cv), cov=float(np.mean(cv)), bias=float(np.median(bi)), width=float(np.median(wd)))
        tchk[t] = all(new[k] == ts[t][k] for k in new)
    check["v3_truths"] = tchk
    print(f"[31] v3 truths regenerated: {tchk} ({time.time()-t0:.0f} s)", flush=True)
    del C3

    # ── BEL v4 (L18): held-out regenerated ──
    s6.init(); s6.TX = TX
    C4 = g30.load_v4(); bel4 = b13.BEL(C4["tr"]["D"], C4["tr"]["T"]); P4 = Prior("PRIOR4", C4["tr"]["T"], TX)
    Z4 = np.load(os.path.join(g30.D4(), "bel_v4.npz"), allow_pickle=True); cov, wid = [], []
    for j, (y, t) in enumerate(zip(C4["ho"]["D"], C4["ho"]["T"])):
        P = bel4.posterior(y, g30.N_POST, np.random.default_rng([g30.SEED_BASE, 90_000 + j]), "kde"); lo, hi = np.percentile(P, [5, 95], 0)
        cov.append((t >= lo) & (t <= hi)); wid.append(hi - lo)
        for k in range(len(TX)):
            if np.isfinite(t[k]):
                rows.add("ho_v4", "BEL4", "PRIOR4", j, from_samples(P[:, k], t[k], [160000, SETS["ho_v4"], j * 100 + k]), t[k],
                         col=TX[k], x=TX[k], truth_kind="target")
                rows.add("ho_v4", "PRIOR4", "PRIOR4", j, P4.score(k, t[k]), t[k], col=TX[k], x=TX[k], truth_kind="target")
    check["v4_heldout"] = dict(cov=eq(cov, Z4["ho_cov"]), wid=eq(wid, Z4["ho_wid"]), n=len(cov))
    print(f"[31] v4 held-out regenerated: {check['v4_heldout']} ({time.time()-t0:.0f} s)", flush=True)
    del C4

    # ── BEL v4 (L24): held-out regenerated ──
    g28, _, CL = g30.l24_sets(); belL = b13.BEL(CL["tr"]["D"], CL["tr"]["T"]); PL = Prior("PRIOR4-L24", CL["tr"]["T"], TXL)
    ZL = np.load(os.path.join(g30.D4L(), "bel_v4_l24.npz"), allow_pickle=True); cov = []
    for j, (y, t) in enumerate(zip(CL["ho"]["D"], CL["ho"]["T"])):
        P = belL.posterior(y, g30.N_POST, np.random.default_rng([g30.L24_SEED_BASE, 90_000 + j]), "kde"); lo, hi = np.percentile(P, [5, 95], 0)
        cov.append((t >= lo) & (t <= hi))
        for k in range(len(TXL)):
            if np.isfinite(t[k]):
                rows.add("ho_v4L24", "BEL4-L24", "PRIOR4-L24", j, from_samples(P[:, k], t[k], [160000, SETS["ho_v4L24"], j * 100 + k]), t[k],
                         col=TXL[k], x=TXL[k], truth_kind="target")
                rows.add("ho_v4L24", "PRIOR4-L24", "PRIOR4-L24", j, PL.score(k, t[k]), t[k], col=TXL[k], x=TXL[k], truth_kind="target")
    check["v4L24_heldout"] = dict(cov=eq(cov, ZL["ho_cov"]), n=len(cov))
    print(f"[31] v4 L24 held-out regenerated: {check['v4L24_heldout']} ({time.time()-t0:.0f} s)", flush=True)
    del CL
    _, _, SS, _ = g30.stretch_sets(); PS = Prior("PRIOR4-s", SS["tr"]["T"], TX)
    Ys = np.array([g14.derived_targets(t) for t in SS["tr"]["th"]])[:, 3]; del SS

    ok = all(bool(vv) for v in check.values() for kk, vv in v.items() if kk != "n")
    check["all_identical"] = bool(ok)
    json.dump(check, open(os.path.join(D6(args.out), "regen_check.json"), "w", encoding="utf-8"), indent=1)
    if not ok:
        raise SystemExit(f"regenerated BEL posteriors do not reproduce the stored summaries: {check} — stopping (decision log v1.6-0)")

    # ── field ──
    DPs = dp_list()
    Pf3 = B3["Pf"]; Pf4 = Z4["Pf"]; PfL = ZL["Pf"]; Zs = np.load(os.path.join(g30.DLT(), "layout_vs_time.npz"), allow_pickle=True)
    Pfs = Zs["sub_Pf"]
    V4 = np.load(s18.LIN_V4); SD = np.load(os.path.join(wc.ROOT, "data", "processed", "v1_4", "samedate", "cpt.npz"), allow_pickle=True)
    for i, d in enumerate(DPs):
        for tk, y in (("logmid", d["logmid"]), ("thr3", d["thr3"])):
            if y is None or not np.isfinite(y):
                continue
            kw = dict(dp=d["dp"], x=d["x"], truth_kind=tk)
            if tk == "logmid":
                for meth, Pf, TXm, pr in (("BEL3-18", Pf3, TX, P3), ("BEL4-18", Pf4, TX, P4), ("BEL4-18s", Pfs, TX, PS), ("BEL4-24", PfL, TXL, PL)):
                    j = int(np.argmin(np.abs(TXm - d["x"])))
                    rows.add("field", meth, pr.name, i, from_samples(Pf[:, j], y, [160000, SETS["field"], 100 * i + j]), y, col=TXm[j], **kw)
                for pr in (P3, P4, PS, PL):
                    j = pr.col(d["x"]); rows.add("field", pr.name, pr.name, i, pr.score(j, y), y, col=pr.TX[j], **kw)
            k = int(np.argmin(np.abs(V4["x"] - d["x"]))); ks = int(np.argmin(np.abs(SD["xcol"] - d["x"])))
            lin = (("LIN-LM-18", V4["pooled_logmid"][:, k], V4["x"][k], P4), ("LIN-3-18", V4["single_thr3"][:, k], V4["x"][k], P4),
                   ("LIN-MG-18", V4["pooled_maxgrad"][:, k], V4["x"][k], P4),
                   ("LIN-LM-24", SD["pooled_band_logmid"][:, ks], SD["xcol"][ks], PL), ("LIN-3-24", SD["lam50_band_thr3"][:, ks], SD["xcol"][ks], PL),
                   ("LIN-MG-24", SD["pooled_band_maxgrad"][:, ks], SD["xcol"][ks], PL))
            for meth, b, xc, pr in lin:
                if tk == "thr3" and "-3-" not in meth:
                    continue
                sc = from_band(b, y)
                if sc is None:
                    rows.rows.append(dict(set="field", method=meth, prior=pr.name, case=i, y=y, l=np.nan, m=np.nan, u=np.nan,
                                          **{"is": np.nan, "crps": np.nan, "approx": 1}, col=float(xc), **kw))
                else:
                    rows.add("field", meth, pr.name, i, sc, y, col=float(xc), **kw)
            if tk == "thr3":
                for pr in (P4, PL):
                    j = pr.col(d["x"]); rows.add("field", pr.name, pr.name, i, pr.score(j, y), y, col=pr.TX[j], **kw)
    print(f"[31] field done ({time.time()-t0:.0f} s)", flush=True)

    # ── v1.0 truths, linearised (A2 runs) ──
    cols10 = sb.COLS
    for ti, t in enumerate(c11.A2_TRUTHS):
        tv = c11.truth_defs(spec4, t, cols10)
        for r in range(sb.N_REAL):
            p = os.path.join(c11.OUT_DATA, "a2_runs", f"{t}_r{r:02d}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=True)
            for dname in ("thr3", "maxgrad", "logmid"):
                rr = z[f"lam50_r_{dname}"]
                for j in range(len(cols10)):
                    y = tv[dname][j]
                    if not (np.isfinite(y) and np.isfinite(rr[j]) and rr[j] <= 10):
                        continue
                    kw = dict(truth=t, real=r, col=cols10[j], x=cols10[j], r=float(rr[j]), truth_kind=dname)
                    for tr in ("lam50", "pooled"):
                        key = f"{tr}_band_{dname}"
                        if key in z.files:
                            rows.add("v10_lin", f"LIN-{dname}-{tr}", "PRIOR3", f"{t}_r{r}", from_band(z[key][:, j], y), y, **kw)
                    rows.add("v10_lin", f"PRIOR3|{dname}", "PRIOR3", f"{t}_r{r}", P3.score(P3.col(cols10[j]), y), y, **kw)
    print(f"[31] v1.0 linearised done ({time.time()-t0:.0f} s)", flush=True)

    # ── v1.3 factorial (posterior depth samples) ──
    g25 = _load("g25", "25_generalisation.py"); d13 = g25.dirs(wc.ROOT)[0]
    for ti, name in enumerate(g25.TRUTHS):
        zb = g25.params(name)["z_b"]
        for r in range(g25.N_REAL):
            p = os.path.join(d13, name, f"r{r}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=True); rt = z["lam50_r_true"]; cl = z["cols"]
            for dname in ("thr3", "maxgrad", "logmid"):
                for j in range(len(cl)):
                    if not (np.isfinite(rt[j]) and rt[j] <= 10):
                        continue
                    kw = dict(truth=name, real=r, col=cl[j], x=cl[j], r=float(rt[j]), truth_kind=dname)
                    seed = [160000, SETS["v13_lin"], ti * 10 ** 6 + r * 10 ** 4 + ("thr3", "maxgrad", "logmid").index(dname) * 1000 + j]
                    b = z[f"lam50_band_{dname}"][:, j]
                    if b[0] >= 0.5:
                        rows.add("v13_lin", f"LIN-{dname}-lam50", "PRIOR3", f"{name}_r{r}", from_samples(z[f"lam50_depths_{dname}"][j], zb, seed), zb, **kw)
                    if r < g25.N_REAL_LAM and z[f"pooled_band_{dname}"][0, j] >= 0.5:
                        S_ = np.concatenate([z[f"lam{int(l)}_depths_{dname}"][j] for l in g25.LAMS_ALL])
                        rows.add("v13_lin", f"LIN-{dname}-pooled", "PRIOR3", f"{name}_r{r}", from_samples(S_, zb, seed + [1]), zb, **kw)
                    rows.add("v13_lin", f"PRIOR3|{dname}", "PRIOR3", f"{name}_r{r}", P3.score(P3.col(cl[j]), zb), zb, **kw)
    print(f"[31] v1.3 factorial done ({time.time()-t0:.0f} s)", flush=True)

    # ── v1.5 DP-based truths: linearised (bands) and BEL v4 (stored samples) ──
    Dg = {d["name"]: d for d in g28.dp_inputs()}; xdp = g28.dp_x(); names = [d["name"] for d in g28.dp_inputs()]
    S5 = np.load(os.path.join(g30.D4(), "synth.npz")); T1s = np.load(os.path.join(g30.DLT(), "t1_subset.npz"))
    belsets = {("T1", "L18"): (S5, "T1_r", TX, P4), ("T3", "L18"): (S5, "T3_r", TX, P4), ("T1", "L24"): (Zs, "T1L24_r", TXL, PL),
               ("T1", "L18s"): (T1s, "T1sub_r", TX, PS)}
    bel_id = {k: i for i, k in enumerate(belsets)}
    for t in g28.TRUTHS:
        for lay in g28.LAYOUTS:
            for r in range(g28.N_REAL):
                z = np.load(os.path.join(g28.SRC(), f"{t}_{lay}_r{r}.npz"), allow_pickle=True); rl = z["lam50_r_logmid"]
                for j, n in enumerate(names):
                    if n == "DP1" or not (np.isfinite(rl[j]) and rl[j] <= 10):
                        continue
                    y = Dg[n]["base"]; kw = dict(truth=f"{t}/{lay}", real=r, dp=n, col=float(xdp[j]), x=float(xdp[j]), r=float(rl[j]))
                    for dname in ("logmid", "thr3"):
                        rows.add("v15_lin", f"LIN-{dname}-lam50", "PRIOR3", f"{t}_{lay}_r{r}", from_band(z[f"lam50_band_{dname}"][:, j], y), y,
                                 truth_kind=dname, **kw)
                        if r < g28.N_REAL_LAM:
                            rows.add("v15_lin", f"LIN-{dname}-pooled", "PRIOR3", f"{t}_{lay}_r{r}", from_band(z[f"pooled_band_{dname}"][:, j], y), y,
                                     truth_kind=dname, **kw)
                    rows.add("v15_lin", "PRIOR3", "PRIOR3", f"{t}_{lay}_r{r}", P3.score(P3.col(xdp[j]), y), y, truth_kind="base", **kw)
    for (t, lay), (Z_, pre, TXm, pr) in belsets.items():
        for r in range(g28.N_REAL):
            key = f"{pre}{r}_base"
            if key not in Z_.files:
                continue
            for n in names:
                if n == "DP1":
                    continue
                y = Dg[n]["base"]; j = int(np.argmin(np.abs(TXm - Dg[n]["x"])))
                kw = dict(truth=f"{t}/{lay}", real=r, dp=n, col=float(TXm[j]), x=float(Dg[n]["x"]), truth_kind="base")
                rows.add("v15_bel", "BEL4", pr.name, f"{t}_{lay}_r{r}", from_samples(Z_[key][:, j], y, [160000, SETS["v15_bel"], bel_id[(t, lay)] * 1000 + 10 * r + j]),
                         y, **kw)
                rows.add("v15_bel", pr.name, pr.name, f"{t}_{lay}_r{r}", pr.score(j, y), y, **kw)
    print(f"[31] v1.5 sets done ({time.time()-t0:.0f} s)", flush=True)

    # ── ρ_below (descriptive): log10 posteriors and priors ──
    rb = {"BEL3-18": B3["P2"][:, 3], "BEL4-18": Z4["P2"][:, 3], "BEL4-18s": Zs["sub_P2"][:, 3], "BEL4-24": ZL["P2"][:, 3],
          "PRIOR3": B3["Y2_train"][:, 3], "PRIOR4": Z4["Y2_train"][:, 3], "PRIOR4-s": Ys, "PRIOR4-L24": ZL["Y2_train"][:, 3]}
    np.savez(os.path.join(D6(args.out), "rho_below_log10.npz"), **{k.replace("-", "_"): np.asarray(v, float) for k, v in rb.items()})
    with open(os.path.join(D6(args.out), "cases.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader()
        for row in rows.rows:
            w.writerow({k: ("" if row.get(k) is None else (repr(float(row[k])) if isinstance(row.get(k), (float, np.floating)) else row[k]))
                        for k in FIELDS})
    print(f"[31] {len(rows.rows)} case rows written ({time.time()-t0:.0f} s)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# report
# ─────────────────────────────────────────────────────────────────────────────
FIELD_METHODS = ["BEL3-18", "BEL4-18", "BEL4-18s", "BEL4-24", "LIN-LM-18", "LIN-3-18", "LIN-MG-18", "LIN-LM-24", "LIN-3-24", "LIN-MG-24",
                 "PRIOR3", "PRIOR4", "PRIOR4-s", "PRIOR4-L24"]
PAIRS = [("BEL4-24", "LIN-LM-24"), ("BEL4-24", "LIN-3-24"), ("BEL4-24", "PRIOR4-L24"), ("BEL4-18", "PRIOR4"), ("BEL3-18", "PRIOR3")]
DPS = ["DP2", "DP3", "DP4", "DP5", "DP6", "DP7"]
FAM = lambda m: "BEL" if m.startswith("BEL") else ("LIN" if m.startswith("LIN") else "PRIOR")
COLF = {"BEL": "#1f4e9c", "LIN": "#d35400", "PRIOR": "0.45"}


def load_cases(root=None):
    rows = list(csv.DictReader(open(os.path.join(D6(root), "cases.csv"), encoding="utf-8")))
    for r in rows:
        for k in ("y", "l", "m", "u", "is", "crps", "x", "col", "r"):
            r[k] = float(r[k]) if r[k] not in ("", None) else np.nan
        r["approx"] = int(r["approx"])
    return rows


def field_vec(C, method, tk="logmid", dps=DPS):
    d = {r["dp"]: r for r in C if r["set"] == "field" and r["method"] == method and r["truth_kind"] == tk}
    g = lambda k: np.array([d[n][k] if n in d else np.nan for n in dps])
    return dict(is_=g("is"), crps=g("crps"), l=g("l"), m=g("m"), u=g("u"), y=g("y"), prior=next(iter(d.values()))["prior"] if d else None,
                approx=int(any(r["approx"] for r in d.values())))


def boot(v, idx):
    return np.nanmean(v[idx], 1)


def fsum(v, p=None, idx=None):
    """Mean with bootstrap 90 % interval; skill against p (vectors over DPs) with interval."""
    o = dict(mean=float(np.nanmean(v)), lo=float(np.percentile(boot(v, idx), 5)), hi=float(np.percentile(boot(v, idx), 95)))
    if p is not None:
        bs = 1 - boot(v, idx) / boot(p, idx)
        o.update(skill=float(1 - np.nanmean(v) / np.nanmean(p)), skill_lo=float(np.percentile(bs, 5)), skill_hi=float(np.percentile(bs, 95)))
    return o


def keyof(r):
    """Pairing key: (case, DP) where a DP is set, else (case, column)."""
    return (r["case"], r["dp"]) if r["dp"] else (r["case"], round(r["col"], 3))


def matched(C, set_, meth, prior, filt=None):
    A = [r for r in C if r["set"] == set_ and r["method"] == meth and (filt is None or filt(r)) and np.isfinite(r["is"])]
    P = {keyof(r): r for r in C if r["set"] == set_ and r["method"] == prior and (filt is None or filt(r))}
    A = [r for r in A if keyof(r) in P]
    return A, [P[keyof(r)] for r in A]


def gsum(A, Pr=None):
    a = lambda R, k: np.array([r[k] for r in R], float)
    o = dict(n=len(A), cov=float(np.mean([(r["l"] <= r["y"] <= r["u"]) for r in A])), width=float(np.median(a(A, "u") - a(A, "l"))),
             mae=float(np.mean(np.abs(a(A, "m") - a(A, "y")))), IS=float(np.mean(a(A, "is"))), CRPS=float(np.mean(a(A, "crps"))),
             approx=int(any(r["approx"] for r in A)))
    if Pr is not None:
        o.update(prior_IS=float(np.mean(a(Pr, "is"))), prior_CRPS=float(np.mean(a(Pr, "crps"))),
                 prior_cov=float(np.mean([(r["l"] <= r["y"] <= r["u"]) for r in Pr])), prior_width=float(np.median(a(Pr, "u") - a(Pr, "l"))))
        o.update(skill_IS=1 - o["IS"] / o["prior_IS"], skill_CRPS=1 - o["CRPS"] / o["prior_CRPS"])
    return o


def syn_groups():
    G = [("held-out", "BEL v3 (L18)", "ho_v3", "BEL3", "PRIOR3", None), ("held-out", "BEL v4 (L18)", "ho_v4", "BEL4", "PRIOR4", None),
         ("held-out", "BEL v4 (L24)", "ho_v4L24", "BEL4-L24", "PRIOR4-L24", None),
         ("v1.0 truths", "BEL v3, log-midpoint truth", "v10_bel", "BEL3", "PRIOR3", None)]
    for st, lab in (("v10_lin", "v1.0 truths"), ("v13_lin", "v1.3 factorial")):
        for d in ("logmid", "maxgrad", "thr3"):
            for tr in ("lam50", "pooled"):
                G.append((lab, f"LIN {d}, {'λ = 50' if tr == 'lam50' else 'λ-pooled'}", st, f"LIN-{d}-{tr}", f"PRIOR3|{d}", None))
    for lay in ("L18", "L24"):
        for d in ("logmid", "thr3"):
            for tr in ("lam50", "pooled"):
                G.append((f"v1.5 T1–T3 ({lay})", f"LIN {d}, {'λ = 50' if tr == 'lam50' else 'λ-pooled'}", "v15_lin", f"LIN-{d}-{tr}", "PRIOR3",
                          (lambda r, lay=lay: r["truth"].endswith("/" + lay))))
    for t, pr in (("T1/L18", "PRIOR4"), ("T3/L18", "PRIOR4"), ("T1/L24", "PRIOR4-L24"), ("T1/L18s", "PRIOR4-s")):
        G.append(("v1.5 BEL v4", f"BEL v4 on {t}" + (" (L24-stretch subset)" if t.endswith("s") else ""), "v15_bel", "BEL4", pr,
                  (lambda r, t=t: r["truth"] == t)))
    return G


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = os.path.join(args.out, "results", "v1_6"); figs = os.path.join(res, "figs"); os.makedirs(figs, exist_ok=True)
    dpp = D6(args.out); os.makedirs(dpp, exist_ok=True)
    C = load_cases(); chk = json.load(open(os.path.join(D6(), "regen_check.json"), encoding="utf-8"))
    idx = np.random.default_rng([160100, 0]).integers(0, len(DPS), (N_BOOT, len(DPS)))
    out = dict(regen=chk)
    f2 = lambda v: f"{v:.2f}"
    ci = lambda o, k="": f"{o['mean' if not k else k]:.2f} [{o['lo' if not k else k + '_lo']:.2f}, {o['hi' if not k else k + '_hi']:.2f}]"
    # ── field, primary truth ──
    F = {m: field_vec(C, m) for m in FIELD_METHODS}
    FS = {}
    for m, v in F.items():
        p = F[v["prior"]] if not m.startswith("PRIOR") else None
        FS[m] = dict(prior=v["prior"], n=int(np.isfinite(v["is_"]).sum()), cov=int(np.sum((v["l"] <= v["y"]) & (v["y"] <= v["u"]))),
                     width=float(np.nanmedian(v["u"] - v["l"])), mae=float(np.nanmean(np.abs(v["m"] - v["y"]))), approx=v["approx"],
                     IS=fsum(v["is_"], None if p is None else p["is_"], idx), CRPS=fsum(v["crps"], None if p is None else p["crps"], idx))
    out["field"] = FS
    pairs = {}
    for a, b in PAIRS:
        pairs[f"{a} vs {b}"] = {k: dict(diff=float(np.nanmean(F[a][kk]) - np.nanmean(F[b][kk])),
                                       p=float(np.mean(boot(F[a][kk], idx) < boot(F[b][kk], idx))))
                                for k, kk in (("IS", "is_"), ("CRPS", "crps"))}
    out["pairs"] = pairs
    # secondary (3 Ω·m crossing)
    sec = {}
    for m in ("LIN-3-18", "LIN-3-24"):
        v = field_vec(C, m, "thr3"); p = field_vec(C, v["prior"], "thr3") if v["prior"] else None
        ok = np.isfinite(v["is_"]) & np.isfinite(p["is_"]) if p else np.isfinite(v["is_"])
        sec[m] = dict(prior=v["prior"], n=int(ok.sum()), cov=int(np.sum(((v["l"] <= v["y"]) & (v["y"] <= v["u"]))[ok])),
                      IS=float(np.mean(v["is_"][ok])), CRPS=float(np.mean(v["crps"][ok])), prior_IS=float(np.mean(p["is_"][ok])),
                      prior_CRPS=float(np.mean(p["crps"][ok])))
        sec[m].update(skill_IS=1 - sec[m]["IS"] / sec[m]["prior_IS"], skill_CRPS=1 - sec[m]["CRPS"] / sec[m]["prior_CRPS"])
    out["field_thr3_truth"] = sec
    D1 = {m: field_vec(C, m, dps=["DP1"]) for m in FIELD_METHODS}
    # ── synthetic ──
    SG = []
    for grp, lab, st, meth, prior, filt in syn_groups():
        A, Pr = matched(C, st, meth, prior, filt)
        if A:
            SG.append(dict(group=grp, label=lab, set=st, method=meth, prior=prior, **gsum(A, Pr)))
    out["synthetic"] = SG
    # S4 paired comparisons
    S4 = []
    def pair_bel_lin(set_b, set_l, meth_l, filt_b, filt_l, lab):
        B = {keyof(r): r for r in C if r["set"] == set_b and r["method"].startswith("BEL") and filt_b(r) and np.isfinite(r["is"])}
        L = [r for r in C if r["set"] == set_l and r["method"] == meth_l and filt_l(r) and np.isfinite(r["is"]) and keyof(r) in B]
        if L:
            S4.append(dict(label=lab, lin=meth_l, n=len(L), IS_lin=float(np.mean([r["is"] for r in L])),
                           IS_bel=float(np.mean([B[keyof(r)]["is"] for r in L])), CRPS_lin=float(np.mean([r["crps"] for r in L])),
                           CRPS_bel=float(np.mean([B[keyof(r)]["crps"] for r in L])), lin_higher=bool(np.mean([r["is"] for r in L]) > np.mean([B[keyof(r)]["is"] for r in L]))))
    for tr in ("lam50", "pooled"):
        pair_bel_lin("v10_bel", "v10_lin", f"LIN-logmid-{tr}", lambda r: True, lambda r: r["truth_kind"] == "logmid",
                     "v1.0 truths (T1, T1h, T2a, T3), log-midpoint truth: BEL v3")
    for t in ("T1/L18", "T3/L18", "T1/L24"):
        for d in ("logmid", "thr3"):
            for tr in ("lam50", "pooled"):
                pair_bel_lin("v15_bel", "v15_lin", f"LIN-{d}-{tr}", lambda r, t=t: r["truth"] == t, lambda r, t=t: r["truth"] == t,
                             f"v1.5 {t}, DP2–DP7: BEL v4")
    out["s4"] = S4
    # ── ρ_below (descriptive) ──
    R = json.load(open(os.path.join(wc.ROOT, "data", "processed", "v1_4", "dp_results.json"), encoding="utf-8"))
    rho_dp = [R["dps"][n]["per"]["primary|A"]["rho_below"] for n in ("DP3", "DP4", "DP5", "DP6")]
    y_rb = float(np.log10(np.median(rho_dp)))
    Zr = np.load(os.path.join(D6(), "rho_below_log10.npz"))
    RB = {}
    for k_, nm in enumerate(["BEL3_18", "PRIOR3", "BEL4_18", "PRIOR4", "BEL4_18s", "PRIOR4_s", "BEL4_24", "PRIOR4_L24"]):
        x = Zr[nm]
        RB[nm.replace("_", "-")] = dict(crps=float(crps_sorted(cap(x, [160000, 10, k_]), y_rb)[0]), med=float(10 ** np.median(x)),
                                        p5=float(10 ** np.percentile(x, 5)), p95=float(10 ** np.percentile(x, 95)))
    out["rho_below"] = dict(truth=float(10 ** y_rb), dp_values=rho_dp, rows=RB)
    # ── verdicts ──
    b, f_ = FS["BEL4-24"], lambda m, k: FS[m][k]["mean"]
    s1 = dict(vs_prior=bool(f_("BEL4-24", "IS") < f_("PRIOR4-L24", "IS") and f_("BEL4-24", "CRPS") < f_("PRIOR4-L24", "CRPS")),
              vs_lin_lm=bool(f_("BEL4-24", "IS") < f_("LIN-LM-24", "IS") and f_("BEL4-24", "CRPS") < f_("LIN-LM-24", "CRPS")),
              vs_lin_3=bool(f_("BEL4-24", "IS") < f_("LIN-3-24", "IS") and f_("BEL4-24", "CRPS") < f_("LIN-3-24", "CRPS")))
    s1["holds"] = all(s1.values())
    s2 = {m: bool(FS[m]["IS"]["skill"] > 0 and FS[m]["CRPS"]["skill"] > 0) for m in ("BEL4-18", "BEL3-18")}; s2["holds"] = all(s2.values())
    sg = {g["set"]: g for g in SG if g["group"] == "held-out"}
    s3 = {"v3": bool(sg["ho_v3"]["skill_IS"] >= 0.3), "v4": bool(sg["ho_v4"]["skill_IS"] >= 0.3)}; s3["holds"] = all(s3.values())
    s4 = dict(holds=all(r["lin_higher"] for r in S4), n_true=sum(r["lin_higher"] for r in S4), n=len(S4),
              exceptions=[f"{r['label']} vs {r['lin']}" for r in S4 if not r["lin_higher"]])
    out["verdicts"] = dict(S1=s1, S2=s2, S3=s3, S4=s4)
    weak = [m for m in ("BEL3-18", "BEL4-18", "BEL4-18s", "BEL4-24") if FS[m]["IS"]["skill"] < 0.1]
    out["weak_field_skill"] = weak
    json.dump(out, open(os.path.join(dpp, "scores.json"), "w", encoding="utf-8"), indent=1)

    # ── markdown ──
    yn = lambda v: "**yes**" if v else "**no**"
    L = ["# 1 — Proper scores for the lens base (v1.6)", "", HEADER, "",
         "IS = 90 % interval score (α = 0.1; lower is better); CRPS from samples (energy form, ≤ 2000 samples) or, for stored "
         "quantile bands, the quantile-score approximation (marked ‡); sharpness = median 90 % width; skill SS = 1 − mean score / "
         "mean prior score. Units: m.", "",
         f"Regenerated BEL posteriors (decision log v1.6-0) reproduce the stored summaries exactly: {yn(chk['all_identical'])} "
         f"(v3 held-out {chk['v3_heldout']['n']}, v4 held-out {chk['v4_heldout']['n']}, v4-L24 held-out {chk['v4L24_heldout']['n']}, "
         "v3 on the four v1.0 truths).", "",
         "## Field — DP2–DP7 (n = 6), truth = DP log-midpoint base", "",
         "Bootstrap: paired over the 6 DPs, 10 000 resamples (`default_rng([160100, 0])`); [90 % interval].", "",
         "| method | prior | n | inside 90 % | sharpness | mean abs. error | mean IS | IS skill | mean CRPS | CRPS skill |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in FIELD_METHODS:
        o = FS[m]; pr = "—" if m.startswith("PRIOR") else o["prior"]
        sk = lambda k: "—" if m.startswith("PRIOR") else f"{o[k]['skill']:+.2f} [{o[k]['skill_lo']:+.2f}, {o[k]['skill_hi']:+.2f}]"
        L.append(f"| {m}{' ‡' if o['approx'] else ''} | {pr} | {o['n']} | {o['cov']} of {o['n']} | {o['width']:.2f} | {o['mae']:.2f} | "
                 f"{ci(o['IS'])} | {sk('IS')} | {ci(o['CRPS'])} | {sk('CRPS')} |")
    L += ["", "Linearised methods are scored against the v4 prior of the matching geometry as a common reference; BEL methods against "
          "their own prior. ‡ CRPS from three stored quantiles (P5, P50, P95).", "",
          "### Paired comparisons (bootstrap over DPs)", "",
          "| pair | Δ mean IS (A − B) | P(IS_A < IS_B) | Δ mean CRPS | P(CRPS_A < CRPS_B) |", "|---|---|---|---|---|"]
    for k, v in pairs.items():
        L.append(f"| {k} | {v['IS']['diff']:+.2f} | {v['IS']['p']:.3f} | {v['CRPS']['diff']:+.2f} | {v['CRPS']['p']:.3f} |")
    L += ["", "### Secondary truth for the 3 Ω·m bands: the DP 3 Ω·m crossing", "",
          "| method | prior | n | inside | mean IS | prior IS | IS skill | mean CRPS ‡ | prior CRPS | CRPS skill |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, o in sec.items():
        L.append(f"| {m} | {o['prior']} | {o['n']} | {o['cov']} of {o['n']} | {o['IS']:.2f} | {o['prior_IS']:.2f} | {o['skill_IS']:+.2f} | "
                 f"{o['CRPS']:.2f} | {o['prior_CRPS']:.2f} | {o['skill_CRPS']:+.2f} |")
    L += ["", "### DP1 (outside the model class; reported separately)", "", "| method | DP base | 90 % interval | inside | IS | CRPS |", "|---|---|---|---|---|---|"]
    for m in FIELD_METHODS:
        v = D1[m]
        if np.isfinite(v["is_"][0]):
            L.append(f"| {m} | {v['y'][0]:.2f} | [{v['l'][0]:.2f}, {v['u'][0]:.2f}] | {'yes' if v['l'][0] <= v['y'][0] <= v['u'][0] else 'no'} | "
                     f"{v['is_'][0]:.2f} | {v['crps'][0]:.2f} |")
        else:
            L.append(f"| {m} | {v['y'][0]:.2f} | — (no estimate) | — | — | — |")
    L += ["", "## Synthetic", "",
          "Linearised sets: columns with r ≤ 10 m; baseline = the BEL v3 prior predictive (PRIOR3) at the nearest BEL column (a common "
          "reference, not the linearised method's own prior). BEL sets: their own prior predictive.", "",
          "| data | method | n | inside 90 % | sharpness | mean abs. error | mean IS | prior IS | IS skill | mean CRPS | prior CRPS | CRPS skill |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in SG:
        L.append(f"| {g['group']} | {g['label']}{' ‡' if g['approx'] else ''} | {g['n']} | {100*g['cov']:.0f} % | {g['width']:.2f} | {g['mae']:.2f} | "
                 f"{g['IS']:.2f} | {g['prior_IS']:.2f} | {g['skill_IS']:+.2f} | {g['CRPS']:.2f} | {g['prior_CRPS']:.2f} | {g['skill_CRPS']:+.2f} |")
    L += ["", "### S4 — linearised vs BEL on the same truths and columns (r ≤ 10 m)", "",
          "| data | linearised | n (paired) | mean IS, linearised | mean IS, BEL | mean CRPS, linearised | mean CRPS, BEL | linearised IS higher? |",
          "|---|---|---|---|---|---|---|---|"]
    for r in S4:
        L.append(f"| {r['label']} | {r['lin']} | {r['n']} | {r['IS_lin']:.2f} | {r['IS_bel']:.2f} | {r['CRPS_lin']:.2f} | {r['CRPS_bel']:.2f} | "
                 f"{'yes' if r['lin_higher'] else '**no**'} |")
    L += ["", "## ρ_below (descriptive only)", "",
          "BEL's ρ_below is a single profile-wide, layout-weighted effective value; the DP values are point measurements that vary "
          f"laterally. Reference: median of the DP3–DP6 clean-sand values ({', '.join(f'{v:.2f}' for v in rho_dp)} Ω·m) = "
          f"{10 ** y_rb:.2f} Ω·m. CRPS in log10 Ω·m. No skill score is computed.", "",
          "| posterior / prior | median [P5, P95] (Ω·m) | CRPS (log10) |", "|---|---|---|"]
    for k, o in RB.items():
        L.append(f"| {k} | {o['med']:.2f} [{o['p5']:.2f}, {o['p95']:.2f}] | {o['crps']:.3f} |")
    L += ["", "## Expectations (logged before computing)", "",
          f"- **S1** BEL4-24 better than PRIOR4-L24 (IS and CRPS): {yn(s1['vs_prior'])}; than LIN-LM-24: {yn(s1['vs_lin_lm'])}; than LIN-3-24: "
          f"{yn(s1['vs_lin_3'])} → S1 {'holds' if s1['holds'] else 'does not hold'}.",
          f"- **S2** skill > 0 (IS and CRPS): BEL4-18 {yn(s2['BEL4-18'])}, BEL3-18 {yn(s2['BEL3-18'])} → S2 {'holds' if s2['holds'] else 'does not hold'}.",
          f"- **S3** held-out IS skill ≥ 0.3: v3 {sg['ho_v3']['skill_IS']:+.2f} {yn(s3['v3'])}, v4 {sg['ho_v4']['skill_IS']:+.2f} {yn(s3['v4'])} "
          f"(v4-L24 {sg['ho_v4L24']['skill_IS']:+.2f}, not part of S3) → S3 {'holds' if s3['holds'] else 'does not hold'}.",
          f"- **S4** linearised mean IS higher than BEL in {s4['n_true']} of {s4['n']} paired comparisons → S4 "
          f"{'holds' if s4['holds'] else 'does not hold everywhere'}" + (f"; exceptions: {'; '.join(s4['exceptions'])}." if s4["exceptions"] else ".")]
    if weak:
        L += ["", f"**Field skill of {', '.join(weak)} against the prior is below 0.1 (IS): their 6/6 coverage comes mainly from width.**"]
    with open(os.path.join(res, "1_scores.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    # ── T_S12 ──
    T = [["scale", "data", "method", "prior", "n", "coverage", "sharpness_m", "mae_m", "mean_IS", "IS_lo90", "IS_hi90", "skill_IS", "skill_IS_lo90",
          "skill_IS_hi90", "mean_CRPS", "CRPS_lo90", "CRPS_hi90", "skill_CRPS", "skill_CRPS_lo90", "skill_CRPS_hi90", "crps_from_quantiles"]]
    g_ = lambda o, k, kk: o[k].get(kk, "") if isinstance(o[k], dict) else ""
    for m in FIELD_METHODS:
        o = FS[m]
        T.append(["field DP2-DP7", "field", m, o["prior"] if not m.startswith("PRIOR") else "", o["n"], o["cov"] / o["n"], o["width"], o["mae"],
                  o["IS"]["mean"], o["IS"]["lo"], o["IS"]["hi"], g_(o, "IS", "skill"), g_(o, "IS", "skill_lo"), g_(o, "IS", "skill_hi"),
                  o["CRPS"]["mean"], o["CRPS"]["lo"], o["CRPS"]["hi"], g_(o, "CRPS", "skill"), g_(o, "CRPS", "skill_lo"), g_(o, "CRPS", "skill_hi"), o["approx"]])
    for g in SG:
        T.append(["synthetic", g["group"], g["label"], g["prior"], g["n"], g["cov"], g["width"], g["mae"], g["IS"], "", "", g["skill_IS"], "", "",
                  g["CRPS"], "", "", g["skill_CRPS"], "", "", g["approx"]])
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    with open(os.path.join(res, "T_S12_scores.csv"), "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([[fmt(v) for v in row] for row in T])
    with open(os.path.join(res, "T_S12_scores.md"), "w", encoding="utf-8") as f:
        f.write("# T_S12 — proper scores (Markdown copy of `T_S12_scores.csv`)\n\n" + HEADER + "\n\n"
                + "| " + " | ".join(T[0]) + " |\n|" + "---|" * len(T[0]) + "\n"
                + "\n".join("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |" for row in T[1:]) + "\n")
    # ── F_20 ──
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    ys = np.arange(len(FIELD_METHODS))[::-1]
    for ax, k, ttl in ((axes[0], "IS", "(a) mean 90 % interval score (m)"), (axes[1], "CRPS", "(b) mean CRPS (m)")):
        for yv, m in zip(ys, FIELD_METHODS):
            o = FS[m][k]; c = COLF[FAM(m)]
            ax.errorbar(o["mean"], yv, xerr=[[o["mean"] - o["lo"]], [o["hi"] - o["mean"]]], fmt="o", color=c, ms=4, lw=1, capsize=2)
        ax.set_yticks(ys); ax.set_yticklabels([m + (" ‡" if FS[m]["approx"] and k == "CRPS" else "") for m in FIELD_METHODS], fontsize=6.5)
        ax.set_title(ttl + "\nfield, DP2–DP7, bootstrap 90 %", fontsize=7); ax.tick_params(labelsize=6.5); ax.grid(axis="x", lw=0.3, color="0.85")
    axes[1].set_yticklabels([])
    ax = axes[2]
    for m in FIELD_METHODS:
        o = FS[m]; ax.plot(o["width"], o["cov"] / o["n"], "o", color=COLF[FAM(m)], ms=5)
        ax.annotate(m, (o["width"], o["cov"] / o["n"]), fontsize=4.8, xytext=(2, 2), textcoords="offset points")
    for g in SG:
        fam = "BEL" if g["method"].startswith("BEL") else "LIN"
        ax.plot(g["width"], g["cov"], "s", mfc="none", color=COLF[fam], ms=4, alpha=0.7)
        ax.plot(g["prior_width"], g["prior_cov"], "s", mfc="none", color=COLF["PRIOR"], ms=3, alpha=0.5)
    ax.axhline(0.9, color="0.5", lw=0.6, ls=":"); ax.set_xscale("log")
    ax.set_xlabel("sharpness: median 90 % width (m)", fontsize=7); ax.set_ylabel("share inside the 90 % interval", fontsize=7)
    ax.set_title("(c) coverage vs sharpness\nfilled: field; open squares: synthetic sets (grey: their priors)", fontsize=7); ax.tick_params(labelsize=6.5)
    fig.suptitle("Scores chosen after the DP results were known; metric definitions logged before computing (decision log v1.6-0). "
                 "Blue BEL, orange linearised, grey prior.", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(figs, "F_20_scores.png"), dpi=300); plt.close(fig)
    # ── F_S13 ──
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 0.22 * len(SG) + 1.4), sharey=True)
    ys = np.arange(len(SG))[::-1]; labs = [f"{g['group']}: {g['label']}" + (" ‡" if g["approx"] else "") for g in SG]
    for yv, g in zip(ys, SG):
        c = COLF["BEL" if g["method"].startswith("BEL") else "LIN"]
        axes[0].plot(g["IS"], yv, "o", color=c, ms=4); axes[0].plot(g["prior_IS"], yv, "|", color=COLF["PRIOR"], ms=7)
        axes[1].barh(yv, g["skill_IS"], color=c, height=0.6)
    axes[0].set_xscale("log"); axes[0].set_yticks(ys); axes[0].set_yticklabels(labs, fontsize=5.8)
    axes[0].set_xlabel("mean IS (m); grey tick: prior", fontsize=7); axes[1].set_xlabel("IS skill against the prior", fontsize=7)
    axes[1].axvline(0, color="k", lw=0.6); axes[1].axvline(0.3, color="0.5", lw=0.5, ls=":")
    for ax in axes:
        ax.tick_params(labelsize=6.5); ax.grid(axis="x", lw=0.3, color="0.85")
    fig.suptitle("Synthetic scores (r ≤ 10 m for linearised sets). Scores chosen after the DP results were known; definitions logged first.", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(figs, "F_S13_synthetic_scores.png"), dpi=300); plt.close(fig)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").add_argument("--out", default=wc.ROOT)
    sub.add_parser("report").add_argument("--out", default=wc.ROOT)
    a = ap.parse_args()
    dict(run=cmd_run, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
