"""
26_raw_abem.py
==============
Westhoek ERT v1.4 — Task 1: the raw 2018 Terrameter LS project of the K1 land line (7 March 2018, from T. Hermans).
Pre-registration: docs/decision_log.md v1.4-3 (E13, E14, definitions).

  run    [--out ROOT]  parse data/raw/Project12_LS210110079_2018-03-07_11-22-23/project.db (read-only), match to
                       land_data.dat, timing, stacking and reciprocal errors, error-model fit, outlier comparison, E14
                       charge-up test, dune electrodes; (f) re-runs g1_lam50 with the reciprocal-based error model as a new
                       run `g1_lam50_recip` (skipped if it exists) → ROOT/data/processed/v1_4/raw2018.json, raw2018_data.csv
  report [--out ROOT]  ROOT/results/v1_4/1_raw_abem.md, T_S8_raw_2018.csv / .md, figs/F_S10_errors.png

Definitions (v1.4-3): stacking error = DataSDev / |R| (DPV DatatypeID 5, Channel > 0); reciprocal error =
|R_n − R_r| / (|R_n + R_r| / 2); our electrode number = 102 − p; task 2 is the reciprocal of station 3 (p = p₂ + 64,
AB ↔ MN). Seeds: bootstrap default_rng([138000]) (E14 odds ratios). No other random draws.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import read_trace  # noqa: E402
read_trace.install("26_raw_abem.py")
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


DB = os.path.join(wc.ROOT, "data", "raw", "Project12_LS210110079_2018-03-07_11-22-23", "project.db")
LAND = os.path.join(wc.ROOT, "data", "raw", "land_data.dat")
DROP = os.path.join(wc.ROOT, "results", "deep_all_droplist.txt")
N_OUR = 102
REC_SHIFT = 64
RECIP_TAG = "g1_lam50_recip"
XREP = (150, 200, 250, 300, 350)
N_BOOT = 2000


def connect():
    return sqlite3.connect(f"file:{DB.replace(os.sep, '/')}?mode=ro", uri=True)


def qkey(r):
    """Quadrupole irrespective of the order within each dipole."""
    return (frozenset((r["A"], r["B"])), frozenset((r["M"], r["N"])))


def our_e(p):
    return int(N_OUR - p) if p <= N_OUR - 1 else None


# ─────────────────────────────────────────────────────────────────────────────
# parsing
# ─────────────────────────────────────────────────────────────────────────────
def read_task(cur, task):
    rows = cur.execute("""
        select v.ID, v.MeasureID, v.DPID, v.Channel, v.DataValue, v.DataSDev, v.Status, v.MCycles, v.N,
               m.StationID, m.Time, a.APosX, a.BPosX, a.MPosX, a.NPosX
        from DPV v join Measures m on v.MeasureID = m.ID join DP_ABMN a on a.ID = v.DPID
        where v.TaskID = ? and v.DatatypeID = 5 and v.Channel > 0 order by m.Time, m.ID, v.Channel""", (task,)).fetchall()
    ip = {}
    for mid, dpid, n in cur.execute("""select MeasureID, DPID, count(*) from DPV where TaskID = ? and DatatypeID = 3
                                       and Channel > 0 group by MeasureID, DPID""", (task,)).fetchall():
        ip[(mid, dpid)] = n
    out = []
    for k, r in enumerate(rows):
        (vid, mid, dpid, ch, R, sd, st, mc, n, sta, t, A, B, M, N) = r
        out.append(dict(id=vid, measure=mid, dpid=dpid, channel=ch, R=R, sdev=sd, status=st, stacks=mc, n=n,
                        station=sta, time=t, A=int(A), B=int(B), M=int(M), N=int(N), order=k,
                        ip_windows=ip.get((mid, dpid), 0),
                        stack_err=(abs(sd / R) if R not in (0, None) and sd is not None else np.nan)))
    return out


def read_land():
    L = open(LAND).read().split("\n")
    n = int(L[0].split()[0])
    D = np.array([[float(v) for v in l.split()[:5]] for l in L[1:1 + n]])
    return D


def tsec(t):
    import datetime as dt
    return dt.datetime.fromisoformat(t).timestamp()


def match_published(t1, D):
    """land_data.dat rows → task-1 rows (our electrode = 102 − p); dipole order swaps flip the sign of R."""
    idx = {}
    for i, r in enumerate(t1):
        idx.setdefault((r["A"], r["B"], r["M"], r["N"]), []).append(i)
    pub = np.full(len(D), -1); rel = np.full(len(D), np.nan); ndup = 0; dup_rows = []
    for j, (a, b, m, n, R) in enumerate(D):
        pa, pb, pm, pn = (int(N_OUR - v) for v in (a, b, m, n))
        cands = []
        for key, sg in (((pa, pb, pm, pn), 1), ((pb, pa, pm, pn), -1), ((pa, pb, pn, pm), -1), ((pb, pa, pn, pm), 1)):
            cands += [(i, sg) for i in idx.get(key, [])]
        if not cands:
            continue
        if len({t1[i]["station"] for i, _ in cands}) > 1:
            ndup += 1
        best = min(cands, key=lambda c: abs(c[1] * t1[c[0]]["R"] - R) / abs(R))
        pub[j] = best[0]; rel[j] = abs(best[1] * t1[best[0]]["R"] - R) / abs(R)
        if len(cands) > 1:
            dup_rows.append(dict(row=j, stations=[t1[i]["station"] for i, _ in cands], chosen=t1[best[0]]["station"],
                                 rel_diffs=[abs(s * t1[i]["R"] - R) / abs(R) for i, s in cands]))
    return pub, rel, ndup, dup_rows


def reciprocal_pairs(t1, t2):
    idx = {}
    for i, r in enumerate(t1):
        if r["station"] == 3:
            idx.setdefault((r["A"], r["B"], r["M"], r["N"]), []).append(i)
    pairs = []
    for k, r in enumerate(t2):
        A, B, M, N = (v + REC_SHIFT for v in (r["M"], r["N"], r["A"], r["B"]))
        for key, sg in (((A, B, M, N), 1), ((B, A, M, N), -1), ((A, B, N, M), -1), ((B, A, N, M), 1)):
            for i in idx.get(key, []):
                Rn, Rr = t1[i]["R"], sg * r["R"]
                pairs.append(dict(i1=i, i2=k, Rn=Rn, Rr=Rr, rec_err=abs(Rn - Rr) / (abs(Rn + Rr) / 2),
                                  published_electrodes=max(t1[i]["A"], t1[i]["B"], t1[i]["M"], t1[i]["N"]) <= N_OUR - 1,
                                  dune=any(N_OUR <= v for v in (t1[i]["A"], t1[i]["B"], t1[i]["M"], t1[i]["N"]))))
    return pairs


def q3(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return dict(n=int(len(v)), median=float(np.median(v)), p90=float(np.percentile(v, 90)), p99=float(np.percentile(v, 99)))


def fit_models(Rbar, d):
    """d ~ N(0, 2σ²); A: σ = a + b|R̄|; B: σ = √((c|R̄|)² + f²). Returns fits, AIC, and the v1.0 log-likelihood."""
    from scipy.optimize import minimize
    R = np.abs(Rbar)
    def nll(sig):
        s2 = 2 * sig ** 2
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + d ** 2 / s2)
    sA = lambda th: np.exp(th[0]) + np.exp(th[1]) * R
    sB = lambda th: np.sqrt((np.exp(th[0]) * R) ** 2 + np.exp(th[1]) ** 2)
    out = {}
    for nm, fn, x0 in (("A", sA, [np.log(1e-4), np.log(0.01)]), ("B", sB, [np.log(0.01), np.log(1e-4)])):
        best = None
        for s in ([0, 0], [1, -1], [-1, 1], [2, 2], [-2, -2]):
            r = minimize(lambda th: nll(fn(th)), np.array(x0) + s, method="Nelder-Mead",
                         options=dict(xatol=1e-8, fatol=1e-8, maxiter=20000))
            if best is None or r.fun < best.fun:
                best = r
        p = np.exp(best.x)
        out[nm] = dict(params=[float(v) for v in p], logL=float(-best.fun), aic=float(2 * 2 + 2 * best.fun))
    v10 = np.sqrt((0.03 * R) ** 2 + 1e-4 ** 2)
    out["v1.0"] = dict(params=[0.03, 1e-4], logL=float(-nll(v10)), aic=float(2 * nll(v10)))
    out["best"] = min(("A", "B"), key=lambda k: out[k]["aic"])
    return out


def sigma_model(fit, R):
    R = np.abs(R); a, b = fit[fit["best"]]["params"]
    return a + b * R if fit["best"] == "A" else np.sqrt((a * R) ** 2 + b ** 2)


def mannwhitney(x, y):
    from scipy.stats import mannwhitneyu
    x = np.asarray(x, float); y = np.asarray(y, float); x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return None
    u = mannwhitneyu(x, y, alternative="two-sided")
    return dict(n_out=int(len(x)), n_ret=int(len(y)), U=float(u.statistic), p=float(u.pvalue),
                p_superiority=float(u.statistic / (len(x) * len(y))),
                rank_biserial=float(1 - 2 * u.statistic / (len(x) * len(y))),
                median_out=float(np.median(x)), median_ret=float(np.median(y)))


def odds(o, g):
    o = np.asarray(o, bool); g = np.asarray(g, bool)
    a, b = (o & g).sum(), (~o & g).sum(); c, d = (o & ~g).sum(), (~o & ~g).sum()
    return (a / b) / (c / d) if b and c and d else np.nan, a / max(a + b, 1), c / max(c + d, 1)


# ─────────────────────────────────────────────────────────────────────────────
def cmd_run(args):
    dpp = os.path.join(args.out, "data", "processed", "v1_4"); os.makedirs(dpp, exist_ok=True)
    con = connect(); cur = con.cursor()
    t1, t2 = read_task(cur, 1), read_task(cur, 2)
    D = read_land()
    pub, rel, ndup, dup_rows = match_published(t1, D)
    for r in t1:
        r["published"] = -1
    for j, i in enumerate(pub):
        if i >= 0:
            t1[i]["published"] = int(j)
    drop = {int(l.split("#")[0]) for l in open(DROP) if l.split("#")[0].strip()}
    out = dict(source=os.path.relpath(DB, wc.ROOT).replace(os.sep, "/"))
    out["counts"] = dict(task1=len(t1), task2=len(t2), published=int(len(D)), matched=int((pub >= 0).sum()),
                         unmatched=int((pub < 0).sum()), max_rel_diff=float(np.nanmax(rel)),
                         dup_quadrupoles=int(ndup), task1_on_published_electrodes=int(sum(max(r["A"], r["B"], r["M"], r["N"]) <= N_OUR - 1 for r in t1)))
    reps = {}
    for i, r in enumerate(t1):
        reps.setdefault(qkey(r), []).append(i)
    pubkeys = {qkey(t1[int(i)]) for i in pub if i >= 0}
    rep_pub = [k for k in pubkeys if len(reps[k]) > 1]
    last = sum(1 for k in rep_pub if max(reps[k], key=lambda i: t1[i]["order"]) in {int(i) for i in pub})
    out["repeats"] = dict(n_repeated_quadrupoles=sum(len(v) > 1 for v in reps.values()), n_published_repeated=len(rep_pub),
                          published_is_last=int(last),
                          station_combinations=sorted({"/".join(str(t1[i]["station"]) for i in sorted(v, key=lambda i: t1[i]["order"]))
                                                       for k, v in reps.items() if len(v) > 1}))
    # selection rule: stacking error ≤ 1 % and R > 0, on published electrodes
    rule = [i for i, r in enumerate(t1) if max(r["A"], r["B"], r["M"], r["N"]) <= N_OUR - 1 and r["R"] > 0
            and np.isfinite(r["stack_err"]) and r["stack_err"] <= 0.01]
    pubset = {int(i) for i in pub if i >= 0}
    rule_q = {qkey(t1[i]) for i in rule}
    q_extra = sorted(rule_q - pubkeys, key=lambda k: min(reps[k]))
    out["rule_quadrupoles"] = dict(n=len(rule_q), n_published=len(pubkeys),
                                   rule_not_published=[dict(row=i, A=our_e(t1[i]["A"]), B=our_e(t1[i]["B"]), M=our_e(t1[i]["M"]),
                                                            N=our_e(t1[i]["N"]), R=t1[i]["R"], stack_err=t1[i]["stack_err"],
                                                            stacks=t1[i]["stacks"], station=t1[i]["station"])
                                                       for k in q_extra for i in [min(j for j in reps[k] if j in rule)]],
                                   published_not_rule=len(pubkeys - rule_q))
    out["rule"] = dict(n_rule=len(rule), rule_not_published=[dict(row=i, A=our_e(t1[i]["A"]), B=our_e(t1[i]["B"]),
                       M=our_e(t1[i]["M"]), N=our_e(t1[i]["N"]), R=t1[i]["R"], stack_err=t1[i]["stack_err"], station=t1[i]["station"])
                       for i in sorted(set(rule) - pubset)],
                       published_not_rule=[dict(row=i, R=t1[i]["R"], stack_err=t1[i]["stack_err"], station=t1[i]["station"])
                                           for i in sorted(pubset - set(rule))])
    # (b) timing
    tim = {}
    for s in (1, 2, 3):
        tt = sorted(t1[i]["time"] for i in pubset if t1[i]["station"] == s)
        tim[str(s)] = [tt[0], tt[-1], len(tt)]
    allpub = sorted(t1[i]["time"] for i in pubset)
    sess = cur.execute("select min(Time), max(Time) from Measures").fetchone()
    out["timing"] = dict(stations=tim, published=[allpub[0], allpub[-1]], session=list(sess),
                         task2=list(cur.execute("select min(m.Time), max(m.Time) from Measures m join Stations s on m.StationID=s.ID where s.TaskID=2").fetchone()),
                         temp=list(cur.execute("select min(Temp), max(Temp) from Measures").fetchone()))
    # settings
    sets = {}
    for k in ("ErrorLimit", "StackLimitsLow", "StackLimitsHigh", "CurrentLimitHighAmpere"):
        sets[k] = sorted({v for (v,) in cur.execute("select Value from AcqSettings where Setting=?", (k,)).fetchall()})
    sets["acq_time"] = sorted({v for (v,) in cur.execute("select Value from AcqSettings where Setting like '%AcqTime%' or Setting like '%Acquisition%' or Setting like '%OnTime%'").fetchall()})
    sets["task_settings"] = [list(r) for r in cur.execute("select key1, Setting, Value from TaskSettings").fetchall()]
    out["settings"] = sets
    # (c) errors
    se_all = [r["stack_err"] for r in t1]; se_pub = [t1[i]["stack_err"] for i in pubset]
    pairs = reciprocal_pairs(t1, t2)
    pe = [p for p in pairs if p["published_electrodes"]]
    out["errors"] = dict(stack_all=q3(se_all), stack_published=q3(se_pub), recip_all=q3([p["rec_err"] for p in pairs]),
                         recip_published=q3([p["rec_err"] for p in pe]), n_pairs=len(pairs), n_pairs_published=len(pe),
                         stack_on_pairs=q3([t1[p["i1"]]["stack_err"] for p in pairs]),
                         n_task2_paired=len({p["i2"] for p in pairs}), n_unique_normal=len({qkey(t1[p["i1"]]) for p in pairs}),
                         n_pairs_dune=sum(p["dune"] for p in pairs),
                         robust_sigma_rel_all=float(1.4826 * np.median([p["rec_err"] for p in pairs]) / np.sqrt(2)),
                         robust_sigma_rel_published=float(1.4826 * np.median([p["rec_err"] for p in pe]) / np.sqrt(2)),
                         recip_over_stack_median=float(np.median([p["rec_err"] / t1[p["i1"]]["stack_err"] for p in pairs
                                                                  if t1[p["i1"]]["stack_err"] > 0])))
    Rbar = np.array([(abs(p["Rn"]) + abs(p["Rr"])) / 2 for p in pairs]); dd = np.array([p["Rn"] - p["Rr"] for p in pairs])
    out["fit_all"] = fit_models(Rbar, dd)
    Rb2 = np.array([(abs(p["Rn"]) + abs(p["Rr"])) / 2 for p in pe]); d2 = np.array([p["Rn"] - p["Rr"] for p in pe])
    out["fit_published"] = fit_models(Rb2, d2) if len(pe) >= 10 else None
    E13 = out["errors"]["stack_published"]["median"]
    out["E13"] = dict(median_published=E13, median_all=out["errors"]["stack_all"]["median"], holds=bool(E13 < 0.03))
    # (d) outliers
    is_out = np.array([j in drop for j in range(len(D))])
    rec_of = {}
    for p in pairs:
        rec_of[p["i1"]] = p["rec_err"]
    se_j = np.array([t1[pub[j]]["stack_err"] if pub[j] >= 0 else np.nan for j in range(len(D))])
    re_j = np.array([rec_of.get(int(pub[j]), np.nan) if pub[j] >= 0 else np.nan for j in range(len(D))])
    out["outliers"] = dict(n_out=int(is_out.sum()), stack=mannwhitney(se_j[is_out], se_j[~is_out]),
                           recip=mannwhitney(re_j[is_out], re_j[~is_out]),
                           n_recip_out=int(np.isfinite(re_j[is_out]).sum()), n_recip_ret=int(np.isfinite(re_j[~is_out]).sum()))
    # (e) E14 charge-up
    meas = cur.execute("""select m.ID, m.Time, min(a.APosX), max(a.BPosX), a.APosX, a.BPosX from Measures m
                          join DPV v on v.MeasureID = m.ID join DP_ABMN a on a.ID = v.DPID
                          join Stations s on m.StationID = s.ID where s.TaskID = 1 and v.Channel > 0
                          group by m.ID order by m.Time, m.ID""").fetchall()
    nab = cur.execute("""select count(*) from (select v.MeasureID from DPV v join DP_ABMN a on a.ID = v.DPID
                         where v.TaskID = 1 and v.Channel > 0 group by v.MeasureID having count(distinct a.APosX || ';' || a.BPosX) > 1)""").fetchone()[0]
    if nab:
        raise SystemExit(f"{nab} injections with more than one A/B pair")
    morder = {mid: k for k, (mid, *_r) in enumerate(meas)}
    mtime = [tsec(t) for _, t, *_r in meas]
    mab = [(int(a), int(b)) for _, _, _, _, a, b in meas]
    last_use = {}                                                     # electrode p → list of measure indices (sorted)
    for k, (a, b) in enumerate(mab):
        last_use.setdefault(a, []).append(k); last_use.setdefault(b, []).append(k)
    import bisect
    dt_, dn_ = np.full(len(D), np.inf), np.full(len(D), np.inf)
    for j in range(len(D)):
        if pub[j] < 0:
            continue
        r = t1[pub[j]]; k = morder[r["measure"]]
        for e in (r["M"], r["N"]):
            L = last_use.get(e, []); q = bisect.bisect_left(L, k) - 1
            if q >= 0:
                dt_[j] = min(dt_[j], mtime[k] - mtime[L[q]]); dn_[j] = min(dn_[j], k - L[q])
    e14 = {}
    rng = np.random.default_rng([138000])
    for nm, v in (("time", dt_), ("count", dn_)):
        fin = np.isfinite(v)
        thr = float(np.percentile(v[fin], 25)) if fin.any() else np.nan
        g = v <= thr
        orr, r1, r0 = odds(is_out, g)
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(D), len(D))
            bs.append(odds(is_out[i], g[i])[0])
        bs = np.array(bs); bs = bs[np.isfinite(bs)]
        e14[nm] = dict(q1=thr, n_low=int(g.sum()), n_rest=int((~g).sum()), n_never=int((~fin).sum()), rate_low=float(r1),
                       rate_rest=float(r0), odds_ratio=float(orr), or90=[float(np.percentile(bs, 5)), float(np.percentile(bs, 95))])
    e14["holds"] = bool(e14["time"]["rate_low"] > e14["time"]["rate_rest"])
    out["E14"] = e14
    # (g) dune electrodes
    dune = list(range(N_OUR, 126))
    etd = cur.execute("select StationID, PosX, ResistanceValue, TestStatus from ElectrodeTestData where PosX >= ?", (N_OUR,)).fetchall()
    out["dune"] = dict(positions=dune, n_data=int(sum(any(v in dune for v in (r["A"], r["B"], r["M"], r["N"])) for r in t1)),
                       n_data_all_dune=int(sum(all(v >= N_OUR for v in (r["A"], r["B"], r["M"], r["N"])) for r in t1)),
                       contact=[dict(station=s, p=int(p), ohm=float(rv), status=int(ts)) for s, p, rv, ts in etd])
    # (f) error-model sensitivity: g1_lam50 with the reciprocal-based error model
    Rp = D[:, 4]
    v10 = np.sqrt((0.03 * Rp) ** 2 + 1e-4 ** 2)
    scale = sigma_model(out["fit_all"], Rp) / v10
    sf = os.path.join(dpp, "recip_err_scale.txt")
    with open(sf, "w", encoding="utf-8") as f:
        f.write(f"# v1.4 Task 1 (f): reciprocal-based error model {out['fit_all']['best']} {out['fit_all'][out['fit_all']['best']]['params']} "
                "÷ v1.0 (3 % + 1e-4 Ω)\n# <0-based index in land_data.dat> <factor>\n")
        for j, s in enumerate(scale):
            f.write(f"{j} {s:.6f}\n")
    out["recip_scale"] = q3(scale)
    run_dir = os.path.join(wc.PROCESSED_DIR, "runs", RECIP_TAG)
    if not os.path.exists(os.path.join(run_dir, "westhoek_inversion_results.npz")):
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "02_land_inversion.py"), "--tag", RECIP_TAG, "--para-depth", "60",
               "--para-dx", "0.33", "--err-rel", "0.03", "--lam", "50", "--max-iter", "30", "--drop-list",
               os.path.join("..", "results", "deep_all_droplist.txt"), "--err-scale-file", sf]
        print("[f] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=SCRIPT_DIR, check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, MPLBACKEND="Agg"))
    out["sensitivity"] = sensitivity()
    with open(os.path.join(dpp, "raw2018.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    cols = ["row", "p_A", "p_B", "p_M", "p_N", "e_A", "e_B", "e_M", "e_N", "station", "time", "order", "R", "stack_err",
            "stacks", "n_cycles_pts", "ip_windows", "status", "published_row", "v10_outlier"]
    with open(os.path.join(dpp, "raw2018_data.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for i, r in enumerate(t1):
            e = [our_e(r[k]) for k in ("A", "B", "M", "N")]
            f.write(",".join(str(v) for v in [i, r["A"], r["B"], r["M"], r["N"], *["" if x is None else x for x in e], r["station"],
                                               r["time"], r["order"], f"{r['R']:.6g}", f"{r['stack_err']:.5g}", r["stacks"], r["n"],
                                               r["ip_windows"], r["status"], r["published"] if r["published"] >= 0 else "",
                                               (r["published"] in drop) if r["published"] >= 0 else ""]) + "\n")
    np.savez(os.path.join(dpp, "raw2018_pairs.npz"), Rn=np.array([p["Rn"] for p in pairs]), Rr=np.array([p["Rr"] for p in pairs]),
             stack=np.array([t1[p["i1"]]["stack_err"] for p in pairs]),
             published_electrodes=np.array([p["published_electrodes"] for p in pairs]))
    print(json.dumps({k: out[k] for k in ("counts", "E13", "E14")}, indent=1, default=float))


def sensitivity():
    """(f): χ² and the 3 Ω·m / log-midpoint bases at XREP for g1_lam50 and g1_lam50_recip (same mesh, same functions)."""
    c11 = _load("c11", "11_stage4_closeout.py")
    res = {}
    ref = wc.load_run("g1_lam50"); mesh, _ = wc.rebuild_paradomain(ref)
    cs = wc.ColumnSampler(mesh, np.array(XREP, float))
    for tag in ("g1_lam50", RECIP_TAG):
        run = wc.load_run(tag)
        if len(run["model"]) != mesh.cellCount():
            raise SystemExit(f"{tag}: mesh differs")
        itf = c11.interfaces_all(cs(np.log10(run["model"]))[:, :, None], cs.z)
        r = (np.log(run["d_obs"]) - np.log(run["d_pred"])) / run["d_err"]
        res[tag] = dict(chi2=float(np.mean(r ** 2)), thr3=[float(v) for v in itf["thr3"][:, 0]],
                        logmid=[float(v) for v in itf["logmid"][:, 0]], median_err=float(np.median(run["d_err"])))
    return res


# ─────────────────────────────────────────────────────────────────────────────
def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    src = os.path.join(args.out, "data", "processed", "v1_4")
    J = json.load(open(os.path.join(src, "raw2018.json"), encoding="utf-8"))
    res = os.path.join(args.out, "results", "v1_4"); figs = os.path.join(res, "figs"); os.makedirs(figs, exist_ok=True)
    s18 = _load("p18", "18_paper.py"); s18.mpl()
    import datetime as dt
    sh = lambda t, h: (dt.datetime.fromisoformat(t) + dt.timedelta(hours=h)).strftime("%H:%M")
    hm = lambda t: t[11:16]
    c, e, t = J["counts"], J["errors"], J["timing"]
    pc = lambda v: f"{100*v:.2f} %"
    dist = lambda d: f"{pc(d['median'])} / {pc(d['p90'])} / {pc(d['p99'])} (n = {d['n']})"
    L = ["# 1 — The raw 2018 file (Task 1)", "",
         f"Source: `{J['source']}` (Terrameter LS project \"Project12\", instrument LS210110079; T. Hermans, October 2026; read-only). "
         "Pre-registration: `docs/decision_log.md` v1.4-3. Disclosure (v1.4-1): the median stacking error of the published "
         "data, a median reciprocal error on the landward spread, the selection rule and the GPS positions were seen before "
         "the pre-registration; anything computed from them is descriptive.", "",
         "## (a) Parse and match", "",
         f"- Task 1 (grad8tri_1, normal): {c['task1']} R values (DatatypeID 5, Channel > 0), three roll-along stations at p = 0, 32, 64; "
         f"{c['task1_on_published_electrodes']} on p ≤ 101 (our electrodes, our number = 102 − p). Task 2 (grad8tri_R_1): {c['task2']} R values.",
         f"- **All {c['matched']} of the {c['published']} published data** of `land_data.dat` match task-1 quadrupoles "
         f"(unmatched: {c['unmatched']}); largest relative difference in R {c['max_rel_diff']:.1e} (< 0.01 %).",
         f"- Repeats: {J['repeats']['n_repeated_quadrupoles']} task-1 quadrupoles were measured more than once, always "
         f"within the same station (sequences {', '.join(J['repeats']['station_combinations'])}), {J['repeats']['n_published_repeated']} "
         f"of them published; the published value is the last one measured in {J['repeats']['published_is_last']} of "
         f"{J['repeats']['n_published_repeated']} cases. (Not \"measured at two stations\": no quadrupole was measured at two stations.)",
         f"- Selection rule \"stacking error ≤ 1 % and R > 0\" on p ≤ 101: {J['rule']['n_rule']} rows = "
         f"**{J['rule_quadrupoles']['n']} distinct quadrupoles** (published: {J['rule_quadrupoles']['n_published']}). The "
         f"{len(J['rule_quadrupoles']['rule_not_published'])} quadrupoles selected by the rule but not published: " + "; ".join(
             f"row {d['row']}: E{d['A']}–E{d['B']} / E{d['M']}–E{d['N']} (station {d['station']}, {d['stacks']} stack(s), "
             f"R = {d['R']:.4g} Ω, stacking error {pc(d['stack_err'])})" for d in J['rule_quadrupoles']['rule_not_published'])
         + f". Published but not selected: {J['rule_quadrupoles']['published_not_rule']}. The 16 single-stack measurements have "
         "stacking error 0 by construction (DataSDev undefined); with ≥ 2 stacks the rule would give 2332 quadrupoles.",
         f"- Settings: ErrorLimit {', '.join(J['settings']['ErrorLimit'])}, stack limits {'/'.join(J['settings']['StackLimitsLow'])} (low) – "
         f"{'/'.join(J['settings']['StackLimitsHigh'])} (high), current limit {' / '.join(J['settings']['CurrentLimitHighAmpere'])} A "
         "(both values occur in the settings); protocols "
         + ", ".join(f"task {k}: {v}" for k, s_, v in J['settings']['task_settings'] if s_ in ("ProtocolFile", "SpreadFile")) + ".",
         "- Per-datum table: `T_S8_raw_2018.csv` (all task-1 data).", "",
         "## (b) Survey timing (instrument clock; time zone unknown — asked T. Hermans)", "",
         "| station | published data, instrument clock | if the clock is UTC: UTC+1 (CET) | if the clock is UTC+1: UTC | n |", "|---|---|---|---|---|"]
    for s, (a, b, n) in t["stations"].items():
        L.append(f"| {s} | {hm(a)}–{hm(b)} | {sh(a, 1)}–{sh(b, 1)} | {sh(a, -1)}–{sh(b, -1)} | {n} |")
    L += ["", f"Published data {hm(t['published'][0])}–{hm(t['published'][1])}; the whole session (incl. task 2, the reciprocal "
          f"measurements of station 3, {hm(t['task2'][0])}–{hm(t['task2'][1])}) {hm(t['session'][0])}–{hm(t['session'][1])} "
          f"instrument clock, 7 March 2018. Instrument temperature {t['temp'][0]:.1f}–{t['temp'][1]:.1f} °C. Tide levels not "
          "fetched in this round.", "",
          "## (c) Errors", "",
          "| quantity | median / P90 / P99 |", "|---|---|",
          f"| stacking error, all task-1 data | {dist(e['stack_all'])} |",
          f"| stacking error, published data | {dist(e['stack_published'])} |",
          f"| reciprocal error, all pairs | {dist(e['recip_all'])} |",
          f"| reciprocal error, pairs on published electrodes | {dist(e['recip_published'])} |",
          f"| stacking error of the normal datum, same pairs | {dist(e['stack_on_pairs'])} |", "",
          f"Pairs: {e['n_pairs']} normal–reciprocal pairings = {e['n_task2_paired']} task-2 data with a partner ({e['n_unique_normal']} "
          f"distinct normal quadrupoles; the extra pairings come from repeated station-3 normals); {e['n_pairs_dune']} involve dune "
          f"electrodes (p 102–125), {e['n_pairs_published']} lie on published electrodes. So reciprocals **were** measured, for the "
          "landward spread (station 3), 13:56–16:14.", "",
          f"Median ratio reciprocal / stacking error on the same pairs: {e['recip_over_stack_median']:.1f}.", "",
          f"Robust scale (descriptive, not pre-registered): 1.4826 × median reciprocal error / √2 = {pc(e['robust_sigma_rel_all'])} "
          f"(all pairs), {pc(e['robust_sigma_rel_published'])} (pairs on published electrodes) — close to the v1.0 3 %.", ""]
    fa = J["fit_all"]
    lab = {"A": "σ = a + b·|R|", "B": "σ = √((c·|R|)² + f²)"}
    L += ["Error models fitted to the reciprocal differences (d = R_n − R_r ~ N(0, 2σ²); descriptive):", "",
          "| model | parameters | log-likelihood | AIC |", "|---|---|---|---|"]
    for k in ("A", "B"):
        p = fa[k]["params"]
        L.append(f"| {k}: {lab[k]} | " + (f"a = {p[0]:.3g} Ω, b = {100*p[1]:.3g} %" if k == "A" else f"c = {100*p[0]:.3g} %, f = {p[1]:.3g} Ω")
                 + f" | {fa[k]['logL']:.1f} | {fa[k]['aic']:.1f} |")
    L.append(f"| v1.0: 3 % + 1e-4 Ω (quadrature) | fixed | {fa['v1.0']['logL']:.1f} | {fa['v1.0']['aic']:.1f} |")
    L += ["", "The Gaussian maximum-likelihood fit is set by the heavy tail of the reciprocal errors (P90 and P99 above), not "
          "by their bulk: it gives much larger errors than the median-based scale.", "",
          f"Better by AIC: **model {fa['best']}**. On the pairs on published electrodes ({e['n_pairs_published']} pairs): " +
          (", ".join(f"{k} " + (f"a = {J['fit_published'][k]['params'][0]:.3g} Ω, b = {100*J['fit_published'][k]['params'][1]:.3g} %" if k == "A"
                                 else f"c = {100*J['fit_published'][k]['params'][0]:.3g} %, f = {J['fit_published'][k]['params'][1]:.3g} Ω")
                     + f" (AIC {J['fit_published'][k]['aic']:.1f})" for k in ("A", "B")) if J["fit_published"] else "too few pairs") + ".",
          f"Ratio of the reciprocal-based error to the v1.0 error on the published data: median {J['recip_scale']['median']:.2f} "
          f"(P90 {J['recip_scale']['p90']:.2f}).", "",
          f"**E13** (median stacking error < 3 %): **{'holds' if J['E13']['holds'] else 'does not hold'}** — {pc(J['E13']['median_published'])} "
          f"on the published data ({pc(J['E13']['median_all'])} on all task-1 data).", ""]
    o = J["outliers"]
    mw = lambda m: "—" if m is None else (f"median {pc(m['median_out'])} (outliers, n = {m['n_out']}) vs {pc(m['median_ret'])} "
                                         f"(retained, n = {m['n_ret']}); U = {m['U']:.0f}, p = {m['p']:.2g}, rank-biserial "
                                         f"1 − 2U/(n₁n₂) = {m['rank_biserial']:+.2f} (P(outlier > retained) = {m['p_superiority']:.2f})")
    L += ["## (d) Outliers (357 v1.0 outliers vs 1974 retained)", "",
          f"- Stacking error: {mw(o['stack'])}.", f"- Reciprocal error (data with a reciprocal pair): {mw(o['recip'])}.", "",
          "## (e) E14 — charge-up test", ""]
    E = J["E14"]
    L += ["| quantity since a potential electrode was last a current electrode | lowest quartile (≤) | outlier rate, lowest quartile | outlier rate, rest | odds ratio [90 % bootstrap] | never used before |",
          "|---|---|---|---|---|---|"]
    for nm, u in (("time", "s"), ("count", "injections")):
        v = E[nm]
        L.append(f"| {nm} | {v['q1']:.0f} {u} (n = {v['n_low']}) | {100*v['rate_low']:.1f} % | {100*v['rate_rest']:.1f} % (n = {v['n_rest']}) | "
                 f"{v['odds_ratio']:.2f} [{v['or90'][0]:.2f}, {v['or90'][1]:.2f}] | {v['n_never']} |")
    L += ["", f"**E14** (outlier rate higher in the lowest time quartile): **{'holds' if E['holds'] else 'does not hold'}**.", ""]
    S = J["sensitivity"]; a, b = S["g1_lam50"], S[RECIP_TAG]
    L += ["## (f) Error-model sensitivity (g1_lam50 re-run with the reciprocal-based error model; nothing else changed)", "",
          f"χ² (with each run's own errors): v1.0 {a['chi2']:.2f}, reciprocal-based {b['chi2']:.2f}; median relative error "
          f"{pc(a['median_err'])} vs {pc(b['median_err'])}.", "",
          "| x (m) | 3 Ω·m v1.0 | 3 Ω·m reciprocal | log-midpoint v1.0 | log-midpoint reciprocal |", "|---|---|---|---|---|"]
    for i, x in enumerate(XREP):
        L.append(f"| {x} | {a['thr3'][i]:.2f} | {b['thr3'][i]:.2f} | {a['logmid'][i]:.2f} | {b['logmid'][i]:.2f} |")
    d = J["dune"]
    L += ["", "This is a sensitivity check; the v1.0 numbers stay as they are.", "",
          "## (g) Unpublished dune electrodes", "",
          f"p = {d['positions'][0]}–{d['positions'][-1]} ({len(d['positions'])} electrodes, x ≈ −5 to −120 m, landward of E1); "
          f"{d['n_data']} task-1 data involve at least one of them ({d['n_data_all_dune']} only these). Contact resistances "
          "(electrode test, kΩ): " + ", ".join(f"p{x['p']} {x['ohm']/1000:.2f}" + ("" if x["status"] == 4 else f" (status {x['status']})")
                                              for x in d["contact"] if x["p"] < 126) + ". p 126–127: " +
          ", ".join(f"{x['ohm']/1000:.0f} kΩ (status {x['status']})" for x in d["contact"] if x["p"] >= 126) +
          ". Not added to any inversion."]
    with open(os.path.join(res, "1_raw_abem.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # T_S8 csv (copy of the per-datum table) + Markdown summary
    import shutil
    shutil.copy2(os.path.join(src, "raw2018_data.csv"), os.path.join(res, "T_S8_raw_2018.csv"))
    rows = list(__import__("csv").DictReader(open(os.path.join(res, "T_S8_raw_2018.csv"), encoding="utf-8")))
    M = ["# T_S8 — raw 2018 data (summary of `T_S8_raw_2018.csv`)", "",
         "| station | data | on our electrodes | published | v1.0 outliers | median stacking error | ≤ 1 % | stacks 2 / 3 | time window (instrument clock) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in ("1", "2", "3"):
        rs = [r for r in rows if r["station"] == s]
        se = np.array([float(r["stack_err"]) for r in rs])
        M.append(f"| {s} | {len(rs)} | {sum(r['e_A'] != '' and r['e_B'] != '' and r['e_M'] != '' and r['e_N'] != '' for r in rs)} | "
                 f"{sum(r['published_row'] != '' for r in rs)} | {sum(r['v10_outlier'] == 'True' for r in rs)} | "
                 f"{100*np.nanmedian(se):.2f} % | {int(np.sum(se <= 0.01))} | {sum(r['stacks'] == '2' for r in rs)} / {sum(r['stacks'] == '3' for r in rs)} | "
                 f"{min(r['time'] for r in rs)[11:16]}–{max(r['time'] for r in rs)[11:16]} |")
    with open(os.path.join(res, "T_S8_raw_2018.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(M) + "\n")

    # F_S10
    P = np.load(os.path.join(src, "raw2018_pairs.npz"))
    Rb = (np.abs(P["Rn"]) + np.abs(P["Rr"])) / 2; dd = np.abs(P["Rn"] - P["Rr"]) / np.sqrt(2)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axes[0]
    pe = P["published_electrodes"]
    ax.loglog(Rb[~pe], dd[~pe], ".", ms=2.5, color="0.65", label="pairs (dune electrodes involved)")
    ax.loglog(Rb[pe], dd[pe], ".", ms=3, color="#1f4e9c", label="pairs on published electrodes")
    rr = np.logspace(np.log10(Rb.min()), np.log10(Rb.max()), 100)
    for k, col, ls in (("A", "#d35400", "-"), ("B", "#6a3d9a", "--")):
        p = fa[k]["params"]
        s = p[0] + p[1] * rr if k == "A" else np.sqrt((p[0] * rr) ** 2 + p[1] ** 2)
        ax.loglog(rr, s, ls, color=col, lw=1.2, label=f"model {k}" + (" (best AIC)" if fa["best"] == k else ""))
    ax.loglog(rr, np.sqrt((0.03 * rr) ** 2 + 1e-4 ** 2), "k:", lw=1.3, label="v1.0: 3 % + 1e-4 Ω")
    ax.set_xlabel("|R| (Ω)"); ax.set_ylabel("|R_n − R_r| / √2 (Ω)"); ax.legend(fontsize=5.8); ax.set_title("(a) reciprocal errors vs models", fontsize=8)
    ax = axes[1]
    re = np.abs(P["Rn"] - P["Rr"]) / ((np.abs(P["Rn"] + P["Rr"])) / 2)
    ax.loglog(100 * P["stack"], 100 * re, ".", ms=2.5, color="#1f4e9c")
    lim = [1e-3, 1e3]; ax.plot(lim, lim, "k-", lw=0.7); ax.axhline(3, color="k", ls=":", lw=1); ax.axvline(1, color="0.5", ls="--", lw=0.8)
    ax.set_xlim(1e-3, 1e2); ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel("stacking error (%)"); ax.set_ylabel("reciprocal error (%)"); ax.set_title("(b) same pairs; dotted: 3 %, dashed: 1 % ErrorLimit", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "F_S10_errors.png"), dpi=300); plt.close(fig)
    print("\n".join(L))
    report_followup(args)


# ─────────────────────────────────────────────────────────────────────────────
# Task H — E14 follow-up (descriptive; decision log v1.4-9)
# ─────────────────────────────────────────────────────────────────────────────
def logit_fit(X, y, iters=100):
    """Logistic regression by Newton–Raphson; returns coefficients, standard errors, log-likelihood."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = p * (1 - p); H = X.T @ (X * W[:, None]); g = X.T @ (y - p)
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), g); b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-(X @ b))), 1e-12, 1 - 1e-12)
    W = p * (1 - p); H = X.T @ (X * W[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se, float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cmd_followup(args):
    import bisect
    dpp = os.path.join(args.out, "data", "processed", "v1_4")
    con = connect(); cur = con.cursor()
    t1 = read_task(cur, 1); D = read_land()
    pub, _, _, _ = match_published(t1, D)
    drop = {int(l.split("#")[0]) for l in open(DROP) if l.split("#")[0].strip()}
    y = np.array([j in drop for j in range(len(D))], float)
    # recent-use flag (identical to E14, time quantity)
    meas = cur.execute("""select m.ID, m.Time, a.APosX, a.BPosX from Measures m join DPV v on v.MeasureID = m.ID
                          join DP_ABMN a on a.ID = v.DPID join Stations s on m.StationID = s.ID
                          where s.TaskID = 1 and v.Channel > 0 group by m.ID order by m.Time, m.ID""").fetchall()
    morder = {mid: k for k, (mid, *_r) in enumerate(meas)}; mtime = [tsec(t) for _, t, _, _ in meas]
    use = {}
    for k, (_, _, a, b) in enumerate(meas):
        use.setdefault(int(a), []).append(k); use.setdefault(int(b), []).append(k)
    dt = np.full(len(D), np.inf)
    for j in range(len(D)):
        r = t1[pub[j]]; k = morder[r["measure"]]
        for e in (r["M"], r["N"]):
            L = use.get(e, []); q = bisect.bisect_left(L, k) - 1
            if q >= 0:
                dt[j] = min(dt[j], mtime[k] - mtime[L[q]])
    q1 = np.percentile(dt[np.isfinite(dt)], 25); flag = (dt <= q1).astype(float)
    J0 = json.load(open(os.path.join(dpp, "raw2018.json"), encoding="utf-8"))["E14"]["time"]
    assert abs(q1 - J0["q1"]) < 1e-9 and int(flag.sum()) == J0["n_low"], "recent-use flag differs from E14"
    # covariates
    land, _ = wc.read_electrodes(); xe = land[:, 0]
    du = {(m, d): v for m, d, v in cur.execute("select MeasureID, DPID, DataValue from DPV where TaskID = 1 and DatatypeID = 7 and Channel > 0").fetchall()}
    etd = {(s, int(p)): v for s, p, v in cur.execute("select StationID, PosX, ResistanceValue from ElectrodeTestData").fetchall()}
    rows = []
    for j in range(len(D)):
        r = t1[pub[j]]; xa, xb, xm, xn = (xe[N_OUR - 1 - r[k]] for k in ("A", "B", "M", "N"))
        rr = lambda p_, q_: abs(p_ - q_)
        K = 2 * np.pi / (1 / rr(xa, xm) - 1 / rr(xb, xm) - 1 / rr(xa, xn) + 1 / rr(xb, xn))
        cr = max(etd.get((r["station"], r[k]), np.nan) for k in ("A", "B", "M", "N"))
        rows.append([np.log10(abs(K)), np.log10(abs(du[(r["measure"], r["dpid"])])), abs(xa - xb), abs(xm - xn), r["station"], np.log10(cr)])
    R_ = np.array(rows, float)
    names = ["recent_use", "log10|K|", "log10|dU|", "AB span", "MN span", "station 2", "station 3", "log10 max contact R"]
    cont = R_[:, [0, 1, 2, 3, 5]]; mu, sd = cont.mean(0), cont.std(0)
    Z = (cont - mu) / sd
    X = np.column_stack([np.ones(len(D)), flag, Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3], (R_[:, 4] == 2).astype(float),
                         (R_[:, 4] == 3).astype(float), Z[:, 4]])
    b, se, ll = logit_fit(X, y)
    groups = {"recent_use": [1], "log10|K|": [2], "log10|dU|": [3], "AB span": [4], "MN span": [5], "station": [6, 7], "log10 max contact R": [8]}
    drops = {}
    for g, idx in groups.items():
        keep = [i for i in range(X.shape[1]) if i not in idx]
        drops[g] = ll - logit_fit(X[:, keep], y)[2]
    rng = np.random.default_rng([138100]); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(D), len(D))
        try:
            bs.append(np.exp(logit_fit(X[i], y[i])[0][1]))
        except np.linalg.LinAlgError:
            pass
    b0, _, _ = logit_fit(X[:, :2], y)
    out = dict(n=len(D), n_out=int(y.sum()), q1=float(q1), coef={n_: float(v) for n_, v in zip(["intercept"] + names, b)},
               se={n_: float(v) for n_, v in zip(["intercept"] + names, se)}, z={n_: float(v / s_) for n_, v, s_ in zip(["intercept"] + names, b, se)},
               or_recent_adj=float(np.exp(b[1])), or_recent_adj90=[float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
               or_recent_crude=float(np.exp(b0[1])), loglik=ll, ll_drop={k: float(v) for k, v in drops.items()},
               covariate_means=dict(zip(["log10|K|", "log10|dU|", "AB span", "MN span", "log10 max contact R"], map(float, mu))),
               covariate_sd=dict(zip(["log10|K|", "log10|dU|", "AB span", "MN span", "log10 max contact R"], map(float, sd))),
               n_boot=len(bs))
    out["by_station"] = {str(s): dict(n=int((R_[:, 4] == s).sum()), rate=float(y[R_[:, 4] == s].mean()),
                                      recent_share=float(flag[R_[:, 4] == s].mean())) for s in (1, 2, 3)}
    dec = np.quantile(R_[:, 1], np.linspace(0, 1, 11)); di = np.clip(np.searchsorted(dec, R_[:, 1], side="right") - 1, 0, 9)
    out["by_dU_decile"] = [dict(decile=int(k + 1), dU_lo=float(10 ** dec[k]), dU_hi=float(10 ** dec[k + 1]), n=int((di == k).sum()),
                                rate=float(y[di == k].mean()), recent_share=float(flag[di == k].mean())) for k in range(10)]
    out["recent_vs_dU"] = dict(median_dU_recent=float(10 ** np.median(R_[flag == 1, 1])), median_dU_rest=float(10 ** np.median(R_[flag == 0, 1])),
                               median_AB_recent=float(np.median(R_[flag == 1, 2])), median_AB_rest=float(np.median(R_[flag == 0, 2])))
    with open(os.path.join(dpp, "e14_followup.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("or_recent_crude", "or_recent_adj", "or_recent_adj90", "ll_drop", "by_station")}, indent=1))


def report_followup(args):
    src = os.path.join(args.out, "data", "processed", "v1_4", "e14_followup.json")
    if not os.path.exists(src):
        return
    J = json.load(open(src, encoding="utf-8")); res = os.path.join(args.out, "results", "v1_4")
    L = ["# 7 — E14 follow-up: does geometry or contact explain the reversed result? (Task H, descriptive)", "",
         "Logged before running: `docs/decision_log.md` v1.4-9. Logistic regression of the v1.0 outlier status "
         f"({J['n_out']} of {J['n']} published data) on the recent-current-use flag (E14 lowest time quartile, ≤ {J['q1']:.0f} s) and "
         "standardised covariates. Nothing here is a test.", "",
         f"- **Recent-use flag:** crude odds ratio {J['or_recent_crude']:.2f}; adjusted for all covariates **{J['or_recent_adj']:.2f}** "
         f"[90 % bootstrap {J['or_recent_adj90'][0]:.2f}, {J['or_recent_adj90'][1]:.2f}] ({J['n_boot']} resamples).", "",
         "| covariate | coefficient (per SD; flag and station: per unit) | z | drop in log-likelihood when removed |", "|---|---|---|---|"]
    grp = {"recent_use": "recent_use", "log10|K|": "log10|K|", "log10|dU|": "log10|dU|", "AB span": "AB span", "MN span": "MN span",
           "station 2": "station", "station 3": "station", "log10 max contact R": "log10 max contact R"}
    for k in ("recent_use", "log10|K|", "log10|dU|", "AB span", "MN span", "station 2", "station 3", "log10 max contact R"):
        L.append(f"| {k} | {J['coef'][k]:+.2f} | {J['z'][k]:+.1f} | {J['ll_drop'][grp[k]]:.1f} |")
    top = sorted(J["ll_drop"].items(), key=lambda kv: -kv[1])
    L += ["", "Covariates by drop in log-likelihood: " + ", ".join(f"{k} ({v:.1f})" for k, v in top) + ".", "",
          "## Outlier rate by station", "", "| station | data | outlier rate | share flagged 'recent use' |", "|---|---|---|---|"]
    for s, v in J["by_station"].items():
        L.append(f"| {s} | {v['n']} | {100*v['rate']:.1f} % | {100*v['recent_share']:.0f} % |")
    L += ["", "## Outlier rate by |ΔU| decile", "", "| decile | |ΔU| range (V) | data | outlier rate | share flagged 'recent use' |", "|---|---|---|---|---|"]
    for d in J["by_dU_decile"]:
        L.append(f"| {d['decile']} | {d['dU_lo']:.2e} – {d['dU_hi']:.2e} | {d['n']} | {100*d['rate']:.1f} % | {100*d['recent_share']:.0f} % |")
    rv = J["recent_vs_dU"]
    L += ["", f"Median |ΔU|: {rv['median_dU_recent']:.2e} V for data flagged recent use vs "
          f"{rv['median_dU_rest']:.2e} V (rest); median AB span {rv['median_AB_recent']:.0f} vs {rv['median_AB_rest']:.0f} m."]
    with open(os.path.join(res, "7_e14_followup.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "report", "followup"):
        sub.add_parser(c).add_argument("--out", default=wc.ROOT)
    a = ap.parse_args()
    dict(run=cmd_run, report=cmd_report, followup=cmd_followup)[a.cmd](a)


if __name__ == "__main__":
    main()
