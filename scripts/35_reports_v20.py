#!/usr/bin/env python3
"""v2.0 reports (decision log v2.0-0): one Markdown report per task in results/v2_0/, generated from the JSON / npz
outputs of 33_review_v20.py in data/processed/v2_0/ (copies of the JSON go to results/v2_0/json/).

    python 35_reports_v20.py a2 a5 a6 a7 a9 ...
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DP = os.path.join(ROOT, "data", "processed", "v2_0")
RS = os.path.join(ROOT, "results", "v2_0")
HDR = ("*Post hoc analysis requested by the internal review of WRR v4; pre-registered in `docs/decision_log.md` v2.0-0 "
       "before any computation. No earlier result is changed.*")
DPS6 = ["DP2", "DP3", "DP4", "DP5", "DP6", "DP7"]


def J(name):
    return json.load(open(os.path.join(DP, name), encoding="utf-8"))


def keep_json(name):
    os.makedirs(os.path.join(RS, "json"), exist_ok=True)
    shutil.copy2(os.path.join(DP, name), os.path.join(RS, "json", name))


def write(name, L):
    os.makedirs(RS, exist_ok=True)
    open(os.path.join(RS, name), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", name)


f2 = lambda v: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2f}"
f3 = lambda v: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"
iv = lambda q: f"{q[0]:.2f} [{q[1]:.2f}, {q[2]:.2f}]"          # (P50, P5, P95)


# ─────────────────────────────────────────────────────────────────────────────
def rep_a2():
    d = J("a2.json"); keep_json("a2.json")
    L = ["# A2 — BEL posterior support: truncation above the ground surface", "", HDR, "",
         "**Rule (v2.0-0).** At each column, posterior samples whose lens base lies above the ground surface of that column are removed; "
         "the surface is the one each prior was drawn on (2018 electrode surface for the L18 BELs, CPT topography for L24). "
         "Priors are truncated by the same rule. Scores are recomputed exactly as `31_scores.py` (same subsample seeds).", "",
         f"*Checks.* The untruncated rows reproduce `data/processed/v1_6/cases.csv` (max |ΔIS| + |ΔCRPS| = "
         f"{d['orig_reproduces_cases_csv_maxdiff']:.1e} over {d['orig_n_compared']} rows). Refitting each BEL from its stored training set "
         f"reproduces the stored field posterior bit for bit ({', '.join(k + ': ' + ('yes' if v else 'no') for k, v in d['regen'].items())}); "
         "this needs the default BLAS threading — with `OMP_NUM_THREADS=2` the refit differs in the last digits (summation order), "
         "so the BEL fits of v2.0 run with default threading. "
         f"Prior samples above the ground: {', '.join(f'{k} {v}' for k, v in d['prior_samples_above_ground'].items())} "
         "(base depth is positive by construction), so the priors do not change.", "",
         "## Where the truncation acts (Table 2 BEL rows; P50 [P5, P95], mTAW)", "",
         "| BEL | DP | column x (m) | surface (mTAW) | samples removed (of 2000) | stored | truncated | DP base | inside, stored → truncated |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m, t in d["dp_table"].items():
        for dp, v in t.items():
            if v["n_removed"] == 0:
                continue
            y = v["y"]; ins = lambda q: q[1] <= y <= q[2]
            L.append(f"| {m} | {dp} | {v['col']:.0f} | {v['surface']:.2f} | {v['n_removed']} | {iv(v['orig'])} | {iv(v['trunc'])} | {y:.2f} | "
                     f"{'yes' if ins(v['orig']) else 'no'} → {'yes' if ins(v['trunc']) else 'no'} |")
    L += ["", "No sample is removed at any other DP column except one sample each at DP3 and DP4 for BEL4-18s (no change at two decimals).", "",
          "## Field scores, DP2–DP7 (Table 3 / scores figure BEL rows)", "",
          "| method | prior | inside (stored → trunc.) | median width (m) | mean IS stored → trunc. | IS skill stored → trunc. | mean CRPS stored → trunc. | CRPS skill stored → trunc. |",
          "|---|---|---|---|---|---|---|---|"]
    O, T = d["field_summary"]["orig"], d["field_summary"]["trunc"]
    for m in O:
        o, t = O[m], T[m]
        sk = lambda s, k: (f"{s[k]['skill']:+.2f} [{s[k]['skill_lo']:+.2f}, {s[k]['skill_hi']:+.2f}]" if "skill" in s[k] else "—")
        L.append(f"| {m} | {o['prior']} | {o['cov']}/{o['n']} → {t['cov']}/{t['n']} | {o['width']:.2f} → {t['width']:.2f} | "
                 f"{o['IS']['mean']:.2f} → {t['IS']['mean']:.2f} | {sk(o, 'IS')} → {sk(t, 'IS')} | {o['CRPS']['mean']:.3f} → {t['CRPS']['mean']:.3f} | "
                 f"{sk(o, 'CRPS')} → {sk(t, 'CRPS')} |")
    chg = [m for m in O if any(O[m]["inside"][n] != T[m]["inside"][n] for n in O[m]["inside"])]
    L += ["", f"Inside/outside verdicts changed by the truncation: **{', '.join(chg) if chg else 'none'}**.", "",
          "## Canonical correlation, KDE bandwidth and the field's position in canonical space", "",
          "Canonical variates have unit variance on the training set, so the Gaussian-KDE bandwidth (Scott's rule, `scipy.stats.gaussian_kde`, "
          "2-D per pair) equals the Scott factor n^(−1/6) in both canonical coordinates. Field percentile = share of training samples with a "
          "smaller data canonical score; Mahalanobis percentile = share of training samples closer to the training mean in the leading five "
          "data canonical scores (small = central).", "",
          "| BEL | n train | data / target PCs | r₁ … r₅ | KDE bandwidth (c_d, c_t) | field c_d,1 … c_d,5 | field percentiles | Mahalanobis² field (percentile) |",
          "|---|---|---|---|---|---|---|---|"]
    for m, c in d["canon"].items():
        p = c["pairs"]
        L.append(f"| {m} | {c['n_train']} | {c['kd']} / {c['kt']} | {', '.join(f'{x:.3f}' for x in c['r'])} | {p[0]['bw_cd']:.3f}, {p[0]['bw_ct']:.3f} | "
                 f"{', '.join('%+.2f' % q['field_cd'] for q in p)} | {', '.join('%.2f' % q['field_pct'] for q in p)} | "
                 f"{c['mahal2_field']:.2f} ({c['mahal2_pct']:.3f}) |")
    exp_ok = (set(n for m in d["dp_table"] for n, v in d["dp_table"][m].items() if v["n_removed"] > 1) <= {"DP2"}) and not chg
    L += ["", "## Expectation A2", "",
          "*Logged:* the truncation changes a BEL interval at DP2 only, and no inside/outside verdict.", "",
          f"*Outcome:* **{'holds' if exp_ok else 'does not hold as written'}** — no verdict changes, but the truncation also moves the BEL "
          "intervals at DP1 (its base lies at −16.6 mTAW, outside every BEL interval before and after; DP1 is not part of the DP2–DP7 scores). "
          "At DP2 the P95 drops from above the ground to below it, and the interval still contains the DP base.", ""]
    write("A2_bel_support.md", L)


# ─────────────────────────────────────────────────────────────────────────────
def rep_a5():
    d = J("a5.json"); keep_json("a5.json")
    L = ["# A5 — damping with matched windows (2025 line, λ = 50)", "", HDR, "",
         "**Matched window.** Exactly the elevations of the sounding's ρ_below,DP samples: 1 cm samples from 2 m below the DP log-midpoint base "
         "to the end of the log or −25 mTAW, clean sand only (nearest CPT sample I_SBT < 2.05). The image value is 10^(median of the λ = 50 "
         "image log10 ρ interpolated at those elevations). **Old window (E20):** 4–10 m below the image's own λ = 50 log-midpoint base. "
         "% seawater: the sounding's Archie draws (`default_rng([137200, DP index])`, 20 000), applied to each ρ. "
         "Cumulative sensitivity: Σ_i (J_ij/ε_i)² of the λ = 50 inversion, normalised by its maximum (as `02_land_inversion.py`), log10, "
         "median over the cells in the window and over the lens (surface to DP base).", ""]
    if d.get("image_rerun_maxdiff_log10_at_dps") is not None:
        L += [f"*Image check:* the λ = 50 rerun (A3) differs from the stored `lam50_col` by at most {d['image_rerun_maxdiff_log10_at_dps']:.2e} "
              f"in log10 at the DP columns → {'accepted' if d['image_rerun_accepted'] else 'NOT accepted'} (rule: ≤ 0.01).", ""]
    L += ["*Sounding check:* the recomputed ρ_below,DP equals the stored value at every DP; the recomputed old-window image values equal the stored E20 values.", "",
          "| DP | DP base (mTAW) | window (mTAW) | clean sand (m) | ρ_below,DP (Ω·m) | image, matched | ratio | image, old window | ratio | "
          "% seawater DP | % seawater image matched | % seawater image old | log10 cum. sens. window | lens |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    sw = lambda q: "—" if not q else f"{q[0]:.1f} [{q[1]:.1f}, {q[2]:.1f}]"
    for n, v in d["dps"].items():
        if "window_top" not in v:
            L.append(f"| {n} | — | — | — | — | — | — | — | — | — | — | — | — | — |"); continue
        L.append(f"| {n} | {v['base']:.2f} | {v['window_top']:.2f} … {v['window_bottom']:.2f} | {v['clean_m']:.2f} | {f2(v['rho_below_dp'])} | "
                 f"{f2(v.get('rho_img_matched'))} | {f2(v.get('ratio_matched'))} | {f2(v['rho_img_old'])} | {f2(v['ratio_old'])} | {sw(v['sw_dp'])} | "
                 f"{sw(v.get('sw_img_matched'))} | {sw(v['sw_img_old'])} | {f2(v.get('log10_cumsens_window'))} | {f2(v.get('log10_cumsens_lens'))} |")
    e = d["expectation"]
    L += ["", "## Expectation A5", "", "*Logged:* with the matched window, the image is below half the sounding value at ≥ 4 of DP3–DP7.", "",
          f"*Outcome:* **{'holds' if e['holds'] else 'does not hold'}** — {e['n_below_half']} of {e['n']} ({', '.join(e['below_half'])}).", ""]
    write("A5_damping_matched.md", L)


# ─────────────────────────────────────────────────────────────────────────────
def rep_a6():
    d = J("a6.json"); keep_json("a6.json")
    m = lambda x: (f"n = {x['n']}, RMSE {x['rmse']:.2f} m, centred R² {f2(x['r2_centred'])}, sign {100 * x['sign_all']:.0f} % "
                   f"(with |q − 0.5| > 0.1: {100 * x['sign_margin']:.0f} % of {x['n_margin']})") if x else "—"
    L = ["# A6 — the bias rule out of sample", "", HDR, "",
         f"Rule (v1.3): bias = −a · r̄ · Φ⁻¹(q); full fit on the 33 sharp-truth cases (thr2, thr3, thr5; 0 < q < 1; λ = 50; r ≤ 10 m): "
         f"a = {d['a_full']:.3f}, R² = {d['r2_full']:.2f}.", "",
         "## Leave-one-truth-out (a refitted without each truth)", "", f"- {m(d['loto'])}", f"- in sample (for comparison): {m(d['in_sample'])}", "",
         "| truth | contour | q | r̄ (m) | observed bias (m) | a without the truth | predicted (m) |", "|---|---|---|---|---|---|---|"]
    for c in d["loto"]["cases"]:
        L.append(f"| {c['truth']} | {c['d']} | {c['q']:.3f} | {c['rbar']:.2f} | {c['y']:+.2f} | {c['a_loto']:.3f} | {c['pred']:+.2f} |")
    L += ["", "## Out of sample, full-fit a", "",
          "Site-like truths: v1.0 T1, T1h, T2a, T3, 3 Ω·m main-lens base, λ = 50, all (realisation, column) pairs with r ≤ 10 m of the stored "
          "A2 runs (median bias, mean r). Sounding-based truths: v1.5 T1–T3 on L18 and L24, 3 Ω·m, λ = 50, DP2–DP7 columns with median r ≤ 10 m "
          "(r of the log-midpoint — the only r stored in the v1.5 runs). Recovered q: median log10 ρ of m̂ from 4 m above the true base to the "
          "surface, and 4–10 m below it (v1.3-1).", "", "| set | q | metrics |", "|---|---|---|"]
    for k, v in d["oos_metrics"].items():
        s, q = k.split("|"); L.append(f"| {s} | {q} | {m(v)} |")
    L += ["", "| site-like truth | n | r̄ (m) | observed (m) | q true | predicted (q true) | q recovered | predicted (q recovered) |", "|---|---|---|---|---|---|---|---|"]
    for c in d["oos"]["v10_site_like"]:
        L.append(f"| {c['truth']} | {c['n']} | {c['rbar']:.2f} | {c['y']:+.2f} | {c['q_true']:.3f} | {c['pred_true']:+.2f} | {c['q_rec']:.3f} | {f2(c['pred_rec'])} |")
    L += ["", "| sounding truth | layout | DP | r̄ (m) | observed (m) | q true | predicted | q recovered | predicted |", "|---|---|---|---|---|---|---|---|---|"]
    for c in d["oos"]["v15_sounding"]:
        L.append(f"| {c['truth']} | {c['layout']} | {c['dp']} | {c['rbar']:.2f} | {c['y']:+.2f} | {c['q_true']:.3f} | {f2(c['pred_true'])} | {f3(c['q_rec'])} | {f2(c['pred_rec'])} |")
    e9 = d["E9"]; nr = d["near_midpoint_sharp"]
    L += ["", f"## E9: the {e9['n']} cases counted", "",
          "Rows of the v1.3 factorial with contour ∈ {2, 3, 5} Ω·m, λ = 50, r ≤ 10 m group, ≥ 5 (realisation, column) pairs, 0 < q < 1 and |q − 0.5| > 0.1 "
          f"(sharp and graded truths). {e9['n_ok']} of {e9['n']} have the predicted sign.", "",
          "| truth | contour | q | n pairs | median bias (m) | predicted sign | ok |", "|---|---|---|---|---|---|---|"]
    for c in e9["cases"]:
        L.append(f"| {c['truth']} | {c['contour']} | {c['q']:.3f} | {c['n']} | {c['bias']:+.2f} | {'+' if c['pred_sign'] > 0 else '−'} | {'yes' if c['ok'] else 'no'} |")
    L += ["", f"**Sharp near-midpoint cases (|q − 0.5| ≤ 0.1), excluded from E9:** {len(nr)}; too deep (bias < 0): "
          f"{sum(c['too_deep'] for c in nr)} of {len(nr)}.", "", "| truth | contour | q | median bias (m) |", "|---|---|---|---|"]
    for c in nr:
        L.append(f"| {c['truth']} | {c['contour']} | {c['q']:.3f} | {c['bias']:+.2f} |")
    lo = d["loto"]; ok = lo["sign_margin"] >= 0.9 and lo["r2_centred"] >= 0.5
    L += ["", "## Expectation A6", "", "*Logged:* LOTO sign accuracy with the margin ≥ 90 % and centred R² ≥ 0.5.", "",
          f"*Outcome:* **{'holds' if ok else 'does not hold'}** — {100 * lo['sign_margin']:.0f} % of {lo['n_margin']}, centred R² {lo['r2_centred']:.2f}.", ""]
    write("A6_bias_rule_oos.md", L)


# ─────────────────────────────────────────────────────────────────────────────
def rep_a7():
    d = J("a7.json"); keep_json("a7.json")
    meths = list(next(iter(d["counts"].values()))["DP2–DP7"])
    L = ["# A7 — sounding-base definition", "", HDR, "",
         f"Alternative bases: the 2.5 and 3 Ω·m crossings of each log (as stored); DP7 first step {d['dp7_first_step']:.2f} (stored log-midpoint) "
         f"and main step {d['dp7_main_step']:.2f} mTAW (steep-increase log-midpoint restricted to −25 … −15 mTAW); DP1 deep step "
         f"{d['dp1_deep_step']:.2f} (stored) and shallow saline layer {d['dp1_shallow']:.2f} mTAW (restricted to the top of the log … −8 mTAW; "
         "the brief's \"about −2 mTAW\" is DP1's 2.5 Ω·m crossing, −1.96). Coverage = the stored field interval of each method contains the "
         "alternative base (intervals unchanged).", ""]
    for dset in ("DP2–DP7", "DP1–DP7"):
        L += [f"## Inside the 90 % interval, {dset}", "", "| base | " + " | ".join(meths) + " |", "|---|" + "---|" * len(meths)]
        for an, v in d["counts"].items():
            L.append(f"| {an} | " + " | ".join(f"{v[dset][m]['k']}/{v[dset][m]['n']}" for m in meths) + " |")
        L.append("")
    L += ["## E15 (BEL v3, DPs at x = 100–450 m)", "", "| base | inside | holds (≥ 2/3) |", "|---|---|---|"]
    for an, v in d["e15"].items():
        L.append(f"| {an} | {v['n_ok']}/{v['n']} ({', '.join(sorted(set(v['dps']) - set(v['ok']))) or 'all'}{' outside' if set(v['dps']) - set(v['ok']) else ''}) | {'yes' if v['holds'] else 'no'} |")
    L += ["", "## 2024 bases by the 2.5 Ω·m threshold vs the 1980 saline base", "", "| DP | 2024, 2.5 Ω·m (mTAW) | 1980 (mTAW) | 2024 − 1980 (m) |", "|---|---|---|---|"]
    for n, v in d["thr2p5_vs_1980"].items():
        L.append(f"| {n} | {v['thr2p5_2024']:.2f} | {v['sal1980']:.2f} | {v['diff']:+.2f} |")
    L += ["", "At DP7 the 2.5 Ω·m rule takes the first crossing of a thin conductive layer just below the surface (−0.04 mTAW), so the DP7 "
          "row does not describe the lens base.", ""]
    mn = min(v["DP2–DP7"]["BEL3-18"]["k"] for v in d["counts"].values())
    L += ["## Expectation A7", "", "*Logged:* BEL3-18 keeps ≥ 5 of 6 under every alternative base.", "",
          f"*Outcome:* **{'holds' if mn >= 5 else 'does not hold'}** — minimum {mn}/6.", ""]
    write("A7_sounding_base.md", L)


# ─────────────────────────────────────────────────────────────────────────────
def rep_a9():
    d = J("a9.json"); keep_json("a9.json")
    p24 = os.path.join(DP, "a3_lin24.npz"); Z = np.load(p24, allow_pickle=True) if os.path.exists(p24) else None
    stop_txt = {"chi2<=1": "χ² ≤ 1 (stopAtChi1)", "dPhi<2%": "ΔΦ < 2 %", "iteration cap": "iteration cap"}
    L = ["# A9 — χ², objective and stopping for every λ; mesh", "", HDR, "",
         "pyGIMLi 1.5.5 `Inversion.run` stops (i) when χ² ≤ 1 if `stopAtChi1` is true — the default, not changed by `02_land_inversion.py` — "
         "(ii) when the relative decrease of Φ is below `minDPhi` = 2 % after iteration 2, or (iii) at `maxIter` (30). Φd = N·χ² (χ² = mean squared "
         "normalised log residual), Φm = ‖C ln m‖² with C the weighted constraint matrix (zWeight 0.5) used in the normal matrix, Φ = Φd + λΦm. "
         "Values are the final iterate, i.e. the stored model.", "",
         "## 2018 G1 line (1974 data, final mesh 2627 cells)", "",
         "| λ | run | iterations | stop | χ² | Φd | Φm | λΦm | Φ |", "|---|---|---|---|---|---|---|---|---|"]
    for o in d["pool_2018"]:
        L.append(f"| {o['lam']:g} | {o['tag']} | {o['n_iter']} | {stop_txt.get(o['stop'], o['stop'])} | {o['chi2']:.3f} | {o['phi_d']:.0f} | {o['phi_m']:.1f} | {o['lam_phi_m']:.0f} | {o['phi']:.0f} |")
    if Z is not None:
        L += ["", f"## 2025 line (ERT.dat, CPT topography, {int(Z['n_kept'])} kept data; A3 rerun with the stored seeds)", "",
              "| λ | iterations | stop | χ² | Φd | Φm | λΦm | Φ |", "|---|---|---|---|---|---|---|---|"]
        for lam in (5, 10, 20, 50, 100):
            p = f"lam{lam}_"
            L.append(f"| {lam} | {int(Z[p + 'n_iter'])} | {stop_txt.get(str(Z[p + 'stop']), str(Z[p + 'stop']))} | {float(Z[p + 'chi2']):.3f} | "
                     f"{float(Z[p + 'phi_d']):.0f} | {float(Z[p + 'phi_m']):.1f} | {lam * float(Z[p + 'phi_m']):.0f} | {float(Z[p + 'phi']):.0f} |")
    else:
        L += ["", "## 2025 line", "", "*Pending: the A3 rerun of the five λ inversions.*"]
    m = d.get("mesh", {})
    L += ["", "## Why χ² = 0.770 at λ = 20 and 0.752 at λ = 50", "",
          "Same mesh (2627 cells), same 1974 data, same error model. λ = 5–50 all stopped at the **first iterate with χ² ≤ 1** "
          "(stopAtChi1). The final χ² is then where the Gauss–Newton step that first crosses 1 happens to land, not the minimum of Φ for "
          "that λ; a larger λ can land lower. The stored models are therefore not the minimisers of their objectives; χ² < 1 by 0.23–0.25 "
          "at λ = 10–50 means they fit the data more closely than the error model implies. λ = 100 stopped on ΔΦ < 2 % with χ² = 1.02.", "",
          "**A⁻¹.** The linearised covariance A⁻¹ (A = JᵀWd²J + λCᵀC) of every λ was evaluated at these stopped models: J is the "
          "Jacobian stored after the final iteration, and `wc.cholesky_A` builds A from it. No reference inversion was rerun to convergence "
          "(as instructed).", ""]
    if m:
        L += ["## Mesh: vertical cell size at −10 … −20 mTAW (cells with centres there, x = 100–450 m)", "",
              f"2018 mesh: {m.get('n_cells_window')} of {m.get('n_cells')} cells in the window (triangles); vertical extent median {m.get('dz_median', float('nan')):.2f} m "
              f"(P5–P95 {m.get('dz_p5', float('nan')):.2f}–{m.get('dz_p95', float('nan')):.2f}, range {m.get('dz_min', float('nan')):.2f}–{m.get('dz_max', float('nan')):.2f}); "
              f"horizontal extent median {m.get('dx_median', float('nan')):.2f} m."]
        if Z is not None:
            dz = Z["mesh_dz_window"]
            L.append(f"2025 mesh: {len(dz)} cells; vertical extent median {np.median(dz):.2f} m (P5–P95 {np.percentile(dz, 5):.2f}–{np.percentile(dz, 95):.2f}).")
        L.append("")
    write("A9_chi2_stopping.md", L)


def rep_a3():
    d = J("a3.json"); keep_json("a3.json")
    S = d["summary"]; P = d["per_sounding"]; meths = list(S)
    cp = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]"
    L = ["# A3 — scores from samples, per-sounding table, exact paired tests", "", HDR, "",
         "Truth: DP log-midpoint base, DP2–DP7. Predictive samples: BEL posteriors as stored; linearised bands from the regenerated "
         "samples (2018: `11_stage4_closeout.cmd_a4` procedure, v2.0-1; 2025: the five λ inversions rerun, accepted — max band difference "
         f"{max(v for k, v in d['lin24_check']['check'].items() if k.endswith('_m')):.3f} m, model columns ≤ "
         f"{max(v for k, v in d['lin24_check']['check'].items() if 'col_maxdiff' in k):.1e} log10); priors = training targets. "
         "IS from the sample P5/P95; CRPS in energy form, E|X − y| − ½E|X − X′|, on ≤ 2000 samples (subsample seeds [200300, method, DP]). "
         "A band is scored only where the interface is defined in ≥ 50 % of the samples (as Table 3). Coverage intervals: Clopper–Pearson, 90 %.", "",
         "## Summary (Table 3 with CRPS from samples)", "",
         "| method | inside (90 % CP) | median width (m) | mean IS (m) | mean CRPS, samples (m) | mean CRPS, Table 3 (m) | CRPS skill | MAE (m) |",
         "|---|---|---|---|---|---|---|---|"]
    for m in meths:
        o = S[m]; st = np.nanmean([P[m][n]["stored_crps"] for n in DPS6])
        L.append(f"| {m} | {o['k']}/{o['n_dp']} {cp(o['cp90'])} | {o['width']:.2f} | {o['IS']:.2f} | {o['CRPS']:.3f} | {st:.3f}"
                 f"{' (3 quantiles)' if any(P[m][n]['stored_approx'] for n in DPS6) else ''} | {f2(o.get('skill_CRPS'))} | {o['MAE']:.2f} |")
    L += ["", "## Per sounding: IS / CRPS / |median − y| (m), inside", ""]
    L += ["| method | " + " | ".join(DPS6) + " |", "|---|" + "---|" * 6]
    for m in meths:
        L.append(f"| {m} | " + " | ".join(
            (f"{P[m][n]['is_']:.2f} / {P[m][n]['crps']:.2f} / {P[m][n]['ae']:.2f} {'✓' if P[m][n]['inside'] else '✗'}" if np.isfinite(P[m][n]["is_"]) else "undefined ✗")
            for n in DPS6) + " |")
    L += ["", "## Exact sign-flip tests (64 patterns; d = BEL − LIN per sounding; one-sided H1: BEL lower)", "",
          "| comparison | IS: mean d (m) | BEL lower at | p one-sided | p two-sided | CRPS: mean d (m) | BEL lower at | p one-sided | p two-sided |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in d["sign_flip"].items():
        a, b = v["IS"], v["CRPS"]
        L.append(f"| {k} | {a['mean_diff']:+.2f} | {a['n_lower']}/{a['n']} | {a['p_one']:.3f} | {a['p_two']:.3f} | {b['mean_diff']:+.3f} | "
                 f"{b['n_lower']}/{b['n']} | {b['p_one']:.3f} | {b['p_two']:.3f} |")
    L += ["", "With six soundings the smallest attainable one-sided p is 1/64 = 0.016.", "",
          "## Leave-one-sounding-out ranking (14 methods)", ""]
    for k in ("IS", "CRPS"):
        lo = d["loo"][k]
        L.append(f"- **{k}**: full-data rank BEL3-18 {lo['full_rank']['BEL3-18']}, BEL4-24 {lo['full_rank']['BEL4-24']}; with DP2 … DP7 left out: "
                 f"BEL3-18 {lo['ranks']['BEL3-18']}, BEL4-24 {lo['ranks']['BEL4-24']}. Still lower than the linearised method in all six leave-outs: "
                 + ", ".join(f"{kk} {'yes' if vv else 'no (' + str(lo['beats_count'][kk]) + '/6)'}" for kk, vv in lo["beats_in_all_6"].items()) + ".")
    L += ["", "## Representativeness error σ_rep added to every predictive distribution", "",
          "| method | inside, 0 / 0.5 / 1.0 m | mean IS, 0 / 0.5 / 1.0 m | mean CRPS, 0 / 0.5 / 1.0 m |", "|---|---|---|---|"]
    R5, R1 = d["sigma_rep"]["0.5"]["summary"], d["sigma_rep"]["1.0"]["summary"]
    for m in meths:
        L.append(f"| {m} | {S[m]['k']} / {R5[m]['k']} / {R1[m]['k']} | {S[m]['IS']:.2f} / {R5[m]['IS']:.2f} / {R1[m]['IS']:.2f} | "
                 f"{S[m]['CRPS']:.3f} / {R5[m]['CRPS']:.3f} / {R1[m]['CRPS']:.3f} |")
    sf = d["sign_flip_sigma_rep"]
    L += ["", "Sign-flip p (one-sided, IS) with σ_rep 0.5 / 1.0 m: " + "; ".join(
        f"{k} {sf['0.5'][k]['IS']['p_one']:.3f} / {sf['1.0'][k]['IS']['p_one']:.3f}" for k in sf["0.5"]) + ".", "",
          "## Clopper–Pearson 90 % intervals for other coverage counts", "",
          "Held-out sets: " + "; ".join(f"{k} {v['k']}/{v['n']} {cp(v['cp90'])}" for k, v in d["cp90_heldout"].items()) + ".", ""]
    if "cp90_a7" in d:
        L += ["A7 alternative bases, BEL3-18 and BEL4-24 (DP2–DP7 | DP1–DP7):", ""]
        for an, v in d["cp90_a7"].items():
            L.append(f"- {an}: " + "; ".join(f"{m} {v[ds][m]['k']}/{v[ds][m]['n']} {cp(v[ds][m]['cp90'])}" for m in ("BEL3-18", "BEL4-24") for ds in ("DP2–DP7", "DP1–DP7")))
    e1 = d["sign_flip"]["BEL4-24 vs LIN-LM-24"]["IS"]; e2 = d["sign_flip"]["BEL3-18 vs LIN-3-18"]["IS"]
    L += ["", "## Expectation A3", "", "*Logged:* BEL4-24 has lower IS than LIN-LM-24 at all 6 DPs (one-sided p = 1/64); the BEL3-18 vs LIN-3-18 differences are not significant (p > 0.1).", "",
          f"*Outcome:* first part **{'holds' if e1['n_lower'] == 6 else 'does not hold'}** — BEL4-24 lower at {e1['n_lower']} of 6 (p = {e1['p_one']:.3f}; "
          "LIN-LM-24 is lower at DP2, where its band is narrow and covers); second part **"
          f"{'holds' if e2['p_two'] > 0.1 else 'does not hold'}** (two-sided p = {e2['p_two']:.2f}).", ""]
    write("A3_scores_tests.md", L)


def rep_a4():
    d = J("a4.json"); keep_json("a4.json")
    T = d["treatments"]
    L = ["# A4 — outlier treatments and error model at the soundings (Table S7)", "", HDR, "",
         "λ pools of 5 runs (λ = 5, 10, 20, 50, 100) per treatment on the same mesh; 1000 linearised samples per run (seeds [200400, treatment, λ]; "
         "G1: the A3 regeneration); bands at the 2 m column nearest each DP. BEL v3 can be retrained only within the 1974 G1 quadrupoles, "
         "because training responses exist only for them (posterior seeds [200500, treatment]).", "",
         "## What the BEL retraining changes", "",
         "| treatment | data for the inversions | BEL v3 |", "|---|---|---|",
         "| G1 | 1974 (357 outliers dropped) | the paper's BEL (1718 training samples) |",
         f"| G1b | {T['G1b']['runs']['50']['n_data'] if '50' in T['G1b']['runs'] else T['G1b']['runs'][50]['n_data']} | **identical to G1's BEL**: the 62 data G1b keeps beyond G1 have no training responses |"]
    def b(n):
        return T[n]["bel"]
    g2, g3, rc = b("G2"), b("G3"), b("RECIP")
    L.append(f"| G2 | 1456 (all data on the 11 flagged electrodes dropped) | {g2['n_dropped_cols']} of the 1974 G1 quadrupoles dropped (those on the 11 "
             f"electrodes); training data built on all 1974 and restricted to the remaining {g2['n_cols']}; field vector restricted the same way; "
             f"{g2['n_train']} training samples, {g2['kd']} data PCs |")
    L.append(f"| G3 | 2331 (none dropped; 875 down-weighted ×2.07) | all 1974 columns; training noise σ_i × the G3 factor (2.07 on the {g3['n_scaled']} G1 data "
             f"that use a flagged electrode, 1 elsewhere), linear space as pre-registered; {g3['n_train']} training samples, {g3['kd']} data PCs |")
    L.append(f"| RECIP | 1974, reciprocal error model B | all 1974 columns; training noise σ_i × the reciprocal factor ({rc['scale_min']:.2f}–{rc['scale_max']:.1f}), "
             f"applied in log space (decision log v2.0-2); {rc['n_train']} training samples, {rc['kd']} data PCs |")
    L += ["", "## Inversion runs", "", "| treatment | λ | run | data | χ² | iterations | stop |", "|---|---|---|---|---|---|---|"]
    for n, v in T.items():
        if n == "G1":
            continue
        for lam, r in v["runs"].items():
            L.append(f"| {n} | {lam} | {r['tag']} | {r['n_data']} | {r['chi2']:.3f} | {r['n_iter']} | {r['stop']} |")
    names = list(T)
    L += ["", "## Table S7 — lens base (median, mTAW) and inside the 90 % interval at DP1–DP7", "",
          "| DP | DP base | " + " | ".join(f"{n}: 3 Ω·m λ50 / log-mid. pooled / BEL" for n in names) + " |", "|---|---|" + "---|" * len(names)]
    dps = list(T["G1"]["rows"])
    mk = lambda r: (f"{r['p50']:.2f}{'✓' if r['inside'] else '✗'}" if np.isfinite(r["p50"]) and r["share"] >= 0.5 else "—✗")
    for n_ in dps:
        y = T["G1"]["rows"][n_]["y"]
        L.append(f"| {n_} | {y:.2f} | " + " | ".join(f"{mk(T[t]['rows'][n_]['thr3_lam50'])} / {mk(T[t]['rows'][n_]['logmid_pooled'])} / {mk(T[t]['rows'][n_]['bel'])}" for t in names) + " |")
    L.append("| **inside, DP2–DP7** | | " + " | ".join(f"**{T[t]['counts']['thr3_lam50']} / {T[t]['counts']['logmid_pooled']} / {T[t]['counts']['bel']}**" for t in names) + " |")
    e = d["expectation"]
    L += ["", "## Expectation A4", "", "*Logged:* BEL v3 inside at ≥ 5 of 6 for every treatment; the linearised counts change by at most one DP across treatments.", "",
          f"*Outcome:* **{'holds' if e['holds'] else 'does not hold'}** — BEL minimum {e['bel_min']}/6; linearised count ranges: 3 Ω·m {e['lin_range']['thr3_lam50']}, "
          f"log-midpoint {e['lin_range']['logmid_pooled']}.", ""]
    write("A4_treatments_TableS7.md", L)


def rep_a10():
    L = ["# A10 (optional) — blocky inversion and BEL v3 on factorial truths", "", HDR, ""]
    if os.path.exists(os.path.join(DP, "a10a.json")):
        a = J("a10a.json"); keep_json("a10a.json")
        L += ["## (a) G1 data, λ = 50, L1 (IRLS) model roughness", "",
              f"{a['n_data']} data; χ² = {a['chi2']:.3f} ({a['stop']}, {a['n_iter']} iterations); run time {a['t_inversion_s'] / 60:.1f} min (inversion), "
              f"{a['t_total_s'] / 60:.1f} min in total. 1000 linearised samples, seed [200600, 0]. *Caveat:* {a['caveat']}", "",
              "| DP | DP base | 3 Ω·m P50 [P5, P95] | inside | log-midpoint P50 [P5, P95] | inside |", "|---|---|---|---|---|---|"]
        for n, r in a["rows"].items():
            c = lambda k: (f"{r[k]['p50']:.2f} [{r[k]['p5']:.2f}, {r[k]['p95']:.2f}] | {'✓' if r[k]['inside_logmid'] else '✗'}" if "p50" in r[k] else "undefined | ✗")
            L.append(f"| {n} | {r['y_logmid']:.2f} | {c('thr3')} | {c('logmid')} |")
        cn = a["counts_dp2_dp7"]
        L += ["", f"Inside at DP2–DP7 (truth: DP log-midpoint): 3 Ω·m {cn['thr3']['vs_logmid']}/6, log-midpoint {cn['logmid']['vs_logmid']}/6 "
                  f"(against the DP 3 Ω·m crossing: {cn['thr3']['vs_thr3']}/6 and {cn['logmid']['vs_thr3']}/6).", ""]
    if os.path.exists(os.path.join(DP, "a10b.json")):
        b = J("a10b.json"); keep_json("a10b.json")
        L += ["## (b) BEL v3 (paper fit) on the 8 sharp factorial truths with z_b = −12 m", "",
              f"Realisations 0–4 of the stored v1.3 noisy data (G1 quadrupoles); posterior seeds [200700, truth, realisation]; columns x = 130–450 m; "
              f"run time {b['t_total_s'] / 60:.1f} min.", "",
              "| truth | ρ_lens | k (ρ_below) | coverage | mean IS (m) | prior IS (m) | IS skill | median bias (m) |", "|---|---|---|---|---|---|---|---|"]
        for t, v in b["truths"].items():
            L.append(f"| {t} | {v['rho_lens']:g} | {v['k']} ({v['rho_lens'] * v['k']:.2f} Ω·m) | {v['cov']:.2f} | {v['IS']:.2f} | {v['IS_prior']:.2f} | "
                     f"{v['skill_IS']:+.2f} | {v['median_bias']:+.2f} |")
        L += ["", "For k = 2 the water below the lens is 1.6 Ω·m (ρ_lens 0.8) or 2.84 Ω·m (ρ_lens 1.42); prior v3 draws ρ_below from 2–20 Ω·m and requires "
                  "ρ_lens ≥ 0.69 Ω·m, so the 0.8 × 2 truth lies outside the prior and the 1.42 × 2 truth at its low edge. The low-contrast "
                  "truths are where BEL misses (coverage 0.44, medians ≈ 5 m too deep); for k ≥ 4 it covers 97–100 % with IS skill ≈ +0.58.", ""]
    write("A10_optional.md", L)


def rep_a1():
    d = J("a1.json"); keep_json("a1.json")
    s = d["summary"]
    L = ["# A1 — BEL with prior v3 (unchanged) on the 2025 line", "", HDR, "",
         f"Ensemble: {d['n_have']} of the {d['n_expected']} prior-v3 draws forward-modelled on the L24 geometry; {d['n_dual']} dual (fine/medium) samples "
         f"for the approximation error (median |mean| {d['approx']['mean_abs_median']:.4f}, median sd {d['approx']['sd_median']:.4f} log10). "
         f"Training {d['n']['tr']['n_kept']} (of {d['n']['tr']['n_all']}), held-out {d['n']['ho']['n_kept']} (of {d['n']['ho']['n_all']}) after the "
         f"site and physical filters; {d['kd']} data / {d['kt']} target PCs.", "",
         f"*Checks.* Prior falsification of the 2025 data: share of training samples with a smaller statistic — T² {d['falsification']['t2']:.3f}, "
         f"robust T² {d['falsification']['t2r']:.3f}, Q {d['falsification']['q']:.3f} (not outside). Held-out coverage {100 * d['heldout']['cov']:.1f} % "
         f"({d['heldout']['cov_k']}/{d['heldout']['cov_n']} columns), median width {d['heldout']['width_median']:.2f} m; second BEL ρ_below held-out "
         f"coverage {100 * d['heldout']['cov2_rho_below']:.1f} %. ρ_below (second BEL, v3 bounds): {d['rho_below'][0]:.2f} [{d['rho_below'][1]:.2f}, {d['rho_below'][2]:.2f}] Ω·m.", "",
         "## Field read-out at DP2–DP7 (truth: DP log-midpoint base)", "",
         "| DP | column | DP base | BEL P50 [P5, P95] | inside | IS | CRPS | prior P5–P95 | prior IS | prior CRPS |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in d["rows"]:
        bb, pp = r["bel"], r["prior"]
        L.append(f"| {r['dp']} | {r['col']:.0f} | {r['y']:.2f} | {bb['m']:.2f} [{bb['l']:.2f}, {bb['u']:.2f}] | {'✓' if r['inside'] else '✗'} | {bb['is_']:.2f} | "
                 f"{bb['crps']:.3f} | {pp['l']:.2f} – {pp['u']:.2f} | {pp['is_']:.2f} | {pp['crps']:.3f} |")
    L += ["", f"**Coverage {s['k']}/6** (90 % CP [{s['cp90'][0]:.2f}, {s['cp90'][1]:.2f}]); median width {s['median_width']:.2f} m; mean IS {s['IS']:.2f} m "
              f"(prior {s['prior_IS']:.2f}; skill {s['skill_IS']:+.2f}); mean CRPS {s['CRPS']:.3f} m (prior {s['prior_CRPS']:.3f}; skill {s['skill_CRPS']:+.2f}). "
              "Prior CRPS subsample seeds [160000, prior set, 500 + column] (not fixed in v2.0-0).", "",
          "## Expectation A1", "", "*Logged:* ≥ 5 of 6 inside and IS skill > 0.", "",
          f"*Outcome:* **{'holds' if d['expectation']['holds'] else 'does not hold'}**.", ""]
    write("A1_bel_v3_2025.md", L)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        globals()["rep_" + a]()
