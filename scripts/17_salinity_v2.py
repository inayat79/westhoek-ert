"""
17_salinity_v2.py
=================
Westhoek ERT — Stage 6 revision: seawater resistivity tied to temperature, per-unit salinity products.

  run [--out ROOT]   field + held-out salinity with the v2 Archie model → ROOT/results/stage6_v2/*.md, *.csv and
                     ROOT/data/processed/stage6_v2/*.npz (ROOT defaults to the project; figures are made by 18_paper.py)

Archie v2 (per posterior sample): F ~ U(2.8, 3.6); salinity S ~ U(31, 34) PSU → EC25_sw from the practical salinity
scale PSS-78 (UNESCO 1981; conductivity ratio inverted at 25 °C, C(35, 15 °C, 0) = 42.914 mS/cm); groundwater
temperature T ~ U(8, 14) °C; ρ_sw = 1 / (EC25_sw · (1 + 0.02 (T − 25))) with the SAME T as the groundwater EC25
correction EC25 = EC_T / (1 + 0.02 (T − 25)). % seawater = 100 ρ_sw / ρ_w, ρ_w = ρ_bulk / F.

Physical constraint (joint): a draw (posterior sample, F, S, T) is REJECTED if any unit present has ρ_w < ρ_sw
(> 100 % seawater). Posterior samples are paired with fresh Archie draws in rounds (the same stream) until the target
number of accepted draws is reached — rejection sampling of p(θ, F, S, T | constraint). Held-out truths: Archie redrawn
(up to 1000 tries) until physical; truths that no draw can make physical are excluded from the salinity coverage.

Posterior: the Stage-6 joint BEL (site subset, prior v3), identical samples (seeds default_rng([129000, k]) as in
stage6_analysis). Archie v2 draws default_rng([131000, k]): field k = 0; fixed F k = 10·F; ρ_below bins k = 10 + bin;
held-out posterior k = 200000 + j; held-out truth k = 500000 + j. Draw order F, S, T (F overridden when fixed, so S and
T are paired across the F cases).

Units (geometry from the joint sample; 1 m grid, water table / surface down to the deep-unit top):
  saline lens      x ≥ x_L, above the lens base (incl. the saline skin over the discharge zone)
  below the lens   x ≥ x_L, below the base, outside the discharge body
  discharge zone   discharge body (x_S … x_S + width, skin base … dis_bottom), where present
  dune aquifer     x < x_L, below the water table (the dune conductive layer is not built)
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import read_trace  # noqa: E402
read_trace.install("17_salinity_v2.py")
import westhoek_common as wc  # noqa: E402
import stage6_analysis as s6  # noqa: E402

ARCHIE2_SEED = 131000
JOINT_SEED = 129000
N_POST = 2000
N_HELD_POST = 500
S_RANGE, T_RANGE, F_RANGE = (31.0, 34.0), (8.0, 14.0), (2.8, 3.6)
ALPHA = 0.02
UNITS = ("lens", "below", "dis", "dune")
UNIT_NAMES = {"lens": "saline lens", "below": "water below the lens", "dis": "discharge-zone water",
              "dune": "dune aquifer"}
UNIT_RHO = {"lens": "rho_lens", "below": "rho_below", "dis": "rho_discharge", "dune": "rho_aquifer"}
XREP = (150, 200, 250, 300, 350, 400, 450)


# ─────────────────────────────────────────────────────────────────────────────
# PSS-78
# ─────────────────────────────────────────────────────────────────────────────
_A = (0.0080, -0.1692, 25.3851, 14.0941, -7.0261, 2.7081)
_B = (0.0005, -0.0056, -0.0066, -0.0375, 0.0636, -0.0144)
_K = 0.0162
_C = (0.6766097, 2.00564e-2, 1.104259e-4, -6.9698e-7, 1.0031e-9)
C35_15 = 42.914                     # mS/cm


def _S_of_Rt(Rt, t):
    r = np.sqrt(Rt)
    a = sum(_A[i] * r ** i for i in range(6)); b = sum(_B[i] * r ** i for i in range(6))
    return a + (t - 15) / (1 + _K * (t - 15)) * b


def ec_pss78(S, t):
    """Seawater conductivity (mS/cm, p = 0) for practical salinity S at t °C (PSS-78 inverted)."""
    rt = sum(_C[i] * t ** i for i in range(5))
    return brentq(lambda R: _S_of_Rt(R, t) - S, 1e-4, 3.0) * rt * C35_15


_SG = np.linspace(30.0, 35.0, 501)
_ECG = np.array([ec_pss78(s, 25.0) for s in _SG])


def ec25_sw(S):
    """EC25 of seawater (S/m) for salinity S (PSU), PSS-78 (tabulated, linear interpolation, 0.01 PSU step)."""
    return np.interp(S, _SG, _ECG) / 10.0


def archie_v2(n, rng, F=None):
    Fd = rng.uniform(*F_RANGE, n); S = rng.uniform(*S_RANGE, n); T = rng.uniform(*T_RANGE, n)
    if F is not None:
        Fd = np.full(n, float(F))
    ec = ec25_sw(S)
    return dict(F=Fd, S=S, T=T, ec25_sw=ec, rsw=1.0 / (ec * (1 + ALPHA * (T - 25))))


# ─────────────────────────────────────────────────────────────────────────────
# sections
# ─────────────────────────────────────────────────────────────────────────────
def unit_masks(base, par):
    ex, ez = wc.land_surface()
    s = np.interp(s6.XG, ex, ez)
    Xc, Zc = np.meshgrid(s6.XG, s6.ZG)
    bx = np.array([np.interp(s6.XG, s6.TX, b) for b in base])
    P = lambda k: np.asarray(par[k], float)
    top = np.minimum(s[None, :], P("z_wt")[:, None])
    valid = (Zc[None] < top[:, None, :]) & (Zc[None] > P("z_deep")[:, None, None])
    L = lambda k: P(k)[:, None, None]
    dune = Xc[None] < L("x_L")
    lens = ~dune & (Zc[None] > bx[:, None, :])
    x1 = np.minimum(L("x_S") + L("dis_width"), 505.0)
    dis = (np.asarray(par["dis_present"])[:, None, None] & ~dune & ~lens & (Xc[None] >= L("x_S")) & (Xc[None] < x1)
           & (Zc[None] > L("dis_bottom")))
    below = ~dune & ~lens & ~dis
    return dict(valid=valid, lens=lens & valid, below=below & valid, dis=dis & valid, dune=dune & valid)


def unit_values(par, a):
    """Per-sample % seawater and EC25 (μS/cm) of each unit (homogeneous units)."""
    out = {}
    for u in UNITS:
        rho = np.asarray(par[UNIT_RHO[u]], float)
        rw = rho / a["F"]
        out[u] = dict(sw=100 * a["rsw"] / rw, ec25=1e4 / rw / (1 + ALPHA * (a["T"] - 25)))
    return out


def sections(base, par, a):
    M = unit_masks(base, par)
    uv = unit_values(par, a)
    sw = np.full(M["valid"].shape, np.nan); ec = np.full(M["valid"].shape, np.nan)
    for u in UNITS:
        sw = np.where(M[u], uv[u]["sw"][:, None, None], sw)
        ec = np.where(M[u], uv[u]["ec25"][:, None, None], ec)
    return M, uv, sw, ec


def presence(base, par, chunk=250):
    """Per posterior sample: does each unit exist in the section?"""
    out = {u: [] for u in UNITS}
    for i in range(0, base.shape[0], chunk):
        sl = slice(i, i + chunk)
        M = unit_masks(base[sl], {k: np.asarray(v)[sl] for k, v in par.items()})
        for u in UNITS:
            out[u].append(M[u].any(axis=(1, 2)))
    return {u: np.concatenate(v) for u, v in out.items()}


def violations(par, pres, a, idx=None):
    """Boolean (n,) per unit: unit present and ρ_w = ρ_unit / F < ρ_sw (> 100 % seawater)."""
    sel = slice(None) if idx is None else idx
    return {u: pres[u][sel] & (np.asarray(par[UNIT_RHO[u]], float)[sel] / a["F"] < a["rsw"]) for u in UNITS}


def constrained(base, par, rng, F=None, n_target=None, max_rounds=100):
    """Rejection sampling of (posterior sample, Archie draw) pairs under the ρ_w ≥ ρ_sw constraint (all units).
    Returns (base, par, draws) of the accepted pairs (≤ n_target) and the rejection statistics."""
    n = base.shape[0]; n_target = n_target or n
    pres = presence(base, par)
    idx, draws = [], []
    n_prop = n_rej = 0; viol = {u: 0 for u in UNITS}; npres = {u: 0 for u in UNITS}
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        a = archie_v2(n, rng, F)
        v = violations(par, pres, a)
        bad = np.zeros(n, bool)
        for u in UNITS:
            viol[u] += int(v[u].sum()); npres[u] += int(pres[u].sum()); bad |= v[u]
        n_prop += n; n_rej += int(bad.sum())
        ok = np.where(~bad)[0]
        idx.append(ok); draws.append({k: x[ok] for k, x in a.items()})
        if sum(len(x) for x in idx) >= n_target:
            break
    idx = np.concatenate(idx)[:n_target]
    A = {k: np.concatenate([d[k] for d in draws])[:n_target] for k in draws[0]}
    info = dict(n_proposed=n_prop, n_accepted=len(idx), rounds=rounds, joint_rate=n_rej / n_prop,
                unit_rate={u: viol[u] / npres[u] if npres[u] else float("nan") for u in UNITS},
                n_distinct=int(len(np.unique(idx))))
    return base[idx], {k: np.asarray(v)[idx] for k, v in par.items()}, A, info


def truth_draw(tpar, tpres, rng, max_tries=1000):
    """Archie draw for a held-out truth, redrawn until physical; None if no draw within max_tries is."""
    for k in range(1, max_tries + 1):
        a = archie_v2(1, rng)
        if not any(v[0] for v in violations(tpar, tpres, a).values()):
            return a, k
    return None, max_tries


def stats(base, par, DR, chunk=250):
    """Section statistics for aligned (posterior sample, Archie draw) pairs; DR = accepted Archie draws."""
    n = base.shape[0]
    Psal = Pfr = Nv = None; iso = []; car = None
    ua = {u: [] for u in UNITS}; usw = {u: [] for u in UNITS}; uec = {u: [] for u in UNITS}; draws = []
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        p = {k: np.asarray(v)[sl] for k, v in par.items()}
        a = {k: v[sl] for k, v in DR.items()}
        draws.append(a)
        M, uv, sw, ec = sections(base[sl], p, a)
        fin = np.isfinite(sw)
        ps, pf, nv = (sw > 50).sum(0), (ec < 1500).sum(0), fin.sum(0)
        Psal = ps if Psal is None else Psal + ps; Pfr = pf if Pfr is None else Pfr + pf; Nv = nv if Nv is None else Nv + nv
        iso.append(s6.isohaline(sw))
        A = s6.areas(sw, ec)
        car = A if car is None else {k: np.concatenate([car[k], A[k]]) for k in A}
        for u in UNITS:
            ua[u].append(M[u].sum(axis=(1, 2)).astype(float))
            pres = M[u].any(axis=(1, 2))
            usw[u].append(np.where(pres, uv[u]["sw"], np.nan)); uec[u].append(np.where(pres, uv[u]["ec25"], np.nan))
    cat = lambda d: {u: np.concatenate(v) for u, v in d.items()}
    with np.errstate(invalid="ignore"):
        return dict(Psal=Psal / Nv, Pfresh=Pfr / Nv, Nv=Nv, iso=np.vstack(iso), areas=car, uarea=cat(ua),
                    usw=cat(usw), uec=cat(uec), draws={k: np.concatenate([d[k] for d in draws]) for k in draws[0]})


def q(v, d=1):
    return s6.q(v, d)


def uprob(st, u):
    sw, ec = st["usw"][u], st["uec"][u]
    ok = np.isfinite(sw)
    return (float(np.mean(ok)), float(np.mean(sw[ok] > 50)) if ok.any() else np.nan,
            float(np.mean(ec[ok] < 1500)) if ok.any() else np.nan)


# ─────────────────────────────────────────────────────────────────────────────
def cmd_run(args):
    t_start = time.time()
    root = os.path.abspath(args.out)
    RES = os.path.join(root, "results", "stage6_v2"); DC = os.path.join(root, "data", "processed", "stage6_v2")
    for d in (RES, DC):
        os.makedirs(d, exist_ok=True)
    s6.init(); s6.TX = s6.S16.TX
    # ── 1. seawater resistivity ─────────────────────────────────────────────
    ec_lo, ec_hi = ec25_sw(S_RANGE[0]) * 10, ec25_sw(S_RANGE[1]) * 10
    rsw_min = 1 / (ec25_sw(S_RANGE[1]) * (1 + ALPHA * (T_RANGE[1] - 25)))
    rsw_max = 1 / (ec25_sw(S_RANGE[0]) * (1 + ALPHA * (T_RANGE[0] - 25)))
    chk = [(S, T, ec_pss78(S, T) / 10, ec25_sw(S) * (1 + ALPHA * (T - 25))) for S in (31.0, 34.0) for T in (8.0, 14.0)]
    C = s6.load_combined()
    obs, _ = s6.b13.observed_sets()
    J = s6.JointBEL(C["tr"]["D"], C["tr"]["T"], C["tr"]["th"])
    base_f, par_f, pdis_f = J.posterior(obs["field"], N_POST, np.random.default_rng([JOINT_SEED, 0]))
    base_c, par_c, A_c, info_f = constrained(base_f, par_f, np.random.default_rng([ARCHIE2_SEED, 0]))
    st = stats(base_c, par_c, A_c)
    rs = st["draws"]["rsw"]
    L1 = ["# Stage 6 revision — 1. Seawater resistivity tied to temperature", "",
          "Salinity S ~ U(31, 34) PSU → EC25 of seawater from the **standard seawater relation PSS-78** (UNESCO 1981: the "
          "practical-salinity polynomial inverted for the conductivity ratio at 25 °C, times C(35, 15 °C, 0) = 42.914 "
          f"mS/cm · r_t(25 °C); C(35, 25 °C) = {ec_pss78(35, 25):.3f} mS/cm). The approximation 53.1 mS/cm × S/35 gives "
          f"{53.1*31/35:.2f} / {53.1*34/35:.2f} mS/cm at S = 31 / 34 (PSS-78: {ec_lo:.2f} / {ec_hi:.2f}); it is not used.",
          "", f"ρ_sw = 1 / (EC25_sw · (1 + {ALPHA} (T − 25))) with T ~ U(8, 14) °C, the SAME T sample as the groundwater "
          "EC25 correction; F ~ U(2.8, 3.6). Draws default_rng([131000, k]), order F, S, T.", "",
          f"- EC25 of seawater: **{ec_lo:.2f}–{ec_hi:.2f} mS/cm** ({ec_lo/10:.3f}–{ec_hi/10:.3f} S/m).",
          f"- ρ_sw range (bounds of the inputs): **{rsw_min:.3f}–{rsw_max:.3f} Ω·m** (S = 34, T = 14 °C … S = 31, T = 8 °C); "
          f"field draws ({len(rs)}): median {np.median(rs):.3f}, P5–P95 {np.percentile(rs, 5):.3f}–{np.percentile(rs, 95):.3f} "
          "Ω·m. (Stage 6 used ρ_sw ~ U(0.2, 0.3) Ω·m, independent of T.)",
          "- Check of the linear temperature compensation against PSS-78 at in-situ T (EC_T in S/m, PSS-78 vs "
          "linear): " + "; ".join(f"S {S:g}, T {T:g} °C: {a:.3f} vs {b:.3f} ({100*(b/a-1):+.1f} %)" for S, T, a, b in chk) + "."]
    with open(os.path.join(RES, "1_seawater_resistivity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L1) + "\n")
    print("\n".join(L1), flush=True)

    # ── 2. per-unit products (field) ────────────────────────────────────────
    L = ["# Stage 6 revision — 2. Salinity products (Archie v2: ρ_sw tied to T)", "",
         f"Joint BEL (site subset, prior v3; {len(C['tr']['D'])} training samples), {N_POST} field posterior samples "
         f"(seeds as Stage 6, P(discharge present) = {pdis_f:.2f}); Archie v2 per sample (see 1_seawater_resistivity.md). "
         "Units as defined in 17_salinity_v2.py; the dune zone (x < 90 m) is weakly constrained.", "",
         "## Per-unit products (field)", "",
         "## Physical constraint: ρ_w ≥ ρ_sw (≤ 100 % seawater) in every unit", "",
         f"Joint rejection of (posterior sample, F, S, T) draws: {info_f['n_proposed']} proposed in {info_f['rounds']} "
         f"round(s), **{100*info_f['joint_rate']:.1f} % rejected**, {info_f['n_accepted']} accepted ({info_f['n_distinct']} "
         "distinct posterior samples). Rejection rate per unit (share of proposed draws in which the unit exists and "
         "exceeds 100 % seawater):", "",
         "| unit | rejection rate |", "|---|---|"]
    for u in UNITS:
        L.append(f"| {UNIT_NAMES[u]} | {100*info_f['unit_rate'][u]:.1f} % |")
    lens_rate = info_f["unit_rate"]["lens"]
    L += ["", (f"**More than 10 % of the lens draws are rejected ({100*lens_rate:.1f} %).**" if lens_rate > 0.10 else
               f"Lens rejection rate {100*lens_rate:.1f} % — below 10 %."),
          f"Posterior samples of ρ_lens below F·ρ_sw can only be accepted with a large F / small ρ_sw: the lower bound "
          f"ρ_lens ≥ {F_RANGE[0] * rsw_min:.3f} Ω·m (F = 2.8, ρ_sw = {rsw_min:.3f}) is physically binding for any draw; "
          f"the prior on ρ_lens (0.4–3.0 Ω·m, log-uniform) puts "
          f"{100*np.log(F_RANGE[0]*rsw_min/0.4)/np.log(3.0/0.4):.0f} % of its mass below it.", "",
          "## Per-unit products (field, constrained; median [P5, P95])", "",
          "| unit | area (m² per m of coast) | % seawater | EC25 (μS/cm) | P(> 50 % seawater) | P(EC25 < 1500 μS/cm) |",
          "|---|---|---|---|---|---|"]
    rows_csv = ["unit,area_p5,area_p50,area_p95,sw_p5,sw_p50,sw_p95,ec25_p5,ec25_p50,ec25_p95,p_present,p_sw50,p_fresh"]
    for u in UNITS:
        pres, psal, pfr = uprob(st, u)
        L.append(f"| {UNIT_NAMES[u]}{' (weakly constrained)' if u == 'dune' else ''} | {q(st['uarea'][u], 0)} | "
                 f"{q(st['usw'][u], 1)} | {q(st['uec'][u], 0)} | {psal:.2f} | {pfr:.2f} |")
        pc = lambda v: np.nanpercentile(v, [5, 50, 95])
        rows_csv.append(",".join([u] + [f"{v:.4g}" for v in (*pc(st["uarea"][u]), *pc(st["usw"][u]), *pc(st["uec"][u]))]
                                 + [f"{pres:.4f}", f"{psal:.4f}", f"{pfr:.4f}"]))
    L += ["", "Area = geometry only (independent of the Archie parameters). % seawater, EC25 and the probabilities "
          "use the samples in which the unit exists (discharge zone: P(present) = "
          f"{uprob(st, 'dis')[0]:.2f})."]
    with open(os.path.join(RES, "per_unit_field.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows_csv) + "\n")

    # ── (b) isohaline ───────────────────────────────────────────────────────
    iso = st["iso"]
    L += ["", "## (b) 50 % seawater isohaline vs the BEL parametric base (mTAW)", "",
          "| x | isohaline median [P5, P95] (share with a saline top) | parametric base (joint) |", "|---|---|---|"]
    for x in XREP:
        k = int(np.argmin(np.abs(s6.XG - x))); j = int(np.argmin(np.abs(s6.TX - x)))
        L.append(f"| {x} | {q(iso[:, k], 2)} ({100*np.mean(np.isfinite(iso[:, k])):.0f} %) | {q(base_c[:, j], 2)} |")
    # ── (c) class areas (appendix) ──────────────────────────────────────────
    A = st["areas"]
    L += ["", "## (c) Class areas (appendix; m² per m of coast)", "",
          "| class | dunes x 0–90 m (weakly constrained) | beach x 90–505 m |", "|---|---|---|"]
    for c in ("saline", "brackish", "fresh"):
        L.append(f"| {c} | {q(A[f'{c}_dunes'], 0)} | {q(A[f'{c}_beach'], 0)} |")
    with open(os.path.join(RES, "class_areas_field.csv"), "w", encoding="utf-8") as f:
        f.write("class,zone,p5,p50,p95\n" + "".join(
            f"{c},{z},{np.percentile(A[f'{c}_{z}'], 5):.4g},{np.percentile(A[f'{c}_{z}'], 50):.4g},"
            f"{np.percentile(A[f'{c}_{z}'], 95):.4g}\n" for c in ("saline", "brackish", "fresh") for z in ("dunes", "beach")))

    # ── (d) sensitivity ─────────────────────────────────────────────────────
    L += ["", "## (d) Sensitivity (medians; constrained draws)", "",
          "| case | joint rejection | isohaline x = 150 / 250 / 350 | % seawater lens / below / discharge | saline beach | "
          "brackish beach | fresh beach | fresh dunes |", "|---|---|---|---|---|---|---|---|"]
    sens = {}

    def srow(name, stt, info):
        a_ = stt["areas"]; iso_ = stt["iso"]
        sens[name] = dict(iso=[float(np.nanmedian(iso_[:, int(np.argmin(np.abs(s6.XG - x)))])) for x in (150, 250, 350)],
                          sw=[float(np.nanmedian(stt["usw"][u])) for u in ("lens", "below", "dis")],
                          areas={k: float(np.median(v)) for k, v in a_.items()}, joint_rate=info["joint_rate"],
                          lens_rate=info["unit_rate"]["lens"])
        L.append(f"| {name} | {100*info['joint_rate']:.1f} % | " + " / ".join(f"{v:.2f}" for v in sens[name]["iso"]) + " | "
                 + " / ".join(f"{v:.1f}" for v in sens[name]["sw"])
                 + f" | {np.median(a_['saline_beach']):.0f} | {np.median(a_['brackish_beach']):.0f} | "
                 f"{np.median(a_['fresh_beach']):.0f} | {np.median(a_['fresh_dunes']):.0f} |")
    srow("reference (F ~ U(2.8, 3.6))", st, info_f)
    for Fv in (2.8, 3.2, 3.6):
        bF, pF, AF, iF = constrained(base_f, par_f, np.random.default_rng([ARCHIE2_SEED, int(round(Fv * 10))]), F=Fv)
        srow(f"F = {Fv}", stats(bF, pF, AF), iF)
    rb = np.array([t["rho_below"] for t in C["tr"]["th"]])
    bins = {}
    for k, (lo_, hi_) in enumerate(((2, 4), (4, 8), (8, 20))):
        m = (rb >= lo_) & (rb < hi_)
        Jb = s6.JointBEL(C["tr"]["D"][m], C["tr"]["T"][m], [t for t, mm in zip(C["tr"]["th"], m) if mm])
        bb, pb, _ = Jb.posterior(obs["field"], N_POST, np.random.default_rng([JOINT_SEED, 10 + k]))
        bB, pB, AB, iB = constrained(bb, pb, np.random.default_rng([ARCHIE2_SEED, 10 + k]))
        sb_ = stats(bB, pB, AB)
        srow(f"ρ_below {lo_}–{hi_} Ω·m (n = {m.sum()})", sb_, iB)
        bins[f"{lo_}-{hi_}"] = dict(Psal=sb_["Psal"], usw={u: sb_["usw"][u] for u in UNITS})
    print("\n".join(L), flush=True)

    # ── held-out validation ─────────────────────────────────────────────────
    nh = min(300, len(C["ho"]["D"]))
    cov_area, rel = [], []
    ucov = {u: {"area": [], "area_phys": [], "sw": [], "ec25": []} for u in UNITS}
    upred = {u: {"p_sal": [], "t_sal": [], "p_fr": [], "t_fr": []} for u in UNITS}
    n_truth_impossible = n_post_empty = 0; truth_tries = []; ho_rates = []
    for j in range(nh):
        y, t, th = C["ho"]["D"][j], C["ho"]["T"][j], C["ho"]["th"][j]
        base, par, _ = J.posterior(y, N_HELD_POST, np.random.default_rng([JOINT_SEED, 200_000 + j]))
        bH, pH, AH, iH = constrained(base, par, np.random.default_rng([ARCHIE2_SEED, 200_000 + j]), n_target=N_HELD_POST)
        ho_rates.append(iH["joint_rate"])
        tb, tp = s6.truth_params(th, t)
        tpres = presence(tb, tp)
        ta, ntry = truth_draw(tp, tpres, np.random.default_rng([ARCHIE2_SEED, 500_000 + j]))
        if len(bH) < 50:                      # no physical posterior draws: geometry-only coverage is not assessed either
            n_post_empty += 1
            continue
        sh = stats(bH, pH, AH)
        tM = unit_masks(tb, tp)
        for u in UNITS:
            ta_ = float(tM[u].sum())
            hit = bool(np.percentile(sh["uarea"][u], 5) <= ta_ <= np.percentile(sh["uarea"][u], 95))
            ucov[u]["area"].append(hit)
            if ta is not None:
                ucov[u]["area_phys"].append(hit)
        if ta is None:                        # truth cannot be physical for any F, S, T: no salinity comparison
            n_truth_impossible += 1
            continue
        truth_tries.append(ntry)
        tM, tuv, tsw, tec = sections(tb, tp, ta)
        tA = s6.areas(tsw, tec)
        cov_area.append({k: bool(np.percentile(sh["areas"][k], 5) <= tA[k][0] <= np.percentile(sh["areas"][k], 95))
                         for k in tA})
        ok = np.isfinite(tsw[0]) & np.isfinite(sh["Psal"])
        rel.append(np.column_stack([sh["Psal"][ok], (tsw[0][ok] > 50).astype(float)]))
        for u in UNITS:
            if tM[u].any() and np.isfinite(sh["usw"][u]).sum() >= 50:
                for key, tv in (("sw", tuv[u]["sw"][0]), ("ec25", tuv[u]["ec25"][0])):
                    v = sh["usw" if key == "sw" else "uec"][u]
                    lo_, hi_ = np.nanpercentile(v, [5, 95]); ucov[u][key].append(lo_ <= tv <= hi_)
                _, ps, pf = uprob(sh, u)
                upred[u]["p_sal"].append(ps); upred[u]["t_sal"].append(float(tuv[u]["sw"][0] > 50))
                upred[u]["p_fr"].append(pf); upred[u]["t_fr"].append(float(tuv[u]["ec25"][0] < 1500))
        if (j + 1) % 50 == 0:
            print(f"[held-out] {j + 1}/{nh} ({time.time() - t_start:.0f} s)", flush=True)
    L += ["", "## Held-out validation", "",
          f"{nh} held-out site samples; joint posterior {N_HELD_POST} constrained draws each (posterior seeds as Stage 6; "
          f"median joint rejection {100*np.median(ho_rates):.1f} %); truth with its own Archie v2 draw "
          "(default_rng([131000, 500000 + j]), redrawn until physical; median "
          f"{np.median(truth_tries) if truth_tries else float('nan'):.0f} tries). "
          f"Excluded: {n_post_empty} held-out samples with < 50 physical posterior draws (from everything); "
          f"{n_truth_impossible} truths that are unphysical for every F, S, T (ρ_unit < F·ρ_sw; from the salinity "
          "columns only — their geometry stays in the area coverage). Coverage of the 90 % intervals.", "",
          "| unit | area coverage: all / physical truths | % seawater coverage | EC25 coverage | Brier P(> 50 %) "
          "(climatology) | Brier P(fresh) (climatology) | n (area all / physical; salinity) |", "|---|---|---|---|---|---|---|"]
    brier = {}
    for u in UNITS:
        c = ucov[u]; pr = {k: np.array(v) for k, v in upred[u].items()}
        bs = lambda p, t: (np.mean((p - t) ** 2), np.mean((t.mean() - t) ** 2)) if t.size else (np.nan, np.nan)
        b1, b1c = bs(pr["p_sal"], pr["t_sal"]); b2, b2c = bs(pr["p_fr"], pr["t_fr"])
        brier[u] = (b1, b1c, b2, b2c)
        L.append(f"| {UNIT_NAMES[u]} | {100*np.mean(c['area']):.1f}% / {100*np.mean(c['area_phys']):.1f}% | "
                 f"{100*np.mean(c['sw']):.1f}% | {100*np.mean(c['ec25']):.1f}% | {b1:.3f} ({b1c:.3f}) | {b2:.3f} ({b2c:.3f}) | "
                 f"{len(c['area'])} / {len(c['area_phys'])}; {len(c['sw'])} |")
    ca = {k: np.mean([c[k] for c in cov_area]) for k in cov_area[0]}
    L += ["", "**Prior–constraint inconsistency.** The held-out samples are drawn from the prior, whose ρ_lens range "
          "(0.4–3.0 Ω·m) extends below the physical bound F·ρ_sw ≥ 0.694 Ω·m: "
          f"{n_truth_impossible} of {nh} truths cannot be physical for any F, S, T, and {n_post_empty} posteriors have "
          f"< 50 physical draws (median rejection {100*np.median(ho_rates):.1f} % vs {100*info_f['joint_rate']:.1f} % for the "
          "field). The constraint truncates those posteriors while their truths stay unphysical, which lowers the area "
          "coverage over all truths; restricted to physical truths it is the second value. A consistent prior would "
          "impose ρ_lens ≥ F·ρ_sw at the prior level (not done here: the BEL training set is unchanged)."]
    L += ["", "Class areas (appendix), coverage of the 90 % intervals: " + "; ".join(f"{k} {100*v:.0f} %" for k, v in ca.items())
          + "."]
    R = np.vstack(rel); bb_ = np.linspace(0, 1, 11); idx = np.clip(np.digitize(R[:, 0], bb_) - 1, 0, 9)
    L.append("Reliability of the P(% seawater > 50) map (cells): " + ", ".join(
        f"{bb_[i]:.1f}–{bb_[i+1]:.1f}: {R[idx == i, 1].mean():.2f}" for i in range(10) if (idx == i).any()) + ".")
    L += ["", "Dune zone (x < 90 m): weakly constrained — largest mesh approximation error (short dipoles under the dune "
          "cap); the dune conductive layer (clay or saltwater) is not built into the sections.",
          f"", f"Runtime {time.time() - t_start:.0f} s."]
    with open(os.path.join(RES, "2_salinity_v2.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    np.savez(os.path.join(DC, "field_v2.npz"), base=base_c, Psal=st["Psal"], Pfresh=st["Pfresh"], Nv=st["Nv"], iso=iso,
             XG=s6.XG, ZG=s6.ZG, TX=s6.TX, pdis=pdis_f,
             **{f"par_{k}": np.asarray(v) for k, v in par_c.items()},
             **{f"area_{k}": v for k, v in A.items()}, **{f"uarea_{u}": st["uarea"][u] for u in UNITS},
             **{f"usw_{u}": st["usw"][u] for u in UNITS}, **{f"uec_{u}": st["uec"][u] for u in UNITS},
             **{f"draw_{k}": v for k, v in st["draws"].items()},
             **{f"bin{k}_Psal": v["Psal"] for k, v in bins.items()},
             **{f"bin{k}_usw_{u}": v["usw"][u] for k, v in bins.items() for u in UNITS})
    json.dump(dict(constraint=dict(joint_rate=info_f["joint_rate"], unit_rate=info_f["unit_rate"],
                                   n_proposed=info_f["n_proposed"], n_accepted=info_f["n_accepted"],
                                   n_distinct=info_f["n_distinct"], heldout_median_joint_rate=float(np.median(ho_rates)),
                                   heldout_truth_impossible=n_truth_impossible, heldout_post_empty=n_post_empty),
                   sens=sens, ucov={u: {k: float(np.mean(v)) for k, v in c.items()} for u, c in ucov.items()},
                   n_heldout=nh, class_cov={k: float(v) for k, v in ca.items()},
                   brier={u: [float(x) for x in v] for u, v in brier.items()},
                   rsw=dict(min=float(rsw_min), max=float(rsw_max), p5=float(np.percentile(rs, 5)),
                            p50=float(np.median(rs)), p95=float(np.percentile(rs, 95))),
                   ec25_sw=[float(ec_lo), float(ec_hi)]),
              open(os.path.join(DC, "summary_v2.json"), "w"), indent=1)
    print("\n".join(L[-12:]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run"); pr.add_argument("--out", default=wc.ROOT)
    a = ap.parse_args()
    dict(run=cmd_run)[a.cmd](a)


if __name__ == "__main__":
    main()
