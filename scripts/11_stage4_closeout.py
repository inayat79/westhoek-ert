"""
11_stage4_closeout.py
=====================
Westhoek ERT — close-out of the linearised-UQ stage (Stage 4 → 5).

Subcommands
  a1           bias / width / coverage of the lens base per λ from the EXISTING Stage-4 λ-pool runs
  drive        parallel, resumable driver for the A2 / A3 inversion jobs (--task a2|a3)
  worker       (internal) runs a list of jobs
  a2metrics    coverage / bias / width for three interface definitions (3 Ω·m, max-gradient, log-midpoint)
  a3metrics    blocky (L1) inversions: lens-base bias with the 3 Ω·m and max-gradient definitions
  a4           the chosen definition(s) on the field run g1_lam50 + λ pool (new _v4 folders)

Interface definitions (per column, applied identically to the truth, to m̂ and to every sample):
  thr3     main-lens base = first ρ = 3 Ω·m crossing (< 3 above, ≥ 3 below) below the lens top
  maxgrad  depth of the maximum d(log10ρ)/d(depth) within ±10 m of the thr3 crossing
           (gradient-weighted centroid of the segments ≥ 90 % of the maximum, so a linear
           transition gives its centre rather than an arbitrary end)
  logmid   crossing — nearest to the maxgrad depth, within ±10 m — of the geometric mean of the
           plateau values (median log10ρ in the 3 m windows above and below the maxgrad depth)

Seeds: A2 re-uses the Stage-4 seeds (noise 1000 + r, posterior 7 000 000 + 100 000·truth_index +
100·r + λ) so the thr3 results must reproduce Stage 4 exactly (checked in a2metrics).
A3 noise 1000 + r; A4 posterior seeds 20260928 + run index. Outputs only in new folders:
data/processed/synthetic_s5/, figures/synthetic_s5/, results/synthetic_s5/, *_v4.
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
import warnings

import numpy as np
import scipy.linalg as sla

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


sb = _load("sb10", "10_synthetic_benchmark.py")

OUT_DATA = os.path.join(wc.PROCESSED_DIR, "synthetic_s5")
OUT_FIG = os.path.join(wc.FIG_DIR, "synthetic_s5")
OUT_RES = os.path.join(wc.RESULTS_DIR, "synthetic_s5")
S4_RUNS = os.path.join(wc.PROCESSED_DIR, "synthetic", "runs")
A2_TRUTHS = ["T1", "T1h", "T2a", "T3"]
A3_TRUTHS = ["T1", "T1h", "T3"]
A3_LAM_GRID = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
A3_NREAL = 10
DEFS = ("thr3", "maxgrad", "logmid")
LAMS = [50.0, 5.0, 10.0, 20.0, 100.0]
Z90 = 1.6449
WIN, PLATE = 10.0, 3.0


# ─────────────────────────────────────────────────────────────────────────────
def interfaces_all(V, z, win=WIN, plate=PLATE):
    """
    Lens-base depth per column and sample for the three definitions.
    V: (nz, nx, ns) log10ρ ordered from the surface down (NaN outside the model).
    Returns dict of (nx, ns) arrays (NaN where undefined).
    """
    _, b3 = wc.main_lens(V, z)
    dzs = (z[:-1] - z[1:])[:, None, None]
    D = (V[1:] - V[:-1]) / dzs                                  # increase of log10ρ with depth
    zm = (0.5 * (z[:-1] + z[1:]))[:, None, None]
    W = np.isfinite(D) & np.isfinite(b3)[None] & (np.abs(zm - b3[None]) <= win)
    gmax = np.where(W, D, -np.inf).max(axis=0)
    ok = np.isfinite(gmax) & (gmax > 0)
    sel = W & (D >= 0.9 * gmax[None])
    wts = np.where(sel, D, 0.0)
    sw = wts.sum(axis=0)
    zg = np.where(ok & (sw > 0), (wts * zm).sum(axis=0) / np.where(sw > 0, sw, 1.0), np.nan)
    Z = z[:, None, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        up = np.nanmedian(np.where((Z > zg[None]) & (Z <= zg[None] + plate), V, np.nan), axis=0)
        lo = np.nanmedian(np.where((Z < zg[None]) & (Z >= zg[None] - plate), V, np.nan), axis=0)
        thr = 0.5 * (up + lo)
        a = V[:-1] - thr[None]; b = V[1:] - thr[None]
        cr = np.isfinite(a) & np.isfinite(b) & ((a <= 0) != (b <= 0)) & (np.abs(zm - zg[None]) <= win)
        t = a / np.where(a != b, a - b, np.nan)
        zc = z[:-1][:, None, None] + t * (z[1:] - z[:-1])[:, None, None]
        dist = np.where(cr, np.abs(zc - zg[None]), np.inf)
        k = dist.argmin(axis=0)
        zl = np.take_along_axis(zc, k[None], 0)[0]
        zl = np.where(np.isfinite(dist.min(axis=0)), zl, np.nan)
    return dict(thr3=b3, maxgrad=zg, logmid=zl, thr=thr)


def fixed_threshold_crossing(V, z, thr, zref, win=WIN):
    """
    A5 adaptive threshold: per column a FIXED log10ρ threshold `thr` (nx,) (from m̂'s plateaus) and a
    reference depth `zref` (nx,) (m̂'s log-midpoint crossing); for every sample the crossing of thr
    nearest zref within ±win. V: (nz, nx, ns). Returns (nx, ns).
    """
    T = thr[None, :, None]; R = zref[None, :, None]
    a = V[:-1] - T; b = V[1:] - T
    zm = (0.5 * (z[:-1] + z[1:]))[:, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = np.isfinite(a) & np.isfinite(b) & ((a <= 0) != (b <= 0)) & (np.abs(zm - R) <= win)
        t = a / np.where(a != b, a - b, np.nan)
        zc = z[:-1][:, None, None] + t * (z[1:] - z[:-1])[:, None, None]
        dist = np.where(cr, np.abs(zc - R), np.inf)
    k = dist.argmin(axis=0)
    out = np.take_along_axis(zc, k[None], 0)[0]
    return np.where(np.isfinite(dist.min(axis=0)), out, np.nan)


def a5_job(m02, spec, fwd, sch, cell_x, cs, truth, real, log):
    out = os.path.join(OUT_DATA, "a5_runs", f"{truth}_r{real:02d}.npz")
    if os.path.exists(out):
        log(f"skip a5 {truth} r{real}"); return
    t0 = time.time()
    R_true = fwd[f"R_{truth}"]
    rng = np.random.default_rng(sb.seeds(truth, real)["noise"])
    sig = np.sqrt((sb.ERR_REL * R_true) ** 2 + sb.ERR_ABS ** 2)
    R = R_true + sig * rng.standard_normal(len(R_true))
    keep = R > 0
    elec = {int(e): (float(x), float(z)) for e, x, z in zip(sch["elec_num"], sch["elec_x"], sch["elec_z"])}
    land = dict({k: sch[k][keep] for k in ("C1", "C2", "P1", "P2")}, R=R[keep], n=int(keep.sum()),
                err=np.zeros(int(keep.sum())))
    lams = [50.0] + ([5.0, 10.0, 20.0, 100.0] if real < sb.N_REAL_LAMBDA else [])
    save = dict(truth=truth, real=real, cols=cs.x)
    pooled = []
    for lam in lams:
        info, S = sb.one_inversion(m02, land, elec, lam, cell_x, cs, sb.seeds(truth, real, lam)["post"])
        hat = interfaces_all(cs(np.log10(info["model"].astype(float)))[:, :, None], cs.z)
        thr, zref = hat["thr"][:, 0], hat["logmid"][:, 0]
        Z = np.full((len(cs.x), S.shape[1]), np.nan)
        for i in range(0, S.shape[1], 100):
            Z[:, i:i + 100] = fixed_threshold_crossing(cs(S[:, i:i + 100] / np.log(10)), cs.z, thr, zref)
        p = f"lam{int(lam)}_"
        save[p + "band_adaptive"] = band_stats(Z); save[p + "mhat_adaptive"] = zref; save[p + "thr"] = thr
        save[p + "r_adaptive"] = r_at(cs, info["radius"], zref)
        if real < sb.N_REAL_LAMBDA:
            pooled.append(Z)
        log(f"a5 {truth} r{real} λ={lam:g}: χ²={info['chi2']:.3f}")
    if pooled:
        save["pooled_band_adaptive"] = band_stats(np.hstack(pooled))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out + ".tmp.npz", **save); os.replace(out + ".tmp.npz", out)
    log(f"done a5 {truth} r{real} in {time.time()-t0:.0f}s")


def cmd_a5metrics(args):
    spec = sb.load_spec()
    L = ["# A5 — adaptive-threshold interface vs log-midpoint (linearised, λ = 50 and pooled λ)", "",
         "Adaptive: per column the threshold = geometric mean of the m̂ plateaus (log-midpoint threshold of m̂); each "
         "posterior sample's crossing of that fixed threshold nearest m̂'s log-midpoint depth (±10 m). Log-midpoint: "
         "each sample's own plateaus (Stage-5 A2). Same seeds, same samples. Truth = log-midpoint truth (base centre). "
         "Grouped by the resolution radius at m̂'s log-midpoint crossing.", "",
         "| truth | r | n | adaptive cov. single / pooled | log-mid cov. single / pooled | adaptive width single / pooled "
         "| log-mid width single / pooled | bias (m̂, same for both) |", "|---|---|---|---|---|---|---|---|"]
    for t in A2_TRUTHS:
        tv = truth_defs(spec, t, sb.COLS)["logmid"]
        for grp in ("≤10", "10–15"):
            rows = []
            for r in range(sb.N_REAL):
                p5, p2 = (os.path.join(OUT_DATA, d, f"{t}_r{r:02d}.npz") for d in ("a5_runs", "a2_runs"))
                if not (os.path.exists(p5) and os.path.exists(p2)):
                    continue
                a5 = np.load(p5, allow_pickle=True); a2 = np.load(p2, allow_pickle=True)
                for j in range(len(sb.COLS)):
                    rr = a2["lam50_r_logmid"][j]
                    if not np.isfinite(tv[j]) or not np.isfinite(rr):
                        continue
                    if ("≤10" if rr <= 10 else ("10–15" if rr <= 15 else None)) != grp:
                        continue
                    ba, bl = a5["lam50_band_adaptive"][:, j], a2["lam50_band_logmid"][:, j]
                    row = dict(ca=ba[0] >= 0.5 and ba[2] <= tv[j] <= ba[3], cl=bl[0] >= 0.5 and bl[2] <= tv[j] <= bl[3],
                               wa=ba[3] - ba[2] if ba[0] >= 0.5 else np.nan, wl=bl[3] - bl[2] if bl[0] >= 0.5 else np.nan,
                               bias=a2["lam50_mhat_logmid"][j] - tv[j])
                    if "pooled_band_adaptive" in a5.files:
                        pa, pl = a5["pooled_band_adaptive"][:, j], a2["pooled_band_logmid"][:, j]
                        row.update(pa=pa[0] >= 0.5 and pa[2] <= tv[j] <= pa[3], pl=pl[0] >= 0.5 and pl[2] <= tv[j] <= pl[3],
                                   wpa=pa[3] - pa[2] if pa[0] >= 0.5 else np.nan, wpl=pl[3] - pl[2] if pl[0] >= 0.5 else np.nan)
                    rows.append(row)
            if not rows:
                continue
            m = lambda k: np.mean([x[k] for x in rows if k in x]) if any(k in x for x in rows) else np.nan
            md = lambda k: np.nanmedian([x[k] for x in rows if k in x]) if any(k in x for x in rows) else np.nan
            L.append(f"| {t} | {grp} | {len(rows)} | {100*m('ca'):.0f}% / {100*m('pa'):.0f}% | {100*m('cl'):.0f}% / "
                     f"{100*m('pl'):.0f}% | {md('wa'):.2f} / {md('wpa'):.2f} | {md('wl'):.2f} / {md('wpl'):.2f} | "
                     f"{md('bias'):+.2f} |")
    os.makedirs(OUT_RES, exist_ok=True)
    with open(os.path.join(OUT_RES, "a5_adaptive_threshold.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def band_stats(Zs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.vstack([np.isfinite(Zs).mean(axis=1), np.nanmedian(Zs, axis=1),
                          np.nanpercentile(Zs, 5, axis=1), np.nanpercentile(Zs, 95, axis=1)])


def sample_defs(cs, S, chunk=100):
    out = {d: np.full((len(cs.x), S.shape[1]), np.nan) for d in DEFS}
    for i in range(0, S.shape[1], chunk):
        r = interfaces_all(cs(S[:, i:i + chunk] / np.log(10)), cs.z)
        for d in DEFS:
            out[d][:, i:i + chunk] = r[d]
    return {d: band_stats(out[d]) for d in DEFS}


def r_at(cs, radius, zc):
    Rg = cs(np.log10(radius)); dz = cs.z[0] - cs.z[1]
    out = np.full(len(zc), np.nan)
    for j, z in enumerate(zc):
        if np.isfinite(z):
            out[j] = 10 ** Rg[int(np.clip(np.floor((cs.z[0] - z) / dz), 0, len(cs.z) - 1)), j]
    return out


def truth_defs(spec, truth, cols, dz=0.02):
    tr = sb.Truth(spec, truth)
    ex, ez = wc.land_surface()
    s = np.interp(cols, ex, ez)
    z = np.arange(np.floor(ez.max() / dz) * dz, wc.Z_FLOOR - 1e-9, -dz)
    X, Zg = np.meshgrid(cols, z)
    V = tr.log10rho(X, Zg); V[Zg > s[None, :] - 1e-6] = np.nan
    r = interfaces_all(V[:, :, None], z)
    return {d: r[d][:, 0] for d in DEFS}


# ─────────────────────────────────────────────────────────────────────────────
def cmd_a1(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_RES, exist_ok=True)
    fwd = np.load(os.path.join(sb.SYN_DATA, "truth_forward.npz"))
    rows = []
    for t in sb.TRUTHS:
        tb = fwd[f"base_{t}"]
        for r in range(sb.N_REAL_LAMBDA):
            d = np.load(os.path.join(S4_RUNS, f"{t}_r{r:02d}.npz"), allow_pickle=True)
            r50 = d["lam50_r_base"]
            for lam in (5, 10, 20, 50, 100):
                k = f"lam{lam}_"
                mh, sw, bb = d[k + "mhat_base"], d[k + "sw_base"], d[k + "band_base"]
                for j in range(len(sb.COLS)):
                    if not np.isfinite(tb[j]) or not np.isfinite(r50[j]):
                        continue
                    g = "≤10" if r50[j] <= 10 else ("10–15" if r50[j] <= 15 else None)
                    if g is None:
                        continue
                    rows.append(dict(truth=t, lam=lam, grp=g, bias=mh[j] - tb[j],
                                     cov_s=bb[0, j] >= 0.5 and bb[2, j] <= tb[j] <= bb[3, j],
                                     cov_a=np.isfinite(mh[j]) and abs(mh[j] - tb[j]) <= Z90 * sw[j],
                                     w=bb[3, j] - bb[2, j] if bb[0, j] >= 0.5 else np.nan))
    L = ["# A1 — lens-base bias, width and coverage per λ (existing Stage-4 λ-pool runs)", "",
         "Realisations 0–4 of every truth model; the SAME (column, realisation) pairs for every λ, grouped by the "
         "resolution radius at the λ = 50 base crossing. Bias = median(m̂ − true) (negative = too deep); width = median "
         "sampled 90 % width; coverage of the sampled / analytic (wᵀA⁻¹w) 90 % band.", ""]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, g in zip(axes, ("≤10", "10–15")):
        L += [f"## r ≤ 10 m" if g == "≤10" else "## 10 < r ≤ 15 m", "",
              "| truth | " + " | ".join(f"λ={l}: bias / width / cov s·a" for l in (5, 10, 20, 50, 100)) + " |",
              "|---|" + "---|" * 5]
        for t in sb.TRUTHS:
            cells, biases = [], []
            for lam in (5, 10, 20, 50, 100):
                rs = [x for x in rows if x["truth"] == t and x["lam"] == lam and x["grp"] == g]
                if not rs:
                    cells.append("—"); biases.append(np.nan); continue
                b = np.nanmedian([x["bias"] for x in rs]); biases.append(b)
                cells.append(f"{b:+.2f} / {np.nanmedian([x['w'] for x in rs]):.2f} / "
                             f"{100*np.mean([x['cov_s'] for x in rs]):.0f}%·{100*np.mean([x['cov_a'] for x in rs]):.0f}% "
                             f"(n={len(rs)})")
            L.append(f"| {t} | " + " | ".join(cells) + " |")
            ax.plot([5, 10, 20, 50, 100], biases, "-o", ms=4, label=t)
        allb = [np.nanmedian([x["bias"] for x in rows if x["lam"] == lam and x["grp"] == g]) for lam in (5, 10, 20, 50, 100)]
        L.append(f"| **all** | " + " | ".join(f"{b:+.2f}" for b in allb) + " |")
        L.append("")
        ax.plot([5, 10, 20, 50, 100], allb, "k-", lw=2.5, label="all")
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 50, 100]); ax.set_xticklabels(["5", "10", "20", "50", "100"])
        ax.set_xlabel("λ"); ax.set_title(f"lens-base bias vs λ — r {'≤ 10 m' if g == '≤10' else '10–15 m'} "
                                         "(at λ = 50 crossing)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("median bias m̂ − true (m); negative = too deep")
    wc.white_legend(axes[1], loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "a1_bias_vs_lambda.png"), dpi=160)
    with open(os.path.join(OUT_RES, "a1_bias_vs_lambda.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
def a2_job(m02, spec, fwd, sch, cell_x, cs, truth, real, log, R_true=None, noise=None, out=None, lams=None,
           post_seed=None, save_extra=False):
    """A2 job. The keyword arguments (added in v1.1 for 22_lebbe1980_synthetic.py) default to the v1.0 behaviour:
    R_true = fwd['R_<truth>']; noise = Gaussian 3 % + 1e-4 Ω with the Stage-4 seed; out = a2_runs/<truth>_r<real>.npz;
    lams = [50] (+ the λ pool for real < N_REAL_LAMBDA); post_seed(λ) = the Stage-4 posterior seed; save_extra adds the
    noisy data (R_noisy, keep) and the λ = 50 model."""
    out = out or os.path.join(OUT_DATA, "a2_runs", f"{truth}_r{real:02d}.npz")
    if os.path.exists(out):
        log(f"skip a2 {truth} r{real}"); return
    t0 = time.time()
    R_true = fwd[f"R_{truth}"] if R_true is None else R_true
    if noise is None:
        rng = np.random.default_rng(sb.seeds(truth, real)["noise"])
        sig = np.sqrt((sb.ERR_REL * R_true) ** 2 + sb.ERR_ABS ** 2)
        R = R_true + sig * rng.standard_normal(len(R_true))
    else:
        R = noise(R_true)
    keep = R > 0
    elec = {int(e): (float(x), float(z)) for e, x, z in zip(sch["elec_num"], sch["elec_x"], sch["elec_z"])}
    land = dict({k: sch[k][keep] for k in ("C1", "C2", "P1", "P2")}, R=R[keep], n=int(keep.sum()),
                err=np.zeros(int(keep.sum())))
    lams = lams or ([50.0] + ([5.0, 10.0, 20.0, 100.0] if real < sb.N_REAL_LAMBDA else []))
    save = dict(truth=truth, real=real, cols=cs.x, n_nonpositive_dropped=int((~keep).sum()))
    if save_extra:
        save.update(R_noisy=R, keep=keep)
    pooled = []
    for lam in lams:
        seed = sb.seeds(truth, real, lam)["post"] if post_seed is None else post_seed(lam)
        info, S = sb.one_inversion(m02, land, elec, lam, cell_x, cs, seed)
        if save_extra and lam == 50.0:
            save["lam50_model"] = info["model"]
        hat = interfaces_all(cs(np.log10(info["model"].astype(float)))[:, :, None], cs.z)
        bands = sample_defs(cs, S)
        p = f"lam{int(lam)}_"
        save.update({p + "chi2": info["chi2"], p + "stop": info["stop"], p + "n_iter": info["n_iter"]})
        for d in DEFS:
            save[p + f"mhat_{d}"] = hat[d][:, 0]
            save[p + f"band_{d}"] = bands[d]
            save[p + f"r_{d}"] = r_at(cs, info["radius"], hat[d][:, 0])
        if len(lams) > 1:                                  # v1.0: the λ pool exists exactly when real < N_REAL_LAMBDA
            pooled.append(S.astype(np.float32))
        log(f"a2 {truth} r{real} λ={lam:g}: χ²={info['chi2']:.3f} ({info['stop']})")
    if pooled:
        pb = sample_defs(cs, np.hstack(pooled).astype(np.float64))
        for d in DEFS:
            save[f"pooled_band_{d}"] = pb[d]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out + ".tmp.npz", **save); os.replace(out + ".tmp.npz", out)
    log(f"done a2 {truth} r{real} in {time.time()-t0:.0f}s")


def a3_job(m02, spec, fwd, sch, cell_x, cs, truth, real, lam, log):
    out = os.path.join(OUT_DATA, "a3_runs", f"{truth}_r{real:02d}_lam{int(lam)}.npz")
    if os.path.exists(out):
        log(f"skip a3 {truth} r{real} λ={lam:g}"); return
    t0 = time.time()
    R_true = fwd[f"R_{truth}"]
    rng = np.random.default_rng(sb.seeds(truth, real)["noise"])
    sig = np.sqrt((sb.ERR_REL * R_true) ** 2 + sb.ERR_ABS ** 2)
    R = R_true + sig * rng.standard_normal(len(R_true))
    keep = R > 0
    elec = {int(e): (float(x), float(z)) for e, x, z in zip(sch["elec_num"], sch["elec_x"], sch["elec_z"])}
    land = dict({k: sch[k][keep] for k in ("C1", "C2", "P1", "P2")}, R=R[keep], n=int(keep.sum()),
                err=np.zeros(int(keep.sum())))
    data = m02.build_datacont(land, elec, err_rel=sb.ERR_REL)[0]
    mgr = m02.run_inversion(data, lam=lam, para_depth=sb.PARA_DEPTH, para_dx=sb.PARA_DX,
                            max_iter=sb.MAX_ITER, blocky=True)
    model = np.array(mgr.model)
    if len(model) != len(cell_x):
        raise RuntimeError("mesh mismatch")
    r = (np.log(np.array(data["rhoa"])) - np.log(np.array(mgr.inv.response))) / np.array(data["err"])
    chi2, chi2_pg, n_iter = float(np.mean(r ** 2)), float(mgr.inv.chi2()), m02.n_iterations(mgr)
    stop = "chi2<=1" if chi2_pg <= 1 else ("iteration cap" if n_iter >= sb.MAX_ITER else "dPhi<2%")
    hat = interfaces_all(cs(np.log10(model))[:, :, None], cs.z)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, truth=truth, real=real, lam=lam, chi2=chi2, chi2_pg=chi2_pg, n_iter=n_iter, stop=stop,
             model=model.astype(np.float32), **{f"mhat_{d}": hat[d][:, 0] for d in DEFS},
             t=time.time() - t0)
    log(f"a3 {truth} r{real} λ={lam:g}: χ²={chi2:.3f} ({stop}, {n_iter} it) {time.time()-t0:.0f}s")


def cmd_worker(args):
    spec = sb.load_spec()
    fwd = np.load(os.path.join(sb.SYN_DATA, "truth_forward.npz"))
    sch = sb.g1_scheme()
    ref = wc.load_run(sb.REF_TAG)
    mesh, _ = wc.rebuild_paradomain(ref)
    cs = wc.ColumnSampler(mesh, sb.COLS)
    m02 = _load("inv02", "02_land_inversion.py")
    m02.VERBOSE = False
    logf = open(os.path.join(OUT_DATA, "logs", f"{args.task}_worker_{args.worker_id}.log"), "a", encoding="utf-8")
    def log(msg):
        logf.write(time.strftime("%H:%M:%S ") + msg + "\n"); logf.flush()
    for job in args.jobs:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                parts = job.split(":")
                if args.task == "a5":
                    a5_job(m02, spec, fwd, sch, np.asarray(ref["cell_x"]), cs, parts[0], int(parts[1]), log)
                elif args.task == "a2":
                    a2_job(m02, spec, fwd, sch, np.asarray(ref["cell_x"]), cs, parts[0], int(parts[1]), log)
                else:
                    a3_job(m02, spec, fwd, sch, np.asarray(ref["cell_x"]), cs, parts[0], int(parts[1]),
                           float(parts[2]), log)
        except Exception as e:
            import traceback
            log(f"FAILED {job}: {e}\n{traceback.format_exc()}")
    log("worker finished")


def a3_lambda_dp():
    """λ by the discrepancy principle per truth: χ² closest to 1 on the realisation-0 grid."""
    out = {}
    for t in A3_TRUTHS:
        grid = []
        for lam in A3_LAM_GRID:
            p = os.path.join(OUT_DATA, "a3_runs", f"{t}_r00_lam{int(lam)}.npz")
            if os.path.exists(p):
                grid.append((lam, float(np.load(p)["chi2"])))
        if len(grid) == len(A3_LAM_GRID):
            out[t] = dict(grid=grid, lam=min(grid, key=lambda g: abs(g[1] - 1.0))[0])
    return out


def cmd_drive(args):
    os.makedirs(os.path.join(OUT_DATA, "logs"), exist_ok=True)
    if args.task == "a2":
        jobs = [f"{t}:{r}" for r in range(sb.N_REAL) for t in A2_TRUTHS
                if not os.path.exists(os.path.join(OUT_DATA, "a2_runs", f"{t}_r{r:02d}.npz"))]
        jobs.sort(key=lambda j: (int(j.split(":")[1]) >= sb.N_REAL_LAMBDA, int(j.split(":")[1])))
    elif args.task == "a5":
        jobs = [f"{t}:{r}" for r in range(sb.N_REAL) for t in A2_TRUTHS
                if not os.path.exists(os.path.join(OUT_DATA, "a5_runs", f"{t}_r{r:02d}.npz"))]
        jobs.sort(key=lambda j: (int(j.split(":")[1]) >= sb.N_REAL_LAMBDA, int(j.split(":")[1])))
    elif args.task == "a3grid":
        jobs = [f"{t}:0:{lam}" for t in A3_TRUTHS for lam in A3_LAM_GRID]
    else:  # a3: 10 realisations at λ_DP
        dp = a3_lambda_dp()
        if len(dp) < len(A3_TRUTHS):
            raise SystemExit("run --task a3grid first")
        jobs = [f"{t}:{r}:{dp[t]['lam']}" for t in A3_TRUTHS for r in range(A3_NREAL)]
    task = args.task if args.task in ("a2", "a5") else "a3"
    parts = [jobs[i::args.workers] for i in range(args.workers)]
    print(f"[drive] {args.task}: {len(jobs)} jobs → {args.workers} workers")
    procs = []
    for w, part in enumerate(parts):
        if not part:
            continue
        wd = os.path.join(OUT_DATA, "work", f"{task}_w{w}")
        os.makedirs(wd, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "worker", "--task", task, "--worker-id", str(w),
               "--jobs"] + part
        procs.append(subprocess.Popen(cmd, cwd=wd, env=dict(os.environ, OMP_NUM_THREADS="2"),
                                      stdout=subprocess.DEVNULL,
                                      stderr=open(os.path.join(OUT_DATA, "logs", f"{task}_worker_{w}.err"), "a")))
    for p in procs:
        p.wait()
    print(f"[drive] {args.task} finished")


# ─────────────────────────────────────────────────────────────────────────────
def cmd_a2metrics(args):
    spec = sb.load_spec()
    cols = sb.COLS
    truths = {t: truth_defs(spec, t, cols) for t in A2_TRUTHS}
    L = ["# A2 — lens-base coverage, bias and width for three interface definitions", "",
         "Same (column, realisation) sets per definition are NOT forced: each definition is grouped by the resolution "
         "radius at its own m̂ depth. Coverage = sampled 90 % band (P5–P95 over 1000 posterior samples; a column needs the "
         "interface in ≥ 50 % of samples) contains the TRUE interface defined the same way on the truth. Pooled = λ ∈ "
         "{5,10,20,50,100} (realisations 0–4). Bias = median(m̂ − true), negative = too deep. Width = median sampled "
         "90 % width.", "",
         "True interface by definition at x = 150 / 250 / 350 m:", ""]
    for t in A2_TRUTHS:
        L.append(f"- {t}: " + "; ".join(f"{d}: " + ", ".join(f"{truths[t][d][np.argmin(np.abs(cols - x))]:.2f}"
                                                              for x in (150, 250, 350)) for d in DEFS))
    # reproduction check against Stage 4 (thr3, λ = 50)
    diffs = []
    recs = {t: [] for t in A2_TRUTHS}
    for t in A2_TRUTHS:
        for r in range(sb.N_REAL):
            p = os.path.join(OUT_DATA, "a2_runs", f"{t}_r{r:02d}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p, allow_pickle=True); recs[t].append(d)
            s4 = np.load(os.path.join(S4_RUNS, f"{t}_r{r:02d}.npz"), allow_pickle=True)
            a, b = d["lam50_mhat_thr3"], s4["lam50_mhat_base"]
            ok = np.isfinite(a) & np.isfinite(b)
            diffs.append(np.max(np.abs(a[ok] - b[ok])) if ok.any() else 0.0)
    L += ["", f"Reproduction check (thr3, λ=50 m̂ vs Stage 4, {len(diffs)} runs): max |Δ| = {max(diffs):.2e} m", ""]
    L += ["| truth | r at est. | definition | n | cov. single | cov. pooled | bias (m) | width (m) | pooled width (m) |",
          "|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for t in A2_TRUTHS:
        for grp in ("≤10", "10–15"):
            for dname in DEFS:
                tv = truths[t][dname]
                rows = []
                for d in recs[t]:
                    mh, bb, rr = d[f"lam50_mhat_{dname}"], d[f"lam50_band_{dname}"], d[f"lam50_r_{dname}"]
                    pb = d[f"pooled_band_{dname}"] if f"pooled_band_{dname}" in d.files else None
                    for j in range(len(cols)):
                        if not np.isfinite(tv[j]) or not np.isfinite(rr[j]):
                            continue
                        g = "≤10" if rr[j] <= 10 else ("10–15" if rr[j] <= 15 else None)
                        if g != grp:
                            continue
                        row = dict(bias=mh[j] - tv[j], cov=bb[0, j] >= 0.5 and bb[2, j] <= tv[j] <= bb[3, j],
                                   w=bb[3, j] - bb[2, j] if bb[0, j] >= 0.5 else np.nan)
                        if pb is not None:
                            row.update(cp=pb[0, j] >= 0.5 and pb[2, j] <= tv[j] <= pb[3, j],
                                       wp=pb[3, j] - pb[2, j] if pb[0, j] >= 0.5 else np.nan)
                        rows.append(row)
                if not rows:
                    continue
                cp = [r["cp"] for r in rows if "cp" in r]
                summ[(t, grp, dname)] = dict(cov=np.mean([r["cov"] for r in rows]), bias=np.nanmedian([r["bias"] for r in rows]))
                L.append(f"| {t} | {grp} | {dname} | {len(rows)} | {100*np.mean([r['cov'] for r in rows]):.0f}% | "
                         + (f"{100*np.mean(cp):.0f}% (n={len(cp)})" if cp else "—")
                         + f" | {np.nanmedian([r['bias'] for r in rows]):+.2f} | {np.nanmedian([r['w'] for r in rows]):.2f} | "
                         + (f"{np.nanmedian([r['wp'] for r in rows if 'wp' in r]):.2f}" if cp else "—") + " |")
    for grp in ("≤10", "10–15"):
        L.append("")
        L.append(f"**All four truths, r {grp} m:** " + "; ".join(
            f"{dname}: coverage {100*np.mean([summ[(t, grp, dname)]['cov'] for t in A2_TRUTHS if (t, grp, dname) in summ]):.0f}% "
            f"(mean over truths), |bias| {np.mean([abs(summ[(t, grp, dname)]['bias']) for t in A2_TRUTHS if (t, grp, dname) in summ]):.2f} m"
            for dname in DEFS))
    os.makedirs(OUT_RES, exist_ok=True)
    with open(os.path.join(OUT_RES, "a2_interface_definitions.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_a3metrics(args):
    spec = sb.load_spec()
    dp = a3_lambda_dp()
    L = ["# A3 — blocky (IRLS L1 model roughness) inversions: lens-base bias", "",
         "pyGIMLi 1.5.5: invert(..., blockyModel=True) → Inversion.blockyModel → RInversion.setBlockyModel; "
         "constrainBlocky() each iteration. Same mesh, data, error model and zWeight as the smooth runs. "
         "λ by the discrepancy principle = the grid value with χ² closest to 1 on realisation 0.", ""]
    for t in A3_TRUTHS:
        if t in dp:
            L.append(f"- {t}: grid " + ", ".join(f"λ={l:g}: χ²={c:.2f}" for l, c in dp[t]["grid"]) + f" → λ_DP = {dp[t]['lam']:g}")
    L += ["", "| truth | model | n runs | χ² median | stop reasons | bias thr3 (m) | bias maxgrad (m) | bias logmid (m) |",
          "|---|---|---|---|---|---|---|---|"]
    import collections
    for t in A3_TRUTHS:
        tv = truth_defs(spec, t, sb.COLS)
        xin = sb.COLS < 370
        # blocky
        runs = [np.load(os.path.join(OUT_DATA, "a3_runs", f"{t}_r{r:02d}_lam{int(dp[t]['lam'])}.npz"), allow_pickle=True)
                for r in range(A3_NREAL) if t in dp and os.path.exists(os.path.join(OUT_DATA, "a3_runs",
                                                                                     f"{t}_r{r:02d}_lam{int(dp[t]['lam'])}.npz"))]
        if runs:
            b = {d: np.nanmedian(np.concatenate([(x[f"mhat_{d}"] - tv[d])[xin] for x in runs])) for d in DEFS}
            st = collections.Counter(str(x["stop"]) for x in runs)
            L.append(f"| {t} | blocky λ={dp[t]['lam']:g} | {len(runs)} | {np.median([float(x['chi2']) for x in runs]):.3f} | "
                     + ", ".join(f"{k}: {v}" for k, v in st.items())
                     + " | " + " | ".join(f"{b[d]:+.2f}" for d in DEFS) + " |")
        # smooth λ=50, same realisations (A2 runs)
        sm = [np.load(os.path.join(OUT_DATA, "a2_runs", f"{t}_r{r:02d}.npz"), allow_pickle=True) for r in range(A3_NREAL)
              if os.path.exists(os.path.join(OUT_DATA, "a2_runs", f"{t}_r{r:02d}.npz"))]
        if sm:
            b = {d: np.nanmedian(np.concatenate([(x[f"lam50_mhat_{d}"] - tv[d])[xin] for x in sm])) for d in DEFS}
            L.append(f"| {t} | smooth λ=50 | {len(sm)} | {np.median([float(x['lam50_chi2']) for x in sm]):.3f} | — | "
                     + " | ".join(f"{b[d]:+.2f}" for d in DEFS) + " |")
    L += ["", "Bias over columns x 100–360 m (the lens), all realisations pooled; negative = too deep."]
    os.makedirs(OUT_RES, exist_ok=True)
    with open(os.path.join(OUT_RES, "a3_blocky.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
def cmd_a4(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = ["g1_lam50", "g1_lam5", "g1_lam10", "deep_drop_data", "g1_lam100"]
    name = "g1_lam50_v4"
    fdir = os.path.join(wc.FIG_DIR, "runs", name); rdir = os.path.join(wc.RESULTS_DIR, "runs", name)
    for dd in (fdir, rdir):
        if os.path.exists(dd) and os.listdir(dd):
            raise SystemExit(f"{dd} exists — not overwriting")
        os.makedirs(dd, exist_ok=True)
    ref = wc.load_run("g1_lam50")
    mesh, _ = wc.rebuild_paradomain(ref)
    ex, ez = wc.land_surface()
    cs = wc.ColumnSampler(mesh, np.arange(np.ceil(ex.min()), np.floor(ex.max()) + 1, 2.0))
    single, pooled, hats, radii = None, [], {}, {}
    for i, t in enumerate(tags):
        run = wc.load_run(t)
        L_, _, _, lam = wc.cholesky_A(run)
        rng = np.random.default_rng(20260928 + i)
        S = np.log(run["model"])[:, None] + sla.solve_triangular(L_, rng.standard_normal((len(run["model"]), 1000)),
                                                                 lower=True, trans="T")
        bands = sample_defs(cs, S)
        hats[t] = interfaces_all(cs(np.log10(run["model"]))[:, :, None], cs.z)
        radii[t] = run["res_radius"]
        if t == "g1_lam50":
            single = bands
        pooled.append(S.astype(np.float32))
        print(f"[a4] {t} λ={lam:g} sampled (seed {20260928 + i})")
    pb = sample_defs(cs, np.hstack(pooled).astype(np.float64))
    rr = {d: r_at(cs, radii["g1_lam50"], hats["g1_lam50"][d][:, 0]) for d in DEFS}
    L = [f"# A4 — field lens-base bands by interface definition (g1_lam50, λ pool {tags})", "",
         "Elevation (mTAW): m̂ | single-λ median [P5, P95] | pooled median [P5, P95] | r at m̂ crossing. "
         "Posterior seeds 20260928 + run index (order as listed).", ""]
    for d in DEFS:
        L += [f"## {d}", "", "| x | m̂ | single λ=50 | pooled λ | r (m) |", "|---|---|---|---|---|"]
        for x in (150, 200, 250, 300, 350):
            j = int(np.argmin(np.abs(cs.x - x)))
            s, p = single[d][:, j], pb[d][:, j]
            fmt = lambda b: (f"{b[1]:.2f} [{b[2]:.2f}, {b[3]:.2f}] ({100*b[0]:.0f}%)" if b[0] > 0 else "absent")
            L.append(f"| {x} | {hats['g1_lam50'][d][j, 0]:.2f} | {fmt(s)} | {fmt(p)} | {rr[d][j]:.1f} |")
        L.append("")
    with open(os.path.join(rdir, "interface_definitions_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    np.savez(os.path.join(rdir, "interface_definitions_bands.npz"), x=cs.x,
             **{f"single_{d}": single[d] for d in DEFS}, **{f"pooled_{d}": pb[d] for d in DEFS},
             **{f"mhat_{d}": hats["g1_lam50"][d][:, 0] for d in DEFS}, **{f"r_{d}": rr[d] for d in DEFS})
    # figure
    polys = wc.cell_polygons(mesh)
    fig, axes = plt.subplots(len(DEFS), 1, figsize=(15, 5 * len(DEFS)))
    import matplotlib.colors as mcolors
    for ax, d in zip(axes, DEFS):
        wc.draw_cells(ax, polys, ref["model"], "RdBu_r", mcolors.LogNorm(0.5, 500), alpha=0.45)
        wc.hatch_cells(ax, polys, ref["res_radius"] > 10)
        for bands, col, lab in ((single[d], "darkorange", "single λ=50"), (pb[d], "purple", "pooled λ")):
            ok = bands[0] >= 0.5
            ax.fill_between(cs.x, np.where(ok, bands[2], np.nan), np.where(ok, bands[3], np.nan), color=col,
                            alpha=0.25, lw=0, zorder=6)
            ax.plot(cs.x, np.where(ok, bands[1], np.nan), color=col, lw=1.4, zorder=7, label=f"{lab}: median, 5–95 %")
        wc.setup_section(ax, f"lens base — definition '{d}' (hatched: r > 10 m)")
        wc.white_legend(ax, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "27_lens_base_definitions.png"), dpi=150, bbox_inches="tight")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("a1"); sub.add_parser("a2metrics"); sub.add_parser("a3metrics"); sub.add_parser("a4")
    sub.add_parser("a5metrics")
    pd = sub.add_parser("drive"); pd.add_argument("--task", choices=("a2", "a3grid", "a3", "a5"), required=True)
    pd.add_argument("--workers", type=int, default=4)
    pw = sub.add_parser("worker"); pw.add_argument("--task", choices=("a2", "a3", "a5"), required=True)
    pw.add_argument("--worker-id", type=int, required=True); pw.add_argument("--jobs", nargs="+", required=True)
    args = ap.parse_args()
    dict(a1=cmd_a1, drive=cmd_drive, worker=cmd_worker, a2metrics=cmd_a2metrics, a3metrics=cmd_a3metrics,
         a4=cmd_a4, a5metrics=cmd_a5metrics)[args.cmd](args)


if __name__ == "__main__":
    main()
