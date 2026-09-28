"""
15_stage5_closeout.py
=====================
Westhoek ERT — Stage 5 close-out.

Subcommands
  conventions   1a  BEL target convention vs the Stage-4 validation truth (both directions)
  twin          1b  BEL-builder twin of T1 (least-squares control points, T1 units), rasterised difference map
  twinbel       1c  twin forwarded on the fine mesh (Stage-4 noise seeds), BEL-KDE vs T1
  site          2a  "site" prior subset (deep-unit top −38…−22 mTAW, lens base above it): lens-base BEL +
                    second BEL; 2b ρ_below-bin conditional lens-base product
  bounded       3   second BEL with logit-bounded resistivity targets; ρ_discharge ceiling check
  archie        4   Archie conversion of the bounded posteriors to pore-water ρ_w, EC25 and % seawater

Seeds: BEL posterior default_rng([127000, k]); Archie / seawater draws default_rng(128000).
Outputs: results/bel_v4/, figures/bel_v4/, data/processed/bel_v4/ (reference unchanged).
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


g14 = _load("g14", "14_gateA_sensitivity.py")
b13, c11 = g14.b13, g14.c11
sb = c11.sb

D4 = os.path.join(wc.PROCESSED_DIR, "bel_v4")
FIG = os.path.join(wc.FIG_DIR, "bel_v4")
RES = os.path.join(wc.RESULTS_DIR, "bel_v4")
POST, ARCHIE_SEED = 127000, 128000
XREP = (150, 200, 250, 300, 350)
TX = b13.TX
N_POST = 2000
RHO_BOUNDS = {"rho_lens": (0.4, 3.0), "rho_below": (2.0, 20.0), "rho_discharge": (2.5, 40.0)}
TNAMES = ["D1", "D2", "rho_lens", "rho_below", "rho_discharge"]


def ensure():
    for d in (D4, FIG, RES):
        os.makedirs(d, exist_ok=True)


def field_extract(log10fun, xs=TX, dz=0.02):
    """Field-based log-midpoint base (the Stage-4 validation 'truth' convention) on vertical columns."""
    ex, ez = wc.land_surface()
    s = np.interp(xs, ex, ez)
    z = np.arange(np.floor(ez.max() / dz) * dz, wc.Z_FLOOR - 1e-9, -dz)
    X, Z = np.meshgrid(xs, z)
    V = log10fun(X, Z); V[Z > s[None, :] - 1e-6] = np.nan
    r = c11.interfaces_all(V[:, :, None], z)
    return {k: r[k][:, 0] for k in ("thr3", "logmid")}


# ─────────────────────────────────────────────────────────────────────────────
def cmd_conventions(args):
    ensure()
    spec4 = sb.load_spec()
    L = ["# 1a — target conventions: BEL target vs the Stage-4 validation truth", "",
         "Validation truth (Stage 4/5 BEL tables) = log-midpoint base extracted from the truth's resistivity field "
         "(`11_stage4_closeout.truth_defs`, dz = 0.02 m). BEL's own target = the parametric base-transition centre "
         "(`13_bel_full.PriorModel2.target`); for the Stage-4 truths the builder's base line b(x) is its analogue. Columns "
         f"x = {TX[0]:g}–{TX[-1]:g} m, mTAW.", "",
         "## Direction 1 — Stage-4 truths: builder base b(x) vs validation truth", "",
         "| truth | columns compared | median (validation − b) | max \\|validation − b\\| | 3 Ω·m − b (median) |", "|---|---|---|---|---|"]
    for t in ("T1", "T1h", "T2a", "T3"):
        tr = sb.Truth(spec4, t)
        fe = field_extract(tr.log10rho)
        b = tr.base(TX)
        ok = np.isfinite(fe["logmid"])
        d = fe["logmid"][ok] - b[ok]; d3 = fe["thr3"][ok] - b[ok]
        L.append(f"| {t} | {ok.sum()} / {len(TX)} | {np.median(d):+.3f} | {np.max(np.abs(d)):.3f} | {np.nanmedian(d3):+.3f} |")
    # direction 2: prior samples
    A = g14.load_all()
    idx = A["ktr"][:300]
    diffs, nan_t, nan_f = [], 0, 0
    for i in idx:
        th = A["th"][i]; pm = b13.PriorModel2(th)
        fe = field_extract(pm.log10rho)["logmid"]; tg = pm.target()
        lens_cols = TX >= th["x_L"]                          # target = surface landward of x_L (no interface)
        ok = lens_cols & np.isfinite(fe)
        nan_f += int((lens_cols & ~np.isfinite(fe)).sum())
        diffs.append(np.where(ok, fe - tg, np.nan))
    D = np.array(diffs)
    L += ["", "## Direction 2 — prior samples: field-based extraction on the rasterised prior model vs BEL's target", "",
          f"{len(idx)} training samples (the first 300 by id), lens columns only (x ≥ x_L).", "",
          f"- median (field − target) {np.nanmedian(D):+.3f} m; P5 / P95 {np.nanpercentile(D, 5):+.3f} / "
          f"{np.nanpercentile(D, 95):+.3f} m; share |Δ| > 0.5 m: {100*np.nanmean(np.abs(D) > 0.5):.1f} %",
          f"- lens columns where the field extraction finds no base: {nan_f} of {int(sum((TX >= A['th'][i]['x_L']).sum() for i in idx))}",
          "- per-column median (field − target), x = 150/200/250/300/350: "
          + ", ".join(f"{np.nanmedian(D[:, int(np.argmin(np.abs(TX - x)))]):+.2f}" for x in XREP)]
    big = np.nanpercentile(np.abs(D), 99)
    L.append(f"- P99 |Δ| = {big:.2f} m (large values occur where the lens base comes within the 3 m plateau window of "
             "the skin or deep unit)")
    with open(os.path.join(RES, "1a_conventions.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
def twin_theta():
    """BEL-builder parameters reproducing T1 as far as the prior parameterisation allows."""
    from scipy.optimize import least_squares
    spec4 = sb.load_spec(); T1 = sb.Truth(spec4, "T1"); v = spec4["values"]["T1"]
    ex, ez = wc.land_surface()
    lens_x = TX[(TX >= sb.X_LENS0) & ((TX < sb.X_DIS0) | (TX >= sb.X_DIS1))]
    tb = T1.base(lens_x)
    s_ctrl = np.interp(b13.CTRL_X, ex, ez)
    def model_base(depths, xs):
        th = dict(base_depth=list(depths), x_L=0.0, dis_present=False)
        from scipy.interpolate import PchipInterpolator
        f = PchipInterpolator(b13.CTRL_X, s_ctrl - np.asarray(depths), extrapolate=False)
        return f(np.clip(xs, b13.CTRL_X[0], b13.CTRL_X[-1]))
    d0 = s_ctrl - np.interp(b13.CTRL_X, lens_x, tb)
    fit = least_squares(lambda d: model_base(d, lens_x) - tb, d0, bounds=(0.5, 40))
    dis_x = TX[(TX >= sb.X_DIS0) & (TX < sb.X_DIS1)]
    skin = float(np.mean(np.interp(dis_x, ex, ez) - T1.base(dis_x)))
    th = dict(x_L=sb.X_LENS0, tau=0.0, z_wt=sb.Z_WT, z_deep=sb.Z_DEEP, base_depth=[float(x) for x in fit.x],
              rho_cap=v["cap"], rho_aquifer=v["fresh"], rho_deep=v["deep"], rho_lens=v["lens"], rho_below=v["brackish"],
              rho_discharge=v["discharge"], n_redraws=0, dis_present=True, x_S=sb.X_DIS0, dis_width=sb.X_DIS1 - sb.X_DIS0,
              dis_skin=skin, dis_bottom=sb.Z_DIS_BOT, clay_present=False, clay_x0=0.0, clay_x1=60.0, clay_top=-5.0,
              clay_thick=1.0, rho_clay=5.0)
    rms = float(np.sqrt(np.mean(fit.fun ** 2)))
    return th, rms, float(np.max(np.abs(fit.fun)))


def cmd_twin(args):
    ensure()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    spec4 = sb.load_spec(); T1 = sb.Truth(spec4, "T1")
    th, rms, mx = twin_theta()
    json.dump(th, open(os.path.join(D4, "twin_theta.json"), "w"), indent=1)
    pm = b13.PriorModel2(th)
    ex, ez = wc.land_surface()
    xg = np.arange(-5, 510.01, 0.5); zg = np.arange(-45, 8.01, 0.1)
    X, Z = np.meshgrid(xg, zg); above = Z > np.interp(X, ex, ez)
    Vt = T1.log10rho(X, Z); Vw = pm.log10rho(X, Z)
    Vt[above] = np.nan; Vw[above] = np.nan
    Dl = Vw - Vt
    fig, axes = plt.subplots(3, 1, figsize=(15, 13))
    for ax, V, ttl in ((axes[0], Vt, "T1 (Stage-4 builder)"), (axes[1], Vw, "BEL-builder twin of T1")):
        pc = ax.pcolormesh(xg, zg, 10 ** V, cmap="RdBu_r", norm=mcolors.LogNorm(0.5, 500), shading="auto", rasterized=True)
        ax.plot(ex, ez, "k-", lw=0.8); ax.plot(TX, T1.base(TX), "y--", lw=1, label="T1 base b(x)")
        ax.plot(TX, pm.target(), "m:", lw=1.3, label="twin target"); ax.set_aspect(3); ax.set_ylim(-45, 9)
        ax.set_title(ttl, fontweight="bold", fontsize=10); fig.colorbar(pc, ax=ax, fraction=0.02, label="ρ (Ω·m)")
        wc.white_legend(ax, loc="lower left")
    pc = axes[2].pcolormesh(xg, zg, Dl, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto", rasterized=True)
    axes[2].plot(ex, ez, "k-", lw=0.8); axes[2].set_aspect(3); axes[2].set_ylim(-45, 9)
    axes[2].set_title("log10ρ(twin) − log10ρ(T1)", fontweight="bold", fontsize=10)
    fig.colorbar(pc, ax=axes[2], fraction=0.02, label="Δ log10ρ")
    for ax in axes:
        ax.set_xlabel("x (m)"); ax.set_ylabel("mTAW")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "1b_twin_vs_T1.png"), dpi=140); plt.close(fig)
    valid = np.isfinite(Dl)
    L = ["# 1b — BEL-builder twin of T1", "",
         f"Control depths (below surface) fitted by least squares to T1's lens base at the lens columns "
         f"(x ≥ {sb.X_LENS0:g}, outside the discharge span): RMS {rms:.2f} m, max {mx:.2f} m. Other twin parameters: "
         f"x_L {th['x_L']:g}, τ 0, water table {th['z_wt']:g}, deep top {th['z_deep']:g} mTAW, T1's six unit resistivities; "
         f"discharge zone x {th['x_S']:g}–{th['x_S'] + th['dis_width']:g} m, skin {th['dis_skin']:.2f} m (constant; T1's "
         f"varies), bottom {th['dis_bottom']:g} mTAW; no dune layer. Parameters: `data/processed/bel_v4/twin_theta.json`.",
         "", f"Rasterised on a common grid (dx 0.5 m, dz 0.1 m, surface to −45 mTAW): share of cells with |Δlog10ρ| > 0.1: "
         f"{100*np.mean(np.abs(Dl[valid]) > 0.1):.1f} %.", "", "Largest systematic offsets (connected zones with |Δ| > 0.1):", ""]
    zones = [("seaward end x 470–510 m (T1 base descends to −25; twin base held at its x = 450 value)", (470, 510), None),
             ("discharge-zone skin x 370–470 m (T1 skin varies −1…−5; twin constant thickness)", (370, 470), (-8, 3)),
             ("lens base x 90–370 m (PCHIP fit residual)", (90, 370), (-25, 0)),
             ("dunes x < 90 m", (-5, 90), None)]
    for name, (x0, x1), zr in zones:
        sel = valid & (X >= x0) & (X < x1)
        if zr:
            sel &= (Z >= zr[0]) & (Z <= zr[1])
        dd = Dl[sel]
        L.append(f"- {name}: {100*np.mean(np.abs(dd) > 0.1):.1f} % of cells differ; area with |Δ| > 0.1 = "
                 f"{0.05*np.sum(np.abs(dd) > 0.1):.0f} m²; mean Δ where different {np.mean(dd[np.abs(dd) > 0.1]) if (np.abs(dd) > 0.1).any() else 0:+.2f}")
    tw = pm.target(); tb = T1.base(TX)
    L += ["", "Target vs T1 base per column (twin target − T1 b(x)), x = 150/200/250/300/350: "
          + ", ".join(f"{tw[int(np.argmin(np.abs(TX - x)))] - tb[int(np.argmin(np.abs(TX - x)))]:+.2f}" for x in XREP)
          + f"; over all columns median {np.median(tw - tb):+.2f}, max |Δ| {np.max(np.abs(tw - tb)):.2f} m."]
    with open(os.path.join(RES, "1b_twin.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_twinbel(args):
    ensure()
    import pygimli.meshtools as mt
    th = json.load(open(os.path.join(D4, "twin_theta.json")))
    pm = b13.PriorModel2(th)
    p = os.path.join(D4, "twin_forward.npz")
    if not os.path.exists(p):
        sch = sb.g1_scheme(); scheme = sb.make_scheme(sch); scheme["k"] = np.ones(scheme.size())
        fmesh = mt.createParaMesh(scheme.sensors(), **sb.FWD)
        cx = np.array([c.center().x() for c in fmesh.cells()]); cz = np.array([c.center().y() for c in fmesh.cells()])
        with contextlib.redirect_stdout(io.StringIO()):
            R = b13.pilot.simulate_R(fmesh, 10 ** pm.log10rho(cx, cz), scheme.size())
        np.savez(p, R=R)
    R_tw = np.load(p)["R"]
    A = g14.load_all(); bel = b13.BEL(A["Dtr"], A["Ttr"])
    spec4 = sb.load_spec()
    R_t1 = np.load(os.path.join(wc.PROCESSED_DIR, "synthetic", "truth_forward.npz"))["R_T1"]
    truths = {"twin (own target)": (R_tw, pm.target()), "twin vs T1 b(x)": (R_tw, sb.Truth(spec4, "T1").base(TX)),
              "T1 (validation truth)": (R_t1, c11.truth_defs(spec4, "T1", TX)["logmid"])}
    L = ["# 1c — BEL-KDE on the twin (fine mesh, Stage-4 noise seeds 1000 + r, 20 realisations)", "",
         "Full prior-v2 BEL (same training set, PCA/CCA/KDE settings as the Stage-5b run). Columns x = 130–450 m. "
         f"Posterior seeds default_rng([{POST}, 100·set + r]).", "",
         "| data / truth | n | coverage | bias (m) | width (m) | bias at x = 150/200/250/300/350 |", "|---|---|---|---|---|---|"]
    cols = TX >= 130
    for si, (name, (R, tv)) in enumerate(truths.items()):
        cov, bias, wid, per = [], [], [], []
        for r in range(20):
            Rn = R + np.sqrt((0.03 * R) ** 2 + 1e-8) * np.random.default_rng(1000 + r).standard_normal(len(R))
            if not np.all(Rn > 0):
                continue
            P = bel.posterior(np.log10(Rn), N_POST, np.random.default_rng([POST, 100 * si + r]), "kde")
            lo, md, hi = np.percentile(P, [5, 50, 95], axis=0)
            cov.append(((tv >= lo) & (tv <= hi))[cols]); bias.append((md - tv)[cols]); wid.append((hi - lo)[cols])
            per.append([md[int(np.argmin(np.abs(TX - x)))] - tv[int(np.argmin(np.abs(TX - x)))] for x in XREP])
        L.append(f"| {name} | {len(cov)}×{cols.sum()} | {100*np.mean(cov):.0f}% | {np.median(bias):+.2f} | {np.median(wid):.2f} | "
                 + ", ".join(f"{v:+.2f}" for v in np.median(per, axis=0)) + " |")
        print(L[-1], flush=True)
    with open(os.path.join(RES, "1c_twin_bel.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
def get_Y(A):
    """Second-BEL targets for training + held-out (cached)."""
    p = os.path.join(D4, "second_bel_targets.npz")
    if os.path.exists(p):
        z = np.load(p); return z["Ytr"], z["Yho"]
    Ytr = np.array([g14.derived_targets(A["th"][i]) for i in A["ktr"]])
    Yho = np.array([g14.derived_targets(A["th"][i]) for i in A["kho"]])
    np.savez(p, Ytr=Ytr, Yho=Yho)
    return Ytr, Yho


class BEL2:
    """Second BEL: data PCA (refit on the given training set) + CCA on standardised (optionally bounded) targets."""

    def __init__(self, D, Y, bounded=False):
        self.bounded = bounded
        self.pd = b13.PCA(D, 0.99)
        self.Sd = self.pd.fwd(D)
        Yt = self.fwdY(Y)
        self.ym, self.ys = Yt.mean(0), Yt.std(0)
        self.cca = b13.CCA(self.Sd, (Yt - self.ym) / self.ys)
        self.Cd = self.cca.cx(self.Sd); self.Ct = self.cca.cy((Yt - self.ym) / self.ys)

    def fwdY(self, Y):
        Y = np.array(Y, float)
        if self.bounded:
            for j, k in ((2, "rho_lens"), (3, "rho_below"), (4, "rho_discharge")):
                lo, hi = np.log10(RHO_BOUNDS[k])
                u = np.clip((Y[:, j] - lo) / (hi - lo), 1e-6, 1 - 1e-6)
                Y[:, j] = np.log(u / (1 - u))
        return Y

    def invY(self, Yt):
        Y = np.array(Yt, float)
        if self.bounded:
            for j, k in ((2, "rho_lens"), (3, "rho_below"), (4, "rho_discharge")):
                lo, hi = np.log10(RHO_BOUNDS[k])
                Y[:, j] = lo + (hi - lo) / (1 + np.exp(-Y[:, j]))
        return Y

    def posterior(self, y, n, rng):
        cd = self.cca.cx(self.pd.fwd(y))[0]
        C = np.column_stack([b13.kde_conditional_sample(self.Cd[:, i], self.Ct[:, i], cd[i], n, rng)
                             for i in range(self.Ct.shape[1])])
        return self.invY(self.cca.y_from_c(C) * self.ys + self.ym)


def fmt(v):
    lo, md, hi = np.percentile(v, [5, 50, 95]); return f"{md:.2f} [{lo:.2f}, {hi:.2f}]"


def cmd_site(args):
    ensure()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = g14.load_all(); obs, _ = b13.observed_sets(); Ytr, Yho = get_Y(A)
    thtr = [A["th"][i] for i in A["ktr"]]; thho = [A["th"][i] for i in A["kho"]]
    def site(t, T):
        lens = TX >= t["x_L"]
        return (-38.0 <= t["z_deep"] <= -22.0) and bool(np.all(T[lens] > t["z_deep"]))
    mtr = np.array([site(t, T) for t, T in zip(thtr, A["Ttr"])]); mho = np.array([site(t, T) for t, T in zip(thho, A["Tho"])])
    full = b13.BEL(A["Dtr"], A["Ttr"]); sbel = b13.BEL(A["Dtr"][mtr], A["Ttr"][mtr])
    def cover(bel, D, T, seed0):
        c = []
        for j, (y, t) in enumerate(zip(D, T)):
            P = bel.posterior(y, N_POST, np.random.default_rng([POST, seed0 + j]), "kde"); lo, hi = np.percentile(P, [5, 95], 0)
            c.append((t >= lo) & (t <= hi))
        return 100 * np.mean(c)
    cov_site = cover(sbel, A["Dho"][mho], A["Tho"][mho], 40_000)
    Pf_full = full.posterior(obs["field"], N_POST, np.random.default_rng([POST, 1]), "kde")
    Pf_site = sbel.posterior(obs["field"], N_POST, np.random.default_rng([POST, 2]), "kde")
    L = ["# 2 — site constraints (Paepen et al. 2020, HESS 24:3539, citing Lebbe 1981)", "",
         "Kortrijk-clay aquifer base −25…−30 mTAW at K1, ≈ −35 mTAW at K0 → training subset **site**: deep-unit top in "
         "[−38, −22] mTAW AND the lens base (all lens columns) above the deep-unit top. PCA/CCA/KDE refitted on the subset; "
         f"held-out filtered the same way. Posterior seeds default_rng([{POST}, k]).", "",
         f"## 2a — site subset: {mtr.sum()} training / {mho.sum()} held-out samples (of {len(mtr)} / {len(mho)})", "",
         f"Held-out 90 % coverage (lens base): {cov_site:.1f} % (full prior: 89.8 %).", "",
         "| x | full prior: median [P5, P95] | site: median [P5, P95] | shift of the median |", "|---|---|---|---|"]
    for x in XREP:
        j = int(np.argmin(np.abs(TX - x)))
        L.append(f"| {x} | {fmt(Pf_full[:, j])} | {fmt(Pf_site[:, j])} | {np.median(Pf_site[:, j]) - np.median(Pf_full[:, j]):+.2f} |")
    # second BEL on the site subset vs full
    b2f = BEL2(A["Dtr"], Ytr); b2s = BEL2(A["Dtr"][mtr], Ytr[mtr])
    P2f = b2f.posterior(obs["field"], N_POST, np.random.default_rng([POST, 3]))
    P2s = b2s.posterior(obs["field"], N_POST, np.random.default_rng([POST, 4]))
    cs = []
    for j, (y, t) in enumerate(zip(A["Dho"][mho], Yho[mho])):
        P = b2s.posterior(y, N_POST, np.random.default_rng([POST, 50_000 + j])); lo, hi = np.percentile(P, [5, 95], 0)
        cs.append((t >= lo) & (t <= hi))
    cs = np.array(cs)
    L += ["", "Second BEL (D1, D2, log10 ρ): field median [P5, P95]", "",
          "| target | full prior | site | site held-out coverage |", "|---|---|---|---|"]
    for i, n in enumerate(TNAMES):
        L.append(f"| {n}{' (log10)' if n.startswith('rho') else ''} | {fmt(P2f[:, i])} | {fmt(P2s[:, i])} | {100*cs[:, i].mean():.1f}% |")
    L += [f"", f"P(D1 > 0): full {np.mean(P2f[:, 0] > 0):.2f}, site {np.mean(P2s[:, 0] > 0):.2f}; "
          f"P(10^D2 < 3 Ω·m): full {np.mean(10**P2f[:, 1] < 3):.2f}, site {np.mean(10**P2s[:, 1] < 3):.2f}"]
    # 2b — ρ_below bins within the site subset
    bins = [(2, 4), (4, 8), (8, 20)]
    L += ["", "## 2b — conditional lens-base product: ρ_below bins within the site subset", "",
          "| ρ_below bin (Ω·m) | n train / held-out | held-out coverage | " + " | ".join(f"x={x}" for x in XREP) + " |",
          "|---|---|---|" + "---|" * len(XREP)]
    rb_tr = np.array([t["rho_below"] for t in thtr]); rb_ho = np.array([t["rho_below"] for t in thho])
    curves = {}
    for k, (lo, hi) in enumerate(bins):
        mt_ = mtr & (rb_tr >= lo) & (rb_tr < hi); mh_ = mho & (rb_ho >= lo) & (rb_ho < hi)
        bb = b13.BEL(A["Dtr"][mt_], A["Ttr"][mt_])
        cv = cover(bb, A["Dho"][mh_], A["Tho"][mh_], 60_000 + 1000 * k)
        P = bb.posterior(obs["field"], N_POST, np.random.default_rng([POST, 10 + k]), "kde")
        curves[(lo, hi)] = {x: np.percentile(P[:, int(np.argmin(np.abs(TX - x)))], [5, 50, 95]) for x in XREP}
        L.append(f"| {lo}–{hi} | {mt_.sum()} / {mh_.sum()} | {cv:.1f}% | "
                 + " | ".join(f"{curves[(lo, hi)][x][1]:.2f} [{curves[(lo, hi)][x][0]:.2f}, {curves[(lo, hi)][x][2]:.2f}]" for x in XREP) + " |")
        print(L[-1], flush=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xpos = np.arange(len(bins))
    for x, c in zip(XREP, plt.cm.viridis(np.linspace(0, 0.9, len(XREP)))):
        md = [curves[b][x][1] for b in bins]; lo_ = [curves[b][x][0] for b in bins]; hi_ = [curves[b][x][2] for b in bins]
        off = (XREP.index(x) - 2) * 0.06
        ax.errorbar(xpos + off, md, yerr=[np.array(md) - lo_, np.array(hi_) - md], fmt="-o", color=c, capsize=3,
                    label=f"x = {x} m")
    ax.set_xticks(xpos); ax.set_xticklabels([f"{a}–{b} Ω·m" for a, b in bins])
    ax.set_xlabel("ρ_below bin (site subset)"); ax.set_ylabel("lens-base elevation (mTAW): median, 90 %")
    ax.set_title("Field lens base conditional on the resistivity of the water below the lens", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3); wc.white_legend(ax, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "2b_base_vs_rho_below.png"), dpi=150); plt.close(fig)
    with open(os.path.join(RES, "2_site_constraints.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
def bounded_posterior():
    A = g14.load_all(); obs, _ = b13.observed_sets(); Ytr, Yho = get_Y(A)
    bb = BEL2(A["Dtr"], Ytr, bounded=True)
    Pf = bb.posterior(obs["field"], N_POST, np.random.default_rng([POST, 5]))
    return A, bb, Pf, Ytr, Yho


def cmd_bounded(args):
    ensure()
    A, bb, Pf, Ytr, Yho = bounded_posterior()
    np.savez(os.path.join(D4, "bounded_field_posterior.npz"), field=Pf, names=np.array(TNAMES))
    cov = []
    for j, (y, t) in enumerate(zip(A["Dho"], Yho)):
        P = bb.posterior(y, N_POST, np.random.default_rng([POST, 70_000 + j])); lo, hi = np.percentile(P, [5, 95], 0)
        cov.append((t >= lo) & (t <= hi))
    cov = np.array(cov)
    lo, hi = np.log10(RHO_BOUNDS["rho_discharge"])
    top = lo + 0.9 * (hi - lo)
    share = float(np.mean(Pf[:, 4] > top)); prior_share = float(np.mean(Ytr[:, 4] > top))
    L = ["# 3 — second BEL with bounded resistivity targets", "",
         "Each bounded resistivity target transformed as logit((log ρ − log lo)/(log hi − log lo)) before CCA/KDE and "
         f"back-transformed after sampling (bounds {RHO_BOUNDS}); D1, D2 unbounded. Posterior seed default_rng([{POST}, 5]).", "",
         "| target | held-out coverage | field median [P5, P95] |", "|---|---|---|"]
    for i, n in enumerate(TNAMES):
        v = Pf[:, i]
        s = (f"{10**np.median(v):.2f} [{10**np.percentile(v, 5):.2f}, {10**np.percentile(v, 95):.2f}] Ω·m" if n.startswith("rho")
             else fmt(v))
        L.append(f"| {n} | {100*cov[:, i].mean():.1f}% | {s} |")
    L += ["", f"ρ_discharge: posterior mass in the top 10 % of its prior range (log-uniform 2.5–40 Ω·m, i.e. above "
          f"{10**top:.1f} Ω·m): **{100*share:.1f} %** (prior: {100*prior_share:.1f} %).",
          ("**FLAG: above 20 % — the 40 Ω·m ceiling is probably too low.**" if share > 0.2 else
           "Below the 20 % flag — the 40 Ω·m ceiling is not contradicted."),
          f"All posterior samples within the prior range by construction (max {10**Pf[:, 4].max():.1f} Ω·m)."]
    with open(os.path.join(RES, "3_bounded_second_bel.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_archie(args):
    ensure()
    Pf = np.load(os.path.join(D4, "bounded_field_posterior.npz"))["field"]
    rng = np.random.default_rng(ARCHIE_SEED)
    n = len(Pf)
    F = rng.uniform(2.8, 3.6, n); rsw = rng.uniform(0.2, 0.3, n)
    T, alpha = 11.0, 0.02
    L = ["# 4 — Stage 6 preview: Archie conversion (table only)", "",
         "ρ_w = ρ_bulk / F with F ~ U(2.8, 3.6) (centred on 3.2, Paepen et al. 2020), drawn independently per posterior "
         f"sample (seed {ARCHIE_SEED}); posteriors from the bounded second BEL (step 3). EC at field temperature "
         "EC_T = 10⁴/ρ_w μS/cm; corrected to 25 °C with the linear compensation EC25 = EC_T / (1 + α(T − 25)), "
         f"α = {alpha} /°C, **assumed groundwater temperature T = {T:g} °C** (typical of shallow coastal groundwater in "
         f"Belgium — please confirm). % seawater = 100·ρ_sw/ρ_w with ρ_sw ~ U(0.2, 0.3) Ω·m at the same temperature "
         "(two-end-member mixing by EC; the fresh end-member's EC neglected). Median [P5, P95].", "",
         "| unit | ρ_bulk (Ω·m) | ρ_w (Ω·m) | EC25 (μS/cm) | % seawater |", "|---|---|---|---|---|"]
    for j, name in ((2, "saline lens"), (3, "water below the lens"), (4, "discharge-zone water")):
        rb = 10 ** Pf[:, j]; rw = rb / F
        ec25 = 1e4 / rw / (1 + alpha * (T - 25))
        sw = 100 * rsw / rw
        q = lambda v, d=2: f"{np.median(v):.{d}f} [{np.percentile(v, 5):.{d}f}, {np.percentile(v, 95):.{d}f}]"
        L.append(f"| {name} | {q(rb)} | {q(rw, 3)} | {q(ec25, 0)} | {q(sw, 0)} |")
    L += ["", "Caveats: Archie with a single F ignores surface conduction (clay-free sands assumed) and applies the same "
          "F to all units; ρ_discharge acts only where a discharge zone exists (P = 1.00 for the field); % seawater > 100 "
          "is not physical and indicates F or ρ_sw at the edge of their ranges."]
    with open(os.path.join(RES, "4_archie_preview.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("conventions", "twin", "twinbel", "site", "bounded", "archie"):
        sub.add_parser(c)
    a = ap.parse_args()
    dict(conventions=cmd_conventions, twin=cmd_twin, twinbel=cmd_twinbel, site=cmd_site, bounded=cmd_bounded,
         archie=cmd_archie)[a.cmd](a)


if __name__ == "__main__":
    main()
