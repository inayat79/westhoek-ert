"""
stage6_analysis.py — analysis half of 16_stage6_salinity.py (refit, joint BEL, salinity sections).

Joint target (one CCA): lens-base PCA scores + logit-bounded unit / geometry parameters. Beyond the list in the
Stage-6 brief, x_L, ρ_aquifer, the discharge-zone width and bottom are included because the salinity section cannot
be built without them (dunes and discharge-zone extent). Discharge presence is sampled per posterior sample from a
cross-validated logistic classifier on the data PCA scores (field P ≈ 1). The base transition τ and the dune
conductive layer are not built into the salinity section (sharp base at the transition centre; clay cannot be told
from saltwater by ERT, so the dune zone x < 90 m is flagged weakly constrained everywhere).
"""

import json
import os

import numpy as np

import westhoek_common as wc

S16 = None          # set by init()


def init():
    global S16, s15, g14, b13
    import importlib.util, contextlib, io
    sp = importlib.util.spec_from_file_location("s16", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "16_stage6_salinity.py"))
    S16 = importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()):
        sp.loader.exec_module(S16)
    s15, g14, b13 = S16.s15, S16.g14, S16.b13


TX = None
BOUNDS = {"rho_lens": (0.4, 3.0, True), "rho_below": (2.0, 20.0, True), "rho_discharge": (2.5, 150.0, True),
          "rho_aquifer": (5.0, 60.0, True), "z_wt": (1.0, 5.0, False), "z_deep": (-38.0, -22.0, False),
          "dis_skin": (0.5, 5.0, False), "x_S": (330.0, 420.0, False), "dis_width": (30.0, 120.0, False),
          "dis_bottom": (-25.0, -8.0, False), "x_L": (50.0, 130.0, False)}
PKEYS = list(BOUNDS)
XG = np.arange(0.0, 505.01, 1.0)
ZG = np.arange(6.5, -44.51, -1.0)            # 1 m cell centres
N_POST = 2000


# ─────────────────────────────────────────────────────────────────────────────
# Physical filter (v1.0 freeze): the lens must be able to hold ≤ 100 % seawater for SOME Archie draw, i.e.
# ρ_lens ≥ min(F) · min(ρ_sw) = 2.8 × 0.247911 Ω·m (PSS-78, S = 34 PSU, T = 14 °C; see 17_salinity_v2.py, which
# asserts this value) = 0.69415 Ω·m.
PHYS_RHO_LENS_MIN = 0.69415


def load_combined(site_only=True, physical=False):
    """Prior v2 + v3 training / held-out: corrected noisy data, targets, parameters (site-filtered; with physical=True
    also ρ_lens ≥ PHYS_RHO_LENS_MIN). Defaults reproduce Stage 6."""
    mu_d, dev = b13.approx_stats()
    v2tr = b13.load_set("train", set(range(b13.N_TRAIN)))
    v2ho = b13.load_set("train", set(range(b13.N_TRAIN, b13.N_TRAIN + b13.N_HELD)))
    d3 = os.path.join(S16.D3, "train")
    v3 = []
    for f in sorted(os.listdir(d3)):
        if f.endswith(".npz") and "tmp" not in f:
            z = np.load(os.path.join(d3, f), allow_pickle=True)
            v3.append((int(f[1:-4]), {k: z[k] for k in z.files}))
    v3tr = [x for x in v3 if x[0] < S16.ID0 + S16.N_NEW_TRAIN]; v3ho = [x for x in v3 if x[0] >= S16.ID0 + S16.N_NEW_TRAIN]
    out = {}
    for name, sets in (("tr", v2tr + v3tr), ("ho", v2ho + v3ho)):
        D, keep = b13.prior_data(sets, mu_d, dev)
        zm = {i: z for i, z in sets}
        th = [json.loads(str(zm[i]["theta"])) for i in keep]
        T = np.array([zm[i]["target"] for i in keep])
        if site_only:
            m = np.array([site(t, tt) and (not physical or t["rho_lens"] >= PHYS_RHO_LENS_MIN) for t, tt in zip(th, T)])
            D, T, th, keep = D[m], T[m], [t for t, k in zip(th, m) if k], [i for i, k in zip(keep, m) if k]
        out[name] = dict(D=D, T=T, th=th, ids=keep)
    out["n_v3"] = (len(v3tr), len(v3ho))
    return out


def site(t, T):
    lens = TX >= t["x_L"]
    return (-38.0 <= t["z_deep"] <= -22.0) and bool(np.all(T[lens] > t["z_deep"]))


def pvec(t):
    v = []
    for k in PKEYS:
        lo, hi, lg = BOUNDS[k]
        x = t[k]
        x = np.log10(x) if lg else x
        a, b = (np.log10(lo), np.log10(hi)) if lg else (lo, hi)
        u = np.clip((x - a) / (b - a), 1e-6, 1 - 1e-6)
        v.append(np.log(u / (1 - u)))
    return np.array(v)


def pinv(V):
    out = {}
    for j, k in enumerate(PKEYS):
        lo, hi, lg = BOUNDS[k]
        a, b = (np.log10(lo), np.log10(hi)) if lg else (lo, hi)
        x = a + (b - a) / (1 + np.exp(-V[:, j]))
        out[k] = 10 ** x if lg else x
    return out


class JointBEL:
    def __init__(self, D, T, th):
        self.pd = b13.PCA(D, 0.99); self.pt = b13.PCA(T, 0.99)
        Sd = self.pd.fwd(D); St = self.pt.fwd(T)
        P = np.array([pvec(t) for t in th])
        Y = np.hstack([St, P])
        self.ym, self.ys = Y.mean(0), Y.std(0)
        self.cca = b13.CCA(Sd, (Y - self.ym) / self.ys)
        self.Cd = self.cca.cx(Sd); self.Ct = self.cca.cy((Y - self.ym) / self.ys)
        self.kt = self.pt.k
        sd = np.sqrt(Sd.var(0, ddof=1)); self.sd = sd
        ydis = np.array([float(t["dis_present"]) for t in th])
        self.p_dis, self.C_dis, _ = g14.logistic_cv(Sd / sd, ydis)

    def posterior(self, y, n, rng):
        cd = self.cca.cx(self.pd.fwd(y))[0]
        C = np.column_stack([b13.kde_conditional_sample(self.Cd[:, i], self.Ct[:, i], cd[i], n, rng)
                             for i in range(self.Ct.shape[1])])
        Y = self.cca.y_from_c(C) * self.ys + self.ym
        base = self.pt.inv(Y[:, :self.kt])
        par = pinv(Y[:, self.kt:])
        pdis = float(self.p_dis(self.pd.fwd(y) / self.sd)[0])
        par["dis_present"] = rng.uniform(size=n) < pdis
        return base, par, pdis


def truth_params(t, T):
    par = {k: np.array([t[k]]) for k in PKEYS}
    par["dis_present"] = np.array([bool(t["dis_present"])])
    return T[None, :], par


# ─────────────────────────────────────────────────────────────────────────────
def build_sections(base, par, archie):
    """Vectorised salinity sections: returns (valid mask (nz,nx), %seawater (ns,nz,nx), EC25 (ns,nz,nx))."""
    ex, ez = wc.land_surface()
    s = np.interp(XG, ex, ez)
    n = base.shape[0]
    Xc, Zc = np.meshgrid(XG, ZG)
    bx = np.array([np.interp(XG, TX, b) for b in base])                   # base held constant beyond 100 / 450
    top = np.minimum(s[None, :], par["z_wt"][:, None])                    # water table / land surface
    valid = (Zc[None] < top[:, None, :]) & (Zc[None] > par["z_deep"][:, None, None])
    L = lambda k: np.asarray(par[k], float)[:, None, None]
    rho = np.broadcast_to(L("rho_below"), (n,) + Xc.shape).copy()
    dune = Xc[None] < L("x_L")
    lens = ~dune & (Zc[None] > bx[:, None, :])
    x0, x1 = L("x_S"), np.minimum(L("x_S") + L("dis_width"), 505.0)
    dis = (np.asarray(par["dis_present"])[:, None, None] & ~dune & ~lens & (Xc[None] >= x0) & (Xc[None] < x1)
           & (Zc[None] > L("dis_bottom")))
    rho = np.where(dune, L("rho_aquifer"), rho)
    rho = np.where(lens, L("rho_lens"), rho)
    rho = np.where(dis, L("rho_discharge"), rho)
    F, rsw, T = (archie[k][:, None, None] for k in ("F", "rsw", "T"))
    rw = rho / F
    sw = 100 * rsw / rw
    ec25 = 1e4 / rw / (1 + 0.02 * (T - 25))
    sw = np.where(valid, sw, np.nan); ec25 = np.where(valid, ec25, np.nan)
    return valid, sw, ec25


def archie_draws(n, rng, F=None):
    return dict(F=np.full(n, F) if F else rng.uniform(2.8, 3.6, n), rsw=rng.uniform(0.2, 0.3, n), T=rng.uniform(8, 14, n))


def isohaline(sw):
    """50 % isohaline per column: lower edge of the saline (> 50 %) zone connected to the top of the saturated
    section; NaN where the top cell is not saline."""
    ns, nz, nx = sw.shape
    fin = np.isfinite(sw); sal = fin & (sw > 50)
    anyv = fin.any(axis=1)
    first = np.argmax(fin, axis=1)                                         # top valid cell
    last = nz - 1 - np.argmax(fin[:, ::-1, :], axis=1)                     # bottom valid cell
    top_sal = np.take_along_axis(sal, first[:, None, :], 1)[:, 0, :]
    ii = np.arange(nz)[None, :, None]
    brk = fin & ~sal & (ii > first[:, None, :])
    has = brk.any(axis=1)
    kb = np.where(has, np.argmax(brk, axis=1), last + 1)                   # first non-saline cell (or below the section)
    zedge = ZG[0] + 0.5 - kb.astype(float)                                 # top edge of cell kb
    return np.where(anyv & top_sal, zedge, np.nan)


def areas(sw, ec25):
    dune = XG < 90
    res = {}
    for zone, m in (("dunes", dune), ("beach", ~dune)):
        s = sw[:, :, m]; e = ec25[:, :, m]; fin = np.isfinite(s)
        sal = (s > 50) & fin; fr = (e < 1500) & fin
        res[f"saline_{zone}"] = sal.sum(axis=(1, 2)).astype(float)
        res[f"fresh_{zone}"] = fr.sum(axis=(1, 2)).astype(float)
        res[f"brackish_{zone}"] = (fin & ~sal & ~fr).sum(axis=(1, 2)).astype(float)
    return res


def sections_stats(base, par, rng, F=None, chunk=250):
    """Maps, isohaline and areas over posterior samples (chunked)."""
    n = base.shape[0]
    Psal = Pfr = Nv = None; iso = []; ar = None
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        p = {k: np.asarray(v)[sl] for k, v in par.items()}
        a = archie_draws(len(base[sl]), rng, F)
        valid, sw, ec = build_sections(base[sl], p, a)
        fin = np.isfinite(sw)
        ps, pf = (sw > 50).sum(0), (ec < 1500).sum(0); nv = fin.sum(0)
        Psal = ps if Psal is None else Psal + ps; Pfr = pf if Pfr is None else Pfr + pf; Nv = nv if Nv is None else Nv + nv
        iso.append(isohaline(sw))
        A = areas(sw, ec)
        ar = A if ar is None else {k: np.concatenate([ar[k], A[k]]) for k in A}
    with np.errstate(invalid="ignore"):
        return dict(Psal=Psal / Nv, Pfresh=Pfr / Nv, Nv=Nv, iso=np.vstack(iso), areas=ar)


def q(v, d=1):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not v.size:
        return "—"
    lo, md, hi = np.percentile(v, [5, 50, 95]); return f"{md:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ─────────────────────────────────────────────────────────────────────────────
def cmd_refit(args):
    init(); S16.ensure()
    global TX
    TX = S16.TX
    C = load_combined()
    obs, _ = b13.observed_sets()
    n3 = C["n_v3"]
    L = ["# Step 1 — prior v3 (ρ_discharge ceiling 150 Ω·m): refit on the site subset", "",
         f"Combined prior v2 + v3 ({n3[0]} + {n3[1]} extra samples forwarded), site subset: {len(C['tr']['D'])} training / "
         f"{len(C['ho']['D'])} held-out.", ""]
    bel = b13.BEL(C["tr"]["D"], C["tr"]["T"])
    cov = []
    for j, (y, t) in enumerate(zip(C["ho"]["D"], C["ho"]["T"])):
        P = bel.posterior(y, N_POST, np.random.default_rng([S16.JOINT_SEED, 90_000 + j]), "kde")
        lo, hi = np.percentile(P, [5, 95], 0); cov.append((t >= lo) & (t <= hi))
    Pf = bel.posterior(obs["field"], N_POST, np.random.default_rng([S16.JOINT_SEED, 1]), "kde")
    L += [f"Lens-base BEL (site, prior v3): held-out 90 % coverage **{100*np.mean(cov):.1f} %**.", "",
          "| x | field lens base median [P5, P95] (mTAW) |", "|---|---|"]
    L += [f"| {x} | {q(Pf[:, int(np.argmin(np.abs(TX - x)))], 2)} |" for x in (150, 200, 250, 300, 350)]
    # bounded second BEL with the new ceiling
    s15.RHO_BOUNDS["rho_discharge"] = (2.5, 150.0)
    Ytr = np.array([g14.derived_targets(t) for t in C["tr"]["th"]]); Yho = np.array([g14.derived_targets(t) for t in C["ho"]["th"]])
    b2 = s15.BEL2(C["tr"]["D"], Ytr, bounded=True)
    P2 = b2.posterior(obs["field"], N_POST, np.random.default_rng([S16.JOINT_SEED, 2]))
    c2 = []
    for j, (y, t) in enumerate(zip(C["ho"]["D"], Yho)):
        P = b2.posterior(y, N_POST, np.random.default_rng([S16.JOINT_SEED, 95_000 + j])); lo, hi = np.percentile(P, [5, 95], 0)
        c2.append((t >= lo) & (t <= hi))
    c2 = np.array(c2)
    lo, hi = np.log10(2.5), np.log10(150.0); top = lo + 0.9 * (hi - lo)
    share = float(np.mean(P2[:, 4] > top)); pshare = float(np.mean(Ytr[:, 4] > top))
    rd = 10 ** P2[:, 4]
    L += ["", "Bounded second BEL (ρ_discharge bounds 2.5–150 Ω·m; site subset):", "",
          "| target | held-out coverage | field median [P5, P95] |", "|---|---|---|"]
    for i, n in enumerate(s15.TNAMES):
        v = 10 ** P2[:, i] if n.startswith("rho") else P2[:, i]
        L.append(f"| {n} | {100*c2[:, i].mean():.1f}% | {q(v, 2)}{' Ω·m' if n.startswith('rho') else ''} |")
    L += ["", f"ρ_discharge posterior {q(rd, 1)} Ω·m; mass in the top 10 % of the new prior range (log-uniform, above "
          f"{10**top:.1f} Ω·m): **{100*share:.1f} %** (prior {100*pshare:.1f} %) → "
          + ("**FLAG: > 20 %, ceiling may still be too low.**" if share > 0.2 else
             "**borderline**: just under the 20 % flag but twice the prior share; the posterior is broad and the data do "
             "not bound ρ_discharge from above." if share > 0.15 else "below the 20 % flag.")]
    np.savez(os.path.join(S16.D3, "refit_field.npz"), lens_base=Pf, second=P2)
    with open(os.path.join(S16.RES, "1_prior_v3_refit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def dune_flag(ax):
    ax.axvspan(0, 90, color="0.85", alpha=0.6, zorder=0, lw=0)
    ax.text(45, 0.97, "dunes: weakly\nconstrained", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=7.5, color="0.2", bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))


def cmd_joint(args):
    init(); S16.ensure()
    global TX
    TX = S16.TX
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = load_combined()
    obs, _ = b13.observed_sets()
    J = JointBEL(C["tr"]["D"], C["tr"]["T"], C["tr"]["th"])
    base_bel = b13.BEL(C["tr"]["D"], C["tr"]["T"])
    L = ["# Steps 2–3 — joint BEL and salinity sections (site subset, prior v3)", "",
         f"Joint target: {J.kt} lens-base PCA scores + {len(PKEYS)} logit-bounded parameters ({', '.join(PKEYS)}); "
         f"x_L, ρ_aquifer, discharge width and bottom added to the brief's list (needed to build the section). "
         f"CCA pairs {J.Ct.shape[1]}, canonical correlations {J.cca.r[0]:.2f} … {J.cca.r[-1]:.2f}. Discharge presence "
         f"from a CV logistic classifier (C = {J.C_dis:g}). Posterior seeds default_rng([{S16.JOINT_SEED}, k]); Archie "
         f"draws default_rng([{S16.ARCHIE_SEED}, k]).", ""]
    # ── held-out coverage per component ─────────────────────────────────────
    nh = min(getattr(args, "n_held", 300) or 300, len(C["ho"]["D"]))
    covb, covp, cov_area, rel = [], [], [], []
    for j in range(nh):
        y, t, th = C["ho"]["D"][j], C["ho"]["T"][j], C["ho"]["th"][j]
        rng = np.random.default_rng([S16.JOINT_SEED, 200_000 + j])
        base, par, _ = J.posterior(y, 500, rng)
        lo, hi = np.percentile(base, [5, 95], 0); covb.append((t >= lo) & (t <= hi))
        cp = []
        for k in PKEYS:
            v = np.asarray(par[k], float); lo_, hi_ = np.percentile(v, [5, 95]); cp.append(lo_ <= th[k] <= hi_)
        covp.append(cp)
        # sections: posterior vs truth (truth with its own Archie draw)
        st = sections_stats(base, par, np.random.default_rng([S16.ARCHIE_SEED, 200_000 + j]))
        tb, tpar = truth_params(th, t)
        ta = archie_draws(1, np.random.default_rng([S16.ARCHIE_SEED, 500_000 + j]))
        _, tsw, tec = build_sections(tb, tpar, ta)
        tA = areas(tsw, tec)
        cov_area.append({k: (np.percentile(st["areas"][k], 5) <= tA[k][0] <= np.percentile(st["areas"][k], 95)) for k in tA})
        ok = np.isfinite(tsw[0]) & np.isfinite(st["Psal"])
        rel.append(np.column_stack([st["Psal"][ok], (tsw[0][ok] > 50).astype(float)]))
    covb, covp = np.array(covb), np.array(covp)
    L += ["## Step 2 — held-out coverage per component (90 % intervals, 500 joint posterior samples each)", "",
          f"{nh} held-out site samples. Lens base (all {len(TX)} columns): **{100*covb.mean():.1f} %** "
          f"(per column {100*covb.mean(0).min():.0f}–{100*covb.mean(0).max():.0f} %).", "",
          "| parameter | coverage |", "|---|---|"]
    L += [f"| {k} | {100*covp[:, i].mean():.1f}% |" for i, k in enumerate(PKEYS)]
    # lens base: joint vs base-only
    rngf = np.random.default_rng([S16.JOINT_SEED, 0])
    base_f, par_f, pdis_f = J.posterior(obs["field"], N_POST, rngf)
    Pb = base_bel.posterior(obs["field"], N_POST, np.random.default_rng([S16.JOINT_SEED, 1]), "kde")
    L += ["", "Lens base, field: joint BEL vs base-only BEL (both site, prior v3):", "",
          "| x | joint median [P5, P95] | base-only median [P5, P95] | shift | within ±1.3 m |", "|---|---|---|---|---|"]
    for x in S16.XREP:
        j = int(np.argmin(np.abs(TX - x))); sh = np.median(base_f[:, j]) - np.median(Pb[:, j])
        L.append(f"| {x} | {q(base_f[:, j], 2)} | {q(Pb[:, j], 2)} | {sh:+.2f} | {'yes' if abs(sh) <= 1.3 else '**no**'} |")
    L += ["", f"Field joint posterior parameters (P(discharge present) = {pdis_f:.2f}): " + "; ".join(
        f"{k} {q(par_f[k], 2)}" for k in PKEYS)]
    # ── Step 3 — field salinity sections ─────────────────────────────────────
    st = sections_stats(base_f, par_f, np.random.default_rng([S16.ARCHIE_SEED, 0]))
    ex, ez = wc.land_surface()
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    for ax, M, ttl, cm in ((axes[0], st["Psal"], "P(% seawater > 50)", "Blues"), (axes[1], st["Pfresh"], "P(EC25 < 1500 μS/cm) — fresh", "Greens")):
        pc = ax.pcolormesh(XG, ZG, np.ma.masked_invalid(np.where(st["Nv"] > 0.5 * N_POST, M, np.nan)), cmap=cm, vmin=0, vmax=1,
                           shading="auto", rasterized=True)
        ax.plot(ex, ez, "k-", lw=0.8); dune_flag(ax)
        ax.set_xlim(0, 505); ax.set_ylim(-45, 8); ax.set_aspect(3)
        ax.set_xlabel("x (m)"); ax.set_ylabel("mTAW"); ax.set_title(f"{ttl} — field, joint BEL + Archie (V.E. 3×)", fontsize=10,
                                                                   fontweight="bold")
        fig.colorbar(pc, ax=ax, fraction=0.02, label="probability")
    fig.tight_layout(); fig.savefig(os.path.join(S16.FIG, "3a_salinity_probability_maps.png"), dpi=150); plt.close(fig)
    # (b) isohaline vs parametric base
    iso = st["iso"]
    L += ["", "## Step 3 — field salinity (Archie: F ~ U(2.8, 3.6), ρ_sw ~ U(0.2, 0.3) Ω·m, T ~ U(8, 14) °C, α = 0.02 /°C)", "",
          "### (b) 50 % seawater isohaline (salinity-defined lens base) vs the BEL parametric base, mTAW", "",
          "| x | isohaline median [P5, P95] (share of samples with a saline top) | parametric base (joint) |", "|---|---|---|"]
    for x in S16.XREP:
        k = int(np.argmin(np.abs(XG - x))); j = int(np.argmin(np.abs(TX - x)))
        L.append(f"| {x} | {q(iso[:, k], 2)} ({100*np.mean(np.isfinite(iso[:, k])):.0f} %) | {q(base_f[:, j], 2)} |")
    L += ["", "The isohaline coincides with the parametric base by construction (homogeneous lens resistivity, sharp base): "
          "its added information is the share of samples whose lens is > 50 % seawater at all (column in brackets); the "
          "remainder (ρ_lens too high for the drawn F and ρ_sw) have no saline zone, which is why the saline-area P5 is 0 "
          "(bimodal distribution). Isohaline quantised to the 1 m grid. Outside x = 100–450 m the base is held constant "
          "(areas only; not plotted)."]
    fig, ax = plt.subplots(figsize=(14, 5))
    lo, md, hi = (np.where((XG >= TX[0]) & (XG <= TX[-1]), np.nanpercentile(iso, p, axis=0), np.nan) for p in (5, 50, 95))
    ax.fill_between(XG, lo, hi, color="navy", alpha=0.2, lw=0); ax.plot(XG, md, color="navy", lw=1.6, label="50 % isohaline: median, 90 %")
    lo2, md2, hi2 = (np.percentile(base_f, p, axis=0) for p in (5, 50, 95))
    ax.fill_between(TX, lo2, hi2, color="purple", alpha=0.15, lw=0); ax.plot(TX, md2, "--", color="purple", lw=1.4, label="BEL parametric base")
    ax.plot(ex, ez, "k-", lw=0.8, label="land surface"); dune_flag(ax)
    ax.set_xlim(0, 505); ax.set_ylim(-38, 8); ax.set_xlabel("x (m)"); ax.set_ylabel("mTAW")
    ax.set_title("Salinity-defined lens base (50 % seawater) vs parametric base — field", fontweight="bold", fontsize=10)
    wc.white_legend(ax, loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(S16.FIG, "3b_isohaline.png"), dpi=150); plt.close(fig)
    # (c) areas
    A = st["areas"]
    L += ["", "### (c) Cross-section areas (m² per metre of coast, 2-D), water table / surface down to the deep-unit top", "",
          "| class | dunes x 0–90 m (weakly constrained) | beach x 90–505 m |", "|---|---|---|"]
    for c in ("saline", "brackish", "fresh"):
        L.append(f"| {c} | {q(A[f'{c}_dunes'], 0)} | {q(A[f'{c}_beach'], 0)} |")
    ca = {k: np.mean([c[k] for c in cov_area]) for k in cov_area[0]}
    L += ["", f"Held-out validation ({nh} site samples; truth with its own Archie draw): coverage of the 90 % area intervals — "
          + "; ".join(f"{k} {100*v:.0f} %" for k, v in ca.items()) + "."]
    R = np.vstack(rel); bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(R[:, 0], bins) - 1, 0, 9)
    L.append("Held-out reliability of P(% seawater > 50) (cells): " + ", ".join(
        f"{bins[i]:.1f}–{bins[i+1]:.1f}: {R[idx == i, 1].mean():.2f} (n={int((idx == i).sum())})" for i in range(10) if (idx == i).any()))
    # (d) sensitivity
    L += ["", "### (d) Sensitivity", "", "| case | isohaline x=150 / 250 / 350 (median) | saline beach | brackish beach | fresh beach | fresh dunes |",
          "|---|---|---|---|---|---|"]
    def srow(name, stt):
        iso_ = stt["iso"]; a = stt["areas"]
        L.append(f"| {name} | " + " / ".join(f"{np.nanmedian(iso_[:, int(np.argmin(np.abs(XG - x)))]):.2f}" for x in (150, 250, 350))
                 + f" | {np.median(a['saline_beach']):.0f} | {np.median(a['brackish_beach']):.0f} | {np.median(a['fresh_beach']):.0f} | "
                 f"{np.median(a['fresh_dunes']):.0f} |")
    srow("reference (F ~ U(2.8, 3.6))", st)
    for Fv in (2.8, 3.2, 3.6):
        srow(f"F = {Fv}", sections_stats(base_f, par_f, np.random.default_rng([S16.ARCHIE_SEED, int(Fv * 10)]), F=Fv))
    rb = np.array([t["rho_below"] for t in C["tr"]["th"]])
    for k, (lo_, hi_) in enumerate(((2, 4), (4, 8), (8, 20))):
        m = (rb >= lo_) & (rb < hi_)
        Jb = JointBEL(C["tr"]["D"][m], C["tr"]["T"][m], [t for t, mm in zip(C["tr"]["th"], m) if mm])
        bb, pb, _ = Jb.posterior(obs["field"], N_POST, np.random.default_rng([S16.JOINT_SEED, 10 + k]))
        srow(f"ρ_below {lo_}–{hi_} Ω·m (n = {m.sum()})", sections_stats(bb, pb, np.random.default_rng([S16.ARCHIE_SEED, 10 + k])))
    L += ["", "Dune zone (x < 90 m): weakly constrained — mesh approximation error is largest there (short dipoles under the "
          "dune cap), and the dune conductive layer (clay or saltwater) is not built into the salinity section."]
    np.savez(os.path.join(S16.D3, "joint_field.npz"), base=base_f, iso=iso, Psal=st["Psal"], Pfresh=st["Pfresh"],
             **{f"par_{k}": np.asarray(v) for k, v in par_f.items()}, **{f"area_{k}": v for k, v in A.items()})
    with open(os.path.join(S16.RES, "2_3_joint_bel_salinity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
