"""
20_petrophysics_sensitivity.py
==============================
Westhoek ERT v1.1 — Task 2: sensitivity of the unit salinities to the petrophysical relation (field posterior, no new
inversion).

  run [--out ROOT]      cases P0–P5 → ROOT/data/processed/v1_1/petrophysics.json (+ .npz)
  report [--out ROOT]   ROOT/results/v1_1/2_petrophysics_sensitivity.md, T_S2_petrophysics.csv / .md,
                        figs/F_S2_petrophysics.png

Field posterior: the v1.0 joint BEL (site + physical, prior v3), regenerated with the v1.0 functions and seed
(default_rng([129000, 0])) — the same samples behind data/processed/stage6_v2/field_v2.npz (checked). Archie cases P0–P4
use the v1.0 rejection sampler (17_salinity_v2.constrained, draws > 100 % seawater in any unit rejected) with the v1.0
seed default_rng([131000, 0]) and draw order F, S, T; only the ranges change. P0 must reproduce v1.0 numbers 101–104
(unit_lens, unit_below, unit_dis, unit_dune) exactly — the run stops otherwise.

P5 (Lebbe 1999, parallel conduction, as calibrated): Ps = 100·(F/ρ_t − F/ρ_mat − 1/ρ_f)/(1/ρ_s − 1/ρ_f), F = 3.98,
ρ_s = 0.434, ρ_f = 20.1, ρ_mat = 320 Ω·m; no temperature correction of Ps and no rejection. EC25 for P5: linear mixing of
the end-member EC25 values, EC25 = x·EC25_s + (1 − x)·EC25_f with x = Ps/100 and EC25_e = 10⁴/ρ_e/(1 + 0.02(T − 25)),
T ~ U(8, 14) °C drawn with default_rng([135000, 5]) (reduces to the v1.0 seawater-only mixing when EC25_f → 0).
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
import read_trace  # noqa: E402
read_trace.install("20_petrophysics_sensitivity.py")
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


s17 = _load("s17", "17_salinity_v2.py")
s6 = s17.s6
UNITS = s17.UNITS
NAMES = {"lens": "saline lens", "below": "water below the lens", "dis": "discharge-zone water", "dune": "dune aquifer"}
CASES = {  # F range (or fixed), S range (PSU), T range (°C)
    "P0": dict(label="Archie, v1.0 reference", F=(2.8, 3.6), S=(31.0, 34.0), T=(8.0, 14.0)),
    "P1": dict(label="Archie, F = 2.7 (Lebbe 1981)", F=2.7, S=(31.0, 34.0), T=(8.0, 14.0)),
    "P2": dict(label="Archie, F ~ U(2.7, 3.98)", F=(2.7, 3.98), S=(31.0, 34.0), T=(8.0, 14.0)),
    "P3": dict(label="Archie, S ~ U(28, 29) (Lebbe 1981, 1999)", F=(2.8, 3.6), S=(28.0, 29.0), T=(8.0, 14.0)),
    "P4": dict(label="Archie, F ~ U(2.7, 3.98), S ~ U(28, 34)", F=(2.7, 3.98), S=(28.0, 34.0), T=(8.0, 14.0)),
    "P5": dict(label="parallel conduction (Lebbe 1999), as calibrated"),
}
P5 = dict(F=3.98, rs=0.434, rf=20.1, rmat=320.0)
XISO = (150, 250, 350)


def archie_case(c):
    """Archie draws with the ranges of case c; same stream and order (F, S, T) as 17_salinity_v2.archie_v2."""
    def draws(n, rng, F=None):
        Fr = c["F"] if isinstance(c["F"], tuple) else (2.8, 3.6)
        Fd = rng.uniform(*Fr, n); S = rng.uniform(*c["S"], n); T = rng.uniform(*c["T"], n)
        if not isinstance(c["F"], tuple):
            Fd = np.full(n, float(c["F"]))
        if F is not None:
            Fd = np.full(n, float(F))
        ec = s17.ec25_sw(S)
        return dict(F=Fd, S=S, T=T, ec25_sw=ec, rsw=1.0 / (ec * (1 + s17.ALPHA * (T - 25))))
    return draws


def field_posterior():
    s6.init(); s6.TX = s6.S16.TX
    C = s6.load_combined(physical=True)
    obs, _ = s6.b13.observed_sets()
    J = s6.JointBEL(C["tr"]["D"], C["tr"]["T"], C["tr"]["th"])
    return J.posterior(obs["field"], s17.N_POST, np.random.default_rng([s17.JOINT_SEED, 0]))


def iso_at(iso):
    return [float(np.nanmedian(iso[:, int(np.argmin(np.abs(s6.XG - x)))])) for x in XISO]


def summarise(usw, uec, extra):
    out = {}
    for u in UNITS:
        v = np.asarray(usw[u], float); e = np.asarray(uec[u], float); ok = np.isfinite(v)
        out[u] = dict(sw=np.percentile(v[ok], [5, 50, 95]).tolist(), p50=float(np.mean(v[ok] > 50)),
                      pfresh=float(np.mean(e[ok] < 1500)), n=int(ok.sum()))
    out.update(extra)
    return out


def cmd_run(args):
    dpp = os.path.join(args.out, "data", "processed", "v1_1"); os.makedirs(dpp, exist_ok=True)
    base_f, par_f, _ = field_posterior()
    res = {}
    arch0 = s17.archie_v2
    for key, c in CASES.items():
        if key == "P5":
            continue
        s17.archie_v2 = archie_case(c)                                   # only the Archie ranges change
        try:
            bc, pc, A, info = s17.constrained(base_f, par_f, np.random.default_rng([s17.ARCHIE2_SEED, 0]))
            st = s17.stats(bc, pc, A)
        finally:
            s17.archie_v2 = arch0
        res[key] = summarise(st["usw"], st["uec"], dict(rej=info["joint_rate"], unit_rej=info["unit_rate"],
                                                        iso=iso_at(st["iso"])))
        if key == "P0":
            # sanity check 1: the accepted draws are those behind field_v2.npz
            z = np.load(os.path.join(wc.PROCESSED_DIR, "stage6_v2", "field_v2.npz"))
            same = bool(np.array_equal(z["base"], bc) and np.array_equal(z["draw_F"], A["F"]))
            # sanity check 2: numbers 101–104 reproduced exactly (formatted as in 18_paper.py)
            stored = {r["key"]: r["value"] for r in json.load(open(os.path.join(wc.RESULTS_DIR, "paper", "numbers.json"),
                                                                   encoding="utf-8"))}
            chk = {}
            for u in UNITS:
                a, v, ec = st["uarea"][u], st["usw"][u], st["uec"][u]; ok = np.isfinite(v)
                val = (f"{s6.q(a, 0)}; {s6.q(v, 1)}; {s6.q(ec, 0)}; {np.mean(v[ok] > 50):.2f}; "
                       f"{np.mean(ec[ok] < 1500):.2f}")
                chk[f"unit_{u}"] = val == stored[f"unit_{u}"]
            res["P0"]["check"] = dict(field_v2_identical=same, numbers_101_104=chk)
            if not same or not all(chk.values()):
                json.dump(res, open(os.path.join(dpp, "petrophysics.json"), "w", encoding="utf-8"), indent=1)
                raise SystemExit(f"P0 does not reproduce v1.0: field_v2 identical {same}, numbers {chk} — stopping")
            print(f"[P0] reproduces v1.0: field_v2 identical {same}; numbers 101–104 {chk}", flush=True)
        print(f"[{key}] done (rejection {100*res[key]['rej']:.1f} %)", flush=True)
    # P5 — parallel conduction, no rejection
    T = np.random.default_rng([135000, 5]).uniform(8.0, 14.0, len(base_f))
    usw, uec, outside, usw_c = {}, {}, {}, {}
    pres = s17.presence(base_f, par_f)
    for u in UNITS:
        rt = np.asarray(par_f[s17.UNIT_RHO[u]], float)
        ps = 100 * (P5["F"] / rt - P5["F"] / P5["rmat"] - 1 / P5["rf"]) / (1 / P5["rs"] - 1 / P5["rf"])
        ecs = 1e4 / P5["rs"] / (1 + s17.ALPHA * (T - 25)); ecf = 1e4 / P5["rf"] / (1 + s17.ALPHA * (T - 25))
        x = ps / 100
        usw[u] = np.where(pres[u], ps, np.nan); uec[u] = np.where(pres[u], x * ecs + (1 - x) * ecf, np.nan)
        usw_c[u] = np.clip(usw[u], 0, 100)
        v = usw[u][np.isfinite(usw[u])]
        outside[u] = dict(below0=float(np.mean(v < 0)), above100=float(np.mean(v > 100)))
    # P5 isohaline: sections from the unit values
    iso = []
    for i in range(0, len(base_f), 250):
        sl = slice(i, i + 250)
        M = s17.unit_masks(base_f[sl], {k: np.asarray(v)[sl] for k, v in par_f.items()})
        sw = np.full(M["valid"].shape, np.nan)
        for u in UNITS:
            sw = np.where(M[u], usw[u][sl][:, None, None], sw)
        iso.append(s6.isohaline(sw))
    res["P5"] = summarise(usw, uec, dict(outside=outside, iso=iso_at(np.vstack(iso))))
    res["P5_clipped"] = summarise(usw_c, uec, dict(iso=res["P5"]["iso"]))
    json.dump(res, open(os.path.join(dpp, "petrophysics.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: {u: v[u]["sw"][1] for u in UNITS} for k, v in res.items()}, indent=1))


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    src = os.path.join(args.out, "data", "processed", "v1_1", "petrophysics.json")
    res_dir = os.path.join(args.out, "results", "v1_1"); os.makedirs(os.path.join(res_dir, "figs"), exist_ok=True)
    R = json.load(open(src, encoding="utf-8"))
    f = lambda s, d=1: f"{s[1]:.{d}f} [{s[0]:.{d}f}, {s[2]:.{d}f}]"
    L = ["# 2 — Petrophysics sensitivity of the unit salinities (field posterior, no new inversion)", "",
         "Field joint-BEL posterior of v1.0 (site + physical, prior v3), 2000 samples, seeds as v1.0. P0–P4: Archie with the "
         "v1.0 per-draw rejection of > 100 % seawater (seed default_rng([131000, 0]); draw order F, S, T). P5: parallel "
         "conduction of Lebbe (1999) as calibrated (F = 3.98, ρ_s = 0.434, ρ_f = 20.1, ρ_mat = 320 Ω·m), no rejection. "
         "% seawater median [P5, P95].", "",
         f"**Sanity check (P0 = v1.0):** accepted draws identical to `field_v2.npz`: "
         f"{R['P0']['check']['field_v2_identical']}; numbers 101–104 reproduced: "
         + ", ".join(f"{k} {v}" for k, v in R["P0"]["check"]["numbers_101_104"].items()) + ".", "",
         ]
    tbl = ["| case | relation | lens | below | discharge | dune | P(> 50 %) lens | P(fresh) discharge | rejected / outside "
           "[0, 100] | 50 % isohaline x = 150 / 250 / 350 (mTAW) |", "|---|---|---|---|---|---|---|---|---|---|"]
    csv = ["case,relation,unit,sw_p5,sw_p50,sw_p95,p_gt50,p_fresh,rejected_or_outside"]
    for key in ("P0", "P1", "P2", "P3", "P4", "P5", "P5_clipped"):
        r = R[key]
        lab = CASES["P5"]["label"] + (", clipped to [0, 100]" if key == "P5_clipped" else ", unclipped") if key.startswith("P5") \
            else CASES[key]["label"]
        if key == "P5":
            rj = "; ".join(f"{u}: < 0 {100*r['outside'][u]['below0']:.1f} %, > 100 {100*r['outside'][u]['above100']:.1f} %"
                           for u in UNITS)
        elif key == "P5_clipped":
            rj = "(as P5)"
        else:
            rj = f"{100*r['rej']:.1f} % (lens {100*r['unit_rej']['lens']:.1f} %)"
        tbl.append(f"| {key} | {lab} | " + " | ".join(f(r[u]["sw"]) for u in UNITS)
                 + f" | {r['lens']['p50']:.2f} | {r['dis']['pfresh']:.2f} | {rj} | " + " / ".join(f"{v:.1f}" for v in r["iso"]) + " |")
        for u in UNITS:
            csv.append(f"{key},\"{lab}\",{u},{r[u]['sw'][0]:.3f},{r[u]['sw'][1]:.3f},{r[u]['sw'][2]:.3f},"
                       f"{r[u]['p50']:.4f},{r[u]['pfresh']:.4f},\"{rj}\"")
    L += tbl
    L += ["", "## Change of the unit medians relative to P0 (percentage points)", "",
          "| case | lens | below | discharge | dune |", "|---|---|---|---|---|"]
    for key in ("P1", "P2", "P3", "P4", "P5"):
        L.append(f"| {key} | " + " | ".join(f"{R[key][u]['sw'][1] - R['P0'][u]['sw'][1]:+.1f}" for u in UNITS) + " |")
    spread = {u: max(R[k][u]["sw"][1] for k in ("P1", "P2", "P3", "P4", "P5")) - min(R[k][u]["sw"][1] for k in ("P1", "P2", "P3", "P4", "P5"))
              for u in UNITS}
    width = {u: R["P0"][u]["sw"][2] - R["P0"][u]["sw"][0] for u in UNITS}
    e5 = [u for u in UNITS if spread[u] > width[u]]
    L += ["", "## Pre-registered statement E5", "",
          "Spread of the unit medians across P1–P5 (P5 unclipped) vs the width of the P0 90 % interval (percentage points): "
          + "; ".join(f"{NAMES[u]} {spread[u]:.1f} vs {width[u]:.1f}" for u in UNITS) + ".",
          f"**E5 {'holds' if e5 else 'does not hold'}**" + (f" (units: {', '.join(NAMES[u] for u in e5)})." if e5 else ".")]
    with open(os.path.join(res_dir, "2_petrophysics_sensitivity.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(res_dir, "T_S2_petrophysics.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(csv) + "\n")
    Tm = ["# Table S2 — petrophysics sensitivity of the unit salinities (% seawater, median [P5, P95])", ""] + tbl
    with open(os.path.join(res_dir, "T_S2_petrophysics.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(Tm) + "\n")
    s18 = _load("p18", "18_paper.py"); s18.mpl()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8), sharey=True)
    keys = ["P0", "P1", "P2", "P3", "P4", "P5"]
    for ax, u in zip(axes, UNITS):
        for i, k in enumerate(keys):
            s = R[k][u]["sw"]
            ax.plot([s[0], s[2]], [i, i], "-", color="#1f4e9c" if k != "P0" else "k", lw=1.2)
            ax.plot(s[1], i, "o", color="#1f4e9c" if k != "P0" else "k", ms=3.5)
        ax.axvline(50, color="0.5", ls=":", lw=0.8)
        ax.set_title(NAMES[u], fontsize=7.5); ax.set_xlabel("% seawater", fontsize=7); ax.grid(True, alpha=0.3)
    axes[0].set_yticks(range(len(keys))); axes[0].set_yticklabels(keys); axes[0].invert_yaxis()
    fig.suptitle("Unit salinity by petrophysical case (median, 90 %; P5 unclipped)", fontsize=8.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, "figs", "F_S2_petrophysics.png"), dpi=300, bbox_inches="tight")
    json.dump(dict(E5=bool(e5), E5_units=e5, spread=spread, width=width),
              open(os.path.join(args.out, "data", "processed", "v1_1", "petrophysics_E5.json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "report"):
        sub.add_parser(c).add_argument("--out", default=wc.ROOT)
    a = ap.parse_args()
    dict(run=cmd_run, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
