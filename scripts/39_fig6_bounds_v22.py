#!/usr/bin/env python3
"""v2.2 (plotting only): prior-v4 bounds of ρ_below for Figure 6(a), and the posterior samples of the six panel-(a) series
outside them. Writes Coral Draw/v2.2/_checks_bounds.json.

    python 39_fig6_bounds_v22.py
"""
import json
import os
import sys

os.environ.pop("WESTHOEK_READ_TRACE", None)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import numpy as np  # noqa: E402

ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
OUT = os.path.join(ROOT, "Coral Draw", "v2.2")


def prior_v4_bounds():
    """ρ_below range of the prior-v4 configuration that generated the ensemble (30_bel_v4.spec_v4, used by draw4)."""
    import paper_figures as pf
    g30 = pf.mod("g30")
    r = g30.spec_v4()["params"]["rho_below"]
    lo, hi = (float(v) for v in r["range"])
    if (lo, hi) != (2.0, 100.0) or r["dist"] != "log-uniform":
        raise SystemExit(f"prior-v4 ρ_below bounds are {lo}–{hi} ({r['dist']}), expected 2–100 log-uniform: stopping")
    return lo, hi


def panel_a_series():
    """The six ρ_below sample sets of Figure 6(a), loaded exactly as paper_figures.fig11 does (Ω·m)."""
    import paper_figures as pf
    g30, g28, D4, D4L, DLT, Dg = pf._v15_inputs()
    Z = np.load(os.path.join(DLT, "layout_vs_time.npz"), allow_pickle=True); S = np.load(os.path.join(D4, "synth.npz"))
    F = np.load(os.path.join(D4, "bel_v4.npz"), allow_pickle=True); L = np.load(os.path.join(D4L, "bel_v4_l24.npz"), allow_pickle=True)
    Z3 = np.load(os.path.join(DLT, "t1_subset.npz"))
    pooled = lambda Zs, pre: np.concatenate([10 ** Zs[k][:, 3] for k in Zs.files if k.startswith(pre) and k.endswith("_P2")])
    return {"Synthetic from soundings, 2018 layout": pooled(S, "T1_r"), "Synthetic from soundings, 2025 layout": pooled(Z, "T1L24_r"),
            "Synthetic from soundings, 2018 beach stretch": pooled(Z3, "T1sub_r"), "2018 data, full line": 10 ** F["P2"][:, 3],
            "2018 data, beach stretch": 10 ** Z["sub_P2"][:, 3], "2025 ERT line": 10 ** L["P2"][:, 3]}, F


def main():
    lo, hi = prior_v4_bounds()
    ser, F = panel_a_series()
    # cross-check against the stored training draws (second-BEL targets: column 3 = log10 ρ_below)
    yt = 10 ** np.asarray(F["Y2_train"])[:, 3]
    out = dict(bounds=[lo, hi], source="30_bel_v4.spec_v4()['params']['rho_below'] (prior-v4 configuration used by draw4)",
               training_draws=dict(n=int(len(yt)), min=float(yt.min()), max=float(yt.max())), series={})
    for k, v in ser.items():
        v = np.asarray(v, float); n = len(v)
        out["series"][k] = dict(n=n, below=int(np.sum(v < lo)), below_pct=float(100 * np.mean(v < lo)), above=int(np.sum(v > hi)),
                                above_pct=float(100 * np.mean(v > hi)), min=float(v.min()), max=float(v.max()))
    old = np.logspace(np.log10(1.5), np.log10(120), 40)
    out["bins_v21"] = dict(edges=[float(e) for e in old], n_bins=len(old) - 1, per_decade=float((len(old) - 1) / np.log10(120 / 1.5)),
                           edge_nearest_2=float(old[np.argmin(np.abs(old - 2))]), edge_nearest_100=float(old[np.argmin(np.abs(old - 100))]))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "_checks_bounds.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "bins_v21"}, indent=1, ensure_ascii=False))
    print("bins v2.1:", out["bins_v21"]["n_bins"], "bins,", round(out["bins_v21"]["per_decade"], 3), "per decade; nearest edges",
          out["bins_v21"]["edge_nearest_2"], out["bins_v21"]["edge_nearest_100"])


if __name__ == "__main__":
    main()
