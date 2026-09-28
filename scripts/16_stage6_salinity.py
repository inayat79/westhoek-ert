"""
16_stage6_salinity.py
=====================
Westhoek ERT — Stage 5 wrap-up (prior v3) and Stage 6 (salinity sections).

Subcommands
  spec3       write results/stage6/prior_spec_v3.md (ρ_discharge ceiling raised to 150 Ω·m)
  sample      parallel, resumable forward modelling of the extra prior-v3 samples on the medium mesh
  refit       step 1: combined prior (v2 + v3), "site" subset, lens-base BEL + bounded second BEL → ρ_discharge
  joint       steps 2–3: joint BEL (lens base + unit / geometry parameters), salinity sections, areas, isohaline,
              sensitivity, held-out validation

Prior v3 = prior v2 with ρ_discharge log-uniform on 40–150 Ω·m for the extra samples, so the combined set is
log-uniform on 2.5–150 Ω·m. Extra samples: ids 20000–21572 (1430 training 20000–21429, 143 held-out 21430–21572);
parameters default_rng([88001, id]); noise / approximation-error seeds as in prior v2 (default_rng([99000, id]),
default_rng([111000, id])) — the id space does not overlap prior v2 (0–3299). Joint-BEL posterior
default_rng([129000, k]); Archie draws default_rng([130000, k]).
Outputs: data/processed/bel_prior_v3/, results/stage6/, figures/stage6/ (reference unchanged).
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
import westhoek_common as wc  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


s15 = _load("s15", "15_stage5_closeout.py")
g14, b13 = s15.g14, s15.b13

D3 = os.path.join(wc.PROCESSED_DIR, "bel_prior_v3")
RES = os.path.join(wc.RESULTS_DIR, "stage6")
FIG = os.path.join(wc.FIG_DIR, "stage6")
PRIOR3_SEED, JOINT_SEED, ARCHIE_SEED = 88001, 129000, 130000
ID0, N_NEW_TRAIN, N_NEW_HELD = 20000, 1430, 143
RHO_DIS_V3 = (40.0, 150.0)
TX = b13.TX
XREP = (150, 200, 250, 300, 350, 400, 450)
N_POST = 2000


def ensure():
    for d in (D3, RES, FIG, os.path.join(D3, "train"), os.path.join(D3, "logs")):
        os.makedirs(d, exist_ok=True)


def spec_v3():
    spec = json.load(open(b13.SPEC2, encoding="utf-8"))
    spec["params"]["rho_discharge"] = dict(range=list(RHO_DIS_V3), dist="log-uniform", unit="Ω·m",
                                           why="prior v3 extra samples (ceiling raised)")
    return spec


def draw3(i):
    """Prior-v3 extra sample (prior-v2 draw with ρ_discharge on 40–150 Ω·m, own seed)."""
    old = b13.PRIOR_SEED
    b13.PRIOR_SEED = PRIOR3_SEED
    try:
        return b13.draw(spec_v3(), i)
    finally:
        b13.PRIOR_SEED = old


# ─────────────────────────────────────────────────────────────────────────────
def cmd_spec3(args):
    ensure()
    n_old, frac = 3300, np.log(150 / 40) / np.log(150 / 2.5)
    L = ["# Prior v3 (Stage 5 wrap-up)", "",
         "Identical to prior v2 (`results/bel/prior_spec_v2.md`) except that the ρ_discharge ceiling is raised to 150 Ω·m. "
         "The 3300 prior-v2 samples (ρ_discharge log-uniform 2.5–40) are kept; extra samples draw ρ_discharge "
         "log-uniform on 40–150 Ω·m, all other parameters from prior v2.", "",
         f"- Required share of new samples: log(150/40)/log(150/2.5) = {frac:.4f} → total = 3300 / (1 − {frac:.4f}) = "
         f"{n_old/(1-frac):.0f} → **{N_NEW_TRAIN + N_NEW_HELD} new** ({N_NEW_TRAIN} training + {N_NEW_HELD} held-out, the "
         "same 10 : 1 ratio as prior v2). Combined: 4430 training + 443 held-out.",
         f"- Contrast rules as prior v2 (ρ_discharge ≥ 1.5 ρ_lens is automatically satisfied above 40 Ω·m).",
         f"- Seeds: parameters default_rng([{PRIOR3_SEED}, id]) with ids {ID0}–{ID0 + N_NEW_TRAIN + N_NEW_HELD - 1}; "
         "noise default_rng([99000, id]); approximation error default_rng([111000, id]) (ids disjoint from prior v2).",
         "- Forward: medium mesh, approximation-error correction (mean + covariance from the prior-v2 dual set), "
         "3 % + 1e-4 Ω noise — as prior v2.",
         "- Headline subset 'site': deep-unit top in [−38, −22] mTAW and the lens base above it (all lens columns)."]
    with open(os.path.join(RES, "prior_spec_v3.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_worker(args):
    import pygimli.meshtools as mt
    scheme = b13.pilot.build_scheme()
    mesh = mt.createParaMesh(scheme.sensors(), **b13.MESH["medium"])
    cx = np.array([c.center().x() for c in mesh.cells()]); cz = np.array([c.center().y() for c in mesh.cells()])
    logf = open(os.path.join(D3, "logs", f"worker_{args.worker_id}.log"), "a", encoding="utf-8")
    for i in args.ids:
        out = os.path.join(D3, "train", f"s{i:06d}.npz")
        if os.path.exists(out):
            continue
        try:
            th = draw3(i); pm = b13.PriorModel2(th); t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                R = b13.pilot.simulate_R(mesh, 10 ** pm.log10rho(cx, cz), scheme.size())
            np.savez(out + ".tmp.npz", R_medium=R, theta=json.dumps(th), target=pm.target(),
                     dis_present=th["dis_present"], rho_discharge=th["rho_discharge"], clay_present=th["clay_present"],
                     retries=b13.pilot.RETRIES["n"])
            os.replace(out + ".tmp.npz", out)
            logf.write(f"{time.strftime('%H:%M:%S')} s{i} {time.time()-t0:.1f}s\n"); logf.flush()
        except Exception as e:
            import traceback
            logf.write(f"FAILED s{i}: {e}\n{traceback.format_exc()}\n"); logf.flush()
    logf.write("worker finished\n")


def cmd_sample(args):
    ensure()
    ids = [i for i in range(ID0, ID0 + N_NEW_TRAIN + N_NEW_HELD)
           if not os.path.exists(os.path.join(D3, "train", f"s{i:06d}.npz"))]
    parts = [ids[k::args.workers] for k in range(args.workers)]
    procs = []
    for w, part in enumerate(parts):
        if not part:
            continue
        wd = os.path.join(D3, "work", f"w{w}"); os.makedirs(wd, exist_ok=True)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", "--worker-id", str(w), "--ids"]
                                      + [str(i) for i in part], cwd=wd, env=dict(os.environ, OMP_NUM_THREADS="2"),
                                      stdout=subprocess.DEVNULL, stderr=open(os.path.join(D3, "logs", f"worker_{w}.err"), "a")))
    for p in procs:
        p.wait()
    print(f"[sample] {len([f for f in os.listdir(os.path.join(D3, 'train')) if f.endswith('.npz') and 'tmp' not in f])} v3 samples")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("spec3", "refit", "joint"):
        sub.add_parser(c)
    ps = sub.add_parser("sample"); ps.add_argument("--workers", type=int, default=4)
    pw = sub.add_parser("worker"); pw.add_argument("--worker-id", type=int, required=True)
    pw.add_argument("--ids", type=int, nargs="+", required=True)
    a = ap.parse_args()
    fns = dict(spec3=cmd_spec3, sample=cmd_sample, worker=cmd_worker)
    if a.cmd in ("refit", "joint"):
        import stage6_analysis as s6
        fns.update(refit=s6.cmd_refit, joint=s6.cmd_joint)
    fns[a.cmd](a)


if __name__ == "__main__":
    main()
