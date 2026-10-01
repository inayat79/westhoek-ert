"""
check_v13_defs_equivalence.py
=============================
v1.3: 25_generalisation.py computes the interface depths of every posterior sample itself (it needs the per-sample
depths and more definitions). This check shows that, on the same samples, its thr3 / maxgrad / logmid bands are
identical to those of the v1.0 function 11_stage4_closeout.sample_defs, and that the oracle-anchored variant moves no
crossing: for a truth whose oracle is 3 Ω·m, maxgrad_o / logmid_o equal maxgrad / logmid exactly.
Samples: the λ = 50 field model g1_lam50 plus Gaussian perturbations (default_rng([136900])); no inversion.
"""
import contextlib
import importlib.util
import io
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


g = _load("g25", "25_generalisation.py")
c11, sb = g.c11, g.sb
ref = wc.load_run(sb.REF_TAG); mesh, _ = wc.rebuild_paradomain(ref); cs = wc.ColumnSampler(mesh, sb.COLS)
rng = np.random.default_rng([136900])
S = np.log(np.asarray(ref["model"], float))[:, None] + 0.3 * rng.standard_normal((len(ref["model"]), 300))
old = c11.sample_defs(cs, S)
p = g.params("G_1.42_4_-12_0")
dep = g.sample_depths(cs, S, p)
ok = True
for d in ("thr3", "maxgrad", "logmid"):
    new = c11.band_stats(dep[d])
    same = np.array_equal(np.nan_to_num(new, nan=-999.0), np.nan_to_num(old[d], nan=-999.0))
    print(f"{d}: bands bit-identical to 11_stage4_closeout.sample_defs: {same}")
    ok &= same
q3 = dict(p, rho_lens=1.5, rho_below=6.0, k=4)                     # oracle √(1.5·6) = 3 Ω·m
d3 = g.sample_depths(cs, S, q3)
for a, b in (("maxgrad_o", "maxgrad"), ("logmid_o", "logmid"), ("oracle", "thr3")):
    e = np.array_equal(np.nan_to_num(d3[a], nan=-999.0), np.nan_to_num(d3[b], nan=-999.0))
    print(f"oracle 3 Ω·m: {a} == {b}: {e}")
    ok &= e
print("RESULT", "PASS" if ok else "FAIL")
