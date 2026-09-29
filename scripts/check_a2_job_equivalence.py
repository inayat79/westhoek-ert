"""
check_a2_job_equivalence.py — v1.1 check that 11_stage4_closeout.a2_job with default arguments behaves exactly as the
v1.0 function (tag v1.0.3-numbers).

    conda run -n pg python check_a2_job_equivalence.py

The v1.0 source is taken from git (`git show v1.0.3-numbers:scripts/11_stage4_closeout.py`). Both versions run two
Stage-4 jobs (T1 r0 with the λ pool, T3 r7 without) with the inversion replaced by the same deterministic stub; every
call to the inversion (hash of the noisy data, n, λ, posterior seed) and every saved array must be bit-identical.
The stub isolates the code change from pyGIMLi's run-to-run floating-point nondeterminism (a real re-run of the v1.0
job T1 r00 differs from the stored run by ≤ 2.6e-5 m in the λ = 50 lens bases; v1.0 already reported 8.9e-4 m for the
same kind of re-run, results/synthetic_s5/a2_interface_definitions.md).
"""

import contextlib
import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()):
        sp.loader.exec_module(m)
    return m


def main():
    tmp = tempfile.mkdtemp(prefix="a2eq_")
    src = subprocess.run(["git", "show", "v1.0.3-numbers:scripts/11_stage4_closeout.py"], cwd=SCRIPT_DIR,
                         capture_output=True, check=True).stdout.decode("utf-8-sig")
    src = src.replace("SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))", f"SCRIPT_DIR = {SCRIPT_DIR!r}")
    old_path = os.path.join(tmp, "c11_v103.py")
    open(old_path, "w", encoding="utf-8").write(src)
    new = load("c11_new", os.path.join(SCRIPT_DIR, "11_stage4_closeout.py")); old = load("c11_old", old_path)
    wc = new.wc
    ref = wc.load_run("g1_lam50"); mesh, _ = wc.rebuild_paradomain(ref); cs = wc.ColumnSampler(mesh, new.sb.COLS)
    spec = new.sb.load_spec(); fwd = np.load(os.path.join(new.sb.SYN_DATA, "truth_forward.npz")); sch = new.sb.g1_scheme()

    def make_stub(calls):
        def stub(m02, land, elec, lam, mesh_check, cs_, seed, max_iter=None):
            calls.append((hashlib.sha256(np.ascontiguousarray(land["R"]).tobytes()).hexdigest(), int(land["n"]),
                          float(lam), int(seed)))
            rng = np.random.default_rng([int(seed) % (2 ** 32), int(lam)])
            model = ref["model"] * np.exp(0.05 * rng.standard_normal(len(ref["model"])))
            S_ = np.log(model)[:, None] + 0.1 * rng.standard_normal((len(model), 200))
            return dict(chi2=1.0 + lam / 1000, stop="stub", n_iter=3, model=model.astype(np.float32),
                        radius=ref["res_radius"]), S_
        return stub

    res = {}
    for label, mod in (("old", old), ("new", new)):
        calls, outs = [], []
        mod.sb.one_inversion = make_stub(calls)
        for truth, real in (("T1", 0), ("T3", 7)):
            out = os.path.join(tmp, f"{label}_{truth}_r{real}.npz")
            if label == "old":
                mod.OUT_DATA = tmp
                mod.a2_job(None, spec, fwd, sch, np.asarray(ref["cell_x"]), cs, truth, real, lambda m: None)
                os.replace(os.path.join(tmp, "a2_runs", f"{truth}_r{real:02d}.npz"), out)
            else:
                mod.a2_job(None, spec, fwd, sch, np.asarray(ref["cell_x"]), cs, truth, real, lambda m: None, out=out)
            outs.append(out)
        res[label] = (calls, outs)
    ok_calls = res["old"][0] == res["new"][0]
    ok_arrays = True
    for fo, fn in zip(res["old"][1], res["new"][1]):
        a, b = np.load(fo, allow_pickle=True), np.load(fn, allow_pickle=True)
        same = sorted(a.files) == sorted(b.files) and all(
            np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a.files)
        ok_arrays &= same
        print(f"{os.path.basename(fn)}: {len(b.files)} arrays bit-identical to v1.0: {same}")
    print(f"inversion calls identical: {ok_calls} ({len(res['new'][0])} calls)")
    print("EQUIVALENCE", "PASSED" if ok_calls and ok_arrays else "FAILED")
    sys.exit(0 if ok_calls and ok_arrays else 1)


if __name__ == "__main__":
    main()
