"""
bug_invlam_repro.py
===================
Minimal reproducer (pyGIMLi 1.5.5): after ERTManager.invert(lam=5), the Python
property ``mgr.inv.lam`` still reports the default (20), while the C++ inversion
used λ = 5 (``mgr.inv.inv.getLambda()``).

Cause (pygimli/frameworks/inversion.py, Inversion.run): the keyword ``lam`` is
popped into a local variable and passed to ``self.inv.setLambda(lam)`` directly,
bypassing the ``lam`` property setter, so ``self._lam`` is never updated.

Consequence: anything that reads ``inv.lam`` after a run — e.g.
``pygimli.frameworks.resolution.resolutionMatrix(inv)`` — silently uses λ = 20.
Workaround: read ``inv.inv.getLambda()`` (or set ``inv.lam = λ`` before calling run).

Run:  python bug_invlam_repro.py      (≈ 10 s; synthetic, no project data needed)
"""

import numpy as np
import pygimli as pg
from pygimli.physics import ert


def main():
    print("pyGIMLi", pg.__version__)
    scheme = ert.createData(elecs=np.linspace(0, 46, 24), schemeName="dd")
    mesh = pg.meshtools.createParaMesh(scheme, paraDepth=12, quality=33)
    data = ert.simulate(mesh, scheme=scheme, res=100.0, noiseLevel=3, noiseAbs=1e-6, seed=1)
    mgr = ert.ERTManager(data, verbose=False)
    mgr.invert(lam=5, maxIter=3, verbose=False)

    lam_py = float(mgr.inv.lam)
    lam_cpp = float(mgr.inv.inv.getLambda())
    print(f"requested lam = 5")
    print(f"mgr.inv.lam (Python property) = {lam_py:g}")
    print(f"mgr.inv.inv.getLambda() (C++) = {lam_cpp:g}")

    from pygimli.frameworks.resolution import resolutionMatrix
    RM_prop = resolutionMatrix(mgr.inv)                   # uses inv.lam (20)
    mgr.inv.lam = lam_cpp                                 # workaround: sync the property
    RM_true = resolutionMatrix(mgr.inv)                   # uses the λ actually inverted with
    print(f"trace(R_m) with inv.lam as left by run(): {np.trace(RM_prop):.2f}")
    print(f"trace(R_m) with inv.lam synced to C++ λ:  {np.trace(RM_true):.2f}")

    if lam_py != lam_cpp:
        print("BUG REPRODUCED: the Python inv.lam property does not reflect the lam passed to invert()")
    else:
        print("not reproduced (property and C++ lambda agree) — fixed in this pyGIMLi version?")


if __name__ == "__main__":
    main()
