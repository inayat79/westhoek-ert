"""
02_land_inversion.py
====================
Westhoek ERT Dataset — Land Profile Inversion with pyGIMLi
+ Resolution Matrix Analysis

What this script does:
  1. Loads the land ERT data (CRTOMO format) into a pyGIMLi DataContainerERT
  2. Creates a 2D FEM mesh appropriate for the Westhoek land profile
  3. Runs a standard Gauss-Newton ERT inversion (replicating the GRL 2020 method)
  4. Extracts the Jacobian (sensitivity) matrix after convergence
  5. Computes:
       - Cumulative sensitivity (data coverage) per cell
       - Model resolution matrix diagonal (Rm diagonal)
       - Linearised posterior model standard deviation
  6. Saves the Jacobian and resolution arrays for plotting in script 03

Requires: pygimli, numpy, scipy, matplotlib
Install:  conda install -c conda-forge pygimli
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.physics.ert as ert
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
# Relative to this scripts/ folder
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "raw"))
LAND_FILE     = os.path.join(DATA_DIR, "land_data.dat")
ELEC_FILE     = os.path.join(DATA_DIR, "elec.dat")
FIG_DIR       = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "figures"))
PROCESSED_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "processed"))
RESULTS_NPZ   = os.path.join(PROCESSED_DIR, "westhoek_inversion_results.npz")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Inversion parameters (matching GRL 2020 paper)
LAM          = 20.0     # regularisation weight (starting value)
ERR_REL      = 0.03     # 3% relative error (no reciprocals available)
ERR_ABS      = 1e-4     # Ohm — absolute noise floor
MAX_ITER     = 10
VERBOSE      = True

# Mesh parameters
DEPTH_FACTOR = 0.3      # depth of investigation ≈ 0.3 × max electrode separation
QUALITY      = 34       # triangle mesh quality angle


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Parse CRTOMO data and build pyGIMLi DataContainerERT
# ══════════════════════════════════════════════════════════════════════════════

def parse_crtomo(filepath):
    """Parse CRTOMO 6-column data file. Returns dict of arrays."""
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    n = int(lines[0].split()[0])
    rows = []
    for line in lines[1:n+1]:
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]),
                     float(parts[4]), float(parts[5]) if len(parts) > 5 else 0.0])
    a = np.array(rows)
    return {'n': len(a), 'C1': a[:,0].astype(int), 'C2': a[:,1].astype(int),
            'P1': a[:,2].astype(int), 'P2': a[:,3].astype(int),
            'R': a[:,4], 'err': a[:,5]}


def parse_electrodes(filepath):
    """Parse electrode file. Returns land and sea dicts."""
    land, sea = {}, {}
    with open(filepath) as f:
        for line in f:
            p = line.strip().split('\t')
            if not p[0].strip().isdigit():
                continue
            idx  = int(p[0])
            x    = float(p[1])
            z    = float(p[2])
            hsea = p[3].strip() if len(p) > 3 else '\\'
            if hsea in ('\\', ''):
                land[idx] = (x, z)
            else:
                sea[idx] = (x, z, float(hsea))
    return land, sea


def apply_drop_list(land_data, drop_path):
    """
    Remove measurements listed in a drop-list file (one 0-based measurement
    index per line, in land_data.dat order; '#' starts a comment).
    Returns (reduced land_data, array of kept original indices).
    """
    keep = np.arange(land_data['n'])
    if drop_path:
        with open(drop_path) as f:
            drop = {int(l.split('#')[0]) for l in f if l.split('#')[0].strip()}
        bad = [i for i in drop if not 0 <= i < land_data['n']]
        if bad:
            raise ValueError(f"Drop-list indices out of range: {bad[:10]}")
        keep = np.array([i for i in keep if i not in drop])
        print(f"   Drop-list {drop_path!r}: removing {len(drop)} of {land_data['n']} measurements")
    reduced = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in land_data.items()}
    reduced['n'] = len(keep)
    return reduced, keep


def build_datacont(land_data, land_elec, err_rel=ERR_REL):
    """
    Build a pyGIMLi DataContainerERT from parsed land data.

    Important:
    - pyGIMLi uses 0-based sensor indices internally
    - CRTOMO uses 1-based electrode indices
    - Sensors are added in order of electrode index
    """
    data = pg.DataContainerERT()

    # Add sensors (electrodes) in ascending index order
    sorted_idx = sorted(land_elec.keys())
    idx_to_sensor = {}   # maps electrode 1-based index → pyGIMLi 0-based sensor number
    for sensor_num, elec_idx in enumerate(sorted_idx):
        x, z = land_elec[elec_idx]
        data.createSensor(pg.Pos(x, z))
        idx_to_sensor[elec_idx] = sensor_num

    # Add measurements (a, b = current; m, n = potential; 0-based sensor numbers)
    data.resize(land_data['n'])
    for key, col in (('a', 'C1'), ('b', 'C2'), ('m', 'P1'), ('n', 'P2')):
        data[key] = [idx_to_sensor[i] for i in land_data[col]]

    # Set resistance values; geometric factors computed numerically because
    # the land profile has ~5 m of topography (analytical K assumes flat ground)
    data['r']    = land_data['R']
    data['k']    = ert.createGeometricFactors(data, numerical=True)
    data['rhoa'] = data['r'] * data['k']
    # Set error: relative% + absolute floor (in Ohm, expressed as fraction)
    data['err'] = np.sqrt((err_rel * land_data['R'])**2 + ERR_ABS**2) / land_data['R']

    # Mark valid data (all positive here)
    data['valid'] = (land_data['R'] > 0).astype(float)

    return data, idx_to_sensor


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Run inversion
# ══════════════════════════════════════════════════════════════════════════════

def run_inversion(data, lam=LAM, para_depth=None, para_dx=None, robust=False,
                  max_iter=MAX_ITER, ctype=None, ref_value=None, blocky=False):
    """
    Run ERT inversion using pyGIMLi ERTManager.

    The mesh is created explicitly (ERTManager.createMesh → createParaMeshPLC,
    pyGIMLi 1.5.5) so only mesh options reach the mesher. Options left as
    None use the pyGIMLi defaults (paraDepth = 0.4 × sensor span, paraDX = 1),
    which reproduces the original reference run.

    - Gauss-Newton inversion in log(resistivity) space
    - robust=True → robustData (IRLS L1 data reweighting)

    Returns: manager object with .model (resistivity) and .fop (forward operator)
    """
    mgr = ert.ERTManager(data, verbose=VERBOSE)

    mesh_kw = {}
    if para_depth is not None:
        mesh_kw['paraDepth'] = para_depth
    if para_dx is not None:
        mesh_kw['paraDX'] = para_dx
    mgr.createMesh(data, **mesh_kw)

    # Optional regularisation type and homogeneous reference model (for DOI):
    # cType=10 → mixed smallness (identity rows) + 1st-order smoothness rows;
    # isReference=True makes the starting model the reference m_ref in ‖C(m − m_ref)‖
    extra = {}
    if ctype is not None:
        extra['cType'] = ctype
    if ref_value is not None:
        extra['startModel'] = float(ref_value)
        extra['isReference'] = True
    if blocky:
        # IRLS L1 reweighting of the model roughness (pyGIMLi 1.5.5: Inversion.blockyModel
        # → RInversion.setBlockyModel; constrainBlocky() is applied every iteration)
        extra['blockyModel'] = True

    # lam: regularisation weight (higher → smoother model)
    # zWeight: vertical vs horizontal smoothing ratio
    mgr.invert(
        data=data,
        lam=lam,
        maxIter=max_iter,
        zWeight=0.5,         # slight vertical emphasis (typical for ERT)
        robustData=robust,
        verbose=VERBOSE,
        **extra
    )
    if ref_value is not None:
        mr = np.array(mgr.inv.inv.modelRef())
        print(f"   Reference model in inversion: {mr.min():.3f}–{mr.max():.3f} Ω·m (requested {ref_value:.3f})")
        if mr.size == 0 or not np.allclose(mr, ref_value, rtol=1e-6):
            raise RuntimeError("Reference model was not applied as requested")

    return mgr


def n_iterations(mgr):
    """Number of Gauss-Newton iterations performed (pyGIMLi 1.5.5)."""
    it = mgr.inv.iter
    return int(it() if callable(it) else it)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Resolution matrix analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_resolution_matrix(mgr, data, lam_expected=None):
    """
    Compute the linearised model resolution matrix after inversion.

    Theory:
    --------
    For the objective function:
        Φ(m) = ||W_d (d - F(m))||² + λ ||C (m - m_ref)||²

    The model resolution matrix (linear approximation around m*) is:
        R_m = (J^T W_d² J + λ C^T C)^{-1}  J^T W_d² J

    where:
        J   = Jacobian (sensitivity) matrix  [n_data × n_cells]
              in log-log space: J_ij = ∂log(d_i)/∂log(m_j)
        W_d = data weighting matrix = diag(1/err_i)
        C   = smoothness/constraint matrix from the mesh
        λ   = regularisation weight (from inversion)

    Interpretation of diag(R_m):
        → 1.0  : cell is perfectly resolved (data uniquely determines it)
        → 0.0  : cell is not resolved (dependent entirely on regularisation)
        → typical range 0.1–0.7 for ERT

    Cumulative sensitivity (data coverage):
        s_j = Σ_i |J_ij| / max(Σ_i |J_ij|)
        This is cheaper to compute and gives similar spatial information.
    """

    print("\n[Resolution Matrix] Extracting Jacobian...")

    # ── Get Jacobian (in log-log space) ──────────────────────────────────
    # pyGIMLi's fop.jacobian() returns ∂ρa/∂ρ (linear units) at the last
    # model iterate; convert to log-log space: ∂lnρa/∂lnρ = J · ρ_j / ρa_i
    J_lin = pg.utils.gmat2numpy(mgr.fop.jacobian())
    model_lin = np.array(mgr.inv.model)
    resp_lin  = np.array(mgr.inv.response)
    J = J_lin * model_lin[None, :] / resp_lin[:, None]   # (n_data, n_cells)
    n_data, n_cells = J.shape
    print(f"   Jacobian shape: {n_data} × {n_cells}")
    # Sanity check: scaling all ρ by c scales ρa by c, so each row of the
    # log-log Jacobian should sum to ≈1 (slightly less due to boundary cells)
    row_sum = J.sum(axis=1)
    print(f"   Row sums of log-log J: median={np.median(row_sum):.3f}, "
          f"range=[{row_sum.min():.3f}, {row_sum.max():.3f}]  (expect ≈1)")

    # ── Data weighting ────────────────────────────────────────────────────
    err = np.array(data['err'])
    # Remove zero or very small errors
    err = np.maximum(err, 1e-6)
    Wd_diag = 1.0 / err
    # Weighted Jacobian: Jw = diag(Wd) @ J
    Jw = Wd_diag[:, None] * J     # shape: (n_data, n_cells)

    # ── Regularisation (constraint) matrix ───────────────────────────────
    # Get the constraint (roughness) matrix from the mesh's region manager
    # In pyGIMLi, this is stored in the forward operator
    try:
        C = pg.utils.sparseMatrix2csr(mgr.fop.constraints())
        # Apply the constraint weights (incl. zWeight anisotropy) used by the inversion
        cw = np.array(mgr.fop.regionManager().constraintWeights())
        if len(cw) == C.shape[0]:
            C = sp.diags(cw) @ C
        else:
            print(f"   Warning: {len(cw)} constraint weights for {C.shape[0]} rows — unweighted C used.")
        use_constraint = True
        print(f"   Constraint matrix shape: {C.shape}")
    except Exception as e:
        print(f"   Warning: could not extract constraint matrix ({e})")
        print("   Falling back to identity regularisation.")
        C = np.eye(n_cells)
        use_constraint = False

    # Get the regularisation weight used in the inversion
    # Read λ from the C++ inversion: in pyGIMLi 1.5.5 Inversion.run() calls
    # inv.setLambda(lam) directly, so the Python property mgr.inv.lam keeps its
    # default (20) and does NOT reflect the λ passed to invert().
    lam = float(mgr.inv.inv.getLambda())
    if lam_expected is not None and not np.isclose(lam, lam_expected):
        raise RuntimeError(f"λ mismatch: inversion reports {lam}, requested {lam_expected}")
    print(f"   Regularisation weight λ = {lam:.4f}")

    # ── Build normal equations matrix A = J^T Wd² J + λ C^T C ───────────
    print(f"   Building normal equations matrix ({n_cells}×{n_cells})...")
    JtWdJ = Jw.T @ Jw       # n_cells × n_cells

    if use_constraint and sp.issparse(C):
        CtC = (C.T @ C).toarray()
    elif use_constraint:
        CtC = C.T @ C
    else:
        CtC = np.eye(n_cells)

    A = JtWdJ + lam * CtC   # n_cells × n_cells

    # ── Model resolution matrix ───────────────────────────────────────────
    # R_m = A^{-1} J^T Wd² J
    # Full matrix is n_cells × n_cells — can be large (>10,000 cells)
    # A is symmetric positive definite, so a single Cholesky factorisation
    # gives both A^{-1} (posterior covariance) and R_m = A^{-1} JtWdJ exactly.
    # Memory: 8·n_cells² bytes per dense matrix (~0.5 GB at 8,000 cells).
    import scipy.linalg as sla
    print(f"   Computing exact resolution matrix via Cholesky ({n_cells} cells)...")
    try:
        cho = sla.cho_factor(A, lower=True)
        A_inv = sla.cho_solve(cho, np.eye(n_cells))
    except np.linalg.LinAlgError as e:
        print(f"   Cholesky failed ({e}) — falling back to pseudo-inverse.")
        A_inv = np.linalg.pinv(A)
    Rm_diag = np.einsum('ij,ji->i', A_inv, JtWdJ)   # diag(A^{-1} JtWdJ)
    C_post_diag = np.diag(A_inv).copy()

    # ── Cumulative sensitivity (data coverage) ────────────────────────────
    # s_j = Σ_i |J_ij|^2 * Wd_i^2   (weighted cumulative sensitivity)
    cum_sens = np.sum((Jw)**2, axis=0)           # shape: (n_cells,)
    cum_sens_norm = cum_sens / (cum_sens.max() + 1e-30)

    # ── Log-space model standard deviation ───────────────────────────────
    # In log(m) space, sigma_log_m = sqrt(C_post_diag)
    # In linear space: approximately ±factor where factor = exp(sigma_log_m)
    sigma_log_m = np.sqrt(np.abs(C_post_diag))

    print(f"\n   Resolution diagonal:")
    print(f"     min = {Rm_diag.min():.4f}")
    print(f"     max = {Rm_diag.max():.4f}")
    print(f"     mean = {Rm_diag.mean():.4f}")
    print(f"   Cells with Rm > 0.5: {(Rm_diag > 0.5).sum()} ({100*(Rm_diag>0.5).mean():.1f}%)")
    print(f"   Cells with Rm > 0.3: {(Rm_diag > 0.3).sum()} ({100*(Rm_diag>0.3).mean():.1f}%)")

    return {
        'J'           : J,
        'C'           : sp.csr_matrix(C) if use_constraint else None,
        # kept in memory only (not saved) for --verify
        'A_inv'       : A_inv,
        'JtWdJ'       : JtWdJ,
        'Wd'          : Wd_diag,
        'err'         : err,
        'Rm_diag'     : Rm_diag,
        'cum_sens'    : cum_sens_norm,
        'sigma_log_m' : sigma_log_m,
        'lam'         : lam,
        'n_data'      : n_data,
        'n_cells'     : n_cells,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
# Same point-spread-function sites as 03_resolution_analysis.py (x, z in m)
PSF_SITES = {
    "Dune freshwater"   : (50,   -5),
    "Saltwater lens"    : (350,  -3),
    "SGD discharge zone": (430,  -4),
    "Offshore"          : (520,  -6),
}


def verify_resolution(mgr, res, cell_x, cell_z, out_path):
    """
    Check that the A = JᵀWd²J + λCᵀC used for R_m matches what pyGIMLi 1.5.5
    actually inverts with, and compare R_m columns at the PSF sites against
    pyGIMLi's built-in resolution tools:
      - C++  RInversion.modelCellResolution(i)   (LSCG column solve, own internals)
      - Py   pygimli.frameworks.resolution.resolutionMatrix(inv)   (dense; uses
             regionManager constraint weights and the Python inv.lam property)
    """
    from pygimli.frameworks.resolution import resolutionMatrix
    inv_cpp = mgr.inv.inv
    L = []
    def rel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))

    lam_cpp, lam_py = float(inv_cpp.getLambda()), float(mgr.inv.lam)
    L.append(f"lambda: C++ getLambda()={lam_cpp:g}   Python inv.lam property={lam_py:g}"
             + ("   <-- MISMATCH (pyGIMLi 1.5.5 property not updated by run())" if lam_cpp != lam_py else ""))

    # constraint weights: region manager (what we use) vs the C++ inversion's own
    cw_rm  = np.array(mgr.fop.regionManager().constraintWeights())
    cw_inv = np.array(inv_cpp.cWeight())
    mw     = np.array(inv_cpp.mWeight())
    L.append(f"constraint weights: n={len(cw_rm)} (regionManager) vs n={len(cw_inv)} (RInversion.cWeight); "
             f"max rel diff = {rel(cw_rm, cw_inv):.2e}")
    vals, cnt = np.unique(np.round(cw_inv, 6), return_counts=True)
    L.append("  distinct cWeight values (count): "
             + ", ".join(f"{v:g} ({c})" for v, c in zip(vals[:8], cnt[:8]))
             + (" ..." if len(vals) > 8 else ""))
    L.append(f"model weights mWeight: min={mw.min():g} max={mw.max():g} (all 1 → no effect)")

    # roughness: RInversion.roughness(m) = (C·m·m_w)·c_w in the inversion's model space
    m = np.array(inv_cpp.model())
    r_cpp = np.array(inv_cpp.roughness(inv_cpp.model()))
    Cw = res['C']
    L.append(f"roughness vs ours (Cw = diag(cw)·C): ‖Cw·ln m − roughness‖/‖·‖ = {rel(Cw @ np.log(m), r_cpp):.2e}   "
             f"(‖Cw·m − roughness‖ rel = {rel(Cw @ m, r_cpp):.2e})")

    # data weights
    dw = np.array(inv_cpp.dataWeight())
    L.append(f"data weights: RInversion.dataWeight vs ours 1/err: max rel diff = {rel(res['Wd'], dw):.2e}; "
             f"median ratio ours/theirs = {np.median(res['Wd'] / dw):.4f}")

    # resolution columns at the PSF sites
    print("   [verify] computing pyGIMLi built-in dense resolutionMatrix ...")
    RM_py = resolutionMatrix(mgr.inv)
    L.append("R_m columns at PSF sites: max|ours − theirs| / max|theirs|")
    for name, (tx, tz) in PSF_SITES.items():
        j = int(np.argmin((cell_x - tx)**2 + (cell_z - tz)**2))
        ours = res['A_inv'] @ res['JtWdJ'][:, j]
        cpp  = np.array(inv_cpp.modelCellResolution(j))
        L.append(f"  {name:<20} cell {j:5d} ({cell_x[j]:6.1f},{cell_z[j]:6.1f}): "
                 f"vs C++ modelCellResolution = {rel(ours, cpp):.2e}   "
                 f"vs Py resolutionMatrix = {rel(ours, RM_py[:, j]):.2e}   "
                 f"diag ours/C++/Py = {ours[j]:.4f}/{cpp[j]:.4f}/{RM_py[j, j]:.4f}")
    txt = "\n".join(L)
    print("   [verify]\n   " + txt.replace("\n", "\n   "))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"   [verify] saved {out_path}")


NEGRESP_FILES = ("negResp.bms", "negResp.vtk", "negResp-Atts.vector", "resp.vec", "respRez.vec")


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Westhoek land ERT inversion + resolution matrix")
    p.add_argument("--lam",        type=float, default=LAM, help=f"regularisation λ (default {LAM})")
    p.add_argument("--err-rel",    type=float, default=ERR_REL,
                   help=f"relative data error (default {ERR_REL})")
    p.add_argument("--para-depth", type=float, default=None,
                   help="paraDepth in m (default: pyGIMLi auto = 0.4 × sensor span)")
    p.add_argument("--para-dx",    type=float, default=None,
                   help="paraDX, relative node spacing between sensors (default: pyGIMLi 1 = none)")
    p.add_argument("--robust",     action="store_true", help="robust (IRLS L1) data weighting")
    p.add_argument("--drop-list",  default=None,
                   help="text file of 0-based measurement indices (land_data.dat order) to exclude")
    p.add_argument("--tag",        default=None,
                   help="run name; outputs go to data/processed/runs/<tag>/ and figures/runs/<tag>/")
    p.add_argument("--max-iter",   type=int, default=MAX_ITER,
                   help=f"maximum Gauss-Newton iterations (default {MAX_ITER})")
    p.add_argument("--err-scale-file", default=None,
                   help="text file of '<0-based index> <factor>' lines; multiplies the relative "
                        "error of those measurements (land_data.dat order)")
    p.add_argument("--ctype",      type=int, default=None,
                   help="pyGIMLi constraint type (default: pyGIMLi's, =1 smoothness; 10 = smallness+smoothness)")
    p.add_argument("--ref-factor", type=float, default=None,
                   help="homogeneous reference AND starting model = factor × median(ρa) (for DOI)")
    p.add_argument("--verify",     action="store_true",
                   help="check R_m ingredients against pyGIMLi internals and built-in resolution tools")
    return p.parse_args()


if __name__ == "__main__":
    import time, shutil
    args = parse_args()
    t_start = time.time()

    # Tagged runs write to their own folders; untagged runs keep the original paths
    if args.tag:
        PROCESSED_DIR = os.path.join(PROCESSED_DIR, "runs", args.tag)
        FIG_DIR       = os.path.join(FIG_DIR, "runs", args.tag)
        RESULTS_NPZ   = os.path.join(PROCESSED_DIR, "westhoek_inversion_results.npz")
        if os.path.exists(RESULTS_NPZ):
            raise SystemExit(f"Run '{args.tag}' already exists ({RESULTS_NPZ}) — "
                             "choose a new --tag; existing runs are never overwritten.")
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 65)
    print("WESTHOEK ERT — LAND PROFILE INVERSION + RESOLUTION MATRIX")
    print("=" * 65)
    print(f"   tag={args.tag}  lam={args.lam}  err_rel={args.err_rel}  "
          f"paraDepth={args.para_depth}  paraDX={args.para_dx}  robust={args.robust}  "
          f"drop_list={args.drop_list}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading and parsing data...")
    land_data_all = parse_crtomo(LAND_FILE)
    land_data, keep_idx = apply_drop_list(land_data_all, args.drop_list)
    land_elec, _ = parse_electrodes(ELEC_FILE)

    # ── Build pyGIMLi data container ───────────────────────────────────────
    print("\n[2] Building pyGIMLi DataContainerERT...")
    data_cont, idx_map = build_datacont(land_data, land_elec, err_rel=args.err_rel)
    print(f"   DataContainerERT: {data_cont.size()} measurements, "
          f"{data_cont.sensorCount()} sensors")

    # Optional per-datum error inflation (e.g. down-weighting suspect electrodes)
    err_scale = np.ones(data_cont.size())
    if args.err_scale_file:
        pos_of = {orig: i for i, orig in enumerate(keep_idx)}
        with open(args.err_scale_file) as f:
            for line in f:
                s = line.split('#')[0].split()
                if len(s) >= 2 and int(s[0]) in pos_of:
                    err_scale[pos_of[int(s[0])]] = float(s[1])
        data_cont['err'] = np.array(data_cont['err']) * err_scale
        print(f"   Error scale file {args.err_scale_file!r}: {int((err_scale != 1).sum())} data "
              f"inflated (factors {err_scale[err_scale != 1].min() if (err_scale != 1).any() else 1:.2f}"
              f"–{err_scale.max():.2f})")

    # ── Run inversion ──────────────────────────────────────────────────────
    print(f"\n[3] Running ERT inversion (λ={args.lam}, max_iter={args.max_iter})...")
    t_inv = time.time()
    ref_value = (args.ref_factor * float(np.median(data_cont['rhoa']))
                 if args.ref_factor is not None else None)
    mgr = run_inversion(data_cont, lam=args.lam, para_depth=args.para_depth,
                        para_dx=args.para_dx, robust=args.robust, max_iter=args.max_iter,
                        ctype=args.ctype, ref_value=ref_value)
    t_inv = time.time() - t_inv

    # Retrieve inverted resistivity model and mesh
    model = np.array(mgr.model)   # resistivity values per cell [Ω·m]
    mesh  = mgr.paraDomain         # pyGIMLi mesh object

    print(f"   Inverted model: {len(model)} cells")
    print(f"   Resistivity range: {model.min():.2f} – {model.max():.2f} Ω·m")

    # ── Fit statistics (with the errors given to the inversion — i.e. the
    #    original errors for robust runs, the inflated ones with --err-scale-file) ──
    d_obs  = np.array(data_cont['rhoa'])
    d_pred = np.array(mgr.inv.response)
    d_err  = np.array(data_cont['err'])
    norm_res = (np.log(d_obs) - np.log(d_pred)) / d_err
    chi2 = float(np.mean(norm_res**2))
    chi2_pg = float(mgr.inv.chi2())
    n_iter = n_iterations(mgr)
    sensor_z = np.array([s.y() for s in data_cont.sensors()])
    para_bottom = mesh.ymin()
    para_depth_used = float(sensor_z.min() - para_bottom)   # depth below lowest electrode
    print(f"   χ² (log space, original errors) = {chi2:.3f}   [pyGIMLi χ² = {chi2_pg:.3f}]")
    # pyGIMLi stops when χ² ≤ 1, when ΔΦ < 2 %, or at maxIter
    if chi2_pg <= 1.0:
        stop_reason = "chi2<=1"
    elif n_iter >= args.max_iter:
        stop_reason = "iteration cap"
    else:
        stop_reason = "dPhi<2%"
    print(f"   Iterations = {n_iter}  (stop: {stop_reason})   paraDomain bottom = {para_bottom:.1f} m "
          f"(≈{para_depth_used:.1f} m below lowest electrode)")

    # ── Did the C++ core dump negResp debug files (written to the CWD)? ────
    negresp_found = [f for f in NEGRESP_FILES
                     if os.path.exists(f) and os.path.getmtime(f) >= t_start]
    negresp = bool(negresp_found)
    print(f"   negResp debug files written: {'YES ' + str(negresp_found) if negresp else 'no'}")
    if negresp and args.tag:
        dbg = os.path.join(PROCESSED_DIR, "negResp_debug")
        os.makedirs(dbg, exist_ok=True)
        for f in negresp_found:
            shutil.move(f, os.path.join(dbg, f))
        print(f"   (moved to {dbg})")

    # ── Resolution matrix analysis ─────────────────────────────────────────
    print("\n[4] Computing resolution matrix...")
    res = compute_resolution_matrix(mgr, data_cont, lam_expected=args.lam)

    # ── Save results ───────────────────────────────────────────────────────
    print("\n[5] Saving results...")

    # Extract cell centres for plotting
    cell_x = np.array([c.center().x() for c in mesh.cells()])
    cell_z = np.array([c.center().y() for c in mesh.cells()])

    # Resolution radius (Friedel 2003): radius of a disc of resolution 1,
    # r_i = sqrt(A_i / (π · max(Rm_ii, 1e-6))), A_i = cell area
    cell_area = np.array(mesh.cellSizes())
    res_radius = np.sqrt(cell_area / (np.pi * np.maximum(res['Rm_diag'], 1e-6)))
    # Model roughness ‖Cw·ln m‖ (the regularisation term the inversion minimises)
    roughness = float(np.linalg.norm(res['C'] @ np.log(model))) if res['C'] is not None else np.nan
    print(f"   Resolution radius: median {np.median(res_radius):.1f} m, "
          f"range [{res_radius.min():.2f}, {res_radius.max():.0f}] m;  roughness ‖C ln m‖ = {roughness:.3f}")

    if args.verify:
        verify_resolution(mgr, res, cell_x, cell_z,
                          os.path.join(PROCESSED_DIR, "resolution_verification.txt"))

    # J, data errors and weighted constraint matrix (CSR parts) are stored so
    # 03_resolution_analysis.py can compute exact point spread functions
    C_extra = {}
    if res['C'] is not None:
        C_extra = dict(C_data=res['C'].data, C_indices=res['C'].indices,
                       C_indptr=res['C'].indptr, C_shape=np.array(res['C'].shape))
    np.savez(RESULTS_NPZ,
             J=res['J'],
             data_err=res['err'],
             **C_extra,
             model=model,
             cell_x=cell_x,
             cell_z=cell_z,
             Rm_diag=res['Rm_diag'],
             cum_sens=res['cum_sens'],
             sigma_log_m=res['sigma_log_m'],
             lam=res['lam'],
             n_data=res['n_data'],
             n_cells=res['n_cells'],
             # data, fit and run metadata (used by 04_misfit_diagnostics.py)
             d_obs=d_obs, d_pred=d_pred, d_err=d_err,
             chi2=chi2, chi2_pygimli=chi2_pg, n_iter=n_iter,
             keep_idx=keep_idx,
             C1=land_data['C1'], C2=land_data['C2'], P1=land_data['P1'], P2=land_data['P2'],
             R=land_data['R'], K=np.array(data_cont['k']),
             elec_num=np.array(sorted(land_elec)),
             elec_x=np.array([land_elec[k][0] for k in sorted(land_elec)]),
             elec_z=np.array([land_elec[k][1] for k in sorted(land_elec)]),
             para_depth_requested=np.nan if args.para_depth is None else args.para_depth,
             para_depth_used=para_depth_used, para_bottom=para_bottom,
             para_dx=np.nan if args.para_dx is None else args.para_dx,
             err_rel=args.err_rel, robust=args.robust,
             tag=args.tag or "", drop_list=args.drop_list or "",
             cell_area=cell_area, res_radius=res_radius, roughness=roughness,
             max_iter=args.max_iter, stop_reason=stop_reason, err_scale=err_scale,
             err_scale_file=args.err_scale_file or "",
             ctype=np.nan if args.ctype is None else args.ctype,
             ref_factor=np.nan if args.ref_factor is None else args.ref_factor,
             ref_value=np.nan if ref_value is None else ref_value,
             negresp=negresp, runtime_inversion_s=t_inv,
             runtime_total_s=time.time() - t_start)

    print(f"   Saved: {RESULTS_NPZ}")

    # Quick plot to verify
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Inverted resistivity
    pg.show(mesh, model, ax=axes[0], cMap='RdYlBu_r',
            logScale=True, label='ρ [Ω·m]',
            cMin=1, cMax=300,
            colorBar=True)
    axes[0].set_title('Inverted Resistivity Model (land profile)', fontsize=11)
    axes[0].set_ylabel('Elevation (mTAW)', fontsize=10)

    # Plot 2: Normalised cumulative sensitivity (log10 — it spans many decades)
    pg.show(mesh, np.log10(np.clip(res['cum_sens'], 1e-6, None)), ax=axes[1], cMap='hot_r',
            label='log10(normalised sensitivity)', cMin=-5, cMax=0,
            colorBar=True, logScale=False)
    axes[1].set_title('Data Coverage (log10 Normalised Cumulative Sensitivity)', fontsize=11)
    axes[1].set_ylabel('Elevation (mTAW)', fontsize=10)

    # Plot 3: Resolution matrix diagonal
    pg.show(mesh, np.clip(res['Rm_diag'], 0, 1), ax=axes[2],
            cMap='viridis', label='Rm diagonal (0=unresolved → 1=resolved)',
            colorBar=True, logScale=False)
    axes[2].set_title('Model Resolution Matrix Diagonal', fontsize=11)
    axes[2].set_xlabel('Distance along profile (m)', fontsize=10)
    axes[2].set_ylabel('Elevation (mTAW)', fontsize=10)

    fig.suptitle('Westhoek Land ERT — Inversion + Resolution Analysis', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "04_inversion_resolution_quickview.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {fig_path}")
    plt.close(fig)

    print(f"\n   Runtime: inversion {t_inv:.0f} s, total {time.time() - t_start:.0f} s")
    print("\n✓ Done. Next: run 03_resolution_analysis.py for detailed plots.")
