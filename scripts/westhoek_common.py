"""
westhoek_common.py
==================
Shared helpers for the Westhoek ERT scripts (03, 09, …).

- load_run / land_surface
- rebuild_paradomain : re-creates a run's inversion paraDomain with the same
  pyGIMLi call as 02 (ERTManager.createMesh) and verifies it cell-by-cell
  against the stored cell centres, so model quantities can be drawn on the
  real mesh instead of griddata interpolation of cell centres
- draw_cells / hatch_cells / setup_section : plotting on the mesh
- ColumnSampler : sparse (points × cells) operator = node interpolation on the
  mesh ∘ cell→node averaging; evaluates any cell vector (or many at once) on
  vertical columns below the land surface
- crossings / main_lens / draw_interfaces : ρ = 3 Ω·m interface extraction
- normal_matrix / cholesky_A : A = JᵀWd²J + λ·CᵀC from exactly the ingredients
  of the verified R_m computation in 02
- fwhm : full-width-at-half-maximum extents of a kernel on the cells
"""

import os

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT          = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
FIG_DIR       = os.path.join(ROOT, "figures")
RESULTS_DIR   = os.path.join(ROOT, "results")
ELEC_FILE     = os.path.join(ROOT, "data", "raw", "elec.dat")

# Inputs that are not public (see data/README.md): the scripts that need them stop with require_input()'s message.
RESTRICTED_INPUTS = {
    "cpt": (os.path.join(ROOT, "data", "raw", "cpt_vrancken_2024"),
            "mechanical CPT logs and surveyed positions of 23 September 2024 (E. Vrancken, Ghent University; not public)"),
    "abem": (os.path.join(ROOT, "data", "raw", "Project12_LS210110079_2018-03-07_11-22-23", "project.db"),
             "2018 Terrameter LS instrument project (T. Hermans, Ghent University; not public)"),
    "cpt_json": (os.path.join(ROOT, "data", "processed", "v1_4", "cpt.json"),
                 "sounding geometry and lithology derived from the mechanical CPTs by `27_dp_groundtruth.py cpt`"),
}


def require_input(key, needed_for, path=None):
    """Stop with a clear message when a restricted input (or a product derived from one) is missing. `path` overrides the
    default location (for scripts writing to another --out tree)."""
    default, what = RESTRICTED_INPUTS[key]
    path = path or default
    if not os.path.exists(path):
        raise SystemExit(f"restricted input missing: {os.path.relpath(path, ROOT)} ({what}). Needed for {needed_for}. "
                         "See data/README.md: this step can be rerun once the data owners publish the data.")
    return path


LOG3 = np.log10(3.0)
Z_FLOOR = -45.0          # sections are drawn down to this elevation (mTAW)
VE = 3.0                 # vertical exaggeration of section figures
X_LIM = (-5.0, 510.0)


# ─────────────────────────────────────────────────────────────────────────────
def run_npz_path(tag=None):
    if tag:
        return os.path.join(PROCESSED_DIR, "runs", tag, "westhoek_inversion_results.npz")
    return os.path.join(PROCESSED_DIR, "westhoek_inversion_results.npz")


def load_run(tag=None, path=None):
    """Load a results npz into a dict; rebuild the weighted constraint matrix."""
    d = np.load(path or run_npz_path(tag), allow_pickle=True)
    out = {k: d[k] for k in d.files}
    if "C_data" in out:
        out["C"] = sp.csr_matrix((out.pop("C_data"), out.pop("C_indices"), out.pop("C_indptr")),
                                 shape=tuple(out.pop("C_shape")))
    return out


def read_electrodes():
    """Land electrodes (x, z) and marine electrodes (x, z, hsea), sorted by x."""
    land, sea = [], []
    with open(ELEC_FILE) as f:
        for line in f:
            p = line.strip().split("\t")
            if not p[0].strip().isdigit():
                continue
            x, z = float(p[1]), float(p[2])
            h = p[3].strip() if len(p) > 3 else "\\"
            (land if h in ("\\", "") else sea).append((x, z) if h in ("\\", "") else (x, z, float(h)))
    return np.array(sorted(land)), np.array(sorted(sea))


def save_publication(fig, out_root, name, clear_axes_titles=()):
    """Publication copy of a report figure → <out_root>/results/figs/<name>.{png,pdf}: no working title (the suptitle and the
    given axes titles are hidden for this save and restored afterwards), 600 dpi PNG, PDF without date metadata."""
    d = os.path.join(out_root, "results", "figs"); os.makedirs(d, exist_ok=True)
    st = getattr(fig, "_suptitle", None); vis = st.get_visible() if st is not None else None
    old = [(ax, ax.get_title()) for ax in clear_axes_titles]
    if st is not None:
        st.set_visible(False)
    for ax, _ in old:
        ax.set_title("")
    fig.savefig(os.path.join(d, name + ".png"), dpi=600, bbox_inches="tight", pad_inches=0.03, metadata={"Software": None})
    fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight", pad_inches=0.03,
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    if st is not None:
        st.set_visible(vis)
    for ax, t in old:
        ax.set_title(t)


def land_surface():
    land, _ = read_electrodes()
    return land[:, 0], land[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
def rebuild_paradomain(run):
    """
    Re-create the paraDomain of a stored run and verify it against the stored
    cell centres (same order). Raises if the rebuild does not match.
    """
    import pygimli as pg
    from pygimli.physics import ert
    if "elec_x" in run:
        ex, ez = run["elec_x"], run["elec_z"]
    else:
        ex, ez = land_surface()
    data = pg.DataContainerERT()
    for x, z in zip(ex, ez):
        data.createSensor(pg.Pos(float(x), float(z)))
    kw = {}
    pdr = float(run.get("para_depth_requested", np.nan))
    pdx = float(run.get("para_dx", np.nan))
    if np.isfinite(pdr):
        kw["paraDepth"] = pdr
    if np.isfinite(pdx):
        kw["paraDX"] = pdx
    mgr = ert.ERTManager(data, verbose=False)
    mgr.createMesh(data, **kw)
    pd = mgr.fop.paraDomain
    cx = np.array([c.center().x() for c in pd.cells()])
    cz = np.array([c.center().y() for c in pd.cells()])
    if len(cx) != len(run["cell_x"]):
        raise RuntimeError(f"Mesh rebuild gave {len(cx)} cells, run has {len(run['cell_x'])}")
    err = float(np.max(np.hypot(cx - run["cell_x"], cz - run["cell_z"])))
    if err > 1e-6:
        raise RuntimeError(f"Mesh rebuild does not match stored cell centres (max offset {err:.3g} m)")
    return pd, err


def cell_polygons(mesh):
    return [np.array([[n.pos().x(), n.pos().y()] for n in c.nodes()]) for c in mesh.cells()]


# ─────────────────────────────────────────────────────────────────────────────
def draw_cells(ax, polys, values, cmap, norm, mask=None, rasterized=True, **kw):
    """Fill each mesh cell with its value (no interpolation). mask=True → blank."""
    from matplotlib.collections import PolyCollection
    v = np.ma.masked_invalid(np.asarray(values, float))
    if mask is not None:
        v = np.ma.masked_where(mask, v)
    pc = PolyCollection(polys, array=v, cmap=cmap, norm=norm, edgecolors="face",
                        linewidths=0.05, rasterized=rasterized, **kw)
    import matplotlib
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    pc.set_cmap(cm.with_extremes(bad=(1, 1, 1, 0)))
    ax.add_collection(pc)
    return pc


def hatch_cells(ax, polys, mask, label=None):
    """Hatch the cells where mask is True (e.g. resolution radius > r_max)."""
    from matplotlib.collections import PolyCollection
    sel = [p for p, m in zip(polys, mask) if m]
    if not sel:
        return None
    pc = PolyCollection(sel, facecolors="none", edgecolors="0.35", hatch="////",
                        linewidths=0.0, label=label)
    ax.add_collection(pc)
    return pc


def setup_section(ax, title=None, ve=VE, z_floor=Z_FLOOR, xlim=X_LIM, show_surface=True):
    ex, ez = land_surface()
    if show_surface:
        ax.plot(ex, ez, "k-", lw=0.8, zorder=6)
    ax.set_xlim(*xlim)
    ax.set_ylim(z_floor, max(ez) + 2)
    ax.set_aspect(ve)
    ax.set_xlabel("Distance along profile (m)")
    ax.set_ylabel("Elevation (mTAW)")
    ax.text(0.995, 0.02, f"V.E. {ve:g}×", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9), zorder=10)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=10)


def white_legend(ax, **kw):
    """Legend with white background and dark text (readable on any colour map)."""
    kw.setdefault("fontsize", 8)
    kw.setdefault("framealpha", 0.95)
    leg = ax.legend(facecolor="white", edgecolor="0.5", labelcolor="0.1", **kw)
    return leg


def label(ax, x, y, text, **kw):
    """Hydrogeological label with a white box so it stays readable."""
    kw.setdefault("fontsize", 8)
    ax.text(x, y, text, ha="center", va="center", color="0.1", zorder=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.85), **kw)


# ─────────────────────────────────────────────────────────────────────────────
def cell_to_node_matrix(mesh):
    """Sparse (nodes × cells): area-weighted mean of the cells around each node."""
    area = np.array(mesh.cellSizes())
    rows, cols, vals = [], [], []
    for n in mesh.nodes():
        cs = [c.id() for c in n.cellSet()]
        w = area[cs] / area[cs].sum()
        rows += [n.id()] * len(cs); cols += cs; vals += list(w)
    return sp.csr_matrix((vals, (rows, cols)), shape=(mesh.nodeCount(), mesh.cellCount()))


class ColumnSampler:
    """
    Evaluate cell vectors on vertical columns x_j, from the land surface down to
    z_floor with step dz, by linear interpolation on the mesh (cell → node
    averaging, then P1 interpolation inside the triangles). Works on a single
    vector (n_cells,) or a block (n_cells, n_samples).
    """

    def __init__(self, mesh, x_cols, z_floor=Z_FLOOR, dz=0.2):
        import pygimli as pg
        ex, ez = land_surface()
        self.x = np.asarray(x_cols, float)
        self.surf = np.interp(self.x, ex, ez)
        self.z = np.arange(np.floor(ez.max() / dz) * dz, z_floor - 1e-9, -dz)   # top → down
        X, Z = np.meshgrid(self.x, self.z)                                        # (nz, nx)
        self.below = Z <= self.surf[None, :] - 1e-6
        pos = [pg.Pos(float(x), float(z)) for x, z in zip(X.ravel(), Z.ravel())]
        I = pg.utils.sparseMatrix2csr(pg.utils.toSparseMatrix(mesh.interpolationMatrix(pos)))
        inside = np.asarray(np.abs(I).sum(axis=1)).ravel() > 0
        self.valid = inside.reshape(X.shape) & self.below
        self.M = (I @ cell_to_node_matrix(mesh)).tocsr()
        self.shape = X.shape

    def __call__(self, cell_values):
        v = np.asarray(cell_values, float)
        out = self.M @ v
        if v.ndim == 1:
            out = out.reshape(self.shape)
            out[~self.valid] = np.nan
        else:
            out = out.reshape(self.shape + (v.shape[1],))
            out[~self.valid] = np.nan
        return out


def crossings(col_vals, z, level=LOG3):
    """
    All crossings of `level` in one column (values ordered from the surface
    down). Returns list of (kind, z) with kind 'top' (≥ level above, < level
    below: top of a low-ρ body) or 'base' (< level above, ≥ level below).
    """
    out = []
    v = col_vals
    for k in range(len(v) - 1):
        a, b = v[k], v[k + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if (a >= level) != (b >= level):
            t = (level - a) / (b - a)
            out.append(("top" if a >= level else "base", z[k] + t * (z[k + 1] - z[k])))
    return out


def draw_interfaces(ax, x_cols, col_crossings, masked=None, colors=("navy", "darkorange"),
                    lw=1.6, labels=("lens top (ρ=3 Ω·m, >3 above)", "lens base (ρ=3 Ω·m, <3 above)")):
    """
    Draw lens-top and lens-base lines. The k-th crossing of neighbouring columns
    is joined only if both columns have the same crossing sequence (so lines are
    never joined where crossings appear or disappear). masked[j][k]=True hides a
    crossing (e.g. resolution radius > r_max) without changing the topology.
    """
    kinds = [tuple(k for k, _ in cc) for cc in col_crossings]
    used = {"top": False, "base": False}
    j = 0
    n = len(x_cols)
    while j < n:
        seq = kinds[j]
        j2 = j
        while j2 + 1 < n and kinds[j2 + 1] == seq:
            j2 += 1
        for k, kind in enumerate(seq):
            xs = np.array(x_cols[j:j2 + 1], float)
            zs = np.array([col_crossings[i][k][1] for i in range(j, j2 + 1)], float)
            if masked is not None:
                zs[[masked[i][k] for i in range(j, j2 + 1)]] = np.nan
            c = colors[0] if kind == "top" else colors[1]
            lab = None
            if not used[kind] and np.isfinite(zs).any():
                lab = labels[0] if kind == "top" else labels[1]
                used[kind] = True
            if len(xs) == 1:
                ax.plot(xs, zs, "o", ms=2, color=c, label=lab, zorder=7)
            else:
                ax.plot(xs, zs, "-", color=c, lw=lw, label=lab, zorder=7)
        j = j2 + 1


def crossing_weights(cs, j, zc):
    """
    Sparse row w (1 × n_cells) such that w·log10(m) is the interpolated log10ρ at
    elevation zc in column j — exactly the weights the interface extraction uses
    (linear between the two grid points around zc, each = M row).
    Returns (w, k, t) with k the upper grid index and t the fraction to k+1.
    """
    nx = len(cs.x)
    k = int(np.clip(np.floor((cs.z[0] - zc) / (cs.z[0] - cs.z[1])), 0, len(cs.z) - 2))
    t = (zc - cs.z[k]) / (cs.z[k + 1] - cs.z[k])
    w = (1 - t) * cs.M[k * nx + j] + t * cs.M[(k + 1) * nx + j]
    return w, k, t


def sigma_z_interp(cs, L, log10m, zcs):
    """
    Analytic σ_z of interface crossings using the variance of the INTERPOLATED
    value: var(w·log10 m) = wᵀ A⁻¹ w / ln(10)²  (A = L Lᵀ, posterior of ln m),
    σ_z = √var / |∂(log10ρ)/∂z| at the crossing. zcs: crossing elevation per
    column (NaN = none). Returns (σ_z, σ_log10 at the crossing) per column.
    """
    V = cs(log10m)
    sz = np.full(len(cs.x), np.nan); sv = np.full(len(cs.x), np.nan)
    for j, zc in enumerate(zcs):
        if not np.isfinite(zc):
            continue
        w, k, t = crossing_weights(cs, j, zc)
        wd = np.asarray(w.todense()).ravel()
        var = float(wd @ sla.cho_solve((L, True), wd)) / np.log(10) ** 2
        g = (V[k + 1, j] - V[k, j]) / (cs.z[k + 1] - cs.z[k])
        sv[j] = np.sqrt(var)
        sz[j] = sv[j] / abs(g) if g != 0 else np.nan
    return sz, sv


def main_lens(V, z, level=LOG3):
    """
    Vectorised main-lens interfaces for columns × samples.
    V: (nz, nx, ns) log10ρ from the surface down (NaN outside the model).
    Lens top  = shallowest 'top' crossing; lens base = shallowest 'base'
    crossing below that top (or the shallowest 'base' if the column starts
    inside a low-ρ body at the surface). Returns (top, base) arrays (nx, ns),
    NaN where absent.
    """
    a, b = V[:-1], V[1:]
    ok = ~(np.isnan(a) | np.isnan(b))
    top_m = ok & (a >= level) & (b < level)
    base_m = ok & (a < level) & (b >= level)
    t = np.where(ok, (level - a) / np.where(b != a, b - a, np.nan), np.nan)
    zk = z[:-1][:, None, None]; dzk = (z[1:] - z[:-1])[:, None, None]
    zc = zk + t * dzk
    nz = V.shape[0] - 1
    kidx = np.arange(nz)[:, None, None]
    has_top = top_m.any(axis=0)
    ktop = np.where(has_top, top_m.argmax(axis=0), -1)
    top = np.where(has_top, np.take_along_axis(zc, np.maximum(ktop, 0)[None], 0)[0], np.nan)
    base_after = base_m & (kidx > ktop[None])
    has_base = base_after.any(axis=0)
    kbase = base_after.argmax(axis=0)
    base = np.where(has_base, np.take_along_axis(zc, kbase[None], 0)[0], np.nan)
    return top, base


# ─────────────────────────────────────────────────────────────────────────────
def normal_matrix(run):
    """
    A = JᵀWd²J + λ·CᵀC with J the log-log Jacobian, Wd = 1/err, C the weighted
    constraint matrix (diag(cWeight)·C, incl. zWeight) and λ the C++ value —
    exactly the ingredients of the R_m computation in 02 (verified against
    pyGIMLi internals with 02 --verify).
    """
    J = np.asarray(run["J"], float)
    Wd = 1.0 / np.maximum(np.asarray(run["data_err"], float), 1e-6)
    Jw = Wd[:, None] * J
    JtWJ = Jw.T @ Jw
    C = run["C"]
    CtC = (C.T @ C).toarray()
    lam = float(run["lam"])
    return JtWJ + lam * CtC, JtWJ, lam


def cholesky_A(run):
    """Lower Cholesky factor L of A (A = L Lᵀ) plus A, JᵀWd²J and λ."""
    A, JtWJ, lam = normal_matrix(run)
    L = sla.cholesky(A, lower=True)
    return L, A, JtWJ, lam


def fwhm(values, cx, cz):
    """Horizontal and vertical extent of cells with value ≥ half the peak (global threshold)."""
    v = np.asarray(values, float)
    pk = np.nanmax(v)
    sel = v >= 0.5 * pk
    return float(cx[sel].max() - cx[sel].min()), float(cz[sel].max() - cz[sel].min()), int(sel.sum())


def lobe_fwhm(mesh, values, j, cx, cz):
    """
    FWHM of the main lobe around target cell j: cells connected to j (through
    shared edges) whose value is ≥ half the value at j. Returns (Δx, Δz, n_cells).
    Robust against side lobes elsewhere (e.g. oscillating near-surface cells).
    """
    v = np.asarray(values, float)
    half = 0.5 * v[j]
    if not hasattr(mesh, "_nb"):
        # cell adjacency through shared edges (boundary left/right cells);
        # the links exist only after createNeighborInfos() (pyGIMLi 1.5.5)
        mesh.createNeighborInfos()
        nb = [[] for _ in range(mesh.cellCount())]
        for b in mesh.boundaries():
            lc, rc = b.leftCell(), b.rightCell()
            if lc is not None and rc is not None:
                nb[lc.id()].append(rc.id()); nb[rc.id()].append(lc.id())
        mesh._nb = nb
    seen = {j}; stack = [j]
    while stack:
        i = stack.pop()
        for n in mesh._nb[i]:
            if n not in seen and v[n] >= half:
                seen.add(n); stack.append(n)
    s = np.fromiter(seen, int)
    return float(cx[s].max() - cx[s].min()), float(cz[s].max() - cz[s].min()), int(len(s))
