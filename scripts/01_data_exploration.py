"""
01_data_exploration.py
======================
Westhoek ERT Dataset – Data Exploration and Pseudosection Plots
De Panne, Belgium (North Sea intertidal zone)

Dataset: Hermans & Paepen (2019), Marine Data Archive, Flanders Marine Institute
Method reference: Hermans & Paepen (2020), GRL, doi:10.1029/2019GL085877

What this script does:
  1. Parses CRTOMO-format land and marine data files
  2. Parses the electrode position file
  3. Plots electrode geometry (profile view)
  4. Plots apparent resistivity pseudosections for both datasets
  5. Estimates data noise statistics

Usage:
  python 01_data_exploration.py

Requires: numpy, matplotlib (no pyGIMLi needed for this script)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ─── File paths — relative to this scripts/ folder ───────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "raw"))
LAND_FILE     = os.path.join(DATA_DIR, "land_data.dat")
SEA_FILE      = os.path.join(DATA_DIR, "sea_data.dat")
ELEC_FILE     = os.path.join(DATA_DIR, "elec.dat")
OUTPUT_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "figures"))            # figures
PROCESSED_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "processed"))  # .npz
PARSED_NPZ    = os.path.join(PROCESSED_DIR, "westhoek_parsed.npz")
SAVE_NPZ      = True

# Optional: python 01_data_exploration.py --out-dir <folder>
#   writes the figures there and does NOT (re)write westhoek_parsed.npz
import argparse as _ap
_p = _ap.ArgumentParser(description="Westhoek ERT — data exploration")
_p.add_argument("--out-dir", default=None, help="figure folder (skips writing the parsed .npz)")
_args, _ = _p.parse_known_args()
if _args.out_dir:
    OUTPUT_DIR = os.path.abspath(_args.out_dir)
    SAVE_NPZ = False
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─── Helper: parse CRTOMO data file ──────────────────────────────────────────
def parse_crtomo(filepath):
    """
    Parse a CRTOMO-format data file.

    CRTOMO format:
      Line 1: number of data points (N)
      Lines 2..N+1: C1  C2  P1  P2  R[Ohm]  err
        where C1,C2 = current electrode indices (1-based)
              P1,P2 = potential electrode indices (1-based)
              R     = measured resistance in Ohm
              err   = error level (0 = not set)

    Returns
    -------
    dict with keys:
        'n'   : number of data points (int)
        'C1','C2','P1','P2' : electrode index arrays (int, 1-based)
        'R'   : resistance array (float, Ohm)
        'err' : error array (float)
    """
    with open(filepath) as f:
        raw = [l.strip() for l in f]

    # First non-empty line is the header
    data_lines = [l for l in raw if l]
    n = int(data_lines[0].split()[0])

    rows = []
    for line in data_lines[1:n+1]:
        parts = line.split()
        if len(parts) < 5:
            continue
        c1, c2, p1, p2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        R  = float(parts[4])
        err = float(parts[5]) if len(parts) > 5 else 0.0
        rows.append((c1, c2, p1, p2, R, err))

    arr = np.array(rows)
    return {
        'n'  : len(arr),
        'C1' : arr[:,0].astype(int),
        'C2' : arr[:,1].astype(int),
        'P1' : arr[:,2].astype(int),
        'P2' : arr[:,3].astype(int),
        'R'  : arr[:,4],
        'err': arr[:,5],
    }


# ─── Helper: parse electrode file ────────────────────────────────────────────
def parse_electrodes(filepath):
    """
    Parse the electrode position file (tab-delimited).

    Columns: index  X(m)  Z_elevation(mTAW)  Hsea(mTAW)
    Land electrodes have '\' or empty in the Hsea column.
    Marine (floating/seafloor) electrodes have a numeric Hsea value.

    Returns
    -------
    land_elec : dict {index: (x, z)}
    sea_elec  : dict {index: (x, z, hsea)}
    """
    land_elec = {}
    sea_elec  = {}

    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if not parts[0].strip().isdigit():
                continue
            idx  = int(parts[0])
            x    = float(parts[1])
            z    = float(parts[2])
            hsea = parts[3].strip() if len(parts) > 3 else '\\'

            if hsea in ('\\', ''):
                land_elec[idx] = (x, z)
            else:
                sea_elec[idx] = (x, z, float(hsea))

    return land_elec, sea_elec


# ─── Helper: compute pseudosection midpoint and depth ────────────────────────
def pseudosection_coords(C1_idx, C2_idx, P1_idx, P2_idx, elec_dict):
    """
    Compute pseudosection midpoint (x) and pseudo-depth (z) for each measurement.
    Uses the convention: mid-x = mean of all 4 electrode positions,
    pseudo-depth = half the maximum electrode separation (a-spacing).
    """
    xC1 = np.array([elec_dict[i][0] for i in C1_idx])
    xC2 = np.array([elec_dict[i][0] for i in C2_idx])
    xP1 = np.array([elec_dict[i][0] for i in P1_idx])
    xP2 = np.array([elec_dict[i][0] for i in P2_idx])

    x_mid     = (xC1 + xC2 + xP1 + xP2) / 4.0
    max_sep   = np.maximum(np.abs(xC2 - xC1), np.abs(xP2 - xP1))
    pseudo_z  = -0.2 * max_sep          # standard ERT pseudo-depth factor

    return x_mid, pseudo_z


# ─── Helper: compute apparent resistivity ────────────────────────────────────
def geometric_factor_wenner_schlumberger(C1_idx, C2_idx, P1_idx, P2_idx, elec_dict):
    """
    Geometric factor K for Wenner-Schlumberger-like arrays.
    K = pi * a * n * (n+1)   for WS with factor n
    Here we compute K from electrode positions directly:
    K = 2*pi / (1/r_C1P1 - 1/r_C1P2 - 1/r_C2P1 + 1/r_C2P2)
    where r_AB = |x_A - x_B|
    """
    def xpos(idx_arr):
        return np.array([elec_dict[i][0] for i in idx_arr])

    xC1 = xpos(C1_idx); xC2 = xpos(C2_idx)
    xP1 = xpos(P1_idx); xP2 = xpos(P2_idx)

    # distances
    r_C1P1 = np.abs(xP1 - xC1)
    r_C1P2 = np.abs(xP2 - xC1)
    r_C2P1 = np.abs(xP1 - xC2)
    r_C2P2 = np.abs(xP2 - xC2)

    # avoid division by zero for collocated electrodes (shouldn't occur)
    eps = 1e-12
    K = 2 * np.pi / (1.0/(r_C1P1+eps) - 1.0/(r_C1P2+eps)
                     - 1.0/(r_C2P1+eps) + 1.0/(r_C2P2+eps))
    return K


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 60)
    print("WESTHOEK ERT DATASET — DATA EXPLORATION")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading data files...")
    land = parse_crtomo(LAND_FILE)
    sea  = parse_crtomo(SEA_FILE)
    land_elec, sea_elec = parse_electrodes(ELEC_FILE)

    # Combine electrode dicts for cross-reference
    all_elec = {}
    all_elec.update(land_elec)
    all_elec.update({k: (v[0], v[1]) for k, v in sea_elec.items()})

    print(f"   Land data: {land['n']} measurements")
    print(f"   Sea  data: {sea['n']} measurements")
    print(f"   Electrodes: {len(land_elec)} land, {len(sea_elec)} marine")

    # ── Data statistics ────────────────────────────────────────────────────
    print("\n[2] Data statistics:")
    print(f"   Land R range : {land['R'].min():.4f} – {land['R'].max():.4f} Ohm")
    print(f"   Land R median: {np.median(land['R']):.4f} Ohm")
    print(f"   Sea  R range : {sea['R'].min():.6f} – {sea['R'].max():.6f} Ohm")
    print(f"   Sea  R median: {np.median(sea['R']):.6f} Ohm")
    print(f"   Negative R values — Land: {(land['R']<0).sum()}, Sea: {(sea['R']<0).sum()}")

    # ── Error estimation ───────────────────────────────────────────────────
    # The GRL paper uses 3% relative error for both datasets.
    # Since reciprocal measurements are not explicitly provided in the files,
    # we assign the standard estimate used in the paper.
    err_abs_floor  = 1e-4      # Ohm  — absolute noise floor
    err_rel        = 0.03      # 3%   — relative error

    land_err = np.sqrt((err_rel * land['R'])**2 + err_abs_floor**2) / land['R']
    sea_err  = np.sqrt((err_rel * sea['R'])**2  + err_abs_floor**2) / sea['R']

    print(f"\n[3] Estimated relative error (3% + floor):")
    print(f"   Land: mean={land_err.mean()*100:.2f}%, median={np.median(land_err)*100:.2f}%")
    print(f"   Sea:  mean={sea_err.mean()*100:.2f}%, median={np.median(sea_err)*100:.2f}%")

    # ── Apparent resistivity ───────────────────────────────────────────────
    print("\n[4] Computing apparent resistivities...")
    K_land = geometric_factor_wenner_schlumberger(
        land['C1'], land['C2'], land['P1'], land['P2'], all_elec)
    K_sea  = geometric_factor_wenner_schlumberger(
        sea['C1'], sea['C2'], sea['P1'], sea['P2'], all_elec)

    rho_a_land = K_land * land['R']
    rho_a_sea  = K_sea  * sea['R']

    # Remove non-physical (negative) apparent resistivities if any
    mask_land = (rho_a_land > 0) & np.isfinite(rho_a_land)
    mask_sea  = (rho_a_sea  > 0) & np.isfinite(rho_a_sea)
    print(f"   Land ρ_a range: {rho_a_land[mask_land].min():.2f} – {rho_a_land[mask_land].max():.2f} Ω·m")
    print(f"   Sea  ρ_a range: {rho_a_sea[mask_sea].min():.2f}  – {rho_a_sea[mask_sea].max():.2f}  Ω·m")

    # ── Pseudosection coordinates ──────────────────────────────────────────
    x_land, z_land = pseudosection_coords(
        land['C1'], land['C2'], land['P1'], land['P2'], all_elec)
    x_sea, z_sea   = pseudosection_coords(
        sea['C1'],  sea['C2'],  sea['P1'],  sea['P2'],  all_elec)

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 1: Electrode geometry profile
    # ══════════════════════════════════════════════════════════════════════
    print("\n[5] Generating figures...")
    fig, ax = plt.subplots(figsize=(14, 3.5))

    sea_x_arr = np.array([v[0] for v in sea_elec.values()])
    hsea_arr  = np.array([v[2] for v in sea_elec.values()])
    h_high    = hsea_arr.mean()

    # Land electrodes (sorted by x)
    lx = np.array([land_elec[k][0] for k in sorted(land_elec, key=lambda k: land_elec[k][0])])
    lz = np.array([land_elec[k][1] for k in sorted(land_elec, key=lambda k: land_elec[k][0])])
    # Marine electrodes (seafloor, sorted by x)
    sx = np.array([sea_elec[k][0] for k in sorted(sea_elec, key=lambda k: sea_elec[k][0])])
    sz = np.array([sea_elec[k][1] for k in sorted(sea_elec, key=lambda k: sea_elec[k][0])])

    # High-tide water column: only between the bed and the high-water level, where
    # the bed is below it (bed = land profile where it exists, seafloor beyond)
    bx = np.unique(np.r_[lx, sx[sx > lx.max()]])
    bz = np.where(bx <= lx.max(), np.interp(bx, lx, lz), np.interp(bx, sx, sz))
    xf = np.linspace(bx.min(), bx.max(), 2000)
    zf = np.interp(xf, bx, bz)
    ax.fill_between(xf, zf, h_high, where=zf < h_high, color='#AADDF0', alpha=0.55,
                    lw=0, label='Water column at high tide', zorder=1)

    # Marine electrodes first, land electrodes drawn on top in the overlap zone
    ax.plot(sx, sz, 'b-', lw=0.8, zorder=2)
    ax.plot(sx, sz, 'b^', ms=3, zorder=3, label=f'Marine electrodes ({len(sea_elec)})')
    ax.plot(lx, lz, 'k-', lw=0.9, zorder=4)
    ax.plot(lx, lz, 'ko', ms=3.5, mfc='white', mew=0.9, zorder=5,
            label=f'Land electrodes ({len(land_elec)})')

    # High water line
    ax.axhline(h_high, color='steelblue', ls='--', lw=0.8, label=f'High water level ({h_high:.2f} mTAW)')
    # Low water line (approx. last land electrode elevation)
    ax.axhline(-0.15, color='gray', ls=':', lw=0.8, label='Low water level (−0.15 mTAW)')

    ax.axvspan(sea_x_arr.min(), lx.max(), alpha=0.08, color='orange', label='Overlap zone')
    ax.set_xlabel('Distance along profile (m)', fontsize=11)
    ax.set_ylabel('Elevation (mTAW)', fontsize=11)
    ax.set_title('Westhoek ERT Profile — Electrode Geometry', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95, facecolor='white', labelcolor='0.1')
    ax.set_xlim(-10, sx.max()+10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname1 = os.path.join(OUTPUT_DIR, "01_electrode_geometry.png")
    fig.savefig(fname1, dpi=150)
    print(f"   Saved: {fname1}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 2: Land pseudosection
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Color scale: log10(ρ_a)
    vmin, vmax = 0, 2.5      # log10 scale: 1 – ~316 Ω·m

    # Land
    sc1 = axes[0].scatter(x_land[mask_land], z_land[mask_land],
                          c=np.log10(rho_a_land[mask_land]),
                          cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                          s=6, marker='s', linewidths=0)
    cb1 = plt.colorbar(sc1, ax=axes[0], pad=0.01)
    cb1.set_label('log₁₀(ρₐ) [Ω·m]', fontsize=9)
    cb1.set_ticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    cb1.set_ticklabels(['1', '3', '10', '32', '100', '316'])
    axes[0].set_xlabel('Distance (m)', fontsize=10)
    axes[0].set_ylabel('Pseudo-depth (m)', fontsize=10)
    axes[0].set_title(f'Land ERT Pseudosection — Multiple Gradient Array ({land["n"]} data points)', fontsize=11)
    axes[0].set_xlim(-5, lx.max()+5)
    axes[0].grid(True, alpha=0.25)

    # Sea
    sc2 = axes[1].scatter(x_sea[mask_sea], z_sea[mask_sea],
                          c=np.log10(rho_a_sea[mask_sea]),
                          cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                          s=6, marker='s', linewidths=0)
    cb2 = plt.colorbar(sc2, ax=axes[1], pad=0.01)
    cb2.set_label('log₁₀(ρₐ) [Ω·m]', fontsize=9)
    cb2.set_ticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    cb2.set_ticklabels(['1', '3', '10', '32', '100', '316'])
    axes[1].set_xlabel('Distance (m)', fontsize=10)
    axes[1].set_ylabel('Pseudo-depth (m)', fontsize=10)
    axes[1].set_title(f'Marine ERT Pseudosection — Reciprocal Wenner-Schlumberger ({sea["n"]} data points)', fontsize=11)
    axes[1].set_xlim(sx.min()-5, sx.max()+5)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle('Westhoek Apparent Resistivity Pseudosections', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fname2 = os.path.join(OUTPUT_DIR, "02_pseudosections.png")
    fig.savefig(fname2, dpi=150, bbox_inches='tight')
    print(f"   Saved: {fname2}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 3: Resistance histogram (data distribution)
    # ══════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(np.log10(land['R']), bins=60, color='#2C6FAC', edgecolor='none', alpha=0.8)
    ax1.set_xlabel('log₁₀(R) [Ω]', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f'Land Data Distribution (n={land["n"]})', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.log10(np.median(land['R'])), color='red', ls='--',
                label=f'Median = {np.median(land["R"]):.4f} Ω')
    ax1.legend(fontsize=9)

    ax2.hist(np.log10(sea['R']), bins=40, color='#1A8C5A', edgecolor='none', alpha=0.8)
    ax2.set_xlabel('log₁₀(R) [Ω]', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title(f'Marine Data Distribution (n={sea["n"]})', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(np.log10(np.median(sea['R'])), color='red', ls='--',
                label=f'Median = {np.median(sea["R"]):.5f} Ω')
    ax2.legend(fontsize=9)

    fig.suptitle('Westhoek ERT — Resistance Distributions', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fname3 = os.path.join(OUTPUT_DIR, "03_data_distributions.png")
    fig.savefig(fname3, dpi=150)
    print(f"   Saved: {fname3}")
    plt.close(fig)

    # ── Save parsed data as numpy arrays for next scripts ─────────────────
    if not SAVE_NPZ:
        print(f"\n   (--out-dir given: {PARSED_NPZ} left unchanged)")
    else:
      np.savez(PARSED_NPZ,
             land_C1=land['C1'], land_C2=land['C2'],
             land_P1=land['P1'], land_P2=land['P2'],
             land_R=land['R'],
             sea_C1=sea['C1'],  sea_C2=sea['C2'],
             sea_P1=sea['P1'],  sea_P2=sea['P2'],
             sea_R=sea['R'],
             land_x=np.array([land_elec[k][0] for k in sorted(land_elec)]),
             land_z=np.array([land_elec[k][1] for k in sorted(land_elec)]),
             land_idx=np.array(sorted(land_elec.keys())),
             sea_x=np.array([sea_elec[k][0] for k in sorted(sea_elec)]),
             sea_z=np.array([sea_elec[k][1] for k in sorted(sea_elec)]),
             sea_idx=np.array(sorted(sea_elec.keys())),
             sea_hsea=np.array([sea_elec[k][2] for k in sorted(sea_elec)]),
             rho_a_land=rho_a_land, rho_a_sea=rho_a_sea,
             K_land=K_land, K_sea=K_sea)
      print(f"\n   Saved: {PARSED_NPZ}  (used by script 03)")

    print("\n✓ Data exploration complete. Check the three figure files.")
    print("  Next: run 02_land_inversion.py")
