# Decision log — v1.1 onwards

The v1.0 decision log (stages 1–7.3, up to tag `v1.0.3-numbers`) is in `results/REFERENCE.md`, section 4. This file
records the decisions of v1.1 and later. **Pre-registered entries are not edited after results are seen; any change is
a new, dated follow-up entry.**

| date | id | entry |
|---|---|---|
| 2026-09-30 | v1.1-0 | **Scope of v1.1** (branch `v1.1` from `v1.0.3-numbers`): (1) synthetic test with the 1980 K1 borehole-log structure (Lebbe 1981, SWIM 7, Fig. 3) as truth; (2) sensitivity of the unit salinities to the petrophysical relation; (3) K0 → K1 correction and the 1980-vs-2018 comparison figure. All additive: no v1.0 result, figure or number changes. New scripts `19_lebbe1980_synthetic.py`, `20_petrophysics_sensitivity.py`, `21_k1_comparison.py`; outputs in `data/processed/v1_1/`, `results/v1_1/`; tag `v1.1-sensitivity`. |

## Pre-registered expectations (written before any v1.1 computation)

| id | expectation / decision rule |
|---|---|
| **E1** | *Supports draft Section 4.3.* For T80a, the median over seeds of the BEL posterior median of ρ_below lies in [2.6, 12.5] Ω·m (the field 90 % interval). |
| **E2** | *Contradicts draft Section 4.3.* For T80a, the median over seeds of the BEL posterior median of ρ_below is ≥ 15 Ω·m. Then Section 4.3 must be rewritten: the pipeline can see fresh water under the lens, and the field result would point to brackish water. |
| E1/E2 | Between 12.5 and 15 Ω·m: **inconclusive**; reported as such. |
| **E3** | For T80a, the BEL lens base at x = 150–350 m falls between the true saline base and the true fresh top (inside the transition), rather than at either boundary. |
| **E4** | For T80b versus T80a, the size of the ρ_below change tells how much of the field value comes from the transition layer rather than from resolution alone. No threshold; reported. |
| **E5** | In the petrophysics sensitivity, the spread of unit medians across P1–P5 exceeds the width of the P0 90 % interval for at least one unit. |
| check | P0 (the v1.0 petrophysics) must reproduce v1.0 numbers 101–104 (`unit_lens`, `unit_below`, `unit_dis`, `unit_dune`) exactly; otherwise stop and report. |

## Follow-up entries

| date | id | entry |
|---|---|---|
| 2026-09-30 | v1.1-1 | Implementation details fixed while writing the scripts, before their results were seen: linearised posterior samples of the synthetic inversions use `default_rng([134050, truth, seed, λ])`; `11_stage4_closeout.a2_job` got keyword arguments (truth data, noise function, output path, λ list, posterior seed, saving the λ = 50 model and the noisy data) whose defaults reproduce v1.0 — checked by re-running the v1.0 job T1 r00 (`results/v1_1/1_lebbe1980_synthetic.md`). Petrophysics P5: EC25 by linear mixing of the two end-member EC25 values, EC25_e = 10⁴/ρ_e/(1 + 0.02(T − 25)) with T ~ U(8, 14) °C, seed `default_rng([135000, 5])` (reduces to the v1.0 seawater-only mixing when EC25_fresh → 0). The P0 reproduction check passed (accepted draws identical to `field_v2.npz`; numbers 101–104 identical). |

## Analysis choices fixed before running (Task 1)

- **Truth geometry.** Boundaries of `data/external/lebbe1981_K1_digitised.csv` interpolated linearly in x between the
  control points and held constant beyond the end points; our topography (the 102 land electrodes). Where the CSV has no
  lower brackish layer (x < 260 m) its top is set to the clay top (zero thickness); where the fresh zone is absent
  (x = 470 m) the fresh top and the lower-brackish top are set to the saline base (brackish from the saline base to the
  clay). Resistivity at each fine-mesh cell centre (as in the Stage-4 benchmark). The clay (deep-unit) resistivity is the
  v1.0 reference-model deep-zone median in `data/processed/synthetic/truth_spec.json`.
- **Forward and noise.** Fine forward mesh of the Stage-4 benchmark (`10_synthetic_benchmark.FWD`), exactly the 1974 G1
  quadrupoles; noise as in v1.0: σ = √((0.03 R)² + (1e-4 Ω)²), Gaussian, seeds `default_rng([134000, truth, seed])`,
  5 seeds per truth.
- **Heavy-tailed run (T80a only, 5 seeds).** In addition to the Gaussian noise, a residual term err_i · r*_i, where r*_i
  is drawn with replacement (seed `default_rng([134500, seed])`) from the standardised residuals of the reference
  inversion g1_lam50 of the data that share datum i's P1 electrode (keeps the electrode-specific tails).
- **Pipeline.** Unchanged v1.0 functions: `11_stage4_closeout.a2_job` (inversions at λ = 50 and the λ pool 5–100,
  interface definitions, linearised posterior bands) with added arguments whose defaults reproduce v1.0; BEL training
  set = prior v3 site + physical (`stage6_analysis.load_combined(physical=True)`); base-only BEL (lens base, thickness),
  second BEL (ρ_lens, ρ_below, ρ_discharge; transform bounds 0.4–3.0 / 2–20 / 2.5–150 Ω·m as v1.0), joint BEL + Archie
  v2 with per-draw rejection (salinity). Posterior seeds `default_rng([134100 + k, truth, seed])`.
- **"True" ρ_below.** Thickness-weighted mean of log10 ρ of the true model between the saline base and the clay top,
  averaged over x = 150, 200, 250, 300, 350 m; the "true" % seawater applies the v1.0 salinity step (Archie v2, same
  draw distribution) to that value.
- **Coverage.** Share of (x, seed) pairs at x = 150–350 m in which the BEL 90 % lens-base interval contains the true
  saline base / the true fresh top.
- **E1/E2 statistic.** For each seed the second-BEL posterior median of ρ_below; the median of these over the 5 seeds.
