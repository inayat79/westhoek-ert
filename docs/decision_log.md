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

| 2026-09-30 | v1.1-2 | **Erratum — the line is K1.** Electrode coordinates place it 1.08–1.11 km north-east of the border along the coast; Paepen et al. (2020, Section 3.1.1) date the K1 land profile to 7 March 2018. No file named our line K0 (every "K0" refers to the border profile or its well; `results/v1_1/3_k1_rename.md`). Survey date recorded in `results/REFERENCE.md` (to be confirmed by the authors); duration not reported. The v1.0 "site" filter (clay top −38 … −22 mTAW, spanning K1 and K0) is unchanged; narrowing it to K1 would be an analysis change. |

| 2026-09-30 | v1.1-3 | **Task 1 outcome (pre-registered rules applied as written).** T80a: median over seeds of the second-BEL ρ_below posterior median = 15.13 Ω·m → **E1 does not hold, E2 holds** (margin 0.13 Ω·m; seeds 14.72–15.25, one of five below 15). **E3 does not hold** (BEL median inside the transition at 3 of 5 positions; the 90 % intervals cover both true boundaries at all positions). **E4:** T80b − T80a = +0.67 Ω·m. *Observations after the results, not pre-registered:* the recovered ρ_below saturates below the 20 Ω·m prior ceiling (13.6–16.3 Ω·m for all 30 data sets although the true log-means are 17–38 Ω·m), so the synthetic test cannot say how fresh the water is, but every synthetic median lies above the field P95 (12.45 Ω·m); the linearised log-midpoint base is shallow for this structure (median +3.1 m). Equivalence of the modified `a2_job`: exact with a stubbed inversion (`scripts/check_a2_job_equivalence.py`); a real re-run of the v1.0 job T1 r00 differs from the stored run by ≤ 2.6e-5 m in the λ = 50 lens bases (pyGIMLi floating-point nondeterminism, as in v1.0). No threshold or analysis choice was changed after the results. |

| 2026-09-30 | v1.1-4 | **v1.1 finished** (tag `v1.1-sensitivity`): 26 numbers appended to `results/paper/numbers.md` with `v11_` keys (the 191 earlier entries byte-identical; v1.0.3 baseline in `results/paper/baseline_v1.0.3/`); clean-folder `--from-cache` rebuild PASS (217/217 numbers, 114/114 cross-checks incl. the v1.1 reports, 18/18 regenerated reports; provenance audit 0 files without a producer). Summary: `results/v1_1/SUMMARY.md`. The draft manuscript is not in the repository, so its Sections 4.2, 4.3 and 2.7 could not be quoted. |

| 2026-09-30 | v1.1.1-0 | **Post hoc robustness checks (added after the v1.1 results; not pre-registered; they do not change the E1–E4 verdicts, which are defined on T80a / T80b).** T80f = T80a with the clay at 4 Ω·m (instead of the v1.0 deep-zone median 8.44 Ω·m). T80g = T80a with the fresh top 5 m deeper and the brackish transition (6 Ω·m) extended from the saline base down to the deeper fresh top; where the deeper fresh top falls below the lower-brackish top the fresh zone vanishes, and it never goes below the clay top. Five Gaussian seeds each; truth indices 6 (T80f) and 7 (T80g) in the v1.1 seed scheme (noise `default_rng([134000, truth, seed])`, posterior seeds as v1.1); unchanged pipeline. Reported like the other variants, with ρ_below compared with the field 90 % interval [2.59, 12.45] Ω·m. Housekeeping in the same release (v1.1.1): `19_lebbe1980_synthetic.py` renamed `22_lebbe1980_synthetic.py` (the number 19 was already used by `19_rho_lens_bound_check.py`); F_S1 panels relabelled (a)–(c); F_S3 revised (log-midpoint masked where r > 10 m, discharge zone x = 380–480 m shaded, legend grouped by source, untitled version) with numbers for x = 400 and 450 m. |

| 2026-09-30 | v1.1.1-1 | **Outcome of the post hoc checks (v1.1.1-0).** T80f (clay 4 Ω·m): ρ_below median of medians 15.63 Ω·m (seeds 15.45–15.89), above the field 90 % interval, as for T80a. T80g (fresh top 5 m deeper): 12.46 Ω·m (seeds 12.01–12.80; true log-mean 12.64), at the upper edge of the field interval [2.59, 12.45] (2 of 5 seeds inside); fresh-top coverage 12 %. Over all 40 data sets, 38 per-seed medians lie above the field P95. The E1–E4 verdicts (v1.1-3) stand as recorded. Existing Task 1 cache arrays byte-identical after adding the variants (271 BEL + 13 forward arrays). K1 comparison at x = 400 / 450 m: the 1980 saline base lies below the BEL 90 % interval at both (BEL − 1980 +12.17 / +7.77 m; log-midpoint − 1980 +8.40 / +12.72 m). numbers.md: the 217 earlier numbers keep keys, order and values; only the "computed by" script name of the 14 v11 Task 1 rows changed (19_ → 22_, rename); 6 `v111_` numbers appended. |

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

# v1.3 — does the fixed-contour failure generalise? (branch `v1.3` from `v1.1.1`)

| date | id | entry |
|---|---|---|
| 2026-10-01 | v1.3-0 | **Scope of v1.3.** Task 0: field like-for-like numbers (λ-pooled 3 Ω·m base and λ = 50 log-midpoint base at x = 150–350 m). Task 1: factorial synthetic study (32 truths, 13 inversions each). Task 2: test of the offset rule Δz = s·Φ⁻¹(q). New script `25_generalisation.py` (run / resume / report); outputs in `data/processed/v1_3/`, `results/v1_3/`; tag `v1.3-general`. All additive. No v1.2 branch exists in this repository (nothing to keep apart). |

## v1.3 pre-registered expectations (written and committed before any v1.3 inversion)

| id | expectation / decision rule |
|---|---|
| **E9** | For the fixed contours (2, 3, 5 Ω·m), the sign of the median bias matches the sign predicted by q in at least 90 % of the (truth × threshold) cases with r ≤ 10 m and \|q − 0.5\| > 0.1. |
| **E10** | The oracle contour (q = 0.5) has \|bias\| ≤ 0.5 m and coverage ≥ 80 % in at least 80 % of sharp truths (r ≤ 10 m). |
| **E11** | The log-midpoint and maximum gradient reach coverage ≥ 85 % (r ≤ 10 m) in at least 80 % of all truths. |
| **E12** | The one-constant fit explains at least half of the variance of the fixed-contour biases (R² ≥ 0.5). |

## v1.3 analysis choices fixed before running

- **Truths.** Name `G_<ρ_lens>_<k>_<z_b>_<τ>` (e.g. `G_1.42_4_-12_0`); order (= truth index ti): ρ_lens (0.8, 1.42) ×
  k (2, 4, 8, 16) × z_b (−8, −12, −18) with τ = 0, then the eight τ = 5 truths (ρ_lens × k at z_b = −12). x ≥ 90 m:
  lens ρ_lens from the surface down to the flat base z_b along the whole lens part (no discharge zone); ρ_below = k·ρ_lens
  from z_b down to −30 mTAW; deep unit 8.44 Ω·m (v1.0 T1 deep median) below −30. τ = 5: log-linear (in log ρ)
  transition of 5 m centred on z_b (as T3). x < 90 m: the v1.0 T1 reference values (cap 211.7 Ω·m above z = 3, fresh
  13.45 Ω·m down to −30, deep below). Truth at the fine forward-mesh cell centres (17 684 cells), exactly the G1
  quadrupoles, as Stage 4.
- **Noise and seeds.** σ = √((0.03 R)² + (1e-4 Ω)²), Gaussian, `default_rng([136000, ti, real])`, realisations 0–4;
  non-positive noisy R dropped (count recorded), as v1.0. Inversion with the unchanged `10_synthetic_benchmark.one_inversion`
  (λ = 50 for realisations 0–4; also λ = 5, 10, 20, 100 for realisations 0–1); 1000 linearised-posterior samples per
  inversion, `default_rng([136100, ti, real, λ])`. Bootstrap of the rule fit `default_rng([136200])` (λ = 50) and
  `default_rng([136201])` (per-λ fits), 2000 resamples of truths.
- **Interface definitions** (to m̂ and to every sample, per column x = 100–450 m every 10 m): `thr2`, `thr3`, `thr5` =
  main-lens base (`westhoek_common.main_lens`) at 2, 3, 5 Ω·m; `oracle` = the same at √(ρ_lens·ρ_below); `maxgrad`,
  `logmid` = `11_stage4_closeout.interfaces_all` unchanged (anchored, as in v1.0, on the 3 Ω·m main-lens crossing).
  **Diagnostic, not used for E9–E12:** `maxgrad_o`, `logmid_o` = the same function anchored on the oracle crossing
  (implemented by adding log10 3 − log10 √(ρ_lens·ρ_below) to log10 ρ, which moves no gradient and no crossing). Known
  before running: where ρ_below ≤ 3 Ω·m (k = 2, both ρ_lens: 8 of 32 truths) the v1.0 anchor falls on the deep unit or
  is absent, so `maxgrad`/`logmid` are expected to fail there by construction. E11 is evaluated as written over all
  truths; the result is also reported for the truths with 3 Ω·m inside the contrast and for the oracle-anchored
  diagnostic. No existing function is changed (the thresholds use the existing `level` argument of `main_lens`).
- **True base** for every definition = z_b (the centre of the transition for τ = 5), not the definition applied to the
  true model (v1.0 A2 used the latter; identical for sharp truths with the threshold inside the contrast).
- **Groups.** By the resolution radius of the λ = 50 inversion of the same realisation at the true base z_b (the same
  group for every definition and λ treatment; v1.0 grouped by r at each definition's own crossing): r ≤ 10 m and
  10 < r ≤ 15 m; columns with r > 15 m are excluded.
- **Metrics** per truth × definition × treatment × group, over (column, realisation) pairs. Coverage: the band is present
  in ≥ 50 % of samples and P5 ≤ z_b ≤ P95 (absent band = not covered). Bias: median of m̂ − z_b (negative = too deep;
  missing m̂ excluded and counted). λ-pooled treatment: realisations 0–1, the 5 × 1000 samples concatenated; its bias is
  the median of the pooled-band medians minus z_b. Scatter ratio (λ = 50): per column, s.d. over the 5 realisations of
  m̂ divided by the median over realisations of the predicted σ_z = (P95 − P5)/(2 × 1.6449) of the sample band; median
  over the columns whose median r over realisations is in the group. r̄ = mean r at the true base over the group's pairs.
- **q and the rule.** q = (log ρ_t − log ρ_lens)/(log ρ_below − log ρ_lens). Cases with q ≤ 0 or q ≥ 1 (threshold
  outside the lens–below contrast; the crossing falls on the deep unit or is absent) have no predicted offset: they are
  excluded from E9, E12 and the fits and reported separately. E9 covers all 32 truths; a case needs ≥ 5 (column,
  realisation) pairs with r ≤ 10 m. Fit (E12): sharp truths × {2, 3, 5} with 0 < q < 1, λ = 50, r ≤ 10 m: bias = −a·r̄·Φ⁻¹(q),
  least squares through the origin; R² = 1 − Σresidual² / Σ(bias − mean bias)². Per-λ fit: realisations 0–1, the
  bias at each λ, the same r̄ (λ = 50), one a per λ. Gradual truths (τ = 5) fitted separately.
- **Verdict treatment.** E9–E12 use λ = 50, r ≤ 10 m (as F_10 and the Task 2 fit); the λ-pooled results are reported
  alongside.
- **Run budget.** Five inversions are timed first; if the projected total exceeds ≈ 36 h, stop and propose a reduced
  design before running.
- **Saved per inversion:** the summaries of the v1.0 synthetic runs (model, radius, χ², m̂ and band per definition, r at
  each crossing), plus the per-sample interface depths per definition (float32). The full posterior samples are not
  saved (disk: ≈ 0.6 GB each set).
