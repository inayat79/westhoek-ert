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

## v1.3 follow-up entries

| date | id | entry |
|---|---|---|
| 2026-10-02 | v1.3-1 | **Task 1–2 outcome (pre-registered rules applied as written).** E9 **holds** (35/35 cases); E10 **does not hold** (1 of 15 sharp truths with r ≤ 10 m data); E11 **does not hold** (log-midpoint and max. gradient 17/22 each = 77 %; pre-registered breakdowns: 14/16 each with 3 Ω·m inside the contrast, 22/22 each with the oracle anchor); E12 **holds** (R² = 0.69, a = 0.621 [0.515, 0.747]). *Implementation details fixed while writing the report, before reading the verdicts:* E10 and E11 count the truths with at least one r ≤ 10 m pair (10 of 32 have none: all nine sharp z_b = −18 truths, G_0.8_16_-12_0 and G_0.8_16_-12_5); the 10–15 m group has data for all 32. *Run:* 416 inversions, no failures; wall-clock 6.3 h on 6 workers against the 2.7 h projected in `1_plan.md` (the assumed 1.5× load factor was too low; still far below the 36 h limit). λ = 50 χ² 1.02–1.43; 22 of 416 inversions end with χ² > 1.6 (20 at λ = 5, mostly ρ_lens = 0.8, stopped on dPhi < 2 %), kept as in v1.0 (they enter only the λ-pooled treatment). *Observation after the results and post hoc diagnostic (not pre-registered):* the oracle bias is always deep and grows with k (−0.7 … −4.4 m for k ≥ 4), and the rule residuals grow more negative with k. The λ = 50 models recover the lens resistivity but a damped ρ_below (recovered/true log-contrast 0.57–0.90 for k ≥ 4). Refitting the rule with q from the recovered plateaus (median log10ρ of m̂ from 4 m above z_b to the surface and 4–10 m below z_b; bootstrap seed `default_rng([136202])`) gives R² = 0.87 (a = 0.366). This diagnostic does not change any verdict. The gradual truths did not behave as a wider s (a = 0.591 vs 0.621; a 5 m ramp widens s by only ≈ 4 %, below what the design can detect). |

# v1.4 — raw ABEM file and direct-push ground truth (branch `v1.4` from `v1.3-general`)

| date | id | entry |
|---|---|---|
| 2026-10-03 | v1.4-0 | **Scope and state at pre-registration.** Task 0 inputs; Task 1 raw ABEM (deferred: the 2018 export has not arrived; the file supplied on 2026-10-03 was byte-identical to the Zenodo `ERT.dat`, i.e. the recent line, not ours); Task 2 DP logs vs the 2018 results; Task 3 outputs. New scripts `26_raw_abem.py` (Task 1, later) and `27_dp_groundtruth.py`; outputs `data/processed/v1_4/`, `results/v1_4/`; tag `v1.4-groundtruth` only after Task 1 is done. Zenodo record 10.5281/zenodo.20792093 (CC-BY-4.0) downloaded 2026-10-03, MD5 verified. **No DP resistivity value had been opened when these entries were written and committed**: only the x and z columns of `DP_R.xzr`, the coordinate columns of `SGS_DP_R.txt` (geostatistical simulations conditioned on the DP data, not measurements) and the sensor block and headers of `ERT.dat` were read. |

## v1.4 pre-registered expectations (written and committed before any DP resistivity value was read)

| id | expectation / decision rule |
|---|---|
| **E13** | The median stacking error is below 3 %, i.e. stacking errors understate the total data error. |
| **E14** | If the order is available: the outlier rate is higher for data whose potential electrode was used as a current electrode shortly before than for the rest. "Shortly before" means the lowest quartile of time or count since last use. |
| **E15** | The DP saline base (log-midpoint) lies inside the 2018 BEL 90 % interval for at least two thirds of the DPs at x = 100–450 m. |
| **E16** | At most DPs within x = 150–350 m, ρ_below,DP is below 12.45 Ω·m (the field BEL P95), and no DP in that range shows a zone above 20 Ω·m thicker than 3 m. If it holds, this supports Section 4.3 (brackish water below the lens); if most DPs show > 20 Ω·m below the lens, Section 4.3 must be revised. |
| **E17** | Where the 2018 log-midpoint and 3 Ω·m bases differ by more than 2 m, the DP base is closer to the log-midpoint at most of those DPs. |

## v1.4 analysis choices fixed before reading any DP value

- **Positions.** DP and recent-ERT-line positions: WGS84 points georeferenced by the user from T. Hermans' map (±10 m).
  The DP files carry local x = 50, 100, …, 350 m; matching against the map distances from the land end (DP7 10 … DP1
  310 m; line 360 m) gives local x = 360 m − distance within ±1 m, so **local x = 0 is the sea end** and DP1 is at local
  x = 50 m. WGS84 → Belgian Lambert 72 with pyproj, operation "BD72 to WGS 84 (3)" (EPSG:15929, 1 m). Each point is
  projected onto the polyline of our 102 land electrodes (GPS_coordinate.xlsx); our x = profile x (elec.dat)
  interpolated along the polyline; perpendicular offset reported with + = north-east (away from the border).
  Sensitivity: every comparison repeated with all DPs shifted by −15 m and +15 m along our line (verdicts reported for
  each; the unshifted positions give the primary verdict).
- **Survey date** of the DP logs and of the recent ERT line: not in the record; "pending (asked T. Hermans)". A
  placeholder (`DP_DATE` in `27_dp_groundtruth.py`) fills the time gap since 7 March 2018 in every report line and
  number once known.
- **Elevation.** DP z = depth below the DP surface (negative down). The record has no DP surface elevations: the 2018
  land surface (our electrodes) at the DP's x is used, flagged (the beach surface may have changed).
- **DP log processing.** r taken as bulk resistivity in Ω·m (units and temperature reference not stated in the record;
  the authors' script treats r as resistivity next to ρa). log10 r; running median over 0.5 m (51 samples at 1 cm,
  centred, edges by nearest value); then sampled at 0.2 m elevation steps (the column step of the paper's interface
  code). Interfaces on that column: 3 Ω·m and 2.5 Ω·m = main-lens base crossing (`westhoek_common.main_lens`, levels
  log10 3 and log10 2.5); log-midpoint as Section 2.4 (`11_stage4_closeout.interfaces_all`: gradient-weighted centroid
  of the segments ≥ 90 % of the steepest increase; plateaus = median log10 ρ in 3 m windows above and below; crossing of
  their mean nearest the steepest increase within ±10 m) but **anchored on the steepest increase of the DP log between
  its top and −25 mTAW** (above the clay; 1980 clay top −25.5 to −28 mTAW) instead of on the 3 Ω·m crossing. Fresh zones:
  contiguous runs of the smoothed log ≥ 20 Ω·m (top, bottom, thickness), all reported; E16 uses the thickest run below
  the DP log-midpoint base. ρ_below,DP = median of the unsmoothed r from 2 m below the DP log-midpoint base down to the
  bottom of the log or −25 mTAW, whichever is shallower (undefined if that interval is empty).
- **Inclusion.** A DP enters the comparisons if the resolution radius of the reference model at the λ = 50
  log-midpoint crossing at its x (`r_logmid` of `results/runs/g1_lam50_v4`, nearest column) is ≤ 15 m; NaN = excluded.
- **2018 comparators at the DP's x** (nearest column): BEL lens base (v1.0 headline: base-only BEL, site + physical,
  `default_rng([129000, 1])`, as F_S3) median and 90 % interval; λ-pooled log-midpoint and λ = 50 3 Ω·m bands (v4);
  1980 saline base (`lebbe1981_K1_digitised.csv`, linear in x); BEL ρ_below posterior 4.93 [2.59, 12.45] Ω·m (a
  profile-wide value); the g1_lam50 column.
- **Verdict rules.** E15: among included DPs at x = 100–450 m, the share with P5 ≤ DP log-midpoint base ≤ P95 of the BEL
  interval ≥ 2/3. E16: among included DPs at x = 150–350 m, more than half with ρ_below,DP < 12.45 Ω·m, and none with a
  ≥ 20 Ω·m run thicker than 3 m below its base. E17: among included DPs with |λ-pooled log-midpoint median − λ = 50
  3 Ω·m median| > 2 m, more than half with the DP base closer to the log-midpoint median; untestable if there is none.
  DPs without a log-midpoint base count as failures for E15 and are excluded from E16's ρ_below count and from E17.
- **Salinity.** % seawater = 100·F·ρ_sw/ρ_below,DP with the v1.0 Archie v2 draws (`17_salinity_v2.archie_v2`: F ~ U(2.8,
  3.6), S ~ U(31, 34) PSU, T ~ U(8, 14) °C, PSS-78), 20 000 draws, `default_rng([137200, dp_index])`, median [P5, P95];
  draws above 100 % reported, not clipped.

## v1.4 follow-up entries

| date | id | entry |
|---|---|---|
| 2026-10-05 | v1.4-1 | **Disclosure (written before any entry below).** While the files were being read, the user and the assistant saw some numbers before these entries were written. Raw file (`project.db`): the median stacking error on the published data (0.13 %), a median reciprocal error of about 2.4 % on the landward spread, the selection rule of the published data, and the instrument GPS positions. Zenodo files: the median ρ in the top 2 m of each DP log, the depth at which each DP log first exceeds 3 Ω·m, and a rough pseudo-section correlation against our data. None of these was compared with the 2018 interface results. Anything computed from these numbers is descriptive, not a test. (The assistant of this session had itself read only the x and z columns of `DP_R.xzr`, as recorded in v1.4-0; the numbers above were seen in the user's own inspection.) |
| 2026-10-05 | v1.4-2 | **Plan changes since v1.4-0 (the v1.4-0 entries stay as written).** (1) DP positions: the map-based positions of v1.4-0 are superseded. The Zenodo line is placed on our profile by fitting its ERT data (Task A, pre-registered below): our_x = X0 − x_local, with x_local = 0 at the sea end. The BD72 reading of T. Hermans' map (X0 ≈ 407 m) is rejected. (2) Our line in Lambert 72: `GPS_coordinate.xlsx` has coordinates for land electrodes 1 and 102 only, so the line is the chord between them (as in F1), with profile x interpolated along it, not "the polyline of the 102 electrodes" written in v1.4-0. (3) The DP comparisons (E15–E17) and the inversion of the Zenodo ERT line (E18–E20) wait (Tasks C and D) for the survey dates and DP ground elevations from T. Hermans, or for the user's go-ahead. No DP resistivity value is read in this round. (4) Tags: `v1.4a-raw` after this round; `v1.4-groundtruth` after Tasks C and D. |
| 2026-10-05 | v1.4-3 | **Task 1 (raw 2018 file) — pre-registration, written before steps (b)–(e).** **E13** and **E14** as pre-registered in v1.4 (unchanged): E13 — the median stacking error is below 3 %, i.e. stacking errors understate the total data error; E14 — if the order is available, the outlier rate is higher for data whose potential electrode was used as a current electrode shortly before than for the rest ("shortly before" = the lowest quartile of time or count since last use). *Definitions:* stacking error = DataSDev / \|R\| (DPV DatatypeID 5, Channel > 0); reciprocal error = \|R_n − R_r\| / (\|R_n + R_r\| / 2). *Reciprocal pairs:* task 2 datum (A₂, B₂, M₂, N₂) ↔ the task-1 station-3 datum with A = M₂ + 64, B = N₂ + 64, M = A₂ + 64, N = B₂ + 64 (an order swap within a dipole flips the sign of R and is allowed); "pairs on published electrodes" = all four positions p ≤ 101. *E13 statistic:* median stacking error of the 2331 published data (primary) and of all task-1 data; holds if < 3 %. *E14:* published data; outlier = one of the 357 data removed by the v1.0 G1 rule; for each datum, over its two potential electrodes (physical p, across stations), the time (s, `Measures.Time`) and the number of current injections (`Measures` rows) since that electrode last served as A or B in an earlier injection, taking the smaller of the two (most recent use); never used before = +∞ (upper group). Lowest quartile = values ≤ the 25th percentile. Primary quantity: time; secondary: count. E14 holds if the outlier rate in the lowest time quartile is higher than in the rest; odds ratio with a 90 % bootstrap interval (2000 resamples of the published data, `default_rng([138000])`), reported for both quantities. *Error-model comparison (descriptive, not a test):* fitted to the reciprocal differences d = R_n − R_r of all pairs, d ~ N(0, 2σ²), maximum likelihood; model A σ = a + b·\|R̄\|; model B σ = √((c·\|R̄\|)² + f²) (relative + floor, the v1.0 form); the better by AIC is used; the log-likelihood of the v1.0 model (c = 0.03, f = 1e-4 Ω) on the same pairs is reported; the fit is repeated on the pairs on published electrodes. *(d)* Mann–Whitney U (two-sided) of the stacking (and, where available, reciprocal) errors of the 357 outliers vs the 1974 retained data; effect size = rank-biserial correlation 1 − 2U/(n₁n₂). *(f)* g1_lam50 re-run with the per-datum relative error σ(\|R\|)/\|R\| of the chosen model, nothing else changed (descriptive sensitivity; v1.0 numbers unchanged). |
| 2026-10-06 | v1.4-4 | **Task A — line placement, pre-registered before the scan was run.** *Primary:* the 72 electrodes of the Zenodo `ERT.dat` (local x = 5(i − 1) m, x = 0 at the sea end) are placed at our_x = X0 − 5(i − 1) on our 2018 topography (elec.dat, linear; constant beyond the ends); data with i ≤ 0, u ≤ 0 or u/i ≤ 0 removed (3 expected). Their quadrupoles are forward-modelled through the cached g1_lam50 model: pyGIMLi `ert.simulate` on a mesh built for the 72 electrodes (`createParaMesh`, paraDX 0.33, paraDepth 60, as the inversion mesh), cell resistivities = the g1_lam50 value of the nearest g1 cell centre. Misfit: RMS of log10 R_pred − log10 R_obs (R_obs = u/i; identical to the RMS of log10 ρa with a common K). Scan X0 = 380–520 m in 2.5 m steps; check orientation our_x = X0′ + 5(i − 1), X0′ = −15–125 m in 2.5 m steps. X0* = argmin (primary orientation); σ_X = half-width of the contiguous interval around X0* where RMS ≤ 1.05 × minimum, floor 10 m. *Secondary:* pseudo-section correlation. Both data sets (ours = the 1974 G1 data; theirs = the positive data of `ERT.dat`): midpoint = mean x of the four electrodes, pseudo-depth = 0.19 × the span of the four electrodes, ρa with the analytical flat-ground geometric factor, log10 ρa; bins of 5 m in x and 2.5 m in pseudo-depth; each pseudo-depth level centred (its mean over the bins both data sets fill removed); Pearson correlation over the shared bins; same X0 grids; X0_pc = argmax. If \|X0_pc − X0*\| > 2σ_X: stop and report. *DP positions:* our_x(DPk) = X0* − x_local(DPk); DP surface = the 2018 surface (elec.dat) at that x, flagged "2018 surface; DP-date surface pending", ±0.5 m as sensitivity. The survey dates differ (2018 vs a later, pending date), so the lens may have changed between the surveys; the placement assumes the resistivity structure is similar enough to locate the line. *Datum check:* instrument GPS (`Measures` lat/lon, median per station) converted with pyproj EPSG:4326 → EPSG:31370 using "BD72 to WGS 84 (3)" and, as a check, without the datum transformation (lat/lon read as BD72, EPSG:4313). |
| 2026-10-06 | v1.4-5 | **Task A secondary method — correction before any result was seen.** The pre-registered pseudo-section binning (5 m × 2.5 m bins, v1.4-4) yields **no shared bin** between the two data sets: in a multiple-gradient array the pseudo-depth is fixed by the current-electrode span, and the spans differ (ours multiples of 50 m, theirs of 55 m), so the depth levels never coincide. The scan crashed on this; neither its correlations nor the primary RMS scan had been printed or saved. Change (secondary method only): each pseudo-section (same midpoint and pseudo-depth rule, log10 ρa, flat-ground K) is interpolated linearly (Delaunay, `scipy.interpolate.griddata`) onto a common grid, x every 5 m and pseudo-depth every 2.5 m, inside each data set's own convex hull; each pseudo-depth row is centred over the nodes both data sets cover; Pearson correlation over those nodes (≥ 10 nodes). Everything else in v1.4-4 (primary method, X0*, σ_X, stop rule) is unchanged. |
| 2026-10-06 | v1.4-6 | **Task B — pre-registration of the DP tests (written before any DP resistivity value is read; not run in this round).** **E15, E16, E17** as pre-registered in v1.4 (unchanged): E15 — the DP saline base (log-midpoint) lies inside the 2018 BEL 90 % interval for at least two thirds of the DPs at x = 100–450 m; E16 — at most DPs within x = 150–350 m, ρ_below,DP is below 12.45 Ω·m, and no DP in that range shows a zone above 20 Ω·m thicker than 3 m (support for Section 4.3 if it holds; revision of 4.3 if most DPs show > 20 Ω·m below the lens); E17 — where the 2018 log-midpoint and 3 Ω·m bases differ by more than 2 m, the DP base is closer to the log-midpoint at most of those DPs. Their verdict rules, inclusion (r ≤ 15 m of g1_lam50 at the λ = 50 log-midpoint crossing) and comparators are as in v1.4-0; the DP positions are now our_x = X0* − x_local with X0* = 502.5 m, σ_X = 10 m (Task A). **Robustness rule:** every expectation (E15–E20) is evaluated in 9 cases: X0* − σ_X, X0*, X0* + σ_X, each with the DP surface at the 2018 surface − 0.5 m, ± 0 and + 0.5 m. It is **held** only if it holds in all 9, **not held** only if it fails in all 9, and **position-dependent** otherwise. **DP definitions** (each log smoothed with a 0.5 m running median, 51 samples at 1 cm, centred, nearest-value edges; then sampled at 0.2 m): 3 Ω·m and 2.5 Ω·m crossings = main-lens base (`westhoek_common.main_lens`); log-midpoint = Section 2.4 (gradient-weighted centroid of the segments ≥ 90 % of the steepest increase; plateaus = median log10 ρ in 3 m windows above and below; crossing of their mean nearest the steepest increase within ±10 m), anchored on the steepest increase between the top of the log and −25 mTAW; fresh zone = contiguous runs ≥ 20 Ω·m (thickest run below the base for E16); ρ_below,DP = median raw ρ from 2 m below the DP log-midpoint base to the bottom of the log or −25 mTAW, whichever is shallower; ρ_lens,DP = median raw ρ from the top of the log to 2 m above that base; q_DP = (log 3 − log ρ_lens,DP) / (log ρ_below,DP − log ρ_lens,DP) (DPs with q_DP ∉ (0, 1) are excluded from E18). **Same-epoch test (Task C):** invert the positive data of `ERT.dat` with our pipeline: Stage-1 G1 outlier rule (all data at λ = 20, maxIter 30, drop \|r\| > 3·1.4826·MAD, re-invert the kept data), v1.0 error model (3 % + 1e-4 Ω), λ = 5, 10, 20, 50, 100 with λ pooling, linearised posterior (1000 samples per λ, `default_rng([139000, λ])`), paraDepth 60 m, paraDX 0.33, the electrodes on our 2018 topography at X0* (check: flat topography); interface definitions of v1.0 (3 Ω·m, max-gradient, log-midpoint anchored on 3 Ω·m) plus the log-midpoint anchored on the steepest increase (secondary); columns at each DP's x; r = the resolution radius of the λ = 50 image at its λ = 50 log-midpoint crossing at that x. **E18.** At DPs with r ≤ 10 m, the image's 3 Ω·m crossing lies on the side of the DP's 3 Ω·m crossing that the v1.3 rule predicts (deeper if q_DP > 0.5), at ≥ 2/3 of those DPs. **E19.** At DPs with r ≤ 10 m, the λ-pooled 90 % log-midpoint band contains the DP log-midpoint base at more DPs than the λ = 50 90 % 3 Ω·m band contains the DP 3 Ω·m crossing. **E20.** The ERT.dat image's median ρ 4–10 m below its log-midpoint base (λ = 50) is lower than ρ_below,DP at ≥ 2/3 of DPs with r ≤ 10 m. If fewer than 3 DPs have r ≤ 10 m, E18–E20 are "not testable", and the 10–15 m results are reported as descriptive only. Salinity from ρ_below,DP (Task C): v1.0 Archie settings with the DP temperature reference unknown, T widened by ±3 °C. |
