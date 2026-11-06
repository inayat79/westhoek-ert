# Scripts

Run from this folder inside the `westhoek-ert` environment (`conda activate westhoek-ert`; some docstrings still say
`conda run -n pg`, the environment's earlier name). Every script derives its paths from its own location, so outputs land
in `data/processed/` and `results/` next to `scripts/`. `run_all.py --from-raw` runs the whole pipeline in order; each
script's docstring lists its subcommands. Scripts are not renamed or moved, so the decision log and the commit history
refer to them by these names.

**P** = part of the final pipeline (run by `run_all.py` or needed for a reported number or figure). **H** = historical
or one-off (kept because the decision log refers to it).

| script | | stage | purpose | main inputs → outputs |
|---|---|---|---|---|
| `run_all.py` | P | all | one-command rebuild: `--from-raw`, `--from-cache`, `--check`, `--audit`, `--check-inputs`, `--fetch-natural-earth` | everything → `data/processed/`, `results/` |
| `westhoek_common.py` | P | all | shared helpers: run loading, column sampling, linearized covariance (`cholesky_A`), restricted-input guard | — |
| `read_trace.py` | P | all | records every file a script reads (provenance audit of `run_all.py`) | — |
| `01_data_exploration.py` | P | v1.0 | first look at the 2018 data | `data/raw/` → figures |
| `02_land_inversion.py` | P | v1.0 | pyGIMLi inversion of the 2018 land line, resolution matrix and Jacobian | `data/raw/land_data.dat`, `elec.dat` → `data/processed/runs/<tag>/` |
| `03_resolution_analysis.py` | P | v1.0 | resolution and linearized uncertainty of a run | run → resolution products |
| `04_misfit_diagnostics.py` | P | v1.0 | data misfit diagnostics | run → misfit report |
| `05_outlier_characterisation.py` | P | v1.0 | outlier electrodes, drop lists (G1, G2, G3) | run → `results/*droplist*.txt` |
| `06_compare_runs.py` | P | v1.0 | compares models of runs on the same mesh | runs → comparison |
| `07_lambda_lcurve.py` | P | v1.0 | λ sweep and L-curve | runs → L-curve |
| `08_doi_index.py` | H | v1.0 | depth-of-investigation index (Oldenburg & Li, 1999); not used in the paper | runs → DOI |
| `09_posterior_sampling.py` | P | v1.0 | samples of the linearized posterior (Equation 2) | run → samples |
| `10_synthetic_benchmark.py` | P | v1.0 Stage 4 | site-like synthetic truths, their inversions and bands (SI Table S3) | reference run → synthetic runs |
| `11_stage4_closeout.py` | P | v1.0 Stage 4 | interface definitions, coverage of the bands, field bands (SI Table S6) | runs → bands |
| `12_bel_pilot.py` | P | v1.0 Stage 5 | BEL pilot: prior, timing, falsification | prior spec → pilot ensemble |
| `13_bel_full.py` | P | v1.0 Stage 5 | BEL full run (PCA–CCA–KDE), held-out calibration | prior ensemble → BEL posterior |
| `14_gateA_sensitivity.py` | P | v1.0 Stage 5 | sensitivity and decisions after the BEL run | BEL products → reports |
| `15_stage5_closeout.py` | P | v1.0 Stage 5 | close-out: twin, site subset, bounded second BEL, Archie | BEL products → reports |
| `16_stage6_salinity.py` | P | v1.0 Stage 6 | prior v3 ensemble, refit, joint BEL, salinity sections | prior v3 → BEL v3 products |
| `stage6_analysis.py` | P | v1.0 Stage 6 | analysis half of `16_stage6_salinity.py` | — |
| `17_salinity_v2.py` | P | v1.0 | salinity products with temperature-dependent seawater resistivity (Figure S10) | BEL v3 → salinity |
| `18_paper.py` | P | all | paper figures, tables and every quoted number | all products → `results/paper/numbers.md` |
| `19_rho_lens_bound_check.py` | P | v1.0 | check of the transform bound on ρ of the lens | BEL v3 → report |
| `20_petrophysics_sensitivity.py` | P | v1.1 | sensitivity of the salinities to the petrophysical relation | BEL v3 → report |
| `21_k1_comparison.py` | P | v1.1 | profile location (K1) and comparison with the 1980 logs (Figure S15) | `data/external/lebbe1981_K1_digitised.csv` → report |
| `22_lebbe1980_synthetic.py` | P | v1.1 | synthetic test with the 1980 K1 structure (expectations E1–E4) | → synthetic runs |
| `25_generalisation.py` | P | v1.3 | factorial synthetic test, bias rule (Equation 3; E9–E12; Figures S6, S7) | → factorial runs, rule fit |
| `26_raw_abem.py` | P | v1.4 | 2018 instrument records: stacking and reciprocal errors, E13–E14 (SI Table S2, Figure S2) | **restricted** `project.db` → reports |
| `27_dp_groundtruth.py` | P | v1.4 | Zenodo soundings and 2025 line: fetch, line placement, CPTs (`cpt`), same-epoch image (`samedate`), DP comparison (`dp`), E15–E21 | Zenodo record, **restricted** CPTs → `dp_results.json`, reports |
| `28_dp_synthetic.py` | P | v1.5 | synthetic truths built from the soundings (exploratory; Figure S13) | `cpt.json` → synthetic runs |
| `29_detectability.py` | P | v1.5 | forward test: can the data tell fresh from brackish water below the lens? | → report |
| `30_bel_v4.py` | P | v1.5 | BEL with prior v4 (exploratory; Figures 6, S14) | → BEL v4 products |
| `31_scores.py` | P | v1.6 | interval score and CRPS of every method (Equation 4; Table 2; Figure S8) | all products → `scores.json` |
| `32_location_map.py` | P | v1.7 | Figure 1 map layers: `fetch` (public map data), `build` | map data, **restricted** positions → Figure 1 |
| `33_review_v20.py` | P | v2.0 | analyses requested by the review (A1–A10; post hoc, decision log v2.0-0) | products → `data/processed/v2_0/` |
| `34_alllog_v20.py` | P | v2.0 | Table S5: every logged expectation with its commit | `docs/decision_log.md`, git history → Table S5 |
| `35_reports_v20.py` | P | v2.0 | reports and `numbers_v20.md` from the A1–A10 outputs | `data/processed/v2_0/` → `results/v2_0/` |
| `36_figs_wrr_v5.py` | P | v2.0–v3.1 | figures under the WRR numbering; CSVs of the plotted values (`--csv-only`) | products → `results/figs_wrr_v5/` |
| `paper_figures.py` | P | v1.8+ | the paper figures in the paper style (every figure function) | products → figures |
| `ert_plot_style.py` | P | v1.8+ | figure style (fonts, sizes, colours, export) | — |
| `make_paper_figures.py` | P | v1.8–v1.10 | figure export and checks (v1.9 WRR export, v1.10 CorelDRAW mode, plotted-value CSV writer) | products → figures |
| `run_v20_a4_inversions.sh` | P | v2.0 | the 16 inversions of the outlier-treatment pools (A4) | `02_land_inversion.py` runs |
| `37_corel_v21.py` | H | v2.1 | CorelDRAW automation (Windows, COM) for editable figure files | SVG → `.cdr` |
| `38_check_v21.py` | H | v2.1 | checks of the v2.1 CorelDRAW exports | — |
| `39_fig6_bounds_v22.py` | H | v2.2 | check of the prior bounds in Figure 6(a) | — |
| `make_release.py` | H | v1.0 | assembled the v1.0 release bundle (not uploaded) | — |
| `bug_invlam_repro.py` | H | v1.0 | minimal reproducer of a pyGIMLi 1.5.5 λ bookkeeping issue | — |
| `check_a2_job_equivalence.py` | H | v1.1 | equivalence check of a refactored Stage-4 function | — |
| `check_v13_defs_equivalence.py` | H | v1.3 | equivalence check of the interface definitions | — |

Numbers 23 and 24 were never used.
