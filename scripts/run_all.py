"""
run_all.py — one-command rebuild of the Westhoek ERT paper products.

Always run inside the `pg` environment:  conda run -n pg python run_all.py <mode> ...

  --from-cache [--out DIR] [--no-check]
        Regenerate every paper figure, table and number from the stored forward runs, prior samples and inversions
        (data/processed/, results/runs/g1_lam50_v3|v4): 17_salinity_v2.py run → 18_paper.py all --recompute, both
        writing under DIR (default: the project itself). Then --check DIR and --audit DIR. Measured: 14.5 min (17: 7.3 min, 18: 7.2 min).

  --from-raw --out DIR [--execute] [--until STEP]
        Rerun everything from data/raw in a fresh tree: DIR/scripts (copied) + DIR/data/raw (copied); every script
        derives its paths from its own location, so all outputs land in DIR. Without --execute the steps are listed
        with their expected runtime on the development machine (Windows 11, 4 worker processes): taken from run logs and stored runtimes where available, otherwise estimated (16-refit, 16-joint, 12-*, 14-tradeoffs/build1c/metrics1c/nuisance, 15-*).

  --audit --out DIR
        Provenance audit of a --from-cache rebuild: every file read by 17_salinity_v2.py, 18_paper.py and the check
        (recorded at run time by read_trace.py audit hooks in DIR/provenance_reads.tsv) is matched to the --from-raw
        step that writes it (producer table below; each producer is verified by finding the file's name token in the
        producing script). Files under data/raw are inputs; anything else without a producer is reported.
        Runs automatically after --from-cache --check.

  --diff OLD.json
        Write results/paper/numbers_diff.md: every number whose value differs between OLD.json (a previous
        results/paper/numbers.json) and the current one, old → new, with the reason (rules in DIFF_REASONS).

  --check --out DIR
        Compare a rebuilt tree with the stored project: (1) every number in DIR/results/paper/numbers.json equals the
        stored results/paper/numbers.json; (2) every number with a cross-check reference is found in that stored report;
        (3) the regenerated reports (results/stage6_v2/*, results/paper/*.md|csv) are identical to the stored ones
        (runtime lines excepted).
"""

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
PY = sys.executable

# (step id, argv, expected runtime, needed for the paper)
RAW_STEPS = [
    ("NE", ["<internal:natural_earth>"], "1 min (download)", True),
    ("01", ["01_data_exploration.py"], "1 min", False),
    ("02-F", ["02_land_inversion.py", "--tag", "deep_all", "--para-depth", "60", "--para-dx", "0.33", "--err-rel", "0.03",
              "--lam", "20", "--max-iter", "30", "--verify"], "2 min", True),
    ("02-Fauto", ["02_land_inversion.py", "--tag", "deep_auto", "--para-dx", "0.33", "--err-rel", "0.03", "--lam", "20",
                  "--max-iter", "30"], "2.3 min", True),
    ("04-F", ["04_misfit_diagnostics.py", "--tag", "deep_all"], "0.5 min", True),
    ("05-F", ["05_outlier_characterisation.py", "--tag", "deep_all"], "0.5 min", True),
    ("02-G1", ["02_land_inversion.py", "--tag", "deep_drop_data", "--para-depth", "60", "--para-dx", "0.33", "--err-rel", "0.03",
               "--lam", "20", "--max-iter", "30", "--drop-list", "../results/deep_all_droplist.txt"], "1.7 min", True),
    ("02-G2", ["02_land_inversion.py", "--tag", "deep_drop_elec", "--para-depth", "60", "--para-dx", "0.33", "--err-rel", "0.03",
               "--lam", "20", "--max-iter", "30", "--drop-list", "../results/deep_all_G2_droplist_flagged_electrodes.txt"],
     "1.3 min", True),
    ("02-G3", ["02_land_inversion.py", "--tag", "deep_downweight", "--para-depth", "60", "--para-dx", "0.33", "--err-rel",
               "0.03", "--lam", "20", "--max-iter", "30", "--err-scale-file", "../results/deep_all_G3_err_scale.txt"], "1.5 min", True),
] + [
    (f"02-lam{l}", ["02_land_inversion.py", "--tag", f"g1_lam{l}", "--para-depth", "60", "--para-dx", "0.33", "--err-rel", "0.03",
                    "--lam", str(l), "--max-iter", "30", "--drop-list", "../results/deep_all_droplist.txt"], "1.3 min", True)
    for l in (5, 10, 50, 100)
] + [
    ("04-G1", ["04_misfit_diagnostics.py", "--tag", "deep_drop_data"], "0.5 min", False),
    ("04-G2", ["04_misfit_diagnostics.py", "--tag", "deep_drop_elec"], "0.5 min", True),
    ("04-G3", ["04_misfit_diagnostics.py", "--tag", "deep_downweight"], "0.5 min", False),
    ("06-outl", ["06_compare_runs.py", "--name", "outlier_treatments_r10", "--ref", "deep_all", "--tags", "deep_drop_data",
                 "deep_drop_elec", "deep_downweight", "--r-max", "10"], "0.5 min", False),
    ("07", ["07_lambda_lcurve.py", "--name", "G1_deep_drop_data", "--tags", "g1_lam5", "g1_lam10", "deep_drop_data", "g1_lam50",
            "g1_lam100"], "0.5 min", False),
    ("03", ["03_resolution_analysis.py", "--tag", "g1_lam50", "--suffix", "v3", "--mask-mode", "radius", "--r-max", "10",
            "--ensemble-tags", "g1_lam5", "g1_lam10", "deep_drop_data", "g1_lam50", "g1_lam100"], "5 min", True),
    ("09", ["09_posterior_sampling.py", "--tag", "g1_lam50", "--suffix", "v3", "--r-max", "10", "--r-max-extra", "15",
            "--mix-tags", "g1_lam5", "g1_lam10", "deep_drop_data", "g1_lam50", "g1_lam100"], "5 min", True),
    ("G1b-list", ["<internal:g1b_droplist>"], "< 1 min", True),
    ("02-G1b", ["02_land_inversion.py", "--tag", "g1b_thr299", "--para-depth", "60", "--para-dx", "0.33", "--err-rel", "0.03",
                "--lam", "50", "--max-iter", "30", "--drop-list", "../results/bel_v3/g1b_droplist_thr299.txt"], "2 min", True),
    ("10-spec", ["10_synthetic_benchmark.py", "spec"], "< 1 min", True),
    ("10-build", ["10_synthetic_benchmark.py", "build"], "15 min", True),
    ("10-run", ["10_synthetic_benchmark.py", "run", "--workers", "3"], "7.5 h (380 inversions, 80 000 CPU-s)", True),
    ("10-metrics", ["10_synthetic_benchmark.py", "metrics"], "3 min", True),
    ("11-a1", ["11_stage4_closeout.py", "a1"], "1 min", True),
    ("11-a2", ["11_stage4_closeout.py", "drive", "--task", "a2", "--workers", "4"], "2.5 h", True),
    ("11-a2m", ["11_stage4_closeout.py", "a2metrics"], "1 min", True),
    ("11-a3g", ["11_stage4_closeout.py", "drive", "--task", "a3grid", "--workers", "4"], "0.3 h", False),
    ("11-a3", ["11_stage4_closeout.py", "drive", "--task", "a3", "--workers", "4"], "0.5 h", False),
    ("11-a3m", ["11_stage4_closeout.py", "a3metrics"], "< 1 min", False),
    ("11-a5", ["11_stage4_closeout.py", "drive", "--task", "a5", "--workers", "4"], "2.5 h", False),
    ("11-a5m", ["11_stage4_closeout.py", "a5metrics"], "< 1 min", False),
    ("11-a4", ["11_stage4_closeout.py", "a4"], "3 min", True),
    ("12-spec", ["12_bel_pilot.py", "spec"], "< 1 min", True),
    ("12-timing", ["12_bel_pilot.py", "timing"], "10 min", True),
    ("12-sample", ["12_bel_pilot.py", "sample", "--n-samples", "300", "--workers", "4"], "1 h", True),
    ("12-falsify", ["12_bel_pilot.py", "falsify"], "2 min", False),
    ("13-spec", ["13_bel_full.py", "spec"], "< 1 min", True),
    ("13-dual", ["13_bel_full.py", "sample", "--set", "dual", "--workers", "4"], "0.5 h", True),
    ("13-train", ["13_bel_full.py", "sample", "--set", "train", "--workers", "4"], "6 h", True),
    ("13-approx", ["13_bel_full.py", "approx"], "1 min", True),
    ("13-falsify", ["13_bel_full.py", "falsify"], "4 min", True),
    ("13-bel", ["13_bel_full.py", "bel"], "15 min", False),
    ("14-tradeoffs", ["14_gateA_sensitivity.py", "tradeoffs"], "5 min", False),
    ("14-subsets", ["14_gateA_sensitivity.py", "subsets"], "22 min", False),
    ("14-build1c", ["14_gateA_sensitivity.py", "build1c"], "5 min", False),
    ("14-drive1c", ["14_gateA_sensitivity.py", "drive1c", "--workers", "4"], "1 h", False),
    ("14-metrics1c", ["14_gateA_sensitivity.py", "metrics1c"], "2 min", False),
    ("14-nuisance", ["14_gateA_sensitivity.py", "nuisance"], "5 min", False),
    ("14-fwdcheck", ["14_gateA_sensitivity.py", "fwdcheck"], "2 min", False),
    ("15-conventions", ["15_stage5_closeout.py", "conventions"], "2 min", False),
    ("15-twin", ["15_stage5_closeout.py", "twin"], "1 min", True),
    ("15-twinbel", ["15_stage5_closeout.py", "twinbel"], "3 min", True),
    ("15-site", ["15_stage5_closeout.py", "site"], "3 min", False),
    ("15-bounded", ["15_stage5_closeout.py", "bounded"], "1 min", False),
    ("15-archie", ["15_stage5_closeout.py", "archie"], "< 1 min", False),
    ("16-spec3", ["16_stage6_salinity.py", "spec3"], "< 1 min", True),
    ("16-sample", ["16_stage6_salinity.py", "sample", "--workers", "4"], "2 h", True),
    ("16-refit", ["16_stage6_salinity.py", "refit"], "8 min", False),
    ("16-joint", ["16_stage6_salinity.py", "joint"], "10 min", False),
    ("17", ["17_salinity_v2.py", "run"], "7 min", True),
    ("22-run", ["22_lebbe1980_synthetic.py", "run", "--workers", "4"], "3–3.5 h (40 data sets × 5 λ)", True),
    ("22-report", ["22_lebbe1980_synthetic.py", "report"], "2 min", True),
    ("20-run", ["20_petrophysics_sensitivity.py", "run"], "3 min", True),
    ("20-report", ["20_petrophysics_sensitivity.py", "report"], "< 1 min", True),
    ("21-location", ["21_k1_comparison.py", "location"], "< 1 min", True),
    ("21-run", ["21_k1_comparison.py", "run"], "1 min", True),
    ("21-report", ["21_k1_comparison.py", "report"], "< 1 min", True),
    ("25-field", ["25_generalisation.py", "field"], "< 1 min", True),
    ("25-run", ["25_generalisation.py", "run", "--workers", "6"], "see results/v1_3/1_plan.md (416 inversions)", True),
    ("25-report", ["25_generalisation.py", "report"], "5 min", True),
    ("26-run", ["26_raw_abem.py", "run"], "2 min (+ 1 min g1_lam50_recip)", True),
    ("26-followup", ["26_raw_abem.py", "followup"], "1 min", True),
    ("26-report", ["26_raw_abem.py", "report"], "< 1 min", True),
    ("27-fetch", ["27_dp_groundtruth.py", "fetch"], "< 1 min (Zenodo download, MD5 checked)", True),
    ("27-run", ["27_dp_groundtruth.py", "run", "--workers", "6"], "3 min", True),
    ("27-cpt", ["27_dp_groundtruth.py", "cpt"], "< 1 min (needs the unpublished CPT archive)", True),
    ("27-samedate", ["27_dp_groundtruth.py", "samedate"], "30 min (12 inversions)", True),
    ("27-dp", ["27_dp_groundtruth.py", "dp"], "3 min", True),
    ("28-run", ["28_dp_synthetic.py", "run", "--workers", "6"], "1.6 h (108 inversions; exploratory)", True),
    ("28-report", ["28_dp_synthetic.py", "report"], "2 min", True),
    ("29-run", ["29_detectability.py", "run", "--workers", "6"], "20 min (42 forward runs)", True),
    ("29-report", ["29_detectability.py", "report"], "< 1 min", True),
    ("30-run", ["30_bel_v4.py", "run", "--workers", "6"], "8 h (4873 forward runs + BEL)", True),
    ("28-ablation", ["28_dp_synthetic.py", "run", "--ablation", "--workers", "3"], "2.3 h (108 inversions; exploratory)", True),
    ("30-synth", ["30_bel_v4.py", "synth"], "10 min", True),
    ("30-l24", ["30_bel_v4.py", "l24", "--workers", "5"], "5.5 h (4873 + 100 forward runs + BEL)", True),
    ("30-layout", ["30_bel_v4.py", "layout"], "15 min (BEL refits; no forward runs)", True),
    ("30-layout-t1", ["30_bel_v4.py", "layout-t1"], "5 min (BEL refits; no forward runs)", True),
    ("30-report", ["30_bel_v4.py", "report"], "1 min", True),
    ("31-run", ["31_scores.py", "run"], "30 min (re-draws of the stored BEL fits, verified; no forward runs)", True),
    ("31-report", ["31_scores.py", "report"], "1 min", True),
    ("32-fetch", ["32_location_map.py", "fetch"], "1 min (network, ~46 MB; skips files already recorded)", True),
    ("32", ["32_location_map.py", "build"], "1 min", True),
    ("27-report", ["27_dp_groundtruth.py", "report"], "< 1 min", True),
    ("19", ["19_rho_lens_bound_check.py"], "3 min", False),
    ("18", ["18_paper.py", "all"], "7 min", True),
]


# ─────────────────────────────────────────────────────────────────────────────
# provenance: which --from-raw step writes each file (pattern on the project-relative path)
# ─────────────────────────────────────────────────────────────────────────────
RUN_STEP = {"deep_all": "02-F", "deep_auto": "02-Fauto", "deep_drop_data": "02-G1", "deep_drop_elec": "02-G2", "deep_downweight": "02-G3",
            "g1_lam5": "02-lam5", "g1_lam10": "02-lam10", "g1_lam50": "02-lam50", "g1_lam100": "02-lam100",
            "g1b_thr299": "02-G1b", "g1_lam50_recip": "26-run"}
PRODUCERS = [   # (regex, step id or callable(match) → step id, script, evidence token that must occur in the script)
    (r"^data/raw/", "raw input", None, None),
    (r"^data/external/natural_earth/ne_10m_", "NE", "run_all.py", "natural_earth"),
    (r"^data/external/lebbe1981_K1_digitised\.csv$", "raw input", None, None),     # hand-digitised (Lebbe 1981, Fig. 3)
    (r"^data/processed/v1_1/lebbe1980/", "22-run", "22_lebbe1980_synthetic.py", "lebbe1980"),
    (r"^results/v1_1/0_profile_location\.md$", "21-location", "21_k1_comparison.py", "0_profile_location.md"),
    (r"^results/v1_1/(1_lebbe1980_synthetic\.md|T_S1_lebbe1980\.csv)$", "22-report", "22_lebbe1980_synthetic.py",
     "1_lebbe1980_synthetic.md"),
    (r"^results/v1_1/(2_petrophysics_sensitivity\.md|T_S2_petrophysics\.(csv|md))$", "20-report",
     "20_petrophysics_sensitivity.py", "2_petrophysics_sensitivity.md"),
    (r"^results/v1_1/3_k1_comparison\.md$", "21-report", "21_k1_comparison.py", "3_k1_comparison.md"),
    (r"^data/processed/v1_3/general/", "25-run", "25_generalisation.py", "general"),
    (r"^data/raw/Project12_LS210110079_2018-03-07_11-22-23/", "raw input", None, None),   # T. Hermans, Oct 2026
    (r"^data/raw/dp_zenodo_20792093/", "27-fetch", "27_dp_groundtruth.py", "zenodo"),
    (r"^data/raw/cpt_vrancken_2024/", "raw input", None, None),            # E. Vrancken, unpublished, Oct 2026
    (r"^data/processed/v1_4/(cpt\.json|T_S6_geometry\.csv)$", "27-cpt", "27_dp_groundtruth.py", "cpt.json"),
    (r"^data/processed/v1_4/samedate/", "27-samedate", "27_dp_groundtruth.py", "samedate"),
    (r"^data/processed/v1_1/k1_field_bel\.npz$", "21-run", "21_k1_comparison.py", "k1_field_bel.npz"),
    (r"^data/processed/paper/bel_products\.npz$", "18", "18_paper.py", "bel_products"),
    (r"^data/processed/v1_5/dp_synth/T2[abc]_", "28-ablation", "28_dp_synthetic.py", "ablation"),
    (r"^data/processed/v1_5/dp_synth/fwd_T2[abc]_", "28-ablation", "28_dp_synthetic.py", "ablation"),
    (r"^data/processed/v1_5/dp_synth/", "28-run", "28_dp_synthetic.py", "dp_synth"),
    (r"^data/processed/v1_5/bel_v4/synth\.npz$", "30-synth", "30_bel_v4.py", "synth.npz"),
    (r"^data/processed/v1_5/bel_v4_l24/", "30-l24", "30_bel_v4.py", "bel_v4_l24"),
    (r"^data/processed/v1_5/bel_v4_layout/t1_subset\.npz$", "30-layout-t1", "30_bel_v4.py", "t1_subset.npz"),
    (r"^data/processed/v1_5/bel_v4_layout/", "30-layout", "30_bel_v4.py", "bel_v4_layout"),
    (r"^data/processed/v1_5/layout_vs_time\.json$", "30-report", "30_bel_v4.py", "layout_vs_time.json"),
    (r"^data/processed/v1_6/(cases\.csv|regen_check\.json|rho_below_log10\.npz)$", "31-run", "31_scores.py", "cases.csv"),
    (r"^data/processed/v1_6/scores\.json$", "31-report", "31_scores.py", "scores.json"),
    (r"^data/external/basemap/", "32-fetch", "32_location_map.py", "basemap"),
    (r"^data/processed/v1_7/location_map\.json$", "32", "32_location_map.py", "location_map.json"),
    (r"^results/figs/F_01_", "32", "32_location_map.py", "F_01"),
    (r"^results/figs/F_12b_dp_vs_2025_ert_line\.(png|pdf)$", "27-report", "27_dp_groundtruth.py", "F_12b publication copy"),
    (r"^results/figs/F_14_synthetic_vs_field\.(png|pdf)$", "28-report", "28_dp_synthetic.py", "F_14 publication copy"),
    (r"^results/figs/(F_17_bel_synthetic_and_2025_ert_line|F_19_layout_vs_time)\.(png|pdf)$", "30-report", "30_bel_v4.py", "F_17/F_19 publication copies"),
    (r"^results/figs/F_20_scores\.(png|pdf)$", "31-report", "31_scores.py", "F_20 publication copy"),
    (r"^results/v1_6/(1_scores\.md|T_S12_scores\.(csv|md))$", "31-report", "31_scores.py", "1_scores.md"),
    (r"^results/v1_5/7_layout_vs_time\.md$", "30-report", "30_bel_v4.py", "7_layout_vs_time.md"),
    (r"^data/processed/v1_5/bel_checks\.json$", "30-report", "30_bel_v4.py", "bel_checks.json"),
    (r"^data/processed/v1_5/ablation\.json$", "28-report", "28_dp_synthetic.py", "ablation.json"),
    (r"^results/v1_5/5_bel_checks\.md$", "30-report", "30_bel_v4.py", "5_bel_checks.md"),
    (r"^results/v1_5/6_ablation\.md$", "28-report", "28_dp_synthetic.py", "6_ablation.md"),
    (r"^data/processed/v1_5/detect/", "29-run", "29_detectability.py", "detect"),
    (r"^data/processed/v1_5/bel_v4/", "30-run", "30_bel_v4.py", "bel_v4"),
    (r"^data/processed/v1_5/dp_synth_summary\.json$", "28-report", "28_dp_synthetic.py", "dp_synth_summary.json"),
    (r"^data/processed/v1_5/detect\.json$", "29-report", "29_detectability.py", "detect.json"),
    (r"^data/processed/v1_5/bel_v4_summary\.json$", "30-report", "30_bel_v4.py", "bel_v4_summary.json"),
    (r"^results/v1_5/(2_dp_synthetic\.md|T_S10_dp_synthetic\.(csv|md))$", "28-report", "28_dp_synthetic.py", "2_dp_synthetic.md"),
    (r"^results/v1_5/3_detectability\.md$", "29-report", "29_detectability.py", "3_detectability.md"),
    (r"^results/v1_5/(4_bel_v4\.md|T_S11_bel_v4\.(csv|md))$", "30-report", "30_bel_v4.py", "4_bel_v4.md"),
    (r"^data/processed/v1_4/dp_results\.json$", "27-dp", "27_dp_groundtruth.py", "dp_results.json"),
    (r"^data/processed/v1_4/e14_followup\.json$", "26-followup", "26_raw_abem.py", "e14_followup.json"),
    (r"^data/processed/v1_4/(raw2018\.json|raw2018_data\.csv|raw2018_pairs\.npz|recip_err_scale\.txt)$", "26-run",
     "26_raw_abem.py", "raw2018.json"),
    (r"^data/processed/v1_4/placement\.json$", "27-run", "27_dp_groundtruth.py", "placement.json"),
    (r"^results/v1_4/(1_raw_abem\.md|T_S8_raw_2018\.(csv|md)|7_e14_followup\.md)$", "26-report", "26_raw_abem.py", "1_raw_abem.md"),
    (r"^results/v1_4/(0_inventory\.md|1_line_placement\.md|2_dp_comparison\.md|3_same_epoch\.md|4_inreview_method\.md|"
     r"5_cpt_inventory\.md|6_lithology\.md|T_S[679]_[a-z0-9_]+\.(csv|md))$", "27-report", "27_dp_groundtruth.py", "1_line_placement.md"),
    (r"^data/processed/v1_3/field\.json$", "25-field", "25_generalisation.py", "field.json"),
    (r"^data/processed/v1_3/v13_summary\.json$", "25-report", "25_generalisation.py", "v13_summary.json"),
    (r"^results/v1_3/0_field_like_for_like\.md$", "25-field", "25_generalisation.py", "0_field_like_for_like.md"),
    (r"^results/v1_3/(2_factorial_results\.md|3_rule_fit\.md|T_S4_factorial\.(csv|md))$", "25-report",
     "25_generalisation.py", "2_factorial_results.md"),
    (r"^data/processed/stage6_v2/field_v2\.npz$", "17", "17_salinity_v2.py", "stage6_v2"),
    (r"^data/processed/runs/([^/]+)/westhoek_inversion_results\.npz$", lambda m: RUN_STEP.get(m.group(1)),
     "02_land_inversion.py", "westhoek_inversion_results.npz"),
    (r"^results/deep_all_droplist\.txt$", "04-F", "04_misfit_diagnostics.py", "_droplist.txt"),
    (r"^results/(deep_all|deep_drop_data|deep_drop_elec|deep_downweight)_misfit_report\.txt$",
     lambda m: {"deep_all": "04-F", "deep_drop_data": "04-G1", "deep_drop_elec": "04-G2", "deep_downweight": "04-G3"}[m.group(1)],
     "04_misfit_diagnostics.py", "_misfit_report.txt"),
    (r"^results/deep_all_G2_droplist_flagged_electrodes\.txt$", "05-F", "05_outlier_characterisation.py", "G2_droplist_flagged_electrodes"),
    (r"^results/deep_all_G3_err_scale\.txt$", "05-F", "05_outlier_characterisation.py", "G3_err_scale"),
    (r"^results/deep_all_outlier_characterisation\.txt$", "05-F", "05_outlier_characterisation.py", "outlier_characterisation"),
    (r"^results/bel_v3/g1b_droplist_thr299\.txt$", "G1b-list", "run_all.py", "g1b_droplist_thr299"),
    (r"^results/runs/g1_lam50_v3/posterior_(interface_bands\.npz|sampling_report\.txt)$", "09", "09_posterior_sampling.py",
     "posterior_interface_bands"),
    (r"^results/synthetic/truth_models\.md$", "10-spec", "10_synthetic_benchmark.py", "truth_models.md"),
    (r"^results/bel_v3/2_nuisance\.md$", "14-nuisance", "14_gateA_sensitivity.py", "2_nuisance.md"),
    (r"^results/runs/g1_lam50_v4/interface_definitions_(bands\.npz|report\.md)$", "11-a4", "11_stage4_closeout.py", "interface_definitions_bands"),
    (r"^data/processed/synthetic/truth_spec\.json$", "10-spec", "10_synthetic_benchmark.py", "truth_spec.json"),
    (r"^data/processed/synthetic/truth_forward\.npz$", "10-build", "10_synthetic_benchmark.py", "truth_forward.npz"),
    (r"^data/processed/synthetic/runs/T[0-9a-z]+_r\d\d\.npz$", "10-run", "10_synthetic_benchmark.py", '"runs"'),
    (r"^data/processed/synthetic_s5/a2_runs/", "11-a2", "11_stage4_closeout.py", "a2_runs"),
    (r"^data/processed/bel/prior_spec\.json$", "12-spec", "12_bel_pilot.py", "prior_spec.json"),
    (r"^data/processed/bel/forward_mesh_choice\.json$", "12-timing", "12_bel_pilot.py", "forward_mesh_choice.json"),
    (r"^data/processed/bel/v2/prior_spec_v2\.json$", "13-spec", "13_bel_full.py", "prior_spec_v2.json"),
    (r"^data/processed/bel/v2/approx_error\.npz$", "13-approx", "13_bel_full.py", "approx_error.npz"),
    (r"^data/processed/bel/v2/train/s\d+\.npz$", "13-train", "13_bel_full.py", '"train"'),
    (r"^data/processed/bel/v2/dual/s\d+\.npz$", "13-dual", "13_bel_full.py", '"dual"'),
    (r"^data/processed/bel_prior_v3/train/s\d+\.npz$", "16-sample", "16_stage6_salinity.py", '"train"'),
    (r"^data/processed/bel_v4/twin_theta\.json$", "15-twin", "15_stage5_closeout.py", "twin_theta.json"),
    (r"^data/processed/bel_v4/twin_forward\.npz$", "15-twinbel", "15_stage5_closeout.py", "twin_forward.npz"),
    (r"^results/synthetic/benchmark_report\.md$", "10-metrics", "10_synthetic_benchmark.py", "benchmark_report.md"),
    (r"^results/synthetic_s5/a1_bias_vs_lambda\.md$", "11-a1", "11_stage4_closeout.py", "a1_bias_vs_lambda.md"),
    (r"^results/synthetic_s5/a2_interface_definitions\.md$", "11-a2m", "11_stage4_closeout.py", "a2_interface_definitions.md"),
    (r"^results/bel_v3/3_forward_accuracy\.md$", "14-fwdcheck", "14_gateA_sensitivity.py", "3_forward_accuracy.md"),
    (r"^results/stage6/1_prior_v3_refit\.md$", "16-refit", "stage6_analysis.py", "1_prior_v3_refit.md"),
    (r"^(results|data/processed)/stage6_v2/", "17", "17_salinity_v2.py", "stage6_v2"),
    (r"^results/paper/", "18", "18_paper.py", "numbers.json"),
]


CURATED = {"results/REFERENCE.md": "hand-maintained reference document (settings, decision log); the numbers "
                                  "cross-checked against it come from data/processed/runs/g1_lam50 (step 02-lam50)"}


def audit(out):
    """Match every recorded read to its --from-raw producer (see PRODUCERS); report files without one."""
    import re
    trace = os.path.join(out, "provenance_reads.tsv")
    if not os.path.exists(trace):
        raise SystemExit(f"no read trace in {trace} — run --from-cache first")
    step_ids = {sid for sid, *_ in RAW_STEPS}
    reads, orig = {}, {}
    import glob
    lines = [l for fn in [trace] + sorted(glob.glob(trace + ".*")) for l in open(fn, encoding="utf-8")]
    for line in lines:
        tag, path = line.rstrip("\n").split("\t", 1)
        key = os.path.normcase(os.path.abspath(path))
        reads.setdefault(key, set()).add(tag); orig.setdefault(key, os.path.abspath(path))
    root_n, out_n = os.path.normcase(ROOT) + os.sep, os.path.normcase(os.path.abspath(out)) + os.sep
    groups, orphans, raw, curated, own, ignored = {}, [], [], [], 0, 0
    src_cache = {}
    for path, tags in sorted(reads.items()):
        if path.startswith(out_n) and out_n != root_n:
            own += 1; continue                               # written by this rebuild (17 → 18)
        if not path.startswith(root_n):
            ignored += 1; continue                           # Python / site-packages / system files
        relp = orig[path][len(root_n):].replace("\\", "/")
        if relp.startswith("scripts/") or relp.endswith((".py", ".pyc")):
            ignored += 1; continue                           # code, not data
        if relp in CURATED:
            curated.append((relp, tags)); continue
        rel_orig = relp
        hit = None
        for rx, step, script, token in PRODUCERS:
            m = re.match(rx, rel_orig, flags=re.I)
            if m:
                sid = step(m) if callable(step) else step
                hit = (rx, sid, script, token); break
        if hit is None:
            orphans.append((relp, tags)); continue
        rx, sid, script, token = hit
        if sid == "raw input":
            raw.append(relp); continue
        if script not in src_cache:
            src_cache[script] = open(os.path.join(SCRIPT_DIR, script), encoding="utf-8").read()
        ok = sid in step_ids and token in src_cache[script]
        g = groups.setdefault((sid, script, rx), dict(n=0, ok=ok, tags=set(), example=relp))
        g["n"] += 1; g["tags"] |= tags; g["ok"] &= ok
    L = ["# Provenance audit — files read by 17_salinity_v2.py, 18_paper.py and the rebuild check", "",
         f"Read trace: `{trace}` (audit hooks, read-mode open() calls). Files written by this rebuild itself and "
         f"read back ({own}), Python/system files and code ({ignored}) are not listed.", "",
         f"## Raw inputs ({len(raw)})", ""] + [f"- `{r}`" for r in raw]
    L += ["", "## Files with a --from-raw producer", "",
          "| producer step | script | files | example | read by | producer verified |", "|---|---|---|---|---|---|"]
    for (sid, script, rx), g in sorted(groups.items()):
        L.append(f"| {sid} | `{script}` | {g['n']} | `{g['example']}` | {', '.join(sorted(g['tags']))} | "
                 f"{'yes' if g['ok'] else '**NO**'} |")
    L += ["", f"## Hand-maintained documents — no producer by design ({len(curated)})", ""]
    L += [f"- `{r}` (read by {', '.join(sorted(t))}): {CURATED[r]}" for r, t in curated] if curated else ["- none"]
    L += ["", f"## Files with NO producer ({len(orphans)})", ""]
    L += [f"- `{r}` (read by {', '.join(sorted(t))})" for r, t in orphans] if orphans else ["- none"]
    bad = [g for g in groups.values() if not g["ok"]]
    L += ["", f"**Result: {'PASS' if not orphans and not bad else 'FAIL'}** — {sum(g['n'] for g in groups.values())} "
          f"produced files in {len(groups)} groups, {len(raw)} raw inputs, {len(curated)} hand-maintained, "
          f"{len(orphans)} without a producer, "
          f"{len(bad)} producer groups not verified."]
    with open(os.path.join(out, "results", "paper", "provenance_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[-(len(orphans) + len(curated) + 8):]))
    return not orphans and not bad


NE_LAYERS = [("physical", "coastline"), ("physical", "land"), ("cultural", "admin_0_boundary_lines_land")]


def fetch_natural_earth(root):
    """Natural Earth 1:10m layers for the F1 location inset → <root>/data/external/natural_earth/ (public domain;
    downloaded through cartopy's natural_earth() from its standard source, then copied with all sidecar files)."""
    import glob
    from cartopy.io import shapereader
    import cartopy
    dst = os.path.join(root, "data", "external", "natural_earth")
    os.makedirs(dst, exist_ok=True)
    got = []
    for cat, name in NE_LAYERS:
        shp = shapereader.natural_earth(resolution="10m", category=cat, name=name)
        for f in glob.glob(os.path.splitext(shp)[0] + ".*"):
            shutil.copy2(f, dst); got.append(os.path.basename(f))
    with open(os.path.join(dst, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# Natural Earth 1:10m layers (F1 location inset)\n\n"
                "Source: Natural Earth (https://www.naturalearthdata.com), public domain — credit: 'Made with Natural Earth'.\n"
                f"Layers: {', '.join(f'ne_10m_{n} ({c})' for c, n in NE_LAYERS)}.\n"
                f"Fetched with cartopy {cartopy.__version__} (cartopy.io.shapereader.natural_earth) by run_all.py "
                "(step NE); files copied unchanged.\n\nFiles: " + ", ".join(sorted(got)) + "\n")
    print(f"[natural earth] {len(got)} files → {dst}")


def g1b_droplist(root):
    """G1b drop-list: run F's normalised log residuals |r| > 2.99 (fixed Stage-4 threshold) instead of 2.495."""
    import numpy as np
    d = np.load(os.path.join(root, "data", "processed", "runs", "deep_all", "westhoek_inversion_results.npz"))
    r = (np.log(d["d_obs"]) - np.log(d["d_pred"])) / d["d_err"]
    drop = sorted(d["keep_idx"].astype(int)[np.abs(r) > 2.99])
    os.makedirs(os.path.join(root, "results", "bel_v3"), exist_ok=True)
    with open(os.path.join(root, "results", "bel_v3", "g1b_droplist_thr299.txt"), "w") as f:
        f.write("# G1b: F (deep_all) per-datum outliers with the FIXED threshold |r| > 2.99 (Stage-4 A value) instead of 2.50\n")
        f.write("# 0-based measurement index in land_data.dat order\n")
        f.writelines(f"{i}\n" for i in drop)
    print(f"[g1b] {len(drop)} data")


def run(argv, cwd, log):
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(argv)}\n"); f.flush()
        p = subprocess.run([PY, "-u"] + argv, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"  {' '.join(argv[:3])} … exit {p.returncode} ({dt:.0f} s)", flush=True)
    if p.returncode != 0:
        raise SystemExit(f"step failed — see {log}")
    return dt


# ─────────────────────────────────────────────────────────────────────────────
def strip(path):
    with open(path, encoding="utf-8") as f:
        return [l for l in f.read().splitlines() if not l.startswith("Runtime ") and "[18_paper] done" not in l]


def check(out):
    ok = True
    L = [f"# Rebuild check — {out} vs the stored project", ""]
    new = json.load(open(os.path.join(out, "results", "paper", "numbers.json"), encoding="utf-8"))
    sp = os.path.join(ROOT, "results", "paper", "numbers.json")
    stored = {r["key"]: r["value"] for r in json.load(open(sp, encoding="utf-8"))} if os.path.exists(sp) else None
    n_same = n_ref = n_ref_ok = 0
    bad = []
    for r in new:
        if stored is not None:
            if stored.get(r["key"]) == r["value"]:
                n_same += 1
            else:
                bad.append(f"value differs: `{r['key']}` rebuilt {r['value']!r} vs stored {stored.get(r['key'])!r}")
        if r.get("ref"):
            n_ref += 1
            rp, ctx, match = r["ref"]
            p = os.path.join(ROOT, rp)
            if ctx.startswith("npz:"):                   # "npz:<key>:<format>" or "npz:len(<key>):<format>"
                import numpy as np
                _, key, fmt = ctx.split(":")
                z = np.load(p, allow_pickle=True)
                v = len(z[key[4:-1]]) if key.startswith("len(") else z[key].item()
                lines = [format(v, fmt)]
                ctx = ""
            else:
                lines = open(p, encoding="utf-8", errors="replace").read().splitlines() if os.path.exists(p) else []
            if any(ctx in l and match in l for l in lines):
                n_ref_ok += 1
            else:
                bad.append(f"cross-check failed: `{r['key']}` = {r['value']!r}: no line in `{rp}` containing {ctx!r} and {match!r}")
    if stored is not None:
        missing = set(stored) - {r["key"] for r in new}
        bad += [f"number missing in the rebuild: `{k}`" for k in sorted(missing)]
    L.append(f"- numbers: {len(new)} rebuilt; identical to the stored numbers.md: "
             f"{n_same if stored is not None else 'n/a (no stored numbers)'}")
    L.append(f"- cross-checks against earlier stored reports: {n_ref_ok} / {n_ref} found")
    files_same = files_diff = 0
    for sub in (os.path.join("results", "stage6_v2"), os.path.join("results", "paper"), os.path.join("results", "v1_1"),
                os.path.join("results", "v1_3"), os.path.join("results", "v1_4"), os.path.join("results", "v1_5"),
                os.path.join("results", "v1_6"), os.path.join("results", "figs")):
        d_new = os.path.join(out, sub)
        if not os.path.isdir(d_new) or os.path.abspath(out) == ROOT:
            continue
        for fn in sorted(os.listdir(d_new)):
            if not fn.endswith((".md", ".csv")):
                continue
            a, b = os.path.join(d_new, fn), os.path.join(ROOT, sub, fn)
            if not os.path.exists(b):
                bad.append(f"no stored counterpart for {sub}/{fn}"); files_diff += 1; continue
            if strip(a) == strip(b):
                files_same += 1
            else:
                files_diff += 1; bad.append(f"report differs: {sub}/{fn}")
    L.append(f"- regenerated reports identical to the stored ones (runtime lines excepted): {files_same}, differing: {files_diff}")
    L += ["", "## Problems", ""] + ([f"- {b}" for b in bad] if bad else ["- none"])
    ok = not bad
    L += ["", f"**Result: {'PASS' if ok else 'FAIL'}**"]
    with open(os.path.join(out, "results", "paper", "rebuild_check.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return ok


DIFF_REASONS = [   # (key prefix, reason) — first match wins
    ("lens_present", "recomputed with the v1.0 headline training set (site + physical, prior v3) instead of the "
                     "Gate-A prior-v2 classifier"),
    ("bel_base_site_only", "pre-freeze (site-only) headline, kept for comparison"),
    ("physical_filter", "new: physical-filter comparison (site + physical vs site only)"),
    ("second_bel", "new: second BEL refitted on the site + physical subset and added to the paper numbers"),
    ("joint_heldout", "new: joint-BEL held-out coverage now reported (site + physical)"),
    ("bel_n_train", "training / held-out sets: site + physical filter"),
    ("bel_", "base-only BEL refitted on the site + physical subset (ρ_lens ≥ 0.69415 Ω·m)"),
    ("falsify_", "falsification recomputed on the site + physical training set"),
    ("unit_lens_heldout", "held-out set changed (site + physical: 173 samples, truths now physical)"),
    ("unit_", "joint BEL refitted on the site + physical subset (transform bound kept at 0.4 Ω·m); Stage 6 rerun"),
    ("constraint_heldout", "held-out set changed (site + physical): truths are now physical, far fewer rejections"),
    ("constraint_", "joint BEL refitted on the site + physical subset; Stage 6 rerun"),
    ("class_area_", "joint BEL refitted on the site + physical subset; Stage 6 rerun"),
    ("pdis_field", "joint BEL refitted on site + physical"),
    ("sens_prior_subset", "headline subset is now site + physical"),
    ("sens_", "derived from the refitted BELs (site + physical)"),
]


def diff(old_json, title=None):
    new_p = os.path.join(ROOT, "results", "paper", "numbers.json")
    import re
    norm = lambda k: re.sub(r"_n_\d+$", "", k)       # pre-freeze keys embedded bin sample counts ("(n = 573)")
    old = {norm(r["key"]): r for r in json.load(open(old_json, encoding="utf-8"))}
    new = {norm(r["key"]): r for r in json.load(open(new_p, encoding="utf-8"))}
    why = lambda k: next((w for pre, w in DIFF_REASONS if k.startswith(pre)), "**unexplained — check**")
    esc = lambda x: str(x).replace("|", "\\|")
    changed = [k for k in new if k in old and new[k]["value"] != old[k]["value"]]
    added = [k for k in new if k not in old]; removed = [k for k in old if k not in new]
    same = [k for k in new if k in old and new[k]["value"] == old[k]["value"]]
    refs = [k for k in same if (new[k].get("ref") or [None])[0] != (old[k].get("ref") or [None])[0]]
    L = [f"# Numbers diff — {title or 'v1.0 freeze (physical filter)'}", "",
         f"Old: `{os.path.relpath(old_json, ROOT)}`; new: `results/paper/numbers.json`. {len(changed)} changed, "
         f"{len(added)} added, {len(removed)} removed, {len(same)} unchanged.", "",
         "## Changed", "", "| key | quantity | old | new | why |", "|---|---|---|---|---|"]
    L += [f"| `{k}` | {esc(new[k]['desc'])} | {esc(old[k]['value'])} | {esc(new[k]['value'])} | {why(k)} |" for k in changed]
    L += ["", "## Added", "", "| key | quantity | value | why |", "|---|---|---|---|"]
    L += [f"| `{k}` | {esc(new[k]['desc'])} | {esc(new[k]['value'])} | {why(k)} |" for k in added]
    chk = os.path.join(ROOT, "results", "paper", "rho_lens_bound_check.md")
    if os.path.exists(chk) and any(k.startswith(("unit_lens", "second_bel_rho_lens")) for k in changed + added):
        body = open(chk, encoding="utf-8").read().split("\n", 2)[2]
        L += ["", "## ρ_lens transform bound: why it stays at 0.4 Ω·m", "",
              "A first v1.0 build also raised the ρ_lens logit bound of the joint and second BEL to the physical value "
              "0.69415 Ω·m. The lens then went from 69.5 to 78.0 % seawater (joint BEL ρ_lens 1.27 [0.91, 1.71] → 1.13 "
              "[0.91, 1.47] Ω·m; second BEL 1.30 [1.03, 1.62] → 1.22 [1.03, 1.46] Ω·m), although removing low-ρ_lens "
              "models should, if anything, make it less saline. `scripts/19_rho_lens_bound_check.py` shows that the "
              "filter alone changes nothing and the shift comes from the bound (a transformation effect, no new data); "
              "v1.0 therefore keeps the transform bound at 0.4 Ω·m. Its report (`results/paper/rho_lens_bound_check.md`):",
              "", body.strip()]
    L += ["", "## Removed", ""] + ([f"- `{k}` ({esc(old[k]['desc'])}): {esc(old[k]['value'])}" for k in removed] or ["- none"])
    L += ["", "## Unchanged values with a new cross-check source", ""]
    L += [f"- `{k}`: {old[k]['ref'][0] if old[k].get('ref') else '—'} → {new[k]['ref'][0] if new[k].get('ref') else '—'}"
          for k in refs] or ["- none"]
    out = os.path.join(ROOT, "results", "paper", "numbers_diff.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:4]) + f"\n[diff] → {out}")
    unexplained = [k for k in changed + added if why(k).startswith("**")]
    if unexplained:
        print("UNEXPLAINED:", unexplained)


def from_cache(out, do_check):
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    in_project = out == ROOT
    logdir = os.path.join(ROOT, "data", "processed", "paper") if in_project else out
    os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, "run_all_from_cache.log")
    if not in_project:                                  # read tracing only for rebuilds into a separate folder
        trace = os.path.join(out, "provenance_reads.tsv")
        import glob
        for fn in [trace] + glob.glob(trace + ".*"):
            if os.path.exists(fn):
                os.remove(fn)
        os.environ["WESTHOEK_READ_TRACE"] = trace
    print(f"[from-cache] → {out} (log {log})")
    t = {}
    t["17"] = run(["17_salinity_v2.py", "run", "--out", out], SCRIPT_DIR, log)
    for sid, argv in (("22r", ["22_lebbe1980_synthetic.py", "report"]), ("20", ["20_petrophysics_sensitivity.py", "run"]),
                      ("20r", ["20_petrophysics_sensitivity.py", "report"]), ("21l", ["21_k1_comparison.py", "location"]),
                      ("21", ["21_k1_comparison.py", "run"]), ("21r", ["21_k1_comparison.py", "report"]),
                      ("25f", ["25_generalisation.py", "field"]), ("25r", ["25_generalisation.py", "report"]),
                      ("26", ["26_raw_abem.py", "run"]), ("26f", ["26_raw_abem.py", "followup"]),
                      ("26r", ["26_raw_abem.py", "report"]), ("27", ["27_dp_groundtruth.py", "run"]),
                      ("27c", ["27_dp_groundtruth.py", "cpt"]), ("27d", ["27_dp_groundtruth.py", "dp"]),
                      ("27r", ["27_dp_groundtruth.py", "report"]), ("28r", ["28_dp_synthetic.py", "report"]),
                      ("29r", ["29_detectability.py", "report"]), ("30r", ["30_bel_v4.py", "report"]),
                      ("31r", ["31_scores.py", "report"]), ("32", ["32_location_map.py", "build"])):
        t[sid] = run(argv + ["--out", out], SCRIPT_DIR, log)
    t["18"] = run(["18_paper.py", "all", "--out", out, "--recompute"], SCRIPT_DIR, log)
    print(f"[from-cache] done: " + ", ".join(f"{k} {v:.0f} s" for k, v in t.items()) + f"; total {sum(t.values())/60:.1f} min")
    if do_check and out != ROOT:
        sys.path.insert(0, SCRIPT_DIR)
        import read_trace
        flush = read_trace.install("check")
        ok = check(out)
        flush()
        ok_audit = audit(out)
        if not (ok and ok_audit):
            raise SystemExit(1)


def from_raw(out, execute, until):
    out = os.path.abspath(out)
    print("[from-raw] steps (expected runtime on the development machine; * = needed for the paper):")
    for sid, argv, rt, paper in RAW_STEPS:
        print(f"  {'*' if paper else ' '} {sid:<14} {rt:<36} {' '.join(argv)}")
    if not execute:
        print("\nList only — add --execute to run. Total ≈ 33 h wall time (≈ 24 h for the paper-relevant steps).")
        return
    if os.path.abspath(out) == ROOT:
        raise SystemExit("--from-raw needs a fresh --out folder (never the project itself)")
    if os.path.exists(os.path.join(out, "scripts")):
        raise SystemExit(f"{out}/scripts exists — use a fresh folder")
    shutil.copytree(SCRIPT_DIR, os.path.join(out, "scripts"), ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(os.path.join(ROOT, "data", "raw"), os.path.join(out, "data", "raw"))
    cwd = os.path.join(out, "scripts"); log = os.path.join(out, "run_all_from_raw.log")
    for sid, argv, rt, paper in RAW_STEPS:
        print(f"[{sid}] expected {rt}", flush=True)
        if argv[0] == "<internal:g1b_droplist>":
            g1b_droplist(out)
        elif argv[0] == "<internal:natural_earth>":
            fetch_natural_earth(out)
        else:
            run(argv, cwd, log)
        if until and sid == until:
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-cache", action="store_true"); g.add_argument("--from-raw", action="store_true")
    g.add_argument("--check", action="store_true"); g.add_argument("--audit", action="store_true")
    g.add_argument("--diff", default=None, metavar="OLD_JSON")
    g.add_argument("--fetch-natural-earth", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--until", default=None)
    ap.add_argument("--diff-title", default=None, help="title of the numbers diff (default: the v1.0 freeze diff)")
    a = ap.parse_args()
    if a.from_cache:
        from_cache(a.out or ROOT, not a.no_check)
    elif a.from_raw:
        if not a.out:
            raise SystemExit("--from-raw needs --out")
        from_raw(a.out, a.execute, a.until)
    elif a.diff:
        diff(os.path.abspath(a.diff), a.diff_title)
    elif a.fetch_natural_earth:
        fetch_natural_earth(os.path.abspath(a.out) if a.out else ROOT)
    elif a.audit:
        if not a.out:
            raise SystemExit("--audit needs --out")
        sys.exit(0 if audit(os.path.abspath(a.out)) else 1)
    else:
        if not a.out:
            raise SystemExit("--check needs --out")
        sys.exit(0 if check(os.path.abspath(a.out)) else 1)


if __name__ == "__main__":
    main()
