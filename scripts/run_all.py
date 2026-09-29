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
    ("19", ["19_rho_lens_bound_check.py"], "3 min", False),
    ("18", ["18_paper.py", "all"], "7 min", True),
]


# ─────────────────────────────────────────────────────────────────────────────
# provenance: which --from-raw step writes each file (pattern on the project-relative path)
# ─────────────────────────────────────────────────────────────────────────────
RUN_STEP = {"deep_all": "02-F", "deep_auto": "02-Fauto", "deep_drop_data": "02-G1", "deep_drop_elec": "02-G2", "deep_downweight": "02-G3",
            "g1_lam5": "02-lam5", "g1_lam10": "02-lam10", "g1_lam50": "02-lam50", "g1_lam100": "02-lam100",
            "g1b_thr299": "02-G1b"}
PRODUCERS = [   # (regex, step id or callable(match) → step id, script, evidence token that must occur in the script)
    (r"^data/raw/", "raw input", None, None),
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
    for line in open(trace, encoding="utf-8"):
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
    for sub in (os.path.join("results", "stage6_v2"), os.path.join("results", "paper")):
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


def diff(old_json):
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
    L = ["# Numbers diff — v1.0 freeze (physical filter)", "",
         f"Old: `{os.path.relpath(old_json, ROOT)}`; new: `results/paper/numbers.json`. {len(changed)} changed, "
         f"{len(added)} added, {len(removed)} removed, {len(same)} unchanged.", "",
         "## Changed", "", "| key | quantity | old | new | why |", "|---|---|---|---|---|"]
    L += [f"| `{k}` | {esc(new[k]['desc'])} | {esc(old[k]['value'])} | {esc(new[k]['value'])} | {why(k)} |" for k in changed]
    L += ["", "## Added", "", "| key | quantity | value | why |", "|---|---|---|---|"]
    L += [f"| `{k}` | {esc(new[k]['desc'])} | {esc(new[k]['value'])} | {why(k)} |" for k in added]
    chk = os.path.join(ROOT, "results", "paper", "rho_lens_bound_check.md")
    if os.path.exists(chk):
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
        if os.path.exists(trace):
            os.remove(trace)
        os.environ["WESTHOEK_READ_TRACE"] = trace
    print(f"[from-cache] → {out} (log {log})")
    t = {}
    t["17"] = run(["17_salinity_v2.py", "run", "--out", out], SCRIPT_DIR, log)
    t["18"] = run(["18_paper.py", "all", "--out", out, "--recompute"], SCRIPT_DIR, log)
    print(f"[from-cache] done: 17 {t['17']:.0f} s, 18 {t['18']:.0f} s, total {sum(t.values())/60:.1f} min")
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
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--until", default=None)
    a = ap.parse_args()
    if a.from_cache:
        from_cache(a.out or ROOT, not a.no_check)
    elif a.from_raw:
        if not a.out:
            raise SystemExit("--from-raw needs --out")
        from_raw(a.out, a.execute, a.until)
    elif a.diff:
        diff(os.path.abspath(a.diff))
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
