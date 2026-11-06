#!/usr/bin/env python3
"""v2.0 A8 (decision log v2.0-0): every logged expectation (E1–E21, X1–X6, S1–S4) with the commit that first added it
(git log -S on its defining token in docs/decision_log.md), the commit date, the counting rule (text of the log) and the
recorded outcome (with the report it is taken from) → results/v2_0/T_alllog.{csv,md} (Table S5 of WRR v5).

    python 34_alllog_v20.py            table
    python 34_alllog_v20.py archive    git bundle --all + git archive zip → release_v2.0/ (git-ignored; not uploaded)
"""
import csv
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "docs", "decision_log.md")

# id → (defining token in the log, outcome, source of the outcome)
ITEMS = [
    ("E1", "| **E1** |", "does not hold (T80a median 15.13 Ω·m, outside [2.6, 12.5])", "docs/decision_log.md v1.1-3"),
    ("E2", "| **E2** |", "holds (15.13 ≥ 15 Ω·m; margin 0.13; one of five seeds below 15)", "docs/decision_log.md v1.1-3"),
    ("E3", "| **E3** |", "does not hold (BEL median inside the transition at 3 of 5 positions)", "docs/decision_log.md v1.1-3"),
    ("E4", "| **E4** |", "descriptive: T80b − T80a = +0.67 Ω·m", "docs/decision_log.md v1.1-3"),
    ("E5", "| **E5** |", "holds (lens: spread 76.6 vs P0 width 43.2 pp)", "results/paper/numbers.md v11_E5"),
    ("E9", "| **E9** |", "holds (35/35 cases)", "docs/decision_log.md v1.3-1"),
    ("E10", "| **E10** |", "does not hold (1 of 15 sharp truths)", "docs/decision_log.md v1.3-1"),
    ("E11", "| **E11** |", "does not hold (log-midpoint and max. gradient 17/22 = 77 % each)", "docs/decision_log.md v1.3-1"),
    ("E12", "| **E12** |", "holds (R² = 0.69, a = 0.621 [0.515, 0.747])", "docs/decision_log.md v1.3-1"),
    ("E13", "| **E13** |", "holds (median stacking error 0.13 %)", "results/v1_4/SUMMARY.md"),
    ("E14", "| **E14** |", "does not hold (opposite direction; odds ratio 0.56 [0.42, 0.72])", "results/v1_4/SUMMARY.md"),
    ("E15", "| **E15** |", "holds (6 of 6, DP2–DP7)", "results/v1_4/2_dp_comparison.md"),
    ("E16", "| **E16** |", "does not hold (1 of 4 estimable DPs below 12.45 Ω·m)", "results/v1_4/2_dp_comparison.md"),
    ("E17", "| **E17** |", "does not hold (0 of 3)", "results/v1_4/2_dp_comparison.md"),
    ("E18", "**E18.**", "does not hold, position-sensitive (2 of 5)", "results/v1_4/3_same_epoch.md"),
    ("E19", "**E19.**", "does not hold, position-sensitive (log-midpoint 1 vs 3 Ω·m 2 of 6)", "results/v1_4/3_same_epoch.md"),
    ("E20", "**E20.**", "holds (6 of 6)", "results/v1_4/3_same_epoch.md"),
    ("E21", "E21.** The DP", "holds (7 of 7 DPs without a fine-grained layer near the base)", "results/v1_4/2_dp_comparison.md"),
    ("X1", "**X1** (T1, L24)", "holds (3 of 5: DP4, DP5, DP6)", "results/v1_5/SUMMARY.md"),
    ("X2", "**X2** (T1 and T2, L24)", "holds (T1 5 of 5, T2 4 of 5)", "results/v1_5/SUMMARY.md"),
    ("X3", "**X3** (T3, L24)", "does not hold (1 of 5)", "results/v1_5/SUMMARY.md"),
    ("X4", "**X4**: T1 vs T2", "descriptive", "results/v1_5/SUMMARY.md"),
    ("X5", "**X5:**", "holds, narrowly (χ²/N 53.48 T1 vs 56.27 T3)", "results/v1_5/SUMMARY.md"),
    ("X6", "**X6:**", "median > 4.93 Ω·m: no (3.52); interval contains 10.89–14.80: yes; base ≥ 5 of 6: yes", "results/v1_5/SUMMARY.md"),
    ("S1", "S1 BEL4-24", "does not hold (better than PRIOR4-L24 and LIN-LM-24, not than LIN-3-24)", "results/v1_6/1_scores.md"),
    ("S2", "S2 BEL4-18", "holds (BEL4-18 and BEL3-18 skill > 0)", "results/v1_6/1_scores.md"),
    ("S3", "S3 BEL", "holds (held-out IS skill v3 +0.36, v4 +0.43)", "results/v1_6/1_scores.md"),
    ("S4", "S4 on r", "does not hold everywhere (8 of 14 paired comparisons)", "results/v1_6/1_scores.md"),
]
NEVER = ["E6", "E7", "E8"]


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def rule_text(lines, token):
    for i, ln in enumerate(lines):
        if token in ln:
            if ln.startswith("|") and token.startswith("|"):
                return i + 1, re.split(r"(?<!\\)\|", ln)[2].strip().replace("\\|", "|")
            k = ln.index(token) + len(token)
            rest = ln[k:]
            m = re.search(r"\*\*[EXS]\d{1,2}[.:*]", rest)
            return i + 1, (token.strip("*| ") + " " + rest[: m.start() if m else 600]).strip(" |")
    return None, None


def table():
    lines = open(LOG, encoding="utf-8").read().splitlines()
    out = []
    for eid, tok, outcome, src in ITEMS:
        ln, rule = rule_text(lines, tok)
        h = git("log", "-S", tok, "--reverse", "--format=%h|%cs|%s", "--", "docs/decision_log.md").splitlines()
        h0 = h[0].split("|", 2) if h else ["", "", ""]
        row = re.search(r"\| (v[\d.]+-\d+) \|", "\n".join(lines[max(0, (ln or 1) - 40): ln or 1])) if ln else None
        out.append(dict(id=eid, commit=h0[0], date=h0[1], commit_subject=h0[2], log_line=ln, rule=re.sub(r"\s+", " ", rule or ""),
                        outcome=outcome, outcome_source=src))
    for eid in NEVER:
        out.append(dict(id=eid, commit="", date="", commit_subject="", log_line="", rule="never assigned (numbering skips E6–E8)",
                        outcome="—", outcome_source=""))
    order = lambda r: (r["id"][0] != "E", r["id"][0], int(r["id"][1:]))
    out.sort(key=order)
    os.makedirs(os.path.join(ROOT, "results", "v2_0"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "v2_0", "T_alllog.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    L = ["# Table S5 — every logged expectation (v2.0 A8)", "",
         "Commit = first commit that added the expectation's defining text to `docs/decision_log.md` (`git log -S`). "
         "Rule = the logged text (abridged where it continues with implementation details). "
         "Outcome = as recorded in the report named. E6–E8 were never assigned.", "",
         "| ID | commit | date | rule | outcome | source |", "|---|---|---|---|---|---|"]
    for r in out:
        rule = r["rule"] if len(r["rule"]) <= 400 else r["rule"][:397] + "…"
        L.append(f"| {r['id']} | {r['commit']} | {r['date']} | {rule.replace('|', chr(92) + '|')} | {r['outcome']} | {r['outcome_source']} |")
    open(os.path.join(ROOT, "results", "v2_0", "T_alllog.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(f"{r['id']} {r['commit']} {r['date']}" for r in out))


def archive():
    d = os.path.join(ROOT, "release_v2.0"); os.makedirs(d, exist_ok=True)
    tag = git("describe", "--tags", "--always")
    subprocess.run(["git", "bundle", "create", os.path.join(d, f"westhoek_ert_{tag}.bundle"), "--all"], cwd=ROOT, check=True)
    subprocess.run(["git", "archive", "--format=zip", "-o", os.path.join(d, f"westhoek_ert_{tag}_source.zip"), "HEAD"], cwd=ROOT, check=True)
    print(os.listdir(d))


if __name__ == "__main__":
    archive() if sys.argv[1:] == ["archive"] else table()
