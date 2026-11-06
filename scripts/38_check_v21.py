#!/usr/bin/env python3
"""v2.1 checks (plotting only).

    python 38_check_v21.py data      plotted values of the four v2.1 figures vs the same figures rebuilt with the v2.0 code
                                     (36_figs_wrr_v5.py of tag v2.0, loaded from git) → Coral Draw/v2.1/_checks_data.json
    python 38_check_v21.py files     SVG/PDF checks of Coral Draw/v2.1 (text elements, glyph paths, fonts, images, clip paths)
    python 38_check_v21.py coral <before.json>   every file of "Coral Draw" that existed before has the same size and mtime
"""
import collections
import importlib.util
import json
import os
import re
import subprocess
import sys

os.environ.pop("WESTHOEK_READ_TRACE", None)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
V21 = os.path.join(ROOT, "Coral Draw", "v2.1")
FIGS = [("Figure5", "Fig09"), ("Figure6", "Fig11"), ("FigureS7", "FigS03"), ("FigureS12", "Fig10")]


def load_v20():
    src = subprocess.run(["git", "show", "v2.0:scripts/36_figs_wrr_v5.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    p = os.path.join(SCRIPT_DIR, "_v20_figs_tmp.py")
    open(p, "w", encoding="utf-8").write(src)
    try:
        spec = importlib.util.spec_from_file_location("figs_v20", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    finally:
        os.remove(p)
    return m


def s12_by_method(v20, pf, v21, byname):
    """Figure S12: (mean, lower, upper) per method label in panels a (IS) and b (CRPS), v2.0 vs v2.1."""
    import matplotlib.pyplot as plt
    import numpy as np

    def read(fig):
        ax0 = fig.axes[0]; labs = {round(t.get_position()[1], 6): t.get_text() for t in ax0.get_yticklabels()}
        clean = lambda s: re.sub(r"^\d+\s+", "", s).rstrip(" *†‡§").strip()
        res = {}
        for pan, ax in (("IS", fig.axes[0]), ("CRPS", fig.axes[1])):
            for ln in ax.lines:
                x, y = ln.get_xdata(), ln.get_ydata()
                if len(x) == 1:
                    res.setdefault(clean(labs[round(float(y[0]), 6)]), {})[pan] = round(float(x[0]), 6)
            for c in ax.collections:
                for seg in c.get_segments():
                    if len(seg) == 2 and seg[0][1] == seg[1][1] and seg[0][0] != seg[1][0]:
                        lab = clean(labs[round(float(seg[0][1]), 6)])
                        res[lab][pan + "_ci"] = [round(float(seg[0][0]), 6), round(float(seg[1][0]), 6)]
        return res
    f0, _ = pf.fig10(); v20.add_figS12(f0); a = read(f0); plt.close(f0)
    f1, _ = v21.fig_s12(); b = read(f1); plt.close(f1)
    out = {}
    for m in sorted(set(a) | set(b)):
        out[m] = dict(v20=a.get(m), v21=b.get(m), IS_same=(a.get(m) or {}).get("IS") == (b.get(m) or {}).get("IS") and
                      (a.get(m) or {}).get("IS_ci") == (b.get(m) or {}).get("IS_ci"),
                      CRPS_same=(a.get(m) or {}).get("CRPS") == (b.get(m) or {}).get("CRPS") and (a.get(m) or {}).get("CRPS_ci") == (b.get(m) or {}).get("CRPS_ci"))
        print("   S12", m, "IS same", out[m]["IS_same"], "CRPS same", out[m]["CRPS_same"], (a.get(m) or {}).get("CRPS"), "→", (b.get(m) or {}).get("CRPS"))
    return out


def cmd_data():
    import matplotlib.pyplot as plt
    import make_paper_figures as mpf
    import paper_figures as pf
    spec = importlib.util.spec_from_file_location("figs_v21", os.path.join(SCRIPT_DIR, "36_figs_wrr_v5.py"))
    v21 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v21)
    v20 = load_v20()
    pf.DATA_DIR = os.path.join(ROOT, "results", "figs_style", "data")
    byname = {n.split("_")[0]: (n, fn) for n, fn, _ in pf.FIGURES}
    s03 = pf.figS03_data(os.path.join(ROOT, "results", "figs_style"))
    out = {}
    for label, short in FIGS:
        # v2.0 build exactly as tag v2.0 did it
        if short == "FigS03":
            f0, _ = pf.figS03(*s03[:3]); v20.add_figS7(f0, s03[0], s03[1])
        elif short == "Fig10":
            f0, _ = pf.fig10(); v20.add_figS12(f0)
        else:
            f0, _ = byname[short][1]()
            (v20.add_fig5 if label == "Figure5" else v20.add_fig6)(f0)
        r0, _ = mpf.figure_rows(f0); plt.close(f0)
        f1, _, _ = v21.build(label, short, byname, s03)
        r1, _ = mpf.figure_rows(f1); plt.close(f1)
        key = lambda r: tuple(r[:6])
        s0, s1 = collections.Counter(map(tuple, r0)), collections.Counter(map(tuple, r1))
        only0, only1 = s0 - s1, s1 - s0
        ser = lambda c: sorted({f"{r[0]} | {r[1]}" for r in c.elements()})
        if short == "Fig10":
            out["FigureS12_by_method"] = s12_by_method(v20, pf, v21, byname)
        out[label] = dict(rows_v20=len(r0), rows_v21=len(r1), identical_rows=int(sum((s0 & s1).values())),
                          only_in_v20=int(sum(only0.values())), only_in_v21=int(sum(only1.values())),
                          series_changed_v20=ser(only0), series_changed_v21=ser(only1))
        print(label, {k: v for k, v in out[label].items() if not k.startswith("series")}, flush=True)
        for k in ("series_changed_v20", "series_changed_v21"):
            print("   ", k, out[label][k][:12])
    os.makedirs(V21, exist_ok=True)
    json.dump(out, open(os.path.join(V21, "_checks_data.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def cmd_files():
    out = {}
    for label, _ in FIGS:
        svg = open(os.path.join(V21, "SVG", label + ".svg"), encoding="utf-8").read()
        fams = sorted(set(re.findall(r"font-family[:=]\s*['\"]?([^;'\"]+)", svg)))
        r = dict(svg_text=len(re.findall(r"<text\b", svg)), svg_glyph_paths=len(re.findall(r'id="(?:DejaVu|STIX|Times)[^"]*-[0-9a-f]+"', svg)) +
                 len(re.findall(r"<use\b[^>]*xlink:href=\"#(?:DejaVu|STIX|Times)", svg)),
                 svg_font_families=[f.strip() for f in fams], svg_images=len(re.findall(r"<image\b", svg)), svg_clip_paths=len(re.findall(r"<clipPath\b", svg)),
                 svg_clip_uses=len(re.findall(r'clip-path="url', svg)))
        raw = open(os.path.join(V21, "PDF", label + ".pdf"), "rb").read()
        subs = re.findall(rb"/Subtype\s*/(Type0|Type1|Type3|TrueType|CIDFontType0|CIDFontType2|Image)", raw)
        r["pdf_font_types"] = sorted({x.decode() for x in subs if x != b"Image"})
        r["pdf_base_fonts"] = sorted({x.decode() for x in re.findall(rb"/BaseFont\s*/([A-Za-z0-9+_-]+)", raw)})
        r["pdf_type3"] = b"Type3" in subs; r["pdf_images"] = subs.count(b"Image")
        r["pdf_truetype_only"] = all(t in ("Type0", "CIDFontType2", "TrueType") for t in r["pdf_font_types"])
        out[label] = r
        print(label, r, flush=True)
    json.dump(out, open(os.path.join(V21, "_checks_files.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def cmd_coral(before):
    b = json.load(open(before, encoding="utf-8")); ch = []
    for p, v in b.items():
        q = os.path.join(ROOT, p)
        if not os.path.exists(q):
            ch.append((p, "missing")); continue
        st = os.stat(q)
        if st.st_size != v["size"] or st.st_mtime_ns != v["mtime_ns"]:
            ch.append((p, "changed"))
    new = [os.path.relpath(os.path.join(dp, f), ROOT) for dp, _, fs in os.walk(os.path.join(ROOT, "Coral Draw")) for f in fs
           if os.path.relpath(os.path.join(dp, f), ROOT) not in b]
    res = dict(n_before=len(b), changed_or_missing=ch, new_files=sorted(new), new_outside_v21=[n for n in new if not n.replace("\\", "/").startswith("Coral Draw/v2.1/")])
    json.dump(res, open(os.path.join(V21, "_checks_coral.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(b)} files before; changed or missing: {len(ch)}; new: {len(new)} (outside v2.1: {len(res['new_outside_v21'])})")


def cmd_readme():
    J = lambda n: json.load(open(os.path.join(V21, n), encoding="utf-8")) if os.path.exists(os.path.join(V21, n)) else None
    D, F, C, L, K = J("_checks_data.json"), J("_checks_files.json"), J("_checks_coral.json"), J("figs_v21_corel_log.json"), J("_cdr_report.json")
    old = {"Figure5": "Figure5", "Figure6": "Figure7", "FigureS7": "FigureS7", "FigureS12": "Figure6"}
    R = ["# CorelDRAW files v2.1 — the four figures changed in v2.0", "",
         "Plotting only (no computation; `results/paper/numbers.md` and `results/v2_0/numbers_v20.md` unchanged). WRR v6 names. Made by "
         "`scripts/36_figs_wrr_v5.py --corel` (CorelDRAW mode of v1.10: all vectors, named groups, live text, Times New Roman, hatches as "
         "separate layers, clip paths removed where the data lie inside the axes, exact print size 190 mm wide); checks by "
         "`scripts/38_check_v21.py`; `.cdr` through COM by `scripts/37_corel_v21.py build`. Nothing else in `Coral Draw/` was opened or written.", "",
         "| v6 file | your earlier edit (v3 numbering) | source function | size (mm) |", "|---|---|---|---|"]
    for f in old:
        R.append(f"| {f} | `CDR/{old[f]}.cdr` | `{L[f]['function']}` | {L[f]['size_mm'][0]} × {L[f]['size_mm'][1]} |")
    R += ["", "## Folders", "", "| folder | content |", "|---|---|",
          "| `CDR/` | native CorelDRAW files (see the status below) |", "| `SVG/` | for CorelDRAW File → Import |", "| `PDF/` | vector PDF |",
          "| `preview/` | 300 dpi PNG; `contact_sheet.png` |", "", "## Changes against v2.0 (step 1, v2.0.1 fixes) and style (step 2)", ""]
    R += [f"- **{f}:** {L[f]['note']}" for f in old]
    R += ["", "## Checks", "", "### Data unchanged", "",
          "Plotted values compared with the same figure rebuilt with the v2.0 code (tag v2.0; v2.0 wrote no CSV for these figures):", ""]
    for f in old:
        d = D[f]; R.append(f"- {f}: {d['identical_rows']} of {d['rows_v20']} v2.0 rows identical; only in v2.0: {d['only_in_v20']}, only in v2.1: {d['only_in_v21']}.")
    s12 = D.get("FigureS12_by_method", {})
    R += ["", "Figure S12 by method: IS unchanged for every row; CRPS unchanged for BEL and priors; changed for the six linearised bands "
          "(now from samples): " + ", ".join(m for m, v in s12.items() if v["v20"] and not v["CRPS_same"]) + "; new row: "
          + ", ".join(m for m, v in s12.items() if not v["v20"]) + ". Figure 6: the only removed series is the grey 8 Ω·m line (the five "
          "DP lines keep their values; their automatic names shift by one).", "",
          "### Text, fonts, images, clip paths", "",
          "| figure | SVG `<text>` | glyph paths | font families (SVG) | PDF font types | PDF Type 3 | images SVG / PDF | clip defs / uses (→ PowerClips) | smallest text (pt) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f in old:
        x = F[f]
        R.append(f"| {f} | {x['svg_text']} | {x['svg_glyph_paths']} | {', '.join(x['svg_font_families'])} | {', '.join(x['pdf_font_types'])} | "
                 f"{'yes' if x['pdf_type3'] else 'no'} | {x['svg_images']} / {x['pdf_images']} | {x['svg_clip_paths']} / {x['svg_clip_uses']} | {L[f]['smallest_pt']} |")
    R += ["", "Remaining clip paths: Figure 5 — the 2025 image column in each of the 7 panels (it continues below −36 mTAW on a log axis, "
          "which the line cutter does not handle); Figure 6 — the two 90 % bands of panel (b), which rise above +5 mTAW. Same as v1.10. "
          "Figure S7: none (v1.10 had 53: open markers kept their clip because matplotlib reports no extent for them; v2.1 tests the marker "
          "positions directly).", "", "### CorelDRAW (.cdr)", ""]
    if K:
        R += ["| figure | text objects after reopening | curves | bitmaps | PowerClips | fonts | .cdr MB |", "|---|---|---|---|---|---|---|"]
        R += [f"| {f} | {v['reopen_counts']['text']} | {v['reopen_counts']['curves']} | {v['reopen_counts']['bitmaps']} | {v['reopen_counts']['powerclips']} | "
              f"{', '.join(v['reopen_counts']['fonts'])} | {v['mb']} |" for f, v in K.items()]
    else:
        R += ["**Not made yet.** CorelDRAW 2023's COM server does not start on this computer (`CO_E_SERVER_EXEC_FAILURE`; started directly, "
              "CorelDRAW runs but does not register for automation — most likely a start-up dialog; a Corel Update Manager update failed "
              "at 19:06 on 21 October 2026). Once CorelDRAW opens normally, `python scripts/37_corel_v21.py build \"Coral Draw/v2.1/SVG\" "
              "\"Coral Draw/v2.1/CDR\" \"Coral Draw/v2.1/_cdr_report.json\"` makes the four `.cdr` files and this table; until then, "
              "File → Import the SVGs."]
    R += ["", "### Your files", "", f"`Coral Draw/` before the run: {C['n_before']} files; changed or missing afterwards: {len(C['changed_or_missing'])}; "
          f"new files outside `Coral Draw/v2.1/`: {len(C['new_outside_v21'])}. Your `.cdr` files were read only as copies in a temporary folder.", "",
          "### Style taken from your edits (step 2)", "",
          "Read from copies of your four `.cdr` files (their embedded page previews and text inventories; COM was not available) against "
          "the v1.10 originals from git.", "",
          "- **Applied:** legend below the panels in a thin black dotted frame with square corners, spanning the figure width; "
          "Figure 5 image line dark grey dashed (as v1.9.1); page size unchanged (your previews have the v1.10 aspect ratios).",
          "- **Not reproduced:** the more compact legend layouts (e.g. Figure S7 in two rows, Figure 7/v6 Figure 6 in three rows of four): "
          "at ≥ 7 pt the long entries need three columns within 190 mm; font sizes, line widths and colours of your edits could not be read "
          "without COM (the previews are 256 px wide) and are left as in v1.10; the label boxes of \"dunes: weakly constrained\" and "
          "\"V.E. 3×\" do not occur in these four figures.", ""]
    open(os.path.join(V21, "README_v2.1.md"), "w", encoding="utf-8").write("\n".join(R) + "\n")
    print("wrote README_v2.1.md")


if __name__ == "__main__":
    if sys.argv[1] == "readme":
        cmd_readme(); sys.exit()
    {"data": cmd_data, "files": cmd_files, "coral": cmd_coral}[sys.argv[1]](*sys.argv[2:])
