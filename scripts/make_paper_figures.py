"""
make_paper_figures.py
=====================
Westhoek ERT v1.8 — redraw the 21 paper figures in the paper style from the saved files of tag v1.7.1 (plotting only).

  python make_paper_figures.py                     all figures → results/figs_style/ (PNG + RGB TIFF at 1000 dpi, PDF,
                                                   300 dpi draft PNG in draft/), then the checks → results/figs_style/checks.json,
                                                   inputs_sha256_after.txt, protected_sha256_{before,after}.txt, contact_sheet.png
  python make_paper_figures.py --only Fig09_ERT2025AtSoundings --draft-only --out <dir>
                                                   quick look: draft PNG only, no checks

Checks (full run): inputs unchanged (SHA-256 vs inputs_sha256_before.txt of Stage 1); numbers.md, captions.md, tables.md
and all table CSVs unchanged; no file under results/ (except results/figs_style/), data/processed/ or figures/ created,
changed or removed; no number registered by importing the analysis scripts; values of Figs. 3, 7, 9 and 10 against
results/paper/numbers.md and Table 3 of the draft; smallest font ≥ 7 pt and nothing clipped at the figure edge.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import zipfile

os.environ.pop("WESTHOEK_READ_TRACE", None)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
OUT = os.path.join(ROOT, "results", "figs_style")
DRAFT_DOCX = os.path.join(ROOT, "Claude outputs", "Westhoek_paper1_full_draft_v3.docx")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def protected_files():
    fs = [os.path.join(ROOT, "results", "paper", n) for n in ("numbers.md", "captions.md", "tables.md")]
    fs += [p for p in sorted(glob.glob(os.path.join(ROOT, "results", "**", "*.csv"), recursive=True))
           if not rel(p).startswith("results/figs_style/")]
    return fs


def snapshot():
    snap = {}
    for top in ("results", os.path.join("data", "processed"), "figures"):
        for d, dirs, files in os.walk(os.path.join(ROOT, top)):
            r = rel(d)
            if r.startswith("results/figs_style"):
                continue
            snap[r + "/"] = ("dir",)
            for f in files:
                p = os.path.join(d, f); st = os.stat(p)
                snap[rel(p)] = (st.st_size, st.st_mtime_ns)
    return snap


def numbers_md():
    out = {}
    for line in open(os.path.join(ROOT, "results", "paper", "numbers.md"), encoding="utf-8"):
        m = re.match(r"\| \d+ \| `([^`]+)` \| (.*?) \| ", line)
        if m:
            out[m.group(1)] = m.group(2).replace("\\|", "|")
    return out


def table3():
    """Table 3 of the draft (python-docx-free): DP → list of the eight cells after the DP name."""
    x = zipfile.ZipFile(DRAFT_DOCX).read("word/document.xml").decode("utf-8")
    paras = ["".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", q)) for q in re.findall(r"<w:p[ >].*?</w:p>", x, flags=re.S)]
    i = next(k for k, p in enumerate(paras) if p.startswith("Table 3."))
    rows, j = {}, i
    while j < len(paras) and len(rows) < 7:
        if re.fullmatch(r"DP[1-7]", paras[j]):
            rows[paras[j]] = paras[j + 1:j + 9]; j += 9
        else:
            j += 1
    return rows


def fmt3(b, share=False):
    s = f"{b[1]:.2f} [{b[2]:.2f}, {b[3]:.2f}]" + (f" ({100*b[0]:.0f}%)" if share else "")
    return s


def spot_checks(meta, pf):
    N = numbers_md(); out = {}
    s18 = pf.mod("s18")
    if "Fig03_LinearisedLensBase" in meta:
        m = meta["Fig03_LinearisedLensBase"]["pooled_logmid"]
        out["Fig03"] = [dict(key=f"lin_logmid_pooled_x{x}", plotted=fmt3(m[x], True), numbers=N.get(f"lin_logmid_pooled_x{x}"),
                             ok=fmt3(m[x], True) == N.get(f"lin_logmid_pooled_x{x}")) for x in (150, 200, 250, 300, 350)]
    if "Fig07_FieldLensBase" in meta:
        Pf, TX = meta["Fig07_FieldLensBase"]["Pf"], meta["Fig07_FieldLensBase"]["TX"]
        import numpy as np
        out["Fig07"] = []
        for x in (150, 200, 250, 300, 350):
            v = s18.q(Pf[:, int(np.argmin(np.abs(TX - x)))])
            out["Fig07"].append(dict(key=f"bel_base_x{x}", plotted=v, numbers=N.get(f"bel_base_x{x}"), ok=v == N.get(f"bel_base_x{x}")))
    if "Fig09_ERT2025AtSoundings" in meta:
        T3 = table3(); out["Fig09"] = []; out["Fig09_median_notes"] = []
        for nm, v in meta["Fig09_ERT2025AtSoundings"].items():
            for k, col in (("logmid", 5), ("thr3", 6)):
                # Fig. 9 draws the P5–P95 bands only (no median line): the drawn values are P5 and P95
                b = v[k]; t = T3[nm][col]
                lims = f"[{b[2]:.2f}, {b[3]:.2f}]".replace("-", "−"); t_lims = t[t.index("["):]
                out["Fig09"].append(dict(key=f"{nm} {k} P5–P95 (Table 3 col '{('linearised log-midpoint 2025', 'linearised 3 Ω·m 2025 (λ = 50)')[col - 5]}')",
                                         plotted=lims, table3=t_lims, ok=lims == t_lims))
                med = f"{b[1]:.2f}".replace("-", "−"); t_med = t[:t.index(" [")]
                if med != t_med:
                    out["Fig09_median_notes"].append(f"{nm} {k}: band median {med} vs Table 3 central value {t_med} (median not drawn in Fig. 9)")
    if "Fig10_Scores" in meta:
        FS = meta["Fig10_Scores"]; out["Fig10"] = []
        ci = lambda o, k: f"{o[k]['mean']:.2f} [{o[k]['lo']:.2f}, {o[k]['hi']:.2f}]"
        sk = lambda o, k: f"{o[k]['skill']:+.2f} [{o[k]['skill_lo']:+.2f}, {o[k]['skill_hi']:+.2f}]"
        for m in ("BEL3-18", "BEL4-18", "BEL4-18s", "BEL4-24"):
            o = FS[m]
            v = (f"IS {ci(o, 'IS')}, skill {sk(o, 'IS')}; CRPS {ci(o, 'CRPS')}, skill {sk(o, 'CRPS')}; "
                 f"{o['cov']} of {o['n']} inside; sharpness {o['width']:.2f} m")
            key = f"v16_field_{m.replace('-', '_')}"
            out["Fig10"].append(dict(key=key, plotted=v, numbers=N.get(key), ok=v == N.get(key)))
        bel = [FS[m] for m in ("BEL3-18", "BEL4-18", "BEL4-18s", "BEL4-24")]
        v = f"{FS['PRIOR4']['width']:.2f} m (PRIOR4); BEL {min(o['width'] for o in bel):.2f}–{max(o['width'] for o in bel):.2f} m"
        out["Fig10"].append(dict(key="v16_field_prior_sharpness", plotted=v, numbers=N.get("v16_field_prior_sharpness"),
                                 ok=v == N.get("v16_field_prior_sharpness")))
        out["Fig10_summary"] = dict(bel_IS_skill=f"{min(o['IS']['skill'] for o in bel):+.2f} to {max(o['IS']['skill'] for o in bel):+.2f}",
                                    bel_width=f"{min(o['width'] for o in bel):.2f}–{max(o['width'] for o in bel):.2f} m",
                                    prior_width=f"{min(FS[m]['width'] for m in FS if m.startswith('PRIOR')):.2f}–{max(FS[m]['width'] for m in FS if m.startswith('PRIOR')):.2f} m",
                                    bel4_24_inside=f"{FS['BEL4-24']['cov']} of {FS['BEL4-24']['n']}",
                                    v15_bel_l24_base_in=N.get("v15_bel_l24_base_in"))
    return out


def contact_sheet(names, out):
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    W, cols, pad = 900, 3, 30
    tiles = []
    for n in names:
        im = Image.open(os.path.join(OUT, "draft", f"{n}.png")).convert("RGB")
        im = im.resize((W, int(im.size[1] * W / im.size[0])))
        t = Image.new("RGB", (W, im.size[1] + 40), "white"); t.paste(im, (0, 40)); ImageDraw.Draw(t).text((5, 8), n, fill="black")
        tiles.append(t)
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    H = sum(max(t.size[1] for t in r) + pad for r in rows)
    sheet = Image.new("RGB", (cols * (W + pad), H), "white"); y = 0
    for r in rows:
        for i, t in enumerate(r):
            sheet.paste(t, (i * (W + pad), y))
        y += max(t.size[1] for t in r) + pad
    sheet.save(out, dpi=(150, 150))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated figure names")
    ap.add_argument("--draft-only", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fig9-color", default="#4D4D4D", help="image column of Fig. 9 (tomato was too close to the 3 Ω·m band)")
    a = ap.parse_args()
    full = not a.only and not a.draft_only and os.path.abspath(a.out) == os.path.abspath(OUT)
    os.makedirs(a.out, exist_ok=True)
    if full:
        with open(os.path.join(OUT, "protected_sha256_before.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines(f"{sha(p)}  {rel(p)}\n" for p in protected_files())
        snap0 = snapshot()
    t_all = time.time()
    import matplotlib.pyplot as plt
    import paper_figures as pf
    import ert_plot_style as eps
    pf.DATA_DIR = os.path.join(a.out, "data") if full else None
    only = set(a.only.split(",")) if a.only else None
    res, meta = {}, {}
    for name, fn, old in pf.FIGURES:
        if only and name not in only:
            continue
        t0 = time.time()
        if name == "FigS03_BiasRule":
            if a.draft_only:
                G_ = pf.g25_rows(); g = G_["g"]
                fit = pf.quiet(g.fit_rule, G_["rows"], "lam50", seed=[136200])
                fit_lam = {int(l): pf.quiet(g.fit_rule, G_["rows"], f"lam{int(l)}_r01", seed=[136201]) for l in [5.0, 10.0, 20.0, 50.0, 100.0]}
                fit_grad = pf.quiet(g.fit_rule, G_["rows"], "lam50", truths=[t for t in g.TRUTHS if g.params(t)["tau"] > 0], seed=[136200])
                info = {}
            else:
                fit, fit_lam, fit_grad, info = pf.figS03_data(a.out)
            fig, m = pf.figS03(fit, fit_lam, fit_grad); m = dict(fit_check=info)
        elif name == "Fig09_ERT2025AtSoundings":
            fig, m = pf.fig09(img_col=a.fig9_color)
        else:
            fig, m = fn()
        if a.draft_only:
            smallest, clipped = eps.layout_checks(fig)
            os.makedirs(os.path.join(a.out, "draft"), exist_ok=True)
            fig.savefig(os.path.join(a.out, "draft", f"{name}.png"), dpi=300); files = []
        else:
            files, smallest, clipped = eps.savefig(fig, name, a.out, 190)
        w, h = fig.get_size_inches() * 25.4
        plt.close(fig)
        res[name] = dict(old=old, size_mm=[round(w, 1), round(h, 1)], smallest_pt=smallest, clipped=clipped,
                         seconds=round(time.time() - t0, 1), files=[rel(p) for p in files],
                         bytes={os.path.splitext(p)[1]: os.path.getsize(p) for p in files})
        meta[name] = m
        print(f"[fig] {name}: {w:.0f}×{h:.0f} mm, smallest {smallest:.2f} pt, clipped {len(clipped)}, {time.time() - t0:.0f} s", flush=True)
    if not full:
        return
    checks = dict(figures=res, seconds=round(time.time() - t_all))
    before = [l.split("  ", 1) for l in open(os.path.join(OUT, "inputs_sha256_before.txt"), encoding="utf-8").read().splitlines() if l]
    after = [(sha(os.path.join(ROOT, p)), p) for _, p in before]
    with open(os.path.join(OUT, "inputs_sha256_after.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{h}  {p}\n" for h, p in after)
    checks["inputs_identical"] = [(h, p) for h, p in before] == list(after)
    checks["n_inputs"] = len(before)
    with open(os.path.join(OUT, "protected_sha256_after.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{sha(p)}  {rel(p)}\n" for p in protected_files())
    checks["protected_identical"] = (open(os.path.join(OUT, "protected_sha256_before.txt"), encoding="utf-8").read()
                                     == open(os.path.join(OUT, "protected_sha256_after.txt"), encoding="utf-8").read())
    checks["n_protected"] = len(protected_files())
    snap1 = snapshot()
    checks["side_effects"] = dict(created=sorted(set(snap1) - set(snap0)), removed=sorted(set(snap0) - set(snap1)),
                                  changed=sorted(k for k in snap0 if k in snap1 and snap0[k] != snap1[k]))
    checks["numbers_registered_by_imports"] = len(pf.mod("s18").NUM.rows)
    checks["spot"] = spot_checks(meta, pf)
    checks["figS03_fit"] = meta.get("FigS03_BiasRule", {}).get("fit_check")
    checks["fig9_image_colour"] = a.fig9_color
    contact_sheet([n for n, _, _ in pf.FIGURES], os.path.join(OUT, "contact_sheet.png"))
    json.dump(checks, open(os.path.join(OUT, "checks.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    ok = (checks["inputs_identical"] and checks["protected_identical"] and not any(checks["side_effects"].values())
          and checks["numbers_registered_by_imports"] == 0
          and all(r["ok"] for k, v in checks["spot"].items() if isinstance(v, list) for r in v if isinstance(r, dict))
          and all(r["smallest_pt"] >= 7.0 - 1e-9 and not r["clipped"] for r in res.values()))
    print(f"[checks] all passed: {ok}  ({checks['seconds']} s)")


if __name__ == "__main__":
    main()
