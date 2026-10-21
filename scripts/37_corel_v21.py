#!/usr/bin/env python3
"""v2.1 (plotting only): CorelDRAW helpers through COM (CorelDRAW 2023, ProgID CorelDRAW.Application.25).

    python 37_corel_v21.py inspect <dir> <out.json>    dump the shape tree (type, name, box, outline, fill, text font/size) of every
                                                        .cdr in <dir> — used on *copies* of the author's files only
    python 37_corel_v21.py build <svg_dir> <cdr_dir> <report.json>
                                                        import each SVG on a page of its size, save as .cdr, reopen and count;
                                                        render each reopened .cdr to PNG for the contact sheet

Files are opened read-only where possible; nothing is ever saved back to an input path.
"""
import json
import os
import sys
import time

MM = 3                     # cdrMillimeter
TYPES = {1: "rectangle", 2: "ellipse", 3: "curve", 4: "polygon", 5: "bitmap", 6: "text", 7: "group", 8: "selection", 9: "guideline",
         10: "blend", 11: "extrude", 12: "ole", 13: "contour", 14: "linear dim", 15: "bevel", 16: "drop shadow", 17: "3D", 18: "artistic media",
         19: "connector", 20: "mesh fill", 21: "perfect shape", 22: "custom", 23: "symbol", 24: "html form", 25: "web control", 26: "EPS"}


def app():
    import win32com.client
    a = win32com.client.Dispatch("CorelDRAW.Application.25")
    a.Visible = False
    try:
        a.Optimization = True
    except Exception:
        pass
    return a


def g(f, default=None):
    try:
        return f()
    except Exception:
        return default


def shape_info(s, depth=0):
    t = g(lambda: s.Type, 0)
    o = dict(type=TYPES.get(t, t), name=g(lambda: s.Name, ""), x=g(lambda: round(s.LeftX, 3)), y=g(lambda: round(s.BottomY, 3)),
             w=g(lambda: round(s.SizeWidth, 3)), h=g(lambda: round(s.SizeHeight, 3)))
    ol = g(lambda: s.Outline)
    if ol is not None and g(lambda: ol.Type) not in (None, 0):
        o["outline"] = dict(width=g(lambda: round(ol.Width, 4)), color=g(lambda: ol.Color.HexValue), dashes=g(lambda: ol.Style.DashCount),
                            style=g(lambda: [round(ol.Style.GetDashAndGapLength(i + 1)[0], 3) for i in range(ol.Style.DashCount)] if ol.Style.DashCount else []),
                            corner=g(lambda: ol.LineJoin))
    f = g(lambda: s.Fill)
    if f is not None:
        ft = g(lambda: f.Type)
        o["fill"] = dict(type=ft, color=g(lambda: f.UniformColor.HexValue) if ft == 1 else None)
        tr = g(lambda: s.Transparency.Uniform) if g(lambda: s.Transparency.Type) else None
        if tr:
            o["fill"]["transparency"] = tr
    if t == 6:
        st = g(lambda: s.Text.Story)
        o["text"] = dict(text=g(lambda: st.Text, ""), font=g(lambda: st.Font), size=g(lambda: round(st.Size, 2)), bold=g(lambda: bool(st.Bold)))
    if g(lambda: s.PowerClip) is not None and g(lambda: s.PowerClip.Shapes.Count, 0):
        o["powerclip"] = [shape_info(c, depth + 1) for c in s.PowerClip.Shapes]
    if t == 7:
        o["children"] = [shape_info(c, depth + 1) for c in s.Shapes]
    return o


def cmd_inspect(d, out):
    a = app(); res = {}
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(".cdr"):
            continue
        t0 = time.time()
        doc = a.OpenDocument(os.path.join(d, fn))
        doc.Unit = MM
        pg = doc.ActivePage
        res[fn] = dict(page_mm=[round(pg.SizeWidth, 3), round(pg.SizeHeight, 3)], n_pages=doc.Pages.Count,
                       layers=[dict(name=l.Name, n=l.Shapes.Count) for l in pg.Layers],
                       shapes=[shape_info(s) for l in pg.Layers for s in l.Shapes], seconds=None)
        doc.Close()
        res[fn]["seconds"] = round(time.time() - t0, 1)
        print(f"[inspect] {fn}: page {res[fn]['page_mm']}, {sum(x['n'] for x in res[fn]['layers'])} top shapes, {res[fn]['seconds']} s", flush=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def walk(shapes):
    for s in shapes:
        yield s
        for k in ("children", "powerclip"):
            if k in s:
                yield from walk(s[k])


def count(doc):
    n = dict(text=0, curves=0, bitmaps=0, powerclips=0, fonts=set(), groups=0)
    def rec(shapes):
        for s in shapes:
            t = g(lambda: s.Type, 0)
            if t == 6:
                n["text"] += 1; n["fonts"].add(g(lambda: s.Text.Story.Font))
            elif t == 5:
                n["bitmaps"] += 1
            elif t == 7:
                n["groups"] += 1; rec(s.Shapes)
            else:
                n["curves"] += 1
            pc = g(lambda: s.PowerClip)
            if pc is not None and g(lambda: pc.Shapes.Count, 0):
                n["powerclips"] += 1; rec(pc.Shapes)
    for l in doc.ActivePage.Layers:
        rec(l.Shapes)
    n["fonts"] = sorted(x for x in n["fonts"] if x)
    return n


def cmd_build(svg_dir, cdr_dir, report):
    a = app(); res = {}
    os.makedirs(cdr_dir, exist_ok=True)
    for fn in sorted(os.listdir(svg_dir)):
        if not fn.endswith(".svg"):
            continue
        name = fn[:-4]; t0 = time.time()
        doc = a.CreateDocument(); doc.Unit = MM
        doc.ActiveLayer.Import(os.path.join(svg_dir, fn))
        sel = doc.ActivePage.Shapes.All()
        w, h = sel.SizeWidth, sel.SizeHeight
        doc.ActivePage.SetSize(w, h); sel.SetPosition(0, h)
        first = count(doc)
        p = os.path.join(cdr_dir, name + ".cdr")
        doc.SaveAs(p); doc.Close()
        doc = a.OpenDocument(p); doc.Unit = MM
        again = count(doc)
        png = os.path.join(cdr_dir, "_render_" + name + ".png")
        try:
            ex = doc.ExportBitmap(png, 802, 0, 3, int(round(190 / 25.4 * 150)), 0, 150, 150)   # cdrPNG, all, RGB, 150 dpi
            ex.Finish()
        except Exception as e:
            png = f"render failed: {e}"
        page = [round(doc.ActivePage.SizeWidth, 2), round(doc.ActivePage.SizeHeight, 2)]
        doc.Close()
        res[name] = dict(import_counts=first, reopen_counts=again, page_mm=page, mb=round(os.path.getsize(p) / 1e6, 2), render=png,
                         seconds=round(time.time() - t0, 1))
        print(f"[build] {name}: {again}, page {page}, {res[name]['seconds']} s", flush=True)
    json.dump(res, open(report, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=list)


if __name__ == "__main__":
    {"inspect": cmd_inspect, "build": cmd_build}[sys.argv[1]](*sys.argv[2:])
