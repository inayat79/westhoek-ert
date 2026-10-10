"""
32_location_map.py
==================
Westhoek ERT v1.7 — Fig. 1 location map (overview, coastal relief, site map on aerial imagery, profile strip).

  fetch            download the open base-map data once into data/external/basemap/ and write README_sources.md
                   (Copernicus DEM GLO-30 COG tiles, EMODnet Bathymetry DTM via WCS, Digitaal Vlaanderen orthophoto via WMS;
                   Natural Earth 1:10m is already in data/external/natural_earth/)
  build [--out ROOT]   draw ROOT/results/figs/F_01_location_map.{png,tif,pdf} and F_01_caption_draft.md from the cache only
                   (offline); checks → ROOT/data/processed/v1_7/location_map.json

Coordinates with pyproj: EPSG:31370 (Belgian Lambert 72) for (b)–(d), EPSG:3035 for (a), WGS84 for graticule labels.
Panel (a) uses cartopy; (b)–(c) use matplotlib on pyproj-projected data (cartopy's EPSG:31370 projection is limited to
the Belgian area of use and would clip Dunkerque).
"""

import argparse
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import read_trace  # noqa: E402
read_trace.install("32_location_map.py")
import westhoek_common as wc  # noqa: E402

BASE = os.path.join(wc.ROOT, "data", "external", "basemap")
NE_DIR = os.path.join(wc.ROOT, "data", "external", "natural_earth")
# panel extents (EPSG:31370, m)
EXT_B = (7000.0, 39500.0, 192500.0, 207000.0)
EXT_C = (22690.0, 23430.0, 199560.0, 200300.0)
GRID_B = 50.0
ORTHO_RES = 0.25
ORTHO_LAYER = "OMWRGB25VL"
WMS_OMW = "https://geo.api.vlaanderen.be/OMW/wms"
WCS_EMOD = "https://ows.emodnet-bathymetry.eu/wcs"
EMOD_COV = "emodnet__mean"
EMOD_BOX = dict(lat=(50.95, 51.30), lon=(2.15, 2.95))
COPDEM_TILES = ("N50_00_E002_00", "N51_00_E002_00")
COPDEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_DEM/Copernicus_DSM_COG_10_{t}_DEM.tif"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# fetch
# ─────────────────────────────────────────────────────────────────────────────
def cmd_fetch(args):
    import requests
    from PIL import Image
    os.makedirs(BASE, exist_ok=True)
    srcp = os.path.join(BASE, "sources.json")
    S = json.load(open(srcp, encoding="utf-8")) if os.path.exists(srcp) else {}
    today = datetime.date.today().isoformat()
    sess = requests.Session()

    def get(url, params=None, out=None, key=None, meta=None):
        if out and os.path.exists(out) and key in S:
            print(f"[fetch] cached {os.path.basename(out)}"); return
        r = sess.get(url, params=params, timeout=300); r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        if "xml" in ct or "html" in ct:
            raise RuntimeError(f"{url}: service error ({ct}): {r.text[:500]}")
        open(out, "wb").write(r.content)
        S[key] = dict(meta or {}, url=r.url, file=os.path.relpath(out, wc.ROOT).replace("\\", "/"), date=today, bytes=len(r.content),
                      sha256=sha256(out))
        print(f"[fetch] {os.path.basename(out)}: {len(r.content)/1e6:.1f} MB")

    # Copernicus DEM GLO-30 (COG tiles)
    for t in COPDEM_TILES:
        get(COPDEM_URL.format(t=t), out=os.path.join(BASE, f"copdem_glo30_{t}.tif"), key=f"copdem_{t}",
            meta=dict(provider="ESA / Copernicus (AWS Open Data Registry)", product="Copernicus DEM GLO-30 (DGED), COG tile " + t,
                      version="tile as served on the AWS Open Data bucket on the download date (release not stated in the file name)", params="none (whole tile)",
                      licence="Copernicus DEM licence (free, worldwide; attribution required)",
                      attribution="Copernicus DEM GLO-30, © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, "
                                  "provided under COPERNICUS by the European Union and ESA"))
    # EMODnet Bathymetry DTM (WCS 2.0.1)
    par = [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"), ("coverageId", EMOD_COV), ("format", "image/tiff"),
           ("subset", f"Lat({EMOD_BOX['lat'][0]},{EMOD_BOX['lat'][1]})"), ("subset", f"Long({EMOD_BOX['lon'][0]},{EMOD_BOX['lon'][1]})")]
    get(WCS_EMOD, params=par, out=os.path.join(BASE, "emodnet_dtm2024_mean.tif"), key="emodnet",
        meta=dict(provider="EMODnet Bathymetry Consortium", product="EMODnet Digital Bathymetry (DTM 2024), mean depth, coverage "
                  + EMOD_COV, version="DTM 2024 (metadata record cf51df64-56f9-4a99-b1aa-36b8d7b743a1)",
                  params="WCS 2.0.1 GetCoverage, " + ", ".join(f"{k}={v}" for k, v in par[3:]),
                  licence="CC-BY 4.0", vertical_datum="Lowest Astronomical Tide (LAT)",
                  attribution="EMODnet Bathymetry Consortium (2024): EMODnet Digital Bathymetry (DTM 2024), "
                              "https://doi.org/10.12770/cf51df64-56f9-4a99-b1aa-36b8d7b743a1"))
    # Digitaal Vlaanderen orthophoto (WMS 1.3.0, EPSG:31370), tiled ≤ 2048 px and stitched
    out = os.path.join(BASE, f"ortho_{ORTHO_LAYER}_{ORTHO_RES:g}m.png")
    if not (os.path.exists(out) and "ortho" in S):
        x0, x1, y0, y1 = EXT_C; nx, ny = int(round((x1 - x0) / ORTHO_RES)), int(round((y1 - y0) / ORTHO_RES))
        tiles_x = int(np.ceil(nx / 2000)); tiles_y = int(np.ceil(ny / 2000))
        img = np.zeros((ny, nx, 3), np.uint8); urls = []
        for i in range(tiles_x):
            for j in range(tiles_y):
                c0, c1 = i * nx // tiles_x, (i + 1) * nx // tiles_x; r0, r1 = j * ny // tiles_y, (j + 1) * ny // tiles_y
                bx0, bx1 = x0 + c0 * ORTHO_RES, x0 + c1 * ORTHO_RES; by1, by0 = y1 - r0 * ORTHO_RES, y1 - r1 * ORTHO_RES
                p = dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS=ORTHO_LAYER, STYLES="", CRS="EPSG:31370",
                         BBOX=f"{bx0},{by0},{bx1},{by1}", WIDTH=c1 - c0, HEIGHT=r1 - r0, FORMAT="image/png")
                r = sess.get(WMS_OMW, params=p, timeout=300); r.raise_for_status()
                if "image" not in r.headers.get("Content-Type", ""):
                    raise RuntimeError(f"WMS error: {r.text[:500]}")
                img[r0:r1, c0:c1] = np.array(Image.open(io.BytesIO(r.content)).convert("RGB")); urls.append(r.url)
        Image.fromarray(img).save(out)
        S["ortho"] = dict(provider="Digitaal Vlaanderen (Informatie Vlaanderen)",
                          product="Orthofotomozaïek, middenschalig, winteropnamen, kleur, 2025, Vlaanderen (layer OMWRGB25VL; the most "
                                  "recent winter mosaic in the service — the brief's layer name OMWRGBMRVL is not offered)",
                          version="2025 winter flights; ground resolution 15 cm (capabilities abstract)", image_year=2025,
                          url=WMS_OMW, params=f"WMS 1.3.0 GetMap, LAYERS={ORTHO_LAYER}, CRS=EPSG:31370, BBOX={x0},{y0},{x1},{y1} "
                          f"(requested in {tiles_x}×{tiles_y} tiles, stitched), {ORTHO_RES} m/pixel, FORMAT=image/png",
                          tile_urls=urls, bbox=[x0, y0, x1, y1], res=ORTHO_RES, file=os.path.relpath(out, wc.ROOT).replace("\\", "/"),
                          date=today, bytes=os.path.getsize(out), sha256=sha256(out),
                          licence="Modellicentie Gratis Hergebruik (Flemish free re-use licence)",
                          attribution="Orthophoto: Digitaal Vlaanderen (winter mosaic 2025)")
        print(f"[fetch] {os.path.basename(out)}: {img.shape}")
    # Natural Earth (already in the repository)
    for f in sorted(os.listdir(NE_DIR)):
        if f.endswith(".shp"):
            S[f"ne_{f[:-4]}"] = dict(provider="Natural Earth", product=f[:-4], version="1:10m (as fetched by cartopy 0.26.0, run_all step NE)",
                                     url="https://www.naturalearthdata.com", params="see data/external/natural_earth/README.md",
                                     file=f"data/external/natural_earth/{f}", date="(earlier round)", sha256=sha256(os.path.join(NE_DIR, f)),
                                     licence="public domain", attribution="Made with Natural Earth")
    json.dump(S, open(srcp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    write_readme(S)


def write_readme(S):
    L = ["# Base-map data for Fig. 1 (`scripts/32_location_map.py fetch`)", "",
         "Downloaded once; `build` uses only these files (offline). Large rasters are git-ignored; this README and "
         "`sources.json` are committed. Re-run `fetch` to re-download (it skips files already recorded).", ""]
    for k, o in S.items():
        L += [f"## {o.get('product', k)}", "",
              f"- Provider: {o.get('provider')}", f"- Version: {o.get('version')}", f"- File: `{o.get('file')}`",
              f"- URL: {o.get('url')}", f"- Request parameters: {o.get('params')}", f"- Download date: {o.get('date')}",
              f"- SHA256: `{o.get('sha256')}`", f"- Licence: {o.get('licence')}", f"- Required attribution: {o.get('attribution')}"]
        if o.get("vertical_datum"):
            L.append(f"- Vertical datum: {o['vertical_datum']}")
        L.append("")
    open(os.path.join(BASE, "README_sources.md"), "w", encoding="utf-8").write("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
# build helpers
# ─────────────────────────────────────────────────────────────────────────────
MM = 1 / 25.4
TOWNS = {"De Panne": (2.5926, 51.0998), "Koksijde": (2.6377, 51.1164), "Nieuwpoort": (2.7517, 51.1300), "Dunkerque": (2.3768, 51.0344)}
COL = dict(land="#c0262d", marine="#ffffff", e24="#00d8ff", cpt="#ffd400", box_b="#d7191c", box_c="k")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def read_gps():
    """GPS_coordinate.xlsx (Belgian Lambert 72), standard-library parser (as 18_paper.read_gps)."""
    import zipfile
    import xml.etree.ElementTree as ET
    z = zipfile.ZipFile(os.path.join(wc.ROOT, "data", "raw", "GPS_coordinate.xlsx"))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
    out = {}
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{{{ns['m']}}}row"):
        vals = []
        for c in row.findall("m:c", ns):
            v = c.find("m:v", ns)
            vals.append(None if v is None else (shared[int(v.text)] if c.get("t") == "s" else float(v.text)))
        if vals and isinstance(vals[0], float):
            out[int(vals[0])] = (vals[1], vals[2])
    return out


def read_cpt_locations():
    L = open(os.path.join(wc.ROOT, "data", "raw", "cpt_vrancken_2024", "CPT", "Location.txt"), encoding="utf-8").read().split("\n")
    out = {}
    for l in L[1:]:
        p = l.strip().split(";")
        if len(p) >= 4 and p[0].startswith("S"):
            out[p[0]] = tuple(float(v.replace(",", ".")) for v in p[1:4])
    return out


def geometry():
    """All feature coordinates (EPSG:31370) and the geometry checks."""
    p04 = _load("p04", "04_misfit_diagnostics.py")
    gps = read_gps()
    land, _ = wc.read_electrodes(); ex, ez = land[:, 0], land[:, 1]
    _, x_hw, x_lw = p04.water_lines(wc.ELEC_FILE, ex, ez)
    P1, P102 = np.array(gps[1]), np.array(gps[102]); M1, M2 = np.array(gps[103]), np.array(gps[538])
    L = np.hypot(*(P102 - P1)); u = (P102 - P1) / L; nrm = np.array([-u[1], u[0]])
    span = ex[-1] - ex[0]
    pos = lambda x: P1 + np.outer((np.asarray(x) - ex[0]) / span * L, u)          # along the chord by along-line x
    E = pos(ex)
    dh = np.hypot(*np.diff(E, axis=0).T); dx = np.diff(ex)
    chk1 = dict(E1=float(np.hypot(*(E[0] - P1))), E102=float(np.hypot(*(E[-1] - P102))), E103=0.0, E538=0.0,
                chord_m=float(L), x_span_m=float(span), max_spacing_dev_m=float(np.max(np.abs(dh - dx))))
    to_x = lambda p: ex[0] + ((np.asarray(p) - P1) @ u) / L * span
    off = lambda p: (np.asarray(p) - P1) @ nrm
    C = read_cpt_locations(); J = json.load(open(os.path.join(wc.ROOT, "data", "processed", "v1_4", "cpt.json"), encoding="utf-8"))
    names = sorted(C); cp = np.array([C[s][:2] for s in names]); cz = np.array([C[s][2] for s in names])
    xs_cpt = to_x(cp); off_cpt = off(cp)
    v14 = {"S02": 454.1, "S03": 404.7, "S04": 352.8, "S05": 304.8, "S06": 253.6, "S07": 203.5, "S08": 152.7}
    chk2 = {s: dict(x=float(xs_cpt[i]), v14=v14[s], dx=float(xs_cpt[i] - v14[s]), offset_m=float(off_cpt[i])) for i, s in enumerate(names)}
    # 2024 line: least-squares (principal-axis) line through S02–S08, electrodes placed by our_x = 503.75 − x_local
    c0 = cp.mean(0); _, _, Vt = np.linalg.svd(cp - c0); d = Vt[0] * np.sign(Vt[0] @ u)
    t_c = (cp - c0) @ d; ox24 = np.array([503.75 - J["cpts"][s]["dp_local_x"] for s in names])
    a = float(np.mean(t_c - ox24)); res = t_c - ox24 - a
    x24 = 503.75 - np.arange(0, 356, 5.0)                       # 72 electrodes, x_local = 0 … 355 m
    pos24 = lambda x: c0 + np.outer(a + np.asarray(x), d)
    E24 = pos24(x24); ends = pos24([148.75, 503.75])
    line_off = [float(off(p)) for p in ends]
    ang = float(np.degrees(np.arctan2(d[0] * u[1] - d[1] * u[0], d @ u)))
    perp24 = (cp - c0) @ np.array([-d[1], d[0]])
    chk24 = dict(rms_along_m=float(np.sqrt(np.mean(res ** 2))), cpt_perp_rms_m=float(np.sqrt(np.mean(perp24 ** 2))),
                 offset_from_2018_at_x148_75_m=line_off[0], offset_from_2018_at_x503_75_m=line_off[1], angle_deg=ang,
                 mean_abs_offset_m=float(np.mean(np.abs(line_off))))
    return dict(gps=gps, ex=ex, ez=ez, x_hw=float(x_hw), x_lw=float(x_lw), E=E, M=(M1, M2), E24=E24, x24=x24, cpt=cp, cpt_z=cz,
                cpt_names=names, cpt_x=xs_cpt, chk1=chk1, chk2=chk2, chk24=chk24, u=u, P1=P1, L=L, span=span)


def read_geotiff(path):
    """Float GeoTIFF via PIL; returns array and functions lon/lat → fractional (row, col)."""
    from PIL import Image
    im = Image.open(path); t = im.tag_v2; a = np.array(im, dtype=np.float64)
    keys = t.get(34735) or ()
    kv = {keys[i]: keys[i + 3] for i in range(4, len(keys), 4)}
    point = kv.get(1025, 1) == 2
    if 34264 in t:
        m = t[34264]; sx, ox, sy, oy = m[0], m[3], m[5], m[7]
    else:
        sx, sy = t[33550][0], -t[33550][1]; ox, oy = t[33922][3], t[33922][4]
    sh = 0.0 if point else 0.5
    return a, (lambda lon, lat: ((np.asarray(lat) - oy) / sy - sh, (np.asarray(lon) - ox) / sx - sh))


def relief_grid():
    """Merged 50 m grid (EPSG:31370): EMODnet DTM 2024 where it is below 0 m (LAT), Copernicus DEM GLO-30 elsewhere."""
    from pyproj import Transformer
    from scipy.ndimage import map_coordinates
    x0, x1, y0, y1 = EXT_B
    xg = np.arange(x0 + GRID_B / 2, x1, GRID_B); yg = np.arange(y1 - GRID_B / 2, y0, -GRID_B)
    X, Y = np.meshgrid(xg, yg)
    lon, lat = Transformer.from_crs(31370, 4326, always_xy=True).transform(X, Y)
    dem = np.full(X.shape, np.nan)
    for t in COPDEM_TILES:
        a, rc = read_geotiff(os.path.join(BASE, f"copdem_glo30_{t}.tif"))
        r, c = rc(lon, lat); ok = (r >= 0) & (r <= a.shape[0] - 1) & (c >= 0) & (c <= a.shape[1] - 1) & np.isnan(dem)
        dem[ok] = map_coordinates(a, [r[ok], c[ok]], order=1)
    a, rc = read_geotiff(os.path.join(BASE, "emodnet_dtm2024_mean.tif")); r, c = rc(lon, lat)
    emo = map_coordinates(a, [r, c], order=1, cval=np.nan)
    from scipy.ndimage import label
    below = np.isfinite(emo) & (emo < 0)
    lab, _ = label(below)                                     # sea = the below-0 m region connected to the open sea (top edge);
    sea = np.isin(lab, np.setdiff1d(np.unique(lab[0]), [0]))  # EMODnet's coarse land fill is negative in some polders
    z = np.where(sea, emo, dem)
    info = dict(sea_mask=None, emodnet_below0_not_connected=float(np.mean(below & ~sea)), n=int(z.size), from_emodnet=float(sea.mean()), dem_nan=int(np.isnan(dem).sum()), z_min=float(np.nanmin(z)),
                z_max=float(np.nanmax(z)))
    info.pop("sea_mask")
    return z, sea, info


class Grat:
    """Graticule on a projected axes: lines + frame labels; records the label positions for check 4."""

    def __init__(self, crs):
        from pyproj import Transformer
        self.fwd = Transformer.from_crs(4326, crs, always_xy=True); self.inv = Transformer.from_crs(crs, 4326, always_xy=True)
        self.rec = []

    def draw(self, ax, ext, dlon, dlat, fmt_lon, fmt_lat, fs=7, lines=True, lc="0.35"):
        """Meridians / parallels every dlon / dlat degrees (round multiples inside the panel)."""
        x0, x1, y0, y1 = ext
        lo0, la0 = self.inv.transform(np.array([x0, x1, x0, x1]), np.array([y0, y0, y1, y1]))
        mult = lambda a, b, st: np.arange(np.ceil(a / st - 1e-9), np.floor(b / st + 1e-9) + 1) * st
        lons, lats = mult(min(lo0), max(lo0), dlon), mult(min(la0), max(la0), dlat)
        for lon in lons:
            la = np.linspace(min(la0) - 1, max(la0) + 1, 400); xx, yy = self.fwd.transform(np.full_like(la, lon), la)
            if lines:
                ax.plot(xx, yy, color=lc, lw=0.35, alpha=0.6, zorder=3, clip_on=True)
            xb = float(np.interp(y0, yy, xx)) if yy[0] < yy[-1] else float(np.interp(y0, yy[::-1], xx[::-1]))
            if x0 < xb < x1:
                ax.annotate(fmt_lon(lon), xy=((xb - x0) / (x1 - x0), 0), xycoords="axes fraction", xytext=(0, -2.5), textcoords="offset points",
                            ha="center", va="top", fontsize=fs)
                ax.plot([xb, xb], [y0, y0 + (y1 - y0) * 0.015], color="k", lw=0.6, clip_on=False, zorder=6)
                self.rec.append(dict(kind="lon", value=lon, edge="bottom", x=xb, y=y0, back=float(self.inv.transform(xb, y0)[0])))
        for lat in lats:
            lo = np.linspace(min(lo0) - 1, max(lo0) + 1, 400); xx, yy = self.fwd.transform(lo, np.full_like(lo, lat))
            if lines:
                ax.plot(xx, yy, color=lc, lw=0.35, alpha=0.6, zorder=3, clip_on=True)
            yb = float(np.interp(x0, xx, yy))
            if y0 < yb < y1:
                ax.annotate(fmt_lat(lat), xy=(0, (yb - y0) / (y1 - y0)), xycoords="axes fraction", xytext=(-2.5, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=fs, rotation=90)
                ax.plot([x0, x0 + (x1 - x0) * 0.015], [yb, yb], color="k", lw=0.6, clip_on=False, zorder=6)
                self.rec.append(dict(kind="lat", value=lat, edge="left", x=x0, y=yb, back=float(self.inv.transform(x0, yb)[1])))


def dm(v, hemi, dec=0):
    d = int(np.floor(abs(v) + 1e-9)); m = (abs(v) - d) * 60
    ms = f"{m:0{3 + dec if dec else 2}.{dec}f}" if dec else f"{int(round(m)):02d}"
    return f"{d}°{ms}′{hemi}"


def north_arrow(ax, xy, crs_tr, length_mm=6.0, fs=7):
    """Arrow pointing to true north at axes position xy (fraction); crs_tr: (fwd, inv) transformers of the axes CRS."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    X = x0 + xy[0] * (x1 - x0); Y = y0 + xy[1] * (y1 - y0)
    lon, lat = crs_tr[1].transform(X, Y); X2, Y2 = crs_tr[0].transform(lon, lat + 0.01)
    ang = np.arctan2(X2 - X, Y2 - Y)
    bb = ax.get_position(); fw, fh = ax.figure.get_size_inches() * 25.4
    dx, dy = np.sin(ang) * length_mm / (bb.width * fw), np.cos(ang) * length_mm / (bb.height * fh)
    ax.annotate("", xy=(xy[0] + dx, xy[1] + dy), xytext=xy, xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="k", mutation_scale=7), zorder=8)
    ax.text(xy[0] + dx * 1.25, xy[1] + dy * 1.25, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=fs, fontweight="bold", zorder=8)
    return float(np.degrees(ang))


def scale_bar(ax, length, label, xy=(0.05, 0.06), fs=7, color="k", halo="w"):
    import matplotlib.patheffects as pe
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    X = x0 + xy[0] * (x1 - x0); Y = y0 + xy[1] * (y1 - y0); h = (y1 - y0) * 0.012
    st = [pe.withStroke(linewidth=1.6, foreground=halo)]
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((X, Y), length / 2, h, facecolor=color, edgecolor=color, lw=0.5, zorder=8))
    ax.add_patch(Rectangle((X + length / 2, Y), length / 2, h, facecolor=halo, edgecolor=color, lw=0.5, zorder=8))
    for k, lab in ((0, "0"), (length, label)):
        ax.text(X + k, Y + 2.2 * h, lab, ha="center", va="bottom", fontsize=fs, color=color, path_effects=st, zorder=8)


def panel_letter(ax, s, fs=8.5):
    ax.text(0.015, 0.985, s, transform=ax.transAxes, ha="left", va="top", fontsize=fs, fontweight="bold", zorder=10,
            bbox=dict(boxstyle="square,pad=0.15", fc="w", ec="none", alpha=0.85))


def pin_marker():
    from matplotlib.path import Path
    th = np.linspace(np.radians(-30), np.radians(210), 40)
    head = np.c_[0.42 * np.cos(th), 1.0 + 0.42 * np.sin(th)]
    v = np.vstack([[0, 0], head, [0, 0]]); codes = [Path.MOVETO] + [Path.LINETO] * (len(v) - 2) + [Path.CLOSEPOLY]
    return Path(v, codes)


def topo_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    sea = ["#08306b", "#1c5a9e", "#3d85c3", "#7db5dd", "#cfe6f3"]
    land = ["#4f8f4a", "#9bbf62", "#e6d77a", "#c9a15a", "#8a5a2e"]
    stops = [(i / 4 * 0.5, c) for i, c in enumerate(sea[:4])] + [(0.4999, sea[4]), (0.5001, land[0])] +         [(0.5 + (i + 1) / 4 * 0.5, c) for i, c in enumerate(land[1:])]
    return LinearSegmentedColormap.from_list("topo_split", stops)


# ─────────────────────────────────────────────────────────────────────────────
# build
# ─────────────────────────────────────────────────────────────────────────────
def cmd_build(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import TwoSlopeNorm, LightSource
    from matplotlib.patches import ConnectionPatch, Rectangle
    from pyproj import Transformer
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader
    from PIL import Image
    S = json.load(open(os.path.join(BASE, "sources.json"), encoding="utf-8"))
    for k, o in S.items():                                     # cache integrity (offline build)
        if k.startswith(("copdem", "emodnet", "ortho")) and sha256(os.path.join(wc.ROOT, o["file"])) != o["sha256"]:
            raise SystemExit(f"cache file changed: {o['file']}")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "svg.hashsalt": "f01",
                         "pdf.fonttype": 42})
    G = geometry(); z, sea, zinfo = relief_grid()
    zinfo["land_below_0_m_frac"] = float(np.mean(~sea & (z < 0)))
    T72 = (Transformer.from_crs(4326, 31370, always_xy=True), Transformer.from_crs(31370, 4326, always_xy=True))
    T35 = (Transformer.from_crs(4326, 3035, always_xy=True), Transformer.from_crs(3035, 4326, always_xy=True))
    fig = plt.figure(figsize=(190 * MM, 160 * MM))
    R = lambda x, y, w, h: [x / 190, y / 160, w / 190, h / 160]
    halo = [pe.withStroke(linewidth=1.8, foreground="w")]
    out = {}

    # ── (b) coastal relief ──
    bx0, bx1, by0, by1 = EXT_B; wb = 122.0; hb = wb * (by1 - by0) / (bx1 - bx0)
    axb = fig.add_axes(R(63, 93, wb, hb))
    cm = topo_cmap(); nrm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    ls = LightSource(azdeg=315, altdeg=40)
    zc = np.where(sea, np.clip(z, -40, -0.01), np.clip(z, 0.01, 40))     # colour by the land/sea mask of the merge
    rgb = ls.shade(zc, cmap=cm, norm=nrm, blend_mode="soft", vert_exag=8, dx=GRID_B, dy=GRID_B)
    axb.imshow(rgb, extent=EXT_B, origin="upper", interpolation="bilinear", zorder=1)
    rd = shapereader.Reader(os.path.join(NE_DIR, "ne_10m_admin_0_boundary_lines_land.shp"))
    for g in rd.geometries():
        for ln in getattr(g, "geoms", [g]):
            lo, la = np.array(ln.coords).T
            if lo.min() < 3.2 and lo.max() > 2.0 and la.min() < 51.4 and la.max() > 50.7:
                xx, yy = T72[0].transform(lo, la); axb.plot(xx, yy, color="k", lw=0.8, ls=(0, (4, 1.5, 1, 1.5)), zorder=4)
    for name, (lo, la) in TOWNS.items():
        xx, yy = T72[0].transform(lo, la); axb.plot(xx, yy, "s", ms=2.6, mfc="k", mec="w", mew=0.4, zorder=6)
        dxy = dict(Dunkerque=(4, -7), Nieuwpoort=(-4, -8)).get(name, (3, -7))
        axb.annotate(name, (xx, yy), xytext=dxy, textcoords="offset points", fontsize=7, ha="left" if dxy[0] > 0 else "right",
                     path_effects=halo, zorder=7)
    for txt, (lo, la), kw in (("North Sea", (2.48, 51.135), dict(style="italic", color="#0b2a55")), ("France", (2.45, 51.046), dict(fontweight="bold")),
                              ("Belgium", (2.66, 51.068), dict(fontweight="bold"))):
        xx, yy = T72[0].transform(lo, la)
        axb.text(xx, yy, txt, fontsize=8, ha="center", va="center", path_effects=halo, zorder=7, clip_on=True, **kw)
    cx0, cx1, cy0, cy1 = EXT_C
    axb.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec=COL["box_c"], lw=0.9, zorder=8))
    axb.set_xlim(bx0, bx1); axb.set_ylim(by0, by1); axb.set_xticks([]); axb.set_yticks([]); axb.set_aspect("equal")
    gb = Grat(31370)
    gb.draw(axb, EXT_B, 5 / 60, 2.5 / 60, lambda v: dm(v, "E"), lambda v: dm(v, "N", 1))
    out["north_b_deg"] = north_arrow(axb, (0.045, 0.60), T72); scale_bar(axb, 5000, "5 km", xy=(0.80, 0.07))
    panel_letter(axb, "(b)")
    cax = fig.add_axes(R(63, 93 + hb + 3.2, 58, 2.0))
    sm = plt.cm.ScalarMappable(norm=nrm, cmap=cm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-40, -20, -10, 0, 10, 20, 40])
    cb.ax.tick_params(labelsize=7, length=2, width=0.5, pad=1); cb.outline.set_linewidth(0.5)
    cb.ax.xaxis.set_ticks_position("top"); cb.ax.xaxis.set_label_position("top")
    cb.set_label("Elevation / depth (m); land clipped at 40 m", fontsize=7, labelpad=1.5)

    # ── (a) overview (cartopy, EPSG:3035) ──
    pa = ccrs.epsg(3035)
    axa = fig.add_axes(R(9, 93, 44, hb), projection=pa)
    lon_a, lat_a = (-1.5, 8.0), (48.6, 54.2)
    xa, ya = T35[0].transform(np.array([lon_a[0], lon_a[1], lon_a[0], lon_a[1], 3.0, 3.0]), np.array([lat_a[0], lat_a[0], lat_a[1], lat_a[1], 48.6, 54.2]))
    ycen = 0.5 * (min(ya) + max(ya)); yh = max(ya) - min(ya); xw = yh * 44 / hb; xcen = 0.5 * (min(xa) + max(xa))
    EXT_A = (xcen - xw / 2, xcen + xw / 2, ycen - yh / 2, ycen + yh / 2)
    axa.set_extent(EXT_A, crs=pa)
    axa.add_feature(cfeature.ShapelyFeature(shapereader.Reader(os.path.join(NE_DIR, "ne_10m_land.shp")).geometries(), ccrs.PlateCarree(),
                                            facecolor="#e9e4d6", edgecolor="none"), zorder=1)
    axa.set_facecolor("#cfe3f0")
    axa.add_feature(cfeature.ShapelyFeature(shapereader.Reader(os.path.join(NE_DIR, "ne_10m_coastline.shp")).geometries(), ccrs.PlateCarree(),
                                            facecolor="none", edgecolor="0.25", lw=0.4), zorder=2)
    axa.add_feature(cfeature.ShapelyFeature(shapereader.Reader(os.path.join(NE_DIR, "ne_10m_admin_0_boundary_lines_land.shp")).geometries(),
                                            ccrs.PlateCarree(), facecolor="none", edgecolor="0.3", lw=0.5, linestyle=(0, (3, 1.2))), zorder=2)
    t = np.linspace(0, 1, 50)
    bxs = np.r_[bx0 + t * (bx1 - bx0), np.full(50, bx1), bx1 - t * (bx1 - bx0), np.full(50, bx0)]
    bys = np.r_[np.full(50, by0), by0 + t * (by1 - by0), np.full(50, by1), by1 - t * (by1 - by0)]
    blo, bla = T72[1].transform(bxs, bys); X3, Y3 = T35[0].transform(blo, bla)
    axa.plot(X3, Y3, color=COL["box_b"], lw=1.2, transform=pa, zorder=5)
    for txt, (lo, la), kw in (("North\nSea", (3.4, 53.0), dict(style="italic", color="#0b2a55")), ("UK", (0.2, 52.4), {}), ("France", (3.2, 49.6), {}),
                              ("Belgium", (5.0, 50.45), {}), ("NL", (5.6, 52.3), {})):
        xx, yy = T35[0].transform(lo, la); axa.text(xx, yy, txt, fontsize=7, ha="center", va="center", transform=pa, zorder=6, **kw)
    ga = Grat(3035)
    ga.draw(axa, EXT_A, 2.0, 2.0, lambda v: dm(v, "E"), lambda v: dm(v, "N"))
    out["north_a_deg"] = north_arrow(axa, (0.88, 0.80), T35); scale_bar(axa, 200000, "200 km", xy=(0.06, 0.06))
    panel_letter(axa, "(a)")

    # ── (c) site map on the orthophoto ──
    wc_ = 70.0; axc = fig.add_axes(R(12, 12, wc_, wc_))
    img = np.array(Image.open(os.path.join(BASE, f"ortho_{ORTHO_LAYER}_{ORTHO_RES:g}m.png")))
    axc.imshow(img, extent=EXT_C, origin="upper", interpolation="lanczos", zorder=1)
    E, (M1, M2), E24, cp = G["E"], G["M"], G["E24"], G["cpt"]
    stroke = lambda w, c="k": [pe.Stroke(linewidth=w, foreground=c), pe.Normal()]
    axc.plot(*E.T, "-", color=COL["land"], lw=2.0, zorder=4, path_effects=stroke(3.0), label="2018 land ERT (K1)")
    axc.plot([M1[0], M2[0]], [M1[1], M2[1]], color=COL["marine"], lw=1.2, ls=(0, (5, 2.5)), zorder=4, path_effects=stroke(2.2),
             label="2018 marine ERT (high tide)")
    axc.plot(*E24[[0, -1]].T, "-", color=COL["e24"], lw=0.8, zorder=5, path_effects=stroke(1.5), label="2024 ERT")
    pin = pin_marker()
    axc.plot(*cp.T, ls="none", marker=pin, ms=9, mfc=COL["cpt"], mec="k", mew=0.6, zorder=7, label="CPT / CPT-EC (2024)†")
    for i, s in enumerate(G["cpt_names"]):
        axc.annotate(f"DP{int(s[1:]) - 1}", cp[i], xytext=(5, 1), textcoords="offset points", fontsize=7, ha="left", va="bottom",
                     path_effects=halo, zorder=8)
    for lab, p, dxy in (("E1", E[0], (6, -2)), ("E102", E[-1], (-6, 4)), ("E103", M1, (-12, -12)), ("E538", M2, (7, -6))):
        axc.annotate(lab, p, xytext=dxy, textcoords="offset points", fontsize=7, ha="left" if dxy[0] > 0 else "right", path_effects=halo, zorder=8)
    for txt, xy, kw in (("Westhoek dunes", (23290, 199590), dict(color="w", fontweight="bold", path_effects=stroke(1.6))),
                        ("Beach", (22960, 199700), dict(rotation=34)), ("North Sea", (22830, 200040), dict(style="italic", color="#0b2a55"))):
        axc.text(*xy, txt, fontsize=8, ha="center", va="center", zorder=6, **({"path_effects": halo} | kw))
    axc.set_xlim(cx0, cx1); axc.set_ylim(cy0, cy1); axc.set_xticks([]); axc.set_yticks([]); axc.set_aspect("equal")
    gc = Grat(31370)
    gc.draw(axc, EXT_C, 0.2 / 60, 0.1 / 60, lambda v: dm(v, "E", 1), lambda v: dm(v, "N", 1),
            lines=False)
    out["north_c_deg"] = north_arrow(axc, (0.06, 0.16), T72); scale_bar(axc, 200, "200 m", xy=(0.06, 0.05))
    leg = axc.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=True, framealpha=0.9, edgecolor="none",
                     handlelength=2.6, borderpad=0.4, labelspacing=0.3)
    panel_letter(axc, "(c)")
    for (xa_, ya_), (xc_, yc_) in (((cx0, cy0), (0, 1)), ((cx1, cy0), (1, 1))):
        fig.add_artist(ConnectionPatch(xyA=(xa_, ya_), coordsA=axb.transData, xyB=(xc_, yc_), coordsB=axc.transAxes, color="k", lw=0.6, zorder=0))

    # ── (d) profile strip ──
    axd = fig.add_axes(R(104, 22, 82, 50))
    axd.axvspan(0, 90, color="#9bbf62", alpha=0.25, lw=0, zorder=0)
    axd.plot(G["ex"], G["ez"], "-", color=COL["land"], lw=1.2, zorder=3, label="2018 electrodes (K1)")
    axd.plot(G["ex"], G["ez"], "o", ms=1.4, color=COL["land"], zorder=3)
    axd.plot(G["cpt_x"], G["cpt_z"], ls="none", marker=pin, ms=8, mfc=COL["cpt"], mec="k", mew=0.6, zorder=5, label="CPT ground level (2024)†")
    for i, s in enumerate(G["cpt_names"]):
        axd.annotate(f"DP{int(s[1:]) - 1}", (G["cpt_x"][i], G["cpt_z"][i]), xytext=(0, 11), textcoords="offset points", fontsize=7, ha="center")
    yb = -1.6
    axd.plot([148.75, 503.75], [yb, yb], "-", color="k", lw=2.4, solid_capstyle="butt", zorder=4)
    axd.plot([148.75, 503.75], [yb, yb], "-", color=COL["e24"], lw=1.4, solid_capstyle="butt", zorder=5)
    axd.text(326, yb + 0.45, "2024 ERT (x = 148.75–503.75 m)", fontsize=7, ha="center", va="bottom")
    for xv, lab in ((90, "dune\nfoot"), (G["x_hw"], f"HW 2018\n(x = {G['x_hw']:.0f} m)"), (G["x_lw"], "LW")):
        axd.axvline(xv, color="0.25", lw=0.6, ls=(0, (3, 2)), zorder=1)
        axd.text(xv + (-4 if xv > 480 else 4), 8.4, lab, fontsize=7, ha="right" if xv > 480 else "left", va="top")
    axd.text(45, 3.6, "dunes", fontsize=7, ha="center", style="italic"); axd.text(215, 5.9, "beach", fontsize=7, ha="center", style="italic")
    axd.set_xlim(0, 505); axd.set_ylim(-2.4, 8.6)
    axd.set_xlabel("x along K1 (m; E1 = 0, dune → sea)", fontsize=7); axd.set_ylabel("Elevation (mTAW)", fontsize=7)
    axd.tick_params(labelsize=7, length=2.5, width=0.6)
    axd.legend(loc="upper right", bbox_to_anchor=(0.99, 0.86), fontsize=7, frameon=False, handlelength=2.0)
    panel_letter(axd, "(d)")
    for s_ in axd.spines.values():
        s_.set_linewidth(0.6)

    # ── save ──
    figd = os.path.join(args.out, "results", "figs"); os.makedirs(figd, exist_ok=True)
    base = os.path.join(figd, "F_01_location_map")
    fig.savefig(base + ".png", dpi=600, metadata={"Software": None})
    fig.savefig(base + ".pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(base + ".tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    # ── checks and caption ──
    g4 = [abs(r["back"] - r["value"]) * (111320 * np.cos(np.radians(51.1)) if r["kind"] == "lon" else 110540) for r in gb.rec + gc.rec] + \
         [abs(r["back"] - r["value"]) * (111320 * np.cos(np.radians(51.1)) if r["kind"] == "lon" else 110540) for r in ga.rec]
    corners = {}
    for nm, ext, tr in (("a", EXT_A, T35), ("b", EXT_B, T72), ("c", EXT_C, T72)):
        lo, la = tr[1].transform(np.array([ext[0], ext[1], ext[0], ext[1]]), np.array([ext[2], ext[2], ext[3], ext[3]]))
        corners[nm] = dict(SW=[float(lo[0]), float(la[0])], SE=[float(lo[1]), float(la[1])], NW=[float(lo[2]), float(la[2])], NE=[float(lo[3]), float(la[3])])
    lab_ok = all(min(corners[p]["SW"][0], corners[p]["NW"][0]) <= r["value"] <= max(corners[p]["SE"][0], corners[p]["NE"][0]) if r["kind"] == "lon"
                 else min(corners[p]["SW"][1], corners[p]["SE"][1]) <= r["value"] <= max(corners[p]["NW"][1], corners[p]["NE"][1])
                 for p, recs in (("a", ga.rec), ("b", gb.rec), ("c", gc.rec)) for r in recs)
    res = dict(check1=G["chk1"], check2=G["chk2"], line2024=G["chk24"], relief=zinfo, north_deg=dict(a=out["north_a_deg"], b=out["north_b_deg"], c=out["north_c_deg"]),
               check4=dict(n_labels=len(gb.rec) + len(gc.rec) + len(ga.rec), max_back_error_m=float(max(g4)), labels_within_corner_ranges=bool(lab_ok),
                           corners=corners, labels={p: [dict(kind=r["kind"], value=r["value"]) for r in recs] for p, recs in (("a", ga.rec), ("b", gb.rec), ("c", gc.rec))}),
               image_year=S["ortho"]["image_year"], ortho_layer=ORTHO_LAYER, x_hw=G["x_hw"], x_lw=G["x_lw"])
    pd_ = os.path.join(args.out, "data", "processed", "v1_7"); os.makedirs(pd_, exist_ok=True)
    json.dump(res, open(os.path.join(pd_, "location_map.json"), "w", encoding="utf-8"), indent=1)
    off = res["line2024"]["mean_abs_offset_m"]
    cap = ["# F_01 — caption draft", "",
           "**Fig. 1.** Location of the Westhoek K1 profile (De Panne, Belgium). (a) Overview; red box: (b). (b) Coastal relief "
           "(sea: EMODnet DTM 2024, depth below LAT; land: Copernicus DEM GLO-30, EGM2008; datums not unified; land clipped at 40 m); "
           "black box: (c). (c) Site on the 2025 winter orthophoto: 2018 land ERT line K1 (E1–E102, 5 m spacing; Paepen et al., 2020; "
           "Hermans et al., 2020), 2018 marine ERT line at high tide (E103–E538; Hermans et al., 2020), 2024 ERT line (72 electrodes; "
           f"Vrancken et al., 2026), on average {off:.1f} m from K1, and the CPT / CPT-EC soundings DP1–DP7 (2024)†. "
           "(d) Elevation along K1: 2018 electrodes and 2024 CPT ground levels; HW/LW: high- and low-water lines of the 2018 survey.",
           "", "† Unpublished data (E. Vrancken), used with permission.", "",
           "Data: Natural Earth (public domain); Copernicus DEM GLO-30, © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH "
           "2014–2018, provided under COPERNICUS by the European Union and ESA; EMODnet Bathymetry Consortium (2024): EMODnet Digital "
           "Bathymetry (DTM 2024), doi:10.12770/cf51df64-56f9-4a99-b1aa-36b8d7b743a1 (CC-BY 4.0); orthophoto: Digitaal Vlaanderen, "
           f"winter mosaic {res['image_year']} (Modellicentie Gratis Hergebruik). Coordinates: Belgian Lambert 72 (EPSG:31370; (a) EPSG:3035); "
           "graticule WGS84."]
    open(os.path.join(figd, "F_01_caption_draft.md"), "w", encoding="utf-8").write("\n".join(cap) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "check4"}, indent=1))
    print("check4:", {k: v for k, v in res["check4"].items() if k in ("n_labels", "max_back_error_m", "labels_within_corner_ranges")})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("build").add_argument("--out", default=wc.ROOT)
    a = ap.parse_args()
    dict(fetch=cmd_fetch, build=cmd_build)[a.cmd](a)


if __name__ == "__main__":
    main()
