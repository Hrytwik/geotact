"""Extract vector layers from the BMC ward map GeoPDF (Ward_Map_1.pdf).

ArcMap exported each map layer as a PDF optional-content block (/OC /OCn BDC ... EMC).
For each leaf layer we render only that block's vector paths, then convert page
coordinates to lon/lat with the GeoPDF /Measure georeference (viewport BBox -> GPTS corners).
"""
import re

import geopandas as gpd
import numpy as np
import pymupdf
from shapely.geometry import LineString, Polygon

from src.config import CRS, INTERIM, ROOT

PDF = ROOT / "Ward_Map_1.pdf"
OUT = INTERIM / "wardmap.gpkg"

# GeoPDF georeference read from the file (/VP BBox and /Measure GPTS, lat-lon order).
VP_BBOX = (59.40052, 3460.80013, 2814.62457, 119.40648)        # x1, y1(top), x2, y2(bottom), PDF y-up
GPTS = np.array([[20.17715, 85.72931], [20.38916, 85.72931], [20.38916, 85.90413], [20.17715, 85.90413]])
LPTS = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])


def _fit_affine():
    """Least-squares affine from normalised viewport coords (LPTS) to lon/lat (GPTS)."""
    A = np.c_[LPTS, np.ones(4)]
    lon, *_ = np.linalg.lstsq(A, GPTS[:, 1], rcond=None)
    lat, *_ = np.linalg.lstsq(A, GPTS[:, 0], rcond=None)
    return lon, lat


LON_C, LAT_C = _fit_affine()


def page_to_lonlat(pts, page_h):
    """pymupdf points (y-down) -> lon/lat."""
    pts = np.asarray(pts, float)
    X, Y = pts[:, 0], page_h - pts[:, 1]
    x1, y1, x2, y2 = VP_BBOX
    u = (X - x1) / (x2 - x1)
    v = (Y - y1) / (y2 - y1)
    A = np.c_[u, v, np.ones(len(u))]
    return np.c_[A @ LON_C, A @ LAT_C]


def layer_spans(raw):
    """Return {OC tag: (start, end)} byte spans of each marked-content block."""
    spans, stack = {}, []
    for m in re.finditer(rb"/OC\s*/(OC\d+)\s*BDC|\bBMC\b|\bEMC\b", raw):
        tok = m.group(0)
        if tok.endswith(b"BDC"):
            stack.append((m.group(1).decode(), m.start()))
        elif tok == b"BMC":
            stack.append((None, m.start()))
        else:
            tag, start = stack.pop()
            if tag:
                spans[tag] = (start, m.end())
    return spans


def layer_names(doc, page):
    props = doc.xref_get_key(page.xref, "Resources")
    res_xref = int(props[1].split()[0])
    names = {}
    for tag, ref in re.findall(r"/(OC\d+)\s+(\d+)\s+0\s+R", doc.xref_object(res_xref)):
        names[tag] = doc.xref_get_key(int(ref), "Name")[1].strip("()").replace("\x00", "")
    return names


def drawings_for(span):
    doc = pymupdf.open(PDF)
    page = doc[0]
    raw = page.read_contents()
    pre = raw[: raw.find(b"/OC")]
    doc.update_stream(int(doc.xref_get_key(page.xref, "Contents")[1].split()[0]), pre + raw[span[0]: span[1]])
    return page.get_drawings(), page.rect.height


def path_subpaths(d):
    """Split one PDF path into its connected sub-paths. A path can contain several
    move-to segments; joining them end-to-end would invent straight connector lines."""
    subs, cur = [], []

    def close():
        if len(cur) >= 2:
            subs.append([(p.x, p.y) for p in cur])

    for item in d["items"]:
        if item[0] in ("re", "qu"):
            close(); cur = []
            g = item[1]
            ring = [g.tl, g.tr, g.br, g.bl, g.tl] if item[0] == "re" else [g.ul, g.ur, g.lr, g.ll, g.ul]
            subs.append([(p.x, p.y) for p in ring])
            continue
        start, end = item[1], item[-1]          # "l": (p1, p2); "c": (p1, c1, c2, p2)
        if cur and (abs(start.x - cur[-1].x) > 1e-3 or abs(start.y - cur[-1].y) > 1e-3):
            close(); cur = []
        if not cur:
            cur.append(start)
        cur.append(end)  # curves keep end points only; ArcMap exports map lines as polylines
    close()
    return subs


def extract(layers=("ward_bnd", "municipal_bnd", "drain_line", "drain_poly", "canal", "river", "wb",
                    "Graticule")):
    doc = pymupdf.open(PDF)
    page = doc[0]
    names = layer_names(doc, page)
    spans = layer_spans(page.read_contents())
    result = {}
    for tag, name in names.items():
        if name not in layers or tag not in spans:
            continue
        drawings, h = drawings_for(spans[tag])
        rows = []
        for d in drawings:
            subs = [page_to_lonlat(s, h) for s in path_subpaths(d)]
            if not subs:
                continue
            meta = dict(layer=name, kind=d.get("type"), stroke=str(d.get("color")),
                        fill=str(d.get("fill")), width=d.get("width"))
            if d.get("fill") is not None:
                # Filled shape: even-odd combination of its rings (islands/holes stay correct).
                rings = [Polygon(s).buffer(0) for s in subs if len(s) >= 4]
                if rings:
                    geom = rings[0]
                    for r in rings[1:]:
                        geom = geom.symmetric_difference(r)
                    if not geom.is_empty:
                        rows.append(dict(meta, geometry=geom))
            else:
                rows += [dict(meta, geometry=LineString(s)) for s in subs]
        if not rows:
            print(f"{name:15s} EMPTY (no vector paths in this layer)")
            continue
        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        result[name] = gdf
        print(f"{name:15s} paths={len(gdf):5d} polys={(gdf.geom_type == 'Polygon').sum():5d}")
    return result


def layer_text(name):
    """Text drawn inside one layer: list of (text, lon, lat). Marker symbols (dotted
    boundaries) are drawn as font glyphs, so they come through here too."""
    doc = pymupdf.open(PDF)
    page = doc[0]
    tag = {v: k for k, v in layer_names(doc, page).items()}[name]
    span = layer_spans(page.read_contents())[tag]
    pre = (page.read_contents())[: page.read_contents().find(b"/OC")]
    raw = page.read_contents()
    doc.update_stream(int(doc.xref_get_key(page.xref, "Contents")[1].split()[0]), pre + raw[span[0]: span[1]])
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                txt = "".join(c["c"] for c in s["chars"])
                if s["font"] == "ESRIDefaultMarker":
                    for c in s["chars"]:   # glyph box centre = symbol location
                        x0, y0, x1, y1 = c["bbox"]
                        out.append(("marker", (x0 + x1) / 2, (y0 + y1) / 2))
                elif txt.strip():
                    x0, y0, x1, y1 = s["bbox"]
                    out.append((txt.strip(), (x0 + x1) / 2, (y0 + y1) / 2))
    if not out:
        return []
    ll = page_to_lonlat([(x, y) for _, x, y in out], page.rect.height)
    return [(t, lon, lat) for (t, _, _), (lon, lat) in zip(out, ll)]


def snap_dangles(lines, targets, tol):
    """Topology clean-up: extend each dangling ward-line end (one touching no other line)
    to the nearest point on any other linework within `tol` metres. Only line ends move,
    so thin features (drains, canals) are never collapsed the way a global snap would."""
    from shapely import STRtree
    from shapely.geometry import Point
    from shapely.ops import nearest_points

    pool = list(lines) + list(targets)
    tree = STRtree(pool)
    out = []
    moved = 0
    for i, ln in enumerate(lines):
        coords = list(ln.coords)
        for end in (0, -1):
            p = Point(coords[end])
            near = [j for j in tree.query(p.buffer(tol)) if j != i]
            best = min(near, key=lambda j: pool[j].distance(p), default=None)
            if best is None or pool[best].distance(p) > tol:
                continue
            q = nearest_points(pool[best], p)[0]
            d = p.distance(q)
            if d < 1e-9:
                continue                                  # exactly on another line: noded by union
            # Overshoot 0.5 m past the target so the lines truly cross and get noded;
            # the tiny stub left beyond the crossing is discarded by polygonize.
            ux, uy = (q.x - p.x) / d, (q.y - p.y) / d
            tip = (q.x + 0.5 * ux, q.y + 0.5 * uy)
            if end == 0:
                coords.insert(0, tip)
            else:
                coords.append(tip)
            moved += 1
        out.append(LineString(coords))
    print(f"snap_dangles: extended {moved} dangling ends (tol {tol} m)")
    return out


def build_wards(lines_gdf, divider_gdfs=()):
    """Municipal boundary from its marker dots; wards = polygonize(ward lines + boundary)."""
    from shapely.geometry import Point
    from shapely.ops import polygonize, unary_union

    dots = gpd.GeoSeries([Point(lon, lat) for t, lon, lat in layer_text("municipal_bnd") if t == "marker"],
                         crs="EPSG:4326").to_crs(CRS)
    spacing = np.median(dots.distance(dots.shift()).dropna())
    r = 40.0
    ring = unary_union(dots.buffer(r))
    ring = max(getattr(ring, "geoms", [ring]), key=lambda g: g.area)
    city = Polygon(ring.exterior).buffer(-r)
    print(f"municipal dots={len(dots)} median gap={spacing:.0f} m  city area={city.area / 1e6:.1f} km2")

    wl = lines_gdf.to_crs(CRS)
    # Where a river/canal/drain forms the ward limit the map draws no separate ward line,
    # so water edges are added as candidate dividers; unlabelled pieces are merged back by Eliminate.
    extra = []
    for g in divider_gdfs:
        for geom in g.to_crs(CRS).geometry:
            extra.append(geom.boundary if geom.geom_type in ("Polygon", "MultiPolygon") else geom)
    ward_lines = snap_dangles(list(wl.geometry), list(extra) + [city.exterior], tol=50.0)
    merged = unary_union(ward_lines + extra + [city.exterior])   # nodes all linework
    polys = [p for p in polygonize(merged) if p.area > 2_000 and city.buffer(50).contains(p.representative_point())]
    wards = gpd.GeoDataFrame(geometry=polys, crs=CRS)

    labels = [(t, lon, lat) for t, lon, lat in layer_text("ward_poly Anno") if re.fullmatch(r"W\d{1,2}", t)]
    lab = gpd.GeoDataFrame(dict(ward=[t for t, _, _ in labels]),
                           geometry=[Point(lon, lat) for _, lon, lat in labels], crs="EPSG:4326").to_crs(CRS)
    # Each ward label is drawn twice (text + halo), so collapse to one label per polygon.
    joined = gpd.sjoin(wards, lab, how="left", predicate="contains")
    clash = joined.groupby(level=0).ward.nunique()
    if (clash > 1).any():
        print("polygons holding >1 ward label (boundary gap):",
              [sorted(joined.loc[i].ward.unique()) for i in clash[clash > 1].index])
    joined = joined[~joined.index.duplicated()].drop(columns="index_right")
    print(f"polygons={len(joined)} labelled={joined.ward.notna().sum()} "
          f"unique labels={joined.ward.nunique()} slivers={joined.ward.isna().sum()}")
    return city, eliminate(joined), lab


def eliminate(polys):
    """GIS 'Eliminate': merge each unlabelled sliver into the labelled neighbour it
    shares the longest border with; repeat until no sliver touches a labelled ward."""
    geoms = dict(zip(polys.index, polys.geometry))
    owner = polys.ward.to_dict()
    pending = [i for i, w in owner.items() if not isinstance(w, str)]
    while pending:
        progressed = False
        for i in sorted(pending, key=lambda k: geoms[k].area):
            best, best_len = None, 0.0
            for j, w in owner.items():
                if isinstance(w, str) and geoms[j].distance(geoms[i]) < 1.0:
                    shared = geoms[j].boundary.intersection(geoms[i].boundary.buffer(1.0)).length
                    if shared > best_len:
                        best, best_len = j, shared
            if best is not None:
                owner[i] = owner[best]
                progressed = True
        pending = [i for i, w in owner.items() if not isinstance(w, str)]
        if not progressed:
            print(f"eliminate: {len(pending)} slivers touch no ward; dropped")
            break
    df = gpd.GeoDataFrame(dict(ward=[owner[i] for i in geoms]), geometry=list(geoms.values()), crs=polys.crs)
    df = df[df.ward.notna()].dissolve("ward").reset_index()
    # Snapping leaves hairline (<1 m) gaps between merged pieces: close them.
    df["geometry"] = df.geometry.buffer(1.0).buffer(-1.0)
    df["ward_no"] = df.ward.str[1:].astype(int)
    df["area_ha"] = (df.area / 1e4).round(1)
    return df.sort_values("ward_no").reset_index(drop=True)


def memory_guard(limit_gb=3.0):
    """Abort the process if resident memory passes limit_gb (8 GB machine safety)."""
    import os
    import threading

    import psutil

    proc = psutil.Process()

    def watch():
        while True:
            if proc.memory_info().rss > limit_gb * 1e9:
                print(f"memory_guard: RSS > {limit_gb} GB, aborting", flush=True)
                os._exit(3)
            threading.Event().wait(1.0)

    threading.Thread(target=watch, daemon=True).start()


if __name__ == "__main__":
    memory_guard()
    res = extract()
    OUT.unlink(missing_ok=True)
    for name, gdf in res.items():
        if len(gdf):
            gdf.to_crs(CRS).to_file(OUT, layer=name, driver="GPKG")
    city, wards, labels = build_wards(res["ward_bnd"], [res[k] for k in ("river", "canal", "drain_line", "drain_poly")])
    gpd.GeoDataFrame(geometry=[city], crs=CRS).to_file(OUT, layer="municipal_poly", driver="GPKG")
    wards.to_file(OUT, layer="wards", driver="GPKG")
    labels.to_file(OUT, layer="ward_labels", driver="GPKG")
    # Place-name labels (used to locate the CSP waterlogging hotspots, src.hotspots).
    import pandas as pd
    place = [(lyr.split()[0], t, lon, lat)
             for lyr in ("location Anno", "mouza_name Anno", "slum_location Anno", "junation Anno", "road_name Anno")
             for t, lon, lat in layer_text(lyr) if t != "marker"]
    pd.DataFrame(place, columns=["layer", "text", "lon", "lat"]).to_csv(INTERIM / "wardmap_labels.csv", index=False)
    print("saved", OUT)
