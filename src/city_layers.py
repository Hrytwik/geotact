"""T11: the layer file a transfer city needs (same layer names as Bhubaneswar's wardmap.gpkg), built from OpenStreetMap only.
municipal_poly = the OSM administrative relation given; wards = the whole municipality as one 'ward'; river = OSM river lines;
the other channel / water layers stay empty because the terrain step already adds OSM waterways and water bodies itself."""
import json

from shapely.geometry import LineString
from shapely.ops import linemerge, polygonize, unary_union

CITY_RELATION = {"cuttack": 10105896}          # 'Cuttack (M.Corp.) P.S', admin level 6: the municipal corporation's area in OSM
HOME_TILE_NORTH = 20.41                         # north edge of the Bhubaneswar tile the models were trained on


def polygon_from_members(members):
    lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]]) for m in members
             if m.get("type") == "way" and m.get("role", "outer") in ("outer", "") and len(m.get("geometry", [])) >= 2]
    polys = list(polygonize(linemerge(unary_union(lines))))
    return unary_union(polys)


def fetch_relation(rel_id):
    import requests
    q = f"[out:json][timeout:200];relation({rel_id});out geom;"
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=300, headers={"User-Agent": "geotact-research"})
    r.raise_for_status()
    return r.json()["elements"][0]


def build(city):
    import geopandas as gpd

    from src.config import BBOX, CRS, INTERIM, RAW
    INTERIM.mkdir(parents=True, exist_ok=True)
    from shapely.geometry import box

    from src.config import CITY
    rel = fetch_relation(CITY_RELATION[city])
    full = polygon_from_members(rel["members"])
    w, s, e, n = BBOX
    poly = full.intersection(box(w, max(s, HOME_TILE_NORTH), e, n))     # outside the Bhubaneswar training tile (no leakage)
    js = json.loads((RAW / "osm_raw.json").read_text())
    rivers = [LineString([(p["lon"], p["lat"]) for p in e["geometry"]]) for e in js["elements"]
              if e["type"] == "way" and e.get("tags", {}).get("waterway") == "river" and len(e.get("geometry", [])) >= 2]
    out = INTERIM / "wardmap.gpkg"
    out.unlink(missing_ok=True)
    gpd.GeoDataFrame(dict(name=[rel["tags"].get("name")]), geometry=[poly], crs=4326).to_crs(CRS).to_file(out, layer="municipal_poly", driver="GPKG")
    gpd.GeoDataFrame(dict(ward_no=[1]), geometry=[poly], crs=4326).to_crs(CRS).to_file(out, layer="wards", driver="GPKG")
    gpd.GeoDataFrame(geometry=rivers, crs=4326).to_crs(CRS).to_file(out, layer="river", driver="GPKG")
    for layer in ("drain_line", "canal", "drain_poly", "wb"):
        gpd.GeoDataFrame(geometry=[], crs=CRS).to_file(out, layer=layer, driver="GPKG")
    area_km2 = float(gpd.GeoSeries([poly], crs=4326).to_crs(CRS).area.iloc[0] / 1e6)
    full_km2 = float(gpd.GeoSeries([full], crs=4326).to_crs(CRS).area.iloc[0] / 1e6)
    print(f"{city}: boundary '{rel['tags'].get('name')}' {full_km2:.1f} km2; study area north of {HOME_TILE_NORTH} N inside the box "
          f"{area_km2:.1f} km2; river lines {len(rivers)}", flush=True)
    return area_km2


if __name__ == "__main__":
    from src.config import CITY
    build(CITY)
