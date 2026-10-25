"""T11: assembling a city boundary polygon from OSM relation member ways (synthetic)."""
from shapely.geometry import Polygon

from src import city_layers as cl


def test_outer_member_ways_are_joined_into_one_polygon():
    members = [dict(type="way", role="outer", geometry=[dict(lon=0, lat=0), dict(lon=1, lat=0)]),
               dict(type="way", role="outer", geometry=[dict(lon=1, lat=0), dict(lon=1, lat=1), dict(lon=0, lat=1)]),
               dict(type="way", role="outer", geometry=[dict(lon=0, lat=1), dict(lon=0, lat=0)]),
               dict(type="node", role="admin_centre", lat=0.5, lon=0.5)]
    poly = cl.polygon_from_members(members)
    assert poly.is_valid and abs(poly.area - 1.0) < 1e-9
    assert poly.equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
