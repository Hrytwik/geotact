"""Locate the 16 waterlogging hotspots (WL1-WL16) listed in the Bhubaneswar City Sanitation
Plan (CDD Society 2017, Fig. 10; derived from the JNNURM stormwater drainage DPR).

Positions come from the BMC ward map's own georeferenced place-name labels
(data/interim/wardmap_labels.csv, extracted by src.wardmap), matched by hand.
Online geocoders were tried first and rejected: Nominatim was unreachable and Photon
returned one unrelated institute for five different neighbourhoods.

Four hotspots whose names are ambiguous (WL3, WL6, WL10, WL11: several Behera Sahi / Sabar Sahi in
the city, Odia spelling variants) were later moved to coordinates supplied by the authors on 2026-10-21
and checked against the BMC beat list, the ward map and press reports (docs/verification_round1.md).
Those entries use ("xy", lat, lon) instead of a map label.

Each point is a locality-level position, so uncertainty is a few hundred metres;
`confidence` records how specific the match is.
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.config import CRS, INTERIM

# id, CSP name, [(label layer, label text)], confidence, note
HOTSPOTS = [
    ("WL1", "Satabdi Nagar", [("location", "Satabdi")], "medium", "label reads 'Satabdi'"),
    ("WL2", "Acharya Vihar", [("junation", "Acharya Vihar")], "high", ""),
    ("WL3", "Behera Sahi (Nayapalli)", [("xy", 20.2849, 85.8200)], "medium",
     "Behera Sahi / Nuasahi, Nayapalli (beat-list ward 25/37); falls in ward map W37, 36 m from a drawn "
     "drain; replaces a label 5.2 km north"),
    ("WL4", "ISKCON Temple, N.H. East, Nayapalli", [("location", "ISKCON")], "high", ""),
    ("WL5", "Ganga Nagar, Aerodrome area, Unit-VI", [("junation", "Ganga Nagar")], "medium", ""),
    ("WL6", "Balitothasahi, near Jama Masjid, Nayapalli", [("xy", 20.2810, 85.8160)], "medium",
     "Balitota Sahi (beat-list ward 34, p.19) / Balitoka Sahi (ward 26, p.13); centroid of a stretch "
     "20.2783,85.8154 to 20.2837,85.8166; falls in W39, 421 m from nearest drawn drain"),
    ("WL7", "Ekambra Park", [("location", "Ekamra Kanan")], "high", "Ekamra Kanan park"),
    ("WL8", "Jharpada-Bamikhal-GGP Colony",
     [("slum_location", "Jharpada"), ("mouza_name", "BOMIKHAL"), ("location", "GGP Colony")], "high",
     "corridor of three named localities; kept as three points"),
    ("WL9", "Laxmisagar", [("location", "Laxmisagar")], "high", ""),
    ("WL10", "Sabarsahi", [("xy", 20.2595, 85.8475)], "medium",
     "Sabar Sahi occurs in beat-list wards 6, 31 and 49; chosen: Laxmisagar/Badagada (ward 49) because "
     "TOI Jul 2025 names Sabar Sahi, Laxmi Sagar, Badagada as flooded. Point sits on the Laxmisagar "
     "mouza label (W45), i.e. a locality centre. Rasulgarh candidate rejected (outside BMC); Siripur kept out"),
    ("WL11", "Gouri Nagar (Old Town)", [("xy", 20.2378, 85.8356), ("xy", 20.2405, 85.8403)], "medium",
     "Gouri Nagar, Old Town (beat-list ward 54, p.38): two ends of the low-lying stretch near "
     "Lingaraj / Kedargouri; both in ward map W60 (supplier said new ward 59, mismatch unresolved)"),
    ("WL12", "Samantarapur", [("junation", "Samantarapur")], "high", ""),
    ("WL13", "Jagamohan Nagar", [("location", "Jagamohan Nagar")], "high", ""),
    ("WL14", "Paika Nagar", [("mouza_name", "PAIKANAGAR")], "high", ""),
    ("WL15", "Shahid Nagar", [("mouza_name", "SAHEEDNAGAR")], "high", ""),
    ("WL16", "Mancheswar Industrial Area", [("location", "Industral")], "high",
     "map label 'Industral' inside Mancheswar"),
]


def main():
    labels = pd.read_csv(INTERIM / "wardmap_labels.csv")
    rows = []
    for wid, name, picks, conf, note in HOTSPOTS:
        for k, pick in enumerate(picks):
            sub = f"{wid}{'abc'[k]}" if len(picks) > 1 else wid
            if pick[0] == "xy":
                _, lat, lon = pick
                rows.append(dict(id=sub, hotspot=wid, name=name, label="manual:user-supplied+checked",
                                 confidence=conf, note=note, lon=lon, lat=lat))
                continue
            layer, text = pick
            hit = labels[(labels.layer == layer) & (labels.text == text)]
            if hit.empty:
                raise ValueError(f"{wid}: label {layer}/{text} not found")
            # labels are drawn twice (text + halo) a few metres apart: take the mean
            rows.append(dict(id=sub, hotspot=wid, name=name, label=f"{layer}:{text}", confidence=conf,
                             note=note, lon=hit.lon.mean(), lat=hit.lat.mean()))
        if not picks:
            rows.append(dict(id=wid, hotspot=wid, name=name, label=None, confidence=conf, note=note))
    df = pd.DataFrame(rows)
    df.to_csv(INTERIM / "hotspots.csv", index=False)
    ok = df.dropna(subset=["lon"])
    gdf = gpd.GeoDataFrame(ok, geometry=[Point(x, y) for x, y in zip(ok.lon, ok.lat)], crs="EPSG:4326")
    gdf.to_crs(CRS).to_file(INTERIM / "hotspots.gpkg", driver="GPKG")
    print(df.groupby("confidence").hotspot.nunique().to_dict(), f"points={len(ok)}")


if __name__ == "__main__":
    main()
