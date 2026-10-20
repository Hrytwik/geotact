"""Locate the 16 waterlogging hotspots (WL1-WL16) listed in the Bhubaneswar City Sanitation
Plan (CDD Society 2017, Fig. 10; derived from the JNNURM stormwater drainage DPR).

Positions come from the BMC ward map's own georeferenced place-name labels
(data/interim/wardmap_labels.csv, extracted by src.wardmap), matched by hand.
Online geocoders were tried first and rejected: Nominatim was unreachable and Photon
returned one unrelated institute for five different neighbourhoods.

Each point is a locality-centre label, so positional uncertainty is a few hundred metres;
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
    ("WL3", "Behera Sahi (Nayapalli)", [("slum_location", "Behera Sahi")], "low",
     "only Behera Sahi on map is ~4 km N of Nayapalli; may be a different place"),
    ("WL4", "ISKCON Temple, N.H. East, Nayapalli", [("location", "ISKCON")], "high", ""),
    ("WL5", "Ganga Nagar, Aerodrome area, Unit-VI", [("junation", "Ganga Nagar")], "medium", ""),
    ("WL6", "Balitothasahi, near Jama Masjid, Nayapalli", [("location", "Nayapalli")], "medium",
     "Balitothasahi not labelled; Nayapalli locality label used"),
    ("WL7", "Ekambra Park", [("location", "Ekamra Kanan")], "high", "Ekamra Kanan park"),
    ("WL8", "Jharpada-Bamikhal-GGP Colony",
     [("slum_location", "Jharpada"), ("mouza_name", "BOMIKHAL"), ("location", "GGP Colony")], "high",
     "corridor of three named localities; kept as three points"),
    ("WL9", "Laxmisagar", [("location", "Laxmisagar")], "high", ""),
    ("WL10", "Sabarsahi", [], "excluded", "four different Sabarsahi labels on the map; ambiguous"),
    ("WL11", "Gouri Nagar (Old Town)", [("location", "Kedargouri")], "medium",
     "Gouri Nagar not labelled; Kedargouri (Old Town) used"),
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
        for k, (layer, text) in enumerate(picks):
            hit = labels[(labels.layer == layer) & (labels.text == text)]
            if hit.empty:
                raise ValueError(f"{wid}: label {layer}/{text} not found")
            # labels are drawn twice (text + halo) a few metres apart: take the mean
            sub = f"{wid}{'abc'[k]}" if len(picks) > 1 else wid
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
