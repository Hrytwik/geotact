# Figure captions — chapter Livelihood065

> **Summary:** Draft caption for each of the 8 chapter figures in docs/figures/ (files, what each shows, numbers and sources); paste into the chapter text, since Springer wants captions in the text file, not in the images.
> **Last updated:** 2026-10-29 (first version; figures made by `python -m src.figures`)

Files: each figure is a 600 dpi LZW-compressed TIFF for submission plus a PNG preview; Fig. 2 (a diagram) also comes as EPS. Springer numbers figures by chapter ("Fig. X.1"), so replace "Fig. 1" with the chapter number once the editors assign it. Every figure is drawn by us from our own data. None reproduces a figure from another publication, so no permission form is needed for the figures themselves.

**Fig. 1 — `fig01_study_area`.** Study area: the Bhubaneswar Municipal Corporation (BMC) boundary and its 67 wards, rivers, canals, water bodies and drains (vector layers extracted from the BMC ward map GeoPDF), and the 16 waterlogging hotspots listed in the City Sanitation Plan (CDD Society 2017), placed at locality level (about 200–400 m). Numbers are hotspot IDs; some hotspots were placed at more than one point (19 points in total).

**Fig. 2 — `fig02_workflow`.** Workflow of the GeoAI urban digital twin. Open data feed three GeoAI layers (land use, vegetation change, flood susceptibility). These build a graph twin of water, green, farm and habitat patches, which drives the scenario engine and the four-objective optimisation. Every test's decision rule was fixed before its result.

**Fig. 3 — `fig03_landuse`.** (a) Land-use product map for 2023: a U-Net ensemble trained on Sentinel-2 and corrected with 300 hand-labelled points; overall accuracy 67.8% (95% CI 61.5–74.2) against 54.0% for the Esri 2023 product on the same points. (b) Each class shown on its own, with its share of the municipal area, as a second encoding for readers with colour-vision deficiency.

**Fig. 4 — `fig04_vegetation`.** (a) Change in normalised NDVI between the 2021–22/2022–23 and 2024–25/2025–26 dry seasons on land that was vegetated in 2023 (red: greenness loss; blue: gain). (b) Pixels flagged as vegetation loss by a threshold fixed in advance from 38 hand-checked places (482.6 ha of 8,300.6 ha vegetated). The index ranks real tree loss well (AUC 0.905), but a blind random check found that only 45% of flagged pixels had truly lost vegetation, so (b) is a screening map, not an area estimate.

**Fig. 5 — `fig05_flood`.** Flood susceptibility (30 m) from a logistic model trained on the 129 field-surveyed flood and 25 non-flood locations digitised from Panigrahi et al. (2026); pooled spatial cross-validation AUC 0.766 (the best ensemble reached 0.785). Circles: field-surveyed flood sites; squares: field-surveyed non-flood sites; triangles: sanitation-plan waterlogging hotspots. The model does not rank the hotspots highly: their mean percentile is 47.6, about what random scores give (about 52) and below the pre-set bar of 70. White areas inside the boundary (rivers, large water bodies) have no score.

**Fig. 6 — `fig06_twin_graph`.** The graph twin on the product map: 475 patches (14 water, 371 green, 90 farm) as nodes, linked by habitat-connectivity, adjacency and downhill drainage edges. Dots mark patch centroids; edges are drawn as straight lines between centroids.

**Fig. 7 — `fig07_scenarios`.** Intervention sites in 1-ha blocks. (a) The hand-made combined scenario (151 ha of wetland restoration, tree corridors and community gardens). (b) The balanced (knee) plan from the four-objective optimisation (149 ha: 67 ha wetland, 45 ha corridors, 37 ha gardens), which lowers storm runoff by 1.96%, adds 94 ha of connected habitat and lowers residents' heat exposure by 0.015 °C, at a cost of 845 GJ of food energy per year.

**Fig. 8 — `fig08_tradeoffs`.** Exact four-objective trade-offs of the optimised 150-ha plans on the final non-dominated front: storm-runoff change against (a) food-energy change and (b) connected-habitat gain, coloured by the change in residents' summer surface-temperature exposure (darker = more cooling). Star: balanced (knee) plan; diamond: hand-made combined scenario (dominated by optimised plans); cross: today. About 30% of front plans improve all four objectives at once. The changes are modest at this budget.
