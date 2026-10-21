# Build plan — GeoAI urban digital twin, Bhubaneswar (tasks T0-T11)

> **Summary:** Dependency-ordered, test-first plan for the build phase (grid, land-use U-Net, vegetation trend, flood model, accuracy estimator, graph twin, scenarios + SDG proxies, NSGA-II, optional transfer test), with each task's interface, tests, success evidence and known limitation. Executed one task at a time under the atomic-workflow discipline (no subagents; full suite run twice before each commit).
> **Last updated:** 2026-10-21

Status key: [ ] todo, [~] in progress, [x] done and verified.

## Dependency graph
T0 grid + study mask + test scaffold  -> everything
T5 vegetation trend        needs T0
T4 land-use U-Net          needs T0
T6 flood model             needs T0 (S1, DEM, hotspots, drains)
T7 accuracy estimator      needs T4 (code + tests now; real run waits for the authors' 300 labels)
T8 graph twin              needs T4, T5, T6 (+ GBIF, annual land cover)
T9 scenarios + SDG proxies needs T8
T10 NSGA-II                needs T9
T11 transfer test (bonus)  needs T4-T10; only if time allows (slow link: needs new downloads)
Order of execution: T0, T5, T4, T6, T7(code), T8, T9, T10, (T11).

## Common rules
- Grid: EPSG:32645, 10 m, the Sentinel-2 composite grid (2895 x 2424). Study mask: BMC municipal polygon (166.5 sq km). Wards raster 1-67.
- Every task: tests written first and seen failing; implementation; full suite twice; interface check against the next task's needs; then commit + doc + memory.
- Randomness: fixed seeds; stochastic results reported over repeated runs.
- Memory: 8 GB machine; memory_guard on every long run; no two heavy jobs at once.
- Honest labelling: every number in the chapter must trace to an artefact in outputs/ produced by a committed script.

## Tasks
### T0 grid and study mask
Interface: src.grid.load_grid() -> (transform, shape, crs); study_mask(); ward_raster(); read_aligned(path, resampling).
Tests: shape equals the S2 composite; mask area within 1 percent of 166.5 sq km; all 67 wards present, no pixel in two wards; DEM read_aligned returns grid shape.
### T5 vegetation health over time (index-based, NOT deep learning)
Method: NDVI from 8 Nov-Mar composites (2018-19 .. 2025-26), per-pixel least-squares slope + Mann-Kendall style significance, latest-year z-score anomaly versus the previous 7 years, ward and land-cover summaries.
Tests: synthetic stack recovers a known slope; NaN handling; anomaly sign; real data: NDVI in [-1,1], class ordering trees > crops > built > water.
Limitation: composites use the 5 clearest days per season, so year-to-year noise is partly scene selection; report slope uncertainty.
### T4 land-use model (U-Net, PyTorch MPS)
Inputs: S2 2023 composite (10 bands, offset-corrected, per-band standardised). Weak labels: Esri/IO LULC 2023 collapsed to the 6 chapter classes (Water, Trees, Cropland, Grass/Scrub, Built-up, Bare ground).
Split: spatial blocks (no patch leakage). Loss: class-weighted cross-entropy. Seeds fixed; train twice.
Tests: label mapping total; patches finite and correct shape; block split disjoint; model forward shape; tiny-batch overfit reaches > 95 percent; tiled inference reconstructs the full map without seams.
Evidence: spatial-validation agreement with the weak labels (agreement, NOT accuracy). Real accuracy only from the authors' 300 hand labels, compared for BOTH our map and the Esri map.
Limitation: the model learns another model's labels; any claim of accuracy waits for T7.
### T6 flood-prone zones
Labels: Sentinel-1 monsoon 2025 water extent (VV dB, speckle filtered, automatic threshold) minus permanent water (dry-season reference), per-pixel flood frequency; scenes chosen after heavy-rain days when ERA5 supports it.
Features: elevation, slope, HAND, TWI, curvature, log flow accumulation, distance to mapped drains/canals/river/water, WorldCover built-up fraction, median NDVI.
Model: gradient boosting with spatially blocked cross-validation. Check: percentile rank of the 19 hotspot points inside BMC, versus elevation-only and HAND-only baselines and a random baseline, with bootstrap intervals.
KNOWN GAP: SAR cannot see street flooding in dense built-up areas (double-bounce); labels mostly capture open-area inundation. The hotspot check is therefore the independent test, and the limitation goes in the chapter.
Tests: synthetic DEM with known HAND; threshold function on a synthetic bimodal image; permanent-water exclusion; spatial CV folds disjoint; hotspot-rank function on toy data.
### T7 accuracy estimator (code now, run when labels arrive)
Stratified estimator (Olofsson 2014, Stehman 2014): overall, user's and producer's accuracy with standard errors, from strata weights in validation/strata_weights.csv. Tests: known confusion matrices from hand calculation; labelled-vs-unsure handling; blind-labelling guard.
### T8 graph twin
Nodes: connected patches (water, green = trees + grass/scrub, farm = cropland) above a minimum area; attributes: area, class, ward, yearly land-cover shares 2017-2023, NDVI by year, flood susceptibility, GBIF records/species within a buffer. Edges: adjacency, habitat connectivity within a dispersal distance (weighted), flow links along drainage.
Tests: node count equals labelled components; edge symmetry where required; attributes complete; GBIF assigned once.
### T9 scenario engine + SDG-aligned proxies
Scenarios: wetland restoration, community gardens, tree corridors, as explicit edits to the land-use map and graph. Metrics: SCS curve-number runoff (uncalibrated; soil group assumed), connectivity (probability of connectivity), food-potential index (cropland area, NOT a yield forecast), public open space share (SDG 11.7.1 proxy from OSM green/open space).
Tests: baseline reproduces itself; each scenario changes only its intended cells; metrics move in the expected direction on toy graphs.
### T10 NSGA-II (pymoo)
Decision: choose intervention sites under an area budget. Objectives: reduce flood exposure, raise food-potential index, raise connectivity. Output: Pareto front, hypervolume, repeated seeds.
Tests: constraint handling; known toy Pareto front recovered; seeds reproduce.
### T11 transfer test (bonus)
Second, less-instrumented city (candidate: Cuttack). Same scripts with a new bbox; needs fresh downloads at about 1 MB/s. Do only if T0-T10 are done and verified.

## Known limitations to carry into the chapter
Weak labels (Esri) for T4; SAR blind to urban street flooding (T6); scene-selection noise (T5); CN soil assumption (T9); GBIF 98 percent birds and 71 percent from 2021+ (T8); hotspot positions locality-level (T6 check); no sensor data (Aurassure dropped); 2021 S2 composite is Jan-Apr only.
