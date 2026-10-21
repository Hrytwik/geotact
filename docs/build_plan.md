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

## Task status
- [x] T0 grid + study mask (7 tests, suite run twice, committed)
- [x] T5 vegetation trend (12 tests; suite 19/19 twice; outputs/vegetation/)
- [x] T4 land-use U-Net (40 tests incl. outputs; suite run twice; outputs/landuse/)
- [ ] T6  - [ ] T7 (code)  - [ ] T8  - [ ] T9  - [ ] T10  - [ ] T11 (bonus)

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
Limitation: composites use the 5 clearest days per season.
RESULT (2026-10-21): (1) The raw yearly NDVI is NOT comparable: unchanged land (same Esri class 2017-2023) shifts together by -0.044..+0.057 between years in every class, and acquisition date explains none of it (R2 0.01). A common additive year offset is estimated on unchanged non-water land and removed (std of city median 0.045 -> 0.011; residual class-specific shifts of about 0.02-0.03 remain, e.g. stable cropland drifts +0.0056/yr). (2) The ORIGINAL pre-written check (all vegetation -> built-up must trend down more than stable built-up) FAILED, and the threshold was not loosened. Cause: November-March cropland and scrub are already bare/low-NDVI, so crop->built leaves almost no signal (0.386 -> 0.372, median slope +0.0004/yr). Tree->built does: 0.505 -> 0.452, median slope -0.0057/yr vs -0.0006/yr for stable built-up (Mann-Whitney p 3e-202, P(converted slope above stable) about 0.40). Test rewritten to the supported claim only. (3) Per-pixel significance is weak (9.7 percent of tree->built pixels significant vs 6.7 percent of stable built); use ward-level aggregates, not pixel maps, in the chapter. The p-values are not corrected for multiple testing or spatial autocorrelation.
### T4 land-use model (U-Net, PyTorch MPS)
Inputs: S2 2023 composite (10 bands, offset-corrected, per-band standardised). Weak labels: Esri/IO LULC 2023 collapsed to the 6 chapter classes (Water, Trees, Cropland, Grass/Scrub, Built-up, Bare ground).
Split: spatial blocks (no patch leakage). Loss: class-weighted cross-entropy. Seeds fixed; train twice.
Tests: label mapping total; patches finite and correct shape; block split disjoint; model forward shape; tiny-batch overfit reaches > 95 percent; tiled inference reconstructs the full map without seams.
Evidence: spatial-validation agreement with the weak labels (agreement, NOT accuracy). Real accuracy only from the authors' 300 hand labels, compared for BOTH our map and the Esri map.
Limitation: the model learns another model's labels; any claim of accuracy waits for T7.
RESULT (2026-10-21), agreement with the weak Esri labels on held-out 512 px blocks, NOT accuracy: U-Net seed 0 overall 84.74 percent, macro-F1 0.745; seed 1 84.73 percent, macro-F1 0.754 (training 91.9 / 92.1 percent). Majority-class baseline 41.5 percent. Per-pixel LightGBM baseline on the same bands and labels: 70.0 percent, macro-F1 0.628, so spatial context adds about 15 points. Per class (seed 0, F1): built-up 0.93, trees 0.88, cropland 0.86, water 0.79, grass/scrub 0.54, bare ground 0.47 (precision 0.31, only 4,576 held-out pixels). Grass/scrub is confused mainly with cropland and trees; water pixels are sometimes called bare. Seeds agree on 94.4 percent of BMC pixels, but only 49.9 percent of pixels either seed calls grass/scrub (built-up 96.6, water 85.3, bare 85.0, trees 80.4, cropland 73.1). Class shares inside BMC differ between seeds by about 1.5 points for cropland (12.4 vs 10.9) and grass/scrub (5.9 vs 7.3); built-up 73.4 vs 73.6. Primary map: seed 0 (declared before running); report seed spread for crop/grass areas. 2023 composite -> 2025-26 composite (same training statistics): built-up rises 0.6-0.9 points (no collapse) but grass/scrub falls about 2 points in both seeds, probably seasonal (Nov-Feb vs Jan-Apr + Oct-Nov scenes), so do not read 2023 -> 2025 grass/crop changes as land change.
INCIDENT: the first seed-0 run crashed (SIGSEGV in libomp) when LightGBM started after PyTorch in one process (two OpenMP runtimes). Fixed by process isolation: src/landuse.py imports torch lazily, the baseline lives in src/landuse_baseline.py and runs separately; two regression tests prove neither module drags torch into the baseline process (both failed on the old code). The U-Net weights had been saved before the crash, so seed 0 was evaluated from the saved weights (identical numbers to the training log).
### T6 flood-prone zones
Labels: Sentinel-1 monsoon 2025 water extent (VV dB, speckle filtered, automatic threshold) minus permanent water (dry-season reference), per-pixel flood frequency; scenes chosen after heavy-rain days when ERA5 supports it.
Features: elevation, slope, HAND, TWI, curvature, log flow accumulation, distance to mapped drains/canals/river/water, WorldCover built-up fraction, median NDVI.
Model: gradient boosting with spatially blocked cross-validation. Check: percentile rank of the 19 hotspot points inside BMC, versus elevation-only and HAND-only baselines and a random baseline, with bootstrap intervals.
DATA FACT (checked 2026-10-21, data/interim/s1_inventory.csv): of 33 monsoon scenes only 25 cover more than 95 percent of the study area (8 cover under 50 percent, edge-of-swath); of 26 dry scenes 18 are usable (1 failed download). Use only full-coverage scenes. Heaviest ERA5 day (3 Sep, 72 mm) has no scene within 3 days; usable scenes with more than 15 mm the day before: 27 Jun, 24 Jul, 26 Aug (24.4 mm). Scenes alternate in 3-day pairs (two orbits), so thresholds are set per scene.
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
