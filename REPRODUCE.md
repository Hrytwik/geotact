# Reproduce the Bhubaneswar GeoAI twin from scratch

> **Summary:** Ordered command runbook (environment -> downloads -> every build step T0-T19 -> every table the chapter quotes), the data sources each download uses, the hand-made inputs that cannot be downloaded, and which steps were actually re-run to check this runbook.
> **Last updated:** 2026-10-29 (GEOTACT_CONTACT for downloads; third-party documents are local-only; written in improvement round 3, item T20)

Every protocol, decision rule and result referred to here is in `docs/results_log.md` (search the task id, e.g. `T7`). Tests: `python -m pytest -q`.

## 1. Environment

Python 3.12.13 in a uv virtual environment. Exact package versions are pinned in `requirements.lock.txt` (output of `uv pip freeze`).

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python -m pytest -q          # 308 tests; 267 run on synthetic data, 41 skip until the rasters are downloaded
```

The machine used was an 8 GB Apple-silicon Mac; heavy steps call `memory_guard(...)` and must be run ONE AT A TIME.

## 2. Inputs that cannot be downloaded

`data/` is git-ignored (8 GB of rasters), so check the "In git" column: inputs marked "no" must be copied over by hand.

| Input | Where | In git | What it is |
|---|---|---|---|
| Ward map GeoPDF | `Ward_Map_1.pdf` | no (third-party document, kept locally) | BMC ward map; wards, drains and water bodies are extracted from it (step T0) |
| City Sanitation Plan | `docs/submission/Bhubaneswar_City_Sanitation_Plan_NIUA.pdf` | no (third-party document, kept locally) | source of the 16 waterlogging hotspots WL1-WL16 (Fig. 10) |
| Field-survey flood points | `data/interim/panigrahi2026_fig3_points.csv` / `.gpkg` | yes (force-added 2026-10-27) | digitised from Fig. 3 of Panigrahi, Pal and Sharma (2026), AIMS Geosciences 12(1); the paper itself is `data/raw/external/panigrahi2026_geosci-12-01-005.pdf` (not in git) |
| Urban waterlogging places | `docs/urban_waterlogging_points.csv` | yes | audited newspaper/report places (audit in docs/results_log.md, 22 Oct) |
| 300 land-use hand labels | `validation/validation_labels.xlsx` + `strata_*.csv` | yes | stratified random sample, labelled by the first author in Google Earth |
| 40 tree-loss hand labels | `validation/tree_loss_labels.xlsx` + key | yes | targeted check of the U-Net tree-loss map |
| 100 vegetation-loss labels | `validation/vegetation_check/` | yes | stratified random check of the T15 loss map |
| State yields | `data/raw/external/moa/` | no (10 MB PDF; re-downloadable with `glance` while the site is up) | Agricultural Statistics at a Glance 2024-25 |
| Food energy values | `data/raw/external/usda_fdc.json` | no (re-fetched by the food code) | USDA FoodData Central values used in T14 |

Files named `*_private_do_not_open.csv` hold the sample design (strata, pixel positions). They are read only by the scoring code.

## 3. Downloads (`python -m src.download <name> ...` or `all`)

Set `GEOTACT_CONTACT=<your email>` before downloading: public APIs (Overpass, GBIF) ask for a contact address in the User-Agent; without it, requests go out without one.

| Name | Source |
|---|---|
| `dem` | Copernicus GLO-30 DEM (Microsoft Planetary Computer STAC) |
| `landcover` | ESA WorldCover 2021; Esri / Impact Observatory annual LULC 2017-2023 (Planetary Computer) |
| `sentinel2`, `ndvi` | Sentinel-2 L2A composites (2021, 2023, 2025-26) and eight dry-season NDVI composites 2018-19 to 2025-26 (Planetary Computer) |
| `sentinel1` | Sentinel-1 GRD monsoon and dry scenes 2025 (Planetary Computer) |
| `lst` | Landsat 8/9 Collection 2 Level-2 thermal band, summers 2025 and 2026 (Planetary Computer) |
| `rain` | ERA5 daily rainfall (Open-Meteo archive API) |
| `gbif` | GBIF bird occurrences (api.gbif.org) |
| `osm`, `roads` | OpenStreetMap open space and highways (Overpass API) |
| `aef` | AlphaEarth Foundations annual embedding 2025 (source.coop, CC-BY 4.0, Google and Google DeepMind) |
| `glance` | Agricultural Statistics at a Glance 2024-25 (nccd.gov.in) |
| GHSL | population / built-up tiles in `data/raw/ghsl` (JRC GHSL, tile R7_C27) |

Imagery searches are date-bounded in `src/download.py` and `src/config.py`, but the archives are live services: a re-download months later can return slightly different scene lists. The scene lists actually used are saved next to each raster (`*_scenes.json`, `data/interim/s1_inventory.csv`).

## 4. Build, in order (each line: task id, command)

```
# T0 study area, wards, drains, hotspots, validation sample
python -m src.wardmap
python -m src.hotspots
python -m src.validation_points                     # 300-point sample (already labelled: do not regenerate over validation/)
# T4 land use (U-Nets on Esri labels, consensus labels, multi-season) and the per-pixel baseline
python -m src.landuse --seed 0 ; python -m src.landuse --seed 1
python -m src.landuse --seed 0 --tag cons_seed0 --labels consensus ; python -m src.landuse --seed 1 --tag cons_seed1 --labels consensus
python -m src.landuse --seed 0 --tag ms_seed0 --labels consensus --extra ndvi ; python -m src.landuse --seed 1 --tag ms_seed1 --labels consensus --extra ndvi
python -m src.landuse_baseline
# T5 vegetation over time; T5b-T5e checks
python -m src.vegetation
python -m src.vegetation_crosscheck
python -m src.tree_loss_points
python -m src.vegetation_dl ; python -m src.vegetation_dl --attempt2 ; python -m src.vegetation_dl --bitemporal
# T6 flood (radar evidence, susceptibility, hotspot test, field-survey models, tuning, urban model, retry)
python -m src.terrain
python -m src.flood_labels
python -m src.flood_model
python -m src.flood_hotspots                        # one-shot test, already spent: re-running repeats, it does not re-test
python -m src.flood_inventory
python -m src.flood_cnn ; python -m src.flood_cnn_baselines ; python -m src.flood_cnn_r3
python -m src.flood_tuning --tabular ; python -m src.flood_cnn --tune-mlp ; python -m src.flood_tuning
python -m src.flood_urban ; python -m src.flood_urban --final
python -m src.flood_dl_retry --transfer ; python -m src.flood_dl_retry --more-positives ; python -m src.flood_dl_retry --learning-curve ; python -m src.flood_dl_retry --tuned
# T7 land-use accuracy; T4c ensemble + correction model (the land-use PRODUCT, landuse_2023_corr.tif)
python -m src.landuse_accuracy
python -m src.landuse_improve --ensemble
python -m src.landuse_improve --score --with-ms   # also fits the correction model, saves correction_model.pkl and writes landuse_2023_corr.tif
# T8-T10 twin, scenarios, optimisation (v2 = on the product map)
python -m src.twin --v2 ; python -m src.scenarios --v2 ; python -m src.optimise --v2
python -m src.sdg
# T12 live twin; T12b bird habitat; T13 land-use push
python -m src.live --auto
python -m src.biodiversity
python -m src.landuse_push
# T14 heat, food, four objectives (v3)
python -m src.heat outputs/landuse/landuse_2023_corr.tif
python -m src.scenarios --v3 landuse_2023_corr.tif
python -m src.optimise --v3 landuse_2023_corr.tif
python -m src.sdg --v3
# T15-T19 round 3
python -m src.vegetation_check            # builds the loss map and the blind sample (already labelled: do not rerun over the sheet)
python -m src.vegetation_check --score
python -m src.vegetation_v2
python -m src.landuse_extra
python -m src.optimise_convergence
```

Seeds are fixed in the code, so the scoring steps are deterministic. U-Net training on a different machine (CPU vs Apple MPS) is NOT expected to reproduce weights bit for bit; everything downstream of the saved maps is.

## 5. Re-run checks of this runbook

See the T20 RESULT paragraph in `docs/results_log.md` for which steps were re-run on 2026-10-27 and whether their outputs matched the committed numbers.
