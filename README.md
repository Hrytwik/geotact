# geotact: GeoAI urban digital twin for Bhubaneswar

Code, tests, validation labels and result tables for the book chapter *GeoAI-Powered Urban Digital Twin for Integrated Blue-Green Infrastructure, Agriculture, and Biodiversity Planning: A Tier-II City Case Study from Bhubaneswar*.

The pipeline builds an open-data urban digital twin of Bhubaneswar, Odisha (67 wards, 166.5 km²) that couples blue-green infrastructure, urban agriculture and biodiversity in one spatial model. Every GeoAI layer is checked against hand labels or field-survey data rather than against another model's map, and every decision rule was written down before its result was computed.

## Authors

Ridanshi Agarwal, Hritwik Bali and Parthiban Krishnamoorthy, School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India. See `CITATION.cff` to cite this work.

## Main results

| Layer | Result | Checked against |
|---|---|---|
| Land use | 67.8% overall accuracy (95% CI 61.5–74.2) from a U-Net corrected with hand labels, against 54.0% for the Esri 2023 product | 300 stratified hand-labelled points |
| Vegetation change | A greenness-drop index ranks real tree loss with AUC 0.905. As an area map it did not meet its pre-set bars (user's accuracy 0.45, producer's 0.18) | 38 hand-checked points and a blind sample of 100 |
| Flood susceptibility | Spatial-CV AUC up to 0.785 on 129 field-surveyed flood sites. It does not reproduce the municipality's 16 documented waterlogging hotspots | Field survey; City Sanitation Plan hotspots |
| Graph twin | 475 nodes (14 water, 371 green, 90 farm) with yearly history. It rebuilds from each new clear Sentinel-2 window | Three dated snapshots |
| Scenarios and optimisation | A four-objective NSGA-II search (runoff, food energy, habitat connectivity, heat exposure). About 30% of front plans improve all four objectives over today. Gains at a 150 ha budget are modest | 16,142 plans re-evaluated exactly; stable under a 33× longer search |
| Reproducibility | A fresh install from the lock file passes the full test suite and rebuilds the land-use map with zero differing pixels | `pytest` on the full data: 308 tests |

Failures are reported as they are. The full record of each test, rule and outcome is in [`docs/results_log.md`](docs/results_log.md); search it by task id (T7, T15, T19b and so on).

## Repository layout

```
geotact/
├── src/                  one module per task; the first docstring line names the task id
├── tests/                308 tests; 267 run on synthetic data alone, the rest skip unless the downloaded rasters are present
├── validation/           hand labels and sampling design (see validation/README.md)
├── data/interim/         small derived inputs: the digitised field-survey flood points
├── docs/
│   ├── results_log.md    protocols, decision rules and results for every task
│   ├── figures/          the eight chapter figures (TIFF, PNG, EPS) and captions
│   └── *.csv             result tables quoted in the chapter
├── REPRODUCE.md          ordered runbook: environment, downloads, commands
└── requirements.lock.txt exact package versions (Python 3.12)
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.lock.txt
python -m pytest -q              # 267 pass on synthetic data; 41 skip without the downloaded rasters
```

Rebuilding the maps needs several gigabytes of satellite and elevation downloads, and heavy steps are run one at a time on an 8 GB machine. `REPRODUCE.md` lists every download, the order of steps and the expected outputs.

## What is not in this repository

- Raw and derived rasters (`data/`, about 8 GB) and model outputs (`outputs/`, about 0.9 GB). They are rebuilt by the commands in `REPRODUCE.md`.
- Third-party documents that are not ours to redistribute: the Bhubaneswar Municipal Corporation ward map and the City Sanitation Plan. `REPRODUCE.md` says how they are used.
- Second-city transfer: the code is prepared, but the transfer was not run, and the chapter says the method is designed to transfer, not shown to.

## Data sources and licence

Inputs are open datasets: Sentinel-1 and Sentinel-2, Copernicus GLO-30, ESA WorldCover, Esri/Impact Observatory land cover, Landsat, ERA5, GBIF, OpenStreetMap and GHSL. Each keeps its own licence; see `REPRODUCE.md`. Code and original content in this repository are released under the MIT licence (`LICENSE`).
