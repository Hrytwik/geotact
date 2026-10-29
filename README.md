# geotact — GeoAI urban digital twin for Bhubaneswar

> **Summary:** Code, results and writing material for the Springer book chapter Livelihood065 (due 31 Oct 2026): an open-data GeoAI digital twin of Bhubaneswar, Odisha, coupling blue-green infrastructure, urban agriculture and biodiversity, with land-use, vegetation and flood models, a graph twin, intervention scenarios and four-objective optimisation. Build finished; chapter writing not started.
> **Last updated:** 2026-10-29

## If you are writing the chapter, start here

1. **This README:** what the project is and where everything lives.
2. **`docs/WRITING_GUIDE.md`:** Springer's rules, the section-by-section outline, every number to quote, tables and figures, citations, declarations and what is still missing.
3. **`docs/claims_restated.md`:** the restated abstract (196 words) and how each promise from the accepted abstract may be worded now.
4. **`docs/claims_status.md`:** each promised claim and how far the build actually got.
5. **`docs/introduction_claims_to_verify.md`:** 26 claims the Introduction must source, soften or drop.
6. **`docs/chapter/01_introduction.md`:** the Introduction draft (needs items 3 and 5 applied).
7. **`docs/references.md`:** verified references (APA-style entries; cite in text as Harvard with no comma, e.g. (Batty 2018)).
8. **`docs/build_plan.md`:** the full record. Every test's rule was written before its result, and every result is recorded with its date. Search by task id (T7, T15, T19b...). It is long; read only the sections you need.

The format documents (Springer guideline, sample chapter, accepted abstract with author contacts, City Sanitation Plan) and the BMC ward map are **not in this public repository**: they are third-party or contain personal details. Ask Hritwik for them; their rules are summarised in `docs/WRITING_GUIDE.md` §2.

## What was built (one paragraph)

The twin is built from open satellite and global data and checked against hand labels, not against other models' maps.

- **Land use:** U-Net land-use maps from Sentinel-2, corrected with 300 hand-labelled points, reach 67.8% accuracy, against 54.0% for the Esri global product on the same points.
- **Vegetation:** an eight-season greenness record and a simple greenness-drop index rank real tree loss well (AUC 0.905). A blind random check showed it is not yet reliable as an area map.
- **Flood:** susceptibility is learned from 129 field-surveyed flood sites (AUC up to 0.785) but does not match the official hotspot list.
- **Graph twin:** water, green, farm and habitat patches are linked in a graph with yearly history, and it rebuilds from each new clear satellite window.
- **Scenarios and trade-offs:** wetland, garden and corridor scenarios are scored on runoff, food energy, habitat connectivity and residents' heat exposure. A four-objective NSGA-II search finds that about 30% of optimal 150-hectare plans improve all four at once; the gains are modest.
- **Reproducibility:** a fresh install from the lock file passed the full test suite (303 tests at the time) and rebuilt the land-use map exactly.

## Status

| Part | State |
|---|---|
| Build (tasks T0–T20) | Done; every result in `docs/build_plan.md` |
| Claims review | Done: `docs/claims_status.md`, `docs/claims_restated.md` |
| Introduction | Draft, on hold for sources (`docs/introduction_claims_to_verify.md`) |
| Figures | Done: 8 figures in `docs/figures/` with draft captions |
| Other sections, abstract, Word file | Not started |
| Second-city transfer (Cuttack) | Dropped by the authors; the chapter says "designed to transfer, not tested" |

## Repository map

```
geotact/
├── README.md                  this file
├── REPRODUCE.md               how to rebuild everything (environment, downloads, commands in order)
├── PLAN.md                    work queue and round-by-round history
├── requirements.lock.txt      exact package versions (Python 3.12)
├── src/                       one module per task (docstring line 1 names the task id)
├── tests/                     308 tests, synthetic data only
├── validation/                hand labels: 300 land-use points, 40 tree-loss points, 100 vegetation-change points
├── data/interim/              small derived inputs kept in git (digitised field-survey flood points)
└── docs/
    ├── WRITING_GUIDE.md       start here to write
    ├── claims_status.md       claims vs achievement
    ├── claims_restated.md     strongest honest wording + restated abstract
    ├── introduction_claims_to_verify.md
    ├── build_plan.md          every protocol and result (T0–T20, T19b)
    ├── references.md          verified reference list
    ├── verification_round1.md source checks of user-supplied claims
    ├── chapter/               chapter text drafts
    ├── figures/               the 8 chapter figures (TIFF/PNG/EPS) + captions.md
    └── *.csv                  result tables quoted in the chapter
```

## What is NOT in this repository

The raw and derived rasters (`data/`, about 8 GB) and the model outputs (`outputs/`, about 0.9 GB) stay on the authors' machine. Downloads are listed in `REPRODUCE.md`. The chapter figures are already exported to `docs/figures/` (600 dpi TIFF + PNG previews, captions in `docs/figures/captions.md`). Any new or changed figure needs Hritwik's machine. All the numbers needed for text and tables are already in `docs/`.

## Ground rules for the writing

- **Only quote numbers that appear in `docs/build_plan.md`** (or the CSV tables in `docs/`), and attach the task id when you take one.
- **Report failures as they are.** The vegetation area check failed; the flood model misses the official hotspots; deep learning did not beat simple models for flood and vegetation; the bird-habitat layer was not adopted. The chapter's credibility comes from saying this plainly.
- **Follow the "do not say" list** in `docs/WRITING_GUIDE.md` §8.
- **Do not add a reference without checking it** by DOI or the original document.
