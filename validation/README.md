# Validation labels

Hand labels used to measure accuracy. None of them is used for training, except where `docs/results_log.md` says a registered correction model was trained under spatial cross-validation.

| File | What it holds |
|---|---|
| `validation_labels.xlsx`, `validation_points.kml` | 300 land-use points, stratified by map class and labelled by the authors in Google Earth |
| `tree_loss_labels.xlsx`, `tree_loss_points.kml` | 40 hand-checked tree-loss locations |
| `vegetation_check/` | 100 stratified random points for the vegetation-loss map |
| `active_learning/`, `labels_to_recheck.csv` | extra labels and a list of points flagged for re-checking |
| `strata_weights.csv` | stratum sizes used by the design-based accuracy estimator |

## Files named `*_private_do_not_open.csv`

These hold the sampling design: stratum membership, pixel positions and the answer keys. They are named this way so that nobody opens them while labelling, which keeps the labels blind to the map's prediction. They are read only by the scoring code. They are published here so the accuracy estimates can be recomputed and audited.
