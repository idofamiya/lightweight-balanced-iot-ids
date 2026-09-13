# scripts/

Standalone Python scripts that reproduce the corpus-scale, cross-validation,
sensitivity, and full-replication experiments reported in the L-CDAIDS paper.
These complement the notebooks in `notebooks/` and `src/`, which cover the
original 200k-sample binary and multiclass experiments.

All scripts follow the same convention: paths are resolved relative to the
repository root via `Path(__file__).resolve().parent.parent`, so they can be
run directly from a clone of this repository (`python3 scripts/<name>.py ...`)
without editing any hardcoded paths. Each script's own `--help` / argparse
flags document its run modes (per-seed run + `--aggregate`, in most cases).

Install requirements: `pip install pandas numpy scikit-learn matplotlib joblib river`

## Table -> script mapping

| Table(s) | Script | Output location |
|---|---|---|
| Table 3 (corpus composition) | `build_sample_fulldata.py` | `outputs/fulldata_drift/sample_meta.json` |
| Table 6 (5-fold cross-validation) | `cross_validation_tables_3_4.py` | `outputs/cross_validation/` |
| Table 7 (corpus-scale covariate shift) | `drift_experiment_fulldata.py --stream affine` | `outputs/fulldata_drift/affine_*` |
| Table 8 (corpus-scale attack-family holdout) | `drift_experiment_fulldata.py --stream family` | `outputs/fulldata_drift/family_*`, `FINAL_family_phasewise.csv` |
| Table 10 (chronological vs. random split) | `chrono_experiment.py` | `outputs/fulldata_drift/chrono_*`, `FINAL_chrono_phasewise.csv` |
| Table 11 (computational cost, ARF row) / Tables 12-13 (three-seed, reduced scale) | `arf_baseline.py` | `outputs/arf_baseline/` |
| Table 12 (three-seed overall) / Table 13 (phase-wise three-seed) / Table 15 (15-seed replication) / Table 16 (drift/retraining events) | `three_seed_concept_drift.py` | `outputs/concept_drift/reproduced/` |
| Table 14 (5-seed disentangling: forest config vs. split) | `disentangle_seed_vs_split.py` | `outputs/concept_drift/disentangle_seed_vs_split/` |

`make_corpus_fig.py` and `plot_fifteen_seed.py` are plotting helpers that turn
the aggregated CSVs above into the paper's corpus-scale and 15-seed comparison
figures; they are included for completeness, but the PNGs they produce are not
checked into this repository.

## Script notes

**`build_sample_fulldata.py`** draws a stratified sample from the full,
un-subsampled CICIoT2023 corpus (34,298,828 flows across 63 capture files)
with explicit per-family quotas, so that rare attack families (Brute Force,
Web/Malware) are present in large enough numbers to train and evaluate on.
Requires the 63 cleaned per-file parquet files under
`data/processed/full_corpus/` (not included in this repository — see
`data/README.md` for how to obtain and clean the raw CICIoT2023 release).
Writes `outputs/fulldata_drift/sample_familyholdout.parquet`, which the two
scripts below consume.

**`drift_experiment_fulldata.py`** runs the corpus-scale drift experiment in
two regimes: `--stream affine` (the paper's covariate-shift transform, applied
to standardised features) and `--stream family` (attack-family holdout: two
families are withheld from training and introduced mid-stream, which is
genuine concept drift rather than a rescalable shift). Both regimes compare
five models: Static RF Top-15, Static RF Full-39, Adaptive-Norm RF Top-15
(online re-standardisation, no retraining), Drift-Triggered RF Top-15 (ADWIN +
recent-window refit), and Adaptive Random Forest (river, prequential).

**`chrono_experiment.py`** evaluates a genuine file-order (chronological)
train/test split against a random-split control of identical size. The
CICIoT2023 CSV release carries no timestamp field, and the 63 files are
statistically near-homogeneous in attack-family composition (see
`per_file_family_counts.csv`), so this script does not expect file order to
carry real temporal drift signal — it measures that directly rather than
assuming it, which is why both arms are reported side by side.

**`cross_validation_tables_3_4.py`** repeats the paper's single 80:20-split RF
configuration (100 trees, max depth 20) across 5-fold stratified
cross-validation, for both the Full-39 and Top-15 feature sets, to report a
confidence interval rather than a single point estimate.

**`three_seed_concept_drift.py`** is a documented reconstruction of the
three-seed / 15-seed, phase-wise, ADWIN drift-triggered retraining experiment.
The original notebook that generated the paper's Tables 12/13/15/16 was not
preserved; this script rebuilds the same experiment from the paper's stated
methodology (fresh stratified 80:20 split per seed, Phase 2/3 covariate
shifts, ADWIN delta=0.002, 20,000-row retrain window). Per its own docstring:
the exact batching/triggering granularity of the original run is not
recoverable from the surviving result files, so this is a faithful
reconstruction rather than a byte-for-byte recovery — treat small numeric
differences from any earlier `results/concept_drift/*.csv` files as expected.

**`disentangle_seed_vs_split.py`** answers a specific follow-up question:
is seed-to-seed instability in the drift results caused by the Random Forest
configuration itself, or by reusing one fixed 80:20 split across all seeds?
It keeps the RF configuration identical to `three_seed_concept_drift.py` but
draws a fresh stratified split per seed as well, isolating the two sources of
variance (Table 14).

**`arf_baseline.py`** adds the Adaptive Random Forest (river) baseline to the
reduced (200k-sample) drift stream, matched to the same fixed 80:20 split,
Top-15 feature space, and three-phase stream used elsewhere in the paper's
Section 4.6. ARF is a pure-Python incremental learner and is comparatively
slow, so this runner is resumable and time-budgeted (`--budget` seconds per
invocation); re-invoke with the same `--seed` until it prints `DONE`.
