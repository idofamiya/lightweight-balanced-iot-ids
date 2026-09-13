# Code and Data Availability

The code for the lightweight and concept-drift-aware IoT IDS experiments (L-CDAIDS) is available in this repository.

## Code

Two complementary layers of code are provided.

**Notebooks and `src/`** (`notebooks/`, `src/`) cover the original 200k-sample binary and multiclass experiments: the static Random Forest Top-15 and full-feature baselines, ADWIN-based drift monitoring, drift-triggered RF retraining, and the multiclass extension. The main notebook is:

```text
notebooks/concept_drift_three_seed_FULL_recovered.ipynb
```

**`scripts/`** contains standalone Python scripts that reproduce the corpus-scale, cross-validation, sensitivity, and full-replication experiments reported in the paper. Each script resolves paths relative to the repository root (`Path(__file__).resolve().parent.parent`), so it can be run directly from a clone with `python3 scripts/<name>.py ...`. See `scripts/README.md` for full usage notes and a table-to-script mapping. Install requirements: `pip install pandas numpy scikit-learn matplotlib joblib river`.

| Table(s) | Script |
|---|---|
| Table 3 (corpus composition) | `scripts/build_sample_fulldata.py` |
| Table 6 (5-fold cross-validation) | `scripts/cross_validation_tables_3_4.py` |
| Table 7 (corpus-scale covariate shift) | `scripts/drift_experiment_fulldata.py --stream affine` |
| Table 8 (corpus-scale attack-family holdout) | `scripts/drift_experiment_fulldata.py --stream family` |
| Table 10 (chronological vs. random split) | `scripts/chrono_experiment.py` |
| Table 11 (computational cost, ARF row) | `scripts/arf_baseline.py` |
| Tables 12/13 (three-seed, reduced scale) | `scripts/three_seed_concept_drift.py` |
| Table 14 (5-seed disentangling: forest config vs. split) | `scripts/disentangle_seed_vs_split.py` |
| Tables 15/16 (15-seed replication, drift/retraining events) | `scripts/three_seed_concept_drift.py` |

## Data

The CICIoT2023 dataset is not included in this repository because of dataset size and external ownership. Users should obtain it from the official dataset provider; see `data/README.md` for the expected layout (`data/raw/` for the original CSVs, `data/processed/` for generated/cleaned files, including the 63 per-file parquet files under `data/processed/full_corpus/` needed by the corpus-scale scripts).

## Reproducibility

Outputs from the `scripts/` runs used for the paper's tables are checked into `outputs/`:

```text
outputs/fulldata_drift/            Tables 3, 7, 8, 10 (corpus-scale drift, chronological-split)
outputs/cross_validation/          Table 6 (5-fold cross-validation)
outputs/arf_baseline/              Table 11 (Adaptive Random Forest baseline)
outputs/concept_drift/reproduced/  Tables 12, 13, 15, 16 (three-seed / 15-seed reconstruction)
outputs/concept_drift/disentangle_seed_vs_split/  Table 14 (seed vs. split sensitivity)
outputs/multiclass/                multiclass extension figures and reports
```

The two 15-seed comparison figures (`phasewise_f1_comparison_15seed.png`, `phasewise_mcc_comparison_15seed.png`) and the large ARF checkpoint file are not checked into the repository; they can be regenerated locally from the CSVs above using `scripts/plot_fifteen_seed.py` and `scripts/arf_baseline.py` respectively.

`scripts/three_seed_concept_drift.py`'s docstring documents that the original notebook behind the paper's three-seed tables was not preserved in full; the script is a faithful, methodology-based reconstruction rather than a byte-for-byte recovery, so small numeric differences from any earlier result files are expected.
