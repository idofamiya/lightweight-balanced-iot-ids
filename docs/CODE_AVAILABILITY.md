# Code and Data Availability

The code for the lightweight and concept-drift-aware IoT IDS experiments is available in this repository.

## Code

The main notebook is:

```text
notebooks/concept_drift_iot_ids_notebook_corrected.ipynb
```

It contains the experimental workflow for:

- static Random Forest Top-15 IDS,
- full-feature RF baseline,
- ADWIN-based drift monitoring,
- drift-triggered RF retraining,
- three-seed evaluation,
- phase-wise F1 and MCC evaluation.

## Data

The CICIoT2023 dataset is not included in this repository because of dataset size and external ownership. Users should obtain it from the official dataset provider and configure local paths in the notebook.

## Reproducibility

The notebook saves the key outputs under:

```text
results/concept_drift/
```

The following files summarize the reported experiments:

```text
three_seed_drift_experiment_summary.csv
phasewise_three_seed_results_summary.csv
phasewise_three_seed_results_paper_table.csv
phasewise_three_seed_events.csv
```
