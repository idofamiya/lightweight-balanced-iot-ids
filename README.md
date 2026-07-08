# Lightweight Balanced IoT IDS

This repository contains reproducible code and experimental artefacts for a lightweight intrusion detection framework for resource-constrained IoT networks.

## Updated Study Direction

The project has been extended from a static lightweight IDS into:

**L-CDAIDS: A Lightweight Concept-Drift-Aware Intrusion Detection Framework for Resource-Constrained IoT Networks**

The extended framework combines:

- balanced binary IDS evaluation,
- Top-15 flow/statistical feature reduction,
- Random Forest based lightweight detection,
- ADWIN-based concept-drift monitoring,
- drift-triggered Random Forest retraining using a recent traffic window,
- phase-wise evaluation under a controlled evolving IoT traffic stream.

## Repository Structure

```text
notebooks/
  concept_drift_iot_ids_notebook_corrected.ipynb

results/
  concept_drift/
    concept_drift_model_comparison.csv
    concept_drift_model_comparison_with_rf_retraining.csv
    final_comparison_full39_top15_drift_rf.csv
    phasewise_three_seed_results_summary.csv
    phasewise_three_seed_results_paper_table.csv
    phasewise_three_seed_events.csv
    three_seed_drift_experiment_summary.csv
    three_seed_drift_experiment_raw.csv
    three_seed_drift_retraining_events.csv

figures/
  preprocessing_workflow.png
  rf_top15_drift_monitoring.png
  phasewise_f1_comparison.png
  phasewise_mcc_comparison.png

docs/
  CODE_AVAILABILITY.md
  UPDATE_NOTES.md
```

## Dataset

The experiments are based on CICIoT2023. The dataset is not redistributed in this repository. Users should download the dataset from the official Canadian Institute for Cybersecurity / UNB source and update the local paths in the notebook.

## Important Methodological Note

The concept-drift experiment uses a **controlled concept-drift simulation** by constructing an evolving stream from processed CICIoT2023 test data and introducing progressive covariate shifts across phases. It should not be described as naturally occurring chronological drift unless timestamp-based chronological ordering is separately verified.

## Main Result Summary

The drift-triggered RF Top-15 model improved robustness in the strongest drift phase compared with both the static Top-15 model and the static Full-39 baseline, while using only 15 features.

## Requirements

Recommended Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib joblib river
```

## How to Run

1. Open the notebook:

```text
notebooks/concept_drift_iot_ids_notebook_corrected.ipynb
```

2. Update the dataset paths according to your local machine.
3. Run all cells.
4. The result files will be generated in the output folder defined in the notebook.

## Citation

A manuscript based on this repository is under preparation/submission. Citation details will be added after acceptance/publication.
