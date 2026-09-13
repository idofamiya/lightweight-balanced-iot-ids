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
- phase-wise evaluation under a controlled evolving IoT traffic stream,
- an extended multiclass attack-category classification study.

## Repository Structure

```text
notebooks/
  concept_drift_three_seed_FULL_recovered.ipynb   # main notebook: three-seed, phase-wise,
                                                    # and drift-triggered-retraining experiment
  concept_drift_iot_ids_notebook_corrected.ipynb   # earlier single-run draft, kept for history
  IoTIDS_balanced_classifier_comparison.ipynb
  IoTIDS_balanced_multiclass_extension.ipynb

src/
  classifier_comparison.py
  multiclass_attack_classification.py              # extended multiclass attack-category study

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

### Concept-drift experiment (three-seed, phase-wise, drift-triggered retraining)

1. Open the notebook:

```text
notebooks/concept_drift_three_seed_FULL_recovered.ipynb
```

2. Update the dataset paths according to your local machine.
3. Run all cells.
4. The result files will be generated in the output folder defined in the notebook (see `results/concept_drift/`).

The earlier `concept_drift_iot_ids_notebook_corrected.ipynb` is a single-run draft kept
for history; it does not contain the seed loop or drift-triggered retraining logic and
does not reproduce the paper's tables.

### Multiclass attack-category extension

1. Run `src/multiclass_attack_classification.py` (or the equivalent
   `notebooks/IoTIDS_balanced_multiclass_extension.ipynb`) after the binary preprocessing
   pipeline has produced `data/processed/final_sample_200000.csv`.
2. Outputs (per-class metrics, confusion matrices, and the paper-ready comparison table)
   are written under `outputs/multiclass/`.

See `docs/CODE_AVAILABILITY.md` for the full mapping between manuscript tables and the
scripts/notebooks that produced them.

## Citation

A manuscript based on this repository is under preparation/submission. Citation details will be added after acceptance/publication.
