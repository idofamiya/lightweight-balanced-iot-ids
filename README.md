# L-CDAIDS: Lightweight Concept-Drift-Aware Intrusion Detection for Resource-Constrained IoT Networks

Reproducible code and experimental artefacts for a lightweight,
concept-drift-aware intrusion detection framework for resource-constrained
IoT networks, evaluated on CICIoT2023. The framework combines:

- balanced binary IDS evaluation,
- Top-15 flow/statistical feature reduction,
- Random Forest based lightweight detection,
- ADWIN-based concept-drift monitoring,
- drift-triggered Random Forest retraining using a recent traffic window,
- phase-wise evaluation under a controlled, evolving IoT traffic stream,
- an extended multiclass attack-category classification study,
- corpus-scale validation (5-fold cross-validation, attack-family holdout,
  chronological-split, 100-tree robustness check) over the full
  34.3M-flow CICIoT2023 release,
- a comparison against an Adaptive Random Forest (river) baseline, and
- hardware validation on a Raspberry Pi 4.

## Start here

**`docs/EXPERIMENT_MAP.md`** is the authoritative mapping from every table
and figure in the manuscript (`paper/L-CDAIDS.tex`) to the folder, script,
or notebook that produced it. If you are trying to reproduce a specific
result, start there.

## Repository structure

```text
paper/                   LaTeX source of the manuscript, bibliography, figures
experiments/             one folder per paper result -- see docs/EXPERIMENT_MAP.md
  01_preprocessing/       raw CICIoT2023 -> balanced 200k binary dataset
  02_static_baseline/     static Full-39 vs Top-15 Random Forest
  03_cross_validation/    5-fold stratified cross-validation
  04_multiclass/          extended multiclass attack-category classification
  05_corpus_scale_drift/  corpus-scale covariate-shift + attack-family holdout
  06_chronological_split/ chronological (file-order) vs. random split
  07_three_seed_reduced_scale/  three-seed phase-wise concept-drift evaluation
  08_disentangle_seed_vs_split/ forest-config vs. split sensitivity
  09_arf_baseline/        Adaptive Random Forest (river) comparison
  10_pi_hardware_validation/    Raspberry Pi 4 timing and model-size measurements
notebooks/                legacy notebooks (see notebooks/NOTE.md)
src/                       legacy scripts (see src/NOTE.md)
data/                      not checked in -- see data/README.md
docs/
  EXPERIMENT_MAP.md         canonical table/figure -> code mapping (start here)
  CODE_AVAILABILITY.md      code/data availability statement
  REPRODUCTION_HISTORY.md   how the three-seed result protocols were investigated
  UPDATE_NOTES.md           dated changelog of updates to this repository
  FULLDATA_FINDINGS.md      findings notes from the corpus-scale drift experiments
```

## Dataset

The experiments use CICIoT2023, which is not redistributed in this
repository. See `data/README.md` for how to obtain it and the exact
layout each experiment expects.

## Requirements

```bash
pip install -r requirements.txt
```

## How to run

1. Obtain and preprocess the dataset as described in `data/README.md`.
2. Pick the experiment you want from `docs/EXPERIMENT_MAP.md`.
3. `cd experiments/<folder>` and follow that folder's `README.md`.

Every script resolves paths relative to the repository root, so it can be
run directly from a clone (`python3 experiments/<folder>/<script>.py`)
once the data is in place.

## Main result summary

The drift-triggered RF Top-15 model achieves a substantial, but only
partial, recovery of detection performance on attack families withheld
from training and introduced mid-stream (genuine concept drift), using
only 15 features and no label access beyond the ADWIN error signal. See
the paper (`paper/L-CDAIDS.tex`) for the full results, including a
Raspberry Pi 4 hardware validation and a comparison against an Adaptive
Random Forest baseline.

## Citation

A manuscript based on this repository is under review. Citation details
will be added after acceptance/publication.
