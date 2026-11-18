# Repository Update Notes

Update date: 2026-07-08

This update adds concept-drift-aware IDS experiments for the repository `lightweight-balanced-iot-ids`.

## Added

- ADWIN-based concept drift monitoring
- Drift-triggered Random Forest retraining
- Full-39 vs Top-15 baseline comparison
- Three-seed evaluation
- Phase-wise F1 and MCC analysis
- Figures for drift monitoring and phase-wise performance

## Files included in this update package

- notebooks/concept_drift_iot_ids_notebook_corrected.ipynb
- results/concept_drift/concept_drift_model_comparison.csv
- results/concept_drift/concept_drift_model_comparison_with_rf_retraining.csv
- results/concept_drift/final_comparison_full39_top15_drift_rf.csv
- results/concept_drift/phasewise_three_seed_results_summary.csv
- results/concept_drift/phasewise_three_seed_results_paper_table.csv
- results/concept_drift/phasewise_three_seed_events.csv
- results/concept_drift/three_seed_drift_experiment_summary.csv
- results/concept_drift/three_seed_drift_experiment_raw.csv
- results/concept_drift/three_seed_drift_retraining_events.csv
- figures/phasewise_f1_comparison.png
- figures/phasewise_mcc_comparison.png
- figures/rf_top15_drift_monitoring.png
- figures/preprocessing_workflow.png

## Suggested Commit Message

```text
Add concept-drift-aware IDS experiments with ADWIN monitoring
```

## Suggested Git Commands

```bash
git status
git checkout -b concept-drift-extension
cp -r <contents-of-this-update-folder>/* .
git add notebooks/ results/ figures/ docs/ README.md .gitignore
git commit -m "Add concept-drift-aware IDS experiments with ADWIN monitoring"
git push origin concept-drift-extension
```

After checking the branch on GitHub, merge it into `main` or create a pull request.

---

## Update: 2026-11-18

This update brings the repository in line with the manuscript after a full
pass against two peer reviews (tracked separately). Summary of what
changed:

### Corrections to the manuscript itself (not this repo's code, but why
some result files below were re-derived)
- Fixed a Section 4.3.2 description mismatch with the actual 5-attack-family
  holdout design the code implements.
- Corrected an overstated standard-deviation comparison and an overstated
  "all gaps are negligible" claim that had folded in a real FPR gap.
- Abstract/conclusion now explicitly say "partial recovery" with the
  residual miss rate stated, rather than implying full recovery.

### New experiments and results added
- **Raspberry Pi 4 hardware validation** (`experiments/10_pi_hardware_validation/`):
  on-device timing and model-size measurements for both the `tab:cost` and
  `tab:fifteen` configurations, not previously in this repository at all.
- **100-tree robustness check** (`experiments/05_corpus_scale_drift/results/robustness_100tree/`):
  re-runs the corpus-scale drift experiments with the published 100-tree
  configuration to check the 50-tree results aren't an artifact of forest size.
- **Per-family sampling-pool quota table** (`tab:quota`, `tab:phasequota`):
  recomputed deterministically from `per_file_family_counts.csv` via the new
  `experiments/05_corpus_scale_drift/recompute_quota_table.py`, after
  discovering the original `sample_meta.json` that would have recorded this
  directly was never preserved.
- **Paired per-seed ARF deltas** (Section 6.8): seed-by-seed F1/MCC/FPR
  deltas between the drift-triggered RF and the ARF baseline, using
  `experiments/09_arf_baseline/results/arf_phasewise_raw.csv` paired against
  `experiments/07_three_seed_reduced_scale/results/phasewise_three_seed_results_raw.csv`.
- **Consolidated experimental settings table** (`tab:settings`, Section 6.14):
  the paper now carries its own script-to-table mapping; this repository's
  `docs/EXPERIMENT_MAP.md` is the maintained version of the same table.

### Reorganization
- Everything under the old flat `scripts/` and `outputs/` folders moved into
  `experiments/<NN>_<name>/`, one folder per paper result, each with its own
  script(s), `results/`, and `README.md`. See `docs/EXPERIMENT_MAP.md`.
- Added `paper/` with the manuscript's LaTeX source, bibliography, and figures.
- Fixed three scripts (`build_sample_fulldata.py`, `drift_experiment_fulldata.py`,
  `chrono_experiment.py`) that had hardcoded absolute paths left over from
  the machine they were originally run on; they now resolve paths relative
  to the repository root like the rest of the scripts.
- Added the previously-missing `iot_ids_preprocessing_pipeline.py` (now
  `experiments/01_preprocessing/`) and `multiclass_paper_sampling.py` (now
  `experiments/04_multiclass/`), both of which produced result files already
  in this repository but had never themselves been checked in.
