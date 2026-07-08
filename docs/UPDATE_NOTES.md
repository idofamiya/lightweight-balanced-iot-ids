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
