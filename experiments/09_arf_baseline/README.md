# Adaptive Random Forest (river) baseline

**Paper reference:** ARF rows of Table `tab:cost`; the paired per-seed ARF-vs-drift-triggered deltas in Section 6.8

## How to run

```bash
python3 arf_baseline.py --seed 42
python3 arf_baseline.py --aggregate
```

## Notes

A `river` Adaptive Random Forest run prequentially over the same 120,000-sample stream as experiment 07, for the ARF comparison in `tab:cost` and the paired-by-seed F1/MCC/FPR deltas against the drift-triggered RF reported in Section 6.8. `results/arf_phasewise_raw.csv` and `../07_three_seed_reduced_scale/results/phasewise_three_seed_results_raw.csv` are the two files paired seed-by-seed for that comparison -- not `../07.../results/reproduced_15seed_robustness_check/`, which is a different protocol (see experiment 07's note).
