# Corpus-scale covariate-shift and attack-family-holdout drift

**Paper reference:** Tables `tab:corpus`, `tab:quota`, `tab:phasequota`, `tab:fullaffine`, `tab:family`, `tab:sweep`, `tab:configrobust`

## How to run

```bash
python3 build_sample_fulldata.py                       # build the 675k-flow sampling pool
python3 drift_experiment_fulldata.py --stream affine --seed 42   # Table tab:fullaffine
python3 drift_experiment_fulldata.py --stream family --seed 42   # Table tab:family
python3 drift_experiment_fulldata.py --aggregate                 # aggregate seeds 42-44
python3 recompute_quota_table.py                        # Table tab:quota, no re-sampling needed
```

## Notes

Runs over the full 34.3M-flow cleaned corpus (`data/processed/full_corpus/`, 63 parquet files -- see `data/README.md`). `tab:configrobust` (the 100-tree robustness check) is the same `drift_experiment_fulldata.py` with the Random Forest constructor swapped from 50 trees to 100; its results are under `results/robustness_100tree/`. `tab:sweep` is the same script's `--novel-frac` mixing-ratio sweep. `recompute_quota_table.py` reproduces Table `tab:quota`'s Achieved column deterministically from `results/per_file_family_counts.csv` -- the original `sample_meta.json` that `build_sample_fulldata.py` writes was not preserved from the run behind the paper, so this script recovers the same numbers from first principles instead of re-drawing a fresh (seed-dependent) sample.
