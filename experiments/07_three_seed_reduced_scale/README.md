# Reduced-scale three-seed concept-drift evaluation

**Paper reference:** Tables `tab:overall`, `tab:phasewise`, `tab:disentangle`-adjacent, `tab:fifteen`, `tab:events`; the static/drift-triggered rows of `tab:cost`

## How to run

```bash
jupyter nbconvert --to notebook --execute concept_drift_three_seed_FULL_recovered.ipynb
```

## Notes

`concept_drift_three_seed_FULL_recovered.ipynb` is the recovered original notebook behind the paper's seed-42/43/44 three-seed tables (`tab:overall`, `tab:phasewise`, `tab:events`) and the static/drift-triggered timing rows of `tab:cost`. It uses one fixed 80:20 split (`random_state=42`) and varies only the 50-tree, `class_weight="balanced"` Random Forest's own seed -- running it reproduces the published seed-44 Phase-3 collapse (F1 0.734002, FPR 71.97%) exactly. `results/` holds that run's raw and summary CSVs. `three_seed_concept_drift.py` is a separate, independent reconstruction that instead draws a *fresh* stratified split per seed with a 100-tree unweighted forest; across 15 seeds (42-56) it finds no comparable instability -- see `results/reproduced_15seed_robustness_check/` and `docs/REPRODUCTION_HISTORY.md` for the full investigation that distinguishes these two protocols. That 15-seed run is Table `tab:fifteen`.
