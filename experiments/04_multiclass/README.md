# Extended multiclass attack-category classification

**Paper reference:** Table `tab:multiclass`

## How to run

```bash
python3 multiclass_paper_sampling.py   # published numbers
python3 multiclass_attack_classification_fixed.py   # base/original sampling
```

## Notes

Two sampling strategies were tried for the multiclass extension. `multiclass_paper_sampling.py` is the "paper strategy" variant whose output (`results/paper_sampling_variant/`) matches the manuscript's Table `tab:multiclass`. `multiclass_attack_classification_fixed.py` is a corrected copy of `../../src/multiclass_attack_classification.py` (the original hardcoded the binary-vs-multiclass comparison numbers instead of reading them from the static baseline's own output; the `_fixed` version reads `../02_static_baseline/results/paper_ready_classifier_comparison.csv`).
