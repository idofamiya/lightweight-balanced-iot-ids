# Static Random Forest baseline (Full-39 vs Top-15)

**Paper reference:** Tables `tab:rf-feature`, `tab:classifiers`

## How to run

```bash
python3 classifier_comparison_fixed.py
```

## Notes

Trains the 100-tree, unweighted, fixed-split (`random_state=42`) Random Forest on the full 39-feature set and on the Top-15 subset, and compares against the other classifiers in Table `tab:classifiers`. `../../src/classifier_comparison.py` is an earlier, superseded version kept for history; this `_fixed` copy is the one behind the published numbers.
