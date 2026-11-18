# Legacy scripts

`classifier_comparison.py` and `multiclass_attack_classification.py` are
the original versions of the static-baseline and multiclass pipelines,
kept for history. Both have corrected, superseding copies used for the
manuscript's published numbers:

- `classifier_comparison.py` -> `../experiments/02_static_baseline/classifier_comparison_fixed.py`
- `multiclass_attack_classification.py` -> `../experiments/04_multiclass/multiclass_attack_classification_fixed.py`
  (the original hardcoded the binary-vs-multiclass comparison numbers
  instead of reading them from the static baseline's actual output --
  see that experiment's `README.md`)

See `../docs/EXPERIMENT_MAP.md` for the full table-to-script mapping.
