# Chronological (file-order) vs. random split

**Paper reference:** Table `tab:chrono`

## How to run

```bash
python3 chrono_experiment.py
```

## Notes

Tests whether capture-file order carries an exploitable temporal signal over the full corpus -- the result behind the paper's claim that the covariate-shift experiments should not be described as naturally occurring chronological drift without this check.
