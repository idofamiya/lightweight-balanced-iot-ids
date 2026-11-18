# 5-fold stratified cross-validation

**Paper reference:** Table `tab:cv`

## How to run

```bash
python3 cross_validation_tables_3_4.py
```

## Notes

Same 100-tree RF configuration as the static baseline, evaluated with 5-fold stratified cross-validation. Each fold reuses one global Top-15 feature list rather than re-ranking features per fold -- see `docs/EXPERIMENT_MAP.md` for why that is called out as a punch-list item rather than silently fixed.
