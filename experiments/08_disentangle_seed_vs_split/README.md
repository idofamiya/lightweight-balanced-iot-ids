# Disentangling model-configuration sensitivity from split sensitivity

**Paper reference:** Table `tab:disentangle`

## How to run

```bash
python3 disentangle_seed_vs_split.py
```

## Notes

The 50-tree weighted configuration with a *fresh* stratified split per seed (rather than one fixed split as in experiment 07) -- isolates whether the Phase-3 instability comes from the forest configuration, the fixed split, or both.
