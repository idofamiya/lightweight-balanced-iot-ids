# Raspberry Pi 4 hardware validation

**Paper reference:** Tables `tab:picost`, `tab:pi15`

## How to run

```bash
See SETUP_INSTRUCTIONS.md -- these run on a physical Raspberry Pi 4, not on a development machine.
python3 pi_benchmark_tabcost.py   # Table tab:picost
python3 pi_benchmark_lcdaids.py   # Table tab:pi15
```

## Notes

Reproduces the exact Table `tab:cost` and 15-seed (`tab:fifteen`) configurations' timing and model-size measurements on a Raspberry Pi 4, to validate that the workstation-measured costs hold on the resource-constrained hardware the paper targets. Needs `data/processed/final_sample_200000.csv` (see experiment 01) copied onto the Pi; see `SETUP_INSTRUCTIONS.md`.
