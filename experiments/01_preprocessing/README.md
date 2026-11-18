# Preprocessing pipeline

**Paper reference:** Table `tab:dataset`; Figure `fig:workflow`

## How to run

```bash
python3 iot_ids_preprocessing_pipeline.py
```

## Notes

Cleans the raw CICIoT2023 capture CSVs (`data/raw/`) into the balanced 200k-row binary dataset used by the static baseline and multiclass experiments. `results/dataset_preprocessing_report.txt` is the report from the run behind the paper's numbers.
