# Data Directory

The CICIoT2023 dataset is **not** redistributed in this repository because
of its size (raw release is tens of GB) and the dataset owner's terms.

## 1. Download

Get the dataset from the official CIC/UNB page:

<https://www.unb.ca/cic/datasets/iotdataset-2023.html>

Place the 63 merged capture CSVs in:

```text
data/raw/
```

## 2. Preprocess

Run `experiments/01_preprocessing/iot_ids_preprocessing_pipeline.py` to
produce the balanced 200k-row binary dataset used by the static baseline
(`experiments/02_static_baseline/`) and multiclass
(`experiments/04_multiclass/`) experiments. Its output lands in:

```text
data/processed/final_sample_200000.csv
```

## 3. Full corpus (for the corpus-scale experiments)

The corpus-scale experiments (`experiments/05_corpus_scale_drift/`,
`experiments/06_chronological_split/`) run over the *complete* cleaned
corpus -- all 34.3M flows across 63 files, each saved as a parquet file for
speed -- expected at:

```text
data/processed/full_corpus/cleaned_Merged01.parquet
data/processed/full_corpus/cleaned_Merged02.parquet
...
data/processed/full_corpus/cleaned_Merged63.parquet
```

Produce these by running the cleaning step of the preprocessing pipeline
over each raw file individually (same cleaning logic as step 2, saved as
parquet instead of being pooled into one sample) and placing the results
here.

## 4. Sampled pool (built by the pipeline itself)

`experiments/05_corpus_scale_drift/build_sample_fulldata.py` draws a
675,000-row, per-family-quota sample from the full corpus (step 3) and
writes it to `data/sampled/sample_familyholdout.parquet`. This is a
generated artifact, not something to download -- running the script
creates it. `experiments/05_corpus_scale_drift/README.md` has the exact
command, and `experiments/05_corpus_scale_drift/recompute_quota_table.py`
reproduces the quota table (`tab:quota`) deterministically without needing
to re-run the sampler.

## Why none of this is checked in

`data/raw/` and `data/processed/` are listed in `.gitignore`. Even the
200k-row balanced sample is ~40MB uncompressed and the full corpus is
multiple gigabytes; none of it belongs in a git repository. Every number
in the paper that depends on this data is reproducible from the scripts in
`experiments/` once the data above is in place -- see
`docs/EXPERIMENT_MAP.md`.
