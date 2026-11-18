# Code and Data Availability

The code for the lightweight and concept-drift-aware IoT IDS experiments
(L-CDAIDS) is available in this repository.

## Code

Three layers of code are provided.

**`notebooks/` and `src/`** cover the original 200k-sample binary and
multiclass experiments. Both are superseded by corrected copies -- see
`src/NOTE.md` and `notebooks/NOTE.md`.

**`experiments/`** contains one folder per result reported in the paper --
the static baseline, cross-validation, corpus-scale drift, chronological
split, the three-seed reduced-scale evaluation, the seed-vs-split
disentangling check, the Adaptive Random Forest baseline, the multiclass
extension, and the Raspberry Pi hardware validation. Each folder has its
own `README.md` with the exact run command. **`docs/EXPERIMENT_MAP.md` is
the authoritative, maintained mapping from every table/figure label in the
manuscript to the folder and script that produced it** -- start there.

Install requirements: `pip install -r requirements.txt`.

## Data

The CICIoT2023 dataset is not included in this repository because of
dataset size and external ownership. See `data/README.md` for how to
obtain it and the exact layout each experiment expects.

## Reproducibility

Every script resolves paths relative to the repository root
(`Path(__file__).resolve().parents[N]`), so the repository can be cloned
and run from anywhere without editing paths, once the data described in
`data/README.md` is in place. Trained-model binaries (`.pkl`, `.joblib`)
and large sampled-data CSVs are not checked in (see `.gitignore`) since
they are regenerable from the scripts; the result CSVs, JSON summaries,
and figures that the paper's tables are built from are checked in under
each experiment's `results/` subfolder.

`docs/REPRODUCTION_HISTORY.md` is kept as a historical record of how the
three-seed concept-drift results were investigated, including the
discovery of the original notebook and the distinction between the three
different experimental protocols documented in `docs/EXPERIMENT_MAP.md`.
It predates, and is superseded as a map by, `docs/EXPERIMENT_MAP.md`.
