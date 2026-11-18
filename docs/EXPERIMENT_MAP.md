# Experiment Map: Paper Table/Figure → Code → Results

This is the canonical mapping between every table and figure in the
manuscript (`paper/L-CDAIDS.tex`) and the script, notebook, or results
folder that produced it. It mirrors the manuscript's own Section 6.14
("Consolidated Experimental Settings", Table `tab:settings`), but is kept
here as the maintained source of truth since it is easier to update than a
typeset table, and it indexes by LaTeX `\label{}` rather than by table
number, which shifts whenever a table is added or removed.

Each experiment folder under `experiments/` has its own `README.md` with
the exact run command and any protocol caveats; this file is the index.

| Label(s) | What it is | Folder |
|---|---|---|
| `tab:related` | Qualitative literature comparison | not code-generated |
| `tab:dataset`, `fig:workflow` | Balanced 200k-row binary dataset summary / preprocessing diagram | `experiments/01_preprocessing/` |
| `tab:corpus` | Composition of the full 34.3M-flow cleaned corpus (63 files) | `experiments/05_corpus_scale_drift/` (`results/per_file_family_counts.csv`) |
| `tab:quota` | Per-family sampling-pool quota / achieved / corpus-total | `experiments/05_corpus_scale_drift/recompute_quota_table.py` |
| `tab:phasequota` | Per-phase stream sample counts for the family-holdout stream | `experiments/05_corpus_scale_drift/` (`drift_experiment_fulldata.py --stream family`) |
| `tab:rf-feature`, `tab:classifiers` | Static Full-39 vs Top-15 Random Forest baseline | `experiments/02_static_baseline/` |
| `tab:cv` | 5-fold stratified cross-validation | `experiments/03_cross_validation/` |
| `tab:fullaffine`, `fig:corpusaffine` | Corpus-scale covariate-shift (affine) regime | `experiments/05_corpus_scale_drift/` (`--stream affine`) |
| `tab:family`, `fig:family` | Corpus-scale attack-family holdout (real concept drift) | `experiments/05_corpus_scale_drift/` (`--stream family`) |
| `tab:sweep` | Mixing-ratio sensitivity sweep | `experiments/05_corpus_scale_drift/` (`--novel-frac`) |
| `tab:configrobust` | 100-tree robustness check of the corpus-scale drift results | `experiments/05_corpus_scale_drift/results/robustness_100tree/` |
| `tab:chrono` | Chronological (file-order) vs. random split | `experiments/06_chronological_split/` |
| `tab:overall`, `tab:phasewise`, `tab:events` | Reduced-scale (120k-sample) three-seed concept-drift evaluation | `experiments/07_three_seed_reduced_scale/` (recovered notebook) |
| `tab:fifteen` | Fifteen-seed replication with independent stratified splits per seed | `experiments/07_three_seed_reduced_scale/results/reproduced_15seed_robustness_check/` (`three_seed_concept_drift.py`) |
| `tab:disentangle` | Disentangling forest-configuration vs. split sensitivity | `experiments/08_disentangle_seed_vs_split/` |
| `tab:cost` | Computational cost on the 120k-sample stream (static / drift-triggered / ARF rows) | `experiments/07_three_seed_reduced_scale/` (static, drift-triggered) + `experiments/09_arf_baseline/` (ARF) |
| `tab:picost`, `tab:pi15` | Raspberry Pi 4 hardware validation of `tab:cost` and `tab:fifteen` | `experiments/10_pi_hardware_validation/` |
| Section 6.8 paired per-seed ARF deltas | Seed-by-seed F1/MCC/FPR deltas, drift-triggered RF vs. ARF | `experiments/09_arf_baseline/results/arf_phasewise_raw.csv` paired against `experiments/07_three_seed_reduced_scale/results/phasewise_three_seed_results_raw.csv` |
| `tab:multiclass` | Extended multiclass attack-category classification | `experiments/04_multiclass/` |
| `tab:settings` | This table itself (Section 6.14) | mirrors this file |

## A caveat worth repeating

`experiments/05_corpus_scale_drift/`, `experiments/06_chronological_split/`,
`experiments/07_three_seed_reduced_scale/` and `experiments/08_disentangle_seed_vs_split/`
cover **three different experimental protocols** that are easy to confuse
because they all produce superficially similar "three-seed" or "phase-wise"
CSVs:

1. **One fixed 80:20 split, 50-tree `class_weight="balanced"` forest, only the
   forest's own seed varies.** This is the recovered original notebook
   (experiment 07's `concept_drift_three_seed_FULL_recovered.ipynb`) and
   `experiments/09_arf_baseline/`. It is the protocol behind `tab:overall`,
   `tab:phasewise`, `tab:events`, `tab:cost`, and the Section 6.8 paired ARF
   deltas.
2. **A fresh stratified split per seed, 100-tree unweighted forest.** This is
   `experiments/07_three_seed_reduced_scale/three_seed_concept_drift.py`
   (15 seeds, results in `reproduced_15seed_robustness_check/`) -- the
   protocol behind `tab:fifteen`.
3. **A fresh stratified split per seed, 50-tree weighted forest** (varying
   both the split and the forest's seed together). This is
   `experiments/08_disentangle_seed_vs_split/` -- the protocol behind
   `tab:disentangle`.

Pairing a result from one protocol against a result from another (for
example, pairing the ARF baseline against `reproduced_15seed_robustness_check/`
instead of against experiment 07's top-level `results/`) silently produces
numbers that do not correspond to any table in the paper. See
`docs/REPRODUCTION_HISTORY.md` for how this distinction was discovered.
