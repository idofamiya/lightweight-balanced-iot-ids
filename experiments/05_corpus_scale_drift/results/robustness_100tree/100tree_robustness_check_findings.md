# Corpus-scale robustness check: 50-tree weighted vs. 100-tree unweighted RF

**Why this was run.** Both peer reviewers independently flagged a real contradiction in the manuscript: Section 6.4 states the corpus-scale experiments (Tables 7/8) use "the same 50-tree class-weight-balanced forests" as the reduced-scale experiment, while a sentence in Section 6.9 claimed "it is the reason the corpus-scale experiments use the configuration they do" right after recommending the more stable 100-tree unweighted configuration — implying the corpus-scale tables had switched configs. They had not; the sentence was simply wrong. Per the author's decision, rather than just fixing the sentence, the corpus-scale experiments were actually rerun under the 100-tree unweighted configuration to check whether the paper's headline "silent failure" finding on unseen attack families is a genuine property of the task, or an artifact of the less stable 50-tree weighted forest.

**Method.** Regenerated the exact same quota-sampled pool from the full 34,298,828-flow / 63-file corpus (seed 42, same per-family quotas as `build_sample_fulldata.py` — confirmed byte-for-byte reproducible: achieved quota fill matches the original run to the row). Reran `drift_experiment_fulldata.py` for both regimes (`family` attack-holdout and `affine` covariate-shift), all three seeds (42/43/44), with the only change being the random-forest constructor: `n_estimators=100, max_depth=20` (no `class_weight`) in place of `n_estimators=50, class_weight="balanced"` — exactly the configuration used in Tables tab:rf-feature/tab:classifiers and in the paper's own tab:fifteen robustness experiment. Everything else (sample pool, phase construction, feature scaling, ADWIN settings, retrain windows) is untouched.

**Result — the central finding is robust.**

| Model | Config | Family holdout, Phase-3 R_novel | Family holdout, Phase-3 FPR | Covariate shift, Phase-3 F1 | Covariate shift, Phase-3 FPR |
|---|---|---|---|---|---|
| Static RF Top-15 | 50-tree weighted (published) | 0.0132 | 0.0010 | 0.7082 | 0.8373 |
| Static RF Top-15 | 100-tree unweighted (new) | 0.0097 ± 0.0029 | 0.0005 ± 0.0001 | 0.6674 ± 0.0008 | 0.9968 ± 0.0036 |
| Static RF Full-39 | 50-tree weighted (published) | 0.0134 | 0.0010 | 0.6716 | 0.9774 |
| Static RF Full-39 | 100-tree unweighted (new) | 0.0130 ± 0.0026 | 0.0004 ± 0.0001 | 0.6918 ± 0.0420 | 0.8953 ± 0.1742 |
| Drift-Triggered RF Top-15 | 50-tree weighted (published) | 0.5220 | 0.0742 | 0.9848 | 0.0047 |
| Drift-Triggered RF Top-15 | 100-tree unweighted (new) | 0.3323 ± 0.0109 | 0.0259 ± 0.0026 | 0.9780 ± 0.0063 | 0.0186 ± 0.0134 |

The static detectors still miss essentially all withheld-family attacks (~1% recall) under the more robust configuration — the "silent failure" is not an artifact of the unstable 50-tree weighted forest. The covariate-shift failure mode (false-alarm storm rather than missed recall) also reproduces.

**What *is* configuration-sensitive.** The size of drift-triggered retraining's recovery on the family-holdout arm: 0.5220 (published, 50-tree weighted) vs. 0.3323 ± 0.0109 (100-tree unweighted) — roughly 40-fold vs. roughly 25–34-fold over the static baseline. Both are still large, substantial recoveries; the exact multiplier is not invariant to forest configuration, unlike the magnitude of the static failure itself. The manuscript now reports this honestly rather than treating "forty-fold" as a fixed property.

**Manuscript changes made (this session):**
1. Section 6.9 (`sec:robustness`): removed the false claim that corpus-scale experiments "use the configuration they [100-tree unweighted] do"; replaced with an accurate statement that corpus-scale retains the 50-tree weighted config, followed by a new paragraph reporting this robustness check and a new table (`tab:configrobust`) with the numbers above.
2. Conclusion: softened "forty-fold in withheld-family recall" to "roughly one to two orders of magnitude (25- to forty-fold, depending on the static forest's configuration, Section 6.9)".

**Files:** raw per-seed results in `family_overall_100tree.csv`, `family_phasewise_100tree.csv`, `affine_overall_100tree.csv`, `affine_phasewise_100tree.csv` (this folder). Updated manuscript committed to `papers/L-CDAIDS.tex`.
