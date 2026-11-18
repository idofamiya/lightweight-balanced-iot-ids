# Changes made to this project (2026-08-29 / updated 2026-08-31)

## IMPORTANT UPDATE (2026-08-31, later same day): the original notebook was found

Everything below this note was written *before* the actual original notebook
behind Tables 5/6/8 turned up. The author supplied it directly and it is now
at `notebooks/concept_drift_three_seed_FULL_recovered.ipynb`. Running its
logic reproduces the paper's seed-44 result (accuracy 0.638675, F1 0.734002,
FPR 71.97%) and the Table 5 summary (0.939222 ± 0.062009) to six decimal
places. **The seed-44 result is genuine, not an error** — the "unverifiable"
conclusion below is retracted.

What the recovered notebook also shows: it loads one fixed 80:20 train/test
split *once*, then the "three seeds" only vary the `random_state` of a
freshly-trained 50-tree, `class_weight="balanced"` Random Forest on that same
fixed split — a different, smaller, more class-weight-sensitive model than
the 100-tree unweighted forest behind Tables 3/4. That specific configuration
is unstable enough that one of its three random initializations (seed 44)
produces the reported collapse. The `three_seed_concept_drift.py` script
below tested something different — a 100-tree unweighted forest re-trained
on a *fresh* split for each seed — and found no such instability across 15
seeds. Put together: the instability looks tied to the small/balanced-weight
forest configuration and/or the single fixed split, not to "IoT traffic
drift breaks Top-15 Random Forest IDS" in general. See the critical review
(Section 3) in the project for the full breakdown and the recommended
disentangling experiment.

`scripts/three_seed_concept_drift.py` and its 15-seed results below are
still useful — they're a genuine, independent robustness check across data
splits, which the paper doesn't currently report — but they are no longer a
"replacement" for Tables 5/6/8. They're answering an additional, different
question.

---

## Original notes (written before the notebook was recovered)

This folder previously contained no scripts or notebooks (only data/ and
outputs/), even though outputs/concept_drift/ and outputs/multiclass/
contain results that must have been produced by code that ran somewhere.
Neither this machine nor the public GitHub repository
(https://github.com/idofamiya/lightweight-balanced-iot-ids) has that code.
See docs/CRITICAL_REVIEW.md (in the project) for the full writeup. Two
things were added here to close part of that gap:

## 1. three_seed_concept_drift.py

A from-scratch, documented reconstruction of the three-seed / phase-wise /
ADWIN drift-triggered Random Forest retraining experiment that the paper
reports in Tables 5, 6, 8 and Figures 2-4. It was built from the paper's
stated methodology (Sections 4.3-4.5) plus the parameters recoverable from
the one experiment notebook that IS public
(concept_drift_iot_ids_notebook_corrected.ipynb, which only covers a single
seed and no retraining).

Usage (run seed-by-seed, since each seed takes ~55-75s and this machine's
shell sessions time out around 3 minutes):

```
python3 scripts/three_seed_concept_drift.py --seed 42
python3 scripts/three_seed_concept_drift.py --seed 43
...
python3 scripts/three_seed_concept_drift.py --seed 56
python3 scripts/three_seed_concept_drift.py --aggregate
```

Results land in outputs/concept_drift/reproduced/, in the same file/column
layout as the paper's tables, so they can be diffed directly against
outputs/concept_drift/*.csv (the numbers currently cited in the paper).

### Update 2026-08-31: scaled from 3 seeds to 15 seeds (42-56)

The original 3-seed reproduction (below) already didn't show the paper's
seed-44 collapse. To rule out "3 seeds just wasn't enough to catch it" and
to give the paper a statistically defensible replacement for Tables 5/6/8
and Figures 2-4, the same script was run for seeds 42 through 56 (15 total,
including the paper's original 3) on this machine's real data.

**Headline result: no seed, out of 15, comes anywhere close to the paper's
reported collapse.** Static RF Top-15 overall accuracy across all 15 seeds
ranges from 0.9751 (seed 53, the *worst* of the 15) to 0.9860 (seed 46) —
mean 0.9829, std 0.0029. Seed 44 specifically — the seed the paper reports
collapsing to 63.9% accuracy / 72% FPR — scores 0.9852 in this
reconstruction, one of the *better* seeds, not the worst.

Overall (all 15 seeds, mean ± std):

| Model | Accuracy | F1 | MCC | FPR |
|---|---|---|---|---|
| Static RF Top-15 | 0.9829 ± 0.0029 | 0.9828 ± 0.0028 | 0.9658 ± 0.0058 | 1.30% ± 0.56% |
| Static RF Full-39 | 0.9876 ± 0.0007 | 0.9875 ± 0.0007 | 0.9754 ± 0.0015 | 0.31% ± 0.10% |
| ADWIN-Monitored RF Top-15 (no retrain) | 0.9829 ± 0.0029 | 0.9828 ± 0.0028 | 0.9658 ± 0.0058 | 1.30% ± 0.56% |
| Drift-Triggered RF Top-15 (retrains on drift) | 0.9871 ± 0.0007 | 0.9869 ± 0.0007 | 0.9744 ± 0.0013 | 0.27% ± 0.08% |

(ADWIN-Monitored and Static Top-15 are identical here by construction —
monitoring alone doesn't change predictions, only retraining does — which
is a useful internal consistency check on the reconstruction.)

Phase 3 (the harshest shift, and the phase the paper's headline number
comes from), mean ± std across all 15 seeds:

| Model | Phase-3 Accuracy | Phase-3 F1 | Phase-3 MCC | Phase-3 FPR |
|---|---|---|---|---|
| Static RF Top-15 | 0.9729 ± 0.0086 | 0.9731 ± 0.0082 | 0.9460 ± 0.0167 | 3.41% ± 1.79% |
| Static RF Full-39 | 0.9854 ± 0.0015 | 0.9853 ± 0.0015 | 0.9709 ± 0.0030 | 0.75% ± 0.29% |
| Drift-Triggered RF Top-15 | 0.9852 ± 0.0010 | 0.9851 ± 0.0011 | 0.9707 ± 0.0021 | 0.39% ± 0.17% |

Drift-triggered retraining gives a real, consistent Phase-3 improvement
over static Top-15 (F1 +1.2 points: 0.9731 -> 0.9851, with the standard
deviation shrinking by ~7x, from 0.0082 to 0.0011 — i.e. retraining doesn't
just improve the average, it makes the model's worst-case behavior far more
predictable). That is a genuine, defensible finding. It is a much smaller
and steadier effect than the paper's claimed jump from 0.8809 to 0.9832
(~10 points), which was driven almost entirely by the single disputed
seed-44 data point.

Two new figures reproduce the paper's Figures 2-3 using all 15 seeds with
error bars (mean ± std): `phasewise_f1_comparison_15seed.png` and
`phasewise_mcc_comparison_15seed.png`, both in
outputs/concept_drift/reproduced/.

**What this does and doesn't prove:** this is still an independent
reconstruction, not a recovery of the original undocumented script, so it
cannot prove the original run was wrong. But going from 3 to 15 seeds — a
5x larger, still-real-data sample, with the paper's own seed 44 scoring
*above* the reconstructed average — makes it substantially harder to
explain the seed-44 collapse as ordinary seed-to-seed variance. The
practical recommendation is unchanged from the original review: either
recover/re-run the original code and confirm the anomaly, or replace
Tables 5/6/8 and Figures 2-4 with the (more modest but verifiable) 15-seed
numbers above.

Original 3-seed table (seeds 42/43/44 only, kept for reference):

| Seed | Model | Phase-3 Accuracy | Phase-3 F1 | Phase-3 FPR |
|---|---|---|---|---|
| 42 | Static RF Top-15 (reproduced) | 0.9755 | 0.9756 | 0.0287 |
| 43 | Static RF Top-15 (reproduced) | 0.9770 | 0.9770 | 0.0257 |
| 44 | Static RF Top-15 (reproduced) | 0.9791 | 0.9791 | 0.0204 |
| 42 | Drift-Triggered RF Top-15 (reproduced) | 0.9845 | 0.9844 | 0.0051 |
| 43 | Drift-Triggered RF Top-15 (reproduced) | 0.9858 | 0.9856 | 0.0035 |
| 44 | Drift-Triggered RF Top-15 (reproduced) | 0.9852 | 0.9850 | 0.0026 |

## 2. multiclass_attack_classification_fixed.py

A corrected copy of src/multiclass_attack_classification.py from the
GitHub repo. The original hardcodes a `binary_rf_top15_summary` dict with
fixed numbers (Accuracy 0.9885, F1 0.99, MCC 0.9772, ...) instead of
reading them from the binary experiment's own saved output
(outputs/paper_ready_classifier_comparison.csv). We confirmed this
locally: outputs/multiclass/binary_vs_multiclass_summary.csv contains
exactly those hardcoded numbers, unchanged, byte-for-byte, regardless of
what the binary experiment on this machine actually produced. The fixed
version loads that CSV and picks out the "Random Forest Top-15" row, so
the binary-vs-multiclass comparison table always reflects the binary run
that was actually done, and fails loudly (with a clear message) instead of
silently reporting stale numbers if that file is missing.

To use it: run src/classifier_comparison.py (or your existing binary
pipeline) first so outputs/paper_ready_classifier_comparison.csv exists,
then run this fixed script in place of the original.
