# Full-Dataset Drift Experiments — L-CDAIDS

Run 2026-09-12 on the complete cleaned CICIoT2023 corpus. Complements the
critical review (`claude/L-CDAIDS_Critical_Review.md`) and the revised
manuscript (`G:\IoT_IDS_Research\papers\L-CDAIDS.tex`, v4).

> **Revision note.** An earlier version of this document reported a
> family-holdout experiment that trained only on DDoS/DoS/Mirai. That design was
> discarded: restricting training to the three most volumetric families made
> benign-vs-attack almost perfectly separable (Phase-1 F1 = 1.0000, **exactly
> zero** false positives in 20,000 benign flows, every seed), which is not a
> representative baseline and would invite a reviewer to suspect leakage. The
> numbers below come from the corrected design, which trains on a proportional
> mix of all five common families and holds out only the rare ones. Any figure
> quoting Phase-1 F1 of 1.0000 or recall of exactly 0.50 is superseded.

## 1. What the full corpus contains

All 63 cleaned Parquet files profiled: **34,298,828 flows**, 35 distinct labels.

| Category | Flows | Share |
|---|---:|---:|
| DDOS | 23,123,908 | 67.42% |
| DOS | 6,457,920 | 18.83% |
| MIRAI | 2,516,501 | 7.34% |
| BENIGN | 1,050,893 | 3.06% |
| SPOOFING | 452,694 | 1.32% |
| VULNERABILITY_SCAN | 357,446 | 1.04% |
| RECON | 303,139 | 0.88% |
| WEB_OR_MALWARE | 23,796 | 0.069% |
| BRUTEFORCE | 12,522 | 0.037% |
| (malformed `NAN` label) | 9 | — |

**The original 200k sample was statistically faithful, not biased.** Its attack
composition matches the corpus almost exactly (DDoS 69.52% vs 69.55%; Brute
Force 0.032% vs 0.038%). The defensible criticism is not skew but absolute
counts: proportional sampling at 200k leaves Brute Force with 32 records and
Web/Malware with 84. Benign is only 3.06% of the corpus, so "train on
everything" is not available without a weighting decision.

## 2. Design

Two drift regimes, both at 200k training scale drawn from all 34.3M flows,
120k stream in three 40k phases, 50:50 benign:attack, seeds 42/43/44, Top-15
features, 50-tree `class_weight="balanced"` forests.

- **Covariate shift (virtual drift)** — the published affine shift, applied to
  standardised features, with the paper's proportional class composition.
- **Attack-family holdout (real concept drift)** — trained on benign plus a
  proportional mix of DDoS, DoS, Mirai, Spoofing and Vulnerability-scan.
  Reconnaissance withheld and introduced in Phase 2; Web/Malware and Brute
  Force withheld and introduced in Phase 3.

Two diagnostic additions:
- **AdaptiveNorm baseline** — static forest, input scaler re-estimated online
  over a 20k rolling window, forest never retrained. Tests whether the drift is
  merely a scaling artefact.
- **Recall split by family origin** — recall reported separately on attack rows
  from seen vs withheld families, because overall recall in a drifted phase
  depends mechanically on the mixing ratio.

## 3. Family-holdout results (mean over 3 seeds)

| Model | P1 F1 | P2 F1 | P3 F1 | P2 R_known | P3 R_known | P2 R_novel | P3 R_novel | P2 FPR | P3 FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static RF Top-15 | 0.9925 | 0.7579 | 0.6660 | 0.9859 | 0.9863 | 0.2354 | **0.0132** | 0.0009 | 0.0010 |
| Static RF Full-39 | 0.9926 | 0.7952 | 0.6660 | 0.9857 | 0.9862 | 0.3357 | 0.0134 | 0.0009 | 0.0010 |
| Adaptive-Norm RF | 0.9683 | 0.6859 | 0.6712 | 0.9985 | 0.9998 | 0.9254 | 0.9905 | 0.8469 | 0.9704 |
| Adaptive Random Forest | 0.9881 | 0.8476 | 0.8354 | 0.9839 | 0.9858 | 0.5799 | 0.5827 | 0.0631 | 0.0934 |
| **Drift-Triggered RF Top-15** | **0.9925** | **0.7944** | **0.8254** | 0.9845 | 0.9881 | 0.3363 | **0.5220** | **0.0023** | **0.0742** |

### Three findings

**1. The failure is silent.** Known-family recall is unchanged across all phases
(0.9845–0.9881) and the static detectors' false-positive rate stays at one in a
thousand. By every signal an operator can observe, nothing has happened — yet
the Top-15 detector recovers only 1.3% of withheld web/malware and brute-force
traffic. This is the opposite of the covariate-shift failure mode, which
announces itself as a false-alarm storm.

**2. Feature reduction has a cost static accuracy conceals.** On reconnaissance,
Full-39 recovers 0.3357 of withheld attacks against 0.2354 for Top-15 — a
ten-point gap between two models that are statistically indistinguishable on
held-out samples of known attacks (98.86% vs 98.85%). The 24 discarded features
carry little information about seen attacks and appreciably more about unseen
ones. This qualifies the paper's own feature-reduction claim. On web/malware and
brute force the distinction vanishes: both are near-blind.

**3. Retraining recovers most of it.** Drift-triggered retraining lifts
withheld-family recall from 0.0132 to 0.5220 in Phase 3 — forty-fold — at
precision 0.9106. In Phase 2 its gain is smaller (0.3363) and essentially equals
what the full feature set achieves statically.

### Not an artefact of the mixing ratio

Withheld families set to 25% / 50% / 75% of each drifted phase (seed 42):

| Model | Phase | 25% | 50% | 75% |
|---|---|---:|---:|---:|
| Static Top-15 | P2 | 0.2524 | 0.2552 | 0.2561 |
| Static Top-15 | P3 | 0.0100 | 0.0120 | 0.0129 |
| Drift-Triggered | P2 | 0.3192 | 0.3262 | 0.3439 |
| Drift-Triggered | P3 | 0.4238 | 0.5672 | 0.6139 |

Withheld-family recall is essentially invariant; only aggregate F1 moves with
the ratio. The deficit is a property of the model, not the stream composition.

## 4. Covariate-shift results (corpus scale, mean over 3 seeds)

| Model | P1 F1 | P2 F1 | P3 F1 | P3 FPR |
|---|---:|---:|---:|---:|
| Static RF Top-15 | 0.9890 | 0.9686 | 0.7082 | 0.8373 |
| Static RF Full-39 | 0.9893 | 0.9860 | 0.6716 | 0.9774 |
| Adaptive-Norm RF | 0.9377 | 0.9466 | 0.7916 | 0.5134 |
| Adaptive Random Forest | 0.9839 | 0.9856 | 0.9843 | 0.0022 |
| **Drift-Triggered RF Top-15** | **0.9890** | **0.9860** | **0.9848** | **0.0047** |

The published degradation reproduces and is more severe at corpus scale. Note
the reversal: **Full-39 is worse than Top-15 in Phase 3** here (0.6716 vs
0.7082), opposite to the reduced-scale result. The reduced-scale numbers differ
(static Top-15 P3 F1 0.8809 vs 0.7082) because that experiment reuses one
held-out fold three times, whereas the corpus experiment draws fresh records per
phase — the corpus figures are the more conservative estimate.

## 5. The renormalisation objection, answered

Adaptive renormalisation does not substitute for retraining on either regime.
Under covariate shift it recovers part of the Phase-3 loss (0.7082 → 0.7916) but
leaves FPR at 0.5134. Under real concept drift it is actively harmful: recall on
withheld families rises to 0.99 but precision falls to 0.51, FPR reaches 0.97
and MCC collapses to 0.06. It also degrades the *undrifted* Phase 1 (0.9683 vs
0.9925). Rescaling does not teach a model a new attack family; it displaces the
boundary until almost everything is flagged.

## 6. Cost (family stream, mean over seeds)

| Model | Fit (s) | Stream 120k (s) |
|---|---:|---:|
| Static RF Top-15 | 12.1 | 0.51 |
| Drift-Triggered RF Top-15 | 12.1 | 5.26 |
| Adaptive Random Forest | 45.5 | 138.7 |

Drift-triggered retraining processes the stream 26.4× faster than ARF, with 2.33
detected drift points and 2.33 retraining events on average.

## 7. Chronological evaluation — a documented negative result

Run 2026-09-12. Two facts established first:

1. **No timestamp exists.** The 42-column CICIoT2023 feature schema has no time
   field. `Time_To_Live` is an IP hop counter. True chronological order is not
   recoverable from this release at all.
2. **The 63 files are homogeneous.** Per-file class shares across all 63 files:

   | Family | min % | max % | spread (pts) | CV % | Spearman ρ vs file index |
   |---|---:|---:|---:|---:|---:|
   | DDOS | 66.860 | 71.677 | 4.817 | 0.99 | +0.220 |
   | DOS | 17.320 | 18.990 | 1.670 | 1.19 | −0.202 |
   | MIRAI | 5.868 | 7.634 | 1.766 | 3.41 | −0.177 |
   | BENIGN | 2.356 | 3.191 | 0.835 | 3.64 | −0.250 |
   | SPOOFING | 1.139 | 1.374 | 0.236 | 2.92 | −0.266 |
   | VULNERABILITY_SCAN | 0.876 | 1.088 | 0.212 | 3.26 | −0.134 |
   | RECON | 0.694 | 0.931 | 0.237 | 3.90 | −0.194 |
   | WEB_OR_MALWARE | 0.040 | 0.079 | 0.039 | 7.35 | −0.058 |
   | BRUTEFORCE | 0.026 | 0.042 | 0.016 | 7.32 | −0.177 |

   Every family appears in every file. No monotone trend (all |ρ| < 0.27). This
   is a random partition of a shuffled pool, not sequential attack episodes.

Ran it anyway, both arms, seeds 42/43/44. Chrono = train files 1–44, stream
files 45–63 in file order. Random = same sizes, all files, shuffled. No affine
shift, no family holdout.

| Model | Arm | F1 | MCC | FPR | Drifts | Stream (s) |
|---|---|---:|---:|---:|---:|---:|
| Static RF Top-15 | chrono | 0.98857 ± 0.00012 | 0.9772 | 0.0014 | — | 0.24 |
| Static RF Top-15 | random | 0.98839 ± 0.00029 | 0.9771 | 0.0016 | — | 0.26 |
| Static RF Full-39 | chrono | 0.98895 ± 0.00007 | 0.9780 | 0.0014 | — | 0.28 |
| Static RF Full-39 | random | 0.98889 ± 0.00020 | 0.9781 | 0.0015 | — | 0.32 |
| Adaptive-Norm RF | chrono | 0.87980 ± 0.06952 | 0.7438 | 0.2682 | — | 3.12 |
| Adaptive-Norm RF | random | 0.96181 ± 0.01448 | 0.9226 | 0.0560 | — | 3.25 |
| Drift-Triggered Top-15 | chrono | 0.98703 ± 0.00023 | 0.9742 | 0.0017 | 19.7 | 13.13 |
| Drift-Triggered Top-15 | random | 0.98813 ± 0.00051 | 0.9766 | 0.0016 | 0.3 | 3.15 |

ΔF1 (chrono − random): Static Top-15 **+0.00018**, Static Full-39 **+0.00006**,
Drift-Triggered **−0.00111**, AdaptiveNorm **−0.08200**.

Phase-wise F1 in the chrono arm is flat, not declining — Static Top-15 gives
0.9885 / 0.9885 / 0.9887 over files 45–51, 51–57, 57–63.

### Three findings

**1. File order carries no usable drift.** Both static detectors differ between
arms by less than their own across-seed SD. A chronological evaluation on
CICIoT2023 is not a stronger test that was omitted — it is a test the data
cannot support. Anyone reporting temporal drift from this file ordering is
reporting sampling noise.

**2. ADWIN fires ~20× and the firing is uninformative.** 19.7 alarms over 120
batches in the chrono arm vs 0.3 in the random arm: file boundaries do perturb
the error stream. But adapting *cost* 0.00111 F1 and 4.2× stream time, because
each retrain replaces a 200k-flow forest with a 20k-window one. **This is the
price of the mechanism when there is nothing to adapt to** — one part in a
thousand, against recoveries of 0.16 F1 (covariate shift) and 40× (withheld
families). Drift-triggered retraining is inexpensive insurance, not free
insurance. This number was missing from the paper and is now in it.

**3. Renormalisation is destabilised by file order.** AdaptiveNorm is the only
model that distinguishes the arms, and it does so by degrading (0.8798 ± 0.0695
vs 0.9618 ± 0.0145; FPR 0.056 → 0.268; per-seed 0.80–0.93). Rolling rescaling
converts between-file sampling variation too small to move either static
detector into large false-alarm swings. Independent support for §5.

## 8. On IoT-23 as a source of temporal drift

IoT-23 (Stratosphere Lab, CTU Prague; 23 scenarios, 20 malware + 3 benign,
captured 2018–2019, ~24 h per scenario; 21 GB full / 8.8 GB lightweight) ships
Zeek `conn.log.labeled` files, which **do** carry a per-flow `ts` timestamp.
It solves the problem CICIoT2023 cannot.

It does not, however, solve it *for this paper*. Zeek connection logs expose
~20 fields (`duration`, `orig_bytes`, `resp_bytes`, `conn_state`, packet and
byte counts per direction, service, protocol) and none of the CICIoT2023
39-feature schema — no `IAT`, no `Header_Length`, no per-flow `Variance`, no
flag-ratio family. The Top-15 subset, which is the paper's central artefact,
cannot be evaluated on IoT-23 at all without first deriving a fresh importance
ranking on Zeek features. That is a cross-dataset generalisation study
answering a different question ("does ranking-then-reducing transfer?") rather
than an edit to this manuscript. Labels also differ (Benign, C&C,
C&C-HeartBeat, DDoS, FileDownload, Attack, Okiru, Torii, Mirai), so the
family-holdout design would need redesigning too.

Recorded in the manuscript's Limitations and Future Work as the named next
dataset, with the schema mismatch stated explicitly.

## 9. Still open

- **Per-family recall for the binary detector** across all families, not only
  the withheld ones.
- **IoT-23 cross-dataset study** — separate paper, see §8.

## 10. Artefacts

- `scripts/build_sample_fulldata.py` — stratified sampler over all 63 files
- `scripts/drift_experiment_fulldata.py` — both regimes, five models, `--novel-frac` sweep
- `scripts/chrono_experiment.py` — chronological vs random arms (§7)
- `outputs/fulldata_drift/FINAL_family_phasewise.csv` — the table in §3
- `outputs/fulldata_drift/chrono_overall_raw.csv`, `FINAL_chrono_phasewise.csv` — §7
- `outputs/fulldata_drift/per_file_family_counts.csv` — the homogeneity profile
- `papers/L-CDAIDS.tex` (v5, 27pp) — manuscript incorporating all of the above
