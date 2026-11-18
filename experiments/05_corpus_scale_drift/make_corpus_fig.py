"""Two-panel figure for the corpus-scale covariate-shift regime (Sec. 6.4).

Panel (a): mean F1 by stream phase, +/- 1 SD over three seeds.
Panel (b): mean false-positive rate by stream phase, same scale of evidence.
Both panels are needed: F1 alone hides that the static detectors fail by
raising alarms, not by missing attacks.

Okabe-Ito colourblind-safe hues, validated with the dataviz palette checker.
Marker shape and line style carry identity redundantly with colour.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURF = "#fcfcfb"
SPEC = [  # model key, legend label, colour, marker, linestyle
    ("StaticRF-Top15",       "Static RF Top-15",         "#E69F00", "s", "--"),
    ("StaticRF-Full39",      "Static RF Full-39",        "#0072B2", "o", "-"),
    ("AdaptiveNorm-Top15",   "Adaptive-Norm RF Top-15",  "#CC79A7", "v", ":"),
    ("ARF-Top15",            "Adaptive Random Forest",   "#009E73", "^", "-."),
    ("DriftTriggered-Top15", "Drift-Triggered RF Top-15","#D55E00", "D", "-"),
]
PHASES = ["Phase 1", "Phase 2", "Phase 3"]

p = pd.read_csv("results/affine_phasewise_raw.csv")
mu = p.pivot_table(index="Model", columns="Phase", values=["F1", "FPR"], aggfunc="mean")
sd = p.pivot_table(index="Model", columns="Phase", values=["F1", "FPR"], aggfunc="std")

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), facecolor=SURF)
x = np.arange(3)

for ax, metric, ylab, title in [
        (axes[0], "F1",  "Mean F$_1$-score",     "(a) Detection quality"),
        (axes[1], "FPR", "Mean false-positive rate", "(b) False-alarm rate")]:
    ax.set_facecolor(SURF)
    for i, (key, lab, col, mk, ls) in enumerate(SPEC):
        # dodge slightly so overlapping series stay readable
        xo = x + (i - 2) * 0.035
        y = [mu.loc[key, (metric, ph)] for ph in PHASES]
        e = [sd.loc[key, (metric, ph)] for ph in PHASES]
        ax.errorbar(xo, y, yerr=e, color=col, marker=mk, linestyle=ls,
                    linewidth=2, markersize=8, capsize=3, elinewidth=1.4,
                    markeredgecolor=SURF, markeredgewidth=1.2, label=lab,
                    zorder=3 + i)
    ax.set_xticks(x); ax.set_xticklabels(PHASES)
    ax.set_xlim(-0.35, 2.35)
    ax.set_xlabel("Stream phase", fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_title(title, fontsize=12, loc="left")
    ax.grid(axis="y", color="#d8d8d4", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b9b4")

axes[0].set_ylim(0.60, 1.02)
axes[1].set_ylim(-0.04, 1.06)

axes[0].legend(loc="lower left", frameon=False, fontsize=9.5, handlelength=2.6)
fig.tight_layout(rect=(0, 0, 1, 1))
fig.savefig("/tmp/claude-0/-home-claude/c6058752-51d7-5049-8eaf-91224f54437e/scratchpad/figures/figures_corpus_affine.png",
            dpi=220, facecolor=SURF, bbox_inches="tight")
print("wrote figures_corpus_affine.png")
