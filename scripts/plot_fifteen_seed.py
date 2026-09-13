import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "outputs" / "concept_drift" / "reproduced"
df = pd.read_csv(BASE / "phasewise_three_seed_results_summary.csv")
# columns: Model, Phase, <metric> Mean, <metric> Std ...
models = ["Static RF Full-39", "Static RF Top-15", "ADWIN-Monitored RF Top-15", "Drift-Triggered RF Top-15"]
phases = ["Phase 1", "Phase 2", "Phase 3"]
colors = {"Static RF Full-39": "#1f77b4", "Static RF Top-15": "#ff7f0e",
          "ADWIN-Monitored RF Top-15": "#2ca02c", "Drift-Triggered RF Top-15": "#d62728"}

def plot_metric(metric, ylabel, title, outfile):
    plt.figure(figsize=(9, 5.5))
    for m in models:
        sub = df[df["Model"] == m].set_index("Phase").reindex(phases)
        plt.errorbar(phases, sub[f"{metric} Mean"], yerr=sub[f"{metric} Std"],
                     marker="o", capsize=4, label=m, color=colors[m])
    plt.xlabel("Stream Phase")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE / outfile, dpi=300)
    plt.close()
    print("saved", BASE / outfile)

plot_metric("F1", "Mean F1-score", "Phase-wise F1-score under Evolving IoT Traffic (15-seed reproduction)", "phasewise_f1_comparison_15seed.png")
plot_metric("MCC", "Mean MCC", "Phase-wise MCC under Evolving IoT Traffic (15-seed reproduction)", "phasewise_mcc_comparison_15seed.png")

