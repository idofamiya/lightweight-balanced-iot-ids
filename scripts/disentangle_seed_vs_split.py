# ============================================================
# Disentangling experiment (review Section 3.4 / priority item 4)
#
# Question: is the seed-44-style collapse in Tables 5/6/8 caused by
# (a) the specific RF configuration (50 trees, class_weight="balanced")
#     being unstable across random initializations, or
# (b) something about the one fixed 80:20 split used for all three seeds
#     in the recovered original notebook?
#
# The recovered notebook (notebooks/concept_drift_three_seed_FULL_recovered.ipynb)
# varies ONLY (a): one fixed split, three RF random_states.
# This script keeps the exact same RF config (50 trees, class_weight="balanced",
# same drift-stream construction, same ADWIN settings) but varies (b) as well:
# each seed gets its own fresh stratified 80:20 split AND its own RF random_state.
# If the instability disappears here, it's mostly about the fixed split, not the
# forest config. If it persists, it's mostly about the forest config itself.
#
# Usage:
#   python3 disentangle_seed_vs_split.py --seed 42
#   python3 disentangle_seed_vs_split.py --seed 43
#   ...
#   python3 disentangle_seed_vs_split.py --aggregate
# ============================================================

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "concept_drift" / "disentangle_seed_vs_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]  # start with 5; extend if useful
TEST_SIZE = 0.20

TOP_15_FEATURES = [
    "Number", "ack_flag_number", "HTTPS", "Tot_size", "Rate",
    "Header_Length", "IAT", "AVG", "ack_count", "Variance",
    "Std", "Max", "Tot_sum", "Time_To_Live", "Min"
]

PHASE2_SHIFT_FEATURES = ["Rate", "IAT", "Tot_size", "Header_Length", "Variance"]
PHASE2_MULT, PHASE2_ADD = 1.35, 0.25
PHASE3_SHIFT_FEATURES = ["Number", "Time_To_Live", "AVG", "Std", "Max", "Tot_sum"]
PHASE3_MULT, PHASE3_ADD = 0.65, -0.15

ADWIN_DELTA = 0.002
BATCH_SIZE = 1000
RETRAIN_WINDOW = 20000
MIN_RETRAIN_SAMPLES = 5000

try:
    from river.drift import ADWIN
except ImportError:
    raise RuntimeError("pip install river --break-system-packages")


def check_adwin_drift(adwin_object):
    if hasattr(adwin_object, "drift_detected"):
        return adwin_object.drift_detected
    elif hasattr(adwin_object, "change_detected"):
        return adwin_object.change_detected
    return False


def load_raw():
    """Load the raw balanced sample so we can make a FRESH split per seed
    (this is the key difference from the recovered notebook, which loads
    one fixed pre-split X_train/X_test and reuses it for all seeds)."""
    final_sample_path = DATA_DIR / "final_sample_200000.csv"
    if not final_sample_path.exists():
        raise FileNotFoundError(f"{final_sample_path} not found")
    df = pd.read_csv(final_sample_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    drop_cols = [c for c in ["Label", "Binary_Label", "Attack_Category"] if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).copy()
    y = df["Binary_Label"].astype(int).values
    return X, y


def build_stream(X_test_15, y_test):
    X_base = X_test_15.reset_index(drop=True).copy()
    y_base = pd.Series(y_test).reset_index(drop=True)

    X1, y1 = X_base.copy(), y_base.copy()

    X2 = X_base.copy()
    for col in PHASE2_SHIFT_FEATURES:
        X2[col] = X2[col] * PHASE2_MULT + PHASE2_ADD
    y2 = y_base.copy()

    X3 = X_base.copy()
    for col in PHASE3_SHIFT_FEATURES:
        X3[col] = X3[col] * PHASE3_MULT + PHASE3_ADD
    y3 = y_base.copy()

    X_stream = pd.concat([X1, X2, X3], ignore_index=True)
    y_stream = pd.concat([y1, y2, y3], ignore_index=True).astype(int).values
    boundaries = [len(X1), len(X1) + len(X2), len(X_stream)]
    return X_stream, y_stream, boundaries


def phase_metrics(y_true, y_pred, boundaries):
    rows = []
    starts = [0] + boundaries[:-1]
    for phase_id, (s, e) in enumerate(zip(starts, boundaries), start=1):
        yt, yp = y_true[s:e], y_pred[s:e]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append({
            "Phase": f"Phase {phase_id}",
            "Accuracy": accuracy_score(yt, yp),
            "F1": f1_score(yt, yp, zero_division=0),
            "MCC": matthews_corrcoef(yt, yp),
            "FPR": fp / (fp + tn) if (fp + tn) else np.nan,
        })
    return rows


def run_seed(seed):
    X, y = load_raw()
    missing = [f for f in TOP_15_FEATURES if f not in X.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # KEY DIFFERENCE from the recovered notebook: fresh split PER SEED.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns)

    X_train_15 = X_train[TOP_15_FEATURES].copy()
    X_test_15 = X_test[TOP_15_FEATURES].copy()

    # SAME RF config as the recovered notebook's three-seed loop: 50 trees,
    # class_weight="balanced", random_state=seed.
    rf = RandomForestClassifier(
        n_estimators=50, random_state=seed, n_jobs=1, class_weight="balanced"
    )
    rf.fit(X_train_15, y_train)

    X_stream, y_stream, boundaries = build_stream(X_test_15, y_test)
    preds = rf.predict(X_stream).astype(int)

    overall = {
        "Seed": seed,
        "Accuracy": accuracy_score(y_stream, preds),
        "F1": f1_score(y_stream, preds, zero_division=0),
        "MCC": matthews_corrcoef(y_stream, preds),
    }
    phases = phase_metrics(y_stream, preds, boundaries)
    for row in phases:
        row["Seed"] = seed

    out = {"overall": overall, "phases": phases}
    with open(OUTPUT_DIR / f"seed_{seed}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Seed {seed}: overall accuracy={overall['Accuracy']:.4f} F1={overall['F1']:.4f}")
    for row in phases:
        print(f"  {row['Phase']}: acc={row['Accuracy']:.4f} F1={row['F1']:.4f} FPR={row['FPR']:.4f}")

    return out


def aggregate():
    all_overall, all_phases = [], []
    for seed in SEEDS:
        p = OUTPUT_DIR / f"seed_{seed}.json"
        if not p.exists():
            print(f"Missing {p}, skipping")
            continue
        d = json.load(open(p))
        all_overall.append(d["overall"])
        all_phases.extend(d["phases"])

    overall_df = pd.DataFrame(all_overall)
    phases_df = pd.DataFrame(all_phases)

    overall_df.to_csv(OUTPUT_DIR / "overall_raw.csv", index=False)
    phases_df.to_csv(OUTPUT_DIR / "phases_raw.csv", index=False)

    print("\n=== Overall (fresh split + fresh RF seed each time) ===")
    print(f"Accuracy: {overall_df['Accuracy'].mean():.4f} +/- {overall_df['Accuracy'].std():.4f}")
    print(f"  range: {overall_df['Accuracy'].min():.4f} to {overall_df['Accuracy'].max():.4f}")
    print(f"F1:       {overall_df['F1'].mean():.4f} +/- {overall_df['F1'].std():.4f}")

    print("\n=== Phase 3 (the harshest shift) ===")
    p3 = phases_df[phases_df["Phase"] == "Phase 3"]
    print(f"Accuracy: {p3['Accuracy'].mean():.4f} +/- {p3['Accuracy'].std():.4f}")
    print(f"  range: {p3['Accuracy'].min():.4f} to {p3['Accuracy'].max():.4f}")
    print(f"F1:       {p3['F1'].mean():.4f} +/- {p3['F1'].std():.4f}")
    print(p3.to_string(index=False))

    summary = phases_df.groupby("Phase").agg(
        Accuracy_Mean=("Accuracy", "mean"), Accuracy_Std=("Accuracy", "std"),
        F1_Mean=("F1", "mean"), F1_Std=("F1", "std"),
        MCC_Mean=("MCC", "mean"), MCC_Std=("MCC", "std"),
        FPR_Mean=("FPR", "mean"), FPR_Std=("FPR", "std"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "phases_summary.csv", index=False)
    print("\nSaved to", OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
    elif args.seed is not None:
        run_seed(args.seed)
    else:
        parser.print_help()

