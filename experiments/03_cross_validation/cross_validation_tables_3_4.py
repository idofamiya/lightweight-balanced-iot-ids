# ============================================================
# Cross-validation for Tables 3/4/9 (review Section 5.2 / priority item 5)
#
# Tables 3 and 4 report a single 80:20 split (random_state=42) point
# estimate. This script repeats that exact RF configuration
# (n_estimators=100, max_depth=20, random_state=<fold seed>, no class
# weighting) across 5-fold stratified cross-validation for both the
# Full-39 and Top-15 feature sets, so the paper can report a confidence
# interval instead of a single point estimate.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "cross_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
TOP_15_FEATURES = [
    "Number", "ack_flag_number", "HTTPS", "Tot_size", "Rate",
    "Header_Length", "IAT", "AVG", "ack_count", "Variance",
    "Std", "Max", "Tot_sum", "Time_To_Live", "Min"
]


def load_raw():
    final_sample_path = DATA_DIR / "final_sample_200000.csv"
    df = pd.read_csv(final_sample_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    drop_cols = [c for c in ["Label", "Binary_Label", "Attack_Category"] if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).copy()
    y = df["Binary_Label"].astype(int).values
    return X, y


def run_cv(X, y, feature_cols, label):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns)

        Xtr = X_train[feature_cols]
        Xte = X_test[feature_cols]

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=1
        )
        rf.fit(Xtr, y_train)
        pred = rf.predict(Xte)
        proba = rf.predict_proba(Xte)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        rows.append({
            "Model": label,
            "Fold": fold_id,
            "Accuracy": accuracy_score(y_test, pred),
            "F1": f1_score(y_test, pred, zero_division=0),
            "Balanced_Accuracy": balanced_accuracy_score(y_test, pred),
            "MCC": matthews_corrcoef(y_test, pred),
            "ROC_AUC": roc_auc_score(y_test, proba),
            "FPR": fp / (fp + tn) if (fp + tn) else np.nan,
            "FNR": fn / (fn + tp) if (fn + tp) else np.nan,
        })
        print(f"{label} fold {fold_id}: acc={rows[-1]['Accuracy']:.4f} F1={rows[-1]['F1']:.4f}")
    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["full", "top15", "both"], default="both")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        parts = []
        for f in ["cv_raw_folds_full.csv", "cv_raw_folds_top15.csv"]:
            p = OUTPUT_DIR / f
            if p.exists():
                parts.append(pd.read_csv(p))
        raw = pd.concat(parts, ignore_index=True)
    else:
        X, y = load_raw()
        full_cols = list(X.columns)
        raw_parts = []
        if args.model in ("full", "both"):
            df_full = run_cv(X, y, full_cols, "RF Full-39 (5-fold CV)")
            df_full.to_csv(OUTPUT_DIR / "cv_raw_folds_full.csv", index=False)
            raw_parts.append(df_full)
        if args.model in ("top15", "both"):
            df_top15 = run_cv(X, y, TOP_15_FEATURES, "RF Top-15 (5-fold CV)")
            df_top15.to_csv(OUTPUT_DIR / "cv_raw_folds_top15.csv", index=False)
            raw_parts.append(df_top15)
        raw = pd.concat(raw_parts, ignore_index=True)

    raw.to_csv(OUTPUT_DIR / "cv_raw_folds.csv", index=False)

    summary = raw.groupby("Model").agg(
        Accuracy_Mean=("Accuracy", "mean"), Accuracy_Std=("Accuracy", "std"),
        F1_Mean=("F1", "mean"), F1_Std=("F1", "std"),
        Balanced_Accuracy_Mean=("Balanced_Accuracy", "mean"), Balanced_Accuracy_Std=("Balanced_Accuracy", "std"),
        MCC_Mean=("MCC", "mean"), MCC_Std=("MCC", "std"),
        ROC_AUC_Mean=("ROC_AUC", "mean"), ROC_AUC_Std=("ROC_AUC", "std"),
        FPR_Mean=("FPR", "mean"), FPR_Std=("FPR", "std"),
        FNR_Mean=("FNR", "mean"), FNR_Std=("FNR", "std"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "cv_summary.csv", index=False)

    print("\n=== 5-fold CV summary (candidate replacement/addition for Tables 3-4 CI) ===")
    print(summary.to_string(index=False))
    print("\nSaved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
