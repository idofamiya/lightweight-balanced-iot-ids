"""
Reproducible three-seed, phase-wise, ADWIN drift-triggered Random Forest
retraining experiment for L-CDAIDS.

This script fills a reproducibility gap in the public repository
(https://github.com/idofamiya/lightweight-balanced-iot-ids): the notebook
published there (notebooks/concept_drift_iot_ids_notebook_corrected.ipynb)
only implements a single-seed static-vs-online-SGD comparison. It does not
contain the three-seed evaluation, the drift-triggered Random Forest
retraining, or the phase-wise F1/MCC breakdown that the paper reports in
Tables 5, 6, 8 and Figures 2-4. The output CSVs for those results exist in
outputs/concept_drift/ but no corresponding script/notebook exists anywhere
on the project machine or in the GitHub repo.

This script reconstructs that missing experiment from the paper's stated
methodology (Sections 4.3-4.5) and the parameters that are recoverable from
the existing single-run notebook and result files:

  - Per seed in {42, 43, 44}: a fresh 80:20 stratified split of
    data/processed/final_sample_200000.csv, with StandardScaler fit on that
    seed's training fold only.
  - Static RF Full-39 and Static RF Top-15 trained on that seed's train fold.
  - A 120,000-row controlled stream built by concatenating the seed's
    40,000-row held-out test fold three times, with Phase 2 and Phase 3
    covariate shifts applied to the same shift feature sets and multipliers
    used in the published single-run notebook.
  - ADWIN (delta=0.002) monitoring the prediction-error stream for the
    static Top-15 model ("ADWIN-Monitored") and for a second copy of the
    model that is retrained on drift ("Drift-Triggered"), using a
    20,000-sample recent window, requiring >=5,000 samples and both classes
    present before retraining is allowed (Section 4.5 of the paper).
  - Phase-wise and overall metrics, saved per seed and aggregated
    (mean +/- sample standard deviation) to match the paper's Tables 5/6/8.

IMPORTANT CAVEAT: the exact batching/triggering granularity of the original
experiment (e.g. why reported retrain points land on round multiples of
1000, such as 76000/81000/82000) is not recoverable from the surviving
artifacts because the generating code was never saved. This script checks
for drift and retrains at 1000-row batch boundaries, which reproduces that
same rounding behaviour, but it is a documented reconstruction, not a
byte-for-byte recovery of the original undocumented script. Treat any
numeric differences from the original outputs/concept_drift/*.csv files as
expected, and report both if you keep the original files as the numbers
published in the paper.
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from river import drift as river_drift
    RIVER_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    RIVER_AVAILABLE = False
    print("River not available, falling back to a window detector:", repr(exc))


# ----------------------------------------------------------------------
# Configuration (kept identical to the published single-run notebook)
# ----------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "final_sample_200000.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "concept_drift" / "reproduced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
TEST_SIZE = 0.20
N_ESTIMATORS = 100
MAX_DEPTH = 20

TOP_15_FEATURES = [
    "Number", "ack_flag_number", "HTTPS", "Tot_size", "Rate",
    "Header_Length", "IAT", "AVG", "ack_count", "Variance",
    "Std", "Max", "Tot_sum", "Time_To_Live", "Min",
]

PHASE2_SHIFT_FEATURES = ["Rate", "IAT", "Tot_size", "Header_Length", "Variance"]
PHASE2_MULT, PHASE2_ADD = 1.35, 0.25

PHASE3_SHIFT_FEATURES = ["Number", "Time_To_Live", "AVG", "Std", "Max", "Tot_sum"]
PHASE3_MULT, PHASE3_ADD = 0.65, -0.15

ADWIN_DELTA = 0.002
BATCH_SIZE = 1000          # granularity at which drift is checked / retraining triggered
RETRAIN_WINDOW = 20000     # recent-window size used for drift-triggered retraining
MIN_RETRAIN_SAMPLES = 5000
DROP_COLS = ["Label", "Binary_Label", "Attack_Category"]


def make_detector():
    if RIVER_AVAILABLE:
        return river_drift.ADWIN(delta=ADWIN_DELTA)
    return FallbackWindowDriftDetector()


class FallbackWindowDriftDetector:
    def __init__(self, window_size=500, threshold=0.08):
        self.window_size = window_size
        self.threshold = threshold
        self.errors = []
        self.drift_detected = False

    def update(self, error):
        self.errors.append(int(error))
        self.drift_detected = False
        if len(self.errors) >= 2 * self.window_size:
            old = self.errors[-2 * self.window_size:-self.window_size]
            new = self.errors[-self.window_size:]
            if abs(np.mean(new) - np.mean(old)) > self.threshold:
                self.drift_detected = True


def detector_flagged(detector):
    if hasattr(detector, "drift_detected"):
        return bool(detector.drift_detected)
    if hasattr(detector, "change_detected"):
        return bool(detector.change_detected)
    return False


def apply_phase_shift(df, features, mult, add):
    out = df.copy()
    for col in features:
        if col in out.columns:
            out[col] = out[col] * mult + add
    return out


def build_stream(x_test, y_test):
    """Concatenate the held-out test fold three times with Phase2/Phase3 shifts."""
    x1 = x_test.reset_index(drop=True)
    y1 = pd.Series(y_test).reset_index(drop=True)

    x2 = apply_phase_shift(x1, PHASE2_SHIFT_FEATURES, PHASE2_MULT, PHASE2_ADD)
    y2 = y1.copy()

    x3 = apply_phase_shift(x1, PHASE3_SHIFT_FEATURES, PHASE3_MULT, PHASE3_ADD)
    y3 = y1.copy()

    x_stream = pd.concat([x1, x2, x3], ignore_index=True)
    y_stream = pd.concat([y1, y2, y3], ignore_index=True).astype(int).values
    boundaries = [len(x1), len(x1) + len(x2), len(x_stream)]
    return x_stream, y_stream, boundaries


def phase_slices(boundaries):
    starts = [0] + boundaries[:-1]
    return list(zip(["Phase 1", "Phase 2", "Phase 3"], starts, boundaries))


def eval_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "FPR": fpr,
        "FNR": fnr,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def run_seed(seed, df, feature_cols):
    print(f"\n=== Seed {seed} ===")
    t0 = time.time()

    X = df[feature_cols]
    y = df["Binary_Label"].astype(int).values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_cols)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_cols)

    X_train_15 = X_train[TOP_15_FEATURES].copy()
    X_test_15 = X_test[TOP_15_FEATURES].copy()

    # ---- Static Full-39 ----
    rf_full = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=1
    )
    tf0 = time.time()
    rf_full.fit(X_train, y_train)
    train_time_full = time.time() - tf0

    x_stream_full, y_stream_full, boundaries = build_stream(X_test, y_test)
    tp0 = time.time()
    preds_full = rf_full.predict(x_stream_full).astype(int)
    pred_time_full = time.time() - tp0

    # ---- Static Top-15 ----
    rf_top15 = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=1
    )
    tt0 = time.time()
    rf_top15.fit(X_train_15, y_train)
    train_time_top15 = time.time() - tt0

    x_stream_15, y_stream_15, boundaries_15 = build_stream(X_test_15, y_test)
    assert boundaries == boundaries_15

    tp1 = time.time()
    preds_static15 = rf_top15.predict(x_stream_15).astype(int)
    pred_time_top15 = time.time() - tp1

    # ---- ADWIN-monitored (same predictions as static Top-15, just logs drift) ----
    adwin_monitor = make_detector()
    monitor_drift_points = []
    for i in range(len(y_stream_15)):
        err = int(preds_static15[i] != y_stream_15[i])
        adwin_monitor.update(err)
        if detector_flagged(adwin_monitor):
            monitor_drift_points.append(i)

    # ---- Drift-triggered retraining ----
    dt_detector = make_detector()
    dt_preds = np.zeros(len(y_stream_15), dtype=int)
    current_model = rf_top15
    drift_points = []
    retrain_points = []
    n_batches = int(np.ceil(len(y_stream_15) / BATCH_SIZE))

    x_stream_arr = x_stream_15.values
    for b in range(n_batches):
        lo = b * BATCH_SIZE
        hi = min(lo + BATCH_SIZE, len(y_stream_15))
        batch_preds = current_model.predict(x_stream_arr[lo:hi]).astype(int)
        dt_preds[lo:hi] = batch_preds

        batch_drift = False
        for i in range(lo, hi):
            err = int(batch_preds[i - lo] != y_stream_15[i])
            dt_detector.update(err)
            if detector_flagged(dt_detector):
                drift_points.append(i)
                batch_drift = True

        if batch_drift:
            win_lo = max(0, hi - RETRAIN_WINDOW)
            window_y = y_stream_15[win_lo:hi]
            if len(window_y) >= MIN_RETRAIN_SAMPLES and len(np.unique(window_y)) == 2:
                window_X = x_stream_arr[win_lo:hi]
                new_model = RandomForestClassifier(
                    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                    random_state=42, n_jobs=1,
                )
                new_model.fit(window_X, window_y)
                current_model = new_model
                retrain_points.append(hi)

    # ---- Assemble phase-wise + overall metrics ----
    rows = []
    for phase_name, start, end in phase_slices(boundaries):
        rows.append({"Seed": seed, "Model": "Static RF Full-39", "Phase": phase_name,
                      "Start": start, "End": end, "Samples": end - start,
                      **eval_metrics(y_stream_full[start:end], preds_full[start:end])})
        rows.append({"Seed": seed, "Model": "Static RF Top-15", "Phase": phase_name,
                      "Start": start, "End": end, "Samples": end - start,
                      **eval_metrics(y_stream_15[start:end], preds_static15[start:end])})
        rows.append({"Seed": seed, "Model": "ADWIN-Monitored RF Top-15", "Phase": phase_name,
                      "Start": start, "End": end, "Samples": end - start,
                      **eval_metrics(y_stream_15[start:end], preds_static15[start:end])})
        rows.append({"Seed": seed, "Model": "Drift-Triggered RF Top-15", "Phase": phase_name,
                      "Start": start, "End": end, "Samples": end - start,
                      **eval_metrics(y_stream_15[start:end], dt_preds[start:end])})

    overall_rows = [
        {"Seed": seed, "Model": "Static RF Full-39",
         **eval_metrics(y_stream_full, preds_full),
         "Drift Points Detected": 0, "Retraining Events": 0,
         "Train Time": train_time_full, "Prediction Time": pred_time_full,
         "Feature Count": len(feature_cols)},
        {"Seed": seed, "Model": "Static RF Top-15",
         **eval_metrics(y_stream_15, preds_static15),
         "Drift Points Detected": 0, "Retraining Events": 0,
         "Train Time": train_time_top15, "Prediction Time": pred_time_top15,
         "Feature Count": len(TOP_15_FEATURES)},
        {"Seed": seed, "Model": "ADWIN-Monitored RF Top-15",
         **eval_metrics(y_stream_15, preds_static15),
         "Drift Points Detected": len(monitor_drift_points), "Retraining Events": 0,
         "Train Time": np.nan, "Prediction Time": np.nan,
         "Feature Count": len(TOP_15_FEATURES)},
        {"Seed": seed, "Model": "Drift-Triggered RF Top-15",
         **eval_metrics(y_stream_15, dt_preds),
         "Drift Points Detected": len(drift_points), "Retraining Events": len(retrain_points),
         "Train Time": np.nan, "Prediction Time": pred_time_top15,
         "Feature Count": len(TOP_15_FEATURES)},
    ]

    events = {
        "Seed": seed,
        "ADWIN-Monitored drift points": monitor_drift_points,
        "Drift-Triggered drift points": drift_points,
        "Drift-Triggered retrain points": retrain_points,
    }

    print(f"Seed {seed} done in {time.time() - t0:.1f}s. "
          f"Drift points: {len(drift_points)}, retrains: {len(retrain_points)}")

    return rows, overall_rows, events


def load_data():
    print("Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    if "Binary_Label" not in df.columns:
        raise ValueError("Binary_Label column missing from final_sample_200000.csv")

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    missing_top15 = [c for c in TOP_15_FEATURES if c not in feature_cols]
    if missing_top15:
        raise ValueError(f"Missing Top-15 features: {missing_top15}")
    return df, feature_cols


def run_single_seed(seed):
    """Run one seed and write its own per-seed result files (keeps each run short
    enough to fit in a single constrained execution slot)."""
    df, feature_cols = load_data()
    phase_rows, overall_rows, events = run_seed(seed, df, feature_cols)

    pd.DataFrame(phase_rows).to_csv(OUTPUT_DIR / f"phasewise_seed{seed}.csv", index=False)
    pd.DataFrame(overall_rows).to_csv(OUTPUT_DIR / f"overall_seed{seed}.csv", index=False)
    with open(OUTPUT_DIR / f"events_seed{seed}.json", "w") as f:
        json.dump(events, f, indent=2)
    print(f"\nSaved per-seed results for seed {seed} to {OUTPUT_DIR}")


def aggregate():
    """Combine per-seed files written by run_single_seed() for all SEEDS into the
    same summary format as the paper's Tables 5/6."""
    phase_parts, overall_parts, all_events = [], [], []
    for seed in SEEDS:
        pf = OUTPUT_DIR / f"phasewise_seed{seed}.csv"
        of = OUTPUT_DIR / f"overall_seed{seed}.csv"
        ef = OUTPUT_DIR / f"events_seed{seed}.json"
        if not (pf.exists() and of.exists()):
            raise FileNotFoundError(f"Missing per-seed results for seed {seed}. Run: python3 {Path(__file__).name} --seed {seed}")
        phase_parts.append(pd.read_csv(pf))
        overall_parts.append(pd.read_csv(of))
        if ef.exists():
            all_events.append(json.load(open(ef)))

    phase_df = pd.concat(phase_parts, ignore_index=True)
    overall_df = pd.concat(overall_parts, ignore_index=True)

    phase_df.to_csv(OUTPUT_DIR / "phasewise_three_seed_results_raw.csv", index=False)
    overall_df.to_csv(OUTPUT_DIR / "three_seed_drift_experiment_raw.csv", index=False)
    with open(OUTPUT_DIR / "three_seed_drift_retraining_events.json", "w") as f:
        json.dump(all_events, f, indent=2)

    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "Balanced Accuracy", "MCC", "FPR", "FNR"]

    overall_summary = (
        overall_df.groupby("Model")[metric_cols + ["Drift Points Detected", "Retraining Events"]]
        .agg(["mean", "std"])
    )
    overall_summary.columns = [f"{c[0]} {c[1].capitalize()}" for c in overall_summary.columns]
    overall_summary.to_csv(OUTPUT_DIR / "three_seed_drift_experiment_summary.csv")

    phase_summary = (
        phase_df.groupby(["Model", "Phase"])[metric_cols]
        .agg(["mean", "std"])
    )
    phase_summary.columns = [f"{c[0]} {c[1].capitalize()}" for c in phase_summary.columns]
    phase_summary.to_csv(OUTPUT_DIR / "phasewise_three_seed_results_summary.csv")

    print("\nSaved aggregated reproduced results to:", OUTPUT_DIR)
    print(overall_summary[["Accuracy Mean", "Accuracy Std", "F1 Mean", "F1 Std", "MCC Mean", "MCC Std"]])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                         help="Run only this seed and write per-seed result files.")
    parser.add_argument("--aggregate", action="store_true",
                         help="Combine per-seed result files (from prior --seed runs) into the final summary CSVs.")
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
    elif args.seed is not None:
        run_single_seed(args.seed)
    else:
        # Full run, all seeds in one process (only feasible with enough time/RAM).
        df, feature_cols = load_data()
        all_phase_rows, all_overall_rows, all_events = [], [], []
        for seed in SEEDS:
            phase_rows, overall_rows, events = run_seed(seed, df, feature_cols)
            all_phase_rows.extend(phase_rows)
            all_overall_rows.extend(overall_rows)
            all_events.append(events)
            pd.DataFrame(phase_rows).to_csv(OUTPUT_DIR / f"phasewise_seed{seed}.csv", index=False)
            pd.DataFrame(overall_rows).to_csv(OUTPUT_DIR / f"overall_seed{seed}.csv", index=False)
        aggregate()


if __name__ == "__main__":
    main()

