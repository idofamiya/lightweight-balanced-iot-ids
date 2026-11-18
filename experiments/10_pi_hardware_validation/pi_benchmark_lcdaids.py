"""
Gateway-hardware benchmark for L-CDAIDS, to run on the Raspberry Pi itself.

Why this exists
----------------
Both peer reviewers asked for the "lightweight / resource-constrained IoT
gateway" claims to be backed by real measurements on gateway-class hardware,
not just a workstation. This script reuses the EXACT reduced-scale drift
protocol that produced the paper's Tables 5, 6 and the L-CDAIDS side of
Table (tab:cost) -- same data, same feature space, same stream construction,
same RF configuration (100 trees, max_depth=20, no class_weight) -- and adds
on-device resource instrumentation (CPU%, peak RSS memory) around every
timed operation: static fit, full-stream inference, and each individual
drift-triggered retrain.

It deliberately does NOT reimplement anything from scratch: the modelling
code is copied unchanged from scripts/three_seed_concept_drift.py (already
in the project) so the numbers this produces are directly comparable to the
existing workstation numbers in the paper. Only the instrumentation is new.

What you need on the Pi
------------------------
1. Python 3.9+ with: pandas, numpy, scikit-learn, river, psutil
   (On Raspberry Pi OS, pip already points at piwheels for prebuilt ARM
   wheels, so `pip install` should NOT need to compile anything from
   source. If it tries to compile numpy/scikit-learn from source, see
   SETUP_INSTRUCTIONS.md.)
2. The file `final_sample_200000.csv` copied into the SAME folder as this
   script (it already exists on your machine at
   G:\\IoT_IDS_Research\\data\\processed\\final_sample_200000.csv -- copy
   it onto the Pi via USB stick, scp, or however you move files to it).

Usage (run each seed separately -- keeps each run short and restartable)
--------------------------------------------------------------------------
    python3 pi_benchmark_lcdaids.py --seed 42
    python3 pi_benchmark_lcdaids.py --seed 43
    python3 pi_benchmark_lcdaids.py --seed 44
    python3 pi_benchmark_lcdaids.py --aggregate

Then send back everything under ./pi_results/ (small CSV/JSON files).
"""

import argparse
import json
import platform
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import psutil
except ImportError:
    raise SystemExit(
        "psutil is required for the resource measurements. Install it with:\n"
        "    pip install psutil\n"
    )

try:
    from river import drift as river_drift
    RIVER_AVAILABLE = True
except Exception as exc:
    RIVER_AVAILABLE = False
    print("river not available, falling back to a simple window detector:", repr(exc))

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "final_sample_200000.csv"
OUTPUT_DIR = HERE / "pi_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
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
BATCH_SIZE = 1000
RETRAIN_WINDOW = 20000
MIN_RETRAIN_SAMPLES = 5000
DROP_COLS = ["Label", "Binary_Label", "Attack_Category"]


# ---------------------------------------------------------------- resource monitor

class ResourceMonitor:
    """Samples this process's RSS memory and system-wide CPU% on a background
    thread while a block of code runs. Use as a context manager:

        with ResourceMonitor() as rm:
            do_the_thing()
        print(rm.peak_rss_mb, rm.mean_cpu_pct)
    """

    def __init__(self, interval=0.1):
        self.interval = interval
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = None
        self.samples_rss = []
        self.samples_cpu = []

    def _run(self):
        # Prime cpu_percent so the first real reading isn't 0.0/garbage.
        psutil.cpu_percent(interval=None)
        self._proc.cpu_percent(interval=None)
        while not self._stop.is_set():
            self.samples_rss.append(self._proc.memory_info().rss)
            self.samples_cpu.append(psutil.cpu_percent(interval=None))
            time.sleep(self.interval)

    def __enter__(self):
        self.t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.wall_time = time.time() - self.t0
        self._stop.set()
        self._thread.join(timeout=2)
        return False

    @property
    def peak_rss_mb(self):
        return max(self.samples_rss) / 1e6 if self.samples_rss else float("nan")

    @property
    def mean_rss_mb(self):
        return (sum(self.samples_rss) / len(self.samples_rss) / 1e6) if self.samples_rss else float("nan")

    @property
    def mean_cpu_pct(self):
        return (sum(self.samples_cpu) / len(self.samples_cpu)) if self.samples_cpu else float("nan")

    @property
    def peak_cpu_pct(self):
        return max(self.samples_cpu) if self.samples_cpu else float("nan")


# ---------------------------------------------------------------- drift detector

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


# ---------------------------------------------------------------- data / stream

def apply_phase_shift(df, features, mult, add):
    out = df.copy()
    for col in features:
        if col in out.columns:
            out[col] = out[col] * mult + add
    return out


def build_stream(x_test, y_test):
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
        "FPR": fpr, "FNR": fnr,
    }


def load_data():
    print("Loading:", DATA_PATH)
    if not DATA_PATH.exists():
        raise SystemExit(
            f"\nCan't find {DATA_PATH.name} next to this script.\n"
            f"Copy it from your machine (G:\\IoT_IDS_Research\\data\\processed\\"
            f"final_sample_200000.csv) into this same folder, then rerun."
        )
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    missing = [c for c in TOP_15_FEATURES if c not in feature_cols]
    if missing:
        raise ValueError(f"Missing Top-15 features: {missing}")
    return df, feature_cols


# ---------------------------------------------------------------- main experiment

def run_seed(seed, df, feature_cols):
    print(f"\n=== Seed {seed} ===")
    result = {"seed": seed, "host": platform.platform(), "python": platform.python_version()}

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

    # ---- Static Top-15 fit: timed + resource-monitored ----
    rf_top15 = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=1
    )
    with ResourceMonitor() as rm_fit:
        rf_top15.fit(X_train_15, y_train)
    result["fit_top15"] = {
        "wall_s": rm_fit.wall_time, "peak_rss_mb": rm_fit.peak_rss_mb,
        "mean_rss_mb": rm_fit.mean_rss_mb, "mean_cpu_pct": rm_fit.mean_cpu_pct,
        "peak_cpu_pct": rm_fit.peak_cpu_pct, "train_rows": len(X_train_15),
    }
    print(f"  fit (Top-15, {len(X_train_15):,} rows): {rm_fit.wall_time:.2f}s, "
          f"peak RSS {rm_fit.peak_rss_mb:.1f} MB, mean CPU {rm_fit.mean_cpu_pct:.0f}%")

    # ---- Static Full-39 fit (for completeness / comparison) ----
    rf_full = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=1
    )
    with ResourceMonitor() as rm_fit_full:
        rf_full.fit(X_train, y_train)
    result["fit_full39"] = {
        "wall_s": rm_fit_full.wall_time, "peak_rss_mb": rm_fit_full.peak_rss_mb,
        "mean_rss_mb": rm_fit_full.mean_rss_mb, "mean_cpu_pct": rm_fit_full.mean_cpu_pct,
        "peak_cpu_pct": rm_fit_full.peak_cpu_pct, "train_rows": len(X_train),
    }

    x_stream_15, y_stream_15, boundaries = build_stream(X_test_15, y_test)
    x_stream_full, y_stream_full, _ = build_stream(X_test, y_test)

    # ---- Static Top-15 inference over the whole stream, per-batch latency ----
    x_arr = x_stream_15.values
    n_batches = int(np.ceil(len(y_stream_15) / BATCH_SIZE))
    batch_latencies_ms = []
    static_preds = np.zeros(len(y_stream_15), dtype=int)
    with ResourceMonitor() as rm_infer:
        for b in range(n_batches):
            lo, hi = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, len(y_stream_15))
            tb0 = time.perf_counter()
            static_preds[lo:hi] = rf_top15.predict(x_arr[lo:hi]).astype(int)
            batch_latencies_ms.append((time.perf_counter() - tb0) * 1000.0)
    result["inference_top15"] = {
        "wall_s": rm_infer.wall_time, "peak_rss_mb": rm_infer.peak_rss_mb,
        "mean_cpu_pct": rm_infer.mean_cpu_pct,
        "batch_size": BATCH_SIZE, "n_batches": n_batches,
        "latency_ms_per_batch_mean": float(np.mean(batch_latencies_ms)),
        "latency_ms_per_batch_p95": float(np.percentile(batch_latencies_ms, 95)),
        "latency_ms_per_instance_mean": float(np.mean(batch_latencies_ms) / BATCH_SIZE),
    }
    print(f"  inference (Top-15 static, {len(y_stream_15):,} rows): "
          f"{rm_infer.wall_time:.2f}s total, "
          f"{result['inference_top15']['latency_ms_per_instance_mean']*1000:.1f} us/instance")

    # ---- Full-39 inference (single pass, for the cost-table comparison) ----
    with ResourceMonitor() as rm_infer_full:
        full_preds = rf_full.predict(x_stream_full).astype(int)
    result["inference_full39"] = {
        "wall_s": rm_infer_full.wall_time, "peak_rss_mb": rm_infer_full.peak_rss_mb,
        "mean_cpu_pct": rm_infer_full.mean_cpu_pct,
    }

    # ---- Drift-triggered retraining: timed + resource-monitored PER RETRAIN ----
    dt_detector = make_detector()
    dt_preds = np.zeros(len(y_stream_15), dtype=int)
    current_model = rf_top15
    drift_points, retrain_points, retrain_events = [], [], []

    with ResourceMonitor() as rm_stream:
        for b in range(n_batches):
            lo, hi = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, len(y_stream_15))
            batch_preds = current_model.predict(x_arr[lo:hi]).astype(int)
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
                    window_X = x_arr[win_lo:hi]
                    new_model = RandomForestClassifier(
                        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                        random_state=42, n_jobs=1,
                    )
                    with ResourceMonitor() as rm_retrain:
                        new_model.fit(window_X, window_y)
                    retrain_events.append({
                        "at_row": hi, "window_rows": len(window_y),
                        "wall_s": rm_retrain.wall_time,
                        "peak_rss_mb": rm_retrain.peak_rss_mb,
                        "mean_cpu_pct": rm_retrain.mean_cpu_pct,
                    })
                    current_model = new_model
                    retrain_points.append(hi)

    result["drift_triggered_stream"] = {
        "wall_s": rm_stream.wall_time, "peak_rss_mb": rm_stream.peak_rss_mb,
        "mean_cpu_pct": rm_stream.mean_cpu_pct,
        "n_drift_points": len(drift_points), "n_retrains": len(retrain_points),
        "retrain_events": retrain_events,
        "retrain_wall_s_mean": float(np.mean([e["wall_s"] for e in retrain_events])) if retrain_events else None,
        "retrain_peak_rss_mb_mean": float(np.mean([e["peak_rss_mb"] for e in retrain_events])) if retrain_events else None,
    }
    if retrain_events:
        print(f"  drift-triggered retraining: {len(retrain_events)} retrain(s), "
              f"mean {result['drift_triggered_stream']['retrain_wall_s_mean']:.2f}s each, "
              f"peak RSS {result['drift_triggered_stream']['retrain_peak_rss_mb_mean']:.1f} MB")
    else:
        print("  drift-triggered retraining: no retrains triggered this seed")

    # ---- accuracy metrics, for sanity-checking against the workstation numbers ----
    rows = []
    for phase_name, start, end in phase_slices(boundaries):
        rows.append({"Seed": seed, "Model": "Static RF Top-15", "Phase": phase_name,
                      **eval_metrics(y_stream_15[start:end], static_preds[start:end])})
        rows.append({"Seed": seed, "Model": "Static RF Full-39", "Phase": phase_name,
                      **eval_metrics(y_stream_full[start:end], full_preds[start:end])})
        rows.append({"Seed": seed, "Model": "Drift-Triggered RF Top-15", "Phase": phase_name,
                      **eval_metrics(y_stream_15[start:end], dt_preds[start:end])})
    result["phase_metrics"] = rows

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
        return

    if args.seed is None:
        raise SystemExit("Pass --seed 42 (or 43 / 44), or --aggregate once all three are done.")

    df, feature_cols = load_data()
    result = run_seed(args.seed, df, feature_cols)
    out = OUTPUT_DIR / f"pi_result_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved {out}")


def aggregate():
    files = sorted(OUTPUT_DIR.glob("pi_result_seed*.json"))
    if not files:
        raise SystemExit(f"No pi_result_seed*.json files found in {OUTPUT_DIR}")
    results = [json.load(open(f)) for f in files]

    summary_rows = []
    for r in results:
        summary_rows.append({
            "Seed": r["seed"],
            "Fit_Top15_s": r["fit_top15"]["wall_s"],
            "Fit_Top15_PeakRSS_MB": r["fit_top15"]["peak_rss_mb"],
            "Fit_Top15_MeanCPU_pct": r["fit_top15"]["mean_cpu_pct"],
            "Inference_Top15_us_per_instance": r["inference_top15"]["latency_ms_per_instance_mean"] * 1000,
            "Inference_Top15_PeakRSS_MB": r["inference_top15"]["peak_rss_mb"],
            "N_Retrains": r["drift_triggered_stream"]["n_retrains"],
            "Retrain_mean_s": r["drift_triggered_stream"]["retrain_wall_s_mean"],
            "Retrain_PeakRSS_MB": r["drift_triggered_stream"]["retrain_peak_rss_mb_mean"],
            "Stream_total_s": r["drift_triggered_stream"]["wall_s"],
            "Stream_MeanCPU_pct": r["drift_triggered_stream"]["mean_cpu_pct"],
            "Host": r["host"],
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "pi_summary_per_seed.csv", index=False)

    all_phase_rows = []
    for r in results:
        all_phase_rows.extend(r["phase_metrics"])
    pd.DataFrame(all_phase_rows).to_csv(OUTPUT_DIR / "pi_phasewise_accuracy.csv", index=False)

    print("\n" + "=" * 70)
    print("PI BENCHMARK SUMMARY (mean over seeds)")
    print("=" * 70)
    numeric = summary.drop(columns=["Host"])
    print(numeric.mean(numeric_only=True).round(3).to_string())
    print(f"\nWrote {OUTPUT_DIR / 'pi_summary_per_seed.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'pi_phasewise_accuracy.csv'}")
    print("\nSend both CSV files (and the pi_result_seed*.json files, if convenient) back.")


if __name__ == "__main__":
    main()
