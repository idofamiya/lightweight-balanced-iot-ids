"""
Gateway-hardware benchmark matching Table `tab:cost` / the ARF-vs-L-CDAIDS
comparison (Section 6.8) EXACTLY -- i.e. the recovered original notebook's
protocol behind Tables 5, 6 and 8, not the fresh-split/100-tree robustness
variant (that one is pi_benchmark_lcdaids.py / tab:fifteen).

Protocol, copied unchanged from notebooks/concept_drift_three_seed_FULL_recovered.ipynb
(cells 5, 19, 25, 26/30) -- confirmed against the actual notebook source, not
reconstructed from the paper's prose:

  - ONE FIXED 80:20 stratified split of final_sample_200000.csv, random_state=42.
    This split is built ONCE and reused for every seed -- "seed" varies only the
    Random Forest's own random_state, not the data split.
  - StandardScaler fit once on that fixed training fold.
  - Top-15 and Full-39 (all numeric columns) feature spaces.
  - RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1,
    class_weight="balanced") for both the static baseline and every
    drift-triggered retrain (random_state offset by the retrain count).
  - The same three-phase, 120,000-row stream (40,000-row test fold x3, with
    the Phase-2/Phase-3 covariate shifts), built once and reused for every seed.
  - ADWIN (delta=0.002), batch_size=1000, retrain_window=20000,
    min_retrain_samples=5000 -- identical to pi_benchmark_lcdaids.py.

Only the resource instrumentation (CPU%, peak RSS memory) is new; the modeling
code reproduces the recovered notebook so these numbers can go directly into
Table tab:cost next to the ARF comparison.

Usage (same pattern as the other benchmark)
--------------------------------------------
    python3 pi_benchmark_tabcost.py --seed 42
    python3 pi_benchmark_tabcost.py --seed 43
    python3 pi_benchmark_tabcost.py --seed 44
    python3 pi_benchmark_tabcost.py --aggregate

Needs final_sample_200000.csv in the same folder (same file as the other
benchmark -- if you already have it from that run, reuse it here too).
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
    raise SystemExit("psutil is required. Install with: pip install psutil")

try:
    from river import drift as river_drift
    RIVER_AVAILABLE = True
except Exception as exc:
    RIVER_AVAILABLE = False
    print("river not available, falling back to a simple window detector:", repr(exc))

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "final_sample_200000.csv"
OUTPUT_DIR = HERE / "pi_results_tabcost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
SPLIT_RANDOM_STATE = 42          # FIXED split, reused for every seed
TEST_SIZE = 0.20
N_ESTIMATORS = 50                # matches the recovered notebook exactly

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
# (identical to pi_benchmark_lcdaids.py)

class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = None
        self.samples_rss = []
        self.samples_cpu = []

    def _run(self):
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


def rf50_weighted(seed):
    """Exactly the recovered notebook's run_static_rf / run_drift_triggered_rf config."""
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1, class_weight="balanced"
    )


def load_fixed_split():
    """Build the ONE fixed 80:20 split, reused for every seed -- matches notebook cell 5."""
    if not DATA_PATH.exists():
        raise SystemExit(f"Can't find {DATA_PATH.name} next to this script.")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop).select_dtypes(include=[np.number]).copy()
    y = df["Binary_Label"].astype(int).values

    missing = [c for c in TOP_15_FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Missing Top-15 features: {missing}")

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr_raw), columns=X.columns)
    X_te = pd.DataFrame(scaler.transform(X_te_raw), columns=X.columns)
    return X_tr, y_tr, X_te, y_te, list(X.columns)


def run_seed(seed, X_train, y_train, X_stream_15, y_stream_15, X_stream_full, y_stream_full,
             boundaries, full_features):
    print(f"\n=== Seed {seed} (tab:cost protocol: fixed split, 50-tree weighted) ===")
    result = {"seed": seed, "host": platform.platform(), "python": platform.python_version()}

    X_train_15 = X_train[TOP_15_FEATURES]

    # ---- Static Top-15 fit ----
    rf_top15 = rf50_weighted(seed)
    with ResourceMonitor() as rm_fit:
        rf_top15.fit(X_train_15, y_train)
    result["fit_top15"] = {
        "wall_s": rm_fit.wall_time, "peak_rss_mb": rm_fit.peak_rss_mb,
        "mean_rss_mb": rm_fit.mean_rss_mb, "mean_cpu_pct": rm_fit.mean_cpu_pct,
        "peak_cpu_pct": rm_fit.peak_cpu_pct, "train_rows": len(X_train_15),
    }
    print(f"  fit (Top-15, {len(X_train_15):,} rows, 50-tree weighted): "
          f"{rm_fit.wall_time:.2f}s, peak RSS {rm_fit.peak_rss_mb:.1f} MB, "
          f"mean CPU {rm_fit.mean_cpu_pct:.0f}%")

    # ---- Static Full-39 fit ----
    rf_full = rf50_weighted(seed)
    with ResourceMonitor() as rm_fit_full:
        rf_full.fit(X_train[full_features], y_train)
    result["fit_full39"] = {
        "wall_s": rm_fit_full.wall_time, "peak_rss_mb": rm_fit_full.peak_rss_mb,
        "mean_rss_mb": rm_fit_full.mean_rss_mb, "mean_cpu_pct": rm_fit_full.mean_cpu_pct,
        "peak_cpu_pct": rm_fit_full.peak_cpu_pct, "train_rows": len(X_train),
    }

    # ---- Static Top-15 inference over the stream, per-batch latency ----
    x_arr = X_stream_15.values
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
    print(f"  inference (Top-15, {len(y_stream_15):,} rows): {rm_infer.wall_time:.2f}s total, "
          f"{result['inference_top15']['latency_ms_per_instance_mean']*1000:.1f} us/instance")

    with ResourceMonitor() as rm_infer_full:
        full_preds = rf_full.predict(X_stream_full.values).astype(int)
    result["inference_full39"] = {
        "wall_s": rm_infer_full.wall_time, "peak_rss_mb": rm_infer_full.peak_rss_mb,
        "mean_cpu_pct": rm_infer_full.mean_cpu_pct,
    }

    # ---- Drift-triggered retraining, starting from THIS seed's static model ----
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
                        n_estimators=N_ESTIMATORS, random_state=seed + len(retrain_points),
                        n_jobs=1, class_weight="balanced",
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

    X_train, y_train, X_test, y_test, full_features = load_fixed_split()
    X_stream_15, y_stream_15, boundaries = build_stream(X_test[TOP_15_FEATURES], y_test)
    X_stream_full, y_stream_full, boundaries_full = build_stream(X_test[full_features], y_test)
    assert boundaries == boundaries_full

    result = run_seed(args.seed, X_train, y_train, X_stream_15, y_stream_15,
                       X_stream_full, y_stream_full, boundaries, full_features)
    out = OUTPUT_DIR / f"pi_tabcost_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved {out}")


def aggregate():
    files = sorted(OUTPUT_DIR.glob("pi_tabcost_seed*.json"))
    if not files:
        raise SystemExit(f"No pi_tabcost_seed*.json files found in {OUTPUT_DIR}")
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
    summary.to_csv(OUTPUT_DIR / "pi_tabcost_summary_per_seed.csv", index=False)

    all_phase_rows = []
    for r in results:
        all_phase_rows.extend(r["phase_metrics"])
    pd.DataFrame(all_phase_rows).to_csv(OUTPUT_DIR / "pi_tabcost_phasewise_accuracy.csv", index=False)

    print("\n" + "=" * 70)
    print("PI TAB:COST BENCHMARK SUMMARY (mean over seeds)")
    print("=" * 70)
    numeric = summary.drop(columns=["Host"])
    print(numeric.mean(numeric_only=True).round(3).to_string())
    print(f"\nWrote {OUTPUT_DIR / 'pi_tabcost_summary_per_seed.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'pi_tabcost_phasewise_accuracy.csv'}")
    print("\nSend both CSV files (and the pi_tabcost_seed*.json files) back.")


if __name__ == "__main__":
    main()
