# ============================================================
# Adaptive Random Forest (ARF) baseline for the L-CDAIDS drift stream
#
# Adds the canonical streaming-ensemble competitor (Gomes et al., 2017,
# river's ARFClassifier) to the comparison in Tables 6 and 7, so that
# drift-triggered batch RF retraining is measured against a genuine
# drift-aware method rather than only against static baselines.
#
# Protocol (matched to the paper's Section 4.6):
#   * the SAME fixed stratified 80:20 split (random_state=42) is used
#     for every seed; the seed varies only the model's own randomness
#   * the SAME Top-15 feature space
#   * the SAME 120,000-sample three-phase drift stream
#   * ARF is warm-started on the identical 160,000-record training set
#     the batch models are fitted on, then run prequentially
#     (test-then-train) over the stream -- i.e. it sees exactly the same
#     stream labels that drift-triggered retraining consumes.
#
# ARF is a pure-Python incremental learner and is far slower than batch
# RF, so this runner is RESUMABLE and TIME-BUDGETED: each invocation
# works for at most --budget seconds, checkpoints, and exits. Re-invoke
# until it prints DONE.
#
#   python3 arf_baseline.py --seed 42          # repeat until DONE
#   python3 arf_baseline.py --aggregate
# ============================================================

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix
)
from river.forest import ARFClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "final_sample_200000.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "arf_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoints are large (80 MB+ once the Hoeffding trees have grown) and are
# rewritten on every invocation, so they go to fast LOCAL scratch rather than
# the project folder. Only the final JSON/CSV results are written to OUTPUT_DIR.
CKPT_DIR = Path.home() / "arf_ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
SPLIT_RANDOM_STATE = 42          # fixed split, as in the paper's protocol
TEST_SIZE = 0.20
N_MODELS = 10                    # river default ensemble size

TOP_15_FEATURES = [
    "Number", "ack_flag_number", "HTTPS", "Tot_size", "Rate",
    "Header_Length", "IAT", "AVG", "ack_count", "Variance",
    "Std", "Max", "Tot_sum", "Time_To_Live", "Min"
]

PHASE2_SHIFT = ["Rate", "IAT", "Tot_size", "Header_Length", "Variance"]
PHASE3_SHIFT = ["Number", "Time_To_Live", "AVG", "Std", "Max", "Tot_sum"]


def load_split():
    """Reproduce the paper's fixed 80:20 split and Top-15 feature space."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    drop = [c for c in ["Label", "Binary_Label", "Attack_Category"] if c in df.columns]
    X = df.drop(columns=drop).select_dtypes(include=[np.number]).copy()
    y = df["Binary_Label"].astype(int).values

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr_raw), columns=X.columns)
    X_te = pd.DataFrame(scaler.transform(X_te_raw), columns=X.columns)
    return X_tr[TOP_15_FEATURES], y_tr, X_te[TOP_15_FEATURES], y_te


def build_stream(X_te_15, y_te):
    base = X_te_15.reset_index(drop=True).copy()
    ybase = pd.Series(y_te).reset_index(drop=True)

    p1 = base.copy()
    p2 = base.copy()
    for c in PHASE2_SHIFT:
        p2[c] = p2[c] * 1.35 + 0.25
    p3 = base.copy()
    for c in PHASE3_SHIFT:
        p3[c] = p3[c] * 0.65 - 0.15

    Xs = pd.concat([p1, p2, p3], ignore_index=True)
    ys = pd.concat([ybase, ybase, ybase], ignore_index=True).astype(int).values
    bounds = [len(p1), len(p1) + len(p2), len(Xs)]
    return Xs, ys, bounds


def metrics_block(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "FPR": fp / (fp + tn) if (fp + tn) else float("nan"),
        "FNR": fn / (fn + tp) if (fn + tp) else float("nan"),
    }


def state_path(seed):
    return CKPT_DIR / f"state_seed{seed}.pkl"


def legacy_state_path(seed):
    """Earlier runs checkpointed into the project folder; migrate that work."""
    return OUTPUT_DIR / f"state_seed{seed}.pkl"


def save_state(seed, st):
    with open(state_path(seed), "wb") as f:
        pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_seed(seed, budget):
    sp = state_path(seed)
    t_start = time.time()

    src = sp if sp.exists() else legacy_state_path(seed)
    st = None
    if src.exists():
        try:
            with open(src, "rb") as f:
                st = pickle.load(f)
            print(f"resumed seed {seed}: stage={st['stage']} idx={st['idx']} "
                  f"({'local' if src == sp else 'migrated from project folder'})")
        except Exception as e:
            # A checkpoint written by an invocation that was killed mid-write is
            # truncated. Discard it and restart the seed rather than crashing.
            print(f"checkpoint at {src} unusable ({type(e).__name__}: {e}); "
                  f"restarting seed {seed} from scratch")
            st = None
    if st is None:
        st = {
            "stage": "warmup", "idx": 0,
            "model": ARFClassifier(n_models=N_MODELS, seed=seed),
            "preds": [], "warmup_time": 0.0,
            "predict_time": 0.0, "learn_time": 0.0,
        }
        print(f"starting seed {seed} fresh")

    X_tr, y_tr, X_te, y_te = load_split()
    Xs, ys, bounds = build_stream(X_te, y_te)

    model = st["model"]

    # ---------- stage 1: warm-up on the batch training set ----------
    if st["stage"] == "warmup":
        recs = X_tr.to_dict(orient="records")
        i = st["idx"]
        t0 = time.time()
        while i < len(recs):
            model.learn_one(recs[i], int(y_tr[i]))
            i += 1
            if (i % 2000 == 0) and (time.time() - t_start > budget):
                break
        st["warmup_time"] += time.time() - t0
        st["idx"] = i
        if i >= len(recs):
            st["stage"] = "stream"
            st["idx"] = 0
            print(f"warm-up complete ({len(recs)} records, "
                  f"{st['warmup_time']:.1f}s cumulative)")
        else:
            st["model"] = model
            save_state(seed, st)
            print(f"PROGRESS seed {seed} warmup {i}/{len(recs)}")
            return False

    # ---------- stage 2: prequential test-then-train on the stream ----------
    if st["stage"] == "stream":
        recs = Xs.to_dict(orient="records")
        i = st["idx"]
        preds = st["preds"]
        while i < len(recs):
            t0 = time.time()
            p = model.predict_one(recs[i])
            st["predict_time"] += time.time() - t0
            preds.append(0 if p is None else int(p))
            t0 = time.time()
            model.learn_one(recs[i], int(ys[i]))
            st["learn_time"] += time.time() - t0
            i += 1
            if (i % 2000 == 0) and (time.time() - t_start > budget):
                break
        st["idx"] = i
        st["preds"] = preds
        if i < len(recs):
            st["model"] = model
            save_state(seed, st)
            print(f"PROGRESS seed {seed} stream {i}/{len(recs)}")
            return False

    # ---------- finalize ----------
    preds = np.asarray(st["preds"], dtype=int)
    model_size_mb = len(pickle.dumps(model)) / 1024 / 1024

    overall = metrics_block(ys, preds)
    overall.update({
        "Seed": seed, "Model": "Adaptive Random Forest (ARF) Top-15",
        "Warmup Time": st["warmup_time"],
        "Predict Time": st["predict_time"],
        "Learn Time": st["learn_time"],
        "Model Size MB": model_size_mb,
        "Trees": N_MODELS,
    })

    phases = []
    starts = [0] + bounds[:-1]
    for pid, (a, b) in enumerate(zip(starts, bounds), start=1):
        row = metrics_block(ys[a:b], preds[a:b])
        row.update({"Seed": seed, "Model": "Adaptive Random Forest (ARF) Top-15",
                    "Phase": f"Phase {pid}"})
        phases.append(row)

    with open(OUTPUT_DIR / f"arf_seed{seed}.json", "w") as f:
        json.dump({"overall": overall, "phases": phases}, f, indent=2)

    print(f"DONE seed {seed}")
    print(f"  overall  acc={overall['Accuracy']:.4f} F1={overall['F1']:.4f} "
          f"MCC={overall['MCC']:.4f} FPR={overall['FPR']:.4f}")
    for r in phases:
        print(f"  {r['Phase']}: acc={r['Accuracy']:.4f} F1={r['F1']:.4f} "
              f"MCC={r['MCC']:.4f} FPR={r['FPR']:.4f}")
    print(f"  warmup={st['warmup_time']:.1f}s predict={st['predict_time']:.1f}s "
          f"learn={st['learn_time']:.1f}s size={model_size_mb:.2f}MB")

    for stale in (sp, legacy_state_path(seed)):
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass  # deletion may be blocked; harmless, state is superseded
    return True


def aggregate():
    overalls, phases = [], []
    for s in SEEDS:
        p = OUTPUT_DIR / f"arf_seed{s}.json"
        if not p.exists():
            print(f"missing {p}, skipping")
            continue
        d = json.load(open(p))
        overalls.append(d["overall"])
        phases.extend(d["phases"])

    if not overalls:
        print("no results yet")
        return

    odf = pd.DataFrame(overalls)
    pdf = pd.DataFrame(phases)
    odf.to_csv(OUTPUT_DIR / "arf_overall_raw.csv", index=False)
    pdf.to_csv(OUTPUT_DIR / "arf_phasewise_raw.csv", index=False)

    print(f"\n=== ARF overall across {len(odf)} seeds (mean +/- sd) ===")
    for m in ["Accuracy", "F1", "MCC", "FPR"]:
        print(f"  {m:9s} {odf[m].mean():.4f} +/- {odf[m].std():.4f}")
    print(f"  warm-up {odf['Warmup Time'].mean():.1f}s | "
          f"stream predict {odf['Predict Time'].mean():.1f}s | "
          f"stream learn {odf['Learn Time'].mean():.1f}s | "
          f"size {odf['Model Size MB'].mean():.2f} MB")

    print("\n=== ARF phase-wise (mean +/- sd) ===")
    summ = pdf.groupby("Phase").agg(
        Acc_Mean=("Accuracy", "mean"), Acc_Std=("Accuracy", "std"),
        F1_Mean=("F1", "mean"), F1_Std=("F1", "std"),
        MCC_Mean=("MCC", "mean"), MCC_Std=("MCC", "std"),
        FPR_Mean=("FPR", "mean"), FPR_Std=("FPR", "std"),
    ).reset_index()
    summ.to_csv(OUTPUT_DIR / "arf_phasewise_summary.csv", index=False)
    print(summ.to_string(index=False))
    print("\nSaved to", OUTPUT_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--budget", type=float, default=95.0,
                    help="seconds of work per invocation before checkpointing")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()

    if a.aggregate:
        aggregate()
    elif a.seed is not None:
        run_seed(a.seed, a.budget)
    else:
        ap.print_help()

