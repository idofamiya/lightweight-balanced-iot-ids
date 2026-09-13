"""
Chronological (file-order) evaluation, and the control that interprets it.

Motivation
----------
The strongest possible drift evidence would be temporal: train on earlier
traffic, test on later traffic, with no injected perturbation at all. The
CICIoT2023 CSV release ships as 63 files named Merged01..Merged63, so file order
is the only ordering available.

Two facts established before running this (see profile output):
  1. The 42-column schema contains NO timestamp field. `Time_To_Live` is an IP
     hop count, not a time. True chronological order is therefore not
     recoverable from these CSVs.
  2. The files are statistically homogeneous. Family shares vary by under one
     percentage point across files for every category except DDoS (4.8 pts), and
     the rare families vary by 0.016-0.039 pts. Web/Malware is 0.07% in every
     single file. That is the signature of a random partition of a shuffled
     pool, not of sequential attack episodes.

This script therefore does not expect to find drift. It runs the file-order
split anyway, alongside a random-split control of identical size, so the
degeneracy can be reported as a measured result rather than asserted. If the
two arms agree, file order carries no temporal information and synthetic drift
regimes are the only option available with this release.

Input data
  Requires the same 63 cleaned per-file CICIoT2023 parquet files as
  build_sample_fulldata.py, under data/processed/full_corpus/ (not included in
  this repository -- see that script's docstring).

Usage
  python3 chrono_experiment.py --arm chrono --seed 42
  python3 chrono_experiment.py --arm random --seed 42
  python3 chrono_experiment.py --aggregate
"""

import warnings
warnings.filterwarnings("ignore")
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from drift_experiment_fulldata import (TOP15, ATTACK_SHARE, mblock, phase_rows,
                              run_static, run_adaptive_norm, run_drift_triggered)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed" / "full_corpus"
BASE = PROJECT_DIR / "outputs" / "fulldata_drift"
OUT = BASE
OUT.mkdir(parents=True, exist_ok=True)
FILES = sorted(glob.glob(str(DATA_DIR / "cleaned_Merged*.parquet")))

SPLIT_FILE = 44          # train on files 1..44, stream on 45..63
TRAIN_N = 200_000        # 100k benign + 100k attack, proportional
PHASE_N = 40_000
N_PHASES = 3
DROP = ("Label", "Binary_Label", "Attack_Category", "source_file")


def sample_from(files, n_benign, n_attack_total, rng):
    """Draw a proportional benign/attack sample spread evenly over `files`."""
    per_file_b = int(np.ceil(n_benign / len(files)))
    want_a = {f: int(np.ceil(n_attack_total * s / len(files)))
              for f, s in ATTACK_SHARE.items()}
    parts = []
    for path in files:
        d = pd.read_parquet(path)
        d = d[d["Attack_Category"] != "OTHER"]
        keep = []
        b = d[d["Binary_Label"] == 0]
        if len(b):
            k = min(per_file_b, len(b))
            keep.append(b.iloc[rng.choice(len(b), k, replace=False)])
        for fam, k in want_a.items():
            sub = d[d["Attack_Category"] == fam]
            if len(sub) and k > 0:
                kk = min(k, len(sub))
                keep.append(sub.iloc[rng.choice(len(sub), kk, replace=False)])
        if keep:
            part = pd.concat(keep, ignore_index=True)
            part["src"] = int(path.split("Merged")[1][:2])
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build(arm, seed):
    rng = np.random.default_rng(seed)
    early = [f for f in FILES if int(f.split("Merged")[1][:2]) <= SPLIT_FILE]
    late = [f for f in FILES if int(f.split("Merged")[1][:2]) > SPLIT_FILE]

    if arm == "chrono":
        tr = sample_from(early, TRAIN_N // 2, TRAIN_N // 2, rng)
        st = sample_from(late, (PHASE_N * N_PHASES) // 2,
                         (PHASE_N * N_PHASES) // 2, rng)
        # order the stream by capture file, so phase 1 is the earliest files
        st = st.sort_values("src", kind="mergesort").reset_index(drop=True)
    else:
        # control: identical sizes, but train and stream drawn from ALL files
        allrows = sample_from(FILES, (TRAIN_N + PHASE_N * N_PHASES) // 2,
                             (TRAIN_N + PHASE_N * N_PHASES) // 2, rng)
        allrows = allrows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        tr = allrows.iloc[:TRAIN_N]
        st = allrows.iloc[TRAIN_N:].reset_index(drop=True)

    # trim to exact sizes, balanced 50:50
    def balance(df, n):
        h = n // 2
        b = df[df.Binary_Label == 0]
        a = df[df.Binary_Label == 1]
        b = b.iloc[rng.choice(len(b), min(h, len(b)), replace=False)]
        a = a.iloc[rng.choice(len(a), min(h, len(a)), replace=False)]
        out = pd.concat([b, a])
        return out.sort_values("src", kind="mergesort") if arm == "chrono" \
            else out.sample(frac=1.0, random_state=seed)

    tr = balance(tr, TRAIN_N).reset_index(drop=True)
    st = balance(st, PHASE_N * N_PHASES).reset_index(drop=True)

    feat = [c for c in tr.columns if c not in DROP and c != "src"]
    bounds = [int(round((i + 1) * len(st) / N_PHASES)) for i in range(N_PHASES)]
    return (tr[feat].astype(np.float64), tr["Binary_Label"].astype(int).to_numpy(),
            st[feat].astype(np.float64), st["Binary_Label"].astype(int).to_numpy(),
            bounds, feat, st["src"].to_numpy())


def run(arm, seed):
    Xtr, ytr, Xst, yst, bounds, feat, src = build(arm, seed)
    spans = []
    starts = [0] + bounds[:-1]
    for a, b in zip(starts, bounds):
        spans.append(f"files {src[a:b].min()}-{src[a:b].max()}")
    print(f"[{arm} seed {seed}] train={Xtr.shape} stream={Xst.shape} "
          f"attack_rate={yst.mean():.3f}")
    print("   phase spans:", "; ".join(spans))

    sc = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=feat)
    Xst_s = pd.DataFrame(sc.transform(Xst), columns=feat)

    overalls, phases = [], []

    def add(res):
        overalls.append(res[0]); phases.extend(res[1])
        o = res[0]
        print(f"   {o['Model']:24s} acc={o['Accuracy']:.4f} F1={o['F1']:.4f} "
              f"MCC={o['MCC']:.4f} FPR={o['FPR']:.4f}")

    add(run_static(Xtr_s[TOP15], ytr, Xst_s[TOP15], yst, bounds, seed, "StaticRF-Top15"))
    add(run_static(Xtr_s, ytr, Xst_s, yst, bounds, seed, "StaticRF-Full39"))
    add(run_adaptive_norm(Xtr[TOP15], ytr, Xst[TOP15], yst, bounds, seed,
                          "AdaptiveNorm-Top15"))
    add(run_drift_triggered(Xtr_s[TOP15], ytr, Xst_s[TOP15], yst, bounds, seed,
                            "DriftTriggered-Top15"))

    for r in overalls + phases:
        r["Arm"] = arm
    json.dump({"overall": overalls, "phases": phases,
               "phase_spans": spans},
              open(OUT / f"chrono_{arm}_seed{seed}.json", "w"), indent=2, default=float)
    print(f"   -> chrono_{arm}_seed{seed}.json")


def aggregate():
    ovs, phs = [], []
    for f in sorted(glob.glob(str(OUT / "chrono_*_seed*.json"))):
        d = json.load(open(f))
        ovs.extend(d["overall"]); phs.extend(d["phases"])
    if not ovs:
        print("no results"); return
    o = pd.DataFrame(ovs); p = pd.DataFrame(phs)
    o.to_csv(OUT / "chrono_overall_raw.csv", index=False)
    p.to_csv(OUT / "chrono_phasewise_raw.csv", index=False)

    print("\n=== overall, by arm (mean over seeds) ===")
    print(o.groupby(["Arm", "Model"])[["Accuracy", "F1", "MCC", "FPR"]]
          .mean().round(4).to_string())
    print("\n=== phase-wise F1, by arm ===")
    print(p.pivot_table(index=["Arm", "Model"], columns="Phase",
                        values="F1", aggfunc="mean").round(4).to_string())
    print("\n=== chrono minus random (F1 difference) ===")
    piv = o.pivot_table(index="Model", columns="Arm", values="F1", aggfunc="mean")
    if {"chrono", "random"} <= set(piv.columns):
        piv["diff"] = (piv["chrono"] - piv["random"]).round(5)
        print(piv.round(5).to_string())
    summ = p.pivot_table(index=["Arm", "Model"], columns="Phase",
                         values=["F1", "FPR"], aggfunc="mean").round(4)
    summ.to_csv(OUT / "FINAL_chrono_phasewise.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["chrono", "random"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    else:
        run(a.arm, a.seed)

