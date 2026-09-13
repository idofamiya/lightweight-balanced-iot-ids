"""
Drift experiment over the FULL CICIoT2023 dataset, with two stream types.

The point of running BOTH streams is the contrast:

  stream=affine  -- the paper's original covariate shift (x -> a*x + b applied
                    to selected features). A reviewer can fairly object that an
                    affine shift of standardised features is exactly what
                    adaptive input normalisation undoes, so retraining may not
                    be needed at all.

  stream=family  -- attack-family holdout. Families withheld from training are
                    introduced mid-stream. This is genuine concept drift: new
                    regions of feature space become "attack". No renormalisation
                    can recover it, because nothing about the scale is wrong.

The AdaptiveNorm baseline is the instrument that makes the contrast measurable:
a static model whose input scaler is re-estimated online but whose forest is
never retrained. If it rescues the affine stream but fails the family stream,
that is direct evidence the family stream tests something the affine stream
cannot.

Models
  StaticRF-Top15     50 trees, class_weight=balanced  (paper's drift config)
  StaticRF-Full39    same, all 39 features
  AdaptiveNorm-Top15 static forest + online re-standardisation, no retraining
  DriftTriggered-Top15  ADWIN on the error stream -> recent-window refit
  ARF-Top15          Adaptive Random Forest (river), prequential

Input data
  Requires SAMPLE (outputs/fulldata_drift/sample_familyholdout.parquet), which
  is produced by running build_sample_fulldata.py first.

Usage
  python3 drift_experiment_fulldata.py --stream family --seed 42
  python3 drift_experiment_fulldata.py --stream affine --seed 42
  python3 drift_experiment_fulldata.py --aggregate
"""

import warnings
warnings.filterwarnings("ignore")
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE = PROJECT_DIR / "outputs" / "fulldata_drift"
BASE.mkdir(parents=True, exist_ok=True)
SAMPLE = BASE / "sample_familyholdout.parquet"
OUT = BASE

SEEDS = [42, 43, 44]

TOP15 = ["Number", "ack_flag_number", "HTTPS", "Tot_size", "Rate",
         "Header_Length", "IAT", "AVG", "ack_count", "Variance",
         "Std", "Max", "Tot_sum", "Time_To_Live", "Min"]

# Families present in training. These are ALL the common families, drawn in
# their natural proportions, so the Phase-1 baseline matches the realistic
# operating point reported in Table 4 rather than an artificially easy one.
# (An earlier design trained only on DDOS/DOS/MIRAI; that made benign-vs-attack
# almost perfectly separable -- Phase-1 FPR was exactly 0 -- and was not a
# representative baseline against which to measure novelty loss.)
KNOWN = ["DDOS", "DOS", "MIRAI", "SPOOFING", "VULNERABILITY_SCAN"]
KNOWN_SHARE = {   # renormalised within the known set
    "DDOS": 0.70266, "DOS": 0.19624, "MIRAI": 0.07647,
    "SPOOFING": 0.01376, "VULNERABILITY_SCAN": 0.01086,
}
# Held out from training entirely; introduced mid-stream.
NOVEL_A = ["RECON"]                            # Phase 2
NOVEL_B = ["WEB_OR_MALWARE", "BRUTEFORCE"]     # Phase 3

# Attack-family shares of the full dataset, attack rows only (33,247,935 flows).
# Used to give the affine arm the same proportional composition as the paper.
ATTACK_SHARE = {
    "DDOS": 0.69550, "DOS": 0.19424, "MIRAI": 0.07569, "SPOOFING": 0.01362,
    "VULNERABILITY_SCAN": 0.01075, "RECON": 0.00912,
    "WEB_OR_MALWARE": 0.00072, "BRUTEFORCE": 0.00038,
}

# training composition (200k, matching the paper's scale but drawn from all 34.3M)
TRAIN_Q = {"BENIGN": 100_000, "DDOS": 60_000, "DOS": 30_000, "MIRAI": 10_000}

PHASE_N = 40_000          # rows per phase, as in the paper
N_PHASES = 3

AFFINE_P2 = (["Rate", "IAT", "Tot_size", "Header_Length", "Variance"], 1.35, 0.25)
AFFINE_P3 = (["Number", "Time_To_Live", "AVG", "Std", "Max", "Tot_sum"], 0.65, -0.15)

ADWIN_DELTA = 0.002
BATCH = 1_000
RETRAIN_WINDOW = 20_000
MIN_RETRAIN = 5_000
NORM_WINDOW = 20_000      # rolling window for the AdaptiveNorm baseline


# ---------------------------------------------------------------- data

def build_splits(stream_kind, seed, novel_frac=0.5):
    """Return (X_train_raw, y_train, stream_raw, y_stream, phase_bounds).

    Rows used for training are removed from the pool before the stream is built,
    so no stream row was ever trained on. Novel families never enter training.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(SAMPLE)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    feat_cols = [c for c in df.columns
                 if c not in ("Label", "Binary_Label", "Attack_Category", "source_file")]

    # ---- training pool ----
    # family stream: train on KNOWN families only, so the novel ones are unseen.
    # affine stream: train on a proportional mix of ALL families, reproducing the
    # published setup so the affine shift is measured against the same baseline
    # difficulty the paper reports.
    if stream_kind == "family":
        quotas = {"BENIGN": 100_000}
        for fam, share in KNOWN_SHARE.items():
            quotas[fam] = int(round(100_000 * share))
    else:
        quotas = {"BENIGN": 100_000}
        for fam, share in ATTACK_SHARE.items():
            quotas[fam] = int(round(100_000 * share))
    train_idx = []
    for fam, q in quotas.items():
        pool = df.index[df["Attack_Category"] == fam].to_numpy()
        take = min(q, len(pool))
        train_idx.append(rng.choice(pool, size=take, replace=False))
    train_idx = np.concatenate(train_idx)
    train = df.loc[train_idx]
    rest = df.drop(index=train_idx)

    def draw(frame, fams, n):
        pool = frame.index[frame["Attack_Category"].isin(fams)].to_numpy()
        if len(pool) == 0 or n <= 0:
            return np.array([], dtype=int)
        return rng.choice(pool, size=min(n, len(pool)), replace=False)

    half = PHASE_N // 2          # 50:50 benign:attack within each phase
    phases = []
    pool = rest

    for p in range(1, N_PHASES + 1):
        if stream_kind == "family":
            if p == 1:
                picks = [draw(pool, ["BENIGN"], half)]
                for fam, share in KNOWN_SHARE.items():
                    picks.append(draw(pool, [fam], int(round(half * share))))
            else:
                nov = NOVEL_A if p == 2 else NOVEL_B
                n_nov = int(round(half * novel_frac))
                n_known = half - n_nov
                picks = [draw(pool, ["BENIGN"], half)]
                for fam, share in KNOWN_SHARE.items():
                    picks.append(draw(pool, [fam], int(round(n_known * share))))
                picks.append(draw(pool, nov, n_nov))
        else:
            # affine: the paper's composition -- 50:50 benign:attack with the
            # attack half drawn PROPORTIONALLY across all families, so the base
            # task has the same difficulty as the published experiment. Only the
            # features shift between phases; the class mix is constant.
            picks = [draw(pool, ["BENIGN"], half)]
            for fam, share in ATTACK_SHARE.items():
                picks.append(draw(pool, [fam], int(round(half * share))))

        idx = np.concatenate(picks)
        rng.shuffle(idx)
        phases.append(pool.loc[idx])
        pool = pool.drop(index=idx)

    stream = pd.concat(phases, ignore_index=True)
    bounds = list(np.cumsum([len(p) for p in phases]))

    X_tr = train[feat_cols].astype(np.float64)
    y_tr = train["Binary_Label"].astype(int).to_numpy()
    X_st = stream[feat_cols].astype(np.float64)
    y_st = stream["Binary_Label"].astype(int).to_numpy()
    # which stream rows belong to a family never seen in training
    novel_fams = set(NOVEL_A) | set(NOVEL_B)
    is_novel = stream["Attack_Category"].isin(novel_fams).to_numpy()

    # NOTE: for the affine arm the shift is applied to STANDARDISED features in
    # run(), not here -- the original notebook shifts X_test *after* scaling, and
    # applying it to raw values instead changes the effective magnitude entirely.
    return X_tr, y_tr, X_st, y_st, bounds, feat_cols, is_novel


def apply_affine(Xst_scaled, bounds):
    """Shift standardised features in phases 2 and 3, as the source notebook does."""
    X = Xst_scaled.copy()
    for (cols, m, a), (lo, hi) in zip(
            [AFFINE_P2, AFFINE_P3], [(bounds[0], bounds[1]), (bounds[1], bounds[2])]):
        for c in cols:
            if c in X.columns:
                X.iloc[lo:hi, X.columns.get_loc(c)] = X.iloc[lo:hi][c].to_numpy() * m + a
    return X


# ---------------------------------------------------------------- metrics

def mblock(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y, p),
        "Precision": precision_score(y, p, zero_division=0),
        "Recall": recall_score(y, p, zero_division=0),
        "F1": f1_score(y, p, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y, p),
        "MCC": matthews_corrcoef(y, p),
        "FPR": fp / (fp + tn) if (fp + tn) else float("nan"),
        "FNR": fn / (fn + tp) if (fn + tp) else float("nan"),
    }


def phase_rows(y, p, bounds, model, seed, extra=None, is_novel=None):
    rows = []
    starts = [0] + bounds[:-1]
    for i, (a, b) in enumerate(zip(starts, bounds), start=1):
        r = mblock(y[a:b], p[a:b])
        if is_novel is not None:
            nv = is_novel[a:b]; yy = y[a:b]; pp = p[a:b]
            # recall measured separately on attack rows from seen vs unseen families
            kn = (yy == 1) & (~nv)
            no = (yy == 1) & nv
            r["Recall_known"] = float(pp[kn].mean()) if kn.sum() else float("nan")
            r["Recall_novel"] = float(pp[no].mean()) if no.sum() else float("nan")
            r["n_novel"] = int(no.sum())
        r.update({"Model": model, "Seed": seed, "Phase": f"Phase {i}"})
        if extra:
            r.update(extra)
        rows.append(r)
    return rows


# ---------------------------------------------------------------- models

def rf(seed, n=50):
    return RandomForestClassifier(n_estimators=n, random_state=seed,
                                  n_jobs=2, class_weight="balanced")


def run_static(Xtr_s, ytr, Xst_s, yst, bounds, seed, name, is_novel=None):
    m = rf(seed)
    t0 = time.time(); m.fit(Xtr_s, ytr); fit_t = time.time() - t0
    t0 = time.time(); pred = m.predict(Xst_s).astype(int); pred_t = time.time() - t0
    ov = mblock(yst, pred); ov.update({"Model": name, "Seed": seed,
                                       "FitTime": fit_t, "StreamTime": pred_t})
    return ov, phase_rows(yst, pred, bounds, name, seed, is_novel=is_novel), pred


def run_adaptive_norm(Xtr_raw, ytr, Xst_raw, yst, bounds, seed, name, is_novel=None):
    """Static forest, but the input scaler is re-estimated online.

    Tests whether the drift can be undone by renormalisation alone -- i.e.
    whether retraining is actually necessary.
    """
    base_scaler = StandardScaler().fit(Xtr_raw)
    m = rf(seed)
    m.fit(base_scaler.transform(Xtr_raw), ytr)

    Xs = Xst_raw.to_numpy()
    preds = np.empty(len(Xs), dtype=int)
    buf = []
    scaler = base_scaler
    t0 = time.time()
    for lo in range(0, len(Xs), BATCH):
        hi = min(lo + BATCH, len(Xs))
        chunk = Xs[lo:hi]
        preds[lo:hi] = m.predict(scaler.transform(chunk)).astype(int)
        buf.append(chunk)
        total = sum(len(b) for b in buf)
        while total - len(buf[0]) >= NORM_WINDOW:
            total -= len(buf.pop(0))
        if total >= MIN_RETRAIN:
            scaler = StandardScaler().fit(np.vstack(buf))
    st = time.time() - t0
    ov = mblock(yst, preds); ov.update({"Model": name, "Seed": seed, "StreamTime": st})
    return ov, phase_rows(yst, preds, bounds, name, seed, is_novel=is_novel), preds


def run_drift_triggered(Xtr_s, ytr, Xst_s, yst, bounds, seed, name, is_novel=None):
    from river.drift import ADWIN
    m = rf(seed)
    m.fit(Xtr_s, ytr)
    ad = ADWIN(delta=ADWIN_DELTA)
    Xs = Xst_s.to_numpy() if hasattr(Xst_s, "to_numpy") else Xst_s
    preds = np.empty(len(Xs), dtype=int)
    memX, memy = [], []
    drifts, retrains = [], []
    t0 = time.time()
    for lo in range(0, len(Xs), BATCH):
        hi = min(lo + BATCH, len(Xs))
        chunk = Xs[lo:hi]
        bp = m.predict(chunk).astype(int)
        preds[lo:hi] = bp
        hit = False
        for j, (yt, yp) in enumerate(zip(yst[lo:hi], bp)):
            ad.update(int(yt != yp))
            if getattr(ad, "drift_detected", False):
                drifts.append(lo + j); hit = True
        memX.append(chunk); memy.append(yst[lo:hi])
        tot = sum(len(b) for b in memX)
        while tot - len(memX[0]) >= RETRAIN_WINDOW:
            tot -= len(memX.pop(0)); memy.pop(0)
        if hit:
            wy = np.concatenate(memy)
            if len(wy) >= MIN_RETRAIN and len(np.unique(wy)) == 2:
                m = rf(seed + len(retrains))
                m.fit(np.vstack(memX), wy)
                retrains.append(hi)
    st = time.time() - t0
    ov = mblock(yst, preds)
    ov.update({"Model": name, "Seed": seed, "StreamTime": st,
               "Drifts": len(drifts), "Retrains": len(retrains)})
    return ov, phase_rows(yst, preds, bounds, name, seed,
                          {"Drifts": len(drifts), "Retrains": len(retrains)},
                          is_novel=is_novel), preds


def run_arf(Xtr_s, ytr, Xst_s, yst, bounds, seed, name, is_novel=None):
    from river.forest import ARFClassifier
    m = ARFClassifier(n_models=10, seed=seed)
    tr = Xtr_s.to_dict(orient="records") if hasattr(Xtr_s, "to_dict") else None
    t0 = time.time()
    for r, lab in zip(tr, ytr):
        m.learn_one(r, int(lab))
    warm = time.time() - t0
    st_recs = Xst_s.to_dict(orient="records")
    preds = np.empty(len(st_recs), dtype=int)
    t0 = time.time()
    for i, r in enumerate(st_recs):
        p = m.predict_one(r)
        preds[i] = 0 if p is None else int(p)
        m.learn_one(r, int(yst[i]))
    st = time.time() - t0
    ov = mblock(yst, preds)
    ov.update({"Model": name, "Seed": seed, "FitTime": warm, "StreamTime": st})
    return ov, phase_rows(yst, preds, bounds, name, seed, is_novel=is_novel), preds


# ---------------------------------------------------------------- driver

def run(stream_kind, seed, only=None, novel_frac=0.5):
    Xtr_raw, ytr, Xst_raw, yst, bounds, feat_cols, is_novel = build_splits(
        stream_kind, seed, novel_frac)
    print(f"[{stream_kind} seed {seed}] train={Xtr_raw.shape} stream={Xst_raw.shape} "
          f"bounds={bounds} attack_rate={yst.mean():.3f}")

    sc = StandardScaler().fit(Xtr_raw)
    Xtr_s = pd.DataFrame(sc.transform(Xtr_raw), columns=feat_cols)
    Xst_s = pd.DataFrame(sc.transform(Xst_raw), columns=feat_cols)
    if stream_kind == "affine":
        Xst_s = apply_affine(Xst_s, bounds)
        # AdaptiveNorm needs the drifted values in its own input space too
        Xst_raw = Xst_s.copy()
        Xtr_raw = Xtr_s.copy()

    overalls, phases = [], []

    def add(res):
        overalls.append(res[0]); phases.extend(res[1])
        o = res[0]
        print(f"   {o['Model']:24s} acc={o['Accuracy']:.4f} F1={o['F1']:.4f} "
              f"MCC={o['MCC']:.4f} FPR={o['FPR']:.4f}")

    todo = only or ["static15", "static39", "adanorm", "drift", "arf"]

    if "static15" in todo:
        add(run_static(Xtr_s[TOP15], ytr, Xst_s[TOP15], yst, bounds, seed, "StaticRF-Top15", is_novel))
    if "static39" in todo:
        add(run_static(Xtr_s, ytr, Xst_s, yst, bounds, seed, "StaticRF-Full39", is_novel))
    if "adanorm" in todo:
        add(run_adaptive_norm(Xtr_raw[TOP15], ytr, Xst_raw[TOP15], yst, bounds,
                              seed, "AdaptiveNorm-Top15", is_novel))
    if "drift" in todo:
        add(run_drift_triggered(Xtr_s[TOP15], ytr, Xst_s[TOP15], yst, bounds,
                                seed, "DriftTriggered-Top15", is_novel))
    if "arf" in todo:
        add(run_arf(Xtr_s[TOP15], ytr, Xst_s[TOP15], yst, bounds, seed, "ARF-Top15", is_novel))

    tag = f"{stream_kind}_seed{seed}"
    if novel_frac != 0.5:
        tag += f"_nf{int(novel_frac*100)}"
    tag += (f"_{'-'.join(todo)}" if only else "")
    json.dump({"overall": overalls, "phases": phases},
              open(OUT / f"{tag}.json", "w"), indent=2, default=float)
    print(f"   -> {OUT / (tag + '.json')}")


def aggregate():
    import glob
    for kind in ("family", "affine"):
        ovs, phs = [], []
        for f in sorted(glob.glob(str(OUT / f"{kind}_seed*.json"))):
            if "_nf" in f:      # mixing-ratio sweep, aggregated separately
                continue
            d = json.load(open(f))
            ovs.extend(d["overall"]); phs.extend(d["phases"])
        if not ovs:
            continue
        o = pd.DataFrame(ovs); p = pd.DataFrame(phs)
        o.to_csv(OUT / f"{kind}_overall_raw.csv", index=False)
        p.to_csv(OUT / f"{kind}_phasewise_raw.csv", index=False)

        print(f"\n{'='*78}\n{kind.upper()} STREAM — overall (mean +/- sd over seeds)\n{'='*78}")
        g = o.groupby("Model").agg(["mean", "std"])[["Accuracy", "F1", "MCC", "FPR"]]
        print(g.round(4).to_string())

        print(f"\n{kind.upper()} STREAM — phase-wise F1 / FPR")
        vals = ["F1", "FPR"] + ([c for c in ("Recall_known", "Recall_novel")
                                 if c in p.columns])
        pv = p.pivot_table(index="Model", columns="Phase", values=vals,
                           aggfunc="mean")
        print(pv.round(4).to_string())
        pv.to_csv(OUT / f"{kind}_phase_pivot.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["family", "affine"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--novel-frac", type=float, default=0.5,
                    help="fraction of each drifted phase's attack traffic drawn "
                         "from held-out families")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    else:
        run(a.stream, a.seed, a.only, a.novel_frac)

