"""
Stratified sampler over the FULL cleaned CICIoT2023 dataset (34,298,828 flows,
63 capture files) for the attack-family holdout drift experiment.

Why this exists
---------------
The paper's original 200k sample was a faithful uniform draw, so it reproduced
the natural family proportions -- including leaving Brute Force with 32 records
and Web/Malware with 84. That is too few to learn or to measure recall on. This
sampler draws from all 34.3M flows with explicit per-family quotas instead, so
the rare families are present in usable numbers and the policy is stated rather
than implicit.

Method: two passes. Pass 1 (already run) gave per-family totals. Pass 2 walks
every file once and takes each family's quota in proportion to that file's share
of the family, so the result is a uniform random sample over the whole dataset
without ever materialising 34M rows in memory.

Input data
----------
This script expects the 63 cleaned per-file CICIoT2023 parquet files (the same
cleaning pipeline used for data/processed/final_sample_200000.csv, but kept as
one file per capture rather than pre-sampled) under data/processed/full_corpus/.
Those files are ~34.3M rows in total and are NOT included in this repository;
obtain the raw CSVs from the CICIoT2023 release per data/README.md, run the
repo's cleaning step on each file, and write the 63 per-file parquet outputs
there (or point DATA_GLOB below at wherever you have them).
"""

import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed" / "full_corpus"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "fulldata_drift"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_GLOB = str(DATA_DIR / "cleaned_Merged*.parquet")
OUT = OUTPUT_DIR / "sample_familyholdout.parquet"

# Families the detector is TRAINED on ("known")
KNOWN_ATTACKS = ["DDOS", "DOS", "MIRAI"]
# Families withheld from training and introduced mid-stream ("novel")
NOVEL_A = ["SPOOFING", "VULNERABILITY_SCAN"]      # introduced in Phase 2
NOVEL_B = ["RECON", "WEB_OR_MALWARE", "BRUTEFORCE"]  # introduced in Phase 3

# Per-family quotas. Training quotas are generous for the known families; the
# novel families only need enough to populate their stream segments.
QUOTA = {
    "BENIGN":             260_000,   # 200k train + 60k stream
    "DDOS":               220_000,
    "DOS":                113_000,
    "MIRAI":               57_000,
    "SPOOFING":             6_000,
    "VULNERABILITY_SCAN":   6_000,
    "RECON":                5_000,
    "WEB_OR_MALWARE":       4_000,
    "BRUTEFORCE":           4_000,
}

# Per-family totals over the full dataset (from the profiling pass).
TOTALS = {
    "DDOS": 23_123_908, "DOS": 6_457_920, "MIRAI": 2_516_501,
    "BENIGN": 1_050_893, "SPOOFING": 452_694, "VULNERABILITY_SCAN": 357_446,
    "RECON": 303_139, "WEB_OR_MALWARE": 23_796, "BRUTEFORCE": 12_522,
}


def main():
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(DATA_GLOB))
    print(f"{len(files)} files")

    taken = Counter()
    parts = []

    for fi, f in enumerate(files, 1):
        d = pd.read_parquet(f)
        # Drop the 9 malformed rows carrying a literal 'NAN' label.
        d = d[d["Attack_Category"] != "OTHER"]
        d = d[d["Label"].astype(str).str.upper() != "NAN"]

        keep = []
        for fam, quota in QUOTA.items():
            sub = d[d["Attack_Category"] == fam]
            if len(sub) == 0:
                continue
            # this file's proportional share of the family's quota
            want = int(round(quota * len(sub) / TOTALS[fam]))
            want = min(want, len(sub), quota - taken[fam])
            if want <= 0:
                continue
            idx = rng.choice(len(sub), size=want, replace=False)
            keep.append(sub.iloc[idx])
            taken[fam] += want

        if keep:
            part = pd.concat(keep, ignore_index=True)
            part["source_file"] = fi          # preserves capture order
            parts.append(part)

        if fi % 10 == 0 or fi == len(files):
            print(f"  {fi}/{len(files)} files; rows so far {sum(len(p) for p in parts):,}")

    df = pd.concat(parts, ignore_index=True)

    # Top up any family that fell short of quota through rounding.
    print("\nquota fill:")
    for fam, q in QUOTA.items():
        got = int((df["Attack_Category"] == fam).sum())
        print(f"  {fam:20s} {got:>8,} / {q:,}  ({100*got/q:5.1f}%)")

    # Shrink float64 -> float32 to halve memory for downstream experiments.
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)

    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  shape={df.shape}")

    meta = {
        "seed": SEED,
        "total_population": 34_298_828,
        "sample_rows": int(len(df)),
        "quota": QUOTA,
        "known_attacks": KNOWN_ATTACKS,
        "novel_a": NOVEL_A,
        "novel_b": NOVEL_B,
        "achieved": {k: int((df["Attack_Category"] == k).sum()) for k in QUOTA},
    }
    with open(OUTPUT_DIR / "sample_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print("wrote sample_meta.json")


if __name__ == "__main__":
    main()

