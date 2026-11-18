"""
Deterministic recomputation of Table tab:quota (the per-family sampling-pool
quota/achieved/corpus-total table in Section 3.3) directly from
per_file_family_counts.csv, without needing the 63 full-corpus parquet files
or a random draw.

Why this exists: the quota table's "Achieved" column is reported from a run
of build_sample_fulldata.py whose sample_meta.json (which build_sample_fulldata.py
itself writes) was not preserved. The exact row COUNT drawn per family is a
deterministic function of per_file_family_counts.csv and build_sample_fulldata.py's
own proportional-allocation rule -- only *which* rows are drawn is randomised
by the seed -- so this script reproduces the Achieved column exactly without
re-running the sampler itself.
"""
import csv
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
COUNTS_CSV = HERE / "results" / "per_file_family_counts.csv"

QUOTA = {
    "BENIGN": 260_000, "DDOS": 220_000, "DOS": 113_000, "MIRAI": 57_000,
    "SPOOFING": 6_000, "VULNERABILITY_SCAN": 6_000, "RECON": 5_000,
    "WEB_OR_MALWARE": 4_000, "BRUTEFORCE": 4_000,
}
TOTALS = {
    "DDOS": 23_123_908, "DOS": 6_457_920, "MIRAI": 2_516_501,
    "BENIGN": 1_050_893, "SPOOFING": 452_694, "VULNERABILITY_SCAN": 357_446,
    "RECON": 303_139, "WEB_OR_MALWARE": 23_796, "BRUTEFORCE": 12_522,
}

def main():
    with open(COUNTS_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["file"]))  # same order build_sample_fulldata.py walks files in

    taken = Counter()
    for r in rows:
        for fam, quota in QUOTA.items():
            file_fam_count = int(float(r[fam]))
            if file_fam_count == 0:
                continue
            want = int(round(quota * file_fam_count / TOTALS[fam]))
            want = min(want, file_fam_count, quota - taken[fam])
            if want > 0:
                taken[fam] += want

    print(f"{'Family':22s} {'Quota':>10s} {'Achieved':>10s} {'Corpus total':>14s}")
    total_quota = total_achieved = total_corpus = 0
    for fam, quota in QUOTA.items():
        print(f"{fam:22s} {quota:>10,d} {taken[fam]:>10,d} {TOTALS[fam]:>14,d}")
        total_quota += quota
        total_achieved += taken[fam]
        total_corpus += TOTALS[fam]
    print(f"{'Total':22s} {total_quota:>10,d} {total_achieved:>10,d} {total_corpus:>14,d}")

if __name__ == "__main__":
    main()
