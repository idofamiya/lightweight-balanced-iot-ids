# ============================================================
# Multiclass experiment with sampling AS DESCRIBED IN THE PAPER
# (review item 6 / Section 6.2)
#
# The repo's src/multiclass_attack_classification.py equalizes every
# class down to the smallest surviving class's size (1,094/class here,
# 7,658 rows total) -- NOT what the paper describes ("up to 20,000
# records from large classes while keeping all available records from
# smaller classes", 70,479 records).
#
# This script implements the paper's described strategy instead:
#   - merge rare sub-categories (< MIN_CLASS_SAMPLES) into "Other",
#     same as the repo code and same as what produced the observed
#     7-class label set (BENIGN, DDOS, DOS, MIRAI, Other, SPOOFING,
#     VULNERABILITY_SCAN)
#   - per class: take min(count, MAX_SAMPLES_PER_CLASS=20000) --
#     i.e. cap large classes, keep ALL of the smaller ones, instead of
#     pulling every class down to the smallest one.
#
# Everything else (RF Full-39 / Top-15 configs: n_estimators=100,
# max_depth=20, random_state=42, no class_weight; 80:20 stratified
# split; StandardScaler) matches the repo code and Tables 3/4 exactly,
# so any difference in the results traces to the sampling strategy.
# ============================================================

from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "final_sample_200000.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "multiclass_paper_sampling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MIN_CLASS_SAMPLES = 1000
MAX_SAMPLES_PER_CLASS = 20000  # paper says "up to 20,000" -- this is the key fix

TOP_15_FEATURES = [
    "Number", "ack_flag_number", "HTTPS", "Tot_size", "Rate",
    "Header_Length", "IAT", "AVG", "ack_count", "Variance",
    "Std", "Max", "Tot_sum", "Time_To_Live", "Min"
]

print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]
print("Shape:", df.shape)

if "Attack_Category" not in df.columns:
    raise ValueError("Attack_Category column missing")

df["Attack_Category"] = df["Attack_Category"].astype(str).str.strip()

counts_before = df["Attack_Category"].value_counts()
print("\nCategory counts before merge:")
print(counts_before)

rare_classes = [cls for cls, cnt in counts_before.items() if cnt < MIN_CLASS_SAMPLES and cls != "BENIGN"]
if rare_classes:
    print("\nMerging rare classes into Other:", rare_classes)
    df.loc[df["Attack_Category"].isin(rare_classes), "Attack_Category"] = "Other"

counts_after_merge = df["Attack_Category"].value_counts()
print("\nCounts after rare-class merge:")
print(counts_after_merge)

valid_classes = counts_after_merge[counts_after_merge >= MIN_CLASS_SAMPLES].index.tolist()
df_valid = df[df["Attack_Category"].isin(valid_classes)].copy()

# ---- THE FIX: asymmetric cap, not equalize-to-smallest ----
sampled_parts = []
for cls, group in df_valid.groupby("Attack_Category"):
    n = min(len(group), MAX_SAMPLES_PER_CLASS)
    sampled = group.sample(n=n, random_state=RANDOM_STATE, replace=False)
    sampled_parts.append(sampled)
    print(f"  {cls}: {len(group)} available -> {n} sampled")

df_multi = pd.concat(sampled_parts, ignore_index=True)
df_multi = df_multi.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("\nFinal multiclass dataset shape:", df_multi.shape)
print("(paper states 70,479 records)")
print(df_multi["Attack_Category"].value_counts())

df_multi.to_csv(OUTPUT_DIR / "multiclass_sample_paper_strategy.csv", index=False)

# ============================================================
# Train-test split, scaling, feature sets -- unchanged from repo code
# ============================================================

non_feature_cols = [c for c in ["Label", "Binary_Label", "Attack_Category"] if c in df_multi.columns]
X_full = df_multi.drop(columns=non_feature_cols).select_dtypes(include=[np.number]).copy()
missing = [c for c in TOP_15_FEATURES if c not in X_full.columns]
if missing:
    raise ValueError(f"Missing Top-15 features: {missing}")
X_top15 = X_full[TOP_15_FEATURES].copy()

le = LabelEncoder()
y = le.fit_transform(df_multi["Attack_Category"])
class_mapping = pd.DataFrame({"Encoded_Label": range(len(le.classes_)), "Attack_Category": le.classes_})
class_mapping.to_csv(OUTPUT_DIR / "multiclass_label_mapping.csv", index=False)
print("\nClass mapping:")
print(class_mapping)

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train_15, X_test_15, _, _ = train_test_split(
    X_top15, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

scaler_full = StandardScaler()
X_train_full_s = scaler_full.fit_transform(X_train_full)
X_test_full_s = scaler_full.transform(X_test_full)

scaler_15 = StandardScaler()
X_train_15_s = scaler_15.fit_transform(X_train_15)
X_test_15_s = scaler_15.transform(X_test_15)

print("\nFull-feature train/test:", X_train_full_s.shape, X_test_full_s.shape)
print("Top-15 train/test:", X_train_15_s.shape, X_test_15_s.shape)

# ============================================================
# Models -- identical configs to the repo code / Tables 3-4
# ============================================================

n_classes = len(le.classes_)

def multiclass_roc_auc(model, Xte, yte, n_classes):
    try:
        y_bin = label_binarize(yte, classes=np.arange(n_classes))
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(Xte)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(Xte)
        else:
            return np.nan, np.nan
        if y_score.shape[1] != n_classes:
            return np.nan, np.nan
        roc_macro = roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr")
        roc_weighted = roc_auc_score(y_bin, y_score, average="weighted", multi_class="ovr")
        return roc_macro, roc_weighted
    except Exception:
        return np.nan, np.nan


def evaluate(model_name, model, Xtr, Xte, ytr, yte, feature_count):
    start = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - start

    start = time.time()
    y_pred = model.predict(Xte)
    inference_time = time.time() - start

    roc_macro, roc_weighted = multiclass_roc_auc(model, Xte, yte, n_classes)

    return {
        "Model": model_name,
        "Features": feature_count,
        "Accuracy": accuracy_score(yte, y_pred),
        "Macro_Precision": precision_score(yte, y_pred, average="macro", zero_division=0),
        "Macro_Recall": recall_score(yte, y_pred, average="macro", zero_division=0),
        "Macro_F1": f1_score(yte, y_pred, average="macro", zero_division=0),
        "Weighted_Precision": precision_score(yte, y_pred, average="weighted", zero_division=0),
        "Weighted_Recall": recall_score(yte, y_pred, average="weighted", zero_division=0),
        "Weighted_F1": f1_score(yte, y_pred, average="weighted", zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(yte, y_pred),
        "MCC": matthews_corrcoef(yte, y_pred),
        "ROC_AUC_Macro": roc_macro,
        "ROC_AUC_Weighted": roc_weighted,
        "Training_Time_sec": train_time,
        "Inference_Time_sec": inference_time,
    }


results = []

print("\nTraining: Random Forest Full-39 Multiclass")
rf_full = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=1)
r = evaluate("Random Forest Full-39", rf_full, X_train_full_s, X_test_full_s, y_train, y_test, X_train_full_s.shape[1])
results.append(r)
print(f"  Accuracy={r['Accuracy']:.4f} Macro-F1={r['Macro_F1']:.4f} Weighted-F1={r['Weighted_F1']:.4f} MCC={r['MCC']:.4f}")

models_top15 = {
    "Random Forest Top-15": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=1),
    "HistGradientBoosting Top-15": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, random_state=RANDOM_STATE),
    "Extra Trees Top-15": ExtraTreesClassifier(n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=1),
    "Decision Tree Top-15": DecisionTreeClassifier(max_depth=20, random_state=RANDOM_STATE),
    "Logistic Regression Top-15": LogisticRegression(max_iter=1000, solver="saga", n_jobs=1, random_state=RANDOM_STATE),
    "Gaussian NB Top-15": GaussianNB(),
}

for name, model in models_top15.items():
    print(f"\nTraining: {name}")
    r = evaluate(name, model, X_train_15_s, X_test_15_s, y_train, y_test, X_train_15_s.shape[1])
    results.append(r)
    print(f"  Accuracy={r['Accuracy']:.4f} Macro-F1={r['Macro_F1']:.4f} Weighted-F1={r['Weighted_F1']:.4f} MCC={r['MCC']:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "multiclass_classifier_comparison_paper_strategy.csv", index=False)

paper_table = results_df.copy()
paper_table["Accuracy (%)"] = paper_table["Accuracy"] * 100
paper_table["Macro_F1 (%)"] = paper_table["Macro_F1"] * 100
paper_table["Weighted_F1 (%)"] = paper_table["Weighted_F1"] * 100
paper_table = paper_table[[
    "Model", "Features", "Accuracy (%)", "Macro_F1 (%)", "Weighted_F1 (%)",
    "Balanced_Accuracy", "MCC", "ROC_AUC_Macro"
]].round(4)
paper_table.to_csv(OUTPUT_DIR / "paper_ready_multiclass_comparison_paper_strategy.csv", index=False)

print("\n=== Final comparison table (paper's described sampling strategy) ===")
print(paper_table.to_string(index=False))

rf_row = results_df[results_df["Model"] == "Random Forest Full-39"].iloc[0]
print("\n=== Compare to Table 9 (as printed in the paper) ===")
print("RF Full-39:  paper Acc=84.85  Macro-F1=73.56  Weighted-F1=84.25  Bal.Acc=0.6988  MCC=0.8024  ROC-AUC=0.9720")
print(
    f"RF Full-39:  ours  Acc={rf_row['Accuracy']*100:.2f}  Macro-F1={rf_row['Macro_F1']*100:.2f}  "
    f"Weighted-F1={rf_row['Weighted_F1']*100:.2f}  Bal.Acc={rf_row['Balanced_Accuracy']:.4f}  "
    f"MCC={rf_row['MCC']:.4f}  ROC-AUC={rf_row['ROC_AUC_Macro']:.4f}"
)
print("\nSaved to", OUTPUT_DIR)
