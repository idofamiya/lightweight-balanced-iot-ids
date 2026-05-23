
# ============================================================
# MULTICLASS EXTENSION - PHASE 1
# Load processed dataset and inspect attack-category labels
# ============================================================

import os
import time
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

# IMPORTANT:
# Run this notebook from your main project directory:
# G:\IoT_IDS_Research
#
# If required, uncomment and modify:
# os.chdir(r"G:\IoT_IDS_Research")

DATA_PATH = Path("data/processed/final_sample_200000.csv")
OUTPUT_DIR = Path("outputs/multiclass")
MODEL_DIR = OUTPUT_DIR / "models"
FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Please run the preprocessing pipeline first "
        "or update DATA_PATH to the location of final_sample_200000.csv."
    )

df_mc = pd.read_csv(DATA_PATH)

print("Loaded dataset:", DATA_PATH)
print("Shape:", df_mc.shape)
print("Columns:", df_mc.columns.tolist()[:10], "...")

# ------------------------------------------------------------
# Create broad attack categories if Attack_Category is missing
# ------------------------------------------------------------

def map_attack_category(label):
    """
    Maps CICIoT2023 fine-grained labels into broad attack categories.
    The function is intentionally rule-based so it can handle labels
    such as DDoS-UDP_Flood, DoS-TCP_Flood, Mirai-greeth_flood, etc.
    """
    label_str = str(label).strip()
    label_low = label_str.lower()

    if label_low in ["benign", "normal"]:
        return "BENIGN"
    if "ddos" in label_low:
        return "DDoS"
    if "dos" in label_low:
        return "DoS"
    if "mirai" in label_low:
        return "Mirai"
    if "recon" in label_low or "scan" in label_low or "osscan" in label_low or "portscan" in label_low:
        return "Reconnaissance"
    if "spoof" in label_low:
        return "Spoofing"
    if "brute" in label_low or "password" in label_low:
        return "Brute Force"
    if "web" in label_low or "sql" in label_low or "xss" in label_low or "upload" in label_low or "injection" in label_low:
        return "Web/Malware"
    if "malware" in label_low or "backdoor" in label_low or "ransom" in label_low:
        return "Web/Malware"
    return "Other"

if "Attack_Category" not in df_mc.columns:
    if "Label" not in df_mc.columns:
        raise ValueError("Neither Attack_Category nor Label column is available.")
    df_mc["Attack_Category"] = df_mc["Label"].apply(map_attack_category)

print("\nAttack category distribution:")
category_counts = df_mc["Attack_Category"].value_counts()
print(category_counts)

print("\nAttack category distribution (%):")
print((df_mc["Attack_Category"].value_counts(normalize=True) * 100).round(3))

category_counts.to_csv(OUTPUT_DIR / "attack_category_distribution.csv")
print("\nSaved:", OUTPUT_DIR / "attack_category_distribution.csv")



# ============================================================
# MULTICLASS EXTENSION - PHASE 2
# Build class-aware multiclass sample
# ============================================================

# Configuration
# Keep broad classes with at least MIN_CLASS_SAMPLES.
# Rare categories can be merged into Other or dropped.
MIN_CLASS_SAMPLES = 1000
MAX_SAMPLES_PER_CLASS = 30000
RANDOM_STATE = 42

# Clean category names
df_mc["Attack_Category"] = df_mc["Attack_Category"].astype(str).str.strip()

# Merge very rare non-BENIGN classes into Other
counts_before = df_mc["Attack_Category"].value_counts()
rare_classes = [
    cls for cls, cnt in counts_before.items()
    if cnt < MIN_CLASS_SAMPLES and cls != "BENIGN"
]

if rare_classes:
    print("Merging rare classes into Other:", rare_classes)
    df_mc.loc[df_mc["Attack_Category"].isin(rare_classes), "Attack_Category"] = "Other"

counts_after_merge = df_mc["Attack_Category"].value_counts()
print("\nCounts after rare-class merge:")
print(counts_after_merge)

# Remove classes still too small after merging, except when needed
valid_classes = counts_after_merge[counts_after_merge >= MIN_CLASS_SAMPLES].index.tolist()
df_valid = df_mc[df_mc["Attack_Category"].isin(valid_classes)].copy()

print("\nClasses retained for multiclass classification:")
print(df_valid["Attack_Category"].value_counts())

# Choose a balanced sample size per class.
# This avoids one huge class dominating macro metrics.
min_available = df_valid["Attack_Category"].value_counts().min()
target_per_class = min(MAX_SAMPLES_PER_CLASS, min_available)

print("\nTarget samples per class:", target_per_class)

balanced_parts = []
for cls, group in df_valid.groupby("Attack_Category"):
    sampled = group.sample(
        n=target_per_class,
        random_state=RANDOM_STATE,
        replace=False
    )
    balanced_parts.append(sampled)

df_multi_balanced = pd.concat(balanced_parts, ignore_index=True)
df_multi_balanced = df_multi_balanced.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("\nBalanced multiclass dataset shape:", df_multi_balanced.shape)
print(df_multi_balanced["Attack_Category"].value_counts())

df_multi_balanced.to_csv(OUTPUT_DIR / "multiclass_balanced_sample.csv", index=False)
print("\nSaved:", OUTPUT_DIR / "multiclass_balanced_sample.csv")



# ============================================================
# MULTICLASS EXTENSION - PHASE 3
# Prepare full-feature and Top-15 feature train-test datasets
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Top-15 features from the binary IDS experiment
top_15_features = [
    "Number",
    "ack_flag_number",
    "HTTPS",
    "Tot_size",
    "Rate",
    "Header_Length",
    "IAT",
    "AVG",
    "ack_count",
    "Variance",
    "Std",
    "Max",
    "Tot_sum",
    "Time_To_Live",
    "Min"
]

# Remove non-feature columns
non_feature_cols = [
    "Label",
    "Binary_Label",
    "Attack_Category"
]

available_non_features = [c for c in non_feature_cols if c in df_multi_balanced.columns]

X_multi_full = df_multi_balanced.drop(columns=available_non_features)
X_multi_full = X_multi_full.select_dtypes(include=[np.number]).copy()

missing_top15 = [c for c in top_15_features if c not in X_multi_full.columns]
if missing_top15:
    raise ValueError(
        "The following Top-15 features are missing from the dataset: "
        + ", ".join(missing_top15)
    )

X_multi_top15 = X_multi_full[top_15_features].copy()

le_multi = LabelEncoder()
y_multi = le_multi.fit_transform(df_multi_balanced["Attack_Category"])

class_mapping = pd.DataFrame({
    "Encoded_Label": range(len(le_multi.classes_)),
    "Attack_Category": le_multi.classes_
})
class_mapping.to_csv(OUTPUT_DIR / "multiclass_label_mapping.csv", index=False)

print("Class mapping:")
display(class_mapping)

# Train-test split
X_train_multi_full, X_test_multi_full, y_train_multi, y_test_multi = train_test_split(
    X_multi_full,
    y_multi,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_multi
)

X_train_multi_15, X_test_multi_15, _, _ = train_test_split(
    X_multi_top15,
    y_multi,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_multi
)

# Scaling
scaler_full_multi = StandardScaler()
X_train_multi_full_scaled = scaler_full_multi.fit_transform(X_train_multi_full)
X_test_multi_full_scaled = scaler_full_multi.transform(X_test_multi_full)

scaler_15_multi = StandardScaler()
X_train_multi_15_scaled = scaler_15_multi.fit_transform(X_train_multi_15)
X_test_multi_15_scaled = scaler_15_multi.transform(X_test_multi_15)

print("Full-feature train shape:", X_train_multi_full_scaled.shape)
print("Full-feature test shape:", X_test_multi_full_scaled.shape)
print("Top-15 train shape:", X_train_multi_15_scaled.shape)
print("Top-15 test shape:", X_test_multi_15_scaled.shape)

print("\nTraining class distribution:")
print(pd.Series(y_train_multi).value_counts().sort_index())

print("\nTesting class distribution:")
print(pd.Series(y_test_multi).value_counts().sort_index())



# ============================================================
# MULTICLASS EXTENSION - PHASE 4
# Define multiclass classifiers and evaluation function
# ============================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import label_binarize

multiclass_models_top15 = {
    "Decision Tree Top-15": DecisionTreeClassifier(
        max_depth=20,
        random_state=RANDOM_STATE
    ),

    "Extra Trees Top-15": ExtraTreesClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Random Forest Top-15": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Logistic Regression Top-15": LogisticRegression(
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),

    "Gaussian NB Top-15": GaussianNB(),

    "HistGradientBoosting Top-15": HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE
    )
}

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    multiclass_models_top15["XGBoost Top-15"] = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    print("XGBoost detected and added.")
except Exception as e:
    print("XGBoost not available. Skipping XGBoost.")
    print("Reason:", e)

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    multiclass_models_top15["LightGBM Top-15"] = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    print("LightGBM detected and added.")
except Exception as e:
    print("LightGBM not available. Skipping LightGBM.")
    print("Reason:", e)

print("Multiclass models selected:")
for name in multiclass_models_top15:
    print("-", name)


def multiclass_roc_auc(model, Xte, yte, n_classes):
    """
    Computes multiclass ROC-AUC using one-vs-rest strategy.
    Returns macro and weighted ROC-AUC when probabilities/scores are available.
    """
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

        roc_macro = roc_auc_score(
            y_bin,
            y_score,
            average="macro",
            multi_class="ovr"
        )

        roc_weighted = roc_auc_score(
            y_bin,
            y_score,
            average="weighted",
            multi_class="ovr"
        )

        return roc_macro, roc_weighted

    except Exception:
        return np.nan, np.nan


def evaluate_multiclass_model(model_name, model, Xtr, Xte, ytr, yte, feature_count):
    """
    Trains and evaluates one multiclass classifier.
    """
    start_train = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = model.predict(Xte)
    inference_time = time.time() - start_pred

    acc = accuracy_score(yte, y_pred)
    macro_precision = precision_score(yte, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(yte, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(yte, y_pred, average="macro", zero_division=0)

    weighted_precision = precision_score(yte, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(yte, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(yte, y_pred, average="weighted", zero_division=0)

    bal_acc = balanced_accuracy_score(yte, y_pred)
    mcc = matthews_corrcoef(yte, y_pred)

    roc_auc_macro, roc_auc_weighted = multiclass_roc_auc(
        model,
        Xte,
        yte,
        n_classes=len(le_multi.classes_)
    )

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    model_path = MODEL_DIR / f"{safe_name}.pkl"
    joblib.dump(model, model_path)
    model_size_mb = os.path.getsize(model_path) / 1024 / 1024

    cm = confusion_matrix(yte, y_pred)
    pd.DataFrame(
        cm,
        index=le_multi.classes_,
        columns=le_multi.classes_
    ).to_csv(OUTPUT_DIR / f"confusion_matrix_{safe_name}.csv")

    report_dict = classification_report(
        yte,
        y_pred,
        target_names=le_multi.classes_,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        OUTPUT_DIR / f"classification_report_{safe_name}.csv"
    )

    return {
        "Model": model_name,
        "Features": feature_count,
        "Accuracy": acc,
        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,
        "Weighted_Precision": weighted_precision,
        "Weighted_Recall": weighted_recall,
        "Weighted_F1": weighted_f1,
        "Balanced_Accuracy": bal_acc,
        "MCC": mcc,
        "ROC_AUC_Macro": roc_auc_macro,
        "ROC_AUC_Weighted": roc_auc_weighted,
        "Training_Time_sec": train_time,
        "Inference_Time_sec": inference_time,
        "Model_Size_MB": model_size_mb
    }



# ============================================================
# MULTICLASS EXTENSION - PHASE 4 continued
# Train full-feature and Top-15 multiclass models
# ============================================================

multiclass_results = []

# Full-feature Random Forest baseline
rf_full_multi = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nTraining: Random Forest Full-39 Multiclass")
result_full = evaluate_multiclass_model(
    model_name="Random Forest Full-39",
    model=rf_full_multi,
    Xtr=X_train_multi_full_scaled,
    Xte=X_test_multi_full_scaled,
    ytr=y_train_multi,
    yte=y_test_multi,
    feature_count=X_train_multi_full_scaled.shape[1]
)

multiclass_results.append(result_full)

print(
    f"Accuracy={result_full['Accuracy']:.5f}, "
    f"Macro-F1={result_full['Macro_F1']:.5f}, "
    f"Weighted-F1={result_full['Weighted_F1']:.5f}, "
    f"MCC={result_full['MCC']:.5f}"
)

# Top-15 classifiers
for model_name, model in multiclass_models_top15.items():
    print("\nTraining:", model_name)

    result = evaluate_multiclass_model(
        model_name=model_name,
        model=model,
        Xtr=X_train_multi_15_scaled,
        Xte=X_test_multi_15_scaled,
        ytr=y_train_multi,
        yte=y_test_multi,
        feature_count=X_train_multi_15_scaled.shape[1]
    )

    multiclass_results.append(result)

    print(
        f"Accuracy={result['Accuracy']:.5f}, "
        f"Macro-F1={result['Macro_F1']:.5f}, "
        f"Weighted-F1={result['Weighted_F1']:.5f}, "
        f"MCC={result['MCC']:.5f}"
    )

multiclass_results_df = pd.DataFrame(multiclass_results)
multiclass_results_df = multiclass_results_df.sort_values(
    by=["Macro_F1", "Weighted_F1", "MCC"],
    ascending=[False, False, False]
).reset_index(drop=True)

multiclass_results_df.to_csv(
    OUTPUT_DIR / "multiclass_classifier_comparison.csv",
    index=False
)

print("\nFinal multiclass classifier comparison:")
display(multiclass_results_df)

print("\nSaved:", OUTPUT_DIR / "multiclass_classifier_comparison.csv")



# ============================================================
# MULTICLASS EXTENSION - PHASE 5
# Binary vs Multiclass comparison table
# ============================================================

# Binary Top-15 results from the previous experiments
binary_rf_top15_summary = {
    "Task": "Binary",
    "Model": "Random Forest Top-15",
    "Classes": 2,
    "Features": 15,
    "Accuracy": 0.9885,
    "Macro_F1": 0.9900,       # From binary class-wise report rounded
    "Weighted_F1": 0.9900,    # From binary class-wise report rounded
    "Balanced_Accuracy": 0.9885,
    "MCC": 0.9772,
    "ROC_AUC": 0.9963
}

# Best multiclass model according to Macro-F1
best_multi = multiclass_results_df.iloc[0].to_dict()

multiclass_best_summary = {
    "Task": "Multiclass",
    "Model": best_multi["Model"],
    "Classes": len(le_multi.classes_),
    "Features": int(best_multi["Features"]),
    "Accuracy": best_multi["Accuracy"],
    "Macro_F1": best_multi["Macro_F1"],
    "Weighted_F1": best_multi["Weighted_F1"],
    "Balanced_Accuracy": best_multi["Balanced_Accuracy"],
    "MCC": best_multi["MCC"],
    "ROC_AUC": best_multi["ROC_AUC_Macro"]
}

binary_vs_multiclass_df = pd.DataFrame([
    binary_rf_top15_summary,
    multiclass_best_summary
])

binary_vs_multiclass_df.to_csv(
    OUTPUT_DIR / "binary_vs_multiclass_summary.csv",
    index=False
)

print("Binary vs Multiclass Summary:")
display(binary_vs_multiclass_df)

print("\nSaved:", OUTPUT_DIR / "binary_vs_multiclass_summary.csv")



# ============================================================
# Export paper-ready tables and plots
# ============================================================

import matplotlib.pyplot as plt

# Paper-ready multiclass table
paper_multi = multiclass_results_df.copy()

percentage_cols = [
    "Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_Precision",
    "Weighted_Recall",
    "Weighted_F1"
]

for col in percentage_cols:
    paper_multi[col + " (%)"] = paper_multi[col] * 100

paper_multi_table = paper_multi[[
    "Model",
    "Features",
    "Accuracy (%)",
    "Macro_F1 (%)",
    "Weighted_F1 (%)",
    "Balanced_Accuracy",
    "MCC",
    "ROC_AUC_Macro",
    "Training_Time_sec",
    "Inference_Time_sec",
    "Model_Size_MB"
]].copy()

paper_multi_table = paper_multi_table.round({
    "Accuracy (%)": 2,
    "Macro_F1 (%)": 2,
    "Weighted_F1 (%)": 2,
    "Balanced_Accuracy": 4,
    "MCC": 4,
    "ROC_AUC_Macro": 4,
    "Training_Time_sec": 3,
    "Inference_Time_sec": 4,
    "Model_Size_MB": 2
})

paper_multi_table.to_csv(
    OUTPUT_DIR / "paper_ready_multiclass_comparison.csv",
    index=False
)

display(paper_multi_table)

latex_table = paper_multi_table.to_latex(
    index=False,
    escape=True,
    caption="Multiclass Attack Category Classification Results",
    label="tab:multiclass_classifier_comparison"
)

with open(OUTPUT_DIR / "multiclass_comparison_table.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print(latex_table)
print("\nSaved:", OUTPUT_DIR / "multiclass_comparison_table.tex")

# Plot Macro-F1 comparison
plt.figure(figsize=(11, 6))
plt.bar(paper_multi_table["Model"], paper_multi_table["Macro_F1 (%)"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Macro F1-score (%)")
plt.title("Multiclass Classifier Comparison")
plt.tight_layout()
plt.savefig(FIG_DIR / "multiclass_macro_f1_comparison.png", dpi=300)
plt.show()

# Plot MCC comparison
plt.figure(figsize=(11, 6))
plt.bar(paper_multi_table["Model"], paper_multi_table["MCC"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("MCC")
plt.title("Multiclass MCC Comparison")
plt.tight_layout()
plt.savefig(FIG_DIR / "multiclass_mcc_comparison.png", dpi=300)
plt.show()

print("Saved figures:")
print(FIG_DIR / "multiclass_macro_f1_comparison.png")
print(FIG_DIR / "multiclass_mcc_comparison.png")



# ============================================================
# Detailed report for best multiclass model
# ============================================================

best_model_name = multiclass_results_df.loc[0, "Model"]
print("Best multiclass model according to Macro-F1:", best_model_name)

if best_model_name == "Random Forest Full-39":
    best_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    best_model.fit(X_train_multi_full_scaled, y_train_multi)
    best_pred = best_model.predict(X_test_multi_full_scaled)
else:
    best_model = multiclass_models_top15[best_model_name]
    best_model.fit(X_train_multi_15_scaled, y_train_multi)
    best_pred = best_model.predict(X_test_multi_15_scaled)

print("\nClassification Report for Best Multiclass Model:")
print(classification_report(
    y_test_multi,
    best_pred,
    target_names=le_multi.classes_,
    zero_division=0
))

cm_best = confusion_matrix(y_test_multi, best_pred)

cm_best_df = pd.DataFrame(
    cm_best,
    index=le_multi.classes_,
    columns=le_multi.classes_
)

display(cm_best_df)

cm_best_df.to_csv(OUTPUT_DIR / "best_multiclass_confusion_matrix.csv")
print("\nSaved:", OUTPUT_DIR / "best_multiclass_confusion_matrix.csv")
