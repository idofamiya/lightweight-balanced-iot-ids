# ============================================================
# CLASSIFIER COMPARISON EXPERIMENT
# Purpose:
# Compare the proposed RF Top-15 model with other ML classifiers
# using the same balanced CICIoT2023 train-test split.
# ============================================================

import os
import time
import joblib
import numpy as np
import pandas as pd

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

# Ensure labels are one-dimensional arrays
y_train_arr = np.ravel(y_train)
y_test_arr = np.ravel(y_test)

# Ensure Top-15 data exists
try:
    X_train_15
    X_test_15
except NameError:
    X_train_15 = X_train[top_15]
    X_test_15 = X_test[top_15]

print("Training set:", X_train_15.shape)
print("Testing set:", X_test_15.shape)
print("Class distribution in training set:")
print(pd.Series(y_train_arr).value_counts().sort_index())
print("Class distribution in testing set:")
print(pd.Series(y_test_arr).value_counts().sort_index())

# ============================================================
# 1. Define comparison classifiers
# ============================================================

models = {
    "Decision Tree Top-15": DecisionTreeClassifier(
        max_depth=20,
        random_state=42
    ),

    "Extra Trees Top-15": ExtraTreesClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest Top-15": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    "Logistic Regression Top-15": LogisticRegression(
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
        random_state=42
    ),

    "Gaussian NB Top-15": GaussianNB(),

    "HistGradientBoosting Top-15": HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42
    )
}

# Optional: XGBoost, only if installed.
try:
    from xgboost import XGBClassifier

    models["XGBoost Top-15"] = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    print("XGBoost detected and added.")
except Exception as e:
    print("XGBoost not available. Skipping XGBoost.")
    print("Reason:", e)

# Optional: LightGBM, only if installed.
try:
    from lightgbm import LGBMClassifier

    models["LightGBM Top-15"] = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    print("LightGBM detected and added.")
except Exception as e:
    print("LightGBM not available. Skipping LightGBM.")
    print("Reason:", e)

print("Models selected for comparison:")
for name in models:
    print("-", name)

# ============================================================
# 2. Evaluation function
# ============================================================

def evaluate_model(model_name, model, Xtr, Xte, ytr, yte, save_model=True):
    """
    Train and evaluate one classifier.
    Returns a dictionary containing performance and efficiency metrics.
    """
    start_train = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = model.predict(Xte)
    inference_time = time.time() - start_pred

    # Confusion matrix format:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(yte, y_pred).ravel()

    acc = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred, zero_division=0)
    f1 = f1_score(yte, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(yte, y_pred)
    mcc = matthews_corrcoef(yte, y_pred)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # ROC-AUC: use predict_proba if available, otherwise decision_function.
    roc_auc = np.nan
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(Xte)[:, 1]
            roc_auc = roc_auc_score(yte, y_score)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(Xte)
            roc_auc = roc_auc_score(yte, y_score)
    except Exception:
        roc_auc = np.nan

    model_size_mb = np.nan
    if save_model:
        os.makedirs("outputs/models", exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        model_path = f"outputs/models/{safe_name}.pkl"
        joblib.dump(model, model_path)
        model_size_mb = os.path.getsize(model_path) / 1024 / 1024

    return {
        "Model": model_name,
        "Features": Xtr.shape[1],
        "Accuracy": acc,
        "Precision_Attack": prec,
        "Recall_Attack": rec,
        "F1_Attack": f1,
        "Balanced_Accuracy": bal_acc,
        "MCC": mcc,
        "ROC_AUC": roc_auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "FPR": fpr,
        "FNR": fnr,
        "Training_Time_sec": train_time,
        "Inference_Time_sec": inference_time,
        "Model_Size_MB": model_size_mb
    }


# ============================================================
# 3. Run comparison on Top-15 feature subset
# ============================================================

results = []

for model_name, model in models.items():
    print("\nTraining:", model_name)
    result = evaluate_model(
        model_name=model_name,
        model=model,
        Xtr=X_train_15,
        Xte=X_test_15,
        ytr=y_train_arr,
        yte=y_test_arr,
        save_model=True
    )
    results.append(result)

    print(
        f"Accuracy={result['Accuracy']:.5f}, "
        f"F1={result['F1_Attack']:.5f}, "
        f"FPR={result['FPR']:.5f}, "
        f"Train={result['Training_Time_sec']:.2f}s, "
        f"Inference={result['Inference_Time_sec']:.4f}s"
    )

comparison_df = pd.DataFrame(results)

# Sort by F1-score and then false positive rate
comparison_df = comparison_df.sort_values(
    by=["F1_Attack", "FPR"],
    ascending=[False, True]
).reset_index(drop=True)

os.makedirs("outputs", exist_ok=True)
comparison_df.to_csv("outputs/classifier_comparison_top15.csv", index=False)

print("\nFinal Classifier Comparison Table:")
display(comparison_df)

print("\nSaved: outputs/classifier_comparison_top15.csv")

# ============================================================
# 4. Add Full Random Forest baseline to comparison
# ============================================================

rf_full_for_comparison = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf_full_result = evaluate_model(
    model_name="Random Forest Full-39",
    model=rf_full_for_comparison,
    Xtr=X_train,
    Xte=X_test,
    ytr=y_train_arr,
    yte=y_test_arr,
    save_model=True
)

full_plus_top15_df = pd.concat(
    [pd.DataFrame([rf_full_result]), comparison_df],
    ignore_index=True
)

full_plus_top15_df.to_csv(
    "outputs/classifier_comparison_full_and_top15.csv",
    index=False
)

print("Full RF baseline added.")
display(full_plus_top15_df)

print("\nSaved: outputs/classifier_comparison_full_and_top15.csv")

# ============================================================
# 5. Print classification report for the best model
# ============================================================

best_model_name = comparison_df.loc[0, "Model"]
print("Best model according to F1-score:", best_model_name)

# Re-train the best model for detailed report
best_model = models[best_model_name]
best_model.fit(X_train_15, y_train_arr)
best_pred = best_model.predict(X_test_15)

print("\nClassification Report:")
print(classification_report(y_test_arr, best_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test_arr, best_pred))

# ============================================================
# 6. Plot comparison figures
# ============================================================

import matplotlib.pyplot as plt

plot_df = comparison_df.copy()

plt.figure(figsize=(11, 6))
plt.bar(plot_df["Model"], plot_df["F1_Attack"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1-score")
plt.title("Classifier Comparison on Top-15 Features")
plt.tight_layout()
plt.savefig("outputs/classifier_comparison_f1_top15.png", dpi=300)
plt.show()

plt.figure(figsize=(11, 6))
plt.bar(plot_df["Model"], plot_df["FPR"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("False Positive Rate")
plt.title("False Positive Rate Comparison on Top-15 Features")
plt.tight_layout()
plt.savefig("outputs/classifier_comparison_fpr_top15.png", dpi=300)
plt.show()

print("Saved figures:")
print("- outputs/classifier_comparison_f1_top15.png")
print("- outputs/classifier_comparison_fpr_top15.png")

# ============================================================
# 7. Generate paper-ready table with percentages
# ============================================================

paper_table = full_plus_top15_df.copy()

paper_table["Accuracy (%)"] = paper_table["Accuracy"] * 100
paper_table["Precision (%)"] = paper_table["Precision_Attack"] * 100
paper_table["Recall (%)"] = paper_table["Recall_Attack"] * 100
paper_table["F1-score (%)"] = paper_table["F1_Attack"] * 100
paper_table["FPR (%)"] = paper_table["FPR"] * 100
paper_table["FNR (%)"] = paper_table["FNR"] * 100

paper_table = paper_table[[
    "Model",
    "Features",
    "Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1-score (%)",
    "Balanced_Accuracy",
    "MCC",
    "FPR (%)",
    "FNR (%)",
    "Training_Time_sec",
    "Inference_Time_sec",
    "Model_Size_MB"
]]

paper_table.to_csv("outputs/paper_ready_classifier_comparison.csv", index=False)

display(paper_table)

print("Saved: outputs/paper_ready_classifier_comparison.csv")

# ============================================================
# 8. Export LaTeX table for paper
# ============================================================

latex_ready = paper_table.copy()

# Round values for paper
for col in latex_ready.columns:
    if col != "Model":
        latex_ready[col] = pd.to_numeric(latex_ready[col], errors="ignore")

rounded = latex_ready.round({
    "Accuracy (%)": 2,
    "Precision (%)": 2,
    "Recall (%)": 2,
    "F1-score (%)": 2,
    "Balanced_Accuracy": 4,
    "MCC": 4,
    "FPR (%)": 4,
    "FNR (%)": 4,
    "Training_Time_sec": 3,
    "Inference_Time_sec": 4,
    "Model_Size_MB": 2
})

latex_table = rounded.to_latex(
    index=False,
    escape=True,
    caption="Comparative Performance of Machine Learning Classifiers on CICIoT2023",
    label="tab:classifier_comparison"
)

with open("outputs/classifier_comparison_table.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print(latex_table)
print("\nSaved: outputs/classifier_comparison_table.tex")