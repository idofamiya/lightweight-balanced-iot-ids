# IoT IDS Preprocessing Pipeline for CIC IoT Attack Dataset 2023
# Author: Prepared for IoT Security IDS research
# Purpose: Clean, optimize, sample, and prepare merged CICIoT2023 CSV files for ML experiments.

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# =========================
# 1. CONFIGURATION
# =========================

RAW_DIR = Path("data/raw")              # Put your 63 merged CSV files here
PROCESSED_DIR = Path("data/processed")  # Cleaned files will be saved here
OUTPUT_DIR = Path("outputs")            # Reports/encoders/scalers

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Label"
CHUNK_SIZE = 100_000

# For your laptop, start with a small sample.
# Later you can increase this to 500000 or 1000000.
FINAL_SAMPLE_SIZE = 200_000

RANDOM_STATE = 42


# =========================
# 2. ATTACK CATEGORY MAPPING
# =========================

def map_attack_category(label: str) -> str:
    """
    Map 34 fine-grained CICIoT2023 labels into broader attack categories.
    This is useful for multiclass category-level IDS.
    """
    label = str(label).upper().strip()

    if label == "BENIGN":
        return "BENIGN"
    elif label.startswith("DDOS"):
        return "DDOS"
    elif label.startswith("DOS"):
        return "DOS"
    elif label.startswith("MIRAI"):
        return "MIRAI"
    elif label.startswith("RECON"):
        return "RECON"
    elif "SPOOFING" in label or "MITM" in label:
        return "SPOOFING"
    elif label in ["SQLINJECTION", "COMMANDINJECTION", "XSS", "BROWSERHIJACKING", "UPLOADING_ATTACK", "BACKDOOR_MALWARE"]:
        return "WEB_OR_MALWARE"
    elif label == "DICTIONARYBRUTEFORCE":
        return "BRUTEFORCE"
    elif label == "VULNERABILITYSCAN":
        return "VULNERABILITY_SCAN"
    else:
        return "OTHER"


# =========================
# 3. MEMORY OPTIMIZATION
# =========================

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce RAM usage by converting float64 to float32 and int64 to smaller integer types.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


# =========================
# 4. CLEAN ONE CHUNK
# =========================

def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Clean one chunk of the dataset.
    """
    # Remove spaces from column names for easier coding
    chunk.columns = [c.strip().replace(" ", "_") for c in chunk.columns]

    target_col = TARGET_COL
    if TARGET_COL not in chunk.columns and "Label" in chunk.columns:
        target_col = "Label"

    # Replace infinity with NaN
    chunk = chunk.replace([np.inf, -np.inf], np.nan)

    # Fill numeric missing values with median of the chunk
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if chunk[col].isna().any():
            chunk[col] = chunk[col].fillna(chunk[col].median())

    # Drop rows without label
    if target_col in chunk.columns:
        chunk[target_col] = chunk[target_col].astype(str).str.upper().str.strip()
        chunk = chunk[chunk[target_col] != ""]
        chunk = chunk.dropna(subset=[target_col])

        # Create binary and category labels
        chunk["Binary_Label"] = chunk[target_col].apply(lambda x: 0 if x == "BENIGN" else 1)
        chunk["Attack_Category"] = chunk[target_col].apply(map_attack_category)

    # Remove duplicate rows inside the chunk
    chunk = chunk.drop_duplicates()

    # Optimize RAM
    chunk = optimize_dtypes(chunk)

    return chunk


# =========================
# 5. PREPROCESS ALL CSV FILES
# =========================

def preprocess_all_files():
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR.resolve()}")

    print(f"Found {len(csv_files)} CSV files.")

    processed_files = []

    for file_index, csv_file in enumerate(csv_files, start=1):
        print(f"\nProcessing file {file_index}/{len(csv_files)}: {csv_file.name}")

        output_file = PROCESSED_DIR / f"cleaned_{csv_file.stem}.parquet"

        cleaned_chunks = []
        total_rows = 0

        for chunk in pd.read_csv(csv_file, chunksize=CHUNK_SIZE):
            total_rows += len(chunk)
            cleaned = clean_chunk(chunk)
            cleaned_chunks.append(cleaned)

        file_df = pd.concat(cleaned_chunks, ignore_index=True)
        file_df = file_df.drop_duplicates()

        file_df.to_parquet(output_file, index=False)
        processed_files.append(output_file)

        print(f"Original rows read: {total_rows}")
        print(f"Cleaned rows saved: {len(file_df)}")
        print(f"Saved: {output_file}")

    return processed_files


# =========================
# 6. CREATE FINAL SAMPLE
# =========================

def create_final_sample(processed_files):
    """
    Create one balanced working sample from all cleaned files.
    This prevents your laptop from loading the full dataset at once.
    """
    print("\nCreating final research sample...")

    sample_parts = []

    per_file_sample = max(1000, FINAL_SAMPLE_SIZE // len(processed_files))

    for file in processed_files:
        df = pd.read_parquet(file)

        if len(df) > per_file_sample:
            df_sample = df.sample(n=per_file_sample, random_state=RANDOM_STATE)
        else:
            df_sample = df

        sample_parts.append(df_sample)

    sample_df = pd.concat(sample_parts, ignore_index=True)

    # Final sample limit
    if len(sample_df) > FINAL_SAMPLE_SIZE:
        sample_df = sample_df.sample(n=FINAL_SAMPLE_SIZE, random_state=RANDOM_STATE)

    sample_df = sample_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    output_sample = PROCESSED_DIR / f"final_sample_{FINAL_SAMPLE_SIZE}.csv"
    sample_df.to_csv(output_sample, index=False)

    print(f"Final sample shape: {sample_df.shape}")
    print(f"Saved final sample: {output_sample}")

    return sample_df, output_sample


# =========================
# 7. CREATE TRAIN-TEST DATA
# =========================

def create_train_test(sample_df: pd.DataFrame):
    """
    Prepare X_train, X_test, y_train, y_test for Binary IDS.
    """
    df = sample_df.copy()

    # Ensure column names are safe
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Columns not used as features
    drop_cols = ["Label", "Binary_Label", "Attack_Category"]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["Binary_Label"]

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Scale features for models that need scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save outputs
    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv(PROCESSED_DIR / "X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv(PROCESSED_DIR / "X_test_scaled.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train_binary.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test_binary.csv", index=False)

    joblib.dump(scaler, OUTPUT_DIR / "standard_scaler.joblib")

    print("\nTrain-test files saved.")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    return X_train, X_test, y_train, y_test


# =========================
# 8. DATASET REPORT
# =========================

def save_dataset_report(sample_df: pd.DataFrame):
    report_path = OUTPUT_DIR / "dataset_preprocessing_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CICIoT2023 Preprocessing Report\n")
        f.write("================================\n\n")
        f.write(f"Final sample shape: {sample_df.shape}\n\n")

        f.write("Columns:\n")
        for col in sample_df.columns:
            f.write(f"- {col}\n")

        f.write("\nFine-grained label distribution:\n")
        f.write(sample_df["Label"].value_counts().to_string())

        f.write("\n\nBinary label distribution:\n")
        f.write(sample_df["Binary_Label"].value_counts().to_string())

        f.write("\n\nAttack category distribution:\n")
        f.write(sample_df["Attack_Category"].value_counts().to_string())

        f.write("\n\nMissing values:\n")
        f.write(sample_df.isna().sum().to_string())

    print(f"Dataset report saved: {report_path}")


# =========================
# 9. MAIN EXECUTION
# =========================

if __name__ == "__main__":
    processed_files = preprocess_all_files()
    sample_df, sample_file = create_final_sample(processed_files)
    save_dataset_report(sample_df)
    create_train_test(sample_df)

    print("\nPreprocessing completed successfully.")
