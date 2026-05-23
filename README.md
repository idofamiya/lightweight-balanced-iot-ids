# Lightweight Balanced IoT IDS

This repository contains the source code, experimental workflow, result templates, and paper files for the research article:

**A Lightweight Balanced Random Forest Intrusion Detection System Using Statistical and Flow-Based Features for Resource-Constrained IoT Networks**

## Overview

This project evaluates a lightweight and class-balanced Random Forest Intrusion Detection System (IDS) using the CICIoT2023 dataset. The main experiment constructs a balanced binary dataset with 100,000 benign and 100,000 attack samples, applies Random Forest feature importance, and compares full-feature and reduced-feature models.

The final binary IDS result reported in the paper is:

- **RF Top-15 accuracy:** 98.85%
- **RF Top-15 F1-score:** 98.84%
- **False positives:** 17 out of 20,000 benign test samples
- **Approximate FPR:** 0.085%
- **Feature reduction:** 61.54% compared with the full 39-feature baseline

An extended multiclass attack-category analysis is also included.

## Dataset

The CICIoT2023 dataset is **not included** in this repository due to its large size and redistribution considerations.

Please download it from the official Canadian Institute for Cybersecurity, University of New Brunswick page:

https://www.unb.ca/cic/datasets/iotdataset-2023.html

After downloading, place the merged CSV files in:

```text
data/raw/
```

The preprocessing scripts expect the raw merged CICIoT2023 CSV files to be available in this folder.

## Repository Structure

```text
lightweight-balanced-iot-ids/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   ├── raw/               # Place raw CICIoT2023 merged CSV files here
│   ├── processed/          # Generated processed files; ignored by Git
│   └── sample/             # Optional tiny sample file for demonstration
├── src/
│   ├── preprocessing_balanced.py
│   ├── classifier_comparison.py
│   └── multiclass_attack_classification.py
├── notebooks/
│   └── experimental notebooks
├── results/
│   └── CSV result tables
├── figures/
│   └── preprocessing_workflow.png
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── IoT_IDS_final_journal_ready.pdf
└── outputs/
    └── generated output files
```

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version: **3.10 or 3.11**.

## Suggested Execution Order

1. Download CICIoT2023 merged CSV files and place them in `data/raw/`.
2. Run preprocessing:

```bash
python src/preprocessing_balanced.py
```

3. Run binary classifier comparison:

```bash
python src/classifier_comparison.py
```

4. Run multiclass attack-category analysis:

```bash
python src/multiclass_attack_classification.py
```

Alternatively, use the notebooks in the `notebooks/` folder.

## Main Experimental Design

### Binary IDS

- Dataset: CICIoT2023
- Files processed: 63 merged CSV files
- Final binary sample: 200,000 records
- Benign: 100,000
- Attack: 100,000
- Split: 80:20 stratified
- Classifier: Random Forest
- Feature configurations: Full-39, Top-5, Top-10, Top-15

### Multiclass Extension

Attack labels are grouped into broader categories:

- BENIGN
- DDOS
- DOS
- MIRAI
- SPOOFING
- VULNERABILITY_SCAN
- Other

A capped sampling strategy is used for extended multiclass evaluation.

## Reproducibility Notes

The scripts use a fixed random seed:

```python
RANDOM_STATE = 42
```

This helps reproduce train-test splits and sampling decisions. Exact results may vary slightly depending on package versions and hardware.

## Citation

If you use this repository, please cite the accompanying research article once published.

## License

This repository is provided for academic and research use. See `LICENSE`.

## Contact

Amiya Kumar Sahu  
Department of Computer Science and Applications  
Sambalpur University, Odisha, India
amiyakumar20072@gmail.com