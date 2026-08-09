# DNS Exfiltration Detection System

> A robust, machine-learning-powered pipeline for detecting DNS data exfiltration tunnels with high precision, built for resource-constrained network defenders.

## 📌 Problem Statement

Network defenders and administrators of small or resource-constrained networks need to know when DNS traffic is carrying an exfiltration tunnel instead of normal lookups. However, they cannot afford to drown in false alerts. Analysts triaging these alerts need a ranked, prioritized list that provides explicit reason codes rather than just a raw probability score. This enables them to quickly and confidently decide whether a domain represents a legitimate look-up or a true exfiltration attempt, avoiding alert fatigue.

## 🚀 Solution Overview

Our solution is an end-to-end detection pipeline leveraging an XGBoost classifier with strict data leakage guards and an interpretable output layer.

By analyzing stateless and stateful DNS features, our model significantly outperforms non-ML baseline heuristics, reducing the False Positive Rate (FPR) from 81% down to ~12%, while retaining near-perfect recall and generating human-readable "reason codes" for every alert.

### Key Features

- **Strict Leakage Controls:** Automated guards ensure no `session_id` or `registrable_domain` crosses train/val/test boundaries.
- **Interpretable Alerts:** Every flagged anomaly is attached to `top_reason_codes` (e.g. `FQDN_count`, `labels`) indicating exactly *why* the model fired.
- **Baseline Comparison:** Integrates a non-ML baseline (query length + entropy heuristics) to transparently prove the value of the ML model.
- **Metrics Module:** Comprehensive PR-AUC, precision, recall, and FPR reporting, split out by Light vs. Heavy attacks.
- **Fully Reproducible:** Environment lockfiles and a single-command CLI ensure results can be regenerated identically.

## ⚙️ Architecture & Pipeline

1. **Data Loader (`src/load.py`):** Parses the Kaggle-style folder layout, extracting stateful/stateless pairs and handling file-group-level stratified splitting.
2. **Leakage Guard (`src/leakage.py`):** Hard-stops the pipeline if data contamination is detected across partitions.
3. **Feature Prep (`src/features.py`):** Median imputation and standard scaling, fitted strictly on the training set to prevent leakage.
4. **Model (`src/model.py`):** XGBoost classifier with threshold selected by maximizing F1 on the validation PR curve.
5. **Metrics (`src/metrics.py`):** Generates reports including confusion matrices and false alerts per 10k benign rows.
6. **Demo / Interactive Mode (`demo.py`):** A front-end script for interactively testing and demonstrating the pipeline.

## 📊 Results Summary (Test Set)

| Model | PR-AUC | Precision | Recall | FPR |
| --- | --- | --- | --- | --- |
| **Baseline (Non-ML)** | 0.428 | 0.405 | 0.992 | 0.809 |
| **XGBoost Model** | 0.810 | 0.823 | 0.997 | **0.119** |

*Note: The model achieves a large, real improvement over the baseline on FPR (81% → ~12%) and on heavy-attack PR-AUC (0.41 → 0.79).*

## 🛠 How to Run

### 1. Install Dependencies

Make sure you have Python 3.10+ installed. Install the exact locked versions:

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

To train the model, evaluate against the baseline, and generate predictions and metrics:

```bash
python pipeline.py --dataset-dir data
```

*(Results, including metrics and predictions, will be saved to the `results/` folder).*

### 3. Run the Test Suite

To verify the integrity of the pipeline (including leakage guards and data transformations):

```bash
python -m pytest tests/ -q
```

## 📁 Repository Structure

```
├── data/                  # Dataset folder (not tracked)
├── src/                   # Core pipeline source code
│   ├── load.py            # Data loading & splitting
│   ├── features.py        # Imputation & scaling
│   ├── leakage.py         # Partition leakage checks
│   ├── model.py           # XGBoost training & reason codes
│   └── metrics.py         # Evaluation & reporting
├── tests/                 # Pytest test suite (16 tests)
├── reports/               # Technical reports & data statements
├── results/               # Auto-generated metrics & predictions
├── pipeline.py            # Main entry point
├── demo.py                # Interactive demo entry point
└── requirements.txt       # Pinned dependencies
```

# Deployment Guide — DNS Exfiltration Detector (Dash frontend)

This guide covers running the Dash app (`dash_app.py`) locally and deploying it
to a free hosting tier. Verified against `dash==4.4.1`, `gunicorn==26.0.0`.

## 1. Prerequisites

- `results/inference_bundle.pkl`, `results/predictions.csv`, `results/scoring_output.json`
  must exist before the app starts — it loads them at startup, not lazily.
  Generate them with:

```bash
  python pipeline.py --dataset-dir data
```

- Files needed in the repo: `dash_app.py`, `src/score_new.py`, `src/model.py`,
  `src/features.py` (with the `return_transformers` parameter), `requirements.txt`, `Procfile`.

## 2. Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python pipeline.py --dataset-dir data
python dash_app.py
```

Open **<http://127.0.0.1:8050>**.

To test it the same way a production host will run it (via gunicorn, not the dev server):

```bash
gunicorn dash_app:server --bind 0.0.0.0:8050
```

## 3. requirements.txt / Procfile 

`requirements.txt` should include, in addition to the ML dependencies:

---
*Developed for the Canadian Institute for Cybersecurity (CIC) DNS Exfiltration Challenge.*
