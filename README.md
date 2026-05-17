# ANNITIA Data Challenge — Survival Analysis Solution

## Overview

This repository contains a complete **survival analysis pipeline** for the **ANNITIA Data Challenge**, which aims to predict two clinical outcomes in patients with metabolic-associated liver disease:

1. **Major Hepatic Events** (`evenements_hepatiques_majeurs`) — e.g. cirrhosis decompensation, hepatocellular carcinoma, liver transplant.
2. **All-cause Death** (`death`).

The task is framed as a **time-to-event (survival analysis)** problem. The evaluation metric is the **Concordance Index (C-index)**, which measures how well the model ranks patients by risk. A C-index of 1.0 is perfect; 0.5 is random.

---

## Dataset Description

| File | Rows | Columns | Description |
|---|---|---|---|
| `trainANNITIA.csv` | 1,253 | 287 | Training set with event labels |
| `test ANNITIA.csv` | 423 | 284 | Test set — no event labels |

### Key Column Groups

**Demographics & Comorbidities (static)**

| Column | Description |
|---|---|
| `patient_id_anon` | Anonymised patient ID |
| `gender` | 1 = Male, 2 = Female |
| `T2DM` | Type 2 Diabetes Mellitus (0/1) |
| `Hypertension` | Hypertension flag (0/1) |
| `Dyslipidaemia` | Dyslipidaemia flag (0/1) |
| `bariatric_surgery` | Bariatric surgery performed (0/1/2) |
| `bariatric_surgery_age` | Age at surgery |
| `Age_v1` … `Age_v22` | Patient age at each clinical visit |

**Non-Invasive Tests (NITs) — Longitudinal (v1..v22 per visit)**

| Prefix | Measurement |
|---|---|
| `BMI_v*` | Body Mass Index |
| `ALT_v*` | Alanine Aminotransferase (liver enzyme) |
| `AST_v*` | Aspartate Aminotransferase (liver enzyme) |
| `GGT_v*` | Gamma-Glutamyl Transferase |
| `bilirubin_v*` | Total Bilirubin |
| `platelets_v*` | Platelet count |
| `glycemia_v*` | Fasting blood glucose |
| `triglycerides_v*` | Triglycerides |
| `total_cholesterol_v*` | Total Cholesterol |
| `FibroTest_v*` | Commercial fibrosis score |
| `fibs_stiffness_med_BM_1_v*` | FibroScan liver stiffness (kPa) |
| `fibs_CAP_BM_1_v*` | FibroScan CAP (steatosis dB/m) |
| `SSI_v*` | Shear-wave stiffness index |

**Target Columns (training only)**

| Column | Description |
|---|---|
| `evenements_hepatiques_majeurs` | Binary: 1 = major hepatic event occurred |
| `evenements_hepatiques_age_occur` | Age at hepatic event (NaN if no event) |
| `death` | Binary: 1 = patient died during follow-up |
| `death_age_occur` | Age at death (NaN if alive/censored) |

> **Censoring**: Patients without an event are right-censored — we know they were event-free up to their last visit but don't know what happened after.

---

## Solution Architecture

### Feature Engineering

Because ANNITIA is fundamentally a **longitudinal study**, the most important step is converting repeated measurements into informative summary statistics per patient.

For each NIT prefix, 9 features are computed across all available visits:

| Feature | Description |
|---|---|
| `*_last` | Most recent observed value |
| `*_first` | Baseline (first visit) value |
| `*_max` | Maximum value across visits |
| `*_min` | Minimum value across visits |
| `*_mean` | Mean across all visits |
| `*_std` | Standard deviation across visits |
| `*_slope` | Linear slope over time (units/year) |
| `*_nobs` | Number of valid observations |
| `*_change` | Absolute change (last − first) |

Static features added: `gender`, `T2DM`, `Hypertension`, `Dyslipidaemia`, `bariatric_surgery`, `bariatric_surgery_age`, `Age_v1`, and derived `follow_up_yrs`.

**Total features: ~35**

### Models

Three complementary survival models are trained for each outcome:

#### 1. CoxNet (Elastic-Net Penalized Cox Regression)
- From `scikit-survival`: `CoxnetSurvivalAnalysis`
- L1 + L2 regularization (`l1_ratio=0.5`) handles collinear features
- Appropriate for high-dimensional, sparse longitudinal summaries
- 5-fold cross-validated C-index reported during training

#### 2. Random Survival Forest (RSF)
- From `scikit-survival`: `RandomSurvivalForest`
- 3,000 trees; captures non-linear interactions
- Robust to extreme class imbalance (47 hepatic events / 1,253 patients)
- 5-fold cross-validated C-index reported

#### 3. DeepSurv (Neural Cox Model)
- Custom PyTorch implementation
- Architecture: Linear(35→32) → BatchNorm → ReLU → Dropout(0.4) → Linear(32→16) → ReLU → Linear(16→1)
- Cox partial likelihood loss (negative log partial likelihood)
- Adam optimizer, `lr=5e-4`, `weight_decay=0.1`
- Best model selected by validation C-index (80/20 stratified split)

### Ensemble

Final risk scores are produced by **rank-normalizing** each model's raw scores to [0, 1] and averaging:

```
risk_final = (rank_norm(CoxNet) + rank_norm(RSF) + rank_norm(DeepSurv)) / 3
```

Rank normalization prevents any single model from dominating due to scale differences.

---

## Validation Results

| Model | Hepatic C-Index (5-fold) | Death C-Index (5-fold) |
|---|---|---|
| CoxNet | 0.7834 ± 0.117 | 0.9677 ± 0.017 |
| RSF | 0.7801 ± 0.114 | 0.9313 ± 0.022 |
| DeepSurv (val) | 0.9416 | 0.9503 |

> **Note on hepatic events**: Only 47 events out of 1,253 patients (~3.7% event rate) makes this a hard imbalanced problem. High C-index variance across folds is expected.

---

## Repository Structure

```
annitia/
├── train_annitia.py        # Full training pipeline — saves model bundles
├── predict_annitia.py      # Inference script — loads bundles, writes submission.csv
├── models/                 # Created by training script
│   ├── bundle_hepatic.pkl  # Imputer + Scaler + CoxNet + RSF + DeepSurv (hepatic)
│   ├── bundle_death.pkl    # Imputer + Scaler + CoxNet + RSF + DeepSurv (death)
│   └── feature_cols.pkl    # Ordered feature column list for alignment
└── README.md
```

---

## Setup & Installation

### Requirements

```
Python 3.10+
pandas >= 2.2
numpy >= 2.0
scikit-learn >= 1.8
scikit-survival >= 0.27
torch >= 2.0
scipy
```

### Install

```bash
pip install scikit-survival torch pandas numpy scipy
```

On Kaggle, scikit-survival needs explicit installation:

```bash
pip install scikit-survival
```

---

## Usage

### Step 1 — Train

```bash
python train_annitia.py
```

This will:
- Load `trainANNITIA.csv`
- Engineer longitudinal + static features
- Build survival targets for both outcomes
- Train CoxNet, RSF, and DeepSurv for each outcome
- Print 5-fold C-index and DeepSurv validation diagnostics
- Save model bundles to `./models/`

**Expected training time:** ~10–20 minutes (RSF with 3,000 trees is the bottleneck)

### Step 2 — Predict

```bash
python predict_annitia.py
```

This will:
- Load `test ANNITIA.csv`
- Apply identical feature engineering
- Load all six saved models
- Generate rank-averaged ensemble predictions
- Save `submission.csv` with columns: `trustii_id`, `risk_hepatic_event`, `risk_death`

---

## Submission Format

```csv
trustii_id,risk_hepatic_event,risk_death
1,0.123,0.456
2,0.789,0.234
...
```

- `trustii_id`: Integer patient ID from the test set
- `risk_hepatic_event`: Continuous risk score (higher = higher risk)
- `risk_death`: Continuous risk score (higher = higher risk)

The competition evaluates submissions using **C-index** independently for each outcome. The final score is typically the mean of both C-indices.

---

## Design Decisions

### Why survival analysis instead of binary classification?

Binary classification (0/1 labels) ignores **when** the event happens and throws away censoring information. A patient censored at age 70 (no event yet) carries real signal that a classifier would misuse as a negative label. Cox-based and tree-based survival models correctly handle this.

### Why rank-normalize before ensembling?

CoxNet produces log-hazard ratios (unbounded), RSF produces mean survival times (strictly positive), and DeepSurv produces clamped log-hazard scores. Direct averaging of these on different scales is misleading. Rank normalization maps each to [0, 1] before combining.

### Why DeepSurv with Dropout + weight_decay?

With only 35 features and ~1,000 training patients (and as few as 47 events), a deep network can memorize quickly. Dropout (0.4) + L2 weight decay (0.1) + early stopping by validation C-index keep generalization in check. The diagnostic table printed during training shows the training/validation C-index gap — a gap > 0.15 signals overfitting.

### Why 3,000 trees in RSF?

With rare events (~3.7% event rate for hepatic), each tree sees very few positive labels. A larger forest averages out variance from individual trees and produces more stable risk rankings. The CV C-index typically stabilizes beyond 500 trees; 3,000 is conservative but safe.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: sksurv` | Run `pip install scikit-survival` |
| `CUDA out of memory` | DeepSurv runs on CPU by default; no GPU needed |
| `ValueError: No events in fold` | Normal with rare outcomes; that fold is skipped |
| Shape mismatch at prediction | Ensure `feature_cols.pkl` from same training run is used |
| Negative follow-up times | Check `Age_v1` is populated; the script clips to 0.01 min |

---

## Citation / Acknowledgements

- **ANNITIA Consortium** — for providing the longitudinal NIT dataset
- `scikit-survival` — Pölsterl S. (2020), *scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn*, JMLR
- DeepSurv architecture inspired by Katzman et al. (2018), *DeepSurv: Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network*
