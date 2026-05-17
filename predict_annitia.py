# ============================================================
# ANNITIA CHALLENGE — PREDICTION / INFERENCE SCRIPT
# Loads saved model bundles and generates submission CSV
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata

# ============================================================
# 0. CONFIG  (update paths if needed)
# ============================================================
DATA_DIR   = "/kaggle/input/datasets/qamarmath/annitia-data-challenge"
MODEL_DIR  = "./models"
TEST_FILE  = os.path.join(DATA_DIR, "test ANNITIA.csv")
OUTPUT_CSV = "submission.csv"

# ============================================================
# 1. DeepSurv architecture  (must match train_annitia.py)
# ============================================================
class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return torch.clamp(self.network(x), min=-10, max=10)


# ============================================================
# 2. FEATURE ENGINEERING  (identical to training)
# ============================================================
NIT_PREFIXES = [
    'bmi',
    'ALT', 'AST', 'GGT', 'bilirubin', 'platelets',
    'glycemia', 'triglycerides', 'total_cholesterol',
    'FibroTest',
    'fibs_stiffness_med_BM_1',
    'fibs_CAP_BM_1',
    'SSI',
]

STATIC_COLS = [
    'gender', 'T2DM', 'Hypertension', 'Dyslipidaemia',
    'bariatric_surgery', 'bariatric_surgery_age', 'Age_v1'
]


def extract_longitudinal_features(df):
    age_cols = sorted(
        [c for c in df.columns if c.startswith('Age_v')],
        key=lambda x: int(x.split('_v')[1])
    )
    features = {}
    for prefix in NIT_PREFIXES:
        visit_cols = sorted(
            [c for c in df.columns
             if c.lower().startswith(prefix.lower() + '_v')],
            key=lambda x: int(x.split('_v')[1])
        )
        if not visit_cols:
            continue

        vals = df[visit_cols].values.astype(float)
        ages = df[age_cols[:len(visit_cols)]].values.astype(float)
        n    = vals.shape[0]

        last_val  = np.full(n, np.nan)
        first_val = np.full(n, np.nan)
        max_val   = np.full(n, np.nan)
        min_val   = np.full(n, np.nan)
        mean_val  = np.full(n, np.nan)
        std_val   = np.full(n, np.nan)
        slope     = np.full(n, np.nan)
        n_obs     = np.zeros(n)

        for i in range(n):
            v, a = vals[i], ages[i]
            mask = ~np.isnan(v) & ~np.isnan(a)
            obs_v, obs_a = v[mask], a[mask]
            if len(obs_v) == 0:
                continue
            n_obs[i]     = len(obs_v)
            last_val[i]  = obs_v[-1]
            first_val[i] = obs_v[0]
            max_val[i]   = obs_v.max()
            min_val[i]   = obs_v.min()
            mean_val[i]  = obs_v.mean()
            std_val[i]   = obs_v.std() if len(obs_v) > 1 else 0.0
            if len(obs_v) >= 2:
                A = np.vstack([obs_a, np.ones(len(obs_a))]).T
                slope[i] = np.linalg.lstsq(A, obs_v, rcond=None)[0][0]

        p = (prefix
             .replace('fibs_stiffness_med_BM_1', 'fibroscan')
             .replace('fibs_CAP_BM_1', 'cap')
             .replace('total_cholesterol', 'chol')
             .lower())

        features[f'{p}_last']   = last_val
        features[f'{p}_first']  = first_val
        features[f'{p}_max']    = max_val
        features[f'{p}_min']    = min_val
        features[f'{p}_mean']   = mean_val
        features[f'{p}_std']    = std_val
        features[f'{p}_slope']  = slope
        features[f'{p}_nobs']   = n_obs
        features[f'{p}_change'] = last_val - first_val

    return pd.DataFrame(features, index=df.index)


def build_static_features(df):
    cols = [c for c in STATIC_COLS if c in df.columns]
    out  = df[cols].copy()
    age_cols = sorted(
        [c for c in df.columns if c.startswith('Age_v')],
        key=lambda x: int(x.split('_v')[1])
    )
    out['follow_up_yrs'] = df[age_cols].max(axis=1) - df['Age_v1']
    return out


def build_feature_matrix(df):
    static = build_static_features(df)
    long_  = extract_longitudinal_features(df)
    return pd.concat(
        [static.reset_index(drop=True), long_.reset_index(drop=True)],
        axis=1
    )


# ============================================================
# 3. PREDICTION HELPERS
# ============================================================
def rank_normalize(arr):
    return rankdata(arr) / len(arr)


def predict_ensemble(X_raw, bundle, feature_cols):
    """
    Run all three models (CoxNet, RSF, DeepSurv) and return
    rank-averaged ensemble risk scores.
    """
    # Align columns to training order
    X_aligned = X_raw.reindex(columns=feature_cols, fill_value=np.nan)

    imputer = bundle['imputer']
    scaler  = bundle['scaler']
    X_proc  = scaler.transform(imputer.transform(X_aligned)).astype(np.float32)

    # --- CoxNet ---
    risk_cox = bundle['cox'].predict(X_proc)

    # --- RSF ---
    risk_rsf = bundle['rsf'].predict(X_proc)

    # --- DeepSurv ---
    ds_model = bundle['deepsurv']
    ds_model.eval()
    with torch.no_grad():
        x_t      = torch.from_numpy(X_proc)
        risk_ds  = ds_model(x_t).squeeze().numpy()

    # Rank-normalize and average (handles scale differences)
    ensemble = (
        rank_normalize(risk_cox) +
        rank_normalize(risk_rsf) +
        rank_normalize(risk_ds)
    ) / 3.0

    return ensemble


# ============================================================
# 4. MAIN
# ============================================================
if __name__ == "__main__":
    # --- Load test data ---
    print("Loading test data...")
    test = pd.read_csv(TEST_FILE)
    print(f"  Test shape: {test.shape}")

    # --- Feature engineering ---
    print("\nEngineering features...")
    X_test = build_feature_matrix(test)
    print(f"  Feature matrix: {X_test.shape}")

    # --- Load models ---
    print("\nLoading model bundles...")
    with open(os.path.join(MODEL_DIR, "bundle_hepatic.pkl"), "rb") as f:
        bundle_hep = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "bundle_death.pkl"), "rb") as f:
        bundle_death = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    print("  Bundles loaded.")

    # --- Predict ---
    print("\nGenerating risk scores...")
    risk_hepatic = predict_ensemble(X_test, bundle_hep,   feature_cols)
    risk_death   = predict_ensemble(X_test, bundle_death, feature_cols)

    print(f"  Hepatic risk — min: {risk_hepatic.min():.4f}, max: {risk_hepatic.max():.4f}")
    print(f"  Death risk   — min: {risk_death.min():.4f},   max: {risk_death.max():.4f}")

    # --- Build submission ---
    submission = pd.DataFrame({
        'trustii_id':          test['trustii_id'].values,
        'risk_hepatic_event':  risk_hepatic,
        'risk_death':          risk_death,
    })

    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSubmission saved to '{OUTPUT_CSV}'")
    print(submission.head(10).to_string(index=False))
