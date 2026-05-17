# ============================================================
# ANNITIA CHALLENGE — TRAINING SCRIPT
# Survival Analysis: Hepatic Events + Death Prediction
# Models: CoxNet, Random Survival Forest, DeepSurv (ensemble)
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import copy
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split

from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# ============================================================
# 0. CONFIG
# ============================================================
DATA_DIR    = "/kaggle/input/datasets/qamarmath/annitia-data-challenge"
OUTPUT_DIR  = "./models"
TRAIN_FILE  = os.path.join(DATA_DIR, "trainANNITIA.csv")
SEED        = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed set to: {seed}")

set_seed(SEED)

# ============================================================
# 2. LONGITUDINAL FEATURE ENGINEERING
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

def extract_longitudinal_features(df):
    """
    For each NIT measurement series (v1..v22), compute:
    last, first, max, min, mean, std, slope, n_obs, absolute change.
    """
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
            n_obs[i]    = len(obs_v)
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


# ============================================================
# 3. STATIC FEATURE BUILDER
# ============================================================
STATIC_COLS = [
    'gender', 'T2DM', 'Hypertension', 'Dyslipidaemia',
    'bariatric_surgery', 'bariatric_surgery_age', 'Age_v1'
]

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
# 4. SURVIVAL TARGET BUILDER
# ============================================================
def build_survival_target(df, event_col, age_occur_col,
                           age_baseline_col='Age_v1'):
    age_cols = sorted(
        [c for c in df.columns if c.startswith('Age_v')],
        key=lambda x: int(x.split('_v')[1])
    )
    last_age     = df[age_cols].max(axis=1)
    baseline     = df[age_baseline_col]
    valid_mask   = df[event_col].notna()
    df_v         = df[valid_mask].copy()
    last_v       = last_age[valid_mask]
    base_v       = baseline[valid_mask]

    events = df_v[event_col].astype(bool).values
    times  = np.where(
        events & df_v[age_occur_col].notna(),
        df_v[age_occur_col].values - base_v.values,
        last_v.values - base_v.values
    )
    times = np.maximum(times, 0.01)

    y = Surv.from_arrays(event=events, time=times)
    return y, valid_mask.values


# ============================================================
# 5. DeepSurv MODEL
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


def cox_loss(risk_pred, event, time):
    idx          = torch.argsort(time, descending=True)
    risk_pred    = risk_pred[idx]
    event        = event[idx]
    max_risk     = torch.max(risk_pred)
    exp_risk     = torch.exp(risk_pred - max_risk)
    log_cum      = torch.log(torch.cumsum(exp_risk, dim=0) + 1e-8) + max_risk
    return -torch.sum(event * (risk_pred - log_cum)) / (event.sum() + 1e-8)


def train_deepsurv(X_scaled, y, label="DeepSurv", epochs=400, lr=5e-4):
    """Train DeepSurv and return best model + val C-index."""
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED,
        stratify=y['event']
    )
    x_tr_t = torch.from_numpy(X_tr.astype(np.float32))
    t_tr_t = torch.from_numpy(y_tr['time'].astype(np.float32).copy())
    e_tr_t = torch.from_numpy(y_tr['event'].astype(np.float32).copy())
    x_va_t = torch.from_numpy(X_va.astype(np.float32))
    t_va_t = torch.from_numpy(y_va['time'].astype(np.float32).copy())
    e_va_t = torch.from_numpy(y_va['event'].astype(np.float32).copy())

    model     = DeepSurv(x_tr_t.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=0.1
    )

    best_ci_va      = 0.0
    best_model_wts  = copy.deepcopy(model.state_dict())

    print(f"\n  {label} training:")
    print(f"  {'Epoch':<8} | {'Tr-Loss':<10} | {'Va-C-Idx':<10}")
    print(f"  {'-'*35}")

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        risk_tr = model(x_tr_t).squeeze()
        loss    = cox_loss(risk_tr, e_tr_t, t_tr_t)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                risk_va = model(x_va_t).squeeze()
                ci_va   = concordance_index_censored(
                    y_va['event'], y_va['time'], risk_va.numpy()
                )[0]
                if ci_va > best_ci_va:
                    best_ci_va     = ci_va
                    best_model_wts = copy.deepcopy(model.state_dict())
                print(f"  {epoch:<8} | {loss.item():<10.4f} | {ci_va:<10.4f}")

    model.load_state_dict(best_model_wts)
    print(f"  Best Val C-Index: {best_ci_va:.4f}")
    return model


# ============================================================
# 6. FULL ENSEMBLE TRAINER
# ============================================================
def rank_normalize(arr):
    from scipy.stats import rankdata
    return rankdata(arr) / len(arr)


def train_full_ensemble(X, y, label="Ensemble"):
    print(f"\n{'='*70}")
    print(f"  TRAINING: {label}")
    print(f"  Patients: {len(y)} | Events: {y['event'].sum()}")
    print(f"{'='*70}")

    # Preprocess
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()
    X_proc  = scaler.fit_transform(imputer.fit_transform(X)).astype(np.float32)

    # ---- CoxNet ----
    print("\n  [1/3] CoxNet (Elastic-Net Cox Regression)...")
    cox = CoxnetSurvivalAnalysis(
        l1_ratio=0.5, fit_baseline_model=True,
        alpha_min_ratio=0.1, max_iter=1000, normalize=False
    )
    cox.fit(X_proc, y)

    # Cross-validate CoxNet
    kf      = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cox_cis = []
    for tr_idx, va_idx in kf.split(X_proc):
        if y[va_idx]['event'].sum() == 0:
            continue
        try:
            m = CoxnetSurvivalAnalysis(
                l1_ratio=0.5, fit_baseline_model=True,
                alpha_min_ratio=0.1, max_iter=1000, normalize=False
            )
            m.fit(X_proc[tr_idx], y[tr_idx])
            ci = concordance_index_censored(
                y[va_idx]['event'], y[va_idx]['time'],
                m.predict(X_proc[va_idx])
            )[0]
            cox_cis.append(ci)
        except Exception:
            pass
    if cox_cis:
        print(f"  CoxNet 5-Fold C-Index: {np.mean(cox_cis):.4f} ± {np.std(cox_cis):.4f}")

    # ---- Random Survival Forest ----
    print("\n  [2/3] Random Survival Forest...")
    rsf = RandomSurvivalForest(
        n_estimators=3000, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt',
        n_jobs=-1, random_state=SEED
    )
    rsf.fit(X_proc, y)
    rsf_cis = []
    for tr_idx, va_idx in kf.split(X_proc):
        if y[va_idx]['event'].sum() == 0:
            continue
        try:
            m = RandomSurvivalForest(
                n_estimators=200, min_samples_split=10,
                min_samples_leaf=5, max_features='sqrt',
                n_jobs=-1, random_state=SEED
            )
            m.fit(X_proc[tr_idx], y[tr_idx])
            ci = concordance_index_censored(
                y[va_idx]['event'], y[va_idx]['time'],
                m.predict(X_proc[va_idx])
            )[0]
            rsf_cis.append(ci)
        except Exception:
            pass
    if rsf_cis:
        print(f"  RSF 5-Fold C-Index:    {np.mean(rsf_cis):.4f} ± {np.std(rsf_cis):.4f}")

    # ---- DeepSurv ----
    print("\n  [3/3] DeepSurv (Neural Cox Model)...")
    ds_model = train_deepsurv(X_proc, y, label=label)

    return {
        'imputer':   imputer,
        'scaler':    scaler,
        'cox':       cox,
        'rsf':       rsf,
        'deepsurv':  ds_model,
    }


# ============================================================
# 7. MAIN: LOAD → FEATURE ENGINEER → TRAIN → SAVE
# ============================================================
if __name__ == "__main__":
    # --- Load ---
    print("Loading data...")
    train = pd.read_csv(TRAIN_FILE)
    print(f"  Train shape: {train.shape}")

    # --- Features ---
    print("\nEngineering features...")
    X_all = build_feature_matrix(train)
    print(f"  Feature matrix: {X_all.shape}")

    # --- Survival targets ---
    print("\nBuilding survival targets...")
    y_hep,   mask_hep   = build_survival_target(
        train, 'evenements_hepatiques_majeurs', 'evenements_hepatiques_age_occur'
    )
    y_death, mask_death = build_survival_target(
        train, 'death', 'death_age_occur'
    )
    X_hep   = X_all[mask_hep].reset_index(drop=True)
    X_death = X_all[mask_death].reset_index(drop=True)

    print(f"  Hepatic — {len(y_hep)} patients, {y_hep['event'].sum()} events")
    print(f"  Death   — {len(y_death)} patients, {y_death['event'].sum()} events")

    # --- Train ---
    bundle_hep   = train_full_ensemble(X_hep,   y_hep,   label="HEPATIC EVENTS")
    bundle_death = train_full_ensemble(X_death, y_death, label="DEATH")

    # --- Save ---
    print("\nSaving model bundles...")
    with open(os.path.join(OUTPUT_DIR, "bundle_hepatic.pkl"), "wb") as f:
        pickle.dump(bundle_hep, f)
    with open(os.path.join(OUTPUT_DIR, "bundle_death.pkl"), "wb") as f:
        pickle.dump(bundle_death, f)

    # Save feature column order for inference alignment
    feature_cols = list(X_all.columns)
    with open(os.path.join(OUTPUT_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    print(f"\nModels saved to '{OUTPUT_DIR}/'")
    print("  - bundle_hepatic.pkl")
    print("  - bundle_death.pkl")
    print("  - feature_cols.pkl")
    print("\nTraining complete.")
