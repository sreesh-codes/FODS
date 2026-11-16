"""
11_predictive.py - Predictive Models for PEAD Trading

Classification task: Predict CAR[+2,+5] up/down (PRIMARY_TARGET)
Walk-forward validation: Expanding window, 2018Q1-2022Q4

Part 1: Baselines + Logistic Regression (STOP HERE for review)
Part 2: Tree models (LightGBM, RF, XGBoost) - run after review

Ground rules enforced:
- Walk-forward validation with 20-day embargo
- No look-ahead bias (all features pre-event)
- Track ALL model variants in model_registry.csv
- Leakage test on shuffled labels
- STOP if AUC > 0.60 (check for leakage)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, roc_curve
from sklearn.preprocessing import StandardScaler
import config
from pathlib import Path
import warnings
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

np.random.seed(config.RANDOM_SEED)

print("=" * 80)
print("11_PREDICTIVE.PY - PART 1: BASELINES + LOGISTIC REGRESSION")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA AND SETUP WALK-FORWARD
# ============================================================================

print("\n[1/7] Loading data...")
panel = pd.read_parquet(config.PROCESSED_DIR / "model_panel.parquet")

# Use train set only (2015-2022)
train = panel[panel['split'] == 'train'].copy()
print(f"  Train events: {len(train):,} (2015-2022)")

# Filter to events with required data
required_cols = config.ALL_FEATURES + [config.PRIMARY_TARGET, 't0', 'ticker', 'year', 'quarter']

train_clean = train.dropna(subset=required_cols).copy()
print(f"  After removing NA: {len(train_clean):,} ({100*len(train_clean)/len(train):.1f}%)")

# Convert t0 to datetime
train_clean['t0'] = pd.to_datetime(train_clean['t0'])

# Define walk-forward splits
print(f"\n[2/7] Setting up walk-forward validation...")
print(f"  Walk-forward: {config.WF_START_YEAR}Q1 to {config.WF_END_YEAR}Q4")
print(f"  Embargo: {config.WF_EMBARGO_DAYS} trading days")

# Generate quarterly test periods
quarters = []
for year in range(config.WF_START_YEAR, config.WF_END_YEAR + 1):
    for q in [1, 2, 3, 4]:
        quarters.append((year, q))

print(f"  Total quarters to evaluate: {len(quarters)}")

# Helper function to create walk-forward split
def get_walk_forward_split(data, test_year, test_quarter, embargo_days=20):
    """
    Create train/test split for walk-forward validation.

    Train: All events before test quarter (with embargo)
    Test: Events in test quarter
    """
    # Define test period
    test_q_str = f"{test_year}Q{test_quarter}"
    test_mask = (data['year'] == test_year) & (data['quarter'] == test_q_str)

    # Define training period (all before test, with embargo)
    if len(data[test_mask]) > 0:
        first_test_t0 = data[test_mask]['t0'].min()
        last_train_t0 = first_test_t0 - timedelta(days=embargo_days)

        train_mask = data['t0'] <= last_train_t0
    else:
        # No test events this quarter
        train_mask = data['year'] < test_year

    train_idx = data[train_mask].index
    test_idx = data[test_mask].index

    return train_idx, test_idx

# ============================================================================
# 2. BASELINE MODELS
# ============================================================================

print(f"\n[3/7] Running baseline models...")

results_list = []

# Baseline 1: Base rate (always predict majority class)
def baseline_base_rate(train_df, test_df, target_col):
    """Always predict the training set base rate."""
    base_rate = train_df[target_col].mean()

    # Predict constant probability
    y_pred_proba = np.full(len(test_df), base_rate)
    y_true = test_df[target_col].values

    # Metrics
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = np.nan

    ll = log_loss(y_true, y_pred_proba)
    bs = brier_score_loss(y_true, y_pred_proba)

    # Rank IC (correlation between predicted and actual)
    rank_ic = np.corrcoef(y_pred_proba, y_true)[0, 1] if len(y_true) > 1 else np.nan

    return {
        'auc': auc,
        'log_loss': ll,
        'brier_score': bs,
        'rank_ic': rank_ic
    }

# Baseline 2: SUE quintile rule (Q5=1, Q1=0, else=base_rate)
def baseline_sue_quintile(train_df, test_df, target_col):
    """Predict based on SUE quintile."""
    base_rate = train_df[target_col].mean()

    # Compute quintile probabilities from train set
    quintile_probs = train_df.groupby('sue_quintile')[target_col].mean().to_dict()

    # Predict using quintile mapping
    y_pred_proba = test_df['sue_quintile'].map(quintile_probs).fillna(base_rate).values
    y_true = test_df[target_col].values

    # Metrics
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = np.nan

    ll = log_loss(y_true, y_pred_proba)
    bs = brier_score_loss(y_true, y_pred_proba)
    rank_ic = np.corrcoef(y_pred_proba, y_true)[0, 1] if len(y_true) > 1 else np.nan

    return {
        'auc': auc,
        'log_loss': ll,
        'brier_score': bs,
        'rank_ic': rank_ic
    }

# Run baselines across all quarters
baseline_results = {
    'base_rate': [],
    'sue_quintile': []
}

for year, quarter in quarters:
    train_idx, test_idx = get_walk_forward_split(train_clean, year, quarter,
                                                   config.WF_EMBARGO_DAYS)

    if len(test_idx) < 10:
        continue

    train_fold = train_clean.loc[train_idx]
    test_fold = train_clean.loc[test_idx]

    # Baseline 1
    res1 = baseline_base_rate(train_fold, test_fold, config.PRIMARY_TARGET)
    res1['year'] = year
    res1['quarter'] = quarter
    baseline_results['base_rate'].append(res1)

    # Baseline 2 (only if sue_quintile exists)
    if 'sue_quintile' in test_fold.columns and test_fold['sue_quintile'].notna().sum() > 0:
        res2 = baseline_sue_quintile(train_fold, test_fold, config.PRIMARY_TARGET)
        res2['year'] = year
        res2['quarter'] = quarter
        baseline_results['sue_quintile'].append(res2)

# Aggregate baseline results
for baseline_name, results in baseline_results.items():
    if len(results) > 0:
        df_res = pd.DataFrame(results)

        # Save to registry
        results_list.append({
            'timestamp': datetime.now().isoformat(),
            'model_type': 'baseline',
            'model_name': baseline_name,
            'features': 'none' if baseline_name == 'base_rate' else 'sue_quintile',
            'n_features': 0 if baseline_name == 'base_rate' else 1,
            'target': config.PRIMARY_TARGET,
            'seed': config.RANDOM_SEED,
            'auc_mean': df_res['auc'].mean(),
            'auc_std': df_res['auc'].std(),
            'log_loss_mean': df_res['log_loss'].mean(),
            'brier_score_mean': df_res['brier_score'].mean(),
            'rank_ic_mean': df_res['rank_ic'].mean(),
            'n_quarters': len(df_res),
            'params': {}
        })

        print(f"  {baseline_name}: AUC={df_res['auc'].mean():.4f} ± {df_res['auc'].std():.4f} ({len(df_res)} quarters)")

# ============================================================================
# 3. LOGISTIC REGRESSION MODELS
# ============================================================================

print(f"\n[4/7] Training logistic regression models...")

# Helper function for logistic regression with walk-forward
def train_logistic_wf(data, features, target, penalty='l2', C=1.0, max_iter=1000):
    """Train logistic regression with walk-forward validation."""

    quarter_results = []

    for year, quarter in quarters:
        train_idx, test_idx = get_walk_forward_split(data, year, quarter,
                                                       config.WF_EMBARGO_DAYS)

        if len(test_idx) < 10:
            continue

        train_fold = data.loc[train_idx]
        test_fold = data.loc[test_idx]

        # Prepare data
        X_train = train_fold[features].values
        y_train = train_fold[target].values
        X_test = test_fold[features].values
        y_test = test_fold[target].values

        # Standardize features (fit on train, transform both)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LogisticRegression(penalty=penalty, C=C, max_iter=max_iter,
                                   random_state=config.RANDOM_SEED, solver='liblinear')
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = np.nan

        ll = log_loss(y_test, y_pred_proba)
        bs = brier_score_loss(y_test, y_pred_proba)
        rank_ic = np.corrcoef(y_pred_proba, y_test)[0, 1] if len(y_test) > 1 else np.nan

        quarter_results.append({
            'year': year,
            'quarter': quarter,
            'auc': auc,
            'log_loss': ll,
            'brier_score': bs,
            'rank_ic': rank_ic,
            'n_train': len(train_fold),
            'n_test': len(test_fold)
        })

    return pd.DataFrame(quarter_results)

# Model 1: Logistic regression with L2 (ridge)
print("  Training Logistic (L2)...")
res_logistic_l2 = train_logistic_wf(train_clean, config.ALL_FEATURES,
                                     config.PRIMARY_TARGET, penalty='l2', C=1.0)

results_list.append({
    'timestamp': datetime.now().isoformat(),
    'model_type': 'logistic',
    'model_name': 'logistic_l2',
    'features': ','.join(config.ALL_FEATURES),
    'n_features': len(config.ALL_FEATURES),
    'target': config.PRIMARY_TARGET,
    'seed': config.RANDOM_SEED,
    'auc_mean': res_logistic_l2['auc'].mean(),
    'auc_std': res_logistic_l2['auc'].std(),
    'log_loss_mean': res_logistic_l2['log_loss'].mean(),
    'brier_score_mean': res_logistic_l2['brier_score'].mean(),
    'rank_ic_mean': res_logistic_l2['rank_ic'].mean(),
    'n_quarters': len(res_logistic_l2),
    'params': {'penalty': 'l2', 'C': 1.0}
})

print(f"    AUC: {res_logistic_l2['auc'].mean():.4f} ± {res_logistic_l2['auc'].std():.4f}")

# Model 2: Logistic regression with L1 (lasso)
print("  Training Logistic (L1)...")
res_logistic_l1 = train_logistic_wf(train_clean, config.ALL_FEATURES,
                                     config.PRIMARY_TARGET, penalty='l1', C=1.0)

results_list.append({
    'timestamp': datetime.now().isoformat(),
    'model_type': 'logistic',
    'model_name': 'logistic_l1',
    'features': ','.join(config.ALL_FEATURES),
    'n_features': len(config.ALL_FEATURES),
    'target': config.PRIMARY_TARGET,
    'seed': config.RANDOM_SEED,
    'auc_mean': res_logistic_l1['auc'].mean(),
    'auc_std': res_logistic_l1['auc'].std(),
    'log_loss_mean': res_logistic_l1['log_loss'].mean(),
    'brier_score_mean': res_logistic_l1['brier_score'].mean(),
    'rank_ic_mean': res_logistic_l1['rank_ic'].mean(),
    'n_quarters': len(res_logistic_l1),
    'params': {'penalty': 'l1', 'C': 1.0}
})

print(f"    AUC: {res_logistic_l1['auc'].mean():.4f} ± {res_logistic_l1['auc'].std():.4f}")

# ============================================================================
# 4. LEAKAGE TEST
# ============================================================================

print(f"\n[5/7] Running leakage test (shuffled labels)...")

# Shuffle labels and re-run logistic L2
train_clean_shuffled = train_clean.copy()
train_clean_shuffled[config.PRIMARY_TARGET] = np.random.permutation(
    train_clean_shuffled[config.PRIMARY_TARGET].values
)

res_shuffled = train_logistic_wf(train_clean_shuffled, config.ALL_FEATURES,
                                  config.PRIMARY_TARGET, penalty='l2', C=1.0)

shuffled_auc = res_shuffled['auc'].mean()
print(f"  Shuffled labels AUC: {shuffled_auc:.4f} ± {res_shuffled['auc'].std():.4f}")

if shuffled_auc > config.LEAKAGE_AUC_THRESHOLD:
    print(f"  ⚠️  WARNING: Shuffled AUC > {config.LEAKAGE_AUC_THRESHOLD} - possible leakage!")
else:
    print(f"  ✓ Shuffled AUC < {config.LEAKAGE_AUC_THRESHOLD} - no leakage detected")

# ============================================================================
# 5. CHECK FOR DRIFT EXPLOITATION
# ============================================================================

print(f"\n[6/7] Checking for exploitable drift...")

max_auc = res_logistic_l2['auc'].mean()

if max_auc > config.DRIFT_AUC_THRESHOLD:
    print(f"  ⚠️  WARNING: Max AUC ({max_auc:.4f}) > {config.DRIFT_AUC_THRESHOLD}")
    print(f"  This suggests predictive power. Check for:")
    print(f"    1. Look-ahead bias in features")
    print(f"    2. Data leakage")
    print(f"    3. Overfitting")
    print(f"\n  🛑 STOPPING before tree models. Review results first!")
else:
    print(f"  ✓ Max AUC ({max_auc:.4f}) < {config.DRIFT_AUC_THRESHOLD}")
    print(f"  Results consistent with efficient markets (weak/no predictability)")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

print(f"\n[7/7] Saving results...")

# Save model registry
registry_df = pd.DataFrame(results_list)

if config.MODEL_REGISTRY_PATH.exists():
    existing = pd.read_csv(config.MODEL_REGISTRY_PATH)
    registry_df = pd.concat([existing, registry_df], ignore_index=True)

registry_df.to_csv(config.MODEL_REGISTRY_PATH, index=False)
print(f"  Saved model registry: {config.MODEL_REGISTRY_PATH}")

# Save detailed quarter-by-quarter results
res_logistic_l2.to_csv(config.LOGS_DIR / "11_logistic_l2_quarters.csv", index=False)
res_logistic_l1.to_csv(config.LOGS_DIR / "11_logistic_l1_quarters.csv", index=False)
print(f"  Saved quarter-by-quarter results: {config.LOGS_DIR}")

# ============================================================================
# 7. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)

print("\nBASELINES:")
for baseline_name, results in baseline_results.items():
    if len(results) > 0:
        df_res = pd.DataFrame(results)
        print(f"  {baseline_name:20s}: AUC = {df_res['auc'].mean():.4f} ± {df_res['auc'].std():.4f}")

print("\nLOGISTIC REGRESSION:")
print(f"  L2 (Ridge):             AUC = {res_logistic_l2['auc'].mean():.4f} ± {res_logistic_l2['auc'].std():.4f}")
print(f"  L1 (Lasso):             AUC = {res_logistic_l1['auc'].mean():.4f} ± {res_logistic_l1['auc'].std():.4f}")

print("\nLEAKAGE TEST:")
print(f"  Shuffled labels:        AUC = {shuffled_auc:.4f} (expect ~0.50)")

print("\nINTERPRETATION:")
if max_auc < 0.52:
    print("  • No predictive power (AUC ≈ 0.50)")
    print("  • Consistent with efficient markets")
    print("  • PEAD is not exploitable in S&P 500 large caps")
elif max_auc < 0.55:
    print("  • Weak predictive power (AUC slightly > 0.50)")
    print("  • May not be economically significant after costs")
elif max_auc < 0.60:
    print("  • Moderate predictive power")
    print("  • Warrants further investigation")
else:
    print("  • Strong predictive power - CHECK FOR LEAKAGE!")

print("\n" + "=" * 80)
print("✓ 11_predictive.py PART 1 completed successfully")
print("\n⏸️  PAUSED: Review results before proceeding to tree models (Part 2)")
print("=" * 80)
