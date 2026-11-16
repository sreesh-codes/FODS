"""
11_predictive_part2.py - Tree Models for PEAD Trading

Part 2: Tree models (LightGBM, Random Forest, XGBoost)
Run this AFTER reviewing Part 1 results.

Models:
- LightGBM with monotone constraint (SUE must increase prediction)
- Random Forest (baseline tree model)
- XGBoost (gradient boosting alternative)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
import config
from pathlib import Path
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Try to import tree models
try:
    import lightgbm as lgb
    HAS_LGBM = True
except:
    HAS_LGBM = False
    print("WARNING: LightGBM not installed. Install with: pip install lightgbm")

try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_RF = True
except:
    HAS_RF = False

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    print("WARNING: XGBoost not installed. Install with: pip install xgboost")

np.random.seed(config.RANDOM_SEED)

print("=" * 80)
print("11_PREDICTIVE.PY - PART 2: TREE MODELS")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n[1/5] Loading data...")
panel = pd.read_parquet(config.PROCESSED_DIR / "model_panel.parquet")
train = panel[panel['split'] == 'train'].copy()

required_cols = config.ALL_FEATURES + [config.PRIMARY_TARGET, 't0', 'ticker', 'year', 'quarter']
train_clean = train.dropna(subset=required_cols).copy()
train_clean['t0'] = pd.to_datetime(train_clean['t0'])

print(f"  Train events: {len(train_clean):,}")

# Generate quarterly test periods
quarters = []
for year in range(config.WF_START_YEAR, config.WF_END_YEAR + 1):
    for q in [1, 2, 3, 4]:
        quarters.append((year, q))

# Helper function
def get_walk_forward_split(data, test_year, test_quarter, embargo_days=20):
    test_q_str = f"{test_year}Q{test_quarter}"
    test_mask = (data['year'] == test_year) & (data['quarter'] == test_q_str)

    if len(data[test_mask]) > 0:
        first_test_t0 = data[test_mask]['t0'].min()
        last_train_t0 = first_test_t0 - timedelta(days=embargo_days)
        train_mask = data['t0'] <= last_train_t0
    else:
        train_mask = data['year'] < test_year

    return data[train_mask].index, data[test_mask].index

results_list = []

# ============================================================================
# 2. LIGHTGBM WITH MONOTONE CONSTRAINT
# ============================================================================

if HAS_LGBM:
    print("\n[2/5] Training LightGBM (with monotone SUE)...")

    # Find SUE feature index
    sue_idx = config.ALL_FEATURES.index('sue_wins_std')

    quarter_results = []

    for year, quarter in quarters:
        train_idx, test_idx = get_walk_forward_split(train_clean, year, quarter,
                                                       config.WF_EMBARGO_DAYS)

        if len(test_idx) < 10:
            continue

        train_fold = train_clean.loc[train_idx]
        test_fold = train_clean.loc[test_idx]

        X_train = train_fold[config.ALL_FEATURES].values
        y_train = train_fold[config.PRIMARY_TARGET].values
        X_test = test_fold[config.ALL_FEATURES].values
        y_test = test_fold[config.PRIMARY_TARGET].values

        # Monotone constraint: SUE must be increasing
        monotone_constraints = [0] * len(config.ALL_FEATURES)
        monotone_constraints[sue_idx] = 1

        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbose': -1,
            'random_state': config.RANDOM_SEED,
            'monotone_constraints': monotone_constraints
        }

        model = lgb.train(params, train_data, num_boost_round=100)

        # Predict
        y_pred_proba = model.predict(X_test)

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
            'rank_ic': rank_ic
        })

    res_lgbm = pd.DataFrame(quarter_results)

    results_list.append({
        'timestamp': datetime.now().isoformat(),
        'model_type': 'tree',
        'model_name': 'lightgbm_monotone',
        'features': ','.join(config.ALL_FEATURES),
        'n_features': len(config.ALL_FEATURES),
        'target': config.PRIMARY_TARGET,
        'seed': config.RANDOM_SEED,
        'auc_mean': res_lgbm['auc'].mean(),
        'auc_std': res_lgbm['auc'].std(),
        'log_loss_mean': res_lgbm['log_loss'].mean(),
        'brier_score_mean': res_lgbm['brier_score'].mean(),
        'rank_ic_mean': res_lgbm['rank_ic'].mean(),
        'n_quarters': len(res_lgbm),
        'params': params
    })

    print(f"  AUC: {res_lgbm['auc'].mean():.4f} ± {res_lgbm['auc'].std():.4f}")
    res_lgbm.to_csv(config.LOGS_DIR / "11_lgbm_quarters.csv", index=False)

else:
    print("\n[2/5] Skipping LightGBM (not installed)")
    res_lgbm = None

# ============================================================================
# 3. RANDOM FOREST
# ============================================================================

if HAS_RF:
    print("\n[3/5] Training Random Forest...")

    quarter_results = []

    for year, quarter in quarters:
        train_idx, test_idx = get_walk_forward_split(train_clean, year, quarter,
                                                       config.WF_EMBARGO_DAYS)

        if len(test_idx) < 10:
            continue

        train_fold = train_clean.loc[train_idx]
        test_fold = train_clean.loc[test_idx]

        X_train = train_fold[config.ALL_FEATURES].values
        y_train = train_fold[config.PRIMARY_TARGET].values
        X_test = test_fold[config.ALL_FEATURES].values
        y_test = test_fold[config.PRIMARY_TARGET].values

        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=20,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred_proba = model.predict_proba(X_test)[:, 1]

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
            'rank_ic': rank_ic
        })

    res_rf = pd.DataFrame(quarter_results)

    results_list.append({
        'timestamp': datetime.now().isoformat(),
        'model_type': 'tree',
        'model_name': 'random_forest',
        'features': ','.join(config.ALL_FEATURES),
        'n_features': len(config.ALL_FEATURES),
        'target': config.PRIMARY_TARGET,
        'seed': config.RANDOM_SEED,
        'auc_mean': res_rf['auc'].mean(),
        'auc_std': res_rf['auc'].std(),
        'log_loss_mean': res_rf['log_loss'].mean(),
        'brier_score_mean': res_rf['brier_score'].mean(),
        'rank_ic_mean': res_rf['rank_ic'].mean(),
        'n_quarters': len(res_rf),
        'params': {'n_estimators': 100, 'max_depth': 5}
    })

    print(f"  AUC: {res_rf['auc'].mean():.4f} ± {res_rf['auc'].std():.4f}")
    res_rf.to_csv(config.LOGS_DIR / "11_rf_quarters.csv", index=False)

else:
    print("\n[3/5] Skipping Random Forest (not installed)")
    res_rf = None

# ============================================================================
# 4. XGBOOST
# ============================================================================

if HAS_XGB:
    print("\n[4/5] Training XGBoost...")

    quarter_results = []

    for year, quarter in quarters:
        train_idx, test_idx = get_walk_forward_split(train_clean, year, quarter,
                                                       config.WF_EMBARGO_DAYS)

        if len(test_idx) < 10:
            continue

        train_fold = train_clean.loc[train_idx]
        test_fold = train_clean.loc[test_idx]

        X_train = train_fold[config.ALL_FEATURES].values
        y_train = train_fold[config.PRIMARY_TARGET].values
        X_test = test_fold[config.ALL_FEATURES].values
        y_test = test_fold[config.PRIMARY_TARGET].values

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=config.RANDOM_SEED,
            eval_metric='logloss',
            use_label_encoder=False
        )
        model.fit(X_train, y_train, verbose=False)

        # Predict
        y_pred_proba = model.predict_proba(X_test)[:, 1]

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
            'rank_ic': rank_ic
        })

    res_xgb = pd.DataFrame(quarter_results)

    results_list.append({
        'timestamp': datetime.now().isoformat(),
        'model_type': 'tree',
        'model_name': 'xgboost',
        'features': ','.join(config.ALL_FEATURES),
        'n_features': len(config.ALL_FEATURES),
        'target': config.PRIMARY_TARGET,
        'seed': config.RANDOM_SEED,
        'auc_mean': res_xgb['auc'].mean(),
        'auc_std': res_xgb['auc'].std(),
        'log_loss_mean': res_xgb['log_loss'].mean(),
        'brier_score_mean': res_xgb['brier_score'].mean(),
        'rank_ic_mean': res_xgb['rank_ic'].mean(),
        'n_quarters': len(res_xgb),
        'params': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05}
    })

    print(f"  AUC: {res_xgb['auc'].mean():.4f} ± {res_xgb['auc'].std():.4f}")
    res_xgb.to_csv(config.LOGS_DIR / "11_xgb_quarters.csv", index=False)

else:
    print("\n[4/5] Skipping XGBoost (not installed)")
    res_xgb = None

# ============================================================================
# 5. SAVE AND SUMMARIZE
# ============================================================================

print("\n[5/5] Saving results...")

# Update model registry
if len(results_list) > 0:
    registry_df = pd.DataFrame(results_list)

    if config.MODEL_REGISTRY_PATH.exists():
        existing = pd.read_csv(config.MODEL_REGISTRY_PATH)
        registry_df = pd.concat([existing, registry_df], ignore_index=True)

    registry_df.to_csv(config.MODEL_REGISTRY_PATH, index=False)
    print(f"  Updated model registry: {config.MODEL_REGISTRY_PATH}")

print("\n" + "=" * 80)
print("SUMMARY - TREE MODELS")
print("=" * 80)

if res_lgbm is not None:
    print(f"\nLightGBM (monotone):  AUC = {res_lgbm['auc'].mean():.4f} ± {res_lgbm['auc'].std():.4f}")

if res_rf is not None:
    print(f"Random Forest:        AUC = {res_rf['auc'].mean():.4f} ± {res_rf['auc'].std():.4f}")

if res_xgb is not None:
    print(f"XGBoost:              AUC = {res_xgb['auc'].mean():.4f} ± {res_xgb['auc'].std():.4f}")

# Compare to baselines
if config.MODEL_REGISTRY_PATH.exists():
    registry = pd.read_csv(config.MODEL_REGISTRY_PATH)

    print("\n" + "=" * 80)
    print("ALL MODELS COMPARISON")
    print("=" * 80)

    for model_type in ['baseline', 'logistic', 'tree']:
        subset = registry[registry['model_type'] == model_type]
        if len(subset) > 0:
            print(f"\n{model_type.upper()}:")
            for _, row in subset.iterrows():
                print(f"  {row['model_name']:25s}: AUC = {row['auc_mean']:.4f} ± {row['auc_std']:.4f}")

print("\n" + "=" * 80)
print("INTERPRETATION:")

# Find best model
if config.MODEL_REGISTRY_PATH.exists():
    registry = pd.read_csv(config.MODEL_REGISTRY_PATH)
    best_model = registry.loc[registry['auc_mean'].idxmax()]

    print(f"\nBest model: {best_model['model_name']}")
    print(f"  AUC: {best_model['auc_mean']:.4f} ± {best_model['auc_std']:.4f}")

    if best_model['auc_mean'] < 0.52:
        print("\n✓ No meaningful predictive power across all models")
        print("  Markets are efficient - PEAD is not exploitable")
    elif best_model['auc_mean'] < 0.55:
        print("\n~ Weak signal detected, but likely not economically significant")
        print("  Transaction costs would likely eliminate any edge")
    else:
        print("\n⚠️  Moderate predictive power - warrants deeper investigation")

print("\n" + "=" * 80)
print("✓ 11_predictive_part2.py completed successfully")
print("=" * 80)
