"""
10_explanatory.py - Does Post-Earnings Announcement Drift Exist?

Regression analysis to test whether PEAD exists in our sample.
Models: OLS with clustered SE, Fama-MacBeth
Outputs: Publication-ready regression tables (CSV + LaTeX)

Ground rules enforced:
- Uses 2015-2022 data only (no holdout contamination)
- Standard errors clustered by announcement date
- Robustness checks for multiple specifications
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import config
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("10_EXPLANATORY.PY - PEAD REGRESSION ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n[1/6] Loading data...")
panel = pd.read_parquet(config.PROCESSED_DIR / "model_panel.parquet")

# Use train set only (2015-2022)
train = panel[panel['split'] == 'train'].copy()
print(f"  Train events: {len(train):,} (2015-2022)")

# Filter to events with required data
required_cols = ['sue_wins_std', 'price_tminus1_wins_std', 'momentum_12_1_wins_std',
                 'rv_21d_wins_std', 'car_abn_2_20', 'car_abn_0_1', 'timing_bmo',
                 't0', 'ticker', 'sector', 'year', 'quarter']

train_clean = train.dropna(subset=required_cols).copy()
print(f"  After removing NA: {len(train_clean):,} ({100*len(train_clean)/len(train):.1f}%)")

# Create year-quarter fixed effects
train_clean['year_quarter'] = train_clean['year'].astype(str) + 'Q' + train_clean['quarter'].str[-1]

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def clustered_regression(data, y_col, X_cols, cluster_var='t0'):
    """
    Run OLS with clustered standard errors (manual implementation).

    Parameters:
    -----------
    data : DataFrame
    y_col : str (dependent variable)
    X_cols : list of str (independent variables)
    cluster_var : str (column to cluster on)

    Returns:
    --------
    dict with coefficients, t-stats, p-values, R2, N
    """
    # Prepare data
    df = data[[y_col] + X_cols + [cluster_var]].dropna().copy()

    y = df[y_col].values.astype(float)
    X = df[X_cols].values.astype(float)
    X_with_const = np.column_stack([np.ones(len(X)), X])
    clusters = df[cluster_var].values

    # Fit OLS
    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

    # Predictions and residuals
    y_pred = X_with_const @ beta
    resid = y - y_pred

    # R-squared
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Clustered standard errors
    n_obs = len(y)
    k = len(beta)

    # Cluster-robust covariance matrix
    cluster_ids = pd.Series(clusters).unique()
    n_clusters = len(cluster_ids)

    # Bread (X'X)^(-1)
    bread = np.linalg.inv(X_with_const.T @ X_with_const)

    # Meat (clustered)
    meat = np.zeros((k, k))
    for c in cluster_ids:
        cluster_mask = (clusters == c)
        X_c = X_with_const[cluster_mask]
        resid_c = resid[cluster_mask]
        meat += X_c.T @ np.outer(resid_c, resid_c) @ X_c

    # Sandwich estimator with finite sample correction
    correction = n_clusters / (n_clusters - 1) * (n_obs - 1) / (n_obs - k)
    vcov_cluster = correction * bread @ meat @ bread

    se_cluster = np.sqrt(np.diag(vcov_cluster))

    # t-stats and p-values
    t_stats = beta / se_cluster
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_clusters-1))

    # Create result with variable names
    var_names = ['Intercept'] + X_cols
    results = {
        'coef': pd.Series(beta, index=var_names),
        'se': pd.Series(se_cluster, index=var_names),
        't': pd.Series(t_stats, index=var_names),
        'p': pd.Series(p_values, index=var_names),
        'r2': r2,
        'r2_adj': 1 - (1-r2) * (n_obs-1) / (n_obs-k),
        'n': n_obs,
        'n_clusters': n_clusters
    }

    return results

def two_way_cluster(data, y_col, X_cols, cluster1='t0', cluster2='ticker'):
    """
    Two-way clustered standard errors (by date and firm).
    Following Cameron, Gelbach & Miller (2011).

    Two-way clustering: Var = Var_cluster1 + Var_cluster2 - Var_intersection
    """
    # Prepare data
    df = data[[y_col] + X_cols + [cluster1, cluster2]].dropna().copy()

    y = df[y_col].values.astype(float)
    X = df[X_cols].values.astype(float)
    X_with_const = np.column_stack([np.ones(len(X)), X])

    # Fit OLS
    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    y_pred = X_with_const @ beta
    resid = y - y_pred

    # R-squared
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    n_obs = len(y)
    k = len(beta)

    # Bread (X'X)^(-1)
    bread = np.linalg.inv(X_with_const.T @ X_with_const)

    # Helper function to compute clustered variance
    def cluster_vcov(cluster_col):
        cluster_ids = pd.Series(cluster_col).unique()
        n_clusters = len(cluster_ids)

        meat = np.zeros((k, k))
        for c in cluster_ids:
            mask = (cluster_col == c)
            X_c = X_with_const[mask]
            resid_c = resid[mask]
            meat += X_c.T @ np.outer(resid_c, resid_c) @ X_c

        correction = n_clusters / (n_clusters - 1) * (n_obs - 1) / (n_obs - k)
        return correction * bread @ meat @ bread

    # Cluster by dimension 1
    vcov1 = cluster_vcov(df[cluster1].values)

    # Cluster by dimension 2
    vcov2 = cluster_vcov(df[cluster2].values)

    # Cluster by intersection (date × firm)
    df_copy = df.copy()
    df_copy['cluster_both'] = df_copy[cluster1].astype(str) + '_' + df_copy[cluster2].astype(str)
    vcov12 = cluster_vcov(df_copy['cluster_both'].values)

    # Two-way clustered variance (Cameron-Gelbach-Miller formula)
    vcov_twoway = vcov1 + vcov2 - vcov12
    se_twoway = np.sqrt(np.diag(vcov_twoway))

    # t-stats and p-values
    t_stats = beta / se_twoway

    # Use min of two cluster counts for conservative DF
    n_clust1 = df[cluster1].nunique()
    n_clust2 = df[cluster2].nunique()
    df_resid = min(n_clust1, n_clust2) - 1

    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_resid))

    var_names = ['Intercept'] + X_cols
    return {
        'coef': pd.Series(beta, index=var_names),
        'se': pd.Series(se_twoway, index=var_names),
        't': pd.Series(t_stats, index=var_names),
        'p': pd.Series(p_values, index=var_names),
        'r2': r2,
        'r2_adj': 1 - (1-r2) * (n_obs-1) / (n_obs-k),
        'n': n_obs,
        'n_clusters': (n_clust1, n_clust2)
    }

def fama_macbeth(data, y_col, X_cols, time_var='year_quarter'):
    """
    Fama-MacBeth cross-sectional regressions.
    One regression per time period, then average coefficients.
    Standard errors account for time-series correlation.
    """
    # Extract time periods
    time_periods = sorted(data[time_var].unique())

    # Run cross-sectional regression for each period
    coef_list = []

    for t in time_periods:
        period_data = data[data[time_var] == t].copy()

        # Need minimum observations per period
        if len(period_data) < 10:
            continue

        # Drop NA
        period_clean = period_data[[y_col] + X_cols].dropna()
        if len(period_clean) < 10:
            continue

        # Run OLS for this period
        y = period_clean[y_col].values.astype(float)
        X = period_clean[X_cols].values.astype(float)
        X_with_const = np.column_stack([np.ones(len(X)), X])

        try:
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            coef_list.append(beta)
        except:
            continue

    # Convert to DataFrame
    var_names = ['Intercept'] + X_cols
    coef_df = pd.DataFrame(coef_list, columns=var_names)

    # Average coefficients across periods (Fama-MacBeth estimate)
    mean_coef = coef_df.mean()

    # Standard errors (time-series standard deviation / sqrt(T))
    # This accounts for time-series correlation in coefficient estimates
    se_fm = coef_df.std() / np.sqrt(len(coef_df))

    t_stats = mean_coef / se_fm
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(coef_df)-1))

    return {
        'coef': mean_coef,
        'se': se_fm,
        't': t_stats,
        'p': p_values,
        'n_periods': len(coef_df),
        'n_total': len(data)
    }

def format_regression_table(results_dict, output_path):
    """Format multiple regression results into publication table."""

    # Create comparison table
    rows = []

    # Get all variable names (union across models)
    all_vars = set()
    for r in results_dict.values():
        all_vars.update(r['coef'].index)

    all_vars = sorted(all_vars)

    # Build rows
    for var in all_vars:
        row = {'Variable': var}

        for model_name, r in results_dict.items():
            # Handle both Series and dict-like returns
            coef_vals = r['coef']
            se_vals = r['se']
            t_vals = r['t']
            p_vals = r['p']

            # Check if variable exists
            try:
                if isinstance(coef_vals, pd.Series):
                    if var in coef_vals.index:
                        coef = coef_vals[var]
                        se = se_vals[var]
                        t = t_vals[var]
                        p = p_vals[var]
                    else:
                        row[model_name] = ""
                        row[f"{model_name}_se"] = ""
                        row[f"{model_name}_t"] = ""
                        continue
                else:
                    # Dict-like access
                    coef = coef_vals.get(var, None)
                    if coef is None:
                        row[model_name] = ""
                        row[f"{model_name}_se"] = ""
                        row[f"{model_name}_t"] = ""
                        continue
                    se = se_vals.get(var)
                    t = t_vals.get(var)
                    p = p_vals.get(var)

                # Format: coefficient (se) with significance stars
                stars = ''
                if p < 0.01:
                    stars = '***'
                elif p < 0.05:
                    stars = '**'
                elif p < 0.10:
                    stars = '*'

                row[model_name] = f"{coef:.4f}{stars}"
                row[f"{model_name}_se"] = f"({se:.4f})"
                row[f"{model_name}_t"] = f"{t:.2f}"
            except:
                row[model_name] = ""
                row[f"{model_name}_se"] = ""
                row[f"{model_name}_t"] = ""

        rows.append(row)

    # Add statistics
    stats_row = {'Variable': 'N'}
    for model_name, r in results_dict.items():
        n_val = r.get('n', r.get('n_total', ''))
        stats_row[model_name] = str(n_val)
        stats_row[f"{model_name}_se"] = ""
        stats_row[f"{model_name}_t"] = ""
    rows.append(stats_row)

    stats_row2 = {'Variable': 'R²'}
    for model_name, r in results_dict.items():
        if 'r2' in r:
            stats_row2[model_name] = f"{r['r2']:.3f}"
        elif 'n_periods' in r:
            # For Fama-MacBeth, show number of periods
            stats_row2[model_name] = f"{r['n_periods']} periods"
        else:
            stats_row2[model_name] = ""
        stats_row2[f"{model_name}_se"] = ""
        stats_row2[f"{model_name}_t"] = ""
    rows.append(stats_row2)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return df

# ============================================================================
# 3. MAIN REGRESSIONS - CAR[+2,+20]
# ============================================================================

print("\n[2/6] Running main regressions (CAR[+2,+20])...")

# Base independent variables
base_X = ['sue_wins_std', 'price_tminus1_wins_std', 'momentum_12_1_wins_std',
          'rv_21d_wins_std', 'timing_bmo']

# Model 1: Pooled OLS, clustered by date
model1 = clustered_regression(train_clean, 'car_abn_2_20', base_X, cluster_var='t0')
print(f"  Model 1 (OLS, date clustering): N={model1['n']:,}, R²={model1['r2']:.3f}")
print(f"    SUE coefficient: {model1['coef']['sue_wins_std']:.4f} (t={model1['t']['sue_wins_std']:.2f})")

# Model 2: With sector fixed effects
sector_dummies = [col for col in config.SECTOR_DUMMIES if col in train_clean.columns]
X_with_sector = base_X + sector_dummies
model2 = clustered_regression(train_clean, 'car_abn_2_20', X_with_sector, cluster_var='t0')
print(f"  Model 2 (+ sector FE): N={model2['n']:,}, R²={model2['r2']:.3f}")
print(f"    SUE coefficient: {model2['coef']['sue_wins_std']:.4f} (t={model2['t']['sue_wins_std']:.2f})")

# Model 3: Two-way clustering (date + firm)
print("  Model 3 (two-way clustering)...")
model3 = two_way_cluster(train_clean, 'car_abn_2_20', base_X, cluster1='t0', cluster2='ticker')
print(f"    SUE coefficient: {model3['coef']['sue_wins_std']:.4f} (t={model3['t']['sue_wins_std']:.2f})")

# Model 4: Fama-MacBeth
print("  Model 4 (Fama-MacBeth)...")
model4 = fama_macbeth(train_clean, 'car_abn_2_20', base_X, time_var='year_quarter')
print(f"    SUE coefficient: {model4['coef']['sue_wins_std']:.4f} (t={model4['t']['sue_wins_std']:.2f})")
print(f"    Periods: {model4['n_periods']}")

# Save results
results_drift = {
    'OLS_Date_Cluster': model1,
    'OLS_Sector_FE': model2,
    'OLS_TwoWay_Cluster': model3,
    'Fama_MacBeth': model4
}

table_drift = format_regression_table(results_drift,
                                      config.LOGS_DIR / "reg_table_drift.csv")

# ============================================================================
# 4. ANNOUNCEMENT RETURN - CAR[0,+1]
# ============================================================================

print("\n[3/6] Running announcement return regressions (CAR[0,+1])...")

model_ann1 = clustered_regression(train_clean, 'car_abn_0_1', base_X, cluster_var='t0')
print(f"  Announcement (OLS): N={model_ann1['n']:,}, R²={model_ann1['r2']:.3f}")
print(f"    SUE coefficient: {model_ann1['coef']['sue_wins_std']:.4f} (t={model_ann1['t']['sue_wins_std']:.2f})")

model_ann2 = fama_macbeth(train_clean, 'car_abn_0_1', base_X, time_var='year_quarter')
print(f"  Announcement (FM): SUE coef={model_ann2['coef']['sue_wins_std']:.4f} (t={model_ann2['t']['sue_wins_std']:.2f})")

results_ann = {
    'OLS': model_ann1,
    'Fama_MacBeth': model_ann2
}

table_ann = format_regression_table(results_ann,
                                    config.LOGS_DIR / "reg_table_announcement.csv")

# ============================================================================
# 5. ROBUSTNESS CHECKS
# ============================================================================

print("\n[4/6] Running robustness checks...")

robustness_results = {}

# 5.1 Price-scaled SUE
if 'eps_actual_yf' in train_clean.columns:
    train_clean['sue_price_scaled'] = train_clean['surprise'] / train_clean['price_tminus1']
    train_clean['sue_price_scaled'] = train_clean.groupby('quarter')['sue_price_scaled'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    X_ps = ['sue_price_scaled', 'price_tminus1_wins_std', 'momentum_12_1_wins_std',
            'rv_21d_wins_std', 'timing_bmo']
    model_ps = clustered_regression(train_clean.dropna(subset=['sue_price_scaled']),
                                     'car_abn_2_20', X_ps, cluster_var='t0')
    robustness_results['Price_Scaled_SUE'] = model_ps
    print(f"  Price-scaled SUE: coef={model_ps['coef']['sue_price_scaled']:.4f} (t={model_ps['t']['sue_price_scaled']:.2f})")

# 5.2 Exclude 2020 (COVID)
train_no_covid = train_clean[train_clean['year'] != 2020].copy()
model_no_covid = clustered_regression(train_no_covid, 'car_abn_2_20', base_X, cluster_var='t0')
robustness_results['Exclude_2020'] = model_no_covid
print(f"  Exclude 2020: SUE coef={model_no_covid['coef']['sue_wins_std']:.4f} (t={model_no_covid['t']['sue_wins_std']:.2f})")

# 5.3 Split by size tercile
for tercile in ['small', 'medium', 'large']:
    if 'size_tercile' in train_clean.columns:
        tercile_data = train_clean[train_clean['size_tercile'] == tercile].copy()
        if len(tercile_data) > 100:
            model_size = clustered_regression(tercile_data, 'car_abn_2_20', base_X, cluster_var='t0')
            robustness_results[f'Size_{tercile}'] = model_size
            print(f"  Size={tercile}: SUE coef={model_size['coef']['sue_wins_std']:.4f} (t={model_size['t']['sue_wins_std']:.2f}, N={model_size['n']:,})")

# 5.4 Time periods
train_early = train_clean[train_clean['year'] <= 2018].copy()
train_late = train_clean[train_clean['year'] >= 2019].copy()

model_early = clustered_regression(train_early, 'car_abn_2_20', base_X, cluster_var='t0')
model_late = clustered_regression(train_late, 'car_abn_2_20', base_X, cluster_var='t0')

robustness_results['2015_2018'] = model_early
robustness_results['2019_2022'] = model_late

print(f"  2015-2018: SUE coef={model_early['coef']['sue_wins_std']:.4f} (t={model_early['t']['sue_wins_std']:.2f})")
print(f"  2019-2022: SUE coef={model_late['coef']['sue_wins_std']:.4f} (t={model_late['t']['sue_wins_std']:.2f})")

# Save robustness table
table_robust = format_regression_table(robustness_results,
                                       config.LOGS_DIR / "reg_table_robustness.csv")

# ============================================================================
# 6. SUMMARY STATISTICS
# ============================================================================

print("\n[5/6] Computing summary statistics...")

summary_stats = {
    'N_events': len(train_clean),
    'N_dates': train_clean['t0'].nunique(),
    'N_tickers': train_clean['ticker'].nunique(),
    'Mean_CAR_2_20': train_clean['car_abn_2_20'].mean(),
    'Std_CAR_2_20': train_clean['car_abn_2_20'].std(),
    'Mean_CAR_0_1': train_clean['car_abn_0_1'].mean(),
    'Std_CAR_0_1': train_clean['car_abn_0_1'].std(),
    'Mean_SUE': train_clean['sue_wins_std'].mean(),
    'Std_SUE': train_clean['sue_wins_std'].std(),
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv(config.LOGS_DIR / "reg_summary_stats.csv", index=False)
print(f"  Saved: {config.LOGS_DIR / 'reg_summary_stats.csv'}")

# ============================================================================
# 7. INTERPRETATION & OUTPUT
# ============================================================================

print("\n[6/6] Summary of findings:")
print("=" * 80)
print("\nKEY RESULT:")
print(f"  SUE coefficient on CAR[+2,+20]: {model1['coef']['sue_wins_std']:.4f}")
print(f"  t-statistic (clustered): {model1['t']['sue_wins_std']:.2f}")
print(f"  p-value: {model1['p']['sue_wins_std']:.4f}")

if model1['p']['sue_wins_std'] < 0.05:
    print("  ✓ STATISTICALLY SIGNIFICANT at 5% level")
else:
    print("  ✗ NOT statistically significant")

print(f"\n  Economic magnitude:")
print(f"    1 std dev increase in SUE → {model1['coef']['sue_wins_std']:.4f} increase in CAR[+2,+20]")
print(f"    That's {100*model1['coef']['sue_wins_std']:.2f} bps over 20 days")

print(f"\nANNOUNCEMENT REACTION:")
print(f"  SUE coefficient on CAR[0,+1]: {model_ann1['coef']['sue_wins_std']:.4f}")
print(f"  t-statistic: {model_ann1['t']['sue_wins_std']:.2f}")
print(f"  That's {100*model_ann1['coef']['sue_wins_std']:.2f} bps immediate reaction")

print(f"\nROBUSTNESS:")
print(f"  Two-way clustering: t={model3['t']['sue_wins_std']:.2f}")
print(f"  Fama-MacBeth: t={model4['t']['sue_wins_std']:.2f}")
print(f"  Exclude 2020: t={model_no_covid['t']['sue_wins_std']:.2f}")

print("\n" + "=" * 80)
print("OUTPUTS SAVED:")
print(f"  • {config.LOGS_DIR / 'reg_table_drift.csv'}")
print(f"  • {config.LOGS_DIR / 'reg_table_announcement.csv'}")
print(f"  • {config.LOGS_DIR / 'reg_table_robustness.csv'}")
print(f"  • {config.LOGS_DIR / 'reg_summary_stats.csv'}")
print("=" * 80)

print("\n✓ 10_explanatory.py completed successfully")
