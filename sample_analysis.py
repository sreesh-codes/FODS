"""
Sample Analysis Script
Shows how to load and analyze the processed data.
Run this AFTER the pipeline completes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the main events table
print("Loading events table...")
events = pd.read_parquet('processed/events.parquet')
print(f"Loaded {len(events):,} earnings events")

# Filter to events with data we need
print("\nFiltering to events with SUE and CAR data...")
events_clean = events[
    events['in_index'] &  # Only stocks in index at announcement
    events['sue'].notna() &  # Has SUE
    events['car_0_5_market_adj'].notna()  # Has CAR
].copy()
print(f"Clean sample: {len(events_clean):,} events")

# ============================================================================
# ANALYSIS 1: Post-Earnings Announcement Drift (PEAD)
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 1: POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD)")
print("="*70)

# Create SUE quintiles
events_clean['sue_quintile'] = pd.qcut(
    events_clean['sue'],
    q=5,
    labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'],
    duplicates='drop'
)

# Average CAR by quintile
sue_analysis = events_clean.groupby('sue_quintile').agg({
    'car_0_5_market_adj': ['mean', 'std', 'count'],
    'car_2_20_market_adj': ['mean', 'std', 'count']
})

print("\nAverage CAR by SUE Quintile:")
print(sue_analysis)

# Spread (Q5 - Q1)
q1_car_short = sue_analysis.loc['Q1 (Low)', ('car_0_5_market_adj', 'mean')]
q5_car_short = sue_analysis.loc['Q5 (High)', ('car_0_5_market_adj', 'mean')]
spread_short = q5_car_short - q1_car

print(f"\nPEAD Test (Short-term CAR[0,+5]):")
print(f"  Q5 (High SUE): {q5_car_short*100:.3f}%")
print(f"  Q1 (Low SUE):  {q1_car_short*100:.3f}%")
print(f"  Spread:        {spread_short*100:.3f}%")

if spread_short > 0:
    print("  → ✓ Evidence of PEAD (high SUE → higher returns)")
else:
    print("  → ✗ No PEAD (or reversal)")

# Drift test (Q5 - Q1 for longer window)
if 'car_2_20_market_adj' in events_clean.columns:
    q1_car_drift = sue_analysis.loc['Q1 (Low)', ('car_2_20_market_adj', 'mean')]
    q5_car_drift = sue_analysis.loc['Q5 (High)', ('car_2_20_market_adj', 'mean')]
    spread_drift = q5_car_drift - q1_car_drift

    print(f"\nDrift Test (CAR[+2,+20]):")
    print(f"  Q5 (High SUE): {q5_car_drift*100:.3f}%")
    print(f"  Q1 (Low SUE):  {q1_car_drift*100:.3f}%")
    print(f"  Spread:        {spread_drift*100:.3f}%")

    if spread_drift > 0:
        print("  → ✓ Drift continues (underreaction)")
    else:
        print("  → ✗ No drift or reversal (efficient markets)")

# ============================================================================
# ANALYSIS 2: Implied Volatility Crush
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 2: IMPLIED VOLATILITY CRUSH")
print("="*70)

try:
    iv_panel = pd.read_parquet('processed/iv_event_panel.parquet')
    print(f"Loaded {len(iv_panel):,} event-day observations")
    print(f"Events with IV: {iv_panel['event_id'].nunique():,}")

    # Average IV by event day
    iv_path = iv_panel.groupby('rel_day').agg({
        'iv': ['mean', 'std', 'count'],
        'rv': ['mean', 'std', 'count'],
        'iv_rv_spread': ['mean', 'std', 'count']
    })

    print("\nIV Path Around Earnings:")
    print(iv_path)

    # Crush test
    if -1 in iv_path.index and 1 in iv_path.index:
        iv_before = iv_path.loc[-1, ('iv', 'mean')]
        iv_after = iv_path.loc[1, ('iv', 'mean')]
        iv_drop = iv_before - iv_after
        iv_drop_pct = 100 * iv_drop / iv_before

        print(f"\nIV Crush Test:")
        print(f"  IV at t-1:  {iv_before:.4f}")
        print(f"  IV at t+1:  {iv_after:.4f}")
        print(f"  Drop:       {iv_drop:.4f} ({iv_drop_pct:.1f}%)")

        if iv_drop > 0:
            print("  → ✓ IV crush confirmed")
        else:
            print("  → ✗ No IV crush (or increase)")

        # IV-RV spread
        spread_before = iv_path.loc[-1, ('iv_rv_spread', 'mean')]
        spread_after = iv_path.loc[1, ('iv_rv_spread', 'mean')]

        print(f"\nIV-RV Spread:")
        print(f"  Before (t-1): {spread_before:.4f}")
        print(f"  After (t+1):  {spread_after:.4f}")
        print(f"  Change:       {spread_after - spread_before:.4f}")

    # Optional: Plot IV path
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        iv_mean = iv_path[('iv', 'mean')]
        rv_mean = iv_path[('rv', 'mean')]

        plt.plot(iv_mean.index, iv_mean.values, 'b-o', label='Implied Vol', markersize=4)
        plt.plot(rv_mean.index, rv_mean.values, 'r--s', label='Realized Vol', markersize=4)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Earnings Date')
        plt.xlabel('Days Relative to Earnings (t=0)')
        plt.ylabel('Volatility')
        plt.title('Implied vs Realized Volatility Around Earnings')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('iv_crush_plot.png', dpi=150)
        print("\n✓ Saved plot to iv_crush_plot.png")
    except:
        print("\n(matplotlib not available for plotting)")

except FileNotFoundError:
    print("✗ IV event panel not found (may not have IV data)")

# ============================================================================
# ANALYSIS 3: Sample Regression Setup
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 3: SAMPLE REGRESSION SETUP")
print("="*70)

print("\nSample regression: CAR[+2,+20] ~ SUE + controls")

# Prepare regression data
if 'car_2_20_market_adj' in events_clean.columns:
    reg_data = events_clean[[
        'ticker', 'sue', 'car_2_20_market_adj', 'price_t_minus_1'
    ]].copy()

    # Log market cap proxy
    reg_data['log_price'] = np.log(reg_data['price_t_minus_1'])

    # Drop missing
    reg_data = reg_data.dropna()

    print(f"\nRegression sample: {len(reg_data):,} observations")
    print("\nCorrelations:")
    print(reg_data[['sue', 'car_2_20_market_adj', 'log_price']].corr())

    # Simple OLS (if statsmodels available)
    try:
        import statsmodels.api as sm

        X = reg_data[['sue', 'log_price']]
        X = sm.add_constant(X)
        y = reg_data['car_2_20_market_adj']

        model = sm.OLS(y, X).fit()
        print("\nOLS Regression Results:")
        print(model.summary())

    except ImportError:
        print("\n(Install statsmodels for regression: pip install statsmodels)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total events: {len(events):,}")
print(f"Clean sample (in-index, has SUE): {len(events_clean):,}")
print(f"Date range: {events['t0'].min()} to {events['t0'].max()}")
print(f"\nKey files:")
print(f"  - processed/events.parquet")
print(f"  - processed/iv_event_panel.parquet")
print(f"  - processed/daily_panel.parquet")
print(f"\nFor more analysis, load the data and customize as needed.")
print("="*70)
