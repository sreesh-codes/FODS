"""
Compute SUE and CAR for events_clean.parquet
This adds the core metrics needed for PEAD analysis.
"""

import pandas as pd
import numpy as np
import config
from datetime import timedelta

print("Loading data...")
events = pd.read_parquet(config.PROCESSED_DIR / "events_clean.parquet")
prices = pd.read_parquet(config.PROCESSED_DIR / "prices_all.parquet")
sp500 = pd.read_parquet(config.PROCESSED_DIR / "prices_sp500.parquet")

# Compute returns
prices = prices.sort_values(['ticker', 'Date'])
prices['ret'] = prices.groupby('ticker')['Close'].pct_change()

sp500 = sp500.sort_values('Date')
sp500['mkt_ret'] = sp500['Close'].pct_change()

print(f"\n=== COMPUTING SUE ===")

# Compute SUE: (actual - estimate) / std(last 8 surprises)
events = events.sort_values(['ticker', 'datetime'])
events['surprise'] = events['eps_actual_yf'] - events['eps_estimate_yf']

def compute_sue(group):
    """Compute SUE for a ticker's events."""
    group = group.sort_values('datetime').copy()

    # Rolling std of last 8 surprises
    group['surprise_std'] = group['surprise'].shift(1).rolling(window=8, min_periods=2).std()

    # SUE = surprise / std
    group['sue'] = group['surprise'] / group['surprise_std']

    return group

# Apply with include_groups=False to prevent ticker column removal
events = events.groupby('ticker', group_keys=False).apply(compute_sue, include_groups=False).reset_index(drop=True)

# Merge ticker back from original
orig_ticker = pd.read_parquet(config.PROCESSED_DIR / "events_clean.parquet")[['ticker']]
events['ticker'] = orig_ticker['ticker'].values

sue_coverage = events['sue'].notna().sum()
print(f"✓ SUE coverage: {sue_coverage} / {len(events)} ({100*sue_coverage/len(events):.1f}%)")

print(f"\n=== COMPUTING CAR WINDOWS ===")

# Merge market returns
events['mkt_ret_t0'] = np.nan

for idx, row in events.iterrows():
    t0 = row['t0']
    mkt_data = sp500[sp500['Date'] == t0]
    if len(mkt_data) > 0:
        events.loc[idx, 'mkt_ret_t0'] = mkt_data.iloc[0]['mkt_ret']

# Compute CAR for each window - use list comprehension for speed
car_results = []

for i, row in events.iterrows():
    result = {'_idx': i}

    ticker = row['ticker']
    t0 = row['t0']

    # Get price data for this ticker
    ticker_prices = prices[prices['ticker'] == ticker].copy()
    ticker_prices = ticker_prices.sort_values('Date')

    for window_start, window_end in config.CAR_WINDOWS:
        # Define window dates
        start_date = t0 + timedelta(days=window_start)
        end_date = t0 + timedelta(days=window_end)

        # Get returns in window
        window_prices = ticker_prices[
            (ticker_prices['Date'] >= start_date) &
            (ticker_prices['Date'] <= end_date)
        ]

        car = np.nan
        if len(window_prices) > 0:
            # Stock CAR (geometric)
            car = (1 + window_prices['ret'].fillna(0)).prod() - 1

        # Get market returns in window
        window_mkt = sp500[
            (sp500['Date'] >= start_date) &
            (sp500['Date'] <= end_date)
        ]

        car_mkt = np.nan
        if len(window_mkt) > 0:
            # Market CAR
            car_mkt = (1 + window_mkt['mkt_ret'].fillna(0)).prod() - 1

        # Abnormal return
        car_abn = car - car_mkt if pd.notna(car) and pd.notna(car_mkt) else np.nan

        result[f'car_{window_start}_{window_end}'] = car
        result[f'car_mkt_{window_start}_{window_end}'] = car_mkt
        result[f'car_abn_{window_start}_{window_end}'] = car_abn

    car_results.append(result)

    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1}/{len(events)} events...")

# Convert to dataframe and merge
car_df = pd.DataFrame(car_results).set_index('_idx')
events = events.join(car_df)

print(f"\n=== CAR COVERAGE ===")
for window_start, window_end in config.CAR_WINDOWS:
    col_name = f'car_{window_start}_{window_end}'
    coverage = events[col_name].notna().sum()
    print(f"  CAR[{window_start},{window_end}]: {coverage} ({100*coverage/len(events):.1f}%)")

# Save
output_path = config.PROCESSED_DIR / "events_clean.parquet"
events.to_parquet(output_path)
print(f"\n✓ Saved events with SUE and CAR to {output_path}")

# Summary
print(f"\n=== SUMMARY ===")
print(f"Total events: {len(events):,}")
print(f"SUE available: {events['sue'].notna().sum():,}")
print(f"CAR[0,1] available: {events['car_0_1'].notna().sum():,}")
print(f"CAR[+2,+20] available: {events['car_2_20'].notna().sum():,}")
