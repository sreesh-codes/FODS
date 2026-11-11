"""
06_build_panels.py
Build derived tables and event panels for analysis.
Creates:
- processed/events.parquet (earnings events with SUE, abnormal returns, etc.)
- processed/iv_event_panel.parquet (IV and RV around earnings, t-10 to t+10)
- processed/daily_panel.parquet (daily stock-level data for all analysis)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / '06_build_panels.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    """Load all required data."""
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)

    data = {}

    # Prices
    try:
        prices = pd.read_parquet(config.PROCESSED_DIR / "prices_all.parquet")
        if 'Date' not in prices.columns and prices.index.name == 'Date':
            prices = prices.reset_index()
        prices['Date'] = pd.to_datetime(prices['Date'])
        # Remove timezone for easier comparison
        if prices['Date'].dt.tz is not None:
            prices['Date'] = prices['Date'].dt.tz_localize(None)
        data['prices'] = prices
        logger.info(f"✓ Prices: {len(prices)} rows, {prices['ticker'].nunique()} tickers")
    except Exception as e:
        logger.error(f"✗ Failed to load prices: {e}")
        data['prices'] = None

    # Market index
    try:
        sp500 = pd.read_parquet(config.PROCESSED_DIR / "prices_sp500.parquet")
        if 'Date' not in sp500.columns and sp500.index.name == 'Date':
            sp500 = sp500.reset_index()
        sp500['Date'] = pd.to_datetime(sp500['Date'])
        # Remove timezone for easier comparison
        if sp500['Date'].dt.tz is not None:
            sp500['Date'] = sp500['Date'].dt.tz_localize(None)
        data['sp500'] = sp500
        logger.info(f"✓ S&P 500: {len(sp500)} rows")
    except Exception as e:
        logger.warning(f"⚠ S&P 500 not found: {e}")
        data['sp500'] = None

    # Earnings
    try:
        earnings = pd.read_parquet(config.PROCESSED_DIR / "earnings_merged.parquet")

        # Handle column naming (might be 'datetime' or 'ann_datetime')
        if 'datetime' in earnings.columns and 'ann_datetime' not in earnings.columns:
            earnings['ann_datetime'] = pd.to_datetime(earnings['datetime'])
        else:
            earnings['ann_datetime'] = pd.to_datetime(earnings['ann_datetime'])

        # Remove timezone for easier comparison
        if earnings['ann_datetime'].dt.tz is not None:
            earnings['ann_datetime'] = earnings['ann_datetime'].dt.tz_localize(None)

        if 'date' in earnings.columns:
            earnings['date'] = pd.to_datetime(earnings['date'])
            if earnings['date'].dt.tz is not None:
                earnings['date'] = earnings['date'].dt.tz_localize(None)

        # Add consensus EPS columns if not present
        if 'eps_estimate' not in earnings.columns:
            earnings['eps_estimate'] = earnings.get('eps_estimate_av', earnings.get('eps_estimate_yf'))
        if 'eps_actual' not in earnings.columns:
            earnings['eps_actual'] = earnings.get('eps_actual_av', earnings.get('eps_actual_yf'))
        if 'surprise' not in earnings.columns and 'eps_actual' in earnings.columns and 'eps_estimate' in earnings.columns:
            earnings['surprise'] = earnings['eps_actual'] - earnings['eps_estimate']

        data['earnings'] = earnings
        logger.info(f"✓ Earnings: {len(earnings)} events")
    except Exception as e:
        logger.error(f"✗ Failed to load earnings: {e}")
        data['earnings'] = None

    # Membership
    try:
        membership = pd.read_parquet(config.PROCESSED_DIR / "membership.parquet")
        membership['start_date'] = pd.to_datetime(membership['start_date'])
        membership['end_date'] = pd.to_datetime(membership['end_date'])
        data['membership'] = membership
        logger.info(f"✓ Membership: {len(membership)} tickers")
    except Exception as e:
        logger.error(f"✗ Failed to load membership: {e}")
        data['membership'] = None

    # Factors
    try:
        factors = pd.read_parquet(config.RAW_DIR / "ff_factors.parquet")
        if 'Date' not in factors.columns:
            # Index might be PeriodIndex, convert to datetime
            factors = factors.reset_index()
            if hasattr(factors.iloc[:, 0], 'to_timestamp'):
                # Convert PeriodIndex to DatetimeIndex
                factors.iloc[:, 0] = factors.iloc[:, 0].apply(lambda x: x.to_timestamp() if hasattr(x, 'to_timestamp') else x)
            factors.columns = ['Date'] + list(factors.columns[1:])
        factors['Date'] = pd.to_datetime(factors['Date'])
        data['factors'] = factors
        logger.info(f"✓ Factors: {len(factors)} rows")
    except Exception as e:
        logger.warning(f"⚠ Factors not found: {e}")
        data['factors'] = None

    # IV data
    try:
        iv = pd.read_parquet(config.RAW_DIR / "iv_data.parquet")
        iv['date'] = pd.to_datetime(iv['date'])
        data['iv'] = iv
        logger.info(f"✓ IV data: {len(iv)} rows, starts {iv['date'].min()}")
    except Exception as e:
        logger.warning(f"⚠ IV data not found: {e}")
        data['iv'] = None

    return data

def compute_returns(prices):
    """Compute daily returns."""
    logger.info("Computing returns...")

    prices = prices.sort_values(['ticker', 'Date'])

    # Log returns
    prices['log_ret'] = prices.groupby('ticker')['Close'].transform(
        lambda x: np.log(x / x.shift(1))
    )

    # Simple returns
    prices['ret'] = prices.groupby('ticker')['Close'].transform(
        lambda x: x.pct_change()
    )

    logger.info(f"  ✓ Computed returns for {prices['ticker'].nunique()} tickers")

    return prices

def compute_market_returns(sp500):
    """Compute market returns."""
    if sp500 is None:
        return None

    logger.info("Computing market returns...")

    sp500 = sp500.sort_values('Date')
    sp500['mkt_ret'] = sp500['Close'].pct_change()
    sp500['mkt_log_ret'] = np.log(sp500['Close'] / sp500['Close'].shift(1))

    return sp500[['Date', 'mkt_ret', 'mkt_log_ret']]

def determine_t0(row):
    """Determine t0: first trading day the market can react."""
    ann_datetime = row['ann_datetime']
    timing = row['timing']

    # If BMO, market reacts same day
    if timing == 'BMO':
        return ann_datetime.date()
    # If AMC, market reacts next trading day
    elif timing == 'AMC':
        return (ann_datetime + timedelta(days=1)).date()
    else:
        # Unknown timing, conservative: next day
        return (ann_datetime + timedelta(days=1)).date()

def build_events_table(data):
    """Build events table with SUE and abnormal returns."""
    logger.info("=" * 60)
    logger.info("BUILDING EVENTS TABLE")
    logger.info("=" * 60)

    earnings = data['earnings'].copy()
    prices = data['prices'].copy()
    membership = data['membership']
    sp500_ret = compute_market_returns(data['sp500'])
    factors = data['factors']

    # Compute returns if not already done
    if 'ret' not in prices.columns:
        prices = compute_returns(prices)

    # Determine t0
    logger.info("Determining t0 (first reaction date)...")
    earnings['t0'] = earnings.apply(determine_t0, axis=1)
    earnings['t0'] = pd.to_datetime(earnings['t0'])

    # Check if in index on announcement date
    logger.info("Checking index membership...")
    if membership is not None:
        def was_in_index(row):
            ticker = row['ticker']
            t0 = row['t0']
            mem = membership[membership['ticker_yahoo'] == ticker]
            if mem.empty:
                return False
            return any((mem['start_date'] <= t0) & (mem['end_date'] >= t0))

        earnings['in_index'] = earnings.apply(was_in_index, axis=1)
    else:
        earnings['in_index'] = True  # Assume all in index

    # Get market cap and price at t-1
    logger.info("Getting market cap and price at t-1...")
    def get_t_minus_1_data(row):
        ticker = row['ticker']
        t0 = row['t0']

        # Get price data around t0
        ticker_prices = prices[(prices['ticker'] == ticker) &
                               (prices['Date'] < t0)].sort_values('Date')

        if ticker_prices.empty:
            return pd.Series({'price_t_minus_1': np.nan, 'market_cap_t_minus_1': np.nan})

        last_row = ticker_prices.iloc[-1]
        price = last_row['Close']

        # Market cap (if we have shares outstanding, otherwise use price as proxy)
        market_cap = np.nan  # We don't have shares outstanding

        return pd.Series({
            'price_t_minus_1': price,
            'market_cap_t_minus_1': market_cap
        })

    tminus1_data = earnings.apply(get_t_minus_1_data, axis=1)
    earnings = pd.concat([earnings, tminus1_data], axis=1)

    # Compute SUE (Standardized Unexpected Earnings)
    logger.info("Computing SUE...")

    def compute_sue(group):
        """Compute SUE for a ticker."""
        group = group.sort_values('t0')

        # Raw surprise
        group['surprise_raw'] = group['surprise']

        # SUE: (actual - estimate) / std of last 8 surprises
        group['surprise_std'] = group['surprise_raw'].rolling(
            window=config.SUE_LOOKBACK, min_periods=4
        ).std().shift(1)  # shift to avoid look-ahead

        group['sue'] = group['surprise_raw'] / group['surprise_std']

        # Price-scaled surprise
        group['surprise_price_scaled'] = group['surprise_raw'] / group['price_t_minus_1']

        return group

    earnings = earnings.groupby('ticker').apply(compute_sue).reset_index(drop=True)

    # Compute abnormal returns
    logger.info("Computing abnormal returns...")

    def compute_abnormal_returns(row, window, prices_df, sp500_ret_df, factors_df):
        """Compute abnormal returns for a window."""
        ticker = row['ticker']
        t0 = row['t0']

        start_day = t0 + timedelta(days=window[0])
        end_day = t0 + timedelta(days=window[1])

        # Get ticker returns
        ticker_rets = prices_df[
            (prices_df['ticker'] == ticker) &
            (prices_df['Date'] >= start_day) &
            (prices_df['Date'] <= end_day)
        ][['Date', 'ret']].copy()

        if ticker_rets.empty:
            return np.nan, np.nan

        # Market-adjusted return
        if sp500_ret_df is not None:
            ticker_rets = ticker_rets.merge(sp500_ret_df, on='Date', how='left')
            ticker_rets['abnormal_ret'] = ticker_rets['ret'] - ticker_rets['mkt_ret']
            car_market_adj = ticker_rets['abnormal_ret'].sum()
        else:
            car_market_adj = np.nan

        # CAPM-adjusted return (estimate beta on [-250, -30])
        beta_start = t0 + timedelta(days=config.BETA_WINDOW[0])
        beta_end = t0 + timedelta(days=config.BETA_WINDOW[1])

        beta_data = prices_df[
            (prices_df['ticker'] == ticker) &
            (prices_df['Date'] >= beta_start) &
            (prices_df['Date'] <= beta_end)
        ][['Date', 'ret']].copy()

        if not beta_data.empty and sp500_ret_df is not None:
            beta_data = beta_data.merge(sp500_ret_df, on='Date', how='left')
            beta_data = beta_data.dropna()

            if len(beta_data) > 30:  # Need enough data
                # Simple OLS beta
                cov = np.cov(beta_data['ret'], beta_data['mkt_ret'])[0, 1]
                var = np.var(beta_data['mkt_ret'])
                beta = cov / var if var > 0 else 1.0

                # CAPM abnormal return
                ticker_rets['capm_abnormal_ret'] = ticker_rets['ret'] - beta * ticker_rets['mkt_ret']
                car_capm = ticker_rets['capm_abnormal_ret'].sum()
            else:
                car_capm = np.nan
        else:
            car_capm = np.nan

        return car_market_adj, car_capm

    # Compute CARs for each window
    for window in config.CAR_WINDOWS:
        logger.info(f"  Computing CAR{window}...")

        def compute_window_car(row):
            market_adj, capm = compute_abnormal_returns(row, window, prices, sp500_ret, factors)
            return pd.Series({
                f'car_{window[0]}_{window[1]}_market_adj': market_adj,
                f'car_{window[0]}_{window[1]}_capm': capm
            })

        car_data = earnings.apply(compute_window_car, axis=1)
        earnings = pd.concat([earnings, car_data], axis=1)

    # Save
    output_path = config.PROCESSED_DIR / "events.parquet"
    earnings.to_parquet(output_path)
    logger.info(f"\n✓ Saved {len(earnings)} events to {output_path}")

    return earnings

def build_iv_event_panel(data, events):
    """Build IV event panel (t-10 to t+10 around earnings)."""
    logger.info("=" * 60)
    logger.info("BUILDING IV EVENT PANEL")
    logger.info("=" * 60)

    if data['iv'] is None:
        logger.warning("No IV data available, skipping")
        return None

    iv = data['iv'].copy()
    prices = data['prices'].copy()

    # Compute realized volatility
    logger.info("Computing realized volatility...")

    if 'log_ret' not in prices.columns:
        prices = compute_returns(prices)

    # 21-day rolling RV (annualized)
    prices['rv_21d'] = prices.groupby('ticker')['log_ret'].transform(
        lambda x: x.rolling(config.RV_WINDOW, min_periods=10).std() * np.sqrt(252)
    )

    # Build panel
    logger.info("Building event-time panel...")

    panels = []

    for idx, event in events.iterrows():
        if idx % 500 == 0:
            logger.info(f"  Progress: {idx}/{len(events)}")

        ticker = event['ticker']
        t0 = event['t0']

        # Skip if event before IV data starts (2019)
        if t0 < pd.to_datetime('2019-01-01'):
            continue

        # Get days from t-10 to t+10
        event_window = range(config.EVENT_WINDOW[0], config.EVENT_WINDOW[1] + 1)

        for rel_day in event_window:
            target_date = t0 + timedelta(days=rel_day)

            # Get IV for this day
            iv_row = iv[(iv['symbol'] == ticker) &
                       (iv['date'] == target_date)]

            # Get RV for this day
            rv_row = prices[(prices['ticker'] == ticker) &
                           (prices['Date'] == target_date)]

            if not iv_row.empty:
                iv_val = iv_row.iloc[0]['iv'] if 'iv' in iv_row.columns else np.nan
            else:
                iv_val = np.nan

            if not rv_row.empty:
                rv_val = rv_row.iloc[0]['rv_21d']
            else:
                rv_val = np.nan

            panels.append({
                'event_id': idx,
                'ticker': ticker,
                't0': t0,
                'rel_day': rel_day,
                'date': target_date,
                'iv': iv_val,
                'rv': rv_val,
                'iv_rv_spread': iv_val - rv_val if not np.isnan(iv_val) and not np.isnan(rv_val) else np.nan
            })

    if panels:
        panel_df = pd.DataFrame(panels)
        output_path = config.PROCESSED_DIR / "iv_event_panel.parquet"
        panel_df.to_parquet(output_path)
        logger.info(f"\n✓ Saved {len(panel_df)} event-day observations to {output_path}")
        logger.info(f"  Events with IV coverage: {panel_df['event_id'].nunique()}")

        return panel_df

    logger.warning("No IV event panel data created")
    return None

def build_daily_panel(data):
    """Build daily panel for all stocks."""
    logger.info("=" * 60)
    logger.info("BUILDING DAILY PANEL")
    logger.info("=" * 60)

    prices = data['prices'].copy()
    sp500_ret = compute_market_returns(data['sp500'])
    factors = data['factors']

    # Compute returns
    if 'ret' not in prices.columns:
        prices = compute_returns(prices)

    # Merge market returns
    if sp500_ret is not None:
        prices = prices.merge(sp500_ret, on='Date', how='left')

    # Merge factors
    if factors is not None:
        prices = prices.merge(factors, on='Date', how='left')

    # Compute realized vol
    prices['rv_21d'] = prices.groupby('ticker')['log_ret'].transform(
        lambda x: x.rolling(config.RV_WINDOW, min_periods=10).std() * np.sqrt(252)
    )

    # Save
    output_path = config.PROCESSED_DIR / "daily_panel.parquet"
    prices.to_parquet(output_path)
    logger.info(f"\n✓ Saved {len(prices)} rows to {output_path}")

    return prices

def main():
    """Main execution."""
    # Load data
    data = load_data()

    # Build events table
    events = build_events_table(data)

    # Build IV event panel
    iv_panel = build_iv_event_panel(data, events)

    # Build daily panel
    daily_panel = build_daily_panel(data)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Events: {len(events)}")
    logger.info(f"  With SUE: {events['sue'].notna().sum()}")
    logger.info(f"  In index: {events['in_index'].sum()}")
    logger.info(f"IV event panel: {len(iv_panel) if iv_panel is not None else 0}")
    logger.info(f"Daily panel: {len(daily_panel)}")

    logger.info("\n✓ Script complete")

if __name__ == "__main__":
    main()
