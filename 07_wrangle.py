"""
07_wrangle.py
Data wrangling for "Do Markets Overreact to Earnings?" project.

This script:
1. Maps earnings announcements to t0 (first reaction date) using NYSE trading calendar
2. Validates event dates via abnormal volume spikes
3. Filters universe to S&P 500 members at t0
4. Removes duplicates (one event per ticker per fiscal quarter)
5. Ensures sufficient price history (250 days before, 20 days after)
6. Flags overlapping/confounding events and clustered announcement days
7. Checks split-adjustment consistency
8. Joins contextual features (market cap, sector, momentum, volatility, IV)

Output: processed/events_clean.parquet, wrangling_log.md
"""

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from pathlib import Path
from datetime import datetime, timedelta
import logging
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / '07_wrangle.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Track wrangling stats
wrangling_stats = {
    'initial_events': 0,
    'after_t0_mapping': 0,
    'after_t0_validation': 0,
    'after_universe_filter': 0,
    'after_dedup': 0,
    'after_price_window': 0,
    'after_confounding': 0,
    'split_adjustments': 0,
    'events_flagged_unknown_timing': 0,
    'events_flagged_wrong_volume_peak': 0,
    'events_flagged_clustered': 0,
    'events_flagged_confounding': 0,
    'final_events': 0,
}

issues_log = []

def load_data():
    """Load all required data."""
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)

    # Earnings
    earnings = pd.read_parquet(config.PROCESSED_DIR / "earnings_merged.parquet")
    earnings['datetime'] = pd.to_datetime(earnings['datetime'])
    logger.info(f"✓ Earnings: {len(earnings)} events")

    # Prices
    prices = pd.read_parquet(config.PROCESSED_DIR / "prices_all.parquet")
    prices['Date'] = pd.to_datetime(prices['Date'])
    logger.info(f"✓ Prices: {len(prices):,} rows, {prices['ticker'].nunique()} tickers")

    # S&P 500 index
    sp500 = pd.read_parquet(config.PROCESSED_DIR / "prices_sp500.parquet")
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    logger.info(f"✓ S&P 500: {len(sp500)} days")

    # Membership
    membership = pd.read_parquet(config.PROCESSED_DIR / "membership.parquet")
    membership['start_date'] = pd.to_datetime(membership['start_date'])
    membership['end_date'] = pd.to_datetime(membership['end_date'])
    logger.info(f"✓ Membership: {len(membership)} records")

    # Factors
    factors = pd.read_parquet(config.RAW_DIR / "ff_factors.parquet")
    if hasattr(factors.index, 'to_timestamp'):
        factors.index = factors.index.to_timestamp()
    else:
        factors.index = pd.to_datetime(factors.index)
    logger.info(f"✓ Factors: {len(factors)} days")

    # IV data (if exists)
    iv_path = config.RAW_DIR / "iv_data.parquet"
    if iv_path.exists():
        iv_data = pd.read_parquet(iv_path)
        iv_data['date'] = pd.to_datetime(iv_data['date'])
        logger.info(f"✓ IV data: {len(iv_data)} rows")
    else:
        iv_data = None
        logger.warning("⚠ IV data not found - will be NA for all events")

    return {
        'earnings': earnings,
        'prices': prices,
        'sp500': sp500,
        'membership': membership,
        'factors': factors,
        'iv_data': iv_data
    }

def get_nyse_calendar():
    """Get NYSE trading calendar."""
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(
        start_date=config.PRICE_START,
        end_date=config.PRICE_END
    )
    trading_days = mcal.date_range(schedule, frequency='1D')
    return pd.DatetimeIndex(trading_days.date).sort_values()

def map_to_t0(earnings, trading_calendar):
    """
    Map each earnings announcement to t0 (first reaction date).
    BMO -> same day, AMC -> next trading day, unknown -> infer from volume.
    """
    logger.info("=" * 60)
    logger.info("MAPPING ANNOUNCEMENTS TO T0")
    logger.info("=" * 60)

    wrangling_stats['initial_events'] = len(earnings)

    # Convert announcement datetime to date
    earnings['ann_date'] = earnings['datetime'].dt.date
    earnings['ann_date'] = pd.to_datetime(earnings['ann_date'])

    # Initialize t0
    earnings['t0'] = pd.NaT
    earnings['t0_method'] = ''

    # BMO: t0 = announcement date (if trading day)
    bmo_mask = earnings['timing'] == 'BMO'
    logger.info(f"BMO events: {bmo_mask.sum()}")

    for idx in earnings[bmo_mask].index:
        ann_date = earnings.loc[idx, 'ann_date']
        if ann_date in trading_calendar:
            earnings.loc[idx, 't0'] = ann_date
            earnings.loc[idx, 't0_method'] = 'BMO_same_day'
        else:
            # If announced on non-trading day, use next trading day
            next_day = trading_calendar[trading_calendar > ann_date].min()
            earnings.loc[idx, 't0'] = next_day
            earnings.loc[idx, 't0_method'] = 'BMO_next_trading_day'

    # AMC: t0 = next trading day
    amc_mask = earnings['timing'] == 'AMC'
    logger.info(f"AMC events: {amc_mask.sum()}")

    for idx in earnings[amc_mask].index:
        ann_date = earnings.loc[idx, 'ann_date']
        next_day = trading_calendar[trading_calendar > ann_date].min()
        earnings.loc[idx, 't0'] = next_day
        earnings.loc[idx, 't0_method'] = 'AMC_next_day'

    # Unknown timing: will infer from volume later
    unknown_mask = earnings['timing'].isin(['unknown', 'trading_hours'])
    logger.info(f"Unknown/trading hours events: {unknown_mask.sum()}")
    earnings.loc[unknown_mask, 't0_method'] = 'infer_from_volume'
    wrangling_stats['events_flagged_unknown_timing'] = unknown_mask.sum()

    wrangling_stats['after_t0_mapping'] = (earnings['t0'].notna()).sum()
    logger.info(f"✓ Mapped {wrangling_stats['after_t0_mapping']} events to t0")

    return earnings, trading_calendar

def infer_t0_from_volume(earnings, prices, trading_calendar):
    """
    For events with unknown timing, infer t0 from abnormal volume.
    Use the day with larger abnormal volume (ann_date vs ann_date+1).
    """
    logger.info("=" * 60)
    logger.info("INFERRING T0 FROM VOLUME (unknown timing)")
    logger.info("=" * 60)

    unknown_mask = earnings['t0_method'] == 'infer_from_volume'
    unknown_events = earnings[unknown_mask].copy()

    logger.info(f"Events to infer: {len(unknown_events)}")

    inferred = 0
    failed = 0

    for idx, row in unknown_events.iterrows():
        ticker = row['ticker']
        ann_date = row['ann_date']

        # Get price data for this ticker
        ticker_prices = prices[prices['ticker'] == ticker].copy()
        ticker_prices = ticker_prices.sort_values('Date')

        # Calculate rolling median volume (60-day lookback)
        ticker_prices['volume_med'] = ticker_prices['Volume'].rolling(
            window=config.VOLUME_LOOKBACK, min_periods=20
        ).median()

        # Get volume on ann_date and next trading day
        day0 = ann_date if ann_date in trading_calendar else trading_calendar[trading_calendar > ann_date].min()
        day1_idx = trading_calendar.get_loc(day0) + 1
        if day1_idx >= len(trading_calendar):
            failed += 1
            continue
        day1 = trading_calendar[day1_idx]

        # Get abnormal volume
        vol0_data = ticker_prices[ticker_prices['Date'] == day0]
        vol1_data = ticker_prices[ticker_prices['Date'] == day1]

        if len(vol0_data) == 0 or len(vol1_data) == 0:
            failed += 1
            continue

        vol0 = vol0_data['Volume'].iloc[0]
        vol1 = vol1_data['Volume'].iloc[0]
        vol_med0 = vol0_data['volume_med'].iloc[0]
        vol_med1 = vol1_data['volume_med'].iloc[0]

        if pd.isna(vol_med0) or pd.isna(vol_med1):
            failed += 1
            continue

        abnormal0 = vol0 / vol_med0 if vol_med0 > 0 else 0
        abnormal1 = vol1 / vol_med1 if vol_med1 > 0 else 0

        # Choose day with higher abnormal volume
        if abnormal0 >= abnormal1:
            earnings.loc[idx, 't0'] = day0
            earnings.loc[idx, 't0_method'] = 'inferred_day0'
        else:
            earnings.loc[idx, 't0'] = day1
            earnings.loc[idx, 't0_method'] = 'inferred_day1'

        inferred += 1

    logger.info(f"✓ Inferred t0 for {inferred} events")
    logger.info(f"✗ Failed to infer {failed} events (will drop)")

    # Drop events where t0 couldn't be determined
    earnings = earnings[earnings['t0'].notna()].copy()

    wrangling_stats['after_t0_validation'] = len(earnings)

    return earnings

def validate_t0_with_volume(earnings, prices):
    """
    Validate t0 by checking that volume peaks at t0 or t0+1.
    Flag events where volume peak in [-3,+3] is elsewhere.
    """
    logger.info("=" * 60)
    logger.info("VALIDATING T0 WITH VOLUME SPIKES")
    logger.info("=" * 60)

    earnings['volume_peak_correct'] = True
    wrong_peak_count = 0
    checked_count = 0

    sample_issues = []

    for idx, row in earnings.iterrows():
        ticker = row['ticker']
        t0 = row['t0']

        # Get price data
        ticker_prices = prices[prices['ticker'] == ticker].copy()
        ticker_prices = ticker_prices.sort_values('Date')

        # Calculate rolling median volume
        ticker_prices['volume_med'] = ticker_prices['Volume'].rolling(
            window=config.VOLUME_LOOKBACK, min_periods=20
        ).median()

        ticker_prices['abnormal_volume'] = ticker_prices['Volume'] / ticker_prices['volume_med']

        # Get window [-3, +3] around t0
        window_data = ticker_prices[
            (ticker_prices['Date'] >= t0 - timedelta(days=5)) &
            (ticker_prices['Date'] <= t0 + timedelta(days=5))
        ].copy()

        if len(window_data) < 3:
            continue

        # Skip if all abnormal volumes are NA
        if window_data['abnormal_volume'].isna().all():
            continue

        checked_count += 1

        # Find day with max abnormal volume
        peak_day = window_data.loc[window_data['abnormal_volume'].idxmax(), 'Date']

        # Check if peak is at t0 or t0+1
        t0_next = t0 + timedelta(days=1)

        if peak_day not in [t0, t0_next]:
            earnings.loc[idx, 'volume_peak_correct'] = False
            wrong_peak_count += 1

            # Sample for manual inspection
            if len(sample_issues) < 10:
                sample_issues.append({
                    'ticker': ticker,
                    't0': t0,
                    'peak_day': peak_day,
                    'timing': row['timing'],
                    't0_method': row['t0_method']
                })

    logger.info(f"Checked {checked_count} events")
    logger.info(f"Volume peak NOT at t0/t0+1: {wrong_peak_count} ({100*wrong_peak_count/checked_count:.1f}%)")

    if sample_issues:
        logger.info("\nSample of events with wrong volume peak:")
        for issue in sample_issues:
            logger.info(f"  {issue['ticker']} t0={issue['t0'].date()} peak={issue['peak_day'].date()} timing={issue['timing']}")

    wrangling_stats['events_flagged_wrong_volume_peak'] = wrong_peak_count

    issues_log.append({
        'rule': 'Volume peak validation',
        'events_flagged': wrong_peak_count,
        'description': f'{wrong_peak_count} events have volume peak outside t0/t0+1 window',
        'action': 'Flagged but not dropped - may indicate incorrect announcement date'
    })

    return earnings

def filter_sp500_universe(earnings, membership):
    """Keep only events where stock was in S&P 500 on t0."""
    logger.info("=" * 60)
    logger.info("FILTERING TO S&P 500 UNIVERSE")
    logger.info("=" * 60)

    before = len(earnings)

    earnings['in_sp500_at_t0'] = False

    for idx, row in earnings.iterrows():
        ticker = row['ticker']
        t0 = row['t0']

        # Check if ticker was in S&P 500 at t0
        ticker_membership = membership[membership['ticker'] == ticker]

        if len(ticker_membership) == 0:
            continue

        for _, mem in ticker_membership.iterrows():
            if pd.isna(mem['start_date']):
                continue

            end_date = mem['end_date'] if pd.notna(mem['end_date']) else pd.Timestamp('2099-12-31')

            if mem['start_date'] <= t0 <= end_date:
                earnings.loc[idx, 'in_sp500_at_t0'] = True
                break

    earnings = earnings[earnings['in_sp500_at_t0']].copy()

    after = len(earnings)
    dropped = before - after

    logger.info(f"✓ Kept {after} events in S&P 500 at t0")
    logger.info(f"✗ Dropped {dropped} events (not in index at t0)")

    wrangling_stats['after_universe_filter'] = after

    issues_log.append({
        'rule': 'S&P 500 membership filter',
        'events_removed': dropped,
        'description': 'Stock must be in S&P 500 index on t0',
        'reason': 'Ensures consistent universe and avoids survivor bias'
    })

    return earnings

def remove_duplicates(earnings):
    """
    Remove duplicate events: one event per ticker per fiscal quarter.
    Keep the event with the most complete data.
    """
    logger.info("=" * 60)
    logger.info("REMOVING DUPLICATES")
    logger.info("=" * 60)

    before = len(earnings)

    # Create fiscal quarter identifier
    earnings['fiscal_qtr'] = earnings['datetime'].dt.to_period('Q')

    # Count duplicates
    duplicates = earnings.groupby(['ticker', 'fiscal_qtr']).size()
    n_duplicates = (duplicates > 1).sum()

    logger.info(f"Found {n_duplicates} ticker-quarters with multiple events")

    # For duplicates, keep the one with the most complete EPS data
    earnings['eps_completeness'] = (
        earnings['eps_actual_yf'].notna().astype(int) +
        earnings['eps_estimate_yf'].notna().astype(int)
    )

    # Sort by completeness (descending) and datetime (ascending)
    earnings = earnings.sort_values(
        ['ticker', 'fiscal_qtr', 'eps_completeness', 'datetime'],
        ascending=[True, True, False, True]
    )

    # Keep first (most complete, earliest)
    earnings = earnings.drop_duplicates(subset=['ticker', 'fiscal_qtr'], keep='first')

    after = len(earnings)
    dropped = before - after

    logger.info(f"✓ Kept {after} unique events")
    logger.info(f"✗ Dropped {dropped} duplicate events")

    wrangling_stats['after_dedup'] = after

    issues_log.append({
        'rule': 'Duplicate removal',
        'events_removed': dropped,
        'description': 'One event per ticker per fiscal quarter',
        'reason': 'Multiple announcements for same quarter (restatements, duplicates in source data)'
    })

    return earnings

def check_price_data_window(earnings, prices):
    """
    Ensure each event has sufficient price history.
    Need 250 days before t0 and 20 days after for beta estimation and CAR.
    """
    logger.info("=" * 60)
    logger.info("CHECKING PRICE DATA WINDOWS")
    logger.info("=" * 60)

    before = len(earnings)

    earnings['has_price_history'] = False
    earnings['has_price_future'] = False

    for idx, row in earnings.iterrows():
        ticker = row['ticker']
        t0 = row['t0']

        ticker_prices = prices[prices['ticker'] == ticker].copy()
        ticker_prices = ticker_prices.sort_values('Date')

        # Check history
        history = ticker_prices[ticker_prices['Date'] < t0]
        if len(history) >= config.MIN_PRICE_HISTORY:
            earnings.loc[idx, 'has_price_history'] = True

        # Check future
        future = ticker_prices[ticker_prices['Date'] > t0]
        if len(future) >= config.MIN_PRICE_FUTURE:
            earnings.loc[idx, 'has_price_future'] = True

    earnings = earnings[
        earnings['has_price_history'] & earnings['has_price_future']
    ].copy()

    after = len(earnings)
    dropped = before - after

    logger.info(f"✓ Kept {after} events with sufficient price data")
    logger.info(f"✗ Dropped {dropped} events (insufficient price history/future)")

    wrangling_stats['after_price_window'] = after

    issues_log.append({
        'rule': 'Price data window',
        'events_removed': dropped,
        'description': f'Need {config.MIN_PRICE_HISTORY} days before t0 and {config.MIN_PRICE_FUTURE} days after',
        'reason': 'Required for beta estimation and CAR calculation'
    })

    return earnings

def flag_confounding_events(earnings):
    """
    Flag events with potential confounding:
    - Another earnings announcement within +/- 20 days
    - Stock splits, large dividends, etc. (placeholder - would need corporate actions data)
    """
    logger.info("=" * 60)
    logger.info("FLAGGING CONFOUNDING EVENTS")
    logger.info("=" * 60)

    earnings = earnings.sort_values(['ticker', 't0'])
    earnings['has_confounding'] = False
    earnings['confounding_reason'] = ''

    confounding_count = 0

    # Check for overlapping events (same ticker)
    for ticker in earnings['ticker'].unique():
        ticker_events = earnings[earnings['ticker'] == ticker].copy()

        for idx, row in ticker_events.iterrows():
            t0 = row['t0']

            # Check if another event within +/- 20 days
            nearby_events = ticker_events[
                (ticker_events['t0'] != t0) &
                (ticker_events['t0'] >= t0 - timedelta(days=20)) &
                (ticker_events['t0'] <= t0 + timedelta(days=20))
            ]

            if len(nearby_events) > 0:
                earnings.loc[idx, 'has_confounding'] = True
                earnings.loc[idx, 'confounding_reason'] = 'overlapping_earnings'
                confounding_count += 1

    logger.info(f"Flagged {confounding_count} events with potential confounding")

    wrangling_stats['events_flagged_confounding'] = confounding_count
    wrangling_stats['after_confounding'] = len(earnings)  # Don't drop, just flag

    issues_log.append({
        'rule': 'Confounding events',
        'events_flagged': confounding_count,
        'description': 'Events with another earnings announcement within ±20 days',
        'action': 'Flagged but not dropped - can exclude in robustness checks'
    })

    return earnings

def flag_clustered_dates(earnings):
    """Flag dates where > CLUSTER_THRESHOLD S&P names report."""
    logger.info("=" * 60)
    logger.info("FLAGGING CLUSTERED ANNOUNCEMENT DATES")
    logger.info("=" * 60)

    # Count events per date
    events_per_date = earnings.groupby('t0').size()
    clustered_dates = events_per_date[events_per_date > config.CLUSTER_THRESHOLD].index

    earnings['is_clustered'] = earnings['t0'].isin(clustered_dates)

    n_clustered = earnings['is_clustered'].sum()

    logger.info(f"Found {len(clustered_dates)} clustered dates (>{config.CLUSTER_THRESHOLD} events)")
    logger.info(f"Flagged {n_clustered} events on clustered dates")

    wrangling_stats['events_flagged_clustered'] = n_clustered

    issues_log.append({
        'rule': 'Clustered announcement dates',
        'events_flagged': n_clustered,
        'description': f'Dates with >{config.CLUSTER_THRESHOLD} announcements',
        'action': 'Flagged for clustered standard errors in analysis'
    })

    return earnings

def check_split_adjustment(earnings, prices):
    """
    Check that EPS and prices are on same split-adjusted basis.
    Focus on known splits: AAPL (2020), NVDA (2021, 2024), TSLA.
    """
    logger.info("=" * 60)
    logger.info("CHECKING SPLIT ADJUSTMENTS")
    logger.info("=" * 60)

    # Known splits to check (ticker, split_date, ratio)
    known_splits = [
        ('AAPL', '2020-08-31', 4),
        ('NVDA', '2021-07-20', 4),
        ('NVDA', '2024-06-10', 10),
        ('TSLA', '2020-08-31', 5),
        ('TSLA', '2022-08-25', 3),
    ]

    issues_found = 0

    for ticker, split_date, ratio in known_splits:
        split_date = pd.to_datetime(split_date)

        # Get events before and after split
        ticker_events = earnings[earnings['ticker'] == ticker].copy()
        before_split = ticker_events[ticker_events['t0'] < split_date]
        after_split = ticker_events[ticker_events['t0'] >= split_date]

        if len(before_split) == 0 or len(after_split) == 0:
            continue

        # Check if EPS scaled by split ratio
        eps_before = before_split['eps_actual_yf'].median()
        eps_after = after_split['eps_actual_yf'].median()

        if pd.notna(eps_before) and pd.notna(eps_after):
            expected_ratio = eps_before / eps_after

            if abs(expected_ratio - ratio) > 0.5:  # Tolerance
                logger.warning(f"  {ticker} split {split_date.date()}: "
                             f"EPS ratio={expected_ratio:.2f} (expected {ratio})")
                issues_found += 1

    if issues_found == 0:
        logger.info("✓ No split adjustment issues detected in known cases")
    else:
        logger.warning(f"⚠ Found {issues_found} potential split adjustment issues")

    wrangling_stats['split_adjustments'] = issues_found

    return earnings

def join_features(earnings, prices, sp500, membership, factors, iv_data):
    """
    Join contextual features as of t-1:
    - Market cap (price × shares_out proxy)
    - Sector
    - 12-1 month momentum
    - Pre-event realized volatility (21-day)
    - Pre-event IV and IV/RV ratio (if available)
    """
    logger.info("=" * 60)
    logger.info("JOINING CONTEXTUAL FEATURES")
    logger.info("=" * 60)

    # Prepare price data with returns
    prices = prices.sort_values(['ticker', 'Date'])
    prices['ret'] = prices.groupby('ticker')['Close'].pct_change()

    # Join sector from membership
    sector_map = membership[['ticker', 'sector']].drop_duplicates('ticker')
    earnings = earnings.merge(sector_map, on='ticker', how='left')

    # For each event, compute features as of t-1
    feature_cols = [
        'price_tminus1', 'mkt_cap_proxy', 'momentum_12_1',
        'rv_21d', 'iv_tminus1', 'iv_rv_ratio'
    ]

    for col in feature_cols:
        earnings[col] = np.nan

    for idx, row in earnings.iterrows():
        ticker = row['ticker']
        t0 = row['t0']
        tminus1 = t0 - timedelta(days=1)

        # Get ticker prices
        ticker_prices = prices[prices['ticker'] == ticker].copy()

        # Price at t-1
        price_data = ticker_prices[ticker_prices['Date'] <= tminus1]
        if len(price_data) > 0:
            earnings.loc[idx, 'price_tminus1'] = price_data.iloc[-1]['Close']

        # Market cap proxy (price × volume can proxy for shares)
        # Better: use shares outstanding if available
        # For now, just use price as proxy for size

        # 12-1 month momentum
        # Need prices from t-252 to t-21
        mom_start = t0 - timedelta(days=365)
        mom_end = t0 - timedelta(days=30)
        mom_prices = ticker_prices[
            (ticker_prices['Date'] >= mom_start) &
            (ticker_prices['Date'] <= mom_end)
        ]
        if len(mom_prices) >= 2:
            momentum = (mom_prices.iloc[-1]['Close'] / mom_prices.iloc[0]['Close']) - 1
            earnings.loc[idx, 'momentum_12_1'] = momentum

        # Realized volatility (21-day)
        rv_window = ticker_prices[
            (ticker_prices['Date'] > tminus1 - timedelta(days=30)) &
            (ticker_prices['Date'] <= tminus1)
        ]
        if len(rv_window) >= config.RV_WINDOW:
            rv = rv_window['ret'].std() * np.sqrt(252)  # Annualized
            earnings.loc[idx, 'rv_21d'] = rv

        # IV data (if available)
        if iv_data is not None:
            # Get IV at t-1
            ticker_iv = iv_data[
                (iv_data['symbol'] == ticker) &
                (iv_data['date'] == tminus1)
            ]
            if len(ticker_iv) > 0:
                # Assuming column is 'iv' or similar - adjust based on actual data
                if 'iv' in ticker_iv.columns:
                    earnings.loc[idx, 'iv_tminus1'] = ticker_iv.iloc[0]['iv']

                # IV/RV ratio
                if pd.notna(earnings.loc[idx, 'rv_21d']) and pd.notna(earnings.loc[idx, 'iv_tminus1']):
                    earnings.loc[idx, 'iv_rv_ratio'] = (
                        earnings.loc[idx, 'iv_tminus1'] / earnings.loc[idx, 'rv_21d']
                    )

    logger.info("✓ Feature coverage:")
    for col in feature_cols:
        coverage = earnings[col].notna().sum()
        pct = 100 * coverage / len(earnings)
        logger.info(f"  {col}: {coverage} ({pct:.1f}%)")

    return earnings

def save_wrangling_log():
    """Save wrangling statistics and issues log to markdown."""
    logger.info("=" * 60)
    logger.info("SAVING WRANGLING LOG")
    logger.info("=" * 60)

    log_path = config.BASE_DIR / "wrangling_log.md"

    with open(log_path, 'w') as f:
        f.write("# Data Wrangling Log\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary Statistics\n\n")
        f.write("| Stage | Events | Change |\n")
        f.write("|-------|--------|--------|\n")

        stages = [
            ('Initial events', 'initial_events', None),
            ('After t0 mapping', 'after_t0_mapping', 'initial_events'),
            ('After t0 validation', 'after_t0_validation', 'after_t0_mapping'),
            ('After S&P 500 filter', 'after_universe_filter', 'after_t0_validation'),
            ('After deduplication', 'after_dedup', 'after_universe_filter'),
            ('After price window check', 'after_price_window', 'after_dedup'),
            ('After confounding check', 'after_confounding', 'after_price_window'),
            ('Final events', 'final_events', 'after_confounding'),
        ]

        for stage_name, key, prev_key in stages:
            count = wrangling_stats.get(key, 0)
            if prev_key:
                prev_count = wrangling_stats.get(prev_key, 0)
                change = count - prev_count
                change_str = f"{change:+d}" if change != 0 else "0"
            else:
                change_str = ""

            f.write(f"| {stage_name} | {count:,} | {change_str} |\n")

        f.write("\n## Flags\n\n")
        f.write("| Flag | Count | Description |\n")
        f.write("|------|-------|-------------|\n")

        flags = [
            ('Unknown timing (inferred from volume)', 'events_flagged_unknown_timing'),
            ('Wrong volume peak', 'events_flagged_wrong_volume_peak'),
            ('Clustered announcement dates', 'events_flagged_clustered'),
            ('Confounding events', 'events_flagged_confounding'),
        ]

        for flag_name, key in flags:
            count = wrangling_stats.get(key, 0)
            f.write(f"| {flag_name} | {count:,} | |\n")

        f.write("\n## Rules Applied\n\n")

        for issue in issues_log:
            f.write(f"### {issue['rule']}\n\n")
            if 'events_removed' in issue:
                f.write(f"**Events removed**: {issue['events_removed']:,}\n\n")
            if 'events_flagged' in issue:
                f.write(f"**Events flagged**: {issue['events_flagged']:,}\n\n")
            f.write(f"**Description**: {issue['description']}\n\n")
            if 'reason' in issue:
                f.write(f"**Reason**: {issue['reason']}\n\n")
            if 'action' in issue:
                f.write(f"**Action**: {issue['action']}\n\n")
            f.write("\n")

    logger.info(f"✓ Saved wrangling log to {log_path}")

def main():
    """Main execution."""
    # Load data
    data = load_data()

    # Get trading calendar
    trading_calendar = get_nyse_calendar()
    logger.info(f"✓ NYSE trading calendar: {len(trading_calendar)} days")

    # Map to t0
    earnings, trading_calendar = map_to_t0(data['earnings'], trading_calendar)

    # Infer t0 from volume for unknown timing
    earnings = infer_t0_from_volume(earnings, data['prices'], trading_calendar)

    # Validate t0 with volume spikes
    earnings = validate_t0_with_volume(earnings, data['prices'])

    # Filter to S&P 500 universe
    earnings = filter_sp500_universe(earnings, data['membership'])

    # Remove duplicates
    earnings = remove_duplicates(earnings)

    # Check price data window
    earnings = check_price_data_window(earnings, data['prices'])

    # Flag confounding events
    earnings = flag_confounding_events(earnings)

    # Flag clustered dates
    earnings = flag_clustered_dates(earnings)

    # Check split adjustments
    earnings = check_split_adjustment(earnings, data['prices'])

    # Join features
    earnings = join_features(
        earnings,
        data['prices'],
        data['sp500'],
        data['membership'],
        data['factors'],
        data['iv_data']
    )

    # Final count
    wrangling_stats['final_events'] = len(earnings)

    # Save output
    output_path = config.PROCESSED_DIR / "events_clean.parquet"
    earnings.to_parquet(output_path)
    logger.info(f"\n✓ Saved {len(earnings)} clean events to {output_path}")

    # Save wrangling log
    save_wrangling_log()

    logger.info("\n" + "=" * 60)
    logger.info("WRANGLING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final events: {wrangling_stats['final_events']:,}")
    logger.info(f"Retention rate: {100*wrangling_stats['final_events']/wrangling_stats['initial_events']:.1f}%")

if __name__ == "__main__":
    main()
