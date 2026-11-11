"""
03_earnings.py
Download earnings data from yfinance and Alpha Vantage, then merge.
Creates:
- cache/av_earnings/{TICKER}.json (Alpha Vantage cache)
- raw/earnings_yf.parquet (yfinance earnings)
- raw/earnings_av.parquet (Alpha Vantage earnings)
- processed/earnings_merged.parquet (merged earnings with conflict flags)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
from pathlib import Path
from datetime import datetime, time as dt_time
import time
import logging
import config
import pytz

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / '03_earnings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YFinanceEarningsDownloader:
    """Downloads earnings data from yfinance."""

    def __init__(self):
        self.failed = []
        self.downloaded = []

    def classify_timing(self, dt):
        """Classify earnings announcement as BMO, AMC, or unknown."""
        if pd.isna(dt):
            return 'unknown'

        try:
            # Convert to ET
            if dt.tzinfo is None:
                dt = pytz.timezone('US/Eastern').localize(dt)
            else:
                dt = dt.astimezone(pytz.timezone('US/Eastern'))

            hour = dt.hour
            minute = dt.minute

            # BMO: before 9:30 ET
            if hour < 9 or (hour == 9 and minute < 30):
                return 'BMO'
            # AMC: after 16:00 ET
            elif hour >= 16:
                return 'AMC'
            else:
                return 'trading_hours'

        except:
            return 'unknown'

    def download_ticker(self, ticker):
        """Download earnings dates for a single ticker."""
        try:
            t = yf.Ticker(ticker)

            # Try get_earnings_dates first
            try:
                earnings = t.get_earnings_dates(limit=60)
            except:
                # Fallback to earnings_dates property
                try:
                    earnings = t.earnings_dates
                except:
                    earnings = None

            if earnings is None or earnings.empty:
                logger.debug(f"  {ticker}: No earnings data")
                self.failed.append((ticker, "no_data"))
                return None

            # Reset index to get datetime as column
            df = earnings.reset_index()

            # Standardize column names
            df = df.rename(columns={
                'Earnings Date': 'datetime',
                'EPS Estimate': 'eps_estimate_yf',
                'Reported EPS': 'eps_actual_yf',
                'Surprise(%)': 'surprise_pct_yf'
            })

            # Keep only what we need
            cols_to_keep = ['datetime']
            if 'eps_estimate_yf' in df.columns:
                cols_to_keep.append('eps_estimate_yf')
            if 'eps_actual_yf' in df.columns:
                cols_to_keep.append('eps_actual_yf')
            if 'surprise_pct_yf' in df.columns:
                cols_to_keep.append('surprise_pct_yf')

            df = df[cols_to_keep].copy()

            # Classify timing
            df['timing'] = df['datetime'].apply(self.classify_timing)
            df['ticker'] = ticker

            # Filter to analysis window
            # Convert config dates to timezone-aware if datetime column has timezone
            start_date = pd.to_datetime(config.ANALYSIS_START)
            end_date = pd.to_datetime(config.ANALYSIS_END)

            if df['datetime'].dt.tz is not None:
                # Make comparison dates timezone-aware
                start_date = start_date.tz_localize('UTC')
                end_date = end_date.tz_localize('UTC')

            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

            if df.empty:
                logger.debug(f"  {ticker}: No earnings in analysis window")
                self.failed.append((ticker, "no_data_in_window"))
                return None

            logger.info(f"  {ticker}: ✓ {len(df)} earnings dates")
            self.downloaded.append(ticker)
            return df

        except Exception as e:
            logger.error(f"  {ticker}: Failed - {e}")
            self.failed.append((ticker, str(e)))
            return None

    def download_all(self, tickers):
        """Download earnings for all tickers."""
        logger.info("=" * 60)
        logger.info("DOWNLOADING YFINANCE EARNINGS")
        logger.info("=" * 60)
        logger.info(f"Tickers: {len(tickers)}")

        all_data = []

        for i, ticker in enumerate(tickers, 1):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(tickers)} ({100*i/len(tickers):.1f}%)")

            data = self.download_ticker(ticker)
            if data is not None:
                all_data.append(data)

            time.sleep(config.YF_DELAY)

        return all_data

class AlphaVantageEarningsDownloader:
    """Downloads earnings data from Alpha Vantage with caching and quota management."""

    def __init__(self):
        self.cache_dir = config.CACHE_DIR / "av_earnings"
        self.cache_dir.mkdir(exist_ok=True)
        self.api_key = config.AV_KEY
        self.failed = []
        self.downloaded = []
        self.quota_exceeded = False
        self.calls_today = 0
        self.max_calls = config.AV_CALLS_PER_DAY

    def download_ticker(self, ticker):
        """Download earnings data for a single ticker (with caching)."""
        # Check quota
        if self.quota_exceeded or self.calls_today >= self.max_calls:
            logger.warning(f"  {ticker}: Quota exceeded, skipping")
            self.quota_exceeded = True
            return None

        # Check cache
        cache_path = self.cache_dir / f"{ticker}.json"
        if cache_path.exists():
            logger.debug(f"  {ticker}: Loading from cache")
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return self.parse_response(ticker, data)
            except Exception as e:
                logger.warning(f"  {ticker}: Cache corrupted ({e}), re-downloading")

        # Check API key
        if not self.api_key:
            logger.warning("No Alpha Vantage API key, skipping")
            self.quota_exceeded = True
            return None

        # Download
        try:
            url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={self.api_key}"
            logger.info(f"  Downloading {ticker}...")

            response = requests.get(url)
            data = response.json()

            # Save to cache
            with open(cache_path, 'w') as f:
                json.dump(data, f)

            self.calls_today += 1

            # Check for quota message
            if "Note" in data:
                logger.warning(f"  Quota message: {data['Note']}")
                self.quota_exceeded = True
                return None

            if "Error Message" in data:
                logger.error(f"  {ticker}: {data['Error Message']}")
                self.failed.append((ticker, data['Error Message']))
                return None

            return self.parse_response(ticker, data)

        except Exception as e:
            logger.error(f"  {ticker}: Failed - {e}")
            self.failed.append((ticker, str(e)))
            return None

    def parse_response(self, ticker, data):
        """Parse Alpha Vantage JSON response."""
        if "quarterlyEarnings" not in data:
            logger.debug(f"  {ticker}: No quarterly earnings in response")
            self.failed.append((ticker, "no_quarterly_earnings"))
            return None

        df = pd.DataFrame(data["quarterlyEarnings"])

        if df.empty:
            logger.debug(f"  {ticker}: Empty quarterly earnings")
            self.failed.append((ticker, "empty_data"))
            return None

        # Parse columns
        df = df.rename(columns={
            'reportedDate': 'date',
            'reportedEPS': 'eps_actual_av',
            'estimatedEPS': 'eps_estimate_av',
            'surprise': 'surprise_av',
            'surprisePercentage': 'surprise_pct_av'
        })

        # Convert date
        df['date'] = pd.to_datetime(df['date'])

        # Convert numeric fields
        for col in ['eps_actual_av', 'eps_estimate_av', 'surprise_av', 'surprise_pct_av']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['ticker'] = ticker

        # Filter to analysis window
        start_date = pd.to_datetime(config.ANALYSIS_START)
        end_date = pd.to_datetime(config.ANALYSIS_END)

        # Ensure both are timezone-naive for comparison
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)

        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        if df.empty:
            logger.debug(f"  {ticker}: No earnings in analysis window")
            return None

        logger.info(f"  {ticker}: ✓ {len(df)} earnings dates")
        self.downloaded.append(ticker)
        return df

    def download_all(self, tickers):
        """Download earnings for all tickers."""
        logger.info("=" * 60)
        logger.info("DOWNLOADING ALPHA VANTAGE EARNINGS")
        logger.info("=" * 60)
        logger.info(f"Tickers: {len(tickers)}")

        if not self.api_key:
            logger.warning("No AV_KEY environment variable set, skipping Alpha Vantage")
            logger.warning("  Set with: export AV_KEY='your_key'")
            return []

        all_data = []

        for i, ticker in enumerate(tickers, 1):
            if self.quota_exceeded:
                logger.warning(f"\nQuota exceeded after {i-1} calls")
                logger.warning("Resume tomorrow or with a new API key")
                break

            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(tickers)} ({100*i/len(tickers):.1f}%)")

            data = self.download_ticker(ticker)
            if data is not None:
                all_data.append(data)

            # Rate limiting: 5 calls per minute = 12 seconds between calls
            time.sleep(12.5)

        logger.info(f"\nTotal API calls today: {self.calls_today}/{self.max_calls}")

        return all_data

def merge_earnings(yf_df, av_df):
    """Merge yfinance and Alpha Vantage earnings data."""
    logger.info("=" * 60)
    logger.info("MERGING EARNINGS DATA")
    logger.info("=" * 60)

    if yf_df is None or yf_df.empty:
        logger.warning("No yfinance data to merge")
        return av_df if av_df is not None else pd.DataFrame()

    if av_df is None or av_df.empty:
        logger.warning("No Alpha Vantage data to merge")
        return yf_df

    # Prepare yfinance data
    yf_df['date'] = yf_df['datetime'].dt.date
    yf_df['date'] = pd.to_datetime(yf_df['date'])

    # Merge on ticker and date
    merged = pd.merge(
        yf_df,
        av_df,
        on=['ticker', 'date'],
        how='outer',
        suffixes=('_yf', '_av'),
        indicator=True
    )

    logger.info(f"Total events: {len(merged)}")
    logger.info(f"  yfinance only: {(merged['_merge'] == 'left_only').sum()}")
    logger.info(f"  Alpha Vantage only: {(merged['_merge'] == 'right_only').sum()}")
    logger.info(f"  Both sources: {(merged['_merge'] == 'both').sum()}")

    # Flag conflicts
    both = merged['_merge'] == 'both'

    # Date conflict: check if datetime differs by >1 day (NA for AV since it only has date)
    # Since AV only has date, we can't detect intraday discrepancies

    # EPS conflicts
    has_both_actual = both & merged['eps_actual_yf'].notna() & merged['eps_actual_av'].notna()
    has_both_estimate = both & merged['eps_estimate_yf'].notna() & merged['eps_estimate_av'].notna()

    merged['eps_actual_conflict'] = False
    merged.loc[has_both_actual, 'eps_actual_conflict'] = (
        np.abs(merged.loc[has_both_actual, 'eps_actual_yf'] -
               merged.loc[has_both_actual, 'eps_actual_av']) > config.EPS_DIFF_THRESHOLD
    )

    merged['eps_estimate_conflict'] = False
    merged.loc[has_both_estimate, 'eps_estimate_conflict'] = (
        np.abs(merged.loc[has_both_estimate, 'eps_estimate_yf'] -
               merged.loc[has_both_estimate, 'eps_estimate_av']) > config.EPS_DIFF_THRESHOLD
    )

    n_conflicts = (merged['eps_actual_conflict'] | merged['eps_estimate_conflict']).sum()
    if n_conflicts > 0:
        logger.warning(f"Found {n_conflicts} events with EPS conflicts (diff > ${config.EPS_DIFF_THRESHOLD})")

    # Create consensus columns: prefer AV estimate, use YF as fallback
    merged['eps_estimate'] = merged['eps_estimate_av'].fillna(merged['eps_estimate_yf'])
    merged['eps_actual'] = merged['eps_actual_av'].fillna(merged['eps_actual_yf'])

    # Compute surprise
    merged['surprise'] = merged['eps_actual'] - merged['eps_estimate']

    # Use timing from yfinance (AV doesn't have time)
    if 'datetime' in merged.columns:
        merged['ann_datetime'] = merged['datetime']
    else:
        # If only AV data, use date at midnight
        merged['ann_datetime'] = merged['date']

    # Keep timing from yfinance
    merged['timing'] = merged['timing'].fillna('unknown')

    return merged

def main():
    """Main execution."""
    # Load universe
    logger.info("Loading universe...")
    universe = pd.read_parquet(config.PROCESSED_DIR / "universe.parquet")
    tickers = universe['ticker_yahoo'].tolist()
    logger.info(f"  Loaded {len(tickers)} tickers")

    # Download yfinance earnings
    yf_downloader = YFinanceEarningsDownloader()
    yf_data = yf_downloader.download_all(tickers)

    if yf_data:
        yf_df = pd.concat(yf_data, axis=0, ignore_index=True)
        yf_path = config.RAW_DIR / "earnings_yf.parquet"
        yf_df.to_parquet(yf_path)
        logger.info(f"✓ Saved yfinance earnings: {len(yf_df)} rows to {yf_path}")
    else:
        yf_df = None

    # Download Alpha Vantage earnings
    av_downloader = AlphaVantageEarningsDownloader()
    av_data = av_downloader.download_all(tickers)

    if av_data:
        av_df = pd.concat(av_data, axis=0, ignore_index=True)
        av_path = config.RAW_DIR / "earnings_av.parquet"
        av_df.to_parquet(av_path)
        logger.info(f"✓ Saved Alpha Vantage earnings: {len(av_df)} rows to {av_path}")
    else:
        av_df = None

    # Merge
    merged = merge_earnings(yf_df, av_df)

    if not merged.empty:
        merged_path = config.PROCESSED_DIR / "earnings_merged.parquet"
        merged.to_parquet(merged_path)
        logger.info(f"✓ Saved merged earnings: {len(merged)} rows to {merged_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"yfinance: {len(yf_df) if yf_df is not None else 0} events")
    logger.info(f"Alpha Vantage: {len(av_df) if av_df is not None else 0} events")
    logger.info(f"Merged: {len(merged)} events")

    logger.info("\n✓ Script complete")

if __name__ == "__main__":
    main()
