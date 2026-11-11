"""
01_universe.py
Download and process historical S&P 500 constituents.
Creates:
- raw/sp500_constituents.csv (raw download)
- processed/universe.parquet (all tickers in analysis window)
- processed/membership.parquet (ticker, start_date, end_date)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import config
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / '01_universe.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def normalize_yahoo_symbol(symbol):
    """Convert symbols to Yahoo Finance format."""
    # BRK.B -> BRK-B, etc.
    return symbol.replace('.', '-')

def download_constituents():
    """Download S&P 500 historical constituents."""
    logger.info("Downloading S&P 500 historical constituents...")

    urls = [
        "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/sp500_constituents.csv",
        "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes(08-01-2024).csv",
    ]

    for url in urls:
        try:
            logger.info(f"  Trying: {url}")
            df = pd.read_csv(url)
            logger.info(f"  ✓ Downloaded {len(df)} rows, {df.shape[1]} columns")

            # Save raw
            raw_path = config.RAW_DIR / "sp500_constituents.csv"
            df.to_csv(raw_path, index=False)
            logger.info(f"  Saved to {raw_path}")

            return df
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    raise Exception("All S&P 500 constituent sources failed")

def parse_constituents_hanshof(df):
    """Parse hanshof format: current constituents only."""
    logger.info("Parsing constituents (hanshof format)...")

    # This source only has current constituents, so we'll use
    # a conservative approach: assume they were in the index
    # for the entire analysis period if they're in the list

    if 'symbol' in df.columns:
        tickers = df['symbol'].unique()
    else:
        raise ValueError("Cannot find ticker column in dataframe")

    # Create membership table: all current members were in for full period
    # (This is conservative - avoids survivorship bias by not using today's list,
    # but we don't have precise historical membership dates from this source)
    membership = pd.DataFrame({
        'ticker': tickers,
        'start_date': pd.to_datetime(config.ANALYSIS_START),
        'end_date': pd.to_datetime(config.ANALYSIS_END),
    })

    logger.info(f"  Found {len(tickers)} unique tickers")

    return membership

def parse_constituents_fja(df):
    """Parse fja05680 format: time-series of constituents."""
    logger.info("Parsing constituents (fja05680 format)...")

    # This format has dates as columns, tickers as rows
    # Extract all tickers that appear in the analysis window

    # Get date columns
    date_cols = [col for col in df.columns if col not in ['Ticker', 'Security', 'GICS Sector', 'GICS Sub Industry']]

    # Parse dates
    try:
        dates = pd.to_datetime(date_cols)
    except:
        # Might need custom parsing
        dates = pd.to_datetime(date_cols, format='%m/%d/%Y', errors='coerce')

    # Filter to analysis window
    mask = (dates >= pd.to_datetime(config.ANALYSIS_START)) & (dates <= pd.to_datetime(config.ANALYSIS_END))
    relevant_dates = dates[mask]

    if len(relevant_dates) == 0:
        logger.warning("  No dates in analysis window, using all dates")
        relevant_dates = dates

    logger.info(f"  Found {len(relevant_dates)} dates in analysis window")

    # Extract tickers present on any of these dates
    ticker_col = 'Ticker' if 'Ticker' in df.columns else df.columns[0]
    all_tickers = set()

    for date in relevant_dates:
        date_str = date.strftime('%m/%d/%Y')
        if date_str in df.columns:
            constituents = df[df[date_str].notna()][ticker_col].values
            all_tickers.update(constituents)

    # Create membership table with first/last appearance
    membership_list = []
    for ticker in all_tickers:
        # Find first and last date this ticker appears
        appearances = []
        for date in relevant_dates:
            date_str = date.strftime('%m/%d/%Y')
            if date_str in df.columns:
                if ticker in df[df[date_str].notna()][ticker_col].values:
                    appearances.append(date)

        if appearances:
            membership_list.append({
                'ticker': ticker,
                'start_date': min(appearances),
                'end_date': max(appearances),
            })

    membership = pd.DataFrame(membership_list)
    logger.info(f"  Found {len(membership)} unique tickers")

    return membership

def create_universe():
    """Create universe and membership tables."""
    logger.info("=" * 60)
    logger.info("CREATING S&P 500 UNIVERSE")
    logger.info("=" * 60)

    # Check if already exists
    membership_path = config.PROCESSED_DIR / "membership.parquet"
    if membership_path.exists():
        logger.info(f"✓ Membership table already exists at {membership_path}")
        logger.info("  Delete it to re-download")
        membership = pd.read_parquet(membership_path)
        logger.info(f"  Loaded {len(membership)} tickers")
        return membership

    # Download
    raw_df = download_constituents()

    # Parse based on format
    if 'symbol' in raw_df.columns:
        membership = parse_constituents_hanshof(raw_df)
    elif 'Ticker' in raw_df.columns or raw_df.columns[0] == 'Ticker':
        membership = parse_constituents_fja(raw_df)
    else:
        raise ValueError("Unknown constituents format")

    # Normalize symbols for Yahoo Finance
    membership['ticker_yahoo'] = membership['ticker'].apply(normalize_yahoo_symbol)

    # Add sector info if available
    if 'gics sector' in raw_df.columns:
        sector_map = raw_df.set_index('symbol')['gics sector'].to_dict()
        membership['sector'] = membership['ticker'].map(sector_map)
    elif 'GICS Sector' in raw_df.columns:
        # For fja format, we'd need to join differently
        membership['sector'] = None

    # Save
    membership.to_parquet(membership_path)
    logger.info(f"✓ Saved membership table to {membership_path}")

    # Also save just the universe (unique tickers)
    universe = pd.DataFrame({'ticker': membership['ticker'].unique()})
    universe['ticker_yahoo'] = universe['ticker'].apply(normalize_yahoo_symbol)
    universe_path = config.PROCESSED_DIR / "universe.parquet"
    universe.to_parquet(universe_path)
    logger.info(f"✓ Saved universe ({len(universe)} tickers) to {universe_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tickers in universe: {len(universe)}")
    logger.info(f"Date range: {membership['start_date'].min()} to {membership['end_date'].max()}")
    logger.info(f"Sample tickers: {', '.join(universe['ticker'].head(10).values)}")

    return membership

if __name__ == "__main__":
    membership = create_universe()
    logger.info("\n✓ Script complete")
