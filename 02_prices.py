"""
02_prices.py
Download daily price data from yfinance.
Creates:
- raw/prices/{TICKER}.parquet (individual ticker files)
- processed/prices_all.parquet (combined dataset)
- logs/02_prices_failed.csv (list of failed tickers)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime
import time
import logging
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / '02_prices.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PriceDownloader:
    """Downloads and caches price data from yfinance."""

    def __init__(self):
        self.prices_dir = config.RAW_DIR / "prices"
        self.prices_dir.mkdir(exist_ok=True)
        self.failed = []
        self.downloaded = []

    def download_ticker(self, ticker, start, end, retry=0):
        """Download price data for a single ticker."""
        # Check cache first
        cache_path = self.prices_dir / f"{ticker}.parquet"
        if cache_path.exists():
            logger.debug(f"  {ticker}: Loading from cache")
            try:
                df = pd.read_parquet(cache_path)
                # Check if we have the full date range
                if df.index.min() <= pd.to_datetime(start) and df.index.max() >= pd.to_datetime(end):
                    return df
                else:
                    logger.info(f"  {ticker}: Cache incomplete, re-downloading")
            except Exception as e:
                logger.warning(f"  {ticker}: Cache corrupted ({e}), re-downloading")

        # Download
        try:
            logger.info(f"  Downloading {ticker}...")

            # Use Ticker.history() method - more reliable
            t = yf.Ticker(ticker)
            data = t.history(start=start, end=end, auto_adjust=True)

            if data.empty:
                logger.warning(f"  {ticker}: No data returned")
                self.failed.append((ticker, "no_data"))
                return None

            # Reset index to make Date a column
            data = data.reset_index()

            # Flatten multi-index columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Add ticker column
            data['ticker'] = ticker

            # Save to cache
            data.to_parquet(cache_path)
            logger.info(f"  {ticker}: ✓ {len(data)} rows, {data.index.min()} to {data.index.max()}")

            self.downloaded.append(ticker)
            return data

        except Exception as e:
            error_str = str(e)

            # Handle rate limiting
            if "429" in error_str or "Too Many Requests" in error_str:
                if retry < config.YF_MAX_RETRIES:
                    wait_time = config.YF_BACKOFF[retry]
                    logger.warning(f"  {ticker}: Rate limited, waiting {wait_time}s (retry {retry + 1}/{config.YF_MAX_RETRIES})")
                    time.sleep(wait_time)
                    return self.download_ticker(ticker, start, end, retry + 1)
                else:
                    logger.error(f"  {ticker}: Rate limit exceeded, skipping")
                    self.failed.append((ticker, "rate_limit"))
                    return None

            logger.error(f"  {ticker}: Failed - {error_str}")
            self.failed.append((ticker, error_str))
            return None

    def download_all(self, tickers, start, end):
        """Download data for all tickers."""
        logger.info("=" * 60)
        logger.info("DOWNLOADING PRICE DATA")
        logger.info("=" * 60)
        logger.info(f"Tickers: {len(tickers)}")
        logger.info(f"Date range: {start} to {end}")

        all_data = []

        for i, ticker in enumerate(tickers, 1):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(tickers)} ({100*i/len(tickers):.1f}%)")

            data = self.download_ticker(ticker, start, end)
            if data is not None:
                all_data.append(data)

            # Rate limiting
            time.sleep(config.YF_DELAY)

        return all_data

def download_market_index():
    """Download S&P 500 index and SPY."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING MARKET INDICES")
    logger.info("=" * 60)

    indices = {'^GSPC': 'SP500', 'SPY': 'SPY'}
    data = {}

    for symbol, name in indices.items():
        try:
            logger.info(f"Downloading {name} ({symbol})...")
            t = yf.Ticker(symbol)
            df = t.history(start=config.PRICE_START, end=config.PRICE_END, auto_adjust=True)

            if not df.empty:
                df = df.reset_index()
                df['ticker'] = symbol
                data[name] = df
                logger.info(f"  ✓ {name}: {len(df)} rows")
            else:
                logger.warning(f"  ✗ {name}: No data")
        except Exception as e:
            logger.error(f"  ✗ {name}: {e}")

    return data

def main():
    """Main execution."""
    # Load universe
    logger.info("Loading universe...")
    universe = pd.read_parquet(config.PROCESSED_DIR / "universe.parquet")
    tickers = universe['ticker_yahoo'].tolist()
    logger.info(f"  Loaded {len(tickers)} tickers")

    # Download individual stocks
    downloader = PriceDownloader()
    all_data = downloader.download_all(tickers, config.PRICE_START, config.PRICE_END)

    # Download indices
    indices = download_market_index()

    # Combine all data
    if all_data:
        logger.info("\nCombining data...")
        combined = pd.concat(all_data, axis=0)
        combined = combined.sort_values(['ticker', 'Date'] if 'Date' in combined.columns else ['ticker', combined.index.name])

        # Save
        output_path = config.PROCESSED_DIR / "prices_all.parquet"
        combined.to_parquet(output_path)
        logger.info(f"✓ Saved {len(combined)} rows to {output_path}")

    # Save indices separately
    if indices:
        for name, df in indices.items():
            output_path = config.PROCESSED_DIR / f"prices_{name.lower()}.parquet"
            df.to_parquet(output_path)
            logger.info(f"✓ Saved {name}: {len(df)} rows to {output_path}")

    # Save failed tickers
    if downloader.failed:
        failed_df = pd.DataFrame(downloader.failed, columns=['ticker', 'reason'])
        failed_path = config.LOGS_DIR / "02_prices_failed.csv"
        failed_df.to_csv(failed_path, index=False)
        logger.warning(f"✗ {len(downloader.failed)} tickers failed (saved to {failed_path})")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tickers attempted: {len(tickers)}")
    logger.info(f"Successfully downloaded: {len(downloader.downloaded)}")
    logger.info(f"Failed: {len(downloader.failed)}")
    if downloader.failed:
        logger.info("Failed tickers (first 10):")
        for ticker, reason in downloader.failed[:10]:
            logger.info(f"  {ticker}: {reason}")

    logger.info("\n✓ Script complete")

if __name__ == "__main__":
    main()
