"""
05_factors.py
Download Fama-French factors from Ken French Data Library.
Creates:
- raw/ff_factors.parquet (daily factors in decimal form)
"""

import pandas as pd
import numpy as np
import pandas_datareader as pdr
import requests
import zipfile
import io
from pathlib import Path
from datetime import datetime
import logging
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / '05_factors.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_via_pandas_datareader():
    """Download using pandas_datareader (preferred method)."""
    logger.info("Trying pandas_datareader...")

    try:
        ff_factors = pdr.get_data_famafrench(
            'F-F_Research_Data_Factors_daily',
            start=config.PRICE_START,
            end=config.PRICE_END
        )

        df = ff_factors[0]  # First table is daily factors

        logger.info(f"✓ Downloaded {len(df)} rows")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")

        # Convert from percent to decimal
        df = df / 100
        logger.info("  Converted from percent to decimal")

        return df

    except Exception as e:
        logger.warning(f"pandas_datareader failed: {e}")
        return None

def download_direct():
    """Download directly from Ken French website (fallback)."""
    logger.info("Trying direct download from Ken French website...")

    try:
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        logger.info(f"  Downloading from {url}")

        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            logger.error(f"  Failed: HTTP {response.status_code}")
            return None

        # Extract ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            logger.info(f"  Extracting {csv_name}")

            with z.open(csv_name) as f:
                # Read and parse
                lines = f.read().decode('utf-8').split('\n')

                # Find where data starts (after column headers line)
                start_idx = 0
                for i, line in enumerate(lines):
                    if 'Mkt-RF' in line and 'SMB' in line and 'HML' in line:
                        start_idx = i
                        break

                if start_idx == 0:
                    logger.error("  Could not find data start")
                    return None

                # Read from that point
                data_text = '\n'.join(lines[start_idx:])
                df = pd.read_csv(io.StringIO(data_text))

                # Filter to valid dates (YYYYMMDD format)
                df = df[df.iloc[:, 0].astype(str).str.len() == 8].copy()

                # Parse dates
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0].astype(str), format='%Y%m%d')
                df = df.set_index(df.columns[0])
                df.index.name = 'Date'

                # Convert to numeric and divide by 100
                df = df.apply(pd.to_numeric, errors='coerce') / 100

                # Filter to date range
                df = df[(df.index >= pd.to_datetime(config.PRICE_START)) &
                        (df.index <= pd.to_datetime(config.PRICE_END))]

                logger.info(f"✓ Downloaded {len(df)} rows")
                logger.info(f"  Columns: {list(df.columns)}")
                logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")

                return df

    except Exception as e:
        logger.error(f"Direct download failed: {e}")
        return None

def main():
    """Main execution."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING FAMA-FRENCH FACTORS")
    logger.info("=" * 60)

    # Check if already exists
    output_path = config.RAW_DIR / "ff_factors.parquet"
    if output_path.exists():
        logger.info(f"✓ Factors already exist at {output_path}")
        logger.info("  Delete it to re-download")
        df = pd.read_parquet(output_path)
        logger.info(f"  Loaded {len(df)} rows")
        return

    # Try pandas_datareader first
    df = download_via_pandas_datareader()

    # Fallback to direct download
    if df is None:
        df = download_direct()

    if df is None:
        logger.error("✗ All download methods failed")
        return

    # Save
    df.to_parquet(output_path)
    logger.info(f"\n✓ Saved {len(df)} rows to {output_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    logger.info(f"Factors: {', '.join(df.columns)}")
    logger.info("\nSample (first 5 rows):")
    print(df.head())
    logger.info("\nSample statistics:")
    print(df.describe())

    logger.info("\n✓ Script complete")

if __name__ == "__main__":
    main()
