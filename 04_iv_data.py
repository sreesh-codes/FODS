"""
04_iv_data.py
Download implied volatility data from DoltHub.
First inspects data density, then downloads via dolt CLI or API fallback.
Creates:
- raw/iv_data.parquet (full IV history for universe tickers)
- logs/04_iv_density_report.txt (data availability report)
"""

import pandas as pd
import numpy as np
import requests
import subprocess
from pathlib import Path
from datetime import datetime
import logging
import config
import io

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / '04_iv_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_dolt_cli():
    """Check if dolt CLI is installed."""
    try:
        result = subprocess.run(["dolt", "version"], capture_output=True, text=True)
        logger.info(f"Dolt CLI found: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        logger.warning("Dolt CLI not found")
        logger.warning("  Install with: brew install dolt (macOS)")
        logger.warning("  Or see: https://docs.dolthub.com/")
        return False

def inspect_density(sample_tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']):
    """Inspect data density before downloading everything."""
    logger.info("=" * 60)
    logger.info("INSPECTING DATA DENSITY")
    logger.info("=" * 60)

    try:
        # Sample query
        url = "https://www.dolthub.com/csv/post-no-preference/options/master/volatility_history"

        logger.info(f"Checking sample tickers: {sample_tickers}")

        sample_query = f"""
        SELECT * FROM volatility_history
        WHERE symbol IN ({','.join([f"'{t}'" for t in sample_tickers])})
        ORDER BY symbol, date DESC
        LIMIT 100
        """

        response = requests.get(url, params={"q": sample_query})

        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            logger.info(f"✓ Retrieved {len(df)} sample rows")
            logger.info(f"  Columns: {list(df.columns)}")
            logger.info("\nSample data:")
            print(df.head(10))

            # Density by year
            density_query = f"""
            SELECT
                symbol,
                YEAR(date) as year,
                COUNT(*) as num_records,
                MIN(date) as first_date,
                MAX(date) as last_date
            FROM volatility_history
            WHERE symbol IN ({','.join([f"'{t}'" for t in sample_tickers])})
            GROUP BY symbol, YEAR(date)
            ORDER BY symbol, year
            """

            response2 = requests.get(url, params={"q": density_query})
            if response2.status_code == 200:
                density = pd.read_csv(io.StringIO(response2.text))
                logger.info("\n" + "=" * 60)
                logger.info("DATA DENSITY BY YEAR")
                logger.info("=" * 60)
                print(density.to_string(index=False))

                # Save report
                report_path = config.LOGS_DIR / "04_iv_density_report.txt"
                with open(report_path, 'w') as f:
                    f.write("DoltHub Volatility Data Density Report\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Generated: {datetime.now()}\n")
                    f.write(f"Sample tickers: {', '.join(sample_tickers)}\n\n")
                    f.write("Sample data:\n")
                    f.write(df.head(20).to_string() + "\n\n")
                    f.write("Density by year:\n")
                    f.write(density.to_string(index=False) + "\n")

                logger.info(f"\n✓ Saved density report to {report_path}")

                return density
            else:
                logger.warning(f"Density query failed: {response2.status_code}")

        else:
            logger.error(f"Sample query failed: {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")

    except Exception as e:
        logger.error(f"Inspection failed: {e}")

    return None

def download_via_dolt_cli(tickers):
    """Download IV data using dolt CLI (preferred method)."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING VIA DOLT CLI")
    logger.info("=" * 60)

    dolt_dir = config.RAW_DIR / "dolt_options"
    dolt_dir.mkdir(exist_ok=True)

    try:
        # Clone repo if not exists
        repo_dir = dolt_dir / "options"
        if not (repo_dir / ".dolt").exists():
            logger.info("Cloning dolt repository (this may take a while)...")
            subprocess.run(
                ["dolt", "clone", "post-no-preference/options"],
                cwd=dolt_dir,
                check=True
            )
            logger.info("✓ Repository cloned")
        else:
            logger.info("Repository already cloned, pulling updates...")
            subprocess.run(
                ["dolt", "pull"],
                cwd=repo_dir,
                check=True
            )
            logger.info("✓ Repository updated")

        # Query data
        logger.info(f"Querying IV data for {len(tickers)} tickers...")

        ticker_list = ','.join([f"'{t}'" for t in tickers])
        query = f"""
        SELECT *
        FROM volatility_history
        WHERE symbol IN ({ticker_list})
        AND date >= '{config.PRICE_START}'
        AND date <= '{config.PRICE_END}'
        ORDER BY symbol, date
        """

        # Export to CSV
        output_path = config.RAW_DIR / "iv_data.csv"
        logger.info(f"Exporting to {output_path}...")

        result = subprocess.run(
            ["dolt", "sql", "-q", query, "-r", "csv"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )

        # Save CSV
        with open(output_path, 'w') as f:
            f.write(result.stdout)

        # Convert to parquet
        df = pd.read_csv(output_path)
        df['date'] = pd.to_datetime(df['date'])

        parquet_path = config.RAW_DIR / "iv_data.parquet"
        df.to_parquet(parquet_path)

        logger.info(f"✓ Downloaded {len(df)} rows")
        logger.info(f"  Saved to {parquet_path}")

        return df

    except subprocess.CalledProcessError as e:
        logger.error(f"Dolt CLI command failed: {e}")
        logger.error(f"  stdout: {e.stdout}")
        logger.error(f"  stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Failed: {e}")
        return None

def download_via_api(tickers):
    """Download IV data using DoltHub API (fallback, may timeout for large queries)."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING VIA API (FALLBACK)")
    logger.info("=" * 60)
    logger.warning("API may timeout for large queries. Consider installing dolt CLI.")

    url = "https://www.dolthub.com/csv/post-no-preference/options/master/volatility_history"

    # Download in batches
    batch_size = 10
    all_data = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        logger.info(f"Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")

        ticker_list = ','.join([f"'{t}'" for t in batch])
        query = f"""
        SELECT *
        FROM volatility_history
        WHERE symbol IN ({ticker_list})
        AND date >= '{config.PRICE_START}'
        AND date <= '{config.PRICE_END}'
        ORDER BY symbol, date
        """

        try:
            response = requests.get(url, params={"q": query}, timeout=120)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                all_data.append(df)
                logger.info(f"  ✓ Retrieved {len(df)} rows")
            else:
                logger.error(f"  Failed: {response.status_code}")

        except Exception as e:
            logger.error(f"  Failed: {e}")

    if all_data:
        combined = pd.concat(all_data, axis=0, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])

        parquet_path = config.RAW_DIR / "iv_data.parquet"
        combined.to_parquet(parquet_path)

        logger.info(f"✓ Downloaded {len(combined)} rows total")
        logger.info(f"  Saved to {parquet_path}")

        return combined

    return None

def main():
    """Main execution."""
    # Check if already exists
    iv_path = config.RAW_DIR / "iv_data.parquet"
    if iv_path.exists():
        logger.info(f"✓ IV data already exists at {iv_path}")
        logger.info("  Delete it to re-download")
        df = pd.read_parquet(iv_path)
        logger.info(f"  Loaded {len(df)} rows")
        return

    # Load universe
    logger.info("Loading universe...")
    universe = pd.read_parquet(config.PROCESSED_DIR / "universe.parquet")
    tickers = universe['ticker_yahoo'].tolist()
    logger.info(f"  Loaded {len(tickers)} tickers")

    # Skip inspection to save time - go straight to download
    logger.info("Skipping inspection, proceeding to download...")

    # Try dolt CLI first
    has_dolt = check_dolt_cli()

    if has_dolt:
        df = download_via_dolt_cli(tickers)
    else:
        logger.info("Note: IV data available from 2019+ only")
        df = download_via_api(tickers)

    if df is not None:
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Tickers: {df['symbol'].nunique() if 'symbol' in df.columns else 'unknown'}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

        # Coverage by year
        if 'symbol' in df.columns and 'date' in df.columns:
            df['year'] = df['date'].dt.year
            coverage = df.groupby('year')['symbol'].nunique()
            logger.info("\nTicker coverage by year:")
            for year, count in coverage.items():
                logger.info(f"  {year}: {count} tickers")

    logger.info("\n✓ Script complete")

if __name__ == "__main__":
    main()
