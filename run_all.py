"""
run_all.py
Orchestrates the full data pipeline.
Run all scripts in sequence.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'logs' / 'run_all.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SCRIPTS = [
    ("01_universe.py", "Download S&P 500 constituents"),
    ("02_prices.py", "Download price data"),
    ("03_earnings.py", "Download earnings data"),
    ("04_iv_data.py", "Download implied volatility data"),
    ("05_factors.py", "Download Fama-French factors"),
    ("06_build_panels.py", "Build analysis panels"),
]

def run_script(script_name, description):
    """Run a single script."""
    logger.info("=" * 70)
    logger.info(f"RUNNING: {script_name}")
    logger.info(f"Description: {description}")
    logger.info("=" * 70)

    start_time = datetime.now()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False  # Show output in real-time
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ {script_name} completed in {elapsed:.1f}s\n")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {script_name} failed with exit code {e.returncode}")
        logger.error(f"  Consider checking logs/{script_name.replace('.py', '.log')}")
        return False

    except Exception as e:
        logger.error(f"✗ {script_name} failed: {e}")
        return False

def main():
    """Run all scripts."""
    logger.info("=" * 70)
    logger.info("STARTING FULL DATA PIPELINE")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("=" * 70)

    overall_start = datetime.now()
    success_count = 0

    for script_name, description in SCRIPTS:
        success = run_script(script_name, description)

        if success:
            success_count += 1
        else:
            logger.error(f"\n✗ Pipeline stopped due to failure in {script_name}")
            logger.error("  Fix the issue and re-run")
            logger.error("  Note: Already-downloaded data will be reused (cached)")
            sys.exit(1)

    # All scripts succeeded
    total_elapsed = (datetime.now() - overall_start).total_seconds()

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"All {success_count}/{len(SCRIPTS)} scripts completed successfully")
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info("\nNext steps:")
    logger.info("  1. Run: python generate_report.py")
    logger.info("  2. Review: data_report.md")
    logger.info("  3. Start analysis!")

if __name__ == "__main__":
    main()
