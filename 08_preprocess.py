"""
08_preprocess.py
Preprocessing for "Do Markets Overreact to Earnings?" project.

GROUND RULE: All preprocessing decisions use 2015-2022 TRAIN data only.
Cutoffs, thresholds computed on train set, then applied to full sample.

This script:
1. Builds missingness table by column × year
2. Winsorizes SUE and features at 1st/99th percentile (train-set cutoffs)
3. Standardizes features cross-sectionally (z-score within quarter)
4. Encodes categoricals (sector, timing, size terciles)
5. Assigns SUE quintiles WITHIN each calendar quarter
6. Creates target variables (CAR windows + binary up/down)
7. Saves model_panel.parquet and data_dictionary.md

Output: processed/model_panel.parquet, data_dictionary.md
"""

import pandas as pd
import numpy as np
import logging
import config
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / '08_preprocess.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    """Load events_clean data."""
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)

    events = pd.read_parquet(config.PROCESSED_DIR / "events_clean.parquet")
    events['t0'] = pd.to_datetime(events['t0'])
    events['year'] = events['t0'].dt.year
    # Create quarter as string (e.g., "2023Q1") to support groupby
    events['quarter'] = events['t0'].dt.year.astype(str) + 'Q' + events['t0'].dt.quarter.astype(str)

    # Split train/test
    train_mask = (events['year'] >= int(config.TRAIN_START[:4])) & (events['year'] <= int(config.TRAIN_END[:4]))
    test_mask = (events['year'] >= int(config.TEST_START[:4])) & (events['year'] <= int(config.TEST_END[:4]))

    events['split'] = 'train'
    events.loc[test_mask, 'split'] = 'test'

    logger.info(f"Total events: {len(events):,}")
    logger.info(f"Train (2015-2022): {train_mask.sum():,} events")
    logger.info(f"Test (2023-2024): {test_mask.sum():,} events")

    return events

def build_missingness_table(events):
    """Build missingness table by column × year."""
    logger.info("=" * 60)
    logger.info("MISSINGNESS ANALYSIS")
    logger.info("=" * 60)

    # Key columns to check
    cols_to_check = [
        'eps_estimate_yf', 'eps_actual_yf', 'sue',
        'car_-5_-1', 'car_0_1', 'car_0_5', 'car_2_20',
        'price_tminus1', 'momentum_12_1', 'rv_21d',
        'iv_tminus1', 'sector'
    ]

    missingness = []

    for year in sorted(events['year'].unique()):
        year_data = events[events['year'] == year]
        row = {'year': year, 'n_events': len(year_data)}

        for col in cols_to_check:
            if col in events.columns:
                missing_pct = 100 * year_data[col].isna().sum() / len(year_data)
                row[f'{col}_missing_pct'] = missing_pct

        missingness.append(row)

    miss_df = pd.DataFrame(missingness)

    # Save to logs
    miss_path = config.LOGS_DIR / "missingness_table.csv"
    miss_df.to_csv(miss_path, index=False)
    logger.info(f"✓ Saved missingness table to {miss_path}")

    # Print summary
    logger.info("\nMissingness by key variable (% across all years):")
    for col in cols_to_check:
        if col in events.columns:
            missing_pct = 100 * events[col].isna().sum() / len(events)
            logger.info(f"  {col:20s}: {missing_pct:5.1f}%")

    return miss_df

def winsorize_features(events):
    """Winsorize continuous features at 1st/99th percentile (train set cutoffs)."""
    logger.info("=" * 60)
    logger.info("WINSORIZATION (1st/99th percentile)")
    logger.info("=" * 60)

    # Features to winsorize
    features_to_wins = ['sue', 'momentum_12_1', 'rv_21d', 'price_tminus1']

    # Add CAR columns
    for ws, we in config.CAR_WINDOWS:
        features_to_wins.append(f'car_{ws}_{we}')
        features_to_wins.append(f'car_abn_{ws}_{we}')

    train = events[events['split'] == 'train']

    cutoffs = {}

    for col in features_to_wins:
        if col not in events.columns:
            continue

        # Compute cutoffs on TRAIN set only
        p01 = train[col].quantile(0.01)
        p99 = train[col].quantile(0.99)

        cutoffs[col] = {'p01': p01, 'p99': p99}

        # Create winsorized column
        events[f'{col}_wins'] = events[col].clip(lower=p01, upper=p99)

        logger.info(f"  {col:20s}: [{p01:8.4f}, {p99:8.4f}]")

    # Save cutoffs for documentation
    cutoffs_df = pd.DataFrame(cutoffs).T
    cutoffs_df.to_csv(config.LOGS_DIR / "winsorization_cutoffs.csv")

    return events

def standardize_features(events):
    """Standardize features cross-sectionally (z-score within each quarter)."""
    logger.info("=" * 60)
    logger.info("CROSS-SECTIONAL STANDARDIZATION")
    logger.info("=" * 60)

    # Features to standardize (use winsorized versions)
    features_to_std = [
        'sue_wins', 'momentum_12_1_wins', 'rv_21d_wins', 'price_tminus1_wins'
    ]

    for col in features_to_std:
        if col not in events.columns:
            continue

        # Standardize within each quarter using transform (preserves index)
        def calc_z_score(x):
            mean = x.mean()
            std = x.std()
            if std > 0:
                return (x - mean) / std
            else:
                return x * 0  # Return zeros if no variation

        events[f'{col}_std'] = events.groupby('quarter')[col].transform(calc_z_score)

        logger.info(f"  ✓ Standardized {col}")

    return events

def encode_categoricals(events):
    """Encode categorical features."""
    logger.info("=" * 60)
    logger.info("CATEGORICAL ENCODING")
    logger.info("=" * 60)

    # 1. Sector (one-hot encoding)
    if 'sector' in events.columns:
        sector_dummies = pd.get_dummies(events['sector'], prefix='sector', drop_first=True)
        events = pd.concat([events, sector_dummies], axis=1)
        logger.info(f"  ✓ Sector one-hot: {len(sector_dummies.columns)} dummies")

    # 2. Timing (binary: BMO=1, AMC=0, unknown=NA)
    events['timing_bmo'] = (events['timing'] == 'BMO').astype(float)
    events.loc[events['timing'].isin(['unknown', 'trading_hours']), 'timing_bmo'] = np.nan
    logger.info(f"  ✓ Timing binary (BMO=1)")

    # 3. Size terciles (based on price_tminus1 as proxy)
    if 'price_tminus1' in events.columns:
        events['size_tercile'] = None  # Use None for object dtype

        # Assign terciles within each quarter
        for quarter in events['quarter'].unique():
            mask = events['quarter'] == quarter
            quarter_prices = events.loc[mask, 'price_tminus1']

            try:
                terciles = pd.qcut(
                    quarter_prices,
                    q=3,
                    labels=['small', 'medium', 'large'],
                    duplicates='drop'
                ).astype(str)
                events.loc[mask, 'size_tercile'] = terciles
            except ValueError:
                # Skip this quarter if can't create terciles
                pass

        # One-hot encoding for size
        size_dummies = pd.get_dummies(events['size_tercile'], prefix='size', drop_first=True)
        events = pd.concat([events, size_dummies], axis=1)
        logger.info(f"  ✓ Size terciles: {events['size_tercile'].value_counts().to_dict()}")

    return events

def assign_sue_quintiles(events):
    """Assign SUE quintiles WITHIN each calendar quarter (not global)."""
    logger.info("=" * 60)
    logger.info("SUE QUINTILES (within-quarter assignment)")
    logger.info("=" * 60)

    # Initialize quintile column
    events['sue_quintile'] = np.nan

    # Assign quintiles within each quarter
    for quarter in events['quarter'].unique():
        mask = events['quarter'] == quarter
        quarter_data = events.loc[mask, 'sue']
        valid = quarter_data.notna()

        if valid.sum() >= 5:  # Need at least 5 events
            try:
                quintiles = pd.qcut(
                    quarter_data[valid],
                    q=5,
                    labels=[1, 2, 3, 4, 5],
                    duplicates='drop'
                )
                events.loc[mask & valid, 'sue_quintile'] = quintiles.astype(float)
            except ValueError:
                # If can't create 5 quintiles (ties), skip this quarter
                pass

    # Convert to numeric (already done above)
    # events['sue_quintile'] = pd.to_numeric(events['sue_quintile'], errors='coerce')

    logger.info(f"  Total events with SUE quintile: {events['sue_quintile'].notna().sum():,}")
    logger.info(f"  Quintile distribution:")
    for q in [1, 2, 3, 4, 5]:
        count = (events['sue_quintile'] == q).sum()
        pct = 100 * count / events['sue_quintile'].notna().sum()
        logger.info(f"    Q{q}: {count:,} ({pct:.1f}%)")

    return events

def create_targets(events):
    """Create target variables for modeling."""
    logger.info("=" * 60)
    logger.info("CREATING TARGET VARIABLES")
    logger.info("=" * 60)

    # Targets already exist (CAR windows), but add binary targets

    # Binary up/down for CAR[+2,+5]
    events['car_2_5_up'] = (events['car_2_20'] > 0).astype(float)
    events.loc[events['car_2_20'].isna(), 'car_2_5_up'] = np.nan

    logger.info(f"  ✓ Created binary target: car_2_5_up")
    logger.info(f"    Up: {(events['car_2_5_up'] == 1).sum():,}")
    logger.info(f"    Down: {(events['car_2_5_up'] == 0).sum():,}")

    return events

def create_data_dictionary(events):
    """Create data dictionary markdown file."""
    logger.info("=" * 60)
    logger.info("CREATING DATA DICTIONARY")
    logger.info("=" * 60)

    dict_path = config.BASE_DIR / "data_dictionary.md"

    with open(dict_path, 'w') as f:
        f.write("# Data Dictionary - model_panel.parquet\n\n")
        f.write("**Generated**: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write("**Project**: Do Markets Overreact to Earnings?\n\n")
        f.write("---\n\n")

        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total events**: {len(events):,}\n")
        f.write(f"- **Train events (2015-2022)**: {(events['split']=='train').sum():,}\n")
        f.write(f"- **Test events (2023-2024)**: {(events['split']=='test').sum():,}\n")
        f.write(f"- **Tickers**: {events['ticker'].nunique()}\n")
        f.write(f"- **Date range**: {events['t0'].min().date()} to {events['t0'].max().date()}\n\n")

        f.write("---\n\n")
        f.write("## Column Descriptions\n\n")

        f.write("### Identifiers & Metadata\n\n")
        f.write("| Column | Type | Description |\n")
        f.write("|--------|------|-------------|\n")
        f.write("| `ticker` | str | Stock ticker symbol |\n")
        f.write("| `datetime` | datetime | Earnings announcement timestamp |\n")
        f.write("| `ann_date` | date | Announcement date |\n")
        f.write("| `t0` | date | First reaction date (trading day) |\n")
        f.write("| `fiscal_qtr` | period | Fiscal quarter (YYYYQQ) |\n")
        f.write("| `year` | int | Year of t0 |\n")
        f.write("| `quarter` | period | Calendar quarter of t0 |\n")
        f.write("| `split` | str | 'train' (2015-2022) or 'test' (2023-2024) |\n\n")

        f.write("### Raw Earnings Data\n\n")
        f.write("| Column | Type | Description |\n")
        f.write("|--------|------|-------------|\n")
        f.write("| `eps_estimate_yf` | float | EPS consensus estimate (split-adjusted) |\n")
        f.write("| `eps_actual_yf` | float | Reported EPS (split-adjusted) |\n")
        f.write("| `surprise` | float | Actual - estimate |\n")
        f.write("| `surprise_std` | float | Rolling std of last 8 surprises |\n")
        f.write("| `sue` | float | Standardized unexpected earnings (surprise / surprise_std) |\n")
        f.write("| `sue_wins` | float | SUE winsorized at 1%/99% (train cutoffs) |\n")
        f.write("| `sue_quintile` | int | SUE quintile (1-5) assigned within calendar quarter |\n\n")

        f.write("### Returns & CAR (Cumulative Abnormal Returns)\n\n")
        f.write("| Column | Type | Description |\n")
        f.write("|--------|------|-------------|\n")
        for ws, we in config.CAR_WINDOWS:
            f.write(f"| `car_{ws}_{we}` | float | Stock CAR from t{ws} to t{we} (geometric) |\n")
            f.write(f"| `car_mkt_{ws}_{we}` | float | Market CAR from t{ws} to t{we} |\n")
            f.write(f"| `car_abn_{ws}_{we}` | float | Abnormal return = stock - market |\n")
            f.write(f"| `car_{ws}_{we}_wins` | float | Winsorized CAR (1%/99%, train cutoffs) |\n")
            f.write(f"| `car_abn_{ws}_{we}_wins` | float | Winsorized abnormal CAR |\n")
        f.write("| `car_2_5_up` | binary | 1 if CAR[+2,+20] > 0, else 0 |\n\n")

        f.write("### Features (Raw)\n\n")
        f.write("| Column | Type | Description |\n")
        f.write("|--------|------|-------------|\n")
        f.write("| `price_tminus1` | float | Price at t-1 |\n")
        f.write("| `momentum_12_1` | float | 12-1 month momentum |\n")
        f.write("| `rv_21d` | float | 21-day realized volatility (annualized) |\n")
        f.write("| `iv_tminus1` | float | Implied volatility at t-1 (NA - not available) |\n")
        f.write("| `iv_rv_ratio` | float | IV/RV ratio (NA) |\n\n")

        f.write("### Features (Preprocessed)\n\n")
        f.write("| Column | Type | Description |\n")
        f.write("|--------|------|-------------|\n")
        f.write("| `*_wins` | float | Winsorized at 1%/99% using train-set cutoffs |\n")
        f.write("| `*_wins_std` | float | Cross-sectionally standardized (z-score within quarter) |\n")
        f.write("| `sector_*` | binary | One-hot encoding for GICS sectors (reference = first sector) |\n")
        f.write("| `timing_bmo` | binary | 1 if BMO, 0 if AMC, NA if unknown |\n")
        f.write("| `size_tercile` | str | 'small', 'medium', 'large' based on price (within quarter) |\n")
        f.write("| `size_*` | binary | One-hot encoding for size (reference = small) |\n\n")

        f.write("### Quality Flags\n\n")
        f.write("| Column | Type | Description |\n")
        f.write("|--------|------|-------------|\n")
        f.write("| `volume_peak_correct` | bool | True if volume peak at t0 or t0+1 |\n")
        f.write("| `is_clustered` | bool | True if >30 events on same announcement date |\n")
        f.write("| `in_sp500_at_t0` | bool | True if in S&P 500 index on t0 |\n")
        f.write("| `has_confounding` | bool | True if overlapping events within ±20 days |\n")
        f.write("| `t0_method` | str | How t0 was determined (BMO_same_day, AMC_next_day, etc.) |\n\n")

        f.write("---\n\n")
        f.write("## Preprocessing Notes\n\n")
        f.write("### Winsorization\n")
        f.write("- All continuous features winsorized at 1st/99th percentile\n")
        f.write("- Cutoffs computed on TRAINING SET (2015-2022) only\n")
        f.write("- Same cutoffs applied to test set (2023-2024)\n")
        f.write("- See `logs/winsorization_cutoffs.csv` for exact values\n\n")

        f.write("### Standardization\n")
        f.write("- Features standardized WITHIN each calendar quarter\n")
        f.write("- Z-score = (x - quarter_mean) / quarter_std\n")
        f.write("- Prevents look-ahead bias and controls for time variation\n\n")

        f.write("### SUE Quintiles\n")
        f.write("- Assigned WITHIN each calendar quarter (not globally)\n")
        f.write("- Q1 = lowest SUE (most negative surprise)\n")
        f.write("- Q5 = highest SUE (most positive surprise)\n")
        f.write("- Within-quarter assignment prevents look-ahead bias\n\n")

        f.write("### Missing Data\n")
        f.write("- SUE: 6% missing (first few events per ticker)\n")
        f.write("- IV: 100% missing (data not downloaded)\n")
        f.write("- RV: 22% missing (insufficient price history)\n")
        f.write("- See `logs/missingness_table.csv` for details\n\n")

        f.write("---\n\n")
        f.write("## Usage Notes\n\n")
        f.write("### For EDA\n")
        f.write("- Use `split == 'train'` for all exploratory decisions\n")
        f.write("- Test set (2023-2024) should NOT be examined during EDA\n\n")

        f.write("### For Modeling\n")
        f.write("- Use `*_wins_std` versions of features (winsorized + standardized)\n")
        f.write("- Include sector and timing dummies as controls\n")
        f.write("- Cluster standard errors by `t0` (announcement date)\n")
        f.write("- Target: `car_abn_2_20` for drift, `car_abn_0_1` for announcement\n\n")

        f.write("### Quality Controls\n")
        f.write("- Exclude `volume_peak_correct == False` for robustness\n")
        f.write("- Exclude `is_clustered == True` for clean sample\n")
        f.write("- Check `has_confounding == True` for sensitivity\n\n")

    logger.info(f"✓ Saved data dictionary to {dict_path}")

def main():
    """Main execution."""
    # Load data
    events = load_data()

    # Missingness analysis
    miss_df = build_missingness_table(events)

    # Winsorization (using train cutoffs)
    events = winsorize_features(events)

    # Standardization (within quarter)
    events = standardize_features(events)

    # Categorical encoding
    events = encode_categoricals(events)

    # SUE quintiles (within quarter)
    events = assign_sue_quintiles(events)

    # Target variables
    events = create_targets(events)

    # Save model panel
    output_path = config.PROCESSED_DIR / "model_panel.parquet"
    events.to_parquet(output_path)
    logger.info(f"\n✓ Saved model panel to {output_path}")
    logger.info(f"  Columns: {len(events.columns)}")
    logger.info(f"  Rows: {len(events):,}")

    # Create data dictionary
    create_data_dictionary(events)

    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model panel ready: {output_path}")
    logger.info(f"Data dictionary: data_dictionary.md")

if __name__ == "__main__":
    main()
