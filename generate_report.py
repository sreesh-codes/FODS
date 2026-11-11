"""
generate_report.py
Generate data_report.md with statistics and sanity checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import config

def load_all_data():
    """Load all processed data."""
    data = {}

    files = {
        'events': config.PROCESSED_DIR / 'events.parquet',
        'iv_panel': config.PROCESSED_DIR / 'iv_event_panel.parquet',
        'daily_panel': config.PROCESSED_DIR / 'daily_panel.parquet',
        'universe': config.PROCESSED_DIR / 'universe.parquet',
        'membership': config.PROCESSED_DIR / 'membership.parquet',
    }

    for name, path in files.items():
        if path.exists():
            data[name] = pd.read_parquet(path)
            print(f"✓ Loaded {name}: {len(data[name])} rows")
        else:
            data[name] = None
            print(f"✗ {name} not found")

    # Load failed tickers log
    failed_path = config.LOGS_DIR / '02_prices_failed.csv'
    if failed_path.exists():
        data['failed_tickers'] = pd.read_csv(failed_path)
    else:
        data['failed_tickers'] = None

    return data

def generate_report(data):
    """Generate markdown report."""
    lines = []

    lines.append("# Data Report: Do Markets Overreact to Earnings?")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nAnalysis period: {config.ANALYSIS_START} to {config.ANALYSIS_END}")

    # ========================================================================
    # SECTION 1: Data Overview
    # ========================================================================
    lines.append("\n## 1. Data Overview")

    if data['universe'] is not None:
        lines.append(f"\n### Universe")
        lines.append(f"- **Total tickers**: {len(data['universe'])}")

        if data['membership'] is not None:
            lines.append(f"- **Membership records**: {len(data['membership'])}")
            lines.append(f"- **Date range**: {data['membership']['start_date'].min()} to {data['membership']['end_date'].max()}")

    # Events
    if data['events'] is not None:
        events = data['events']
        lines.append(f"\n### Earnings Events")
        lines.append(f"- **Total events**: {len(events):,}")
        lines.append(f"- **Unique tickers**: {events['ticker'].nunique()}")
        lines.append(f"- **Events in index**: {events['in_index'].sum():,} ({100*events['in_index'].mean():.1f}%)")

        # By year
        events['year'] = pd.to_datetime(events['t0']).dt.year
        events_by_year = events.groupby('year').size()

        lines.append(f"\n#### Events per year")
        lines.append("| Year | Events |")
        lines.append("|------|--------|")
        for year, count in events_by_year.items():
            lines.append(f"| {year} | {count:,} |")

        # EPS coverage
        has_yf_eps = events['eps_estimate_yf'].notna().sum()
        has_av_eps = events['eps_estimate_av'].notna().sum()
        has_both_eps = ((events['eps_estimate_yf'].notna()) & (events['eps_estimate_av'].notna())).sum()

        lines.append(f"\n#### EPS Estimate Coverage")
        lines.append(f"- **yfinance**: {has_yf_eps:,} events ({100*has_yf_eps/len(events):.1f}%)")
        lines.append(f"- **Alpha Vantage**: {has_av_eps:,} events ({100*has_av_eps/len(events):.1f}%)")
        lines.append(f"- **Both sources**: {has_both_eps:,} events ({100*has_both_eps/len(events):.1f}%)")

        # Conflicts
        if 'eps_actual_conflict' in events.columns:
            n_conflicts = (events['eps_actual_conflict'] | events['eps_estimate_conflict']).sum()
            lines.append(f"- **Conflicts** (diff > ${config.EPS_DIFF_THRESHOLD}): {n_conflicts} ({100*n_conflicts/max(has_both_eps, 1):.1f}% of events with both sources)")

    # IV coverage
    if data['iv_panel'] is not None:
        iv_panel = data['iv_panel']
        lines.append(f"\n### Implied Volatility Coverage")
        lines.append(f"- **Event-day observations**: {len(iv_panel):,}")
        lines.append(f"- **Events with IV data**: {iv_panel['event_id'].nunique():,}")

        # Coverage by year (if events exist)
        if data['events'] is not None:
            events_with_iv = iv_panel['event_id'].unique()
            events_df = data['events']
            events_df['has_iv'] = events_df.index.isin(events_with_iv)
            events_df['year'] = pd.to_datetime(events_df['t0']).dt.year

            iv_coverage_by_year = events_df.groupby('year')['has_iv'].agg(['sum', 'count'])
            iv_coverage_by_year['pct'] = 100 * iv_coverage_by_year['sum'] / iv_coverage_by_year['count']

            lines.append(f"\n#### IV Coverage by Year")
            lines.append("| Year | Events with IV | Total Events | Coverage % |")
            lines.append("|------|----------------|--------------|------------|")
            for year, row in iv_coverage_by_year.iterrows():
                lines.append(f"| {year} | {int(row['sum']):,} | {int(row['count']):,} | {row['pct']:.1f}% |")
    else:
        lines.append(f"\n### Implied Volatility Coverage")
        lines.append("**No IV data available** (pre-2019 or download failed)")

    # ========================================================================
    # SECTION 2: Dropped Tickers
    # ========================================================================
    lines.append("\n## 2. Dropped/Failed Tickers")

    if data['failed_tickers'] is not None:
        failed = data['failed_tickers']
        lines.append(f"\n**Total failed**: {len(failed)}")

        # Group by reason
        reason_counts = failed['reason'].value_counts()
        lines.append(f"\n### Failure reasons")
        lines.append("| Reason | Count |")
        lines.append("|--------|-------|")
        for reason, count in reason_counts.items():
            lines.append(f"| {reason} | {count} |")

        # Sample of failed tickers
        lines.append(f"\n### Sample failed tickers")
        lines.append("| Ticker | Reason |")
        lines.append("|--------|--------|")
        for _, row in failed.head(20).iterrows():
            lines.append(f"| {row['ticker']} | {row['reason']} |")
    else:
        lines.append("\nNo failed ticker log found.")

    # ========================================================================
    # SECTION 3: Sanity Checks
    # ========================================================================
    lines.append("\n## 3. Sanity Checks")

    # SUE quintiles vs CAR
    if data['events'] is not None:
        events = data['events']

        lines.append("\n### 3.1 Post-Earnings Announcement Drift (PEAD)")
        lines.append("\nExpected pattern: Higher SUE → Higher positive CAR")

        # Create SUE quintiles
        events_with_sue = events[events['sue'].notna() & events[f'car_0_5_market_adj'].notna()].copy()

        if len(events_with_sue) > 0:
            events_with_sue['sue_quintile'] = pd.qcut(
                events_with_sue['sue'],
                q=5,
                labels=['Q1 (Low SUE)', 'Q2', 'Q3', 'Q4', 'Q5 (High SUE)'],
                duplicates='drop'
            )

            # Compute average CAR by quintile
            sue_analysis = events_with_sue.groupby('sue_quintile').agg({
                'car_0_5_market_adj': ['mean', 'std', 'count'],
                'car_2_20_market_adj': ['mean', 'std', 'count'] if 'car_2_20_market_adj' in events_with_sue.columns else ['mean', 'std', 'count']
            })

            lines.append("\n#### Average CAR by SUE Quintile (Market-Adjusted)")
            lines.append("| SUE Quintile | CAR[0,+5] Mean | Std Dev | N | CAR[+2,+20] Mean | Std Dev | N |")
            lines.append("|--------------|----------------|---------|---|------------------|---------|---|")

            for quintile in sue_analysis.index:
                car_0_5_mean = sue_analysis.loc[quintile, ('car_0_5_market_adj', 'mean')]
                car_0_5_std = sue_analysis.loc[quintile, ('car_0_5_market_adj', 'std')]
                car_0_5_n = int(sue_analysis.loc[quintile, ('car_0_5_market_adj', 'count')])

                if 'car_2_20_market_adj' in events_with_sue.columns:
                    car_2_20_mean = sue_analysis.loc[quintile, ('car_2_20_market_adj', 'mean')]
                    car_2_20_std = sue_analysis.loc[quintile, ('car_2_20_market_adj', 'std')]
                    car_2_20_n = int(sue_analysis.loc[quintile, ('car_2_20_market_adj', 'count')])
                else:
                    car_2_20_mean = np.nan
                    car_2_20_std = np.nan
                    car_2_20_n = 0

                lines.append(f"| {quintile} | {car_0_5_mean*100:.3f}% | {car_0_5_std*100:.3f}% | {car_0_5_n} | {car_2_20_mean*100:.3f}% | {car_2_20_std*100:.3f}% | {car_2_20_n} |")

            # Assessment
            q1_car = sue_analysis.loc['Q1 (Low SUE)', ('car_0_5_market_adj', 'mean')]
            q5_car = sue_analysis.loc['Q5 (High SUE)', ('car_0_5_market_adj', 'mean')]

            lines.append(f"\n**Assessment**: Q5 - Q1 = {(q5_car - q1_car)*100:.3f}%")

            if q5_car > q1_car:
                lines.append("- ✓ **Pattern confirmed**: High SUE → Higher CAR (PEAD)")
            else:
                lines.append("- ✗ **Unexpected**: High SUE → Lower CAR")

        else:
            lines.append("\n**Insufficient data** to compute SUE quintiles")

    # IV crush
    if data['iv_panel'] is not None:
        iv_panel = data['iv_panel']

        lines.append("\n### 3.2 Implied Volatility Crush")
        lines.append("\nExpected pattern: IV peaks before earnings (t-1), drops sharply after (t+1)")

        # Average IV by relative day
        iv_by_day = iv_panel.groupby('rel_day').agg({
            'iv': ['mean', 'std', 'count'],
            'rv': ['mean', 'std', 'count'],
            'iv_rv_spread': ['mean', 'std', 'count']
        })

        lines.append("\n#### Average IV, RV, and IV-RV Spread by Event Day")
        lines.append("| Event Day | IV Mean | IV Std | N | RV Mean | RV Std | N | IV-RV Spread | N |")
        lines.append("|-----------|---------|--------|---|---------|--------|---|--------------|---|")

        for day in range(-10, 11):
            if day in iv_by_day.index:
                row = iv_by_day.loc[day]
                iv_mean = row[('iv', 'mean')]
                iv_std = row[('iv', 'std')]
                iv_n = int(row[('iv', 'count')])

                rv_mean = row[('rv', 'mean')]
                rv_std = row[('rv', 'std')]
                rv_n = int(row[('rv', 'count')])

                spread_mean = row[('iv_rv_spread', 'mean')]
                spread_n = int(row[('iv_rv_spread', 'count')])

                lines.append(f"| t{day:+d} | {iv_mean:.4f} | {iv_std:.4f} | {iv_n} | {rv_mean:.4f} | {rv_std:.4f} | {rv_n} | {spread_mean:.4f} | {spread_n} |")

        # Assessment
        if -1 in iv_by_day.index and 1 in iv_by_day.index:
            iv_before = iv_by_day.loc[-1, ('iv', 'mean')]
            iv_after = iv_by_day.loc[1, ('iv', 'mean')]
            iv_drop = iv_before - iv_after

            lines.append(f"\n**Assessment**: IV at t-1 = {iv_before:.4f}, IV at t+1 = {iv_after:.4f}")
            lines.append(f"- **IV drop**: {iv_drop:.4f} ({100*iv_drop/iv_before:.1f}%)")

            if iv_drop > 0:
                lines.append("- ✓ **Pattern confirmed**: IV drops after earnings (IV crush)")
            else:
                lines.append("- ✗ **Unexpected**: IV increases or stays flat after earnings")
        else:
            lines.append("\n**Insufficient data** around t-1 and t+1")

    # ========================================================================
    # SECTION 4: Data Quality Issues
    # ========================================================================
    lines.append("\n## 4. Data Quality Issues")

    issues = []

    # Check for duplicate events
    if data['events'] is not None:
        events = data['events']
        duplicates = events.groupby(['ticker', 't0']).size()
        duplicates = duplicates[duplicates > 1]

        if len(duplicates) > 0:
            issues.append(f"- **Duplicate events**: {len(duplicates)} ticker-date pairs have multiple entries")
        else:
            issues.append(f"- ✓ No duplicate events")

        # EPS outliers
        if 'surprise' in events.columns:
            surprise_abs = events['surprise'].abs()
            outliers = surprise_abs[surprise_abs > 10].count()  # >$10 surprise

            if outliers > 0:
                issues.append(f"- **EPS outliers**: {outliers} events with |surprise| > $10 (possible stock splits)")
            else:
                issues.append(f"- ✓ No extreme EPS outliers")

        # Missing prices
        missing_price = events['price_t_minus_1'].isna().sum()
        if missing_price > 0:
            issues.append(f"- **Missing prices**: {missing_price} events ({100*missing_price/len(events):.1f}%) have no price at t-1")

    # Timezone issues
    issues.append(f"- **Timezone handling**: Earnings times converted to ET for BMO/AMC classification")

    # Stock splits
    issues.append(f"- **Stock splits**: Not explicitly adjusted in EPS data (may cause outliers)")

    if issues:
        lines.append("\n" + "\n".join(issues))

    # ========================================================================
    # SECTION 5: Recommendations
    # ========================================================================
    lines.append("\n## 5. Recommendations")
    lines.append("\n### For Analysis")
    lines.append("- Filter to `in_index == True` to avoid survivorship bias")
    lines.append("- Use `sue` or `surprise_price_scaled` for standardization")
    lines.append("- For IV analysis, restrict to 2019+ where coverage is good")
    lines.append("- Consider winsorizing extreme surprises (>$10) to handle splits")

    lines.append("\n### Data Improvements")
    lines.append("- Obtain more Alpha Vantage API quota for better EPS estimate coverage")
    lines.append("- Consider adding sector/industry controls from constituents data")
    lines.append("- Download trading calendar to properly handle holidays in t0 calculation")

    # ========================================================================
    # Save report
    # ========================================================================
    report_text = "\n".join(lines)

    output_path = Path(__file__).parent / "data_report.md"
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\n✓ Report saved to {output_path}")

    return report_text

if __name__ == "__main__":
    print("Generating data report...")
    data = load_all_data()
    report = generate_report(data)
    print("\n✓ Report generation complete")
