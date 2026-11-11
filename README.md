# FODS Earnings Project: Data Pipeline

Reproducible data pipeline for "Do Markets Overreact to Earnings?" project testing:
1. Post-Earnings Announcement Drift (PEAD)
2. Implied Volatility Crush around earnings

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy yfinance pandas-datareader requests pytz

# 2. (Optional) Set Alpha Vantage API key for better EPS coverage
export AV_KEY='your_api_key_here'

# 3. (Optional) Install dolt CLI for faster IV data download
# macOS: brew install dolt
# See: https://docs.dolthub.com/

# 4. Run the full pipeline
python run_all.py

# 5. Generate report
python generate_report.py

# 6. Review results
cat data_report.md
```

## Pipeline Structure

### Scripts (run in order)

1. **01_universe.py** - Download S&P 500 historical constituents
   - Source: github.com/hanshof/sp500_constituents
   - Creates: `processed/universe.parquet`, `processed/membership.parquet`

2. **02_prices.py** - Download daily OHLCV data
   - Source: yfinance
   - Date range: 2014-10-01 to 2025-01-31 (extra lead-in for rolling windows)
   - Creates: `raw/prices/{TICKER}.parquet`, `processed/prices_all.parquet`
   - Handles rate limiting with retries and backoff

3. **03_earnings.py** - Download and merge earnings data
   - Sources: yfinance + Alpha Vantage
   - Classifies timing (BMO/AMC)
   - Flags conflicts between sources
   - Creates: `raw/earnings_yf.parquet`, `raw/earnings_av.parquet`, `processed/earnings_merged.parquet`
   - **Resumable**: AV calls are cached, stops cleanly at quota limit

4. **04_iv_data.py** - Download implied volatility history
   - Source: DoltHub post-no-preference/options
   - Coverage: 2019+
   - First inspects data density
   - Preferred method: dolt CLI; fallback: web API
   - Creates: `raw/iv_data.parquet`, `logs/04_iv_density_report.txt`

5. **05_factors.py** - Download Fama-French factors
   - Source: Ken French Data Library
   - Converts from percent to decimal
   - Creates: `raw/ff_factors.parquet`

6. **06_build_panels.py** - Build derived tables
   - **events.parquet**: Earnings events with SUE, CAR, market cap, etc.
     - SUE: (actual - estimate) / std of last 8 surprises
     - Abnormal returns: market-adjusted and CAPM-adjusted
     - CAR windows: [-5,-1], [0,+1], [0,+5], [+2,+20]
   - **iv_event_panel.parquet**: IV and RV from t-10 to t+10 around earnings
   - **daily_panel.parquet**: All stocks, all days, with returns and factors

### Configuration

Edit `config.py` to change:
- Date ranges
- CAR windows
- Rate limiting parameters
- Thresholds for conflicts

### Directory Structure

```
fods_data/
├── config.py              # Central configuration
├── run_all.py            # Orchestration script
├── generate_report.py    # Report generator
├── 01_universe.py
├── 02_prices.py
├── 03_earnings.py
├── 04_iv_data.py
├── 05_factors.py
├── 06_build_panels.py
├── raw/                  # Raw downloads (never overwritten)
│   ├── sp500_constituents.csv
│   ├── prices/
│   ├── earnings_yf.parquet
│   ├── earnings_av.parquet
│   ├── iv_data.parquet
│   └── ff_factors.parquet
├── processed/            # Cleaned & merged data
│   ├── universe.parquet
│   ├── membership.parquet
│   ├── prices_all.parquet
│   ├── earnings_merged.parquet
│   ├── events.parquet
│   ├── iv_event_panel.parquet
│   └── daily_panel.parquet
├── cache/                # API response caches
│   └── av_earnings/
├── logs/                 # Execution logs
│   ├── run_all.log
│   ├── 02_prices_failed.csv
│   └── 04_iv_density_report.txt
└── data_report.md        # Final data report
```

## Resumability

All scripts are resumable:
- **Prices**: Individual ticker files cached, skipped if complete
- **Earnings (AV)**: JSON responses cached, quota tracking
- **IV data**: Single file, skipped if exists
- **Panels**: Skipped if output exists

To re-download: Delete the relevant output file.

## No Look-Ahead Bias

Every feature at t0 uses only data available **before** the announcement:
- SUE uses rolling std of **past** surprises only
- Beta estimated on **[-250, -30]** before t0
- RV uses rolling window ending at current date

## Data Sources (All Free)

1. **S&P 500 constituents**: GitHub archive
2. **Prices**: Yahoo Finance (yfinance)
3. **Earnings**:
   - yfinance (has timestamps)
   - Alpha Vantage (free tier: 25 calls/day, better EPS estimates)
4. **IV**: DoltHub options database (open source)
5. **Factors**: Ken French Data Library

## Known Limitations

- **Survivorship bias**: Mitigated by using historical constituents, not today's list
- **IV coverage**: Starts 2019 only
- **Alpha Vantage quota**: 25 calls/day (run over multiple days or get premium key)
- **Yahoo rate limiting**: Pipeline includes delays and retries
- **Stock splits**: Not explicitly handled in EPS (may cause outliers)
- **Trading calendar**: Uses calendar days +1 for AMC, not trading day calendar

## Sanity Checks in Report

The `data_report.md` includes:
1. **SUE quintiles vs CAR**: Should show PEAD (Q5 > Q1)
2. **IV path t-10 to t+10**: Should show crush (peak at t-1, drop after)
3. **Data quality checks**: Duplicates, outliers, missing values
4. **Coverage statistics**: % events with EPS estimates, IV, etc.

## Citation

Data sources:
- S&P 500 constituents: fja05680/sp500, hanshof/sp500_constituents (GitHub)
- Prices & earnings: Yahoo Finance via yfinance
- Earnings estimates: Alpha Vantage
- Implied volatility: post-no-preference/options (DoltHub)
- Risk factors: Ken French Data Library

## Support

Check logs/ for detailed error messages.
Each script has its own .log file.

Failed tickers are logged in `logs/02_prices_failed.csv` with reasons.
