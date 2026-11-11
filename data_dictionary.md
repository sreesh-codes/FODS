# Data Dictionary - model_panel.parquet

**Generated**: 2026-11-11 00:57:26

**Project**: Do Markets Overreact to Earnings?

---

## Dataset Overview

- **Total events**: 17,157
- **Train events (2015-2022)**: 13,318
- **Test events (2023-2024)**: 3,839
- **Tickers**: 485
- **Date range**: 2015-09-30 to 2024-12-20

---

## Column Descriptions

### Identifiers & Metadata

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Stock ticker symbol |
| `datetime` | datetime | Earnings announcement timestamp |
| `ann_date` | date | Announcement date |
| `t0` | date | First reaction date (trading day) |
| `fiscal_qtr` | period | Fiscal quarter (YYYYQQ) |
| `year` | int | Year of t0 |
| `quarter` | period | Calendar quarter of t0 |
| `split` | str | 'train' (2015-2022) or 'test' (2023-2024) |

### Raw Earnings Data

| Column | Type | Description |
|--------|------|-------------|
| `eps_estimate_yf` | float | EPS consensus estimate (split-adjusted) |
| `eps_actual_yf` | float | Reported EPS (split-adjusted) |
| `surprise` | float | Actual - estimate |
| `surprise_std` | float | Rolling std of last 8 surprises |
| `sue` | float | Standardized unexpected earnings (surprise / surprise_std) |
| `sue_wins` | float | SUE winsorized at 1%/99% (train cutoffs) |
| `sue_quintile` | int | SUE quintile (1-5) assigned within calendar quarter |

### Returns & CAR (Cumulative Abnormal Returns)

| Column | Type | Description |
|--------|------|-------------|
| `car_-5_-1` | float | Stock CAR from t-5 to t-1 (geometric) |
| `car_mkt_-5_-1` | float | Market CAR from t-5 to t-1 |
| `car_abn_-5_-1` | float | Abnormal return = stock - market |
| `car_-5_-1_wins` | float | Winsorized CAR (1%/99%, train cutoffs) |
| `car_abn_-5_-1_wins` | float | Winsorized abnormal CAR |
| `car_0_1` | float | Stock CAR from t0 to t1 (geometric) |
| `car_mkt_0_1` | float | Market CAR from t0 to t1 |
| `car_abn_0_1` | float | Abnormal return = stock - market |
| `car_0_1_wins` | float | Winsorized CAR (1%/99%, train cutoffs) |
| `car_abn_0_1_wins` | float | Winsorized abnormal CAR |
| `car_0_5` | float | Stock CAR from t0 to t5 (geometric) |
| `car_mkt_0_5` | float | Market CAR from t0 to t5 |
| `car_abn_0_5` | float | Abnormal return = stock - market |
| `car_0_5_wins` | float | Winsorized CAR (1%/99%, train cutoffs) |
| `car_abn_0_5_wins` | float | Winsorized abnormal CAR |
| `car_2_20` | float | Stock CAR from t2 to t20 (geometric) |
| `car_mkt_2_20` | float | Market CAR from t2 to t20 |
| `car_abn_2_20` | float | Abnormal return = stock - market |
| `car_2_20_wins` | float | Winsorized CAR (1%/99%, train cutoffs) |
| `car_abn_2_20_wins` | float | Winsorized abnormal CAR |
| `car_2_5_up` | binary | 1 if CAR[+2,+20] > 0, else 0 |

### Features (Raw)

| Column | Type | Description |
|--------|------|-------------|
| `price_tminus1` | float | Price at t-1 |
| `momentum_12_1` | float | 12-1 month momentum |
| `rv_21d` | float | 21-day realized volatility (annualized) |
| `iv_tminus1` | float | Implied volatility at t-1 (NA - not available) |
| `iv_rv_ratio` | float | IV/RV ratio (NA) |

### Features (Preprocessed)

| Column | Type | Description |
|--------|------|-------------|
| `*_wins` | float | Winsorized at 1%/99% using train-set cutoffs |
| `*_wins_std` | float | Cross-sectionally standardized (z-score within quarter) |
| `sector_*` | binary | One-hot encoding for GICS sectors (reference = first sector) |
| `timing_bmo` | binary | 1 if BMO, 0 if AMC, NA if unknown |
| `size_tercile` | str | 'small', 'medium', 'large' based on price (within quarter) |
| `size_*` | binary | One-hot encoding for size (reference = small) |

### Quality Flags

| Column | Type | Description |
|--------|------|-------------|
| `volume_peak_correct` | bool | True if volume peak at t0 or t0+1 |
| `is_clustered` | bool | True if >30 events on same announcement date |
| `in_sp500_at_t0` | bool | True if in S&P 500 index on t0 |
| `has_confounding` | bool | True if overlapping events within ±20 days |
| `t0_method` | str | How t0 was determined (BMO_same_day, AMC_next_day, etc.) |

---

## Preprocessing Notes

### Winsorization
- All continuous features winsorized at 1st/99th percentile
- Cutoffs computed on TRAINING SET (2015-2022) only
- Same cutoffs applied to test set (2023-2024)
- See `logs/winsorization_cutoffs.csv` for exact values

### Standardization
- Features standardized WITHIN each calendar quarter
- Z-score = (x - quarter_mean) / quarter_std
- Prevents look-ahead bias and controls for time variation

### SUE Quintiles
- Assigned WITHIN each calendar quarter (not globally)
- Q1 = lowest SUE (most negative surprise)
- Q5 = highest SUE (most positive surprise)
- Within-quarter assignment prevents look-ahead bias

### Missing Data
- SUE: 6% missing (first few events per ticker)
- IV: 100% missing (data not downloaded)
- RV: 22% missing (insufficient price history)
- See `logs/missingness_table.csv` for details

---

## Usage Notes

### For EDA
- Use `split == 'train'` for all exploratory decisions
- Test set (2023-2024) should NOT be examined during EDA

### For Modeling
- Use `*_wins_std` versions of features (winsorized + standardized)
- Include sector and timing dummies as controls
- Cluster standard errors by `t0` (announcement date)
- Target: `car_abn_2_20` for drift, `car_abn_0_1` for announcement

### Quality Controls
- Exclude `volume_peak_correct == False` for robustness
- Exclude `is_clustered == True` for clean sample
- Check `has_confounding == True` for sensitivity

