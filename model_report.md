# Do Markets Overreact to Earnings?
## PEAD Analysis Report: S&P 500, 2015-2024

**Date:** November 12, 2026
**Author:** [Student Name]
**Data:** 17,157 earnings events, 488 S&P 500 constituents
**Period:** Train 2015-2022 (13,318 events), Holdout 2023-2024 (3,839 events)

---

## Executive Summary

This study investigates whether markets overreact or underreact to earnings announcements by testing for Post-Earnings Announcement Drift (PEAD) in S&P 500 stocks from 2015-2024.

### Main Findings:

1. **No Exploitable Drift**: Markets do NOT overreact to earnings. There is no statistically significant post-announcement drift (CAR[+2,+20] ≈ 0%, p=0.75).

2. **Efficient Immediate Reaction**: Markets react efficiently and immediately to earnings surprises (CAR[0,+1] = 1.36% per std dev SUE, p<0.001).

3. **No Predictive Power**: Machine learning models achieve AUC ≈ 0.52 (barely above random), confirming no exploitable trading signals.

4. **Conclusion**: **Markets are efficient** for S&P 500 large caps. PEAD has disappeared, consistent with Martineau (2022).

---

## 1. Research Questions

### Primary Hypotheses:

**H1 (PEAD Exists)**: Do earnings surprises predict subsequent drift?
→ **REJECTED**. No significant drift detected (t=0.31, p=0.75).

**H2 (Predictability)**: Can we predict drift direction using SUE and firm characteristics?
→ **REJECTED**. Best model AUC = 0.52 (no better than random).

**H3 (Economic Significance)**: Is any detected drift large enough to trade profitably after costs?
→ **N/A**. No drift to exploit.

---

## 2. Data & Methodology

### 2.1 Sample Construction

| **Component** | **Details** |
|---------------|-------------|
| Universe | S&P 500 historical constituents (2015-2024) |
| Tickers | 503 unique tickers, 488 with complete data |
| Events (raw) | 19,153 earnings announcements |
| **Events (clean)** | **17,157 (89.6% retention)** |
| Train period | 2015-2022 (13,318 events, 77.6%) |
| Test period (held out) | 2023-2024 (3,839 events, 22.4%) |

### 2.2 Data Sources (All Free)

1. **S&P 500 constituents**: GitHub archive (fja05680/sp500, hanshof/sp500_constituents)
2. **Prices (OHLCV)**: Yahoo Finance via `yfinance` (1.23M rows)
3. **Earnings**: yfinance + Alpha Vantage API
4. **Risk factors**: Ken French Data Library (Mkt-RF, SMB, HML)

### 2.3 Event Alignment & Validation

**t0 Mapping**:
- BMO (before market open): t0 = announcement date
- AMC (after market close): t0 = next trading day
- Unknown timing: Inferred from volume spikes

**Quality Checks**:
- Volume peak validation: 79.7% have peak at t0/t0+1 ✓
- Clustered dates: 39.4% on high-announcement days (flagged, not dropped)
- Duplicate quarters: Removed 85 events (0.4%)
- Insufficient price history: Removed 1,866 events (9.7%)

### 2.4 Feature Construction (No Look-Ahead Bias)

**Standardized Unexpected Earnings (SUE)**:
```
SUE_t = (Actual_t - Estimate_t) / σ(Surprise_{t-1, ..., t-8})
```
- Rolling std uses PAST 8 quarters only
- Within-quarter standardization (cross-sectional z-score)
- Coverage: 94.0% of events (16,132/17,157)

**Abnormal Returns**:
- Market-adjusted: CAR_abn = CAR_stock - CAR_market
- Windows: [-5,-1] (pre-announcement), [0,+1] (announcement), [+2,+20] (drift)

**Control Variables**:
- Size: Market cap proxy, tercile dummies
- Momentum: 12-1 month returns (excluding recent month)
- Volatility: 21-day realized volatility
- Sector: GICS sector dummies (10 sectors)

All features winsorized at 1%/99% (train-set cutoffs) and cross-sectionally standardized.

---

## 3. Explanatory Analysis Results

### 3.1 Regression Analysis

**Dependent Variable**: CAR[+2,+20] (abnormal return, drift window)
**Method**: OLS with clustered standard errors (by announcement date)

| **Model** | **SUE Coefficient** | **t-statistic** | **p-value** | **R²** | **N** |
|-----------|---------------------|-----------------|-------------|--------|-------|
| OLS (date cluster) | 0.0002 | 0.31 | 0.75 | 0.004 | 9,628 |
| OLS (sector FE) | 0.0002 | 0.35 | 0.73 | 0.007 | 9,628 |
| OLS (two-way cluster) | 0.0002 | 0.32 | 0.75 | 0.004 | 9,628 |
| Fama-MacBeth | 0.0008 | 1.63 | 0.11 | 27 periods | 9,628 |

**Interpretation**:
- SUE coefficient ≈ **0.02 bps** over 20 days (economically zero)
- NOT statistically significant at any conventional level
- Robust to clustering method, sector controls, and time-series regression

### 3.2 Announcement Reaction (Immediate)

**Dependent Variable**: CAR[0,+1] (announcement window)

| **Model** | **SUE Coefficient** | **t-statistic** | **R²** |
|-----------|---------------------|-----------------|--------|
| OLS (date cluster) | **0.0136** | **19.58*** | 0.059 |
| Fama-MacBeth | **0.0137** | **22.98*** | 27 periods |

**Interpretation**:
- 1 std dev SUE → **1.36% immediate reaction** (highly significant)
- Market reacts quickly and completely to earnings news
- No underreaction → no subsequent drift

### 3.3 Robustness Checks

| **Specification** | **SUE Coefficient** | **t-statistic** |
|-------------------|---------------------|-----------------|
| **Base case** | 0.0002 | 0.31 |
| Exclude 2020 (COVID) | 0.0005 | 0.86 |
| 2015-2018 only | 0.0010 | 1.26 |
| 2019-2022 only | -0.0003 | -0.33 |
| Small tercile | 0.0015 | 1.23 |
| Medium tercile | 0.0003 | 0.23 |
| Large tercile | -0.0010 | -1.01 |

**All robustness checks confirm null result** (no drift in any subsample).

---

## 4. Predictive Models Results

### 4.1 Walk-Forward Validation Setup

- **Method**: Expanding window, quarterly retraining
- **Test periods**: 2018Q1 to 2022Q4 (20 quarters)
- **Embargo**: 20 trading days between last train t0 and first test t0
- **Target**: CAR[+2,+5] up/down (binary classification)
- **Metric**: AUC-ROC (area under curve)

### 4.2 Model Performance

| **Model Type** | **Model Name** | **AUC Mean** | **AUC Std** | **Quarters** |
|----------------|----------------|--------------|-------------|--------------|
| **Baseline** | Base rate (always 50%) | 0.5000 | 0.0000 | 20 |
| **Baseline** | SUE quintile rule | 0.5026 | 0.0241 | 20 |
| **Logistic** | L2 (Ridge) | **0.5200** | 0.0613 | 20 |
| **Logistic** | L1 (Lasso) | 0.5195 | 0.0613 | 20 |
| **Tree** | LightGBM (monotone) | 0.5115 | 0.0489 | 20 |
| **Tree** | Random Forest | 0.5125 | 0.0714 | 20 |
| **Tree** | XGBoost | 0.5117 | 0.0530 | 20 |

**Best Model**: Logistic L2 (AUC = 0.52 ± 0.06)

### 4.3 Leakage Test

**Shuffled labels AUC**: 0.5181 ± 0.0251
**Status**: ✓ PASSED (AUC < 0.55, no data leakage detected)

### 4.4 Interpretation

- **AUC ≈ 0.50**: No better than random coin flip
- **Economic significance**: Zero. Transaction costs (10-20 bps per side) would eliminate any edge.
- **Comparison to literature**: Consistent with Martineau (2022) showing PEAD disappeared post-2005 for large caps.

---

## 5. Why No PEAD in S&P 500?

### 5.1 Academic Consensus

**Classic PEAD** (Bernard & Thomas 1989, Foster et al. 1984):
- Small caps, 1970s-1990s
- SUE Q5-Q1 spread: 4-6% over 60 days
- Attributed to: Limited attention, costly arbitrage, slow diffusion

**Modern Evidence** (Martineau 2022, Chordia et al. 2009):
- PEAD has **disappeared** for S&P 500 large caps
- Why? Algorithmic trading, lower costs, faster information diffusion
- Remains only in: Small caps, illiquid stocks, international markets

### 5.2 Our Findings Support Efficiency

| **Observation** | **Implication** |
|-----------------|-----------------|
| No drift (CAR[+2,+20] ≈ 0) | Market fully impounds news at announcement |
| Strong immediate reaction (1.36%) | No underreaction |
| AUC ≈ 0.50 | No predictable patterns |
| Robust across time/size | Not a subsample artifact |

**Conclusion**: S&P 500 markets are **informationally efficient** with respect to earnings announcements (consistent with semi-strong EMH).

---

## 6. Limitations & Future Work

### 6.1 Limitations

1. **Survivorship bias** (mitigated): Used historical constituents, but index changes may introduce bias.
2. **Sample period**: 2015-2024 may be atypically efficient (algorithmic trading era).
3. **Transaction costs**: Assumed 10-20 bps; real costs vary by broker, size, market conditions.
4. **IV data**: Not downloaded (DoltHub API issues). Cannot test IV crush hypothesis.
5. **Alternative targets**: Did not test CAR[+2,+60] (longer drift window) or regression on CAR magnitude.

### 6.2 Extensions

1. **Compare to small caps**: Test Russell 2000 or micro-caps (PEAD may still exist).
2. **International markets**: Test emerging markets (slower information diffusion).
3. **Textual analysis**: Incorporate earnings call transcripts (sentiment, tone).
4. **Intraday data**: Test high-frequency reaction patterns (first hour, day).
5. **Options strategies**: Test volatility crush trades (short straddles pre-earnings).

---

## 7. Viva Defense Prep

### Key Talking Points:

**Q: "Your title asks if markets overreact, but you find efficient markets. Isn't that contradictory?"**

A: The title poses a research question. The answer is "No, markets do not overreact - they react efficiently." This is a valid scientific finding. The title reflects the historical debate (PEAD literature), and our contribution is confirming that PEAD has disappeared in modern large-cap markets.

**Q: "Why not test smaller stocks where PEAD might still exist?"**

A: Our focus on S&P 500 is deliberate:
1. Data quality is highest for large caps
2. Tests the strongest form of market efficiency
3. Directly comparable to Martineau (2022)
4. Most relevant for institutional investors

Small caps would be an excellent extension (acknowledged in Section 6.2).

**Q: "Your AUC is 0.52. Isn't that slightly above random?"**

A: Yes, but not statistically or economically significant:
1. Standard error is ±0.06 → not distinguishable from 0.50
2. Leakage test shows shuffled labels also get 0.518
3. Transaction costs (20-40 bps round-trip) >> predicted edge
4. Classic PEAD would show AUC > 0.60

**Q: "Why use walk-forward validation instead of simple train/test split?"**

A: Walk-forward is the gold standard for time-series prediction:
1. Mimics real trading (expanding window, quarterly retraining)
2. Prevents look-ahead bias (20-day embargo)
3. Tests model stability over time
4. More realistic than single hold-out test

**Q: "You didn't test the IV crush hypothesis. Isn't that incomplete?"**

A: Acknowledged limitation. IV data download failed (DoltHub API issue). However:
1. Primary hypothesis (PEAD) is fully tested
2. IV analysis would be a separate paper (options pricing, not equity drift)
3. Acknowledged in Section 6.1
4. Could be future work with better data source

---

## 8. References

1. **Bernard, V. L., & Thomas, J. K. (1989)**. Post-earnings-announcement drift: Delayed price response or risk premium? *Journal of Accounting Research*, 27, 1-36.

2. **Foster, G., Olsen, C., & Shevlin, T. (1984)**. Earnings releases, anomalies, and the behavior of security returns. *The Accounting Review*, 59(4), 574-603.

3. **Martineau, C. (2022)**. Rest in peace post-earnings announcement drift. *Critical Finance Review*, 11(1), 153-184.

4. **Chordia, T., Goyal, A., Sadka, G., Sadka, R., & Shivakumar, L. (2009)**. Liquidity and the post-earnings-announcement drift. *Financial Analysts Journal*, 65(4), 18-32.

5. **Fama, E. F. (1970)**. Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2), 383-417.

---

## Appendix: Data Quality Summary

| **Metric** | **Value** |
|------------|-----------|
| **Raw events** | 19,153 |
| **Final events** | 17,157 (89.6%) |
| **Filtering breakdown** | |
| - Failed t0 inference | 5 (0.03%) |
| - Not in S&P 500 at t0 | 40 (0.2%) |
| - Duplicate quarters | 85 (0.4%) |
| - Insufficient price history | 1,866 (9.7%) |
| **Quality flags** | |
| - Volume peak mismatch | 3,480 (20.3%) |
| - Clustered dates | 6,765 (39.4%) |
| **SUE coverage** | 16,132 (94.0%) |
| **Complete feature coverage** | 9,628 (56.1%) |

**Train/Test Split (held out)**:
- Train 2015-2022: 13,318 events (77.6%)
- Test 2023-2024: 3,839 events (22.4%)

---

**Report Generated**: November 12, 2026
**Contact**: [Student Email]
**Code Repository**: [GitHub Link]
**Data Availability**: Reproducible pipeline (all free data sources)
