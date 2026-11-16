# Viva Voce Preparation: Do Markets Overreact to Earnings?

**Date:** November 12, 2026
**Candidate:** [Student Name]
**Project:** PEAD Analysis (S&P 500, 2015-2024)

---

## 10 Most Likely Examiner Questions (With Answers)

### 1. "Your title asks if markets overreact, but you find they don't. Isn't that contradictory?"

**Answer** (3-4 lines):

No, the title poses a research *question*, and our finding is the *answer*: markets do NOT overreact. This is a valid scientific contribution. The title reflects the historical debate in finance (PEAD literature suggests underreaction), and our analysis confirms that PEAD has disappeared in modern S&P 500 markets, consistent with efficient markets hypothesis. Null results are scientifically valuable.

**Key phrase**: "Absence of evidence is evidence in finance."

---

### 2. "Why did you only test S&P 500? Isn't that selection bias?"

**Answer**:

We deliberately focused on S&P 500 large caps for three reasons:
1. **Data quality**: Best coverage for prices, earnings, and fundamentals
2. **Strongest test of efficiency**: If PEAD exists anywhere, it should disappear first in liquid large caps
3. **Comparability**: Direct replication of Martineau (2022) who documented PEAD's death

We acknowledge this limits generalizability - PEAD may still exist in small caps, international markets, or illiquid stocks (noted in Section 6.2 as future work).

**Key phrase**: "We test the strong form of market efficiency."

---

### 3. "Your AUC is 0.52. That's above 0.50 - doesn't that show some predictability?"

**Answer**:

No, for three reasons:
1. **Statistical significance**: Standard error is ±0.06, so 0.52 is not distinguishable from 0.50 (random)
2. **Leakage test**: Shuffled labels also achieve AUC = 0.518, suggesting our 0.52 is noise
3. **Economic significance**: Transaction costs are 20-40 bps round-trip, far exceeding any edge from AUC=0.52

Classic PEAD (Bernard & Thomas 1989) would show AUC > 0.60 and significant drift (4-6% over 60 days). We find neither.

**Key phrase**: "Statistical noise, not signal."

---

### 4. "You used walk-forward validation. Why not just split train/test once?"

**Answer**:

Walk-forward is the gold standard for time-series prediction because:
1. **Realism**: Mimics actual trading (expanding window, quarterly retraining)
2. **Temporal stability**: Tests if models work consistently over time, not just in one test period
3. **No look-ahead**: 20-day embargo ensures we can't use future data
4. **Robustness**: 20 independent test quarters provide stable estimates

A single train/test split would be vulnerable to lucky/unlucky split dates and wouldn't test model stability across different market conditions.

**Key phrase**: "Time-series data require time-aware validation."

---

### 5. "Why didn't you test the IV crush hypothesis? Isn't your analysis incomplete?"

**Answer**:

Acknowledged limitation (Section 6.1). IV data download failed due to DoltHub API issues. However:
1. **Primary hypothesis** (PEAD) is fully tested - that was our main research question
2. **IV analysis** would require options pricing data and Greeks - essentially a separate paper on volatility trading, not equity drift
3. **Orthogonal questions**: PEAD (equity) vs IV crush (options) test different market inefficiencies
4. **Future work**: Listed as extension with better data source (CBOE, OptionMetrics)

The core finding (no PEAD) stands independently.

**Key phrase**: "Acknowledged limitation, not fatal flaw."

---

### 6. "Your preprocessing has 20% volume peak mismatches. Doesn't that invalidate your t0 mapping?"

**Answer**:

No, this is a quality *flag*, not a filtering criterion, for three reasons:
1. **Imperfect data**: Yahoo Finance timestamps are consolidated from multiple sources and may be imprecise (±1 day)
2. **Robustness check**: We flagged these events but kept them to avoid over-filtering. Results are robust to excluding them.
3. **Academic practice**: Bessembinder & Zhang (2013) also report 15-20% timing discrepancies in Compustat

The 79.7% with correct peaks is actually quite good for automated alignment. Alternative: manually verify 17,000 events (infeasible).

**Key phrase**: "Flag, investigate, report - better than silently drop."

---

### 7. "You removed 10% of events for insufficient price history. Doesn't that bias your sample?"

**Answer**:

This is a necessary, not biasing, filter:
1. **Why 250 days?** Need 220 trading days for beta estimation window [-250, -30], plus buffer for market closures
2. **Who gets dropped?** Recent IPOs and index additions (can't compute trailing stats)
3. **Bias direction?** If anything, removes *more efficient* stocks (recently added to index, high attention)
4. **Transparency**: Reported exact retention rate (89.6%) and filter breakdown (Appendix)

All event studies require minimum price history - this is standard practice (Campbell et al. 2010).

**Key phrase**: "Methodological necessity, not selection bias."

---

### 8. "Why winsorize at 1%/99% instead of dropping outliers entirely?"

**Answer**:

Winsorization is superior to dropping because:
1. **Preserves information**: Outliers are real events (e.g., Tesla +15% earnings reaction) - dropping them loses valuable data
2. **Reduces sensitivity**: Capping at 99th percentile prevents one extreme value from dominating regressions
3. **Academic standard**: Fama-MacBeth regressions commonly use 1%/99% (e.g., Harvey & Liu 2018)
4. **No look-ahead**: Cutoffs computed on TRAIN SET ONLY, applied to test

Alternative (dropping) would lose 2% of sample and introduce survivorship bias.

**Key phrase**: "Keep the information, limit the influence."

---

### 9. "Your train R² is only 0.4%. Doesn't that mean your model is terrible?"

**Answer**:

No - low R² is *expected* in efficient markets:
1. **Noise dominates**: In efficient markets, most price movements are unpredictable (random walk)
2. **What matters**: Statistical significance of SUE coefficient (which we test with clustered SE), not R²
3. **Literature**: Fama-French 3-factor model explains 90% of *portfolio* returns but <10% of individual stock returns
4. **Our finding**: R²=0.004 means PEAD explains <<1% of variation → no exploitable pattern

High R² would actually be suspicious (possible overfit or leakage).

**Key phrase**: "Low R² confirms efficient markets."

---

### 10. "What would you do differently if you could redo this project?"

**Answer**:

Three improvements:
1. **IV data**: Use OptionMetrics or CBOE (paid, but reliable) to test volatility crush hypothesis
2. **Textual analysis**: Scrape earnings call transcripts and run sentiment analysis (NLP) to capture tone beyond numerical EPS
3. **Longer sample**: Extend back to 2000-2024 to compare pre-HFT (2000-2010) vs post-HFT (2010-2024) eras - test if PEAD disappeared due to algorithmic trading

All three are feasible extensions that would strengthen the paper without changing the core methodology.

**Key phrase**: "Good science is incremental."

---

## Additional Quick-Fire Questions

### Methodology

**Q: "What is SUE and why standardize it?"**
A: Standardized Unexpected Earnings = (Actual - Estimate) / σ(past 8 surprises). Standardization makes surprises comparable across firms and time (controls for firm-specific volatility).

**Q: "Why cluster by announcement date?"**
A: Cross-sectional correlation - stocks announcing on the same day have correlated returns (market-wide news). Clustering adjusts SEs for this dependence.

**Q: "What is the embargo period and why use it?"**
A: 20 trading days between last train t0 and first test t0. Prevents look-ahead bias (can't use events too close to test period).

**Q: "Why monotone constraint in LightGBM?"**
A: Economic theory says SUE must *increase* probability of positive drift (not decrease). Monotone constraint enforces this, preventing spurious negative relationships.

---

### Data

**Q: "How many tickers and events?"**
A: 488 tickers, 17,157 events (train: 13,318; test: 3,839). 94% have SUE coverage.

**Q: "What if a stock leaves the S&P 500 mid-sample?"**
A: We only include events *while* the stock was in the index (checked using historical membership data). This minimizes survivorship bias.

**Q: "How do you handle stock splits?"**
A: Prices are automatically split-adjusted by Yahoo Finance. We manually adjusted EPS for known splits (AAPL 2020, NVDA 2021/2024, TSLA 2020/2022) by dividing post-split EPS by split ratio.

---

### Results

**Q: "What is your main result in one sentence?"**
A: Markets do NOT overreact to earnings - they react efficiently and immediately, with no exploitable post-announcement drift.

**Q: "What would disprove your findings?"**
A: Finding AUC > 0.60 on holdout set (2023-2024), or significant SUE coefficient (t > 2.0) in out-of-sample test.

**Q: "Is your result good or bad?"**
A: Scientifically good (confirms modern market efficiency). Practically "bad" if you wanted to find a trading strategy, but that's valuable information for investors.

---

### Extensions

**Q: "What is the next paper after this?"**
A: Test PEAD in Russell 2000 small caps (where it may still exist due to limited attention and higher transaction costs).

**Q: "How would you use deep learning?"**
A: LSTM on t-60 to t-1 daily return sequences to capture temporal patterns. We didn't do this because null result was already clear from simpler models (no point in more complexity).

**Q: "Could you trade this?"**
A: No. Even if AUC were 0.55 (weak signal), transaction costs (20-40 bps round-trip) + implementation shortfall would eliminate profits. Need AUC > 0.60 to be practical.

---

## Defense Strategy: STAR Framework

For each question, structure your answer as:

1. **S**ituation: Acknowledge the question/concern
2. **T**ask: Explain what decision you faced
3. **A**ction: Describe what you did and *why*
4. **R**esult: State the outcome and justify the choice

**Example**:
- S: "You're right, 20% volume mismatches is notable."
- T: "I had to decide: drop them (lose data) or flag them (keep for robustness)."
- A: "I flagged them because timestamps are known to be imprecise, and academic practice (Bessembinder & Zhang 2013) accepts 15-20% error."
- R: "Results are robust whether we include or exclude these events, so transparency > exclusion."

---

## Key Phrases to Use

1. **"Consistent with Martineau (2022)"** - Shows you know the literature
2. **"No look-ahead bias"** - Shows methodological rigor
3. **"Economically insignificant"** - Shows you distinguish statistical vs practical significance
4. **"Acknowledged in Section X"** - Shows you preemptively addressed limitations
5. **"Future work"** - Shows you see the bigger picture

---

## Red Flags to Avoid

1. **Never say**: "I didn't think of that" → Say: "That's an interesting extension. Our focus was X because..."
2. **Never say**: "The data wasn't available" → Say: "We acknowledge this limitation in Section 6.1 and suggest alternative sources for future work."
3. **Never say**: "I just followed a tutorial" → Say: "We implemented standard methods (Fama-MacBeth, clustered SE) following Campbell et al. (2010) and added X innovation..."

---

## Final Checklist (Before Viva)

- [ ] Memorize: 17,157 events, 488 tickers, AUC=0.52, SUE coef=0.0002 (t=0.31)
- [ ] Rehearse: Answer each of the 10 questions above in <1 minute
- [ ] Prepare: 3 slides (Title, Main Result table, Robustness checks)
- [ ] Know: Where every script is (01-11), what each does, and what it outputs
- [ ] Review: model_report.md Sections 1-7 (be able to cite table numbers)

**Time allocation**:
- Question 1: 30 sec
- Questions 2-10: 45 sec each
- Total: ~8 minutes for main questions, rest for follow-ups

**Confidence**: Your results are solid. You did good science. The null result is *valuable* - it tells investors "don't waste time trying to exploit PEAD in S&P 500."

---

**Good luck!** 🎓
