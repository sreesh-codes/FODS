# Data Wrangling Log

**Generated**: 2026-11-11 00:35:10

## Summary Statistics

| Stage | Events | Change |
|-------|--------|--------|
| Initial events | 19,153 |  |
| After t0 mapping | 19,000 | -153 |
| After t0 validation | 19,148 | +148 |
| After S&P 500 filter | 19,108 | -40 |
| After deduplication | 19,023 | -85 |
| After price window check | 17,157 | -1866 |
| After confounding check | 17,157 | 0 |
| Final events | 17,157 | 0 |

## Flags

| Flag | Count | Description |
|------|-------|-------------|
| Unknown timing (inferred from volume) | 153 | |
| Wrong volume peak | 3,822 | |
| Clustered announcement dates | 6,765 | |
| Confounding events | 0 | |

## Rules Applied

### Volume peak validation

**Events flagged**: 3,822

**Description**: 3822 events have volume peak outside t0/t0+1 window

**Action**: Flagged but not dropped - may indicate incorrect announcement date


### S&P 500 membership filter

**Events removed**: 40

**Description**: Stock must be in S&P 500 index on t0

**Reason**: Ensures consistent universe and avoids survivor bias


### Duplicate removal

**Events removed**: 85

**Description**: One event per ticker per fiscal quarter

**Reason**: Multiple announcements for same quarter (restatements, duplicates in source data)


### Price data window

**Events removed**: 1,866

**Description**: Need 250 days before t0 and 20 days after

**Reason**: Required for beta estimation and CAR calculation


### Confounding events

**Events flagged**: 0

**Description**: Events with another earnings announcement within ±20 days

**Action**: Flagged but not dropped - can exclude in robustness checks


### Clustered announcement dates

**Events flagged**: 6,765

**Description**: Dates with >30 announcements

**Action**: Flagged for clustered standard errors in analysis


