"""Configuration for FODS earnings project."""

import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
FIGURES_DIR = BASE_DIR / "figures"

# Date ranges
PRICE_START = "2014-10-01"  # Extra lead-in for rolling windows
PRICE_END = "2025-01-31"
ANALYSIS_START = "2015-01-01"
ANALYSIS_END = "2024-12-31"

# Train/test split for EDA (ground rule: hold out test years)
TRAIN_START = "2015-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

# API settings
AV_KEY = os.getenv("AV_KEY", "")
AV_CALLS_PER_MIN = 5
AV_CALLS_PER_DAY = 25

# Yahoo Finance settings
YF_DELAY = 0.5  # seconds between calls
YF_MAX_RETRIES = 3
YF_BACKOFF = [1, 5, 15]  # retry delays in seconds

# Market hours (ET)
BMO_CUTOFF = "09:30"
AMC_CUTOFF = "16:00"

# Analysis parameters
BETA_WINDOW = (-250, -30)  # days before t0
SUE_LOOKBACK = 8  # quarters
RV_WINDOW = 21  # days for realized volatility
EVENT_WINDOW = (-10, 10)  # days around earnings for IV analysis

# CAR windows
CAR_WINDOWS = [(-5, -1), (0, 1), (0, 5), (2, 20)]

# Data quality thresholds
EPS_DIFF_THRESHOLD = 0.02  # dollars
DATE_DIFF_THRESHOLD = 1  # trading days

# Wrangling parameters
MIN_PRICE_HISTORY = 250  # trading days before t0
MIN_PRICE_FUTURE = 20  # trading days after t0
VOLUME_LOOKBACK = 60  # days for median volume calculation
ABNORMAL_VOLUME_THRESHOLD = 1.5  # spike threshold for t0 validation
CLUSTER_THRESHOLD = 30  # max events per day before flagging
WINSORIZE_PCTILE = (1, 99)  # percentiles for winsorization

# Figure settings
FIG_DPI = 300
FIG_FORMAT_RASTER = "png"
FIG_FORMAT_VECTOR = "svg"

# Symbol mappings (Yahoo format)
SYMBOL_MAPPINGS = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}

# Known ticker renames/mergers (old -> new)
TICKER_RENAMES = {
    "FB": "META",
    "GOOGL": "GOOGL",  # exists
    "GOOG": "GOOG",  # exists
}

# ============================================================================
# MODELING PARAMETERS (Ground Rules)
# ============================================================================

# Random seeds for reproducibility
RANDOM_SEEDS = [42, 123, 456, 789, 1011]  # 5 seeds for deep learning
RANDOM_SEED = RANDOM_SEEDS[0]  # Default seed

# Walk-forward validation parameters
WF_START_YEAR = 2018  # First test quarter: 2018Q1
WF_END_YEAR = 2022    # Last test quarter: 2022Q4
WF_EMBARGO_DAYS = 20  # Trading days between last train t0 and first test t0
WF_VALIDATION_QUARTERS = 4  # Use last 4 quarters of training for hyperparameter tuning

# Holdout test set (run ONCE at the end)
HOLDOUT_START = "2023-01-01"
HOLDOUT_END = "2024-12-31"

# Model targets
PRIMARY_TARGET = "car_2_5_binary"  # CAR[+2,+5] up/down
SECONDARY_TARGET = "car_2_20_binary"  # CAR[+2,+20] up/down
REGRESSION_TARGET = "car_abn_2_20"  # For explanatory regressions

# Feature groups (for no-look-ahead validation)
PRE_EVENT_FEATURES = [
    # SUE and earnings
    'sue_wins_std',
    # Size and momentum
    'price_tminus1_wins_std', 'momentum_12_1_wins_std',
    # Volatility
    'rv_21d_wins_std',
    # Timing and sector (dummies handled separately)
    'timing_bmo',
]

# Categorical features (one-hot encoded)
SECTOR_DUMMIES = ['sector_Consumer Discretionary', 'sector_Consumer Staples',
                  'sector_Energy', 'sector_Financials', 'sector_Health Care',
                  'sector_Industrials', 'sector_Information Technology',
                  'sector_Materials', 'sector_Real Estate', 'sector_Utilities']
SIZE_DUMMIES = ['size_medium', 'size_small']

# All model features
ALL_FEATURES = PRE_EVENT_FEATURES + SECTOR_DUMMIES + SIZE_DUMMIES

# Clustering for standard errors
CLUSTER_VAR = 't0'  # Cluster by announcement date
FIRM_CLUSTER_VAR = 'ticker'  # For two-way clustering

# Transaction costs (basis points per side)
STOCK_COST_LOW = 10   # 10 bps for liquid large caps
STOCK_COST_HIGH = 20  # 20 bps for higher cost scenario

# Model registry
MODEL_REGISTRY_PATH = LOGS_DIR / "model_registry.csv"

# Deep learning parameters
DL_BATCH_SIZE = 64
DL_MAX_EPOCHS = 100
DL_PATIENCE = 10  # Early stopping patience
DL_LEARNING_RATE = 0.001

# Sequence model parameters
SEQ_LENGTH = 60  # Days from t-60 to t-1
SEQ_FEATURES = ['ret_abn', 'log_vol_abn', 'rv', 'iv']  # Per-day features

# Tree model parameters (initial search grid - will be tuned)
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': RANDOM_SEED,
    'monotone_constraints': {0: 1},  # SUE must be monotone increasing
}

# Metrics to track
METRICS = ['auc', 'log_loss', 'brier_score', 'rank_ic']

# Leakage test thresholds
LEAKAGE_AUC_THRESHOLD = 0.55  # Shuffled labels should get AUC ~0.50 ± 0.05
DRIFT_AUC_THRESHOLD = 0.60    # Above this, check for leakage before continuing
