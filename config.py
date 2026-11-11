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
