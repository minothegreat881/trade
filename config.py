"""
Configuration for ML Trading System
Fill in your decisions from Step 1
"""

# YOUR DECISIONS FROM STEP 1
TICKER = "SPY"  # S&P 500 ETF
HOLDING_PERIOD = 5  # Target holding period (days)

# Backtesting parameters
INITIAL_CAPITAL = 10000  # Starting capital in USD
TRANSACTION_COST = 0.001  # Transaction cost (0.1%)

# Data period
START_DATE = "2020-01-01"  # Backtest start
END_DATE = "2024-12-31"    # Backtest end

# Train/test split
TRAIN_RATIO = 0.70  # 70% train, 30% test

# Directories
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
TRAIN_TEST_DIR = "data/train_test"
LOG_DIR = "logs"

# Features to create
BASIC_FEATURES = [
    'return_1d',
    'return_5d',
    'return_10d',
    'return_20d',
    'volatility_20d',
    'volatility_60d',
    'volume_ratio',
    'price_position',
    'sma_20',
    'sma_50',
    'trend'
]

# Validation settings
MAX_MISSING_PCT = 0.01  # Maximum 1% missing data allowed
OUTLIER_THRESHOLD = 5   # Z-score threshold for outliers

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

print(f"[OK] Config loaded: {TICKER} from {START_DATE} to {END_DATE}")
print(f"[OK] Holding period: {HOLDING_PERIOD} days")
print(f"[OK] Train/test split: {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}")
