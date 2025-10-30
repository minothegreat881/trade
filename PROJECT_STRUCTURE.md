# ML Trading System - Project Structure

## Directory Tree

```
ml_trading_system/
│
├── 📄 pipeline.py                     # Main entry point - run this!
├── 📄 data_collector.py               # Download & validate data
├── 📄 feature_engineering.py          # Create features & target
├── 📄 data_validator.py               # Data quality checks
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Usage instructions
├── 📄 PHASE1_COMPLETION_SUMMARY.md    # This completion summary
├── 📄 PROJECT_STRUCTURE.md            # This file
│
├── 📁 data/                           # All data files
│   ├── 📁 raw/                        # Raw downloaded data
│   │   └── SPY_historical.csv         # 1,257 rows × 7 columns
│   │
│   ├── 📁 processed/                  # Processed data with features
│   │   ├── SPY_featured.csv           # 1,192 rows × 20 columns
│   │   └── data_quality_report.txt    # Quality analysis
│   │
│   └── 📁 train_test/                 # Train/test splits
│       ├── train_data.csv             # Full train set (834 rows)
│       ├── test_data.csv              # Full test set (358 rows)
│       ├── train_X.csv                # Train features only
│       ├── train_y.csv                # Train target only
│       ├── test_X.csv                 # Test features only
│       └── test_y.csv                 # Test target only
│
└── 📁 tests/                          # Unit tests
    ├── test_data_collector.py         # Data collection tests
    └── test_feature_engineering.py    # Feature & bias tests
```

## Module Dependency Graph

```
pipeline.py (orchestrator)
    │
    ├─→ data_collector.py
    │       └─→ yfinance (Yahoo Finance API)
    │
    ├─→ feature_engineering.py
    │       └─→ pandas, numpy
    │
    └─→ data_validator.py
            └─→ pandas, numpy
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE FLOW                       │
└─────────────────────────────────────────────────────────────┘

    1. DOWNLOAD
    ┌─────────────┐
    │ Yahoo       │
    │ Finance API │
    └──────┬──────┘
           │ yfinance
           ↓
    ┌─────────────┐
    │ Raw OHLCV   │ 1,257 rows × 7 columns
    │ Data        │ (Date, Open, High, Low, Close, Volume, Adj Close)
    └──────┬──────┘
           │
           │ 2. VALIDATE
           ↓
    ┌─────────────┐
    │ Data        │ ✓ No missing values
    │ Validation  │ ✓ No date gaps
    └──────┬──────┘ ✓ No invalid data
           │
           │ 3. CREATE FEATURES
           ↓
    ┌─────────────┐
    │ 11 Features │ Returns (4)
    │ Created     │ Volatility (2)
    └──────┬──────┘ Volume (1), Position (1), Trend (3)
           │
           │ 4. CREATE TARGET
           ↓
    ┌─────────────┐
    │ Target      │ 5-day forward return
    │ Variable    │ (Properly shifted forward!)
    └──────┬──────┘
           │
           │ 5. CLEAN DATA
           ↓
    ┌─────────────┐
    │ Remove NaN  │ 1,192 rows remaining
    │ Rows        │ (65 rows lost - 5.2%)
    └──────┬──────┘
           │
           │ 6. TEMPORAL SPLIT
           ↓
    ┌──────────────┬──────────────┐
    │              │              │
    ↓              ↓              ↓
┌────────┐  ┌────────┐  ┌────────┐
│ Train  │  │ Test   │  │ Ready  │
│ 834    │  │ 358    │  │ for ML │
│ rows   │  │ rows   │  │ Models │
└────────┘  └────────┘  └────────┘
  70%         30%
```

## Feature Engineering Pipeline

```
Raw OHLCV Data
    │
    ├─→ [Returns Features]
    │   ├─→ return_1d    (1-day momentum)
    │   ├─→ return_5d    (5-day momentum)
    │   ├─→ return_10d   (10-day momentum)
    │   └─→ return_20d   (20-day momentum)
    │
    ├─→ [Volatility Features]
    │   ├─→ volatility_20d  (short-term regime)
    │   └─→ volatility_60d  (long-term regime)
    │
    ├─→ [Volume Features]
    │   └─→ volume_ratio  (current vs 20-day avg)
    │
    ├─→ [Price Position Features]
    │   └─→ price_position  (position in 20-day range)
    │
    ├─→ [Trend Features]
    │   ├─→ sma_20   (20-day moving average)
    │   ├─→ sma_50   (50-day moving average)
    │   └─→ trend    (1 if sma_20 > sma_50, else 0)
    │
    └─→ [Target Variable]
        └─→ target   (5-day forward return)
                     ⚠️ NO LOOK-AHEAD BIAS!
```

## Usage Flow

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline
python pipeline.py

# 3. Verify output
ls data/train_test/

# 4. Run tests
pytest tests/ -v
```

### Custom Configuration
```python
# Edit pipeline.py
TICKER = 'SPY'              # ← Change ticker
START_DATE = '2020-01-01'   # ← Change start
END_DATE = '2024-12-31'     # ← Change end
HOLDING_PERIOD = 5          # ← Change horizon
TRAIN_SPLIT = 0.7           # ← Change split
```

### Use as Library
```python
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from data_validator import generate_data_quality_report

# Step 1: Collect data
collector = DataCollector('AAPL')
df = collector.download_historical('2020-01-01', '2024-12-31')

# Step 2: Create features
engineer = FeatureEngineer()
df = engineer.create_basic_features(df)
df = engineer.create_target(df, horizon=5)

# Step 3: Validate
report = generate_data_quality_report(df, engineer.get_feature_names())

# Step 4: Use for ML
X = df[engineer.get_feature_names()]
y = df['target']
```

## Testing Structure

```
tests/
│
├── test_data_collector.py
│   ├─→ test_initialization_valid_ticker()
│   ├─→ test_download_historical_valid_dates()
│   ├─→ test_download_historical_invalid_date_format()
│   ├─→ test_validate_data_clean_data()
│   └─→ ... (12 tests total)
│
└── test_feature_engineering.py
    ├─→ test_create_returns()
    ├─→ test_create_volatility()
    ├─→ test_create_target_forward_shift() ⚠️ CRITICAL!
    ├─→ test_no_look_ahead_bias_in_features() ⚠️ CRITICAL!
    └─→ ... (15 tests total)
```

## Key Classes and Methods

### DataCollector
```python
class DataCollector:
    def __init__(ticker: str)
    def download_historical(start_date: str, end_date: str) -> pd.DataFrame
    def get_latest_data(days_back: int = 100) -> pd.DataFrame
    def validate_data(df: pd.DataFrame) -> dict
    def save_to_csv(df: pd.DataFrame, filename: str) -> None
```

### FeatureEngineer
```python
class FeatureEngineer:
    def create_basic_features(df: pd.DataFrame) -> pd.DataFrame
    def create_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame
    def create_returns(df: pd.DataFrame) -> pd.DataFrame
    def create_volatility(df: pd.DataFrame) -> pd.DataFrame
    def create_volume_features(df: pd.DataFrame) -> pd.DataFrame
    def create_price_position(df: pd.DataFrame) -> pd.DataFrame
    def create_trend_features(df: pd.DataFrame) -> pd.DataFrame
    def validate_features(df: pd.DataFrame) -> dict
    def get_feature_names() -> List[str]
```

### Data Validator Functions
```python
def check_missing_values(df: pd.DataFrame) -> dict
def check_date_gaps(df: pd.DataFrame, max_gap_days: int = 5) -> dict
def check_outliers(df: pd.DataFrame, threshold: float = 5.0) -> dict
def generate_data_quality_report(df: pd.DataFrame, feature_cols: List[str]) -> str
```

## Configuration Files

### requirements.txt
```
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.4.0
```

## Output File Formats

### Raw Data (CSV)
```
Date,Open,High,Low,Close,Volume,Dividends,Stock Splits
2020-01-02,324.87,325.46,321.53,324.87,123345600,0.0,0.0
...
```

### Featured Data (CSV)
```
Date,Open,High,Low,Close,Volume,...,return_1d,return_5d,...,target
2020-03-30,257.24,259.79,...,...,...,0.032476,0.173581,...,0.014532
...
```

### Train/Test Features (train_X.csv)
```
Date,return_1d,return_5d,return_10d,...,trend
2020-03-30,0.032476,0.173581,...,0
...
```

### Train/Test Target (train_y.csv)
```
Date,target
2020-03-30,0.014532
...
```

## Performance Metrics

### Pipeline Execution Time
- Download: ~5-10 seconds
- Feature Creation: ~1-2 seconds
- Validation: <1 second
- **Total**: ~10-15 seconds

### Memory Usage
- Raw data: ~100 KB
- Featured data: ~500 KB
- Train/test sets: ~400 KB
- **Total**: ~1 MB

### Data Dimensions
```
Raw Data:        1,257 rows × 7 columns
Featured Data:   1,192 rows × 20 columns
Train Features:  834 rows × 11 columns
Train Target:    834 rows × 1 column
Test Features:   358 rows × 11 columns
Test Target:     358 rows × 1 column
```

## Next Phase Preview

### Phase 2 Structure (Coming Soon)
```
ml_trading_system/
├── models/
│   ├── baseline_ridge.py      # Ridge Regression
│   ├── xgboost_model.py       # XGBoost
│   └── model_evaluator.py     # Performance metrics
│
├── backtesting/
│   ├── backtest_engine.py     # Backtest framework
│   └── performance_metrics.py # Sharpe, Sortino, etc.
│
└── features/
    ├── sentiment_features.py  # Fear & Greed Index
    └── crypto_features.py     # Bitcoin momentum
```

---

**Status**: Phase 1 Complete ✅
**Next**: Phase 2 - Model Development
**Date**: 2025-10-29
