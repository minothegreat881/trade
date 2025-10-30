# ML Trading System - Phase 1: Data Pipeline

## Project Overview
Implementation of academic ML trading strategies based on:
- **Yan (2025)**: XGBoost momentum strategy with 5-day holding period
- **Gómez-Martínez et al. (2023)**: Sentiment-augmented ML trading
- **Kelly & Xiu (2023)**: Financial machine learning theoretical framework
- **Suárez-Cetrulo et al. (2023)**: Regime change detection

## Configuration
- **Ticker**: SPY (S&P 500 ETF)
- **Period**: 2020-01-01 to 2024-12-31 (5 years)
- **Holding**: 5 days (forward returns)
- **Train/Test**: 70/30 temporal split

## Data Pipeline Status
✅ Data collection (Yahoo Finance)
✅ Feature engineering (11 features)
✅ Target creation (5-day forward return)
✅ Train/test split (temporal, no overlap)
✅ Validation (automated tests)
✅ Data quality: 100/100

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Pipeline
```bash
# Run full data pipeline
python pipeline.py

# Expected output:
# [INFO] Downloaded 1257 rows
# [INFO] Created 11 features
# [INFO] Train: 834 rows, Test: 358 rows
# [INFO] ✓ Pipeline complete!
```

### 3. Validate Output
```bash
# Run automated validation
python validate_pipeline.py

# Expected output:
# 🎉 ALL TESTS PASSED!
# ✓ Pipeline output is valid
# ✓ Ready for modeling
```

### 4. Inspect Data
```bash
# Visual inspection
python inspect_data.py

# Opens matplotlib plots showing:
# - Target distribution
# - Returns over time
# - Volatility patterns
# - Trend indicators
```

### 5. Explore Interactively
```bash
# Launch Jupyter notebook
jupyter notebook explore_data_pipeline.ipynb

# Interactive exploration with 9 visualizations
# Experiments: uptrend vs downtrend, volatility analysis, etc.
```

## Project Structure

```
ml_trading_system/
├── config.py                      # Configuration file
├── pipeline.py                    # Main pipeline script
├── data_collector.py              # Data download module
├── feature_engineering.py         # Feature creation module
├── data_validator.py              # Data quality checks
├── validate_pipeline.py           # Pipeline validation
├── inspect_data.py                # Visual data inspection
├── explore_data_pipeline.ipynb    # Interactive Jupyter notebook
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/
│   ├── raw/
│   │   └── SPY_historical.csv         # Raw OHLCV data (1257 rows)
│   ├── processed/
│   │   ├── SPY_featured.csv           # Data with features (1192 rows)
│   │   ├── data_quality_report.txt    # Quality analysis
│   │   └── data_inspection.png        # Visual inspection plots
│   └── train_test/
│       ├── train_data.csv             # Full train set (834 rows)
│       ├── test_data.csv              # Full test set (358 rows)
│       ├── train_X.csv                # Train features only
│       ├── train_y.csv                # Train target only
│       ├── test_X.csv                 # Test features only
│       └── test_y.csv                 # Test target only
│
└── tests/
    ├── test_data_collector.py         # Data collection tests
    └── test_feature_engineering.py    # Feature & bias tests
```

## Features Created (11 total)

### Returns (4 features)
1. **return_1d**: 1-day return (daily momentum)
2. **return_5d**: 5-day return (weekly momentum)
3. **return_10d**: 10-day return (bi-weekly momentum)
4. **return_20d**: 20-day return (monthly momentum)

### Volatility (2 features)
5. **volatility_20d**: 20-day rolling volatility (short-term regime)
6. **volatility_60d**: 60-day rolling volatility (long-term regime)

### Volume (1 feature)
7. **volume_ratio**: Current volume vs 20-day average

### Price Position (1 feature)
8. **price_position**: Price position in 20-day range (0 = low, 1 = high)

### Trend (3 features)
9. **sma_20**: 20-day simple moving average
10. **sma_50**: 50-day simple moving average
11. **trend**: Binary indicator (1 if SMA20 > SMA50, else 0)

### Target Variable
12. **target**: 5-day forward return (properly shifted, NO look-ahead bias!)

## Data Quality Report

### Raw Data
- **Rows**: 1,257 (2020-01-02 to 2024-12-30)
- **Missing values**: 0
- **Date gaps**: None
- **Outliers**: 38 (mostly COVID crash - valid)

### Processed Data
- **Rows**: 1,192 (65 rows lost to rolling windows - 5.2%)
- **Features**: 11
- **Target**: 5-day forward return
- **NaN values**: 0
- **Quality Score**: 100/100 ✅

### Train Set
- **Rows**: 834 (70%)
- **Period**: 2020-03-30 to 2023-07-21
- **Covers**: COVID crash, 2021 bull, 2022 bear
- **Mean return**: 0.41% (5-day avg)
- **Std**: 2.54%

### Test Set
- **Rows**: 358 (30%)
- **Period**: 2023-07-24 to 2024-12-20
- **Fresh unseen data**: Yes
- **Mean return**: 0.42% (5-day avg)
- **Std**: 1.81%
- **Similar to train**: ✅ Yes

## Critical Validations ✅

### 1. No Look-Ahead Bias
Target at time `t` = return from `t` to `t+5` (properly shifted forward)
- ✅ Verified with manual calculation
- ✅ Test confirms features don't change with future data
- ✅ Correlation test passes (target vs return_1d < 0.95)

### 2. Temporal Train/Test Split
- ✅ Train: 2020-03-30 to 2023-07-21
- ✅ Test: 2023-07-24 to 2024-12-20
- ✅ No overlap (verified)
- ✅ Chronologically ordered

### 3. Feature Quality
- ✅ All features have variance
- ✅ No perfect correlations (avoiding multicollinearity)
- ✅ Ranges are reasonable
- ✅ COVID crash outliers present (valid, not removed)

## Usage Examples

### Basic Usage
```python
from data_collector import DataCollector
from feature_engineering import FeatureEngineer

# Download data
collector = DataCollector('SPY')
df = collector.download_historical('2020-01-01', '2024-12-31')

# Create features
engineer = FeatureEngineer()
df = engineer.create_basic_features(df)
df = engineer.create_target(df, horizon=5)

# Features are ready!
X = df[engineer.get_feature_names()]
y = df['target']
```

### Custom Configuration
```python
# Edit config.py
TICKER = "QQQ"  # NASDAQ-100 ETF
HOLDING_PERIOD = 10  # 10-day returns
START_DATE = "2015-01-01"  # Earlier start

# Run pipeline
python pipeline.py
```

### Run Tests
```bash
# Unit tests
pytest tests/ -v

# Critical tests included:
# ✓ Data download works
# ✓ Features created correctly
# ✓ NO look-ahead bias in target
# ✓ NO look-ahead bias in features
# ✓ Temporal split correct
```

## Troubleshooting

### Issue: Pipeline fails to download data
**Symptoms**: ConnectionError, empty DataFrame
**Fix**:
- Check internet connection
- Verify ticker symbol is correct
- Try: `python -c "import yfinance; print(yfinance.download('SPY', period='5d'))"`

### Issue: All NaN after feature creation
**Symptoms**: df_clean is empty
**Fix**:
- Need at least 60 days of data (for 60-day volatility)
- Check: `len(df_raw) > 100`
- Adjust date range to earlier start

### Issue: Tests fail
**Symptoms**: pytest failures
**Fix**:
- Run: `python validate_pipeline.py` to see which test fails
- Read error message carefully
- Most common: need internet connection for real data

### Issue: "ModuleNotFoundError"
**Symptoms**: Can't import modules
**Fix**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install missing module
pip install [module_name]
```

## Next Steps - Phase 2

After completing Phase 1, proceed to:

### Phase 2A: Baseline Model
- [ ] Implement Ridge Regression (Kelly & Xiu 2023)
- [ ] Cross-validation framework
- [ ] Performance metrics (Sharpe, Sortino)
- [ ] Baseline results

### Phase 2B: Advanced Model
- [ ] XGBoost implementation (Yan 2025)
- [ ] Hyperparameter tuning (Optuna)
- [ ] Feature importance analysis
- [ ] Model comparison

### Phase 2C: Enhanced Features
- [ ] Sentiment data (Fear & Greed Index)
- [ ] Crypto momentum (Bitcoin)
- [ ] Feature selection
- [ ] Regime detection

### Phase 3: Backtesting
- [ ] Transaction costs
- [ ] Slippage modeling
- [ ] Position sizing
- [ ] Risk management

### Phase 4: Live Trading
- [ ] Paper trading
- [ ] Real-time data feed
- [ ] Order execution
- [ ] Monitoring & alerts

## Performance Expectations

Based on academic literature:
- **Ridge Regression Baseline**: Sharpe ~0.5-0.8
- **XGBoost**: Sharpe ~0.8-1.2
- **With Sentiment**: Sharpe ~1.0-1.5

⚠️ **Important Notes:**
- These are realistic expectations
- Beware of overfitting (use cross-validation!)
- Look-ahead bias prevented (tested)
- Transaction costs will reduce performance
- Past performance ≠ future results

## Educational Value

This project demonstrates:

✅ **Production ML Pipeline**
- Modular design
- Error handling
- Logging and validation
- Comprehensive testing

✅ **Financial ML Best Practices**
- Temporal data handling
- Look-ahead bias prevention
- Proper train/test splits
- Feature engineering for markets

✅ **Code Quality**
- Type hints
- Google-style docstrings
- Unit tests (27 tests total)
- Configuration management

## References

1. **Yan (2025)**: "Machine Learning for Stock Prediction Based on Extreme Gradient Boosting"
2. **Gómez-Martínez et al. (2023)**: "Sentiment-Augmented ML Trading Strategies"
3. **Kelly & Xiu (2023)**: "Financial Machine Learning: Theoretical Foundations"
4. **Suárez-Cetrulo et al. (2023)**: "Regime Detection in Financial Time Series"

## License

For educational purposes only. Not financial advice.

## Contact & Support

- Issues: Check troubleshooting section
- Tests: `pytest tests/ -v`
- Validation: `python validate_pipeline.py`

---

**Status**: ✅ Phase 1 Complete
**Quality Score**: 100/100
**Ready for**: Phase 2 - Modeling
**Last Updated**: 2025-10-29
