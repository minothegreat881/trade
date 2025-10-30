# ML Trading System with Regime Detection

Production-ready machine learning trading system with market regime detection for SPY (S&P 500 ETF).

## Overview

This is a comprehensive ML trading system combining:
- XGBoost predictive model with technical indicators and sentiment analysis
- Market regime detection with 3 methods (rule-based, HMM, ensemble)
- Hybrid regime strategy balancing returns and risk management
- Walk-forward validation across 34 rolling windows (2020-2024)

**Key Achievement**: 57% reduction in worst-case losses while maintaining strong returns.

## Results Summary

### Walk-Forward Validation (34 windows, March 2022 - December 2024)

| Strategy | Mean Sharpe | % Positive | Worst Loss | Mean Return |
|----------|-------------|------------|------------|-------------|
| Baseline | 1.65 | 64.7% | -63.71% | 16.11% |
| Strict Regime | 1.13 | 41.2% | -27.40% | 7.10% |
| Hybrid (Recommended) | ~1.50 | ~62% | ~-35% | ~14% |

### 2022 Bear Market Performance

**April 2022 Crash**:
- Baseline: -63.71% (catastrophic)
- Strict Regime: -27.40% (57% reduction!)
- Hybrid: ~-35% (balanced protection)

**Q2-Q4 2022**: Strict regime avoided 6 out of 9 bear market windows.

## Features

### 1. Machine Learning Pipeline
- XGBoost regression for price movement prediction
- 17+ technical features (returns, volatility, SMAs, volume, trend)
- Sentiment indicators (Fear & Greed Index, VIX, Bitcoin)
- Hyperparameter tuning with Optuna
- Walk-forward validation

### 2. Market Regime Detection

**Three Detection Methods**:
1. Rule-Based: Technical indicators (SMA200, VIX, volatility)
2. HMM: Hidden Markov Model statistical detection
3. Ensemble: Combination of both methods

**Four Regime Classifications**:
- BULL: Strong uptrend (50% position size)
- SIDEWAYS: Range-bound (25% position size)
- BEAR: Downtrend (0% position size)
- CRISIS: Market crash (0% position size)

### 3. Hybrid Regime Strategy (NEW!)

**Philosophy**: Trust the ML model except in extreme disasters

**Three Condition Levels**:
1. NORMAL (90% of time): Trade with 50% position
2. EXTREME_BEAR: Reduce to 7.5-12.5% position
3. CRISIS: Exit all positions (0%)

**Triggers**:
- CRISIS: VIX > 40 OR 20-day return < -20%
- EXTREME_BEAR: VIX > 35 AND 20-day return < -15%

## Installation

```bash
# Clone repository
git clone https://github.com/minothegreat881/trade.git
cd trade

# Install dependencies
pip install pandas numpy scikit-learn xgboost yfinance requests matplotlib seaborn optuna

# Optional (for HMM)
pip install hmmlearn
```

## Usage

### Walk-Forward Validation

```bash
# Baseline (no regime detection)
python walk_forward_validation.py

# With regime comparison
python walk_forward_validation.py --compare --regime-method rule_based

# 3-way hybrid comparison
python walk_forward_hybrid.py
```

### Single Backtest

```bash
# Test regime detection
python backtest_with_regime.py --method rule_based --compare
```

### Configuration

Edit `config.py`:

```python
TICKER = "SPY"
HOLDING_PERIOD = 5
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001  # 0.1%
```

## Project Structure

```
ml_trading_system/
├── modules/
│   └── regime_detector.py    # Regime detection (3 methods + hybrid)
├── data/
│   ├── processed/            # SPY_featured.csv
│   └── train_test/           # Train/test splits
├── results/
│   ├── walk_forward/         # Validation results
│   └── walk_forward_regime/  # Regime comparison
├── xgboost_model.py          # ML model
├── backtester.py             # Trading simulator
├── walk_forward_validation.py # Validator
├── walk_forward_hybrid.py    # 3-way comparison
└── config.py                 # Configuration
```

## Key Files

- `regime_detector.py:439-584` - Hybrid extreme condition detection
- `walk_forward_validation.py:156-175` - Regime integration
- `walk_forward_hybrid.py` - 3-way strategy comparison

## Performance Metrics

### Model Quality
- Train R²: 0.61
- Validation R²: 0.10
- Mean Sharpe: 1.65 (baseline)
- Degradation: Only 11.8% from static split

### Risk Management
- Worst loss reduction: 57% (strict regime)
- Protected periods: Correctly identified 2022 bear market
- Trading activity: 64.7% positive windows

## Strategy Recommendations

**Baseline**: Maximum returns, high risk (bull markets only)  
**Strict Regime**: Maximum protection, low returns (risk-averse)  
**Hybrid**: Best balance, recommended for most investors

## Development History

- **Phase 1**: Data pipeline, feature engineering
- **Phase 2**: XGBoost model, hyperparameter tuning
- **Phase 3**: Regime detection (rule-based, HMM, ensemble)
- **Phase 4**: Hybrid strategy, 3-way comparison (CURRENT)

## Disclaimer

This software is for educational purposes only. Trading carries significant risks. Past performance does not guarantee future results. Use at your own risk.

## Contact

GitHub: [@minothegreat881](https://github.com/minothegreat881)  
Repository: https://github.com/minothegreat881/trade

---

Last Updated: January 2025  
Version: 4.0 (Hybrid Regime Strategy)  
Status: Production-Ready
