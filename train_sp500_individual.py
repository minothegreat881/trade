"""
TRAIN INDIVIDUAL XGBOOST MODELS FOR TOP 50 S&P 500 STOCKS
==========================================================

Strategy: One model per stock
- 50 separate XGBoost models
- Each learns stock-specific patterns
- Walk-forward validation
- Save all models and metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from datetime import datetime
import joblib
import json
import warnings
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

print("="*80)
print("TRAIN INDIVIDUAL XGBOOST MODELS - TOP 50 S&P 500")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# 1. LOAD TOP 50 STOCKS
# ================================================================

print("\n[1/5] Loading Top 50 stocks...")

summary = pd.read_csv('data/sp500_top50_summary.csv')
print(f"  Found {len(summary)} stocks")

# Load all stocks
stocks = {}
for ticker in summary['Ticker']:
    file_path = f'data/sp500_top50/{ticker}_features.csv'
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    stocks[ticker] = df

print(f"  Loaded {len(stocks)} stocks successfully")


# ================================================================
# 2. XGBOOST PARAMETERS (OPTIMIZED FOR SPY)
# ================================================================

print("\n[2/5] Setting up XGBoost parameters...")

# These params worked well for SPY - use same for all stocks
XGB_PARAMS = {
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.05,
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

print(f"  Parameters: {XGB_PARAMS}")


# ================================================================
# 3. TRAIN MODELS
# ================================================================

print("\n[3/5] Training models for each stock...")
print("  (This will take several minutes...)\n")

# Create output directories
Path('models/sp500').mkdir(parents=True, exist_ok=True)
Path('results/sp500').mkdir(parents=True, exist_ok=True)

all_results = []

for idx, ticker in enumerate(summary['Ticker'], 1):
    print(f"  [{idx}/50] Training {ticker}...", end=' ')

    try:
        df = stocks[ticker].copy()

        # Exclude columns
        exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Adj Close',
                       'Volume', 'Dividends', 'Stock Splits', 'fear_greed_classification']

        # Get feature columns (only those that exist)
        feature_cols = [col for col in df.columns
                       if col not in exclude_cols and col != 'target']

        if len(feature_cols) == 0:
            print("ERROR - No features!")
            continue

        # Check if target exists
        if 'target' not in df.columns:
            print("ERROR - No target column!")
            continue

        X = df[feature_cols]
        y = df['target']

        # Drop NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            print(f"ERROR - Only {len(X)} samples!")
            continue

        # Train/Test split (chronological)
        split_idx = int(len(X) * 0.8)

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        if len(X_test) < 20:
            print(f"ERROR - Test set too small ({len(X_test)})!")
            continue

        # Train XGBoost
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],
                 verbose=False)

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Metrics
        train_corr, _ = spearmanr(y_train, train_pred)
        test_corr, _ = spearmanr(y_test, test_pred)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        # Trading signals
        test_signals = (test_pred > 0).astype(int)
        test_returns = y_test.values

        # Strategy returns
        strategy_returns = test_returns * (test_signals * 2 - 1)  # Long only when signal=1

        # Calculate metrics
        avg_return = strategy_returns.mean()
        volatility = strategy_returns.std()
        sharpe = (avg_return / volatility) * np.sqrt(252) if volatility > 0 else 0

        # Win rate
        trades = strategy_returns[test_signals == 1]
        win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0

        # Max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Total return
        total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0

        # Save model
        model_file = f'models/sp500/{ticker}_xgb.pkl'
        joblib.dump(model, model_file)

        # Save metrics
        metrics = {
            'ticker': ticker,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(feature_cols),
            'train_corr': float(train_corr),
            'test_corr': float(test_corr),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'sharpe': float(sharpe),
            'total_return': float(total_return),
            'avg_return': float(avg_return),
            'volatility': float(volatility),
            'win_rate': float(win_rate),
            'max_drawdown': float(max_drawdown),
            'num_trades': int(trades.sum() if len(trades) > 0 else 0),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        metrics_file = f'results/sp500/{ticker}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        all_results.append(metrics)

        print(f"OK | Sharpe: {sharpe:.2f} | Corr: {test_corr:.3f} | Win: {win_rate:.1%}")

    except Exception as e:
        print(f"ERROR - {str(e)}")
        continue


# ================================================================
# 4. CREATE SUMMARY
# ================================================================

print(f"\n[4/5] Creating summary report...")

if len(all_results) == 0:
    print("  ERROR: No models trained successfully!")
    exit(1)

results_df = pd.DataFrame(all_results)

# Sort by Sharpe ratio
results_df = results_df.sort_values('sharpe', ascending=False)

# Save summary
summary_file = 'results/sp500/training_summary.csv'
results_df.to_csv(summary_file, index=False)

print(f"  Saved summary to {summary_file}")
print(f"  Successfully trained {len(results_df)} models")


# ================================================================
# 5. DISPLAY RESULTS
# ================================================================

print("\n[5/5] Training Results")
print("="*80)

print(f"\nSuccessfully trained: {len(results_df)}/50 models")
print(f"Average test correlation: {results_df['test_corr'].mean():.3f}")
print(f"Average Sharpe ratio: {results_df['sharpe'].mean():.2f}")
print(f"Average win rate: {results_df['win_rate'].mean():.1%}")

print("\n" + "="*80)
print("TOP 10 MODELS BY SHARPE RATIO")
print("="*80)

top10 = results_df.head(10)
print("\nTicker | Sharpe | Return | WinRate | Corr | Trades")
print("-" * 60)
for _, row in top10.iterrows():
    print(f"{row['ticker']:6s} | {row['sharpe']:6.2f} | {row['total_return']:6.1%} | "
          f"{row['win_rate']:7.1%} | {row['test_corr']:4.2f} | {row['num_trades']:6.0f}")

print("\n" + "="*80)
print("BOTTOM 10 MODELS BY SHARPE RATIO")
print("="*80)

bottom10 = results_df.tail(10)
print("\nTicker | Sharpe | Return | WinRate | Corr | Trades")
print("-" * 60)
for _, row in bottom10.iterrows():
    print(f"{row['ticker']:6s} | {row['sharpe']:6.2f} | {row['total_return']:6.1%} | "
          f"{row['win_rate']:7.1%} | {row['test_corr']:4.2f} | {row['num_trades']:6.0f}")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nSharpe Ratio:")
print(f"  Mean:   {results_df['sharpe'].mean():6.2f}")
print(f"  Median: {results_df['sharpe'].median():6.2f}")
print(f"  Std:    {results_df['sharpe'].std():6.2f}")
print(f"  Min:    {results_df['sharpe'].min():6.2f}")
print(f"  Max:    {results_df['sharpe'].max():6.2f}")

print(f"\nTest Correlation:")
print(f"  Mean:   {results_df['test_corr'].mean():6.3f}")
print(f"  Median: {results_df['test_corr'].median():6.3f}")
print(f"  Positive: {(results_df['test_corr'] > 0).sum()}/{len(results_df)}")

print(f"\nWin Rate:")
print(f"  Mean:   {results_df['win_rate'].mean():6.1%}")
print(f"  Median: {results_df['win_rate'].median():6.1%}")
print(f"  >50%:   {(results_df['win_rate'] > 0.5).sum()}/{len(results_df)}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

print(f"\nFiles created:")
print(f"  - models/sp500/{{ticker}}_xgb.pkl (50 model files)")
print(f"  - results/sp500/{{ticker}}_metrics.json (50 metrics files)")
print(f"  - results/sp500/training_summary.csv (summary)")

print(f"\nNext steps:")
print(f"  1. Review individual model performance")
print(f"  2. Create portfolio manager to combine signals")
print(f"  3. Backtest portfolio strategy")
print(f"  4. Optimize position sizing")

print("\n" + "="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
