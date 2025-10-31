"""
Train Complete Model with ALL Features (Technical + Sentiment)
Saves model + metadata for dashboard monitoring
"""

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
from pathlib import Path

from xgboost_model import XGBoostModel
from backtester import Backtester

print("="*60)
print("TRAINING COMPLETE MODEL (Technical + Sentiment)")
print("="*60)

# Load data
print("\n[1/6] Loading data with sentiment...")

# Check which data exists
if Path('data/train_test_sentiment/train_X.csv').exists():
    print("  Loading sentiment features from train_test_sentiment/...")
    train_X = pd.read_csv('data/train_test_sentiment/train_X.csv', index_col=0, parse_dates=True)
    train_y = pd.read_csv('data/train_test_sentiment/train_y.csv', index_col=0, parse_dates=True)
    test_X = pd.read_csv('data/train_test_sentiment/test_X.csv', index_col=0, parse_dates=True)
    test_y = pd.read_csv('data/train_test_sentiment/test_y.csv', index_col=0, parse_dates=True)

    # Load original data for OHLCV columns
    print("  Loading OHLCV data from train_test/...")
    train_ohlcv = pd.read_csv('data/train_test/train_data.csv', index_col=0, parse_dates=True)
    test_ohlcv = pd.read_csv('data/train_test/test_data.csv', index_col=0, parse_dates=True)

    # Convert indices to DatetimeIndex and remove timezone (normalize to match sentiment data)
    train_ohlcv.index = pd.to_datetime(train_ohlcv.index, utc=True).tz_localize(None).normalize()
    test_ohlcv.index = pd.to_datetime(test_ohlcv.index, utc=True).tz_localize(None).normalize()

    # Merge features with OHLCV data
    train = train_ohlcv[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    train = train.join(train_X, how='inner')
    train['target'] = train_y['target']

    test = test_ohlcv[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    test = test.join(test_X, how='inner')
    test['target'] = test_y['target']

elif Path('data/train_test/train_data.csv').exists():
    print("  Loading from train_test/ (no sentiment)...")
    train = pd.read_csv('data/train_test/train_data.csv', index_col=0, parse_dates=True)
    test = pd.read_csv('data/train_test/test_data.csv', index_col=0, parse_dates=True)
else:
    raise FileNotFoundError("Data file not found.")

print(f"  Train: {len(train)} days ({train.index[0].date()} to {train.index[-1].date()})")
print(f"  Test:  {len(test)} days ({test.index[0].date()} to {test.index[-1].date()})")
print(f"  Total: {len(train) + len(test)} days")

# Define features
print("\n[2/6] Preparing features...")
exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains']
feature_cols = [col for col in train.columns if col not in exclude_cols]

# Separate technical vs sentiment
technical_features = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'volatility_20d', 'volatility_60d', 'volume_ratio',
    'price_position', 'sma_20', 'sma_50', 'trend'
]

sentiment_features = [col for col in feature_cols if col not in technical_features]

print(f"\n  Total Features: {len(feature_cols)}")
print(f"  - Technical: {len(technical_features)}")
print(f"  - Sentiment: {len(sentiment_features)}")

# Load best hyperparameters
print("\n[3/6] Loading hyperparameters...")
best_params = {
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'min_child_weight': 5
}

print(f"  Parameters:")
for key, val in best_params.items():
    print(f"    - {key}: {val}")

# Train model
print("\n[4/6] Training XGBoost model...")
model = XGBoostModel(params=best_params)

# Split train into train/val
split_idx = int(len(train) * 0.8)
X_train_split = train[feature_cols].iloc[:split_idx]
y_train_split = train['target'].iloc[:split_idx]
X_val = train[feature_cols].iloc[split_idx:]
y_val = train['target'].iloc[split_idx:]

model.train(X_train_split, y_train_split, X_val, y_val)
print("  [OK] Training complete")

# Get feature importances
feature_importance = model.model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance,
    'type': ['technical' if f in technical_features else 'sentiment'
             for f in feature_cols]
}).sort_values('importance', ascending=False)

print(f"\n  Top 10 Most Important Features:")
for idx, row in importance_df.head(10).iterrows():
    print(f"    {row['feature']:30s} {row['importance']:.4f} ({row['type']})")

# Test performance
print("\n[5/6] Testing model performance...")
predictions = model.predict(test[feature_cols])

backtester = Backtester()
results = backtester.run_backtest(
    predictions,
    test['target'],
    test['Close']
)

# Calculate metrics from backtester results
total_return = (results['final_capital'] / results['initial_capital']) - 1
returns = results['returns']

# Sharpe ratio (annualized)
if len(returns) > 0 and returns.std() > 0:
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
else:
    sharpe = 0

# Max drawdown
equity_curve = results['equity_curve']
rolling_max = equity_curve.expanding().max()
drawdowns = (equity_curve - rolling_max) / rolling_max
max_drawdown = drawdowns.min()

# Win rate
trades_df = results['trades']
if len(trades_df) > 0:
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df)
else:
    win_rate = 0

print(f"\n  Test Results:")
print(f"    Sharpe Ratio:     {sharpe:.2f}")
print(f"    Annual Return:    {total_return*100:.2f}%")
print(f"    Max Drawdown:     {max_drawdown*100:.2f}%")
print(f"    Win Rate:         {win_rate*100:.1f}%")
print(f"    Total Trades:     {results['n_trades']}")

# Save model
print("\n[6/6] Saving model and metadata...")
Path('models').mkdir(exist_ok=True)

# Save model
joblib.dump(model.model, 'models/xgboost_sentiment_model.pkl')
print("  [OK] Model saved: models/xgboost_sentiment_model.pkl")

# Save metadata
metadata = {
    'model_version': '1.0.0',
    'created_at': datetime.now().isoformat(),
    'training_period': {
        'start': train.index[0].isoformat(),
        'end': train.index[-1].isoformat(),
        'days': len(train)
    },
    'test_period': {
        'start': test.index[0].isoformat(),
        'end': test.index[-1].isoformat(),
        'days': len(test)
    },
    'features': {
        'total': len(feature_cols),
        'technical': len(technical_features),
        'sentiment': len(sentiment_features),
        'list': feature_cols,
        'technical_list': technical_features,
        'sentiment_list': sentiment_features
    },
    'hyperparameters': best_params,
    'feature_importances': importance_df.to_dict('records'),
    'test_performance': {
        'sharpe_ratio': float(sharpe),
        'annual_return': float(total_return),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'num_trades': int(results['n_trades'])
    }
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("  [OK] Metadata saved: models/model_metadata.json")

# Save feature importances separately
importance_df.to_csv('models/feature_importances.csv', index=False)
print("  [OK] Feature importances saved: models/feature_importances.csv")

print("\n" + "="*60)
print("[OK] MODEL TRAINING COMPLETE!")
print("="*60)
print(f"Features:  {len(feature_cols)} (Technical: {len(technical_features)}, Sentiment: {len(sentiment_features)})")
print(f"Sharpe:    {sharpe:.2f}")
print(f"Return:    {total_return*100:.2f}%")
print("\nModel is ready for live trading simulator!")
print("="*60)
