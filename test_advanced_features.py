"""
TEST #5 - WITH ADVANCED FEATURES
Train and test model with 80+ advanced technical indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys

from sentiment_collector import SentimentCollector
from sentiment_features import SentimentFeatureEngineer
from feature_engineering import FeatureEngineer
from modules.advanced_features import create_advanced_features
from xgboost_model import XGBoostModel
from backtester import Backtester
import joblib
import json

print("=" * 70)
print("TEST #5 - WITH ADVANCED FEATURES")
print("=" * 70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Strategy: Add 80+ advanced technical indicators")
print("=" * 70)

# ===== 1. DOWNLOAD DATA =====
print("\n[STEP 1/7] DOWNLOADING FRESH DATA")
print("=" * 70)

end_date = datetime.now()
start_date = end_date - timedelta(days=4*365)

spy = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

print(f"SPY data: {len(spy)} days")

# Download sentiment
collector = SentimentCollector()
sentiment_data = collector.collect_all_sentiment(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)

# ===== 2. MERGE & BASIC FEATURES =====
print("\n[STEP 2/7] CREATING BASIC FEATURES")
print("=" * 70)

spy.index = pd.to_datetime(spy.index).normalize()
sentiment_data.index = pd.to_datetime(sentiment_data.index).normalize()

combined = spy[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
combined = combined.join(sentiment_data, how='inner')

# Create basic features
fe = FeatureEngineer()
data = fe.create_basic_features(combined)

# Add sentiment features
sentiment_eng = SentimentFeatureEngineer()
sentiment_features_df = sentiment_eng.create_all_sentiment_features(combined)

for col in sentiment_features_df.columns:
    data[col] = sentiment_features_df[col]

print(f"Basic features: {data.shape[1]}")

# ===== 3. ADD ADVANCED FEATURES =====
print("\n[STEP 3/7] CREATING ADVANCED FEATURES")
print("=" * 70)

# Create advanced features (needs OHLCV)
data_ohlcv = combined[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
advanced = create_advanced_features(data_ohlcv)

# Merge advanced features with existing data
for col in advanced.columns:
    if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] and col not in data.columns:
        data[col] = advanced[col]

print(f"Total features after advanced: {data.shape[1]}")

# Create target
data['target'] = data['Close'].pct_change(5).shift(-5)
data = data.dropna()

print(f"Final dataset: {len(data)} days")
print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")

# ===== 4. SELECT FEATURES =====
print("\n[STEP 4/7] SELECTING FEATURES")
print("=" * 70)

exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_value', 'fear_greed_classification',
                'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']

all_features = [col for col in data.columns if col not in exclude_cols]

# Categorize features
basic_features = [f for f in all_features if any(x in f for x in ['return_', 'volatility_', 'volume_ratio', 'price_position', 'sma_', 'trend'])]
sentiment_features = [f for f in all_features if any(x in f for x in ['fear_greed', 'vix_', 'btc_', 'composite_sentiment'])]
momentum_features = [f for f in all_features if any(x in f for x in ['rsi', 'macd', 'stoch', 'willr', 'roc', 'momentum'])]
volatility_features = [f for f in all_features if any(x in f for x in ['atr', 'bb_', 'hist_vol', 'vol_ratio', 'parkinson'])]
trend_features = [f for f in all_features if any(x in f for x in ['adx', 'aroon', 'di_', 'linreg', 'trend_strength', 'trend_alignment'])]
volume_features = [f for f in all_features if any(x in f for x in ['obv', 'mfi', 'ad', 'vwap', 'volume_'])]
pattern_features = [f for f in all_features if any(x in f for x in ['gap', 'body', 'shadow', 'doji', 'engulfing', 'resistance', 'support', 'higher_high', 'lower_low'])]
timeframe_features = [f for f in all_features if any(x in f for x in ['weekly_', 'monthly_', 'position_in_'])]
micro_features = [f for f in all_features if any(x in f for x in ['spread', 'price_impact', 'liquidity', 'intraday', 'overnight'])]

print(f"\nFeature Breakdown:")
print(f"  Basic (baseline):     {len(basic_features)}")
print(f"  Sentiment:            {len(sentiment_features)}")
print(f"  Momentum:             {len(momentum_features)}")
print(f"  Volatility:           {len(volatility_features)}")
print(f"  Trend:                {len(trend_features)}")
print(f"  Volume:               {len(volume_features)}")
print(f"  Pattern:              {len(pattern_features)}")
print(f"  Multi-timeframe:      {len(timeframe_features)}")
print(f"  Microstructure:       {len(micro_features)}")
print(f"  TOTAL:                {len(all_features)}")

# ===== 5. SPLIT DATA =====
print("\n[STEP 5/7] SPLITTING DATA")
print("=" * 70)

split_idx = int(len(data) * 0.7)
train = data.iloc[:split_idx]
test = data.iloc[split_idx:]

print(f"Train: {len(train)} days ({train.index[0].date()} to {train.index[-1].date()})")
print(f"Test:  {len(test)} days ({test.index[0].date()} to {test.index[-1].date()})")

X_train = train[all_features]
y_train = train['target']
X_test = test[all_features]
y_test = test['target']

# ===== 6. TRAIN MODEL =====
print("\n[STEP 6/7] TRAINING MODEL WITH ADVANCED FEATURES")
print("=" * 70)

model = XGBoostModel(params={
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8
})

val_split = int(len(X_train) * 0.8)
model.train(
    X_train.iloc[:val_split],
    y_train.iloc[:val_split],
    X_train.iloc[val_split:],
    y_train.iloc[val_split:]
)

# Get feature importances
feature_importance = model.model.feature_importances_
importance_df = pd.DataFrame({
    'feature': all_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nTop 20 Most Important Features:")
for idx, row in importance_df.head(20).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

# ===== 7. EVALUATE =====
print("\n[STEP 7/7] MODEL EVALUATION")
print("=" * 70)

predictions = model.predict(X_test)

print(f"\nPredictions: {len(predictions)}")
print(f"  Mean: {predictions.mean():.6f}")
print(f"  Positive: {(predictions > 0).sum()} / {len(predictions)}")

print("\nRunning backtest...")
backtester = Backtester(
    initial_capital=10000,
    transaction_cost=0.001,
    holding_period=5,
    position_size_pct=0.5,
    prediction_threshold=0.001
)

results = backtester.run_backtest(predictions, y_test, test['Close'])

# Calculate metrics
total_return = (results['final_capital'] / results['initial_capital']) - 1
returns = results['returns']

if len(returns) > 0 and returns.std() > 0:
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
else:
    sharpe = 0

equity_curve = results['equity_curve']
rolling_max = equity_curve.expanding().max()
drawdowns = (equity_curve - rolling_max) / rolling_max
max_drawdown = drawdowns.min()

trades_df = results['trades']
if len(trades_df) > 0:
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df)
else:
    win_rate = 0

# ===== RESULTS =====
print("\n" + "=" * 70)
print("FINAL RESULTS - WITH ADVANCED FEATURES")
print("=" * 70)

print(f"\nRETURNS:")
print(f"  Total Return:           {total_return*100:.2f}%")
print(f"  Annual Return (CAGR):   {(((1 + total_return) ** (252/len(test))) - 1)*100:.2f}%")

print(f"\nRISK:")
print(f"  Maximum Drawdown:       {max_drawdown*100:.2f}%")
print(f"  Daily Volatility:       {returns.std()*100:.2f}%")

print(f"\nRISK-ADJUSTED:")
print(f"  Sharpe Ratio:           {sharpe:.2f}")

print(f"\nTRADING:")
print(f"  Total Trades:           {results['n_trades']}")
print(f"  Win Rate:               {win_rate*100:.1f}%")

print("\n" + "=" * 70)
print("COMPARISON WITH ALL TESTS")
print("=" * 70)
print(f"Test #1 (WITH VIX/BTC, 28 features):         Sharpe 0.61, 172 trades")
print(f"Test #2 (NO VIX/BTC, 17 features):           Sharpe 0.55, 101 trades")
print(f"Test #3 (REFERENCE, 17 features):            Sharpe 0.78, 1 trade")
print(f"Test #4 (WITH REGIME, 28 features):          Sharpe 0.63, 152 trades")
print(f"Test #5 (ADVANCED, {len(all_features)} features):    Sharpe {sharpe:.2f}, {results['n_trades']} trades")

print("\n" + "=" * 70)
print("IMPROVEMENT ANALYSIS:")
print("=" * 70)

baseline_sharpe = 0.63
improvement = ((sharpe / baseline_sharpe) - 1) * 100

print(f"Baseline (Test #4 with regime):  Sharpe 0.63")
print(f"Current (Test #5 advanced):       Sharpe {sharpe:.2f}")
print(f"Improvement:                      {improvement:+.1f}%")

if sharpe > baseline_sharpe:
    print(f"\n  Advanced features IMPROVED performance!")
else:
    print(f"\n  Advanced features did not improve (possible overfitting)")

# Save model
print("\n[SAVING] Saving model with advanced features...")
joblib.dump(model, 'models/xgboost_advanced_features.pkl')

# Save metadata
metadata = {
    'model_version': '5.0.0-advanced-features',
    'created_at': datetime.now().isoformat(),
    'features': {
        'total': len(all_features),
        'basic': len(basic_features),
        'sentiment': len(sentiment_features),
        'momentum': len(momentum_features),
        'volatility': len(volatility_features),
        'trend': len(trend_features),
        'volume': len(volume_features),
        'pattern': len(pattern_features),
        'timeframe': len(timeframe_features),
        'microstructure': len(micro_features),
        'list': all_features
    },
    'test_performance': {
        'sharpe_ratio': float(sharpe),
        'annual_return': float((((1 + total_return) ** (252/len(test))) - 1)),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'num_trades': int(results['n_trades'])
    },
    'top_features': importance_df.head(30).to_dict('records')
}

with open('models/model_metadata_advanced.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("  Model and metadata saved!")
print("=" * 70)
