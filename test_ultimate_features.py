"""
TEST #6 - ULTIMATE MODEL
Combines VŠETKY features:
- 108 advanced features from Test #5
- 10 top research-backed features (RSI, MACD, ATR, Bollinger, ADX, OBV, Stochastic)
= 118 TOTAL FEATURES

Expected: Sharpe > 1.3 (beat Test #5's 1.28)
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
from modules.top_features import create_top_features
from xgboost_model import XGBoostModel
from backtester import Backtester
import joblib
import json

print("=" * 70)
print("TEST #6 - ULTIMATE MODEL WITH ALL FEATURES")
print("=" * 70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Strategy: Combine 108 advanced + 10 research-backed features")
print("=" * 70)

# ===== 1. DOWNLOAD DATA =====
print("\n[STEP 1/8] DOWNLOADING FRESH DATA")
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
print("\n[STEP 2/8] CREATING BASIC FEATURES")
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

print(f"Basic + Sentiment features: {data.shape[1]}")

# ===== 3. ADD ADVANCED FEATURES =====
print("\n[STEP 3/8] CREATING ADVANCED FEATURES")
print("=" * 70)

# Create advanced features (needs OHLCV)
data_ohlcv = combined[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
advanced = create_advanced_features(data_ohlcv)

# Merge advanced features with existing data
for col in advanced.columns:
    if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] and col not in data.columns:
        data[col] = advanced[col]

print(f"Total features after advanced: {data.shape[1]}")

# ===== 4. ADD TOP RESEARCH FEATURES =====
print("\n[STEP 4/8] CREATING RESEARCH-BACKED FEATURES")
print("=" * 70)

# Create target BEFORE adding research features to avoid index issues
data['target'] = data['Close'].pct_change(5).shift(-5)

# Now create top research features on the ORIGINAL combined data (before any dropna)
top_research = create_top_features(combined[['Open', 'High', 'Low', 'Close', 'Volume']])

# Merge research features with existing data (they should have same index)
for col in top_research.columns:
    if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] and col not in data.columns:
        data[col] = top_research[col]

print(f"Total features after research: {data.shape[1]}")

# Check for NaN values before dropping
print(f"\nBefore dropna: {len(data)} days")
print("\nChecking for NaN values by column...")
nan_counts = data.isna().sum()
nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)

if len(nan_cols) > 0:
    print(f"\nColumns with NaN values (showing top 20):")
    for col, count in nan_cols.head(20).items():
        pct = (count / len(data)) * 100
        print(f"  {col:40s} {count:5d} NaN ({pct:5.1f}%)")

    # Check if any column has ALL NaN values
    all_nan_cols = nan_counts[nan_counts == len(data)]
    if len(all_nan_cols) > 0:
        print(f"\n[ERROR] {len(all_nan_cols)} columns have ALL NaN values:")
        for col in all_nan_cols.index[:10]:
            print(f"  - {col}")
        print("\nThis is likely a bug in feature creation. Dropping these columns...")
        data = data.drop(columns=all_nan_cols.index)

# Now drop rows with any NaN
data = data.dropna()
print(f"After dropna:  {len(data)} days")

if len(data) == 0:
    print("\n[ERROR] All rows were dropped due to NaN values!")
    print("This usually happens when rolling windows are too long for the dataset.")
    import sys
    sys.exit(1)

print(f"Final dataset: {len(data)} days")
print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")

# ===== 5. SELECT FEATURES =====
print("\n[STEP 5/8] SELECTING FEATURES")
print("=" * 70)

exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_value', 'fear_greed_classification',
                'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']

all_features = [col for col in data.columns if col not in exclude_cols]

# Categorize features
basic_features = [f for f in all_features if any(x in f for x in ['return_', 'volatility_', 'volume_ratio', 'price_position', 'sma_', 'trend']) and 'research' not in f]
sentiment_features = [f for f in all_features if any(x in f for x in ['fear_greed', 'vix_', 'btc_', 'composite_sentiment'])]
momentum_features = [f for f in all_features if any(x in f for x in ['rsi', 'macd', 'stoch', 'willr', 'roc', 'momentum']) and 'research' not in f]
volatility_features = [f for f in all_features if any(x in f for x in ['atr', 'bb_', 'hist_vol', 'vol_ratio', 'parkinson']) and 'research' not in f]
trend_features = [f for f in all_features if any(x in f for x in ['adx', 'aroon', 'di_', 'linreg', 'trend_strength', 'trend_alignment']) and 'research' not in f]
volume_features = [f for f in all_features if any(x in f for x in ['obv', 'mfi', 'ad', 'vwap', 'volume_']) and 'research' not in f]
pattern_features = [f for f in all_features if any(x in f for x in ['gap', 'body', 'shadow', 'doji', 'engulfing', 'resistance', 'support', 'higher_high', 'lower_low'])]
timeframe_features = [f for f in all_features if any(x in f for x in ['weekly_', 'monthly_', 'position_in_'])]
micro_features = [f for f in all_features if any(x in f for x in ['spread', 'price_impact', 'liquidity', 'intraday', 'overnight'])]
research_features = [f for f in all_features if 'research' in f]

print(f"\nFeature Breakdown:")
print(f"  Basic (baseline):         {len(basic_features)}")
print(f"  Sentiment:                {len(sentiment_features)}")
print(f"  Momentum:                 {len(momentum_features)}")
print(f"  Volatility:               {len(volatility_features)}")
print(f"  Trend:                    {len(trend_features)}")
print(f"  Volume:                   {len(volume_features)}")
print(f"  Pattern:                  {len(pattern_features)}")
print(f"  Multi-timeframe:          {len(timeframe_features)}")
print(f"  Microstructure:           {len(micro_features)}")
print(f"  Research-backed (NEW):    {len(research_features)}")
print(f"  TOTAL:                    {len(all_features)}")

# ===== 6. SPLIT DATA =====
print("\n[STEP 6/8] SPLITTING DATA")
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

# ===== 7. TRAIN MODEL =====
print("\n[STEP 7/8] TRAINING ULTIMATE MODEL")
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
    is_research = '[RESEARCH]' if 'research' in row['feature'] else ''
    print(f"  {row['feature']:40s} {row['importance']:.4f} {is_research}")

# Check research features performance
research_importances = importance_df[importance_df['feature'].str.contains('research')]
if len(research_importances) > 0:
    print(f"\nResearch Features Performance:")
    print(f"  Total research features:     {len(research_importances)}")
    print(f"  Avg importance:              {research_importances['importance'].mean():.6f}")
    print(f"  Max importance:              {research_importances['importance'].max():.6f}")
    print(f"  In top 20:                   {len(research_importances.head(20))}")

# ===== 8. EVALUATE =====
print("\n[STEP 8/8] MODEL EVALUATION")
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
print("FINAL RESULTS - ULTIMATE MODEL")
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
print(f"Test #5 (ADVANCED, 108 features):            Sharpe 1.28, 127 trades")
print(f"Test #6 (ULTIMATE, {len(all_features)} features):    Sharpe {sharpe:.2f}, {results['n_trades']} trades")

print("\n" + "=" * 70)
print("IMPROVEMENT ANALYSIS:")
print("=" * 70)

test5_sharpe = 1.28
improvement = ((sharpe / test5_sharpe) - 1) * 100

print(f"Test #5 (Advanced):       Sharpe 1.28")
print(f"Test #6 (Ultimate):       Sharpe {sharpe:.2f}")
print(f"Improvement:              {improvement:+.1f}%")

if sharpe > test5_sharpe:
    print(f"\n  SUCCESS! Research features IMPROVED performance!")
    print(f"  Ultimate model is the BEST performing model!")
elif sharpe > 1.20:
    print(f"\n  GOOD! Model performance is excellent (Sharpe > 1.2)")
    print(f"  Research features maintained strong performance")
else:
    print(f"\n  Research features did not improve performance")
    print(f"  Test #5 remains the best model")

# Save model
print("\n[SAVING] Saving ultimate model...")
joblib.dump(model, 'models/xgboost_ultimate_features.pkl')

# Save metadata
metadata = {
    'model_version': '6.0.0-ultimate-features',
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
        'research': len(research_features),
        'list': all_features
    },
    'test_performance': {
        'sharpe_ratio': float(sharpe),
        'annual_return': float((((1 + total_return) ** (252/len(test))) - 1)),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'num_trades': int(results['n_trades'])
    },
    'top_features': importance_df.head(30).to_dict('records'),
    'research_features': {
        'count': len(research_features),
        'avg_importance': float(research_importances['importance'].mean()) if len(research_importances) > 0 else 0,
        'top_research': research_importances.head(10).to_dict('records') if len(research_importances) > 0 else []
    }
}

with open('models/model_metadata_ultimate.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("  Model and metadata saved!")

# Save comparison
print("\n[SAVING] Saving test comparison...")
comparison = {
    'test_date': datetime.now().isoformat(),
    'test_period': {
        'start': test.index[0].strftime('%Y-%m-%d'),
        'end': test.index[-1].strftime('%Y-%m-%d'),
        'days': len(test)
    },
    'models': {
        'test1': {'sharpe': 0.61, 'features': 28, 'trades': 172, 'description': 'WITH VIX/BTC'},
        'test2': {'sharpe': 0.55, 'features': 17, 'trades': 101, 'description': 'NO VIX/BTC'},
        'test3': {'sharpe': 0.78, 'features': 17, 'trades': 1, 'description': 'REFERENCE'},
        'test4': {'sharpe': 0.63, 'features': 28, 'trades': 152, 'description': 'WITH REGIME'},
        'test5': {'sharpe': 1.28, 'features': 108, 'trades': 127, 'description': 'ADVANCED'},
        'test6': {'sharpe': float(sharpe), 'features': len(all_features), 'trades': int(results['n_trades']), 'description': 'ULTIMATE'}
    },
    'best_model': 'test6' if sharpe > 1.28 else 'test5',
    'recommendation': 'Use Test #6 (Ultimate)' if sharpe > 1.28 else 'Use Test #5 (Advanced)'
}

with open('models/test_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print("  Comparison saved to: models/test_comparison.json")

print("\n" + "=" * 70)
print("TEST #6 COMPLETE!")
print("=" * 70)
