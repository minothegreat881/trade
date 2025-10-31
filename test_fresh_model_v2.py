"""
TEST MODEL #2 - FRESH DATA (4 roky až do DNES)
LEN Fear & Greed features (BEZ VIX, BEZ BTC)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys

from sentiment_collector import SentimentCollector
from sentiment_features import SentimentFeatureEngineer
from feature_engineering import FeatureEngineer
from xgboost_model import XGBoostModel
from backtester import Backtester

print("=" * 70)
print("TEST MODEL #2 - FRESH DATA WITHOUT VIX/BTC")
print("=" * 70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Features: 11 technical + 6 Fear & Greed ONLY")
print("=" * 70)

# ===== 1. DOWNLOAD FRESH DATA =====
print("\n[STEP 1/6] DOWNLOADING FRESH DATA")
print("=" * 70)

end_date = datetime.now()
start_date = end_date - timedelta(days=4*365)

print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Download SPY
spy = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

print(f"SPY data: {len(spy)} days")

# Download ONLY Fear & Greed (skip VIX and BTC)
print("\nDownloading Fear & Greed ONLY...")
collector = SentimentCollector()
sentiment_data = collector.get_fear_greed_index(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)
print(f"Fear & Greed data: {len(sentiment_data)} days")

# ===== 2. MERGE DATA =====
print("\n[STEP 2/6] MERGING PRICE + SENTIMENT DATA")
print("=" * 70)

spy.index = pd.to_datetime(spy.index).normalize()
sentiment_data.index = pd.to_datetime(sentiment_data.index).normalize()

combined = spy[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
combined = combined.join(sentiment_data, how='inner')

print(f"Merged data: {len(combined)} days")

# ===== 3. FEATURE ENGINEERING =====
print("\n[STEP 3/6] CREATING FEATURES")
print("=" * 70)

# Technical features
fe = FeatureEngineer()
data = fe.create_basic_features(combined)

# Sentiment features - ONLY Fear & Greed
print("Creating Fear & Greed features ONLY...")
sentiment_eng = SentimentFeatureEngineer()

# Create only F&G features
fg_features = sentiment_eng.create_fear_greed_features(combined)
print(f"  Created: {len(fg_features.columns)} Fear & Greed features")

# Add composite sentiment (uses F&G only when VIX/BTC not available)
composite = sentiment_eng.create_composite_sentiment(fg_features)
print(f"  Created: {len(composite.columns)} composite feature")

# Merge
for col in fg_features.columns:
    data[col] = fg_features[col]
for col in composite.columns:
    data[col] = composite[col]

# Create target
data['target'] = data['Close'].pct_change(5).shift(-5)
data = data.dropna()

print(f"\nFinal dataset: {len(data)} days")

# ===== 4. FEATURES =====
print("\n[STEP 4/6] FEATURE VERIFICATION")
print("=" * 70)

all_features = [col for col in data.columns
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target',
                               'Dividends', 'Stock Splits', 'Capital Gains',
                               'fear_greed_value', 'fear_greed_classification']]

technical = [f for f in all_features
             if not any(x in f.lower() for x in ['fear', 'greed', 'sentiment'])]
sentiment = [f for f in all_features if f not in technical]

print(f"\nTOTAL FEATURES: {len(all_features)}")
print(f"  Technical: {len(technical)}")
print(f"  Sentiment: {len(sentiment)} (Fear & Greed ONLY)")

print(f"\nTECHNICAL FEATURES ({len(technical)}):")
for i, f in enumerate(technical, 1):
    print(f"  {i:2d}. {f}")

print(f"\nSENTIMENT FEATURES ({len(sentiment)}):")
for i, f in enumerate(sentiment, 1):
    print(f"  {i:2d}. {f}")

print(f"\nLATEST DATA (last 5 days):")
latest = data.tail(5)
print(f"Latest date: {latest.index[-1].date()}")
print(f"Days old: {(datetime.now().date() - latest.index[-1].date()).days} days")

print("\nFear & Greed (last 5 days):")
if 'fear_greed_value' in combined.columns and 'fear_greed_ma10' in data.columns:
    print(combined[['fear_greed_value']].tail(5).join(data[['fear_greed_ma10']].tail(5)).to_string())

missing = data[all_features].isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print(f"\nWARNING! Missing values:")
    for col, count in missing.items():
        print(f"  {col}: {count}")
else:
    print(f"\nNo missing values!")

# ===== 5. TRAIN MODEL =====
print("\n[STEP 5/6] TRAINING MODEL")
print("=" * 70)

split_idx = int(len(data) * 0.7)
train = data.iloc[:split_idx]
test = data.iloc[split_idx:]

print(f"\nData split:")
print(f"  Train: {len(train)} days ({train.index[0].date()} to {train.index[-1].date()})")
print(f"  Test:  {len(test)} days ({test.index[0].date()} to {test.index[-1].date()})")

X_train = train[all_features]
y_train = train['target']
X_test = test[all_features]
y_test = test['target']

print(f"\nTraining XGBoost with {len(all_features)} features...")
model = XGBoostModel(params={
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'min_child_weight': 5
})

val_split = int(len(X_train) * 0.8)
X_train_split = X_train.iloc[:val_split]
y_train_split = y_train.iloc[:val_split]
X_val = X_train.iloc[val_split:]
y_val = y_train.iloc[val_split:]

model.train(X_train_split, y_train_split, X_val, y_val)
print("  Model trained!")

# ===== 6. EVALUATE =====
print("\n[STEP 6/6] MODEL EVALUATION")
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

test_prices = test['Close']
results = backtester.run_backtest(predictions, y_test, test_prices)

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

print("\n" + "=" * 70)
print("FINAL RESULTS - TEST #2 (NO VIX/BTC)")
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
print("COMPARISON:")
print("=" * 70)
print(f"Test #1 (WITH VIX/BTC, 28 features):    Sharpe 0.61, 172 trades")
print(f"Test #2 (NO VIX/BTC, {len(all_features)} features):      Sharpe {sharpe:.2f}, {results['n_trades']} trades")
print(f"Reference (NO VIX/BTC, 17 features):    Sharpe 1.28, 222 trades")
print("=" * 70)
