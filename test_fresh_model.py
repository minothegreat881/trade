"""
TEST MODEL - FRESH DATA (4 roky až do DNES)
Všetky features vrátane VIX a BTC
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
from metrics import generate_metrics_report

print("=" * 70)
print("TEST MODEL - FRESH DATA WITH ALL FEATURES")
print("=" * 70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Period: 4 years back to TODAY")
print("=" * 70)

# ===== 1. DOWNLOAD FRESH DATA =====
print("\n[STEP 1/7] DOWNLOADING FRESH DATA")
print("=" * 70)

# Calculate dates
end_date = datetime.now()
start_date = end_date - timedelta(days=4*365)  # 4 years back

print(f"Start date: {start_date.strftime('%Y-%m-%d')}")
print(f"End date:   {end_date.strftime('%Y-%m-%d')}")
print(f"Total period: {(end_date - start_date).days} days")

# Download SPY
print("\nDownloading SPY price data...")
spy = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=False)

# Flatten MultiIndex columns if needed
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

print(f"  Downloaded: {len(spy)} days")
print(f"  Period: {spy.index[0].date()} to {spy.index[-1].date()}")

# Download sentiment data
print("\nDownloading sentiment data (Fear&Greed, VIX, BTC)...")
collector = SentimentCollector()
sentiment_data = collector.collect_all_sentiment(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)
print(f"  Downloaded: {len(sentiment_data)} days")
print(f"  Period: {sentiment_data.index[0].date()} to {sentiment_data.index[-1].date()}")

# ===== 2. MERGE DATA =====
print("\n[STEP 2/7] MERGING PRICE + SENTIMENT DATA")
print("=" * 70)

# Normalize indices
spy.index = pd.to_datetime(spy.index).normalize()
sentiment_data.index = pd.to_datetime(sentiment_data.index).normalize()

# Merge
combined = spy[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
combined = combined.join(sentiment_data, how='inner')

print(f"Merged data: {len(combined)} days")
print(f"Period: {combined.index[0].date()} to {combined.index[-1].date()}")

# ===== 3. FEATURE ENGINEERING =====
print("\n[STEP 3/7] CREATING FEATURES")
print("=" * 70)

# Technical features
print("Creating technical features...")
fe = FeatureEngineer()
data_with_technical = fe.create_basic_features(combined)
tech_features = [col for col in data_with_technical.columns
                 if col not in ['Open', 'High', 'Low', 'Close', 'Volume',
                                'Dividends', 'Stock Splits', 'Capital Gains']]
tech_features = [col for col in tech_features
                 if not any(x in col.lower() for x in ['fear', 'greed', 'vix', 'btc', 'sentiment'])]
print(f"  Created: {len(tech_features)} technical features")

# Sentiment features
print("\nCreating sentiment features...")
sentiment_eng = SentimentFeatureEngineer()
sentiment_features_df = sentiment_eng.create_all_sentiment_features(combined)
print(f"  Created: {len(sentiment_features_df.columns)} sentiment features")

# Merge all features
print("\nMerging all features...")
data = data_with_technical.copy()
for col in sentiment_features_df.columns:
    data[col] = sentiment_features_df[col]

# Create target
data['target'] = data['Close'].pct_change(5).shift(-5)
data = data.dropna()

print(f"\nFinal dataset: {len(data)} days")
print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")

# ===== 4. SHOW FEATURES =====
print("\n[STEP 4/7] FEATURE VERIFICATION")
print("=" * 70)

# Separate feature types
all_features = [col for col in data.columns
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target',
                               'Dividends', 'Stock Splits', 'Capital Gains',
                               'fear_greed_classification', 'fear_greed_value',
                               'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']]

technical = [f for f in all_features
             if not any(x in f.lower() for x in ['fear', 'greed', 'vix', 'btc', 'sentiment'])]
sentiment = [f for f in all_features if f not in technical]

print(f"\nTOTAL FEATURES: {len(all_features)}")
print(f"  Technical: {len(technical)}")
print(f"  Sentiment: {len(sentiment)}")

print(f"\nTECHNICAL FEATURES ({len(technical)}):")
for i, f in enumerate(technical, 1):
    print(f"  {i:2d}. {f}")

print(f"\nSENTIMENT FEATURES ({len(sentiment)}):")
for i, f in enumerate(sentiment, 1):
    print(f"  {i:2d}. {f}")

# Check for VIX and BTC
vix_features = [f for f in sentiment if 'vix' in f.lower()]
btc_features = [f for f in sentiment if 'btc' in f.lower()]
fg_features = [f for f in sentiment if 'fear' in f.lower() or 'greed' in f.lower()]

print(f"\nFEATURE BREAKDOWN:")
print(f"  Fear & Greed: {len(fg_features)} features")
print(f"  VIX: {len(vix_features)} features")
print(f"  BTC: {len(btc_features)} features")
print(f"  Other: {len(sentiment) - len(fg_features) - len(vix_features) - len(btc_features)} features")

# ===== 5. SHOW LATEST DATA =====
print("\n[STEP 5/7] LATEST DATA VERIFICATION (Last 5 days)")
print("=" * 70)

latest = data.tail(5)
print(f"\nLatest date: {latest.index[-1].date()}")
print(f"Days old: {(datetime.now().date() - latest.index[-1].date()).days} days")

print("\nPrice data (last 5 days):")
print(latest[['Close']].to_string())

print("\nFear & Greed (last 5 days):")
if 'fear_greed_value' in data.columns:
    print(latest[['fear_greed_value', 'fear_greed_ma10']].to_string())
else:
    fg_cols = [c for c in sentiment if 'fear_greed' in c][:2]
    if fg_cols:
        print(latest[fg_cols].to_string())

print("\nVIX (last 5 days):")
if 'VIX' in data.columns:
    print(latest[['VIX']].to_string())
elif vix_features:
    print(latest[[vix_features[0]]].to_string())
else:
    print("  NO VIX DATA!")

print("\nBTC (last 5 days):")
if 'BTC_Close' in data.columns:
    print(latest[['BTC_Close']].to_string())
elif btc_features:
    print(latest[[btc_features[0]]].to_string())
else:
    print("  NO BTC DATA!")

# Check for missing values
print("\n\nMissing values in features:")
missing = data[all_features].isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print("  WARNING! Missing values found:")
    for col, count in missing.items():
        pct = (count / len(data)) * 100
        print(f"    {col}: {count} ({pct:.1f}%)")
else:
    print("  [OK] No missing values!")

# ===== 6. TRAIN MODEL =====
print("\n[STEP 6/7] TRAINING MODEL")
print("=" * 70)

# Split data
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

# Train model
print(f"\nTraining XGBoost with {len(all_features)} features...")
model = XGBoostModel(params={
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'min_child_weight': 5
})

# Split train into train/val
val_split = int(len(X_train) * 0.8)
X_train_split = X_train.iloc[:val_split]
y_train_split = y_train.iloc[:val_split]
X_val = X_train.iloc[val_split:]
y_val = y_train.iloc[val_split:]

model.train(X_train_split, y_train_split, X_val, y_val)
print("  Model trained!")

# ===== 7. EVALUATE =====
print("\n[STEP 7/7] MODEL EVALUATION")
print("=" * 70)

# Predict
predictions = model.predict(X_test)
print(f"\nPredictions: {len(predictions)}")
print(f"  Mean: {predictions.mean():.6f}")
print(f"  Std:  {predictions.std():.6f}")
print(f"  Min:  {predictions.min():.6f}")
print(f"  Max:  {predictions.max():.6f}")
print(f"  Positive: {(predictions > 0).sum()} / {len(predictions)}")

# Backtest
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

# Print results
print("\n" + "=" * 70)
print("FINAL RESULTS")
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
print("TEST COMPLETE!")
print("=" * 70)
