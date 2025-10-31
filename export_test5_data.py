"""
Export Test #5 training and test data for optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

from sentiment_collector import SentimentCollector
from sentiment_features import SentimentFeatureEngineer
from feature_engineering import FeatureEngineer
from modules.advanced_features import create_advanced_features

print("="*70)
print("EXPORTING TEST #5 DATA FOR OPTIMIZATION")
print("="*70)

# ===== 1. DOWNLOAD DATA =====
print("\n[1/4] DOWNLOADING DATA")
print("="*70)

end_date = datetime.now()
start_date = end_date - timedelta(days=4*365)

spy = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

# Download sentiment
collector = SentimentCollector()
sentiment_data = collector.collect_all_sentiment(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)

# ===== 2. CREATE FEATURES =====
print("\n[2/4] CREATING FEATURES")
print("="*70)

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

# Create advanced features
data_ohlcv = combined[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
advanced = create_advanced_features(data_ohlcv)

for col in advanced.columns:
    if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] and col not in data.columns:
        data[col] = advanced[col]

# Create target
data['target'] = data['Close'].pct_change(5).shift(-5)
data = data.dropna()

print(f"Total features: {data.shape[1]}")
print(f"Total samples: {len(data)}")
print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")

# ===== 3. SPLIT DATA =====
print("\n[3/4] SPLITTING DATA")
print("="*70)

split_idx = int(len(data) * 0.7)
train = data.iloc[:split_idx]
test = data.iloc[split_idx:]

print(f"Train: {len(train)} samples ({train.index[0].date()} to {train.index[-1].date()})")
print(f"Test:  {len(test)} samples ({test.index[0].date()} to {test.index[-1].date()})")

# ===== 4. SAVE =====
print("\n[4/4] SAVING DATA")
print("="*70)

train.to_csv('data/train_advanced.csv')
test.to_csv('data/test_advanced.csv')

print("  Saved: data/train_advanced.csv")
print("  Saved: data/test_advanced.csv")

# Also save feature names for reference
exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_value', 'fear_greed_classification',
                'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']

feature_cols = [col for col in train.columns if col not in exclude_cols]

with open('data/feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_cols))

print(f"  Saved: data/feature_names.txt ({len(feature_cols)} features)")

print("\n" + "="*70)
print("EXPORT COMPLETE!")
print("="*70)
print("\nYou can now run: python optimize_models.py")
