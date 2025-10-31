"""
CREATE FULL DATASET 2020-2025
==============================

Vytvorí kompletný dataset:
- Od: 2020-01-01
- Do: 2025-10-31
- Všetky features (118+)
- S target

Uloží ako: data/full_dataset_2020_2025.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

from sentiment_collector import SentimentCollector
from sentiment_features import SentimentFeatureEngineer
from feature_engineering import FeatureEngineer
from modules.advanced_features import create_advanced_features
from modules.top_features import create_top_features

print("="*80)
print("CREATE FULL DATASET 2020-2025")
print("="*80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ================================================================
# 1. STIAHNUTIE DAT
# ================================================================

print("\n[1/7] Stiahnutie SPY data...")

spy = yf.download('SPY', start='2020-01-01', end='2025-11-01', progress=False, auto_adjust=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

spy.index = pd.to_datetime(spy.index).normalize()

print(f"  SPY: {len(spy)} dni")
print(f"  Obdobie: {spy.index.min().strftime('%Y-%m-%d')} -> {spy.index.max().strftime('%Y-%m-%d')}")

# Stiahnutie sentiment
print("\n[2/7] Stiahnutie sentiment data...")

collector = SentimentCollector()
sentiment_data = collector.collect_all_sentiment('2020-01-01', '2025-10-31')
sentiment_data.index = pd.to_datetime(sentiment_data.index).normalize()

print(f"  Sentiment: {len(sentiment_data)} dni")
print(f"  Columns: {', '.join(sentiment_data.columns[:5])}...")


# ================================================================
# 3. MERGE & BASIC FEATURES
# ================================================================

print("\n[3/7] Vytvorenie basic features...")

combined = spy[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
combined = combined.join(sentiment_data, how='inner')

print(f"  Po merge: {len(combined)} dni")

# Create basic features
fe = FeatureEngineer()
data = fe.create_basic_features(combined)

# Add sentiment features
sentiment_eng = SentimentFeatureEngineer()
sentiment_features_df = sentiment_eng.create_all_sentiment_features(combined)

for col in sentiment_features_df.columns:
    data[col] = sentiment_features_df[col]

print(f"  Basic + Sentiment: {data.shape[1]} columns")


# ================================================================
# 4. ADVANCED FEATURES
# ================================================================

print("\n[4/7] Vytvorenie advanced features...")

data_ohlcv = combined[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
advanced = create_advanced_features(data_ohlcv)

# Merge advanced features
for col in advanced.columns:
    if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] and col not in data.columns:
        data[col] = advanced[col]

print(f"  Po advanced: {data.shape[1]} columns")


# ================================================================
# 5. TOP RESEARCH FEATURES
# ================================================================

print("\n[5/7] Vytvorenie research features...")

# Create target FIRST
data['target'] = data['Close'].pct_change(5).shift(-5)

# Create top research features
top_research = create_top_features(combined[['Open', 'High', 'Low', 'Close', 'Volume']])

# Merge research features
for col in top_research.columns:
    if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] and col not in data.columns:
        data[col] = top_research[col]

print(f"  Po research: {data.shape[1]} columns")


# ================================================================
# 6. CLEAN DATA
# ================================================================

print("\n[6/7] Cistenie dat...")

print(f"  Pred dropna: {len(data)} dni")

# Check for NaN
nan_counts = data.isna().sum()
nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)

if len(nan_cols) > 0:
    print(f"\n  Columns s NaN (top 10):")
    for col, count in nan_cols.head(10).items():
        pct = (count / len(data)) * 100
        print(f"    {col[:40]:40s} {count:5d} ({pct:5.1f}%)")

    # Drop columns with ALL NaN
    all_nan_cols = nan_counts[nan_counts == len(data)]
    if len(all_nan_cols) > 0:
        print(f"\n  Mazem {len(all_nan_cols)} columns s ALL NaN")
        data = data.drop(columns=all_nan_cols.index)

# Drop rows with NaN
data = data.dropna()
print(f"  Po dropna:  {len(data)} dni")
print(f"  Obdobie: {data.index[0].strftime('%Y-%m-%d')} -> {data.index[-1].strftime('%Y-%m-%d')}")


# ================================================================
# 7. SAVE DATASET
# ================================================================

print("\n[7/7] Ukladanie datasetu...")

# Define exclude columns
exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_value', 'fear_greed_classification',
                'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']

all_features = [col for col in data.columns if col not in exclude_cols]

print(f"\n  Total features: {len(all_features)}")
print(f"  Rows: {len(data)}")
print(f"  Columns (total): {data.shape[1]}")

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

print(f"\n  Feature Breakdown:")
print(f"    Basic (baseline):         {len(basic_features)}")
print(f"    Sentiment:                {len(sentiment_features)}")
print(f"    Momentum:                 {len(momentum_features)}")
print(f"    Volatility:               {len(volatility_features)}")
print(f"    Trend:                    {len(trend_features)}")
print(f"    Volume:                   {len(volume_features)}")
print(f"    Pattern:                  {len(pattern_features)}")
print(f"    Multi-timeframe:          {len(timeframe_features)}")
print(f"    Microstructure:           {len(micro_features)}")
print(f"    Research-backed:          {len(research_features)}")
print(f"    =" * 40)
print(f"    TOTAL:                    {len(all_features)}")

# Save full dataset
data.to_csv('data/full_dataset_2020_2025.csv')

print(f"\n  [OK] Ulozene:")
print(f"    - data/full_dataset_2020_2025.csv")
print(f"      Rows: {len(data)}")
print(f"      Columns: {data.shape[1]}")
print(f"      Features: {len(all_features)}")
print(f"      Obdobie: {data.index[0].strftime('%Y-%m-%d')} -> {data.index[-1].strftime('%Y-%m-%d')}")

# Save feature list
import json
feature_metadata = {
    'created_at': datetime.now().isoformat(),
    'data_period': {
        'start': data.index[0].strftime('%Y-%m-%d'),
        'end': data.index[-1].strftime('%Y-%m-%d'),
        'total_days': len(data)
    },
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
        'research': len(research_features)
    },
    'feature_list': all_features,
    'exclude_columns': exclude_cols
}

with open('data/full_dataset_2020_2025_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

print(f"    - data/full_dataset_2020_2025_metadata.json")

print("\n" + "="*80)
print("HOTOVO!")
print("="*80)
print(f"\nDATASET:")
print(f"  File: data/full_dataset_2020_2025.csv")
print(f"  Obdobie: 2020-01-01 -> 2025-10-31")
print(f"  Dni: {len(data)}")
print(f"  Features: {len(all_features)}")
print(f"  Target: forward 5-day return")
print("="*80)
