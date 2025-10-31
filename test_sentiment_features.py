"""
KROK 1: Verify & Fix Sentiment Features
Test if sentiment feature engineering works in LIVE mode
"""

# Set UTF-8 encoding for console output
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*60)
print("TESTING SENTIMENT FEATURE CREATION")
print("="*60)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test 1: Check if sentiment_features module exists
print("\n1. Checking for sentiment feature module...")
try:
    from sentiment_features import SentimentFeatureEngineer
    print("   [OK] sentiment_features.py found")
    use_class = True
except ImportError:
    print("   [WARN] sentiment_features.py not found as module")
    try:
        # Try importing from modules
        sys.path.insert(0, 'modules')
        from sentiment_features import create_sentiment_features
        print("   [OK] modules/sentiment_features.py found")
        use_class = False
    except ImportError:
        print("   [ERROR] No sentiment feature module found!")
        print("\n   Checking trading_engine.py for inline implementation...")

        # Check what trading_engine uses
        try:
            from trading_engine import TradingEngine
            print("   [OK] trading_engine.py exists")
            print("   -> Need to inspect how it creates sentiment features")
        except ImportError:
            print("   [ERROR] trading_engine.py not found!")

        sys.exit(1)

# Test 2: Create sample data (simulating live data)
print("\n2. Creating sample live data...")
dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
sample_data = pd.DataFrame({
    'Close': np.random.randn(100).cumsum() + 100,
    'Open': np.random.randn(100).cumsum() + 100,
    'High': np.random.randn(100).cumsum() + 102,
    'Low': np.random.randn(100).cumsum() + 98,
    'Volume': np.random.randint(1000000, 10000000, 100),
    'fear_greed_value': np.random.randint(20, 80, 100),
    'VIX': np.random.randn(100) * 5 + 15,
}, index=dates)

print(f"   Input data:")
print(f"   - Rows: {len(sample_data)}")
print(f"   - Columns: {sample_data.columns.tolist()}")
print(f"   - Has fear_greed_value: {'fear_greed_value' in sample_data.columns}")
print(f"   - Has VIX: {'VIX' in sample_data.columns}")

# Test 3: Apply sentiment feature engineering
print(f"\n3. Applying sentiment feature engineering...")
try:
    if use_class:
        engineer = SentimentFeatureEngineer()
        enriched_data = engineer.create_all_sentiment_features(sample_data)
    else:
        enriched_data = create_sentiment_features(sample_data)

    # Check output
    sentiment_cols = [col for col in enriched_data.columns
                      if any(x in col.lower() for x in ['fear', 'vix', 'sentiment', 'greed'])]

    print(f"   [OK] Success!")
    print(f"\n   Output:")
    print(f"   - Total columns: {len(enriched_data.columns)}")
    print(f"   - Sentiment columns: {len(sentiment_cols)}")

    print(f"\n   All sentiment features created:")
    for i, col in enumerate(sorted(sentiment_cols), 1):
        print(f"      {i:2d}. {col}")

    # Check for expected features (based on model metadata)
    expected = [
        'fear_greed_value', 'fear_greed_ma5', 'fear_greed_ma10',
        'fear_greed_extreme_fear', 'fear_greed_extreme_greed',
        'fear_greed_change_5d', 'composite_sentiment',
        'VIX', 'VIX_ma5', 'VIX_ma10', 'VIX_momentum',
        'BTC_return_1d', 'BTC_return_5d', 'BTC_return_10d',
        'sentiment_regime'
    ]

    present = [col for col in expected if col in enriched_data.columns]
    missing = [col for col in expected if col not in enriched_data.columns]

    print(f"\n4. Feature validation:")
    print(f"   Expected features: {len(expected)}")
    print(f"   Present: {len(present)}")
    print(f"   Missing: {len(missing)}")

    if missing:
        print(f"\n   [WARN] Missing features:")
        for col in missing:
            print(f"      - {col}")
    else:
        print(f"\n   [OK] ALL EXPECTED FEATURES PRESENT!")

    # Check for NaN values
    nan_cols = enriched_data[sentiment_cols].isnull().sum()
    nan_cols = nan_cols[nan_cols > 0]

    if len(nan_cols) > 0:
        print(f"\n   [WARN] Some features have NaN values:")
        for col, count in nan_cols.items():
            print(f"      - {col}: {count} NaN values")
    else:
        print(f"   [OK] No NaN values in sentiment features")

    # Final verdict
    print("\n" + "="*60)
    print("VERDICT:")
    print("="*60)

    min_required = 15  # At least 15 sentiment features needed

    if len(sentiment_cols) >= min_required and len(missing) <= 3:
        print(f"[PASS] Sentiment feature creation works!")
        print(f"   Generated {len(sentiment_cols)} sentiment features")
        print(f"   Live simulator should work with copied model!")
        print("\n-> NEXT STEP: Copy existing model to models/ directory")
    else:
        print(f"[FAIL] Insufficient sentiment features!")
        print(f"   Generated only {len(sentiment_cols)} features")
        print(f"   Need at least {min_required} features")
        print(f"   Missing critical features: {len(missing)}")
        print("\n-> NEXT STEP: Fix sentiment feature creation first!")

    print("="*60)

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

    print("\n" + "="*60)
    print("VERDICT:")
    print("="*60)
    print(f"[CRITICAL FAILURE] Sentiment features don't work!")
    print(f"   Error: {str(e)}")
    print("\n-> NEXT STEP: Must fix sentiment feature creation!")
    print("="*60)

    sys.exit(1)
