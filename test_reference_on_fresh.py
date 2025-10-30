"""
TEST #3 - Referencny model (GitHub) na FRESH datach
Otestujeme ten ISTY model ktory mal Sharpe 1.28
na nasom volatilnom obdobi (2024-09 -> 2025-10)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import sys
import os

print("=" * 70)
print("TEST #3 - REFERENCE MODEL ON FRESH DATA")
print("=" * 70)
print(f"Loading model from: ../trade_reference/")
print(f"Testing on: 2024-09-10 to 2025-10-23 (FRESH)")
print("=" * 70)

# ===== 1. LOAD REFERENCE MODEL =====
print("\n[STEP 1/5] LOADING REFERENCE MODEL")
print("=" * 70)

# Check if reference repo exists
if not os.path.exists('../trade_reference'):
    print("[ERROR] trade_reference not found!")
    print("Please ensure you cloned the GitHub repo.")
    sys.exit(1)

# Load reference data to get features
print("Loading reference training data to extract features...")
ref_train = pd.read_csv('../trade_reference/data/train_test_sentiment/train_X.csv', index_col=0, parse_dates=True)
reference_features = list(ref_train.columns)

print(f"Reference model features: {len(reference_features)}")
print("Features:")
for i, f in enumerate(reference_features, 1):
    print(f"  {i:2d}. {f}")

# Try to find the actual model file
model_path = None
possible_paths = [
    '../trade_reference/models/xgboost_sentiment_model.pkl',
    '../trade_reference/results/xgboost_sentiment/xgboost_model.pkl',
]

for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    print("\n[INFO] Model file not found. Will train using reference data...")
    # Load and train using reference approach
    from xgboost_model import XGBoostModel

    ref_train = pd.read_csv('../trade_reference/data/train_test_sentiment/train_X.csv', index_col=0, parse_dates=True)
    ref_train_y = pd.read_csv('../trade_reference/data/train_test_sentiment/train_y.csv', index_col=0, parse_dates=True)

    print(f"Training on reference data: {len(ref_train)} samples")

    model = XGBoostModel(params={
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'min_child_weight': 5
    })

    split = int(len(ref_train) * 0.8)
    X_train = ref_train.iloc[:split]
    y_train = ref_train_y.iloc[:split]
    X_val = ref_train.iloc[split:]
    y_val = ref_train_y.iloc[split:]

    model.train(X_train, y_train, X_val, y_val)
    reference_model = model.model
    print("  Model trained using reference approach!")
else:
    print(f"Loading model from: {model_path}")
    reference_model = joblib.load(model_path)
    print("  Model loaded!")

# ===== 2. LOAD FRESH TEST DATA =====
print("\n[STEP 2/5] LOADING FRESH TEST DATA")
print("=" * 70)

# Use the same fresh data we created earlier
print("Loading fresh test data (2024-09 -> 2025-10)...")

# Load from our test #2 since we know it has the right features
import yfinance as yf
from sentiment_collector import SentimentCollector
from sentiment_features import SentimentFeatureEngineer
from feature_engineering import FeatureEngineer

end_date = datetime.now()
start_date = end_date - timedelta(days=4*365)

# Download data
spy = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

collector = SentimentCollector()
sentiment_data = collector.get_fear_greed_index(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)

# Merge
spy.index = pd.to_datetime(spy.index).normalize()
sentiment_data.index = pd.to_datetime(sentiment_data.index).normalize()
combined = spy[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
combined = combined.join(sentiment_data, how='inner')

# Create features exactly like reference
fe = FeatureEngineer()
data = fe.create_basic_features(combined)

sentiment_eng = SentimentFeatureEngineer()
fg_features = sentiment_eng.create_fear_greed_features(combined)
composite = sentiment_eng.create_composite_sentiment(fg_features)

for col in fg_features.columns:
    data[col] = fg_features[col]
for col in composite.columns:
    data[col] = composite[col]

data['target'] = data['Close'].pct_change(5).shift(-5)
data = data.dropna()

print(f"Full dataset: {len(data)} days ({data.index[0].date()} to {data.index[-1].date()})")

# ===== 3. EXTRACT TEST PERIOD =====
print("\n[STEP 3/5] EXTRACTING TEST PERIOD")
print("=" * 70)

# Use 70% split like reference, but we want to test on fresh period
split_idx = int(len(data) * 0.7)
test = data.iloc[split_idx:]

print(f"Test period: {test.index[0].date()} to {test.index[-1].date()}")
print(f"Test samples: {len(test)} days")

# Check feature alignment
data_features = [col for col in data.columns
                 if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target',
                                'Dividends', 'Stock Splits', 'Capital Gains',
                                'fear_greed_value', 'fear_greed_classification']]

print(f"\nFeature alignment:")
print(f"  Reference model expects: {len(reference_features)} features")
print(f"  Our data has: {len(data_features)} features")

missing = set(reference_features) - set(data_features)
extra = set(data_features) - set(reference_features)

if missing:
    print(f"  Missing from our data: {missing}")
if extra:
    print(f"  Extra in our data: {extra}")

# Use only features the model expects
X_test = test[reference_features]
y_test = test['target']
test_prices = test['Close']

print(f"\nTest data ready: {len(X_test)} samples, {len(reference_features)} features")

# ===== 4. MAKE PREDICTIONS =====
print("\n[STEP 4/5] MAKING PREDICTIONS WITH REFERENCE MODEL")
print("=" * 70)

predictions = reference_model.predict(X_test)

print(f"Predictions: {len(predictions)}")
print(f"  Mean: {predictions.mean():.6f}")
print(f"  Std:  {predictions.std():.6f}")
print(f"  Min:  {predictions.min():.6f}")
print(f"  Max:  {predictions.max():.6f}")
print(f"  Positive: {(predictions > 0).sum()} / {len(predictions)}")

# ===== 5. BACKTEST =====
print("\n[STEP 5/5] BACKTESTING")
print("=" * 70)

from backtester import Backtester

backtester = Backtester(
    initial_capital=10000,
    transaction_cost=0.001,
    holding_period=5,
    position_size_pct=0.5,
    prediction_threshold=0.001
)

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

# ===== RESULTS =====
print("\n" + "=" * 70)
print("RESULTS - REFERENCE MODEL ON FRESH DATA")
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
print("COMPLETE COMPARISON")
print("=" * 70)
print(f"\nREFERENCE MODEL:")
print(f"  On reference period (2023-07 -> 2024-12):  Sharpe 1.28, 222 trades")
print(f"  On fresh period (2024-09 -> 2025-10):      Sharpe {sharpe:.2f}, {results['n_trades']} trades")

print(f"\nOUR MODELS (on fresh period):")
print(f"  Test #1 (WITH VIX/BTC, 28 features):      Sharpe 0.61, 172 trades")
print(f"  Test #2 (NO VIX/BTC, 17 features):        Sharpe 0.55, 101 trades")
print(f"  Test #3 (REFERENCE model):                Sharpe {sharpe:.2f}, {results['n_trades']} trades")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)

if sharpe < 0.7:
    print("The reference model ALSO performs worse on fresh data!")
    print("This confirms: The problem is NOT our model or features.")
    print("The fresh period (2024-09 -> 2025-10) is simply HARDER to predict.")
else:
    print("The reference model performs BETTER on fresh data!")
    print("This suggests: Our model implementation differs from reference.")
    print("We should investigate training/preprocessing differences.")

print("=" * 70)
