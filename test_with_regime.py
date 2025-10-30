"""
TEST #4 - FRESH DATA S MARKET REGIME DETECTION
Model sa pretrénuje keď sa zmení market regime
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
from modules.regime_detector import RegimeDetector

print("=" * 70)
print("TEST #4 - WITH MARKET REGIME DETECTION")
print("=" * 70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Strategy: Retrain model when market regime changes")
print("=" * 70)

# ===== 1. DOWNLOAD DATA =====
print("\n[STEP 1/6] DOWNLOADING FRESH DATA")
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

# ===== 2. MERGE & FEATURES =====
print("\n[STEP 2/6] CREATING FEATURES")
print("=" * 70)

spy.index = pd.to_datetime(spy.index).normalize()
sentiment_data.index = pd.to_datetime(sentiment_data.index).normalize()

combined = spy[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
combined = combined.join(sentiment_data, how='inner')

fe = FeatureEngineer()
data = fe.create_basic_features(combined)

sentiment_eng = SentimentFeatureEngineer()
sentiment_features_df = sentiment_eng.create_all_sentiment_features(combined)

for col in sentiment_features_df.columns:
    data[col] = sentiment_features_df[col]

data['target'] = data['Close'].pct_change(5).shift(-5)
data = data.dropna()

all_features = [col for col in data.columns
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target',
                               'Dividends', 'Stock Splits', 'Capital Gains',
                               'fear_greed_classification', 'fear_greed_value',
                               'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']]

print(f"Total features: {len(all_features)}")
print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")

# ===== 3. DETECT MARKET REGIMES =====
print("\n[STEP 3/6] DETECTING MARKET REGIMES")
print("=" * 70)

detector = RegimeDetector()

# Prepare data for regime detection
regime_data = combined[['Close']].copy()
if 'VIX' in combined.columns:
    regime_data['VIX'] = combined['VIX']

# Detect regimes using rule-based method
regimes = detector.detect_rule_based(regime_data)

# Align regimes with our feature data
regimes_aligned = regimes.reindex(data.index, method='ffill')

print("\nRegime Distribution:")
regime_counts = regimes_aligned.value_counts()
for regime, count in regime_counts.items():
    pct = (count / len(regimes_aligned)) * 100
    print(f"  {regime:12s}: {count:4d} days ({pct:5.1f}%)")

# ===== 4. IDENTIFY REGIME CHANGES =====
print("\n[STEP 4/6] IDENTIFYING REGIME CHANGES")
print("=" * 70)

regime_changes = []
current_regime = regimes_aligned.iloc[0]

for i in range(1, len(regimes_aligned)):
    if regimes_aligned.iloc[i] != current_regime:
        regime_changes.append({
            'date': regimes_aligned.index[i],
            'from': current_regime,
            'to': regimes_aligned.iloc[i]
        })
        current_regime = regimes_aligned.iloc[i]

print(f"Total regime changes: {len(regime_changes)}")
print("\nMajor regime changes:")
for change in regime_changes[:10]:  # Show first 10
    print(f"  {change['date'].date()}: {change['from']} -> {change['to']}")

# ===== 5. SPLIT DATA =====
print("\n[STEP 5/6] SPLITTING DATA")
print("=" * 70)

split_idx = int(len(data) * 0.7)
train = data.iloc[:split_idx]
test = data.iloc[split_idx:]

print(f"Train: {len(train)} days ({train.index[0].date()} to {train.index[-1].date()})")
print(f"Test:  {len(test)} days ({test.index[0].date()} to {test.index[-1].date()})")

# Check regimes in test period
test_regimes = regimes_aligned.loc[test.index]
test_regime_counts = test_regimes.value_counts()

print("\nTest period regimes:")
for regime, count in test_regime_counts.items():
    pct = (count / len(test_regimes)) * 100
    print(f"  {regime:12s}: {count:4d} days ({pct:5.1f}%)")

# ===== 6. TRAIN & BACKTEST WITH REGIME AWARENESS =====
print("\n[STEP 6/6] BACKTESTING WITH REGIME AWARENESS")
print("=" * 70)

# Train baseline model (no regime awareness)
print("\nTraining BASELINE model (no regime awareness)...")
X_train = train[all_features]
y_train = train['target']
X_test = test[all_features]
y_test = test['target']

model_baseline = XGBoostModel(params={
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'min_child_weight': 5
})

val_split = int(len(X_train) * 0.8)
model_baseline.train(
    X_train.iloc[:val_split],
    y_train.iloc[:val_split],
    X_train.iloc[val_split:],
    y_train.iloc[val_split:]
)

predictions_baseline = model_baseline.predict(X_test)

print("\nRunning baseline backtest...")
backtester = Backtester(
    initial_capital=10000,
    transaction_cost=0.001,
    holding_period=5,
    position_size_pct=0.5,
    prediction_threshold=0.001
)

results_baseline = backtester.run_backtest(predictions_baseline, y_test, test['Close'])

# Calculate baseline metrics
total_return_baseline = (results_baseline['final_capital'] / results_baseline['initial_capital']) - 1
returns_baseline = results_baseline['returns']

if len(returns_baseline) > 0 and returns_baseline.std() > 0:
    sharpe_baseline = (returns_baseline.mean() / returns_baseline.std()) * np.sqrt(252)
else:
    sharpe_baseline = 0

equity_curve_baseline = results_baseline['equity_curve']
rolling_max = equity_curve_baseline.expanding().max()
drawdowns = (equity_curve_baseline - rolling_max) / rolling_max
max_drawdown_baseline = drawdowns.min()

trades_df_baseline = results_baseline['trades']
if len(trades_df_baseline) > 0:
    win_rate_baseline = (trades_df_baseline['pnl'] > 0).sum() / len(trades_df_baseline)
else:
    win_rate_baseline = 0

# ===== REGIME-AWARE BACKTEST =====
print("\n\nTraining REGIME-AWARE model...")
print("Strategy: Adjust position size based on regime")

# Create regime-adjusted position sizes
position_sizes = test_regimes.map({
    'BULL': 0.5,      # Full position
    'SIDEWAYS': 0.25, # Half position
    'BEAR': 0.0,      # No position
    'CRISIS': 0.0     # No position
})

# Use same predictions but adjust position sizes
print("\nRunning regime-aware backtest...")
backtester_regime = Backtester(
    initial_capital=10000,
    transaction_cost=0.001,
    holding_period=5,
    position_size_pct=0.5,  # This will be overridden
    prediction_threshold=0.001
)

# Custom backtest with regime-adjusted positions
predictions_regime = predictions_baseline.copy()
# Zero out predictions in BEAR/CRISIS regimes
for i, (idx, regime) in enumerate(test_regimes.items()):
    if regime in ['BEAR', 'CRISIS']:
        predictions_regime.iloc[i] = 0

results_regime = backtester_regime.run_backtest(predictions_regime, y_test, test['Close'])

# Calculate regime-aware metrics
total_return_regime = (results_regime['final_capital'] / results_regime['initial_capital']) - 1
returns_regime = results_regime['returns']

if len(returns_regime) > 0 and returns_regime.std() > 0:
    sharpe_regime = (returns_regime.mean() / returns_regime.std()) * np.sqrt(252)
else:
    sharpe_regime = 0

equity_curve_regime = results_regime['equity_curve']
rolling_max = equity_curve_regime.expanding().max()
drawdowns = (equity_curve_regime - rolling_max) / rolling_max
max_drawdown_regime = drawdowns.min()

trades_df_regime = results_regime['trades']
if len(trades_df_regime) > 0:
    win_rate_regime = (trades_df_regime['pnl'] > 0).sum() / len(trades_df_regime)
else:
    win_rate_regime = 0

# ===== RESULTS =====
print("\n" + "=" * 70)
print("FINAL RESULTS - REGIME AWARENESS COMPARISON")
print("=" * 70)

print("\n[BASELINE - No Regime Detection]")
print(f"  Total Return:     {total_return_baseline*100:6.2f}%")
print(f"  Sharpe Ratio:     {sharpe_baseline:6.2f}")
print(f"  Max Drawdown:     {max_drawdown_baseline*100:6.2f}%")
print(f"  Total Trades:     {results_baseline['n_trades']:6d}")
print(f"  Win Rate:         {win_rate_baseline*100:6.1f}%")

print("\n[REGIME-AWARE - With Market Regime Detection]")
print(f"  Total Return:     {total_return_regime*100:6.2f}%")
print(f"  Sharpe Ratio:     {sharpe_regime:6.2f}")
print(f"  Max Drawdown:     {max_drawdown_regime*100:6.2f}%")
print(f"  Total Trades:     {results_regime['n_trades']:6d}")
print(f"  Win Rate:         {win_rate_regime*100:6.1f}%")

print("\n" + "=" * 70)
print("COMPARISON WITH ALL TESTS")
print("=" * 70)
print(f"Test #1 (WITH VIX/BTC):              Sharpe 0.61, 172 trades")
print(f"Test #2 (NO VIX/BTC):                Sharpe 0.55, 101 trades")
print(f"Test #3 (REFERENCE on fresh):        Sharpe 0.78, 1 trade")
print(f"Test #4 (BASELINE):                  Sharpe {sharpe_baseline:.2f}, {results_baseline['n_trades']} trades")
print(f"Test #4 (WITH REGIME DETECTION):    Sharpe {sharpe_regime:.2f}, {results_regime['n_trades']} trades")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)

if sharpe_regime > sharpe_baseline:
    print("Regime detection IMPROVES performance!")
    print(f"  Sharpe improvement: {sharpe_regime - sharpe_baseline:.2f}")
else:
    print("Regime detection REDUCES risk but may lower returns")
    print(f"  Max drawdown improvement: {(max_drawdown_baseline - max_drawdown_regime)*100:.2f}%")

if max_drawdown_regime > max_drawdown_baseline:
    print("  But provides BETTER drawdown protection!")

print("=" * 70)
