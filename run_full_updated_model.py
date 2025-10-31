"""
FULL UPDATED MODEL - Všetky dáta po 31.10.2025
================================================

Stiahne VŠETKY dáta k aktuálnemu dátumu:
1. SPY (stock prices)
2. Fear & Greed Index
3. VIX (volatility index)
4. Bitcoin prices

Vytvorí všetkých 26 features a natrénuje s baseline parametrami (Sharpe 1.28)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import xgboost as xgb
from datetime import datetime, timedelta
import json
from scipy.stats import spearmanr

print("="*80)
print("FULL UPDATED MODEL - Všetky dáta po 31.10.2025")
print("="*80)
print(f"Datum spustenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# 1. STIAHNUTIE SPY DATA
# ================================================================

print("\n[1/5] Stiahnutie SPY data...")

spy = yf.download('SPY', start='2020-01-01', end='2025-11-01', progress=False)
spy.index = spy.index.tz_localize(None)

print(f"  SPY: {len(spy)} dni")
print(f"  Obdobie: {spy.index.min().strftime('%Y-%m-%d')} -> {spy.index.max().strftime('%Y-%m-%d')}")


# ================================================================
# 2. STIAHNUTIE SENTIMENT DATA
# ================================================================

print("\n[2/5] Stiahnutie sentiment data...")
print("  (Fear & Greed, VIX, Bitcoin)")

from sentiment_collector import SentimentCollector

collector = SentimentCollector()
start_str = spy.index.min().strftime('%Y-%m-%d')
end_str = spy.index.max().strftime('%Y-%m-%d')

print(f"  Zber od {start_str} po {end_str}...")

sentiment_data = collector.collect_all_sentiment(start_str, end_str)

# sentiment_data je DataFrame, nie dictionary
print(f"  Sentiment data: {len(sentiment_data)} dni")
print(f"  Columns: {', '.join(sentiment_data.columns)}")


# ================================================================
# 3. VYTVORENIE TECHNICAL FEATURES (11)
# ================================================================

print("\n[3/5] Vytvorenie technical features...")

data = spy.copy()

# Returns
data['return_1d'] = data['Close'].pct_change()
data['return_5d'] = data['Close'].pct_change(5)
data['return_10d'] = data['Close'].pct_change(10)
data['return_20d'] = data['Close'].pct_change(20)

# Volatility
data['volatility_20d'] = data['return_1d'].rolling(20).std()
data['volatility_60d'] = data['return_1d'].rolling(60).std()

# Volume
data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()

# Price position
high_20 = data['High'].rolling(20).max()
low_20 = data['Low'].rolling(20).min()
data['price_position'] = (data['Close'] - low_20) / (high_20 - low_20 + 1e-8)

# Moving averages
data['sma_20'] = data['Close'].rolling(20).mean()
data['sma_50'] = data['Close'].rolling(50).mean()

# Trend
data['trend'] = (data['sma_20'] > data['sma_50']).astype(int)

# Target
data['target'] = data['Close'].pct_change(5).shift(-5)

tech_features = ['return_1d', 'return_5d', 'return_10d', 'return_20d',
                 'volatility_20d', 'volatility_60d', 'volume_ratio',
                 'price_position', 'sma_20', 'sma_50', 'trend']

print(f"  Technical features: {len(tech_features)}")
for i, f in enumerate(tech_features, 1):
    print(f"    {i:2d}. {f}")


# ================================================================
# 4. SENTIMENT FEATURES (15)
# ================================================================

print("\n[4/5] Vytvorenie sentiment features...")

from sentiment_features import SentimentFeatureEngineer

engineer = SentimentFeatureEngineer()
sentiment_features = engineer.create_all_sentiment_features(sentiment_data)

print(f"  Sentiment features: {len(sentiment_features.columns)}")
for i, f in enumerate(sentiment_features.columns, 1):
    print(f"    {i:2d}. {f}")


# ================================================================
# 5. MERGE A CLEAN
# ================================================================

print("\n[5/5] Merge technical + sentiment...")

# Prepare data
data_clean = data[tech_features + ['target', 'Close']].copy()
data_clean = data_clean.dropna()

# Normalize dates
data_clean.index = data_clean.index.normalize()
sentiment_features.index = sentiment_features.index.normalize()

# Merge
combined = data_clean[tech_features].join(sentiment_features, how='inner')
combined['target'] = data_clean['target']
combined['Close'] = data_clean['Close']

combined = combined.dropna()

all_features = tech_features + list(sentiment_features.columns)

print(f"  Pred merge: {len(data_clean)}")
print(f"  Po merge: {len(combined)}")
print(f"  Total features: {len(all_features)} (11 tech + {len(sentiment_features.columns)} sent)")

# Verify we have exactly 26 features
if len(all_features) != 26:
    print(f"\n  [WARNING] Expected 26 features, got {len(all_features)}")
    print(f"  Missing or extra features!")
else:
    print(f"\n  [OK] Exactly 26 features as expected!")


# ================================================================
# 6. TRAIN/TEST SPLIT (70/30)
# ================================================================

print("\n[6/8] Train/Test split (70/30)...")

split_idx = int(len(combined) * 0.7)

train_data = combined.iloc[:split_idx]
test_data = combined.iloc[split_idx:]

X_train = train_data[all_features]
y_train = train_data['target']
X_test = test_data[all_features]
y_test = test_data['target']

print(f"\n  TRAIN:")
print(f"    Vzorky:   {len(X_train)}")
print(f"    Features: {len(all_features)}")
print(f"    Obdobie:  {train_data.index.min().strftime('%Y-%m-%d')} -> {train_data.index.max().strftime('%Y-%m-%d')}")
print(f"    Dni:      {(train_data.index.max() - train_data.index.min()).days}")

print(f"\n  TEST:")
print(f"    Vzorky:   {len(X_test)}")
print(f"    Features: {len(all_features)}")
print(f"    Obdobie:  {test_data.index.min().strftime('%Y-%m-%d')} -> {test_data.index.max().strftime('%Y-%m-%d')}")
print(f"    Dni:      {(test_data.index.max() - test_data.index.min()).days}")


# ================================================================
# 7. TRAINING (BASELINE PARAMS - Sharpe 1.28)
# ================================================================

print("\n[7/8] Training s BASELINE parametrami...")

baseline_params = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 42,
    'verbosity': 0
}

print("\n  Baseline parametre (original Sharpe 1.28):")
for k, v in baseline_params.items():
    if k not in ['verbosity', 'random_state']:
        print(f"    {k:20s} = {v}")

model = xgb.XGBRegressor(**baseline_params)
model.fit(X_train, y_train)

print(f"\n  [OK] Model natreovany!")

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Correlations
train_corr, _ = spearmanr(y_train, train_pred)
test_corr, _ = spearmanr(y_test, test_pred)

print(f"\n  Korelacia:")
print(f"    Train: {train_corr:.4f}")
print(f"    Test:  {test_corr:.4f}")


# ================================================================
# 8. BACKTEST
# ================================================================

print("\n[8/8] Backtest...")

from backtester import Backtester

backtester = Backtester(
    initial_capital=10000,
    transaction_cost=0.001,
    holding_period=5,
    position_size_pct=0.5,
    prediction_threshold=0.001
)

results = backtester.run_backtest(test_pred, y_test, test_data['Close'])

# Calculate metrics
returns = results['returns']
sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

equity = results['equity_curve']
rolling_max = equity.expanding().max()
drawdowns = (equity - rolling_max) / rolling_max
max_dd = drawdowns.min()

trades_df = results['trades']
win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) if len(trades_df) > 0 else 0

total_return = (results['final_capital'] / results['initial_capital']) - 1
annual_return = (1 + total_return) ** (252 / len(test_data)) - 1

print(f"\n  VYSLEDKY:")
print(f"    Sharpe Ratio:     {sharpe:.2f}")
print(f"    Annual Return:    {annual_return*100:.2f}%")
print(f"    Total Return:     {total_return*100:.2f}%")
print(f"    Max Drawdown:     {max_dd*100:.2f}%")
print(f"    Win Rate:         {win_rate*100:.1f}%")
print(f"    Total Trades:     {results['n_trades']}")
print(f"    Final Capital:    ${results['final_capital']:.2f}")


# ================================================================
# 9. POROVNANIE S ORIGINALNYM
# ================================================================

print("\n" + "="*80)
print("POROVNANIE S ORIGINALNYM MODELOM")
print("="*80)

print(f"\n  ORIGINAL MODEL (2020-2024):")
print(f"    Train vzorky:     834")
print(f"    Test vzorky:      358")
print(f"    Obdobie:          2020-03-30 -> 2024-12-20")
print(f"    Sharpe:           1.28")
print(f"    Annual Return:    12.7%")
print(f"    Max Drawdown:     -8.0%")
print(f"    Win Rate:         59.0%")
print(f"    Total Trades:     222")

print(f"\n  UPDATED MODEL (2020-2025.10.31):")
print(f"    Train vzorky:     {len(X_train)}")
print(f"    Test vzorky:      {len(X_test)}")
print(f"    Obdobie:          {train_data.index.min().strftime('%Y-%m-%d')} -> {test_data.index.max().strftime('%Y-%m-%d')}")
print(f"    Sharpe:           {sharpe:.2f}")
print(f"    Annual Return:    {annual_return*100:.2f}%")
print(f"    Max Drawdown:     {max_dd*100:.2f}%")
print(f"    Win Rate:         {win_rate*100:.1f}%")
print(f"    Total Trades:     {results['n_trades']}")

# Calculate differences
train_diff = len(X_train) - 834
test_diff = len(X_test) - 358
sharpe_diff = sharpe - 1.28
sharpe_pct = ((sharpe / 1.28) - 1) * 100 if sharpe > 0 else -100

print(f"\n  ROZDIEL:")
print(f"    Train vzorky:     {train_diff:+d} ({(len(X_train)/834-1)*100:+.1f}%)")
print(f"    Test vzorky:      {test_diff:+d} ({(len(X_test)/358-1)*100:+.1f}%)")
print(f"    Sharpe:           {sharpe_diff:+.2f} ({sharpe_pct:+.1f}%)")
print(f"    Annual Return:    {(annual_return-0.127)*100:+.2f} pp")
print(f"    Max Drawdown:     {(max_dd+0.08)*100:+.2f} pp")

if sharpe > 1.35:
    print(f"\n  [VYBORNE] Updated model je vyrazne lepsi! ({sharpe_pct:+.1f}%)")
elif sharpe > 1.20:
    print(f"\n  [DOBRE] Updated model je podobny originalovi")
elif sharpe > 1.00:
    print(f"\n  [OK] Updated model je o nieco horsi, ale stale dobry")
else:
    print(f"\n  [WARNING] Updated model je vyrazne horsi")
    print(f"    Mozne priciny:")
    print(f"      - Iny casovy usek (2025 ma ine trzne podmienky)")
    print(f"      - Sentiments data sa zmenili")
    print(f"      - Trh je volatilnejsi v 2025")


# ================================================================
# 10. ULOZENIE VYSLEDKOV
# ================================================================

print("\n[UKLADANIE] Ukladam vysledky...")

# Create directories
save_dir = Path('results/full_updated_model_2025')
save_dir.mkdir(parents=True, exist_ok=True)

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

# Save model
import joblib
joblib.dump(model, models_dir / 'xgboost_full_updated_2025.pkl')

# Save metrics
metrics = {
    'created_at': datetime.now().isoformat(),
    'data_source': {
        'spy_start': spy.index.min().strftime('%Y-%m-%d'),
        'spy_end': spy.index.max().strftime('%Y-%m-%d'),
        'spy_days': len(spy),
        'train_start': train_data.index.min().strftime('%Y-%m-%d'),
        'train_end': train_data.index.max().strftime('%Y-%m-%d'),
        'test_start': test_data.index.min().strftime('%Y-%m-%d'),
        'test_end': test_data.index.max().strftime('%Y-%m-%d'),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    },
    'features': {
        'total': len(all_features),
        'technical': len(tech_features),
        'sentiment': len(sentiment_features.columns),
        'list': all_features
    },
    'parameters': {k: v for k, v in baseline_params.items() if k != 'verbosity'},
    'performance': {
        'sharpe_ratio': float(sharpe),
        'annual_return': float(annual_return),
        'total_return': float(total_return),
        'max_drawdown': float(max_dd),
        'max_drawdown_pct': float(max_dd * 100),
        'win_rate': float(win_rate),
        'total_trades': int(results['n_trades']),
        'final_capital': float(results['final_capital']),
        'initial_capital': float(results['initial_capital'])
    },
    'correlation': {
        'train': float(train_corr),
        'test': float(test_corr)
    },
    'comparison_with_original': {
        'original_sharpe': 1.28,
        'original_annual_return': 0.127,
        'original_max_dd': -0.08,
        'original_win_rate': 0.59,
        'original_trades': 222,
        'updated_sharpe': float(sharpe),
        'sharpe_difference': float(sharpe_diff),
        'sharpe_percent_change': float(sharpe_pct)
    }
}

with open(save_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save trades
trades_df.to_csv(save_dir / 'trades.csv', index=False)

# Save equity curve
equity_df = pd.DataFrame({
    'date': test_data.index,
    'equity': equity.values
})
equity_df.to_csv(save_dir / 'equity_curve.csv', index=False)

# Save predictions
pred_df = pd.DataFrame({
    'date': test_data.index,
    'actual': y_test.values,
    'predicted': test_pred
}, index=test_data.index)
pred_df.to_csv(save_dir / 'predictions.csv')

# Save train/test data
X_train.to_csv('data/train_full_updated_2025.csv')
X_test.to_csv('data/test_full_updated_2025.csv')
pd.DataFrame({'target': y_train}).to_csv('data/train_y_full_updated_2025.csv')
pd.DataFrame({'target': y_test}).to_csv('data/test_y_full_updated_2025.csv')

print(f"  [OK] Ulozene:")
print(f"    - models/xgboost_full_updated_2025.pkl")
print(f"    - results/full_updated_model_2025/metrics.json")
print(f"    - results/full_updated_model_2025/trades.csv")
print(f"    - results/full_updated_model_2025/equity_curve.csv")
print(f"    - results/full_updated_model_2025/predictions.csv")
print(f"    - data/train_full_updated_2025.csv")
print(f"    - data/test_full_updated_2025.csv")

print("\n" + "="*80)
print("HOTOVO!")
print("="*80)

print(f"\nSUMMARY:")
print(f"  Data updated do:  31.10.2025")
print(f"  Train samples:    {len(X_train)} (original: 834)")
print(f"  Test samples:     {len(X_test)} (original: 358)")
print(f"  Features:         {len(all_features)}")
print(f"  Sharpe Ratio:     {sharpe:.2f} (original: 1.28)")
print(f"  Change:           {sharpe_diff:+.2f} ({sharpe_pct:+.1f}%)")

print("="*80)
