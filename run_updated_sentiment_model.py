"""
UPDATED SENTIMENT MODEL - Data az po 31.10.2025
================================================

Krok po kroku:
1. Stiahne SPY data od 2020-01-01 po 2025-10-31
2. Stiahne sentiment data (Fear & Greed, VIX, BTC)
3. Vytvori features (26 features - rovnake ako original)
4. Train/test split (70/30)
5. Natrenuje XGBoost s baseline parametrami (Sharpe 1.28)
6. Backtest a metriky
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import xgboost as xgb
from datetime import datetime
import json

print("="*80)
print("UPDATED SENTIMENT MODEL - Data do 31.10.2025")
print("="*80)
print(f"Datum spustenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# 1. STIAHNUTIE DAT
# ================================================================

print("\n[1/7] Stiahnutie SPY data...")

# Stiahneme SPY od 2020 po dnes (31.10.2025)
spy = yf.download('SPY', start='2020-01-01', end='2025-11-01', progress=False)
spy.index = spy.index.tz_localize(None)  # Remove timezone

print(f"  Stiahnutych dni: {len(spy)}")
print(f"  Obdobie: {spy.index.min().strftime('%Y-%m-%d')} -> {spy.index.max().strftime('%Y-%m-%d')}")


# ================================================================
# 2. VYTVORENIE TECHNICKYCH FEATURES
# ================================================================

print("\n[2/7] Vytvorenie technickych features...")

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

# Price position in range
high_20 = data['High'].rolling(20).max()
low_20 = data['Low'].rolling(20).min()
data['price_position'] = (data['Close'] - low_20) / (high_20 - low_20 + 1e-8)

# Moving averages
data['sma_20'] = data['Close'].rolling(20).mean()
data['sma_50'] = data['Close'].rolling(50).mean()

# Trend
data['trend'] = (data['sma_20'] > data['sma_50']).astype(int)

# Target (forward 5-day return)
data['target'] = data['Close'].pct_change(5).shift(-5)

print(f"  Technickych features: 11")
print(f"  Target: forward 5-day return")


# ================================================================
# 3. STIAHNUTIE SENTIMENT DAT
# ================================================================

print("\n[3/7] Stiahnutie sentiment data...")
print("  (Fear & Greed, VIX, BTC)")

from sentiment_collector import SentimentCollector

collector = SentimentCollector()
start_str = spy.index.min().strftime('%Y-%m-%d')
end_str = spy.index.max().strftime('%Y-%m-%d')

sentiment_data = collector.collect_all_sentiment(start_str, end_str)

print(f"  Fear & Greed: {len(sentiment_data['fear_greed'])} dni")
print(f"  VIX: {len(sentiment_data['vix'])} dni")
print(f"  BTC: {len(sentiment_data['btc'])} dni")


# ================================================================
# 4. SENTIMENT FEATURES
# ================================================================

print("\n[4/7] Vytvorenie sentiment features...")

from sentiment_features import SentimentFeatureEngineer

engineer = SentimentFeatureEngineer()
sentiment_features = engineer.create_all_sentiment_features(sentiment_data)

print(f"  Sentiment features: {len(sentiment_features.columns)}")
print(f"  Features:")
for i, col in enumerate(sentiment_features.columns, 1):
    print(f"    {i:2d}. {col}")


# ================================================================
# 5. MERGE TECHNICAL + SENTIMENT
# ================================================================

print("\n[5/7] Merge technical + sentiment features...")

# Technical features
tech_features = ['return_1d', 'return_5d', 'return_10d', 'return_20d',
                 'volatility_20d', 'volatility_60d', 'volume_ratio',
                 'price_position', 'sma_20', 'sma_50', 'trend']

# Prepare data
data_clean = data[tech_features + ['target', 'Close']].copy()
data_clean = data_clean.dropna()

# Normalize dates (remove time)
data_clean.index = data_clean.index.normalize()
sentiment_features.index = sentiment_features.index.normalize()

# Merge
combined = data_clean[tech_features].join(sentiment_features, how='inner')
combined['target'] = data_clean['target']
combined['Close'] = data_clean['Close']

combined = combined.dropna()

print(f"  Pred merge: {len(data_clean)} vzoriek")
print(f"  Po merge: {len(combined)} vzoriek")
print(f"  Celkovo features: {len(tech_features) + len(sentiment_features.columns)} (11 tech + {len(sentiment_features.columns)} sent)")


# ================================================================
# 6. TRAIN/TEST SPLIT
# ================================================================

print("\n[6/7] Train/Test split...")

# 70% train, 30% test
split_idx = int(len(combined) * 0.7)

train_data = combined.iloc[:split_idx]
test_data = combined.iloc[split_idx:]

feature_cols = tech_features + list(sentiment_features.columns)

X_train = train_data[feature_cols]
y_train = train_data['target']
X_test = test_data[feature_cols]
y_test = test_data['target']

print(f"\n  TRAIN:")
print(f"    Vzorky:   {len(X_train)}")
print(f"    Features: {len(feature_cols)}")
print(f"    Obdobie:  {train_data.index.min().strftime('%Y-%m-%d')} -> {train_data.index.max().strftime('%Y-%m-%d')}")
print(f"    Dni:      {(train_data.index.max() - train_data.index.min()).days}")

print(f"\n  TEST:")
print(f"    Vzorky:   {len(X_test)}")
print(f"    Features: {len(feature_cols)}")
print(f"    Obdobie:  {test_data.index.min().strftime('%Y-%m-%d')} -> {test_data.index.max().strftime('%Y-%m-%d')}")
print(f"    Dni:      {(test_data.index.max() - test_data.index.min()).days}")


# ================================================================
# 7. TRENOVANIE MODELU (BASELINE PARAMS)
# ================================================================

print("\n[7/7] Trenovanie modelu s baseline parametrami...")

# BASELINE PARAMETERS (Sharpe 1.28)
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

print("\n  Baseline parametre:")
for k, v in baseline_params.items():
    if k != 'verbosity':
        print(f"    {k:20s} = {v}")

model = xgb.XGBRegressor(**baseline_params)
model.fit(X_train, y_train)

print(f"\n  Model natreovany!")

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Correlations
from scipy.stats import spearmanr
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

# Sharpe
returns = results['returns']
sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

# Max Drawdown
equity = results['equity_curve']
rolling_max = equity.expanding().max()
drawdowns = (equity - rolling_max) / rolling_max
max_dd = drawdowns.min()

# Win Rate
trades_df = results['trades']
win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) if len(trades_df) > 0 else 0

# Annual Return
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
print("POROVNANIE")
print("="*80)

print(f"\n  ORIGINAL (2020-2024):")
print(f"    Train vzorky:     834")
print(f"    Test vzorky:      358")
print(f"    Sharpe:           1.28")
print(f"    Annual Return:    12.7%")
print(f"    Max Drawdown:     -8.0%")

print(f"\n  UPDATED (2020-2025.10.31):")
print(f"    Train vzorky:     {len(X_train)}")
print(f"    Test vzorky:      {len(X_test)}")
print(f"    Sharpe:           {sharpe:.2f}")
print(f"    Annual Return:    {annual_return*100:.2f}%")
print(f"    Max Drawdown:     {max_dd*100:.2f}%")

sharpe_diff = sharpe - 1.28
sharpe_pct = (sharpe / 1.28 - 1) * 100

print(f"\n  ROZDIEL:")
print(f"    Train vzorky:     {len(X_train)-834:+d} ({(len(X_train)/834-1)*100:+.1f}%)")
print(f"    Test vzorky:      {len(X_test)-358:+d} ({(len(X_test)/358-1)*100:+.1f}%)")
print(f"    Sharpe:           {sharpe_diff:+.2f} ({sharpe_pct:+.1f}%)")

if sharpe > 1.28:
    print(f"\n  [USPECH] Updated model je lepsi! (+{sharpe_pct:.1f}%)")
elif sharpe > 1.15:
    print(f"\n  [OK] Updated model je podobny originalovi")
else:
    print(f"\n  [WARNING] Updated model je horsi. Mozne priciny:")
    print(f"    - Iny casovy usek (2025 data mozu byt volatilnejsie)")
    print(f"    - Zmeny v trznom spravani")
    print(f"    - Sentiment data mozu byt ine")


# ================================================================
# 10. ULOZENIE
# ================================================================

print("\n[UKLADANIE] Ukladam vysledky...")

# Save directory
save_dir = Path('results/updated_sentiment_model')
save_dir.mkdir(parents=True, exist_ok=True)

# Save model
import joblib
joblib.dump(model, 'models/xgboost_updated_sentiment.pkl')

# Save metrics
metrics = {
    'created_at': datetime.now().isoformat(),
    'data_period': {
        'start': spy.index.min().strftime('%Y-%m-%d'),
        'end': spy.index.max().strftime('%Y-%m-%d'),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    },
    'features': {
        'total': len(feature_cols),
        'technical': len(tech_features),
        'sentiment': len(sentiment_features.columns),
        'list': feature_cols
    },
    'parameters': baseline_params,
    'performance': {
        'sharpe_ratio': float(sharpe),
        'annual_return': float(annual_return),
        'total_return': float(total_return),
        'max_drawdown': float(max_dd),
        'max_drawdown_pct': float(max_dd * 100),
        'win_rate': float(win_rate),
        'total_trades': int(results['n_trades']),
        'final_capital': float(results['final_capital'])
    },
    'correlation': {
        'train': float(train_corr),
        'test': float(test_corr)
    },
    'comparison_with_original': {
        'original_sharpe': 1.28,
        'updated_sharpe': float(sharpe),
        'difference': float(sharpe_diff),
        'percent_change': float(sharpe_pct)
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

# Save data
X_train.to_csv('data/train_updated_sentiment.csv')
X_test.to_csv('data/test_updated_sentiment.csv')
pd.DataFrame({'target': y_train}).to_csv('data/train_y_updated.csv')
pd.DataFrame({'target': y_test}).to_csv('data/test_y_updated.csv')

print(f"  [OK] Ulozene:")
print(f"    - models/xgboost_updated_sentiment.pkl")
print(f"    - results/updated_sentiment_model/metrics.json")
print(f"    - results/updated_sentiment_model/trades.csv")
print(f"    - results/updated_sentiment_model/equity_curve.csv")
print(f"    - results/updated_sentiment_model/predictions.csv")
print(f"    - data/train_updated_sentiment.csv")
print(f"    - data/test_updated_sentiment.csv")

print("\n" + "="*80)
print("HOTOVO!")
print("="*80)

print(f"\nSUMMARY:")
print(f"  Data do:          31.10.2025")
print(f"  Train vzorky:     {len(X_train)}")
print(f"  Test vzorky:      {len(X_test)}")
print(f"  Features:         {len(feature_cols)}")
print(f"  Sharpe:           {sharpe:.2f}")
print(f"  vs Original:      {sharpe_diff:+.2f} ({sharpe_pct:+.1f}%)")

print("="*80)
