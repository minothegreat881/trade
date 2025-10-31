"""
TRAIN BASELINE MODEL ON FULL DATASET 2020-2025
================================================

Pouzije uspesny baseline model (Sharpe 1.28) na novy kompletny dataset
- Dataset: full_dataset_2020_2025.csv
- Features: 117
- Obdobie: 2020-10-15 to 2025-10-23
- Baseline XGBoost parametre
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import json
from pathlib import Path
import joblib
from scipy.stats import spearmanr

print("="*80)
print("TRAIN BASELINE MODEL ON FULL DATASET 2020-2025")
print("="*80)
print(f"Datum spustenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# 1. LOAD DATASET
# ================================================================

print("\n[1/6] Loading dataset...")

df = pd.read_csv('data/full_dataset_2020_2025.csv', index_col=0, parse_dates=True)

print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")
print(f"  Period: {df.index.min().strftime('%Y-%m-%d')} -> {df.index.max().strftime('%Y-%m-%d')}")


# ================================================================
# 2. PREPARE FEATURES
# ================================================================

print("\n[2/6] Preparing features...")

# Exclude columns
exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_value', 'fear_greed_classification',
                'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']

# Get feature columns
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"  Total features: {len(feature_cols)}")
print(f"  Target: forward 5-day return")

# Check for NaN
if df[feature_cols + ['target']].isna().sum().sum() > 0:
    print(f"\n  WARNING: Found NaN values, dropping...")
    df = df.dropna(subset=feature_cols + ['target'])
    print(f"  After dropna: {len(df)} rows")


# ================================================================
# 3. TRAIN/TEST SPLIT
# ================================================================

print("\n[3/6] Train/Test split (70/30)...")

split_idx = int(len(df) * 0.7)

train_data = df.iloc[:split_idx]
test_data = df.iloc[split_idx:]

X_train = train_data[feature_cols]
y_train = train_data['target']
X_test = test_data[feature_cols]
y_test = test_data['target']

print(f"\n  TRAIN:")
print(f"    Vzorky:   {len(X_train)}")
print(f"    Features: {len(feature_cols)}")
print(f"    Obdobie:  {train_data.index.min().strftime('%Y-%m-%d')} -> {train_data.index.max().strftime('%Y-%m-%d')}")

print(f"\n  TEST:")
print(f"    Vzorky:   {len(X_test)}")
print(f"    Features: {len(feature_cols)}")
print(f"    Obdobie:  {test_data.index.min().strftime('%Y-%m-%d')} -> {test_data.index.max().strftime('%Y-%m-%d')}")


# ================================================================
# 4. TRAIN MODEL (BASELINE PARAMETERS)
# ================================================================

print("\n[4/6] Training XGBoost s baseline parametrami...")

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
train_corr, _ = spearmanr(y_train, train_pred)
test_corr, _ = spearmanr(y_test, test_pred)

print(f"\n  Korelacia (Spearman):")
print(f"    Train: {train_corr:.4f}")
print(f"    Test:  {test_corr:.4f}")

# Prediction stats
print(f"\n  Prediction Stats:")
print(f"    Train mean: {train_pred.mean():.6f}")
print(f"    Test mean:  {test_pred.mean():.6f}")
print(f"    Train std:  {train_pred.std():.6f}")
print(f"    Test std:   {test_pred.std():.6f}")


# ================================================================
# 5. BACKTEST
# ================================================================

print("\n[5/6] Backtest...")

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
days_in_test = (test_data.index[-1] - test_data.index[0]).days
annual_return = (1 + total_return) ** (365 / days_in_test) - 1 if days_in_test > 0 else 0

print(f"\n  VYSLEDKY:")
print(f"    Sharpe Ratio:     {sharpe:.2f}")
print(f"    Annual Return:    {annual_return*100:.2f}%")
print(f"    Total Return:     {total_return*100:.2f}%")
print(f"    Max Drawdown:     {max_dd*100:.2f}%")
print(f"    Win Rate:         {win_rate*100:.1f}%")
print(f"    Total Trades:     {results['n_trades']}")
print(f"    Final Capital:    ${results['final_capital']:.2f}")


# ================================================================
# 6. POROVNANIE S ORIGINALNYM
# ================================================================

print("\n" + "="*80)
print("POROVNANIE")
print("="*80)

print(f"\n  ORIGINAL MODEL (26 features, 2020-2024):")
print(f"    Train vzorky:     834")
print(f"    Test vzorky:      358")
print(f"    Features:         26")
print(f"    Sharpe:           1.28")
print(f"    Annual Return:    12.7%")
print(f"    Max Drawdown:     -8.0%")

print(f"\n  NEW MODEL (117 features, 2020-2025):")
print(f"    Train vzorky:     {len(X_train)}")
print(f"    Test vzorky:      {len(X_test)}")
print(f"    Features:         {len(feature_cols)}")
print(f"    Sharpe:           {sharpe:.2f}")
print(f"    Annual Return:    {annual_return*100:.2f}%")
print(f"    Max Drawdown:     {max_dd*100:.2f}%")

sharpe_diff = sharpe - 1.28
sharpe_pct = (sharpe / 1.28 - 1) * 100

print(f"\n  ROZDIEL:")
print(f"    Train vzorky:     {len(X_train)-834:+d} ({(len(X_train)/834-1)*100:+.1f}%)")
print(f"    Test vzorky:      {len(X_test)-358:+d} ({(len(X_test)/358-1)*100:+.1f}%)")
print(f"    Features:         {len(feature_cols)-26:+d} (+{(len(feature_cols)-26)/26*100:.0f}%)")
print(f"    Sharpe:           {sharpe_diff:+.2f} ({sharpe_pct:+.1f}%)")

if sharpe > 1.28:
    print(f"\n  [SUCCESS] New model je lepsi! (+{sharpe_pct:.1f}%)")
elif sharpe > 1.15:
    print(f"\n  [OK] New model je podobny originalovi")
else:
    print(f"\n  [WARNING] New model je horsi")
    print(f"    Mozne priciny:")
    print(f"    - Viac features != lepsi model (overfitting)")
    print(f"    - Rozny casovy usek")
    print(f"    - Potrebna feature selection")


# ================================================================
# 7. FEATURE IMPORTANCE
# ================================================================

print("\n[6/6] Feature importance (top 20)...")

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  TOP 20 FEATURES:")
for i, row in importance_df.head(20).iterrows():
    print(f"    {row['feature']:40s} {row['importance']:.4f}")

# Check sentiment features in top 20
top_20_features = importance_df.head(20)['feature'].tolist()
sentiment_in_top20 = [f for f in top_20_features if any(x in f.lower() for x in ['fear', 'greed', 'vix', 'btc', 'sentiment'])]

print(f"\n  Sentiment features v top 20: {len(sentiment_in_top20)}")
if sentiment_in_top20:
    for f in sentiment_in_top20:
        print(f"    - {f}")


# ================================================================
# 8. SAVE RESULTS
# ================================================================

print("\n[UKLADANIE] Ukladam vysledky...")

# Save directory
save_dir = Path('results/full_dataset_baseline_2025')
save_dir.mkdir(parents=True, exist_ok=True)

# Save model
joblib.dump(model, 'models/xgboost_full_dataset_2025.pkl')

# Save metrics
metrics = {
    'created_at': datetime.now().isoformat(),
    'dataset': 'full_dataset_2020_2025.csv',
    'data_period': {
        'start': df.index.min().strftime('%Y-%m-%d'),
        'end': df.index.max().strftime('%Y-%m-%d'),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'train_period': {
            'start': train_data.index.min().strftime('%Y-%m-%d'),
            'end': train_data.index.max().strftime('%Y-%m-%d')
        },
        'test_period': {
            'start': test_data.index.min().strftime('%Y-%m-%d'),
            'end': test_data.index.max().strftime('%Y-%m-%d')
        }
    },
    'features': {
        'total': len(feature_cols),
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
        'original_features': 26,
        'original_train_samples': 834,
        'original_test_samples': 358,
        'new_sharpe': float(sharpe),
        'new_features': len(feature_cols),
        'new_train_samples': int(len(X_train)),
        'new_test_samples': int(len(X_test)),
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

# Save feature importance
importance_df.to_csv(save_dir / 'feature_importance.csv', index=False)

print(f"\n  [OK] Ulozene:")
print(f"    - models/xgboost_full_dataset_2025.pkl")
print(f"    - results/full_dataset_baseline_2025/metrics.json")
print(f"    - results/full_dataset_baseline_2025/trades.csv")
print(f"    - results/full_dataset_baseline_2025/equity_curve.csv")
print(f"    - results/full_dataset_baseline_2025/predictions.csv")
print(f"    - results/full_dataset_baseline_2025/feature_importance.csv")

print("\n" + "="*80)
print("HOTOVO!")
print("="*80)

print(f"\nSUMMARY:")
print(f"  Dataset:          full_dataset_2020_2025.csv")
print(f"  Period:           2020-10-15 -> 2025-10-23")
print(f"  Train/Test:       {len(X_train)}/{len(X_test)}")
print(f"  Features:         {len(feature_cols)}")
print(f"  Sharpe Ratio:     {sharpe:.2f}")
print(f"  vs Original:      {sharpe_diff:+.2f} ({sharpe_pct:+.1f}%)")

if sharpe > 1.28:
    print(f"\n  VYSLEDOK: Model s 117 features prekonal original 26 features!")
else:
    print(f"\n  VYSLEDOK: Viac features nepomohlo. Feature selection by mohla pomoct.")

print("="*80)
