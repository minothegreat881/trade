"""
OPTIMALIZACIA PARAMETROV - FULL DATASET 2020-2025
==================================================

Hlada najlepsie XGBoost parametre pre maximalizaciu Sharpe ratio
- Dataset: full_dataset_2020_2025.csv (117 features)
- Baseline: Sharpe 1.34
- Target: Sharpe > 1.34
- Metoda: Optuna optimization
- Iterations: 50 (potom viac)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import json
from pathlib import Path
import joblib
import optuna
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr

print("="*80)
print("OPTIMALIZACIA PARAMETROV - FULL DATASET 2020-2025")
print("="*80)
print(f"Datum spustenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# 1. LOAD DATASET
# ================================================================

print("\n[1/5] Loading dataset...")

df = pd.read_csv('data/full_dataset_2020_2025.csv', index_col=0, parse_dates=True)

print(f"  Rows: {len(df)}")
print(f"  Period: {df.index.min().strftime('%Y-%m-%d')} -> {df.index.max().strftime('%Y-%m-%d')}")

# Exclude columns
exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_value', 'fear_greed_classification',
                'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']

# Get feature columns
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"  Features: {len(feature_cols)}")

# Check for NaN
if df[feature_cols + ['target']].isna().sum().sum() > 0:
    df = df.dropna(subset=feature_cols + ['target'])
    print(f"  After dropna: {len(df)} rows")


# ================================================================
# 2. TRAIN/TEST SPLIT
# ================================================================

print("\n[2/5] Train/Test split (70/30)...")

split_idx = int(len(df) * 0.7)

train_data = df.iloc[:split_idx]
test_data = df.iloc[split_idx:]

X_train = train_data[feature_cols]
y_train = train_data['target']
X_test = test_data[feature_cols]
y_test = test_data['target']
test_close = test_data['Close']

print(f"  Train: {len(X_train)} samples ({train_data.index.min().strftime('%Y-%m-%d')} -> {train_data.index.max().strftime('%Y-%m-%d')})")
print(f"  Test:  {len(X_test)} samples ({test_data.index.min().strftime('%Y-%m-%d')} -> {test_data.index.max().strftime('%Y-%m-%d')})")


# ================================================================
# 3. BASELINE PERFORMANCE
# ================================================================

print("\n[3/5] Baseline performance...")

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

baseline_model = xgb.XGBRegressor(**baseline_params)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)

# Calculate baseline Sharpe
from backtester import Backtester

backtester = Backtester(
    initial_capital=10000,
    transaction_cost=0.001,
    holding_period=5,
    position_size_pct=0.5,
    prediction_threshold=0.001
)

baseline_results = backtester.run_backtest(baseline_pred, y_test, test_close)
baseline_returns = baseline_results['returns']
baseline_sharpe = (baseline_returns.mean() / baseline_returns.std()) * np.sqrt(252) if len(baseline_returns) > 0 and baseline_returns.std() > 0 else 0

print(f"  Baseline Sharpe: {baseline_sharpe:.2f}")
print(f"  Target: Beat {baseline_sharpe:.2f}")


# ================================================================
# 4. OPTUNA OPTIMIZATION
# ================================================================

print("\n[4/5] Optuna optimization (50 trials)...")
print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")


def objective(trial):
    """
    Objective funkcia pre Optuna - maximalizuje Sharpe ratio
    """
    # Suggest hyperparameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
        'random_state': 42,
        'verbosity': 0
    }

    # Time Series Cross-Validation (3 folds)
    tscv = TimeSeriesSplit(n_splits=3)
    sharpe_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)

        # Predict
        val_pred = model.predict(X_val)

        # Calculate Sharpe on validation fold
        # Simple Sharpe based on predictions vs actual
        # Create mock trades based on predictions
        threshold = 0.001
        positions = (val_pred > threshold).astype(int)

        if positions.sum() > 0:
            # Calculate returns when we have positions
            actual_returns = y_val.values
            strategy_returns = positions * actual_returns

            if strategy_returns.std() > 0:
                sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0

        sharpe_scores.append(sharpe)

    # Return mean Sharpe across folds
    mean_sharpe = np.mean(sharpe_scores)

    return mean_sharpe


# Create Optuna study
study = optuna.create_study(
    direction='maximize',
    study_name='xgboost_sharpe_optimization',
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Optimize
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n  End: {datetime.now().strftime('%H:%M:%S')}")


# ================================================================
# 5. BEST PARAMETERS
# ================================================================

print("\n[5/5] Best parameters found...")

best_params = study.best_params
best_params['random_state'] = 42
best_params['verbosity'] = 0

print(f"\n  Best Sharpe (CV): {study.best_value:.4f}")
print(f"\n  Best Parameters:")
for k, v in best_params.items():
    if k not in ['random_state', 'verbosity']:
        print(f"    {k:20s} = {v}")


# ================================================================
# 6. TRAIN FINAL MODEL
# ================================================================

print("\n[6/6] Training final model with best parameters...")

best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)
best_pred = best_model.predict(X_test)

# Test correlation
test_corr, _ = spearmanr(y_test, best_pred)

print(f"  Test correlation: {test_corr:.4f}")


# ================================================================
# 7. BACKTEST
# ================================================================

print("\n[7/7] Backtest with best model...")

best_results = backtester.run_backtest(best_pred, y_test, test_close)

# Sharpe
best_returns = best_results['returns']
best_sharpe = (best_returns.mean() / best_returns.std()) * np.sqrt(252) if len(best_returns) > 0 and best_returns.std() > 0 else 0

# Max Drawdown
best_equity = best_results['equity_curve']
rolling_max = best_equity.expanding().max()
drawdowns = (best_equity - rolling_max) / rolling_max
best_max_dd = drawdowns.min()

# Win Rate
best_trades_df = best_results['trades']
best_win_rate = (best_trades_df['pnl'] > 0).sum() / len(best_trades_df) if len(best_trades_df) > 0 else 0

# Annual Return
best_total_return = (best_results['final_capital'] / best_results['initial_capital']) - 1
days_in_test = (test_data.index[-1] - test_data.index[0]).days
best_annual_return = (1 + best_total_return) ** (365 / days_in_test) - 1 if days_in_test > 0 else 0

print(f"\n  BEST MODEL RESULTS:")
print(f"    Sharpe Ratio:     {best_sharpe:.2f}")
print(f"    Annual Return:    {best_annual_return*100:.2f}%")
print(f"    Total Return:     {best_total_return*100:.2f}%")
print(f"    Max Drawdown:     {best_max_dd*100:.2f}%")
print(f"    Win Rate:         {best_win_rate*100:.1f}%")
print(f"    Total Trades:     {best_results['n_trades']}")
print(f"    Final Capital:    ${best_results['final_capital']:.2f}")


# ================================================================
# 8. POROVNANIE
# ================================================================

print("\n" + "="*80)
print("POROVNANIE BASELINE vs OPTIMIZED")
print("="*80)

print(f"\n                     BASELINE    OPTIMIZED    ROZDIEL")
print(f"Sharpe Ratio:        {baseline_sharpe:7.2f}     {best_sharpe:7.2f}      {best_sharpe-baseline_sharpe:+6.2f}")
print(f"Annual Return:       {baseline_results['final_capital']/baseline_results['initial_capital']-1:7.1%}     {best_annual_return:7.1%}      {best_annual_return-(baseline_results['final_capital']/baseline_results['initial_capital']-1):+6.1%}")
print(f"Max Drawdown:        {baseline_results['equity_curve'].expanding().max().sub(baseline_results['equity_curve']).div(baseline_results['equity_curve'].expanding().max()).min():7.1%}     {best_max_dd:7.1%}      {best_max_dd-(baseline_results['equity_curve'].expanding().max().sub(baseline_results['equity_curve']).div(baseline_results['equity_curve'].expanding().max()).min()):+6.1%}")
print(f"Win Rate:            {(baseline_results['trades']['pnl'] > 0).sum() / len(baseline_results['trades']):7.1%}     {best_win_rate:7.1%}      {best_win_rate - ((baseline_results['trades']['pnl'] > 0).sum() / len(baseline_results['trades'])):+6.1%}")
print(f"Total Trades:        {baseline_results['n_trades']:7d}     {best_results['n_trades']:7d}      {best_results['n_trades']-baseline_results['n_trades']:+6d}")

sharpe_improvement = ((best_sharpe / baseline_sharpe) - 1) * 100

if best_sharpe > baseline_sharpe:
    print(f"\n  [SUCCESS] Optimalizacia zlepsila Sharpe o {sharpe_improvement:.1f}%!")
elif best_sharpe > baseline_sharpe * 0.95:
    print(f"\n  [OK] Podobny vysledok ako baseline")
else:
    print(f"\n  [WARNING] Optimalizacia nezlepsila vysledok")


# ================================================================
# 9. SAVE RESULTS
# ================================================================

print("\n[UKLADANIE] Ukladam vysledky...")

# Save directory
save_dir = Path('results/optimized_full_dataset')
save_dir.mkdir(parents=True, exist_ok=True)

# Save model
joblib.dump(best_model, 'models/xgboost_optimized_full_dataset.pkl')

# Save Optuna study
study_path = save_dir / 'optuna_study.pkl'
joblib.dump(study, study_path)

# Save optimization history
trials_df = study.trials_dataframe()
trials_df.to_csv(save_dir / 'optimization_history.csv', index=False)

# Save metrics
metrics = {
    'created_at': datetime.now().isoformat(),
    'dataset': 'full_dataset_2020_2025.csv',
    'optimization': {
        'n_trials': 50,
        'best_trial': study.best_trial.number,
        'best_cv_sharpe': float(study.best_value)
    },
    'baseline': {
        'sharpe': float(baseline_sharpe),
        'parameters': baseline_params
    },
    'optimized': {
        'sharpe': float(best_sharpe),
        'annual_return': float(best_annual_return),
        'total_return': float(best_total_return),
        'max_drawdown': float(best_max_dd),
        'win_rate': float(best_win_rate),
        'total_trades': int(best_results['n_trades']),
        'final_capital': float(best_results['final_capital']),
        'parameters': best_params
    },
    'improvement': {
        'sharpe_diff': float(best_sharpe - baseline_sharpe),
        'sharpe_pct': float(sharpe_improvement)
    }
}

with open(save_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save trades
best_trades_df.to_csv(save_dir / 'trades.csv', index=False)

# Save equity curve
equity_df = pd.DataFrame({
    'date': test_data.index,
    'baseline_equity': baseline_results['equity_curve'].values,
    'optimized_equity': best_equity.values
})
equity_df.to_csv(save_dir / 'equity_curve.csv', index=False)

# Save predictions
pred_df = pd.DataFrame({
    'date': test_data.index,
    'actual': y_test.values,
    'baseline_pred': baseline_pred,
    'optimized_pred': best_pred
}, index=test_data.index)
pred_df.to_csv(save_dir / 'predictions.csv')

print(f"\n  [OK] Ulozene:")
print(f"    - models/xgboost_optimized_full_dataset.pkl")
print(f"    - results/optimized_full_dataset/metrics.json")
print(f"    - results/optimized_full_dataset/optimization_history.csv")
print(f"    - results/optimized_full_dataset/optuna_study.pkl")
print(f"    - results/optimized_full_dataset/trades.csv")
print(f"    - results/optimized_full_dataset/equity_curve.csv")
print(f"    - results/optimized_full_dataset/predictions.csv")

print("\n" + "="*80)
print("HOTOVO!")
print("="*80)

print(f"\nSUMMARY:")
print(f"  Trials:           50")
print(f"  Baseline Sharpe:  {baseline_sharpe:.2f}")
print(f"  Optimized Sharpe: {best_sharpe:.2f}")
print(f"  Improvement:      {sharpe_improvement:+.1f}%")

if best_sharpe > baseline_sharpe:
    print(f"\n  VYSLEDOK: Optimalizacia uspesna! Sharpe {baseline_sharpe:.2f} -> {best_sharpe:.2f}")
else:
    print(f"\n  VYSLEDOK: Baseline je stale najlepsi. Mozeme skusit viac trials.")

print("="*80)
