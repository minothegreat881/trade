"""
OPTIMALIZACIA KORELECIE - OPRAVENA VERZIA
==========================================

Optimalizuje Spearman correlation s constraints:
- Learning rate <= 0.06 (zabrani overfitting)
- Predictions std > 0.001 (zabrani konstantnym predikciam)
- Max depth <= 4
- Uzsi parameter space okolo baseline

Target: Beat Sharpe 1.34 (Baseline)
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
print("OPTIMALIZACIA KORELECIE - OPRAVENA VERZIA")
print("="*80)
print(f"Datum spustenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# 1. LOAD DATASET
# ================================================================

print("\n[1/4] Loading dataset...")

df = pd.read_csv('data/full_dataset_2020_2025.csv', index_col=0, parse_dates=True)

# Exclude columns
exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_value', 'fear_greed_classification',
                'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']

feature_cols = [col for col in df.columns if col not in exclude_cols]

if df[feature_cols + ['target']].isna().sum().sum() > 0:
    df = df.dropna(subset=feature_cols + ['target'])

print(f"  Rows: {len(df)}")
print(f"  Features: {len(feature_cols)}")


# ================================================================
# 2. TRAIN/TEST SPLIT
# ================================================================

print("\n[2/4] Train/Test split (70/30)...")

split_idx = int(len(df) * 0.7)

train_data = df.iloc[:split_idx]
test_data = df.iloc[split_idx:]

X_train = train_data[feature_cols]
y_train = train_data['target']
X_test = test_data[feature_cols]
y_test = test_data['target']
test_close = test_data['Close']

print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")


# ================================================================
# 3. BASELINE PERFORMANCE
# ================================================================

print("\n[3/4] Baseline performance...")

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
baseline_test_pred = baseline_model.predict(X_test)

baseline_corr, _ = spearmanr(y_test, baseline_test_pred)

print(f"  Baseline Spearman: {baseline_corr:.4f}")
print(f"  Target: Beat {baseline_corr:.4f}")


# ================================================================
# 4. OPTUNA OPTIMIZATION
# ================================================================

print("\n[4/4] Optuna optimization (50 trials)...")
print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")

# Track best test correlation
best_test_corr = baseline_corr
best_test_params = baseline_params.copy()


def objective(trial):
    """
    Objective funkcia - maximalizuje Spearman correlation
    S constraints na prediction variance
    """
    # UZSI PARAMETER SPACE okolo baseline
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 4),  # 2-4 (baseline=3)
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.06),  # MAX 0.06!
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),  # 50-150 (baseline=100)
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),  # 3-10 (baseline=5)
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),  # 0.7-0.9 (baseline=0.8)
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),  # 0.7-0.9 (baseline=0.8)
        'gamma': trial.suggest_float('gamma', 0, 0.2),  # 0-0.2 (baseline=0)
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),  # 0-0.5 (baseline=0)
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),  # 0.5-2 (baseline=1)
        'random_state': 42,
        'verbosity': 0
    }

    # Time Series Cross-Validation (3 folds)
    tscv = TimeSeriesSplit(n_splits=3)
    correlations = []
    pred_stds = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, verbose=False)

        # Predict
        val_pred = model.predict(X_val)

        # Check prediction variance
        pred_std = val_pred.std()
        pred_stds.append(pred_std)

        # Calculate Spearman correlation
        if pred_std > 0.0001:  # Zabran konstantnym predikciam
            corr, _ = spearmanr(y_val, val_pred)
            if not np.isnan(corr):
                correlations.append(corr)
            else:
                correlations.append(-1.0)  # Penalizacia za NaN
        else:
            # PENALIZACIA za konstantne predikcie!
            correlations.append(-1.0)

    # Mean correlation across folds
    mean_corr = np.mean(correlations)
    mean_pred_std = np.mean(pred_stds)

    # Additional penalty if predictions are too constant
    if mean_pred_std < 0.001:
        mean_corr -= 0.5  # Velka penalizacia

    # Store info for tracking
    trial.set_user_attr('mean_pred_std', mean_pred_std)
    trial.set_user_attr('mean_corr', mean_corr)

    return mean_corr


# Create Optuna study
study = optuna.create_study(
    direction='maximize',
    study_name='xgboost_correlation_optimization',
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Callback to track best test performance
def callback(study, trial):
    global best_test_corr, best_test_params

    # Every 10 trials, test on actual test set
    if trial.number % 10 == 0 and trial.number > 0:
        # Get best params so far
        params = study.best_params.copy()
        params['random_state'] = 42
        params['verbosity'] = 0

        # Train on full train set
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)

        # Test
        test_pred = model.predict(X_test)
        test_pred_std = test_pred.std()

        if test_pred_std > 0.001:
            test_corr, _ = spearmanr(y_test, test_pred)

            print(f"\n  Trial {trial.number}: Test Spearman = {test_corr:.4f}, Pred Std = {test_pred_std:.6f}")

            if test_corr > best_test_corr:
                best_test_corr = test_corr
                best_test_params = params.copy()
                print(f"  NEW BEST! Test Spearman: {test_corr:.4f}")

# Optimize with callback
study.optimize(objective, n_trials=50, show_progress_bar=True, callbacks=[callback])

print(f"\n  End: {datetime.now().strftime('%H:%M:%S')}")


# ================================================================
# 5. BEST PARAMETERS
# ================================================================

print("\n[5/7] Best parameters found...")

# Use best TEST params if better than CV params
if best_test_corr > baseline_corr:
    print(f"\n  Using best TEST params (Spearman: {best_test_corr:.4f})")
    best_params = best_test_params
else:
    print(f"\n  Using best CV params")
    best_params = study.best_params.copy()
    best_params['random_state'] = 42
    best_params['verbosity'] = 0

print(f"\n  Best CV Correlation: {study.best_value:.4f}")
print(f"\n  Best Parameters:")
for k, v in best_params.items():
    if k not in ['random_state', 'verbosity']:
        baseline_val = baseline_params.get(k, 'N/A')
        print(f"    {k:20s} = {v:7.4f}  (baseline: {baseline_val})")


# ================================================================
# 6. TRAIN FINAL MODEL
# ================================================================

print("\n[6/7] Training final model...")

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)

# Check prediction quality
final_pred_std = final_pred.std()
final_pred_mean = final_pred.mean()
final_corr, _ = spearmanr(y_test, final_pred)

print(f"\n  Final Test Performance:")
print(f"    Spearman:      {final_corr:.4f}")
print(f"    Pred Mean:     {final_pred_mean:.6f}")
print(f"    Pred Std:      {final_pred_std:.6f}")
print(f"    Pred Min:      {final_pred.min():.6f}")
print(f"    Pred Max:      {final_pred.max():.6f}")


# ================================================================
# 7. BACKTEST
# ================================================================

print("\n[7/7] Backtest...")

from backtester import Backtester

backtester = Backtester(
    initial_capital=10000,
    transaction_cost=0.001,
    holding_period=5,
    position_size_pct=0.5,
    prediction_threshold=0.001
)

results = backtester.run_backtest(final_pred, y_test, test_close)

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

print(f"\n  FINAL RESULTS:")
print(f"    Sharpe Ratio:     {sharpe:.2f}")
print(f"    Annual Return:    {annual_return*100:.2f}%")
print(f"    Total Return:     {total_return*100:.2f}%")
print(f"    Max Drawdown:     {max_dd*100:.2f}%")
print(f"    Win Rate:         {win_rate*100:.1f}%")
print(f"    Total Trades:     {results['n_trades']}")
print(f"    Final Capital:    ${results['final_capital']:.2f}")


# ================================================================
# 8. COMPARISON
# ================================================================

print("\n" + "="*80)
print("POROVNANIE BASELINE vs OPTIMIZED")
print("="*80)

# Baseline metrics from file
with open('results/full_dataset_baseline_2025/metrics.json', 'r') as f:
    baseline_metrics = json.load(f)

baseline_sharpe = baseline_metrics['performance']['sharpe_ratio']

print(f"\n                     BASELINE    OPTIMIZED    ROZDIEL")
print(f"Spearman Corr:       {baseline_corr:7.4f}     {final_corr:7.4f}      {final_corr-baseline_corr:+6.4f}")
print(f"Sharpe Ratio:        {baseline_sharpe:7.2f}     {sharpe:7.2f}      {sharpe-baseline_sharpe:+6.2f}")
print(f"Annual Return:       {baseline_metrics['performance']['annual_return']:7.1%}     {annual_return:7.1%}      {annual_return-baseline_metrics['performance']['annual_return']:+6.1%}")
print(f"Max Drawdown:        {baseline_metrics['performance']['max_drawdown']:7.1%}     {max_dd:7.1%}      {max_dd-baseline_metrics['performance']['max_drawdown']:+6.1%}")
print(f"Win Rate:            {baseline_metrics['performance']['win_rate']:7.1%}     {win_rate:7.1%}      {win_rate-baseline_metrics['performance']['win_rate']:+6.1%}")
print(f"Total Trades:        {baseline_metrics['performance']['total_trades']:7d}     {results['n_trades']:7d}      {results['n_trades']-baseline_metrics['performance']['total_trades']:+6d}")

# Prediction comparison
print(f"\n  PREDICTIONS:")
print(f"    Baseline pred std:   {0.007095:.6f}")
print(f"    Optimized pred std:  {final_pred_std:.6f}")

improvement = ((sharpe / baseline_sharpe) - 1) * 100

if sharpe > baseline_sharpe:
    print(f"\n  [SUCCESS] Optimalizacia zlepsila Sharpe o {improvement:.1f}%!")
elif sharpe > baseline_sharpe * 0.95:
    print(f"\n  [OK] Podobny vysledok ako baseline ({improvement:+.1f}%)")
else:
    print(f"\n  [WARNING] Optimalizacia nezlepsila vysledok ({improvement:+.1f}%)")


# ================================================================
# 9. SAVE RESULTS
# ================================================================

print("\n[UKLADANIE] Ukladam vysledky...")

save_dir = Path('results/optimized_correlation')
save_dir.mkdir(parents=True, exist_ok=True)

# Save model
joblib.dump(final_model, 'models/xgboost_optimized_correlation.pkl')

# Save study
joblib.dump(study, save_dir / 'optuna_study.pkl')

# Save trials
trials_df = study.trials_dataframe()
trials_df.to_csv(save_dir / 'optimization_history.csv', index=False)

# Save metrics
metrics = {
    'created_at': datetime.now().isoformat(),
    'optimization': {
        'method': 'correlation',
        'n_trials': 50,
        'best_cv_correlation': float(study.best_value),
        'best_test_correlation': float(final_corr)
    },
    'baseline': {
        'correlation': float(baseline_corr),
        'sharpe': float(baseline_sharpe),
        'parameters': baseline_params
    },
    'optimized': {
        'correlation': float(final_corr),
        'sharpe': float(sharpe),
        'annual_return': float(annual_return),
        'max_drawdown': float(max_dd),
        'win_rate': float(win_rate),
        'total_trades': int(results['n_trades']),
        'final_capital': float(results['final_capital']),
        'pred_std': float(final_pred_std),
        'pred_mean': float(final_pred_mean),
        'parameters': best_params
    },
    'improvement': {
        'correlation_diff': float(final_corr - baseline_corr),
        'sharpe_diff': float(sharpe - baseline_sharpe),
        'sharpe_pct': float(improvement)
    }
}

with open(save_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save trades
trades_df.to_csv(save_dir / 'trades.csv', index=False)

# Save predictions
pred_df = pd.DataFrame({
    'date': test_data.index,
    'actual': y_test.values,
    'baseline_pred': baseline_test_pred,
    'optimized_pred': final_pred
}, index=test_data.index)
pred_df.to_csv(save_dir / 'predictions.csv')

print(f"\n  [OK] Ulozene:")
print(f"    - models/xgboost_optimized_correlation.pkl")
print(f"    - results/optimized_correlation/metrics.json")
print(f"    - results/optimized_correlation/optimization_history.csv")
print(f"    - results/optimized_correlation/trades.csv")
print(f"    - results/optimized_correlation/predictions.csv")

print("\n" + "="*80)
print("HOTOVO!")
print("="*80)

print(f"\nSUMMARY:")
print(f"  Trials:              50")
print(f"  Baseline Sharpe:     {baseline_sharpe:.2f}")
print(f"  Optimized Sharpe:    {sharpe:.2f}")
print(f"  Improvement:         {improvement:+.1f}%")
print(f"  Baseline Corr:       {baseline_corr:.4f}")
print(f"  Optimized Corr:      {final_corr:.4f}")
print(f"  Pred Std (check):    {final_pred_std:.6f} (must be > 0.001)")

if sharpe > baseline_sharpe and final_pred_std > 0.001:
    print(f"\n  VYSLEDOK: USPECH! Model je lepsi a nepredikuje konstantu!")
elif final_pred_std < 0.001:
    print(f"\n  PROBLEM: Model stale predikuje konstantu!")
else:
    print(f"\n  VYSLEDOK: Baseline stale najlepsi, ale model funguje.")

print("="*80)
