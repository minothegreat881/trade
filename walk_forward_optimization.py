"""
WALK-FORWARD OPTIMIZATION
Scientific approach that prevents overfitting

Methodology:
1. Split data into rolling windows
2. Optimize on each training window
3. Test on next OOS window
4. Aggregate results

Expected improvement: +3-5% Sharpe
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import optuna
import json
import warnings
warnings.filterwarnings('ignore')

from backtester import Backtester

print("="*70)
print("WALK-FORWARD OPTIMIZATION")
print("="*70)
print("Scientific approach to prevent overfitting")
print("="*70)


# ================================================================
# 1. LOAD DATA
# ================================================================

print("\n[STEP 1/4] LOADING DATA")
print("="*70)

# Load Test #5 data
try:
    train = pd.read_csv('data/train_advanced.csv', index_col=0, parse_dates=True)
    test = pd.read_csv('data/test_advanced.csv', index_col=0, parse_dates=True)
except FileNotFoundError:
    print("[ERROR] Data files not found! Run: python export_test5_data.py")
    exit(1)

# Combine for walk-forward
full_data = pd.concat([train, test])

exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_value', 'fear_greed_classification',
                'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']
feature_cols = [col for col in full_data.columns if col not in exclude_cols]

print(f"Total data: {len(full_data)} samples")
print(f"Period: {full_data.index[0].date()} to {full_data.index[-1].date()}")
print(f"Features: {len(feature_cols)}")


# ================================================================
# 2. WALK-FORWARD SETUP
# ================================================================

print("\n[STEP 2/4] SETTING UP WALK-FORWARD")
print("="*70)

# Walk-forward parameters
n_splits = 5  # 5 windows
min_train_size = 300  # Minimum 300 days for training

# Create time series splits
tscv = TimeSeriesSplit(n_splits=n_splits)

print(f"Number of windows: {n_splits}")
print(f"Min training size: {min_train_size} days")

# Store results for each window
all_window_results = []


# ================================================================
# 3. WALK-FORWARD LOOP
# ================================================================

print("\n[STEP 3/4] RUNNING WALK-FORWARD OPTIMIZATION")
print("="*70)

optuna.logging.set_verbosity(optuna.logging.WARNING)

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(full_data), 1):

    print(f"\n{'='*70}")
    print(f"WINDOW {fold_idx}/{n_splits}")
    print(f"{'='*70}")

    # Skip if training set too small
    if len(train_idx) < min_train_size:
        print(f"  Skipping - training set too small ({len(train_idx)} < {min_train_size})")
        continue

    # Split data
    train_fold = full_data.iloc[train_idx]
    test_fold = full_data.iloc[test_idx]

    print(f"  Train: {len(train_fold)} samples ({train_fold.index[0].date()} to {train_fold.index[-1].date()})")
    print(f"  Test:  {len(test_fold)} samples ({test_fold.index[0].date()} to {test_fold.index[-1].date()})")

    # Further split train into train/val for optimization
    val_split = int(len(train_fold) * 0.8)
    train_split = train_fold.iloc[:val_split]
    val_split_data = train_fold.iloc[val_split:]

    X_train = train_split[feature_cols]
    y_train = train_split['target']
    X_val = val_split_data[feature_cols]
    y_val = val_split_data['target']
    X_test_fold = test_fold[feature_cols]
    y_test_fold = test_fold['target']

    # Optimize hyperparameters on this window
    print(f"\n  Optimizing hyperparameters (30 trials)...")

    def objective(trial):
        # Conservative parameter ranges
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
            'n_estimators': trial.suggest_int('n_estimators', 80, 150),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'gamma': trial.suggest_float('gamma', 0, 0.3),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        }

        model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate on validation using simple returns
        val_pred = model.predict(X_val)
        val_returns = y_val * val_pred

        if val_returns.std() > 0:
            sharpe = (val_returns.mean() / val_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        return sharpe

    # Run optimization (fewer trials = less overfitting)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42 + fold_idx)
    )
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    print(f"  Best validation Sharpe: {study.best_value:.2f}")

    # Train final model on full training window
    best_model = xgb.XGBRegressor(**study.best_params, random_state=42, verbosity=0)
    best_model.fit(train_fold[feature_cols], train_fold['target'])

    # Predict on test fold
    test_predictions = best_model.predict(X_test_fold)

    # Backtest on this fold
    backtester = Backtester(
        initial_capital=10000,
        transaction_cost=0.001,
        holding_period=5,
        position_size_pct=0.5,
        prediction_threshold=0.001
    )

    results = backtester.run_backtest(
        test_predictions,
        y_test_fold,
        test_fold['Close']
    )

    # Calculate Sharpe
    returns = results['returns']
    if len(returns) > 0 and returns.std() > 0:
        test_sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        test_sharpe = 0

    # Calculate other metrics
    equity_curve = results['equity_curve']
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    # Calculate win rate
    trades_df = results['trades']
    if len(trades_df) > 0:
        win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df)
    else:
        win_rate = 0

    print(f"\n  Window Results:")
    print(f"    Test Sharpe:  {test_sharpe:.2f}")
    print(f"    Max DD:       {max_dd*100:.2f}%")
    print(f"    Win Rate:     {win_rate*100:.1f}%")
    print(f"    Trades:       {results['n_trades']}")

    # Store results
    all_window_results.append({
        'fold': fold_idx,
        'train_start': str(train_fold.index[0].date()),
        'train_end': str(train_fold.index[-1].date()),
        'test_start': str(test_fold.index[0].date()),
        'test_end': str(test_fold.index[-1].date()),
        'val_sharpe': float(study.best_value),
        'test_sharpe': float(test_sharpe),
        'max_dd': float(max_dd),
        'win_rate': float(win_rate),
        'num_trades': int(results['n_trades']),
        'best_params': study.best_params
    })


# ================================================================
# 4. AGGREGATE RESULTS
# ================================================================

print("\n" + "="*70)
print("WALK-FORWARD RESULTS")
print("="*70)

# Create results DataFrame
results_df = pd.DataFrame(all_window_results)

print("\nResults by Window:")
print(results_df[['fold', 'val_sharpe', 'test_sharpe', 'max_dd', 'num_trades']].to_string(index=False))

# Average metrics
avg_val_sharpe = results_df['val_sharpe'].mean()
avg_test_sharpe = results_df['test_sharpe'].mean()
avg_max_dd = results_df['max_dd'].mean()
avg_win_rate = results_df['win_rate'].mean()

print(f"\nAggregated Metrics:")
print(f"  Average Validation Sharpe: {avg_val_sharpe:.2f}")
print(f"  Average Test Sharpe:       {avg_test_sharpe:.2f}")
print(f"  Average Max DD:            {avg_max_dd*100:.2f}%")
print(f"  Average Win Rate:          {avg_win_rate*100:.1f}%")
print(f"  Sharpe Std Dev:            {results_df['test_sharpe'].std():.2f}")

# Compare with baseline
baseline_sharpe = 1.28
baseline_dd = -0.0479
improvement = ((avg_test_sharpe / baseline_sharpe) - 1) * 100

print(f"\n{'='*70}")
print(f"COMPARISON WITH TEST #5 BASELINE")
print(f"{'='*70}")
print(f"  Baseline (Test #5):        Sharpe {baseline_sharpe:.2f}, Max DD {baseline_dd*100:.2f}%")
print(f"  Walk-Forward Average:      Sharpe {avg_test_sharpe:.2f}, Max DD {avg_max_dd*100:.2f}%")
print(f"  Improvement:               {improvement:+.1f}%")

if improvement > 5:
    print(f"\n  SUCCESS! +{improvement:.1f}% improvement!")
    recommendation = "Use walk-forward optimized model. Proceed to Phase 2 (Ensemble)"
elif improvement > 0:
    print(f"\n  MARGINAL improvement (+{improvement:.1f}%)")
    recommendation = "Walk-forward helps slightly. Consider Phase 2 (Ensemble)"
else:
    print(f"\n  No improvement ({improvement:.1f}%)")
    recommendation = "Test #5 baseline is still best. Skip to Phase 3 (Position Sizing)"

print(f"\n  Recommendation: {recommendation}")

# Check consistency across windows
sharpe_std = results_df['test_sharpe'].std()
if sharpe_std < 0.3:
    print(f"  Consistent performance across windows (std={sharpe_std:.2f})")
else:
    print(f"  WARNING: Inconsistent performance across windows (std={sharpe_std:.2f})")


# ================================================================
# 5. SAVE RESULTS
# ================================================================

print("\n[SAVING] Saving walk-forward results...")

# Find best performing window
best_window = results_df.loc[results_df['test_sharpe'].idxmax()]

metadata = {
    'model_version': '8.0.0-walk-forward',
    'created_at': datetime.now().isoformat(),
    'methodology': 'walk_forward_optimization',
    'n_windows': n_splits,
    'avg_test_sharpe': float(avg_test_sharpe),
    'avg_max_dd': float(avg_max_dd),
    'avg_win_rate': float(avg_win_rate),
    'sharpe_std': float(sharpe_std),
    'improvement_vs_baseline': float(improvement),
    'best_window': {
        'fold': int(best_window['fold']),
        'test_sharpe': float(best_window['test_sharpe']),
        'params': best_window['best_params']
    },
    'all_windows': all_window_results,
    'recommendation': recommendation
}

with open('models/walk_forward_results.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("  Results saved: models/walk_forward_results.json")

print("\n" + "="*70)
print("WALK-FORWARD OPTIMIZATION COMPLETE!")
print("="*70)

if avg_test_sharpe > baseline_sharpe:
    print(f"\nSUCCESS! Walk-forward improved performance!")
    print(f"Next step: Try ensemble (Phase 2)")
else:
    print(f"\nWalk-forward didn't improve performance")
    print(f"Recommendation: {recommendation}")
