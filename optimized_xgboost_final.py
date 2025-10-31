"""
OPTIMIZED XGBOOST - FINAL VERSION
- 100 trials (instead of 200) to prevent crashes
- Better progress reporting
- Saves checkpoints every 20 trials
- Cleaner output
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import optuna
import joblib
import json
import warnings
import sys
warnings.filterwarnings('ignore')

print("="*80)
print("OPTIMIZED XGBOOST - FINAL VERSION")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("100 trials with checkpointing")
print("="*80)


# ================================================================
# 1. LOAD DATA
# ================================================================

print("\n[1/6] Loading data...")

train = pd.read_csv('data/train_advanced.csv', index_col=0, parse_dates=True)
test = pd.read_csv('data/test_advanced.csv', index_col=0, parse_dates=True)

exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_classification']
feature_cols = [col for col in train.columns if col not in exclude_cols]

X_train = train[feature_cols]
y_train = train['target']
X_test = test[feature_cols]
y_test = test['target']

print(f"  Train: {len(X_train)} samples, {len(feature_cols)} features")
print(f"  Test:  {len(X_test)} samples")


# ================================================================
# 2. SETUP CV
# ================================================================

print("\n[2/6] Setting up 3-fold CV...")
tscv = TimeSeriesSplit(n_splits=3)


# ================================================================
# 3. OBJECTIVE FUNCTION
# ================================================================

print("\n[3/6] Defining objective function...")

def objective(trial):
    """Correlation-based objective with minimal regularization"""

    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
        'n_estimators': trial.suggest_int('n_estimators', 80, 150),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 2),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.1),
    }

    cv_correlations = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        try:
            model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            val_pred = model.predict(X_val)

            # Check for constant predictions
            if val_pred.std() < 1e-8:
                corr = -0.1  # Penalty
            else:
                corr, _ = spearmanr(y_val, val_pred)
                if np.isnan(corr):
                    corr = 0

            cv_correlations.append(corr)
        except Exception as e:
            # If model training fails, return bad score
            cv_correlations.append(-0.1)

    avg_correlation = np.mean(cv_correlations)

    trial.set_user_attr('cv_correlations', cv_correlations)
    trial.set_user_attr('cv_std', np.std(cv_correlations))

    return avg_correlation


# ================================================================
# 4. RUN OPTIMIZATION
# ================================================================

print("\n[4/6] Running optimization (100 trials)...")
print("  Progress updates every 10 trials")
print("  Checkpoints saved every 20 trials")
print()

optuna.logging.set_verbosity(optuna.logging.ERROR)

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
)

trial_count = [0]
best_so_far = [-999]

def callback(study, trial):
    trial_count[0] += 1

    # Update best
    if trial.value > best_so_far[0]:
        best_so_far[0] = trial.value

    # Progress every 10 trials
    if trial_count[0] % 10 == 0:
        elapsed = (datetime.now() - start_time).total_seconds()
        trials_per_sec = trial_count[0] / elapsed
        remaining_trials = 100 - trial_count[0]
        eta_seconds = remaining_trials / trials_per_sec if trials_per_sec > 0 else 0
        eta_minutes = eta_seconds / 60

        print(f"  [{trial_count[0]:3d}/100] Best: {best_so_far[0]:.4f} | "
              f"Current: {trial.value:.4f} | ETA: {eta_minutes:.1f}m")
        sys.stdout.flush()

    # Checkpoint every 20 trials
    if trial_count[0] % 20 == 0:
        joblib.dump(study, f'models/checkpoint_trial_{trial_count[0]}.pkl')

start_time = datetime.now()

try:
    study.optimize(objective, n_trials=100, callbacks=[callback], show_progress_bar=False)
    print(f"\n  [OK] Optimization completed!")
except Exception as e:
    print(f"\n  [WARNING] Optimization stopped early: {e}")
    print(f"  Completed {trial_count[0]}/100 trials")

print(f"  Best correlation: {study.best_value:.4f}")


# ================================================================
# 5. TRAIN FINAL MODEL
# ================================================================

print("\n[5/6] Training final model with best parameters...")

best_params = study.best_params
best_trial = study.best_trial

print(f"\n  Best parameters:")
for param, value in best_params.items():
    print(f"    {param:20s} = {value}")

cv_corrs = best_trial.user_attrs['cv_correlations']
cv_std = best_trial.user_attrs['cv_std']

print(f"\n  CV Correlations:")
for i, corr in enumerate(cv_corrs, 1):
    print(f"    Fold {i}: {corr:.4f}")
print(f"    Mean:   {np.mean(cv_corrs):.4f}")
print(f"    Std:    {cv_std:.4f}")

# Train final model
final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
final_model.fit(X_train, y_train)

# Test predictions
test_predictions = final_model.predict(X_test)

# Calculate test correlation
test_corr, _ = spearmanr(y_test, test_predictions)

print(f"\n  Test correlation: {test_corr:.4f}")


# ================================================================
# 6. BACKTEST
# ================================================================

print("\n[6/6] Backtesting...")

from backtester import Backtester
backtester = Backtester(
    initial_capital=10000,
    transaction_cost=0.001,
    holding_period=5,
    position_size_pct=0.5,
    prediction_threshold=0.001
)

results = backtester.run_backtest(test_predictions, y_test, test['Close'])

returns = results['returns']
if len(returns) > 0 and returns.std() > 0:
    test_sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
else:
    test_sharpe = 0

equity_curve = results['equity_curve']
rolling_max = equity_curve.expanding().max()
drawdowns = (equity_curve - rolling_max) / rolling_max
max_dd = drawdowns.min()

total_return = (results['final_capital'] / results['initial_capital']) - 1
annual_return = (1 + total_return) ** (252 / len(test)) - 1

trades_df = results['trades']
win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) if len(trades_df) > 0 else 0

print(f"\n  Results:")
print(f"    Sharpe Ratio:     {test_sharpe:.2f}")
print(f"    Annual Return:    {annual_return*100:.2f}%")
print(f"    Max Drawdown:     {max_dd*100:.2f}%")
print(f"    Win Rate:         {win_rate*100:.1f}%")
print(f"    Total Trades:     {results['n_trades']}")

# Compare with baseline
baseline_sharpe = 1.28
improvement = ((test_sharpe / baseline_sharpe) - 1) * 100

print(f"\n  Comparison:")
print(f"    Baseline:         {baseline_sharpe:.2f}")
print(f"    Optimized:        {test_sharpe:.2f}")
print(f"    Improvement:      {improvement:+.1f}%")

if improvement > 10:
    status = "SUCCESS"
elif improvement > 5:
    status = "GOOD"
elif improvement > 0:
    status = "MARGINAL"
else:
    status = "NO_IMPROVEMENT"

print(f"    Status:           {status}")


# ================================================================
# 7. SAVE
# ================================================================

print("\n[SAVING] Saving results...")

joblib.dump(final_model, 'models/xgboost_optimized_final.pkl')
joblib.dump(study, 'models/xgboost_study_final.pkl')

# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

importance_df.to_csv('models/xgboost_feature_importance_final.csv', index=False)

# Metadata
metadata = {
    'model': 'XGBoost',
    'version': 'optimized_final',
    'created_at': datetime.now().isoformat(),
    'optimization': {
        'n_trials': trial_count[0],
        'n_folds': 3,
        'best_cv_correlation': float(study.best_value)
    },
    'best_params': best_params,
    'cv_performance': {
        'mean_correlation': float(np.mean(cv_corrs)),
        'std_correlation': float(cv_std),
        'fold_correlations': [float(c) for c in cv_corrs]
    },
    'test_performance': {
        'correlation': float(test_corr),
        'sharpe': float(test_sharpe),
        'annual_return': float(annual_return),
        'max_drawdown': float(max_dd),
        'win_rate': float(win_rate),
        'num_trades': int(results['n_trades'])
    },
    'comparison': {
        'baseline_sharpe': float(baseline_sharpe),
        'improvement_pct': float(improvement),
        'status': status
    }
}

with open('models/xgboost_optimized_final_results.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("  [OK] Saved:")
print("    - models/xgboost_optimized_final.pkl")
print("    - models/xgboost_study_final.pkl")
print("    - models/xgboost_optimized_final_results.json")
print("    - models/xgboost_feature_importance_final.csv")


# ================================================================
# SUMMARY
# ================================================================

print(f"\n{'='*80}")
print("OPTIMIZATION COMPLETE")
print("="*80)

print(f"\n  Summary:")
print(f"    Trials completed:    {trial_count[0]}/100")
print(f"    Best CV correlation: {study.best_value:.4f}")
print(f"    Test Sharpe:         {test_sharpe:.2f}")
print(f"    Improvement:         {improvement:+.1f}%")
print(f"    Status:              {status}")

if status in ["SUCCESS", "GOOD"]:
    print(f"\n  [OK] Great! Optimization significantly improved performance!")
elif status == "MARGINAL":
    print(f"\n  [OK] Marginal improvement. Consider using optimized model.")
else:
    print(f"\n  Baseline is still better. Keep using Test #5 baseline.")

print(f"\n  Top 5 features:")
for i, row in importance_df.head(5).iterrows():
    print(f"    {i+1}. {row['feature']:30s} {row['importance']:.4f}")

print("="*80)
