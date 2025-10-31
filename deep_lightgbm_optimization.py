"""
DEEP LIGHTGBM OPTIMIZATION
500 trials with 5-fold cross-validation
Expected time: ~1.5 hours (faster than XGBoost)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import optuna
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEEP LIGHTGBM OPTIMIZATION - 500 TRIALS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Expected duration: ~1.5 hours")
print("="*80)


# ================================================================
# LOAD DATA & SETUP (same as XGBoost)
# ================================================================

print("\n[1/6] Loading data...")

train = pd.read_csv('data/train_advanced.csv', index_col=0, parse_dates=True)
test = pd.read_csv('data/test_advanced.csv', index_col=0, parse_dates=True)

exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains']
feature_cols = [col for col in train.columns if col not in exclude_cols]

X_train = train[feature_cols]
y_train = train['target']
X_test = test[feature_cols]
y_test = test['target']

print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")
print(f"  Features: {len(feature_cols)}")

tscv = TimeSeriesSplit(n_splits=5)


# ================================================================
# OBJECTIVE FUNCTION
# ================================================================

print("\n[3/6] Defining optimization objective...")

def objective(trial):
    """LightGBM-specific parameters"""

    params = {
        # Tree structure
        'num_leaves': trial.suggest_int('num_leaves', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10, log=True),

        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),

        # Sampling
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),

        # Regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10, log=True),

        # LightGBM specific
        'min_split_gain': trial.suggest_float('min_split_gain', 0, 5),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
    }

    cv_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=-1)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        val_pred = model.predict(X_val)
        val_returns = y_val * np.sign(val_pred)

        if val_returns.std() > 0:
            sharpe = (val_returns.mean() / val_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        cv_scores.append(sharpe)

    trial.set_user_attr('cv_scores', cv_scores)
    trial.set_user_attr('cv_std', np.std(cv_scores))

    return np.mean(cv_scores)


# ================================================================
# RUN OPTIMIZATION
# ================================================================

print("\n[4/6] Running 500 trials optimization...")

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
)

def callback(study, trial):
    if trial.number % 50 == 0:
        print(f"  Trial {trial.number}/500 - Best: {study.best_value:.3f}, Current: {trial.value:.3f}")

study.optimize(objective, n_trials=500, callbacks=[callback], show_progress_bar=True)

print(f"\n✅ Optimization complete! Best Sharpe: {study.best_value:.3f}")


# ================================================================
# TRAIN & TEST (same structure as XGBoost)
# ================================================================

print("\n[6/6] Training final model and testing...")

best_params = study.best_params
final_model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1, n_jobs=-1)
final_model.fit(X_train, y_train)

test_predictions = final_model.predict(X_test)

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
test_sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

equity_curve = results['equity_curve']
rolling_max = equity_curve.expanding().max()
drawdowns = (equity_curve - rolling_max) / rolling_max
max_dd = drawdowns.min()

total_return = (results['final_capital'] / results['initial_capital']) - 1
annual_return = (1 + total_return) ** (252 / len(test)) - 1

trades_df = results['trades']
win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) if len(trades_df) > 0 else 0

baseline_sharpe = 1.28
improvement = ((test_sharpe / baseline_sharpe) - 1) * 100

print(f"\n{'='*80}")
print(f"LIGHTGBM RESULTS")
print(f"{'='*80}")
print(f"  Test Sharpe:      {test_sharpe:.2f}")
print(f"  Annual Return:    {annual_return*100:.2f}%")
print(f"  Max Drawdown:     {max_dd*100:.2f}%")
print(f"  Win Rate:         {win_rate*100:.1f}%")
print(f"  Improvement:      {improvement:+.1f}%")

# Save
joblib.dump(final_model, 'models/lightgbm_deep_optimized.pkl')
joblib.dump(study, 'models/lightgbm_study.pkl')

metadata = {
    'model': 'LightGBM',
    'best_params': best_params,
    'test_performance': {
        'sharpe': float(test_sharpe),
        'annual_return': float(annual_return),
        'max_drawdown': float(max_dd),
        'win_rate': float(win_rate),
        'improvement_vs_baseline': float(improvement)
    }
}

with open('models/lightgbm_deep_results.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ LightGBM optimization complete!")
