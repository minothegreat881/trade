"""
DEEP XGBOOST OPTIMIZATION
500 trials with 5-fold cross-validation
Expected time: ~2 hours
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEEP XGBOOST OPTIMIZATION - 500 TRIALS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Expected duration: ~2 hours")
print("="*80)


# ================================================================
# 1. LOAD DATA
# ================================================================

print("\n[1/6] Loading data...")

train = pd.read_csv('data/train_advanced.csv', index_col=0, parse_dates=True)
test = pd.read_csv('data/test_advanced.csv', index_col=0, parse_dates=True)

exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_value', 'fear_greed_classification',
                'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']
feature_cols = [col for col in train.columns if col not in exclude_cols]

X_train = train[feature_cols]
y_train = train['target']
X_test = test[feature_cols]
y_test = test['target']

print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")
print(f"  Features: {len(feature_cols)}")


# ================================================================
# 2. CROSS-VALIDATION SETUP
# ================================================================

print("\n[2/6] Setting up 5-fold time series cross-validation...")

# 5-fold time series split
tscv = TimeSeriesSplit(n_splits=5)

print(f"  Folds: 5")
print(f"  Method: Time Series Split (no data leakage)")


# ================================================================
# 3. DEFINE OBJECTIVE FUNCTION
# ================================================================

print("\n[3/6] Defining optimization objective...")

def objective(trial):
    """
    Optuna objective for XGBoost
    Wide parameter space for thorough search
    """

    # Sample hyperparameters - WIDE RANGE
    params = {
        # Tree structure
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 5),

        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),

        # Sampling
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),

        # Regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

        # Other
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
    }

    # Cross-validation scores
    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        # Train model
        model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Predict on validation
        val_pred = model.predict(X_val)

        # Calculate Sharpe on validation set
        val_returns = y_val * np.sign(val_pred)  # Direction-based returns

        if val_returns.std() > 0:
            sharpe = (val_returns.mean() / val_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        cv_scores.append(sharpe)

    # Return average Sharpe across all folds
    avg_sharpe = np.mean(cv_scores)

    # Store fold details
    trial.set_user_attr('cv_scores', cv_scores)
    trial.set_user_attr('cv_std', np.std(cv_scores))

    return avg_sharpe


# ================================================================
# 4. RUN OPTIMIZATION
# ================================================================

print("\n[4/6] Running 500 trials optimization...")
print("  This will take approximately 2 hours")
print("  Progress will be shown every 50 trials")
print()

# Create study
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=50,  # Random search first 50 trials
        n_ei_candidates=50    # Then Bayesian optimization
    ),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_min_trials=5)
)

# Callback to show progress
trial_count = [0]

def callback(study, trial):
    trial_count[0] += 1
    if trial_count[0] % 50 == 0:
        print(f"  Trial {trial_count[0]}/500")
        print(f"    Current best: {study.best_value:.3f}")
        print(f"    This trial: {trial.value:.3f}")
        print()

# Run optimization
study.optimize(
    objective,
    n_trials=500,
    callbacks=[callback],
    show_progress_bar=True
)

print(f"\nOptimization complete!")
print(f"   Total trials: {len(study.trials)}")
print(f"   Best Sharpe: {study.best_value:.3f}")


# ================================================================
# 5. ANALYZE RESULTS
# ================================================================

print("\n[5/6] Analyzing results...")

# Best parameters
best_params = study.best_params
best_trial = study.best_trial

print(f"\n  BEST PARAMETERS:")
for param, value in best_params.items():
    print(f"    {param:20s} = {value}")

# Best trial details
cv_scores = best_trial.user_attrs['cv_scores']
cv_std = best_trial.user_attrs['cv_std']

print(f"\n  CROSS-VALIDATION SCORES:")
for i, score in enumerate(cv_scores, 1):
    print(f"    Fold {i}: {score:.3f}")
print(f"    Mean:   {np.mean(cv_scores):.3f}")
print(f"    Std:    {cv_std:.3f}")

# Check for stability
if cv_std < 0.3:
    print(f"    Stable across folds!")
elif cv_std < 0.5:
    print(f"    Moderate stability")
else:
    print(f"    WARNING: Unstable - high variance!")


# ================================================================
# 6. TRAIN FINAL MODEL & TEST
# ================================================================

print("\n[6/6] Training final model on full training set...")

# Train with best parameters
final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
final_model.fit(X_train, y_train)

# Predict on test set
test_predictions = final_model.predict(X_test)

# Backtest
from backtester import Backtester
backtester = Backtester(
    initial_capital=10000,
    transaction_cost=0.001,
    holding_period=5,
    position_size_pct=0.5,
    prediction_threshold=0.001
)

results = backtester.run_backtest(test_predictions, y_test, test['Close'])

# Calculate metrics
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
if len(trades_df) > 0:
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df)
else:
    win_rate = 0

print(f"\n{'='*80}")
print(f"TEST SET PERFORMANCE")
print(f"{'='*80}")
print(f"  Sharpe Ratio:     {test_sharpe:.2f}")
print(f"  Annual Return:    {annual_return*100:.2f}%")
print(f"  Max Drawdown:     {max_dd*100:.2f}%")
print(f"  Win Rate:         {win_rate*100:.1f}%")
print(f"  Total Trades:     {results['n_trades']}")

# Compare with baseline
baseline_sharpe = 1.28
improvement = ((test_sharpe / baseline_sharpe) - 1) * 100

print(f"\n{'='*80}")
print(f"COMPARISON WITH BASELINE")
print(f"{'='*80}")
print(f"  Baseline (Test #5):    {baseline_sharpe:.2f}")
print(f"  Optimized XGBoost:     {test_sharpe:.2f}")
print(f"  Improvement:           {improvement:+.1f}%")

if improvement > 10:
    print(f"\n  EXCELLENT! {improvement:.1f}% improvement!")
    status = "SUCCESS"
elif improvement > 5:
    print(f"\n  GOOD! {improvement:.1f}% improvement")
    status = "GOOD"
elif improvement > 0:
    print(f"\n  MARGINAL {improvement:.1f}% improvement")
    status = "MARGINAL"
else:
    print(f"\n  NO IMPROVEMENT ({improvement:.1f}%)")
    status = "FAILED"


# ================================================================
# 7. SAVE RESULTS
# ================================================================

print("\n[SAVING] Saving model and results...")

# Save model
joblib.dump(final_model, 'models/xgboost_deep_optimized.pkl')

# Save study
joblib.dump(study, 'models/xgboost_study.pkl')

# Save metadata
metadata = {
    'model': 'XGBoost',
    'optimization': {
        'method': 'Optuna TPE',
        'n_trials': 500,
        'n_folds': 5,
        'best_trial': best_trial.number,
        'completed_at': datetime.now().isoformat()
    },
    'best_params': best_params,
    'cv_performance': {
        'mean_sharpe': float(np.mean(cv_scores)),
        'std_sharpe': float(cv_std),
        'fold_scores': [float(s) for s in cv_scores]
    },
    'test_performance': {
        'sharpe': float(test_sharpe),
        'annual_return': float(annual_return),
        'max_drawdown': float(max_dd),
        'win_rate': float(win_rate),
        'num_trades': int(results['n_trades']),
        'improvement_vs_baseline': float(improvement),
        'status': status
    }
}

with open('models/xgboost_deep_results.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  Model saved: models/xgboost_deep_optimized.pkl")
print(f"  Study saved: models/xgboost_study.pkl")
print(f"  Results saved: models/xgboost_deep_results.json")


# ================================================================
# 8. VISUALIZATIONS
# ================================================================

print("\n[VISUALIZING] Creating optimization plots...")

try:
    # Plot optimization history
    fig1 = plot_optimization_history(study)
    fig1.write_html('models/xgboost_optimization_history.html')
    print(f"  History plot: models/xgboost_optimization_history.html")

    # Plot parameter importances
    fig2 = plot_param_importances(study)
    fig2.write_html('models/xgboost_param_importance.html')
    print(f"  Param importance: models/xgboost_param_importance.html")

except Exception as e:
    print(f"  Visualization failed: {e}")


# ================================================================
# 9. FEATURE IMPORTANCE
# ================================================================

print("\n[ANALYZING] Feature importances...")

feature_importance = final_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n  Top 15 features:")
for idx, row in importance_df.head(15).iterrows():
    print(f"    {row['feature']:30s} {row['importance']:.4f}")

# Save feature importance
importance_df.to_csv('models/xgboost_feature_importance.csv', index=False)
print(f"\n  Feature importance saved: models/xgboost_feature_importance.csv")


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "="*80)
print("XGBOOST DEEP OPTIMIZATION - COMPLETE")
print("="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nResults:")
print(f"  Trials completed:     500")
print(f"  CV Sharpe:           {np.mean(cv_scores):.2f} +/- {cv_std:.2f}")
print(f"  Test Sharpe:         {test_sharpe:.2f}")
print(f"  Status:              {status}")
print(f"\nFiles created:")
print(f"  - models/xgboost_deep_optimized.pkl")
print(f"  - models/xgboost_study.pkl")
print(f"  - models/xgboost_deep_results.json")
print(f"  - models/xgboost_feature_importance.csv")
print(f"  - models/xgboost_optimization_history.html")
print(f"  - models/xgboost_param_importance.html")
print("="*80)

if test_sharpe > baseline_sharpe:
    print(f"\nXGBoost optimization SUCCESSFUL!")
    print(f"Improved from {baseline_sharpe:.2f} to {test_sharpe:.2f}")
else:
    print(f"\nXGBoost optimization did not improve baseline")
    print(f"Baseline {baseline_sharpe:.2f} still better than {test_sharpe:.2f}")
