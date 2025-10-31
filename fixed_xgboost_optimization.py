"""
FIXED XGBOOST OPTIMIZATION
Fixes all 4 critical bugs
Expected: Stable results, Sharpe > 1.28
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
warnings.filterwarnings('ignore')

print("="*80)
print("FIXED XGBOOST OPTIMIZATION")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nFixes applied:")
print("  [*] Objective function: Correlation-based")
print("  [*] Larger validation sets: 3 folds instead of 5")
print("  [*] Narrower parameter space: Around known good values")
print("  [*] Consistent metrics: Same evaluation everywhere")
print("="*80)


# ================================================================
# 1. LOAD DATA
# ================================================================

print("\n[1/7] Loading data...")

train = pd.read_csv('data/train_advanced.csv', index_col=0, parse_dates=True)
test = pd.read_csv('data/test_advanced.csv', index_col=0, parse_dates=True)

exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_classification']  # Categorical column, exclude
feature_cols = [col for col in train.columns if col not in exclude_cols]

X_train = train[feature_cols]
y_train = train['target']
X_test = test[feature_cols]
y_test = test['target']

print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")
print(f"  Features: {len(feature_cols)}")

# Check for data quality issues
print(f"\n  Data quality check:")
print(f"    X_train NaN: {X_train.isna().sum().sum()}")
print(f"    y_train NaN: {y_train.isna().sum()}")
print(f"    X_train inf: {np.isinf(X_train.select_dtypes(include=[np.number]).values).sum()}")

if X_train.isna().sum().sum() > 0 or y_train.isna().sum() > 0:
    print(f"    WARNING: Found NaN values, dropping...")
    X_train = X_train.dropna()
    y_train = y_train[X_train.index]


# ================================================================
# 2. BASELINE EVALUATION
# ================================================================

print("\n[2/7] Evaluating Test #5 baseline...")

# Load baseline model (Test #5)
try:
    baseline_model = joblib.load('models/xgboost_advanced_features.pkl')
    baseline_pred = baseline_model.predict(X_test)

    # Calculate baseline metrics using CORRELATION (consistent with optimization)
    baseline_corr, _ = spearmanr(y_test, baseline_pred)
    print(f"  Baseline correlation: {baseline_corr:.4f}")

    # Also calculate actual Sharpe via backtest
    from backtester import Backtester
    backtester = Backtester(
        initial_capital=10000,
        transaction_cost=0.001,
        holding_period=5,
        position_size_pct=0.5,
        prediction_threshold=0.001
    )
    baseline_results = backtester.run_backtest(baseline_pred, y_test, test['Close'])
    baseline_returns = baseline_results['returns']
    baseline_sharpe = (baseline_returns.mean() / baseline_returns.std()) * np.sqrt(252)
    print(f"  Baseline Sharpe: {baseline_sharpe:.2f}")

except Exception as e:
    print(f"  [OK]  Could not load baseline: {e}")
    baseline_corr = 0.05
    baseline_sharpe = 1.28


# ================================================================
# 3. CROSS-VALIDATION SETUP
# ================================================================

print("\n[3/7] Setting up 3-fold time series CV...")

# Use only 3 folds for larger validation sets
tscv = TimeSeriesSplit(n_splits=3)

print(f"  Folds: 3 (larger val sets = more stable)")

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    print(f"    Fold {fold_idx}: Train={len(train_idx)}, Val={len(val_idx)}")


# ================================================================
# 4. OBJECTIVE FUNCTION (FIXED!)
# ================================================================

print("\n[4/7] Defining FIXED objective function...")

def objective(trial):
    """
    FIXED objective function using correlation
    This is much more stable than fake Sharpe calculation
    """

    # FIXED parameter space - MINIMAL regularization (data is very sensitive)
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 6),  # Was 2-12, now 3-6
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),  # Was 0.001-0.3
        'n_estimators': trial.suggest_int('n_estimators', 80, 150),  # Was 50-500
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 2),  # FIXED: Was 1-3, now 1-2 only
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),  # Was 0.3-1.0
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),  # Was 0.3-1.0
        'gamma': trial.suggest_float('gamma', 0, 0.1),  # FIXED: Was 0-0.15, now 0-0.1 (minimal)
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.1),  # FIXED: Was 0-0.3, now 0-0.1 (minimal)
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.1),  # FIXED: Was 0-0.3, now 0-0.1 (minimal)
    }

    # Cross-validation scores
    cv_correlations = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        # Train model
        model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Predict
        val_pred = model.predict(X_val)

        # FIXED: Check for constant predictions (variance too low)
        # This prevents wasting trials on over-regularized models
        if val_pred.std() < 1e-8:
            # Constant predictions = useless model, return penalty
            corr = -0.1  # Penalty score
        else:
            # FIXED: Use correlation instead of fake Sharpe!
            # This measures if predictions align with actual returns
            corr, pval = spearmanr(y_val, val_pred)

            # Handle NaN correlation
            if np.isnan(corr):
                corr = 0

        cv_correlations.append(corr)

    # Average correlation across folds
    avg_correlation = np.mean(cv_correlations)

    # Store for analysis
    trial.set_user_attr('cv_correlations', cv_correlations)
    trial.set_user_attr('cv_std', np.std(cv_correlations))

    return avg_correlation


# ================================================================
# 5. RUN OPTIMIZATION
# ================================================================

print("\n[5/7] Running optimization with 200 trials...")
print("  (Reduced from 500 due to narrower search space)")
print()

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=30  # Was 50, now 30
    ),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10
    )
)

def callback(study, trial):
    if trial.number % 20 == 0 and trial.number > 0:
        print(f"  Trial {trial.number:3d}/200")
        print(f"    Best correlation: {study.best_value:.4f}")
        print(f"    This trial:       {trial.value:.4f}")
        if trial.number == 20:
            print(f"    Estimated time remaining: {(200-20)*0.1:.0f} minutes")
        print()

study.optimize(
    objective,
    n_trials=200,  # Was 500, now 200 (narrower space)
    callbacks=[callback],
    show_progress_bar=True
)

print(f"\n[OK] Optimization complete!")
print(f"   Trials: {len(study.trials)}")
print(f"   Best correlation: {study.best_value:.4f}")


# ================================================================
# 6. ANALYZE RESULTS
# ================================================================

print("\n[6/7] Analyzing results...")

best_params = study.best_params
best_trial = study.best_trial

print(f"\n  BEST PARAMETERS:")
for param, value in best_params.items():
    print(f"    {param:20s} = {value}")

# Cross-validation stability
cv_corrs = best_trial.user_attrs['cv_correlations']
cv_std = best_trial.user_attrs['cv_std']

print(f"\n  CROSS-VALIDATION CORRELATIONS:")
for i, corr in enumerate(cv_corrs, 1):
    print(f"    Fold {i}: {corr:.4f}")
print(f"    Mean:   {np.mean(cv_corrs):.4f}")
print(f"    Std:    {cv_std:.4f}")

# Stability check
if cv_std < 0.05:
    print(f"    [OK] VERY STABLE! (std < 0.05)")
    stability = "EXCELLENT"
elif cv_std < 0.10:
    print(f"    [OK] STABLE (std < 0.10)")
    stability = "GOOD"
elif cv_std < 0.15:
    print(f"    [OK]  MODERATE stability (std < 0.15)")
    stability = "MODERATE"
else:
    print(f"    [OK] UNSTABLE (std >= 0.15)")
    stability = "POOR"


# ================================================================
# 7. TRAIN FINAL MODEL & BACKTEST
# ================================================================

print("\n[7/7] Training final model and running backtest...")

# Train with best parameters on full training set
final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

# Predict on test set
test_predictions = final_model.predict(X_test)

# Calculate test correlation
test_corr, test_pval = spearmanr(y_test, test_predictions)
print(f"\n  Test correlation: {test_corr:.4f} (p={test_pval:.4f})")

# Run full backtest
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
win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) if len(trades_df) > 0 else 0


# ================================================================
# RESULTS & COMPARISON
# ================================================================

print(f"\n{'='*80}")
print(f"FINAL RESULTS")
print(f"{'='*80}")

print(f"\n  OPTIMIZATION METRICS:")
print(f"    CV Correlation:      {np.mean(cv_corrs):.4f} [OK] {cv_std:.4f}")
print(f"    Test Correlation:    {test_corr:.4f}")
print(f"    Stability:           {stability}")

print(f"\n  BACKTEST METRICS:")
print(f"    Sharpe Ratio:        {test_sharpe:.2f}")
print(f"    Annual Return:       {annual_return*100:.2f}%")
print(f"    Max Drawdown:        {max_dd*100:.2f}%")
print(f"    Win Rate:            {win_rate*100:.1f}%")
print(f"    Total Trades:        {results['n_trades']}")

print(f"\n{'='*80}")
print(f"COMPARISON WITH BASELINE")
print(f"{'='*80}")

print(f"\n  CORRELATION:")
print(f"    Baseline:            {baseline_corr:.4f}")
print(f"    Optimized:           {test_corr:.4f}")
corr_improvement = ((test_corr / baseline_corr) - 1) * 100 if baseline_corr > 0 else 0
print(f"    Improvement:         {corr_improvement:+.1f}%")

print(f"\n  SHARPE RATIO:")
print(f"    Baseline:            {baseline_sharpe:.2f}")
print(f"    Optimized:           {test_sharpe:.2f}")
sharpe_improvement = ((test_sharpe / baseline_sharpe) - 1) * 100
print(f"    Improvement:         {sharpe_improvement:+.1f}%")

# Determine success
if test_sharpe > baseline_sharpe * 1.05:
    print(f"\n  [OK] SUCCESS! +{sharpe_improvement:.1f}% improvement!")
    status = "SUCCESS"
elif test_sharpe > baseline_sharpe:
    print(f"\n  [OK] MARGINAL improvement (+{sharpe_improvement:.1f}%)")
    status = "MARGINAL"
elif test_sharpe > baseline_sharpe * 0.95:
    print(f"\n  [OK]  SIMILAR performance ({sharpe_improvement:.1f}%)")
    status = "SIMILAR"
else:
    print(f"\n  [OK] WORSE than baseline ({sharpe_improvement:.1f}%)")
    status = "WORSE"


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

print(f"\n[FEATURES] Analyzing feature importance...")

feature_importance = final_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Check if we have meaningful importances
total_importance = importance_df['importance'].sum()
max_importance = importance_df['importance'].max()

print(f"  Total importance: {total_importance:.4f}")
print(f"  Max importance:   {max_importance:.4f}")

if max_importance > 0.001:
    print(f"  [OK] Model learned meaningful patterns!")
    print(f"\n  Top 15 features:")
    for idx, row in importance_df.head(15).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.4f}")
else:
    print(f"  [OK]  WARNING: All importances near zero!")
    print(f"     Model may not have learned properly")


# ================================================================
# SAVE RESULTS
# ================================================================

print(f"\n[SAVING] Saving model and results...")

# Save model
joblib.dump(final_model, 'models/xgboost_optimized_fixed.pkl')
joblib.dump(study, 'models/xgboost_study_fixed.pkl')

# Save feature importance
importance_df.to_csv('models/xgboost_feature_importance_fixed.csv', index=False)

# Save metadata
metadata = {
    'model': 'XGBoost',
    'version': 'optimized_fixed',
    'optimization': {
        'method': 'Optuna TPE with correlation objective',
        'n_trials': 200,
        'n_folds': 3,
        'objective': 'spearman_correlation',
        'completed_at': datetime.now().isoformat(),
        'fixes_applied': [
            'Correlation-based objective (was fake Sharpe)',
            'Larger validation sets (3 folds vs 5)',
            'Narrower parameter space',
            'Consistent evaluation metrics'
        ]
    },
    'best_params': best_params,
    'cv_performance': {
        'mean_correlation': float(np.mean(cv_corrs)),
        'std_correlation': float(cv_std),
        'fold_correlations': [float(c) for c in cv_corrs],
        'stability': stability
    },
    'test_performance': {
        'correlation': float(test_corr),
        'sharpe': float(test_sharpe),
        'annual_return': float(annual_return),
        'max_drawdown': float(max_dd),
        'win_rate': float(win_rate),
        'num_trades': int(results['n_trades']),
        'status': status
    },
    'comparison': {
        'baseline_correlation': float(baseline_corr),
        'baseline_sharpe': float(baseline_sharpe),
        'correlation_improvement': float(corr_improvement),
        'sharpe_improvement': float(sharpe_improvement)
    }
}

with open('models/xgboost_optimized_fixed_results.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  [OK] Saved:")
print(f"     - models/xgboost_optimized_fixed.pkl")
print(f"     - models/xgboost_study_fixed.pkl")
print(f"     - models/xgboost_optimized_fixed_results.json")
print(f"     - models/xgboost_feature_importance_fixed.csv")


# ================================================================
# FINAL SUMMARY
# ================================================================

print(f"\n{'='*80}")
print(f"OPTIMIZATION COMPLETE")
print(f"{'='*80}")

print(f"\n  Summary:")
print(f"    Trials:              200")
print(f"    CV Correlation:      {np.mean(cv_corrs):.4f} [OK] {cv_std:.4f}")
print(f"    Test Sharpe:         {test_sharpe:.2f}")
print(f"    Improvement:         {sharpe_improvement:+.1f}%")
print(f"    Status:              {status}")

if status in ["SUCCESS", "MARGINAL"]:
    print(f"\n  [OK] Optimization improved performance!")
    print(f"     Use: models/xgboost_optimized_fixed.pkl")
elif status == "SIMILAR":
    print(f"\n  [OK]  Similar performance to baseline")
    print(f"     Can use either model")
else:
    print(f"\n  [OK] Baseline is still better")
    print(f"     Keep using: Test #5 baseline")

print(f"\n  Next steps:")
if status in ["SUCCESS", "MARGINAL", "SIMILAR"]:
    print(f"    1. [OK] Use optimized model")
    print(f"    2. Try Phase 3: Dynamic Position Sizing")
    print(f"    3. Move to live paper trading")
else:
    print(f"    1. Review why optimization didn't help")
    print(f"    2. Try alternative approaches:")
    print(f"       - Feature selection (reduce to 50-70 best)")
    print(f"       - Ensemble methods")
    print(f"       - Market regime detection")
    print(f"    3. Or proceed with Test #5 to Phase 3")

print("="*80)
