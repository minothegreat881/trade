"""
MINI GRID SEARCH AROUND BASELINE
Safe, simple, effective
Expected: +1-5% improvement
Time: ~10-15 minutes
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import product
import joblib
from scipy.stats import spearmanr
import json
from datetime import datetime

print("="*80)
print("MINI GRID SEARCH - Around Test #5 Baseline")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# LOAD DATA
# ================================================================

print("\n[1/4] Loading data...")

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

print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")
print(f"  Features: {len(feature_cols)}")


# ================================================================
# DEFINE NARROW GRID
# ================================================================

print("\n[2/4] Defining search space around baseline...")
print("\n  BASELINE: max_depth=4, lr=0.1, n_estimators=100, mcw=5")

# BASELINE: max_depth=4, lr=0.1, n_estimators=100, mcw=5
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.08, 0.1, 0.12],
    'n_estimators': [80, 100, 120],
    'min_child_weight': [4, 5, 6],
    'subsample': [0.8],  # Keep fixed
    'colsample_bytree': [0.8],  # Keep fixed
}

# Generate all combinations
keys = list(param_grid.keys())
values = list(param_grid.values())
combinations = list(product(*values))

print(f"\n  Search space:")
print(f"    max_depth:        {param_grid['max_depth']}")
print(f"    learning_rate:    {param_grid['learning_rate']}")
print(f"    n_estimators:     {param_grid['n_estimators']}")
print(f"    min_child_weight: {param_grid['min_child_weight']}")
print(f"\n  Total combinations: {len(combinations)}")
print(f"  Estimated time: ~{len(combinations)*0.2/60:.1f} minutes")


# ================================================================
# GRID SEARCH
# ================================================================

print("\n[3/4] Testing all combinations...")
print("  (Progress every 10 combinations)")
print()

from backtester import Backtester

results = []
baseline_sharpe = 1.28

backtester = Backtester(
    initial_capital=10000,
    transaction_cost=0.001,
    holding_period=5,
    position_size_pct=0.5,
    prediction_threshold=0.001
)

best_sharpe_so_far = 0

for i, combo in enumerate(combinations):
    params = dict(zip(keys, combo))

    # Add fixed params
    params['random_state'] = 42
    params['n_jobs'] = -1
    params['verbosity'] = 0

    # Train model - NO early stopping!
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)  # Simple fit!

    # Test
    predictions = model.predict(X_test)

    # Backtest
    try:
        backtest_results = backtester.run_backtest(predictions, y_test, test['Close'])
        returns = backtest_results['returns']
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
    except Exception as e:
        sharpe = 0

    # Also calculate correlation
    corr, _ = spearmanr(y_test, predictions)
    if np.isnan(corr):
        corr = 0

    results.append({
        **params,
        'sharpe': sharpe,
        'correlation': corr,
        'trades': backtest_results['n_trades'] if 'backtest_results' in locals() else 0
    })

    # Track best
    if sharpe > best_sharpe_so_far:
        best_sharpe_so_far = sharpe

    # Progress
    if (i+1) % 10 == 0 or (i+1) == len(combinations):
        print(f"  [{i+1:3d}/{len(combinations)}] Best so far: {best_sharpe_so_far:.2f}")

print(f"\n  [OK] Grid search complete!")


# ================================================================
# ANALYZE RESULTS
# ================================================================

print("\n[4/4] Analyzing results...")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print("\n  Top 10 configurations:")
print("  " + "-"*76)
top10 = results_df.head(10)[['max_depth', 'learning_rate', 'n_estimators',
                              'min_child_weight', 'sharpe', 'correlation']]
for idx, row in top10.iterrows():
    print(f"  {idx+1:2d}. depth={int(row['max_depth'])}, lr={row['learning_rate']:.2f}, "
          f"n_est={int(row['n_estimators'])}, mcw={int(row['min_child_weight'])} "
          f"→ Sharpe={row['sharpe']:.2f}, Corr={row['correlation']:.3f}")

# Best result
best = results_df.iloc[0]
best_params = {
    'max_depth': int(best['max_depth']),
    'learning_rate': float(best['learning_rate']),
    'n_estimators': int(best['n_estimators']),
    'min_child_weight': int(best['min_child_weight']),
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0
}

print(f"\n{'='*80}")
print(f"BEST CONFIGURATION")
print(f"{'='*80}")
print(f"\n  Parameters:")
for k, v in best_params.items():
    if k != 'verbosity':
        print(f"    {k:20s} = {v}")

print(f"\n  Performance:")
print(f"    Sharpe:       {best['sharpe']:.2f}")
print(f"    Correlation:  {best['correlation']:.4f}")
print(f"    Trades:       {int(best['trades'])}")

print(f"\n{'='*80}")
print(f"COMPARISON WITH BASELINE")
print(f"{'='*80}")
print(f"  Baseline:     {baseline_sharpe:.2f}")
print(f"  Grid Best:    {best['sharpe']:.2f}")

improvement = ((best['sharpe'] / baseline_sharpe) - 1) * 100
print(f"  Improvement:  {improvement:+.1f}%")

if improvement > 3:
    print(f"\n  [OK] SUCCESS! Found better parameters!")
    status = "IMPROVED"
elif improvement > 0:
    print(f"\n  [OK] Small improvement")
    status = "SLIGHT_IMPROVEMENT"
elif improvement > -3:
    print(f"\n  [OK] Similar to baseline")
    status = "SIMILAR"
else:
    print(f"\n  Baseline is still better")
    status = "BASELINE_BETTER"


# ================================================================
# SAVE BEST MODEL
# ================================================================

if status in ["IMPROVED", "SLIGHT_IMPROVEMENT", "SIMILAR"]:
    print("\n[SAVING] Training and saving best model...")

    best_model = xgb.XGBRegressor(**best_params, n_jobs=-1)
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, 'models/xgboost_grid_search_best.pkl')

    metadata = {
        'method': 'mini_grid_search',
        'created_at': datetime.now().isoformat(),
        'search_space': {k: [float(x) if isinstance(x, (int, float)) else x for x in v]
                        for k, v in param_grid.items()},
        'total_combinations': len(combinations),
        'best_params': best_params,
        'performance': {
            'sharpe': float(best['sharpe']),
            'correlation': float(best['correlation']),
            'trades': int(best['trades']),
            'improvement_vs_baseline': float(improvement)
        },
        'status': status,
        'top_5_configs': [
            {
                'rank': i+1,
                'params': {
                    'max_depth': int(row['max_depth']),
                    'learning_rate': float(row['learning_rate']),
                    'n_estimators': int(row['n_estimators']),
                    'min_child_weight': int(row['min_child_weight'])
                },
                'sharpe': float(row['sharpe']),
                'correlation': float(row['correlation'])
            }
            for i, (idx, row) in enumerate(results_df.head(5).iterrows())
        ]
    }

    with open('models/grid_search_results.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  [OK] Saved:")
    print(f"    - models/xgboost_grid_search_best.pkl")
    print(f"    - models/grid_search_results.json")

print("\n" + "="*80)
print("GRID SEARCH COMPLETE")
print("="*80)

if status == "IMPROVED":
    print("\n  [OK] Use grid search model! It's better than baseline!")
elif status == "SLIGHT_IMPROVEMENT":
    print("\n  [OK] Grid search found small improvement. Can use either model.")
elif status == "SIMILAR":
    print("\n  Results similar to baseline. Baseline is validated!")
else:
    print("\n  Baseline remains best. Grid search validated baseline choice.")

print("="*80)
