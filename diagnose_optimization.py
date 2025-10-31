"""
Diagnose why XGBoost optimization is returning 0.0 correlations
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr

print("="*70)
print("DIAGNOSING OPTIMIZATION FAILURE")
print("="*70)

# Load data
train = pd.read_csv('data/train_advanced.csv', index_col=0, parse_dates=True)

exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_classification']
feature_cols = [col for col in train.columns if col not in exclude_cols]

X_train = train[feature_cols]
y_train = train['target']

print(f"\nData shape: {X_train.shape}")
print(f"Target stats:")
print(f"  Mean: {y_train.mean():.6f}")
print(f"  Std:  {y_train.std():.6f}")
print(f"  Min:  {y_train.min():.6f}")
print(f"  Max:  {y_train.max():.6f}")

# Test a few parameter combinations
tscv = TimeSeriesSplit(n_splits=3)

test_params = [
    # Conservative (from optimization)
    {
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 80,
        'min_child_weight': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
    },
    # Middle ground
    {
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
    },
    # More aggressive
    {
        'max_depth': 6,
        'learning_rate': 0.15,
        'n_estimators': 150,
        'min_child_weight': 8,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'gamma': 0.3,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
    }
]

for idx, params in enumerate(test_params, 1):
    print(f"\n{'='*70}")
    print(f"TEST {idx}/3: {params}")
    print(f"{'='*70}")

    cv_correlations = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        # Train model
        model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        # Predict
        val_pred = model.predict(X_val)

        # Check prediction statistics
        print(f"\n  Fold {fold_idx}:")
        print(f"    Val size:      {len(X_val)}")
        print(f"    Pred mean:     {val_pred.mean():.6f}")
        print(f"    Pred std:      {val_pred.std():.6f}")
        print(f"    Pred min:      {val_pred.min():.6f}")
        print(f"    Pred max:      {val_pred.max():.6f}")
        print(f"    Pred unique:   {len(np.unique(val_pred))}")

        print(f"    Target mean:   {y_val.mean():.6f}")
        print(f"    Target std:    {y_val.std():.6f}")

        # Check if predictions have variance
        if val_pred.std() < 1e-8:
            print(f"    WARNING: Predictions are constant!")
            corr = 0.0
        else:
            corr, pval = spearmanr(y_val, val_pred)
            if np.isnan(corr):
                print(f"    WARNING: Correlation is NaN!")
                corr = 0.0

        print(f"    Correlation:   {corr:.4f}")
        cv_correlations.append(corr)

    avg_corr = np.mean(cv_correlations)
    print(f"\n  Average correlation: {avg_corr:.4f}")

    if avg_corr < 0.01:
        print(f"  PROBLEM: This parameter set produces near-zero correlations!")
    elif avg_corr < 0.1:
        print(f"  WEAK: Very low predictive power")
    else:
        print(f"  OK: Reasonable correlation")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
print("\nNext step: Check if any parameter set works well")
