"""
Test if the FIXED parameter ranges prevent constant predictions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr

print("="*70)
print("TESTING FIXED PARAMETER RANGES")
print("="*70)

# Load data
train = pd.read_csv('data/train_advanced.csv', index_col=0, parse_dates=True)

exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_classification']
feature_cols = [col for col in train.columns if col not in exclude_cols]

X_train = train[feature_cols]
y_train = train['target']

tscv = TimeSeriesSplit(n_splits=3)

# Test FIXED parameter ranges (same as in optimization)
test_params = [
    # Low regularization (should work)
    {
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 80,
        'min_child_weight': 1,  # FIXED: Was 3, now 1
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
    },
    # Medium regularization (should work now)
    {
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'min_child_weight': 3,  # FIXED: Was 4, now 3 (max)
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.2,  # FIXED: Was 0.3, now 0.2
        'reg_lambda': 0.2,  # FIXED: Was 0.3, now 0.2
    },
    # High regularization at boundary (should work)
    {
        'max_depth': 6,
        'learning_rate': 0.15,
        'n_estimators': 150,
        'min_child_weight': 3,  # FIXED: Max is now 3
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'gamma': 0.15,  # FIXED: Max is now 0.15
        'reg_alpha': 0.3,  # FIXED: Max is now 0.3
        'reg_lambda': 0.3,  # FIXED: Max is now 0.3
    }
]

constant_count = 0
working_count = 0

for idx, params in enumerate(test_params, 1):
    print(f"\n{'='*70}")
    print(f"TEST {idx}/3")
    print(f"{'='*70}")

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        val_pred = model.predict(X_val)

        # Check if constant
        is_constant = val_pred.std() < 1e-8

        if is_constant:
            corr = -0.1  # Penalty
            fold_results.append("CONSTANT")
        else:
            corr, _ = spearmanr(y_val, val_pred)
            if np.isnan(corr):
                corr = 0
            fold_results.append(f"Corr={corr:.4f}")

        print(f"  Fold {fold_idx}: {fold_results[-1]} (std={val_pred.std():.6f})")

    # Summary for this parameter set
    if "CONSTANT" in fold_results:
        print(f"\n  RESULT: CONSTANT PREDICTIONS DETECTED")
        constant_count += 1
    else:
        print(f"\n  RESULT: WORKING (non-constant predictions)")
        working_count += 1

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Working parameter sets: {working_count}/3")
print(f"Constant prediction sets: {constant_count}/3")

if constant_count == 0:
    print("\n[OK] All parameter sets produce non-constant predictions!")
    print("The optimization should work now.")
else:
    print(f"\n[WARNING] {constant_count} parameter sets still produce constant predictions")
    print("Need to adjust ranges further.")
