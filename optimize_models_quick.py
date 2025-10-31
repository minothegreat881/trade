"""
QUICK MODEL OPTIMIZATION - 20 trials per model
Tests 5 different models + hyperparameter tuning

Expected: Sharpe 1.28 → 1.35-1.45
Runtime: ~15-30 minutes
"""

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Models
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False
    print("[WARNING] LightGBM not installed, skipping...")

try:
    import catboost as cb
    HAS_CATBOOST = True
except:
    HAS_CATBOOST = False
    print("[WARNING] CatBoost not installed, skipping...")

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except:
    HAS_OPTUNA = False
    print("[ERROR] Optuna not installed! Install with: pip install optuna")
    exit(1)

# Custom modules
from backtester import Backtester

print("="*70)
print("QUICK MODEL OPTIMIZATION FRAMEWORK")
print("="*70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Strategy: Test models + hyperparameter tuning (20 trials each)")
print("="*70)


# ================================================================
# LOAD DATA
# ================================================================

print("\n[STEP 1/7] LOADING DATA")
print("="*70)

try:
    train = pd.read_csv('data/train_advanced.csv', index_col=0, parse_dates=True)
    test = pd.read_csv('data/test_advanced.csv', index_col=0, parse_dates=True)
except FileNotFoundError:
    print("[ERROR] Data files not found!")
    print("Run first: python export_test5_data.py")
    exit(1)

# Separate features and target
exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_value', 'fear_greed_classification',
                'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']
feature_cols = [col for col in train.columns if col not in exclude_cols]

X_train_full = train[feature_cols]
y_train_full = train['target']
X_test = test[feature_cols]
y_test = test['target']
test_prices = test['Close']

# Split train into train/val
val_split = int(len(X_train_full) * 0.8)
X_train = X_train_full.iloc[:val_split]
y_train = y_train_full.iloc[:val_split]
X_val = X_train_full.iloc[val_split:]
y_val = y_train_full.iloc[val_split:]

print(f"Train: {len(X_train)} samples")
print(f"Val:   {len(X_val)} samples")
print(f"Test:  {len(X_test)} samples")
print(f"Features: {len(feature_cols)}")


# ================================================================
# EVALUATION FUNCTION
# ================================================================

def evaluate_model(predictions, y_test, test_prices):
    """Evaluate predictions using backtesting"""
    backtester = Backtester(
        initial_capital=10000,
        transaction_cost=0.001,
        holding_period=5,
        position_size_pct=0.5,
        prediction_threshold=0.001
    )

    results = backtester.run_backtest(predictions, y_test, test_prices)

    # Calculate Sharpe
    returns = results['returns']
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Calculate max drawdown
    equity_curve = results['equity_curve']
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    return {
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': results['n_trades'],
        'results': results
    }


# ================================================================
# MODEL 1: XGBOOST
# ================================================================

print("\n[STEP 2/7] OPTIMIZING XGBOOST")
print("="*70)

def objective_xgboost(trial):
    """Optuna objective for XGBoost"""
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
    }

    model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate on validation set using backtest
    val_pred = model.predict(X_val)
    val_results = evaluate_model(val_pred, y_val, train.iloc[val_split:]['Close'])

    return val_results['sharpe']

# Run optimization
study_xgb = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='xgboost_optimization'
)
study_xgb.optimize(objective_xgboost, n_trials=20, show_progress_bar=True)

print(f"\nBest XGBoost Val Sharpe: {study_xgb.best_value:.3f}")
print(f"Best params: {study_xgb.best_params}")

# Train best XGBoost on full training data
best_xgb = xgb.XGBRegressor(**study_xgb.best_params, random_state=42, verbosity=0)
best_xgb.fit(X_train_full, y_train_full)
xgb_pred = best_xgb.predict(X_test)
xgb_results = evaluate_model(xgb_pred, y_test, test_prices)

print(f"Test Sharpe: {xgb_results['sharpe']:.3f}")


# ================================================================
# MODEL 2: LIGHTGBM
# ================================================================

if HAS_LIGHTGBM:
    print("\n[STEP 3/7] OPTIMIZING LIGHTGBM")
    print("="*70)

    def objective_lightgbm(trial):
        """Optuna objective for LightGBM"""
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        }

        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, force_col_wise=True)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_results = evaluate_model(val_pred, y_val, train.iloc[val_split:]['Close'])

        return val_results['sharpe']

    study_lgb = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='lightgbm_optimization'
    )
    study_lgb.optimize(objective_lightgbm, n_trials=20, show_progress_bar=True)

    print(f"\nBest LightGBM Val Sharpe: {study_lgb.best_value:.3f}")

    best_lgb = lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, verbose=-1)
    best_lgb.fit(X_train_full, y_train_full)
    lgb_pred = best_lgb.predict(X_test)
    lgb_results = evaluate_model(lgb_pred, y_test, test_prices)

    print(f"Test Sharpe: {lgb_results['sharpe']:.3f}")
else:
    print("\n[STEP 3/7] SKIPPING LIGHTGBM (not installed)")
    lgb_results = None


# ================================================================
# MODEL 3: CATBOOST
# ================================================================

if HAS_CATBOOST:
    print("\n[STEP 4/7] OPTIMIZING CATBOOST")
    print("="*70)

    def objective_catboost(trial):
        """Optuna objective for CatBoost"""
        params = {
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'iterations': trial.suggest_int('iterations', 50, 200),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 128),
        }

        model = cb.CatBoostRegressor(**params, random_state=42, verbose=False)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_results = evaluate_model(val_pred, y_val, train.iloc[val_split:]['Close'])

        return val_results['sharpe']

    study_cb = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='catboost_optimization'
    )
    study_cb.optimize(objective_catboost, n_trials=20, show_progress_bar=True)

    print(f"\nBest CatBoost Val Sharpe: {study_cb.best_value:.3f}")

    best_cb = cb.CatBoostRegressor(**study_cb.best_params, random_state=42, verbose=False)
    best_cb.fit(X_train_full, y_train_full)
    cb_pred = best_cb.predict(X_test)
    cb_results = evaluate_model(cb_pred, y_test, test_prices)

    print(f"Test Sharpe: {cb_results['sharpe']:.3f}")
else:
    print("\n[STEP 4/7] SKIPPING CATBOOST (not installed)")
    cb_results = None


# ================================================================
# MODEL 4: RANDOM FOREST
# ================================================================

print("\n[STEP 5/7] OPTIMIZING RANDOM FOREST")
print("="*70)

def objective_rf(trial):
    """Optuna objective for Random Forest"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
        'max_features': trial.suggest_float('max_features', 0.4, 1.0),
    }

    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_results = evaluate_model(val_pred, y_val, train.iloc[val_split:]['Close'])

    return val_results['sharpe']

study_rf = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='random_forest_optimization'
)
study_rf.optimize(objective_rf, n_trials=20, show_progress_bar=True)

print(f"\nBest Random Forest Val Sharpe: {study_rf.best_value:.3f}")

best_rf = RandomForestRegressor(**study_rf.best_params, random_state=42, n_jobs=-1)
best_rf.fit(X_train_full, y_train_full)
rf_pred = best_rf.predict(X_test)
rf_results = evaluate_model(rf_pred, y_test, test_prices)

print(f"Test Sharpe: {rf_results['sharpe']:.3f}")


# ================================================================
# MODEL 5: NEURAL NETWORK
# ================================================================

print("\n[STEP 6/7] TRAINING NEURAL NETWORK")
print("="*70)

# Normalize features
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# Simple neural network
best_nn = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=500,
    early_stopping=True,
    random_state=42,
    verbose=False
)

best_nn.fit(X_train_full_scaled, y_train_full)
nn_pred = best_nn.predict(X_test_scaled)
nn_results = evaluate_model(nn_pred, y_test, test_prices)

print(f"Neural Network Test Sharpe: {nn_results['sharpe']:.3f}")


# ================================================================
# RESULTS COMPARISON
# ================================================================

print("\n" + "="*70)
print("FINAL RESULTS - ALL MODELS ON TEST SET")
print("="*70)

# Collect results
results_data = [
    ('XGBoost', xgb_results['sharpe'], xgb_results['num_trades'], xgb_results['max_drawdown']),
]

if lgb_results:
    results_data.append(('LightGBM', lgb_results['sharpe'], lgb_results['num_trades'], lgb_results['max_drawdown']))

if cb_results:
    results_data.append(('CatBoost', cb_results['sharpe'], cb_results['num_trades'], cb_results['max_drawdown']))

results_data.append(('Random Forest', rf_results['sharpe'], rf_results['num_trades'], rf_results['max_drawdown']))
results_data.append(('Neural Network', nn_results['sharpe'], nn_results['num_trades'], nn_results['max_drawdown']))

# Create DataFrame
comparison = pd.DataFrame(
    results_data,
    columns=['Model', 'Sharpe', 'Trades', 'Max DD']
).sort_values('Sharpe', ascending=False)

print("\n" + comparison.to_string(index=False))

# Find best model
best_model_name = comparison.iloc[0]['Model']
best_sharpe = comparison.iloc[0]['Sharpe']

print(f"\n{'='*70}")
print(f"WINNER: {best_model_name.upper()}")
print(f"Sharpe: {best_sharpe:.2f}")
print(f"Max Drawdown: {comparison.iloc[0]['Max DD']*100:.2f}%")
print(f"Trades: {int(comparison.iloc[0]['Trades'])}")
print(f"{'='*70}")

# Compare with Test #5 baseline
baseline_sharpe = 1.28
baseline_dd = -0.0479
improvement = ((best_sharpe / baseline_sharpe) - 1) * 100

print(f"\nComparison with Test #5 Baseline:")
print(f"  Baseline Sharpe:         {baseline_sharpe:.2f}")
print(f"  Best Optimized Sharpe:   {best_sharpe:.2f}")
print(f"  Improvement:             {improvement:+.1f}%")
print(f"")
print(f"  Baseline Max DD:         {baseline_dd*100:.2f}%")
print(f"  Best Optimized Max DD:   {comparison.iloc[0]['Max DD']*100:.2f}%")

if improvement > 5:
    print(f"\n  🎉 SUCCESS! {improvement:.1f}% Sharpe improvement!")
elif improvement > 0:
    print(f"\n  ✅ MARGINAL improvement (+{improvement:.1f}%)")
else:
    print(f"\n  ⚠️ No improvement. Test #5 remains best.")


# ================================================================
# SAVE BEST MODEL
# ================================================================

print("\n[STEP 7/7] SAVING BEST MODEL")
print("="*70)

# Map model objects
model_objects = {
    'XGBoost': best_xgb,
    'LightGBM': best_lgb if lgb_results else None,
    'CatBoost': best_cb if cb_results else None,
    'Random Forest': best_rf,
    'Neural Network': (best_nn, scaler)
}

best_model_obj = model_objects[best_model_name]

# Save
if best_model_name == 'Neural Network':
    joblib.dump(best_model_obj, 'models/best_optimized_model.pkl')
else:
    joblib.dump(best_model_obj, 'models/best_optimized_model.pkl')

# Save metadata
metadata = {
    'model_version': '7.0.0-optimized',
    'created_at': datetime.now().isoformat(),
    'best_model': best_model_name,
    'optimization': {
        'method': 'optuna_bayesian',
        'trials_per_model': 20,
        'models_tested': len(results_data)
    },
    'test_performance': {
        'sharpe_ratio': float(best_sharpe),
        'max_drawdown': float(comparison.iloc[0]['Max DD']),
        'num_trades': int(comparison.iloc[0]['Trades']),
        'improvement_vs_baseline': float(improvement)
    },
    'all_models': {
        row['Model']: {
            'sharpe': float(row['Sharpe']),
            'trades': int(row['Trades']),
            'max_dd': float(row['Max DD'])
        }
        for _, row in comparison.iterrows()
    }
}

# Add best hyperparameters
if best_model_name == 'XGBoost':
    metadata['best_hyperparameters'] = study_xgb.best_params
elif best_model_name == 'LightGBM' and lgb_results:
    metadata['best_hyperparameters'] = study_lgb.best_params
elif best_model_name == 'CatBoost' and cb_results:
    metadata['best_hyperparameters'] = study_cb.best_params
elif best_model_name == 'Random Forest':
    metadata['best_hyperparameters'] = study_rf.best_params

with open('models/model_metadata_optimized.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  Model saved: models/best_optimized_model.pkl")
print(f"  Metadata saved: models/model_metadata_optimized.json")

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE!")
print("="*70)
