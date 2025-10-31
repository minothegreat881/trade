import joblib

model = joblib.load('models/xgboost_advanced_features.pkl')

print("="*70)
print("TEST #5 BASELINE - WINNING MODEL (Sharpe 1.28)")
print("="*70)

print("\nModel Parameters:")
print("-"*70)
params = model.get_params()

# Key parameters first
key_params = ['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight',
              'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda']

print("\nKEY PARAMETERS:")
for k in key_params:
    if k in params:
        print(f"  {k:25s} = {params[k]}")

print("\nOTHER PARAMETERS:")
for k, v in sorted(params.items()):
    if k not in key_params:
        print(f"  {k:25s} = {v}")

print("\n" + "="*70)
