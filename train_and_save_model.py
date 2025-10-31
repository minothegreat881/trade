"""
Quick script to train and save XGBoost model for live trading
"""

import pandas as pd
import joblib
from pathlib import Path
from xgboost_model import XGBoostModel
import config

print("=" * 60)
print("TRAINING MODEL FOR LIVE TRADING")
print("=" * 60)

# Load data
print("\n[1/4] Loading data...")
train_data = pd.read_csv('data/train_test/train_data.csv', index_col=0, parse_dates=True)
test_data = pd.read_csv('data/train_test/test_data.csv', index_col=0, parse_dates=True)

print(f"  Train: {len(train_data)} samples")
print(f"  Test: {len(test_data)} samples")

# Prepare features (only technical features, exclude OHLCV, dividends, etc.)
exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Capital Gains']
feature_cols = [col for col in train_data.columns if col not in exclude_cols]
X_train = train_data[feature_cols]
y_train = train_data['target']
X_test = test_data[feature_cols]
y_test = test_data['target']

print(f"  Features: {len(feature_cols)}")

# Train model
print("\n[2/4] Training XGBoost model...")
params = {
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'min_child_weight': 5
}
model = XGBoostModel(params=params)

# Split train into train/val
split_idx = int(len(X_train) * 0.8)
X_train_split = X_train.iloc[:split_idx]
y_train_split = y_train.iloc[:split_idx]
X_val = X_train.iloc[split_idx:]
y_val = y_train.iloc[split_idx:]

model.train(X_train_split, y_train_split, X_val, y_val)

print("  Model trained successfully!")

# Test model
print("\n[3/4] Testing model...")
y_pred = model.predict(X_test)
r2 = model.model.score(X_test, y_test)
print(f"  Test R2: {r2:.4f}")
print(f"  Mean prediction: {y_pred.mean():.6f}")

# Save model
print("\n[4/4] Saving model...")
Path('models').mkdir(exist_ok=True)
model_path = 'models/xgboost_sentiment_model.pkl'
joblib.dump(model.model, model_path)

print(f"  [OK] Model saved to: {model_path}")
print(f"  File size: {Path(model_path).stat().st_size / 1024:.1f} KB")

print("\n" + "=" * 60)
print("[OK] MODEL READY FOR LIVE TRADING!")
print("=" * 60)
print(f"\nNow you can run:")
print("  python run_live_simulator.py")
