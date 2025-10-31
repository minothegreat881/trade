"""
TRAIN NEURAL NETWORK MODELS - HYBRID APPROACH
==============================================

Replaces XGBoost with Neural Networks while keeping everything else the same:
- Same features
- Same train/test split
- Same HYBRID approach (ORIGINAL/MULTI-SCALE/ADAPTIVE per stock)
- Same evaluation metrics

Architecture:
- Input Layer → Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.3) → Dense(32) → Output(1)
- Early stopping to prevent overfitting
- Adam optimizer with learning rate scheduling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
import json
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

# Sklearn metrics
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("TRAIN NEURAL NETWORK MODELS - HYBRID APPROACH")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"TensorFlow version: {tf.__version__}")
print("="*80)


# ================================================================
# NEURAL NETWORK ARCHITECTURE
# ================================================================

def create_nn_model(input_dim, l1_reg=0.001, l2_reg=0.001, dropout=0.3, learning_rate=0.001):
    """
    Create Neural Network regression model

    Architecture:
    - Dense(128) + Dropout(0.3)
    - Dense(64) + Dropout(0.3)
    - Dense(32)
    - Output(1)
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),

        # Hidden layer 1
        layers.Dense(128, activation='relu',
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout),

        # Hidden layer 2
        layers.Dense(64, activation='relu',
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout),

        # Hidden layer 3
        layers.Dense(32, activation='relu',
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),

        # Output layer (regression)
        layers.Dense(1, activation='linear')
    ])

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


# ================================================================
# LOAD BEST MODEL SELECTION
# ================================================================

print("\n[1/5] Loading best model selection (from XGBoost hybrid)...")

best_models = pd.read_csv('results/best_model_per_stock.csv')
print(f"  Loaded selection for {len(best_models)} stocks")

approach_counts = best_models['best_approach'].value_counts()
print(f"\n  Distribution (from XGBoost):")
for approach, count in approach_counts.items():
    print(f"    {approach:12s}: {count:2d} stocks")


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def load_stock_data(ticker, approach, group):
    """Load correct data based on approach"""

    exclude_cols = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
        'Dividends', 'Stock Splits', 'fear_greed_classification',
        'target', 'target_5d_return', 'target_profit_3pct',
        'target_profit_any', 'target_max_drawdown_5d', 'target_max_profit_5d'
    ]

    # Determine data source
    if approach == 'ORIGINAL':
        data_path = f'data/sp500_top50/{ticker}_features.csv'
        target_col = 'target'
    elif approach == 'MULTI-SCALE':
        data_path = f'data/sp500_multiscale/{ticker}_multiscale.csv'
        target_col = 'target_5d_return'
    elif approach == 'ADAPTIVE':
        if group == 'HIGH':
            data_path = f'data/sp500_top50/{ticker}_features.csv'
            target_col = 'target'
        else:
            data_path = f'data/sp500_multiscale/{ticker}_multiscale.csv'
            target_col = 'target_5d_return'
    else:
        raise ValueError(f"Unknown approach: {approach}")

    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Get features
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    return X, y, feature_cols


# ================================================================
# TRAIN NEURAL NETWORKS
# ================================================================

print("\n[2/5] Training Neural Network models...")
print("  (This will take several minutes...)\\n")

# Create output directories
Path('models/neural_network').mkdir(parents=True, exist_ok=True)
Path('results/neural_network').mkdir(parents=True, exist_ok=True)

all_results = []

for idx, row in best_models.iterrows():
    ticker = row['ticker']
    approach = row['best_approach']
    group = row['group']

    print(f"  [{idx+1}/50] {ticker:6s} ({approach:12s}) ", end='')

    try:
        # Load data
        X, y, feature_cols = load_stock_data(ticker, approach, group)

        if len(X) < 100:
            print(f"ERROR - Only {len(X)} samples!")
            continue

        # Train/test split (80/20 chronological)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        if len(X_test) < 20:
            print(f"ERROR - Test set too small!")
            continue

        # Standardize features (important for NN)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create model
        model = create_nn_model(
            input_dim=X_train_scaled.shape[1],
            l1_reg=0.001,
            l2_reg=0.001,
            dropout=0.3,
            learning_rate=0.001
        )

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )

        # Train model
        history = model.fit(
            X_train_scaled, y_train.values,
            validation_data=(X_test_scaled, y_test.values),
            epochs=200,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        # Predictions
        train_pred = model.predict(X_train_scaled, verbose=0).flatten()
        test_pred = model.predict(X_test_scaled, verbose=0).flatten()

        # Metrics
        train_corr, _ = spearmanr(y_train, train_pred)
        test_corr, _ = spearmanr(y_test, test_pred)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        # Trading signals
        test_signals = (test_pred > 0).astype(int)
        test_returns = y_test.values

        # Strategy returns
        strategy_returns = test_returns * (test_signals * 2 - 1)

        # Calculate metrics
        avg_return = strategy_returns.mean()
        volatility = strategy_returns.std()
        sharpe = (avg_return / volatility) * np.sqrt(252) if volatility > 0 else 0

        # Win rate
        trades = strategy_returns[test_signals == 1]
        win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0

        # Max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Total return
        total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0

        # Save model and scaler
        model_file = f'models/neural_network/{ticker}_nn.h5'
        model.save(model_file)

        scaler_file = f'models/neural_network/{ticker}_scaler.pkl'
        import joblib
        joblib.dump(scaler, scaler_file)

        # Save metrics
        metrics = {
            'ticker': ticker,
            'approach': approach,
            'group': group,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(feature_cols),
            'train_corr': float(train_corr),
            'test_corr': float(test_corr),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'sharpe': float(sharpe),
            'total_return': float(total_return),
            'avg_return': float(avg_return),
            'volatility': float(volatility),
            'win_rate': float(win_rate),
            'max_drawdown': float(max_drawdown),
            'num_trades': int(test_signals.sum()),
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        metrics_file = f'results/neural_network/{ticker}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        all_results.append(metrics)

        print(f"OK | Sharpe: {sharpe:5.2f} | Corr: {test_corr:.3f} | Epochs: {metrics['epochs_trained']:3d}")

    except Exception as e:
        print(f"ERROR - {str(e)}")
        continue


# ================================================================
# CREATE SUMMARY
# ================================================================

print(f"\n[3/5] Creating summary report...")

if len(all_results) == 0:
    print("  ERROR: No models trained successfully!")
    exit(1)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('sharpe', ascending=False)

# Save summary
summary_file = 'results/neural_network/training_summary.csv'
results_df.to_csv(summary_file, index=False)

print(f"  Saved summary to {summary_file}")
print(f"  Successfully trained {len(results_df)} models")


# ================================================================
# COMPARISON WITH XGBOOST
# ================================================================

print("\n[4/5] Comparison with XGBoost")
print("="*80)

# Load XGBoost results (from hybrid approach)
xgb_original = pd.read_csv('results/sp500/training_summary.csv')
xgb_multiscale = pd.read_csv('results/sp500_multiscale/training_summary.csv')
xgb_adaptive = pd.read_csv('results/sp500_adaptive/training_summary.csv')

# Calculate XGBoost hybrid performance (using best approach per stock)
xgb_hybrid_sharpes = []
for _, row in best_models.iterrows():
    ticker = row['ticker']
    approach = row['best_approach']

    if approach == 'ORIGINAL':
        sharpe = xgb_original[xgb_original['ticker'] == ticker]['sharpe'].values
    elif approach == 'MULTI-SCALE':
        sharpe = xgb_multiscale[xgb_multiscale['ticker'] == ticker]['sharpe'].values
    else:  # ADAPTIVE
        sharpe = xgb_adaptive[xgb_adaptive['ticker'] == ticker]['sharpe'].values

    if len(sharpe) > 0:
        xgb_hybrid_sharpes.append(sharpe[0])

xgb_hybrid_avg = np.mean(xgb_hybrid_sharpes) if xgb_hybrid_sharpes else 0

print(f"\nOverall Performance Comparison:")
print(f"  XGBoost (Hybrid Best): {xgb_hybrid_avg:.3f}")
print(f"  Neural Network:        {results_df['sharpe'].mean():.3f}")
print(f"  Difference:            {results_df['sharpe'].mean() - xgb_hybrid_avg:+.3f}")

# Stock-by-stock comparison
print(f"\n  Stock-by-Stock Wins:")
nn_wins = 0
xgb_wins = 0

for _, row in results_df.iterrows():
    ticker = row['ticker']
    nn_sharpe = row['sharpe']

    # Find XGBoost sharpe for this stock
    approach = best_models[best_models['ticker'] == ticker]['best_approach'].values[0]

    if approach == 'ORIGINAL':
        xgb_sharpe = xgb_original[xgb_original['ticker'] == ticker]['sharpe'].values
    elif approach == 'MULTI-SCALE':
        xgb_sharpe = xgb_multiscale[xgb_multiscale['ticker'] == ticker]['sharpe'].values
    else:
        xgb_sharpe = xgb_adaptive[xgb_adaptive['ticker'] == ticker]['sharpe'].values

    if len(xgb_sharpe) > 0:
        if nn_sharpe > xgb_sharpe[0]:
            nn_wins += 1
        else:
            xgb_wins += 1

print(f"    Neural Network wins: {nn_wins}/{nn_wins + xgb_wins} stocks ({nn_wins/(nn_wins+xgb_wins)*100:.1f}%)")
print(f"    XGBoost wins:        {xgb_wins}/{nn_wins + xgb_wins} stocks ({xgb_wins/(nn_wins+xgb_wins)*100:.1f}%)")


# ================================================================
# TOP PERFORMERS
# ================================================================

print("\n[5/5] Neural Network Top Performers")
print("="*80)

print(f"\nTOP 10 NEURAL NETWORK MODELS:")
print("\nTicker | Approach     | Sharpe | Corr | WinRate | Epochs")
print("-" * 70)
for _, row in results_df.head(10).iterrows():
    print(f"{row['ticker']:6s} | {row['approach']:12s} | {row['sharpe']:6.2f} | "
          f"{row['test_corr']:4.2f} | {row['win_rate']:7.1%} | {row['epochs_trained']:3.0f}")

print(f"\nBOTTOM 5 NEURAL NETWORK MODELS:")
print("\nTicker | Approach     | Sharpe | Corr | WinRate | Epochs")
print("-" * 70)
for _, row in results_df.tail(5).iterrows():
    print(f"{row['ticker']:6s} | {row['approach']:12s} | {row['sharpe']:6.2f} | "
          f"{row['test_corr']:4.2f} | {row['win_rate']:7.1%} | {row['epochs_trained']:3.0f}")


# ================================================================
# SUMMARY STATISTICS
# ================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nNeural Network Models:")
print(f"  Mean Sharpe:     {results_df['sharpe'].mean():6.2f}")
print(f"  Median Sharpe:   {results_df['sharpe'].median():6.2f}")
print(f"  Std Sharpe:      {results_df['sharpe'].std():6.2f}")
print(f"  Min Sharpe:      {results_df['sharpe'].min():6.2f}")
print(f"  Max Sharpe:      {results_df['sharpe'].max():6.2f}")

print(f"\nTraining Stats:")
print(f"  Avg Epochs:      {results_df['epochs_trained'].mean():.0f}")
print(f"  Avg Train Loss:  {results_df['final_train_loss'].mean():.4f}")
print(f"  Avg Val Loss:    {results_df['final_val_loss'].mean():.4f}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

print(f"\nFiles created:")
print(f"  - models/neural_network/{{ticker}}_nn.h5")
print(f"  - models/neural_network/{{ticker}}_scaler.pkl")
print(f"  - results/neural_network/{{ticker}}_metrics.json")
print(f"  - results/neural_network/training_summary.csv")

print(f"\nNext steps:")
print(f"  1. Compare NN vs XGBoost performance in detail")
print(f"  2. Create ensemble (combine NN + XGBoost predictions)")
print(f"  3. Analyze which approach works best for which stocks")

print("\n" + "="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
