"""
RETRAIN XGBOOST MODELS WITH MULTI-SCALE FEATURES
=================================================

Trains new models with enhanced multi-scale features and compares
performance with original single-scale models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from datetime import datetime
import joblib
import json
import warnings
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

print("="*80)
print("RETRAIN XGBOOST WITH MULTI-SCALE FEATURES")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# LOAD SUMMARY DATA
# ================================================================

print("\n[1/5] Loading summary data...")

summary = pd.read_csv('data/sp500_top50_summary.csv')
print(f"  Found {len(summary)} stocks")

# Load old results for comparison
old_results_path = Path('results/sp500/training_summary.csv')
if old_results_path.exists():
    old_results = pd.read_csv(old_results_path)
    print(f"  Loaded old results: {len(old_results)} stocks")
else:
    old_results = None
    print(f"  No old results found")


# ================================================================
# XGBOOST PARAMETERS
# ================================================================

print("\n[2/5] Setting up XGBoost parameters...")

XGB_PARAMS = {
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.05,
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

print(f"  Parameters: {XGB_PARAMS}")


# ================================================================
# TRAIN MODELS
# ================================================================

print("\n[3/5] Training models with multi-scale features...")
print("  (This will take a few minutes...)\n")

# Create output directories
Path('models/sp500_multiscale').mkdir(parents=True, exist_ok=True)
Path('results/sp500_multiscale').mkdir(parents=True, exist_ok=True)

all_results = []

for idx, ticker in enumerate(summary['Ticker'], 1):
    print(f"  [{idx}/50] Training {ticker}...", end=' ')

    try:
        # Load multiscale data
        input_file = f'data/sp500_multiscale/{ticker}_multiscale.csv'
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)

        # Exclude columns
        exclude_cols = [
            # Original base columns
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'Dividends', 'Stock Splits', 'fear_greed_classification',
            # All target variables
            'target', 'target_5d_return', 'target_profit_3pct',
            'target_profit_any', 'target_max_drawdown_5d', 'target_max_profit_5d'
        ]

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Use target_5d_return as target
        if 'target_5d_return' not in df.columns:
            print(f"ERROR - No target_5d_return!")
            continue

        X = df[feature_cols]
        y = df['target_5d_return']

        # Drop NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            print(f"ERROR - Only {len(X)} samples!")
            continue

        # Train/Test split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        if len(X_test) < 20:
            print(f"ERROR - Test set too small!")
            continue

        # Train XGBoost
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],
                 verbose=False)

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Metrics
        train_corr, _ = spearmanr(y_train, train_pred)
        test_corr, _ = spearmanr(y_test, test_pred)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        # Trading signals (buy when prediction > 0)
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

        # Get old metrics for comparison
        old_sharpe = None
        old_corr = None
        old_win_rate = None
        if old_results is not None:
            old_row = old_results[old_results['ticker'] == ticker]
            if len(old_row) > 0:
                old_sharpe = old_row.iloc[0]['sharpe']
                old_corr = old_row.iloc[0]['test_corr']
                old_win_rate = old_row.iloc[0]['win_rate']

        # Calculate improvements
        sharpe_improvement = sharpe - old_sharpe if old_sharpe is not None else None
        corr_improvement = test_corr - old_corr if old_corr is not None and not np.isnan(old_corr) else None

        # Save model
        model_file = f'models/sp500_multiscale/{ticker}_xgb.pkl'
        joblib.dump(model, model_file)

        # Save metrics
        metrics = {
            'ticker': ticker,
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
            # Old metrics for comparison
            'old_sharpe': float(old_sharpe) if old_sharpe is not None else None,
            'old_test_corr': float(old_corr) if old_corr is not None else None,
            'old_win_rate': float(old_win_rate) if old_win_rate is not None else None,
            'sharpe_improvement': float(sharpe_improvement) if sharpe_improvement is not None else None,
            'corr_improvement': float(corr_improvement) if corr_improvement is not None else None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        metrics_file = f'results/sp500_multiscale/{ticker}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        all_results.append(metrics)

        # Display with comparison
        if sharpe_improvement is not None:
            improvement_str = f"(Δ{sharpe_improvement:+.2f})"
        else:
            improvement_str = ""

        print(f"OK | Sharpe: {sharpe:.2f} {improvement_str} | Corr: {test_corr:.3f} | Win: {win_rate:.1%}")

    except Exception as e:
        print(f"ERROR - {str(e)}")
        continue


# ================================================================
# CREATE SUMMARY
# ================================================================

print(f"\n[4/5] Creating summary report...")

if len(all_results) == 0:
    print("  ERROR: No models trained successfully!")
    exit(1)

results_df = pd.DataFrame(all_results)

# Sort by Sharpe ratio
results_df = results_df.sort_values('sharpe', ascending=False)

# Save summary
summary_file = 'results/sp500_multiscale/training_summary.csv'
results_df.to_csv(summary_file, index=False)

print(f"  Saved summary to {summary_file}")
print(f"  Successfully trained {len(results_df)} models")


# ================================================================
# COMPARISON ANALYSIS
# ================================================================

print("\n[5/5] Performance Comparison")
print("="*80)

# Filter results with old data for comparison
results_with_old = results_df[results_df['old_sharpe'].notna()].copy()

if len(results_with_old) > 0:
    print(f"\nComparing {len(results_with_old)} stocks (with old metrics)")

    # Calculate improvements
    sharpe_improvements = results_with_old['sharpe_improvement'].dropna()
    improved_count = (sharpe_improvements > 0).sum()
    degraded_count = (sharpe_improvements < 0).sum()

    print(f"\nSHARPE RATIO:")
    print(f"  Improved:  {improved_count}/{len(sharpe_improvements)} stocks ({improved_count/len(sharpe_improvements)*100:.1f}%)")
    print(f"  Degraded:  {degraded_count}/{len(sharpe_improvements)} stocks ({degraded_count/len(sharpe_improvements)*100:.1f}%)")
    print(f"  Avg Δ:     {sharpe_improvements.mean():+.3f}")
    print(f"  Old avg:   {results_with_old['old_sharpe'].mean():.3f}")
    print(f"  New avg:   {results_with_old['sharpe'].mean():.3f}")

    # Correlation improvements
    corr_improvements = results_with_old['corr_improvement'].dropna()
    if len(corr_improvements) > 0:
        corr_improved = (corr_improvements > 0).sum()
        print(f"\nTEST CORRELATION:")
        print(f"  Improved:  {corr_improved}/{len(corr_improvements)} stocks")
        print(f"  Avg Δ:     {corr_improvements.mean():+.3f}")

print("\n" + "="*80)
print("TOP 10 MODELS (Multi-Scale)")
print("="*80)

top10 = results_df.head(10)
print("\nTicker | Sharpe | Δ      | Corr | WinRate | Features")
print("-" * 70)
for _, row in top10.iterrows():
    delta = row['sharpe_improvement']
    delta_str = f"{delta:+.2f}" if pd.notna(delta) else "  NEW"
    print(f"{row['ticker']:6s} | {row['sharpe']:6.2f} | {delta_str:6s} | "
          f"{row['test_corr']:4.2f} | {row['win_rate']:7.1%} | {row['features']:3.0f}")

print("\n" + "="*80)
print("BIGGEST IMPROVEMENTS")
print("="*80)

if len(results_with_old) > 0:
    improvements = results_with_old.sort_values('sharpe_improvement', ascending=False)
    print("\nTicker | Old Sharpe | New Sharpe | Improvement")
    print("-" * 60)
    for _, row in improvements.head(10).iterrows():
        print(f"{row['ticker']:6s} | {row['old_sharpe']:10.2f} | {row['sharpe']:10.2f} | "
              f"{row['sharpe_improvement']:+11.2f}")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nMulti-Scale Models:")
print(f"  Mean Sharpe:   {results_df['sharpe'].mean():6.2f}")
print(f"  Median Sharpe: {results_df['sharpe'].median():6.2f}")
print(f"  Std Sharpe:    {results_df['sharpe'].std():6.2f}")
print(f"  Min Sharpe:    {results_df['sharpe'].min():6.2f}")
print(f"  Max Sharpe:    {results_df['sharpe'].max():6.2f}")

print(f"\nFeature Count:")
print(f"  Old models:  ~124 features")
print(f"  New models:  {results_df['features'].mean():.0f} features (avg)")
print(f"  Increase:    +{results_df['features'].mean() - 124:.0f} features")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

print(f"\nFiles created:")
print(f"  - models/sp500_multiscale/{{ticker}}_xgb.pkl")
print(f"  - results/sp500_multiscale/{{ticker}}_metrics.json")
print(f"  - results/sp500_multiscale/training_summary.csv")

print(f"\nNext steps:")
print(f"  1. Analyze feature importance by time scale")
print(f"  2. Create portfolio with best multi-scale models")
print(f"  3. Backtest portfolio strategy")

print("\n" + "="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
