"""
TRAIN ADAPTIVE XGBOOST MODELS
===============================

Uses volatility-based adaptive feature selection:
- HIGH VOL: 80 simple features + strong regularization
- LOW VOL: 185 multi-scale features + allow complexity
- MEDIUM VOL: 100 hybrid features + balanced params
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
print("TRAIN ADAPTIVE XGBOOST MODELS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# LOAD CLASSIFICATION AND STRATEGY
# ================================================================

print("\n[1/5] Loading classification and feature strategy...")

# Load stock classification
classification = pd.read_csv('data/stock_volatility_classification.csv')
print(f"  Loaded classification for {len(classification)} stocks")

# Load feature strategy
with open('data/adaptive_feature_strategy.json', 'r') as f:
    feature_strategy = json.load(f)

# Load old results for comparison
old_original = pd.read_csv('results/sp500/training_summary.csv')
old_multiscale = pd.read_csv('results/sp500_multiscale/training_summary.csv')
print(f"  Loaded old results for comparison")

# Display strategy
print("\n  Strategy summary:")
for group in ['HIGH', 'LOW', 'MEDIUM']:
    stocks = classification[classification['group'] == group]
    print(f"    {group:6s}: {len(stocks):2d} stocks -> {feature_strategy[group]['feature_count']:3d} features")


# ================================================================
# PREPARE DATA SOURCES
# ================================================================

print("\n[2/5] Preparing data sources...")

# Exclude columns
exclude_cols = [
    'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
    'Dividends', 'Stock Splits', 'fear_greed_classification',
    'target', 'target_5d_return', 'target_profit_3pct',
    'target_profit_any', 'target_max_drawdown_5d', 'target_max_profit_5d'
]

print(f"  Excluding {len(exclude_cols)} columns")


# ================================================================
# TRAIN MODELS
# ================================================================

print("\n[3/5] Training adaptive models...")
print("  (This will take a few minutes...)\n")

# Create output directories
Path('models/sp500_adaptive').mkdir(parents=True, exist_ok=True)
Path('results/sp500_adaptive').mkdir(parents=True, exist_ok=True)

all_results = []

for idx, row in classification.iterrows():
    ticker = row['ticker']
    group = row['group']
    volatility = row['volatility_pct']

    print(f"  [{idx+1}/50] {ticker:6s} (VOL: {volatility:.2f}% | {group:6s}) ", end='')

    try:
        # Load appropriate data source
        if group == 'HIGH':
            # HIGH VOL: Use original simple features
            data_file = f'data/sp500_top50/{ticker}_features.csv'
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            target_col = 'target'
        else:
            # LOW/MEDIUM: Use multiscale features
            data_file = f'data/sp500_multiscale/{ticker}_multiscale.csv'
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            target_col = 'target_5d_return'

        # Check target exists
        if target_col not in df.columns:
            print(f"ERROR - No {target_col} column!")
            continue

        # Get features for this group
        available_features = [col for col in df.columns if col not in exclude_cols]
        group_features = feature_strategy[group]['features']

        # Intersect available and desired features
        feature_cols = [f for f in group_features if f in available_features]

        if len(feature_cols) < 10:
            print(f"ERROR - Only {len(feature_cols)} features available!")
            continue

        # Prepare X and y
        X = df[feature_cols]
        y = df[target_col]

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

        # Get group-specific XGBoost params
        xgb_params = feature_strategy[group]['xgb_params']

        # Train XGBoost
        model = xgb.XGBRegressor(**xgb_params)
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

        # Trading signals
        test_signals = (test_pred > 0).astype(int)
        test_returns = y_test.values

        # Strategy returns
        strategy_returns = test_returns * (test_signals * 2 - 1)

        # Calculate metrics
        avg_return = strategy_returns.mean()
        volatility_metric = strategy_returns.std()
        sharpe = (avg_return / volatility_metric) * np.sqrt(252) if volatility_metric > 0 else 0

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
        old_original_row = old_original[old_original['ticker'] == ticker]
        old_multiscale_row = old_multiscale[old_multiscale['ticker'] == ticker]

        old_original_sharpe = old_original_row.iloc[0]['sharpe'] if len(old_original_row) > 0 else None
        old_multiscale_sharpe = old_multiscale_row.iloc[0]['sharpe'] if len(old_multiscale_row) > 0 else None

        # Calculate improvements
        improvement_vs_original = sharpe - old_original_sharpe if old_original_sharpe is not None else None
        improvement_vs_multiscale = sharpe - old_multiscale_sharpe if old_multiscale_sharpe is not None else None

        # Save model
        model_file = f'models/sp500_adaptive/{ticker}_xgb.pkl'
        joblib.dump(model, model_file)

        # Save metrics
        metrics = {
            'ticker': ticker,
            'group': group,
            'volatility_pct': float(volatility),
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
            'volatility': float(volatility_metric),
            'win_rate': float(win_rate),
            'max_drawdown': float(max_drawdown),
            'num_trades': int(test_signals.sum()),
            # Comparison metrics
            'old_original_sharpe': float(old_original_sharpe) if old_original_sharpe is not None else None,
            'old_multiscale_sharpe': float(old_multiscale_sharpe) if old_multiscale_sharpe is not None else None,
            'improvement_vs_original': float(improvement_vs_original) if improvement_vs_original is not None else None,
            'improvement_vs_multiscale': float(improvement_vs_multiscale) if improvement_vs_multiscale is not None else None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        metrics_file = f'results/sp500_adaptive/{ticker}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        all_results.append(metrics)

        # Display
        improvement_str = ""
        if improvement_vs_original is not None:
            sign = "+" if improvement_vs_original > 0 else ""
            improvement_str = f"(vs orig: {sign}{improvement_vs_original:.2f})"

        print(f"OK | Sharpe: {sharpe:5.2f} {improvement_str} | Corr: {test_corr:.3f} | Features: {len(feature_cols):3d}")

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
summary_file = 'results/sp500_adaptive/training_summary.csv'
results_df.to_csv(summary_file, index=False)

print(f"  Saved summary to {summary_file}")
print(f"  Successfully trained {len(results_df)} models")


# ================================================================
# COMPARISON ANALYSIS
# ================================================================

print("\n[5/5] Performance Comparison")
print("="*80)

# Overall statistics
print(f"\nOVERALL PERFORMANCE:")
print(f"  Original (124 features):     Avg Sharpe = {old_original['sharpe'].mean():.3f}")
print(f"  Multi-Scale (185 features):  Avg Sharpe = {old_multiscale['sharpe'].mean():.3f}")
print(f"  Adaptive (custom features):  Avg Sharpe = {results_df['sharpe'].mean():.3f}")

# Improvement statistics
improvements_vs_original = results_df['improvement_vs_original'].dropna()
improvements_vs_multiscale = results_df['improvement_vs_multiscale'].dropna()

print(f"\nIMPROVEMENT VS ORIGINAL:")
print(f"  Better:  {(improvements_vs_original > 0).sum()}/{len(improvements_vs_original)} stocks")
print(f"  Worse:   {(improvements_vs_original < 0).sum()}/{len(improvements_vs_original)} stocks")
print(f"  Avg Delta: {improvements_vs_original.mean():+.3f}")

print(f"\nIMPROVEMENT VS MULTI-SCALE:")
print(f"  Better:  {(improvements_vs_multiscale > 0).sum()}/{len(improvements_vs_multiscale)} stocks")
print(f"  Worse:   {(improvements_vs_multiscale < 0).sum()}/{len(improvements_vs_multiscale)} stocks")
print(f"  Avg Delta: {improvements_vs_multiscale.mean():+.3f}")

# Performance by group
print(f"\nPERFORMANCE BY GROUP:")
for group in ['HIGH', 'LOW', 'MEDIUM']:
    group_data = results_df[results_df['group'] == group]
    if len(group_data) > 0:
        avg_sharpe = group_data['sharpe'].mean()
        avg_improvement_orig = group_data['improvement_vs_original'].mean()
        avg_improvement_ms = group_data['improvement_vs_multiscale'].mean()
        print(f"\n  {group} VOL ({len(group_data)} stocks):")
        print(f"    Avg Sharpe: {avg_sharpe:.3f}")
        print(f"    vs Original: {avg_improvement_orig:+.3f}")
        print(f"    vs Multi-Scale: {avg_improvement_ms:+.3f}")

# Top 10 models
print("\n" + "="*80)
print("TOP 10 MODELS (ADAPTIVE)")
print("="*80)

top10 = results_df.head(10)
print("\nTicker | Group  | Sharpe | vs Orig | vs Multi | Corr | WinRate | Features")
print("-" * 85)
for _, row in top10.iterrows():
    delta_orig = row['improvement_vs_original']
    delta_multi = row['improvement_vs_multiscale']
    delta_orig_str = f"{delta_orig:+.2f}" if pd.notna(delta_orig) else "  N/A"
    delta_multi_str = f"{delta_multi:+.2f}" if pd.notna(delta_multi) else "  N/A"
    print(f"{row['ticker']:6s} | {row['group']:6s} | {row['sharpe']:6.2f} | "
          f"{delta_orig_str:7s} | {delta_multi_str:8s} | "
          f"{row['test_corr']:4.2f} | {row['win_rate']:7.1%} | {row['features']:3.0f}")

# Biggest improvements
print("\n" + "="*80)
print("BIGGEST IMPROVEMENTS VS ORIGINAL")
print("="*80)

improvements = results_df.sort_values('improvement_vs_original', ascending=False)
print("\nTicker | Group  | Old Sharpe | New Sharpe | Improvement")
print("-" * 70)
for _, row in improvements.head(10).iterrows():
    print(f"{row['ticker']:6s} | {row['group']:6s} | {row['old_original_sharpe']:10.2f} | "
          f"{row['sharpe']:10.2f} | {row['improvement_vs_original']:+11.2f}")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nAdaptive Models:")
print(f"  Mean Sharpe:   {results_df['sharpe'].mean():6.2f}")
print(f"  Median Sharpe: {results_df['sharpe'].median():6.2f}")
print(f"  Std Sharpe:    {results_df['sharpe'].std():6.2f}")
print(f"  Min Sharpe:    {results_df['sharpe'].min():6.2f}")
print(f"  Max Sharpe:    {results_df['sharpe'].max():6.2f}")

print(f"\nFeature Usage:")
print(f"  HIGH VOL:   {results_df[results_df['group']=='HIGH']['features'].mean():.0f} features avg")
print(f"  MEDIUM VOL: {results_df[results_df['group']=='MEDIUM']['features'].mean():.0f} features avg")
print(f"  LOW VOL:    {results_df[results_df['group']=='LOW']['features'].mean():.0f} features avg")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

print(f"\nFiles created:")
print(f"  - models/sp500_adaptive/{{ticker}}_xgb.pkl")
print(f"  - results/sp500_adaptive/{{ticker}}_metrics.json")
print(f"  - results/sp500_adaptive/training_summary.csv")

print(f"\nNext steps:")
print(f"  1. Analyze which approach works best for which stock types")
print(f"  2. Create ensemble portfolio combining best models")
print(f"  3. Backtest adaptive portfolio strategy")

print("\n" + "="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
