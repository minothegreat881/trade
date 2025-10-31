"""
HYBRID PORTFOLIO MANAGER
=========================

Uses the BEST model for each stock (from best_model_per_stock.csv)
- Loads appropriate model (ORIGINAL/MULTI-SCALE/ADAPTIVE)
- Uses correct data source for each model
- Generates trading signals
- Backtests hybrid portfolio
- Compares with uniform approaches

Expected Sharpe: 2.437 (vs 1.084 original)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*80)
print("HYBRID PORTFOLIO MANAGER")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# LOAD BEST MODEL SELECTION
# ================================================================

print("\n[1/6] Loading best model selection...")

best_models = pd.read_csv('results/best_model_per_stock.csv')
print(f"  Loaded selection for {len(best_models)} stocks")

# Count by approach
approach_counts = best_models['best_approach'].value_counts()
print(f"\n  Distribution:")
for approach, count in approach_counts.items():
    print(f"    {approach:12s}: {count:2d} stocks")

print(f"\n  Expected portfolio Sharpe: {best_models['best_sharpe'].mean():.3f}")


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def load_model_and_data(ticker, approach, group):
    """
    Load correct model and data based on approach
    """
    # Exclude columns
    exclude_cols = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
        'Dividends', 'Stock Splits', 'fear_greed_classification',
        'target', 'target_5d_return', 'target_profit_3pct',
        'target_profit_any', 'target_max_drawdown_5d', 'target_max_profit_5d'
    ]

    if approach == 'ORIGINAL':
        # Load original model and data
        model_path = f'models/sp500/{ticker}_xgb.pkl'
        data_path = f'data/sp500_top50/{ticker}_features.csv'
        target_col = 'target'

    elif approach == 'MULTI-SCALE':
        # Load multi-scale model and data
        model_path = f'models/sp500_multiscale/{ticker}_xgb.pkl'
        data_path = f'data/sp500_multiscale/{ticker}_multiscale.csv'
        target_col = 'target_5d_return'

    elif approach == 'ADAPTIVE':
        # Load adaptive model and appropriate data
        model_path = f'models/sp500_adaptive/{ticker}_xgb.pkl'

        # HIGH VOL uses original data, LOW/MEDIUM use multiscale
        if group == 'HIGH':
            data_path = f'data/sp500_top50/{ticker}_features.csv'
            target_col = 'target'
        else:
            data_path = f'data/sp500_multiscale/{ticker}_multiscale.csv'
            target_col = 'target_5d_return'
    else:
        raise ValueError(f"Unknown approach: {approach}")

    # Load model
    model = joblib.load(model_path)

    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Get model's expected features
    if hasattr(model, 'feature_names_in_'):
        # XGBoost stores the feature names it was trained with
        feature_cols = list(model.feature_names_in_)
    else:
        # Fallback: use all non-excluded columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Check if all required features are present
    missing_features = set(feature_cols) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")

    # Get X and y (select only features model expects)
    X = df[feature_cols]
    y = df[target_col]

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    return model, X, y


# ================================================================
# GENERATE SIGNALS FOR ALL STOCKS
# ================================================================

print("\n[2/6] Generating trading signals for all stocks...")

all_signals = {}
all_returns = {}
stock_info = []

for idx, row in best_models.iterrows():
    ticker = row['ticker']
    approach = row['best_approach']
    group = row['group']

    print(f"  [{idx+1}/50] {ticker:6s} ({approach:12s}) ", end='')

    try:
        # Load model and data
        model, X, y = load_model_and_data(ticker, approach, group)

        # Train/test split (80/20 chronological)
        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        if len(X_test) < 20:
            print(f"SKIP - test set too small ({len(X_test)})")
            continue

        # Generate predictions
        predictions = model.predict(X_test)

        # Trading signals (buy when prediction > 0)
        signals = (predictions > 0).astype(int)

        # Store signals and returns with common index
        all_signals[ticker] = pd.Series(signals, index=y_test.index)
        all_returns[ticker] = pd.Series(y_test.values, index=y_test.index)

        stock_info.append({
            'ticker': ticker,
            'approach': approach,
            'group': group,
            'test_samples': len(X_test),
            'features': X.shape[1],
            'signals': signals.sum(),
            'signal_rate': signals.mean()
        })

        print(f"OK | Test: {len(X_test):3d} | Signals: {signals.sum():3d} ({signals.mean():.1%})")

    except Exception as e:
        print(f"ERROR - {str(e)}")
        continue

print(f"\n  Successfully generated signals for {len(all_signals)} stocks")


# ================================================================
# ALIGN DATA ACROSS ALL STOCKS
# ================================================================

print("\n[3/6] Aligning data across all stocks...")

# Convert to DataFrames
signals_df = pd.DataFrame(all_signals)
returns_df = pd.DataFrame(all_returns)

# Find common dates (intersection)
common_dates = signals_df.index.intersection(returns_df.index)
print(f"  Common test period: {len(common_dates)} days")
print(f"  Date range: {common_dates[0].date()} to {common_dates[-1].date()}")

# Align both dataframes
signals_df = signals_df.loc[common_dates]
returns_df = returns_df.loc[common_dates]

# Handle any remaining NaN
signals_df = signals_df.fillna(0)
returns_df = returns_df.fillna(0)

print(f"  Final shape: {signals_df.shape}")


# ================================================================
# PORTFOLIO STRATEGY
# ================================================================

print("\n[4/6] Calculating portfolio performance...")

# Equal weight portfolio
num_stocks = len(signals_df.columns)

# Calculate per-stock returns (long only when signal=1)
stock_returns = returns_df * (signals_df * 2 - 1)  # Convert 0/1 to -1/+1

# Portfolio return = average of all stocks
portfolio_returns = stock_returns.mean(axis=1)

# Calculate metrics
avg_return = portfolio_returns.mean()
volatility = portfolio_returns.std()
sharpe = (avg_return / volatility) * np.sqrt(252) if volatility > 0 else 0

# Win rate
win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)

# Max drawdown
cumulative = np.cumprod(1 + portfolio_returns)
running_max = np.maximum.accumulate(cumulative)
drawdown = (cumulative - running_max) / running_max
max_drawdown = np.min(drawdown)

# Total return
total_return = cumulative.iloc[-1] - 1

# Annualized return
days = len(portfolio_returns)
years = days / 252
annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0

print(f"\n  HYBRID PORTFOLIO PERFORMANCE:")
print(f"    Sharpe Ratio:       {sharpe:.3f}")
print(f"    Total Return:       {total_return:.1%}")
print(f"    Annualized Return:  {annualized_return:.1%}")
print(f"    Volatility (daily): {volatility*100:.2f}%")
print(f"    Win Rate:           {win_rate:.1%}")
print(f"    Max Drawdown:       {max_drawdown:.1%}")
print(f"    Avg Daily Return:   {avg_return*100:.3f}%")


# ================================================================
# PER-STOCK CONTRIBUTION
# ================================================================

print("\n[5/6] Analyzing per-stock contribution...")

stock_performance = []

for ticker in stock_returns.columns:
    stock_ret = stock_returns[ticker]

    # Calculate metrics for this stock
    stock_sharpe = (stock_ret.mean() / stock_ret.std()) * np.sqrt(252) if stock_ret.std() > 0 else 0
    stock_total = (np.cumprod(1 + stock_ret).iloc[-1] - 1) if len(stock_ret) > 0 else 0
    stock_win = (stock_ret > 0).sum() / len(stock_ret)

    # Find approach used
    approach = best_models[best_models['ticker'] == ticker]['best_approach'].values[0]

    stock_performance.append({
        'ticker': ticker,
        'approach': approach,
        'sharpe': stock_sharpe,
        'total_return': stock_total,
        'win_rate': stock_win,
        'avg_return': stock_ret.mean(),
        'volatility': stock_ret.std()
    })

stock_perf_df = pd.DataFrame(stock_performance)
stock_perf_df = stock_perf_df.sort_values('sharpe', ascending=False)

print(f"\n  TOP 10 CONTRIBUTORS:")
print("\n  Ticker | Approach     | Sharpe | Return | WinRate")
print("  " + "-" * 60)
for _, row in stock_perf_df.head(10).iterrows():
    print(f"  {row['ticker']:6s} | {row['approach']:12s} | {row['sharpe']:6.2f} | "
          f"{row['total_return']:6.1%} | {row['win_rate']:7.1%}")

print(f"\n  BOTTOM 5 CONTRIBUTORS:")
print("\n  Ticker | Approach     | Sharpe | Return | WinRate")
print("  " + "-" * 60)
for _, row in stock_perf_df.tail(5).iterrows():
    print(f"  {row['ticker']:6s} | {row['approach']:12s} | {row['sharpe']:6.2f} | "
          f"{row['total_return']:6.1%} | {row['win_rate']:7.1%}")


# ================================================================
# COMPARISON WITH UNIFORM APPROACHES
# ================================================================

print("\n[6/6] Comparison with uniform approaches")
print("="*80)

# Load uniform approach results
original_results = pd.read_csv('results/sp500/training_summary.csv')
multiscale_results = pd.read_csv('results/sp500_multiscale/training_summary.csv')
adaptive_results = pd.read_csv('results/sp500_adaptive/training_summary.csv')

comparison_data = {
    'Approach': ['ORIGINAL', 'MULTI-SCALE', 'ADAPTIVE', 'HYBRID (Best)'],
    'Avg Sharpe': [
        original_results['sharpe'].mean(),
        multiscale_results['sharpe'].mean(),
        adaptive_results['sharpe'].mean(),
        sharpe  # Hybrid portfolio sharpe
    ],
    'Win Rate': [
        original_results['win_rate'].mean(),
        multiscale_results['win_rate'].mean(),
        adaptive_results['win_rate'].mean(),
        win_rate
    ],
    'Max Sharpe': [
        original_results['sharpe'].max(),
        multiscale_results['sharpe'].max(),
        adaptive_results['sharpe'].max(),
        stock_perf_df['sharpe'].max()
    ],
    'Stocks Used': [50, 50, 50, len(signals_df.columns)]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Highlight improvements
hybrid_sharpe = sharpe
original_sharpe = original_results['sharpe'].mean()
improvement = hybrid_sharpe - original_sharpe

print(f"\n  HYBRID vs ORIGINAL:")
print(f"    Sharpe improvement: {improvement:+.3f} ({improvement/original_sharpe*100:+.1f}%)")

print(f"\n  HYBRID vs Individual Stock Avg:")
print(f"    Hybrid Portfolio: {hybrid_sharpe:.3f}")
print(f"    Avg Stock Sharpe: {stock_perf_df['sharpe'].mean():.3f}")
print(f"    Diversification benefit: {hybrid_sharpe - stock_perf_df['sharpe'].mean():.3f}")


# ================================================================
# SAVE RESULTS
# ================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save portfolio performance
output_dir = Path('results/hybrid_portfolio')
output_dir.mkdir(parents=True, exist_ok=True)

# Portfolio metrics
portfolio_metrics = {
    'sharpe_ratio': float(sharpe),
    'total_return': float(total_return),
    'annualized_return': float(annualized_return),
    'volatility': float(volatility),
    'win_rate': float(win_rate),
    'max_drawdown': float(max_drawdown),
    'avg_daily_return': float(avg_return),
    'num_stocks': int(num_stocks),
    'test_days': int(len(portfolio_returns)),
    'date_range': f"{common_dates[0].date()} to {common_dates[-1].date()}"
}

import json
with open(output_dir / 'portfolio_metrics.json', 'w') as f:
    json.dump(portfolio_metrics, f, indent=2)

# Stock performance
stock_perf_df.to_csv(output_dir / 'stock_contributions.csv', index=False)

# Portfolio returns time series
portfolio_returns_df = pd.DataFrame({
    'date': portfolio_returns.index,
    'return': portfolio_returns.values,
    'cumulative': cumulative.values
})
portfolio_returns_df.to_csv(output_dir / 'portfolio_returns.csv', index=False)

# Signals and returns
signals_df.to_csv(output_dir / 'all_signals.csv')
returns_df.to_csv(output_dir / 'all_returns.csv')

# Approach usage summary
approach_summary = best_models['best_approach'].value_counts().to_dict()
with open(output_dir / 'approach_summary.json', 'w') as f:
    json.dump(approach_summary, f, indent=2)

print(f"\n  Saved results to: {output_dir}/")
print(f"    - portfolio_metrics.json")
print(f"    - stock_contributions.csv")
print(f"    - portfolio_returns.csv")
print(f"    - all_signals.csv")
print(f"    - all_returns.csv")
print(f"    - approach_summary.json")


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "="*80)
print("HYBRID PORTFOLIO COMPLETE!")
print("="*80)

print(f"\n  Portfolio Performance:")
print(f"    Sharpe Ratio:  {sharpe:.3f}")
print(f"    Total Return:  {total_return:.1%}")
print(f"    Win Rate:      {win_rate:.1%}")

print(f"\n  Comparison:")
print(f"    vs ORIGINAL:    {hybrid_sharpe - original_sharpe:+.3f} ({(hybrid_sharpe - original_sharpe)/original_sharpe*100:+.1f}%)")
print(f"    vs MULTI-SCALE: {hybrid_sharpe - multiscale_results['sharpe'].mean():+.3f}")
print(f"    vs ADAPTIVE:    {hybrid_sharpe - adaptive_results['sharpe'].mean():+.3f}")

print(f"\n  Approach Distribution:")
for approach, count in approach_counts.items():
    print(f"    {approach:12s}: {count:2d} stocks ({count/len(best_models)*100:.0f}%)")

print(f"\n  Next steps:")
print(f"    1. Analyze portfolio_returns.csv for detailed performance")
print(f"    2. Review stock_contributions.csv to identify best/worst performers")
print(f"    3. Consider rebalancing strategy (equal weight vs Sharpe-weighted)")
print(f"    4. Implement position sizing based on individual stock Sharpe")

print("\n" + "="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
