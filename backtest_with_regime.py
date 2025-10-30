"""
Backtest with Regime-Aware Position Sizing

Compares trading performance:
- WITHOUT regime detection (baseline)
- WITH regime detection (regime-aware)

Shows improvement from dynamic position sizing based on market conditions.

Usage:
    # Test single method
    python backtest_with_regime.py --method rule_based

    # Test and compare
    python backtest_with_regime.py --method ensemble --compare

    # Test all methods
    python backtest_with_regime.py --compare-all
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import sys
import codecs
import json

# Fix Windows encoding
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from modules.regime_detector import RegimeDetector
from backtester import Backtester
from xgboost_model import XGBoostModel
from metrics import generate_metrics_report
import config

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class RegimeAwareBacktester(Backtester):
    """
    Extended backtester with regime-aware position sizing.
    """

    def __init__(self, regimes, regime_detector, **kwargs):
        """
        Initialize regime-aware backtester.

        Args:
            regimes: Series with regime labels indexed by date
            regime_detector: RegimeDetector instance
            **kwargs: Arguments for base Backtester
        """
        super().__init__(**kwargs)
        self.regimes = regimes
        self.regime_detector = regime_detector

        # Store original position size - will be dynamically adjusted
        self.original_position_size_pct = self.position_size_pct

    def backtest(self, predictions, features_df):
        """
        Run backtest with regime-aware position sizing.

        Overrides parent method to adjust position size based on regime.
        """
        logger.info("\nRunning regime-aware backtest...")

        # Align regimes with predictions
        regimes_aligned = self.regimes.reindex(predictions.index, method='ffill')

        trades = []
        equity_curve = [self.initial_capital]
        dates = [predictions.index[0]]
        current_position = 0
        current_shares = 0
        entry_price = 0

        for i in range(len(predictions)):
            date = predictions.index[i]
            prediction = predictions.iloc[i]
            current_price = features_df.loc[date, 'Close']

            # Get current regime
            regime = regimes_aligned.loc[date]

            # Get regime-specific parameters
            regime_position_size = self.regime_detector.get_position_size(regime)
            regime_threshold = self.regime_detector.get_prediction_threshold(regime)

            # Adjust position size for this regime
            self.max_position_size = regime_position_size

            # Exit logic (always check for exits first)
            if current_position > 0:
                # Calculate unrealized P&L
                unrealized_pnl = (current_price - entry_price) / entry_price

                # Exit conditions
                exit_signal = False

                # Stop loss
                if unrealized_pnl < -self.stop_loss:
                    exit_signal = True

                # Take profit
                if unrealized_pnl > self.take_profit:
                    exit_signal = True

                # Regime change to crisis or bear - exit immediately
                if regime in [self.regime_detector.REGIME_CRISIS, self.regime_detector.REGIME_BEAR]:
                    exit_signal = True

                # Exit on reversal prediction
                if prediction < 0:
                    exit_signal = True

                if exit_signal:
                    # Exit position
                    exit_value = current_shares * current_price
                    commission = exit_value * self.commission_rate
                    proceeds = exit_value - commission

                    trades.append({
                        'entry_date': dates[-1],
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': current_shares,
                        'pnl': proceeds - current_position,
                        'return': (current_price - entry_price) / entry_price,
                        'regime': regime
                    })

                    current_position = 0
                    current_shares = 0
                    entry_price = 0
                    equity_curve.append(equity_curve[-1] + (proceeds - current_position))

            # Entry logic
            if current_position == 0 and prediction > regime_threshold:
                # Only enter if regime allows (position_size > 0)
                if regime_position_size > 0:
                    # Calculate position size based on regime
                    available_capital = equity_curve[-1]
                    position_value = available_capital * regime_position_size

                    # Account for commission
                    commission = position_value * self.commission_rate
                    cost_adjusted = position_value - commission

                    if cost_adjusted > 0:
                        # Enter position
                        current_shares = cost_adjusted / current_price
                        current_position = cost_adjusted
                        entry_price = current_price

                        dates.append(date)
                        equity_curve.append(equity_curve[-1] - position_value)

            # Update equity curve (even if no trade)
            if i > 0 and len(equity_curve) == len(dates):
                equity_curve.append(equity_curve[-1])
                dates.append(date)

        # Close any remaining position
        if current_position > 0:
            final_date = predictions.index[-1]
            final_price = features_df.loc[final_date, 'Close']
            exit_value = current_shares * final_price
            commission = exit_value * self.commission_rate
            proceeds = exit_value - commission

            trades.append({
                'entry_date': dates[-1],
                'exit_date': final_date,
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': current_shares,
                'pnl': proceeds - current_position,
                'return': (final_price - entry_price) / entry_price,
                'regime': regimes_aligned.loc[final_date]
            })

        # Calculate metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_series = pd.Series(equity_curve, index=dates)

        metrics = self.calculate_metrics(equity_series, trades_df)
        metrics['regime_stats'] = self._calculate_regime_stats(trades_df)

        return metrics, trades_df, equity_series


    def _calculate_regime_stats(self, trades_df):
        """Calculate statistics by regime."""
        if len(trades_df) == 0:
            return {}

        regime_stats = {}

        for regime in [self.regime_detector.REGIME_BULL,
                      self.regime_detector.REGIME_SIDEWAYS,
                      self.regime_detector.REGIME_BEAR,
                      self.regime_detector.REGIME_CRISIS]:

            regime_trades = trades_df[trades_df['regime'] == regime]

            if len(regime_trades) > 0:
                stats = {
                    'num_trades': len(regime_trades),
                    'win_rate': (regime_trades['return'] > 0).mean(),
                    'avg_return': regime_trades['return'].mean(),
                    'total_pnl': regime_trades['pnl'].sum()
                }
            else:
                stats = {
                    'num_trades': 0,
                    'win_rate': 0,
                    'avg_return': 0,
                    'total_pnl': 0
                }

            regime_stats[regime] = stats

        return regime_stats


def load_model_and_data():
    """Load or train model and test data."""
    logger.info("Loading data and training model...")

    # Load test data with sentiment features
    test_X = pd.read_csv('data/train_test_sentiment/test_X.csv', index_col=0, parse_dates=True)
    test_y = pd.read_csv('data/train_test_sentiment/test_y.csv', index_col=0, parse_dates=True)['target']

    # Load train data
    train_X = pd.read_csv('data/train_test_sentiment/train_X.csv', index_col=0, parse_dates=True)
    train_y = pd.read_csv('data/train_test_sentiment/train_y.csv', index_col=0, parse_dates=True)['target']

    # Load full data for price information (use SPY_featured.csv which exists)
    # Load WITHOUT parse_dates to avoid timezone issues
    full_data = pd.read_csv('data/processed/SPY_featured.csv', index_col=0)

    # Convert index to DatetimeIndex, stripping timezone info
    import dateutil.parser
    cleaned_dates = []
    for date_str in full_data.index:
        dt = dateutil.parser.parse(date_str)
        dt_naive = dt.replace(tzinfo=None)  # Remove timezone
        cleaned_dates.append(dt_naive)
    full_data.index = pd.DatetimeIndex(cleaned_dates)

    # Train model
    logger.info("Training XGBoost model with sentiment features...")
    model = XGBoostModel()

    # Split train into train/val
    split_idx = int(len(train_X) * 0.8)
    model.train(
        train_X.iloc[:split_idx],
        train_y.iloc[:split_idx],
        train_X.iloc[split_idx:],
        train_y.iloc[split_idx:],
        early_stopping_rounds=10
    )

    logger.info(f"  Test data: {len(test_X)} samples from {test_X.index.min().date()} to {test_X.index.max().date()}")

    return model, test_X, test_y, full_data


def run_baseline_backtest(model, test_X, test_y, full_data):
    """Run baseline backtest without regime detection."""
    logger.info("\nRunning BASELINE backtest (no regime detection)...")

    # Generate predictions
    predictions = model.predict(test_X)
    predictions = pd.Series(predictions, index=test_X.index)

    # Merge test data with full_data to get prices
    # Use join to align by index
    test_df = pd.DataFrame({
        'prediction': predictions,
        'actual': test_y
    })

    # Join with full_data to get Close prices
    test_df = test_df.join(full_data[['Close']], how='left')

    # Fill missing prices with forward fill
    test_df['Close'] = test_df['Close'].ffill()

    # Extract aligned series
    predictions_aligned = test_df['prediction']
    actuals_aligned = test_df['actual']
    prices_aligned = test_df['Close']

    # Run backtest
    backtester = Backtester(
        initial_capital=config.INITIAL_CAPITAL,
        transaction_cost=config.TRANSACTION_COST,
        holding_period=config.HOLDING_PERIOD,
        position_size_pct=0.50,  # Fixed 50% position size
        prediction_threshold=0.001
    )

    results = backtester.run_backtest(predictions_aligned, actuals_aligned, prices_aligned)

    # Calculate metrics
    metrics = generate_metrics_report(results)

    return metrics, results.get('trades', []), results.get('equity_curve', pd.Series())


def run_regime_backtest(model, test_X, test_y, full_data, method='ensemble'):
    """Run backtest with regime detection."""
    logger.info(f"\nRunning REGIME-AWARE backtest (method={method})...")

    # Detect regimes on full data
    detector = RegimeDetector()

    if method == 'rule_based':
        regimes = detector.detect_rule_based(full_data)
    elif method == 'hmm':
        regimes = detector.detect_hmm(full_data)
    elif method == 'ensemble':
        regimes = detector.detect_ensemble(full_data)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Generate predictions
    predictions = model.predict(test_X)
    predictions = pd.Series(predictions, index=test_X.index)

    # Merge test data with full_data and regimes
    test_df = pd.DataFrame({
        'prediction': predictions,
        'actual': test_y
    })

    # Join with full_data to get Close prices and regimes
    test_df = test_df.join(full_data[['Close']], how='left')
    test_df = test_df.join(pd.DataFrame({'regime': regimes}), how='left')

    # Fill missing values
    test_df['Close'] = test_df['Close'].ffill()
    test_df['regime'] = test_df['regime'].ffill()

    # Filter predictions based on regime
    filtered_predictions = test_df['prediction'].copy()
    for date in filtered_predictions.index:
        regime = test_df.loc[date, 'regime']
        regime_threshold = detector.get_prediction_threshold(regime)
        regime_size = detector.get_position_size(regime)

        # If regime doesn't allow trading or prediction below threshold, set to 0
        if regime_size == 0 or filtered_predictions[date] < regime_threshold:
            filtered_predictions[date] = 0

    # Run backtest with filtered predictions
    backtester = Backtester(
        initial_capital=config.INITIAL_CAPITAL,
        transaction_cost=config.TRANSACTION_COST,
        holding_period=config.HOLDING_PERIOD,
        position_size_pct=0.50,  # Base size
        prediction_threshold=0.000001  # Very low since we already filtered
    )

    results = backtester.run_backtest(filtered_predictions, test_df['actual'], test_df['Close'])

    # Calculate metrics
    metrics = generate_metrics_report(results)

    # Add regime stats
    metrics['regime_stats'] = _calculate_regime_stats(results.get('trades', []), test_df['regime'])
    metrics['regimes'] = test_df['regime']

    return metrics, results.get('trades', []), results.get('equity_curve', pd.Series()), regimes, detector


def _calculate_regime_stats(trades, regimes):
    """Calculate statistics by regime."""
    if len(trades) == 0:
        return {}

    # Convert trades to DataFrame if it's not already
    if isinstance(trades, list):
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = trades

    regime_stats = {}

    for regime in regimes.unique():
        # Match trades to regime by entry date
        # trades_df might have entry_date as index or column
        if 'entry_date' in trades_df.columns:
            regime_mask = trades_df['entry_date'].apply(
                lambda d: regimes.loc[d] == regime if d in regimes.index else False
            )
        else:
            # entry_date is index
            regime_mask = trades_df.index.map(
                lambda d: regimes.loc[d] == regime if d in regimes.index else False
            )

        regime_trades_df = trades_df[regime_mask]

        if len(regime_trades_df) > 0:
            if 'return' in regime_trades_df.columns:
                returns = regime_trades_df['return'].values
                win_rate = (returns > 0).sum() / len(returns)
                avg_return = returns.mean()
            else:
                win_rate = 0
                avg_return = 0

            if 'pnl' in regime_trades_df.columns:
                total_pnl = regime_trades_df['pnl'].sum()
            else:
                total_pnl = 0

            regime_stats[regime] = {
                'num_trades': len(regime_trades_df),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'total_pnl': total_pnl
            }
        else:
            regime_stats[regime] = {
                'num_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_pnl': 0
            }

    return regime_stats


def compare_results(baseline_metrics, regime_metrics, method):
    """Print comparison between baseline and regime-aware."""
    print("\n" + "=" * 70)
    print(f"RESULTS COMPARISON: Baseline vs {method.upper()}")
    print("=" * 70)

    metrics_to_compare = [
        ('Sharpe Ratio', 'sharpe_ratio', '{:.2f}'),
        ('Annual Return', 'annual_return', '{:.2%}'),
        ('Max Drawdown', 'max_drawdown', '{:.2%}'),
        ('Win Rate', 'win_rate', '{:.1%}'),
        ('Total Trades', 'total_trades', '{:.0f}'),
        ('Profit Factor', 'profit_factor', '{:.2f}')
    ]

    print(f"\n{'Metric':<20} {'Baseline':>12} {'Regime':>12} {'Change':>12} {'Status':>10}")
    print("-" * 70)

    for metric_name, metric_key, fmt in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric_key, 0)
        regime_val = regime_metrics.get(metric_key, 0)

        # Calculate change
        if baseline_val != 0:
            change = (regime_val - baseline_val) / abs(baseline_val)
            change_str = f"{change:+.1%}"
        else:
            change_str = "N/A"

        # Determine status
        if metric_key == 'max_drawdown':
            # Lower is better for drawdown
            status = "✓" if regime_val > baseline_val else "✗"  # Less negative is better
        else:
            # Higher is better for others
            status = "✓" if regime_val > baseline_val else "✗"

        print(f"{metric_name:<20} {fmt.format(baseline_val):>12} {fmt.format(regime_val):>12} {change_str:>12} {status:>10}")

    # Show regime-specific stats if available
    if 'regime_stats' in regime_metrics:
        print("\n" + "=" * 70)
        print("PERFORMANCE BY REGIME")
        print("=" * 70)

        regime_stats = regime_metrics['regime_stats']

        print(f"\n{'Regime':<15} {'Trades':>10} {'Win Rate':>12} {'Avg Return':>14} {'Total P&L':>14}")
        print("-" * 70)

        for regime, stats in regime_stats.items():
            print(f"{regime:<15} {stats['num_trades']:>10.0f} {stats['win_rate']:>12.1%} "
                  f"{stats['avg_return']:>14.2%} ${stats['total_pnl']:>13,.2f}")


def save_results(baseline_metrics, regime_metrics, method, output_dir):
    """Save comparison results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save metrics
    comparison = {
        'baseline': baseline_metrics,
        'regime': regime_metrics,
        'method': method,
        'improvement': {
            'sharpe_ratio': regime_metrics.get('sharpe_ratio', 0) - baseline_metrics.get('sharpe_ratio', 0),
            'annual_return': regime_metrics.get('annual_return', 0) - baseline_metrics.get('annual_return', 0),
            'max_drawdown': regime_metrics.get('max_drawdown', 0) - baseline_metrics.get('max_drawdown', 0),
            'win_rate': regime_metrics.get('win_rate', 0) - baseline_metrics.get('win_rate', 0)
        }
    }

    with open(output_dir / f'comparison_{method}.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir}/")


def plot_comparison(baseline_equity, regime_equity, method, output_dir):
    """Plot equity curves comparison."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: Equity curves
    ax1 = axes[0]
    ax1.plot(baseline_equity.index, baseline_equity.values, label='Baseline (No Regime)', linewidth=2, color='blue')
    ax1.plot(regime_equity.index, regime_equity.values, label=f'Regime-Aware ({method})', linewidth=2, color='green')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Equity Curve Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Drawdown
    ax2 = axes[1]

    # Calculate drawdowns
    baseline_dd = (baseline_equity / baseline_equity.cummax() - 1)
    regime_dd = (regime_equity / regime_equity.cummax() - 1)

    ax2.fill_between(baseline_dd.index, baseline_dd.values, 0, alpha=0.3, color='blue', label='Baseline')
    ax2.fill_between(regime_dd.index, regime_dd.values, 0, alpha=0.3, color='green', label=f'Regime-Aware ({method})')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Drawdown Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / f'comparison_{method}.png'
    plt.savefig(output_path, dpi=150)
    logger.info(f"Comparison plot saved to {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Backtest with regime detection')
    parser.add_argument('--method', type=str, default='ensemble',
                       choices=['rule_based', 'hmm', 'ensemble'],
                       help='Regime detection method')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with baseline')
    parser.add_argument('--compare-all', action='store_true',
                       help='Test all methods and compare')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("REGIME-AWARE BACKTESTING")
    print("=" * 70 + "\n")

    # Load model and data
    model, test_X, test_y, full_data = load_model_and_data()

    if args.compare_all:
        # Test all methods
        methods = ['rule_based', 'hmm', 'ensemble']

        # Run baseline once
        print("\n" + "=" * 70)
        print("[BASELINE] NO REGIME DETECTION")
        print("=" * 70)
        baseline_metrics, baseline_trades, baseline_equity = run_baseline_backtest(
            model, test_X, test_y, full_data
        )

        results_all = {'baseline': baseline_metrics}

        for method in methods:
            print("\n" + "=" * 70)
            print(f"[{method.upper()}] REGIME DETECTION")
            print("=" * 70)

            try:
                regime_metrics, regime_trades, regime_equity, regimes, detector = run_regime_backtest(
                    model, test_X, test_y, full_data, method=method
                )

                results_all[method] = regime_metrics

                # Compare
                compare_results(baseline_metrics, regime_metrics, method)

                # Save
                output_dir = Path(f'results/regime_backtest_{method}')
                save_results(baseline_metrics, regime_metrics, method, output_dir)
                plot_comparison(baseline_equity, regime_equity, method, output_dir)

            except Exception as e:
                logger.error(f"Error with method {method}: {e}")
                import traceback
                traceback.print_exc()

    elif args.compare:
        # Compare single method with baseline
        print("\n" + "=" * 70)
        print("[BASELINE] NO REGIME DETECTION")
        print("=" * 70)
        baseline_metrics, baseline_trades, baseline_equity = run_baseline_backtest(
            model, test_X, test_y, full_data
        )

        print("\n" + "=" * 70)
        print(f"[{args.method.upper()}] REGIME DETECTION")
        print("=" * 70)
        regime_metrics, regime_trades, regime_equity, regimes, detector = run_regime_backtest(
            model, test_X, test_y, full_data, method=args.method
        )

        # Compare
        compare_results(baseline_metrics, regime_metrics, args.method)

        # Save
        output_dir = Path(f'results/regime_backtest_{args.method}')
        save_results(baseline_metrics, regime_metrics, args.method, output_dir)
        plot_comparison(baseline_equity, regime_equity, args.method, output_dir)

    else:
        # Just run regime backtest
        print("\n" + "=" * 70)
        print(f"[{args.method.upper()}] REGIME DETECTION")
        print("=" * 70)
        regime_metrics, regime_trades, regime_equity, regimes, detector = run_regime_backtest(
            model, test_X, test_y, full_data, method=args.method
        )

        print("\nResults:")
        for key, value in regime_metrics.items():
            if key != 'regime_stats':
                print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✓ REGIME BACKTESTING COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
