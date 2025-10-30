"""
Walk-Forward Testing for Trading Model
Time-series cross-validation with rolling windows

Simulates real-world monthly retraining scenario.
Tests model robustness across different time periods.

Usage:
    python walk_forward_validation.py              # Default (24-month train)
    python walk_forward_validation.py --train 18   # 18-month train window
    python walk_forward_validation.py --test 3     # 3-month test window
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import argparse
import sys
import codecs

# Fix Windows encoding
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from xgboost_model import XGBoostModel
from backtester import Backtester
from metrics import generate_metrics_report
from modules.regime_detector import RegimeDetector
import config


class WalkForwardValidator:
    """
    Walk-forward validation with rolling windows.

    Simulates real trading where:
    - Train on past N months
    - Test on next M months
    - Roll forward, retrain, test again
    """

    def __init__(self, train_months=24, test_months=1, step_months=1, use_regime=False, regime_method='rule_based'):
        """
        Args:
            train_months: Size of training window (months)
            test_months: Size of test window (months)
            step_months: How many months to step forward (default: 1)
            use_regime: Whether to use regime-aware position sizing
            regime_method: Regime detection method ('rule_based', 'hmm', or 'ensemble')
        """
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.use_regime = use_regime
        self.regime_method = regime_method
        self.results = []

        if use_regime:
            self.regime_detector = RegimeDetector()

    def create_windows(self, df):
        """
        Create rolling train/test windows.

        Returns:
            List of window dicts with train/test date ranges
        """
        # Convert to monthly periods
        df_monthly = df.copy()
        df_monthly['year_month'] = df_monthly.index.to_period('M')

        # Get unique months
        unique_months = df_monthly['year_month'].unique()
        unique_months = sorted(unique_months)

        print(f"Total months in data: {len(unique_months)}")
        print(f"First month: {unique_months[0]}")
        print(f"Last month: {unique_months[-1]}")

        windows = []

        # Create windows
        i = 0
        while i + self.train_months + self.test_months <= len(unique_months):
            # Get month ranges
            train_start_month = unique_months[i]
            train_end_month = unique_months[i + self.train_months - 1]
            test_start_month = unique_months[i + self.train_months]
            test_end_month = unique_months[i + self.train_months + self.test_months - 1]

            # Convert to timestamps
            windows.append({
                'train_start': train_start_month.to_timestamp(),
                'train_end': train_end_month.to_timestamp('M'),  # End of month
                'test_start': test_start_month.to_timestamp(),
                'test_end': test_end_month.to_timestamp('M')
            })

            i += self.step_months

        return windows

    def run_single_window(self, df, window, window_num, best_params=None):
        """
        Train and test on a single window.

        Returns:
            dict with results for this window
        """
        try:
            # Split data by date ranges
            train_mask = (df.index >= window['train_start']) & (df.index <= window['train_end'])
            test_mask = (df.index >= window['test_start']) & (df.index <= window['test_end'])

            train_data = df[train_mask].copy()
            test_data = df[test_mask].copy()

            if len(train_data) < 100:
                print(f"  ⚠️ SKIP: Too few train samples ({len(train_data)})")
                return None

            if len(test_data) < 5:
                print(f"  ⚠️ SKIP: Too few test samples ({len(test_data)})")
                return None

            # Prepare features
            feature_cols = [col for col in df.columns if col not in ['target', 'Close', 'Open', 'High', 'Low', 'Volume']]

            X_train = train_data[feature_cols]
            y_train = train_data['target']
            X_test = test_data[feature_cols]
            y_test = test_data['target']

            # Train model
            model = XGBoostModel(params=best_params)

            # Use 80% of train for training, 20% for early stopping
            split_idx = int(len(X_train) * 0.8)
            X_train_split = X_train.iloc[:split_idx]
            y_train_split = y_train.iloc[:split_idx]
            X_val_split = X_train.iloc[split_idx:]
            y_val_split = y_train.iloc[split_idx:]

            model.train(X_train_split, y_train_split, X_val_split, y_val_split,
                       early_stopping_rounds=10, verbose=False)

            # Predict
            test_pred = model.predict(X_test)
            test_pred = pd.Series(test_pred, index=X_test.index)

            # NEW: Apply regime detection if enabled
            if self.use_regime:
                # Detect regimes on full dataframe (need historical context)
                regime_series = self.regime_detector.detect_rule_based(df)  # Returns a Series

                # Filter predictions based on regime
                for date in test_pred.index:
                    # Try to match date (handle timezone issues)
                    try:
                        # regime_series is already a Series, so just access by date
                        regime = regime_series.loc[date]
                        regime_threshold = self.regime_detector.get_prediction_threshold(regime)
                        regime_pos_size = self.regime_detector.get_position_size(regime)

                        # Filter out unfavorable regime predictions
                        if test_pred[date] < regime_threshold or regime_pos_size == 0:
                            test_pred[date] = 0  # Don't trade
                    except (KeyError, IndexError, TypeError):
                        # Date not found in regime data, keep prediction as-is
                        pass

            # Get prices for backtest
            if 'Close' in test_data.columns:
                test_prices = test_data['Close']
            else:
                # If no Close, use index as placeholder
                test_prices = pd.Series(100, index=test_data.index)

            # Backtest
            backtester = Backtester(
                initial_capital=10000,
                transaction_cost=config.TRANSACTION_COST,
                holding_period=config.HOLDING_PERIOD,
                position_size_pct=0.5,
                prediction_threshold=0.000001  # Very low since we already filtered
            )

            backtest_results = backtester.run_backtest(test_pred, y_test, test_prices)

            # Calculate metrics
            metrics = generate_metrics_report(backtest_results)

            # Store results
            result = {
                'window_num': window_num,
                'window': window,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'sharpe': metrics['sharpe_ratio'],
                'annual_return': metrics['annual_return'],
                'max_drawdown': metrics['max_drawdown_pct'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'total_trades': metrics['total_trades'],
                'total_return': metrics['total_return'],
                'equity_curve': backtest_results['equity_curve']
            }

            return result

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_walk_forward(self, df, best_params=None):
        """
        Run complete walk-forward analysis.
        """
        print("\n" + "="*70)
        print("WALK-FORWARD VALIDATION")
        if self.use_regime:
            print(f"  WITH REGIME DETECTION ({self.regime_method})")
        else:
            print("  WITHOUT REGIME DETECTION (Baseline)")
        print("="*70)
        print(f"Train window: {self.train_months} months")
        print(f"Test window:  {self.test_months} month(s)")
        print(f"Step size:    {self.step_months} month(s)")
        print()

        # Create windows
        windows = self.create_windows(df)
        print(f"\nTotal windows: {len(windows)}")
        print(f"Data range: {df.index.min().date()} to {df.index.max().date()}")
        print()

        # Run each window
        for i, window in enumerate(windows, 1):
            print(f"\n[Window {i}/{len(windows)}]")
            print(f"  Train: {window['train_start'].date()} to {window['train_end'].date()}")
            print(f"  Test:  {window['test_start'].date()} to {window['test_end'].date()}")

            result = self.run_single_window(df, window, i, best_params)

            if result is None:
                continue

            self.results.append(result)

            print(f"  Sharpe: {result['sharpe']:.2f}")
            print(f"  Return: {result['annual_return']*100:.2f}%")
            print(f"  Trades: {result['total_trades']}")

        print("\n" + "="*70)
        print(f"COMPLETED: {len(self.results)}/{len(windows)} windows")
        print("="*70)

    def calculate_aggregate_metrics(self):
        """
        Calculate aggregate statistics across all windows.
        """
        if len(self.results) == 0:
            return None

        sharpes = [r['sharpe'] for r in self.results]
        returns = [r['annual_return'] for r in self.results]
        dds = [r['max_drawdown'] for r in self.results]
        win_rates = [r['win_rate'] for r in self.results]

        aggregate = {
            'num_windows': len(self.results),
            'mean_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'min_sharpe': np.min(sharpes),
            'max_sharpe': np.max(sharpes),
            'median_sharpe': np.median(sharpes),
            'positive_sharpe_pct': sum(1 for s in sharpes if s > 0) / len(sharpes) * 100,
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_drawdown': np.mean(dds),
            'worst_drawdown': np.min(dds),
            'mean_win_rate': np.mean(win_rates)
        }

        return aggregate

    def create_visualizations(self, save_dir='results/walk_forward'):
        """
        Create comprehensive visualizations.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        print("\nGenerating visualizations...")

        # 1. Overview figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Sharpe ratio over time
        ax = axes[0, 0]
        test_dates = [r['window']['test_start'] for r in self.results]
        sharpes = [r['sharpe'] for r in self.results]

        ax.plot(test_dates, sharpes, marker='o', linewidth=2, markersize=6, color='#2E86AB')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=np.mean(sharpes), color='green', linestyle='--', alpha=0.7,
                  linewidth=2, label=f'Mean: {np.mean(sharpes):.2f}')
        ax.set_title('Sharpe Ratio Over Time', fontsize=13, fontweight='bold')
        ax.set_xlabel('Test Period Start', fontsize=11)
        ax.set_ylabel('Sharpe Ratio', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: Distribution of Sharpe ratios
        ax = axes[0, 1]
        ax.hist(sharpes, bins=min(15, len(sharpes)//2), edgecolor='black', alpha=0.7, color='#A23B72')
        ax.axvline(x=np.mean(sharpes), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(sharpes):.2f}')
        ax.axvline(x=np.median(sharpes), color='green', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(sharpes):.2f}')
        ax.set_title('Sharpe Ratio Distribution', fontsize=13, fontweight='bold')
        ax.set_xlabel('Sharpe Ratio', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Annual returns over time
        ax = axes[1, 0]
        returns = [r['annual_return'] * 100 for r in self.results]
        colors = ['green' if r > 0 else 'red' for r in returns]
        ax.bar(range(len(returns)), returns, alpha=0.7, color=colors, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=np.mean(returns), color='blue', linestyle='--', linewidth=2,
                  alpha=0.7, label=f'Mean: {np.mean(returns):.1f}%')
        ax.set_title('Annual Returns by Window', fontsize=13, fontweight='bold')
        ax.set_xlabel('Window #', fontsize=11)
        ax.set_ylabel('Annual Return (%)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: Win rate distribution
        ax = axes[1, 1]
        win_rates = [r['win_rate'] * 100 for r in self.results]
        ax.scatter(range(len(win_rates)), win_rates, s=100, alpha=0.6, color='#F18F01', edgecolors='black')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1, label='50% (random)')
        ax.axhline(y=np.mean(win_rates), color='green', linestyle='--', linewidth=2,
                  alpha=0.7, label=f'Mean: {np.mean(win_rates):.1f}%')
        ax.set_title('Win Rate by Window', fontsize=13, fontweight='bold')
        ax.set_xlabel('Window #', fontsize=11)
        ax.set_ylabel('Win Rate (%)', fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'walk_forward_overview.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: walk_forward_overview.png")
        plt.close()

        # 2. Detailed results table
        self._create_results_table(save_dir)

    def _create_results_table(self, save_dir):
        """Create detailed results table visualization."""
        fig, ax = plt.subplots(figsize=(14, max(8, len(self.results) * 0.4)))
        ax.axis('tight')
        ax.axis('off')

        # Create table data
        table_data = []
        table_data.append(['#', 'Test Period', 'Sharpe', 'Return', 'DD', 'Win%', 'Trades'])

        for result in self.results:
            window_num = result['window_num']
            test_start = result['window']['test_start'].strftime('%Y-%m')
            test_end = result['window']['test_end'].strftime('%Y-%m')
            test_period = f"{test_start}"

            sharpe = f"{result['sharpe']:.2f}"
            ret = f"{result['annual_return']*100:.1f}%"
            dd = f"{result['max_drawdown']:.1f}%"
            wr = f"{result['win_rate']*100:.0f}%"
            trades = result['total_trades']

            table_data.append([window_num, test_period, sharpe, ret, dd, wr, trades])

        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.08, 0.18, 0.12, 0.14, 0.12, 0.12, 0.10])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(7):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color code Sharpe ratios
        for i in range(1, len(table_data)):
            sharpe_val = float(table_data[i][2])
            if sharpe_val > 0.8:
                table[(i, 2)].set_facecolor('#A5D6A7')  # Light green
            elif sharpe_val > 0.5:
                table[(i, 2)].set_facecolor('#C8E6C9')  # Lighter green
            elif sharpe_val > 0:
                table[(i, 2)].set_facecolor('#FFF9C4')  # Yellow
            else:
                table[(i, 2)].set_facecolor('#FFCDD2')  # Light red

        plt.title('Walk-Forward Results by Window', fontsize=15, fontweight='bold', pad=20)
        plt.savefig(save_dir / 'results_table.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: results_table.png")
        plt.close()

    def save_results(self, save_dir='results/walk_forward'):
        """
        Save results to files.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        # Save aggregate metrics
        aggregate = self.calculate_aggregate_metrics()

        with open(save_dir / 'aggregate_metrics.json', 'w') as f:
            # Convert numpy types
            agg_save = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in aggregate.items()}
            json.dump(agg_save, f, indent=2)

        # Save individual window results
        results_df = pd.DataFrame([
            {
                'window_num': r['window_num'],
                'test_start': r['window']['test_start'],
                'test_end': r['window']['test_end'],
                'train_samples': r['train_samples'],
                'test_samples': r['test_samples'],
                'sharpe': r['sharpe'],
                'annual_return': r['annual_return'],
                'total_return': r['total_return'],
                'max_drawdown': r['max_drawdown'],
                'win_rate': r['win_rate'],
                'profit_factor': r['profit_factor'],
                'total_trades': r['total_trades']
            }
            for r in self.results
        ])

        results_df.to_csv(save_dir / 'window_results.csv', index=False)

        print(f"\n✓ Results saved to: {save_dir}/")


def main(train_months=24, test_months=1, step_months=1, compare=False, regime_method='rule_based'):
    """
    Main walk-forward validation.

    Args:
        train_months: Training window size
        test_months: Test window size
        step_months: Step size
        compare: If True, compare WITH vs WITHOUT regime detection
        regime_method: Regime detection method to use
    """
    print("\n" + "="*70)
    print("WALK-FORWARD VALIDATION SETUP")
    if compare:
        print("  MODE: COMPARISON (WITH vs WITHOUT REGIME DETECTION)")
    print("="*70)

    # Load data
    print("\n[1/4] Loading data...")

    # Try to load combined data
    train_file = Path('data/train_test_sentiment/train_X.csv')
    test_file = Path('data/train_test_sentiment/test_X.csv')

    if not train_file.exists() or not test_file.exists():
        print("❌ ERROR: Could not find sentiment data files!")
        print("   Expected: data/train_test_sentiment/train_X.csv")
        print("   Run integrate_sentiment.py first!")
        sys.exit(1)

    # Load features and targets
    train_X = pd.read_csv(train_file, index_col=0, parse_dates=True)
    train_y = pd.read_csv('data/train_test_sentiment/train_y.csv', index_col=0, parse_dates=True)
    test_X = pd.read_csv(test_file, index_col=0, parse_dates=True)
    test_y = pd.read_csv('data/train_test_sentiment/test_y.csv', index_col=0, parse_dates=True)

    # Load original data for Close prices
    train_orig = pd.read_csv('data/train_test/train_data.csv', index_col=0, parse_dates=True)
    test_orig = pd.read_csv('data/train_test/test_data.csv', index_col=0, parse_dates=True)

    # Normalize dates
    for df in [train_X, train_y, test_X, test_y, train_orig, test_orig]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        elif hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()

    # Combine features + target + prices
    train_full = train_X.copy()
    train_full['target'] = train_y['target']
    train_full['Close'] = train_orig['Close']

    test_full = test_X.copy()
    test_full['target'] = test_y['target']
    test_full['Close'] = test_orig['Close']

    # Concatenate train and test
    df = pd.concat([train_full, test_full]).sort_index()

    print(f"  ✓ Loaded {len(df)} rows")
    print(f"  ✓ Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  ✓ Features: {len([c for c in df.columns if c not in ['target', 'Close']])}")

    # Load best hyperparameters
    print("\n[2/4] Loading best hyperparameters...")

    # Use tuned params from Phase 3.5
    best_params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    print("  ✓ Using tuned parameters from Phase 3.5")

    # Run walk-forward
    print(f"\n[3/4] Running walk-forward validation...")
    num_models = (len(df.index.to_period('M').unique()) - train_months - test_months + 1) // step_months
    print(f"  This will train {num_models} models...")

    if compare:
        print(f"  Running TWICE (without and with regime detection)")
        print(f"  Estimated time: 20-40 minutes\n")

        # Run WITHOUT regime detection
        print("\n" + "="*70)
        print("[BASELINE] WITHOUT REGIME DETECTION")
        print("="*70)

        validator_no_regime = WalkForwardValidator(
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            use_regime=False
        )
        validator_no_regime.run_walk_forward(df, best_params=best_params)

        if len(validator_no_regime.results) == 0:
            print("\n❌ ERROR: No windows completed successfully!")
            sys.exit(1)

        aggregate_no_regime = validator_no_regime.calculate_aggregate_metrics()

        # Run WITH regime detection
        print("\n" + "="*70)
        print(f"[REGIME-AWARE] WITH REGIME DETECTION ({regime_method})")
        print("="*70)

        validator_with_regime = WalkForwardValidator(
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            use_regime=True,
            regime_method=regime_method
        )
        validator_with_regime.run_walk_forward(df, best_params=best_params)

        if len(validator_with_regime.results) == 0:
            print("\n❌ ERROR: No windows completed successfully!")
            sys.exit(1)

        aggregate_with_regime = validator_with_regime.calculate_aggregate_metrics()

        # Use regime results as primary for saving
        validator = validator_with_regime
        aggregate = aggregate_with_regime

    else:
        print(f"  Estimated time: 10-20 minutes")

        validator = WalkForwardValidator(
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            use_regime=False
        )
        validator.run_walk_forward(df, best_params=best_params)

        if len(validator.results) == 0:
            print("\n❌ ERROR: No windows completed successfully!")
            sys.exit(1)

        # Calculate and display results
        print("\n[4/4] Calculating aggregate metrics...")
        aggregate = validator.calculate_aggregate_metrics()

    print("\n" + "="*70)
    print("WALK-FORWARD RESULTS")
    print("="*70)
    print(f"\nWindows tested: {aggregate['num_windows']}")
    print(f"\n📊 Sharpe Ratio:")
    print(f"  Mean:       {aggregate['mean_sharpe']:.2f}")
    print(f"  Median:     {aggregate['median_sharpe']:.2f}")
    print(f"  Std Dev:    {aggregate['std_sharpe']:.2f}")
    print(f"  Min:        {aggregate['min_sharpe']:.2f}")
    print(f"  Max:        {aggregate['max_sharpe']:.2f}")
    print(f"  % Positive: {aggregate['positive_sharpe_pct']:.1f}%")

    print(f"\n💰 Annual Return:")
    print(f"  Mean: {aggregate['mean_return']*100:.2f}%")
    print(f"  Std:  {aggregate['std_return']*100:.2f}%")

    print(f"\n📉 Max Drawdown:")
    print(f"  Mean:  {aggregate['mean_drawdown']:.2f}%")
    print(f"  Worst: {aggregate['worst_drawdown']:.2f}%")

    print(f"\n🎯 Win Rate:")
    print(f"  Mean: {aggregate['mean_win_rate']*100:.1f}%")

    # Show comparison if enabled
    if compare:
        print("\n" + "="*70)
        print("WALK-FORWARD COMPARISON: WITH vs WITHOUT REGIME")
        print("="*70)

        comparison = {
            'Mean Sharpe': (aggregate_no_regime['mean_sharpe'], aggregate_with_regime['mean_sharpe']),
            '% Positive': (aggregate_no_regime['positive_sharpe_pct'], aggregate_with_regime['positive_sharpe_pct']),
            'Worst Loss': (aggregate_no_regime['worst_drawdown'], aggregate_with_regime['worst_drawdown']),
            'Mean Return': (aggregate_no_regime['mean_return']*100, aggregate_with_regime['mean_return']*100)
        }

        print(f"\n{'Metric':<15s}  {'Without':>8s}  {'With':>8s}  {'Change':>10s}")
        print("-" * 50)

        for metric, (no, yes) in comparison.items():
            change = ((yes - no) / abs(no) * 100) if no != 0 else 0
            if 'Loss' in metric:
                # For loss, improvement is less negative
                status = "✓" if yes > no else "✗"
            else:
                # For others, improvement is higher
                status = "✓" if yes > no else "✗"

            print(f"{metric:<15s}  {no:8.2f}  {yes:8.2f}  {change:+9.1f}% {status}")

        # Save comparison
        save_dir = Path('results/walk_forward_regime')
        save_dir.mkdir(exist_ok=True, parents=True)

        comparison_df = pd.DataFrame({
            'Metric': list(comparison.keys()),
            'Without_Regime': [v[0] for v in comparison.values()],
            'With_Regime': [v[1] for v in comparison.values()]
        })
        comparison_df.to_csv(save_dir / 'regime_comparison.csv', index=False)
        print(f"\n  ✓ Comparison saved to {save_dir / 'regime_comparison.csv'}")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if aggregate['mean_sharpe'] > 0.8:
        print("\n✅ EXCELLENT! Model is robust across time")
        print(f"   Mean Sharpe {aggregate['mean_sharpe']:.2f} exceeds 0.8")
    elif aggregate['mean_sharpe'] > 0.5:
        print("\n✅ GOOD! Model performs consistently")
        print(f"   Mean Sharpe {aggregate['mean_sharpe']:.2f} is solid")
    elif aggregate['mean_sharpe'] > 0.3:
        print("\n⚠️ MODERATE performance")
        print(f"   Mean Sharpe {aggregate['mean_sharpe']:.2f} is below target")
    else:
        print("\n❌ WEAK! Model not robust across time")
        print(f"   Mean Sharpe {aggregate['mean_sharpe']:.2f} is too low")

    # Compare with static split
    static_sharpe = 1.28
    print(f"\n📈 Comparison:")
    print(f"  Static split Sharpe:      {static_sharpe:.2f}")
    print(f"  Walk-forward mean Sharpe: {aggregate['mean_sharpe']:.2f}")

    degradation = (static_sharpe - aggregate['mean_sharpe']) / static_sharpe * 100
    print(f"  Degradation: {degradation:.1f}%")

    if degradation < 20:
        print("  → ✅ Excellent! Degradation < 20%")
    elif degradation < 40:
        print("  → ⚠️ Moderate degradation (20-40%)")
    else:
        print("  → ❌ High degradation (> 40%) - possible overfit to test period")

    # Save results
    print("\n[SAVING] Generating visualizations...")
    validator.create_visualizations()
    validator.save_results()

    print("\n" + "="*70)
    print("✓ Walk-forward validation complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Walk-forward validation for trading model'
    )
    parser.add_argument('--train', type=int, default=24,
                       help='Training window size (months), default: 24')
    parser.add_argument('--test', type=int, default=1,
                       help='Test window size (months), default: 1')
    parser.add_argument('--step', type=int, default=1,
                       help='Step size (months), default: 1')
    parser.add_argument('--compare', action='store_true',
                       help='Compare WITH vs WITHOUT regime detection')
    parser.add_argument('--regime-method', type=str, default='rule_based',
                       choices=['rule_based', 'hmm', 'ensemble'],
                       help='Regime detection method, default: rule_based')

    args = parser.parse_args()

    main(
        train_months=args.train,
        test_months=args.test,
        step_months=args.step,
        compare=args.compare,
        regime_method=args.regime_method
    )
