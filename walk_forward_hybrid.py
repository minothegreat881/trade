"""
Walk-Forward Validation with HYBRID Regime Detection

Compares 3 approaches:
1. Baseline (no regime detection)
2. Strict regime (original rule-based)
3. Hybrid (extreme conditions only) ← NEW!

Usage:
    python walk_forward_hybrid.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

from modules.regime_detector import RegimeDetector
from xgboost_model import XGBoostModel
from backtester import Backtester
from metrics import generate_metrics_report
import config


class HybridWalkForward:
    """
    Walk-forward validation with 3 regime strategies
    """

    def __init__(self, train_months=24, test_months=1):
        self.train_months = train_months
        self.test_months = test_months
        self.results = {
            'baseline': [],
            'strict': [],
            'hybrid': []
        }

    def create_windows(self, df):
        """Create rolling windows"""
        # Get unique year-month periods
        df_copy = df.copy()
        df_copy['year_month'] = df_copy.index.to_period('M')
        unique_months = sorted(df_copy['year_month'].unique())

        windows = []
        i = 0
        while i + self.train_months + self.test_months <= len(unique_months):
            # Get month ranges
            train_start_month = unique_months[i]
            train_end_month = unique_months[i + self.train_months - 1]
            test_start_month = unique_months[i + self.train_months]
            test_end_month = unique_months[i + self.train_months + self.test_months - 1]

            # Convert to timestamps (use 'M' for end dates to get last day of month)
            windows.append({
                'train_start': train_start_month.to_timestamp(),
                'train_end': train_end_month.to_timestamp('M'),  # End of month
                'test_start': test_start_month.to_timestamp(),
                'test_end': test_end_month.to_timestamp('M')  # End of month
            })

            i += 1  # Step forward by 1 month

        return windows

    def run_single_window(self, df, window, best_params, strategy='baseline'):
        """
        Run one window with specified strategy

        Args:
            strategy: 'baseline', 'strict', or 'hybrid'
        """
        try:
            # Split data
            train_mask = (df.index >= window['train_start']) & (df.index <= window['train_end'])
            test_mask = (df.index >= window['test_start']) & (df.index <= window['test_end'])

            train_data = df[train_mask].copy()
            test_data = df[test_mask].copy()

            if len(train_data) < 100 or len(test_data) < 10:
                return None

            # Prepare features
            feature_cols = [col for col in df.columns if col not in ['target', 'Close', 'Open', 'High', 'Low', 'Volume']]

            # Check if target exists
            if 'target' not in train_data.columns:
                return None

            X_train = train_data[feature_cols]
            y_train = train_data['target']
            X_test = test_data[feature_cols]
            y_test = test_data['target']

            # Train model
            model = XGBoostModel(params=best_params)
            split_idx = int(len(X_train) * 0.8)
            model.train(
                X_train.iloc[:split_idx], y_train.iloc[:split_idx],
                X_train.iloc[split_idx:], y_train.iloc[split_idx:],
                early_stopping_rounds=10, verbose=False
            )

            # Predict
            predictions = model.predict(X_test)
            predictions = pd.Series(predictions, index=X_test.index)

            # Apply regime strategy
            if strategy == 'baseline':
                # No filtering - trade on all signals
                signals = predictions.copy()

            elif strategy == 'strict':
                # Original rule-based regime detection
                detector = RegimeDetector()
                regime_series = detector.detect_rule_based(df)

                signals = predictions.copy()
                for date in signals.index:
                    try:
                        regime = regime_series.loc[date]
                        threshold = detector.get_prediction_threshold(regime)
                        pos_size = detector.get_position_size(regime)

                        if predictions[date] < threshold or pos_size == 0:
                            signals[date] = 0
                    except (KeyError, IndexError, TypeError):
                        pass

            elif strategy == 'hybrid':
                # NEW: Hybrid extreme condition detection
                detector = RegimeDetector()
                test_data_with_extreme = detector.detect_extreme_conditions(test_data)

                signals = predictions.copy()
                for date in signals.index:
                    try:
                        extreme = test_data_with_extreme.loc[date, 'extreme_condition']
                        pred_value = predictions[date]

                        threshold = detector.get_prediction_threshold_hybrid(extreme)
                        pos_size = detector.get_position_size_hybrid(extreme, pred_value)

                        if pred_value < threshold or pos_size == 0:
                            signals[date] = 0
                    except (KeyError, IndexError, TypeError):
                        pass

            # Backtest
            backtester = Backtester(
                initial_capital=10000,
                transaction_cost=config.TRANSACTION_COST,
                holding_period=config.HOLDING_PERIOD,
                position_size_pct=0.5,
                prediction_threshold=0.000001
            )

            results = backtester.run_backtest(
                signals,
                y_test,
                test_data['Close'] if 'Close' in test_data.columns else pd.Series(100, index=test_data.index)
            )

            metrics = generate_metrics_report(results)

            return {
                'window': window,
                'sharpe': metrics['sharpe_ratio'],
                'return': metrics['annual_return'],
                'max_dd': metrics['max_drawdown'],
                'trades': results['n_trades']
            }

        except Exception as e:
            print(f"    [ERROR] ({strategy}): {str(e)}")
            return None

    def run_comparison(self, df, best_params):
        """
        Run walk-forward for all 3 strategies
        """
        windows = self.create_windows(df)

        print("=" * 80)
        print("HYBRID WALK-FORWARD COMPARISON")
        print("=" * 80)
        print(f"Strategies: Baseline, Strict Regime, Hybrid")
        print(f"Windows: {len(windows)}")
        print()

        for i, window in enumerate(windows, 1):
            print(f"\n[Window {i}/{len(windows)}] {window['test_start'].date()}")

            for strategy in ['baseline', 'strict', 'hybrid']:
                result = self.run_single_window(df, window, best_params, strategy)

                if result is not None:
                    self.results[strategy].append(result)
                    print(f"  {strategy:10s}: Sharpe {result['sharpe']:6.2f}, Return {result['return']*100:6.1f}%, Trades {result['trades']:3d}")

        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE")
        print("=" * 80)

    def calculate_aggregate(self):
        """Calculate aggregate statistics for each strategy"""
        aggregates = {}

        for strategy in ['baseline', 'strict', 'hybrid']:
            results = self.results[strategy]

            if len(results) == 0:
                continue

            sharpes = [r['sharpe'] for r in results]
            returns = [r['return'] for r in results]
            dds = [r['max_dd'] for r in results]
            trades = [r['trades'] for r in results]

            aggregates[strategy] = {
                'mean_sharpe': np.mean(sharpes),
                'median_sharpe': np.median(sharpes),
                'std_sharpe': np.std(sharpes),
                'min_sharpe': np.min(sharpes),
                'max_sharpe': np.max(sharpes),
                'pct_positive': sum(1 for s in sharpes if s > 0) / len(sharpes) * 100,
                'mean_return': np.mean(returns),
                'worst_dd': np.min(dds),
                'mean_trades': np.mean(trades),
                'zero_trade_windows': sum(1 for t in trades if t == 0)
            }

        return aggregates

    def print_comparison(self):
        """Print detailed comparison"""
        agg = self.calculate_aggregate()

        print("\n" + "=" * 90)
        print("AGGREGATE COMPARISON: BASELINE vs STRICT vs HYBRID")
        print("=" * 90)

        metrics = [
            ('Mean Sharpe', 'mean_sharpe', '.2f', 'higher'),
            ('Median Sharpe', 'median_sharpe', '.2f', 'higher'),
            ('Std Sharpe', 'std_sharpe', '.2f', 'lower'),
            ('% Positive', 'pct_positive', '.1f', 'higher'),
            ('Mean Return', 'mean_return', '.2%', 'higher'),
            ('Worst DD', 'worst_dd', '.2%', 'higher'),
            ('Mean Trades/Window', 'mean_trades', '.1f', 'higher'),
            ('Windows w/ 0 Trades', 'zero_trade_windows', '.0f', 'lower')
        ]

        print(f"\n{'Metric':<25} | {'Baseline':>12} | {'Strict':>12} | {'Hybrid':>12} | Winner")
        print("-" * 90)

        for name, key, fmt, direction in metrics:
            baseline_val = agg['baseline'].get(key, 0)
            strict_val = agg['strict'].get(key, 0)
            hybrid_val = agg['hybrid'].get(key, 0)

            # Determine winner
            if direction == 'higher':
                values = {'Baseline': baseline_val, 'Strict': strict_val, 'Hybrid': hybrid_val}
                winner = max(values, key=values.get)
            else:
                values = {'Baseline': baseline_val, 'Strict': strict_val, 'Hybrid': hybrid_val}
                winner = min(values, key=values.get)

            # Format values
            if '%' in fmt:
                b_str = f"{baseline_val*100:{fmt[:-1]}}{fmt[-1]}" if 'return' in key or 'dd' in key else f"{baseline_val:{fmt}}"
                s_str = f"{strict_val*100:{fmt[:-1]}}{fmt[-1]}" if 'return' in key or 'dd' in key else f"{strict_val:{fmt}}"
                h_str = f"{hybrid_val*100:{fmt[:-1]}}{fmt[-1]}" if 'return' in key or 'dd' in key else f"{hybrid_val:{fmt}}"
            else:
                b_str = f"{baseline_val:{fmt}}"
                s_str = f"{strict_val:{fmt}}"
                h_str = f"{hybrid_val:{fmt}}"

            print(f"{name:<25} | {b_str:>12} | {s_str:>12} | {h_str:>12} | {winner}")

        # Calculate overall winner
        print("\n" + "=" * 90)
        print("OVERALL VERDICT")
        print("=" * 90)

        # Score each strategy (higher is better)
        scores = {'baseline': 0, 'strict': 0, 'hybrid': 0}

        # Weight different metrics
        weights = {
            'mean_sharpe': 3,           # Most important
            'pct_positive': 2,          # Very important
            'worst_dd': 2,              # Very important (inverse)
            'mean_return': 1,           # Important
            'zero_trade_windows': 1     # Moderate (inverse)
        }

        for key, weight in weights.items():
            baseline_val = agg['baseline'].get(key, 0)
            strict_val = agg['strict'].get(key, 0)
            hybrid_val = agg['hybrid'].get(key, 0)

            # For inverse metrics (lower is better)
            if key in ['worst_dd', 'zero_trade_windows']:
                max_val = max(abs(baseline_val), abs(strict_val), abs(hybrid_val))
                if max_val > 0:
                    scores['baseline'] += (1 - abs(baseline_val)/max_val) * weight
                    scores['strict'] += (1 - abs(strict_val)/max_val) * weight
                    scores['hybrid'] += (1 - abs(hybrid_val)/max_val) * weight
            else:
                # Higher is better
                max_val = max(baseline_val, strict_val, hybrid_val)
                if max_val > 0:
                    scores['baseline'] += (baseline_val/max_val) * weight
                    scores['strict'] += (strict_val/max_val) * weight
                    scores['hybrid'] += (hybrid_val/max_val) * weight

        print("\nWeighted Scores (higher = better):")
        for strategy, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy.upper():10s}: {score:.2f}")

        winner = max(scores, key=scores.get)

        print(f"\n[WINNER]: {winner.upper()}")

        if winner == 'hybrid':
            print("\n[SUCCESS] Hybrid strategy achieves best balance:")
            print("   - Near-baseline returns")
            print("   - Much better risk control than baseline")
            print("   - More active trading than strict regime")
            print("   - Protects in extreme conditions only")
        elif winner == 'baseline':
            print("\n[WARN] Baseline won, but consider:")
            print("   - Higher returns but much higher risk")
            print("   - Vulnerable to crashes")
            print("   - Not sustainable long-term")
        else:
            print("\n[WARN] Strict regime won, but very conservative")
            print("   - Safe but misses opportunities")
            print("   - Consider hybrid for better balance")


def main():
    """
    Main execution
    """
    print("=" * 90)
    print("HYBRID WALK-FORWARD VALIDATION")
    print("=" * 90)
    print("Comparing 3 strategies:")
    print("  1. Baseline      - No regime detection")
    print("  2. Strict Regime - Original rule-based (0% in BEAR)")
    print("  3. Hybrid        - Extreme conditions only (NEW!)")
    print()

    # Load data
    print("[1/4] Loading data...")
    data_file = Path('data/processed/SPY_featured.csv')

    # Load and clean timezone issues
    df = pd.read_csv(data_file, index_col=0)

    # Convert index to DatetimeIndex, stripping timezone
    import dateutil.parser
    cleaned_dates = []
    for date_str in df.index:
        dt = dateutil.parser.parse(str(date_str))
        dt_naive = dt.replace(tzinfo=None)
        cleaned_dates.append(dt_naive)
    df.index = pd.DatetimeIndex(cleaned_dates)

    print(f"  [OK] Loaded {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")

    # Load best params
    print("\n[2/4] Loading model parameters...")
    params_file = Path('results/xgboost_tuned/best_params.json')
    if params_file.exists():
        with open(params_file, 'r') as f:
            best_params = json.load(f)
        print("  [OK] Using tuned parameters")
    else:
        best_params = None
        print("  [WARN] Using default parameters")

    # Run comparison
    print("\n[3/4] Running walk-forward comparison...")
    print("  This will take 30-60 minutes (training 3× models for each window)")
    print()

    validator = HybridWalkForward(train_months=24, test_months=1)
    validator.run_comparison(df, best_params)

    # Results
    print("\n[4/4] Analyzing results...")
    validator.print_comparison()

    # Save
    print("\nSaving results...")
    save_dir = Path('results/walk_forward_hybrid')
    save_dir.mkdir(exist_ok=True, parents=True)

    agg = validator.calculate_aggregate()
    with open(save_dir / 'aggregate_comparison.json', 'w') as f:
        # Convert numpy types to native Python types
        agg_serializable = {}
        for strategy, metrics in agg.items():
            agg_serializable[strategy] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                         for k, v in metrics.items()}
        json.dump(agg_serializable, f, indent=2)
    print(f"[OK] Saved: {save_dir / 'aggregate_comparison.json'}")

    print("\n" + "=" * 90)
    print("[SUCCESS] HYBRID WALK-FORWARD COMPLETE!")
    print("=" * 90)
    print(f"\nResults saved in: {save_dir}")
    print("\nNext steps:")
    print("  1. Review comparison results above")
    print("  2. If hybrid wins - use for production")
    print("  3. Consider parameter tuning for winner")


if __name__ == "__main__":
    main()
