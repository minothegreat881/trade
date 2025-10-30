"""
XGBoost Model Execution Script - Phase 3

Complete pipeline:
1. Load train/test data
2. (Optional) Hyperparameter tuning
3. Train XGBoost model
4. Generate predictions
5. Run backtest
6. Calculate metrics
7. Compare with baseline (Ridge)
8. Generate visualizations

Usage:
    python run_xgboost.py              # Use default params
    python run_xgboost.py --tune       # Hyperparameter tuning first

Expected: Sharpe 0.09 → 0.4-0.6 (4-6× improvement!)
"""

import pandas as pd
import numpy as np
import json
import argparse
import sys
import codecs
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Import project modules
from xgboost_model import XGBoostModel
from hyperparameter_tuner import XGBoostTuner
from model_comparator import ModelComparator
from baseline_model import BaselineModel
from backtester import Backtester
from metrics import generate_metrics_report, print_metrics_summary
import visualizer
import config
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_baseline_results():
    """
    Load baseline (Ridge) results for comparison.

    Returns:
        dict with baseline_results and baseline_metrics
    """
    logger.info("Loading baseline (Ridge) results...")

    baseline_dir = Path('results/baseline')

    # Load metrics from CSV
    metrics_csv = pd.read_csv(baseline_dir / 'metrics.csv').iloc[0].to_dict()

    # Load equity curve
    equity_df = pd.read_csv(baseline_dir / 'equity_curve.csv',
                            index_col=0, parse_dates=True)

    baseline_results = {
        'equity_curve': equity_df['strategy'],
        'initial_capital': 10000,
        'final_capital': equity_df['strategy'].iloc[-1],
        'benchmark': equity_df['benchmark']
    }

    logger.info(f"  Baseline Sharpe: {metrics_csv['sharpe_ratio']:.2f}")
    logger.info(f"  Baseline Return: {metrics_csv['total_return']*100:.2f}%")

    return baseline_results, metrics_csv


def main(tune_hyperparameters=False):
    """
    Main execution pipeline.

    Args:
        tune_hyperparameters: If True, run hyperparameter tuning first
    """
    print("\n" + "=" * 70)
    print("XGBOOST MODEL - PHASE 3")
    print("=" * 70)
    print("\nTarget: Sharpe 0.4-0.6 (4-6× improvement over baseline)")
    print("=" * 70 + "\n")

    try:
        # ===== 1. LOAD DATA =====
        print("\n" + "=" * 70)
        print("[1/7] LOADING DATA")
        print("=" * 70)

        train_X = pd.read_csv('data/train_test/train_X.csv',
                              index_col=0, parse_dates=True)
        train_y = pd.read_csv('data/train_test/train_y.csv',
                              index_col=0, parse_dates=True)['target']
        test_X = pd.read_csv('data/train_test/test_X.csv',
                             index_col=0, parse_dates=True)
        test_y = pd.read_csv('data/train_test/test_y.csv',
                             index_col=0, parse_dates=True)['target']

        # Load full data for prices
        train_full = pd.read_csv('data/train_test/train_data.csv',
                                 index_col=0, parse_dates=True)
        test_full = pd.read_csv('data/train_test/test_data.csv',
                                index_col=0, parse_dates=True)

        test_prices = test_full['Close']

        logger.info(f"Train set: {len(train_X)} samples")
        logger.info(f"Test set: {len(test_X)} samples")
        logger.info(f"Features: {len(train_X.columns)}")

        # Split train into train/val for early stopping
        split_idx = int(len(train_X) * 0.8)
        X_train = train_X.iloc[:split_idx]
        y_train = train_y.iloc[:split_idx]
        X_val = train_X.iloc[split_idx:]
        y_val = train_y.iloc[split_idx:]

        logger.info(f"After split:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Validation: {len(X_val)} samples")
        logger.info(f"  Test: {len(test_X)} samples")

        # ===== 2. HYPERPARAMETER TUNING (OPTIONAL) =====
        best_params = None

        if tune_hyperparameters:
            print("\n" + "=" * 70)
            print("[2/7] HYPERPARAMETER TUNING")
            print("=" * 70)
            print("This will take 10-30 minutes...")

            tuner = XGBoostTuner(n_splits=3)

            param_grid = {
                'max_depth': [2, 3, 4],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200],
                'min_child_weight': [3, 5, 7]
            }

            best_params = tuner.tune(train_X, train_y, param_grid)

            # Save tuning results
            results_dir = Path('results/xgboost')
            results_dir.mkdir(exist_ok=True, parents=True)

            results_df = tuner.get_results_df()
            results_df.to_csv(results_dir / 'tuning_results.csv', index=False)
            logger.info(f"  Tuning results saved to {results_dir}/tuning_results.csv")

        else:
            print("\n" + "=" * 70)
            print("[2/7] USING DEFAULT PARAMETERS")
            print("=" * 70)
            print("(Use --tune flag for hyperparameter optimization)")
            print("Default params: max_depth=3, learning_rate=0.05, n_estimators=100")

        # ===== 3. TRAIN XGBOOST =====
        print("\n" + "=" * 70)
        print("[3/7] TRAINING XGBOOST MODEL")
        print("=" * 70)

        xgb_model = XGBoostModel(params=best_params)
        xgb_model.train(X_train, y_train, X_val, y_val, early_stopping_rounds=10)

        # Evaluate on test
        test_eval = xgb_model.evaluate(test_X, test_y)
        print(f"\nTest Set Performance:")
        print(f"  R²: {test_eval['r2']:.4f}")
        print(f"  MAE: {test_eval['mae']:.6f}")
        print(f"  RMSE: {test_eval['rmse']:.6f}")

        # ===== 4. GENERATE PREDICTIONS =====
        print("\n" + "=" * 70)
        print("[4/7] GENERATING PREDICTIONS")
        print("=" * 70)

        test_pred = xgb_model.predict(test_X)
        logger.info(f"Predictions generated: {len(test_pred)} samples")
        logger.info(f"  Mean prediction: {test_pred.mean():.6f}")
        logger.info(f"  Std prediction: {test_pred.std():.6f}")
        logger.info(f"  Positive predictions: {(test_pred > 0).sum()} / {len(test_pred)}")

        # ===== 5. BACKTEST =====
        print("\n" + "=" * 70)
        print("[5/7] RUNNING BACKTEST")
        print("=" * 70)

        backtester = Backtester(
            initial_capital=config.INITIAL_CAPITAL,
            transaction_cost=config.TRANSACTION_COST,
            holding_period=config.HOLDING_PERIOD,
            position_size_pct=0.5,
            prediction_threshold=0.001
        )

        xgb_backtest_results = backtester.run_backtest(test_pred, test_y, test_prices)

        # Calculate benchmark
        benchmark = backtester.calculate_benchmark(test_prices)
        xgb_backtest_results['benchmark'] = benchmark

        # ===== 6. CALCULATE METRICS =====
        print("\n" + "=" * 70)
        print("[6/7] CALCULATING METRICS")
        print("=" * 70)

        xgb_metrics = generate_metrics_report(xgb_backtest_results)
        print_metrics_summary(xgb_metrics)

        # ===== 7. COMPARE WITH BASELINE =====
        print("\n" + "=" * 70)
        print("[7/7] COMPARING WITH BASELINE")
        print("=" * 70)

        # Load baseline results
        baseline_results, baseline_metrics = load_baseline_results()

        # Create comparator
        comparator = ModelComparator()
        comparator.add_model('Ridge (Baseline)', None, baseline_results, baseline_metrics)
        comparator.add_model('XGBoost', xgb_model, xgb_backtest_results, xgb_metrics)

        # Print comparison
        comparator.print_comparison_summary()

        # Calculate improvements
        sharpe_improvement = xgb_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']
        return_improvement = xgb_metrics['annual_return'] - baseline_metrics['annual_return']

        print("\n" + "=" * 70)
        print("KEY IMPROVEMENTS")
        print("=" * 70)
        print(f"Sharpe Ratio: {baseline_metrics['sharpe_ratio']:.2f} → "
              f"{xgb_metrics['sharpe_ratio']:.2f} "
              f"({sharpe_improvement:+.2f})")
        print(f"Annual Return: {baseline_metrics['annual_return']*100:.2f}% → "
              f"{xgb_metrics['annual_return']*100:.2f}% "
              f"({return_improvement*100:+.2f}%)")

        # ===== 8. SAVE RESULTS =====
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)

        results_dir = Path('results/xgboost')
        results_dir.mkdir(exist_ok=True, parents=True)

        # Save equity curve
        equity_df = pd.DataFrame({
            'strategy': xgb_backtest_results['equity_curve'],
            'benchmark': xgb_backtest_results['benchmark']
        })
        equity_df.to_csv(results_dir / 'equity_curve.csv')
        logger.info("  Saved equity_curve.csv")

        # Save trades
        if len(xgb_backtest_results['trades']) > 0:
            xgb_backtest_results['trades'].to_csv(results_dir / 'trades.csv', index=False)
            logger.info("  Saved trades.csv")

        # Save metrics
        metrics_df = pd.DataFrame([xgb_metrics])
        metrics_df.to_csv(results_dir / 'metrics.csv', index=False)
        logger.info("  Saved metrics.csv")

        # Save feature importance
        feature_importance = xgb_model.get_feature_importance()
        feature_importance.to_csv(results_dir / 'feature_importance.csv', index=False)
        logger.info("  Saved feature_importance.csv")

        # Save comparison table
        comparison_table = comparator.create_comparison_table()
        comparison_table.to_csv(results_dir / 'model_comparison.csv', index=False)
        logger.info("  Saved model_comparison.csv")

        # ===== 9. GENERATE VISUALIZATIONS =====
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        import matplotlib.pyplot as plt

        # 1. Equity curve
        fig1 = visualizer.plot_equity_curve(
            xgb_backtest_results,
            benchmark=xgb_backtest_results['benchmark'],
            save_path=results_dir / 'equity_curve.png'
        )
        plt.close()

        # 2. Drawdown
        fig2 = visualizer.plot_drawdown(
            xgb_backtest_results,
            save_path=results_dir / 'drawdown.png'
        )
        plt.close()

        # 3. Feature importance
        fig3 = xgb_model.plot_feature_importance(
            top_n=10,
            save_path=results_dir / 'feature_importance.png'
        )
        plt.close()

        # 4. Comparison: Equity curves
        fig4 = comparator.plot_equity_curves(
            save_path=results_dir / 'comparison_equity_curves.png'
        )
        plt.close()

        # 5. Comparison: Metrics
        fig5 = comparator.plot_metrics_comparison(
            save_path=results_dir / 'comparison_metrics.png'
        )
        plt.close()

        # 6. Complete tearsheet
        fig6 = visualizer.generate_tearsheet(
            xgb_backtest_results, xgb_metrics, xgb_model,
            benchmark=xgb_backtest_results['benchmark'],
            save_path=results_dir / 'tearsheet.png'
        )
        plt.close()

        logger.info(f"\nAll visualizations saved to {results_dir}/")

        # ===== FINAL SUMMARY =====
        print("\n" + "=" * 70)
        print("🎉 XGBOOST MODEL COMPLETE!")
        print("=" * 70)

        print(f"\nXGBoost Performance:")
        print(f"  Sharpe Ratio:    {xgb_metrics['sharpe_ratio']:>8.2f}")
        print(f"  Annual Return:   {xgb_metrics['annual_return']*100:>8.2f}%")
        print(f"  Max Drawdown:    {xgb_metrics['max_drawdown_pct']:>8.2f}%")
        print(f"  Win Rate:        {xgb_metrics['win_rate']*100:>8.1f}%")
        print(f"  Profit Factor:   {xgb_metrics['profit_factor']:>8.2f}")
        print(f"  Total Trades:    {xgb_metrics['total_trades']:>8d}")

        print(f"\nImprovement over Baseline:")
        print(f"  Sharpe:  {baseline_metrics['sharpe_ratio']:.2f} → "
              f"{xgb_metrics['sharpe_ratio']:.2f} "
              f"({sharpe_improvement:+.2f}x)")
        print(f"  Return:  {baseline_metrics['annual_return']*100:.2f}% → "
              f"{xgb_metrics['annual_return']*100:.2f}% "
              f"({return_improvement*100:+.2f}%)")

        print(f"\nAll results saved to: {results_dir}/")
        print("\nKey files:")
        print("  - tearsheet.png (complete performance overview)")
        print("  - comparison_equity_curves.png (Ridge vs XGBoost)")
        print("  - comparison_metrics.png (metric comparison)")
        print("  - feature_importance.png (top features)")
        print("  - model_comparison.csv (detailed comparison)")

        # Assessment
        print("\n" + "=" * 70)
        print("ASSESSMENT")
        print("=" * 70)

        if xgb_metrics['sharpe_ratio'] > 0.6:
            print("\n⭐⭐⭐ EXCELLENT! Sharpe > 0.6 achieved!")
            print("Ready for Phase 4 (Sentiment Data)")
        elif xgb_metrics['sharpe_ratio'] > 0.4:
            print("\n✓ GOOD! Solid improvement over baseline")
            print("Target achieved. Ready for Phase 4 (Sentiment Data)")
        elif xgb_metrics['sharpe_ratio'] > 0.3:
            print("\n✓ MODERATE improvement. Consider:")
            print("  1. Run with --tune flag for hyperparameter optimization")
            print("  2. More feature engineering")
            print("  3. Proceed to Phase 4 (sentiment might help)")
        else:
            print("\n⚠️ LIMITED improvement. Recommendations:")
            print("  1. MUST run with --tune flag")
            print("  2. Check feature correlations with target")
            print("  3. Review prediction distribution")
            print("  4. Consider different prediction threshold")

        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Make sure you've run pipeline.py and run_baseline.py first!")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run XGBoost trading model')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning (takes 10-30 min)')
    args = parser.parse_args()

    main(tune_hyperparameters=args.tune)
