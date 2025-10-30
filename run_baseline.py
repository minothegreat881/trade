"""
Baseline Model Execution Script

Complete pipeline:
1. Load train/test data
2. Train Ridge Regression model
3. Generate predictions
4. Run backtest
5. Calculate metrics
6. Generate visualizations
7. Save results

Based on Kelly & Xiu (2023) regularization framework.
"""

import pandas as pd
import numpy as np
import logging
import sys
import codecs
import os
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Import project modules
from baseline_model import BaselineModel
from backtester import Backtester
from metrics import generate_metrics_report, print_metrics_summary
from visualizer import generate_tearsheet, plot_equity_curve, plot_drawdown, plot_monthly_returns
import config

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """
    Load train and test datasets.

    Returns:
        tuple: (train_X, train_y, test_X, test_y, train_prices, test_prices, dates)
    """
    logger.info("Loading data...")

    # Load train data
    train_X = pd.read_csv('data/train_test/train_X.csv',
                          index_col=0, parse_dates=True)
    train_y = pd.read_csv('data/train_test/train_y.csv',
                          index_col=0, parse_dates=True)['target']

    # Load test data
    test_X = pd.read_csv('data/train_test/test_X.csv',
                         index_col=0, parse_dates=True)
    test_y = pd.read_csv('data/train_test/test_y.csv',
                         index_col=0, parse_dates=True)['target']

    # Load full data for prices
    train_full = pd.read_csv('data/train_test/train_data.csv',
                             index_col=0, parse_dates=True)
    test_full = pd.read_csv('data/train_test/test_data.csv',
                            index_col=0, parse_dates=True)

    train_prices = train_full['Close']
    test_prices = test_full['Close']

    logger.info(f"Train set: {len(train_X)} samples")
    logger.info(f"Test set: {len(test_X)} samples")
    logger.info(f"Features: {len(train_X.columns)}")

    return train_X, train_y, test_X, test_y, train_prices, test_prices


def train_model(train_X, train_y, alpha=1.0):
    """
    Train Ridge Regression baseline model.

    Args:
        train_X: Training features
        train_y: Training target
        alpha: Ridge regularization parameter

    Returns:
        BaselineModel: Trained model
    """
    logger.info("Training baseline model...")

    model = BaselineModel(alpha=alpha)
    model.train(train_X, train_y)

    # Print summary
    model.summary()

    return model


def run_backtest_on_data(model, test_X, test_y, test_prices):
    """
    Run backtest on test data.

    Args:
        model: Trained BaselineModel
        test_X: Test features
        test_y: Actual returns
        test_prices: Price data

    Returns:
        dict: Backtest results
    """
    logger.info("Running backtest on test set...")

    # Generate predictions
    predictions = model.predict(test_X)

    # Initialize backtester
    backtester = Backtester(
        initial_capital=config.INITIAL_CAPITAL,
        transaction_cost=config.TRANSACTION_COST,
        holding_period=config.HOLDING_PERIOD,
        position_size_pct=0.5,
        prediction_threshold=0.001
    )

    # Run backtest
    results = backtester.run_backtest(predictions, test_y, test_prices)

    # Calculate benchmark
    benchmark = backtester.calculate_benchmark(test_prices)
    results['benchmark'] = benchmark

    return results


def save_results(backtest_results, metrics, model):
    """
    Save all results to files.

    Args:
        backtest_results: Dict from backtester
        metrics: Dict from metrics module
        model: Trained model
    """
    logger.info("Saving results...")

    # Create results directory
    results_dir = Path('results/baseline')
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save equity curve
    equity_df = pd.DataFrame({
        'strategy': backtest_results['equity_curve'],
        'benchmark': backtest_results['benchmark']
    })
    equity_df.to_csv(results_dir / 'equity_curve.csv')
    logger.info(f"  Saved equity_curve.csv")

    # 2. Save trades
    if len(backtest_results['trades']) > 0:
        backtest_results['trades'].to_csv(results_dir / 'trades.csv', index=False)
        logger.info(f"  Saved trades.csv")

    # 3. Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / 'metrics.csv', index=False)
    logger.info(f"  Saved metrics.csv")

    # 4. Save feature importance
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': importance.index,
        'importance': importance.values
    })
    importance_df.to_csv(results_dir / 'feature_importance.csv', index=False)
    logger.info(f"  Saved feature_importance.csv")

    # 5. Save comprehensive report
    with open(results_dir / 'performance_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BASELINE MODEL PERFORMANCE REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Model Type: Ridge Regression\n")
        f.write(f"Regularization (alpha): {model.alpha}\n")
        f.write(f"Number of Features: {len(model.feature_names)}\n")
        f.write(f"Training R2: {model.train_r2:.4f}\n\n")

        f.write("BACKTEST CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Initial Capital: ${backtest_results['initial_capital']:,.0f}\n")
        f.write(f"Holding Period: {config.HOLDING_PERIOD} days\n")
        f.write(f"Transaction Cost: {config.TRANSACTION_COST*100:.2f}%\n")
        f.write(f"Position Size: 50% of capital\n\n")

        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Return:        {metrics['total_return']*100:>10.2f}%\n")
        f.write(f"Annual Return:       {metrics['annual_return']*100:>10.2f}%\n")
        f.write(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}\n")
        f.write(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}\n")
        f.write(f"Max Drawdown:        {metrics['max_drawdown_pct']:>10.2f}%\n")
        f.write(f"Return/DD Ratio:     {metrics['return_dd_ratio']:>10.2f}\n")
        f.write(f"Win Rate:            {metrics['win_rate']*100:>10.1f}%\n")
        f.write(f"Profit Factor:       {metrics['profit_factor']:>10.2f}\n")
        f.write(f"Total Trades:        {metrics['total_trades']:>10d}\n\n")

        f.write("BENCHMARK COMPARISON\n")
        f.write("-" * 70 + "\n")
        benchmark = backtest_results['benchmark']
        bench_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
        strategy_return = metrics['total_return'] * 100
        outperformance = strategy_return - bench_return

        f.write(f"Strategy Return:     {strategy_return:>10.2f}%\n")
        f.write(f"Benchmark Return:    {bench_return:>10.2f}%\n")
        f.write(f"Outperformance:      {outperformance:>10.2f}%\n\n")

        f.write("TOP 10 FEATURES\n")
        f.write("-" * 70 + "\n")
        top_features = model.get_feature_importance(top_n=10)
        coefficients = model.get_coefficients()

        for i, (feat, imp) in enumerate(top_features.items(), 1):
            coef = coefficients[feat]
            sign = "+" if coef > 0 else "-"
            f.write(f"{i:2d}. {feat:25s} ({sign}) {imp:>8.4f}\n")

    logger.info(f"  Saved performance_report.txt")
    logger.info(f"\nAll results saved to {results_dir}/")


def generate_visualizations(backtest_results, metrics, model):
    """
    Generate and save all visualizations.

    Args:
        backtest_results: Dict from backtester
        metrics: Dict from metrics module
        model: Trained model
    """
    logger.info("Generating visualizations...")

    results_dir = Path('results/baseline')
    results_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    # 1. Equity curve
    fig1 = plot_equity_curve(backtest_results,
                             benchmark=backtest_results['benchmark'],
                             save_path=results_dir / 'equity_curve.png')
    plt.close()

    # 2. Drawdown
    fig2 = plot_drawdown(backtest_results,
                         save_path=results_dir / 'drawdown.png')
    plt.close()

    # 3. Monthly returns (optional - skip if fails due to date issues)
    try:
        fig3 = plot_monthly_returns(backtest_results,
                                    save_path=results_dir / 'monthly_returns.png')
        plt.close()
    except Exception as e:
        logger.warning(f"Skipping monthly returns plot: {e}")

    # 4. Complete tearsheet
    fig4 = generate_tearsheet(backtest_results, metrics, model,
                              benchmark=backtest_results['benchmark'],
                              save_path=results_dir / 'tearsheet.png')
    plt.close()

    logger.info(f"Visualizations saved to {results_dir}/")


def main():
    """
    Main execution pipeline.
    """
    print("\n" + "=" * 70)
    print("BASELINE MODEL - RIDGE REGRESSION")
    print("=" * 70)
    print("\nBased on Kelly & Xiu (2023) regularization framework")
    print(f"Ticker: {config.TICKER}")
    print(f"Holding Period: {config.HOLDING_PERIOD} days")
    print("\n" + "=" * 70 + "\n")

    try:
        # Step 1: Load data
        train_X, train_y, test_X, test_y, train_prices, test_prices = load_data()

        # Step 2: Train model
        print("\n" + "=" * 70)
        print("STEP 1: TRAINING MODEL")
        print("=" * 70)
        model = train_model(train_X, train_y, alpha=1.0)

        # Step 3: Run backtest
        print("\n" + "=" * 70)
        print("STEP 2: BACKTESTING")
        print("=" * 70)
        backtest_results = run_backtest_on_data(model, test_X, test_y, test_prices)

        # Step 4: Calculate metrics
        print("\n" + "=" * 70)
        print("STEP 3: CALCULATING METRICS")
        print("=" * 70)
        metrics = generate_metrics_report(backtest_results)
        print_metrics_summary(metrics)

        # Step 5: Save results
        print("\n" + "=" * 70)
        print("STEP 4: SAVING RESULTS")
        print("=" * 70)
        save_results(backtest_results, metrics, model)

        # Step 6: Generate visualizations
        print("\n" + "=" * 70)
        print("STEP 5: GENERATING VISUALIZATIONS")
        print("=" * 70)
        generate_visualizations(backtest_results, metrics, model)

        # Final summary
        print("\n" + "=" * 70)
        print("BASELINE MODEL COMPLETE!")
        print("=" * 70)
        print(f"\nFinal Portfolio Value: ${backtest_results['final_capital']:,.2f}")
        print(f"Total Return: {metrics['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"\nAll results saved to: results/baseline/")
        print(f"\nKey files:")
        print(f"  - tearsheet.png (complete performance overview)")
        print(f"  - performance_report.txt (detailed metrics)")
        print(f"  - equity_curve.csv (portfolio values)")
        print(f"  - trades.csv (all executed trades)")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        logger.error("Please run pipeline.py first to generate data!")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
