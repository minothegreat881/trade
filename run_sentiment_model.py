"""
XGBoost Model with Sentiment Features - Phase 4

Complete pipeline:
1. Load train/test data with technical + sentiment features
2. Train XGBoost with tuned hyperparameters
3. Generate predictions
4. Run backtest
5. Calculate metrics
6. Compare with previous models (Ridge, XGBoost, XGBoost Tuned)
7. Generate visualizations

Expected improvement: Sharpe 0.42 → 0.55-0.70 (+30-65%)

Usage:
    python run_sentiment_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import codecs
import logging

# Fix Windows encoding
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from xgboost_model import XGBoostModel
from model_comparator import ModelComparator
from backtester import Backtester
from metrics import generate_metrics_report, print_metrics_summary
import visualizer
import config

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_previous_results():
    """
    Load results from previous models for comparison.

    Returns:
        dict with model names as keys and (results, metrics) as values
    """
    logger.info("Loading previous model results...")

    previous_models = {}

    # 1. Baseline (Ridge)
    try:
        baseline_dir = Path('results/baseline')
        baseline_metrics = pd.read_csv(baseline_dir / 'metrics.csv').iloc[0].to_dict()
        baseline_equity = pd.read_csv(baseline_dir / 'equity_curve.csv',
                                      index_col=0, parse_dates=True)

        baseline_results = {
            'equity_curve': baseline_equity['strategy'],
            'initial_capital': 10000,
            'final_capital': baseline_equity['strategy'].iloc[-1],
            'benchmark': baseline_equity['benchmark']
        }

        previous_models['Ridge (Baseline)'] = (baseline_results, baseline_metrics)
        logger.info(f"  Ridge: Sharpe {baseline_metrics['sharpe_ratio']:.2f}")

    except FileNotFoundError:
        logger.warning("  Baseline results not found, skipping")

    # 2. XGBoost (Tuned)
    try:
        xgb_dir = Path('results/xgboost_tuned')
        xgb_metrics = pd.read_csv(xgb_dir / 'metrics.csv').iloc[0].to_dict()
        xgb_equity = pd.read_csv(xgb_dir / 'equity_curve.csv',
                                 index_col=0, parse_dates=True)

        xgb_results = {
            'equity_curve': xgb_equity['strategy'],
            'initial_capital': 10000,
            'final_capital': xgb_equity['strategy'].iloc[-1],
            'benchmark': xgb_equity['benchmark']
        }

        previous_models['XGBoost (Tuned)'] = (xgb_results, xgb_metrics)
        logger.info(f"  XGBoost Tuned: Sharpe {xgb_metrics['sharpe_ratio']:.2f}")

    except FileNotFoundError:
        logger.warning("  XGBoost tuned results not found, skipping")

    return previous_models


def main():
    """
    Main execution pipeline.
    """
    print("\n" + "=" * 70)
    print("XGBOOST MODEL WITH SENTIMENT - PHASE 4")
    print("=" * 70)
    print("\nTarget: Sharpe 0.55-0.70 (+30-65% over current best 0.42)")
    print("Features: Technical (11) + Sentiment (15) = 26 total")
    print("=" * 70 + "\n")

    try:
        # ===== 1. LOAD DATA WITH SENTIMENT =====
        print("\n" + "=" * 70)
        print("[1/7] LOADING DATA WITH SENTIMENT")
        print("=" * 70)

        data_dir = Path('data/train_test_sentiment')

        train_X = pd.read_csv(data_dir / 'train_X.csv',
                              index_col=0, parse_dates=True)
        train_y = pd.read_csv(data_dir / 'train_y.csv',
                              index_col=0, parse_dates=True)['target']
        test_X = pd.read_csv(data_dir / 'test_X.csv',
                             index_col=0, parse_dates=True)
        test_y = pd.read_csv(data_dir / 'test_y.csv',
                             index_col=0, parse_dates=True)['target']

        # Load prices for backtest
        original_test_data = pd.read_csv('data/train_test/test_data.csv', index_col=0)
        original_test_data.index = pd.to_datetime(original_test_data.index, utc=True).tz_localize(None).normalize()
        test_prices = original_test_data['Close']

        # Align prices with test_X dates (some may have been dropped)
        test_prices = test_prices.loc[test_X.index]

        logger.info(f"Train set: {len(train_X)} samples, {len(train_X.columns)} features")
        logger.info(f"Test set: {len(test_X)} samples, {len(test_X.columns)} features")
        logger.info(f"Date range: {train_X.index.min().date()} to {test_X.index.max().date()}")

        # Count feature types
        technical_features = ['return_1d', 'return_5d', 'return_20d', 'volatility_20d',
                             'rsi_14', 'macd', 'macd_signal', 'bb_position',
                             'volume_ratio', 'atr_14', 'obv_change']
        sentiment_features = [col for col in train_X.columns if col not in technical_features]

        logger.info(f"\nFeature breakdown:")
        logger.info(f"  Technical: {len(technical_features)} features")
        logger.info(f"  Sentiment: {len(sentiment_features)} features")

        # Split train into train/val for early stopping
        split_idx = int(len(train_X) * 0.8)
        X_train = train_X.iloc[:split_idx]
        y_train = train_y.iloc[:split_idx]
        X_val = train_X.iloc[split_idx:]
        y_val = train_y.iloc[split_idx:]

        logger.info(f"\nAfter split:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Validation: {len(X_val)} samples")
        logger.info(f"  Test: {len(test_X)} samples")

        # ===== 2. CONFIGURE MODEL WITH TUNED PARAMS =====
        print("\n" + "=" * 70)
        print("[2/7] CONFIGURING XGBOOST MODEL")
        print("=" * 70)

        # Use tuned hyperparameters from Phase 3.5
        tuned_params = {
            'objective': 'reg:squarederror',
            'max_depth': 4,              # Tuned
            'learning_rate': 0.1,        # Tuned
            'n_estimators': 100,         # Tuned
            'min_child_weight': 5,       # Tuned
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42
        }

        logger.info("Using tuned hyperparameters from Phase 3.5:")
        logger.info(f"  max_depth: {tuned_params['max_depth']}")
        logger.info(f"  learning_rate: {tuned_params['learning_rate']}")
        logger.info(f"  n_estimators: {tuned_params['n_estimators']}")
        logger.info(f"  min_child_weight: {tuned_params['min_child_weight']}")

        # ===== 3. TRAIN MODEL =====
        print("\n" + "=" * 70)
        print("[3/7] TRAINING XGBOOST WITH SENTIMENT")
        print("=" * 70)

        model = XGBoostModel(params=tuned_params)
        model.train(X_train, y_train, X_val, y_val, early_stopping_rounds=10)

        # Evaluate on test
        test_eval = model.evaluate(test_X, test_y)
        print(f"\nTest Set Performance:")
        print(f"  R²: {test_eval['r2']:.4f}")
        print(f"  MAE: {test_eval['mae']:.6f}")
        print(f"  RMSE: {test_eval['rmse']:.6f}")

        # ===== 4. GENERATE PREDICTIONS =====
        print("\n" + "=" * 70)
        print("[4/7] GENERATING PREDICTIONS")
        print("=" * 70)

        test_pred = model.predict(test_X)
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

        backtest_results = backtester.run_backtest(test_pred, test_y, test_prices)

        # Calculate benchmark
        benchmark = backtester.calculate_benchmark(test_prices)
        backtest_results['benchmark'] = benchmark

        # ===== 6. CALCULATE METRICS =====
        print("\n" + "=" * 70)
        print("[6/7] CALCULATING METRICS")
        print("=" * 70)

        metrics = generate_metrics_report(backtest_results)
        print_metrics_summary(metrics)

        # ===== 7. COMPARE WITH PREVIOUS MODELS =====
        print("\n" + "=" * 70)
        print("[7/7] COMPARING WITH PREVIOUS MODELS")
        print("=" * 70)

        # Load previous results
        previous_models = load_previous_results()

        # Create comparator
        comparator = ModelComparator()

        # Add previous models
        for model_name, (results, prev_metrics) in previous_models.items():
            comparator.add_model(model_name, None, results, prev_metrics)

        # Add current model
        comparator.add_model('XGBoost + Sentiment', model, backtest_results, metrics)

        # Print comparison
        comparator.print_comparison_summary()

        # Calculate improvements
        if 'XGBoost (Tuned)' in previous_models:
            _, prev_best_metrics = previous_models['XGBoost (Tuned)']
            sharpe_improvement = metrics['sharpe_ratio'] - prev_best_metrics['sharpe_ratio']
            sharpe_pct_improvement = (sharpe_improvement / prev_best_metrics['sharpe_ratio']) * 100

            return_improvement = metrics['annual_return'] - prev_best_metrics['annual_return']

            print("\n" + "=" * 70)
            print("IMPROVEMENT OVER PREVIOUS BEST (XGBoost Tuned)")
            print("=" * 70)
            print(f"Sharpe Ratio: {prev_best_metrics['sharpe_ratio']:.2f} → "
                  f"{metrics['sharpe_ratio']:.2f} "
                  f"({sharpe_improvement:+.2f}, {sharpe_pct_improvement:+.1f}%)")
            print(f"Annual Return: {prev_best_metrics['annual_return']*100:.2f}% → "
                  f"{metrics['annual_return']*100:.2f}% "
                  f"({return_improvement*100:+.2f}%)")

        # ===== 8. SAVE RESULTS =====
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)

        results_dir = Path('results/xgboost_sentiment')
        results_dir.mkdir(exist_ok=True, parents=True)

        # Save equity curve
        equity_df = pd.DataFrame({
            'strategy': backtest_results['equity_curve'],
            'benchmark': backtest_results['benchmark']
        })
        equity_df.to_csv(results_dir / 'equity_curve.csv')
        logger.info("  Saved equity_curve.csv")

        # Save trades
        if len(backtest_results['trades']) > 0:
            backtest_results['trades'].to_csv(results_dir / 'trades.csv', index=False)
            logger.info("  Saved trades.csv")

        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(results_dir / 'metrics.csv', index=False)
        logger.info("  Saved metrics.csv")

        # Save feature importance
        feature_importance = model.get_feature_importance()
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
            backtest_results,
            benchmark=backtest_results['benchmark'],
            save_path=results_dir / 'equity_curve.png'
        )
        plt.close()

        # 2. Drawdown
        fig2 = visualizer.plot_drawdown(
            backtest_results,
            save_path=results_dir / 'drawdown.png'
        )
        plt.close()

        # 3. Feature importance
        fig3 = model.plot_feature_importance(
            top_n=15,
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
            backtest_results, metrics, model,
            benchmark=backtest_results['benchmark'],
            save_path=results_dir / 'tearsheet.png'
        )
        plt.close()

        logger.info(f"\nAll visualizations saved to {results_dir}/")

        # ===== FINAL SUMMARY =====
        print("\n" + "=" * 70)
        print("🎉 SENTIMENT MODEL COMPLETE!")
        print("=" * 70)

        print(f"\nXGBoost + Sentiment Performance:")
        print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Annual Return:   {metrics['annual_return']*100:>8.2f}%")
        print(f"  Max Drawdown:    {metrics['max_drawdown_pct']:>8.2f}%")
        print(f"  Win Rate:        {metrics['win_rate']*100:>8.1f}%")
        print(f"  Profit Factor:   {metrics['profit_factor']:>8.2f}")
        print(f"  Total Trades:    {metrics['total_trades']:>8d}")

        if 'XGBoost (Tuned)' in previous_models:
            _, prev_best = previous_models['XGBoost (Tuned)']
            print(f"\nImprovement over XGBoost (Tuned):")
            print(f"  Sharpe:  {prev_best['sharpe_ratio']:.2f} → "
                  f"{metrics['sharpe_ratio']:.2f} "
                  f"({sharpe_improvement:+.2f})")
            print(f"  Return:  {prev_best['annual_return']*100:.2f}% → "
                  f"{metrics['annual_return']*100:.2f}% "
                  f"({return_improvement*100:+.2f}%)")

        print(f"\nAll results saved to: {results_dir}/")
        print("\nKey files:")
        print("  - tearsheet.png (complete performance overview)")
        print("  - comparison_equity_curves.png (all models)")
        print("  - comparison_metrics.png (metric comparison)")
        print("  - feature_importance.png (top 15 features)")
        print("  - model_comparison.csv (detailed comparison)")

        # Assessment
        print("\n" + "=" * 70)
        print("ASSESSMENT")
        print("=" * 70)

        if metrics['sharpe_ratio'] > 0.70:
            print("\n⭐⭐⭐ OUTSTANDING! Sharpe > 0.70 achieved!")
            print("Exceeded target! System ready for live testing.")
        elif metrics['sharpe_ratio'] > 0.55:
            print("\n✓ EXCELLENT! Target achieved (Sharpe 0.55-0.70)")
            print("Sentiment data significantly improved performance!")
            print("Phase 4 complete. System ready for validation.")
        elif metrics['sharpe_ratio'] > 0.45:
            print("\n✓ GOOD improvement. Close to target.")
            print("Sentiment features added value. Consider:")
            print("  1. Additional sentiment sources (news, social media)")
            print("  2. Feature selection (remove weak features)")
            print("  3. Alternative feature engineering approaches")
        else:
            print("\n⚠️ LIMITED improvement from sentiment. Recommendations:")
            print("  1. Review feature importance (which sentiment features matter?)")
            print("  2. Check correlation between sentiment and returns")
            print("  3. Consider regime-based models (different strategies for different conditions)")
            print("  4. May need different sentiment sources or higher quality data")

        print("=" * 70 + "\n")

        # Show top sentiment features
        print("\n" + "=" * 70)
        print("TOP SENTIMENT FEATURES")
        print("=" * 70)

        top_15 = model.get_feature_importance(top_n=15)
        sentiment_in_top = top_15[top_15['feature'].isin(sentiment_features)]

        if len(sentiment_in_top) > 0:
            print(f"\n{len(sentiment_in_top)} sentiment features in top 15:")
            for i, (idx, row) in enumerate(sentiment_in_top.iterrows(), 1):
                print(f"  {i}. {row['feature']:30s} {row['importance']:.4f}")
        else:
            print("\nNo sentiment features in top 15.")
            print("This suggests sentiment may not be adding significant value.")

        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Make sure you've run integrate_sentiment.py first!")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
