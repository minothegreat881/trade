"""
XGBoost Hyperparameter Tuning Script

Uses TimeSeriesSplit cross-validation to find optimal parameters
that maximize out-of-sample Sharpe ratio.

Usage:
    python tune_xgboost.py              # Full grid search (90-180 min)
    python tune_xgboost.py --quick      # Quick search (10-30 min)
    python tune_xgboost.py --random     # Random search (30-60 min)
"""

import pandas as pd
import numpy as np
import json
import argparse
import sys
import codecs
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from itertools import product
import time
from datetime import datetime
import logging

# Fix Windows encoding
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from xgboost_model import XGBoostModel
from backtester import Backtester
from metrics import generate_metrics_report, print_metrics_summary
import visualizer
from model_comparator import ModelComparator
import config

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class XGBoostTuner:
    """
    Hyperparameter tuning with time-series cross-validation.

    Uses TimeSeriesSplit to preserve temporal order (critical!).
    Optimizes for Sharpe ratio (best metric for trading strategies).
    """

    def __init__(self, n_splits=3, metric='sharpe'):
        """
        Initialize tuner.

        Args:
            n_splits: Number of CV folds (default: 3)
            metric: Optimization metric ('sharpe', 'return', 'correlation', 'r2')
        """
        self.n_splits = n_splits
        self.metric = metric
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.results = []
        self.best_params = None
        self.best_score = -np.inf

        logger.info(f"Initialized XGBoostTuner")
        logger.info(f"  Metric: {metric}")
        logger.info(f"  CV folds: {n_splits}")

    def evaluate_params(self, params, X_train, y_train):
        """
        Evaluate parameter set using time-series CV.

        Args:
            params: Dict of XGBoost parameters
            X_train: Training features
            y_train: Training target

        Returns:
            dict with mean_score, std_score, cv_scores
        """
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.tscv.split(X_train)):
            # Split data (preserving temporal order)
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Train model with early stopping
            full_params = {
                **params,
                'objective': 'reg:squarederror',
                'random_state': 42,
                'early_stopping_rounds': 20
            }

            model = xgb.XGBRegressor(**full_params)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )

            # Predict on validation set
            val_pred = model.predict(X_fold_val)

            # Calculate metric
            if self.metric == 'correlation':
                # Pearson correlation (direction prediction)
                score = np.corrcoef(y_fold_val, val_pred)[0, 1]
                if np.isnan(score):
                    score = 0.0

            elif self.metric == 'r2':
                score = r2_score(y_fold_val, val_pred)

            elif self.metric == 'sharpe':
                # Approximate Sharpe from predictions
                # Buy when prediction > 0, hold cash otherwise
                signals = (val_pred > 0).astype(float)
                strategy_returns = y_fold_val.values * signals

                if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                    sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                    score = sharpe
                else:
                    score = 0.0

            else:
                # Default: negative MSE
                score = -mean_squared_error(y_fold_val, val_pred)

            cv_scores.append(score)

        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)

        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'cv_scores': cv_scores
        }

    def grid_search(self, X_train, y_train, param_grid, verbose=True):
        """
        Grid search over parameter grid.

        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Dict of parameters to search
            verbose: Print progress

        Returns:
            best_params: Dict with best parameters
        """
        print("\n" + "=" * 70)
        print("HYPERPARAMETER TUNING - GRID SEARCH")
        print("=" * 70)
        print(f"Optimization metric: {self.metric}")
        print(f"Cross-validation folds: {self.n_splits}")
        print(f"\nParameter grid:")
        for key, values in param_grid.items():
            print(f"  {key}: {values}")

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        total_combinations = len(param_combinations)
        print(f"\nTotal combinations to test: {total_combinations}")
        print(f"Estimated time: {total_combinations * 0.5:.1f} - {total_combinations * 1:.1f} minutes")
        print("\nStarting search...\n")

        start_time = time.time()

        for i, param_vals in enumerate(param_combinations, 1):
            # Create params dict
            params = dict(zip(param_names, param_vals))

            if verbose:
                print(f"[{i}/{total_combinations}] Testing:")
                for key, val in params.items():
                    print(f"  {key}: {val}")

            # Evaluate
            eval_result = self.evaluate_params(params, X_train, y_train)

            if verbose:
                print(f"  → Score: {eval_result['mean_score']:.4f} "
                      f"(+/- {eval_result['std_score']:.4f})")

            # Store result
            result = {
                'params': params.copy(),
                'mean_score': eval_result['mean_score'],
                'std_score': eval_result['std_score'],
                'cv_scores': eval_result['cv_scores']
            }
            self.results.append(result)

            # Update best
            if eval_result['mean_score'] > self.best_score:
                self.best_score = eval_result['mean_score']
                self.best_params = params.copy()
                if verbose:
                    print(f"  ⭐ NEW BEST! Score: {self.best_score:.4f}")

            if verbose:
                print()

        elapsed_time = time.time() - start_time
        print("=" * 70)
        print("TUNING COMPLETE")
        print("=" * 70)
        print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
        print(f"\nBest parameters found:")
        for key, val in self.best_params.items():
            print(f"  {key}: {val}")
        print(f"\nBest CV score: {self.best_score:.4f}")
        print("=" * 70)

        return self.best_params

    def random_search(self, X_train, y_train, param_distributions, n_iter=50, verbose=True):
        """
        Random search (faster alternative to grid search).

        Args:
            X_train: Training features
            y_train: Training target
            param_distributions: Dict with parameter ranges
            n_iter: Number of random combinations
            verbose: Print progress

        Returns:
            best_params: Dict with best parameters
        """
        print("\n" + "=" * 70)
        print("HYPERPARAMETER TUNING - RANDOM SEARCH")
        print("=" * 70)
        print(f"Random iterations: {n_iter}")
        print(f"Optimization metric: {self.metric}")

        np.random.seed(42)
        start_time = time.time()

        for i in range(n_iter):
            # Sample random parameters
            params = {}
            for key, values in param_distributions.items():
                if isinstance(values, list):
                    params[key] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    # Assume (min, max) range
                    if isinstance(values[0], int):
                        params[key] = np.random.randint(values[0], values[1] + 1)
                    else:
                        params[key] = np.random.uniform(values[0], values[1])

            if verbose:
                print(f"\n[{i+1}/{n_iter}] Testing: {params}")

            # Evaluate
            eval_result = self.evaluate_params(params, X_train, y_train)

            if verbose:
                print(f"  Score: {eval_result['mean_score']:.4f} "
                      f"(+/- {eval_result['std_score']:.4f})")

            # Store and update best
            result = {
                'params': params.copy(),
                'mean_score': eval_result['mean_score'],
                'std_score': eval_result['std_score'],
                'cv_scores': eval_result['cv_scores']
            }
            self.results.append(result)

            if eval_result['mean_score'] > self.best_score:
                self.best_score = eval_result['mean_score']
                self.best_params = params.copy()
                if verbose:
                    print(f"  ⭐ NEW BEST!")

        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TUNING COMPLETE")
        print("=" * 70)
        print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best score: {self.best_score:.4f}")
        print("=" * 70)

        return self.best_params

    def get_results_df(self):
        """Return results as DataFrame sorted by score."""
        if not self.results:
            return pd.DataFrame()

        results_clean = []
        for r in self.results:
            row = r['params'].copy()
            row['mean_score'] = r['mean_score']
            row['std_score'] = r['std_score']
            results_clean.append(row)

        df = pd.DataFrame(results_clean)
        df = df.sort_values('mean_score', ascending=False)
        return df


def load_current_xgboost_results():
    """Load current XGBoost results for comparison."""
    logger.info("Loading current XGBoost results...")

    xgb_dir = Path('results/xgboost')

    # Load metrics
    metrics_df = pd.read_csv(xgb_dir / 'metrics.csv')
    current_metrics = metrics_df.iloc[0].to_dict()

    # Load equity curve
    equity_df = pd.read_csv(xgb_dir / 'equity_curve.csv',
                            index_col=0, parse_dates=True)

    logger.info(f"  Current Sharpe: {current_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Current Return: {current_metrics['total_return']*100:.2f}%")

    return current_metrics, equity_df


def main(search_type='grid', quick=False):
    """
    Main tuning function.

    Args:
        search_type: 'grid' or 'random'
        quick: If True, use smaller parameter grid
    """
    print("\n" + "=" * 70)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Search type: {search_type}")
    print(f"Quick mode: {quick}")
    print(f"Goal: Improve Sharpe from 0.30 to 0.40+")
    print("=" * 70)

    try:
        # ===== 1. LOAD DATA =====
        print("\n[1/5] Loading data...")

        train = pd.read_csv(f'{config.TRAIN_TEST_DIR}/train_data.csv',
                            index_col=0, parse_dates=True)
        test = pd.read_csv(f'{config.TRAIN_TEST_DIR}/test_data.csv',
                           index_col=0, parse_dates=True)

        feature_cols = [f for f in train.columns if f != 'target']
        X_train_full = train[feature_cols]
        y_train_full = train['target']
        X_test = test[feature_cols]
        y_test = test['target']

        logger.info(f"Train: {len(X_train_full)} samples")
        logger.info(f"Test: {len(X_test)} samples")
        logger.info(f"Features: {len(feature_cols)}")

        # ===== 2. DEFINE PARAMETER GRID =====
        print("\n[2/5] Setting up parameter grid...")

        if quick:
            # Quick search (16 combinations)
            param_grid = {
                'max_depth': [3, 4],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [100, 200],
                'min_child_weight': [3, 5],
                'subsample': [0.8],
                'colsample_bytree': [0.8]
            }
        else:
            # Full search (192 combinations)
            param_grid = {
                'max_depth': [2, 3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200, 300],
                'min_child_weight': [1, 3, 5, 7],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }

        # ===== 3. RUN TUNING =====
        print("\n[3/5] Running hyperparameter search...")
        print("(This will take a while - be patient!)")

        tuner = XGBoostTuner(n_splits=3, metric='sharpe')

        if search_type == 'grid':
            best_params = tuner.grid_search(X_train_full, y_train_full, param_grid)
        else:
            # Random search
            param_distributions = {
                'max_depth': [2, 3, 4, 5, 6],
                'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
                'n_estimators': [50, 100, 150, 200, 300],
                'min_child_weight': [1, 3, 5, 7, 10],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
            }
            n_iter = 20 if quick else 50
            best_params = tuner.random_search(X_train_full, y_train_full,
                                             param_distributions, n_iter)

        # ===== 4. TRAIN FINAL MODEL =====
        print("\n[4/5] Training final model with best parameters...")

        # Split for early stopping
        split_idx = int(len(X_train_full) * 0.8)
        X_train = X_train_full.iloc[:split_idx]
        y_train = y_train_full.iloc[:split_idx]
        X_val = X_train_full.iloc[split_idx:]
        y_val = y_train_full.iloc[split_idx:]

        tuned_model = XGBoostModel(params=best_params)
        tuned_model.train(X_train, y_train, X_val, y_val, early_stopping_rounds=20)

        # Evaluate on test
        test_eval = tuned_model.evaluate(X_test, y_test)

        # ===== 5. BACKTEST =====
        print("\n[5/5] Backtesting tuned model...")

        test_pred = tuned_model.predict(X_test)

        backtester = Backtester(
            initial_capital=config.INITIAL_CAPITAL,
            transaction_cost=config.TRANSACTION_COST,
            holding_period=config.HOLDING_PERIOD,
            position_size_pct=0.5,
            prediction_threshold=0.001
        )

        test_prices = test['Close']
        tuned_backtest = backtester.run_backtest(test_pred, y_test, test_prices)

        # Calculate benchmark
        benchmark = backtester.calculate_benchmark(test_prices)
        tuned_backtest['benchmark'] = benchmark

        tuned_metrics = generate_metrics_report(tuned_backtest)

        # Load current XGBoost for comparison
        current_metrics, current_equity = load_current_xgboost_results()

        # ===== COMPARISON =====
        print("\n" + "=" * 70)
        print("RESULTS COMPARISON")
        print("=" * 70)

        comparison_data = []
        metrics_to_compare = [
            ('sharpe_ratio', 'Sharpe Ratio', False),
            ('annual_return', 'Annual Return', True),
            ('max_drawdown_pct', 'Max Drawdown', True),
            ('win_rate', 'Win Rate', True),
            ('profit_factor', 'Profit Factor', False),
            ('total_trades', 'Total Trades', False)
        ]

        print(f"\n{'Metric':<20} {'Current':<12} {'Tuned':<12} {'Change':<12}")
        print("-" * 70)

        for key, label, is_pct in metrics_to_compare:
            current = current_metrics[key]
            tuned = tuned_metrics[key]
            change = tuned - current

            if current != 0:
                change_pct = (change / abs(current)) * 100
            else:
                change_pct = 0

            if is_pct and 'trades' not in key.lower():
                print(f"{label:<20} {current:>10.2f}% {tuned:>10.2f}% {change_pct:>+10.1f}%")
            elif 'trades' in key.lower():
                print(f"{label:<20} {int(current):>11d} {int(tuned):>11d} {change_pct:>+10.1f}%")
            else:
                print(f"{label:<20} {current:>11.2f} {tuned:>11.2f} {change_pct:>+10.1f}%")

        # ===== SAVE RESULTS =====
        print("\n[SAVING] Saving tuned model results...")

        tuned_dir = Path('results/xgboost_tuned')
        tuned_dir.mkdir(exist_ok=True, parents=True)

        # Save metrics
        metrics_df = pd.DataFrame([tuned_metrics])
        metrics_df.to_csv(tuned_dir / 'metrics.csv', index=False)
        logger.info("  Saved metrics.csv")

        # Save best params
        with open(tuned_dir / 'best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        logger.info("  Saved best_params.json")

        # Save tuning results
        results_df = tuner.get_results_df()
        results_df.to_csv(tuned_dir / 'tuning_results.csv', index=False)
        logger.info(f"  Saved tuning_results.csv ({len(results_df)} combinations tested)")

        # Save equity curve
        equity_df = pd.DataFrame({
            'strategy': tuned_backtest['equity_curve'],
            'benchmark': tuned_backtest['benchmark']
        })
        equity_df.to_csv(tuned_dir / 'equity_curve.csv')
        logger.info("  Saved equity_curve.csv")

        # Save trades
        if len(tuned_backtest['trades']) > 0:
            tuned_backtest['trades'].to_csv(tuned_dir / 'trades.csv', index=False)
            logger.info("  Saved trades.csv")

        # Save feature importance
        feature_importance = tuned_model.get_feature_importance()
        feature_importance.to_csv(tuned_dir / 'feature_importance.csv', index=False)
        logger.info("  Saved feature_importance.csv")

        # ===== VISUALIZATIONS =====
        print("\n[VISUALIZING] Creating comparison charts...")

        import matplotlib.pyplot as plt

        # Individual tearsheet
        fig1 = visualizer.generate_tearsheet(
            tuned_backtest, tuned_metrics, tuned_model,
            benchmark=tuned_backtest['benchmark'],
            save_path=tuned_dir / 'tearsheet.png'
        )
        plt.close()

        # Feature importance
        fig2 = tuned_model.plot_feature_importance(
            top_n=10,
            save_path=tuned_dir / 'feature_importance.png'
        )
        plt.close()

        # Three-way comparison
        comparator = ModelComparator()

        # Load baseline (Ridge)
        baseline_dir = Path('results/baseline')
        baseline_metrics_df = pd.read_csv(baseline_dir / 'metrics.csv')
        baseline_metrics = baseline_metrics_df.iloc[0].to_dict()
        baseline_equity = pd.read_csv(baseline_dir / 'equity_curve.csv',
                                      index_col=0, parse_dates=True)

        # Add all three models
        comparator.add_model('Ridge (Baseline)', None,
                            {'equity_curve': baseline_equity['strategy'],
                             'initial_capital': 10000,
                             'final_capital': baseline_equity['strategy'].iloc[-1]},
                            baseline_metrics)

        comparator.add_model('XGBoost (Default)', None,
                            {'equity_curve': current_equity['strategy'],
                             'initial_capital': 10000,
                             'final_capital': current_equity['strategy'].iloc[-1]},
                            current_metrics)

        comparator.add_model('XGBoost (Tuned)', tuned_model,
                            tuned_backtest, tuned_metrics)

        # Comparison plots
        fig3 = comparator.plot_equity_curves(
            save_path=tuned_dir / 'comparison_equity_curves.png'
        )
        plt.close()

        fig4 = comparator.plot_metrics_comparison(
            save_path=tuned_dir / 'comparison_metrics.png'
        )
        plt.close()

        # Save comparison table
        comparison_df = comparator.create_comparison_table()
        comparison_df.to_csv(tuned_dir / 'three_way_comparison.csv', index=False)
        logger.info("  Saved three_way_comparison.csv")

        logger.info(f"\nAll results saved to {tuned_dir}/")

        # ===== FINAL SUMMARY =====
        print("\n" + "=" * 70)
        print("🎉 TUNING COMPLETE!")
        print("=" * 70)

        sharpe_improvement = tuned_metrics['sharpe_ratio'] - current_metrics['sharpe_ratio']
        sharpe_vs_baseline = tuned_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']

        print(f"\nTuned Model Performance:")
        print(f"  Sharpe Ratio:    {tuned_metrics['sharpe_ratio']:>8.2f} ({sharpe_improvement:+.2f} vs current)")
        print(f"  Annual Return:   {tuned_metrics['annual_return']*100:>8.2f}%")
        print(f"  Max Drawdown:    {tuned_metrics['max_drawdown_pct']:>8.2f}%")
        print(f"  Win Rate:        {tuned_metrics['win_rate']*100:>8.1f}%")
        print(f"  Profit Factor:   {tuned_metrics['profit_factor']:>8.2f}")
        print(f"  Total Trades:    {tuned_metrics['total_trades']:>8d}")

        print(f"\nBest Parameters:")
        for key, val in best_params.items():
            print(f"  {key}: {val}")

        print(f"\nOverall Improvement (vs Ridge Baseline):")
        print(f"  Sharpe: {baseline_metrics['sharpe_ratio']:.2f} → {tuned_metrics['sharpe_ratio']:.2f} "
              f"({sharpe_vs_baseline/baseline_metrics['sharpe_ratio']*100:+.0f}%)")
        print(f"  Return: {baseline_metrics['annual_return']*100:.2f}% → {tuned_metrics['annual_return']*100:.2f}% ")

        # Assessment
        print("\n" + "=" * 70)
        print("ASSESSMENT")
        print("=" * 70)

        if tuned_metrics['sharpe_ratio'] >= 0.40:
            print("\n⭐⭐⭐ EXCELLENT! Target Sharpe (0.40+) achieved!")
            print("Ready for Phase 4: Sentiment Integration")
        elif tuned_metrics['sharpe_ratio'] >= 0.35:
            print("\n⭐⭐ VERY GOOD! Close to target Sharpe (0.40)")
            print("Consider Phase 4 or further feature engineering")
        elif tuned_metrics['sharpe_ratio'] > current_metrics['sharpe_ratio']:
            print("\n⭐ IMPROVEMENT! Better than default params")
            print("Proceed with tuned model. Consider Phase 4 for further gains.")
        else:
            print("\n⚠️ No improvement. Default params were already good.")
            print("This can happen - stick with default XGBoost model")

        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Make sure you've run pipeline.py and run_xgboost.py first!")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error during tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XGBoost Hyperparameter Tuning')
    parser.add_argument('--quick', action='store_true',
                        help='Quick search with fewer parameters (16 combinations, ~10-30 min)')
    parser.add_argument('--random', action='store_true',
                        help='Use random search instead of grid search (faster)')

    args = parser.parse_args()

    search_type = 'random' if args.random else 'grid'
    main(search_type=search_type, quick=args.quick)
