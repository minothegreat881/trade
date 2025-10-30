"""
Hyperparameter Tuning Module for XGBoost

Uses time-series cross-validation to find optimal parameters.
CRITICAL: Uses TimeSeriesSplit to maintain temporal order.
"""

from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import numpy as np
import pandas as pd
from itertools import product
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class XGBoostTuner:
    """
    Grid search for XGBoost hyperparameters with time-series CV.

    IMPORTANT: Uses TimeSeriesSplit instead of KFold to preserve
    temporal order in financial data.

    Attributes:
        n_splits (int): Number of CV folds
        tscv (TimeSeriesSplit): Time-series cross-validator
        best_params (dict): Best parameter combination found
        results (list): All tuning results

    Example:
        >>> tuner = XGBoostTuner(n_splits=3)
        >>> param_grid = {
        ...     'max_depth': [2, 3, 4],
        ...     'learning_rate': [0.01, 0.05, 0.1]
        ... }
        >>> best_params = tuner.tune(X_train, y_train, param_grid)
    """

    def __init__(self, n_splits=3):
        """
        Initialize tuner with time-series cross-validation.

        Args:
            n_splits: Number of CV splits (default: 3)
                     More splits = more robust but slower
        """
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.best_params = None
        self.results = []

        logger.info(f"Initialized XGBoostTuner with {n_splits} time-series splits")

    def tune(self, X_train, y_train, param_grid):
        """
        Grid search over parameter grid using time-series CV.

        Args:
            X_train: Training features (DataFrame)
            y_train: Training target (Series)
            param_grid: Dict of parameters to search
                Example:
                {
                    'max_depth': [2, 3, 4],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'min_child_weight': [3, 5, 7]
                }

        Returns:
            best_params: Dict with best parameter combination

        Example:
            >>> param_grid = {'max_depth': [2, 3, 4]}
            >>> best_params = tuner.tune(X_train, y_train, param_grid)
            >>> print(f"Best max_depth: {best_params['max_depth']}")
        """
        logger.info("="*60)
        logger.info("HYPERPARAMETER TUNING")
        logger.info("="*60)
        logger.info(f"\nParameter grid:")
        for key, values in param_grid.items():
            logger.info(f"  {key}: {values}")

        # Generate all combinations
        param_combinations = [
            dict(zip(param_grid.keys(), values))
            for values in product(*param_grid.values())
        ]

        total_combinations = len(param_combinations)
        logger.info(f"\nTesting {total_combinations} combinations × {self.n_splits} folds")
        logger.info(f"= {total_combinations * self.n_splits} total model fits\n")

        best_score = -np.inf

        # Test each combination
        for i, params in enumerate(param_combinations, 1):
            logger.info(f"[{i}/{total_combinations}] Testing: {params}")

            # Time-series cross-validation
            cv_scores = []

            for fold, (train_idx, val_idx) in enumerate(self.tscv.split(X_train), 1):
                # Split data
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]

                # Add fixed params
                full_params = {
                    'objective': 'reg:squarederror',
                    'random_state': 42,
                    **params
                }

                # Train model
                model = xgb.XGBRegressor(**full_params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )

                # Evaluate (use correlation as metric)
                val_pred = model.predict(X_fold_val)
                score = np.corrcoef(y_fold_val, val_pred)[0, 1]  # Correlation
                cv_scores.append(score)

                logger.info(f"  Fold {fold}: {score:.4f}")

            # Calculate mean and std
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            logger.info(f"  Mean CV Score: {mean_score:.4f} (+/- {std_score:.4f})")

            # Store results
            self.results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores
            })

            # Update best
            if mean_score > best_score:
                best_score = mean_score
                self.best_params = params
                logger.info(f"  ⭐ New best score!")

            logger.info("")  # Blank line

        # Print final results
        logger.info("="*60)
        logger.info("TUNING COMPLETE")
        logger.info("="*60)
        logger.info(f"\nBest parameters:")
        for key, value in self.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"\nBest CV Score: {best_score:.4f}")
        logger.info("="*60)

        return self.best_params

    def get_results_df(self):
        """
        Return tuning results as DataFrame.

        Returns:
            DataFrame with columns: params, mean_score, std_score

        Example:
            >>> results_df = tuner.get_results_df()
            >>> results_df.sort_values('mean_score', ascending=False)
        """
        if not self.results:
            raise ValueError("No tuning results available. Run tune() first!")

        # Extract params into separate columns
        results_list = []
        for result in self.results:
            row = {**result['params'],
                   'mean_score': result['mean_score'],
                   'std_score': result['std_score']}
            results_list.append(row)

        df = pd.DataFrame(results_list)
        df = df.sort_values('mean_score', ascending=False)

        return df

    def plot_tuning_results(self, param_name, save_path=None):
        """
        Plot how a specific parameter affects CV score.

        Args:
            param_name: Parameter name to plot (e.g., 'max_depth')
            save_path: Path to save plot (optional)

        Returns:
            matplotlib Figure

        Example:
            >>> fig = tuner.plot_tuning_results('max_depth')
            >>> plt.show()
        """
        import matplotlib.pyplot as plt

        df = self.get_results_df()

        if param_name not in df.columns:
            raise ValueError(f"Parameter '{param_name}' not found in results!")

        # Group by parameter value
        grouped = df.groupby(param_name).agg({
            'mean_score': ['mean', 'std']
        })

        fig, ax = plt.subplots(figsize=(10, 6))

        x = grouped.index
        y = grouped[('mean_score', 'mean')]
        err = grouped[('mean_score', 'std')]

        ax.errorbar(x, y, yerr=err, marker='o', capsize=5,
                    linewidth=2, markersize=8)

        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('CV Score', fontsize=12)
        ax.set_title(f'Effect of {param_name} on CV Performance',
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig


if __name__ == "__main__":
    # Example usage
    print("Testing XGBoostTuner...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    X = pd.DataFrame({
        'return_1d': np.random.randn(500),
        'return_5d': np.random.randn(500),
        'volatility': np.random.randn(500)
    }, index=dates)
    y = pd.Series(0.5 * X['return_5d'] + 0.3 * X['volatility'] +
                  np.random.randn(500) * 0.1, index=dates)

    # Define small param grid for testing
    param_grid = {
        'max_depth': [2, 3],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [50, 100]
    }

    # Run tuning
    tuner = XGBoostTuner(n_splits=3)
    best_params = tuner.tune(X, y, param_grid)

    # Get results
    print("\nAll Results:")
    results_df = tuner.get_results_df()
    print(results_df.to_string(index=False))

    # Plot (if matplotlib available)
    try:
        fig = tuner.plot_tuning_results('max_depth')
        plt.close()
        print("\n✓ Plotting works!")
    except:
        print("\n(Skipped plotting)")

    print("\n✓ XGBoostTuner test complete!")
