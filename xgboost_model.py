"""
XGBoost Model Module - Advanced Trading Prediction

Based on Yan (2025) methodology:
- Non-linear feature interactions
- Early stopping for optimal iteration
- Feature importance analysis
- Robust to overfitting with proper regularization
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost regression model for stock return prediction.

    Features:
    - Time-series aware training
    - Early stopping to prevent overfitting
    - Feature importance analysis
    - Hyperparameter tuning support

    Attributes:
        params (dict): XGBoost parameters
        model (XGBRegressor): Trained model
        feature_importance_ (DataFrame): Feature importance scores
        best_iteration (int): Best iteration from early stopping
    """

    def __init__(self, params=None):
        """
        Initialize XGBoost model.

        Args:
            params: Dict of XGBoost parameters. If None, use defaults.

        Default parameters (conservative to avoid overfitting):
            - max_depth: 3 (shallow trees)
            - learning_rate: 0.05 (slow learning)
            - n_estimators: 100 (with early stopping)
            - min_child_weight: 5 (prevent overfitting)
            - subsample: 0.8 (row sampling)
            - colsample_bytree: 0.8 (feature sampling)

        Example:
            >>> model = XGBoostModel()
            >>> model.train(X_train, y_train, X_val, y_val)
        """
        if params is None:
            # Default parameters - conservative to avoid overfitting
            self.params = {
                'objective': 'reg:squarederror',
                'max_depth': 3,              # Shallow trees
                'learning_rate': 0.05,       # Slow learning
                'n_estimators': 100,         # Will use early stopping
                'min_child_weight': 5,       # Prevent overfitting
                'subsample': 0.8,            # Row sampling (80%)
                'colsample_bytree': 0.8,     # Feature sampling (80%)
                'gamma': 0,                  # Min loss reduction
                'reg_alpha': 0,              # L1 regularization
                'reg_lambda': 1,             # L2 regularization
                'random_state': 42
            }
        else:
            self.params = params

        self.model = None
        self.feature_names = None
        self.feature_importance_ = None
        self.best_iteration = None
        self.is_trained = False

        logger.info(f"Initialized XGBoostModel")
        logger.info(f"  max_depth: {self.params.get('max_depth', 3)}")
        logger.info(f"  learning_rate: {self.params.get('learning_rate', 0.05)}")
        logger.info(f"  n_estimators: {self.params.get('n_estimators', 100)}")

    def train(self, X_train, y_train, X_val=None, y_val=None,
              early_stopping_rounds=10, verbose=False):
        """
        Train XGBoost model with optional early stopping.

        Args:
            X_train: Training features (DataFrame or array)
            y_train: Training target (Series or array)
            X_val: Validation features (for early stopping)
            y_val: Validation target
            early_stopping_rounds: Stop if no improvement for N rounds
            verbose: Print training progress

        Returns:
            self (for method chaining)

        Example:
            >>> model.train(X_train, y_train, X_val, y_val)
            >>> print(f"Best iteration: {model.best_iteration}")
        """
        logger.info("Training XGBoost model...")

        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Train with or without early stopping
        if X_val is not None and y_val is not None:
            logger.info("  Using validation set for early stopping")

            # Create model with early stopping in params
            params_with_es = {
                **self.params,
                'early_stopping_rounds': early_stopping_rounds
            }
            self.model = xgb.XGBRegressor(**params_with_es)

            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=verbose
            )

            # Get best iteration
            if hasattr(self.model, 'best_iteration'):
                self.best_iteration = self.model.best_iteration
                logger.info(f"  Best iteration: {self.best_iteration}")
            else:
                self.best_iteration = self.params.get('n_estimators', 100)
                logger.info(f"  Completed {self.best_iteration} iterations")

        else:
            logger.info("  Training without early stopping")

            # Create and train model
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(X_train, y_train, verbose=verbose)

        # Store feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        train_mse = mean_squared_error(y_train, train_pred)

        self.is_trained = True

        logger.info(f"Training metrics:")
        logger.info(f"  Train R²: {train_r2:.4f}")
        logger.info(f"  Train MSE: {train_mse:.6f}")

        # Validation metrics if available
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            val_mse = mean_squared_error(y_val, val_pred)

            logger.info(f"  Val R²: {val_r2:.4f}")
            logger.info(f"  Val MSE: {val_mse:.6f}")

            # Check for overfitting
            overfit_gap = train_r2 - val_r2
            if overfit_gap > 0.10:
                logger.warning(f"  ⚠️ Possible overfitting detected! Train-Val gap: {overfit_gap:.4f}")
                logger.warning("     Consider: decrease max_depth, increase min_child_weight")

        return self

    def predict(self, X):
        """
        Generate predictions for new data.

        Args:
            X: Features to predict on (DataFrame or array)

        Returns:
            predictions (pd.Series): Predicted values with dates as index

        Raises:
            ValueError: If model not trained yet

        Example:
            >>> predictions = model.predict(X_test)
            >>> print(predictions.head())
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")

        # Generate predictions
        predictions = self.model.predict(X)

        # Return as Series with original index if DataFrame
        if isinstance(X, pd.DataFrame):
            return pd.Series(predictions, index=X.index, name='prediction')
        else:
            return pd.Series(predictions, name='prediction')

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test set.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            dict with metrics: r2, mse, mae, rmse

        Example:
            >>> metrics = model.evaluate(X_test, y_test)
            >>> print(f"Test R²: {metrics['r2']:.4f}")
        """
        logger.info("Evaluating model on test set...")

        # Generate predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }

        logger.info(f"  Test R²: {metrics['r2']:.4f}")
        logger.info(f"  Test RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  Test MAE: {metrics['mae']:.6f}")

        return metrics

    def get_feature_importance(self, top_n=None):
        """
        Get feature importance scores.

        Args:
            top_n: Return only top N features (default: all)

        Returns:
            pd.DataFrame: Feature importance sorted by importance

        Example:
            >>> importance = model.get_feature_importance(top_n=10)
            >>> print(importance)
        """
        if self.feature_importance_ is None:
            raise ValueError("Model must be trained first!")

        if top_n is not None:
            return self.feature_importance_.head(top_n)

        return self.feature_importance_

    def plot_feature_importance(self, top_n=10, save_path=None):
        """
        Plot feature importance chart.

        Args:
            top_n: Number of top features to show
            save_path: Path to save plot (optional)

        Returns:
            matplotlib Figure

        Example:
            >>> fig = model.plot_feature_importance(top_n=10)
            >>> plt.show()
        """
        logger.info(f"Plotting top {top_n} feature importance...")

        top_features = self.get_feature_importance(top_n=top_n)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Horizontal bar chart
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'].values,
                color='#2E86AB', alpha=0.8, edgecolor='black')

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (Gain)', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance (XGBoost)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  Saved to {save_path}")

        return fig

    def summary(self):
        """
        Print model summary with key statistics.

        Example:
            >>> model.summary()
        """
        print("=" * 60)
        print("XGBOOST MODEL SUMMARY")
        print("=" * 60)
        print(f"\nModel Type: XGBoost Regression")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Best iteration: {self.best_iteration}")
        print(f"\nHyperparameters:")
        for key, value in self.params.items():
            if key != 'random_state':
                print(f"  {key}: {value}")
        print(f"\nTop 5 Most Important Features:")
        top_5 = self.get_feature_importance(top_n=5)
        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            print(f"  {i}. {row['feature']:20} {row['importance']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    # Example usage
    print("Testing XGBoostModel...")

    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame({
        'return_1d': np.random.randn(100),
        'return_5d': np.random.randn(100),
        'volatility': np.random.randn(100)
    })
    y = pd.Series(0.5 * X['return_5d'] + 0.3 * X['volatility'] +
                  np.random.randn(100) * 0.1)

    # Split data
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    # Train model
    model = XGBoostModel()
    model.train(X_train, y_train, X_val, y_val)

    # Test predictions
    predictions = model.predict(X_val)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"First 5 predictions:\n{predictions.head()}")

    # Evaluate
    metrics = model.evaluate(X_val, y_val)

    # Feature importance
    print("\nFeature Importance:")
    print(model.get_feature_importance())

    # Summary
    model.summary()

    print("\n✓ XGBoostModel test complete!")
