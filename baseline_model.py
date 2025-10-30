"""
Baseline Model Module - Ridge Regression

Based on Kelly & Xiu (2023) regularization framework.
Simple, interpretable baseline for trading predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Ridge Regression baseline model for trading predictions.

    Uses L2 regularization to prevent overfitting and ensure stable
    coefficient estimates.

    Attributes:
        alpha (float): Ridge regularization parameter
        model (Ridge): Fitted Ridge regression model
        scaler (StandardScaler): Feature scaling
        feature_names (list): Names of features used
        coefficients (dict): Feature importance coefficients
    """

    def __init__(self, alpha=1.0):
        """
        Initialize baseline model.

        Args:
            alpha: Ridge regularization parameter (default: 1.0)
                  Higher alpha = more regularization

        Example:
            >>> model = BaselineModel(alpha=1.0)
            >>> model.train(X_train, y_train)
        """
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.coefficients = None
        self.is_trained = False

        logger.info(f"Initialized BaselineModel with alpha={alpha}")

    def train(self, X_train, y_train):
        """
        Train Ridge Regression model on training data.

        Args:
            X_train: Training features (DataFrame or array)
            y_train: Training target (Series or array)

        Returns:
            self (for method chaining)

        Example:
            >>> model.train(X_train, y_train)
            >>> print(f"Model trained with R² = {model.train_r2:.4f}")
        """
        logger.info("Training Ridge Regression model...")

        # Store feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_scaled, y_train)

        # Store coefficients
        self.coefficients = dict(zip(self.feature_names, self.model.coef_))

        # Calculate training metrics
        train_pred = self.model.predict(X_scaled)
        self.train_r2 = r2_score(y_train, train_pred)
        self.train_mse = mean_squared_error(y_train, train_pred)

        self.is_trained = True

        logger.info(f"✓ Model trained successfully")
        logger.info(f"  Train R²: {self.train_r2:.4f}")
        logger.info(f"  Train MSE: {self.train_mse:.6f}")

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

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Generate predictions
        predictions = self.model.predict(X_scaled)

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
        Get feature importance (absolute coefficients).

        Args:
            top_n: Return only top N features (default: all)

        Returns:
            pd.Series: Feature importance sorted by absolute value

        Example:
            >>> importance = model.get_feature_importance(top_n=5)
            >>> print(importance)
        """
        if self.coefficients is None:
            raise ValueError("Model must be trained first!")

        # Create Series and sort by absolute value
        importance = pd.Series(self.coefficients).abs().sort_values(ascending=False)

        if top_n is not None:
            importance = importance.head(top_n)

        return importance

    def get_coefficients(self):
        """
        Get raw model coefficients (with signs).

        Returns:
            pd.Series: Feature coefficients with signs

        Example:
            >>> coefs = model.get_coefficients()
            >>> positive_features = coefs[coefs > 0]
        """
        if self.coefficients is None:
            raise ValueError("Model must be trained first!")

        return pd.Series(self.coefficients).sort_values(ascending=False)

    def summary(self):
        """
        Print model summary with key statistics.

        Example:
            >>> model.summary()
        """
        print("=" * 60)
        print("BASELINE MODEL SUMMARY")
        print("=" * 60)
        print(f"\nModel Type: Ridge Regression")
        print(f"Regularization (alpha): {self.alpha}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"\nTraining Performance:")
        print(f"  R² Score: {self.train_r2:.4f}")
        print(f"  MSE: {self.train_mse:.6f}")
        print(f"\nTop 5 Most Important Features:")
        importance = self.get_feature_importance(top_n=5)
        for i, (feat, imp) in enumerate(importance.items(), 1):
            coef = self.coefficients[feat]
            sign = "+" if coef > 0 else "-"
            print(f"  {i}. {feat:20} ({sign}) {imp:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    # Example usage
    print("Testing BaselineModel...")

    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame({
        'return_1d': np.random.randn(100),
        'return_5d': np.random.randn(100),
        'volatility': np.random.randn(100)
    })
    y = pd.Series(0.5 * X['return_5d'] + 0.3 * X['volatility'] + np.random.randn(100) * 0.1)

    # Train model
    model = BaselineModel(alpha=1.0)
    model.train(X, y)

    # Test predictions
    predictions = model.predict(X)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"First 5 predictions:\n{predictions.head()}")

    # Evaluate
    metrics = model.evaluate(X, y)

    # Feature importance
    print("\nFeature Importance:")
    print(model.get_feature_importance())

    # Summary
    model.summary()

    print("\n✓ BaselineModel test complete!")
