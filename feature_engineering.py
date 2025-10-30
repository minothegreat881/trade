"""
Feature Engineering Module for ML Trading System

This module creates technical indicators and features from raw OHLCV data,
following academic research by Yan (2025), Kelly & Xiu (2023), and others.

CRITICAL: All features avoid look-ahead bias. Target variable is properly
shifted forward to predict future returns.
"""

import logging
from typing import List
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates technical features and target variables for ML trading models.

    This class implements feature engineering based on academic research:
    - Returns (momentum features)
    - Volatility (regime detection)
    - Volume metrics
    - Price position indicators
    - Trend indicators

    All features are designed to avoid look-ahead bias.
    """

    def __init__(self):
        """Initialize FeatureEngineer."""
        self.feature_columns = []
        logger.info("Initialized FeatureEngineer")

    def create_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create return features for multiple time horizons.

        Returns are the most important features according to Kelly & Xiu (2023)
        and Yan (2025).

        Args:
            df: DataFrame with OHLCV data (must have 'Close' column)

        Returns:
            DataFrame with added return columns:
            - return_1d: 1-day return
            - return_5d: 5-day return
            - return_10d: 10-day return
            - return_20d: 20-day return

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = engineer.create_returns(df)
            >>> print(df[['Close', 'return_1d', 'return_5d']].head())
        """
        logger.info("Creating return features...")

        # 1-day return
        df['return_1d'] = df['Close'].pct_change(1)

        # 5-day return
        df['return_5d'] = df['Close'].pct_change(5)

        # 10-day return
        df['return_10d'] = df['Close'].pct_change(10)

        # 20-day return (monthly momentum)
        df['return_20d'] = df['Close'].pct_change(20)

        logger.info("Created 4 return features")
        return df

    def create_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility features for regime detection.

        Volatility is important for detecting market regimes (Suárez-Cetrulo, 2023).

        Args:
            df: DataFrame with return features

        Returns:
            DataFrame with added volatility columns:
            - volatility_20d: 20-day rolling std of returns
            - volatility_60d: 60-day rolling std of returns

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = engineer.create_volatility(df)
            >>> print(df[['return_1d', 'volatility_20d']].head(25))
        """
        logger.info("Creating volatility features...")

        # Need returns to calculate volatility
        if 'return_1d' not in df.columns:
            df = self.create_returns(df)

        # 20-day volatility
        df['volatility_20d'] = df['return_1d'].rolling(window=20).std()

        # 60-day volatility (longer-term regime)
        df['volatility_60d'] = df['return_1d'].rolling(window=60).std()

        logger.info("Created 2 volatility features")
        return df

    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features.

        Volume can indicate strength of price movements.

        Args:
            df: DataFrame with OHLCV data (must have 'Volume' column)

        Returns:
            DataFrame with added volume feature:
            - volume_ratio: Current volume / 20-day average volume

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = engineer.create_volume_features(df)
            >>> print(df[['Volume', 'volume_ratio']].head(25))
        """
        logger.info("Creating volume features...")

        # Volume ratio: current volume vs 20-day average
        volume_ma_20 = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / volume_ma_20

        logger.info("Created 1 volume feature")
        return df

    def create_price_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price position indicator.

        Shows where current price is relative to recent high/low range.
        Range: 0 (at 20-day low) to 1 (at 20-day high)

        Args:
            df: DataFrame with OHLCV data (must have 'Close', 'High', 'Low')

        Returns:
            DataFrame with added feature:
            - price_position: (Close - 20d_low) / (20d_high - 20d_low)

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = engineer.create_price_position(df)
            >>> print(df[['Close', 'price_position']].head(25))
        """
        logger.info("Creating price position feature...")

        # Calculate 20-day high and low
        high_20d = df['High'].rolling(window=20).max()
        low_20d = df['Low'].rolling(window=20).min()

        # Price position (0 to 1)
        # Handle division by zero case
        price_range = high_20d - low_20d
        df['price_position'] = np.where(
            price_range > 0,
            (df['Close'] - low_20d) / price_range,
            0.5  # Default to middle if no range
        )

        logger.info("Created 1 price position feature")
        return df

    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend indicators using moving averages.

        Simple moving averages are classic trend indicators.

        Args:
            df: DataFrame with OHLCV data (must have 'Close' column)

        Returns:
            DataFrame with added features:
            - sma_20: 20-day simple moving average
            - sma_50: 50-day simple moving average
            - trend: Binary (1 if sma_20 > sma_50, else 0)

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = engineer.create_trend_features(df)
            >>> print(df[['Close', 'sma_20', 'sma_50', 'trend']].head(60))
        """
        logger.info("Creating trend features...")

        # Simple moving averages
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()

        # Trend: 1 if short-term MA > long-term MA (uptrend), else 0
        df['trend'] = (df['sma_20'] > df['sma_50']).astype(int)

        logger.info("Created 3 trend features")
        return df

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all basic features in one call.

        This is the main method to call for feature creation.

        Args:
            df: DataFrame with raw OHLCV data

        Returns:
            DataFrame with all features added (12 new columns)

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = engineer.create_basic_features(df)
            >>> print(f"Features created: {len(engineer.get_feature_names())}")
        """
        logger.info("Creating all basic features...")

        # Create all feature groups
        df = self.create_returns(df)
        df = self.create_volatility(df)
        df = self.create_volume_features(df)
        df = self.create_price_position(df)
        df = self.create_trend_features(df)

        # Store feature column names
        self.feature_columns = [
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            'volatility_20d', 'volatility_60d',
            'volume_ratio',
            'price_position',
            'sma_20', 'sma_50', 'trend'
        ]

        logger.info(f"Created {len(self.feature_columns)} features total")
        return df

    def create_target(
        self,
        df: pd.DataFrame,
        horizon: int = 5
    ) -> pd.DataFrame:
        """
        Create forward return target variable.

        CRITICAL: This properly shifts the target FORWARD to avoid look-ahead bias.
        We're predicting the return from t to t+horizon.

        Formula: (Close[t+horizon] / Close[t]) - 1

        Args:
            df: DataFrame with OHLCV data (must have 'Close' column)
            horizon: Number of days to look forward (default: 5)

        Returns:
            DataFrame with added 'target' column

        Raises:
            ValueError: If horizon < 1

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = engineer.create_target(df, horizon=5)
            >>> # Target at time t is the return from t to t+5
            >>> print(df[['Close', 'target']].head(10))

        Note:
            The last 'horizon' rows will have NaN target values because
            we don't have future data for them. These should be removed
            before training.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1")

        logger.info(f"Creating target variable with {horizon}-day forward return")

        # Calculate forward return
        # shift(-horizon) moves Close prices backward, so we're looking at future prices
        df['target'] = (df['Close'].shift(-horizon) / df['Close']) - 1

        # Log how many NaN values created
        nan_count = df['target'].isna().sum()
        logger.info(f"Target created. {nan_count} NaN values at end (expected: {horizon})")

        return df

    def validate_features(self, df: pd.DataFrame) -> dict:
        """
        Validate features for data quality issues.

        Checks for:
        - NaN values
        - Infinite values
        - Extreme outliers (>5 std from mean)

        Args:
            df: DataFrame with features

        Returns:
            Dictionary with validation results:
            {
                'is_valid': bool,
                'nan_count': int,
                'inf_count': int,
                'outlier_count': int,
                'issues': list of str
            }

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = engineer.create_basic_features(df)
            >>> report = engineer.validate_features(df)
            >>> print(f"NaN values: {report['nan_count']}")
        """
        issues = []

        # Check for NaN values
        nan_count = df[self.feature_columns].isna().sum().sum()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values in features")

        # Check for infinite values
        inf_count = np.isinf(df[self.feature_columns].select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values")

        # Check for extreme outliers (>5 std from mean)
        outlier_count = 0
        for col in self.feature_columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                mean = df[col].mean()
                std = df[col].std()
                outliers = ((df[col] - mean).abs() > 5 * std).sum()
                outlier_count += outliers

        if outlier_count > 0:
            issues.append(f"Found {outlier_count} extreme outliers (>5 std)")

        is_valid = len(issues) == 0

        validation_report = {
            'is_valid': is_valid,
            'nan_count': int(nan_count),
            'inf_count': int(inf_count),
            'outlier_count': int(outlier_count),
            'issues': issues
        }

        if is_valid:
            logger.info("Feature validation: ✓ All checks passed")
        else:
            logger.warning(f"Feature validation issues: {', '.join(issues)}")

        return validation_report

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature column names.

        Returns:
            List of feature column names

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = engineer.create_basic_features(df)
            >>> features = engineer.get_feature_names()
            >>> print(f"Features: {features}")
        """
        return self.feature_columns


if __name__ == "__main__":
    # Example usage with sample data
    from data_collector import DataCollector

    collector = DataCollector('SPY')
    df = collector.download_historical('2020-01-01', '2024-12-31')

    engineer = FeatureEngineer()
    df = engineer.create_basic_features(df)
    df = engineer.create_target(df, horizon=5)

    print(f"\nFeatures created: {len(engineer.get_feature_names())}")
    print(f"Feature names: {engineer.get_feature_names()}")
    print(f"\nData shape: {df.shape}")
    print(f"\nFirst few rows with features:")
    print(df[['Close', 'return_1d', 'volatility_20d', 'trend', 'target']].head(70))

    # Validate features
    report = engineer.validate_features(df)
    print(f"\nValidation: {'✓ PASSED' if report['is_valid'] else '✗ FAILED'}")
    print(f"NaN values: {report['nan_count']}")
