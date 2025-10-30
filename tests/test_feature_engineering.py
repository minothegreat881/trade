"""
Unit tests for FeatureEngineer module.

Critical tests:
- Feature creation correctness
- Target variable forward shift (NO LOOK-AHEAD BIAS!)
- NaN handling
- Feature validation
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_engineering import FeatureEngineer
from data_collector import DataCollector


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    df = pd.DataFrame({
        'Open': np.random.uniform(100, 110, 100),
        'High': np.random.uniform(110, 120, 100),
        'Low': np.random.uniform(90, 100, 100),
        'Close': np.random.uniform(100, 110, 100),
        'Volume': np.random.uniform(1e6, 1e7, 100)
    }, index=dates)

    # Ensure High >= Low
    df['High'] = df[['High', 'Low']].max(axis=1)

    return df


@pytest.fixture
def real_data():
    """Download real data for testing."""
    collector = DataCollector('SPY')
    return collector.download_historical('2023-01-01', '2023-12-31')


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""

    def test_initialization(self):
        """Test initialization."""
        engineer = FeatureEngineer()
        assert engineer is not None
        assert engineer.feature_columns == []

    def test_create_returns(self, sample_data):
        """Test return feature creation."""
        engineer = FeatureEngineer()
        df = engineer.create_returns(sample_data.copy())

        # Check that return columns exist
        assert 'return_1d' in df.columns
        assert 'return_5d' in df.columns
        assert 'return_10d' in df.columns
        assert 'return_20d' in df.columns

        # Check that first row has NaN for returns (no previous data)
        assert pd.isna(df['return_1d'].iloc[0])

        # Check that returns are calculated correctly
        # return_1d should be (Close[t] - Close[t-1]) / Close[t-1]
        expected_return = (df['Close'].iloc[2] / df['Close'].iloc[1]) - 1
        actual_return = df['return_1d'].iloc[2]
        assert np.isclose(expected_return, actual_return, rtol=1e-5)

    def test_create_volatility(self, sample_data):
        """Test volatility feature creation."""
        engineer = FeatureEngineer()
        df = engineer.create_volatility(sample_data.copy())

        # Check columns exist
        assert 'volatility_20d' in df.columns
        assert 'volatility_60d' in df.columns

        # First 20 rows should have NaN for 20-day volatility
        assert df['volatility_20d'].iloc[:19].isna().all()

        # After 20 rows, should have values
        assert not pd.isna(df['volatility_20d'].iloc[20])

    def test_create_volume_features(self, sample_data):
        """Test volume feature creation."""
        engineer = FeatureEngineer()
        df = engineer.create_volume_features(sample_data.copy())

        # Check column exists
        assert 'volume_ratio' in df.columns

        # First 20 rows should have NaN (need 20-day window)
        assert df['volume_ratio'].iloc[:19].isna().all()

        # After 20 rows, should have values
        assert not pd.isna(df['volume_ratio'].iloc[20])

    def test_create_price_position(self, sample_data):
        """Test price position feature creation."""
        engineer = FeatureEngineer()
        df = engineer.create_price_position(sample_data.copy())

        # Check column exists
        assert 'price_position' in df.columns

        # Price position should be between 0 and 1
        valid_data = df['price_position'].dropna()
        assert (valid_data >= 0).all()
        assert (valid_data <= 1).all()

    def test_create_trend_features(self, sample_data):
        """Test trend feature creation."""
        engineer = FeatureEngineer()
        df = engineer.create_trend_features(sample_data.copy())

        # Check columns exist
        assert 'sma_20' in df.columns
        assert 'sma_50' in df.columns
        assert 'trend' in df.columns

        # Trend should be binary (0 or 1)
        valid_trend = df['trend'].dropna()
        assert set(valid_trend.unique()).issubset({0, 1})

    def test_create_basic_features(self, real_data):
        """Test creating all features at once."""
        engineer = FeatureEngineer()
        df = engineer.create_basic_features(real_data.copy())

        # Check that all expected features exist
        expected_features = [
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            'volatility_20d', 'volatility_60d',
            'volume_ratio',
            'price_position',
            'sma_20', 'sma_50', 'trend'
        ]

        for feature in expected_features:
            assert feature in df.columns

        # Check feature count
        assert len(engineer.get_feature_names()) == 11

    def test_create_target_forward_shift(self, sample_data):
        """
        CRITICAL TEST: Verify target is shifted FORWARD (no look-ahead bias).

        This ensures we're predicting future returns, not explaining past returns.
        """
        engineer = FeatureEngineer()
        df = sample_data.copy()

        # Create 5-day forward target
        df = engineer.create_target(df, horizon=5)

        # Calculate what the target should be manually
        # Target at time t should be return from t to t+5
        # target[t] = (Close[t+5] / Close[t]) - 1

        # Check a specific row (row 10)
        t = 10
        expected_target = (df['Close'].iloc[t + 5] / df['Close'].iloc[t]) - 1
        actual_target = df['target'].iloc[t]

        assert np.isclose(expected_target, actual_target, rtol=1e-5), \
            f"Target at t={t} should be forward return to t+5"

        # Last 5 rows should have NaN target (no future data)
        assert df['target'].iloc[-5:].isna().all(), \
            "Last 5 rows should have NaN target (no future data available)"

    def test_create_target_different_horizons(self, sample_data):
        """Test target creation with different horizons."""
        engineer = FeatureEngineer()

        # Test horizon = 1
        df1 = engineer.create_target(sample_data.copy(), horizon=1)
        assert df1['target'].iloc[-1:].isna().all()

        # Test horizon = 10
        df10 = engineer.create_target(sample_data.copy(), horizon=10)
        assert df10['target'].iloc[-10:].isna().all()

    def test_create_target_invalid_horizon(self, sample_data):
        """Test target creation with invalid horizon."""
        engineer = FeatureEngineer()

        with pytest.raises(ValueError):
            engineer.create_target(sample_data, horizon=0)

        with pytest.raises(ValueError):
            engineer.create_target(sample_data, horizon=-1)

    def test_validate_features_clean_data(self, real_data):
        """Test feature validation with clean data."""
        engineer = FeatureEngineer()
        df = engineer.create_basic_features(real_data.copy())

        # Drop NaN rows first
        df = df.dropna(subset=engineer.get_feature_names())

        report = engineer.validate_features(df)

        # Should have no issues after dropping NaN
        assert report['nan_count'] == 0
        assert report['inf_count'] == 0

    def test_no_look_ahead_bias_in_features(self, real_data):
        """
        CRITICAL TEST: Ensure no features use future data.

        All features at time t should only use data from t and earlier.
        """
        engineer = FeatureEngineer()
        df = engineer.create_basic_features(real_data.copy())
        df = engineer.create_target(df, horizon=5)

        # For any row t, features should only depend on data up to t
        # We test this by verifying that feature[t] doesn't change
        # if we modify data after t

        # Select a row in the middle
        t = 100
        original_features = df[engineer.get_feature_names()].iloc[t].copy()

        # Modify future data
        df_modified = df.copy()
        df_modified.loc[df_modified.index[t+1:], 'Close'] = 999.99

        # Recreate features
        df_modified = engineer.create_basic_features(df_modified)

        # Features at time t should be unchanged
        modified_features = df_modified[engineer.get_feature_names()].iloc[t]

        for feature in engineer.get_feature_names():
            original_val = original_features[feature]
            modified_val = modified_features[feature]

            # Allow for NaN == NaN
            if pd.isna(original_val) and pd.isna(modified_val):
                continue

            assert np.isclose(original_val, modified_val, rtol=1e-5), \
                f"Feature {feature} at time t changed when future data was modified! " \
                f"This indicates look-ahead bias."

    def test_feature_shapes(self, real_data):
        """Test that feature shapes are correct."""
        engineer = FeatureEngineer()
        df = engineer.create_basic_features(real_data.copy())

        # DataFrame should have same number of rows
        assert len(df) == len(real_data)

        # Should have original columns + feature columns
        assert len(df.columns) > len(real_data.columns)

    def test_get_feature_names(self, real_data):
        """Test getting feature names."""
        engineer = FeatureEngineer()
        df = engineer.create_basic_features(real_data.copy())

        feature_names = engineer.get_feature_names()

        # Should return a list
        assert isinstance(feature_names, list)

        # All features should exist in dataframe
        for feature in feature_names:
            assert feature in df.columns


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
