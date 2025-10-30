"""
Unit tests for DataCollector module.

Tests verify:
- Data download functionality
- Date range validation
- Data validation
- CSV saving
- Error handling
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_collector import DataCollector


class TestDataCollector:
    """Test suite for DataCollector class."""

    def test_initialization_valid_ticker(self):
        """Test initialization with valid ticker."""
        collector = DataCollector('SPY')
        assert collector.ticker == 'SPY'

    def test_initialization_lowercase_ticker(self):
        """Test that ticker is converted to uppercase."""
        collector = DataCollector('spy')
        assert collector.ticker == 'SPY'

    def test_initialization_invalid_ticker(self):
        """Test initialization with invalid ticker."""
        with pytest.raises(ValueError):
            DataCollector('')

        with pytest.raises(ValueError):
            DataCollector(None)

    def test_download_historical_valid_dates(self):
        """Test downloading data with valid date range."""
        collector = DataCollector('SPY')
        df = collector.download_historical('2023-01-01', '2023-12-31')

        # Verify DataFrame is not empty
        assert not df.empty
        assert len(df) > 200  # Should have ~252 trading days per year

        # Verify required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            assert col in df.columns

        # Verify index is DatetimeIndex
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_download_historical_invalid_date_format(self):
        """Test with invalid date format."""
        collector = DataCollector('SPY')

        with pytest.raises(ValueError):
            collector.download_historical('2023/01/01', '2023/12/31')

        with pytest.raises(ValueError):
            collector.download_historical('01-01-2023', '12-31-2023')

    def test_download_historical_start_after_end(self):
        """Test with start date after end date."""
        collector = DataCollector('SPY')

        with pytest.raises(ValueError):
            collector.download_historical('2023-12-31', '2023-01-01')

    def test_get_latest_data(self):
        """Test getting latest data."""
        collector = DataCollector('SPY')
        df = collector.get_latest_data(days_back=30)

        assert not df.empty
        assert len(df) >= 20  # Should have at least 20 trading days in 30 calendar days

    def test_get_latest_data_invalid_days(self):
        """Test get_latest_data with invalid days_back."""
        collector = DataCollector('SPY')

        with pytest.raises(ValueError):
            collector.get_latest_data(days_back=0)

        with pytest.raises(ValueError):
            collector.get_latest_data(days_back=-1)

    def test_validate_data_clean_data(self):
        """Test validation with clean data."""
        collector = DataCollector('SPY')
        df = collector.download_historical('2023-01-01', '2023-12-31')

        report = collector.validate_data(df)

        assert report['is_valid'] == True
        assert report['missing_values'] == 0
        assert report['negative_values'] == False

    def test_validate_data_with_missing_values(self):
        """Test validation with missing values."""
        collector = DataCollector('SPY')
        df = collector.download_historical('2023-01-01', '2023-12-31')

        # Introduce missing values
        df.loc[df.index[0], 'Close'] = None

        report = collector.validate_data(df)

        assert report['is_valid'] == False
        assert report['missing_values'] > 0

    def test_save_to_csv(self, tmp_path):
        """Test saving data to CSV."""
        collector = DataCollector('SPY')
        df = collector.download_historical('2023-01-01', '2023-01-31')

        # Save to temporary file
        test_file = tmp_path / "test_data.csv"
        collector.save_to_csv(df, str(test_file))

        # Verify file exists and can be read
        assert test_file.exists()

        df_loaded = pd.read_csv(test_file, index_col=0, parse_dates=True)
        assert len(df_loaded) == len(df)

    def test_date_range_respected(self):
        """Test that downloaded data respects date range."""
        collector = DataCollector('SPY')
        start_date = '2023-06-01'
        end_date = '2023-08-31'

        df = collector.download_historical(start_date, end_date)

        # First date should be on or after start_date
        assert df.index[0] >= pd.Timestamp(start_date)

        # Last date should be on or before end_date
        assert df.index[-1] <= pd.Timestamp(end_date)

    def test_data_types(self):
        """Test that data has correct types."""
        collector = DataCollector('SPY')
        df = collector.download_historical('2023-01-01', '2023-12-31')

        # Check numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])

        # Check index is datetime
        assert isinstance(df.index, pd.DatetimeIndex)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
