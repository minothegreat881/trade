"""
Data Collector Module for ML Trading System

This module handles downloading historical stock data from Yahoo Finance,
validating data quality, and saving data to CSV files.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Handles downloading and managing historical stock market data.

    This class provides methods to download historical OHLCV data from
    Yahoo Finance, validate data quality, and save data to CSV files.

    Attributes:
        ticker (str): Stock ticker symbol (e.g., 'SPY', 'AAPL')
    """

    def __init__(self, ticker: str):
        """
        Initialize DataCollector with a ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g., 'SPY', 'AAPL')

        Raises:
            ValueError: If ticker is empty or invalid
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")

        self.ticker = ticker.upper()
        logger.info(f"Initialized DataCollector for {self.ticker}")

    def download_historical(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data for the specified date range.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume, Adj Close
            Index: Date (datetime)

        Raises:
            ValueError: If date format is invalid or start_date > end_date
            ConnectionError: If unable to download data from Yahoo Finance

        Example:
            >>> collector = DataCollector('SPY')
            >>> df = collector.download_historical('2020-01-01', '2024-12-31')
            >>> print(df.head())
        """
        # Validate date format
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD': {e}")

        if start_dt >= end_dt:
            raise ValueError("start_date must be before end_date")

        logger.info(f"Downloading {self.ticker} data from {start_date} to {end_date}")

        try:
            # Download data from Yahoo Finance
            ticker_obj = yf.Ticker(self.ticker)
            df = ticker_obj.history(start=start_date, end=end_date)

            if df.empty:
                raise ConnectionError(
                    f"No data returned for {self.ticker}. "
                    "Check if ticker is valid and dates are correct."
                )

            # Reset index to make Date a column
            df.reset_index(inplace=True)

            # Ensure we have all required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Set Date as index
            df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)

            logger.info(f"Downloaded {len(df)} rows of data")

            return df

        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise ConnectionError(f"Failed to download data for {self.ticker}: {e}")

    def get_latest_data(self, days_back: int = 100) -> pd.DataFrame:
        """
        Download latest data for the specified number of days.

        This is useful for live trading scenarios where you need recent data.

        Args:
            days_back: Number of days of historical data to fetch (default: 100)

        Returns:
            DataFrame with latest OHLCV data

        Raises:
            ValueError: If days_back < 1
            ConnectionError: If unable to download data

        Example:
            >>> collector = DataCollector('SPY')
            >>> df = collector.get_latest_data(days_back=30)
            >>> print(f"Latest data from {df.index[0]} to {df.index[-1]}")
        """
        if days_back < 1:
            raise ValueError("days_back must be at least 1")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        return self.download_historical(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate data quality and return a report.

        Checks for:
        - Missing values
        - Date gaps (missing trading days)
        - Negative prices or volumes
        - Duplicate dates

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with validation results:
            {
                'is_valid': bool,
                'missing_values': int,
                'date_gaps': int,
                'negative_values': bool,
                'duplicate_dates': int,
                'issues': list of str
            }

        Example:
            >>> collector = DataCollector('SPY')
            >>> df = collector.download_historical('2020-01-01', '2024-12-31')
            >>> report = collector.validate_data(df)
            >>> if report['is_valid']:
            ...     print("Data is valid!")
        """
        issues = []

        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            issues.append(f"Found {missing_count} missing values")

        # Check for date gaps (more than 5 days between consecutive dates)
        # Note: Market is closed on weekends, so we allow up to 5 days gap
        date_diffs = df.index.to_series().diff()
        large_gaps = (date_diffs > timedelta(days=5)).sum()
        if large_gaps > 0:
            issues.append(f"Found {large_gaps} date gaps > 5 days")

        # Check for negative prices or volumes
        has_negative = (
            (df['Open'] < 0).any() or
            (df['High'] < 0).any() or
            (df['Low'] < 0).any() or
            (df['Close'] < 0).any() or
            (df['Volume'] < 0).any()
        )
        if has_negative:
            issues.append("Found negative prices or volumes")

        # Check for duplicate dates
        duplicate_dates = df.index.duplicated().sum()
        if duplicate_dates > 0:
            issues.append(f"Found {duplicate_dates} duplicate dates")

        is_valid = len(issues) == 0

        validation_report = {
            'is_valid': is_valid,
            'missing_values': int(missing_count),
            'date_gaps': int(large_gaps),
            'negative_values': has_negative,
            'duplicate_dates': int(duplicate_dates),
            'issues': issues
        }

        if is_valid:
            logger.info("Data validation: ✓ All checks passed")
        else:
            logger.warning(f"Data validation issues: {', '.join(issues)}")

        return validation_report

    def save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            filename: Path to save the CSV file

        Raises:
            IOError: If unable to write file

        Example:
            >>> collector = DataCollector('SPY')
            >>> df = collector.download_historical('2020-01-01', '2024-12-31')
            >>> collector.save_to_csv(df, 'data/raw/SPY_historical.csv')
        """
        try:
            df.to_csv(filename)
            logger.info(f"Data saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise IOError(f"Unable to save data to {filename}: {e}")


if __name__ == "__main__":
    # Example usage
    collector = DataCollector('SPY')
    df = collector.download_historical('2020-01-01', '2024-12-31')

    print(f"\nDownloaded {len(df)} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nFirst few rows:\n{df.head()}")

    # Validate data
    report = collector.validate_data(df)
    print(f"\nValidation: {'✓ PASSED' if report['is_valid'] else '✗ FAILED'}")

    # Save to CSV
    collector.save_to_csv(df, 'data/raw/SPY_historical.csv')
