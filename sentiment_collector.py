"""
Sentiment Data Collection Module

Collects sentiment data from multiple sources:
- Fear & Greed Index (Alternative.me API)
- VIX (CBOE Volatility Index from Yahoo Finance)
- Bitcoin (BTC-USD from Yahoo Finance)

Uses caching to avoid repeated downloads.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class SentimentCollector:
    """
    Collect sentiment data from multiple sources.

    Uses caching to avoid repeated API calls.
    Provides fallbacks if primary sources fail.
    """

    def __init__(self, cache_dir='data/sentiment_cache'):
        """
        Initialize sentiment collector.

        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Initialized SentimentCollector")
        logger.info(f"  Cache directory: {self.cache_dir}")

    def get_fear_greed_index(self, start_date, end_date):
        """
        Get Fear & Greed Index from Alternative.me API.

        Note: Using Crypto Fear & Greed as proxy for CNN Fear & Greed
        (CNN index not publicly available via free API)

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: [fear_greed_value, fear_greed_classification]
        """
        logger.info("Downloading Fear & Greed Index...")

        cache_file = self.cache_dir / f'fear_greed_{start_date}_{end_date}.csv'

        # Check cache first
        if cache_file.exists():
            logger.info(f"  Loading from cache: {cache_file.name}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

        try:
            # Alternative.me Crypto Fear & Greed API (free, no auth)
            url = "https://api.alternative.me/fng/?limit=0"

            logger.info("  Fetching from Alternative.me API...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'data' not in data:
                raise ValueError("Unexpected API response format")

            # Parse data
            records = []
            for item in data['data']:
                timestamp = int(item['timestamp'])
                date = pd.to_datetime(timestamp, unit='s')
                value = int(item['value'])
                classification = item['value_classification']

                records.append({
                    'Date': date,
                    'fear_greed_value': value,
                    'fear_greed_classification': classification
                })

            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()

            # Filter by date range
            df = df.loc[start_date:end_date]

            # Save to cache
            df.to_csv(cache_file)
            logger.info(f"  Downloaded {len(df)} days of Fear & Greed data")

            return df

        except Exception as e:
            logger.warning(f"  Could not fetch Fear & Greed Index: {e}")
            logger.info("  Creating synthetic Fear & Greed data from VIX...")

            # Fallback: Synthesize F&G from VIX
            vix_data = self.get_vix_data(start_date, end_date)

            if vix_data is not None and len(vix_data) > 0:
                # Inverse relationship: High VIX = Low F&G (fear)
                # VIX 10-40 → F&G 0-100 (inverted)
                fear_greed_value = 100 - ((vix_data['VIX'].clip(10, 40) - 10) / 30 * 100)
                fear_greed_value = fear_greed_value.round()

                df = pd.DataFrame({
                    'fear_greed_value': fear_greed_value,
                })

                # Classification
                def classify_fg(value):
                    if value <= 25:
                        return 'Extreme Fear'
                    elif value <= 45:
                        return 'Fear'
                    elif value <= 55:
                        return 'Neutral'
                    elif value <= 75:
                        return 'Greed'
                    else:
                        return 'Extreme Greed'

                df['fear_greed_classification'] = df['fear_greed_value'].apply(classify_fg)

                # Save to cache
                df.to_csv(cache_file)
                logger.info(f"  Created synthetic F&G data ({len(df)} days)")

                return df

            logger.warning("  Could not create Fear & Greed data")
            return None

    def get_vix_data(self, start_date, end_date):
        """
        Get VIX (CBOE Volatility Index) from Yahoo Finance.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with VIX values
        """
        logger.info("Downloading VIX data...")

        cache_file = self.cache_dir / f'vix_{start_date}_{end_date}.csv'

        # Check cache
        if cache_file.exists():
            logger.info(f"  Loading from cache: {cache_file.name}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

        try:
            vix = yf.download('^VIX', start=start_date, end=end_date,
                             progress=False, auto_adjust=False)

            if vix is None or len(vix) == 0:
                raise ValueError("No VIX data returned")

            # Handle single-column or multi-column case
            if 'Close' in vix.columns:
                vix_close = vix['Close']
            elif isinstance(vix.columns, pd.MultiIndex):
                vix_close = vix['Close']['^VIX']
            else:
                vix_close = vix.iloc[:, 0]  # First column if structure unclear

            df = pd.DataFrame({
                'VIX': vix_close
            })

            # Save to cache
            df.to_csv(cache_file)
            logger.info(f"  Downloaded {len(df)} days of VIX data")

            return df

        except Exception as e:
            logger.error(f"  Error downloading VIX: {e}")
            return None

    def get_bitcoin_data(self, start_date, end_date):
        """
        Get Bitcoin price data from Yahoo Finance.

        Bitcoin as leading indicator for risk sentiment.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with BTC prices and returns
        """
        logger.info("Downloading Bitcoin data...")

        cache_file = self.cache_dir / f'btc_{start_date}_{end_date}.csv'

        # Check cache
        if cache_file.exists():
            logger.info(f"  Loading from cache: {cache_file.name}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

        try:
            btc = yf.download('BTC-USD', start=start_date, end=end_date,
                             progress=False, auto_adjust=False)

            if btc is None or len(btc) == 0:
                raise ValueError("No Bitcoin data returned")

            # Handle single-column or multi-column case
            if 'Close' in btc.columns and 'Volume' in btc.columns:
                btc_close = btc['Close']
                btc_volume = btc['Volume']
            elif isinstance(btc.columns, pd.MultiIndex):
                btc_close = btc['Close']['BTC-USD']
                btc_volume = btc['Volume']['BTC-USD']
            else:
                btc_close = btc.iloc[:, 3]  # Close is usually 4th column
                btc_volume = btc.iloc[:, 5]  # Volume is usually 6th column

            df = pd.DataFrame({
                'BTC_Close': btc_close,
                'BTC_Volume': btc_volume
            })

            # Calculate returns
            df['BTC_return_1d'] = df['BTC_Close'].pct_change()
            df['BTC_return_5d'] = df['BTC_Close'].pct_change(5)
            df['BTC_return_10d'] = df['BTC_Close'].pct_change(10)

            # Save to cache
            df.to_csv(cache_file)
            logger.info(f"  Downloaded {len(df)} days of Bitcoin data")

            return df

        except Exception as e:
            logger.error(f"  Error downloading Bitcoin: {e}")
            return None

    def get_put_call_ratio(self, start_date, end_date):
        """
        Get Put/Call Ratio (optional - requires paid data source).

        Placeholder for future implementation.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            None (not implemented)
        """
        logger.info("Put/Call Ratio - Skipping (requires paid data)")
        return None

    def collect_all_sentiment(self, start_date, end_date):
        """
        Collect all sentiment data sources.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with all sentiment features
        """
        print("\n" + "=" * 70)
        print("COLLECTING SENTIMENT DATA")
        print("=" * 70)
        print(f"Date range: {start_date} to {end_date}\n")

        # Collect each source
        fear_greed = self.get_fear_greed_index(start_date, end_date)
        vix = self.get_vix_data(start_date, end_date)
        btc = self.get_bitcoin_data(start_date, end_date)

        # Merge all sentiment data
        logger.info("\nMerging sentiment data...")

        sentiment_dfs = []

        if fear_greed is not None:
            sentiment_dfs.append(fear_greed)

        if vix is not None:
            sentiment_dfs.append(vix)

        if btc is not None:
            sentiment_dfs.append(btc)

        if len(sentiment_dfs) == 0:
            raise ValueError("Could not collect any sentiment data!")

        # Merge on date index (outer join to keep all dates)
        sentiment_data = sentiment_dfs[0]
        for df in sentiment_dfs[1:]:
            sentiment_data = sentiment_data.join(df, how='outer')

        logger.info(f"Merged sentiment data: {len(sentiment_data)} rows, "
                   f"{len(sentiment_data.columns)} columns")

        print("\n" + "=" * 70)
        print("SENTIMENT DATA COLLECTED")
        print("=" * 70)
        print(f"Total rows: {len(sentiment_data)}")
        print(f"Total columns: {len(sentiment_data.columns)}")
        print(f"Date range: {sentiment_data.index.min().date()} to {sentiment_data.index.max().date()}")
        print("\nColumns:")
        for col in sentiment_data.columns:
            print(f"  - {col}")
        print("=" * 70)

        return sentiment_data


if __name__ == "__main__":
    # Test the sentiment collector
    print("Testing SentimentCollector...")

    collector = SentimentCollector()

    # Test date range (2020-2024)
    start_date = "2020-01-01"
    end_date = "2024-12-31"

    sentiment_data = collector.collect_all_sentiment(start_date, end_date)

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"\nData shape: {sentiment_data.shape}")
    print(f"\nFirst 5 rows:")
    print(sentiment_data.head())
    print(f"\nLast 5 rows:")
    print(sentiment_data.tail())
    print(f"\nMissing values:")
    print(sentiment_data.isnull().sum())

    print("\nSentiment Collector test complete!")
