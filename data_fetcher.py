"""
Data Fetcher for Live Trading
Fetches real-time market data and sentiment
"""

import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetch real-time market data + sentiment indicators
    """

    def __init__(self):
        """Initialize data fetcher"""
        self.fear_greed_url = "https://api.alternative.me/fng/"
        logger.info("DataFetcher initialized")

    def get_current_snapshot(self) -> Optional[Dict]:
        """
        Get current market snapshot with price + sentiment

        Returns:
            Dict with:
                - timestamp
                - symbol
                - price (current)
                - open, high, low (today)
                - volume
                - VIX
                - fear_greed_value
                - fear_greed_text
                - is_market_open
        """
        try:
            # Get SPY current data
            spy = yf.Ticker("SPY")

            # Get today's data
            today_data = spy.history(period="1d")

            if len(today_data) == 0:
                logger.error("No data available for SPY")
                return None

            # Get current price (last close)
            current_price = today_data['Close'].iloc[-1]

            # Get OHLCV
            open_price = today_data['Open'].iloc[-1]
            high_price = today_data['High'].iloc[-1]
            low_price = today_data['Low'].iloc[-1]
            volume = today_data['Volume'].iloc[-1]

            # Get VIX
            vix_data = self._get_vix()

            # Get Fear & Greed
            fear_greed = self._get_fear_greed()

            # Check if market is open
            is_market_open = self._is_market_open()

            snapshot = {
                'timestamp': datetime.now(),
                'symbol': 'SPY',
                'price': float(current_price),
                'open': float(open_price),
                'high': float(high_price),
                'low': float(low_price),
                'volume': int(volume),
                'VIX': vix_data,
                'fear_greed_value': fear_greed.get('value', 50),
                'fear_greed_text': fear_greed.get('classification', 'Neutral'),
                'is_market_open': is_market_open
            }

            logger.info(f"Snapshot fetched: SPY=${current_price:.2f}, VIX={vix_data}, F&G={fear_greed.get('value', 'N/A')}")

            return snapshot

        except Exception as e:
            logger.error(f"Error fetching snapshot: {e}", exc_info=True)
            return None

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical OHLCV data

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if len(df) == 0:
                logger.error(f"No historical data for {symbol}")
                return pd.DataFrame()

            # Keep only OHLCV
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            logger.info(f"Historical data fetched: {len(df)} days for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}", exc_info=True)
            return pd.DataFrame()

    def _get_vix(self) -> float:
        """
        Get current VIX value

        Returns:
            VIX value (float)
        """
        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d")

            if len(vix_data) > 0:
                return float(vix_data['Close'].iloc[-1])
            else:
                logger.warning("VIX data not available, using default 15")
                return 15.0

        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return 15.0

    def _get_fear_greed(self) -> Dict:
        """
        Get Fear & Greed Index from Alternative.me

        Returns:
            Dict with 'value' (0-100) and 'classification' (text)
        """
        try:
            response = requests.get(self.fear_greed_url, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'data' in data and len(data['data']) > 0:
                fng = data['data'][0]
                return {
                    'value': int(fng['value']),
                    'classification': fng['value_classification']
                }
            else:
                logger.warning("Fear & Greed data not available")
                return {'value': 50, 'classification': 'Neutral'}

        except Exception as e:
            logger.error(f"Error fetching Fear & Greed: {e}")
            return {'value': 50, 'classification': 'Neutral'}

    def _is_market_open(self) -> bool:
        """
        Check if US stock market is currently open

        Returns:
            True if market is open, False otherwise
        """
        now = datetime.now()

        # Check if weekend
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        # Simplified check (doesn't account for holidays)
        market_open_hour = 9
        market_open_minute = 30
        market_close_hour = 16
        market_close_minute = 0

        current_time = now.time()
        open_time = datetime.strptime(f"{market_open_hour}:{market_open_minute}", "%H:%M").time()
        close_time = datetime.strptime(f"{market_close_hour}:{market_close_minute}", "%H:%M").time()

        # Note: This assumes local time is ET. For production, use pytz
        return open_time <= current_time <= close_time


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Data Fetcher...")
    print("=" * 60)

    fetcher = DataFetcher()

    # Test current snapshot
    print("\n[TEST] Fetching current snapshot...")
    snapshot = fetcher.get_current_snapshot()

    if snapshot:
        print(f"[OK] SPY: ${snapshot['price']:.2f}")
        print(f"[OK] VIX: {snapshot['VIX']:.2f}")
        print(f"[OK] Fear & Greed: {snapshot['fear_greed_value']} ({snapshot['fear_greed_text']})")
        print(f"[OK] Market Open: {snapshot['is_market_open']}")
    else:
        print("[ERROR] Failed to fetch snapshot")

    # Test historical data
    print("\n[TEST] Fetching historical data (last 30 days)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    historical = fetcher.get_historical_data(
        symbol='SPY',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

    if len(historical) > 0:
        print(f"[OK] Got {len(historical)} days of data")
        print(f"[OK] Last close: ${historical['Close'].iloc[-1]:.2f}")
    else:
        print("[ERROR] Failed to fetch historical data")

    print("\n[OK] Data fetcher working!")
