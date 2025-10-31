"""
REAL-TIME DATA PIPELINE
========================

Stahuje real-time data pre top 50 S&P 500 akcii a uklada do InfluxDB.

Features:
- Kazdu minutu stahuje data z Yahoo Finance
- Vypocitava technicke indikatory (RSI, MACD, MA, volatilita)
- Uklada do InfluxDB pre Grafanu
- Error handling a logging
- Graceful shutdown
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import logging
import time
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# InfluxDB Configuration
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "your-super-secret-token"  # Change this!
INFLUX_ORG = "trading-org"
INFLUX_BUCKET = "stock-data"

# Top 50 S&P 500 stocks
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'PG', 'XOM', 'MA', 'HD', 'CVX', 'LLY', 'ABBV', 'MRK',
    'AVGO', 'PEP', 'KO', 'COST', 'WMT', 'MCD', 'CSCO', 'TMO', 'ACN', 'ABT',
    'ADBE', 'DHR', 'VZ', 'CMCSA', 'NKE', 'CRM', 'NFLX', 'TXN', 'INTC', 'DIS',
    'AMD', 'PFE', 'PM', 'ORCL', 'WFC', 'UPS', 'RTX', 'HON', 'QCOM', 'LIN'
]

# ================================================================
# TECHNICAL INDICATORS
# ================================================================

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_moving_averages(prices):
    """Calculate multiple moving averages"""
    return {
        'sma_10': prices.rolling(window=10).mean(),
        'sma_20': prices.rolling(window=20).mean(),
        'sma_50': prices.rolling(window=50).mean(),
        'sma_200': prices.rolling(window=200).mean()
    }

def calculate_volatility(returns, period=20):
    """Calculate rolling volatility"""
    return returns.rolling(window=period).std() * np.sqrt(252) * 100

# ================================================================
# DATA FETCHING
# ================================================================

def fetch_stock_data(ticker, period='60d'):
    """
    Fetch historical data for a stock

    Args:
        ticker: Stock symbol
        period: Data period (60d, 1mo, etc.)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval='1d')

        if df.empty:
            logger.warning(f"{ticker}: No data returned")
            return None

        return df

    except Exception as e:
        logger.error(f"{ticker}: Error fetching data - {e}")
        return None

def calculate_all_indicators(df):
    """
    Calculate all technical indicators

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with calculated indicators
    """
    if df is None or df.empty or len(df) < 50:
        return None

    # Calculate returns
    df['returns'] = df['Close'].pct_change()

    # RSI
    df['rsi_14'] = calculate_rsi(df['Close'], 14)
    df['rsi_5'] = calculate_rsi(df['Close'], 5)

    # MACD
    macd, signal, histogram = calculate_macd(df['Close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_histogram'] = histogram

    # Moving averages
    mas = calculate_moving_averages(df['Close'])
    for name, values in mas.items():
        df[name] = values

    # Volatility
    df['volatility_20d'] = calculate_volatility(df['returns'], 20)

    # Price momentum
    df['momentum_5d'] = df['Close'].pct_change(5)
    df['momentum_20d'] = df['Close'].pct_change(20)

    return df

# ================================================================
# INFLUXDB OPERATIONS
# ================================================================

def write_to_influxdb(client, ticker, data):
    """
    Write stock data to InfluxDB

    Args:
        client: InfluxDB client
        ticker: Stock symbol
        data: DataFrame with stock data
    """
    if data is None or data.empty:
        return

    write_api = client.write_api(write_options=SYNCHRONOUS)

    # Get latest row
    latest = data.iloc[-1]
    timestamp = data.index[-1]

    # Create point
    point = Point("stock_data") \
        .tag("ticker", ticker) \
        .field("open", float(latest['Open'])) \
        .field("high", float(latest['High'])) \
        .field("low", float(latest['Low'])) \
        .field("close", float(latest['Close'])) \
        .field("volume", int(latest['Volume'])) \
        .time(timestamp, WritePrecision.NS)

    # Add technical indicators if available
    indicators = [
        'rsi_14', 'rsi_5', 'macd', 'macd_signal', 'macd_histogram',
        'sma_10', 'sma_20', 'sma_50', 'sma_200',
        'volatility_20d', 'momentum_5d', 'momentum_20d'
    ]

    for indicator in indicators:
        if indicator in latest and not pd.isna(latest[indicator]):
            point = point.field(indicator, float(latest[indicator]))

    # Write to InfluxDB
    try:
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        logger.info(f"{ticker}: Data written to InfluxDB")
    except Exception as e:
        logger.error(f"{ticker}: Error writing to InfluxDB - {e}")

# ================================================================
# MAIN PIPELINE
# ================================================================

def run_pipeline_iteration(client):
    """
    Run one iteration of the data pipeline

    Fetches data for all tickers and writes to InfluxDB
    """
    logger.info("="*80)
    logger.info("Starting pipeline iteration")
    logger.info("="*80)

    success_count = 0
    error_count = 0

    for ticker in TICKERS:
        try:
            logger.info(f"Processing {ticker}...")

            # Fetch data
            df = fetch_stock_data(ticker, period='60d')

            if df is None:
                error_count += 1
                continue

            # Calculate indicators
            df_indicators = calculate_all_indicators(df)

            if df_indicators is None:
                error_count += 1
                continue

            # Write to InfluxDB
            write_to_influxdb(client, ticker, df_indicators)

            success_count += 1

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"{ticker}: Unexpected error - {e}")
            error_count += 1
            continue

    logger.info("="*80)
    logger.info(f"Pipeline iteration complete")
    logger.info(f"  Success: {success_count}/{len(TICKERS)}")
    logger.info(f"  Errors:  {error_count}/{len(TICKERS)}")
    logger.info("="*80)

def main():
    """
    Main function - runs continuous data pipeline
    """
    # Create logs directory
    Path('logs').mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("REAL-TIME DATA PIPELINE STARTING")
    logger.info("="*80)
    logger.info(f"InfluxDB URL: {INFLUX_URL}")
    logger.info(f"Bucket: {INFLUX_BUCKET}")
    logger.info(f"Tickers: {len(TICKERS)}")
    logger.info("="*80)

    # Initialize InfluxDB client
    try:
        client = InfluxDBClient(
            url=INFLUX_URL,
            token=INFLUX_TOKEN,
            org=INFLUX_ORG
        )
        logger.info("InfluxDB client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize InfluxDB client: {e}")
        sys.exit(1)

    # Run pipeline
    iteration = 0
    try:
        while True:
            iteration += 1
            logger.info(f"\n*** ITERATION {iteration} ***")

            # Run pipeline
            run_pipeline_iteration(client)

            # Wait 1 minute (60 seconds)
            # In production, adjust based on data frequency needs
            logger.info("Waiting 60 seconds until next iteration...")
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("\nShutdown signal received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        client.close()
        logger.info("InfluxDB client closed")
        logger.info("Pipeline stopped")

if __name__ == "__main__":
    main()
