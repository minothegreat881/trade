"""
GET TOP 50 S&P 500 STOCKS BY MARKET CAP
========================================

1. Stiahne S&P 500 zoznam z Wikipedia
2. Získa market cap pre každú akciu
3. Vyberie top 50
4. Vytvorí features pre každú akciu
5. Uloží do CSV

"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import sentiment collector
import sys
sys.path.insert(0, str(Path(__file__).parent))
from sentiment_collector import SentimentCollector

print("="*80)
print("TOP 50 S&P 500 STOCKS - DATA ACQUISITION")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# 1. DOWNLOAD S&P 500 LIST FROM WIKIPEDIA
# ================================================================

print("\n[1/6] Downloading S&P 500 list from Wikipedia...")

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

try:
    # Read table from Wikipedia
    tables = pd.read_html(url)
    sp500_table = tables[0]

    print(f"  Downloaded {len(sp500_table)} companies")

    # Clean column names
    sp500_table.columns = sp500_table.columns.str.strip()

    # Get relevant columns
    sp500_list = sp500_table[['Symbol', 'Security', 'GICS Sector']].copy()
    sp500_list.columns = ['Ticker', 'Company', 'Sector']

    # Clean tickers (remove dots and special chars)
    sp500_list['Ticker'] = sp500_list['Ticker'].str.replace('.', '-', regex=False)

    print(f"  Sample tickers: {', '.join(sp500_list['Ticker'].head(10).tolist())}")

except Exception as e:
    print(f"  ERROR downloading from Wikipedia: {e}")
    print(f"  Using fallback list...")

    # Fallback: Top 100 largest S&P 500 companies
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
               'UNH', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK',
               'ABBV', 'COST', 'PEP', 'AVGO', 'KO', 'LLY', 'WMT', 'BAC', 'TMO',
               'CSCO', 'ACN', 'MCD', 'ABT', 'DHR', 'DIS', 'ADBE', 'VZ', 'CRM',
               'NFLX', 'NKE', 'TXN', 'PM', 'ORCL', 'INTC', 'UPS', 'QCOM', 'RTX',
               'HON', 'NEE', 'BMY', 'LOW', 'AMD', 'AMGN', 'SBUX', 'CAT', 'UNP',
               'T', 'PFE', 'BA', 'GE', 'IBM', 'LMT', 'DE', 'AXP', 'ELV', 'MDT',
               'SPGI', 'GILD', 'BLK', 'ISRG', 'SYK', 'CVS', 'TJX', 'MMC', 'CB',
               'VRTX', 'PLD', 'CI', 'SCHW', 'MDLZ', 'ADI', 'TMUS', 'AMT', 'SO',
               'ZTS', 'CME', 'MO', 'BSX', 'DUK', 'NOW', 'EOG', 'REGN', 'SLB',
               'BDX', 'PNC', 'USB', 'ITW', 'AON', 'CL', 'MMM', 'GD', 'APD', 'CSX']

    sp500_list = pd.DataFrame({
        'Ticker': tickers,
        'Company': ['Unknown'] * len(tickers),
        'Sector': ['Unknown'] * len(tickers)
    })


# ================================================================
# 2. GET MARKET CAP FOR EACH STOCK
# ================================================================

print("\n[2/6] Fetching market cap for all stocks...")

market_caps = []

for idx, row in sp500_list.iterrows():
    ticker = row['Ticker']

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get market cap
        market_cap = info.get('marketCap', None)

        if market_cap is None or market_cap == 0:
            # Try alternative
            market_cap = info.get('market_cap', None)

        market_caps.append({
            'Ticker': ticker,
            'Company': row['Company'],
            'Sector': row['Sector'],
            'MarketCap': market_cap if market_cap else 0
        })

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(sp500_list)} stocks...")

    except Exception as e:
        print(f"  ERROR with {ticker}: {e}")
        market_caps.append({
            'Ticker': ticker,
            'Company': row['Company'],
            'Sector': row['Sector'],
            'MarketCap': 0
        })

market_cap_df = pd.DataFrame(market_caps)

# Remove stocks with no market cap
market_cap_df = market_cap_df[market_cap_df['MarketCap'] > 0]

print(f"\n  Found market cap for {len(market_cap_df)} stocks")


# ================================================================
# 3. SELECT TOP 50 BY MARKET CAP
# ================================================================

print("\n[3/6] Selecting top 50 stocks by market cap...")

# Sort by market cap
market_cap_df = market_cap_df.sort_values('MarketCap', ascending=False)

# Get top 50
top50 = market_cap_df.head(50).copy()

# Format market cap
top50['MarketCap_B'] = (top50['MarketCap'] / 1e9).round(2)

print(f"\n  TOP 10 STOCKS:")
for idx, row in top50.head(10).iterrows():
    print(f"    {row['Ticker']:6s} - {row['Company'][:30]:30s} ${row['MarketCap_B']:8.2f}B")

# Save tickers list
top50[['Ticker', 'Company', 'Sector', 'MarketCap']].to_csv(
    'data/sp500_top50_tickers.csv',
    index=False
)

print(f"\n  [OK] Saved to data/sp500_top50_tickers.csv")


# ================================================================
# 4. DOWNLOAD HISTORICAL DATA
# ================================================================

print("\n[4/6] Downloading historical data for top 50...")

start_date = '2020-01-01'
end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

print(f"  Period: {start_date} -> {end_date}")

stock_data = {}
failed_tickers = []

for idx, row in top50.iterrows():
    ticker = row['Ticker']

    try:
        print(f"  Downloading {ticker}... ({idx - top50.index[0] + 1}/50)", end='\r')

        stock = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

        if len(stock) > 0:
            # Flatten multi-level columns if present (happens with auto_adjust=False)
            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = stock.columns.get_level_values(0)
            stock_data[ticker] = stock
        else:
            print(f"\n  WARNING: No data for {ticker}")
            failed_tickers.append(ticker)

    except Exception as e:
        print(f"\n  ERROR downloading {ticker}: {e}")
        failed_tickers.append(ticker)

print(f"\n\n  Successfully downloaded: {len(stock_data)}/{50} stocks")

if failed_tickers:
    print(f"  Failed tickers: {', '.join(failed_tickers)}")
    # Remove failed tickers from top50
    top50 = top50[~top50['Ticker'].isin(failed_tickers)]


# ================================================================
# 5. CREATE FEATURES FOR EACH STOCK
# ================================================================

print("\n[5/6] Creating features for each stock...")

def create_technical_features(df):
    """
    Create technical features (same as SPY)
    """
    # Returns
    df['return_1d'] = df['Close'].pct_change()
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)
    df['return_20d'] = df['Close'].pct_change(20)

    # Volatility
    df['volatility_20d'] = df['return_1d'].rolling(20).std()
    df['volatility_60d'] = df['return_1d'].rolling(60).std()

    # Volume
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Price position
    df['price_position'] = (df['Close'] - df['Close'].rolling(252).min()) / \
                           (df['Close'].rolling(252).max() - df['Close'].rolling(252).min())

    # Moving averages
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()

    # Momentum
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    df['momentum_60'] = df['Close'] / df['Close'].shift(60) - 1

    # Trend
    df['trend'] = (df['sma_20'] > df['sma_50']).astype(int)

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(14).mean()

    # Target (5-day forward return)
    df['target'] = df['Close'].pct_change(5).shift(-5)

    return df


# Get sentiment data once (shared across all stocks)
print("\n  Fetching sentiment data...")
try:
    collector = SentimentCollector()
    sentiment_df = collector.collect_all_sentiment(start_date, end_date)
    print(f"    [OK] Sentiment data: {len(sentiment_df)} days")
except Exception as e:
    print(f"    [WARNING] Could not fetch sentiment: {e}")
    sentiment_df = None


# Create features for each stock
all_stock_datasets = {}

for ticker in top50['Ticker']:
    if ticker not in stock_data:
        continue

    print(f"  Creating features for {ticker}...", end='\r')

    try:
        # Get stock data
        df = stock_data[ticker].copy()

        # Create technical features
        df = create_technical_features(df)

        # Merge sentiment data if available
        if sentiment_df is not None:
            df = df.merge(sentiment_df, left_index=True, right_index=True, how='left')
            df[sentiment_df.columns] = df[sentiment_df.columns].ffill()

        # Drop NaN
        df = df.dropna()

        if len(df) > 0:
            all_stock_datasets[ticker] = df

    except Exception as e:
        print(f"\n  ERROR creating features for {ticker}: {e}")

print(f"\n\n  Created features for {len(all_stock_datasets)}/{len(top50)} stocks")


# ================================================================
# 6. SAVE TO CSV
# ================================================================

print("\n[6/6] Saving datasets to CSV...")

# Create directory
output_dir = Path('data/sp500_top50')
output_dir.mkdir(parents=True, exist_ok=True)

# Save each stock separately
for ticker, df in all_stock_datasets.items():
    output_file = output_dir / f"{ticker}_features.csv"
    df.to_csv(output_file)
    print(f"  Saved {ticker}: {len(df)} rows, {len(df.columns)} columns")

# Create combined summary
summary_data = []

for ticker in top50['Ticker']:
    if ticker in all_stock_datasets:
        df = all_stock_datasets[ticker]
        row_info = top50[top50['Ticker'] == ticker].iloc[0]

        summary_data.append({
            'Ticker': ticker,
            'Company': row_info['Company'],
            'Sector': row_info['Sector'],
            'MarketCap_B': row_info['MarketCap_B'],
            'DataPoints': len(df),
            'Features': len(df.columns),
            'StartDate': df.index.min().strftime('%Y-%m-%d'),
            'EndDate': df.index.max().strftime('%Y-%m-%d'),
            'AvgReturn': df['return_1d'].mean(),
            'Volatility': df['return_1d'].std(),
            'DataFile': f"{ticker}_features.csv"
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('data/sp500_top50_summary.csv', index=False)

print(f"\n  [OK] Summary saved to data/sp500_top50_summary.csv")


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "="*80)
print("SUMMARY - TOP 50 S&P 500 STOCKS")
print("="*80)

print(f"\nTotal stocks with data: {len(all_stock_datasets)}")
print(f"Period: {start_date} -> {end_date}")

if len(all_stock_datasets) > 0:
    sample_ticker = list(all_stock_datasets.keys())[0]
    sample_df = all_stock_datasets[sample_ticker]
    print(f"Features per stock: {len(sample_df.columns)}")
    print(f"Avg data points: {np.mean([len(df) for df in all_stock_datasets.values()]):.0f}")

print(f"\nFiles created:")
print(f"  - data/sp500_top50_tickers.csv (basic info)")
print(f"  - data/sp500_top50_summary.csv (detailed summary)")
print(f"  - data/sp500_top50/{'{ticker}'}_features.csv (individual datasets)")

print("\n" + "="*80)
print("TOP 10 BY MARKET CAP:")
print("="*80)

for idx, row in summary_df.head(10).iterrows():
    print(f"  {idx+1:2d}. {row['Ticker']:6s} - {row['Company'][:25]:25s} ${row['MarketCap_B']:8.2f}B")

print("\n" + "="*80)
print("DONE!")
print("="*80)

print(f"\nNext steps:")
print(f"  1. Use these datasets for portfolio optimization")
print(f"  2. Train models on individual stocks")
print(f"  3. Create multi-stock trading strategy")
