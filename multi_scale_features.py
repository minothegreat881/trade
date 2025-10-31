"""
MULTI-SCALE FEATURE ENGINEERING FOR 5-DAY TRADING HORIZON
===========================================================

Creates 3 time-scale features:
1. SHORT-TERM (1-5 days): Entry timing, momentum
2. MEDIUM-TERM (10-30 days): Prevailing trend
3. LONG-TERM (60-252 days): Overall market regime

Target: Predict 5-day forward return
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MULTI-SCALE FEATURE ENGINEERING")
print("="*80)


# ================================================================
# HELPER FUNCTIONS
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
    """Calculate MACD"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff


def calculate_bollinger_bands(prices, period=20, std=2):
    """Calculate Bollinger Bands"""
    middle = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    bb_width = (upper - lower) / middle
    bb_position = (prices - lower) / (upper - lower)
    return upper, middle, lower, bb_width, bb_position


def count_consecutive(series, direction='up'):
    """Count consecutive up/down days"""
    if direction == 'up':
        changes = (series.diff() > 0).astype(int)
    else:
        changes = (series.diff() < 0).astype(int)

    consecutive = changes * (changes.groupby((changes != changes.shift()).cumsum()).cumcount() + 1)
    return consecutive


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def linear_regression_slope(series, period):
    """Calculate linear regression slope"""
    slopes = []
    for i in range(len(series)):
        if i < period - 1:
            slopes.append(np.nan)
        else:
            y = series.iloc[i-period+1:i+1].values
            x = np.arange(period)
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)
    return pd.Series(slopes, index=series.index)


# ================================================================
# SHORT-TERM FEATURES (1-5 days)
# ================================================================

def create_short_term_features(df):
    """
    SCALE 1: SHORT-TERM MOMENTUM (1-5 days)
    What's happening NOW? Entry timing.
    """
    print("  [1/3] Creating SHORT-TERM features (1-5 days)...")

    # Returns
    df['st_return_1d'] = df['Close'].pct_change(1)
    df['st_return_3d'] = df['Close'].pct_change(3)
    df['st_return_5d'] = df['Close'].pct_change(5)

    # Fast RSI
    df['st_rsi_5'] = calculate_rsi(df['Close'], period=5)

    # Fast MACD
    macd, macd_signal, macd_diff = calculate_macd(df['Close'], fast=6, slow=13, signal=5)
    df['st_macd_fast'] = macd
    df['st_macd_fast_signal'] = macd_signal
    df['st_macd_fast_diff'] = macd_diff

    # Volatility
    df['st_volatility_5d'] = df['st_return_1d'].rolling(5).std()

    # Volume
    df['st_volume_ratio_5d'] = df['Volume'] / df['Volume'].rolling(5).mean()

    # Gaps and ranges
    df['st_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['st_intraday_range'] = (df['High'] - df['Low']) / df['Open']

    # Consecutive patterns
    df['st_consecutive_up'] = count_consecutive(df['Close'], direction='up')
    df['st_consecutive_down'] = count_consecutive(df['Close'], direction='down')

    # Price position in 5-day range
    rolling_5d_high = df['High'].rolling(5).max()
    rolling_5d_low = df['Low'].rolling(5).min()
    df['st_price_position_5d'] = (df['Close'] - rolling_5d_low) / (rolling_5d_high - rolling_5d_low)

    # Price momentum
    df['st_price_acceleration'] = df['st_return_1d'] - df['st_return_1d'].shift(1)

    return df


# ================================================================
# MEDIUM-TERM FEATURES (10-30 days)
# ================================================================

def create_medium_term_features(df):
    """
    SCALE 2: MEDIUM-TERM TREND (10-30 days)
    What's the trend for our 5-day hold?
    """
    print("  [2/3] Creating MEDIUM-TERM features (10-30 days)...")

    # Returns
    df['mt_return_10d'] = df['Close'].pct_change(10)
    df['mt_return_20d'] = df['Close'].pct_change(20)
    df['mt_return_30d'] = df['Close'].pct_change(30)

    # Moving Averages
    df['mt_sma_10'] = df['Close'].rolling(10).mean()
    df['mt_sma_20'] = df['Close'].rolling(20).mean()
    df['mt_sma_30'] = df['Close'].rolling(30).mean()

    # Price to SMA ratios
    df['mt_price_to_sma_10'] = df['Close'] / df['mt_sma_10']
    df['mt_price_to_sma_20'] = df['Close'] / df['mt_sma_20']

    # Standard RSI
    df['mt_rsi_14'] = calculate_rsi(df['Close'], period=14)

    # Standard MACD
    macd, macd_signal, macd_diff = calculate_macd(df['Close'])
    df['mt_macd'] = macd
    df['mt_macd_signal'] = macd_signal
    df['mt_macd_diff'] = macd_diff

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_width, bb_position = calculate_bollinger_bands(df['Close'])
    df['mt_bb_upper'] = bb_upper
    df['mt_bb_middle'] = bb_middle
    df['mt_bb_lower'] = bb_lower
    df['mt_bb_width'] = bb_width
    df['mt_bb_position'] = bb_position

    # ATR (volatility)
    df['mt_atr_14'] = calculate_atr(df['High'], df['Low'], df['Close'], period=14)
    df['mt_atr_20'] = calculate_atr(df['High'], df['Low'], df['Close'], period=20)

    # Volume trend
    df['mt_volume_trend_20d'] = linear_regression_slope(df['Volume'], period=20)

    # Pattern counts
    df['mt_higher_highs_20d'] = (df['High'] > df['High'].shift(1)).rolling(20).sum()
    df['mt_lower_lows_20d'] = (df['Low'] < df['Low'].shift(1)).rolling(20).sum()

    # Win rate (percentage of positive days)
    df['mt_win_rate_20d'] = (df['Close'] > df['Close'].shift(1)).rolling(20).mean()

    # Trend strength
    df['mt_trend_strength'] = abs(linear_regression_slope(df['Close'], period=20))

    return df


# ================================================================
# LONG-TERM FEATURES (60-252 days)
# ================================================================

def create_long_term_features(df):
    """
    SCALE 3: LONG-TERM CONTEXT (60-252 days)
    Overall market regime and positioning.
    """
    print("  [3/3] Creating LONG-TERM features (60-252 days)...")

    # Long-term returns
    df['lt_return_60d'] = df['Close'].pct_change(60)
    df['lt_return_90d'] = df['Close'].pct_change(90)
    df['lt_return_180d'] = df['Close'].pct_change(180)
    df['lt_return_252d'] = df['Close'].pct_change(252)

    # Long-term moving averages
    df['lt_sma_50'] = df['Close'].rolling(50).mean()
    df['lt_sma_200'] = df['Close'].rolling(200).mean()
    df['lt_ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # Golden/Death cross
    df['lt_golden_cross'] = (df['lt_sma_50'] > df['lt_sma_200']).astype(int)

    # Distance to 52-week high/low
    df['lt_52w_high'] = df['High'].rolling(252).max()
    df['lt_52w_low'] = df['Low'].rolling(252).min()
    df['lt_price_to_52w_high'] = df['Close'] / df['lt_52w_high']
    df['lt_price_to_52w_low'] = df['Close'] / df['lt_52w_low']

    # Position in 52-week range
    df['lt_52w_position'] = (df['Close'] - df['lt_52w_low']) / (df['lt_52w_high'] - df['lt_52w_low'])

    # Volatility regime
    df['lt_volatility_60d'] = df['st_return_1d'].rolling(60).std()
    df['lt_volatility_252d'] = df['st_return_1d'].rolling(252).std()

    # Long-term trend
    df['lt_trend_slope_60d'] = linear_regression_slope(df['Close'], period=60)
    df['lt_trend_slope_252d'] = linear_regression_slope(df['Close'], period=252)

    # Moving average alignment (bullish when all MAs aligned)
    df['lt_ma_alignment'] = ((df['Close'] > df['mt_sma_20']) &
                             (df['mt_sma_20'] > df['lt_sma_50']) &
                             (df['lt_sma_50'] > df['lt_sma_200'])).astype(int)

    # Drawdown from peak
    cumulative_max = df['Close'].cummax()
    df['lt_drawdown'] = (df['Close'] - cumulative_max) / cumulative_max

    # Time-based features
    df['lt_month'] = df.index.month
    df['lt_day_of_week'] = df.index.dayofweek
    df['lt_quarter'] = df.index.quarter

    return df


# ================================================================
# TARGET VARIABLES (5-day forward)
# ================================================================

def create_target_variables(df):
    """
    Create target variables for 5-day trading horizon
    """
    print("\n  Creating TARGET variables (5-day forward)...")

    # 5-day forward return
    df['target_5d_return'] = df['Close'].pct_change(5).shift(-5)

    # Binary: Profit > 3% in next 5 days
    df['target_profit_3pct'] = (df['target_5d_return'] > 0.03).astype(int)

    # Binary: Profit > 0% (any profit)
    df['target_profit_any'] = (df['target_5d_return'] > 0).astype(int)

    # Max drawdown during 5-day hold
    for i in range(len(df)):
        if i + 5 >= len(df):
            df.loc[df.index[i], 'target_max_drawdown_5d'] = np.nan
        else:
            future_prices = df['Close'].iloc[i:i+6]
            entry_price = df['Close'].iloc[i]
            max_dd = ((future_prices - entry_price) / entry_price).min()
            df.loc[df.index[i], 'target_max_drawdown_5d'] = max_dd

    # Max profit during 5-day hold
    for i in range(len(df)):
        if i + 5 >= len(df):
            df.loc[df.index[i], 'target_max_profit_5d'] = np.nan
        else:
            future_prices = df['Close'].iloc[i:i+6]
            entry_price = df['Close'].iloc[i]
            max_profit = ((future_prices - entry_price) / entry_price).max()
            df.loc[df.index[i], 'target_max_profit_5d'] = max_profit

    return df


# ================================================================
# MAIN PROCESSING FUNCTION
# ================================================================

def process_stock(ticker, input_path, output_dir):
    """
    Process single stock with multi-scale features
    """
    print(f"\n{'='*80}")
    print(f"Processing: {ticker}")
    print(f"{'='*80}")

    # Load data
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")

    initial_rows = len(df)

    # Create features
    df = create_short_term_features(df)
    df = create_medium_term_features(df)
    df = create_long_term_features(df)
    df = create_target_variables(df)

    # Drop NaN (first 252 rows due to long-term features)
    df = df.dropna()

    final_rows = len(df)
    dropped_rows = initial_rows - final_rows

    print(f"\n  Final dataset:")
    print(f"    Rows: {final_rows} (dropped {dropped_rows} due to NaN)")
    print(f"    Features: {len(df.columns)}")

    # Count features by category
    st_features = [c for c in df.columns if c.startswith('st_')]
    mt_features = [c for c in df.columns if c.startswith('mt_')]
    lt_features = [c for c in df.columns if c.startswith('lt_')]
    target_features = [c for c in df.columns if c.startswith('target_')]

    print(f"\n  Feature breakdown:")
    print(f"    SHORT-TERM:  {len(st_features)} features")
    print(f"    MEDIUM-TERM: {len(mt_features)} features")
    print(f"    LONG-TERM:   {len(lt_features)} features")
    print(f"    TARGETS:     {len(target_features)} features")
    print(f"    OTHER:       {len(df.columns) - len(st_features) - len(mt_features) - len(lt_features) - len(target_features)}")

    # Save
    output_path = output_dir / f"{ticker}_multiscale.csv"
    df.to_csv(output_path)
    print(f"\n  [OK] Saved to: {output_path}")

    return df


# ================================================================
# MAIN EXECUTION
# ================================================================

if __name__ == "__main__":

    print("\n[1/3] Setting up directories...")

    input_dir = Path('data/sp500_top50')
    output_dir = Path('data/sp500_multiscale')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    # Get list of stocks
    stock_files = list(input_dir.glob('*_features.csv'))
    print(f"\n[2/3] Found {len(stock_files)} stocks to process")

    # Process each stock
    print(f"\n[3/3] Processing stocks...")

    results = []

    for idx, stock_file in enumerate(stock_files, 1):
        ticker = stock_file.stem.replace('_features', '')

        try:
            df = process_stock(ticker, stock_file, output_dir)

            results.append({
                'ticker': ticker,
                'rows': len(df),
                'features': len(df.columns),
                'status': 'OK'
            })

        except Exception as e:
            print(f"\n  [ERROR] {ticker}: {str(e)}")
            results.append({
                'ticker': ticker,
                'rows': 0,
                'features': 0,
                'status': f'ERROR: {str(e)}'
            })

    # Summary
    print(f"\n\n{'='*80}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*80}")

    results_df = pd.DataFrame(results)
    successful = len(results_df[results_df['status'] == 'OK'])

    print(f"\nSuccessfully processed: {successful}/{len(stock_files)} stocks")
    print(f"\nOutput directory: {output_dir}")

    # Save summary
    summary_path = output_dir / 'processing_summary.csv'
    results_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review feature descriptions in processing_summary.csv")
    print("2. Retrain XGBoost models with new features")
    print("3. Compare performance: old features vs multi-scale features")
    print("4. Analyze feature importance by time scale")
    print(f"{'='*80}\n")
