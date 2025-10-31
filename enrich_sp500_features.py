"""
ENRICH TOP 50 S&P 500 STOCKS WITH ALL FEATURES
==============================================

Adds all advanced features to match the full_dataset_2020_2025.csv structure
- Reads existing sp500_top50 CSV files (40 features)
- Adds 90+ advanced features
- Saves enriched versions with same features as SPY dataset

"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ENRICHING TOP 50 S&P 500 STOCKS WITH ADVANCED FEATURES")
print("="*80)

# ================================================================
# ADVANCED FEATURE CREATION FUNCTIONS
# ================================================================

def create_advanced_technical_indicators(df):
    """
    Create advanced technical indicators
    """

    # ==== OSCILLATORS ====

    # RSI (additional periods)
    for period in [7, 30]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # Williams %R
    df['willr'] = -100 * (high_14 - df['Close']) / (high_14 - low_14)

    # ROC (Rate of Change)
    df['roc'] = df['Close'].pct_change(10) * 100

    # ==== MOVING AVERAGES ====

    # Additional SMAs
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()

    # Trend alignment
    df['trend_alignment'] = ((df['sma_10'] > df['sma_50']) &
                              (df['sma_50'] > df['sma_200'])).astype(int)

    # ==== VOLATILITY ====

    # Historical volatility
    df['hist_vol_20'] = df['return_1d'].rolling(20).std()
    df['hist_vol_60'] = df['return_1d'].rolling(60).std()
    df['vol_ratio'] = df['hist_vol_20'] / df['hist_vol_60']

    # Parkinson volatility (High-Low estimator)
    df['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) *
                                   ((np.log(df['High']/df['Low']))**2).rolling(20).mean())

    # ATR (additional period)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr_20'] = ranges.max(axis=1).rolling(20).mean()

    # ==== DIRECTIONAL INDICATORS ====

    # Plus/Minus Directional Indicators
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = ranges.max(axis=1)
    atr_14 = tr.rolling(14).mean()

    df['plus_di'] = 100 * (plus_dm.rolling(14).mean() / atr_14)
    df['minus_di'] = 100 * (minus_dm.rolling(14).mean() / atr_14)
    df['adx'] = 100 * (abs(df['plus_di'] - df['minus_di']) /
                       (df['plus_di'] + df['minus_di'])).rolling(14).mean()
    df['di_diff'] = df['plus_di'] - df['minus_di']

    # ==== AROON ====

    def aroon(series, period=25):
        aroon_up = series.rolling(period + 1).apply(
            lambda x: (period - (period - x.argmax())) / period * 100, raw=True
        )
        aroon_down = series.rolling(period + 1).apply(
            lambda x: (period - (period - x.argmin())) / period * 100, raw=True
        )
        return aroon_up, aroon_down

    df['aroon_up'], df['aroon_down'] = aroon(df['Close'])
    df['aroon_osc'] = df['aroon_up'] - df['aroon_down']

    # ==== TREND STRENGTH ====

    df['trend_strength'] = abs(df['aroon_osc']) / 100

    # ==== LINEAR REGRESSION SLOPES ====

    def linreg_slope(series, period):
        def calc_slope(y):
            x = np.arange(len(y))
            if len(y) < 2:
                return 0
            return np.polyfit(x, y, 1)[0]
        return series.rolling(period).apply(calc_slope, raw=False)

    df['linreg_slope_20'] = linreg_slope(df['Close'], 20)
    df['linreg_slope_50'] = linreg_slope(df['Close'], 50)

    return df


def create_volume_indicators(df):
    """
    Create volume-based indicators
    """

    # ==== OBV (On-Balance Volume) ====

    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv

    # ==== MFI (Money Flow Index) ====

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()

    mfi_ratio = positive_flow / negative_flow
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))

    # ==== A/D (Accumulation/Distribution) ====

    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    clv = clv.fillna(0)
    df['ad'] = (clv * df['Volume']).cumsum()

    # ==== VWAP ====

    df['vwap'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

    # ==== Volume Features ====

    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio_adv'] = df['Volume'] / df['volume_sma_20']
    df['volume_trend'] = (df['volume_sma_20'] > df['volume_sma_20'].shift(5)).astype(int)

    # OBV Slope
    df['obv_slope'] = df['obv'].diff(5) / df['obv'].shift(5)

    # Price-Volume Correlation
    df['pv_corr'] = df['Close'].rolling(20).corr(df['Volume'])

    return df


def create_price_patterns(df):
    """
    Create candlestick and price pattern features
    """

    # ==== PRICE PATTERNS ====

    df['higher_high'] = ((df['High'] > df['High'].shift(1)) &
                          (df['High'].shift(1) > df['High'].shift(2))).astype(int)
    df['lower_low'] = ((df['Low'] < df['Low'].shift(1)) &
                        (df['Low'].shift(1) < df['Low'].shift(2))).astype(int)

    # Gaps
    df['gap_up'] = (df['Low'] > df['High'].shift(1)).astype(int)
    df['gap_down'] = (df['High'] < df['Low'].shift(1)).astype(int)

    # ==== CANDLESTICK PATTERNS ====

    body = abs(df['Close'] - df['Open'])
    range_hl = df['High'] - df['Low']

    df['body_size'] = body / df['Close']
    df['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / range_hl
    df['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / range_hl

    # Doji
    df['is_doji'] = (body < (range_hl * 0.1)).astype(int)

    # Engulfing patterns
    df['bullish_engulfing'] = ((df['Close'] > df['Open']) &  # Current is bullish
                                 (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous is bearish
                                 (df['Open'] < df['Close'].shift(1)) &  # Opens below previous close
                                 (df['Close'] > df['Open'].shift(1))).astype(int)  # Closes above previous open

    df['bearish_engulfing'] = ((df['Close'] < df['Open']) &  # Current is bearish
                                 (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous is bullish
                                 (df['Open'] > df['Close'].shift(1)) &  # Opens above previous close
                                 (df['Close'] < df['Open'].shift(1))).astype(int)  # Closes below previous open

    # ==== SUPPORT/RESISTANCE ====

    df['resistance_20'] = df['High'].rolling(20).max()
    df['support_20'] = df['Low'].rolling(20).min()

    df['distance_to_resistance'] = (df['resistance_20'] - df['Close']) / df['Close']
    df['distance_to_support'] = (df['Close'] - df['support_20']) / df['Close']

    return df


def create_timeframe_features(df):
    """
    Create weekly/monthly features
    """

    # ==== WEEKLY FEATURES ====

    df['weekly_return'] = df['return_5d']  # Approximate
    df['weekly_high'] = df['High'].rolling(5).max()
    df['weekly_low'] = df['Low'].rolling(5).min()
    df['weekly_range'] = (df['weekly_high'] - df['weekly_low']) / df['weekly_low']
    df['position_in_weekly_range'] = (df['Close'] - df['weekly_low']) / (df['weekly_high'] - df['weekly_low'])

    # ==== MONTHLY FEATURES ====

    df['monthly_return'] = df['return_20d']  # Approximate
    df['monthly_high'] = df['High'].rolling(20).max()
    df['monthly_low'] = df['Low'].rolling(20).min()
    df['position_in_monthly_range'] = (df['Close'] - df['monthly_low']) / (df['monthly_high'] - df['monthly_low'])

    return df


def create_liquidity_features(df):
    """
    Create liquidity and microstructure features
    """

    # ==== SPREAD FEATURES ====

    df['spread'] = df['High'] - df['Low']
    df['spread_ma'] = df['spread'].rolling(20).mean()
    df['spread_ratio'] = df['spread'] / df['spread_ma']

    # ==== PRICE IMPACT ====

    df['price_impact'] = df['return_1d'] / (df['volume_ratio'] + 0.001)

    # ==== LIQUIDITY ====

    df['liquidity'] = df['Volume'] * df['Close']

    # ==== INTRADAY FEATURES ====

    df['intraday_vol'] = (df['High'] - df['Low']) / df['Open']
    df['overnight_return'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['intraday_momentum'] = (df['Close'] - df['Open']) / df['Open']

    return df


def create_sentiment_features(df):
    """
    Create sentiment-derived features
    """

    # ==== FEAR & GREED FEATURES ====

    if 'fear_greed_value' in df.columns:
        df['fear_greed_ma5'] = df['fear_greed_value'].rolling(5).mean()
        df['fear_greed_ma10'] = df['fear_greed_value'].rolling(10).mean()
        df['fear_greed_extreme_fear'] = (df['fear_greed_value'] < 25).astype(int)
        df['fear_greed_extreme_greed'] = (df['fear_greed_value'] > 75).astype(int)
        df['fear_greed_change_5d'] = df['fear_greed_value'].diff(5)

    # ==== VIX FEATURES ====

    if 'VIX' in df.columns:
        df['vix_ma5'] = df['VIX'].rolling(5).mean()
        df['vix_ma20'] = df['VIX'].rolling(20).mean()
        df['vix_regime'] = (df['VIX'] > 20).astype(int) + (df['VIX'] > 30).astype(int)
        df['vix_spike'] = (df['VIX'] > df['vix_ma20'] * 1.2).astype(int)

        # VIX Z-score
        vix_mean = df['VIX'].rolling(60).mean()
        vix_std = df['VIX'].rolling(60).std()
        df['vix_zscore'] = (df['VIX'] - vix_mean) / vix_std

    # ==== BTC FEATURES ====

    if 'BTC_Close' in df.columns:
        df['btc_return_5d'] = df['BTC_return_5d']
        df['btc_return_10d'] = df['BTC_return_10d']
        df['btc_volatility_10d'] = df['BTC_return_1d'].rolling(10).std()
        df['btc_momentum'] = (df['BTC_Close'] > df['BTC_Close'].rolling(20).mean()).astype(int)

    # ==== COMPOSITE SENTIMENT ====

    if all(col in df.columns for col in ['fear_greed_value', 'VIX', 'BTC_Close']):
        # Normalize to 0-1 scale
        fg_norm = (df['fear_greed_value'] - 0) / 100
        vix_norm = 1 - ((df['VIX'] - 10).clip(0, 80) / 80)  # Inverse: high VIX = low sentiment
        btc_norm = (df['BTC_return_10d'] + 0.5).clip(0, 1)  # Center around 0

        df['composite_sentiment'] = (fg_norm + vix_norm + btc_norm) / 3

    return df


def create_research_features(df):
    """
    Create research-specific features (duplicates with different names)
    """

    # RSI research
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_research_14'] = 100 - (100 / (1 + rs))

    gain_30 = (delta.where(delta > 0, 0)).rolling(30).mean()
    loss_30 = (-delta.where(delta < 0, 0)).rolling(30).mean()
    rs_30 = gain_30 / loss_30
    df['rsi_research_30'] = 100 - (100 / (1 + rs_30))

    # RSI features
    df['rsi_14_ma'] = df['rsi'].rolling(14).mean()
    df['rsi_divergence'] = df['rsi'] - df['rsi_14_ma']

    # MACD research
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd_research'] = exp1 - exp2
    df['macd_signal_research'] = df['macd_research'].ewm(span=9, adjust=False).mean()
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                             (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)

    # ATR research
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr_research_14'] = ranges.max(axis=1).rolling(14).mean()

    # Bollinger research
    df['bollinger_pct_b_research'] = df['bb_position']
    df['bb_width_research'] = df['bb_width']
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.5).astype(int)

    # ADX research
    df['adx_research'] = df['adx']

    # Stochastic research
    df['stochastic_k_research'] = df['stoch_k']

    # Momentum
    df['momentum'] = df['Close'] - df['Close'].shift(10)

    return df


# ================================================================
# MAIN PROCESSING
# ================================================================

def enrich_stock(ticker):
    """
    Enrich single stock with all advanced features
    """

    input_file = Path(f'data/sp500_top50/{ticker}_features.csv')

    if not input_file.exists():
        print(f"  [SKIP] {ticker} - file not found")
        return None

    try:
        # Load existing data
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
        initial_cols = len(df.columns)

        # Add advanced features
        df = create_advanced_technical_indicators(df)
        df = create_volume_indicators(df)
        df = create_price_patterns(df)
        df = create_timeframe_features(df)
        df = create_liquidity_features(df)
        df = create_sentiment_features(df)
        df = create_research_features(df)

        # Drop NaN
        df = df.dropna()

        final_cols = len(df.columns)
        added_cols = final_cols - initial_cols

        # Save enriched version
        df.to_csv(input_file)

        return {
            'ticker': ticker,
            'rows': len(df),
            'initial_cols': initial_cols,
            'final_cols': final_cols,
            'added_cols': added_cols
        }

    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return None


# ================================================================
# PROCESS ALL STOCKS
# ================================================================

print("\n[1/2] Loading stock list...")

summary_file = Path('data/sp500_top50_summary.csv')
summary = pd.read_csv(summary_file)

print(f"  Found {len(summary)} stocks to enrich")

print("\n[2/2] Enriching stocks with advanced features...")

results = []

for idx, row in summary.iterrows():
    ticker = row['Ticker']
    print(f"  [{idx+1}/{len(summary)}] Enriching {ticker}...", end='\r')

    result = enrich_stock(ticker)
    if result:
        results.append(result)

print("\n")

# ================================================================
# SUMMARY
# ================================================================

if len(results) > 0:
    results_df = pd.DataFrame(results)

    print("="*80)
    print("ENRICHMENT SUMMARY")
    print("="*80)

    print(f"\nTotal stocks enriched: {len(results)}/{len(summary)}")
    print(f"Average features added: {results_df['added_cols'].mean():.0f}")
    print(f"Average final columns: {results_df['final_cols'].mean():.0f}")
    print(f"Average rows after dropna: {results_df['rows'].mean():.0f}")

    print(f"\nSample (first 10):")
    print(results_df.head(10).to_string(index=False))

    # Update summary
    for result in results:
        summary.loc[summary['Ticker'] == result['ticker'], 'Features'] = result['final_cols']
        summary.loc[summary['Ticker'] == result['ticker'], 'DataPoints'] = result['rows']

    summary.to_csv(summary_file, index=False)

    print(f"\n[OK] Updated summary: {summary_file}")

else:
    print("\n[ERROR] No stocks were enriched!")

print("\n" + "="*80)
print("DONE!")
print("="*80)
