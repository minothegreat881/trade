"""
Advanced Feature Engineering for Trading Model
Adds 30+ sophisticated technical indicators to improve predictions

Categories:
1. Momentum indicators (RSI, MACD, Stochastic, etc.)
2. Volatility indicators (ATR, Bollinger Bands, etc.)
3. Trend indicators (ADX, Aroon, etc.)
4. Volume indicators (OBV, MFI, etc.)
5. Multi-timeframe features
6. Market microstructure
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Try importing ta-lib, fallback to pandas_ta if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    print("[WARNING] TA-Lib not found, using pandas fallback")
    HAS_TALIB = False


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for financial time series
    """

    def __init__(self):
        self.has_talib = HAS_TALIB

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all advanced features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional features
        """
        print("[INFO] Creating advanced features...")

        # Make a copy
        df = df.copy()

        # 1. Momentum indicators
        df = self.add_momentum_features(df)
        print("[INFO]   Momentum features added")

        # 2. Volatility indicators
        df = self.add_volatility_features(df)
        print("[INFO]   Volatility features added")

        # 3. Trend indicators
        df = self.add_trend_features(df)
        print("[INFO]   Trend features added")

        # 4. Volume indicators
        df = self.add_volume_features(df)
        print("[INFO]   Volume features added")

        # 5. Pattern recognition
        df = self.add_pattern_features(df)
        print("[INFO]   Pattern features added")

        # 6. Multi-timeframe
        df = self.add_multitimeframe_features(df)
        print("[INFO]   Multi-timeframe features added")

        # 7. Market microstructure
        df = self.add_microstructure_features(df)
        print("[INFO]   Microstructure features added")

        # Count new features
        new_features = [col for col in df.columns
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"[INFO] Total advanced features created: {len(new_features)}")

        return df

    # ================================================================
    # 1. MOMENTUM INDICATORS
    # ================================================================

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based indicators
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Stochastic Oscillator
        - Williams %R
        - ROC (Rate of Change)
        - Momentum
        """
        if self.has_talib:
            # RSI - multiple periods
            df['rsi_14'] = talib.RSI(df['Close'], timeperiod=14)
            df['rsi_30'] = talib.RSI(df['Close'], timeperiod=30)
            df['rsi_7'] = talib.RSI(df['Close'], timeperiod=7)

            # MACD
            macd, signal, hist = talib.MACD(df['Close'],
                                           fastperiod=12,
                                           slowperiod=26,
                                           signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist

            # Stochastic
            slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'],
                                      fastk_period=14, slowk_period=3,
                                      slowd_period=3)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd

            # Williams %R
            df['willr'] = talib.WILLR(df['High'], df['Low'], df['Close'],
                                      timeperiod=14)

            # ROC (Rate of Change)
            df['roc'] = talib.ROC(df['Close'], timeperiod=10)

            # Momentum
            df['momentum'] = talib.MOM(df['Close'], timeperiod=10)

        else:
            # Pandas fallback
            # RSI
            df['rsi_14'] = self._calculate_rsi(df['Close'], 14)
            df['rsi_30'] = self._calculate_rsi(df['Close'], 30)
            df['rsi_7'] = self._calculate_rsi(df['Close'], 7)

            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Stochastic
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

            # Williams %R
            df['willr'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))

            # ROC
            df['roc'] = df['Close'].pct_change(periods=10) * 100

            # Momentum
            df['momentum'] = df['Close'] - df['Close'].shift(10)

        # Derived features
        df['rsi_14_ma'] = df['rsi_14'].rolling(window=10).mean()
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_14_ma']
        df['macd_crossover'] = (df['macd'] > df['macd_signal']).astype(int)

        return df

    # ================================================================
    # 2. VOLATILITY INDICATORS
    # ================================================================

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators
        - ATR (Average True Range)
        - Bollinger Bands
        - Historical Volatility
        """
        if self.has_talib:
            # ATR
            df['atr_14'] = talib.ATR(df['High'], df['Low'], df['Close'],
                                     timeperiod=14)
            df['atr_20'] = talib.ATR(df['High'], df['Low'], df['Close'],
                                     timeperiod=20)

            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['Close'],
                                                timeperiod=20,
                                                nbdevup=2,
                                                nbdevdn=2)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower

        else:
            # ATR calculation
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(window=14).mean()
            df['atr_20'] = tr.rolling(window=20).mean()

            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(window=20).mean()
            std = df['Close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (2 * std)
            df['bb_lower'] = df['bb_middle'] - (2 * std)

        # Derived Bollinger features
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(100).quantile(0.2)).astype(int)

        # Historical volatility
        df['hist_vol_20'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['hist_vol_60'] = df['Close'].pct_change().rolling(window=60).std() * np.sqrt(252)

        # Volatility ratio
        df['vol_ratio'] = df['hist_vol_20'] / df['hist_vol_60']

        # Parkinson's volatility (uses High-Low)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) *
            np.log(df['High'] / df['Low'])**2
        ).rolling(window=20).mean()

        return df

    # ================================================================
    # 3. TREND INDICATORS
    # ================================================================

    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators
        - ADX (Average Directional Index)
        - Aroon
        - Linear Regression
        """
        if self.has_talib:
            # ADX
            df['adx'] = talib.ADX(df['High'], df['Low'], df['Close'],
                                  timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'],
                                          timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'],
                                            timeperiod=14)

            # Aroon
            aroon_up, aroon_down = talib.AROON(df['High'], df['Low'],
                                               timeperiod=25)
            df['aroon_up'] = aroon_up
            df['aroon_down'] = aroon_down

        else:
            # ADX approximation
            plus_dm = df['High'].diff()
            minus_dm = -df['Low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            tr = pd.concat([
                df['High'] - df['Low'],
                (df['High'] - df['Close'].shift()).abs(),
                (df['Low'] - df['Close'].shift()).abs()
            ], axis=1).max(axis=1)

            atr = tr.rolling(window=14).mean()
            df['plus_di'] = 100 * (plus_dm.rolling(window=14).mean() / atr)
            df['minus_di'] = 100 * (minus_dm.rolling(window=14).mean() / atr)

            dx = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
            df['adx'] = dx.rolling(window=14).mean()

            # Aroon approximation
            df['aroon_up'] = df['High'].rolling(25).apply(
                lambda x: (24 - x.argmax()) / 24 * 100 if len(x) == 25 else np.nan
            )
            df['aroon_down'] = df['Low'].rolling(25).apply(
                lambda x: (24 - x.argmin()) / 24 * 100 if len(x) == 25 else np.nan
            )

        # Derived features
        df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
        df['trend_strength'] = (df['adx'] > 25).astype(int)
        df['di_diff'] = df['plus_di'] - df['minus_di']

        # Linear regression trend
        for period in [20, 50]:
            df[f'linreg_slope_{period}'] = df['Close'].rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan
            )

        return df

    # ================================================================
    # 4. VOLUME INDICATORS
    # ================================================================

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators
        - OBV (On-Balance Volume)
        - MFI (Money Flow Index)
        - VWAP
        """
        if self.has_talib:
            # OBV
            df['obv'] = talib.OBV(df['Close'], df['Volume'])

            # MFI (Money Flow Index)
            df['mfi'] = talib.MFI(df['High'], df['Low'], df['Close'],
                                  df['Volume'], timeperiod=14)

            # AD (Accumulation/Distribution)
            df['ad'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])

        else:
            # OBV
            df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

            # MFI approximation
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']

            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()

            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            df['mfi'] = mfi

            # AD
            mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            mfv = mfm * df['Volume']
            df['ad'] = mfv.cumsum()

        # VWAP
        cumulative_tpv = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum()
        cumulative_vol = df['Volume'].cumsum()
        df['vwap'] = cumulative_tpv / cumulative_vol

        # Volume trends
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio_adv'] = df['Volume'] / df['volume_sma_20']
        df['volume_trend'] = (df['volume_ratio_adv'] > 1.2).astype(int)

        # OBV slope
        df['obv_slope'] = df['obv'].diff(5) / df['obv'].shift(5)

        # Price-Volume correlation
        df['pv_corr'] = df['Close'].rolling(window=20).corr(df['Volume'])

        return df

    # ================================================================
    # 5. PATTERN RECOGNITION
    # ================================================================

    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern recognition features
        """
        # Higher highs / Lower lows
        df['higher_high'] = ((df['High'] > df['High'].shift(1)) &
                            (df['High'].shift(1) > df['High'].shift(2))).astype(int)
        df['lower_low'] = ((df['Low'] < df['Low'].shift(1)) &
                          (df['Low'].shift(1) < df['Low'].shift(2))).astype(int)

        # Gap detection
        df['gap_up'] = (df['Low'] > df['High'].shift(1)).astype(int)
        df['gap_down'] = (df['High'] < df['Low'].shift(1)).astype(int)

        # Candle body size
        df['body_size'] = np.abs(df['Close'] - df['Open']) / df['Open']
        df['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
        df['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']

        # Doji detection (small body)
        df['is_doji'] = (df['body_size'] < 0.001).astype(int)

        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['Close'] > df['Open']) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Open'] < df['Close'].shift(1)) &
            (df['Close'] > df['Open'].shift(1))
        ).astype(int)

        df['bearish_engulfing'] = (
            (df['Close'] < df['Open']) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['Close'] < df['Open'].shift(1))
        ).astype(int)

        # Support/Resistance
        df['resistance_20'] = df['High'].rolling(window=20).max()
        df['support_20'] = df['Low'].rolling(window=20).min()
        df['distance_to_resistance'] = (df['resistance_20'] - df['Close']) / df['Close']
        df['distance_to_support'] = (df['Close'] - df['support_20']) / df['Close']

        return df

    # ================================================================
    # 6. MULTI-TIMEFRAME FEATURES
    # ================================================================

    def add_multitimeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features from multiple timeframes
        """
        # Weekly trends (5-day)
        df['weekly_return'] = df['Close'].pct_change(5)
        df['weekly_high'] = df['High'].rolling(window=5).max()
        df['weekly_low'] = df['Low'].rolling(window=5).min()
        df['weekly_range'] = (df['weekly_high'] - df['weekly_low']) / df['Close']

        # Monthly trends (20-day)
        df['monthly_return'] = df['Close'].pct_change(20)
        df['monthly_high'] = df['High'].rolling(window=20).max()
        df['monthly_low'] = df['Low'].rolling(window=20).min()

        # Position within ranges
        df['position_in_weekly_range'] = (
            (df['Close'] - df['weekly_low']) / (df['weekly_high'] - df['weekly_low'])
        )
        df['position_in_monthly_range'] = (
            (df['Close'] - df['monthly_low']) / (df['monthly_high'] - df['monthly_low'])
        )

        # Trend alignment
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_50_adv'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()

        df['trend_alignment'] = (
            (df['sma_10'] > df['sma_50_adv']) & (df['sma_50_adv'] > df['sma_200'])
        ).astype(int)

        return df

    # ================================================================
    # 7. MARKET MICROSTRUCTURE
    # ================================================================

    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add microstructure features
        """
        # Spread
        df['spread'] = (df['High'] - df['Low']) / df['Close']
        df['spread_ma'] = df['spread'].rolling(window=20).mean()
        df['spread_ratio'] = df['spread'] / df['spread_ma']

        # Price impact
        df['price_impact'] = df['Close'].pct_change() / (df['Volume'] / df['Volume'].rolling(20).mean() + 1e-10)

        # Liquidity
        df['liquidity'] = df['Volume'] / (df['spread'] + 1e-10)

        # Realized volatility intraday
        df['intraday_vol'] = np.log(df['High'] / df['Low'])

        # Overnight gap
        df['overnight_return'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

        # Intraday momentum
        df['intraday_momentum'] = (df['Close'] - df['Open']) / df['Open']

        return df

    # ================================================================
    # HELPER FUNCTIONS
    # ================================================================

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create all advanced features
    """
    engineer = AdvancedFeatureEngineer()
    return engineer.create_all_features(df)
