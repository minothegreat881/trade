"""
Sentiment Feature Engineering Module

Creates ~15 sentiment features from raw sentiment data:
- Fear & Greed Index features (smoothed, regime, extremes)
- VIX features (regime, spikes, z-score)
- Bitcoin features (momentum, volatility)
- Composite sentiment score

Designed to complement existing technical features.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class SentimentFeatureEngineer:
    """
    Create sentiment-based features for trading models.

    Features capture market psychology and risk sentiment:
    - Fear & Greed Index: Direct measure of market sentiment
    - VIX: Market volatility expectations (fear gauge)
    - Bitcoin: Risk-on/risk-off indicator

    All features designed to be predictive of forward returns.
    """

    def __init__(self):
        """
        Initialize sentiment feature engineer.

        Example:
            >>> engineer = SentimentFeatureEngineer()
            >>> features = engineer.create_all_sentiment_features(sentiment_data)
        """
        logger.info("Initialized SentimentFeatureEngineer")

    def create_fear_greed_features(self, data):
        """
        Create Fear & Greed Index features.

        Features:
        1. fear_greed_ma5: 5-day smoothed moving average
        2. fear_greed_ma10: 10-day smoothed moving average
        3. fear_greed_extreme_fear: Binary flag for extreme fear (<25)
        4. fear_greed_extreme_greed: Binary flag for extreme greed (>75)
        5. fear_greed_change_5d: 5-day change in sentiment

        Args:
            data: DataFrame with 'fear_greed_value' column

        Returns:
            DataFrame with Fear & Greed features
        """
        if 'fear_greed_value' not in data.columns:
            logger.warning("fear_greed_value not found in data, skipping F&G features")
            return pd.DataFrame(index=data.index)

        logger.info("Creating Fear & Greed features...")

        features = pd.DataFrame(index=data.index)

        fg = data['fear_greed_value']

        # 1. Smoothed moving averages
        features['fear_greed_ma5'] = fg.rolling(5, min_periods=1).mean()
        features['fear_greed_ma10'] = fg.rolling(10, min_periods=1).mean()

        # 2. Extreme sentiment flags
        features['fear_greed_extreme_fear'] = (fg <= 25).astype(int)
        features['fear_greed_extreme_greed'] = (fg >= 75).astype(int)

        # 3. Sentiment change (momentum)
        features['fear_greed_change_5d'] = fg.diff(5)

        logger.info(f"  Created {len(features.columns)} Fear & Greed features")

        return features

    def create_vix_features(self, data):
        """
        Create VIX (Volatility Index) features.

        Features:
        1. vix_ma5: 5-day moving average
        2. vix_ma20: 20-day moving average
        3. vix_regime: VIX regime (low/medium/high)
        4. vix_spike: Binary flag for VIX spikes (>1.5 std above MA)
        5. vix_zscore: Z-score of VIX (20-day rolling)

        VIX interpretation:
        - VIX < 15: Low volatility (risk-on)
        - VIX 15-25: Normal volatility
        - VIX > 25: High volatility (risk-off)

        Args:
            data: DataFrame with 'VIX' column

        Returns:
            DataFrame with VIX features
        """
        if 'VIX' not in data.columns:
            logger.warning("VIX not found in data, skipping VIX features")
            return pd.DataFrame(index=data.index)

        logger.info("Creating VIX features...")

        features = pd.DataFrame(index=data.index)

        vix = data['VIX']

        # 1. Smoothed moving averages
        features['vix_ma5'] = vix.rolling(5, min_periods=1).mean()
        features['vix_ma20'] = vix.rolling(20, min_periods=1).mean()

        # 2. VIX regime classification
        def classify_vix_regime(value):
            if pd.isna(value):
                return 1  # Default to medium
            if value < 15:
                return 0  # Low volatility (risk-on)
            elif value > 25:
                return 2  # High volatility (risk-off)
            else:
                return 1  # Medium volatility

        features['vix_regime'] = vix.apply(classify_vix_regime)

        # 3. VIX spikes (sudden fear)
        vix_rolling_mean = vix.rolling(20, min_periods=10).mean()
        vix_rolling_std = vix.rolling(20, min_periods=10).std()
        features['vix_spike'] = (vix > (vix_rolling_mean + 1.5 * vix_rolling_std)).astype(int)

        # 4. VIX z-score (standardized)
        features['vix_zscore'] = (vix - vix_rolling_mean) / vix_rolling_std
        features['vix_zscore'] = features['vix_zscore'].fillna(0)

        logger.info(f"  Created {len(features.columns)} VIX features")

        return features

    def create_bitcoin_features(self, data):
        """
        Create Bitcoin features (risk sentiment indicator).

        Features:
        1. btc_return_5d: 5-day return (already in data, keep)
        2. btc_return_10d: 10-day return (already in data, keep)
        3. btc_volatility_10d: 10-day rolling volatility
        4. btc_momentum: Simple momentum indicator (price > MA20)

        Bitcoin as leading indicator:
        - BTC often leads equity markets in risk-on/risk-off transitions
        - Strong BTC momentum → risk appetite
        - BTC volatility spikes → general market stress

        Args:
            data: DataFrame with BTC columns

        Returns:
            DataFrame with Bitcoin features
        """
        if 'BTC_Close' not in data.columns:
            logger.warning("BTC_Close not found in data, skipping Bitcoin features")
            return pd.DataFrame(index=data.index)

        logger.info("Creating Bitcoin features...")

        features = pd.DataFrame(index=data.index)

        btc_close = data['BTC_Close']

        # 1-2. Returns (keep from collector if exist)
        if 'BTC_return_5d' in data.columns:
            features['btc_return_5d'] = data['BTC_return_5d']

        if 'BTC_return_10d' in data.columns:
            features['btc_return_10d'] = data['BTC_return_10d']

        # 3. Rolling volatility
        btc_returns = btc_close.pct_change()
        features['btc_volatility_10d'] = btc_returns.rolling(10, min_periods=5).std()

        # 4. Momentum indicator (price above 20-day MA)
        btc_ma20 = btc_close.rolling(20, min_periods=10).mean()
        features['btc_momentum'] = (btc_close > btc_ma20).astype(int)

        logger.info(f"  Created {len(features.columns)} Bitcoin features")

        return features

    def create_composite_sentiment(self, data):
        """
        Create composite sentiment score combining all sources.

        Composite score components:
        1. Fear & Greed Index (0-100, higher = more greed)
        2. VIX (inverse: lower VIX = more confidence)
        3. Bitcoin momentum (positive = risk-on)

        Composite score interpretation:
        - High score (>0.5): Risk-on environment (bullish)
        - Low score (<-0.5): Risk-off environment (bearish)

        Args:
            data: DataFrame with sentiment features

        Returns:
            DataFrame with composite sentiment score
        """
        logger.info("Creating composite sentiment score...")

        features = pd.DataFrame(index=data.index)

        # Component 1: Fear & Greed (normalize to -1 to +1)
        if 'fear_greed_ma5' in data.columns:
            fg_normalized = (data['fear_greed_ma5'] - 50) / 50  # 0-100 → -1 to +1
        else:
            fg_normalized = pd.Series(0, index=data.index)

        # Component 2: VIX (inverse, normalize)
        if 'vix_ma5' in data.columns:
            # VIX 10-40 → normalized to -1 (high VIX) to +1 (low VIX)
            vix_normalized = (25 - data['vix_ma5'].clip(10, 40)) / 15
        else:
            vix_normalized = pd.Series(0, index=data.index)

        # Component 3: Bitcoin momentum (already binary 0/1 → rescale to -1/+1)
        if 'btc_momentum' in data.columns:
            btc_normalized = (data['btc_momentum'] * 2) - 1  # 0/1 → -1/+1
        else:
            btc_normalized = pd.Series(0, index=data.index)

        # Weighted composite (weights can be tuned)
        # Current weights: F&G=40%, VIX=40%, BTC=20%
        features['composite_sentiment'] = (
            0.4 * fg_normalized +
            0.4 * vix_normalized +
            0.2 * btc_normalized
        )

        # Clip to [-1, 1] range
        features['composite_sentiment'] = features['composite_sentiment'].clip(-1, 1)

        logger.info(f"  Created composite sentiment score")
        logger.info(f"  Mean: {features['composite_sentiment'].mean():.3f}, "
                   f"Std: {features['composite_sentiment'].std():.3f}")

        return features

    def create_all_sentiment_features(self, sentiment_data):
        """
        Create all sentiment features from raw sentiment data.

        Pipeline:
        1. Fear & Greed features (5 features)
        2. VIX features (5 features)
        3. Bitcoin features (4 features)
        4. Composite sentiment (1 feature)

        Total: ~15 sentiment features

        Args:
            sentiment_data: DataFrame with raw sentiment data
                Required columns: fear_greed_value, VIX, BTC_Close

        Returns:
            DataFrame with all sentiment features

        Example:
            >>> from sentiment_collector import SentimentCollector
            >>> collector = SentimentCollector()
            >>> sentiment_data = collector.collect_all_sentiment('2020-01-01', '2024-12-31')
            >>>
            >>> engineer = SentimentFeatureEngineer()
            >>> features = engineer.create_all_sentiment_features(sentiment_data)
            >>> print(f"Created {len(features.columns)} features")
        """
        print("\n" + "=" * 70)
        print("CREATING SENTIMENT FEATURES")
        print("=" * 70)

        # Create feature groups
        fg_features = self.create_fear_greed_features(sentiment_data)
        vix_features = self.create_vix_features(sentiment_data)
        btc_features = self.create_bitcoin_features(sentiment_data)

        # Merge all features
        all_features = pd.concat([
            fg_features,
            vix_features,
            btc_features
        ], axis=1)

        # Create composite sentiment (uses features created above)
        composite = self.create_composite_sentiment(all_features)
        all_features = pd.concat([all_features, composite], axis=1)

        # Handle missing values
        logger.info("\nHandling missing values...")
        missing_before = all_features.isnull().sum().sum()

        # Forward fill (use last known sentiment)
        all_features = all_features.ffill()

        # Backward fill (for start of series)
        all_features = all_features.bfill()

        # Fill any remaining with 0
        all_features = all_features.fillna(0)

        missing_after = all_features.isnull().sum().sum()

        logger.info(f"  Missing values: {missing_before} → {missing_after}")

        print("\n" + "=" * 70)
        print("SENTIMENT FEATURES COMPLETE")
        print("=" * 70)
        print(f"Total features: {len(all_features.columns)}")
        print(f"Total rows: {len(all_features)}")
        print(f"Date range: {all_features.index.min().date()} to {all_features.index.max().date()}")
        print("\nFeature list:")
        for i, col in enumerate(all_features.columns, 1):
            print(f"  {i:2d}. {col}")
        print("=" * 70)

        return all_features


if __name__ == "__main__":
    # Test the sentiment feature engineer
    print("\nTesting SentimentFeatureEngineer...\n")

    from sentiment_collector import SentimentCollector

    # Collect sentiment data
    collector = SentimentCollector()
    sentiment_data = collector.collect_all_sentiment('2020-01-01', '2024-12-31')

    # Create features
    engineer = SentimentFeatureEngineer()
    features = engineer.create_all_sentiment_features(sentiment_data)

    # Show results
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"\nFeature shape: {features.shape}")
    print(f"\nFirst 5 rows:")
    print(features.head())
    print(f"\nLast 5 rows:")
    print(features.tail())
    print(f"\nFeature statistics:")
    print(features.describe())
    print(f"\nMissing values:")
    print(features.isnull().sum())

    print("\n✓ SentimentFeatureEngineer test complete!")
