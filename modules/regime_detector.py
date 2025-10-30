"""
Market Regime Detection Module

Classifies market conditions into 4 regimes:
- BULL: Strong uptrend, low volatility, high sentiment
- BEAR: Downtrend, elevated volatility, low sentiment
- SIDEWAYS: Range-bound, moderate conditions
- CRISIS: Extreme volatility, market panic

Three detection methods:
1. Rule-based: Simple threshold logic (fast, interpretable)
2. HMM: Hidden Markov Model (best, probabilistic)
3. Ensemble: Combines both methods (most robust)

Usage:
    detector = RegimeDetector()

    # Rule-based
    regimes = detector.detect_rule_based(df)

    # HMM (requires hmmlearn)
    regimes = detector.detect_hmm(df)

    # Ensemble (recommended)
    regimes = detector.detect_ensemble(df)

    # Get position size for regime
    position_size = detector.get_position_size('BULL')  # Returns 0.50
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings

# Try to import hmmlearn for HMM detection
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("hmmlearn not available. HMM detection will not work. Install with: pip install hmmlearn")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Market regime detector with multiple methods.
    """

    # Regime constants
    REGIME_BULL = 'BULL'
    REGIME_BEAR = 'BEAR'
    REGIME_SIDEWAYS = 'SIDEWAYS'
    REGIME_CRISIS = 'CRISIS'

    # Position sizing by regime
    POSITION_SIZES = {
        REGIME_BULL: 0.50,      # 50% capital in bull market
        REGIME_SIDEWAYS: 0.25,  # 25% capital in sideways
        REGIME_BEAR: 0.00,      # 0% - no long positions in bear
        REGIME_CRISIS: 0.00     # 0% - liquidate in crisis
    }

    # Prediction thresholds by regime (more conservative in uncertain markets)
    PREDICTION_THRESHOLDS = {
        REGIME_BULL: 0.001,      # Lower threshold (easier to enter)
        REGIME_SIDEWAYS: 0.002,  # Moderate threshold
        REGIME_BEAR: 0.005,      # Higher threshold (harder to enter, but we use 0% size anyway)
        REGIME_CRISIS: 0.010     # Very high threshold (but we use 0% size)
    }

    def __init__(self):
        """Initialize regime detector."""
        self.regimes = None
        self.method = None

    def detect_rule_based(self, df):
        """
        Rule-based regime detection using simple thresholds.

        Logic:
        1. CRISIS: VIX > 40 OR volatility > 40% (market panic)
        2. BEAR: Price < SMA200 AND (VIX > 25 OR fear_greed < 30)
        3. BULL: Price > SMA200 AND VIX < 20 AND fear_greed > 50
        4. SIDEWAYS: Everything else

        Args:
            df: DataFrame with OHLCV + features (must have Close, sma_200, volatility_20)

        Returns:
            Series with regime labels (indexed by date)
        """
        logger.info("\nDetecting regimes using RULE-BASED method...")

        df = df.copy()

        # Calculate required features if not present
        if 'sma_200' not in df.columns:
            df['sma_200'] = df['Close'].rolling(window=200, min_periods=1).mean()

        if 'volatility_20' not in df.columns:
            df['volatility_20'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

        # Initialize regime column
        df['regime'] = self.REGIME_SIDEWAYS  # Default

        # Rule 1: CRISIS (highest priority)
        # VIX > 40 OR volatility > 40%
        crisis_mask = (df['volatility_20'] > 0.40)

        # If we have VIX data, use it
        if 'vix' in df.columns:
            crisis_mask = crisis_mask | (df['vix'] > 40)

        df.loc[crisis_mask, 'regime'] = self.REGIME_CRISIS

        # Rule 2: BEAR (second priority)
        # Price < SMA200 AND (VIX > 25 OR fear_greed < 30)
        bear_mask = (df['Close'] < df['sma_200'])

        if 'vix' in df.columns:
            bear_mask = bear_mask & (df['vix'] > 25)

        if 'fear_greed' in df.columns:
            bear_mask = bear_mask | ((df['Close'] < df['sma_200']) & (df['fear_greed'] < 30))

        # Only apply bear if not already crisis
        df.loc[bear_mask & (df['regime'] != self.REGIME_CRISIS), 'regime'] = self.REGIME_BEAR

        # Rule 3: BULL (third priority)
        # Price > SMA200 AND VIX < 20 AND fear_greed > 50
        bull_mask = (df['Close'] > df['sma_200'])

        if 'vix' in df.columns:
            bull_mask = bull_mask & (df['vix'] < 20)

        if 'fear_greed' in df.columns:
            bull_mask = bull_mask & (df['fear_greed'] > 50)
        elif 'vix' not in df.columns:
            # If no VIX or fear_greed, just use price vs SMA200
            bull_mask = (df['Close'] > df['sma_200']) & (df['volatility_20'] < 0.20)

        # Only apply bull if not crisis or bear
        df.loc[bull_mask & ~(crisis_mask | bear_mask), 'regime'] = self.REGIME_BULL

        # Rule 4: SIDEWAYS is default (already set above)

        regimes = df['regime']

        # Log regime distribution
        self._log_regime_stats(regimes, "RULE-BASED")

        self.regimes = regimes
        self.method = 'rule_based'

        return regimes

    def detect_hmm(self, df, n_states=4, n_iter=100):
        """
        Hidden Markov Model (HMM) regime detection.

        Uses Gaussian HMM to learn market states from features.
        Then maps learned states to regime labels based on characteristics.

        Args:
            df: DataFrame with features
            n_states: Number of hidden states (default 4 for our 4 regimes)
            n_iter: Number of EM iterations

        Returns:
            Series with regime labels
        """
        if not HMM_AVAILABLE:
            logger.error("hmmlearn not installed! Install with: pip install hmmlearn")
            logger.info("Falling back to rule-based detection...")
            return self.detect_rule_based(df)

        logger.info("\nDetecting regimes using HMM method...")

        df = df.copy()

        # Select features for HMM
        feature_cols = []

        # Price momentum
        if 'returns_20' in df.columns:
            feature_cols.append('returns_20')
        else:
            df['returns_20'] = df['Close'].pct_change(20)
            feature_cols.append('returns_20')

        # Volatility
        if 'volatility_20' in df.columns:
            feature_cols.append('volatility_20')
        else:
            df['volatility_20'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            feature_cols.append('volatility_20')

        # Trend (price vs SMA)
        if 'sma_200' in df.columns:
            df['price_vs_sma200'] = (df['Close'] - df['sma_200']) / df['sma_200']
            feature_cols.append('price_vs_sma200')

        # Sentiment if available
        if 'fear_greed' in df.columns:
            df['fear_greed_norm'] = (df['fear_greed'] - 50) / 50  # Normalize to [-1, 1]
            feature_cols.append('fear_greed_norm')

        # Prepare feature matrix
        X = df[feature_cols].fillna(method='ffill').fillna(method='bfill').values

        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info(f"Training HMM with {n_states} states on {len(feature_cols)} features...")
        logger.info(f"Features: {feature_cols}")

        # Train Gaussian HMM
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42
        )

        model.fit(X_scaled)

        # Predict hidden states
        hidden_states = model.predict(X_scaled)

        # Map hidden states to regime labels
        # Analyze each state's characteristics
        regimes_mapped = self._map_hmm_states_to_regimes(df, hidden_states, feature_cols)

        # Log regime distribution
        self._log_regime_stats(regimes_mapped, "HMM")

        self.regimes = regimes_mapped
        self.method = 'hmm'

        return regimes_mapped

    def _map_hmm_states_to_regimes(self, df, hidden_states, feature_cols):
        """
        Map HMM hidden states to regime labels.

        Analyzes characteristics of each state and assigns regime label.
        """
        df = df.copy()
        df['hidden_state'] = hidden_states

        # Calculate mean characteristics for each state
        state_chars = {}

        for state in range(hidden_states.max() + 1):
            mask = df['hidden_state'] == state
            state_df = df[mask]

            chars = {
                'mean_return': state_df['Close'].pct_change(20).mean() if 'returns_20' not in df.columns else state_df['returns_20'].mean(),
                'mean_volatility': state_df['volatility_20'].mean() if 'volatility_20' in df.columns else 0,
                'count': mask.sum()
            }

            if 'fear_greed' in df.columns:
                chars['mean_sentiment'] = state_df['fear_greed'].mean()

            state_chars[state] = chars

        logger.info("\nHMM State Characteristics:")
        for state, chars in state_chars.items():
            logger.info(f"  State {state}: return={chars['mean_return']:.4f}, vol={chars['mean_volatility']:.2%}, count={chars['count']}")

        # Map states to regimes based on characteristics
        state_to_regime = {}

        for state, chars in state_chars.items():
            ret = chars['mean_return']
            vol = chars['mean_volatility']

            # Crisis: High volatility
            if vol > 0.30:
                state_to_regime[state] = self.REGIME_CRISIS
            # Bear: Negative returns
            elif ret < -0.01:
                state_to_regime[state] = self.REGIME_BEAR
            # Bull: Positive returns + low volatility
            elif ret > 0.01 and vol < 0.20:
                state_to_regime[state] = self.REGIME_BULL
            # Sideways: Everything else
            else:
                state_to_regime[state] = self.REGIME_SIDEWAYS

        logger.info("\nState to Regime Mapping:")
        for state, regime in state_to_regime.items():
            logger.info(f"  State {state} -> {regime}")

        # Map states to regimes
        regimes = df['hidden_state'].map(state_to_regime)

        return regimes

    def detect_ensemble(self, df):
        """
        Ensemble method: Combines rule-based and HMM.

        Uses majority voting between the two methods.
        If HMM not available, falls back to rule-based.

        Args:
            df: DataFrame with features

        Returns:
            Series with regime labels
        """
        logger.info("\nDetecting regimes using ENSEMBLE method...")

        # Get predictions from both methods
        regimes_rule = self.detect_rule_based(df)

        if HMM_AVAILABLE:
            regimes_hmm = self.detect_hmm(df)

            # Combine using simple voting
            # If both agree, use that. If they disagree, prefer HMM (more sophisticated)
            regimes_ensemble = regimes_hmm.copy()

            # Log agreement
            agreement = (regimes_rule == regimes_hmm).sum() / len(regimes_rule)
            logger.info(f"\nRule-based and HMM agreement: {agreement:.1%}")

            self.regimes = regimes_ensemble
            self.method = 'ensemble'

            return regimes_ensemble
        else:
            logger.warning("HMM not available, using only rule-based method")
            self.method = 'ensemble'
            return regimes_rule

    def get_position_size(self, regime):
        """
        Get position size for a given regime.

        Args:
            regime: Regime label (BULL/BEAR/SIDEWAYS/CRISIS)

        Returns:
            Position size as fraction of capital (0.0 to 1.0)
        """
        return self.POSITION_SIZES.get(regime, 0.25)

    def get_prediction_threshold(self, regime):
        """
        Get prediction threshold for a given regime.

        More conservative (higher threshold) in uncertain markets.

        Args:
            regime: Regime label

        Returns:
            Prediction threshold
        """
        return self.PREDICTION_THRESHOLDS.get(regime, 0.002)

    def _log_regime_stats(self, regimes, method_name):
        """Log regime distribution statistics."""
        logger.info(f"\n{method_name} Regime Distribution:")

        regime_counts = regimes.value_counts()
        total = len(regimes)

        for regime in [self.REGIME_BULL, self.REGIME_SIDEWAYS, self.REGIME_BEAR, self.REGIME_CRISIS]:
            count = regime_counts.get(regime, 0)
            pct = count / total * 100
            pos_size = self.POSITION_SIZES[regime]
            logger.info(f"  {regime:10s}: {count:4d} days ({pct:5.1f}%) - Position size: {pos_size:.0%}")

    def plot_regimes(self, df, regimes, save_path=None):
        """
        Plot price with regime backgrounds.

        Args:
            df: DataFrame with Close prices
            regimes: Series with regime labels
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots(figsize=(15, 6))

        # Plot close price
        ax.plot(df.index, df['Close'], linewidth=1, color='black', label='SPY Close')

        # Color map for regimes
        regime_colors = {
            self.REGIME_BULL: 'green',
            self.REGIME_SIDEWAYS: 'gray',
            self.REGIME_BEAR: 'red',
            self.REGIME_CRISIS: 'darkred'
        }

        # Add regime backgrounds
        for regime in regime_colors.keys():
            regime_mask = (regimes == regime)
            regime_dates = regimes[regime_mask].index

            for date in regime_dates:
                ax.axvspan(date, date + pd.Timedelta(days=1),
                          alpha=0.2, color=regime_colors[regime])

        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, alpha=0.3, label=regime)
                          for regime, color in regime_colors.items()]
        ax.legend(handles=legend_elements, loc='upper left')

        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'Market Regimes - {self.method.upper()}')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved regime plot to {save_path}")

        plt.close()

    def detect_extreme_conditions(self, df):
        """
        Detect EXTREME market conditions only (not normal bear markets).

        This is more selective than regime detection - only flags
        truly dangerous conditions that require intervention.

        Returns df with 'extreme_condition' column:
        - 'CRISIS': Emergency (VIX > 40 OR return < -20%)
        - 'EXTREME_BEAR': Very dangerous (VIX > 35 AND return < -15%)
        - 'NORMAL': Trade normally

        Philosophy: Trust the ML model except in extreme disasters
        """
        logger.info("\nDetecting EXTREME market conditions...")

        df = df.copy()

        # Ensure we have needed features (calculate if missing)
        if 'return_20d' not in df.columns and 'Close' in df.columns:
            df['return_20d'] = df['Close'].pct_change(20)

        if 'return_5d' not in df.columns and 'Close' in df.columns:
            df['return_5d'] = df['Close'].pct_change(5)

        if 'volatility_20d' not in df.columns:
            if 'return_1d' in df.columns:
                df['volatility_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252) * 100
            elif 'Close' in df.columns:
                df['volatility_20d'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100

        # Get features (with safe fallbacks)
        vix = df['vix'] if 'vix' in df.columns else df.get('volatility_20d', pd.Series(15, index=df.index))
        return_20d = df.get('return_20d', pd.Series(0, index=df.index))
        return_5d = df.get('return_5d', pd.Series(0, index=df.index))
        volatility_20d = df.get('volatility_20d', pd.Series(15, index=df.index))

        # Initialize all as NORMAL
        condition = pd.Series('NORMAL', index=df.index)

        # CRISIS conditions (most extreme - emergency exit)
        # These are "market is on fire" scenarios
        crisis_mask = (
            (vix > 40) |               # VIX spike above 40 (extreme panic)
            (return_20d < -0.20) |     # Down > 20% in 20 days
            (return_5d < -0.10) |      # Down > 10% in 5 days (sudden crash)
            (volatility_20d > 40)      # Volatility > 40%
        )
        condition[crisis_mask] = 'CRISIS'

        # EXTREME BEAR conditions (very dangerous but not emergency)
        # These are "very risky but not catastrophic"
        extreme_bear_mask = (
            (vix > 35) &               # VIX elevated
            (return_20d < -0.15) &     # Down > 15% in 20 days
            ~crisis_mask               # Not already crisis
        )
        condition[extreme_bear_mask] = 'EXTREME_BEAR'

        # Everything else = NORMAL (trust the model!)

        df['extreme_condition'] = condition

        # Log distribution
        self._log_extreme_stats(condition)

        return df

    def get_position_size_hybrid(self, extreme_condition, prediction=None, base_size=0.5):
        """
        HYBRID position sizing - intervene only in extreme conditions.

        Args:
            extreme_condition: 'CRISIS', 'EXTREME_BEAR', or 'NORMAL'
            prediction: Optional model prediction value
            base_size: Base position size (default 0.5 = 50%)

        Returns:
            Position size multiplier (0 to 1.0)
        """
        if extreme_condition == 'CRISIS':
            # EMERGENCY: Liquidate everything
            return 0.0

        elif extreme_condition == 'EXTREME_BEAR':
            # DANGEROUS: Reduce significantly but don't stop
            if prediction is not None and prediction > 0.003:
                # Model very confident (>0.3% prediction) → small position
                return base_size * 0.25  # 12.5% of capital
            else:
                # Not very confident → minimal exposure
                return base_size * 0.15  # 7.5% of capital

        else:  # NORMAL
            # TRUST MODEL: Normal trading
            return base_size  # 50% of capital

    def get_prediction_threshold_hybrid(self, extreme_condition, base_threshold=0.001):
        """
        HYBRID prediction threshold - higher bar only in extreme conditions.

        Args:
            extreme_condition: Current extreme condition
            base_threshold: Normal threshold (default 0.001 = 0.1%)

        Returns:
            Threshold value
        """
        if extreme_condition == 'CRISIS':
            # Never trade in crisis
            return 999  # Impossible to meet

        elif extreme_condition == 'EXTREME_BEAR':
            # Need stronger signal (2.5× normal)
            return base_threshold * 2.5

        else:  # NORMAL
            # Normal threshold - trust the model
            return base_threshold

    def _log_extreme_stats(self, conditions):
        """Log extreme condition distribution."""
        logger.info("\nExtreme Condition Distribution:")

        condition_counts = conditions.value_counts()
        total = len(conditions)

        for condition in ['NORMAL', 'EXTREME_BEAR', 'CRISIS']:
            count = condition_counts.get(condition, 0)
            pct = count / total * 100

            if condition == 'CRISIS':
                emoji = "🚨"
                action = "EXIT all (0%)"
            elif condition == 'EXTREME_BEAR':
                emoji = "⚠️"
                action = "REDUCE (7.5-12.5%)"
            else:
                emoji = "✅"
                action = "TRADE normally (50%)"

            logger.info(f"  {emoji} {condition:15s}: {count:4d} days ({pct:5.1f}%) → {action}")

        normal_pct = condition_counts.get('NORMAL', 0) / total * 100
        logger.info(f"\nTrading normally: {normal_pct:.1f}% of the time")
        logger.info(f"Protected periods: {100-normal_pct:.1f}% of the time")


def main():
    """
    Test regime detection on historical data.
    """
    print("\n" + "=" * 70)
    print("MARKET REGIME DETECTION - STANDALONE TEST")
    print("=" * 70 + "\n")

    # Load data
    logger.info("Loading data...")
    data_path = Path('data/processed/SPY_processed.csv')

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Run pipeline.py first to generate processed data!")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} days of data from {df.index.min().date()} to {df.index.max().date()}")

    # Initialize detector
    detector = RegimeDetector()

    # Test all three methods
    print("\n" + "=" * 70)
    print("[1/3] RULE-BASED DETECTION")
    print("=" * 70)
    regimes_rule = detector.detect_rule_based(df)

    if HMM_AVAILABLE:
        print("\n" + "=" * 70)
        print("[2/3] HMM DETECTION")
        print("=" * 70)
        regimes_hmm = detector.detect_hmm(df)
    else:
        logger.warning("\nSkipping HMM detection (hmmlearn not installed)")

    print("\n" + "=" * 70)
    print("[3/3] ENSEMBLE DETECTION")
    print("=" * 70)
    regimes_ensemble = detector.detect_ensemble(df)

    # Plot regimes
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    output_dir = Path('results/regime_detection')
    output_dir.mkdir(exist_ok=True, parents=True)

    detector.method = 'rule_based'
    detector.plot_regimes(df, regimes_rule, save_path=output_dir / 'regimes_rule_based.png')

    if HMM_AVAILABLE:
        detector.method = 'hmm'
        detector.plot_regimes(df, regimes_hmm, save_path=output_dir / 'regimes_hmm.png')

    detector.method = 'ensemble'
    detector.plot_regimes(df, regimes_ensemble, save_path=output_dir / 'regimes_ensemble.png')

    print("\n" + "=" * 70)
    print("✓ REGIME DETECTION TEST COMPLETE!")
    print("=" * 70)
    print(f"\nVisualizations saved to: {output_dir}/")
    print("\nNext step: Run backtest with regime detection")
    print("Command: python backtest_with_regime.py --method ensemble --compare")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
