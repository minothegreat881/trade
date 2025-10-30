"""
Trading Engine
Generates signals using trained model + regime detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import logging
from typing import Dict, Optional

from feature_engineering import FeatureEngineer
from modules.regime_detector import RegimeDetector
from sentiment_features import SentimentFeatureEngineer
import config

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Generate trading signals from real-time data
    """

    def __init__(self, model_path: str = 'models/xgboost_sentiment_model.pkl'):
        """
        Args:
            model_path: Path to trained XGBoost model
        """
        # Load trained model
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Load expected features from metadata
        import json
        metadata_path = model_path.replace('.pkl', '_metadata.json').replace('xgboost_sentiment_model', 'model')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.expected_features = metadata['features']['list']
                logger.info(f"Loaded {len(self.expected_features)} expected features from metadata")
        except:
            # Fallback to default feature list
            self.expected_features = [
                'return_1d', 'return_5d', 'return_10d', 'return_20d',
                'volatility_20d', 'volatility_60d', 'volume_ratio',
                'price_position', 'sma_20', 'sma_50', 'trend',
                'fear_greed_ma5', 'fear_greed_ma10', 'fear_greed_extreme_fear',
                'fear_greed_extreme_greed', 'fear_greed_change_5d', 'composite_sentiment'
            ]
            logger.warning("Could not load metadata, using default feature list")

        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.sentiment_engineer = SentimentFeatureEngineer()
        self.regime_detector = RegimeDetector()

        # Trading parameters
        self.base_threshold = 0.001  # 0.1% minimum prediction
        self.base_position_size = 0.5  # 50% of capital

        logger.info("Trading engine initialized")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw data

        Args:
            df: DataFrame with OHLCV + sentiment

        Returns:
            DataFrame with all features
        """
        # Technical features
        df = self.feature_engineer.create_basic_features(df)

        # Sentiment features (if not already present)
        if 'fear_greed_value' in df.columns and 'fear_greed_ma10' not in df.columns:
            sentiment_features = self.sentiment_engineer.create_all_sentiment_features(df)
            # Merge sentiment features with existing df
            for col in sentiment_features.columns:
                df[col] = sentiment_features[col]

        return df

    def generate_signal(self, historical_data: pd.DataFrame,
                       current_snapshot: Dict) -> Dict:
        """
        Generate trading signal for current moment

        Args:
            historical_data: Recent historical data (100+ days)
            current_snapshot: Current price + sentiment snapshot

        Returns:
            Signal dictionary with action, prediction, regime, etc.
        """
        try:
            # Append current snapshot to historical data
            current_row = pd.DataFrame([{
                'Open': current_snapshot.get('open', current_snapshot['price']),
                'High': current_snapshot.get('high', current_snapshot['price']),
                'Low': current_snapshot.get('low', current_snapshot['price']),
                'Close': current_snapshot['price'],
                'Volume': current_snapshot.get('volume', 0),
                'fear_greed_value': current_snapshot.get('fear_greed_value', 50),
                'VIX': current_snapshot.get('VIX', 15)
            }], index=[pd.Timestamp(current_snapshot['timestamp'])])

            df = pd.concat([historical_data, current_row])

            # Engineer features
            df = self.prepare_features(df)

            # Detect regime (hybrid - extreme conditions)
            df = self.regime_detector.detect_extreme_conditions(df)

            # Get today's features
            latest = df.iloc[-1]

            # Use only the features the model was trained on
            feature_cols = [col for col in self.expected_features if col in df.columns]
            missing_features = [col for col in self.expected_features if col not in df.columns]

            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                return {
                    'timestamp': current_snapshot['timestamp'],
                    'action': 'HOLD',
                    'reason': f'Missing features: {len(missing_features)}',
                    'prediction': 0,
                    'extreme_condition': 'UNKNOWN',
                    'position_size': 0,
                    'current_price': current_snapshot['price']
                }

            if latest[feature_cols].isna().any():
                logger.warning("Features contain NaN values, insufficient data")
                return {
                    'timestamp': current_snapshot['timestamp'],
                    'action': 'HOLD',
                    'reason': 'Insufficient historical data for features',
                    'prediction': 0,
                    'extreme_condition': 'UNKNOWN',
                    'position_size': 0,
                    'current_price': current_snapshot['price']
                }

            # Make prediction
            X = latest[feature_cols].values.reshape(1, -1)
            prediction = self.model.predict(X)[0]

            # Get regime
            extreme_condition = latest['extreme_condition']

            # Get regime-adjusted parameters
            threshold = self.regime_detector.get_prediction_threshold_hybrid(
                extreme_condition,
                self.base_threshold
            )

            position_size = self.regime_detector.get_position_size_hybrid(
                extreme_condition,
                prediction,
                self.base_position_size
            )

            # Generate action
            if extreme_condition == 'CRISIS':
                action = 'CLOSE_ALL'
                reason = 'CRISIS detected - emergency exit'
            elif position_size == 0:
                action = 'CLOSE'
                reason = f'{extreme_condition} - no trading'
            elif prediction > threshold:
                action = 'BUY'
                reason = f'Signal: {prediction:.4f} > {threshold:.4f}'
            else:
                action = 'HOLD'
                reason = f'Signal weak: {prediction:.4f} < {threshold:.4f}'

            signal = {
                'timestamp': current_snapshot['timestamp'],
                'action': action,
                'reason': reason,
                'prediction': float(prediction),
                'extreme_condition': extreme_condition,
                'threshold': float(threshold),
                'position_size': float(position_size),
                'current_price': float(current_snapshot['price']),
                'vix': float(current_snapshot.get('VIX', 15)),
                'fear_greed': int(current_snapshot.get('fear_greed_value', 50))
            }

            logger.info(f"Signal generated: {action} (prediction: {prediction:.4f}, "
                       f"regime: {extreme_condition})")

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return {
                'timestamp': current_snapshot['timestamp'],
                'action': 'HOLD',
                'reason': f'Error: {str(e)}',
                'prediction': 0,
                'extreme_condition': 'ERROR',
                'position_size': 0,
                'current_price': current_snapshot['price']
            }

    def evaluate_signal(self, signal: Dict, current_position: int,
                       portfolio_value: float, cash: float) -> Dict:
        """
        Evaluate signal and determine exact trade to execute

        Args:
            signal: Signal from generate_signal()
            current_position: Current shares held
            portfolio_value: Total portfolio value
            cash: Available cash

        Returns:
            Trade instruction with symbol, action, quantity
        """
        action = signal['action']
        position_size = signal['position_size']
        current_price = signal['current_price']

        # Calculate target position
        target_value = portfolio_value * position_size
        target_shares = int(target_value / current_price)

        # Calculate trade
        shares_to_trade = target_shares - current_position

        if action == 'CLOSE_ALL' or action == 'CLOSE':
            # Close entire position
            if current_position > 0:
                return {
                    'symbol': 'SPY',
                    'action': 'SELL',
                    'quantity': current_position,
                    'price': current_price,
                    'reason': signal['reason']
                }
            else:
                return None

        elif action == 'BUY':
            # Buy to reach target
            if shares_to_trade > 0:
                # Check if enough cash
                cost = shares_to_trade * current_price * 1.001  # Include commission
                if cost <= cash:
                    return {
                        'symbol': 'SPY',
                        'action': 'BUY',
                        'quantity': shares_to_trade,
                        'price': current_price,
                        'reason': signal['reason']
                    }
                else:
                    # Buy what we can afford
                    affordable_shares = int(cash / (current_price * 1.001))
                    if affordable_shares > 0:
                        return {
                            'symbol': 'SPY',
                            'action': 'BUY',
                            'quantity': affordable_shares,
                            'price': current_price,
                            'reason': 'Partial buy (limited cash)'
                        }
            elif shares_to_trade < 0:
                # Need to sell to reach target
                return {
                    'symbol': 'SPY',
                    'action': 'SELL',
                    'quantity': abs(shares_to_trade),
                    'price': current_price,
                    'reason': 'Reduce position'
                }

        # No trade needed
        return None


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Trading Engine...")
    print("="*60)

    # This requires trained model to exist
    try:
        engine = TradingEngine()
        print("[OK] Trading engine initialized")

        # Would need real data to test signal generation
        print("\n[WARN] Full test requires historical data and trained model")

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        print("\nNote: Ensure xgboost_sentiment_model.pkl exists in models/")
