"""
Sentiment Data Integration Script

Combines technical features with sentiment features:
1. Load existing technical features from train/test splits
2. Collect sentiment data (Fear & Greed, VIX, Bitcoin)
3. Create sentiment features (~15 features)
4. Merge technical + sentiment features
5. Save combined datasets

Expected output:
- data/train_test_sentiment/train_X.csv (11 technical + 15 sentiment = 26 features)
- data/train_test_sentiment/test_X.csv
- data/train_test_sentiment/train_y.csv (unchanged)
- data/train_test_sentiment/test_y.csv (unchanged)

Usage:
    python integrate_sentiment.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import codecs
import logging

# Fix Windows encoding
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from sentiment_collector import SentimentCollector
from sentiment_features import SentimentFeatureEngineer
import config

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_existing_data():
    """
    Load existing train/test splits with technical features.

    Returns:
        tuple: (train_X, train_y, test_X, test_y, train_data, test_data)
    """
    logger.info("Loading existing train/test data...")

    data_dir = Path('data/train_test')

    train_X = pd.read_csv(data_dir / 'train_X.csv', index_col=0)
    train_X.index = pd.to_datetime(train_X.index, utc=True).tz_localize(None)

    train_y = pd.read_csv(data_dir / 'train_y.csv', index_col=0)['target']
    train_y.index = pd.to_datetime(train_y.index, utc=True).tz_localize(None)

    test_X = pd.read_csv(data_dir / 'test_X.csv', index_col=0)
    test_X.index = pd.to_datetime(test_X.index, utc=True).tz_localize(None)

    test_y = pd.read_csv(data_dir / 'test_y.csv', index_col=0)['target']
    test_y.index = pd.to_datetime(test_y.index, utc=True).tz_localize(None)

    # Load full data (for date ranges)
    train_data = pd.read_csv(data_dir / 'train_data.csv', index_col=0, parse_dates=True)
    test_data = pd.read_csv(data_dir / 'test_data.csv', index_col=0, parse_dates=True)

    logger.info(f"  Train: {len(train_X)} samples, {len(train_X.columns)} features")
    logger.info(f"  Test: {len(test_X)} samples, {len(test_X.columns)} features")
    logger.info(f"  Train date range: {train_X.index.min().date()} to {train_X.index.max().date()}")
    logger.info(f"  Test date range: {test_X.index.min().date()} to {test_X.index.max().date()}")

    return train_X, train_y, test_X, test_y, train_data, test_data


def collect_and_create_sentiment_features(start_date, end_date):
    """
    Collect sentiment data and create features.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with sentiment features
    """
    logger.info(f"\nCollecting sentiment data from {start_date} to {end_date}...")

    # Collect sentiment data
    collector = SentimentCollector()
    sentiment_data = collector.collect_all_sentiment(start_date, end_date)

    # Create features
    engineer = SentimentFeatureEngineer()
    sentiment_features = engineer.create_all_sentiment_features(sentiment_data)

    return sentiment_features


def merge_features(technical_X, sentiment_features):
    """
    Merge technical features with sentiment features.

    Uses inner join on date index to ensure alignment.
    Handles missing values appropriately.

    Args:
        technical_X: DataFrame with technical features
        sentiment_features: DataFrame with sentiment features

    Returns:
        DataFrame with combined features
    """
    logger.info("\nMerging technical and sentiment features...")

    # Remove timezone if present
    if hasattr(technical_X.index, 'tz') and technical_X.index.tz is not None:
        technical_X.index = technical_X.index.tz_localize(None)

    if hasattr(sentiment_features.index, 'tz') and sentiment_features.index.tz is not None:
        sentiment_features.index = sentiment_features.index.tz_localize(None)

    # Normalize to date only (remove time component for matching)
    # Technical data has 04:00:00 (from UTC conversion)
    # Sentiment data has 00:00:00 (midnight)
    # We need to match on date only
    technical_X.index = technical_X.index.normalize()
    sentiment_features.index = sentiment_features.index.normalize()

    # Use inner join to keep only dates with both technical and sentiment data
    combined = technical_X.join(sentiment_features, how='inner')

    logger.info(f"  Technical features: {len(technical_X.columns)}")
    logger.info(f"  Sentiment features: {len(sentiment_features.columns)}")
    logger.info(f"  Combined features: {len(combined.columns)}")
    logger.info(f"  Rows before merge: {len(technical_X)}")
    logger.info(f"  Rows after merge: {len(combined)}")

    # Check for missing values
    missing = combined.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"  Missing values found:")
        for col, count in missing[missing > 0].items():
            logger.warning(f"    {col}: {count}")

        # Fill missing with forward fill then backward fill
        combined = combined.fillna(method='ffill').fillna(method='bfill')

        missing_after = combined.isnull().sum().sum()
        logger.info(f"  Missing values after filling: {missing_after}")

    return combined


def save_combined_data(train_X_combined, train_y, test_X_combined, test_y):
    """
    Save combined datasets to disk.

    Args:
        train_X_combined: Training features (technical + sentiment)
        train_y: Training target
        test_X_combined: Test features (technical + sentiment)
        test_y: Test target
    """
    logger.info("\nSaving combined datasets...")

    output_dir = Path('data/train_test_sentiment')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Normalize train_y and test_y indices to match X indices
    if hasattr(train_y.index, 'tz') and train_y.index.tz is not None:
        train_y.index = train_y.index.tz_localize(None)
    train_y.index = train_y.index.normalize()

    if hasattr(test_y.index, 'tz') and test_y.index.tz is not None:
        test_y.index = test_y.index.tz_localize(None)
    test_y.index = test_y.index.normalize()

    # Align y with X (in case some dates were dropped during merge)
    train_y_aligned = train_y.loc[train_X_combined.index]
    test_y_aligned = test_y.loc[test_X_combined.index]

    # Save combined features
    train_X_combined.to_csv(output_dir / 'train_X.csv')
    test_X_combined.to_csv(output_dir / 'test_X.csv')

    # Save targets (aligned)
    train_y_aligned.to_frame('target').to_csv(output_dir / 'train_y.csv')
    test_y_aligned.to_frame('target').to_csv(output_dir / 'test_y.csv')

    logger.info(f"  Saved train_X.csv: {train_X_combined.shape}")
    logger.info(f"  Saved test_X.csv: {test_X_combined.shape}")
    logger.info(f"  Saved train_y.csv: {len(train_y_aligned)} samples")
    logger.info(f"  Saved test_y.csv: {len(test_y_aligned)} samples")

    logger.info(f"\nAll files saved to: {output_dir}/")


def main():
    """
    Main integration pipeline.

    Steps:
    1. Load existing technical features
    2. Collect sentiment data
    3. Create sentiment features
    4. Merge technical + sentiment
    5. Save combined datasets
    """
    print("\n" + "=" * 70)
    print("SENTIMENT DATA INTEGRATION")
    print("=" * 70)
    print("\nCombining technical features (11) + sentiment features (~15)")
    print("Expected output: ~26 total features")
    print("=" * 70 + "\n")

    try:
        # 1. Load existing data
        print("\n" + "=" * 70)
        print("[1/5] LOADING EXISTING DATA")
        print("=" * 70)

        train_X, train_y, test_X, test_y, train_data, test_data = load_existing_data()

        # Get full date range (need extra dates for feature lags)
        full_start = min(train_X.index.min(), test_X.index.min())
        full_end = max(train_X.index.max(), test_X.index.max())

        # Add buffer for rolling calculations (30 days before)
        full_start = (full_start - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        full_end = full_end.strftime('%Y-%m-%d')

        # 2. Collect and create sentiment features
        print("\n" + "=" * 70)
        print("[2/5] COLLECTING SENTIMENT DATA")
        print("=" * 70)

        sentiment_features = collect_and_create_sentiment_features(full_start, full_end)

        # 3. Merge train features
        print("\n" + "=" * 70)
        print("[3/5] MERGING TRAIN FEATURES")
        print("=" * 70)

        train_X_combined = merge_features(train_X, sentiment_features)

        # 4. Merge test features
        print("\n" + "=" * 70)
        print("[4/5] MERGING TEST FEATURES")
        print("=" * 70)

        test_X_combined = merge_features(test_X, sentiment_features)

        # 5. Save combined data
        print("\n" + "=" * 70)
        print("[5/5] SAVING COMBINED DATA")
        print("=" * 70)

        save_combined_data(train_X_combined, train_y, test_X_combined, test_y)

        # Summary
        print("\n" + "=" * 70)
        print("✓ SENTIMENT INTEGRATION COMPLETE!")
        print("=" * 70)

        print(f"\nCombined datasets:")
        print(f"  Train: {train_X_combined.shape[0]} samples × {train_X_combined.shape[1]} features")
        print(f"  Test:  {test_X_combined.shape[0]} samples × {test_X_combined.shape[1]} features")

        print(f"\nFeature breakdown:")
        print(f"  Technical features:  {len(train_X.columns)}")
        print(f"  Sentiment features:  {len(sentiment_features.columns)}")
        print(f"  Total features:      {len(train_X_combined.columns)}")

        print(f"\nOutput location: data/train_test_sentiment/")

        print(f"\nFeature list:")
        for i, col in enumerate(train_X_combined.columns, 1):
            feature_type = "technical" if col in train_X.columns else "sentiment"
            print(f"  {i:2d}. {col:30s} [{feature_type}]")

        print("\n" + "=" * 70)
        print("NEXT STEP: Run XGBoost with sentiment features")
        print("Command: python run_sentiment_model.py")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Make sure you've run pipeline.py first!")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error during integration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
