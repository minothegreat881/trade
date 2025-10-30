"""
Main Pipeline Script for ML Trading System

This script orchestrates the entire data pipeline:
1. Download historical data
2. Validate raw data quality
3. Create features
4. Create target variable
5. Clean data (remove NaN)
6. Split train/test (temporal split)
7. Save processed data
8. Generate quality report

Usage:
    python pipeline.py

Configuration can be modified in the main() function.
"""

import logging
import os
from datetime import datetime
import pandas as pd

from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from data_validator import generate_data_quality_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories for data storage."""
    directories = [
        'data/raw',
        'data/processed',
        'data/train_test'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ready: {directory}")


def save_split_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    output_dir: str = 'data/train_test'
) -> None:
    """
    Save train and test data to CSV files.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names
        output_dir: Directory to save files
    """
    # Save full datasets
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')

    train_df.to_csv(train_path)
    test_df.to_csv(test_path)

    logger.info(f"Saved train data to {train_path}")
    logger.info(f"Saved test data to {test_path}")

    # Save feature-only versions (for easy model training)
    train_X_path = os.path.join(output_dir, 'train_X.csv')
    train_y_path = os.path.join(output_dir, 'train_y.csv')
    test_X_path = os.path.join(output_dir, 'test_X.csv')
    test_y_path = os.path.join(output_dir, 'test_y.csv')

    train_df[feature_cols].to_csv(train_X_path)
    train_df['target'].to_csv(train_y_path)
    test_df[feature_cols].to_csv(test_X_path)
    test_df['target'].to_csv(test_y_path)

    logger.info("Saved separate X and y files for easy model training")


def main():
    """
    Main pipeline execution.

    Configuration:
    - TICKER: Stock ticker symbol (default: SPY)
    - START_DATE: Start of historical data (default: 2020-01-01)
    - END_DATE: End of historical data (default: 2024-12-31)
    - HOLDING_PERIOD: Target holding period in days (default: 5)
    - TRAIN_SPLIT: Fraction of data for training (default: 0.7)
    """
    logger.info("=" * 60)
    logger.info("ML TRADING SYSTEM - DATA PIPELINE")
    logger.info("=" * 60)

    # ===== CONFIGURATION =====
    TICKER = 'SPY'
    START_DATE = '2020-01-01'
    END_DATE = '2024-12-31'
    HOLDING_PERIOD = 5  # days
    TRAIN_SPLIT = 0.7  # 70% train, 30% test

    logger.info(f"\nConfiguration:")
    logger.info(f"  Ticker: {TICKER}")
    logger.info(f"  Date range: {START_DATE} to {END_DATE}")
    logger.info(f"  Holding period: {HOLDING_PERIOD} days")
    logger.info(f"  Train/test split: {int(TRAIN_SPLIT*100)}/{int((1-TRAIN_SPLIT)*100)}")

    # ===== STEP 1: CREATE DIRECTORIES =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Creating directories")
    logger.info("=" * 60)
    create_directories()

    # ===== STEP 2: DOWNLOAD DATA =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Downloading historical data")
    logger.info("=" * 60)

    collector = DataCollector(TICKER)
    df_raw = collector.download_historical(START_DATE, END_DATE)

    logger.info(f"Downloaded {len(df_raw)} rows")
    logger.info(f"Date range: {df_raw.index[0]} to {df_raw.index[-1]}")

    # Save raw data
    raw_data_path = f'data/raw/{TICKER}_historical.csv'
    collector.save_to_csv(df_raw, raw_data_path)

    # ===== STEP 3: VALIDATE RAW DATA =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Validating raw data")
    logger.info("=" * 60)

    validation_report = collector.validate_data(df_raw)

    if not validation_report['is_valid']:
        logger.warning("⚠️  Raw data has quality issues. Proceeding with caution...")
        for issue in validation_report['issues']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("✓ Raw data validation passed")

    # ===== STEP 4: CREATE FEATURES =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Creating features")
    logger.info("=" * 60)

    engineer = FeatureEngineer()
    df_featured = engineer.create_basic_features(df_raw.copy())

    logger.info(f"Features created: {len(engineer.get_feature_names())}")
    logger.info(f"Feature names: {engineer.get_feature_names()}")

    # ===== STEP 5: CREATE TARGET VARIABLE =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Creating target variable")
    logger.info("=" * 60)

    df_featured = engineer.create_target(df_featured, horizon=HOLDING_PERIOD)
    logger.info(f"Target variable created: {HOLDING_PERIOD}-day forward return")

    # ===== STEP 6: CLEAN DATA (REMOVE NaN) =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Cleaning data")
    logger.info("=" * 60)

    rows_before = len(df_featured)
    logger.info(f"Rows before cleaning: {rows_before}")

    # Remove rows with NaN in features or target
    feature_cols = engineer.get_feature_names()
    cols_to_check = feature_cols + ['target']
    df_clean = df_featured.dropna(subset=cols_to_check)

    rows_after = len(df_clean)
    rows_lost = rows_before - rows_after

    logger.info(f"Rows after cleaning: {rows_after}")
    logger.info(f"Rows lost: {rows_lost} ({rows_lost/rows_before*100:.1f}%)")

    if rows_after == 0:
        logger.error("❌ No data remaining after cleaning! Check your features.")
        return

    # Save featured data
    featured_data_path = f'data/processed/{TICKER}_featured.csv'
    df_clean.to_csv(featured_data_path)
    logger.info(f"Saved featured data to {featured_data_path}")

    # ===== STEP 7: TRAIN/TEST SPLIT (TEMPORAL) =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Splitting train/test")
    logger.info("=" * 60)

    # Temporal split (no shuffling!)
    split_idx = int(len(df_clean) * TRAIN_SPLIT)

    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()

    logger.info(f"Train set: {len(train_df)} rows ({len(train_df)/len(df_clean)*100:.1f}%)")
    logger.info(f"  Period: {train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')}")

    logger.info(f"Test set: {len(test_df)} rows ({len(test_df)/len(df_clean)*100:.1f}%)")
    logger.info(f"  Period: {test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}")

    # Verify no overlap
    if train_df.index[-1] >= test_df.index[0]:
        logger.error("❌ CRITICAL: Train and test sets overlap! This should not happen.")
        return
    else:
        logger.info("✓ No overlap between train and test sets")

    # ===== STEP 8: SAVE SPLIT DATA =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: Saving train/test data")
    logger.info("=" * 60)

    save_split_data(train_df, test_df, feature_cols)

    # ===== STEP 9: GENERATE QUALITY REPORT =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 9: Generating data quality report")
    logger.info("=" * 60)

    report = generate_data_quality_report(df_clean, feature_cols=feature_cols)

    # Save report to file
    report_path = f'data/processed/data_quality_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # ===== PIPELINE COMPLETE =====
    logger.info("\n" + "=" * 60)
    logger.info("✓ PIPELINE COMPLETE!")
    logger.info("=" * 60)

    logger.info("\nSummary:")
    logger.info(f"  - Raw data: {len(df_raw)} rows")
    logger.info(f"  - After feature creation: {rows_before} rows")
    logger.info(f"  - After cleaning: {rows_after} rows ({rows_lost} rows removed)")
    logger.info(f"  - Train set: {len(train_df)} rows")
    logger.info(f"  - Test set: {len(test_df)} rows")
    logger.info(f"  - Features: {len(feature_cols)}")

    logger.info("\nNext steps:")
    logger.info("  1. Review the data quality report")
    logger.info("  2. Run unit tests: pytest tests/ -v")
    logger.info("  3. Proceed to Phase 2: Model development")

    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise
